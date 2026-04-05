#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""Data collection policy for ACT training.

Wraps CheatCode to record expert demonstrations. Each trial produces one
episode saved as numpy arrays + metadata, suitable for later conversion
to a LeRobot dataset.

Observations recorded per timestep:
  - 3 camera images (left, center, right) downsampled to 256x256
  - 26D robot state (matching RunACT.prepare_observations layout)

Actions recorded per timestep:
  - 7D velocity-like vector: TCP pose delta (position 3 + orientation 3)
    normalized to per-timestep velocity, plus gripper state (1).
    This matches RunACT's velocity-mode output format.

Usage:
  POLICY=aic_example_policies.ros.DataCollector scripts/remote-eval.sh
  (requires ground_truth=true for CheatCode)
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from rclpy.node import Node

from .CheatCode import CheatCode

# Global episode counter persists across trials within one engine run
# (engine runs 3 trials per run, each trial calls insert_cable once).
_episode_counter_file = Path.home() / "training_data" / ".episode_counter"


def _next_episode_index() -> int:
    """Atomically increment and return the next episode index."""
    counter_file = _episode_counter_file
    counter_file.parent.mkdir(parents=True, exist_ok=True)
    idx = int(counter_file.read_text().strip()) if counter_file.exists() else 0
    counter_file.write_text(str(idx + 1))
    return idx


class DataCollector(Policy):
    """Records CheatCode demonstrations for ACT training.

    Extends Policy (NOT CheatCode) to avoid diamond inheritance issues.
    Internally creates a CheatCode instance and delegates to it, while
    intercepting observations and actions to save them as training data.

    Each trial is saved as one episode in ~/training_data/episode_NNN/.
    """

    # Target image size for saved data (matches ACT training resolution)
    IMG_SIZE = 256

    # Recording rate target in Hz
    RECORD_HZ = 20.0

    def __init__(self, parent_node: Node):
        super().__init__(parent_node)

        # Create the expert policy we delegate to
        self._expert = CheatCode(parent_node)

        # Data directory
        self._data_dir = Path.home() / "training_data"
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # Per-episode buffers (reset each trial)
        self._images_left: list[np.ndarray] = []
        self._images_center: list[np.ndarray] = []
        self._images_right: list[np.ndarray] = []
        self._states: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []

        # Track previous TCP pose for computing velocity-like deltas
        self._prev_tcp_position: np.ndarray | None = None
        self._prev_tcp_orientation: np.ndarray | None = None
        self._prev_time: float | None = None

        # Gripper state (extracted from observations)
        self._gripper_position: float = 0.0

        self.get_logger().info("DataCollector initialized.")

    def _reset_buffers(self):
        """Clear all recording buffers for a new episode."""
        self._images_left.clear()
        self._images_center.clear()
        self._images_right.clear()
        self._states.clear()
        self._actions.clear()
        self._prev_tcp_position = None
        self._prev_tcp_orientation = None
        self._prev_time = None
        self._gripper_position = 0.0

    @staticmethod
    def _ros_image_to_numpy(raw_img, target_size: int = 256) -> np.ndarray:
        """Convert a ROS Image message to a downsampled (H, W, 3) uint8 numpy array."""
        img_np = np.frombuffer(raw_img.data, dtype=np.uint8).reshape(
            raw_img.height, raw_img.width, 3
        )
        if img_np.shape[0] != target_size or img_np.shape[1] != target_size:
            img_np = cv2.resize(
                img_np,
                (target_size, target_size),
                interpolation=cv2.INTER_AREA,
            )
        return img_np

    def _extract_state(self, obs: Observation) -> np.ndarray:
        """Extract the 26D state vector from an Observation message.

        Layout matches RunACT.prepare_observations exactly:
          [0:3]   TCP position (x, y, z)
          [3:7]   TCP orientation quaternion (x, y, z, w)
          [7:10]  TCP linear velocity (x, y, z)
          [10:13] TCP angular velocity (x, y, z)
          [13:19] TCP tracking error (6D)
          [19:26] Joint positions (7: 6 arm + 1 gripper)
        """
        tcp_pose = obs.controller_state.tcp_pose
        tcp_vel = obs.controller_state.tcp_velocity

        state = np.array(
            [
                # TCP Position (3)
                tcp_pose.position.x,
                tcp_pose.position.y,
                tcp_pose.position.z,
                # TCP Orientation quaternion (4)
                tcp_pose.orientation.x,
                tcp_pose.orientation.y,
                tcp_pose.orientation.z,
                tcp_pose.orientation.w,
                # TCP Linear Velocity (3)
                tcp_vel.linear.x,
                tcp_vel.linear.y,
                tcp_vel.linear.z,
                # TCP Angular Velocity (3)
                tcp_vel.angular.x,
                tcp_vel.angular.y,
                tcp_vel.angular.z,
                # TCP Error (6)
                *obs.controller_state.tcp_error,
                # Joint Positions (7)
                *obs.joint_states.position[:7],
            ],
            dtype=np.float32,
        )
        return state

    def _compute_velocity_action(
        self, obs: Observation, current_time: float
    ) -> np.ndarray | None:
        """Compute a 7D velocity-like action from TCP pose deltas.

        Returns None for the first timestep (no previous pose to diff against).

        Action format (matching RunACT velocity mode):
          [0:3] linear velocity  (m/s)  - TCP position delta / dt
          [3:6] angular velocity (rad/s) - approximate from quaternion delta
          [6]   gripper position (0..1)
        """
        tcp_pose = obs.controller_state.tcp_pose
        pos = np.array(
            [tcp_pose.position.x, tcp_pose.position.y, tcp_pose.position.z],
            dtype=np.float32,
        )
        quat = np.array(
            [
                tcp_pose.orientation.x,
                tcp_pose.orientation.y,
                tcp_pose.orientation.z,
                tcp_pose.orientation.w,
            ],
            dtype=np.float32,
        )

        # Extract gripper position from joint states (last joint)
        joint_positions = list(obs.joint_states.position)
        if len(joint_positions) >= 7:
            self._gripper_position = float(joint_positions[6])

        if self._prev_tcp_position is None:
            self._prev_tcp_position = pos.copy()
            self._prev_tcp_orientation = quat.copy()
            self._prev_time = current_time
            return None

        dt = current_time - self._prev_time
        if dt < 1e-6:
            dt = 1.0 / self.RECORD_HZ  # fallback

        # Linear velocity: simple position delta / dt
        linear_vel = (pos - self._prev_tcp_position) / dt

        # Angular velocity: approximate from quaternion delta
        # Using the small-angle approximation: omega ~= 2 * (q_delta.xyz) / dt
        # where q_delta = q_current * q_prev_inv
        q_prev_inv = np.array(
            [
                -self._prev_tcp_orientation[0],
                -self._prev_tcp_orientation[1],
                -self._prev_tcp_orientation[2],
                self._prev_tcp_orientation[3],
            ],
            dtype=np.float32,
        )
        # Hamilton product: q_delta = q_current * q_prev_inv
        q_delta = _quat_multiply(quat, q_prev_inv)
        # Ensure w > 0 for consistent sign
        if q_delta[3] < 0:
            q_delta = -q_delta
        angular_vel = 2.0 * q_delta[:3] / dt

        self._prev_tcp_position = pos.copy()
        self._prev_tcp_orientation = quat.copy()
        self._prev_time = current_time

        action = np.array(
            [
                linear_vel[0],
                linear_vel[1],
                linear_vel[2],
                angular_vel[0],
                angular_vel[1],
                angular_vel[2],
                self._gripper_position,
            ],
            dtype=np.float32,
        )
        return action

    def _record_observation(self, obs: Observation, current_time: float):
        """Record one timestep of observation and compute the corresponding action."""
        # Images
        self._images_left.append(
            self._ros_image_to_numpy(obs.left_image, self.IMG_SIZE)
        )
        self._images_center.append(
            self._ros_image_to_numpy(obs.center_image, self.IMG_SIZE)
        )
        self._images_right.append(
            self._ros_image_to_numpy(obs.right_image, self.IMG_SIZE)
        )

        # State
        state = self._extract_state(obs)
        self._states.append(state)

        # Action (velocity from pose delta)
        action = self._compute_velocity_action(obs, current_time)
        if action is not None:
            self._actions.append(action)

    def _save_episode(self, task: Task, duration: float):
        """Save the recorded episode to disk as numpy arrays + metadata.

        Aligns states and actions so that states[t] pairs with actions[t].
        The first state has no corresponding action (no previous pose), so we
        drop states[0] to keep lengths equal.
        """
        episode_idx = _next_episode_index()
        episode_dir = self._data_dir / f"episode_{episode_idx:04d}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        num_actions = len(self._actions)
        if num_actions == 0:
            self.get_logger().warn("No actions recorded, skipping save.")
            return

        # Align: drop first state/images (no action for first timestep)
        # states[1:], images[1:] pair with actions[0:]
        states = np.array(self._states[1 : num_actions + 1], dtype=np.float32)
        actions = np.array(self._actions[:num_actions], dtype=np.float32)

        # Save images frame by frame (saves memory vs one big array)
        for i in range(num_actions):
            img_idx = i + 1  # offset by 1 to match aligned indexing
            if img_idx < len(self._images_left):
                np.save(
                    episode_dir / f"images_left_{i:04d}.npy",
                    self._images_left[img_idx],
                )
                np.save(
                    episode_dir / f"images_center_{i:04d}.npy",
                    self._images_center[img_idx],
                )
                np.save(
                    episode_dir / f"images_right_{i:04d}.npy",
                    self._images_right[img_idx],
                )

        # Save states and actions as single arrays
        np.save(episode_dir / "states.npy", states)
        np.save(episode_dir / "actions.npy", actions)

        # Metadata
        metadata = {
            "episode_index": episode_idx,
            "num_timesteps": num_actions,
            "duration_seconds": duration,
            "record_hz": self.RECORD_HZ,
            "image_size": self.IMG_SIZE,
            "state_dim": 26,
            "action_dim": 7,
            "action_format": "velocity: linear_xyz(3) + angular_xyz(3) + gripper(1)",
            "state_format": (
                "tcp_pos(3) + tcp_quat(4) + tcp_lin_vel(3) + tcp_ang_vel(3)"
                " + tcp_error(6) + joint_pos(7)"
            ),
            "task": {
                "cable_name": task.cable_name,
                "plug_name": task.plug_name,
                "target_module_name": task.target_module_name,
                "port_name": task.port_name,
            },
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(episode_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        self.get_logger().info(
            f"Episode {episode_idx} saved to {episode_dir} "
            f"({num_actions} timesteps, {duration:.1f}s)"
        )

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        """Run CheatCode while recording observations and actions.

        We wrap get_observation to intercept and record data at ~20Hz,
        then delegate the actual control to CheatCode.
        """
        self.get_logger().info(f"DataCollector.insert_cable() task: {task}")
        self._reset_buffers()

        start_time = time.time()
        last_record_time = 0.0

        def recording_get_observation() -> Observation:
            """Wrapper that records observation data before returning it."""
            nonlocal last_record_time
            obs = get_observation()
            if obs is None:
                return obs

            current_time = time.time()
            # Throttle recording to target Hz
            if current_time - last_record_time >= (1.0 / self.RECORD_HZ):
                self._record_observation(obs, current_time)
                last_record_time = current_time

            return obs

        def recording_move_robot(**kwargs):
            """Wrapper that records the current observation when a command is sent."""
            nonlocal last_record_time
            # Record observation at each move command (CheatCode doesn't call get_observation)
            obs = get_observation()
            if obs is not None:
                current_time = time.time()
                if current_time - last_record_time >= (1.0 / self.RECORD_HZ):
                    self._record_observation(obs, current_time)
                    last_record_time = current_time
            move_robot(**kwargs)

        # Delegate to CheatCode expert, passing our recording wrappers
        try:
            result = self._expert.insert_cable(
                task,
                recording_get_observation,
                recording_move_robot,
                send_feedback,
            )
        except Exception as ex:
            self.get_logger().error(f"CheatCode failed: {ex}")
            result = False

        duration = time.time() - start_time

        # Save the recorded episode
        try:
            self._save_episode(task, duration)
        except Exception as ex:
            self.get_logger().error(f"Failed to save episode: {ex}")

        return result


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions in (x, y, z, w) format."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float32,
    )
