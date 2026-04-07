#
#  DataCollectorV2 - Records velocity commands computed from CheatCode's
#  position targets. This produces better training labels than observed
#  TCP velocity because it maintains a nonzero signal during insertion.
#

import os
import time

import cv2
import numpy as np
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Pose
from rclpy.node import Node

from .CheatCode import CheatCode


class DataCollectorV2(CheatCode):
    """Records velocity actions computed from CheatCode's position targets.

    Instead of recording the observed TCP velocity (which goes to ~0 near
    insertion as the controller settles), this computes the velocity as:

        v_linear = K_lin * (target_position - current_position)
        v_angular = observed angular velocity (hard to compute from quaternions)

    This gives a nonzero velocity signal that points toward the insertion
    target, which is what a velocity-mode policy should output.
    """

    def __init__(self, parent_node: Node):
        super().__init__(parent_node)
        self._recording = False
        self._get_observation = None
        self._last_target_pose = None
        self._episode_data = {
            "states": [],
            "actions": [],
            "images_left": [],
            "images_center": [],
            "images_right": [],
            "timestamps": [],
        }
        self._save_dir = os.environ.get(
            "TRAINING_DATA_DIR",
            os.path.expanduser("~/training_data_v2"),
        )
        os.makedirs(self._save_dir, exist_ok=True)
        self._episode_count = self._count_existing_episodes()
        self._image_scaling = 0.25
        record_hz = float(os.environ.get("RECORD_HZ", "4"))
        self._record_interval = 1.0 / record_hz
        self._last_record_time = 0.0
        # Gain for converting position error to velocity command
        self._k_lin = float(os.environ.get("K_LIN", "2.0"))
        self.get_logger().info(
            f"DataCollectorV2 initialized. Save dir: {self._save_dir}, "
            f"K_lin: {self._k_lin}, record_hz: {record_hz}"
        )

    def _count_existing_episodes(self) -> int:
        count = 0
        if os.path.exists(self._save_dir):
            for d in os.listdir(self._save_dir):
                if d.startswith("episode_"):
                    count += 1
        return count

    def _extract_state(self, obs: Observation) -> np.ndarray:
        tcp_pose = obs.controller_state.tcp_pose
        tcp_vel = obs.controller_state.tcp_velocity
        return np.array(
            [
                tcp_pose.position.x,
                tcp_pose.position.y,
                tcp_pose.position.z,
                tcp_pose.orientation.x,
                tcp_pose.orientation.y,
                tcp_pose.orientation.z,
                tcp_pose.orientation.w,
                tcp_vel.linear.x,
                tcp_vel.linear.y,
                tcp_vel.linear.z,
                tcp_vel.angular.x,
                tcp_vel.angular.y,
                tcp_vel.angular.z,
                *obs.controller_state.tcp_error,
                *obs.joint_states.position[:7],
            ],
            dtype=np.float32,
        )

    def _compute_velocity_action(
        self, obs: Observation, target_pose: Pose
    ) -> np.ndarray:
        """Compute velocity command from position error to target.

        v_lin = K * (target_pos - current_pos)
        v_ang = observed angular velocity (from state)

        This produces a nonzero velocity signal that points toward
        the insertion target, even when the robot is very close.
        """
        tcp_pose = obs.controller_state.tcp_pose
        tcp_vel = obs.controller_state.tcp_velocity

        # Linear velocity: P-controller on position error
        pos_error = np.array(
            [
                target_pose.position.x - tcp_pose.position.x,
                target_pose.position.y - tcp_pose.position.y,
                target_pose.position.z - tcp_pose.position.z,
            ]
        )
        lin_vel = self._k_lin * pos_error

        # Angular velocity: use observed (computing from quaternion diff is noisy)
        ang_vel = np.array(
            [
                tcp_vel.angular.x,
                tcp_vel.angular.y,
                tcp_vel.angular.z,
            ]
        )

        return np.array(
            [*lin_vel, *ang_vel, 0.012],
            dtype=np.float32,
        )

    def _extract_image(self, raw_img) -> np.ndarray:
        img_np = np.frombuffer(raw_img.data, dtype=np.uint8).reshape(
            raw_img.height, raw_img.width, 3
        )
        if self._image_scaling != 1.0:
            img_np = cv2.resize(
                img_np,
                None,
                fx=self._image_scaling,
                fy=self._image_scaling,
                interpolation=cv2.INTER_AREA,
            )
        return img_np

    def _save_episode(self):
        n_steps = len(self._episode_data["states"])
        if n_steps < 10:
            self.get_logger().warn(
                f"Episode too short ({n_steps} steps), skipping save."
            )
            return

        ep_dir = os.path.join(
            self._save_dir, f"episode_{self._episode_count:04d}"
        )
        os.makedirs(ep_dir, exist_ok=True)

        np.save(
            os.path.join(ep_dir, "states.npy"),
            np.array(self._episode_data["states"]),
        )
        np.save(
            os.path.join(ep_dir, "actions.npy"),
            np.array(self._episode_data["actions"]),
        )
        np.save(
            os.path.join(ep_dir, "timestamps.npy"),
            np.array(self._episode_data["timestamps"]),
        )

        for i in range(n_steps):
            np.save(
                os.path.join(ep_dir, f"images_left_{i:04d}.npy"),
                self._episode_data["images_left"][i],
            )
            np.save(
                os.path.join(ep_dir, f"images_center_{i:04d}.npy"),
                self._episode_data["images_center"][i],
            )
            np.save(
                os.path.join(ep_dir, f"images_right_{i:04d}.npy"),
                self._episode_data["images_right"][i],
            )

        self.get_logger().info(
            f"Saved episode {self._episode_count} with {n_steps} steps "
            f"to {ep_dir}"
        )
        self._episode_count += 1

    def _reset_episode(self):
        self._episode_data = {
            "states": [],
            "actions": [],
            "images_left": [],
            "images_center": [],
            "images_right": [],
            "timestamps": [],
        }
        self._last_record_time = 0.0
        self._last_target_pose = None

    def set_pose_target(
        self,
        move_robot,
        pose,
        frame_id="base_link",
        stiffness=None,
        damping=None,
    ):
        """Override to capture the target pose for velocity computation."""
        if stiffness is None:
            stiffness = [90.0, 90.0, 90.0, 50.0, 50.0, 50.0]
        if damping is None:
            damping = [50.0, 50.0, 50.0, 20.0, 20.0, 20.0]
        self._last_target_pose = pose
        super().set_pose_target(move_robot, pose, frame_id, stiffness, damping)

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        self.get_logger().info(
            f"DataCollectorV2.insert_cable() task: {task}, "
            f"recording episode {self._episode_count}"
        )
        self._reset_episode()
        self._recording = True
        self._get_observation = get_observation

        result = super().insert_cable(
            task, get_observation, move_robot, send_feedback
        )

        self._recording = False
        self._save_episode()

        self.get_logger().info(
            f"DataCollectorV2.insert_cable() done, result={result}"
        )
        return result

    def sleep_for(self, duration_sec: float) -> None:
        """Record during sleeps, using target pose for velocity computation."""
        if (
            self._recording
            and self._get_observation is not None
            and self._last_target_pose is not None
        ):
            now = time.time()
            if (now - self._last_record_time) >= self._record_interval:
                obs = self._get_observation()
                if obs is not None:
                    self._last_record_time = now
                    state = self._extract_state(obs)
                    action = self._compute_velocity_action(
                        obs, self._last_target_pose
                    )
                    img_left = self._extract_image(obs.left_image)
                    img_center = self._extract_image(obs.center_image)
                    img_right = self._extract_image(obs.right_image)

                    self._episode_data["states"].append(state)
                    self._episode_data["actions"].append(action)
                    self._episode_data["images_left"].append(img_left)
                    self._episode_data["images_center"].append(img_center)
                    self._episode_data["images_right"].append(img_right)
                    self._episode_data["timestamps"].append(now)

        super().sleep_for(duration_sec)
