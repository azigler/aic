#
#  DataCollector - Wraps CheatCode to record velocity-mode training data
#  for LeRobot ACTPolicy fine-tuning.
#
#  Records: 3 camera images (288x256), 26D state, 7D velocity action
#  Saves episodes as numpy arrays for later conversion to LeRobot format.
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
from rclpy.node import Node

from .CheatCode import CheatCode


class DataCollector(CheatCode):
    """Wraps CheatCode to record expert demonstrations for ACT training.

    Records observations and velocity actions while CheatCode solves the task.
    The velocity action is the observed TCP velocity (what the robot actually does),
    which is the correct label for training a velocity-mode policy like RunACT.
    """

    def __init__(self, parent_node: Node):
        super().__init__(parent_node)
        self._recording = False
        self._get_observation = None
        self._episode_data = {
            "states": [],
            "actions": [],
            "images_left": [],
            "images_center": [],
            "images_right": [],
            "timestamps": [],
        }
        self._save_dir = os.environ.get(
            "TRAINING_DATA_DIR", os.path.expanduser("~/aic/data/velocity")
        )
        os.makedirs(self._save_dir, exist_ok=True)
        self._episode_count = self._count_existing_episodes()
        self._image_scaling = 0.25  # Match RunACT: 1152x1024 -> 288x256
        # Record at 10Hz for better insertion dynamics capture
        # RunACT runs at 4Hz but we want higher fidelity training data
        record_hz = float(os.environ.get("RECORD_HZ", "10"))
        self._record_interval = 1.0 / record_hz
        self._last_record_time = 0.0
        self.get_logger().info(
            f"DataCollector initialized. Save dir: {self._save_dir}, "
            f"Existing episodes: {self._episode_count}"
        )

    def _count_existing_episodes(self) -> int:
        """Count existing episode directories."""
        count = 0
        if os.path.exists(self._save_dir):
            for d in os.listdir(self._save_dir):
                if d.startswith("episode_"):
                    count += 1
        return count

    def _extract_state(self, obs: Observation) -> np.ndarray:
        """Extract 26D state vector matching RunACT's prepare_observations."""
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

    def _extract_velocity_action(self, obs: Observation) -> np.ndarray:
        """Extract 7D velocity action from observation.

        Uses the actual TCP velocity as the expert action.
        This is what the robot achieved while following CheatCode's position targets.
        action[0:3] = linear velocity (x, y, z)
        action[3:6] = angular velocity (x, y, z)
        action[6] = gripper (constant, matching pretrained model)
        """
        tcp_vel = obs.controller_state.tcp_velocity
        return np.array(
            [
                tcp_vel.linear.x,
                tcp_vel.linear.y,
                tcp_vel.linear.z,
                tcp_vel.angular.x,
                tcp_vel.angular.y,
                tcp_vel.angular.z,
                0.012,  # Gripper constant (matches pretrained action_mean[6])
            ],
            dtype=np.float32,
        )

    def _extract_image(self, raw_img) -> np.ndarray:
        """Convert ROS Image to scaled numpy array (H, W, 3) uint8."""
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

    def _record_observation(self, get_observation: GetObservationCallback):
        """Record a single observation + action pair."""
        obs = get_observation()
        if obs is None:
            return

        state = self._extract_state(obs)
        action = self._extract_velocity_action(obs)
        img_left = self._extract_image(obs.left_image)
        img_center = self._extract_image(obs.center_image)
        img_right = self._extract_image(obs.right_image)

        self._episode_data["states"].append(state)
        self._episode_data["actions"].append(action)
        self._episode_data["images_left"].append(img_left)
        self._episode_data["images_center"].append(img_center)
        self._episode_data["images_right"].append(img_right)
        self._episode_data["timestamps"].append(time.time())

    def _save_episode(self):
        """Save recorded episode data as numpy arrays."""
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

        # Save images as individual files (more memory-efficient for large datasets)
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
            f"Saved episode {self._episode_count} with {n_steps} steps to {ep_dir}"
        )
        self._episode_count += 1

    def _reset_episode(self):
        """Clear episode buffer."""
        self._episode_data = {
            "states": [],
            "actions": [],
            "images_left": [],
            "images_center": [],
            "images_right": [],
            "timestamps": [],
        }
        self._last_record_time = 0.0

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        """Run CheatCode while recording observations and velocity actions."""
        self.get_logger().info(
            f"DataCollector.insert_cable() task: {task}, "
            f"recording episode {self._episode_count}"
        )
        self._reset_episode()
        self._recording = True
        self._get_observation = get_observation

        # Run CheatCode's insertion (which calls our overridden sleep_for)
        result = super().insert_cable(
            task, get_observation, move_robot, send_feedback
        )

        self._recording = False

        # Save the episode
        self._save_episode()

        self.get_logger().info(
            f"DataCollector.insert_cable() done, result={result}"
        )
        return result

    def sleep_for(self, duration_sec: float) -> None:
        """Override sleep_for to record observations during CheatCode's sleeps."""
        if self._recording and self._get_observation is not None:
            now = time.time()
            if (now - self._last_record_time) >= self._record_interval:
                obs = self._get_observation()
                if obs is not None:
                    self._last_record_time = now
                    state = self._extract_state(obs)
                    action = self._extract_velocity_action(obs)
                    img_left = self._extract_image(obs.left_image)
                    img_center = self._extract_image(obs.center_image)
                    img_right = self._extract_image(obs.right_image)

                    self._episode_data["states"].append(state)
                    self._episode_data["actions"].append(action)
                    self._episode_data["images_left"].append(img_left)
                    self._episode_data["images_center"].append(img_center)
                    self._episode_data["images_right"].append(img_right)
                    self._episode_data["timestamps"].append(now)

        # Call the actual sleep
        super().sleep_for(duration_sec)
