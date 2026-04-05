#
# OurACT: Run our trained SimpleACT model for cable insertion.
# Loads model from ~/models/act_micro (or configured path).
#

import json
import os

# Import SimpleACT from training script
import sys
import time

import cv2
import numpy as np
import torch
from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Twist, Vector3, Wrench
from std_msgs.msg import Header

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts")
)
from train_act import SimpleACT


class OurACT(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load model
        model_dir = os.environ.get(
            "ACT_MODEL_DIR", os.path.expanduser("~/models/act_micro")
        )
        self.get_logger().info(f"Loading ACT model from {model_dir}")

        # Load normalization stats
        stats_path = os.path.join(model_dir, "normalization_stats.json")
        with open(stats_path) as f:
            stats = json.load(f)

        self.state_mean = torch.tensor(
            stats["state_mean"], dtype=torch.float32, device=self.device
        )
        self.state_std = torch.tensor(
            stats["state_std"], dtype=torch.float32, device=self.device
        )
        self.action_mean = torch.tensor(
            stats["action_mean"], dtype=torch.float32, device=self.device
        )
        self.action_std = torch.tensor(
            stats["action_std"], dtype=torch.float32, device=self.device
        )

        self.img_stats = {}
        for cam in ["left", "center", "right"]:
            self.img_stats[cam] = {
                "mean": torch.tensor(
                    stats[f"img_{cam}_mean"],
                    dtype=torch.float32,
                    device=self.device,
                ).view(1, 3, 1, 1),
                "std": torch.tensor(
                    stats[f"img_{cam}_std"],
                    dtype=torch.float32,
                    device=self.device,
                ).view(1, 3, 1, 1),
            }

        # Load model
        self.model = SimpleACT(state_dim=26, action_dim=7, chunk_size=50)
        checkpoint = torch.load(
            os.path.join(model_dir, "best_model.pt"), map_location=self.device
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.get_logger().info(f"ACT model loaded on {self.device}")

    def _process_image(self, img_msg, cam_name):
        """Convert ROS Image to normalized tensor."""
        img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
            img_msg.height, img_msg.width, 3
        )
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        tensor = (
            torch.from_numpy(img)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(self.device)
        )
        stats = self.img_stats[cam_name]
        return (tensor - stats["mean"]) / (stats["std"] + 1e-8)

    def _extract_state(self, obs):
        """Extract 26D state vector (matching training format)."""
        tcp = obs.controller_state.tcp_pose
        vel = obs.controller_state.tcp_velocity
        state = np.array(
            [
                tcp.position.x,
                tcp.position.y,
                tcp.position.z,
                tcp.orientation.x,
                tcp.orientation.y,
                tcp.orientation.z,
                tcp.orientation.w,
                vel.linear.x,
                vel.linear.y,
                vel.linear.z,
                vel.angular.x,
                vel.angular.y,
                vel.angular.z,
                *obs.controller_state.tcp_error,
                *list(obs.joint_states.position[:7]),
            ],
            dtype=np.float32,
        )
        tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
        return (tensor - self.state_mean) / (self.state_std + 1e-8)

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        self.get_logger().info(
            f"OurACT: port={task.port_name} plug={task.plug_type}"
        )

        start = time.time()
        step = 0

        while time.time() - start < 60.0:  # 60s max
            obs = get_observation()
            if obs is None:
                continue

            # Process observations
            img_left = self._process_image(obs.left_image, "left")
            img_center = self._process_image(obs.center_image, "center")
            img_right = self._process_image(obs.right_image, "right")
            state = self._extract_state(obs)

            # Inference
            with torch.inference_mode():
                action_norm = self.model(img_left, img_center, img_right, state)

            # Un-normalize action
            action = (
                (action_norm[0] * self.action_std + self.action_mean)
                .cpu()
                .numpy()
            )

            # Send velocity command
            motion = MotionUpdate()
            motion.header = Header(
                frame_id="base_link",
                stamp=self.get_clock().now().to_msg(),
            )
            motion.velocity = Twist(
                linear=Vector3(
                    x=float(action[0]), y=float(action[1]), z=float(action[2])
                ),
                angular=Vector3(
                    x=float(action[3]), y=float(action[4]), z=float(action[5])
                ),
            )
            motion.target_stiffness = np.diag(
                [100.0, 100.0, 100.0, 50.0, 50.0, 50.0]
            ).flatten()
            motion.target_damping = np.diag(
                [40.0, 40.0, 40.0, 15.0, 15.0, 15.0]
            ).flatten()
            motion.feedforward_wrench_at_tip = Wrench()
            motion.wrench_feedback_gains_at_tip = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
            motion.trajectory_generation_mode.mode = (
                TrajectoryGenerationMode.MODE_VELOCITY
            )
            move_robot(motion_update=motion)

            if step % 20 == 0:
                self.get_logger().info(
                    f"Step {step}: action=[{action[0]:.4f},{action[1]:.4f},{action[2]:.4f}]"
                )

            step += 1
            self.sleep_for(0.25)

        return True
