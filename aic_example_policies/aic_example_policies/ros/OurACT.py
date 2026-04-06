#
# OurACT: Run our trained SimpleACT model for cable insertion.
# Loads model from ~/models/act_micro (or configured path).
#
# Outputs POSITION targets (7D: x,y,z,qx,qy,qz,qw) via set_pose_target,
# matching the position-mode actions recorded by DataCollector.
#

import json
import os
import time

import cv2
import numpy as np
import torch
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion

from ..simple_act import SimpleACT


class OurACT(Policy):
    # Default image size for inference (must match training resolution)
    IMG_SIZE = 256

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

        # Read image size and offset mode from model config
        config_path = os.path.join(model_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            self.img_size = config.get("img_size", self.IMG_SIZE)
            self.offset_mode = config.get("offset_mode", False)
        else:
            self.img_size = self.IMG_SIZE
            self.offset_mode = False

        self.get_logger().info(f"Offset mode: {self.offset_mode}")

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

        # Load model (pretrained=False: we load full checkpoint, don't need ImageNet init)
        self.model = SimpleACT(
            state_dim=26,
            action_dim=7,
            chunk_size=50,
            img_size=self.img_size,
            pretrained=False,
        )
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
        img = cv2.resize(
            img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA
        )
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

            # Inference (SimpleACT.forward expects: state, img_left, img_center, img_right)
            with torch.inference_mode():
                action_norm = self.model(state, img_left, img_center, img_right)

            # Un-normalize action chunk: shape (chunk_size, 7)
            action_chunk = (
                (action_norm[0] * self.action_std + self.action_mean)
                .cpu()
                .numpy()
            )
            # Use first timestep from the chunk
            action = action_chunk[0]  # shape (7,): [x, y, z, qx, qy, qz, qw]

            # In offset mode, action[:3] is a delta — add to current TCP position
            if self.offset_mode:
                tcp = obs.controller_state.tcp_pose
                target_xyz = np.array(
                    [
                        tcp.position.x + action[0],
                        tcp.position.y + action[1],
                        tcp.position.z + action[2],
                    ]
                )
            else:
                target_xyz = action[:3]

            # Normalize quaternion to ensure unit length
            quat = action[3:7]
            quat_norm = np.linalg.norm(quat)
            if quat_norm > 1e-6:
                quat = quat / quat_norm
            else:
                quat = np.array([0.0, 0.0, 0.0, 1.0])

            # Send position target via set_pose_target (uses MODE_POSITION)
            target_pose = Pose(
                position=Point(
                    x=float(target_xyz[0]),
                    y=float(target_xyz[1]),
                    z=float(target_xyz[2]),
                ),
                orientation=Quaternion(
                    x=float(quat[0]),
                    y=float(quat[1]),
                    z=float(quat[2]),
                    w=float(quat[3]),
                ),
            )
            self.set_pose_target(move_robot, pose=target_pose)

            if step % 20 == 0:
                self.get_logger().info(
                    f"Step {step}: pos=[{target_xyz[0]:.4f},{target_xyz[1]:.4f},{target_xyz[2]:.4f}]"
                )

            step += 1
            self.sleep_for(0.25)

        return True
