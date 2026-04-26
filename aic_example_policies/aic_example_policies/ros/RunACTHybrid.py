#
#  RunACTHybrid - ACT model for approach + constant descent for insertion.
#
#  Uses the fine-tuned ACT model for the first APPROACH_TIME seconds,
#  then switches to a constant downward velocity to push through insertion.
#

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import draccus
import numpy as np
from aic_control_interfaces.msg import (
    MotionUpdate,
    TrajectoryGenerationMode,
)
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Twist, Vector3, Wrench
from rclpy.duration import Duration
from rclpy.node import Node

# NOTE: torch, lerobot, safetensors are deferred into __init__ (which
# runs inside on_configure's 60s budget) so module discovery on the portal
# stays under the 30s model_discovery_timeout. See bd-crw / upstream PR #500.
if TYPE_CHECKING:
    pass


class RunACTHybrid(Policy):
    def __init__(self, parent_node: Node):
        super().__init__(parent_node)

        # Heavy imports: deferred from module top to keep module-load fast
        # for portal discovery (30s budget). on_configure has its own 60s
        # budget which is where __init__ runs.
        import torch
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from safetensors.torch import load_file

        # Stash for use in methods.
        self._torch = torch
        self._load_file = load_file

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        model_path = os.environ.get("MODEL_PATH", "")
        if model_path:
            policy_path = Path(model_path)
        else:
            from huggingface_hub import snapshot_download

            policy_path = Path(
                snapshot_download(
                    repo_id="grkw/aic_act_policy",
                    allow_patterns=[
                        "config.json",
                        "model.safetensors",
                        "*.safetensors",
                    ],
                )
            )

        with open(policy_path / "config.json") as f:
            config_dict = json.load(f)
            if "type" in config_dict:
                del config_dict["type"]

        config = draccus.decode(ACTConfig, config_dict)
        self.policy = ACTPolicy(config)
        self.policy.load_state_dict(
            load_file(policy_path / "model.safetensors")
        )
        self.policy.eval()
        self.policy.to(self.device)

        stats_path = (
            policy_path
            / "policy_preprocessor_step_3_normalizer_processor.safetensors"
        )
        stats = load_file(stats_path)

        def get_stat(key, shape):
            return stats[key].to(self.device).view(*shape)

        self.img_stats = {
            "left": {
                "mean": get_stat(
                    "observation.images.left_camera.mean", (1, 3, 1, 1)
                ),
                "std": get_stat(
                    "observation.images.left_camera.std", (1, 3, 1, 1)
                ),
            },
            "center": {
                "mean": get_stat(
                    "observation.images.center_camera.mean", (1, 3, 1, 1)
                ),
                "std": get_stat(
                    "observation.images.center_camera.std", (1, 3, 1, 1)
                ),
            },
            "right": {
                "mean": get_stat(
                    "observation.images.right_camera.mean", (1, 3, 1, 1)
                ),
                "std": get_stat(
                    "observation.images.right_camera.std", (1, 3, 1, 1)
                ),
            },
        }

        self.state_mean = get_stat("observation.state.mean", (1, -1))
        self.state_std = get_stat("observation.state.std", (1, -1))
        self.action_mean = get_stat("action.mean", (1, -1))
        self.action_std = get_stat("action.std", (1, -1))

        self.image_scaling = 0.25

        # Hybrid parameters
        self.approach_time = float(os.environ.get("APPROACH_TIME", "25"))
        self.descent_speed = float(os.environ.get("DESCENT_SPEED", "0.01"))
        self.total_time = float(os.environ.get("TIME_LIMIT", "45"))

        self.get_logger().info(
            f"RunACTHybrid loaded. approach={self.approach_time}s, "
            f"descent_speed={self.descent_speed}, total={self.total_time}s"
        )

    def _img_to_tensor(self, raw_img, device, scale, mean, std):
        img_np = np.frombuffer(raw_img.data, dtype=np.uint8).reshape(
            raw_img.height, raw_img.width, 3
        )
        if scale != 1.0:
            img_np = cv2.resize(
                img_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )
        tensor = (
            self._torch.from_numpy(img_np)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(device)
        )
        return (tensor - mean) / std

    def prepare_observations(self, obs_msg: Observation):
        obs = {
            "observation.images.left_camera": self._img_to_tensor(
                obs_msg.left_image,
                self.device,
                self.image_scaling,
                self.img_stats["left"]["mean"],
                self.img_stats["left"]["std"],
            ),
            "observation.images.center_camera": self._img_to_tensor(
                obs_msg.center_image,
                self.device,
                self.image_scaling,
                self.img_stats["center"]["mean"],
                self.img_stats["center"]["std"],
            ),
            "observation.images.right_camera": self._img_to_tensor(
                obs_msg.right_image,
                self.device,
                self.image_scaling,
                self.img_stats["right"]["mean"],
                self.img_stats["right"]["std"],
            ),
        }

        tcp_pose = obs_msg.controller_state.tcp_pose
        tcp_vel = obs_msg.controller_state.tcp_velocity

        state_np = np.array(
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
                *obs_msg.controller_state.tcp_error,
                *obs_msg.joint_states.position[:7],
            ],
            dtype=np.float32,
        )

        raw_state_tensor = (
            self._torch.from_numpy(state_np)
            .float()
            .unsqueeze(0)
            .to(self.device)
        )
        obs["observation.state"] = (
            raw_state_tensor - self.state_mean
        ) / self.state_std

        return obs

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        **kwargs,
    ):
        self.policy.reset()
        self.get_logger().info(
            f"RunACTHybrid.insert_cable() enter. Task: {task}"
        )

        # Sim-clock-aware deadline + pacing. The portal enforces
        # task.time_limit on the ROS clock (sim time), so we must use
        # Node.get_clock() rather than time.time(). See bd-spw / upstream
        # PR #500 / docs/troubleshooting.md.
        clock = self.get_clock()
        start_time = clock.now()

        # Prefer task.time_limit (uint64 seconds) when set; fall back to
        # the env-configurable self.total_time we recorded in __init__.
        if task is not None and getattr(task, "time_limit", 0):
            time_limit = float(task.time_limit)
        else:
            time_limit = self.total_time
        deadline = Duration(seconds=time_limit)
        period = Duration(seconds=0.25)  # 4 Hz control rate

        while (clock.now() - start_time) < deadline:
            loop_start = clock.now()
            # Sim-time elapsed in seconds for phase switching.
            elapsed = (loop_start - start_time).nanoseconds * 1e-9

            observation_msg = get_observation()
            if observation_msg is None:
                continue

            if elapsed < self.approach_time:
                # Phase 1: ACT model approach
                obs_tensors = self.prepare_observations(observation_msg)
                with self._torch.inference_mode():
                    normalized_action = self.policy.select_action(obs_tensors)
                raw_action = (
                    normalized_action * self.action_std
                ) + self.action_mean
                action = raw_action[0].cpu().numpy()
                phase = "ACT"
            else:
                # Phase 2: Constant descent velocity
                # Keep a small downward (-z) velocity to push into the port
                action = np.array(
                    [0.0, 0.0, -self.descent_speed, 0.0, 0.0, 0.0, 0.012]
                )
                phase = "DESCENT"

            self.get_logger().info(
                f"[{phase} t={elapsed:.1f}s] Action: {action[:6]}"
            )

            twist = Twist(
                linear=Vector3(
                    x=float(action[0]),
                    y=float(action[1]),
                    z=float(action[2]),
                ),
                angular=Vector3(
                    x=float(action[3]),
                    y=float(action[4]),
                    z=float(action[5]),
                ),
            )
            motion_update = self._make_motion_update(twist)
            move_robot(motion_update=motion_update)
            send_feedback(f"{phase} t={elapsed:.1f}s")

            # Maintain 4 Hz control rate using the ROS clock so we honor
            # sim time on the portal.
            loop_elapsed = clock.now() - loop_start
            remaining_ns = period.nanoseconds - loop_elapsed.nanoseconds
            if remaining_ns > 0:
                clock.sleep_for(Duration(nanoseconds=remaining_ns))

        self.get_logger().info("RunACTHybrid.insert_cable() exiting...")
        return True

    def _make_motion_update(self, twist, frame_id="base_link"):
        msg = MotionUpdate()
        msg.velocity = twist
        msg.header.frame_id = frame_id
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.target_stiffness = np.diag(
            [100.0, 100.0, 100.0, 50.0, 50.0, 50.0]
        ).flatten()
        msg.target_damping = np.diag(
            [40.0, 40.0, 40.0, 15.0, 15.0, 15.0]
        ).flatten()

        msg.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )
        msg.wrench_feedback_gains_at_tip = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
        msg.trajectory_generation_mode.mode = (
            TrajectoryGenerationMode.MODE_VELOCITY
        )

        return msg
