#
# HybridPolicy: Combine IBVS initial alignment with ACT insertion.
# IBVS handles coarse XY alignment (which ACT struggles with),
# then ACT handles the fine insertion (which IBVS can't do).
#

import json
import math
import os
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
from geometry_msgs.msg import Point, Pose, Twist, Vector3, Wrench
from std_msgs.msg import Header

from aic_example_policies.perception import detect_port

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts")
)
from train_act import SimpleACT


class HybridPolicy(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load ACT model
        model_dir = os.environ.get(
            "ACT_MODEL_DIR", os.path.expanduser("~/models/act_v6_diverse27")
        )
        try:
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

            self.model = SimpleACT(state_dim=26, action_dim=7, chunk_size=50)
            ckpt = torch.load(
                os.path.join(model_dir, "best_model.pt"),
                map_location=self.device,
            )
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()
            self.has_act = True
            self.get_logger().info(f"ACT model loaded from {model_dir}")
        except Exception as ex:
            self.has_act = False
            self.get_logger().warn(
                f"ACT model not loaded: {ex}. Using IBVS only."
            )

    def _get_force(self, obs):
        if obs is None or obs.wrist_wrench is None:
            return 0.0
        w = obs.wrist_wrench.wrench
        return math.sqrt(w.force.x**2 + w.force.y**2 + w.force.z**2)

    def _process_image(self, img_msg, cam_name):
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
        s = self.img_stats[cam_name]
        return (tensor - s["mean"]) / (s["std"] + 1e-8)

    def _extract_state(self, obs):
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
            f"HybridPolicy: port={task.port_name} plug={task.plug_type}"
        )

        obs = get_observation()
        if obs is None:
            return False

        tcp = obs.controller_state.tcp_pose
        ci = obs.center_camera_info
        img_cx = ci.width / 2.0
        img_cy = ci.height / 2.0
        fx, fy = ci.k[0], ci.k[4]
        cx_k, cy_k = ci.k[2], ci.k[5]

        # Baseline force
        forces = []
        for _ in range(5):
            obs = get_observation()
            if obs:
                forces.append(self._get_force(obs))
            self.sleep_for(0.03)
        baseline = sum(forces) / len(forces) if forces else 25.0

        pixel_gain = 0.00004
        current_x = tcp.position.x
        current_y = tcp.position.y
        current_z = tcp.position.z
        orientation = tcp.orientation

        # Phase 1: IBVS alignment (10 samples, averaged correction)
        self.get_logger().info("Phase 1: IBVS alignment")
        send_feedback("IBVS alignment")

        corrections_dx = []
        corrections_dy = []
        frame_counter = 0

        for i in range(10):
            obs = get_observation()
            if obs is None:
                continue
            frame_counter += 1
            det = detect_port(
                obs.center_image,
                task.plug_type,
                ci,
                frame_counter,
                save_debug=(i < 3),
                port_name=task.port_name,
            )
            if det is None:
                continue

            dz = det[2] if abs(det[2]) > 0.01 else 0.18
            px = det[0] * fx / dz + cx_k
            py = det[1] * fy / dz + cy_k
            err_px = px - img_cx
            err_py = py - img_cy
            corrections_dx.append(err_py * pixel_gain * 0.5)
            corrections_dy.append(-err_px * pixel_gain * 0.5)
            self.sleep_for(0.1)

        if corrections_dx:
            avg_dx = sum(corrections_dx) / len(corrections_dx)
            avg_dy = sum(corrections_dy) / len(corrections_dy)
            current_x += avg_dx
            current_y += avg_dy
            self.get_logger().info(
                f"IBVS correction: ({avg_dx:.4f},{avg_dy:.4f})"
            )
            pose = Pose(
                position=Point(x=current_x, y=current_y, z=current_z),
                orientation=orientation,
            )
            self.set_pose_target(move_robot=move_robot, pose=pose)
            self.sleep_for(0.5)

        # Phase 2: ACT insertion (if model loaded)
        if self.has_act:
            self.get_logger().info("Phase 2: ACT insertion")
            send_feedback("ACT insertion")

            start = time.time()
            step = 0
            while time.time() - start < 45.0:
                obs = get_observation()
                if obs is None:
                    continue

                img_l = self._process_image(obs.left_image, "left")
                img_c = self._process_image(obs.center_image, "center")
                img_r = self._process_image(obs.right_image, "right")
                state = self._extract_state(obs)

                with torch.inference_mode():
                    action_norm = self.model(img_l, img_c, img_r, state)

                action = (
                    (action_norm[0] * self.action_std + self.action_mean)
                    .cpu()
                    .numpy()
                )

                motion = MotionUpdate()
                motion.header = Header(
                    frame_id="base_link", stamp=self.get_clock().now().to_msg()
                )
                motion.velocity = Twist(
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
                motion.target_stiffness = np.diag(
                    [100.0, 100.0, 100.0, 50.0, 50.0, 50.0]
                ).flatten()
                motion.target_damping = np.diag(
                    [40.0, 40.0, 40.0, 15.0, 15.0, 15.0]
                ).flatten()
                motion.feedforward_wrench_at_tip = Wrench()
                motion.wrench_feedback_gains_at_tip = [
                    0.5,
                    0.5,
                    0.5,
                    0.0,
                    0.0,
                    0.0,
                ]
                motion.trajectory_generation_mode.mode = (
                    TrajectoryGenerationMode.MODE_VELOCITY
                )
                move_robot(motion_update=motion)

                step += 1
                self.sleep_for(0.25)

        else:
            # Fallback: descent + spiral (from DirectApproach)
            self.get_logger().info("Phase 2: Classical descent (no ACT)")
            force_limit = 15.0
            stiff = [80.0, 80.0, 60.0, 40.0, 40.0, 40.0]
            damp = [50.0, 50.0, 40.0, 25.0, 25.0, 25.0]
            max_d = 0.3 if task.plug_type == "sc" else 0.22

            for _ in range(int(max_d / 0.0005)):
                obs = get_observation()
                if obs is None:
                    continue
                if self._get_force(obs) - baseline > force_limit:
                    self.sleep_for(0.03)
                    continue
                current_z -= 0.0005
                pose = Pose(
                    position=Point(x=current_x, y=current_y, z=current_z),
                    orientation=orientation,
                )
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=pose,
                    stiffness=stiff,
                    damping=damp,
                )
                self.sleep_for(0.03)

        self.sleep_for(2.0)
        return True
