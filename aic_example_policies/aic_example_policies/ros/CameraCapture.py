#
# CameraCapture: Save camera images to disk for analysis
# Run this as a policy, capture images at start of each trial,
# then return so the trial completes quickly.
#

import os

import numpy as np
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task


class CameraCapture(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._trial_count = 0

    def _save_image(self, img_msg, name):
        """Save a sensor_msgs/Image to a PNG file."""
        try:
            # Convert ROS Image to numpy array
            h, w = img_msg.height, img_msg.width
            if img_msg.encoding == "rgb8":
                channels = 3
            elif img_msg.encoding == "rgba8":
                channels = 4
            elif img_msg.encoding == "bgr8":
                channels = 3
            else:
                self.get_logger().warn(f"Unknown encoding: {img_msg.encoding}")
                channels = 3

            arr = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
                h, w, channels
            )

            # Save as raw numpy (no PIL dependency needed)
            out_dir = os.path.expanduser("~/aic_results/camera_captures")
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"{name}.npy")
            np.save(path, arr)

            # Also save as PPM (simple image format, no dependencies)
            ppm_path = os.path.join(out_dir, f"{name}.ppm")
            with open(ppm_path, "wb") as f:
                f.write(f"P6\n{w} {h}\n255\n".encode())
                if channels == 3:
                    f.write(arr.tobytes())
                elif channels == 4:
                    f.write(arr[:, :, :3].tobytes())

            self.get_logger().info(
                f"Saved {name}: {w}x{h} {img_msg.encoding} -> {ppm_path}"
            )
        except Exception as ex:
            self.get_logger().error(f"Failed to save {name}: {ex}")

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        self._trial_count += 1
        self.get_logger().info(
            f"CameraCapture trial {self._trial_count}: "
            f"port={task.port_name} plug={task.plug_name}"
        )

        # Wait a moment for cameras to stabilize
        self.sleep_for(1.0)

        obs = get_observation()
        if obs is None:
            self.get_logger().error("No observation")
            return False

        prefix = f"trial{self._trial_count}"

        # Log TCP pose
        tcp = obs.controller_state.tcp_pose
        self.get_logger().info(
            f"TCP: ({tcp.position.x:.4f}, {tcp.position.y:.4f}, {tcp.position.z:.4f})"
        )

        # Save all 3 camera images
        self._save_image(obs.left_image, f"{prefix}_left")
        self._save_image(obs.center_image, f"{prefix}_center")
        self._save_image(obs.right_image, f"{prefix}_right")

        # Log camera info for calibration
        ci = obs.center_camera_info
        self.get_logger().info(
            f"Center camera: {ci.width}x{ci.height}, "
            f"fx={ci.k[0]:.1f} fy={ci.k[4]:.1f} cx={ci.k[2]:.1f} cy={ci.k[5]:.1f}"
        )

        # Quickly return so trial completes fast
        self.sleep_for(2.0)
        return True
