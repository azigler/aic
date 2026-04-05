#
# exp-017: Image-Based Visual Servoing (IBVS) Policy
#
# Strategy: Instead of computing 3D offsets (which require accurate
# camera-to-base transforms), use the camera image directly:
# 1. Detect port in center camera image (color detection)
# 2. Compute pixel error from image center
# 3. Move TCP proportionally to pixel error (small steps)
# 4. Repeat until port is centered in image
# 5. Then descend straight down with compliant control
#
# No 3D transforms needed! The camera is roughly aligned with the
# insertion axis (both point down), so centering in the image means
# aligning the TCP with the port.
#

import math

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose

from aic_example_policies.perception import detect_port


class IBVSPolicy(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)

    def _get_force(self, obs):
        if obs is None or obs.wrist_wrench is None:
            return 0.0
        w = obs.wrist_wrench.wrench
        return math.sqrt(w.force.x**2 + w.force.y**2 + w.force.z**2)

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        self.get_logger().info(
            f"=== IBVSPolicy === port={task.port_name} plug={task.plug_type}"
        )

        obs = get_observation()
        if obs is None:
            return False

        tcp = obs.controller_state.tcp_pose
        self.get_logger().info(
            f"Start TCP: ({tcp.position.x:.4f}, {tcp.position.y:.4f}, {tcp.position.z:.4f})"
        )

        # Baseline force
        forces = []
        for _ in range(5):
            obs = get_observation()
            if obs:
                forces.append(self._get_force(obs))
            self.sleep_for(0.03)
        baseline = sum(forces) / len(forces) if forces else 25.0
        force_limit = 15.0

        # Get camera image center
        obs = get_observation()
        ci = obs.center_camera_info
        img_cx = ci.width / 2.0  # Image center X
        img_cy = ci.height / 2.0  # Image center Y

        # Phase 1: IBVS -- move TCP until port is centered in image
        self.get_logger().info("Phase 1: IBVS centering")
        send_feedback("IBVS: centering port in camera")

        current_x = tcp.position.x
        current_y = tcp.position.y
        current_z = tcp.position.z
        orientation = tcp.orientation

        # Pixel-to-meters gain: how many meters to move per pixel of error
        # At ~18cm distance, 1 pixel ≈ 0.16mm (from FOV and resolution)
        # We want to be conservative, so use a smaller gain
        pixel_gain = 0.00008  # meters per pixel

        stiffness = [90.0, 90.0, 70.0, 50.0, 50.0, 50.0]
        damping = [50.0, 50.0, 40.0, 25.0, 25.0, 25.0]

        frame_counter = 0
        for iteration in range(30):  # Max 30 servo iterations
            obs = get_observation()
            if obs is None:
                self.sleep_for(0.05)
                continue

            frame_counter += 1
            detection = detect_port(
                image_msg=obs.center_image,
                port_type=task.plug_type,
                camera_info=obs.center_camera_info,
                frame_id=frame_counter,
                save_debug=(frame_counter <= 3),
                port_name=task.port_name,
            )

            if detection is None:
                self.get_logger().info(f"IBVS iter {iteration}: no detection")
                self.sleep_for(0.1)
                continue

            # detection returns (dx_cam, dy_cam, dz_cam) in camera frame meters
            # But we want pixel error. Let me use the raw pixel position instead.
            # The detect_port function converts pixels to meters, but for IBVS
            # we want the pixel error from center.
            # Reconstruct pixel position from camera offset:
            fx = ci.k[0]
            fy = ci.k[4]
            cx = ci.k[2]
            cy = ci.k[5]

            # detection = (dx_cam, dy_cam, dz_cam) where dx_cam = (pixel_x - cx) * dz / fx
            # So pixel_x = dx_cam * fx / dz + cx
            dz = detection[2]
            if abs(dz) < 0.01:
                dz = 0.18
            pixel_x = detection[0] * fx / dz + cx
            pixel_y = detection[1] * fy / dz + cy

            # Pixel error from image center
            err_px = pixel_x - img_cx
            err_py = pixel_y - img_cy

            self.get_logger().info(
                f"IBVS iter {iteration}: pixel=({pixel_x:.0f},{pixel_y:.0f}) "
                f"err=({err_px:.0f},{err_py:.0f})"
            )

            # If error is small enough, we're centered
            if abs(err_px) < 15 and abs(err_py) < 15:
                self.get_logger().info("Port centered! Moving to descent.")
                break

            # Move TCP to reduce pixel error
            # From data: err_px=17 means port is right of center -> move TCP right (+Y base)
            # err_py=-36 means port is above center -> move TCP forward (-X base)
            # Gains reduced to prevent overshoot
            move_dx = err_py * pixel_gain * 0.5  # image Y -> base X (converges)
            move_dy = (
                -err_px * pixel_gain * 0.5
            )  # image X -> base -Y (flip X sign)

            current_x += move_dx
            current_y += move_dy

            pose = Pose(
                position=Point(x=current_x, y=current_y, z=current_z),
                orientation=orientation,
            )
            self.set_pose_target(
                move_robot=move_robot,
                pose=pose,
                stiffness=stiffness,
                damping=damping,
            )
            self.sleep_for(0.15)  # Wait for arm to move before next image

        # Phase 2: Descent with spiral search
        self.get_logger().info("Phase 2: Descent + spiral")
        send_feedback("Descending toward port")

        max_descent = 0.3 if task.plug_type == "sc" else 0.22

        target_x = current_x
        target_y = current_y
        target_z = current_z
        step = 0.0005
        total = 0.0

        descent_stiff = [80.0, 80.0, 60.0, 40.0, 40.0, 40.0]
        descent_damp = [50.0, 50.0, 40.0, 25.0, 25.0, 25.0]

        for i in range(int(max_descent / step)):
            obs = get_observation()
            if obs is None:
                continue

            rel_f = self._get_force(obs) - baseline
            if rel_f > force_limit:
                self.sleep_for(0.03)
                continue

            target_z -= step
            total += step

            pose = Pose(
                position=Point(x=target_x, y=target_y, z=target_z),
                orientation=orientation,
            )
            self.set_pose_target(
                move_robot=move_robot,
                pose=pose,
                stiffness=descent_stiff,
                damping=descent_damp,
            )

            if i % 60 == 0:
                self.get_logger().info(
                    f"Descent: {total * 1000:.0f}mm z={target_z:.4f} f={rel_f:.1f}N"
                )
            self.sleep_for(0.03)

        # Phase 3: Spiral + push
        insert_stiff = [40.0, 40.0, 25.0, 25.0, 25.0, 25.0]
        insert_damp = [35.0, 35.0, 25.0, 18.0, 18.0, 18.0]

        spiral_z = target_z
        for ring in range(1, 6):
            radius = ring * 0.003
            points = max(8, ring * 6)
            for p in range(points):
                angle = 2 * math.pi * p / points
                sx = target_x + radius * math.cos(angle)
                sy = target_y + radius * math.sin(angle)
                spiral_z -= 0.0002

                obs = get_observation()
                if obs is None:
                    continue
                rel_f = self._get_force(obs) - baseline
                if rel_f > force_limit:
                    self.sleep_for(0.03)
                    continue

                pose = Pose(
                    position=Point(x=sx, y=sy, z=spiral_z),
                    orientation=orientation,
                )
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=pose,
                    stiffness=insert_stiff,
                    damping=insert_damp,
                )
                self.sleep_for(0.03)

        # Final push
        final_z = spiral_z
        for _ in range(100):
            obs = get_observation()
            if obs is None:
                continue
            rel_f = self._get_force(obs) - baseline
            if rel_f > force_limit:
                self.sleep_for(0.03)
                continue
            final_z -= 0.0004
            pose = Pose(
                position=Point(x=target_x, y=target_y, z=final_z),
                orientation=orientation,
            )
            self.set_pose_target(
                move_robot=move_robot,
                pose=pose,
                stiffness=insert_stiff,
                damping=insert_damp,
            )
            self.sleep_for(0.03)

        obs = get_observation()
        if obs:
            p = obs.controller_state.tcp_pose.position
            self.get_logger().info(f"Final: ({p.x:.4f}, {p.y:.4f}, {p.z:.4f})")

        self.sleep_for(2.0)
        return True
