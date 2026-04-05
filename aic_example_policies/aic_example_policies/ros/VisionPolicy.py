"""Vision-guided cable insertion policy.

Uses camera-based port detection for visual servoing, then compliant
insertion. Falls back to DirectApproach-style blind descent if the
camera cannot detect the port.
"""

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


class VisionPolicy(Policy):
    """Camera-guided insertion with DirectApproach fallback."""

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._frame_counter = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_force(self, obs) -> float:
        if obs is None or obs.wrist_wrench is None:
            return 0.0
        w = obs.wrist_wrench.wrench
        return math.sqrt(w.force.x**2 + w.force.y**2 + w.force.z**2)

    def _measure_baseline_force(self, get_observation) -> float:
        """Sample force over 0.5s to get baseline (cable weight ~21N)."""
        forces = []
        for _ in range(10):
            obs = get_observation()
            if obs:
                forces.append(self._get_force(obs))
            self.sleep_for(0.05)
        return sum(forces) / len(forces) if forces else 25.0

    def _stabilize(self, get_observation, move_robot, steps: int = 10) -> None:
        """Hold current pose briefly to let the arm settle."""
        for _ in range(steps):
            obs = get_observation()
            if obs:
                self.set_pose_target(
                    move_robot=move_robot, pose=obs.controller_state.tcp_pose
                )
            self.sleep_for(0.05)

    def _detect_port_from_obs(
        self, obs, port_type: str, port_name: str = "sfp_port_0"
    ):
        """Run port detection on the center camera image.

        Returns (dx, dy, dz) in camera optical frame or None.
        """
        if obs is None:
            return None

        self._frame_counter += 1
        result = detect_port(
            image_msg=obs.center_image,
            port_type=port_type,
            camera_info=obs.center_camera_info,
            frame_id=self._frame_counter,
            save_debug=(self._frame_counter % 10 == 1),
            port_name=port_name,
        )
        return result

    def _camera_offset_to_base_xy(self, cam_offset, tcp_pose):
        """Convert camera-frame XY offset to approximate base-frame XY offset.

        The center camera optical frame has X=right, Y=down, Z=forward.
        When the gripper is pointing straight down (nominal orientation),
        camera X maps roughly to base -Y, camera Y maps roughly to base -X,
        and camera Z maps roughly to base -Z.

        This is a simplified transform that works for the nominal downward
        orientation used during insertion. A full quaternion rotation would
        be needed for arbitrary poses.
        """
        # Camera optical frame -> base frame (approximate for downward pose)
        # camera X (right) ~ base -Y
        # camera Y (down)  ~ base -X
        dx_base = -cam_offset[1]  # camera Y -> base -X
        dy_base = -cam_offset[0]  # camera X -> base -Y
        return dx_base, dy_base

    # ------------------------------------------------------------------
    # Main policy
    # ------------------------------------------------------------------

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        self.get_logger().info("=== VisionPolicy v1 ===")
        self.get_logger().info(
            f"Task: port={task.port_name} plug={task.plug_name} "
            f"type={task.plug_type} module={task.target_module_name}"
        )

        obs = get_observation()
        if obs is None:
            return False

        tcp = obs.controller_state.tcp_pose
        self.get_logger().info(
            f"Start TCP: x={tcp.position.x:.5f} y={tcp.position.y:.5f} "
            f"z={tcp.position.z:.5f}"
        )

        # Baseline force measurement
        baseline = self._measure_baseline_force(get_observation)
        self.get_logger().info(f"Baseline force: {baseline:.2f}N")

        # Stabilize
        self._stabilize(get_observation, move_robot)

        obs = get_observation()
        start_pose = obs.controller_state.tcp_pose
        force_limit = 15.0

        # ------------------------------------------------------------------
        # Phase 1: Camera-based XY alignment
        # ------------------------------------------------------------------
        self.get_logger().info("Phase 1: Camera-based port detection")

        # Determine fallback offsets (from DirectApproach, used if vision fails)
        if task.plug_type == "sc":
            fallback_dx = -0.10
            fallback_dy = 0.08
            max_descent = 0.30
        else:
            fallback_dx = -0.01
            fallback_dy = 0.02
            max_descent = 0.22

        # Try to detect port from current position
        detection_count = 0
        accumulated_dx = 0.0
        accumulated_dy = 0.0

        # Sample multiple frames for robustness
        for i in range(5):
            obs = get_observation()
            if obs is None:
                self.sleep_for(0.1)
                continue

            cam_offset = self._detect_port_from_obs(
                obs, task.plug_type, task.port_name
            )
            if cam_offset is not None:
                dx_base, dy_base = self._camera_offset_to_base_xy(
                    cam_offset, obs.controller_state.tcp_pose
                )
                accumulated_dx += dx_base
                accumulated_dy += dy_base
                detection_count += 1
                self.get_logger().info(
                    f"Detection {i}: cam=({cam_offset[0]:.4f}, {cam_offset[1]:.4f}, "
                    f"{cam_offset[2]:.4f}) -> base_xy=({dx_base:.4f}, {dy_base:.4f})"
                )
            else:
                self.get_logger().info(f"Detection {i}: no port found")
            self.sleep_for(0.1)

        # Decide XY offset
        if detection_count >= 2:
            # Use averaged camera detections
            avg_dx = accumulated_dx / detection_count
            avg_dy = accumulated_dy / detection_count
            self.get_logger().info(
                f"Vision alignment: dx={avg_dx:.4f} dy={avg_dy:.4f} "
                f"({detection_count} detections)"
            )
            target_x = start_pose.position.x + avg_dx
            target_y = start_pose.position.y + avg_dy
        else:
            # Fall back to hardcoded offsets
            self.get_logger().info(
                f"Vision fallback: using hardcoded offsets "
                f"dx={fallback_dx} dy={fallback_dy}"
            )
            target_x = start_pose.position.x + fallback_dx
            target_y = start_pose.position.y + fallback_dy

        target_z = start_pose.position.z

        # ------------------------------------------------------------------
        # Phase 2: Descent with visual servo refinement
        # ------------------------------------------------------------------
        self.get_logger().info("Phase 2: Descent with visual refinement")

        stiffness = [80.0, 80.0, 60.0, 40.0, 40.0, 40.0]
        damping = [50.0, 50.0, 40.0, 25.0, 25.0, 25.0]
        step = 0.0005
        total_descent = 0.0
        servo_interval = 40  # Re-detect every 40 steps (~2s)

        for i in range(int(max_descent / step)):
            obs = get_observation()
            if obs is None:
                continue

            rel_f = self._get_force(obs) - baseline
            if rel_f > force_limit:
                self.sleep_for(0.05)
                continue

            # Periodic visual servo refinement during descent
            if (
                i > 0
                and i % servo_interval == 0
                and total_descent < max_descent * 0.7
            ):
                cam_offset = self._detect_port_from_obs(
                    obs, task.plug_type, task.port_name
                )
                if cam_offset is not None:
                    dx_base, dy_base = self._camera_offset_to_base_xy(
                        cam_offset, obs.controller_state.tcp_pose
                    )
                    # Apply a fraction of the correction (damped servo)
                    correction_gain = 0.3
                    target_x += dx_base * correction_gain
                    target_y += dy_base * correction_gain
                    self.get_logger().info(
                        f"Servo correction at {total_descent * 1000:.0f}mm: "
                        f"dx={dx_base * correction_gain:.4f} "
                        f"dy={dy_base * correction_gain:.4f}"
                    )

            target_z -= step
            total_descent += step

            pose = Pose(
                position=Point(x=target_x, y=target_y, z=target_z),
                orientation=start_pose.orientation,
            )
            self.set_pose_target(
                move_robot=move_robot,
                pose=pose,
                stiffness=stiffness,
                damping=damping,
            )

            if i % 60 == 0:
                self.get_logger().info(
                    f"Descent: {total_descent * 1000:.1f}mm z={target_z:.4f} "
                    f"f={rel_f:.1f}N"
                )
            self.sleep_for(0.05)

        self.get_logger().info(
            f"Descent complete: {total_descent * 1000:.1f}mm"
        )

        # ------------------------------------------------------------------
        # Phase 3: Spiral search (same as DirectApproach)
        # ------------------------------------------------------------------
        self.get_logger().info("Phase 3: Spiral search")

        spiral_stiff = [50.0, 50.0, 40.0, 30.0, 30.0, 30.0]
        spiral_damp = [40.0, 40.0, 35.0, 20.0, 20.0, 20.0]
        center_x = target_x
        center_y = target_y
        spiral_z = target_z

        for ring in range(1, 6):
            radius = ring * 0.002
            points = max(8, ring * 8)
            for p in range(points):
                angle = 2 * math.pi * p / points
                sx = center_x + radius * math.cos(angle)
                sy = center_y + radius * math.sin(angle)
                spiral_z -= 0.0001

                obs = get_observation()
                if obs is None:
                    continue
                rel_f = self._get_force(obs) - baseline
                if rel_f > force_limit:
                    self.sleep_for(0.04)
                    continue

                pose = Pose(
                    position=Point(x=sx, y=sy, z=spiral_z),
                    orientation=start_pose.orientation,
                )
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=pose,
                    stiffness=spiral_stiff,
                    damping=spiral_damp,
                )
                self.sleep_for(0.04)

        # ------------------------------------------------------------------
        # Phase 4: Compliant push (same as DirectApproach)
        # ------------------------------------------------------------------
        self.get_logger().info("Phase 4: Compliant push")

        insert_stiff = [40.0, 40.0, 25.0, 20.0, 20.0, 20.0]
        insert_damp = [35.0, 35.0, 25.0, 15.0, 15.0, 15.0]
        final_z = spiral_z

        for _ in range(200):
            obs = get_observation()
            if obs is None:
                continue
            rel_f = self._get_force(obs) - baseline
            if rel_f > force_limit:
                self.sleep_for(0.04)
                continue

            final_z -= 0.0003
            pose = Pose(
                position=Point(x=center_x, y=center_y, z=final_z),
                orientation=start_pose.orientation,
            )
            self.set_pose_target(
                move_robot=move_robot,
                pose=pose,
                stiffness=insert_stiff,
                damping=insert_damp,
            )
            self.sleep_for(0.04)

        # Log final position
        obs = get_observation()
        if obs:
            p = obs.controller_state.tcp_pose.position
            self.get_logger().info(
                f"Final TCP: x={p.x:.5f} y={p.y:.5f} z={p.z:.5f}"
            )

        self.sleep_for(3.0)
        self.get_logger().info("=== VisionPolicy v1 complete ===")
        return True
