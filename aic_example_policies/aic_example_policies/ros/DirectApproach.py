#
# exp-004: DirectApproach v2
#
# Improvements over v1:
# 1. Increase max descent from 8cm to 15cm
# 2. After initial descent, wiggle in XY to find the port (spiral search)
# 3. Lower force limit to detect contact earlier -> indicates we're near the port
# 4. If force contact detected, switch to very compliant mode and push gently
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


class DirectApproach(Policy):
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
            f"DirectApproach v2: port={task.port_name} plug={task.plug_name} "
            f"type={task.plug_type} module={task.target_module_name}"
        )

        obs = get_observation()
        if obs is None:
            return False

        tcp = obs.controller_state.tcp_pose
        self.get_logger().info(
            f"Start TCP: ({tcp.position.x:.4f}, {tcp.position.y:.4f}, {tcp.position.z:.4f})"
        )

        # Sample baseline force
        forces = []
        for _ in range(20):
            obs = get_observation()
            if obs:
                forces.append(self._get_force(obs))
            self.sleep_for(0.05)
        baseline = sum(forces) / len(forces) if forces else 25.0
        self.get_logger().info(f"Force baseline: {baseline:.1f}N")

        # Stabilize
        for _ in range(20):
            obs = get_observation()
            if obs:
                self.set_pose_target(
                    move_robot=move_robot, pose=obs.controller_state.tcp_pose
                )
            self.sleep_for(0.05)

        obs = get_observation()
        start_pose = obs.controller_state.tcp_pose
        target_x = start_pose.position.x
        target_y = start_pose.position.y
        target_z = start_pose.position.z

        # Phase 1: Descend in -Z
        send_feedback("Phase 1: Descending...")
        step = 0.0005  # 0.5mm per step
        max_descent = 0.15  # 15cm
        contact_threshold = 5.0  # 5N above baseline = contact
        force_limit = 15.0  # 15N above baseline = stop
        total = 0.0
        contact_detected = False

        stiffness = [80.0, 80.0, 60.0, 40.0, 40.0, 40.0]
        damping = [50.0, 50.0, 40.0, 25.0, 25.0, 25.0]

        for i in range(int(max_descent / step)):
            obs = get_observation()
            if obs is None:
                continue

            rel_f = self._get_force(obs) - baseline

            if rel_f > force_limit:
                self.sleep_for(0.1)
                continue

            if rel_f > contact_threshold and not contact_detected:
                contact_detected = True
                self.get_logger().info(
                    f"Contact at {total * 1000:.1f}mm descent, "
                    f"force={rel_f:.1f}N above baseline"
                )

            target_z -= step
            total += step

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
                    f"Descent: {total * 1000:.1f}mm, z={target_z:.4f}, rel_f={rel_f:.1f}N"
                )
            self.sleep_for(0.05)

        self.get_logger().info(f"Phase 1 done: descended {total * 1000:.1f}mm")

        # Phase 2: Spiral search in XY at current Z
        # This helps if we're close but slightly offset from the port
        send_feedback("Phase 2: Spiral search...")
        spiral_stiffness = [50.0, 50.0, 40.0, 30.0, 30.0, 30.0]
        spiral_damping = [40.0, 40.0, 35.0, 20.0, 20.0, 20.0]

        center_x = target_x
        center_y = target_y
        spiral_z = target_z

        for ring in range(1, 6):  # 5 rings of increasing radius
            radius = ring * 0.002  # 2mm per ring
            points = max(8, ring * 8)  # More points for larger rings

            for p in range(points):
                angle = 2 * math.pi * p / points
                sx = center_x + radius * math.cos(angle)
                sy = center_y + radius * math.sin(angle)

                # Also descend slightly during spiral
                spiral_z -= 0.0001  # 0.1mm per point

                obs = get_observation()
                if obs is None:
                    continue

                rel_f = self._get_force(obs) - baseline

                if rel_f > force_limit:
                    self.sleep_for(0.05)
                    continue

                pose = Pose(
                    position=Point(x=sx, y=sy, z=spiral_z),
                    orientation=start_pose.orientation,
                )
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=pose,
                    stiffness=spiral_stiffness,
                    damping=spiral_damping,
                )
                self.sleep_for(0.04)

            self.get_logger().info(
                f"Spiral ring {ring}: radius={radius * 1000:.1f}mm, z={spiral_z:.4f}"
            )

        # Phase 3: Final push down with very compliant control
        send_feedback("Phase 3: Insertion push...")
        insert_stiffness = [30.0, 30.0, 20.0, 20.0, 20.0, 20.0]
        insert_damping = [30.0, 30.0, 25.0, 15.0, 15.0, 15.0]

        final_z = spiral_z
        for _ in range(100):
            obs = get_observation()
            if obs is None:
                continue

            rel_f = self._get_force(obs) - baseline
            if rel_f > force_limit:
                self.sleep_for(0.05)
                continue

            final_z -= 0.0003
            pose = Pose(
                position=Point(x=center_x, y=center_y, z=final_z),
                orientation=start_pose.orientation,
            )
            self.set_pose_target(
                move_robot=move_robot,
                pose=pose,
                stiffness=insert_stiffness,
                damping=insert_damping,
            )
            self.sleep_for(0.05)

        # Hold
        send_feedback("Holding...")
        self.get_logger().info(f"Final z={final_z:.4f}. Holding 5s.")
        self.sleep_for(5.0)

        return True
