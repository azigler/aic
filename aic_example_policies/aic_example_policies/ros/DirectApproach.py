#
# exp-007: DirectApproach v5 (diagnostic + best-of approach)
#
# This version:
# 1. Uses exp-004's parameters (our best: 93.4)
# 2. Adds detailed logging of TCP pose at key moments
# 3. Uses the CheatCode's insight: port is accessible via TF during training
#    but NOT during eval. However, the gripper offset from sample_config tells us
#    the grasp pose relative to plug. Combined with task info, we can infer
#    approximate insertion direction.
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
        self.get_logger().info("=== DirectApproach v5 ===")
        self.get_logger().info(
            f"Task: port={task.port_name} plug={task.plug_name} "
            f"type={task.plug_type} module={task.target_module_name} "
            f"cable={task.cable_name} cable_type={task.cable_type}"
        )

        obs = get_observation()
        if obs is None:
            return False

        tcp = obs.controller_state.tcp_pose
        self.get_logger().info(
            f"DIAG start_tcp: x={tcp.position.x:.5f} y={tcp.position.y:.5f} z={tcp.position.z:.5f}"
        )
        self.get_logger().info(
            f"DIAG start_quat: x={tcp.orientation.x:.5f} y={tcp.orientation.y:.5f} "
            f"z={tcp.orientation.z:.5f} w={tcp.orientation.w:.5f}"
        )

        # Log joint states
        js = obs.joint_states
        if js and js.position:
            self.get_logger().info(
                f"DIAG joints: {[f'{p:.4f}' for p in js.position]}"
            )

        # Log wrench
        if obs.wrist_wrench:
            w = obs.wrist_wrench.wrench
            self.get_logger().info(
                f"DIAG wrench: fx={w.force.x:.2f} fy={w.force.y:.2f} fz={w.force.z:.2f} "
                f"tx={w.torque.x:.2f} ty={w.torque.y:.2f} tz={w.torque.z:.2f}"
            )

        # Baseline force
        forces = []
        for _ in range(10):
            obs = get_observation()
            if obs:
                forces.append(self._get_force(obs))
            self.sleep_for(0.05)
        baseline = sum(forces) / len(forces) if forces else 25.0
        self.get_logger().info(f"DIAG baseline_force: {baseline:.2f}N")

        # Stabilize briefly
        for _ in range(10):
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

        force_limit = 15.0

        # Phase 1: Descent (exp-004 settings: 0.5mm steps, 15cm max)
        stiffness = [80.0, 80.0, 60.0, 40.0, 40.0, 40.0]
        damping = [50.0, 50.0, 40.0, 25.0, 25.0, 25.0]

        step = 0.0005
        max_descent = 0.15
        total = 0.0

        for i in range(int(max_descent / step)):
            obs = get_observation()
            if obs is None:
                continue

            rel_f = self._get_force(obs) - baseline
            if rel_f > force_limit:
                self.sleep_for(0.05)
                continue

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
                    f"Descent: {total * 1000:.1f}mm z={target_z:.4f} f={rel_f:.1f}N"
                )
            self.sleep_for(0.05)

        self.get_logger().info(f"DIAG descent_total: {total * 1000:.1f}mm")

        # Log position after descent
        obs = get_observation()
        if obs:
            p = obs.controller_state.tcp_pose.position
            self.get_logger().info(
                f"DIAG after_descent: x={p.x:.5f} y={p.y:.5f} z={p.z:.5f}"
            )

        # Phase 2: Spiral search
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

        # Phase 3: Compliant push
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
                f"DIAG final_tcp: x={p.x:.5f} y={p.y:.5f} z={p.z:.5f}"
            )

        self.sleep_for(3.0)
        self.get_logger().info("=== DirectApproach v5 complete ===")
        return True
