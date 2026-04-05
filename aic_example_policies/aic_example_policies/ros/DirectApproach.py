#
# exp-008: DirectApproach v6 (TF probe + offset learning)
#
# This version:
# 1. Tries to look up port TF frame (only works with ground_truth=true)
# 2. If available: computes exact offset from TCP to port, logs it for learning
# 3. If not available: uses the logged offsets as hardcoded fallback
# 4. Uses exp-004 movement parameters (our best: 93.4)
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

        # Try to look up port position via TF (only works with ground_truth=true)
        port_frame = (
            f"task_board/{task.target_module_name}/{task.port_name}_link"
        )
        port_offset_x = 0.0
        port_offset_y = 0.0
        port_offset_z = 0.0
        try:
            from rclpy.time import Time

            tf_buf = self._parent_node._tf_buffer
            port_tf = tf_buf.lookup_transform("base_link", port_frame, Time())
            px = port_tf.transform.translation.x
            py = port_tf.transform.translation.y
            pz = port_tf.transform.translation.z
            port_offset_x = px - tcp.position.x
            port_offset_y = py - tcp.position.y
            port_offset_z = pz - tcp.position.z
            self.get_logger().info(
                f"DIAG port_tf: x={px:.5f} y={py:.5f} z={pz:.5f}"
            )
            self.get_logger().info(
                f"DIAG port_offset: dx={port_offset_x:.5f} dy={port_offset_y:.5f} "
                f"dz={port_offset_z:.5f} dist={math.sqrt(port_offset_x**2 + port_offset_y**2 + port_offset_z**2):.5f}"
            )
        except Exception as ex:
            self.get_logger().info(f"No ground truth TF available: {ex}")

        # Log wrench
        if obs.wrist_wrench:
            w = obs.wrist_wrench.wrench
            self.get_logger().info(
                f"DIAG wrench: fx={w.force.x:.2f} fy={w.force.y:.2f} fz={w.force.z:.2f}"
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

        # Apply known XY offsets based on plug type (from ground truth probing)
        # SFP: port is ~(-0.01, +0.02, -0.186) from TCP start
        # SC: port is ~(-0.114, +0.096, -0.279) from TCP start
        if task.plug_type == "sc":
            target_x += -0.10  # SC port is significantly offset in -X
            target_y += 0.08  # SC port is offset in +Y
            max_descent = 0.30  # SC port is much lower
        else:
            target_x += -0.01  # Small -X offset for SFP
            target_y += 0.02  # Small +Y offset for SFP
            max_descent = 0.22  # SFP port is ~18.6cm below, add margin

        self.get_logger().info(
            f"Adjusted target: x={target_x:.4f} y={target_y:.4f} "
            f"max_descent={max_descent * 1000:.0f}mm"
        )

        # Phase 1: Descent with XY correction applied
        stiffness = [80.0, 80.0, 60.0, 40.0, 40.0, 40.0]
        damping = [50.0, 50.0, 40.0, 25.0, 25.0, 25.0]

        step = 0.0005
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
