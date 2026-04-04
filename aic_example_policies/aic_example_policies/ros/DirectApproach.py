#
# exp-003: DirectApproach
#
# Strategy: Instead of computing forward from quaternion, observe that:
# 1. The CheatCode shows: it descends in Z (z_offset goes from +0.2 to -0.015)
# 2. The board is at roughly z=1.14 (from sample_config), robot is above it
# 3. Insertion means pushing DOWN (-Z in base_link frame)
# 4. But we also need XY alignment with the port
#
# New approach: Log the starting TCP pose and slowly push DOWN (-Z) while
# maintaining XY. The robot starts "a few cm" above the port. Unlike BlindPush,
# we handle the baseline force correctly.
#
# Also try small XY search pattern if initial descent doesn't make progress.
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

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        self.get_logger().info(
            f"DirectApproach: port={task.port_name} plug={task.plug_name} "
            f"type={task.plug_type} module={task.target_module_name}"
        )

        # Read initial observation
        obs = get_observation()
        if obs is None:
            return False

        tcp = obs.controller_state.tcp_pose
        self.get_logger().info(
            f"Start TCP: ({tcp.position.x:.4f}, {tcp.position.y:.4f}, {tcp.position.z:.4f})"
        )
        self.get_logger().info(
            f"Start orientation: ({tcp.orientation.x:.4f}, {tcp.orientation.y:.4f}, "
            f"{tcp.orientation.z:.4f}, {tcp.orientation.w:.4f})"
        )

        # Sample baseline force
        forces = []
        for _ in range(20):
            obs = get_observation()
            if obs and obs.wrist_wrench:
                w = obs.wrist_wrench.wrench
                forces.append(
                    math.sqrt(w.force.x**2 + w.force.y**2 + w.force.z**2)
                )
            self.sleep_for(0.05)
        baseline = sum(forces) / len(forces) if forces else 25.0
        self.get_logger().info(f"Force baseline: {baseline:.1f}N")

        # Phase 1: Stabilize (1s)
        send_feedback("Stabilizing...")
        for _ in range(20):
            obs = get_observation()
            if obs:
                self.set_pose_target(
                    move_robot=move_robot, pose=obs.controller_state.tcp_pose
                )
            self.sleep_for(0.05)

        # Phase 2: Descend in -Z (toward the board/port)
        # Use compliant stiffness for safe approach
        send_feedback("Descending toward port...")
        obs = get_observation()
        if obs is None:
            return False

        start_pose = obs.controller_state.tcp_pose
        target_x = start_pose.position.x
        target_y = start_pose.position.y
        target_z = start_pose.position.z

        # Softer stiffness for approach -- compliant in all axes
        stiffness = [70.0, 70.0, 50.0, 40.0, 40.0, 40.0]
        damping = [50.0, 50.0, 40.0, 25.0, 25.0, 25.0]

        step = 0.0003  # 0.3mm per step
        max_descent = 0.08  # 8cm max
        force_limit = 12.0  # 12N above baseline
        total = 0.0

        for i in range(int(max_descent / step)):
            obs = get_observation()
            if obs is None:
                continue

            # Force check (relative to baseline)
            w = obs.wrist_wrench.wrench
            f = math.sqrt(w.force.x**2 + w.force.y**2 + w.force.z**2)
            rel_f = f - baseline

            if rel_f > force_limit:
                # Contact detected -- hold and try smaller step
                self.sleep_for(0.1)
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

            if i % 50 == 0:
                self.get_logger().info(
                    f"Descent: {total * 1000:.1f}mm, z={target_z:.4f}, "
                    f"force={f:.1f}N (rel={rel_f:.1f}N)"
                )
            self.sleep_for(0.05)

        # Phase 3: Hold for insertion detection
        send_feedback("Holding at depth...")
        self.get_logger().info(f"Descended {total * 1000:.1f}mm. Holding 5s.")
        self.sleep_for(5.0)

        return True
