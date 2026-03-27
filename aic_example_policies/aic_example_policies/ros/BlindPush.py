#
# exp-001: BlindPush policy
#
# Strategy: The robot starts a few cm from the port with the plug in hand.
# Without ground truth, we don't know exactly where the port is, but we
# know the general direction is "forward" from current TCP position.
#
# This policy:
# 1. Reads current TCP pose from observation
# 2. Pushes gently forward (negative Z in TCP frame = toward board)
# 3. Uses low stiffness for compliance during contact
# 4. Monitors force to avoid penalties (>20N for >1s = -12 pts)
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


class BlindPush(Policy):
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
            f"BlindPush.insert_cable() task: port={task.port_name} "
            f"plug={task.plug_name} module={task.target_module_name}"
        )

        # Get initial observation to find our current pose
        obs = get_observation()
        if obs is None:
            self.get_logger().error("No observation available")
            return False

        tcp_pose = obs.controller_state.tcp_pose
        self.get_logger().info(
            f"Initial TCP pose: ({tcp_pose.position.x:.4f}, "
            f"{tcp_pose.position.y:.4f}, {tcp_pose.position.z:.4f})"
        )

        # Phase 1: Hold position for 1 second to stabilize
        send_feedback("Phase 1: Stabilizing...")
        for _ in range(20):
            obs = get_observation()
            if obs is None:
                continue
            self.set_pose_target(
                move_robot=move_robot,
                pose=obs.controller_state.tcp_pose,
            )
            self.sleep_for(0.05)

        # Phase 2: Slowly push forward (descend in Z)
        # The port is roughly "below" / "in front of" the current position.
        # In base_link frame, the insertion direction depends on the grasp.
        # Let's try descending in Z (toward the board) in small increments.
        send_feedback("Phase 2: Pushing toward port...")
        obs = get_observation()
        if obs is None:
            return False

        start_pose = obs.controller_state.tcp_pose
        target_x = start_pose.position.x
        target_y = start_pose.position.y
        target_z = start_pose.position.z

        # Compliant stiffness for insertion
        insertion_stiffness = [50.0, 50.0, 30.0, 30.0, 30.0, 30.0]
        insertion_damping = [40.0, 40.0, 30.0, 20.0, 20.0, 20.0]

        total_descent = 0.0
        max_descent = 0.10  # 10cm max -- we start a few cm away
        step_size = 0.0005  # 0.5mm per step
        max_force = 15.0  # Back off before the 20N penalty threshold

        for i in range(int(max_descent / step_size)):
            obs = get_observation()
            if obs is None:
                continue

            # Check force -- avoid the 20N penalty
            wrench = obs.wrist_wrench.wrench
            force_magnitude = math.sqrt(
                wrench.force.x**2 + wrench.force.y**2 + wrench.force.z**2
            )

            if force_magnitude > max_force:
                self.get_logger().warn(
                    f"Force {force_magnitude:.1f}N > {max_force}N threshold, pausing descent"
                )
                # Hold position, don't push further
                self.sleep_for(0.1)
                continue

            total_descent += step_size
            target_z -= step_size

            pose = Pose(
                position=Point(x=target_x, y=target_y, z=target_z),
                orientation=start_pose.orientation,
            )

            self.set_pose_target(
                move_robot=move_robot,
                pose=pose,
                stiffness=insertion_stiffness,
                damping=insertion_damping,
            )

            if i % 40 == 0:
                self.get_logger().info(
                    f"Descent: {total_descent * 1000:.1f}mm, "
                    f"force: {force_magnitude:.1f}N, "
                    f"z: {target_z:.4f}"
                )

            self.sleep_for(0.05)

        # Phase 3: Hold final position
        send_feedback("Phase 3: Holding position...")
        self.get_logger().info(
            f"Total descent: {total_descent * 1000:.1f}mm. Holding for 3s..."
        )
        self.sleep_for(3.0)

        self.get_logger().info("BlindPush.insert_cable() complete")
        return True
