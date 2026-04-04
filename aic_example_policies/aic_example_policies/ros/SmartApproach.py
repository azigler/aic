#
# exp-002: SmartApproach policy
#
# Strategy: Read initial TCP pose, compute the gripper's forward axis from
# the quaternion orientation, and push along that axis toward the board.
# The robot starts a few cm from the port with the plug oriented toward it,
# so "forward" in the gripper frame should be approximately the insertion direction.
#
# Also: tare the force readings by sampling baseline, then use relative force
# for safety monitoring.
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


def quat_to_forward(q):
    """Extract the forward direction (local -Z axis) from a quaternion.

    The gripper TCP frame has -Z pointing "forward" (toward insertion).
    Given quaternion (x, y, z, w), compute the world-frame direction
    of the local -Z axis.
    """
    x, y, z, w = q.x, q.y, q.z, q.w
    # Rotation matrix third column (Z axis in world frame)
    fwd_x = 2 * (x * z + w * y)
    fwd_y = 2 * (y * z - w * x)
    fwd_z = 1 - 2 * (x * x + y * y)
    # Negate for -Z (insertion direction)
    return -fwd_x, -fwd_y, -fwd_z


class SmartApproach(Policy):
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
            f"SmartApproach: port={task.port_name} plug={task.plug_name} "
            f"module={task.target_module_name}"
        )

        # Phase 0: Sample initial state
        obs = get_observation()
        if obs is None:
            self.get_logger().error("No observation")
            return False

        tcp = obs.controller_state.tcp_pose
        start_x = tcp.position.x
        start_y = tcp.position.y
        start_z = tcp.position.z

        # Get gripper forward direction (insertion axis)
        fwd_x, fwd_y, fwd_z = quat_to_forward(tcp.orientation)
        fwd_len = math.sqrt(fwd_x**2 + fwd_y**2 + fwd_z**2)
        if fwd_len > 0.001:
            fwd_x /= fwd_len
            fwd_y /= fwd_len
            fwd_z /= fwd_len

        self.get_logger().info(
            f"Start TCP: ({start_x:.4f}, {start_y:.4f}, {start_z:.4f})"
        )
        self.get_logger().info(
            f"Forward dir: ({fwd_x:.4f}, {fwd_y:.4f}, {fwd_z:.4f})"
        )

        # Sample baseline force (cable weight)
        force_samples = []
        for _ in range(10):
            obs = get_observation()
            if obs and obs.wrist_wrench:
                w = obs.wrist_wrench.wrench
                mag = math.sqrt(w.force.x**2 + w.force.y**2 + w.force.z**2)
                force_samples.append(mag)
            self.sleep_for(0.05)

        baseline_force = (
            sum(force_samples) / len(force_samples) if force_samples else 0
        )
        self.get_logger().info(f"Baseline force: {baseline_force:.1f}N")

        # Phase 1: Stabilize at current position (1s)
        send_feedback("Stabilizing...")
        for _ in range(20):
            obs = get_observation()
            if obs:
                self.set_pose_target(
                    move_robot=move_robot, pose=obs.controller_state.tcp_pose
                )
            self.sleep_for(0.05)

        # Phase 2: Push forward along insertion axis
        send_feedback("Approaching port...")
        target_x = start_x
        target_y = start_y
        target_z = start_z

        step_size = 0.0005  # 0.5mm per step
        max_distance = 0.10  # 10cm max travel
        total_distance = 0.0
        force_limit = baseline_force + 10.0  # 10N above baseline

        # Use default stiffness for approach (firm but not too stiff)
        approach_stiffness = [90.0, 90.0, 90.0, 50.0, 50.0, 50.0]
        approach_damping = [50.0, 50.0, 50.0, 20.0, 20.0, 20.0]

        for i in range(int(max_distance / step_size)):
            obs = get_observation()
            if obs is None:
                continue

            # Check relative force (above baseline)
            w = obs.wrist_wrench.wrench
            force_mag = math.sqrt(w.force.x**2 + w.force.y**2 + w.force.z**2)
            relative_force = force_mag - baseline_force

            if relative_force > force_limit:
                self.get_logger().warn(
                    f"Force {force_mag:.1f}N (relative {relative_force:.1f}N) > limit, pausing"
                )
                self.sleep_for(0.1)
                continue

            # Advance along forward axis
            target_x += fwd_x * step_size
            target_y += fwd_y * step_size
            target_z += fwd_z * step_size
            total_distance += step_size

            pose = Pose(
                position=Point(x=target_x, y=target_y, z=target_z),
                orientation=tcp.orientation,  # Keep original orientation
            )

            self.set_pose_target(
                move_robot=move_robot,
                pose=pose,
                stiffness=approach_stiffness,
                damping=approach_damping,
            )

            if i % 40 == 0:
                self.get_logger().info(
                    f"Distance: {total_distance * 1000:.1f}mm, "
                    f"force: {force_mag:.1f}N (rel: {relative_force:.1f}N), "
                    f"pos: ({target_x:.4f}, {target_y:.4f}, {target_z:.4f})"
                )

            self.sleep_for(0.05)

        # Phase 3: Hold and settle
        send_feedback("Holding position...")
        self.get_logger().info(
            f"Total distance: {total_distance * 1000:.1f}mm. Holding 5s..."
        )
        self.sleep_for(5.0)

        self.get_logger().info("SmartApproach complete")
        return True
