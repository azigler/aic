#
# exp-006: DirectApproach v4
#
# Back to basics from exp-004 (best score 93.4) but faster:
# 1. Larger step size (1mm instead of 0.5mm) for faster descent
# 2. Skip the stabilization phase (wasted 1s)
# 3. Shorter spiral search (compact, fast)
# 4. Single focused push phase
# Goal: same insertion quality as exp-004 but in <15s for max duration bonus
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
            f"DirectApproach v4: port={task.port_name} type={task.plug_type}"
        )

        obs = get_observation()
        if obs is None:
            return False

        tcp = obs.controller_state.tcp_pose
        self.get_logger().info(
            f"TCP: ({tcp.position.x:.4f}, {tcp.position.y:.4f}, {tcp.position.z:.4f})"
        )

        # Quick baseline force sample (0.5s instead of 1s)
        forces = []
        for _ in range(10):
            obs = get_observation()
            if obs:
                forces.append(self._get_force(obs))
            self.sleep_for(0.05)
        baseline = sum(forces) / len(forces) if forces else 25.0
        self.get_logger().info(f"Baseline: {baseline:.1f}N")

        obs = get_observation()
        start_pose = obs.controller_state.tcp_pose
        target_x = start_pose.position.x
        target_y = start_pose.position.y
        target_z = start_pose.position.z

        force_limit = 15.0  # N above baseline

        # Phase 1: Fast descent (1mm steps, ~10cm in ~5s)
        stiffness = [90.0, 90.0, 70.0, 50.0, 50.0, 50.0]
        damping = [50.0, 50.0, 40.0, 25.0, 25.0, 25.0]

        step = 0.001  # 1mm per step
        max_descent = 0.12  # 12cm
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

            if i % 20 == 0:
                self.get_logger().info(
                    f"Descent: {total * 1000:.0f}mm, z={target_z:.4f}, f={rel_f:.1f}N"
                )
            self.sleep_for(0.04)

        self.get_logger().info(f"Descended {total * 1000:.0f}mm")

        # Phase 2: Quick spiral at current depth
        spiral_stiff = [60.0, 60.0, 40.0, 30.0, 30.0, 30.0]
        spiral_damp = [45.0, 45.0, 35.0, 20.0, 20.0, 20.0]

        center_x = target_x
        center_y = target_y
        spiral_z = target_z

        for ring in range(1, 6):
            radius = ring * 0.002
            points = max(8, ring * 6)

            for p in range(points):
                angle = 2 * math.pi * p / points
                sx = center_x + radius * math.cos(angle)
                sy = center_y + radius * math.sin(angle)
                spiral_z -= 0.0002  # Gentle descent during spiral

                obs = get_observation()
                if obs is None:
                    continue

                rel_f = self._get_force(obs) - baseline
                if rel_f > force_limit:
                    self.sleep_for(0.03)
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
                self.sleep_for(0.03)

        # Phase 3: Final compliant push
        insert_stiff = [40.0, 40.0, 25.0, 20.0, 20.0, 20.0]
        insert_damp = [35.0, 35.0, 25.0, 15.0, 15.0, 15.0]

        final_z = spiral_z
        for _ in range(150):
            obs = get_observation()
            if obs is None:
                continue

            rel_f = self._get_force(obs) - baseline
            if rel_f > force_limit:
                self.sleep_for(0.04)
                continue

            final_z -= 0.0005
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
            self.sleep_for(0.03)

        # Brief hold
        self.sleep_for(3.0)
        self.get_logger().info(f"Done. Final z={final_z:.4f}")
        return True
