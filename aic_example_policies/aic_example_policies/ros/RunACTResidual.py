#
#  RunACTResidual - RunACTLocal + classical spiral-search residual overlay.
#
#  When the fine-tuned ACT policy parks the end-effector near the port but
#  can't push through the final contact-rich insertion (the "last-cm"
#  plateau we've observed at 1-5cm from target in v5 evals), this policy
#  overlays a parametric spiral + small rotational wiggle on the ACT
#  velocity output. The overlay fires only when two conditions are met:
#
#    1. Contact is detected: wrist wrench F_z magnitude exceeds
#       RESIDUAL_FZ_THRESHOLD (default 2.0 N).
#    2. Vertical progress has stalled: |dz/dt| averaged over
#       RESIDUAL_STALL_WINDOW seconds (default 0.5s) is below
#       RESIDUAL_STALL_VZ_THRESHOLD (default 2 mm/s).
#
#  Rationale (AugInsert, IROS 2025): F/T is the most informative modality
#  for contact-rich assembly. Our ACT policy ignores wrist_wrench entirely;
#  a classical F/T-gated search is a zero-retrain way to use that signal.
#
#  Tunable via env vars:
#    RESIDUAL_ENABLED              "1" | "0"   (default "1")
#    RESIDUAL_FZ_THRESHOLD         N           (default 2.0)
#    RESIDUAL_STALL_WINDOW         s           (default 0.5)
#    RESIDUAL_STALL_VZ_THRESHOLD   m/s         (default 0.002)
#    RESIDUAL_SPIRAL_RADIUS        m           (default 0.003)
#    RESIDUAL_SPIRAL_RATE          Hz          (default 1.0)
#    RESIDUAL_DOWNWARD_BIAS        m/s         (default 0.005)
#    RESIDUAL_WIGGLE_AMPLITUDE     rad/s       (default 0.05)
#
#  All other behavior (observation construction, model loading, chunking)
#  is inherited from RunACTLocal.
#

import math
import os

from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Twist, Vector3
from rclpy.duration import Duration

from aic_example_policies.ros.RunACTLocal import RunACTLocal


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


class RunACTResidual(RunACTLocal):
    def __init__(self, parent_node):
        super().__init__(parent_node)

        self.residual_enabled = _env_bool("RESIDUAL_ENABLED", True)
        self.fz_threshold = _env_float("RESIDUAL_FZ_THRESHOLD", 2.0)
        self.stall_window_s = _env_float("RESIDUAL_STALL_WINDOW", 0.5)
        self.stall_vz_threshold = _env_float(
            "RESIDUAL_STALL_VZ_THRESHOLD", 0.002
        )
        self.spiral_radius = _env_float("RESIDUAL_SPIRAL_RADIUS", 0.003)
        self.spiral_rate_hz = _env_float("RESIDUAL_SPIRAL_RATE", 1.0)
        self.downward_bias = _env_float("RESIDUAL_DOWNWARD_BIAS", 0.005)
        self.wiggle_amplitude = _env_float("RESIDUAL_WIGGLE_AMPLITUDE", 0.05)

        self.get_logger().info(
            f"ResidualOverlay: enabled={self.residual_enabled} "
            f"fz_th={self.fz_threshold}N window={self.stall_window_s}s "
            f"vz_th={self.stall_vz_threshold}m/s "
            f"spiral_r={self.spiral_radius}m rate={self.spiral_rate_hz}Hz "
            f"down_bias={self.downward_bias}m/s "
            f"wiggle={self.wiggle_amplitude}rad/s"
        )

        # Rolling buffer of (timestamp, z_pos) for stall detection.
        self._z_history: list[tuple[float, float]] = []
        self._residual_phase: float = 0.0
        self._residual_active_since: float | None = None

    def _observed_fz_magnitude(self, obs_msg: Observation) -> float:
        w = obs_msg.wrist_wrench.wrench
        return math.sqrt(
            w.force.x * w.force.x
            + w.force.y * w.force.y
            + w.force.z * w.force.z
        )

    def _z_speed_estimate(self, now: float, z_now: float) -> float:
        """Return |dz/dt| averaged over stall_window_s, or +inf if not enough data."""
        self._z_history.append((now, z_now))
        cutoff = now - self.stall_window_s
        self._z_history = [(t, z) for (t, z) in self._z_history if t >= cutoff]
        if len(self._z_history) < 2:
            return float("inf")
        t0, z0 = self._z_history[0]
        tN, zN = self._z_history[-1]
        dt = max(tN - t0, 1e-6)
        return abs(zN - z0) / dt

    def _should_engage_residual(
        self, obs_msg: Observation, now: float, z_now: float
    ) -> bool:
        if not self.residual_enabled:
            return False
        fz_mag = self._observed_fz_magnitude(obs_msg)
        if fz_mag < self.fz_threshold:
            return False
        vz = self._z_speed_estimate(now, z_now)
        return vz < self.stall_vz_threshold

    def _residual_twist(self, dt: float) -> tuple[Vector3, Vector3]:
        """Return (linear, angular) overlay vectors in base frame."""
        self._residual_phase += 2.0 * math.pi * self.spiral_rate_hz * dt
        cos_p = math.cos(self._residual_phase)
        sin_p = math.sin(self._residual_phase)
        # Parametric spiral in the horizontal plane with small down bias.
        lin_x = (
            self.spiral_radius * self.spiral_rate_hz * 2.0 * math.pi * (-sin_p)
        )
        lin_y = (
            self.spiral_radius * self.spiral_rate_hz * 2.0 * math.pi * (cos_p)
        )
        lin_z = -self.downward_bias
        # Small rotational wiggle around Z to break static friction.
        ang_z = self.wiggle_amplitude * cos_p
        return (
            Vector3(x=float(lin_x), y=float(lin_y), z=float(lin_z)),
            Vector3(x=0.0, y=0.0, z=float(ang_z)),
        )

    def insert_cable(
        self,
        task: Task,
        get_observation,
        move_robot,
        send_feedback,
        **kwargs,
    ):
        self.policy.reset()
        self.get_logger().info(
            f"RunACTResidual.insert_cable() enter. Task: {task}"
        )

        # Sim-clock-aware deadline + pacing. The portal enforces
        # task.time_limit on the ROS clock (sim time), so we must use
        # Node.get_clock() rather than time.time(). See bd-spw / upstream
        # PR #500 / docs/troubleshooting.md.
        clock = self.get_clock()
        start_time = clock.now()

        # Prefer task.time_limit (uint64 seconds) when set; fall back to
        # the legacy TIME_LIMIT env var (default 30s) for local overrides.
        if task is not None and getattr(task, "time_limit", 0):
            time_limit = float(task.time_limit)
        else:
            time_limit = float(os.environ.get("TIME_LIMIT", "30"))
        deadline = Duration(seconds=time_limit)
        period = Duration(seconds=0.25)  # 4 Hz control rate

        # Sim-time floats (seconds since loop start) for the residual
        # stall detector — the buffer math in _z_speed_estimate is
        # frame-relative, so any consistent monotonic clock works as long
        # as it ticks at sim rate (not wall rate).
        last_loop_sec = 0.0

        residual_fired_count = 0

        while (clock.now() - start_time) < deadline:
            loop_start = clock.now()
            now_sec = (loop_start - start_time).nanoseconds * 1e-9
            dt = max(now_sec - last_loop_sec, 1e-3)
            last_loop_sec = now_sec

            observation_msg = get_observation()
            if observation_msg is None:
                self.get_logger().info("No observation received.")
                continue

            obs_tensors = self.prepare_observations(observation_msg)

            # torch is stashed on self by RunACTLocal.__init__ (deferred
            # import to keep module discovery fast — see bd-crw).
            with self._torch.inference_mode():
                normalized_action = self.policy.select_action(obs_tensors)

            raw_action_tensor = (
                normalized_action * self.action_std
            ) + self.action_mean
            action = raw_action_tensor[0].cpu().numpy()

            twist = Twist(
                linear=Vector3(
                    x=float(action[0]),
                    y=float(action[1]),
                    z=float(action[2]),
                ),
                angular=Vector3(
                    x=float(action[3]),
                    y=float(action[4]),
                    z=float(action[5]),
                ),
            )

            z_now = observation_msg.controller_state.tcp_pose.position.z
            if self._should_engage_residual(observation_msg, now_sec, z_now):
                residual_fired_count += 1
                if self._residual_active_since is None:
                    self._residual_active_since = now_sec
                    self.get_logger().info(
                        f"Residual ENGAGED (|F|≥{self.fz_threshold}N, "
                        f"|vz|<{self.stall_vz_threshold}m/s)"
                    )
                r_lin, r_ang = self._residual_twist(dt)
                twist.linear.x = twist.linear.x + r_lin.x
                twist.linear.y = twist.linear.y + r_lin.y
                twist.linear.z = twist.linear.z + r_lin.z
                twist.angular.x = twist.angular.x + 0.0
                twist.angular.y = twist.angular.y + 0.0
                twist.angular.z = twist.angular.z + r_ang.z
            elif self._residual_active_since is not None:
                self.get_logger().info("Residual DISENGAGED.")
                self._residual_active_since = None
                self._residual_phase = 0.0

            motion_update = self.set_cartesian_twist_target(twist)
            move_robot(motion_update=motion_update)
            send_feedback("in progress...")

            # Maintain 4 Hz control rate using the ROS clock so we honor
            # sim time on the portal.
            elapsed = clock.now() - loop_start
            remaining_ns = period.nanoseconds - elapsed.nanoseconds
            if remaining_ns > 0:
                clock.sleep_for(Duration(nanoseconds=remaining_ns))

        self.get_logger().info(
            f"RunACTResidual exiting. Residual fired {residual_fired_count} steps."
        )
        return True
