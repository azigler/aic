# AI for Industry Challenge -- Hackathon Repo

## What This Is

Competition entry for the [AI for Industry Challenge](https://www.intrinsic.ai/events/ai-for-industry-challenge/) ($180K prize pool). The task: train an AI policy to control a UR5e robot to insert fiber optic cables (SFP and SC connectors) into ports on a randomized task board. Evaluation is in Gazebo simulation.

## Challenge Summary

- **Robot:** UR5e + Robotiq Hand-E gripper + ATI F/T sensor + 3 wrist cameras
- **Task:** Insert cable connector (SFP or SC) into the correct port on a randomized task board
- **Scoring:** 100 pts/trial max (validity 1 + performance 24 + insertion 75). 3 trials per eval. Penalties for collisions (-24) and excessive force (-12).
- **Qualification deadline:** May 15, 2026. Eval: May 18-27. Top 30 advance.

## Repository Structure

```
aic/
├── aic_adapter/          # Sensor fusion -> Observation at 20Hz
├── aic_assets/           # 3D models (SFP, SC, NIC, cables, task board)
├── aic_bringup/          # Launch files for simulation environment
├── aic_controller/       # Impedance controller (C++, ros2_control)
├── aic_description/      # URDF/SDF for robot, task board, world
├── aic_engine/           # Trial orchestration and validation
├── aic_example_policies/ # Baselines: WaveArm, CheatCode, RunACT
├── aic_gazebo/           # Gazebo plugins
├── aic_interfaces/       # ROS 2 msg/srv/action definitions
├── aic_model/            # Policy framework (Python, lifecycle node)
├── aic_scoring/          # Scoring implementation
├── aic_utils/            # MuJoCo + Isaac Lab integrations, teleoperation
├── docker/               # Dockerfiles for eval and submission
├── docs/                 # Full documentation
└── .claude/
    ├── refs/             # Challenge description, research notes
    └── skills/           # Development workflow skills
```

## Key Technical Details

### Policy Interface

Your policy extends `aic_model.Policy` and implements `insert_cable()`:

```python
def insert_cable(self, task, get_observation, move_robot, send_feedback):
    obs = get_observation()  # Observation msg at up to 20Hz
    move_robot(motion_update)  # MotionUpdate or JointMotionUpdate
```

### Observation Contents (20Hz)
- `left_image`, `center_image`, `right_image` (1152x1024 RGB)
- `left_camera_info`, `center_camera_info`, `right_camera_info`
- `joint_states` (6 arm joints + gripper)
- `wrist_wrench` (3D force + 3D torque)
- `controller_state` (TCP pose, velocity, tracking error)

### Control Modes
- **Cartesian position:** Send target TCP pose in base_link or gripper/tcp frame
- **Cartesian velocity:** Send TCP velocity
- **Joint position:** Send target joint angles
- **Joint velocity:** Send target joint velocities
- Switch modes via `/aic_controller/change_target_mode` service

### Scoring Breakdown (per trial, max 100)
- Tier 1: Model validity (0-1)
- Tier 2: Smoothness (0-6) + Duration (0-12) + Efficiency (0-6) + Force penalty (0 to -12) + Contact penalty (0 to -24)
- Tier 3: Correct insertion (75) / Wrong port (-12) / Partial (38-50) / Proximity (0-25)

## Critical Controller Details (from source analysis)

- **Controller update rate:** 500 Hz (aic_ros2_controllers.yaml)
- **Observation rate:** 20 Hz (aic_adapter, triggered by camera sync)
- **FT sensor rate:** 50 Hz
- **Cartesian velocity limits:** ±0.25 m/s translational, 2.0 rad/s rotational
- **Max wrench (safety clamp):** ±10 N force, ±10 Nm torque
- **Tracking error timeout:** 2.0s (controller resets if target unreachable)
- **Nullspace control:** Disabled by default (stiffness=0), damping=10
- **Controller starts in Cartesian mode** -- switch via service before sending joint commands

### Policy.set_pose_target() Defaults
- Stiffness: [90, 90, 90, 50, 50, 50] (translation N/m, rotation Nm/rad)
- Damping: [50, 50, 50, 20, 20, 20]

### Gripper Joint Detail
- Adapter reorders joints: 6 arm + 1 gripper (position/2 for finger gap)
- Joint order: shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3, gripper

### Observation Assembly (aic_adapter)
- Cameras synchronized within ±1ms
- Joint/wrench/controller state: finds most recent message ≤ image timestamp
- Buffers: 128 messages per non-camera stream

## Development Commands

```bash
# Enter pixi env
pixi shell

# Run eval container
export DBX_CONTAINER_MANAGER=docker
distrobox enter -r aic_eval -- /entrypoint.sh

# Run a policy
pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.CheatCode

# Build submission container
docker compose -f docker/docker-compose.yaml build model

# Local eval
docker compose -f docker/docker-compose.yaml up
```

## Tech Stack

- **Language:** Python (policy), C++ (controller)
- **Framework:** ROS 2 Kilted Kaiju
- **Simulators:** Gazebo (eval), MuJoCo, Isaac Lab (training)
- **ML:** PyTorch, ACT (Action Chunking with Transformers)
- **Build:** Pixi (conda + ROS), Docker

## Conventions

- Python code follows ruff formatting
- Commit messages use gitmoji (see /commit skill)
- Task tracking via beads-rust (`br`)
- See `.claude/skills/` for all workflow skills
