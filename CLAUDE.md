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

## Methodology

See `.claude/refs/methodology.md` for the full experiment protocol.

**Core loop:** HYPOTHESIZE -> IMPLEMENT -> SCORE -> LOG -> ANALYZE -> ADJUST

Every experiment gets a bead. Beads are the research log -- hypothesis, changes,
results table, analysis, and next steps. See `/experiment` skill.

**Exploration tree:**
- Branch A: Classical (hardcoded -> vision -> force control) -- **start here**
- Branch B: Imitation (demos -> ACT training)
- Branch C: Hybrid (vision + learned insertion)
- Branch D: RL (Isaac Lab)

**Local scoring:** `docker compose -f docker/docker-compose.yaml up` runs headless
eval. Results in `aic_results/scoring.yaml`. Unlimited local runs; 1/day cloud submit.

## Skills Inventory

| Skill | Purpose |
|-------|---------|
| `/experiment` | **Core loop** -- propose, run, log, analyze experiments |
| `/sim` | Gazebo lifecycle, scenarios, cross-simulator training |
| `/train` | ML training, data collection, experiment tracking |
| `/eval-policy` | Run trials, parse scores, compare experiments |
| `/impl` | Policy development phases and control interface reference |
| `/orient` | Session entry, state discovery, routing |
| `/lint` | Python/ruff code quality |
| `/test` | Sim-based eval + unit tests |
| `/spec` | Design documents for policy approaches |
| `/review` | Experiment-driven design decisions |
| `/release` | Docker submission workflow |
| `/beads` | Task tracking and research log |
| `/commit` | Gitmoji commit conventions |
| `/branch` | Branching strategy |
| `/orchestrator` | Subagent delegation |
| `/housekeeping` | Cleanup workflows |

## Key References

- `.claude/refs/challenge-description.md` -- full challenge spec
- `.claude/refs/methodology.md` -- experiment protocol
- `.claude/refs/experiment-log.md` -- score leaderboard
- `.claude/refs/decisions.md` -- design decision log

<<<<<<< HEAD
## Remote Runner

The eval stack runs in Docker (`docker compose up` with eval + model containers).
A remote Linux machine with a GPU is the recommended way to speed up iteration.

**Why Linux GPU, not macOS:**
- The eval container (`ghcr.io/intrinsic-dev/aic/aic_eval`) is linux/amd64 only
- Docker on macOS requires x86 emulation (Rosetta), which is slow and fragile
- Native Gazebo on macOS fails: conda ogre2 crashes during Metal rendering init,
  rosidl Python bindings have flat-namespace symbol issues, and macOS SIP blocks
  the DYLD workarounds
- Linux GPU (nvidia-container-toolkit) gives native Docker + GPU passthrough

**How it works:**
1. Scale up the VPS with a GPU (L4 or similar), or use any Linux box with Docker + nvidia GPU
2. `rsync` code to remote
3. `ssh` to run `docker compose -f docker/docker-compose.yaml up`
4. `rsync` results back
5. Analyze `scoring.yaml` locally

**Configuration:** `scripts/runner-config.sh` defines remote host and SSH settings.
`scripts/remote-eval.sh` orchestrates the rsync + eval + fetch cycle.
=======
## Cloud GPU Runner (OVH L4-90)

An OVH cloud GPU instance (NVIDIA L4, 24GB VRAM) serves as the remote eval,
training, and submission runner. This matches the **official cloud eval hardware**
exactly, eliminating sim-to-sim GPU discrepancy. Cost: ~$1.00/hr.

**How it works:**
1. Edit policy code locally (or SSH directly into the GPU instance)
2. `rsync` code to GPU instance
3. `ssh` to run eval (Gazebo native + ROS 2 via pixi, GPU-accelerated)
4. `rsync` results back
5. Analyze `scoring.yaml` locally

**Configuration:** `scripts/runner-config.sh` defines `GPU_HOST` and SSH settings.
An SSH config entry named `gpu` should exist in `~/.ssh/config`.
>>>>>>> worktree-agent-ab72a33a

**Usage:**
```bash
scripts/remote-eval.sh <policy_class>              # e.g. aic_example_policies.ros.BlindPush
<<<<<<< HEAD
scripts/remote-eval.sh <policy_class> <remote_host> # override host
=======
scripts/remote-eval.sh <policy_class> <gpu_host>   # override host (default: gpu)
>>>>>>> worktree-agent-ab72a33a
```

**Performance comparison:**

| Setup | Eval Time | Experiments/Hour |
|-------|-----------|-----------------|
| Local CPU-only (Docker) | ~27 min | ~2 |
<<<<<<< HEAD
| Linux GPU (L4, Docker) | ~5-8 min | ~8-10 |
=======
| Cloud GPU (L4) | ~5-10 min | ~6-12 |
>>>>>>> worktree-agent-ab72a33a

**Training:** The L4 with 24GB VRAM and CUDA supports training directly on the
instance (ACT batch 32-64, diffusion batch 16-32). See `/train` for details.

The cloud GPU runner handles **dev iteration, training, and Docker builds**. Final
submission Docker images can be built and pushed to ECR directly from the GPU
instance (see `/release`).

## Conventions

- Python code follows ruff formatting
- Commit messages use gitmoji (see /commit skill)
- Task tracking via beads-rust (`br`) -- each experiment = one bead
- Always push commits after each experiment
- See `.claude/skills/` for all workflow skills
