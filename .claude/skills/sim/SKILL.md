---
name: sim
description: Launch Gazebo simulation environment, run trials, and collect training data
argument-hint: "[launch|teleop|export|scenario]"
---

# /sim - Simulation Environment

Manage the Gazebo simulation for training and testing cable insertion policies.

## Architecture

```
Terminal 0: Zenoh router (required)
Terminal 1: aic_eval container (Gazebo + aic_engine + scoring)
Terminal 2: pixi env (your aic_model policy node)
```

## Quick Reference

### Launch Eval Environment (Distrobox)

```bash
# One-time setup
export DBX_CONTAINER_MANAGER=docker
docker pull ghcr.io/intrinsic-dev/aic/aic_eval:latest
distrobox create -r --nvidia -i ghcr.io/intrinsic-dev/aic/aic_eval:latest aic_eval

# Start with engine (full eval pipeline)
distrobox enter -r aic_eval -- /entrypoint.sh ground_truth:=false start_aic_engine:=true

# Start without engine (free exploration)
distrobox enter -r aic_eval -- /entrypoint.sh ground_truth:=true start_aic_engine:=false
```

### Run a Policy

```bash
pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=<package>.<PolicyClass>
```

Example policies:
- `aic_example_policies.ros.WaveArm` -- minimal arm waving
- `aic_example_policies.ros.CheatCode` -- ground truth insertion (60pts/trial)
- `aic_example_policies.ros.RunACT` -- ACT transformer policy

### Save Results to Unique Directory

```bash
AIC_RESULTS_DIR=~/aic_results/<experiment_name> \
ros2 launch aic_bringup aic_gz_bringup.launch.py start_aic_engine:=true
```

Results are written to `$AIC_RESULTS_DIR/scoring.yaml`. Each run **overwrites** the
previous scoring.yaml, so always set a unique directory per experiment.

## Creating Training Scenarios

### Custom Task Board Configuration

```bash
/entrypoint.sh spawn_task_board:=true \
    task_board_x:=0.3 task_board_y:=-0.1 task_board_z:=1.2 \
    task_board_roll:=0.0 task_board_pitch:=0.0 task_board_yaw:=0.785 \
    sfp_mount_rail_0_present:=true sfp_mount_rail_0_translation:=-0.08 \
    sc_mount_rail_0_present:=true sc_mount_rail_0_translation:=-0.09 \
    nic_card_mount_0_present:=true nic_card_mount_0_translation:=0.005 \
    sc_port_0_present:=true sc_port_0_translation:=-0.04 \
    spawn_cable:=true cable_type:=sfp_sc_cable attach_cable_to_gripper:=true \
    ground_truth:=true start_aic_engine:=false
```

### Export World State for Cross-Simulator Training

World state auto-saves to `/tmp/aic.sdf` after spawning. Copy to preserve:

```bash
cp /tmp/aic.sdf ~/training_scenarios/scenario_001.sdf
```

Import into MuJoCo (`aic_utils/aic_mujoco/`) or Isaac Lab (`aic_utils/aic_isaac/`).

## Key Parameters

| Parameter | Values | Description |
|-----------|--------|-------------|
| `ground_truth` | true/false | Enable ground truth TF frames |
| `start_aic_engine` | true/false | Run trial orchestration |
| `spawn_task_board` | true/false | Auto-spawn task board |
| `spawn_cable` | true/false | Spawn cable in scene |
| `attach_cable_to_gripper` | true/false | Attach cable to gripper |
| `cable_type` | sfp_sc_cable, sfp_sc_cable_reversed | Cable type |
| `task_board_x/y/z` | float | Board position |
| `task_board_yaw` | float | Board orientation |
| `nic_card_mount_N_present` | true/false | NIC card on rail N (0-4) |
| `sc_port_N_present` | true/false | SC port N |

## Teleoperation

```bash
# See aic_utils/aic_teleoperation/README.md for setup
# Always tare F/T sensor before training episodes:
ros2 service call /aic_controller/tare_force_torque_sensor std_srvs/srv/Trigger
```

## Trial Lifecycle (when engine is running)

1. Engine spawns task board + cable
2. Engine sends InsertCable action to aic_model
3. Policy executes (observe + command loop)
4. Policy returns (or times out)
5. Engine scores the trial
6. Repeat for next trial (3 trials in qualification)

## Remote Runner (Mac Studio)

Run Gazebo eval natively on a Mac Studio M1 Max for faster iteration (~10-12 min
vs ~27 min local Docker).

### Mac Setup

Prerequisites: macOS 13+, Xcode Command Line Tools, Homebrew.

```bash
# 1. Install dependencies
brew install cmake git wget curl

# 2. Install pixi
curl -fsSL https://pixi.sh/install.sh | sh

# 3. Clone and install
mkdir -p ~/ws_aic/src && cd ~/ws_aic/src
git clone <repo-url> aic && cd aic
pixi install

# 4. Verify
pixi run gz sim --version
pixi run ros2 --version

# 5. Create ~/run-eval.sh (see .claude/refs/spec-mac-runner.md for full script)
```

### Running Headless on Mac

The remote eval script runs Gazebo with `gazebo_gui:=false` and `launch_rviz:=false`.
No display is needed -- the Mac can run fully headless over SSH.

### Differences from Docker-Based Eval

| Aspect | Docker (local) | Mac Native |
|--------|---------------|------------|
| Renderer | OGRE2 + OpenGL | OGRE2 + Metal |
| Physics | Identical | Identical |
| Scoring | Identical | Identical |
| Eval time | ~27 min | ~10-12 min |
| Setup | `docker compose up` | SSH + rsync |

Visual appearance may differ slightly (Metal vs OpenGL), but physics and scoring
are identical. Always do a final verification in Docker before submission.

### Usage from Dev Host

```bash
scripts/remote-eval.sh <policy_class>           # uses 'mac' SSH host
scripts/remote-eval.sh <policy_class> <host>    # custom host
```

## Cross-Simulator Training

Train in multiple simulators for domain randomization:

| Simulator | Guide | Best For |
|-----------|-------|----------|
| Gazebo | Default env | Evaluation-identical physics |
| MuJoCo | `aic_utils/aic_mujoco/` | Fast contact-rich simulation |
| Isaac Lab | `aic_utils/aic_isaac/` | GPU-accelerated parallel training |
