---
name: train
description: ML training loop for cable insertion policies -- data collection, training, and experiment management
argument-hint: "[collect|train|checkpoint|compare]"
---

# /train - Policy Training

Orchestrate training loops for cable insertion policies. This skill covers data
collection, model training, checkpointing, and experiment tracking.

## Training Approaches

### 1. Imitation Learning (ACT / Diffusion Policy)

The primary approach: learn from demonstrations.

**Data collection:**
1. Launch sim with ground truth enabled (`/sim launch` with `ground_truth:=true`)
2. Teleoperate the robot to perform insertions
3. Record observation-action pairs at 20Hz
4. Vary task board configurations for domain randomization

**Observation space (per timestep):**
- 3x camera images (1152x1024 RGB) -- may need resizing for model input
- 6 joint positions + 1 gripper position
- 6D wrench (force + torque)
- TCP pose (position + quaternion)
- TCP velocity (linear + angular)

**Action space:**
- Cartesian: 6D pose (position + quaternion) or 6D velocity (linear + angular)
- Joint: 6 joint positions or 6 joint velocities
- Stiffness/damping parameters (optional, can be fixed)

**Key training parameters for ACT:**
- Chunk size: number of future actions predicted at once (typically 10-100)
- Camera image resolution: downsample for speed (224x224 typical)
- Learning rate: ~1e-4 with cosine schedule
- Batch size: 32-128 depending on GPU memory

### 2. Reinforcement Learning

Alternative: learn from reward signal.

**Reward shaping for cable insertion:**
- Distance to target port (dense, proximity reward)
- Alignment with port axis (orientation reward)
- Successful insertion (sparse, large bonus)
- Force penalty (penalize excessive contact force)
- Smoothness penalty (penalize jerk)

**Recommended RL framework:** Isaac Lab (GPU-parallel) or MuJoCo (fast CPU)

### 3. Classical Control + Perception

Hybrid approach: use ML for perception, classical control for insertion.

- Vision model: detect port location from camera images
- Estimate port pose in TCP frame
- Spiral search + compliant insertion controller
- Impedance control with force feedback for final insertion

## Data Collection Workflow

### 1. Collect Demonstrations via Teleoperation

```bash
# Terminal 1: Launch sim
distrobox enter -r aic_eval -- /entrypoint.sh \
  ground_truth:=true start_aic_engine:=false \
  spawn_task_board:=true spawn_cable:=true attach_cable_to_gripper:=true

# Terminal 2: Record (implement your own recorder or use rosbag)
pixi run ros2 bag record \
  /left_camera/image /center_camera/image /right_camera/image \
  /joint_states /gripper_state \
  /fts_broadcaster/wrench \
  /aic_controller/controller_state \
  /aic_controller/pose_commands \
  /tf /tf_static \
  -o ~/training_data/demo_001
```

### 2. Collect via CheatCode Policy

Use CheatCode as an expert demonstrator:

```bash
# Terminal 1: Sim + engine
distrobox enter -r aic_eval -- /entrypoint.sh \
  ground_truth:=true start_aic_engine:=true

# Terminal 2: CheatCode policy (record its actions)
pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=aic_example_policies.ros.CheatCode

# Terminal 3: Record
pixi run ros2 bag record -a -o ~/training_data/cheatcode_demo_001
```

### 3. Domain Randomization

Vary these parameters across collection runs:
- Task board position (`task_board_x/y/z/yaw`)
- NIC card rail and offset
- SC port position
- Grasp pose noise (~2mm, ~0.04 rad as in evaluation)

## Experiment Tracking

### Directory Structure

```
experiments/
├── exp_001_act_baseline/
│   ├── config.yaml          # Hyperparameters
│   ├── checkpoints/         # Model weights
│   │   ├── epoch_010.pt
│   │   └── best.pt
│   ├── logs/                # Training logs
│   └── eval/                # Evaluation results
│       └── scoring.yaml     # From aic_engine
├── exp_002_act_larger/
└── ...
```

### Checkpoint Management

```python
# Save checkpoint
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': config,
    'eval_score': score,
}, f'experiments/{exp_name}/checkpoints/epoch_{epoch:03d}.pt')
```

### Evaluation Loop

For each checkpoint:
1. Export model to policy format
2. Run 3-trial eval via aic_engine
3. Parse `scoring.yaml` for total score
4. Compare against best checkpoint
5. Save best model

## Scoring Quick Reference

| Metric | Max | Threshold |
|--------|-----|-----------|
| Correct insertion | 75/trial | Contact sensor verification |
| Partial insertion | 38-50/trial | Within port bounding box |
| Proximity | 0-25/trial | Distance to port |
| Smoothness | 0-6/trial | Jerk < 50 m/s³ |
| Duration | 0-12/trial | < 5s = max, > 60s = 0 |
| Efficiency | 0-6/trial | Short path length |
| Force penalty | 0 to -12/trial | > 20N for > 1s |
| Contact penalty | 0 to -24/trial | Any robot-enclosure contact |

**Target: 100 pts/trial x 3 trials = 300 max**

## Cloud GPU Training (OVH L4-90)

The OVH L4-90 instance has an NVIDIA L4 GPU with 24GB VRAM and CUDA support,
matching the official eval hardware. Training runs directly on this instance.

### Recommended Batch Sizes for L4 (24GB VRAM)

| Policy Type | Batch Size | Notes |
|-------------|-----------|-------|
| ACT (3 cameras) | 32-64 | 224x224 image input |
| Diffusion Policy | 16-32 | Higher memory per sample |
| RL (Isaac Lab) | GPU-parallel envs | Scales with VRAM |

### Training on the GPU Instance

```bash
# SSH in and run training directly
ssh gpu
cd ~/ws_aic/src/aic

# ACT training example
pixi run python train_act.py --batch-size 32 --epochs 100

# Or from dev host via SSH
ssh gpu "cd ~/ws_aic/src/aic && pixi run python train_act.py --batch-size 32 --epochs 100"
```

### Data Collection on GPU Instance

Data collection (teleop or CheatCode demos) can run on the GPU instance with
Gazebo headless. The GPU accelerates rendering for faster-than-realtime collection.

```bash
ssh gpu
cd ~/ws_aic/src/aic
# Launch headless sim + record demos
pixi run ros2 bag record -a -o ~/training_data/demo_001 &
pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true -p policy:=aic_example_policies.ros.CheatCode
```

### Cost Note

The GPU instance costs ~$1.00/hr. Stop the instance when not training or
experimenting. Budget: ~$1.00/hr x 5hr/day x 50 days = ~$250 total.

## Rules

- **Iterate fast:** Short training runs with quick evals beat long monolithic runs
- **Track everything:** Every experiment gets a config, checkpoints, and eval scores
- **Domain randomize:** Vary board pose, port positions, grasp noise
- **Watch for overfitting:** If scores are high on fixed configs but low on random, diversify training
- **Submission limit:** 1 per day to cloud eval. Local eval is unlimited.
