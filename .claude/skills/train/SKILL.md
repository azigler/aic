---
name: train
description: ACT training pipeline -- data collection, model training, hyperparameter tuning, and evaluation
argument-hint: "[collect|train|eval|tune|status]"
---

# /train - ACT Training Pipeline

The **primary skill** for score improvement. After 24 experiments, classical control
(Branch A, plateau 93.4) and camera perception (Branch B, plateau ~100) are exhausted.
ACT imitation learning (Branch C) is the path forward.

## Overview

```
COLLECT DEMOS -> TRAIN MODEL -> EVALUATE -> TUNE CONFIG -> REPEAT
     (~30 min)     (~1-2 hrs)     (~10 min)     (analysis)
```

Each full cycle takes ~2-3 hours on the L4 GPU. The iteration variable is the
training config, not the policy source code. The RunACT policy code stays fixed.

## Step 1: Collect Demonstrations

Demonstrations are collected by running CheatCode (the ground-truth expert) with
domain randomization. Data is stored on the GPU instance.

### Automated Collection

```bash
# On GPU instance
ssh gpu
cd ~/ws_aic/src/aic
scripts/collect_demos.sh --num-demos 100 --output-dir ~/training_data/batch_001
```

### Manual Collection (CheatCode)

```bash
# Terminal 1: Sim + engine
distrobox enter -r aic_eval -- /entrypoint.sh \
  ground_truth:=true start_aic_engine:=true

# Terminal 2: CheatCode policy
pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=aic_example_policies.ros.CheatCode

# Terminal 3: Record
pixi run ros2 bag record \
  /left_camera/image /center_camera/image /right_camera/image \
  /joint_states /gripper_state \
  /fts_broadcaster/wrench \
  /aic_controller/controller_state \
  /aic_controller/pose_commands \
  /tf /tf_static \
  -o ~/training_data/demo_001
```

### Domain Randomization

Vary these parameters across collection runs for generalization:
- Task board position (`task_board_x/y/z/yaw`)
- NIC card rail and offset
- SC port position
- Grasp pose noise (~2mm, ~0.04 rad as in evaluation)

### Data Guidelines

| Quantity | Expected Quality | Notes |
|----------|-----------------|-------|
| 50 demos | Minimum viable | May overfit to seen configs |
| 100 demos | Good baseline | Reasonable generalization |
| 200 demos | Strong | Covers most board configurations |
| 500 demos | Comprehensive | Diminishing returns beyond this |

Data lives in `~/training_data/` on the GPU instance.

## Step 2: Train ACT Model

### Command Reference

```bash
# Basic training
ssh gpu "cd ~/ws_aic/src/aic && scripts/train_act.py \
  --data-dir ~/training_data \
  --output-dir ~/models/exp-NNN \
  --chunk-size 50 \
  --lr 1e-4 \
  --batch-size 32 \
  --epochs 100"

# Resume from checkpoint
ssh gpu "cd ~/ws_aic/src/aic && scripts/train_act.py \
  --data-dir ~/training_data \
  --output-dir ~/models/exp-NNN \
  --resume ~/models/exp-NNN/checkpoints/epoch_050.pt \
  --epochs 100"

# Or run training directly on the GPU
ssh gpu
cd ~/ws_aic/src/aic
pixi run python scripts/train_act.py --batch-size 32 --epochs 100
```

### Hyperparameter Guide

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `chunk_size` | 50 | 10-100 | Larger = smoother but less reactive |
| `lr` | 1e-4 | 1e-5 to 1e-3 | Standard with cosine schedule |
| `batch_size` | 32 | 32-64 | Limited by L4 VRAM (24GB) |
| `epochs` | 100 | 50-500 | More epochs if more data |
| `img_size` | 224 | 128-320 | Larger = more detail but slower |
| `num_cameras` | 3 | 1-3 | More = better perception, more VRAM |
| `kl_weight` | 10 | 1-100 | Higher = more regularized latent space |
| `hidden_dim` | 512 | 256-1024 | Model capacity |
| `dim_feedforward` | 3200 | 1600-6400 | Transformer FFN width |
| `num_layers` | 4 | 2-8 | Transformer depth |

### Recommended Experiment Families

Sweep one parameter at a time:

```
Family: Chunk Size Sweep
  - exp-A: chunk_size=10 -> score X
  - exp-B: chunk_size=25 -> score Y
  - exp-C: chunk_size=50 -> score Z
  - exp-D: chunk_size=100 -> score W
  -> Pick best, move to next family

Family: Data Quantity Sweep
  - exp-A: 50 demos -> score X
  - exp-B: 100 demos -> score Y
  - exp-C: 200 demos -> score Z
  -> Pick best, move to next family
```

### L4 GPU Memory Budget (24GB VRAM)

| Configuration | Approx VRAM | Notes |
|--------------|-------------|-------|
| ACT, 3 cameras, batch 32, 224px | ~12GB | Comfortable |
| ACT, 3 cameras, batch 64, 224px | ~20GB | Near limit |
| ACT, 3 cameras, batch 32, 320px | ~18GB | Higher res |
| ACT, 1 camera, batch 64, 224px | ~10GB | Fast iteration |

## Step 3: Evaluate Trained Model

### Run Evaluation

```bash
# Ensure the model weights are accessible to the policy
# RunACT loads from a configured model path
scripts/remote-eval.sh aic_example_policies.ros.RunACT
```

### Parse Results

```bash
cat aic_results/scoring.yaml
# Look at per-trial breakdown: T1 (validity), T2 (performance), T3 (insertion)
```

### Compare Against Best

| Metric | Current Best | This Run | Delta |
|--------|-------------|----------|-------|
| Total | 110.4 | ? | ? |
| Trial 1 (SFP) | ? | ? | ? |
| Trial 2 (SFP) | ? | ? | ? |
| Trial 3 (SC) | ? | ? | ? |

## Step 4: Tune and Iterate

After evaluation, decide what to change:

| Symptom | Likely Fix |
|---------|-----------|
| Low score, all trials | More demos, longer training |
| Good SFP, bad SC | More SC demos, domain randomization |
| Jerky motion | Larger chunk size, lower stiffness |
| Slow insertion | Smaller chunk size, higher stiffness |
| Overfitting (train good, eval bad) | More domain randomization |
| Underfitting (train bad) | More epochs, higher LR, larger model |
| OOM during training | Reduce batch size or image resolution |

## Directory Structure

```
# On GPU instance
~/training_data/                    # Raw demonstration data
├── batch_001/                      # First collection batch
│   ├── demo_001/                   # Individual demo (rosbag)
│   ├── demo_002/
│   └── ...
├── batch_002/
└── ...

~/models/                           # Trained models
├── exp_025_act_baseline/
│   ├── config.yaml                 # Training hyperparameters
│   ├── checkpoints/
│   │   ├── epoch_010.pt
│   │   ├── epoch_050.pt
│   │   └── best.pt                # Best by validation loss
│   ├── logs/                       # Tensorboard logs
│   └── eval/
│       └── scoring.yaml            # Eval results
├── exp_026_act_more_demos/
└── ...
```

## Checkpoint Management

```python
# Save checkpoint (handled by train_act.py)
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': config,
    'eval_score': score,
}, f'~/models/{exp_name}/checkpoints/epoch_{epoch:03d}.pt')
```

Always keep the best checkpoint by validation loss. Evaluate the top 2-3
checkpoints per training run (not just the final one).

## Scoring Quick Reference

| Metric | Max | Threshold |
|--------|-----|-----------|
| Correct insertion | 75/trial | Contact sensor verification |
| Partial insertion | 38-50/trial | Within port bounding box |
| Proximity | 0-25/trial | Distance to port |
| Smoothness | 0-6/trial | Jerk < 50 m/s^3 |
| Duration | 0-12/trial | < 5s = max, > 60s = 0 |
| Efficiency | 0-6/trial | Short path length |
| Force penalty | 0 to -12/trial | > 20N for > 1s |
| Contact penalty | 0 to -24/trial | Any robot-enclosure contact |

**Target: 100 pts/trial x 3 trials = 300 max**

## Cloud GPU (OVH L4-90)

The L4 with 24GB VRAM and CUDA runs both training and evaluation. Cost: ~$1.00/hr.
Stop the instance when not in use.

```bash
# Check GPU status
ssh gpu "nvidia-smi"

# Monitor training
ssh gpu "tail -f ~/models/exp-NNN/logs/train.log"
```

## Rules

- **Iterate fast:** Short training runs with quick evals beat long monolithic runs
- **Track everything:** Every experiment gets a config, checkpoints, and eval scores
- **Domain randomize:** Vary board pose, port positions, grasp noise during data collection
- **Watch for overfitting:** If scores are high on fixed configs but low on random, diversify training data
- **One variable at a time:** Change chunk size OR learning rate, not both
- **Submission limit:** 1 per day to cloud eval. Local eval is unlimited.

## Related Skills

- `/experiment` -- Experiment loop (propose, run, log, analyze)
- `/sim` -- Launch simulation for data collection
- `/eval-policy` -- Score parsing and comparison
- `/impl` -- Policy code reference (RunACT)
- `/commit` -- Commit conventions
