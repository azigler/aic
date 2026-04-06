---
name: train
description: Training pipeline using LeRobot ACTPolicy. Follow competition patterns.
argument-hint: "[collect|train|convert]"
---

# /train - LeRobot ACT Training

## IMPORTANT: Use LeRobot, NOT custom models

The competition provides RunACT.py which uses LeRobot's ACTPolicy. Fine-tune
this model on our demonstration data. Do NOT build custom model architectures.

## Step 1: Collect Demonstrations

Use DataCollector (wraps CheatCode) to record expert demos:
```bash
# Via distrobox on GPU (ground_truth=true required for CheatCode)
ssh gpu "export PATH=\$HOME/.pixi/bin:\$PATH && cd ~/ws_aic/src/aic && \
  POLICY=aic_example_policies.ros.DataCollector \
  GROUND_TRUTH=true ~/run-eval.sh"
```

Data saved to `~/training_data/` on GPU. 3 episodes per eval run (3 trials).

## Step 2: Convert to LeRobot Format

TODO: Create conversion script from our numpy episode format to LeRobot
HDF5 dataset format. LeRobot expects:
```
dataset/
├── episode_0/
│   ├── observation.images.left_camera  # (T, C, H, W)
│   ├── observation.state               # (T, 26)
│   └── action                          # (T, 7)
└── meta/
    └── stats.json
```

## Step 3: Fine-tune ACTPolicy

```bash
# Use LeRobot's training script
ssh gpu "nohup pixi run python -m lerobot.scripts.train \
  --dataset.path=~/training_data_lerobot \
  --policy=act \
  --output_dir=~/models/act_finetuned \
  > /tmp/train.log 2>&1 &"
```

## Step 4: Evaluate

```bash
# Point RunACT to fine-tuned weights instead of HuggingFace
# (requires modifying RunACT.py to load local weights)
```

## Existing Training Data on GPU

- `~/training_data_pos/`: 39 episodes from 13 configs (position-mode, numpy format)
  - These were collected via pixi+distrobox (good distribution)
  - Need conversion to LeRobot format
- `~/training_data_new/`: 68 episodes from Docker collection (BAD -- don't use)

## Rules

- **ALWAYS nohup** for training >10 min
- **ONE variable at a time** when tuning hyperparameters
- **Check val_loss BUT also eval score** -- they don't always correlate
- **50 epochs** was the sweet spot for 24 demos with the old approach. May differ with LeRobot.
