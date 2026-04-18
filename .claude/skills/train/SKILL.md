---
name: train
description: Training pipeline using LeRobot ACTPolicy. Follow competition patterns.
argument-hint: "[collect|train|convert]"
---

# /train - LeRobot ACT Training

## IMPORTANT: Use LeRobot, NOT custom models

The competition provides RunACT.py which uses LeRobot's ACTPolicy. Fine-tune
this model on our demonstration data. Do NOT build custom model architectures.

## Pre-flight checklist (BEFORE any training run)

1. **Run `/audit-train`** to verify the training script is sane (val split
   present, stats saved with correct shape, seed fixed, etc.)
2. **Confirm data distribution** — never mix Docker-collected + pixi-collected
   episodes, never mix high-friction + low-friction datasets. Our best
   baseline (v5) is 66 episodes of high-friction pixi-collected velocity data.
3. **GPU headroom** — `nvidia-smi`. Training a 100-epoch v5-class model on
   66 eps uses ~18GB peak; leave other workloads space.
4. **Scoped output** — write all artifacts under `~/aic-work/models/<name>/`
   so we don't pollute home dirs shared with other agents.

## Winning recipe (v5 baseline to beat)

- 66 episodes, high-friction pixi-collected, velocity mode
- 100 epochs max, lr=5e-6, batch=8, weight_decay=1e-4, grad_clip=10.0
- chunk_size=100 (from ACTConfig), n_action_steps default
- insertion_weight=1.0 (NO reweighting — exp-047/059/060 proved any value > 1
  regresses)
- 80/20 train/val split by episode, early stopping patience=10 on val_loss

Score range: 110-170/300, median ~140, Docker-verified 124.2/300.

## Training command (post-scripts/train_act.py rewrite)

```bash
ssh gpu "cd ~/aic-work/src && \
  nohup pixi run python scripts/train_act.py \
    --data-dir ~/aic-work/data/velocity \
    --output-dir ~/aic-work/models/act_velocity_v13 \
    --epochs 100 --batch-size 8 --lr 5e-6 \
    --val-frac 0.2 --patience 10 --seed 42 \
    > ~/aic-work/logs/train_v13.log 2>&1 &"
```

## Step 1: Collect Demonstrations

Use DataCollector (wraps CheatCode) to record expert demos via distrobox:

```bash
ssh gpu "export PATH=\$HOME/.pixi/bin:\$PATH && cd ~/aic-work/src && \
  OUT_DIR=~/aic-work/data/velocity_new \
  POLICY=aic_example_policies.ros.DataCollector \
  GROUND_TRUTH=true ~/aic-work/bin/run-eval.sh"
```

3 episodes per run (3 trials). Budget: ~2 minutes per run.

## Step 2: (Historical) Convert to LeRobot Format

scripts/convert_to_lerobot.py exists but we don't use LeRobot's HDF5 format —
train_act.py loads our numpy format directly. Kept for reference.

## Step 3: Fine-tune ACTPolicy

Use the command above. Training time: ~4 hours for 100 epochs on L4, may early-stop sooner.

## Step 4: Evaluate

```bash
MODEL_PATH=~/aic-work/models/act_velocity_v13/best /eval
```

Always 3-seed (see /eval). If mean improves ≥5 points over v5, promote.

## Rules

- **ALWAYS nohup** for training >10 min
- **ONE variable at a time** when tuning hyperparameters
- **Check val_loss AND eval score** — they correlate loosely. v6 (200 epochs)
  had best val loss but worst eval score.
- **Never retrain with insertion_weight > 1.0** without an explicit ablation bead
- **Never overwrite** an existing model directory. Always new name.
