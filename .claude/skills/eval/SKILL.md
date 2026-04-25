---
name: eval
description: Run policy evaluation on GPU via distrobox+pixi. NEVER use Docker compose for iteration.
argument-hint: "[policy_class] [--seeds N]"
---

# /eval - Run Policy Evaluation

## Quick Eval (distrobox + pixi on GPU)

All paths assume the `~/aic/` scoping convention (see CLAUDE.md "Project layout").

```bash
# 1. Rsync code to GPU (target: ~/aic/src/, leaves ~/aic/data ~/aic/models alone)
rsync -az --exclude='.git' --exclude='.pixi' --exclude='__pycache__' \
  --exclude='data' --exclude='models' --exclude='results' --exclude='logs' \
  /home/ubuntu/aic/ gpu:~/aic/src/

# 2. Reinstall pixi package (REQUIRED after code changes)
ssh gpu "export PATH=\$HOME/.pixi/bin:\$PATH && cd ~/aic/src && \
  pixi reinstall ros-kilted-aic-example-policies 2>&1 | tail -3"

# 3. Run eval (example: residual policy on v5 weights)
ssh gpu "export PATH=\$HOME/.pixi/bin:\$PATH && cd ~/aic/src && \
  MODEL_PATH=~/aic/models/act_velocity_v5/best \
  TIME_LIMIT=30 \
  POLICY=aic_example_policies.ros.RunACTResidual ~/aic/bin/run-eval.sh"

# 4. Read results
ssh gpu "cat ~/aic/results/scoring.yaml"
```

## ALWAYS do a 3-seed variance run

Single-run eval is unreliable. We measured v5 variance of 139-168 across runs
(29-point spread from randomized task configs). Default to 3 seeds:

```bash
for i in 1 2 3; do
  ssh gpu "export PATH=\$HOME/.pixi/bin:\$PATH && cd ~/aic/src && \
    MODEL_PATH=~/aic/models/act_velocity_v5/best \
    RANDOM_SEED=$i \
    POLICY=aic_example_policies.ros.RunACTLocal ~/aic/bin/run-eval.sh"
  ssh gpu "cp ~/aic/results/scoring.yaml ~/aic/results/scoring_seed_$i.yaml"
done
```

Report mean AND range. A score bump of <5 points across seeds is noise.

## After EVERY eval, you MUST:
1. Parse scoring.yaml for per-trial breakdown
2. Update the experiment bead with results
3. Compare against best known score (check `.claude/refs/experiment-log.md`)
4. Analyze what worked/didn't
5. Decide: KEEP or DISCARD (≥5 pt mean improvement = KEEP)

## NEVER use Docker compose for iteration eval
Docker compose is ONLY for pre-submission verification.
Docker and pixi give different results due to:
- Different data distribution when collecting training data via Docker
- Different sim timing/sync behavior
- Different random seeds by default

## Residual overlay eval (Phase 2c)

Residual tunable env vars (see RunACTResidual.py docstring for full list):

```bash
RESIDUAL_ENABLED=1 \
RESIDUAL_FZ_THRESHOLD=2.0 \
RESIDUAL_STALL_VZ_THRESHOLD=0.002 \
RESIDUAL_SPIRAL_RADIUS=0.003 \
POLICY=aic_example_policies.ros.RunACTResidual ~/aic/bin/run-eval.sh
```

Ablate residual on/off with the same seed to isolate the overlay's contribution.
