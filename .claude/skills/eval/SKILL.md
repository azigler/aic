---
name: eval
description: Run policy evaluation on GPU via distrobox+pixi. NEVER use Docker compose for iteration.
argument-hint: "[policy_class]"
---

# /eval - Run Policy Evaluation

## Quick Eval (distrobox + pixi on GPU)

```bash
# 1. Rsync code to GPU
rsync -az --exclude='.git' --exclude='.pixi' --exclude='__pycache__' \
  /home/ubuntu/aic/ gpu:~/ws_aic/src/aic/

# 2. Reinstall pixi package (REQUIRED after code changes)
ssh gpu "export PATH=\$HOME/.pixi/bin:\$PATH && cd ~/ws_aic/src/aic && \
  pixi reinstall ros-kilted-aic-example-policies 2>&1 | tail -3"

# 3. Run eval
ssh gpu "export PATH=\$HOME/.pixi/bin:\$PATH && cd ~/ws_aic/src/aic && \
  POLICY=aic_example_policies.ros.RunACT ~/run-eval.sh"

# 4. Read results
ssh gpu "cat ~/aic_results/scoring.yaml"
```

## After EVERY eval, you MUST:
1. Parse scoring.yaml for per-trial breakdown
2. Update the experiment bead with results
3. Compare against best known score
4. Analyze what worked/didn't
5. Decide: KEEP or DISCARD

## NEVER use Docker compose for iteration eval
Docker compose is ONLY for pre-submission verification.
