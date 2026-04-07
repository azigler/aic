# Experiment Log

## Session 1 Summary (Previous Agent)

Explored 3 branches: Classical (A), Camera Perception (B), ACT Training (C).
Branches A and B plateaued. Branch C (ACT) became primary approach.

Best claimed score: 136/300 (position-mode ACT, 24 demos, 8 configs, 50 epochs).
However: this was through pixi eval with a custom SimpleACT model that had
argument order bugs. The true score is uncertain.

## Session 2 Summary (This Agent)

Ran 10 experiments, mostly regressions due to:
1. Using Docker compose for eval instead of distrobox+pixi
2. Custom SimpleACT model had bugs (argument order, import path)
3. Docker-collected training data had different distribution

**Key finding:** We should use the competition's LeRobot ACTPolicy (RunACT.py)
instead of our custom SimpleACT. This is the fundamental reset for Session 3.

## Session 3: Velocity-Mode Fine-Tuning

### Infrastructure Fixes
- Fixed run-eval.sh timing: start policy BEFORE eval container (not after)
- Discovered `lerobot_robot_aic` — competition's LeRobot integration (missed in S1/S2)
- Created DataCollector.py wrapping CheatCode for automated demo collection
- Created RunACTLocal.py for evaluating local fine-tuned weights
- Created train_act.py for fine-tuning from pretrained HuggingFace weights

### Baseline Results
- Pretrained RunACT (HuggingFace): **-21/300** (gets ~16-20cm from port)
- CheatCode (ground truth): **~221/300** (successful insertions)

### exp-050: First velocity-mode fine-tune (21 episodes, 50 epochs)
- Score: **86.1/300** (+107 from pretrained)
- Trial 1 (SFP): 44 pts, 5cm from port, proximity points
- Trial 3 (SC): 41 pts, 3cm from port, nearly inserted
- Trial 2: regression (14cm), config not well represented

### exp-051: Training v2 (36 episodes, 50 epochs)
- Score: **96.8/300** — trial 2 fixed (was 1.0 in v1)
- All trials getting 3-5cm from ports

### exp-052: Controller params experiment — DISCARD
- lerobot params (high damping, no wrench): 39.6 — worse
- RunACT original params are better for our model

### exp-053: v2 with 30s limit (best config)
- Score: **118.4/300** — 30s better than 60s (trajectory timing)

### exp-054: v4 model (66 episodes, 50 epochs)
- Score: **126.8/300** — SC trial improved to 39 pts
- All 3 trials now consistently 5cm from port (25 tier3 each)
- More data → better generalization across configs

### exp-055: v5 (66 eps, 100 epochs, lr=5e-6) — PARTIAL INSERTION!
- Score: **152.9/300** — SFP trials achieved PARTIAL INSERTION (38 tier3)
- Key: 100 epochs + lower lr pushed through the 5cm barrier

### exp-056: v6 (66 eps, 200 epochs, lr=5e-6) — OVERFITTING
- Score: **83.9/300** — REGRESSION, trial 1 went to 14cm
- Confirmed: 200 epochs on 66 episodes is too much
- Lesson: need MORE DATA not more epochs

### Current best: v5 at 152.9/300 (2 partial insertions + 1 proximity)
Next: collect ~120+ episodes total, retrain at 100 epochs
