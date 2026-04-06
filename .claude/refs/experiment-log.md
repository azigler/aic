# Experiment Log

Running summary of all experiments. Updated after each `/experiment log`.

## Score Leaderboard (Top 15)

| Rank | Experiment | Total | Branch | Notes |
|------|-----------|-------|--------|-------|
| 1 | exp-035 Position ACT (24 demos, 8 configs, 50ep) | 136.0 | C | **BEST** — position-mode breakthrough |
| 2 | exp-034 Position ACT micro (9 demos, 20ep) | 131.1 | C | Position mode proof-of-concept |
| 3 | exp-029 ACT v6 (27 demos, 8 configs, 100ep) | 121.9 | C | Best velocity-mode ACT |
| 4 | exp-027 ACT v4 diverse (6 demos, 2 configs) | 120.7 | C | First diverse ACT |
| 5 | exp-017c IBVS (lucky run) | 110.4 | B | Lucky alignment, not reliable |
| 6 | exp-024 OurACT micro (3 demos, 5ep) | 101.9 | C | First ACT working |
| 7 | exp-018b Averaged IBVS | 100.1 | B | Best reliable camera |
| 8 | exp-010 VisionPolicy color | 98.5 | B | Color detection breakthrough |
| 9 | exp-004 DirectApproach v2 | 93.4 | A | Best classical |
| 10 | exp-015 IBVS larger spiral | 85.0 | B | |
| 11 | exp-003 DirectApproach v1 | 78.4 | A | First proximity points |
| 12 | WaveArm baseline | 42.3 | -- | |
| 13 | exp-001 BlindPush | 3.0 | A | Force threshold too low |

## Current Best (Docker Verified)

**Score:** 118.6/300 (position-mode ACT + temporal ensembling, exp-044)
**Docker baseline:** 114.4 (fixed argument order + chunk indexing bugs)
**Model:** ~/models/act_pos_50ep on GPU (24 demos, 8 configs, 50 epochs)
**Bottleneck:** ~5cm XY gap to port. No full insertions yet (75 pts/trial).

Note: Previous 136.0 claim was from pixi eval with broken argument order.
Docker eval with fixed code establishes 114.4 as the true baseline.

## Session 2 Experiments (Docker-verified)

| Exp | Description | Score | vs Baseline | Verdict |
|-----|-------------|-------|-------------|---------|
| 042 | Docker baseline (fixed bugs) | 114.4 | — | BASELINE |
| 044 | Temporal ensembling | **118.6** | **+4.2** | **KEPT** |
| 039 | Offset actions | 111.6 | -2.8 | DISCARD |
| 043 | Data augmentation | 88.4 | -26.0 | DISCARD |
| 045 | Adaptive stiffness | 111.5 | -2.9 | DISCARD |
| 046 | 120s time limit | 86.3 | -28.1 | DISCARD |
| 047 | Bottleneck oversampling | 106.1 | -8.3 | DISCARD |
| 048 | More diverse data | in progress | — | COLLECTING |

## Branch Status

| Branch | Experiments | Best Score | Status |
|--------|-----------|------------|--------|
| A: Classical | 8 | 93.4 | Plateau — exhausted |
| B: Camera Perception | 16 | 110.4 (lucky) / ~100 reliable | Plateau — exhausted |
| C: ACT Training | 15 | 136.0 (position mode) | **Active — primary approach** |
| D: RL | 0 | -- | Not started (fallback) |

## ACT Progression (Branch C)

| Phase | Score Range | Key Change |
|-------|-----------|------------|
| Micro-train (3 demos) | 101.9 | Proved pipeline works |
| Diverse data (2 configs) | 120.7 | Config diversity >> quantity |
| Diverse data (8 configs) | 121.9 | Marginal over 2 configs |
| **Position mode** | **131-136** | **Biggest single improvement** |
| Overfitting experiments | 120.9 | More data/epochs hurts |

## Key Learnings

### Position-Mode ACT Breakthrough
- Switching from velocity to position actions: val_loss dropped 50-100× (0.29 vs 14-27)
- Score jumped from ~120 to 131-136
- Model predicts absolute TCP target poses matching CheatCode's output
- Sweet spot: 24 demos from 8 configs, 50 epochs. More = overfitting.

### Data & Training
- **Config diversity > quantity**: Each new board config helps more than more demos from same config
- **50 epochs is the sweet spot** for 24 demos. >50 overfits (train=0.08 vs val=0.17)
- **Score variance: ~15 points** (120-136 from randomization). Need 3-5 runs.
- **Weight decay** didn't help (exp-038: 120.9 = variance, not improvement)

### Infrastructure
- **ALWAYS nohup** for GPU training >10 min (SSH timeout killed epoch 70/100)
- **Docker ACT_MODEL_DIR** must be verified before every eval
- **Docker --no-cache** if code changes aren't reflected
- **Never distrobox over SSH** for eval — use run-eval.sh or docker compose

## Active Experiments

- **exp-039** (P1): Offset actions — predict TCP delta instead of absolute position
- **exp-040** (P1): Higher image resolution (384×384)
- **exp-041** (P2): Task encoding in observation
- **exp-042** (P2): Variance baseline (5 runs)
