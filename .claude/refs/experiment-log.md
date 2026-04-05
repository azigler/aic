# Experiment Log

Running summary of all experiments. Updated after each `/experiment log`.

## Score Leaderboard

| Rank | Experiment | Total | T1 | T2 | T3 | Branch | Date |
|------|-----------|-------|----|----|----|---------| -----|
| 1 | exp-004 DirectApproach v2 | 93.4 | 3 | 32.4 | 38.4+16.5+0 | A1 | 2026-04-04 |
| 2 | exp-006 DirectApproach v4 | 89.3 | 3 | ~30 | ~56 | A1 | 2026-04-04 |
| 3 | exp-005 DirectApproach v3 | 83.3 | 3 | ~25 | ~55 | A1 | 2026-04-04 |
| 4 | exp-007 DirectApproach v5 | 81.6 | 3 | ~28 | ~50 | A2 | 2026-04-04 |
| 5 | exp-008 XY correction | 79.9 | 3 | ~25 | ~52 | A2 | 2026-04-05 |
| 6 | exp-003 DirectApproach v1 | 78.4 | 3 | 29 | 46.4 | A1 | 2026-04-04 |
| 7 | WaveArm baseline | 42.3 | 3 | 21.2 | 18.1 | -- | 2026-04-04 |
| 8 | exp-001 BlindPush | 3 | 3 | 0 | 0 | A0 | 2026-04-04 |
| 9 | exp-002 SmartApproach | 3 | 3 | 0 | 0 | A1 | 2026-04-04 |

## Current Best

**Score:** 93.4
**Policy:** exp-004 DirectApproach v2
**Branch:** A1 (Classical Control)

## Branch Status

| Branch | Experiments | Best Score | Status |
|--------|-----------|------------|--------|
| A: Classical | 8 | 93.4 | Plateau -- pivoting to B |
| B: Camera Perception | 0 | -- | Next up |
| C: Hybrid | 0 | -- | Not started |
| D: RL | 0 | -- | Not started |

## Key Learnings

- **Z descent works for SFP but not SC.** SFP ports are tall (Z~0.133) and forgiving; SC ports are flush (Z~0.015) and require precise XY alignment before any Z motion.
- **Hardcoded offsets don't generalize to randomized configs.** The task board position/orientation is randomized each trial. Ground-truth TF offsets only work for one config.
- **Force baseline ~21N from cable weight must be compensated.** The FT sensor reads ~21N in Z even when hovering due to cable and gripper weight. Force control thresholds must account for this offset.
- **Speed matters: faster = higher Tier 2 score.** Duration scoring rewards fast completion (0-12 pts). Minimizing approach time directly improves T2.
- **XY alignment is the bottleneck (~5cm offset).** Classical approaches plateau because they cannot correct XY misalignment without visual feedback. The remaining error is too large for blind insertion.
- **Port positions from ground truth:** SFP Z~0.133, SC Z~0.015. These heights are reliable but XY positions vary with board randomization.
- **Camera perception is needed for the next breakthrough.** To go beyond 93.4, the policy must detect port positions from camera images rather than relying on hardcoded offsets.
