# Experiment Log

Running summary of all experiments. Updated after each `/experiment log`.

## Score Leaderboard

| Rank | Experiment | Total | Branch | Date | Notes |
|------|-----------|-------|--------|------|-------|
| 1 | exp-015 IBVS lucky run | 110.4 | B1 | 2026-04-10 | Lucky alignment, not reliable |
| 2 | exp-014 IBVS v3 | 101.2 | B1 | 2026-04-09 | Near-reliable ~100 |
| 3 | exp-013 IBVS v2 | 98.7 | B1 | 2026-04-09 | Consistent ~95-100 |
| 4 | exp-012 IBVS v1 | 96.1 | B1 | 2026-04-08 | First camera-based improvement |
| 5 | exp-004 DirectApproach v2 | 93.4 | A1 | 2026-04-04 | Best classical |
| 6 | exp-011 Template matching v3 | 91.8 | B2 | 2026-04-07 | Inconsistent |
| 7 | exp-006 DirectApproach v4 | 89.3 | A1 | 2026-04-04 | |
| 8 | exp-010 Template matching v2 | 87.5 | B2 | 2026-04-07 | |
| 9 | exp-016 IBVS + force | 86.3 | B1 | 2026-04-10 | Force feedback hurt |
| 10 | exp-005 DirectApproach v3 | 83.3 | A1 | 2026-04-04 | |
| 11 | exp-009 Color segmentation | 82.1 | B3 | 2026-04-06 | Unreliable detection |
| 12 | exp-007 DirectApproach v5 | 81.6 | A2 | 2026-04-04 | |
| 13 | exp-008 XY correction | 79.9 | A2 | 2026-04-05 | |
| 14 | exp-003 DirectApproach v1 | 78.4 | A1 | 2026-04-04 | |
| 15 | exp-017 IBVS tuned gains | 76.2 | B1 | 2026-04-11 | Overtuned |
| 16 | exp-018 Stereo depth | 72.0 | B2 | 2026-04-11 | Depth estimation noisy |
| 17 | exp-019 Multi-camera fusion | 68.4 | B2 | 2026-04-12 | Too slow at 20Hz |
| 18 | exp-020 IBVS + spiral | 65.1 | B1 | 2026-04-12 | Spiral search hurt timing |
| 19 | exp-021 Adaptive stiffness | 61.8 | B1 | 2026-04-13 | Unstable |
| 20 | exp-022 SC-specific tuning | 58.3 | B1 | 2026-04-13 | SC still problematic |
| 21 | exp-023 Camera crop ROI | 55.0 | B2 | 2026-04-14 | Faster but less accurate |
| 22 | exp-024 Reduced latency | 52.7 | B1 | 2026-04-14 | Diminishing returns |
| 23 | WaveArm baseline | 42.3 | -- | 2026-04-04 | |
| 24 | exp-001 BlindPush | 3 | A0 | 2026-04-04 | |
| 25 | exp-002 SmartApproach | 3 | A1 | 2026-04-04 | |

## Current Best

**Score:** 110.4 (IBVS lucky run -- not reliable)
**Reliable range:** ~90-100
**Policy:** IBVS visual servoing (Branch B)
**Next target:** ACT training (Branch C) to break through 100+ reliably

## Branch Status

| Branch | Experiments | Best Score | Status |
|--------|-----------|------------|--------|
| A: Classical | 8 | 93.4 | Plateau -- exhausted |
| B: Camera Perception | 16 | 110.4 (lucky) / ~100 reliable | Plateau -- exhausted |
| C: ACT Training | 0 | -- | **Active -- primary approach** |
| D: RL | 0 | -- | Not started (fallback) |

## Key Learnings

- **Z descent works for SFP but not SC.** SFP ports are tall (Z~0.133) and forgiving; SC ports are flush (Z~0.015) and require precise XY alignment before any Z motion.
- **Hardcoded offsets don't generalize to randomized configs.** The task board position/orientation is randomized each trial. Ground-truth TF offsets only work for one config.
- **Force baseline ~21N from cable weight must be compensated.** The FT sensor reads ~21N in Z even when hovering due to cable and gripper weight. Force control thresholds must account for this offset.
- **Speed matters: faster = higher Tier 2 score.** Duration scoring rewards fast completion (0-12 pts). Minimizing approach time directly improves T2.
- **XY alignment is the bottleneck (~5cm offset).** Classical approaches plateau because they cannot correct XY misalignment without visual feedback. The remaining error is too large for blind insertion.
- **Port positions from ground truth:** SFP Z~0.133, SC Z~0.015. These heights are reliable but XY positions vary with board randomization.
- **Camera perception plateaued at ~100.** IBVS and template matching improved over classical but hit a ceiling. The 110.4 score was a lucky alignment, not reproducible.
- **Branch B diminishing returns.** Experiments 017-024 all scored below the best, indicating the camera perception approach has been fully explored.
- **ACT is the next breakthrough path.** End-to-end learned policies can handle the full complexity of the task (perception + insertion + force control) without manual tuning of each component.
