# Scoreboard — Strategy Predictions vs. Actual

This file tracks the score predictions I (Opus 4.7, entered session on 2026-04-18)
made before running experiments, so future-me can calibrate.

## Baseline

- **v5** (pre-upstream-sync): 110–170/300 range across runs, median ~140
- **v5** (post-friction-sync): partial insertion at 1cm on SC trials
- **v5 Docker submission**: 124.2/300 (verified)

## Plan predictions (committed 2026-04-18)

| Phase | Change | Predicted range | Actual | Grounded? |
|-------|--------|-----------------|--------|-----------|
| 1a | Fix image-stats squeeze bug (defensive only — my read says it was a false alarm from the audit, but the check prevents future regression) | +0–5 | | |
| 1b | Add val split + early stopping; retrain v5 recipe → **v13** | +10–30 (mostly from avoiding silent overfit; expected v13 near the top of v5's range) | | |
| 2a | BID wrapper (N=8 chunks, backward-coherence) at inference | +15–30 | | |
| 2b | Temporal ensemble λ/k sweep on v5 | +5–15 | | |
| 2c | Classical spiral residual (RunACTResidual) gated by F/T + z-stall | +20–50 (last-cm is our bottleneck; residual targets it directly) | | |
| 3 | Add wrist_wrench to observation (state 26→32), retrain → **v14** | +15–40 | | |
| 4 | JUICER bottleneck augmentation | +10–25 | | |

**Cumulative expected ceiling** if Phases 1–3 land at their midpoints:
140 + 2 + 20 + 25 + 10 + 35 + 25 = ~260/300. This is optimistic — each
phase will likely land below its midpoint, and effects are not purely
additive. Realistic composite: 200–220/300.

**Submission threshold self-target**: 200/300 mean (Docker-verified).

## How I'll update this

After each experiment lands, append Actual + a one-sentence honest note on
whether my estimate was grounded. The goal is to calibrate: if I'm always
optimistic by 2x, future plans should be discounted.

## Risks that can invalidate the plan

- Residual overlay might engage during approach (pre-contact F/T noise) →
  score could regress. Mitigation: the `RESIDUAL_ENABLED=0` A/B ablation is
  the first sanity check.
- v13 retrain could reveal v5 was genuinely overfit and was carried by lucky
  trial configs → new val-loss-selected checkpoint could score below v5.
  Mitigation: this is the right answer if true; keep v5 as submission fallback.
- Adding wrist_wrench to observation requires re-collecting training data
  that includes the wrench field OR backfilling from existing episodes if
  the raw data has it. Need to check `~/aic/data/velocity/episode_*/`
  contents before committing to Phase 3.
