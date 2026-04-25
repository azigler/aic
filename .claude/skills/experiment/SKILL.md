---
name: experiment
description: The core experiment loop. Every change goes through this.
argument-hint: "[propose|run|log|next]"
---

# /experiment - Experiment Loop

## Pre-flight checks (BEFORE creating a new experiment bead)

1. **Check experiment-log.md and bead history** for the hypothesis. If we've
   already tried it and it regressed, the answer is in `.claude/refs/experiment-log.md`.
2. **Verify we're changing ONE variable** from the known-good baseline (v5):
   66 episodes, 100 epochs, lr=5e-6, batch=8, insertion_weight=1.0,
   velocity-mode data, high-friction, 30s trial time limit.
3. **GPU status** — if suspended, note the ~$1/hr cost of resuming and confirm
   with user before starting long runs. Workspace lives under `~/aic/`
   (data, models, results all under that root on both local and GPU).
4. **No mixing data distributions** — never combine Docker-collected with
   pixi-collected, never combine high-friction with low-friction.

## BEFORE starting ANY experiment:

1. Create a bead with hypothesis:

```bash
br create -p 1 "exp-NNN: brief description"
br update <id> --description "## Hypothesis
[what and why]
## Baseline
[v5 at 110-170/300, median 140, Docker 124.2/300]
## Variable changed
[exactly one — e.g., added wrist_wrench to observation state 26→32]
## Expected Outcome
[target score range with reasoning]"
br update <id> --status=in_progress
```

## Running an experiment:

1. Make the code change locally (one variable!)
2. Rsync to GPU under `~/aic/src/`
3. Run /eval — ALWAYS 3 seeds for variance
4. Parse `~/aic/results/scoring.yaml` for mean + range
5. Log immediately to the bead and experiment-log.md

## AFTER every experiment:

1. Update bead with FULL results:

```bash
br update <id> --description "[existing content]
## Results
| Seed | Score | Trial 1 | Trial 2 | Trial 3 |
| 1 | ... | ... | ... | ... |
| 2 | ... | ... | ... | ... |
| 3 | ... | ... | ... | ... |
Mean: X.X, Range: [Y, Z]
## Analysis
[why it worked/didn't — include v_loss curve summary for train experiments]
## Next
[what specifically to try next]"
```

2. Update `.claude/refs/experiment-log.md` with a new section
3. **Compare to your pre-experiment score estimate.** Note if grounded or off.
4. Commit with gitmoji + bead trailer (`Bead: bd-xxx`)
5. If KEEP: update baseline in /train and /experiment skills. If DISCARD:
   revert code changes, close bead with DISCARD verdict.

## Decision rules:

- **KEEP:** Mean score improved by ≥5 points over baseline across 3 seeds
- **DISCARD:** Mean score declined or within noise. Revert immediately.
- **PIVOT:** 3 consecutive discards from the same hypothesis family → try a
  fundamentally different approach
- **STOP:** Mean score ≥200 → prepare submission

## Known failed hypotheses (DO NOT re-run without a new angle)

- More epochs than 100 on 66-episode dataset (v6 @ 200 epochs → 83.9)
- Any insertion_weight > 1.0 (1.5x, 3x, 3x-with-decay all regressed)
- Low-friction-only training (exp-061 → 60)
- Combined high+low friction training (exp-062 → 50)
- Image augmentation (exp-043 → 88.4)
- 120s time limit (exp-046 → 86.3; oscillation penalties accumulate)
- P-controller velocity in V2 collector (exp-057 → 3; velocity 10× too large)
- Lerobot controller params (exp-052 → 39.6; RunACT params are better)
