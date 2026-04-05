# Autoresearch Methodology (Applied to AIC)

Adapted from https://github.com/karpathy/autoresearch

## Core Loop (HARD RULES)

### Classical / Camera Experiments (~10 min per cycle)

```
LOOP FOREVER:
1. Read git state + experiments.tsv (what's best, what's been tried)
2. Formulate hypothesis (one sentence: "X will improve because Y")
3. Edit policy code (ONE change per experiment)
4. git commit (with hypothesis as message)
5. scripts/autorun.sh <policy> (run 3 trials, ~7 min)
6. Parse score from aic_results/scoring.yaml
7. If score > best: KEEP (update experiments.tsv, git push)
8. If score < best: DISCARD (git reset --hard HEAD~1)
9. NEVER pause, ask human, or wait for permission
10. If stuck (5+ discards in a row): try radical change or pivot branch
```

### ACT Training Experiments (~2-3 hours per cycle)

```
LOOP FOREVER:
1. Read prior ACT experiment results (training configs + eval scores)
2. Formulate hypothesis about training config change
3. Adjust training config (ONE variable: chunk size, LR, demos, etc.)
4. Collect demos if needed (scripts/collect_demos.sh, ~30 min)
5. Train model (scripts/train_act.py, ~1-2 hours)
6. Evaluate trained model (scripts/remote-eval.sh, ~10 min)
7. Parse score from aic_results/scoring.yaml
8. If score > best: KEEP (save model, update experiments.tsv)
9. If score < best: DISCARD (revert config change, try different variable)
10. If stuck (3+ discards in a row): try different experiment family
```

Note: ACT iteration changes training configs, not policy code. The RunACT policy
stays fixed. The "code" being modified is the training hyperparameters.

## Decision Rules

- **KEEP:** Score improved by ≥ 1.0 point over current best
- **KEEP_EXPERIMENTAL:** Score within 0.5 of best, novel approach (tag but don't update best)
- **DISCARD:** Score declined. Revert immediately.

## Crash Triage

- Trivial fix (typo, import): Fix and retry ONCE
- Resource issue (OOM, timeout >15 min): Mark "crash", discard
- Logic bug: Investigate ≤5 min. If unclear, discard and try different approach
- Maximum 2 retries per experiment

## Experiment Families

Don't try random ideas. Batch into families:

```
Family: Force Threshold Sweep
  - exp-A: threshold=12N → score X
  - exp-B: threshold=15N → score Y
  - exp-C: threshold=18N → score Z
  → Pick best, move to next family
```

## Convergence Detection

- Branch A (Classical): Plateaued at 93.4 after 8 experiments -- DONE
- Branch B (Camera): Plateaued at ~100 after 16 experiments -- DONE
- Branch C (ACT): Active. Each cycle is ~2-3 hours (collect + train + eval)
- Global: If May 10 and best < 150 → radical reset (try RL in Isaac Lab)
- If best > 200 → submit daily and optimize

## experiments.tsv Format

```
commit	score	time_s	status	branch	description
```

Status values: best, keep, keep_experimental, discard, crash, timeout

## Key Differences from Karpathy's Setup

| Aspect | Autoresearch | Our Setup |
|--------|-------------|-----------|
| Time per experiment | 5 min fixed | ~7 min (classical) / ~2-3 hrs (ACT) |
| Metric | val_bpb (lower=better) | score 0-300 (higher=better) |
| Code to modify | train.py only | Training config (ACT) or policy code (classical) |
| Cost | Free (local GPU) | $1/hr cloud GPU |
| Deadline | None | May 15, 2026 |
