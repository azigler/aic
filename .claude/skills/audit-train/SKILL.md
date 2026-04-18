---
name: audit-train
description: Pre-train sanity check on scripts/train_act.py. Catches regressions to the pipeline hygiene fixed after v5.
---

# /audit-train - Training Script Audit

Run this BEFORE every training experiment. It checks that the training
script still has the hygiene that made v5 reproducible: validation split,
early stopping on val loss, fixed seed, no default insertion weight > 1,
correct stats shape on save, gradient clipping.

## Checklist

Read `scripts/train_act.py` and verify each of the following holds. If any
fails, fix before training. History of past regressions that caused the
checklist to be needed in the first place is in square brackets.

### Data & stats

- [ ] `VelocityDemoDataset.__init__` default `insertion_weight=1.0`
      [exp-047/059/060 all regressed at >1.0]
- [ ] `_build_sample_index` early-returns when `insertion_weight == 1.0`
      (no duplicated entries)
- [ ] Image stats saved with shape `(1, 3, 1, 1)` and `.numel() == 3` asserted
      in `_save_checkpoint`
- [ ] `state_std`/`action_std` clamped at 1e-6 minimum to avoid divide-by-zero
- [ ] `_compute_image_stats` samples across episodes (not just first N steps)

### Split & val loop

- [ ] `_split_episodes(val_frac, seed)` produces episode-level split, deterministic
- [ ] `val_dataset.set_preprocess_stats(...)` applied with train-set stats
      (val must NOT recompute its own stats — creates distribution mismatch)
- [ ] `_evaluate(policy, val_loader, device)` uses `torch.no_grad()` AND
      `policy.eval()` / `policy.train()` bracket
- [ ] Early stopping uses `val_loss < best_val_loss - min_delta` not
      `<=` (otherwise every zero-improvement epoch resets patience)

### Optimization

- [ ] `torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)`
      present in train step
- [ ] `optimizer.zero_grad()` called before `loss.backward()`
- [ ] `args.lr` default = 5e-6 (v5 recipe)
- [ ] `args.weight_decay` default = 1e-4

### Reproducibility

- [ ] `_seed_everything(args.seed)` called first in `train()`
- [ ] `args.seed` default = 42
- [ ] Splits pass `seed` to `random.Random` (not global state)

### Output

- [ ] `best/` checkpoint saved on val improvement, not train improvement
- [ ] `training_history.json` written at end with (epoch, train_loss, val_loss, time_s) per epoch
- [ ] All outputs under user-provided `--output-dir`, not hardcoded paths

### CLI

- [ ] `--val-frac`, `--patience`, `--min-delta`, `--insertion-weight`,
      `--seed`, `--grad-clip` are ALL CLI args (not hardcoded)

## Quick run

```bash
cd /home/ubuntu/aic
ruff check scripts/train_act.py
python3 -c "import ast; ast.parse(open('scripts/train_act.py').read()); print('syntax OK')"
grep -E "insertion_weight: float = 1.0|patience|min_delta|val_frac" scripts/train_act.py
```

Fail fast: if this skill's script grep doesn't show those flags, do NOT start training.
