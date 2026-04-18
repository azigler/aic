---
name: residual
description: Classical spiral-search + F/T-gated overlay on top of the ACT policy for the last-cm insertion problem.
argument-hint: "[tune|disable|enable]"
---

# /residual - Spiral-Search Residual Overlay

## What this is

A classical (no-ML) overlay on top of RunACTLocal that fires when the ACT
policy has the end-effector near the port but isn't making progress:

- **Contact detected**: `|wrist_wrench.force|` ≥ threshold (default 2.0 N)
- **Vertical stall**: `|dz/dt|` averaged over 0.5s is below threshold (default 2 mm/s)

When both hold, the overlay adds a parametric spiral (small lateral motion)
+ downward bias + rotational wiggle to the ACT velocity command. This uses
the ATI F/T sensor signal that the ACT policy currently ignores entirely.

**Code**: `aic_example_policies/aic_example_policies/ros/RunACTResidual.py`

**Rationale**: AugInsert (IROS 2025) found F/T is the most informative
modality for contact-rich assembly. The residual is the cheapest way to use
that signal without a policy retrain.

## Activation

```bash
POLICY=aic_example_policies.ros.RunACTResidual MODEL_PATH=... /eval
```

All behavior is inherited from RunACTLocal — same ACT weights, same
observation construction, same control loop rate.

## Tuning knobs (env vars)

| Var | Default | Notes |
|-----|---------|-------|
| `RESIDUAL_ENABLED` | 1 | Set to 0 for clean ablation against RunACTLocal |
| `RESIDUAL_FZ_THRESHOLD` | 2.0 N | Lower → engages sooner (more false positives on pre-contact) |
| `RESIDUAL_STALL_WINDOW` | 0.5 s | Averaging window for z-speed |
| `RESIDUAL_STALL_VZ_THRESHOLD` | 0.002 m/s | Lower → stricter stall definition |
| `RESIDUAL_SPIRAL_RADIUS` | 0.003 m | 3mm radius; port tolerance is ~1-2mm |
| `RESIDUAL_SPIRAL_RATE` | 1.0 Hz | Faster → more exploration per second but less dwell time per point |
| `RESIDUAL_DOWNWARD_BIAS` | 0.005 m/s | Always pushes into the port |
| `RESIDUAL_WIGGLE_AMPLITUDE` | 0.05 rad/s | Breaks static friction around Z axis |

## Experiment protocol

1. **Ablate baseline**: 3-seed eval with `RESIDUAL_ENABLED=0` → re-confirm v5 range
2. **Ablate enabled default**: 3-seed eval with `RESIDUAL_ENABLED=1` and defaults
3. If improved, sweep `RESIDUAL_FZ_THRESHOLD` ∈ {1.0, 2.0, 4.0} to find the
   engagement point that maximizes mean score without triggering on approach
4. Then sweep `RESIDUAL_SPIRAL_RADIUS` ∈ {0.002, 0.003, 0.005}

## Failure modes to watch for

- **Engaging during approach**: log lines will show "Residual ENGAGED" too early
  (before policy gets near port). Fix: raise FZ threshold.
- **Oscillation into penalty**: force spikes in scoring.yaml. Fix: lower
  spiral radius or downward bias.
- **Never engaging**: no "Residual ENGAGED" log lines. Probably threshold too
  high or policy isn't reaching contact.

## Integration with RunACT.py submission

If this wins, update the Docker submission `CMD` in
`docker/aic_model/Dockerfile` to point to `RunACTResidual`. The env-var
defaults should stay tuned for the best-performing configuration.
