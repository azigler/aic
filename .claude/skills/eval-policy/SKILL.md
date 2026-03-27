---
name: eval-policy
description: Evaluate cable insertion policies -- run trials, parse scores, compare experiments
argument-hint: "[run|parse|compare|report]"
---

# /eval-policy - Policy Evaluation

Run, score, and compare cable insertion policy experiments.

## Quick Eval

### Run a policy through 3 qualification trials

```bash
# Terminal 1: Sim + engine (set unique results dir)
AIC_RESULTS_DIR=~/aic_results/<experiment> \
distrobox enter -r aic_eval -- /entrypoint.sh \
  start_aic_engine:=true

# Terminal 2: Your policy
pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=<your_package>.<YourPolicy>
```

### Parse Scoring Results

Results are in `$AIC_RESULTS_DIR/scoring.yaml`.

```bash
cat ~/aic_results/<experiment>/scoring.yaml
```

### Score Breakdown (per trial)

```yaml
# Expected structure in scoring.yaml:
trial_N:
  tier_1:
    model_validity: 0 or 1
  tier_2:
    trajectory_smoothness: 0-6
    task_duration: 0-12
    trajectory_efficiency: 0-6
    insertion_force_penalty: 0 to -12
    off_limit_contacts_penalty: 0 to -24
  tier_3:
    cable_insertion: -12 to 75
  total: <sum>
```

## Evaluation Protocol

### 1. Local Development Eval

Fast iteration loop:
1. Fix task board config (deterministic)
2. Run policy
3. Parse score
4. Modify policy
5. Repeat

```bash
# Fixed config for deterministic testing
AIC_RESULTS_DIR=~/aic_results/dev_$(date +%Y%m%d_%H%M%S)
```

### 1b. Remote Eval (Mac Studio)

Faster iteration via remote Mac Studio runner (~10-12 min vs ~27 min local):

```bash
scripts/remote-eval.sh <your_package>.<YourPolicy>
```

Results land in `./aic_results/scoring.yaml` (rsynced back from Mac).

Remote and local (Docker) results should be comparable -- physics and scoring are
identical. If you see discrepancies, verify with a Docker run before drawing
conclusions. Always use Docker for pre-submission verification.

### 2. Randomized Eval

Test generalization:
1. Run with default engine config (randomized board pose)
2. Run multiple times
3. Average scores across runs
4. Look for failure modes

### 3. Pre-Submission Eval

Full eval matching cloud conditions:
1. Build Docker container
2. Run via docker-compose
3. Verify all 3 trials complete
4. Check scores match local results

```bash
# Build
docker compose -f docker/docker-compose.yaml build model

# Run full evaluation
docker compose -f docker/docker-compose.yaml up
```

## Comparing Experiments

### Score Table

```
| Experiment | Trial 1 | Trial 2 | Trial 3 | Total |
|------------|---------|---------|---------|-------|
| baseline   |    45   |    42   |    38   |  125  |
| exp_002    |    72   |    68   |    55   |  195  |
| exp_003    |    85   |    82   |    70   |  237  |
```

### Key Metrics to Track

- **Insertion rate:** What fraction of trials achieve full insertion (75pts)?
- **Partial vs miss:** When it fails, how close does it get? (proximity score)
- **Safety:** Any force or contact penalties? These are -12 and -24 respectively.
- **Efficiency:** How fast? (duration score is 0-12, <=5s = max)
- **Generalization:** Do scores hold across randomized configs?

### Failure Analysis

When a trial scores low, diagnose:

1. **Score 0:** Policy didn't move toward target. Check perception.
2. **Score 0-25 (proximity only):** Got close but missed. Check alignment.
3. **Score 25-50 (partial insertion):** Almost! Check insertion force/angle.
4. **Score 75 but low Tier 2:** Inserted but messy. Optimize trajectory.
5. **Negative penalty:** Collision (-24) or force (-12). Check safety bounds.
6. **Wrong port (-12):** Perception error. Check port identification.

## Baseline Scores

| Policy | Trial 1 (SFP) | Trial 2 (SFP) | Trial 3 (SC) | Description |
|--------|---------------|---------------|--------------|-------------|
| WaveArm | ~1 | ~1 | ~1 | Just waves, no insertion |
| CheatCode | ~80 | ~80 | ~80 | Ground truth, reference score |

CheatCode is the upper bound for what's achievable. Your policy should aim to
match or exceed its insertion rate while being robust to randomization.

## Cloud Evaluation

- **Hardware:** 64 vCPU, 256 GiB RAM, 1x NVIDIA L4 (24GB VRAM)
- **Submission:** Docker container to Amazon ECR
- **Limit:** 1 submission per day
- **Scoring:** Same engine, same config, same metrics

### Pre-Submission Checklist

- [ ] Policy loads without errors (Tier 1 passes)
- [ ] All 3 trials complete without timeout
- [ ] No force penalties (>20N for >1s)
- [ ] No off-limit contact penalties
- [ ] Insertion succeeds on at least some trials
- [ ] Docker container builds and runs locally
- [ ] Results match between pixi and Docker runs
