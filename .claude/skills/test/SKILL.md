---
description: Testing policies via simulation trials and unit tests
---

# Test Workflow

Testing in this project means two things: simulation-based evaluation (primary)
and unit tests for utility code (secondary).

## Simulation Testing (Primary)

The real test is: does the policy insert cables successfully?

### Quick Smoke Test

```bash
# Does the policy load and pass Tier 1?
pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true -p policy:=<your_package>.<YourPolicy>
```

### Full Evaluation

See `/eval-policy` skill for running 3-trial evaluations with scoring.

### Test Matrix

| Test | What It Validates | Expected Score |
|------|-------------------|---------------|
| Tier 1: loads | Policy activates, accepts goals | 1 |
| Tier 1: commands | Sends valid MotionUpdate msgs | 1 |
| SFP insertion | Trial 1 or 2 | 75+ |
| SC insertion | Trial 3 | 75+ |
| Randomized board | Different configs | Consistent |
| Force safety | No >20N sustained | No penalty |
| Collision safety | No enclosure contact | No penalty |

### Regression Testing

After any policy change:
1. Run full 3-trial eval
2. Compare scores against previous best
3. Check for regressions in any trial

## Unit Testing (Secondary)

For utility code (data processing, perception, etc.):

```bash
pixi run pytest <test_file>
```

### What to Unit Test

- Image preprocessing functions
- Pose transformation utilities
- Reward/score computation helpers
- Data loading and augmentation
- Neural network forward pass (shape checks)

### What NOT to Unit Test

- The full policy loop (use simulation instead)
- ROS message construction (too tightly coupled)
- Controller behavior (that's the eval container's job)

## Bead Protocol

```bash
br create -p 2 "test: smoke test Tier 1 validation"
br create -p 2 "test: full 3-trial eval on randomized configs"
```
