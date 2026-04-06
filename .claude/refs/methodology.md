# AIC Experiment Methodology

## Goal

Qualify for Phase 1 (top 30 of all teams). This requires maximizing total score
across 3 trials (max 300 points). The qualification eval uses the same
`sample_config.yaml` structure but with randomized board poses and port offsets.

## Local Scoring Loop

We can score ourselves **unlimited times locally** before the 1/day cloud submission.
This is our primary iteration tool.

### Two Ways to Score Locally

**Method 1: Pixi + Distrobox (fast iteration)**
```
Terminal 1: distrobox enter -r aic_eval -- /entrypoint.sh start_aic_engine:=true
Terminal 2: pixi run ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=<pkg>.<Class>
```
Results: `~/aic_results/scoring.yaml` (overwritten each run)

**Method 2: Docker Compose (submission-identical)**
```
docker compose -f docker/docker-compose.yaml up
```
This runs eval + model containers with networking, identical to cloud.

### Score Parsing

After each run, parse `scoring.yaml` for per-trial breakdown:
- Tier 1 (validity), Tier 2 (performance), Tier 3 (insertion)
- Total across 3 trials
- Penalty flags (force, contacts)

## Experiment Protocol

Every experiment follows this cycle:

```
HYPOTHESIZE → IMPLEMENT → SCORE → LOG → ANALYZE → ADJUST
     ↑                                                |
     +------------------------------------------------+
```

### 1. HYPOTHESIZE

Before writing code, state:
- **What** you're trying (one sentence)
- **Why** you think it will improve score (rationale)
- **Expected outcome** (target score or specific improvement)

### 2. IMPLEMENT

- Create/modify policy code
- Keep changes small and testable
- One variable at a time when possible

### 3. SCORE

- Run full 3-trial local eval
- Parse scoring.yaml
- Record per-trial and total scores

### 4. LOG

Create a bead for each experiment with results in the description:

```bash
br create -p 2 "exp: [brief description]"
br update <id> --description "$(cat <<'EOF'
## Hypothesis
[What and why]

## Changes
[What was modified]

## Results
| Trial | T1 | T2 | T3 | Total |
|-------|----|----|----| ------|
| 1 (SFP) | X | X | X | X |
| 2 (SFP) | X | X | X | X |
| 3 (SC)  | X | X | X | X |
| **Sum** | | | | **X** |

## Penalties
- Force: [yes/no, details]
- Contacts: [yes/no, details]

## Analysis
[What worked, what didn't, why]

## Next
[What to try next based on this result]
EOF
)"
```

### 5. ANALYZE

Compare against previous best:
- Did total score improve?
- Which trials improved/regressed?
- Any new penalties introduced?
- What's the bottleneck now? (perception, alignment, insertion, speed)

### 6. ADJUST

Based on analysis, choose next experiment:
- If insertion fails: fix perception or alignment
- If insertion works but slow: optimize trajectory
- If penalties: reduce stiffness or add safety bounds
- If stuck: try a fundamentally different approach

## Approach Exploration Strategy

We explored multiple branches and converged on the most promising path. After 38
experiments the exploration tree looks like this:

```
Root: Get Tier 1 passing (valid policy)
├── Branch A: Classical Control -- PLATEAUED at 93.4 (8 experiments)
│   ├── A1: Hardcoded approach -- best 93.4
│   └── A2: XY correction -- best 79.9
├── Branch B: Camera Perception -- PLATEAUED at ~100 (16 experiments)
│   ├── B1: IBVS (image-based visual servoing) -- best 110.4 (lucky), reliable ~90-100
│   ├── B2: Template matching -- inconsistent
│   └── B3: Color segmentation -- unreliable
├── Branch C: ACT Imitation Learning -- ACTIVE, best 136.0 (15+ experiments)
│   ├── C1: Velocity-mode ACT -- plateaued at 121.9 (8 configs, 100ep)
│   ├── C2: **Position-mode ACT** -- best 136.0 (24 demos, 8 configs, 50ep)
│   ├── C3: Overfitting experiments -- more data/epochs hurts (120.9)
│   └── C4: Next: offset actions, higher resolution, task encoding
└── Branch D: Reinforcement Learning -- Not started
    └── D1: Isaac Lab parallel training (fallback if ACT fails)
```

**Current status:** Branch C (ACT) with position-mode actions is the active primary
approach. Branches A and B are exhausted. The breakthrough was switching from velocity
to position actions (D-005). Sweet spot: 24 demos from 8 configs, 50 epochs.

**Decision points:**
- After each experiment, assess: is this branch worth continuing?
- If 3 experiments on a branch show no progress, pivot
- If a branch scores >50/trial, invest more time optimizing it
- If a branch scores >75/trial, it's our submission candidate

## ACT Training Cycle

Each ACT experiment follows a longer cycle than classical experiments (~2-3 hours
per cycle instead of ~10 minutes):

```
COLLECT DATA -> TRAIN MODEL -> EVALUATE -> ADJUST CONFIG -> REPEAT
```

1. **Collect data:** Run `scripts/collect_demos.sh` on GPU instance. CheatCode
   generates expert demonstrations with domain randomization. Sweet spot: 24 demos
   from 8 board configs (3 demos per config). Data stored in `~/training_data/`.

2. **Train model:** Run `scripts/train_act.py` on GPU instance with position-mode
   actions (default). ~11 min for 50 epochs on L4 GPU. Models stored in `~/models/`.

3. **Evaluate:** Run 3-trial eval with the trained ACT policy. Parse scoring.yaml.

4. **Adjust config:** Change hyperparameters based on results:
   - Action mode: position (default, best) or offset (experimental)
   - Chunk size (10-100): larger = smoother but less reactive
   - Learning rate (~1e-4): cosine schedule
   - Batch size (32-64): limited by L4 VRAM (24GB)
   - Number of configs: diversity matters more than quantity (8 configs sweet spot)
   - Domain randomization: board pose, port offset, grasp noise

Key difference from classical experiments: the iteration variable is the training
config, not the policy source code. The policy code (RunACT) stays fixed.

## Score Targets

| Milestone | Score/Trial | Total | Meaning |
|-----------|------------|-------|---------|
| Tier 1 pass | 1 | 3 | Policy loads and runs |
| Proximity | 10-25 | 30-75 | Getting close to port |
| **Current best** | **~45/trial** | **136** | **Position ACT, ~2cm from port** |
| Partial insertion | 38-50 | 114-150 | In the port, not fully seated |
| Full insertion | 75 | 225 | Connector fully inserted |
| Optimized | 90+ | 270+ | Fast, smooth, efficient insertion |
| Perfect | 100 | 300 | Theoretical max |

**Qualification target:** We don't know the cutoff, but CheatCode scores ~88/trial
(264 total). Current best is 136/300. Next milestone: partial insertion on all trials
(150+). A competitive entry should target 200+.

## Experiment Naming Convention

```
exp-NNN-<branch>-<brief>
```

Examples:
- `exp-001-a1-hardcoded-approach`
- `exp-002-a2-camera-port-detection`
- `exp-003-b1-cheatcode-demo-collection`

## Beads as Research Log

Each experiment gets a bead. The bead description is the lab notebook entry.
Beads are never deleted -- even failed experiments are valuable data.

- **Open beads** = experiments in progress or planned
- **Closed beads** = experiments completed with results logged
- **P1 beads** = current priority experiment
- **P2 beads** = queued experiments
- **P3 beads** = ideas for later

## Automation: Self-Running Harness

The ideal loop runs autonomously:

1. **Agent proposes** next experiment based on all prior results
2. **Agent implements** the policy change
3. **Agent runs** local eval (if sim environment is available)
4. **Agent logs** results to bead
5. **Agent analyzes** and proposes next experiment
6. **Repeat**

**Cloud GPU eval performance:** ~5-10 min per 3-trial eval on the OVH L4
instance (NVIDIA L4, matches official eval hardware). This gives 6-12 experiments
per hour. Cost: ~$1.00/hr -- stop the instance when not experimenting.

When sim isn't available (no GPU, no Gazebo), the agent can still:
- Write policy code
- Analyze prior results
- Plan next experiments
- Prepare training data pipelines
- Optimize code for speed

## Decision Framework: When to Submit

Submit to cloud when:
- [ ] Local score is our personal best
- [ ] Score is consistent across 3+ local runs
- [ ] No regressions from previous submission
- [ ] Docker container verified locally
- [ ] No obvious failure modes remaining

**Never waste the 1/day submission on untested code.**

## File Structure

```
.claude/refs/
├── challenge-description.md   # Challenge spec
├── methodology.md             # This document
├── decisions.md               # Design decisions log
└── experiment-log.md          # Running summary of all experiments

experiments/                   # Policy code and configs per experiment
├── exp-001-a1-hardcoded/
│   ├── policy.py
│   └── results.yaml
├── exp-002-a2-camera/
│   ├── policy.py
│   └── results.yaml
└── ...
```
