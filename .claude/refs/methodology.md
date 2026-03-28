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

We don't commit to one approach. We explore multiple paths and double down on
what works. The exploration follows a tree:

```
Root: Get Tier 1 passing (valid policy)
├── Branch A: Classical Control
│   ├── A1: Hardcoded approach (fixed offset from task.target_module_name)
│   ├── A2: Vision-based port detection + PID control
│   ├── A3: Force-feedback spiral search for insertion
│   └── A4: Combine A2 + A3
├── Branch B: Imitation Learning
│   ├── B1: Collect demos via CheatCode
│   ├── B2: Train ACT on demos
│   ├── B3: Fine-tune with domain randomization
│   └── B4: Distill to smaller/faster model
├── Branch C: Hybrid
│   ├── C1: Vision for coarse alignment + classical for insertion
│   ├── C2: Classical for approach + learned for insertion
│   └── C3: Full pipeline with learned components
└── Branch D: Reinforcement Learning
    ├── D1: Isaac Lab parallel training
    ├── D2: Reward shaping for insertion
    └── D3: Sim-to-sim transfer
```

**Decision points:**
- After each experiment, assess: is this branch worth continuing?
- If 3 experiments on a branch show no progress, pivot
- If a branch scores >50/trial, invest more time optimizing it
- If a branch scores >75/trial, it's our submission candidate

## Score Targets

| Milestone | Score/Trial | Total | Meaning |
|-----------|------------|-------|---------|
| Tier 1 pass | 1 | 3 | Policy loads and runs |
| Proximity | 10-25 | 30-75 | Getting close to port |
| Partial insertion | 38-50 | 114-150 | In the port, not fully seated |
| Full insertion | 75 | 225 | Connector fully inserted |
| Optimized | 90+ | 270+ | Fast, smooth, efficient insertion |
| Perfect | 100 | 300 | Theoretical max |

**Qualification target:** We don't know the cutoff, but CheatCode scores ~88/trial
(264 total). Assume top 30 needs at least 150+ (partial insertion on all trials).
A competitive entry should target 200+.

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
