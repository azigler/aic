---
description: Session entrypoint -- discover state, classify work, route to sub-skill
---

# Orient

Entry point for every session. Discovers current state, classifies remaining work,
and routes to the appropriate sub-skill.

## Step 1: Read Foundation

Read **every** item below before taking any action:

1. **`CLAUDE.md`** (root) -- project definition, challenge summary, tech stack
2. **`~/.claude/CLAUDE.md`** (global) -- delegation patterns, bead lifecycle
3. **`MEMORY.md`** -- user preferences, operational lessons
4. **Every skill file** in `.claude/skills/*/SKILL.md` -- especially:
   - `/experiment` -- the core iteration loop
   - `/sim` -- simulation environment management
   - `/impl` -- policy development phases
   - `/train` -- ML training orchestration
   - `/eval-policy` -- scoring and comparison
5. **`.claude/refs/challenge-description.md`** -- full challenge specification
6. **`.claude/refs/methodology.md`** -- experiment protocol and exploration tree
7. **`.claude/refs/experiment-log.md`** -- running leaderboard of all experiments
8. **`.claude/refs/decisions.md`** -- active design decisions

## Step 2: Discover Live State

```bash
git log --oneline -5                        # recent work
git branch -a | grep -v worktree            # active branches
br list                                     # open beads
git status --short                          # dirty files
docker images 2>/dev/null | head -5         # available containers
docker ps 2>/dev/null                       # running containers

# Check cloud GPU runner status
if [ -f scripts/runner-config.sh ]; then
    echo "Cloud GPU runner config: found"
    ssh -o ConnectTimeout=3 gpu true 2>/dev/null \
        && echo "Cloud GPU runner: reachable" \
        || echo "Cloud GPU runner: UNREACHABLE"
else
    echo "Cloud GPU runner: not configured"
fi
```

Determine:
- **Active branch**: any version branch means work in flight
- **Open beads**: in-progress beads = interrupted experiments
- **Dirty files**: uncommitted changes need attention first
- **Docker status**: eval container available? model built?
- **Runner status**: cloud GPU instance configured and reachable?
- **Current best score**: from experiment-log.md

## Step 3: Find Current Position

Check what's been built so far:

```bash
ls aic_example_policies/aic_example_policies/ros/  # existing policies
ls experiments/ 2>/dev/null                          # our experiment policies
cat .claude/refs/experiment-log.md                   # score leaderboard
```

Assess which phase we're in:
1. **Bootstrap** -- harness setup, Docker install, first build
2. **Baseline** -- getting Tier 1 passing, first non-zero score
3. **Perception** -- working on port detection from cameras
4. **Insertion** -- working on approach/insertion strategy
5. **Training** -- training a learned policy (ACT/diffusion/RL)
6. **Optimization** -- tuning for max score (speed, smoothness)
7. **Submission** -- packaging, local verification, cloud submit

## Step 4: Classify Work

| Domain | Skill | When |
|--------|-------|------|
| **Experiment Loop** | `/experiment` | Proposing, running, logging, analyzing experiments |
| **Simulation** | `/sim` | Launching Gazebo, creating scenarios |
| **Implementation** | `/impl` | Building policy code |
| **Training** | `/train` | Data collection, model training |
| **Evaluation** | `/eval-policy` | Running trials, parsing scores |
| **Review** | `/review` | Comparing approaches, making design decisions |
| **Submission** | `/release` | Docker build, ECR push, cloud submission |
| **Housekeeping** | `/housekeeping` | Cleanup, docs, config |

## Step 5: Route to Next Action

**Priority order:**

1. If there's an **in-progress experiment bead** -- resume it
2. If Docker eval is running and **results are ready** -- log them (`/experiment log`)
3. If there's a **P1 bead** -- start it
4. Otherwise -- run `/experiment next` to propose the highest-leverage experiment

## Step 6: Present and Route

```
## Orientation Report

**Position**: [phase and current experiment]
**Best score**: [from experiment-log.md]
**Active beads**: [list or none]
**Docker status**: [eval image pulled? model built? running?]
**Runner status**: [not configured / reachable / unreachable]
**Blockers**: [none / list]

**Recommended action**: [what to do next]
```

Then invoke the appropriate skill.

## Post-Compaction Recovery

If resuming after context compaction:

1. Read CLAUDE.md, methodology.md, and experiment-log.md first
2. `br list` to find interrupted experiments
3. Check git log for what's already done
4. Check Docker status (`docker ps`, `docker images`)
5. Present findings before taking action

The most common post-compaction mistake is starting a new experiment when one
is already in progress. Always check beads first.
