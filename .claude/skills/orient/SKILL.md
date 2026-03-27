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
4. **Every skill file** in `.claude/skills/*/SKILL.md`
5. **`.claude/refs/challenge-description.md`** -- full challenge specification

## Step 2: Discover Live State

```bash
git log --oneline -5                        # recent work
git branch -a | grep -v worktree            # active branches
br list                                     # open beads
git status --short                          # dirty files
```

Determine:
- **Active branch**: any version branch means work in flight
- **Open beads**: any in-progress beads mean interrupted work
- **Dirty files**: uncommitted changes need attention first

## Step 3: Find Current Position

Check what's been built so far:

```bash
ls aic_example_policies/aic_example_policies/ros/  # existing policies
ls experiments/ 2>/dev/null                          # training experiments
```

Assess which implementation phase we're in:
1. **Bootstrap** -- harness setup, no custom policy yet
2. **Perception** -- working on port detection
3. **Control** -- working on approach/insertion strategy
4. **Training** -- training a learned policy
5. **Optimization** -- tuning for max score
6. **Submission** -- packaging and submitting

## Step 4: Classify Work

| Domain | Skill | When |
|--------|-------|------|
| **Simulation** | `/sim` | Launching, configuring, exploring the environment |
| **Implementation** | `/impl` | Building policy code |
| **Training** | `/train` | Data collection, model training |
| **Evaluation** | `/eval-policy` | Running trials, parsing scores, comparing |
| **Housekeeping** | `/housekeeping` | Cleanup, docs, config |

## Step 5: Present and Route

```
## Orientation Report

**Position**: [where we are in development]
**Active beads**: [list or none]
**Blockers**: [none / list]

**Recommended action**: [what to do next]
```

Then invoke the appropriate skill.

## Post-Compaction Recovery

If resuming after context compaction:

1. Read CLAUDE.md and challenge description first
2. Check git log for what's already been done
3. Check open beads for interrupted work
4. Present findings before taking action
