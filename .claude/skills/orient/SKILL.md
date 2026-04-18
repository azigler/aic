---
name: orient
description: Session entrypoint -- discover state, classify work, route to experiment/train/eval
---

# /orient - Session Entrypoint

Entry point for every session. Discovers live state, classifies remaining work
by skill domain, and routes to the appropriate sub-skill.

**No hardcoded references.** Everything discovered fresh from live state.

## Step 1: Read Foundation

Read in order (absorb each before continuing):

1. `CLAUDE.md` -- project rules, workflows, infra facts
2. `.claude/projects/-home-ubuntu-aic/memory/MEMORY.md` -- auto-memory (already loaded)
3. Available skill files in `.claude/skills/*/SKILL.md` -- experiment, train, eval

## Step 2: Discover Live State

```bash
git log --oneline -10                       # recent experiments/commits
git status --short                          # dirty files
br list                                     # open beads (P1 = active work)
ls /home/ubuntu/aic_models/ 2>/dev/null     # local model checkpoints
ls /home/ubuntu/training_data_*/ -d 2>/dev/null  # local training data
```

Also check if the last session's JSONL references GPU state (suspended vs
running) -- GPU is ~$1/hr and should be off when idle.

## Step 3: Identify Current Position

From live state, determine:

- **Champion model**: latest committed best score (git log for score mentions)
- **Last experiment**: most recent bead with results, or `exp-NNN` in commit log
- **Open experiments**: P1 beads in OPEN or in_progress state
- **GPU state**: suspended (no-op) or running (ready for eval)
- **Data inventory**: what datasets exist locally for training

## Step 4: Classify Next Work

Every task falls into one domain:

| Domain | Skill | When |
|--------|-------|------|
| **Experiment** | `/experiment` | Propose hypothesis, log results, close a bead |
| **Train** | `/train` | Fine-tune ACTPolicy on existing demo data |
| **Eval** | `/eval` | Run policy on GPU via distrobox+pixi |
| **Submit** | (manual) | Docker compose verification for final submission |

The pipeline is typically:

```
experiment (propose) -> train -> eval -> experiment (log results) -> ...
```

## Step 5: Check Blockers

Before routing, verify:

1. **GPU suspended?** -> Must resume before train/eval
2. **Dirty git state?** -> Resolve before new experiment
3. **Open in_progress beads?** -> Finish logging before starting new work
4. **Champion score >200?** -> STOP. Route to submission, not more experiments.
5. **3+ consecutive discards?** -> Pivot strategy before running another

## Step 6: Present and Route

Show the user:

```
## Orientation Report

**Champion**: [model name] at [score]/300 (verified in [Docker/pixi])
**Last experiment**: [exp-NNN: result]
**Open beads**: [count] P1 ([list])
**GPU**: [suspended | running]
**Data**: [datasets available locally]
**Blockers**: [none | list]

**Recommended action**: [submit | run exp-NNN | pivot | collect more data]
```

Then either wait for user direction or invoke the appropriate skill.

## Post-Compaction Recovery

If resuming after context compaction:

1. **Do NOT start training or eval immediately.** Orient first.
2. **Read the last bead closed** to understand the most recent finding.
3. **Check git log for score mentions** -- champion may have changed.
4. **Verify local data still exists** before planning GPU work.
5. **Present findings** before taking action. User may have off-record context.

The most common post-compaction mistake is re-running an experiment that was
already tried and discarded. Always check bead history for the hypothesis
before proposing a new experiment.

## Hackathon-Specific Rules

- **Every experiment = one bead + one variable changed.** No exceptions.
- **GPU cost matters.** If suspended, ask before resuming.
- **Champion is sacred.** Never overwrite v5 weights. New experiments go to
  new model directories.
- **Docker eval != pixi eval.** Docker compose is submission-only.
- **Log every score.** Both the scoring.yaml numbers and the per-trial breakdown.
