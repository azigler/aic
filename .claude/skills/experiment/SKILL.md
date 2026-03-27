---
name: experiment
description: Run the experiment loop -- propose, implement, score, log, analyze, adjust
argument-hint: "[propose|run|log|analyze|next|status]"
---

# /experiment - Experiment Loop

The core iteration skill. Every improvement to the policy goes through this loop.
See `.claude/refs/methodology.md` for full protocol.

## Commands

### /experiment propose

Generate the next experiment based on prior results.

1. Read all closed experiment beads (`br list` + `br show`)
2. Read the current best score
3. Identify the bottleneck (perception? alignment? insertion? speed? safety?)
4. Propose a specific, testable change
5. Create a bead for it:

```bash
br create -p 2 "exp: NNN-<branch>-<brief>"
br update <id> --description "$(cat <<'EOF'
## Hypothesis
[What change and why it should improve score]

## Branch
[A/B/C/D from methodology.md exploration tree]

## Changes Planned
[Specific files and modifications]

## Expected Outcome
[Target score or specific metric improvement]
EOF
)"
```

### /experiment run

Execute a proposed experiment.

1. Claim the bead: `br update <id> --status=in_progress`
2. Implement the policy change
3. Run eval -- detect whether remote runner is available:

   **Remote runner (preferred, ~10-12 min):**
   ```bash
   # Check if runner is configured and reachable
   if [ -f scripts/runner-config.sh ] && ssh -o ConnectTimeout=3 mac true 2>/dev/null; then
       scripts/remote-eval.sh <pkg>.<Class>
   fi
   ```

   **Local fallback (~27 min):**
   ```bash
   docker compose -f docker/docker-compose.yaml up
   ```

   **Manual local (distrobox):**
   ```bash
   # Terminal 1 (eval container must already be running):
   # distrobox enter -r aic_eval -- /entrypoint.sh start_aic_engine:=true

   # Terminal 2 (policy):
   AIC_RESULTS_DIR=~/aic_results/exp-NNN \
   pixi run ros2 run aic_model aic_model --ros-args \
     -p use_sim_time:=true -p policy:=<pkg>.<Class>
   ```
4. Parse results: `cat ~/aic_results/exp-NNN/scoring.yaml`
   (remote eval places results in `./aic_results/scoring.yaml`)
5. Log results to the bead (see /experiment log)

### /experiment log

Record results in the bead description.

```bash
br update <id> --description "$(cat <<'EOF'
## Hypothesis
[Original hypothesis]

## Changes
[What was actually modified -- files, parameters, approach]

## Results
| Trial | Tier1 | Tier2 | Tier3 | Total | Notes |
|-------|-------|-------|-------|-------|-------|
| 1 (SFP) | | | | | |
| 2 (SFP) | | | | | |
| 3 (SC)  | | | | | |
| **Sum** | | | | **X** | |

## Penalties
- Force: [yes/no]
- Off-limit contacts: [yes/no]

## Analysis
[What worked, what didn't, why]

## Next
[What to try next based on this result]
EOF
)"
```

Then close the bead and commit:
```bash
br close <id>
git add .beads/issues.jsonl <changed-files>
git commit -m ":test_tube: exp-NNN: <brief result summary>

Bead: <id>
Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
git push
```

### /experiment analyze

Review all experiments and identify patterns.

1. List all closed experiment beads
2. Sort by score (best to worst)
3. Identify:
   - Which branches are most promising?
   - What's the current score ceiling?
   - What's the bottleneck preventing higher scores?
   - Are there unexplored approaches worth trying?
4. Update `.claude/refs/experiment-log.md` with summary table

### /experiment next

Shortcut: analyze + propose in one step.
Read all prior results, determine the highest-leverage next experiment, create the bead.

### /experiment status

Quick summary of where we stand.

```bash
br list                          # Open/in-progress experiments
echo "---"
echo "Best score: [from experiment-log.md]"
echo "Current branch: [which approach we're on]"
echo "Experiments run: [count]"
echo "Next planned: [bead title]"
```

## Automation Protocol

When running autonomously (no human in the loop), the agent should:

1. Run `/experiment next` to pick the next experiment
2. Run `/experiment run` to implement and test
3. Run `/experiment log` to record results
4. Check if score improved:
   - **Yes**: continue optimizing this branch
   - **No after 3 tries**: pivot to different branch
5. Repeat from step 1

### Stopping Conditions

- Score > 250 total (strong qualification candidate) -- notify human, prepare submission
- Score plateaued for 5+ experiments -- notify human, request guidance
- All branches explored without >150 -- fundamental rethink needed

### Constraints

- Never submit to cloud without human approval
- Always push commits after each experiment
- Always log results even if experiment failed
- Keep experiment beads as permanent record

## Branches Quick Reference

| Branch | Approach | Difficulty | Potential |
|--------|----------|------------|-----------|
| A: Classical | Hardcoded + vision + force control | Low-Medium | 50-80/trial |
| B: Imitation | ACT/diffusion on demonstrations | Medium-High | 60-95/trial |
| C: Hybrid | Vision approach + learned insertion | Medium | 70-95/trial |
| D: RL | Reward-shaped reinforcement learning | High | 50-100/trial |

**Start with A** (fastest to get a score > 0), then branch based on results.

## Related Skills

- `/sim` -- Launch simulation
- `/train` -- Training loop
- `/eval-policy` -- Score parsing
- `/beads` -- Task tracking
- `/commit` -- Commit conventions
