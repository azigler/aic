---
name: experiment
description: The core experiment loop. Every change goes through this.
argument-hint: "[propose|run|log|next]"
---

# /experiment - Experiment Loop

## BEFORE starting ANY experiment:

1. Create a bead with hypothesis:
```bash
br create -p 2 "exp-NNN: brief description"
br update <id> --description "## Hypothesis\n[what and why]\n## Expected Outcome\n[target]"
br update <id> --status=in_progress
```

2. Verify you're changing ONLY ONE variable from the known-good baseline.

## Running an experiment:

1. Make the code change locally
2. Run /eval to test on GPU
3. Parse results immediately

## AFTER every experiment:

1. Update bead with FULL results:
```bash
br update <id> --description "## Results\n| Trial | Score |\n...\n## Analysis\n[why]\n## Next\n[what]"
```

2. Update experiment-log.md
3. Commit with gitmoji + bead trailer
4. If KEEP: update baseline. If DISCARD: revert changes.
5. Close bead: `br close <id>`

## Autoresearch rules:

- **KEEP:** Score improved by ≥5 points over baseline
- **DISCARD:** Score declined or within noise. Revert immediately.
- **PIVOT:** 3 consecutive discards → try fundamentally different approach
- **STOP:** Score >200 → prepare submission
