---
description: Design documents for policy approaches and experiments
---

# Spec / Design Documents

For this hackathon, specs are lightweight design docs that capture the approach
before implementing. They live in `.claude/refs/` alongside the challenge description.

## When to Write a Spec

- Before starting a new policy approach (Option A/B/C from `/impl`)
- Before a major architectural change
- When comparing multiple design alternatives

## Spec Structure

```markdown
# Design: [Policy Approach Name]

## Approach
[1-2 paragraph summary of the approach]

## Architecture
[Diagram or description of data flow]

## Key Decisions
- [Decision 1 and rationale]
- [Decision 2 and rationale]

## Expected Performance
- Target score per trial: X
- Expected failure modes: [list]

## Experiment Plan
1. [Step 1]
2. [Step 2]
3. ...

## Risks
- [Risk 1 and mitigation]
```

## Output

Write to `.claude/refs/design-<name>.md`.
