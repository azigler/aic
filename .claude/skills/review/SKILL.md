---
description: Review design decisions and experiment results before committing to an approach
---

# Review

Lightweight review for hackathon decisions. Use when:
- Choosing between policy approaches
- Evaluating experiment results to decide next steps
- Deciding on architecture changes

## Process

1. **State the decision** needed clearly
2. **Present options** with concrete pros/cons
3. **Show evidence** (scores, timing, failure modes)
4. **Recommend** an approach with rationale
5. **Record** the decision in `.claude/refs/decisions.md`

## Experiment Review Template

```markdown
## Decision: [What we're deciding]

### Options
| Option | Pros | Cons | Evidence |
|--------|------|------|----------|
| A | ... | ... | Score: X |
| B | ... | ... | Score: Y |

### Decision: [Choice]
### Rationale: [Why]
### Next steps: [What to implement]
```
