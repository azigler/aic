# Design Decisions

## Active Decisions

### D-001: Start with Branch A (Classical Control)
**Date:** 2026-03-27
**Rationale:** Fastest path to a non-zero score. We can get Tier 1 passing and
proximity points within one session. This establishes the baseline that all other
approaches must beat.
**Status:** DECIDED

### D-002: Use beads as experiment research log
**Date:** 2026-03-27
**Rationale:** Each experiment is a discrete unit of work with a hypothesis,
changes, results, and analysis. Beads give us a structured, searchable record
that persists across sessions.
**Status:** DECIDED

### D-003: Local scoring as primary iteration loop
**Date:** 2026-03-27
**Rationale:** Cloud submissions are limited to 1/day. Local eval via
distrobox + pixi is unlimited and runs the exact same scoring engine.
Only submit to cloud when we have a verified personal best.
**Status:** DECIDED

### D-004: Which vision approach for port detection?
**Date:** 2026-03-27
**Options:**
- Template matching on camera images
- Neural network (pretrained detector fine-tuned on port images)
- Stereo depth estimation from 3 wrist cameras
- Use TF frames during training, learn to infer at eval time
**Decision:** End-to-end ACT. After 16 experiments on Branch B (camera perception),
all explicit vision approaches plateaued at ~100. ACT learns implicit perception
from raw images, bypassing the need for a separate vision pipeline.
**Status:** DECIDED

### D-005: Position-mode ACT over velocity-mode
**Date:** 2026-04-05
**Rationale:** Switching ACT from velocity actions to position (absolute TCP target
pose) actions dropped val_loss 50-100x (0.29 vs 14-27) and raised score from ~120
to 131-136. Position targets match CheatCode's output directly and don't accumulate
drift like velocity predictions.
**Status:** DECIDED

## Pending Decisions

### D-006: Offset actions (relative position deltas)
**Options:**
- Keep absolute position targets (current best, 136.0)
- Switch to relative TCP deltas (translation-invariant, may generalize better)
**Rationale:** Absolute positions encode board location implicitly. Offset actions
would decouple the learned policy from specific board positions, potentially
improving generalization to unseen randomizations.
**Status:** OPEN -- highest priority next experiment
