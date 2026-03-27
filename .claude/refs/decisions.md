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

## Pending Decisions

### D-004: Which vision approach for port detection?
**Options:**
- Template matching on camera images
- Neural network (pretrained detector fine-tuned on port images)
- Stereo depth estimation from 3 wrist cameras
- Use TF frames during training, learn to infer at eval time
**Status:** OPEN -- depends on Branch A results
