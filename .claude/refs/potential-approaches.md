# Potential Approaches: How to Win This Competition

## Current Status (April 5, 2026)

**Best score: 93.4/300** (exp-004). We've achieved partial insertion on one SFP trial.
We've been iterating for ~4 hours on the GPU ($4 spent so far).

**Time remaining:** ~40 days until May 15 qualification deadline.
**Budget:** at $1/hr, ~$250 if using 6hr/day. We've used ~$4 so far.

## What We've Learned (7 experiments)

1. **-Z descent works** -- the gripper points straight down, port is below
2. **We're not descending far enough** -- port is 18.6cm below TCP, we only go 15cm
3. **XY offset matters** -- port is offset 1-6cm in XY from TCP start position
4. **SC trial is completely different geometry** -- port is 31.7cm away, much more offset
5. **Speed matters** -- duration score (12 pts) is significant. Faster = better score
6. **Force baseline is ~21N** -- must be compensated, not absolute threshold
7. **Partial insertion possible** at 4cm depth (trial 1, exp-004)

## The Bottleneck

Our blind descent in -Z gets to ~5cm from the port but can't get closer because:
- We run out of descent range before reaching the port
- XY offset means we miss the port entrance even at the right Z
- No perception = no ability to correct course

## Approach Options (Ranked by Effort vs Impact)

### A. Fix Depth + Hardcode Offsets (1-2 hours, est. score: 120-150)

**What:** Increase descent to 20cm, add average XY offset from ground truth data.

**Pros:** Trivially easy, immediate improvement. Gets us to port Z level.
**Cons:** Hardcoded offsets won't match randomized board positions exactly.
Still blind -- proximity points but probably not full insertion.

**Expected:** 30-50 pts/trial for SFP (proximity + partial), still ~0 for SC.
**Cost:** ~$2 (2 experiments)

### B. Camera-Based Port Detection (1-2 days, est. score: 150-200)

**What:** Use the 3 wrist cameras to detect the port location, servo toward it.

**Pros:** Works regardless of randomization. Enables full insertion.
**Cons:** Requires computer vision (template matching, feature detection, or neural).
Camera images are 1152x1024 -- processing cost. Need to know what ports look like.

**Approaches within this:**
- Template matching (OpenCV, simple, may work for SFP/SC shapes)
- Feature detection (ORB/SIFT + matching against known port geometry)
- Neural network (train a detector, needs data)
- Depth estimation from stereo cameras (3 cameras = stereo pairs)

**Expected:** If port detection works reliably, 60-80 pts/trial (full insertion possible).
**Cost:** ~$10-20 (10-20 experiments over 1-2 days)

### C. ACT / Imitation Learning (3-5 days, est. score: 200-250)

**What:** Collect demonstrations via CheatCode, train ACT policy.

**Pros:** End-to-end, handles complex dynamics. Reference implementation provided.
**Cons:** Needs training data collection (~100 demos), training time (~hours),
hyperparameter tuning. Existing RunACT baseline exists but needs training data.

**Expected:** With good data + tuning, 60-85 pts/trial across all trial types.
**Cost:** ~$30-50 (training + eval)

### D. Hybrid: Camera Coarse + Classical Fine (2-3 days, est. score: 200-270)

**What:** Camera detects port location (approach), then compliant insertion
using force feedback and spiral search.

**Pros:** Best of perception + control. Robust to randomization.
**Cons:** More complex to implement. Two-phase system needs careful tuning.

**Expected:** 70-90 pts/trial if both phases work well.
**Cost:** ~$20-30

### E. Reinforcement Learning (5-10 days, est. score: 150-270)

**What:** Train a policy via RL in simulation (Isaac Lab or MuJoCo).

**Pros:** Can learn optimal behavior from scratch. Scalable.
**Cons:** Expensive (GPU hours for training), reward shaping is hard,
sim-to-sim transfer uncertain. High variance.

**Expected:** Wide range -- could be amazing or terrible.
**Cost:** ~$50-100 (days of GPU time for training)

## Recommended Strategy

### Phase 1: Quick Wins (Today, ~$2-4)
1. **Fix descent depth to 20cm** (currently 15cm, port is at 18.6cm below)
2. **Add average XY offset** from ground truth data
3. **Target: 120+ total** (beat WaveArm's 42 and our 93.4 on all trials)

### Phase 2: Camera Perception (Next 3-5 days, ~$10-15)
1. Capture camera images at various positions
2. Implement port detection (start with simple template matching)
3. Servo toward detected port before insertion
4. **Target: 180+ total** (consistent partial/full insertion on SFP)

### Phase 3: SC Trial Fix (Same period)
1. SC port geometry is different -- 31cm away, large XY offset
2. Camera perception should handle this if trained on SC shapes
3. **Target: 200+ total** (scoring on all 3 trials)

### Phase 4: Full Insertion + Optimization (Days 5-15, ~$20-30)
1. Tune insertion parameters for full 75-point insertion
2. Optimize trajectory for speed (duration bonus)
3. Smooth motion (jerk minimization)
4. **Target: 250+ total** (competitive for top 30)

### Phase 5: Learned Policy (Days 15-40 if needed, ~$50)
1. If classical approach plateaus, train ACT on successful demos
2. Use our classical controller as the demonstration expert
3. **Target: 270+ total** (prize contention)

## Cost Summary

| Phase | Duration | GPU Hours | Cost |
|-------|----------|-----------|------|
| 1. Quick wins | 1 day | 2-4h | $2-4 |
| 2. Camera perception | 3-5 days | 15-25h | $15-25 |
| 3. SC fix | Included in 2 | -- | -- |
| 4. Full insertion | 5-10 days | 20-40h | $20-40 |
| 5. Learned policy | 10-20 days | 30-60h | $30-60 |
| **Total** | **~20-40 days** | **67-129h** | **$67-129** |

Well under the $250 budget.

## How Smart Teams Iterate

1. **Data-driven:** Every experiment produces numbers. Analyze before coding.
2. **One variable at a time:** Don't change step size AND stiffness AND spiral radius.
3. **Quick validation:** Test on one trial first, then full 3-trial eval.
4. **Keep what works:** When something improves score, commit it. Don't rewrite.
5. **Know when to pivot:** If 3 experiments on an approach show no progress, try something fundamentally different.
6. **Parallelize thinking and testing:** Write next experiment while current one runs.
7. **Submission strategy:** Submit every day once we have a stable score > 100.
   The leaderboard shows where we stand relative to others.

## Key Insight from Ground Truth Data

The port positions we measured tell us:
- SFP ports are at Z≈0.133 in base_link frame (board height + mount)
- SC ports are at Z≈0.015 (much lower, near board base)
- TCP starts at Z≈0.30-0.32
- **SFP needs ~18cm descent, SC needs ~28cm descent**
- **XY offsets are small for SFP (~2cm) but large for SC (~15cm)**

This data alone should dramatically improve our score with just a descent fix.
