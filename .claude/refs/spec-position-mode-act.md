# Spec: Position-Mode ACT + Quick Experiment Strategy

## Problem
We're stuck at ~120/300. The 5cm XY gap to the port persists across all approaches.
Root cause: ACT trains on velocity actions (lossy conversion from CheatCode's
position commands), causing accumulated drift.

## Changes (ONE at a time, 20-epoch micro-trains to find signal)

### Change 1: Position-Mode Actions (HIGHEST PRIORITY)
Record CheatCode's actual `set_pose_target` pose as the action (7D: x,y,z,qx,qy,qz,qw).
At inference, output position targets via `set_pose_target` instead of velocity mode.

**DataCollector changes:**
- In `recording_move_robot`: capture `motion_update.pose` as the action
- Remove velocity computation from `_record_observation`
- Action dimension: 7D (position xyz + quaternion xyzw)

**OurACT changes:**
- Output 7D position target instead of 7D velocity
- Use `set_pose_target(move_robot, pose=predicted_pose)` instead of velocity MotionUpdate
- Keep same stiffness/damping as CheatCode defaults

**Training:**
- Re-collect 9 episodes (3 configs × 3 trials) with position-mode recording
- Micro-train: 20 epochs, batch_size=8 (~15 min)
- If score > 125: scale to 50 epochs. If > 140: scale to 100 epochs.

### Change 2: Higher Image Resolution
Increase from 256×256 to 384×384 or 512×512.
At 5cm distance, port is ~11px at 256 but ~22px at 512.

**DataCollector:** change IMG_SIZE from 256 to 384
**train_act.py:** update input shape
**OurACT:** match resize

**Micro-train:** 20 epochs with same data but larger images.

### Change 3: Task Encoding
Add task port_type ("sfp" or "sc") and target_module_name to the observation.
This tells the model WHICH port to target.

**DataCollector:** append task encoding to state vector (28D instead of 26D)
**train_act.py:** update state_dim to 28
**OurACT:** include task encoding in observation

### Change 4: Last-Mile Curriculum
Oversample the last 20% of each episode (the insertion approach phase).
This forces the model to learn the hard part (fine alignment).

**train_act.py:** weight samples from the last 100 timesteps 3× higher

## Experiment Plan

```
1. Collect 9 new episodes with POSITION-MODE recording (~30 min)
2. Micro-train Change 1 (20 epochs, ~15 min). Eval (~8 min). Total: ~53 min
3. If improvement: scale to 50 epochs. Else: try Change 2.
4. Micro-train Change 2 (20 epochs). Eval. Total: ~30 min
5. Combine best changes. Full train (50-100 epochs).
```

Each micro-experiment takes ~30 min including eval. We can test 4-6 ideas per
2-hour GPU session instead of waiting 2 hours for one 100-epoch run.
