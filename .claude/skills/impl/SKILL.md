---
description: Policy implementation workflow for cable insertion challenge
---

# Implementation Workflow

## What We're Building

A Python policy class that controls a UR5e robot to insert fiber optic connectors
(SFP and SC) into ports on a randomized task board. The policy:

1. Receives sensor observations at 20Hz (cameras, joint states, wrench, TCP pose)
2. Outputs robot commands (Cartesian or joint targets with impedance params)
3. Must handle randomized board poses and port positions
4. Must be safe (no collisions, no excessive force)

## Policy Architecture Options

### Option A: End-to-End Learned (ACT / Diffusion)

```
cameras + proprioception -> neural network -> action chunks -> controller
```

- Train on demonstrations (teleoperation or CheatCode expert)
- ACT baseline provided in `aic_example_policies/`
- Pros: can learn complex behaviors, handles uncertainty
- Cons: needs lots of data, can fail on OOD configs

### Option B: Perception + Classical Control

```
cameras -> vision model -> port pose estimate
port pose + TCP pose -> motion planner -> controller
```

- ML for perception only (detect port position/orientation)
- Classical controller for approach + insertion (spiral search, impedance)
- Pros: interpretable, robust, less data needed
- Cons: may not handle complex cable physics

### Option C: Hybrid

```
cameras -> vision model -> coarse alignment
proprioception + wrench -> learned insertion policy -> controller
```

- Vision for coarse positioning
- Learned fine-grained insertion using force feedback
- Best of both worlds

## Implementation Phases

### Phase 1: Get the Loop Running

1. Create a policy class extending `aic_model.Policy`
2. Implement `insert_cable()` with basic motion commands
3. Verify Tier 1 passes (model validity)
4. Score > 0 on at least one trial (proximity points)

### Phase 2: Perception

1. Process camera images to locate target port
2. Estimate port pose relative to TCP
3. Use force/torque for contact detection
4. Score > 25 (proximity/partial insertion)

### Phase 3: Insertion

1. Implement approach trajectory (move toward port)
2. Implement insertion strategy (comply + push)
3. Tune impedance parameters (lower stiffness for compliance)
4. Score > 50 (partial to full insertion)

### Phase 4: Optimization

1. Optimize trajectory for speed (duration score 0-12)
2. Smooth motion (reduce jerk, smoothness score 0-6)
3. Short path (efficiency score 0-6)
4. Target: 90+ per trial

## Policy Template

```python
from aic_model.policy import Policy

class MyPolicy(Policy):
    def __init__(self):
        super().__init__()
        # Load model weights, initialize state

    def insert_cable(self, task, get_observation, move_robot, send_feedback):
        """Called per trial. Return when insertion is complete or giving up."""
        obs = get_observation()

        # 1. Parse task to know target port
        target_port = task.target_port_name
        cable_type = task.cable_type

        # 2. Perception: locate port from cameras
        # ...

        # 3. Plan approach trajectory
        # ...

        # 4. Execute approach + insertion
        while not done:
            obs = get_observation()
            motion_cmd = self.compute_action(obs)
            move_robot(motion_cmd)

        return  # Signal task complete
```

## Control Interface Reference

### Cartesian Position (most common for insertion)

```python
from aic_control_interfaces.msg import MotionUpdate
from geometry_msgs.msg import Pose
from std_msgs.msg import Header

cmd = MotionUpdate()
cmd.header = Header(frame_id='base_link')  # or 'gripper/tcp'
cmd.pose = Pose(position=..., orientation=...)
cmd.target_stiffness = [85.0, 0, 0, 0, 0, 0,
                        0, 85.0, 0, 0, 0, 0,
                        0, 0, 85.0, 0, 0, 0,
                        0, 0, 0, 85.0, 0, 0,
                        0, 0, 0, 0, 85.0, 0,
                        0, 0, 0, 0, 0, 85.0]
cmd.target_damping = [75.0, 0, 0, 0, 0, 0,
                      0, 75.0, 0, 0, 0, 0,
                      0, 0, 75.0, 0, 0, 0,
                      0, 0, 0, 75.0, 0, 0,
                      0, 0, 0, 0, 75.0, 0,
                      0, 0, 0, 0, 0, 75.0]
cmd.trajectory_generation_mode.mode = 2  # MODE_POSITION
move_robot(cmd)
```

### Stiffness/Damping Tuning

| Context | Stiffness | Damping | Notes |
|---------|-----------|---------|-------|
| Free space motion | 85-200 | 75-100 | High stiffness for precise tracking |
| Approach | 50-85 | 50-75 | Medium compliance |
| Insertion | 10-50 | 30-50 | Low stiffness for compliance |
| Contact detection | 5-20 | 20-40 | Very compliant, feel the port |

Lower stiffness = more compliance = safer during contact but less precise.
Higher damping = less oscillation = smoother but slower.

## Scoring Optimization Priority

1. **Insertion success (75 pts)** -- this is 75% of the score. Get this first.
2. **Duration (12 pts)** -- be fast. <=5s = full marks.
3. **Smoothness (6 pts)** -- low jerk motion
4. **Efficiency (6 pts)** -- short path
5. **Avoid penalties** -- no collisions (-24), no excessive force (-12)

## Bead Protocol

Each implementation phase gets a bead:

```bash
br create -p 2 "impl: basic policy loop (Tier 1)"
br create -p 2 "impl: perception -- port detection"
br create -p 2 "impl: insertion strategy"
br create -p 2 "impl: optimization -- speed and smoothness"
```

## Related Skills

- `/sim` -- Launch simulation environment
- `/train` -- Training loop for learned policies
- `/eval-policy` -- Run evaluation and compare experiments
- `/commit` -- Commit conventions
