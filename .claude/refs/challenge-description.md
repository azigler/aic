# AI for Industry Challenge - Complete Reference

## Overview

The **AI for Industry Challenge** is an open competition sponsored by Intrinsic and Open Robotics targeting a critical bottleneck in modern manufacturing: electronics assembly. Specifically, it focuses on **dexterous cable management and insertion** -- a task that currently remains largely manual and repetitive. From a robotics perspective, this task is notoriously difficult due to the complex physics involved in manipulating flexible cables and the extreme precision required to perceive, handle, and insert connectors.

## Prize Pool: $180,000

| Place | Prize |
|-------|-------|
| 1st | $100,000 |
| 2nd | $40,000 |
| 3rd | $20,000 |
| 4th | $10,000 |
| 5th | $10,000 |

## Timeline

**Overall: March 2 - September 8, 2026**

| Phase | Dates | Description |
|-------|-------|-------------|
| **Qualification** | Mar 2 - May 15 | Train cable assembly models in simulation. Eval: May 18-27. Top 30 announced May 28. |
| **Phase 1** | May 28 - Jul 14 | 30 qualified teams get Intrinsic Flowstate + Vision Model. Eval: Jul 14-21. Top 10 announced Jul 22. |
| **Phase 2** | Jul 27 - Aug 25 | 10 teams deploy to physical robot at Intrinsic HQ. Eval: Aug 26 - Sep 4. Winner Sep 8. |

**Registration deadline: April 17, 2026. Team size: up to 10.**

## The Task

### Qualification Phase (What We're Building For)

Single cable insertion per trial. Robot starts with plug already in-hand, within a few cm of the target.

**Two insertion types:**
1. **SFP Module -> SFP Port** (on NIC cards in Zone 1) -- Trials 1 & 2
2. **SC Plug -> SC Port** (on patch panel in Zone 2) -- Trial 3

**Randomization per trial:**
- Task board pose (position + yaw)
- NIC card rail position (5 rails available)
- NIC card translation and yaw offset on rail
- SC port translation along rail

**Grasp pose:** Approximately fixed with ~2mm, ~0.04 rad deviations. Policies must be robust to minor variations.

### Robot Setup

- **Robot:** Universal Robots UR5e
- **Gripper:** Robotiq Hand-E
- **F/T Sensor:** ATI AXIA80-M20
- **Camera:** Basler acA2440-20gc (1152x1024, 20 FPS) -- 3 cameras on wrist (left, center, right)

### Observation Data (20 Hz)

- 3x camera images (left, center, right) + camera info
- Joint states (arm + gripper)
- 3D force + 3D torque at wrist
- Target and actual TCP poses
- TCP velocity

### Control Interface

- **Cartesian control:** `/aic_controller/pose_commands` (MotionUpdate -- position or velocity mode)
- **Joint control:** `/aic_controller/joint_commands` (JointMotionUpdate -- position or velocity mode)
- Switch modes via `/aic_controller/change_target_mode` service (Cartesian=1, Joint=2)
- Controller starts in Cartesian mode by default
- Impedance control with configurable stiffness/damping

## Scoring (per trial, max 100 points)

### Tier 1: Model Validity (0-1 point)
- Policy loads, activates, responds to InsertCable action, sends valid commands

### Tier 2: Performance & Convergence (-36 to +24 points)
- **Trajectory smoothness** (0-6): Low jerk = better. Only if Tier 3 > 0
- **Task duration** (0-12): <=5s = max, >=60s = 0. Only if Tier 3 > 0
- **Trajectory efficiency** (0-6): Shorter path = better. Only if Tier 3 > 0
- **Insertion force penalty** (0 to -12): >20N for >1s = -12
- **Off-limit contact penalty** (0 to -24): Any robot-enclosure/task-board contact = -24

### Tier 3: Cable Insertion (-12 to +75 points)
- **Correct port insertion:** 75 points
- **Wrong port insertion:** -12 points
- **Partial insertion:** 38-50 points (proportional to depth, within 5mm x-y tolerance)
- **Proximity:** 0-25 points (inversely proportional to distance from port)

**Total across 3 trials: max 300 points**

## Technical Requirements (aic_model)

- ROS 2 Lifecycle node named `aic_model`
- Start unconfigured, configure within 60s, activate within 60s
- Accept `/insert_cable` action goals when active, goals must be cancellable
- Complete within `time_limit` from Task message
- No publishing when unconfigured/configured/shutdown
- Clean deactivate/cleanup/shutdown transitions

## Cloud Evaluation Hardware

- 64 vCPU, 256 GiB RAM
- 1x NVIDIA L4 Tensor Core (24 GiB VRAM)

## Submission

- Docker container via Amazon ECR
- 1 submission per day
- Final submission before deadline is scored
- Must pass local verification first

## Key Baseline Policies

- **WaveArm** - Minimal example, just waves the arm
- **CheatCode** - Uses ground truth data, scores 60/trial (full insertion)
- **RunACT** - ACT (Action Chunking with Transformers) policy

## Simulators Supported

- **Gazebo** (official evaluation)
- **MuJoCo** (via aic_mujoco integration)
- **Isaac Lab** (via aic_isaac integration)
- Cross-simulator training encouraged for domain randomization

## Sources

- Event page: https://www.intrinsic.ai/events/ai-for-industry-challenge
- Toolkit: https://github.com/intrinsic-dev/aic
- Discourse: https://discourse.openrobotics.org/c/competitions/ai-for-industry-challenge/
