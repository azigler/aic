# AI for Industry Challenge -- Hackathon Repo

## What This Is

Competition entry for the [AI for Industry Challenge](https://www.intrinsic.ai/events/ai-for-industry-challenge/) ($180K prize pool). Train an AI policy to control a UR5e robot to insert fiber optic cables (SFP and SC connectors) into ports on a randomized task board in Gazebo simulation.

## CRITICAL RULES (Do NOT deviate)

1. **Use the competition's provided patterns.** RunACT.py, LeRobot ACTPolicy, distrobox,
   pixi, Docker compose — use what they gave us. Do NOT build custom alternatives.
2. **distrobox + pixi for iteration.** Docker compose is ONLY for submission verification.
3. **LeRobot ACTPolicy for training and inference.** Do NOT build a custom model architecture.
4. **Follow the tutorial.** Read `docs/tutorial.md` and the README before changing anything.
5. **One change at a time.** Never change multiple variables in an experiment.
6. **Log everything in beads.** Every experiment gets a bead with hypothesis, results, analysis.
7. **Verify on GPU before concluding.** Never trust local-only analysis.

## Challenge Summary

- **Robot:** UR5e + Robotiq Hand-E gripper + ATI F/T sensor + 3 wrist cameras
- **Task:** Insert cable connector (SFP or SC) into the correct port on a randomized task board
- **Scoring:** 100 pts/trial max (validity 1 + performance 24 + insertion 75). 3 trials per eval.
- **Qualification deadline:** May 15, 2026. Top 30 advance.

## Competition-Provided Reference Implementation

**RunACT.py** is the reference ACT policy:
- Uses **LeRobot's ACTPolicy** (`lerobot.policies.act.modeling_act`)
- Downloads pretrained weights from HuggingFace (`grkw/aic_act_policy`)
- Uses **velocity mode** (7D: linear_xyz + angular_xyz + gripper)
- Uses **`policy.select_action(obs_dict)`** for inference (handles chunking, ensembling)
- Uses **image_scaling = 0.25** (1152×1024 → 288×256)
- Uses **wrench feedback gains** [0.5, 0.5, 0.5, 0, 0, 0]
- 26D robot state observation
- 4Hz control loop (0.25s sleep)

**This is the pattern to follow.** Fine-tune this, don't replace it.

## Repository Structure

```
aic/
├── aic_adapter/          # Sensor fusion -> Observation at 20Hz
├── aic_assets/           # 3D models
├── aic_bringup/          # Launch files for simulation
├── aic_controller/       # Impedance controller (C++)
├── aic_description/      # URDF/SDF
├── aic_engine/           # Trial orchestration and validation
├── aic_example_policies/ # Baselines: WaveArm, CheatCode, RunACT, DataCollector
├── aic_gazebo/           # Gazebo plugins
├── aic_interfaces/       # ROS 2 msg/srv/action
├── aic_model/            # Policy framework (lifecycle node)
├── aic_scoring/          # Scoring implementation
├── aic_utils/            # MuJoCo + Isaac Lab integrations
├── docker/               # Dockerfiles for eval and submission
├── docs/                 # Full documentation (READ THESE)
└── scripts/              # Helper scripts
```

## Development Workflow

### Iteration Loop (on GPU via SSH)

```bash
# 1. Edit code locally, rsync to GPU
rsync -az --exclude='.git' --exclude='.pixi' --exclude='__pycache__' \
  /home/ubuntu/aic/ gpu:~/ws_aic/src/aic/

# 2. Reinstall package on GPU
ssh gpu "export PATH=\$HOME/.pixi/bin:\$PATH && cd ~/ws_aic/src/aic && \
  pixi reinstall ros-kilted-aic-example-policies 2>&1 | tail -3"

# 3. Run eval via distrobox + pixi
ssh gpu "export PATH=\$HOME/.pixi/bin:\$PATH && cd ~/ws_aic/src/aic && \
  POLICY=aic_example_policies.ros.RunACT ~/run-eval.sh"

# 4. Check results
ssh gpu "cat ~/aic_results/scoring.yaml"
```

### Training Loop

```bash
# 1. Collect demos (CheatCode with ground_truth=true)
# 2. Convert to LeRobot dataset format
# 3. Fine-tune ACTPolicy using LeRobot training
# 4. Evaluate fine-tuned model
# 5. Iterate on hyperparameters
```

### Submission (Docker compose -- ONLY for final verification)

```bash
ssh gpu "cd ~/ws_aic/src/aic && docker compose -f docker/docker-compose.yaml build model"
ssh gpu "cd ~/ws_aic/src/aic && docker compose -f docker/docker-compose.yaml up"
```

## GPU Instance (OVH L4-90)

- NVIDIA L4, 24GB VRAM, 22 CPU cores, 90GB RAM
- SSH config entry: `gpu`
- Cost: ~$1/hr -- STOP when not using
- Training data: `~/training_data_pos/` (39 position-mode episodes from session 1)
- Models: `~/models/` (various from session 1)
- distrobox aic_eval container: created and ready
- pixi: installed at `~/.pixi/bin/pixi`

## Key Technical Details

### Policy Interface
```python
def insert_cable(self, task, get_observation, move_robot, send_feedback):
    obs = get_observation()  # Observation msg at up to 20Hz
    move_robot(motion_update)  # MotionUpdate or JointMotionUpdate
```

### Scoring (per trial, max 100)
- Tier 1: Model validity (0-1)
- Tier 2: Smoothness (0-6) + Duration (0-12) + Efficiency (0-6) + Force penalty (0 to -12) + Contact penalty (0 to -24)
- Tier 3: Correct insertion (75) / Wrong port (-12) / Partial (38-50) / Proximity (0-25)

### Infrastructure Rules
- **ALWAYS nohup** for GPU training >10 min (SSH timeout kills processes)
- **ALWAYS verify** pixi reinstall picked up changes before eval
- **NEVER collect training data through Docker compose** (different distribution)
- **NEVER assume Docker eval = pixi eval** (they give different results)

## Conventions

- Python code follows ruff formatting
- Commit messages use gitmoji
- Task tracking via beads-rust (`br`)
- Always push commits after experiments
