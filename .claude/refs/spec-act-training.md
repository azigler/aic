# Spec: ACT Training Pipeline for Cable Insertion

## 1. Overview

Train an ACT (Action Chunking with Transformers) policy to perform cable insertion
by imitating CheatCode demonstrations. The pipeline: collect demonstrations →
process into LeRobot dataset → train ACT → deploy as policy.

### 1.1 Why ACT

Our camera perception approach plateaued at ~90-100/300 because the hand-coded
camera-to-base transform is not accurate enough for the 5mm port tolerance.
ACT learns the mapping from raw pixels to actions automatically, potentially
achieving full insertion (75 pts/trial) on all 3 trials.

### 1.2 The Reference Implementation

RunACT.py already exists in the repo. It loads a pre-trained model from
HuggingFace (`grkw/aic_act_policy`) and runs inference. We need to:
1. Collect our own demonstrations
2. Train our own model
3. Deploy using the same RunACT inference pipeline

## 2. Data Collection

### 2.1 Expert: CheatCode

CheatCode achieves full insertion with ground_truth=true. We use it as our expert
demonstrator. The key data to collect per timestep:

**Observations (input):**
- 3x camera images (left, center, right) at 1152x1024
- 26D robot state:
  - TCP position (3), orientation (4), linear vel (3), angular vel (3)
  - TCP error (6), joint positions (7)
- Wrist wrench (6D force+torque)

**Actions (output):**
- 7D velocity command: linear (3) + angular (3) + gripper (1)
- OR: 7D position command: TCP pose (6) + gripper (1)
- The RunACT baseline uses velocity mode

### 2.2 Data Collection Script

```python
# Pseudocode for data collection
for episode in range(100):
    # Launch sim with randomized board config
    # Run CheatCode policy
    # Record all observations + actions at each timestep
    # Save as episode in LeRobot HDF5 format
```

**Key requirements:**
- At least 50-100 episodes for ACT to generalize
- Diverse board configurations (randomized pose, port positions)
- Both SFP and SC insertion trials
- ~20Hz recording rate (matching observation frequency)
- Each episode: ~20-30s = 400-600 timesteps

### 2.3 Recording Format

LeRobot uses HDF5 datasets with this structure:
```
dataset/
├── episode_0/
│   ├── observation.images.left_camera  # (T, C, H, W) uint8
│   ├── observation.images.center_camera
│   ├── observation.images.right_camera
│   ├── observation.state  # (T, 26) float32
│   └── action  # (T, 7) float32
├── episode_1/
│   └── ...
└── meta/
    ├── stats.json  # normalization statistics
    └── info.json   # dataset metadata
```

### 2.4 Implementation Approach

Rather than building a custom recorder, modify CheatCode to:
1. Log observations + actions to a buffer
2. After each trial, save the buffer as an HDF5 episode
3. Run 100+ trials with different random configs

The data collection policy:
```python
class DataCollector(CheatCode):
    def insert_cable(self, task, get_observation, move_robot, send_feedback):
        self.observations = []
        self.actions = []

        # Wrap move_robot to capture actions
        def recording_move_robot(**kwargs):
            self.actions.append(kwargs)
            move_robot(**kwargs)

        # Run CheatCode's insert_cable with recording
        result = super().insert_cable(task, get_observation, recording_move_robot, send_feedback)

        # Save episode
        self.save_episode(self.observations, self.actions)
        return result
```

## 3. Training

### 3.1 ACT Architecture

From the RunACT implementation:
- **Vision encoder:** ResNet (or similar) for each camera → feature vectors
- **State encoder:** Linear layers for 26D robot state
- **Transformer:** Cross-attention between vision and state features
- **Action decoder:** Predicts chunks of 10-100 future actions at once
- **Training loss:** L1 loss on action predictions

### 3.2 Training Configuration

```python
config = ACTConfig(
    input_shapes={
        "observation.images.left_camera": (3, 256, 256),  # resized from 1024
        "observation.images.center_camera": (3, 256, 256),
        "observation.images.right_camera": (3, 256, 256),
        "observation.state": (26,),
    },
    output_shapes={"action": (7,)},
    chunk_size=50,  # predict 50 future actions
    n_action_steps=1,  # execute 1 action per step
    # ... other hyperparameters
)
```

### 3.3 Training on L4 GPU

- **Batch size:** 32-64 (24GB VRAM)
- **Epochs:** 100-500
- **Learning rate:** 1e-4 with cosine schedule
- **Image resolution:** 256x256 (downsample from 1024)
- **Estimated time:** 1-4 hours on L4

### 3.4 Training Command

```bash
ssh gpu "cd ~/ws_aic/src/aic && pixi run python scripts/train_act.py \
    --data-dir ~/training_data \
    --output-dir ~/models/act_v1 \
    --batch-size 32 \
    --epochs 200 \
    --lr 1e-4"
```

## 4. Deployment

### 4.1 Inference Policy

Modify RunACT.py to load our trained model instead of the HuggingFace one:

```python
class OurACT(RunACT):
    def __init__(self, parent_node):
        # Load from local path instead of HuggingFace
        policy_path = Path("~/models/act_v1")
        # ... rest of loading logic
```

### 4.2 Evaluation

```bash
POLICY=aic_example_policies.ros.OurACT scripts/remote-eval.sh
```

## 5. Implementation Plan

### Phase 1: Data Collection (2-3 hours)
1. Create `DataCollector` policy that wraps CheatCode
2. Run 100+ episodes with diverse configs
3. Save in LeRobot HDF5 format
4. Verify data quality (images, state, actions aligned)

### Phase 2: Training (2-4 hours)
1. Create `scripts/train_act.py` training script
2. Configure ACT for our observation/action spaces
3. Train on collected data
4. Monitor loss curves, save best checkpoint

### Phase 3: Deployment (1 hour)
1. Create `OurACT` policy that loads our model
2. Run eval
3. Compare against perception-based policy (current best ~100)

### Phase 4: Iteration (ongoing)
1. If score < expectation: collect more data, adjust hyperparameters
2. If score > current best: use as primary submission candidate
3. Can also combine: ACT for SFP trials, IBVS for SC trials

## 6. Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Data collection takes too long | Use parallel sim instances |
| ACT doesn't generalize to randomized configs | More diverse training data |
| Model too large for 24GB VRAM | Reduce image resolution, batch size |
| Training unstable | Use pre-trained vision backbone |
| Inference too slow (>250ms/step) | Optimize model, reduce resolution |

## 7. Expected Outcomes

| Metric | Current (IBVS) | Expected (ACT) |
|--------|---------------|----------------|
| SFP Trial 1 | 30-50 (proximity/partial) | 60-85 (partial/full insertion) |
| SFP Trial 2 | 20-40 (proximity) | 60-85 (partial/full insertion) |
| SC Trial 3 | 0-15 (far/proximity) | 40-70 (proximity/partial) |
| **Total** | **80-110** | **160-240** |
