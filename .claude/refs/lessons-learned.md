# Lessons Learned: AIC Hackathon Testing Loop

Accumulated from 32 experiments across 3 branches, ~$15 GPU cost, 2 days.

## Infrastructure Lessons

### Docker is for submission, not iteration
- Docker layer caching causes stale code (pixi build cache persists across rebuilds)
- Fix: removed `--mount=type=cache,target=.pixi/build` from Dockerfile
- Fix: always `docker compose build --no-cache` if code changes aren't reflected
- The `pixi reinstall ros-kilted-aic-example-policies` step is REQUIRED after any code change
- For iteration: use pixi natively on GPU. Docker only for final submission build.

### Distrobox doesn't work over non-interactive SSH
- Zenoh router in distrobox can't communicate with pixi processes over SSH
- The `~/run-eval.sh` script uses distrobox but ONLY works in interactive sessions
- Docker compose handles Zenoh networking correctly between containers
- **RULE: always use `docker compose` for eval, never distrobox over SSH**

### Volume mounts for Docker
- Can't mount a file onto a file (Docker error). Must mount directories.
- The eval container's config is at `/ws_aic/install/share/aic_engine/config/`
- Mount a directory containing `sample_config.yaml` over it for custom configs
- Training data volume: mount host dir to `/root/training_data/` in model container

### GPU instance management
- OVH L4-90: $1/hr, 23GB VRAM, 22 CPU cores, 90GB RAM
- Training uses ~1-3GB VRAM (SimpleACT is small). GPU mainly for rendering in eval.
- Running 3 training jobs in parallel: each slows ~3x due to CPU contention, NOT GPU
- Docker eval + pixi training CAN run simultaneously
- ALWAYS stop the instance when not actively using it

## Experiment Methodology Lessons

### One variable at a time
- Changing step size AND stiffness AND spiral radius = can't tell what helped
- exp-005 (wider spiral + more push + slower) scored WORSE than exp-004
- Best results came from isolated changes: just fix depth, just add color detection

### Scores have ~30-point variance
- Same policy, different randomized board config → 80 to 110 range
- Never draw conclusions from a single eval run
- The 110.4 "best" was a lucky randomization; true average was ~95
- When comparing: a 5-point difference is noise. Need 15+ to be significant.

### Speed matters more than you think
- Duration score (0-12 pts per trial) × 3 trials = 36 potential points
- CheatCode takes ~38s per trial. ACT v6 takes 1.1s.
- Faster policies get higher Tier 2 scores even with same Tier 3

### Don't over-invest in a plateaued branch
- Branch A (classical) plateaued at 93.4 after 8 experiments
- Branch B (camera) plateaued at ~100 after 16 experiments (24 total)
- Should have pivoted to ACT after 5-6 experiments on Branch B, not 16
- The autoresearch rule (5 consecutive discards → pivot) was right

## ACT Training Lessons

### Data diversity > data quantity > training duration
- 3 demos from 1 config: 102 (5 epochs OR 50 epochs -- same!)
- 6 demos from 2 configs: 120.7 (huge jump from diversity)
- 9 demos from 3 configs: 120.7 (saturated at that diversity level)
- 27 demos from 8 configs: 121.9 (marginal improvement)
- **Each new CONFIG helps more than more demos from the same config**

### Training epochs have diminishing returns
- 5 epochs (45s): 101.9
- 50 epochs (11min): 101.9 (same score, lower loss)
- 100 epochs (80min): 121.9 (with diverse data)
- 300 epochs: killed at 29ep (SSH timeout). Not worth the time.
- Sweet spot: 100 epochs is enough for 27-42 episodes

### SimpleACT model is sufficient (for now)
- 29.6M params, ResNet-18 backbone, transformer decoder
- Trains in ~80 min on L4 GPU with 27 episodes
- Inference at 4Hz (0.25s per step), which is fine
- No need for larger model until we have 100+ diverse demos

### The CheatCode expert has limitations
- CheatCode uses ground_truth TF frames (not available during eval)
- CheatCode's trajectory is a smooth descent + integrator feedback
- When trained on CheatCode demos, ACT learns to descend smoothly
- BUT: CheatCode's XY correction relies on TF, which ACT can't replicate
- ACT must learn XY alignment from camera images alone

## Data Collection Lessons

### Config randomization must happen BETWEEN runs, not WITHIN
- The `sample_config.yaml` defines fixed board positions
- The engine doesn't randomize beyond what the config specifies
- To get diverse data: create multiple config files with different board poses
- Use `scripts/generate_random_configs.py` to create variants
- Mount different configs into the eval container for each collection run

### Volume mount for training data
- DataCollector saves to `/root/training_data/` inside the model container
- The Docker volume mount (`../../training_data:/root/training_data`) didn't work
  reliably due to path resolution issues
- Reliable approach: `docker cp aic-model-1:/root/training_data/` after each run
- Rename episodes to avoid counter collisions between runs

### DataCollector must wrap move_robot, not just get_observation
- CheatCode doesn't call `get_observation()` -- it uses TF lookups
- The recording wrapper on `get_observation` captured ZERO data initially
- Fix: also wrap `move_robot` to trigger observation recording at each command

## What Worked Best (Score Leaderboard)

| Rank | Score | Experiment | Branch | Key Insight |
|------|-------|-----------|--------|-------------|
| 1 | 121.9 | ACT v6 (27 demos, 8 configs) | C | Diverse data + ACT |
| 2 | 120.7 | ACT v4 (6 demos, 2 configs) | C | First diverse ACT |
| 3 | 115 | ACT v7 / chunk20 | C | Not enough training/diversity |
| 4 | 110.4 | IBVS (lucky run) | B | Oscillation helped by accident |
| 5 | 102 | ACT micro (3 demos, 5ep) | C | Proved ACT pipeline works |
| 6 | 100.1 | IBVS averaged | B | Stable camera approach |
| 7 | 98.5 | VisionPolicy color | B | Color detection breakthrough |
| 8 | 93.4 | DirectApproach v2 | A | Best classical, partial insertion |
| 9 | 78.4 | DirectApproach v1 | A | First proximity points |

## What to Try Next

1. **More diverse configs** (14→30+) with ACT training
2. **Position-mode actions** instead of velocity (match CheatCode more directly)
3. **Combine IBVS + ACT** (HybridPolicy already written, not yet tested)
4. **Image augmentation** during training (crops, brightness, noise)
5. **Larger model** if diverse data saturates SimpleACT capacity
6. **Train on SC-specific demos** separately if SC trial remains weak
