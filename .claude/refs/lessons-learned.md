# Lessons Learned from Session 1

Hard-won knowledge from ~50 experiments across 2 agent sessions.

## Infrastructure

- **distrobox + pixi for iteration, Docker compose ONLY for submission.** Docker eval
  gives different scores. Docker-collected training data has different distribution.
- **ALWAYS nohup** for GPU training >10 min. SSH timeout killed training at epoch 70.
- **pixi reinstall** after every code change, or changes won't be reflected.
- **`gpus: all`** required in docker-compose.yaml for both eval and model services.

## Training

- **Config diversity > data quantity.** Each new board config helps more than more
  demos from the same config. 8 configs → 120, 13 configs same score.
- **50 epochs is the sweet spot** for ~24 demos. >50 overfits.
- **Data augmentation hurts.** The model needs precise visual features for port
  localization. Random brightness/contrast destroys these.
- **Bottleneck oversampling hurts.** The model needs the full trajectory equally.
- **val_loss does NOT predict score.** A model with 40% better val_loss scored 10×
  worse. The training distribution must match eval distribution.

## Architecture

- **Use LeRobot ACTPolicy.** The competition provides RunACT.py with LeRobot's
  battle-tested ACT implementation. Our custom SimpleACT had argument order bugs,
  import issues, and never worked reliably across pixi/Docker environments.
- **Velocity mode** is what RunACT uses (7D twist). Position mode was a custom deviation.
- **Image scaling 0.25** (288×256), not fixed 256×256 resize.

## What NOT to do

- Don't build custom model architectures when a standard one is provided
- Don't use Docker compose for iteration (only submission)
- Don't collect training data through Docker (distribution shift)
- Don't mix data from different collection methods
- Don't assume local analysis matches GPU eval results
- Don't change multiple variables per experiment
- Don't run long training without checking intermediate results
