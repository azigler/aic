# Spec: Mac Studio Remote Runner

## Overview

Use a Mac Studio M1 Max (64GB RAM, 1TB SSD) as a remote GPU-accelerated eval
runner. Experiments are developed on this machine, synced to the Mac, executed
with Gazebo using Metal GPU, and results fetched back for analysis.

## Architecture

```
┌─────────────────────┐         SSH/rsync          ┌──────────────────────────┐
│  Development Host   │ ───────────────────────────▶│    Mac Studio Runner     │
│  (this machine)     │                             │  M1 Max, 64GB, Metal GPU │
│                     │  1. rsync policy code ─────▶│                          │
│  - Edit policy      │                             │  - Gazebo (native)       │
│  - Analyze results  │  2. ssh run-eval.sh ───────▶│  - ROS 2 Kilted (pixi)  │
│  - Track beads      │                             │  - aic_engine + scoring  │
│  - Git push         │  3. rsync results back ◀────│                          │
│                     │                             │  ~/aic_results/          │
└─────────────────────┘                             └──────────────────────────┘
```

## Mac Studio Setup (native, no Docker)

### Prerequisites
- macOS 13+ (Ventura or later)
- Xcode Command Line Tools
- Homebrew

### Install Script Steps

1. **Homebrew packages:**
   ```bash
   brew install cmake git wget curl
   ```

2. **Pixi:**
   ```bash
   curl -fsSL https://pixi.sh/install.sh | sh
   ```

3. **Clone repo and install:**
   ```bash
   mkdir -p ~/ws_aic/src
   cd ~/ws_aic/src
   git clone <repo-url> aic
   cd aic
   pixi install
   ```

4. **Verify Gazebo + ROS 2:**
   ```bash
   pixi run gz sim --version
   pixi run ros2 --version
   ```

5. **Create run script** at `~/run-eval.sh`:
   ```bash
   #!/bin/bash
   cd ~/ws_aic/src/aic
   export AIC_RESULTS_DIR=~/aic_results
   mkdir -p $AIC_RESULTS_DIR

   # Terminal 1 equivalent: launch eval environment
   pixi run ros2 launch aic_bringup aic_gz_bringup.launch.py \
     gazebo_gui:=false launch_rviz:=false \
     ground_truth:=false start_aic_engine:=true \
     shutdown_on_aic_engine_exit:=true &
   EVAL_PID=$!

   # Wait for engine to be ready
   sleep 10

   # Terminal 2 equivalent: run policy
   pixi run ros2 run aic_model aic_model --ros-args \
     -p use_sim_time:=true \
     -p policy:=${POLICY:-aic_example_policies.ros.BlindPush}

   # Wait for eval to finish
   wait $EVAL_PID

   echo "Results at: $AIC_RESULTS_DIR/scoring.yaml"
   ```

### Caveats
- Gazebo on macOS uses OGRE2 with Metal backend -- visual appearance may differ
  from Linux/OpenGL slightly, but physics and scoring are identical
- ROS 2 Kilted via robostack/pixi should work on Apple Silicon
- If pixi can't resolve some packages for osx-arm64, may need to build from source
- MuJoCo is fully native on Apple Silicon and could be an alternative for training

## Remote Execution Flow

### From Development Host

```bash
# 1. Sync code changes to Mac
rsync -avz --exclude='.pixi' --exclude='__pycache__' --exclude='.git' \
  ~/aic/ mac:~/ws_aic/src/aic/

# 2. Run eval remotely
ssh mac "cd ~/ws_aic/src/aic && POLICY=aic_example_policies.ros.BlindPush bash ~/run-eval.sh"

# 3. Fetch results
rsync -avz mac:~/aic_results/scoring.yaml ~/aic/aic_results/

# 4. Parse results locally
cat ~/aic/aic_results/scoring.yaml
```

### Wrapper Script: `scripts/remote-eval.sh`

Combines steps 1-3 into a single command:

```bash
#!/bin/bash
# Usage: ./scripts/remote-eval.sh <policy_class> [mac_host]
POLICY=${1:-aic_example_policies.ros.BlindPush}
MAC_HOST=${2:-mac}

echo "=== Syncing code to $MAC_HOST ==="
rsync -avz --exclude='.pixi' --exclude='__pycache__' --exclude='.git' \
  --exclude='aic_results' --exclude='.beads' \
  . $MAC_HOST:~/ws_aic/src/aic/

echo "=== Running eval on $MAC_HOST with policy=$POLICY ==="
ssh $MAC_HOST "cd ~/ws_aic/src/aic && POLICY=$POLICY bash ~/run-eval.sh" 2>&1 | \
  tee /tmp/remote-eval.log

echo "=== Fetching results ==="
rsync -avz $MAC_HOST:~/aic_results/ ./aic_results/

echo "=== Results ==="
cat ./aic_results/scoring.yaml 2>/dev/null || echo "No scoring.yaml found"
```

## SSH Configuration

On the development host, add to `~/.ssh/config`:

```
Host mac
    HostName <mac-ip-or-hostname>
    User <username>
    IdentityFile ~/.ssh/id_ed25519
    ForwardAgent yes
```

## Skill Integration

The `/experiment run` skill should detect whether a remote runner is configured
and use it automatically:

```
if SSH_RUNNER configured in .claude/refs/runner-config:
    rsync -> ssh run -> rsync results
else:
    docker compose up (local)
```

## Expected Performance

| Setup | Eval Time | Experiments/Hour |
|-------|-----------|-----------------|
| Local CPU-only | ~27 min | ~2 |
| Mac Docker (Rosetta) | ~12-15 min | ~4 |
| Mac Native (Metal) | ~10-12 min | ~5-6 |
| Cloud GPU (L4) | ~5-8 min | ~8-10 |
