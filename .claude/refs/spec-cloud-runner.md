# Spec: OVH Cloud L4 GPU Runner

## Overview

Use an OVH L4-90 cloud instance as the primary development, training, and
submission machine. This instance has the **exact same GPU** (NVIDIA L4, 24GB VRAM)
as the official cloud evaluation, eliminating sim-to-sim GPU discrepancy.

## Instance Specs

| Spec | L4-90 | Cloud Eval (reference) |
|------|-------|----------------------|
| GPU | NVIDIA L4 (24GB VRAM) | NVIDIA L4 (24GB VRAM) |
| CPU | 22 cores | 64 cores |
| RAM | 90 GB | 256 GB |
| Storage | 400 GB NVMe | -- |
| OS | Ubuntu 24.04 | Ubuntu 24.04 |
| Price | $1.00/hr | -- |

## Architecture

```
┌─────────────────────┐         SSH              ┌──────────────────────────┐
│  Development Host   │ ────────────────────────▶│   OVH L4-90 Cloud GPU   │
│  (this machine)     │                           │  NVIDIA L4, 22 cores     │
│                     │  1. rsync policy code ──▶│                          │
│  - Edit policy      │                           │  - Gazebo (native, GPU)  │
│  - Analyze results  │  2. ssh run-eval.sh ────▶│  - ROS 2 Kilted (pixi)  │
│  - Track beads      │                           │  - aic_engine + scoring  │
│  - Git push         │  3. rsync results back ◀─│  - PyTorch (CUDA)       │
│                     │                           │  - Docker (submissions)  │
│  OR: SSH directly   │                           │                          │
│  and work on GPU    │                           │  ~/aic_results/          │
│  instance itself    │                           │  ~/ws_aic/src/aic/       │
└─────────────────────┘                           └──────────────────────────┘
```

## Two Usage Modes

### Mode A: Remote Runner (rsync + SSH)
Edit code locally, run eval remotely. Best for quick policy tweaks.

```bash
scripts/remote-eval.sh <policy_class>
```

### Mode B: Direct SSH
SSH into the GPU instance and work directly. Best for training, debugging,
interactive exploration. Claude Code can SSH in and run everything there.

```bash
ssh gpu   # then work directly on the instance
```

## Instance Setup (one-time)

### 1. OS + NVIDIA Drivers
OVH L4-90 comes with Ubuntu 24.04 and NVIDIA drivers pre-installed.
Verify with `nvidia-smi`.

### 2. Core Tools
```bash
sudo apt update && sudo apt install -y git curl build-essential
```

### 3. Pixi (manages ROS 2 + all deps)
```bash
curl -fsSL https://pixi.sh/install.sh | sh
source ~/.bashrc
```

### 4. Clone Repo + Install
```bash
mkdir -p ~/ws_aic/src
cd ~/ws_aic/src
git clone <repo-url> aic
cd aic
pixi install    # Downloads ROS 2, Gazebo, PyTorch, etc.
```

### 5. Docker (for submission builds only)
```bash
sudo apt install -y docker.io docker-compose-v2
sudo usermod -aG docker $USER
# Log out and back in for group to take effect
```

### 6. NVIDIA Container Toolkit (for Docker GPU access)
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 7. Verify
```bash
nvidia-smi                                    # GPU visible
pixi run gz sim --version                     # Gazebo works
pixi run ros2 --version                       # ROS 2 works
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi  # Docker GPU
```

### 8. Create run-eval.sh
The setup script creates `~/run-eval.sh` which:
- Starts Zenoh router
- Launches Gazebo + aic_engine headless
- Runs the specified policy
- Writes results to ~/aic_results/scoring.yaml

## Eval Performance Estimate

| Phase | Time | Notes |
|-------|------|-------|
| Gazebo startup | ~30s | GPU-accelerated rendering |
| Trial setup (spawn board + cable) | ~15s | Per trial |
| Trial execution | ~60-180s | Depends on policy (sim-time = wall-time) |
| Scoring | ~5s | Per trial |
| **Total 3 trials** | **~5-10 min** | Near real-time with L4 GPU |

This gives **6-12 experiments per hour** vs 2/hr on CPU-only.

## Training on GPU

The L4 with 24GB VRAM supports:
- **ACT training**: batch size 32-64 with 3 cameras
- **Diffusion policy**: batch size 16-32
- **RL (Isaac Lab)**: parallel envs on GPU
- **PyTorch MPS**: N/A (CUDA instead, faster)

```bash
# Example: train ACT policy
ssh gpu "cd ~/ws_aic/src/aic && pixi run python train_act.py --batch-size 32 --epochs 100"
```

## Submission from GPU Instance

```bash
# On the GPU instance:
docker compose -f docker/docker-compose.yaml build model
docker tag my-solution:v1 973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/<team>:vN
docker push 973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/<team>:vN
```

Docker build on the GPU instance will be fast (NVMe storage, 22 cores).

## Cost Management

- **Only pay when instance is running** -- stop when not experimenting
- Budget: ~$1.00/hr × 5hr/day × 50 days = ~$250 total
- OVH bills hourly; stop the instance when done for the day
- Storage persists when stopped (small monthly cost)

## SSH Configuration

On dev host (`~/.ssh/config`):
```
Host gpu
    HostName <ovh-instance-ip>
    User ubuntu
    IdentityFile ~/.ssh/id_ed25519
    ForwardAgent yes
```

## Replaces

This spec **replaces** the Mac Studio runner spec (`spec-mac-runner.md`).
All references to "Mac runner" in skills should be updated to "cloud GPU runner".
The `scripts/runner-config.sh` should default to `GPU_HOST` instead of `MAC_HOST`.
