#!/usr/bin/env bash
# ============================================================================
# gpu-setup.sh -- One-shot setup for OVH L4-90 cloud GPU eval runner
# ============================================================================
#
# Run this script ON the OVH L4-90 instance to prepare it as a remote eval
# runner and training machine.
#
# Prerequisites:
#   - Ubuntu 24.04 (OVH L4-90 default)
#   - NVIDIA drivers pre-installed (OVH default)
#   - Network access for apt, pixi, and Docker
#
# Usage:
#   bash gpu-setup.sh                # Full setup
#   bash gpu-setup.sh --verify-only  # Just check if everything works
#
# What this script does:
#   1. Verifies NVIDIA GPU is present (nvidia-smi)
#   2. Installs system deps (git, curl, build-essential)
#   3. Installs Pixi package manager
#   4. Installs Docker + NVIDIA Container Toolkit
#   5. Creates workspace and prompts for repo clone
#   6. Runs pixi install
#   7. Disables GlobalIllumination in aic.sdf for headless performance
#   8. Creates ~/run-eval.sh helper script
#   9. Verifies GPU, Gazebo, ROS 2, and Docker with GPU access
# ============================================================================

set -euo pipefail

# -- Colors for output -------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; }

# -- Configuration -----------------------------------------------------------
WORKSPACE_DIR="$HOME/ws_aic/src/aic"
RESULTS_DIR="$HOME/aic_results"
RUN_EVAL_SCRIPT="$HOME/run-eval.sh"
SDF_FILE="$WORKSPACE_DIR/aic_description/world/aic.sdf"

# -- Pre-flight checks -------------------------------------------------------
check_linux() {
    if [[ "$(uname)" != "Linux" ]]; then
        err "This script is intended for Linux only."
        exit 1
    fi
    local version
    version=$(lsb_release -ds 2>/dev/null || cat /etc/os-release | grep PRETTY_NAME | cut -d= -f2 | tr -d '"')
    info "OS: $version"
    ok "Linux detected."
}

check_nvidia_gpu() {
    info "Checking for NVIDIA GPU..."
    if ! command -v nvidia-smi &>/dev/null; then
        err "nvidia-smi not found. NVIDIA drivers must be installed."
        echo ""
        echo "On OVH L4-90, drivers should be pre-installed."
        echo "If not, install them with:"
        echo "  sudo apt install -y nvidia-driver-550"
        echo "  sudo reboot"
        exit 1
    fi

    local gpu_name
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    local gpu_mem
    gpu_mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)

    if [[ -z "$gpu_name" ]]; then
        err "nvidia-smi found but no GPU detected."
        exit 1
    fi

    ok "GPU detected: $gpu_name ($gpu_mem)"
}

# -- Installation steps -------------------------------------------------------
install_system_deps() {
    info "Installing system dependencies..."
    sudo apt update
    sudo apt install -y git curl build-essential wget
    ok "System dependencies installed."
}

install_pixi() {
    info "Checking Pixi..."
    if ! command -v pixi &>/dev/null; then
        info "Installing Pixi..."
        curl -fsSL https://pixi.sh/install.sh | sh
        # Add pixi to PATH for current session
        export PATH="$HOME/.pixi/bin:$PATH"
    fi
    if command -v pixi &>/dev/null; then
        ok "Pixi is available: $(pixi --version)"
    else
        err "Pixi installation failed. Please install manually: https://pixi.sh"
        exit 1
    fi
}

install_docker() {
    info "Checking Docker..."
    if command -v docker &>/dev/null; then
        ok "Docker is available: $(docker --version)"
    else
        info "Installing Docker..."
        sudo apt install -y docker.io docker-compose-v2
        ok "Docker installed."
    fi

    # Ensure current user is in docker group
    if ! groups | grep -q docker; then
        info "Adding $USER to docker group..."
        sudo usermod -aG docker "$USER"
        warn "You may need to log out and back in for the docker group to take effect."
        warn "Alternatively, run: newgrp docker"
    fi
    ok "Docker group membership configured."
}

install_nvidia_container_toolkit() {
    info "Checking NVIDIA Container Toolkit..."
    if dpkg -l nvidia-container-toolkit &>/dev/null 2>&1; then
        ok "NVIDIA Container Toolkit is already installed."
        return
    fi

    info "Installing NVIDIA Container Toolkit..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
        sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 2>/dev/null || true
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
    sudo apt update
    sudo apt install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    ok "NVIDIA Container Toolkit installed and configured."
}

setup_workspace() {
    info "Setting up workspace at $WORKSPACE_DIR..."
    mkdir -p "$(dirname "$WORKSPACE_DIR")"

    if [[ -d "$WORKSPACE_DIR" && -f "$WORKSPACE_DIR/pixi.toml" ]]; then
        ok "Workspace already exists at $WORKSPACE_DIR."
    else
        warn "Repo not found at $WORKSPACE_DIR."
        echo ""
        echo "Please clone the repo manually:"
        echo "  mkdir -p $(dirname "$WORKSPACE_DIR")"
        echo "  cd $(dirname "$WORKSPACE_DIR")"
        echo "  git clone <repo-url> aic"
        echo ""
        echo "Then re-run this script."
        exit 1
    fi
}

run_pixi_install() {
    info "Running pixi install (this may take several minutes)..."
    cd "$WORKSPACE_DIR"
    pixi install
    ok "Pixi install completed."
}

disable_global_illumination() {
    info "Disabling GlobalIllumination in aic.sdf for headless performance..."
    if [[ ! -f "$SDF_FILE" ]]; then
        warn "SDF file not found at $SDF_FILE -- skipping GI disable."
        return
    fi

    # Replace <enabled>true</enabled> within GlobalIllumination blocks
    if grep -q 'GlobalIllumination' "$SDF_FILE"; then
        # Linux sed (GNU) does not need '' after -i
        sed -i '/<plugin.*GlobalIllumination/,/<\/plugin>/ s/<enabled>true<\/enabled>/<enabled>false<\/enabled>/g' "$SDF_FILE"
        ok "GlobalIllumination disabled in aic.sdf."
        warn "Visual appearance will differ from default -- physics and scoring are unaffected."
    else
        info "No GlobalIllumination plugin found in SDF -- nothing to change."
    fi
}

create_run_eval_script() {
    info "Creating eval runner script at $RUN_EVAL_SCRIPT..."
    cat > "$RUN_EVAL_SCRIPT" << 'EVALSCRIPT'
#!/usr/bin/env bash
# ============================================================================
# run-eval.sh -- Run a single evaluation on the OVH L4-90 GPU instance
# ============================================================================
#
# Usage:
#   POLICY=aic_example_policies.ros.BlindPush bash ~/run-eval.sh
#
# Environment variables:
#   POLICY       -- Policy class name (default: aic_example_policies.ros.BlindPush)
#   AIC_RESULTS_DIR -- Where to write results (default: ~/aic_results)
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$HOME/ws_aic/src/aic"
export AIC_RESULTS_DIR="${AIC_RESULTS_DIR:-$HOME/aic_results}"
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
export ZENOH_CONFIG_OVERRIDE="${ZENOH_CONFIG_OVERRIDE:-transport/shared_memory/enabled=false}"

POLICY="${POLICY:-aic_example_policies.ros.BlindPush}"

mkdir -p "$AIC_RESULTS_DIR"
cd "$SCRIPT_DIR"

echo "=== Starting evaluation ==="
echo "  Policy:  $POLICY"
echo "  Results: $AIC_RESULTS_DIR"
echo "  Time:    $(date)"
echo ""

# Start the Zenoh router (needed for ROS 2 rmw_zenoh_cpp communication)
echo "--- Starting Zenoh router ---"
pixi run ros2 run rmw_zenoh_cpp rmw_zenohd &
ZENOH_PID=$!
sleep 3

# Launch the eval environment (Gazebo + aic_engine)
echo "--- Launching eval environment ---"
pixi run ros2 launch aic_bringup aic_gz_bringup.launch.py \
    gazebo_gui:=false launch_rviz:=false \
    ground_truth:=false start_aic_engine:=true \
    shutdown_on_aic_engine_exit:=true &
EVAL_PID=$!

# Wait for the environment to be ready
echo "--- Waiting for environment to initialize ---"
sleep 15

# Run the policy
echo "--- Running policy: $POLICY ---"
pixi run ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true \
    -p "policy:=$POLICY"

# Wait for eval to finish
wait $EVAL_PID 2>/dev/null || true

# Clean up Zenoh router
kill $ZENOH_PID 2>/dev/null || true
wait $ZENOH_PID 2>/dev/null || true

echo ""
echo "=== Evaluation complete ==="
echo "  Time: $(date)"
if [[ -f "$AIC_RESULTS_DIR/scoring.yaml" ]]; then
    echo "  Results: $AIC_RESULTS_DIR/scoring.yaml"
    echo ""
    cat "$AIC_RESULTS_DIR/scoring.yaml"
else
    echo "  WARNING: scoring.yaml not found in $AIC_RESULTS_DIR"
fi
EVALSCRIPT

    chmod +x "$RUN_EVAL_SCRIPT"
    ok "Eval runner script created at $RUN_EVAL_SCRIPT."
}

# -- Verification -------------------------------------------------------------
verify_installation() {
    info "Verifying installation..."

    local failed=0

    echo -n "  Checking NVIDIA GPU... "
    if nvidia-smi &>/dev/null; then
        local gpu_name
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        ok "$gpu_name"
    else
        err "nvidia-smi failed."
        failed=1
    fi

    if [[ -d "$WORKSPACE_DIR" ]]; then
        cd "$WORKSPACE_DIR"

        echo -n "  Checking Gazebo... "
        if pixi run gz sim --version &>/dev/null; then
            ok "$(pixi run gz sim --version 2>&1 | head -1)"
        else
            err "Gazebo not working."
            failed=1
        fi

        echo -n "  Checking ROS 2... "
        if pixi run ros2 --help &>/dev/null; then
            ok "ROS 2 is functional."
        else
            err "ROS 2 not working."
            failed=1
        fi
    else
        warn "Workspace not found at $WORKSPACE_DIR -- skipping Gazebo/ROS 2 checks."
    fi

    echo -n "  Checking Docker... "
    if command -v docker &>/dev/null; then
        ok "$(docker --version)"
    else
        warn "Docker not installed."
    fi

    echo -n "  Checking Docker GPU access... "
    if docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi &>/dev/null 2>&1; then
        ok "Docker can access GPU."
    else
        warn "Docker GPU access not working (may need group reload or toolkit install)."
    fi

    echo -n "  Checking run-eval.sh... "
    if [[ -x "$RUN_EVAL_SCRIPT" ]]; then
        ok "$RUN_EVAL_SCRIPT is executable."
    else
        err "$RUN_EVAL_SCRIPT missing or not executable."
        failed=1
    fi

    echo -n "  Checking results directory... "
    mkdir -p "$RESULTS_DIR"
    ok "$RESULTS_DIR exists."

    if [[ "$failed" -eq 0 ]]; then
        echo ""
        ok "All checks passed. GPU instance is ready for remote evaluation."
        echo ""
        echo "To test manually:"
        echo "  POLICY=aic_example_policies.ros.BlindPush bash ~/run-eval.sh"
    else
        echo ""
        err "Some checks failed. Please fix the issues above and re-run:"
        echo "  bash gpu-setup.sh --verify-only"
        exit 1
    fi
}

# -- Main ---------------------------------------------------------------------
main() {
    echo "============================================"
    echo "  OVH L4-90 Cloud GPU Eval Runner Setup"
    echo "============================================"
    echo ""

    if [[ "${1:-}" == "--verify-only" ]]; then
        check_linux
        check_nvidia_gpu
        verify_installation
        return
    fi

    check_linux
    check_nvidia_gpu
    install_system_deps
    install_pixi
    install_docker
    install_nvidia_container_toolkit
    setup_workspace
    run_pixi_install
    disable_global_illumination
    create_run_eval_script
    verify_installation

    echo ""
    info "Setup complete. Configure SSH on your development host:"
    echo "  Host gpu"
    echo "      HostName <ovh-instance-ip>"
    echo "      User $(whoami)"
    echo "      IdentityFile ~/.ssh/id_ed25519"
    echo "      ForwardAgent yes"
}

main "$@"
