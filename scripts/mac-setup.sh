#!/usr/bin/env bash
# ============================================================================
# mac-setup.sh -- One-shot setup for Mac Studio M1 Max eval runner
# ============================================================================
#
# Run this script ON the Mac Studio to prepare it as a remote eval runner.
#
# Prerequisites:
#   - macOS 13+ (Ventura or later)
#   - Xcode Command Line Tools: xcode-select --install
#   - Network access for Homebrew and pixi
#
# Usage:
#   bash mac-setup.sh                # Full setup
#   bash mac-setup.sh --verify-only  # Just check if everything works
#
# What this script does:
#   1. Installs Homebrew dependencies (cmake, git, wget, curl)
#   2. Installs Pixi package manager
#   3. Installs Docker Desktop (for submission builds)
#   4. Creates workspace and prompts for repo clone
#   5. Runs pixi install
#   6. Creates ~/run-eval.sh helper script
#   7. Disables GlobalIllumination in aic.sdf for headless performance
#   8. Verifies Gazebo and ROS 2 are functional
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
check_macos() {
    if [[ "$(uname)" != "Darwin" ]]; then
        err "This script is intended for macOS only."
        exit 1
    fi
    local version
    version=$(sw_vers -productVersion)
    info "macOS version: $version"

    local major
    major=$(echo "$version" | cut -d. -f1)
    if [[ "$major" -lt 13 ]]; then
        err "macOS 13 (Ventura) or later is required. Found: $version"
        exit 1
    fi
    ok "macOS version is supported."
}

check_xcode_cli() {
    if ! xcode-select -p &>/dev/null; then
        warn "Xcode Command Line Tools not found. Installing..."
        xcode-select --install
        echo "Please complete the Xcode CLI tools installation and re-run this script."
        exit 1
    fi
    ok "Xcode Command Line Tools are installed."
}

# -- Installation steps -------------------------------------------------------
install_homebrew_deps() {
    info "Checking Homebrew..."
    if ! command -v brew &>/dev/null; then
        info "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        # Add brew to PATH for Apple Silicon
        if [[ -f /opt/homebrew/bin/brew ]]; then
            eval "$(/opt/homebrew/bin/brew shellenv)"
        fi
    fi
    ok "Homebrew is available."

    info "Installing Homebrew packages: cmake git wget curl..."
    brew install cmake git wget curl 2>/dev/null || true
    ok "Homebrew packages installed."
}

install_pixi() {
    info "Checking Pixi..."
    if ! command -v pixi &>/dev/null; then
        info "Installing Pixi..."
        curl -fsSL https://pixi.sh/install.sh | sh
        # Add pixi to PATH
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
        return
    fi

    warn "Docker not found. Docker Desktop is needed for submission builds."
    echo ""
    if command -v brew &>/dev/null; then
        info "Installing Docker Desktop via Homebrew..."
        brew install --cask docker
        echo ""
        info "Docker Desktop installed. Please:"
        echo "  1. Open Docker Desktop from Applications"
        echo "  2. Complete the setup wizard"
        echo "  3. Ensure Docker is running (whale icon in menu bar)"
        echo ""
        warn "Docker is only needed for final submission -- not for iteration."
    else
        echo "Please install Docker Desktop manually:"
        echo "  https://www.docker.com/products/docker-desktop/"
        echo ""
        warn "Docker is only needed for final submission -- not for iteration."
    fi
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
    # Use sed to find GlobalIllumination sections and disable them
    if grep -q 'GlobalIllumination' "$SDF_FILE"; then
        # macOS sed requires '' after -i
        sed -i '' '/<plugin.*GlobalIllumination/,/<\/plugin>/ s/<enabled>true<\/enabled>/<enabled>false<\/enabled>/g' "$SDF_FILE"
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
# run-eval.sh -- Run a single evaluation on the Mac Studio
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
    cd "$WORKSPACE_DIR"

    local failed=0

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
        ok "All checks passed. Mac Studio is ready for remote evaluation."
        echo ""
        echo "To test manually:"
        echo "  POLICY=aic_example_policies.ros.BlindPush bash ~/run-eval.sh"
    else
        echo ""
        err "Some checks failed. Please fix the issues above and re-run:"
        echo "  bash mac-setup.sh --verify-only"
        exit 1
    fi
}

# -- Main ---------------------------------------------------------------------
main() {
    echo "============================================"
    echo "  Mac Studio Eval Runner Setup"
    echo "============================================"
    echo ""

    if [[ "${1:-}" == "--verify-only" ]]; then
        check_macos
        verify_installation
        return
    fi

    check_macos
    check_xcode_cli
    install_homebrew_deps
    install_pixi
    install_docker
    setup_workspace
    run_pixi_install
    disable_global_illumination
    create_run_eval_script
    verify_installation

    echo ""
    info "Setup complete. Configure SSH on your development host:"
    echo "  Host mac"
    echo "      HostName $(hostname)"
    echo "      User $(whoami)"
    echo "      IdentityFile ~/.ssh/id_ed25519"
    echo "      ForwardAgent yes"
}

main "$@"
