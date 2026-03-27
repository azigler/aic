#!/usr/bin/env bash
# ============================================================================
# remote-eval.sh -- Run evaluation on a remote Mac Studio from this machine
# ============================================================================
#
# Usage:
#   ./scripts/remote-eval.sh                              # defaults from runner-config.sh
#   ./scripts/remote-eval.sh <policy_class>               # custom policy
#   ./scripts/remote-eval.sh <policy_class> <mac_host>    # custom policy + host
#
# Examples:
#   ./scripts/remote-eval.sh aic_example_policies.ros.BlindPush
#   ./scripts/remote-eval.sh aic_example_policies.ros.CheatCode mac-studio
#
# What this script does:
#   1. Checks SSH connectivity to the remote Mac
#   2. Rsyncs the current codebase (excluding build artifacts)
#   3. Runs ~/run-eval.sh on the Mac via SSH
#   4. Fetches scoring.yaml back to local aic_results/
#   5. Prints results and total elapsed time
#
# Prerequisites:
#   - SSH key-based auth to the Mac (see runner-config.sh for SSH setup)
#   - Mac has been set up with mac-setup.sh
#   - runner-config.sh has correct MAC_HOST
# ============================================================================

set -euo pipefail

# -- Load configuration ------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck source=runner-config.sh
source "$SCRIPT_DIR/runner-config.sh"

# -- Parse arguments ----------------------------------------------------------
POLICY="${1:-$DEFAULT_POLICY}"
MAC_HOST="${2:-$MAC_HOST}"

# -- Colors -------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; }

# -- Timer --------------------------------------------------------------------
START_TIME=$(date +%s)

elapsed() {
    local now
    now=$(date +%s)
    local secs=$((now - START_TIME))
    printf '%dm%02ds' $((secs / 60)) $((secs % 60))
}

# -- Connectivity check -------------------------------------------------------
info "Checking SSH connectivity to $MAC_HOST..."
if ! ssh -o ConnectTimeout="$SSH_TIMEOUT" -o BatchMode=yes "$MAC_HOST" "echo ok" &>/dev/null; then
    err "Cannot connect to $MAC_HOST via SSH."
    echo ""
    echo "Troubleshooting:"
    echo "  1. Ensure the Mac is powered on and reachable"
    echo "  2. Check your ~/.ssh/config has a 'Host $MAC_HOST' entry"
    echo "  3. Test manually: ssh $MAC_HOST echo hello"
    echo "  4. Ensure SSH key auth is set up (no password prompts)"
    exit 1
fi
ok "SSH connection to $MAC_HOST successful."

# -- Build rsync exclude args -------------------------------------------------
RSYNC_ARGS=(-avz --delete)
for pattern in $RSYNC_EXCLUDES; do
    RSYNC_ARGS+=(--exclude="$pattern")
done

# -- Step 1: Sync code --------------------------------------------------------
echo ""
info "=== Syncing code to $MAC_HOST:$REMOTE_DIR ==="
info "  Excludes: $RSYNC_EXCLUDES"
rsync "${RSYNC_ARGS[@]}" "$REPO_DIR/" "$MAC_HOST:$REMOTE_DIR/"
ok "Code synced. [$(elapsed)]"

# -- Step 2: Run eval ---------------------------------------------------------
echo ""
info "=== Running eval on $MAC_HOST ==="
info "  Policy: $POLICY"
info "  Remote dir: $REMOTE_DIR"
echo ""

ssh -o ConnectTimeout="$SSH_TIMEOUT" "$MAC_HOST" \
    "cd $REMOTE_DIR && POLICY='$POLICY' bash ~/run-eval.sh" 2>&1 | \
    tee /tmp/remote-eval.log

EVAL_EXIT=${PIPESTATUS[0]}
echo ""
if [[ "$EVAL_EXIT" -ne 0 ]]; then
    warn "Remote eval exited with code $EVAL_EXIT. Check /tmp/remote-eval.log for details."
fi
info "Eval finished. [$(elapsed)]"

# -- Step 3: Fetch results ----------------------------------------------------
echo ""
info "=== Fetching results from $MAC_HOST ==="
mkdir -p "$REPO_DIR/$LOCAL_RESULTS_DIR"
rsync -avz "$MAC_HOST:$REMOTE_RESULTS_DIR/" "$REPO_DIR/$LOCAL_RESULTS_DIR/"
ok "Results fetched to $LOCAL_RESULTS_DIR/. [$(elapsed)]"

# -- Step 4: Print results ----------------------------------------------------
echo ""
info "=== Results ==="
SCORING_FILE="$REPO_DIR/$LOCAL_RESULTS_DIR/scoring.yaml"
if [[ -f "$SCORING_FILE" ]]; then
    echo ""
    cat "$SCORING_FILE"
    echo ""
    ok "Scoring results displayed above."
else
    warn "No scoring.yaml found in $LOCAL_RESULTS_DIR/."
    echo "  Check /tmp/remote-eval.log for errors."
fi

# -- Summary ------------------------------------------------------------------
echo ""
echo "============================================"
echo "  Remote Eval Summary"
echo "============================================"
echo "  Host:     $MAC_HOST"
echo "  Policy:   $POLICY"
echo "  Results:  $LOCAL_RESULTS_DIR/scoring.yaml"
echo "  Duration: $(elapsed)"
echo "  Log:      /tmp/remote-eval.log"
echo "============================================"
