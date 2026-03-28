#!/usr/bin/env bash
# ============================================================================
# remote-eval.sh -- Run evaluation on a remote GPU host from this machine
# ============================================================================
#
# Usage:
#   ./scripts/remote-eval.sh                              # defaults from runner-config.sh
#   ./scripts/remote-eval.sh <policy_class>               # custom policy
#   ./scripts/remote-eval.sh <policy_class> <host>        # custom policy + host
#
# Examples:
#   ./scripts/remote-eval.sh aic_example_policies.ros.BlindPush
#   ./scripts/remote-eval.sh aic_example_policies.ros.CheatCode gpu
#   GPU_HOST=my-gpu ./scripts/remote-eval.sh
#
# What this script does:
#   1. Checks SSH connectivity to the remote host
#   2. Rsyncs the current codebase (excluding build artifacts)
#   3. Runs ~/run-eval.sh on the host via SSH
#   4. Fetches scoring.yaml back to local aic_results/
#   5. Prints results and total elapsed time
#
# Prerequisites:
#   - SSH key-based auth to the host (see runner-config.sh for SSH setup)
#   - Host has been set up with gpu-setup.sh (or mac-setup.sh)
#   - runner-config.sh has correct GPU_HOST
# ============================================================================

set -euo pipefail

# -- Load configuration ------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck source=runner-config.sh
source "$SCRIPT_DIR/runner-config.sh"

# -- Parse arguments ----------------------------------------------------------
POLICY="${1:-$DEFAULT_POLICY}"
# Accept host override as second argument; prefer GPU_HOST (falls back to MAC_HOST)
REMOTE_HOST="${2:-$GPU_HOST}"

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
info "Checking SSH connectivity to $REMOTE_HOST..."
if ! ssh -o ConnectTimeout="$SSH_TIMEOUT" -o BatchMode=yes "$REMOTE_HOST" "echo ok" &>/dev/null; then
    err "Cannot connect to $REMOTE_HOST via SSH."
    echo ""
    echo "Troubleshooting:"
    echo "  1. Ensure the host is powered on and reachable"
    echo "  2. Check your ~/.ssh/config has a 'Host $REMOTE_HOST' entry"
    echo "  3. Test manually: ssh $REMOTE_HOST echo hello"
    echo "  4. Ensure SSH key auth is set up (no password prompts)"
    exit 1
fi
ok "SSH connection to $REMOTE_HOST successful. [$(elapsed)]"

# -- Build rsync exclude args -------------------------------------------------
RSYNC_ARGS=(-avz --delete)
for pattern in $RSYNC_EXCLUDES; do
    RSYNC_ARGS+=(--exclude="$pattern")
done

# -- Step 1: Sync code --------------------------------------------------------
echo ""
info "=== Syncing code to $REMOTE_HOST:$REMOTE_DIR ==="
info "  Excludes: $RSYNC_EXCLUDES"
SYNC_START=$(date +%s)
rsync "${RSYNC_ARGS[@]}" "$REPO_DIR/" "$REMOTE_HOST:$REMOTE_DIR/"
SYNC_END=$(date +%s)
SYNC_SECS=$((SYNC_END - SYNC_START))
ok "Code synced in ${SYNC_SECS}s. [$(elapsed)]"

# -- Step 2: Run eval ---------------------------------------------------------
echo ""
info "=== Running eval on $REMOTE_HOST ==="
info "  Policy: $POLICY"
info "  Remote dir: $REMOTE_DIR"
echo ""

EVAL_START=$(date +%s)
ssh -o ConnectTimeout="$SSH_TIMEOUT" "$REMOTE_HOST" \
    "cd $REMOTE_DIR && POLICY='$POLICY' bash ~/run-eval.sh" 2>&1 | \
    tee /tmp/remote-eval.log

EVAL_EXIT=${PIPESTATUS[0]}
EVAL_END=$(date +%s)
EVAL_SECS=$((EVAL_END - EVAL_START))
echo ""
if [[ "$EVAL_EXIT" -ne 0 ]]; then
    warn "Remote eval exited with code $EVAL_EXIT. Check /tmp/remote-eval.log for details."
fi
info "Eval finished in ${EVAL_SECS}s. [$(elapsed)]"

# -- Step 3: Fetch results ----------------------------------------------------
echo ""
info "=== Fetching results from $REMOTE_HOST ==="
mkdir -p "$REPO_DIR/$LOCAL_RESULTS_DIR"
rsync -avz "$REMOTE_HOST:$REMOTE_RESULTS_DIR/" "$REPO_DIR/$LOCAL_RESULTS_DIR/"
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
echo "  Host:     $REMOTE_HOST"
echo "  Policy:   $POLICY"
echo "  Results:  $LOCAL_RESULTS_DIR/scoring.yaml"
echo "  Sync:     ${SYNC_SECS}s"
echo "  Eval:     ${EVAL_SECS}s"
echo "  Total:    $(elapsed)"
echo "  Log:      /tmp/remote-eval.log"
echo "============================================"
