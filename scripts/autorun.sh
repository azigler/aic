#!/usr/bin/env bash
# ============================================================================
# autorun.sh -- Autoresearch-style autonomous experiment runner
# ============================================================================
#
# Usage: scripts/autorun.sh <policy_class>
#
# Runs a single experiment: sync, eval, parse score, log to experiments.tsv.
# Returns the score. The calling agent decides keep/discard.
#
# This script is the atomic unit of the experiment loop.
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
source "$SCRIPT_DIR/runner-config.sh"

POLICY="${1:-aic_example_policies.ros.VisionPolicy}"
REMOTE_HOST="${GPU_HOST:-gpu}"
REMOTE_DIR="${REMOTE_DIR:-$HOME/ws_aic/src/aic}"
TSV_FILE="$REPO_DIR/experiments.tsv"

echo "=== Autorun: $POLICY ==="
echo "  Host: $REMOTE_HOST"
echo "  Time: $(date)"

START=$(date +%s)

# Step 1: Sync code
rsync -avz --quiet \
    --exclude='.pixi' --exclude='__pycache__' --exclude='.git' \
    --exclude='aic_results' --exclude='.beads' --exclude='.claude/worktrees' \
    "$REPO_DIR/" "$REMOTE_HOST:$REMOTE_DIR/"

# Step 2: Reinstall + run eval
ssh -o ConnectTimeout=10 "$REMOTE_HOST" \
    "export PATH=\$HOME/.pixi/bin:\$PATH && \
     cd $REMOTE_DIR && \
     pixi reinstall ros-kilted-aic-example-policies 2>&1 | tail -1 && \
     rm -f ~/aic_results/scoring.yaml && \
     rm -rf ~/aic_results/bag_* && \
     POLICY='$POLICY' ~/run-eval.sh 2>&1" | tee /tmp/autorun.log

END=$(date +%s)
ELAPSED=$((END - START))

# Step 3: Fetch results
mkdir -p "$REPO_DIR/aic_results"
rsync -avz --quiet "$REMOTE_HOST:~/aic_results/scoring.yaml" "$REPO_DIR/aic_results/" 2>/dev/null || true

# Step 4: Parse score
SCORE="0"
if [[ -f "$REPO_DIR/aic_results/scoring.yaml" ]]; then
    SCORE=$(grep "^total:" "$REPO_DIR/aic_results/scoring.yaml" | head -1 | awk '{print $2}' | cut -c1-6)
fi

# Step 5: Get current git commit
COMMIT=$(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo "-")

echo ""
echo "=== Result ==="
echo "  Score: $SCORE"
echo "  Time:  ${ELAPSED}s"
echo "  Commit: $COMMIT"

# Output score for the calling script to capture
echo "$SCORE"
