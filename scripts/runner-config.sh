#!/usr/bin/env bash
# ============================================================================
# runner-config.sh -- Configuration for the Mac Studio remote eval runner
# ============================================================================
#
# Source this file from other scripts to pick up defaults:
#   source "$(dirname "$0")/runner-config.sh"
#
# Override any value via environment variables before sourcing:
#   MAC_HOST=mac-studio source scripts/runner-config.sh
#
# SSH setup:
#   Add an entry to ~/.ssh/config:
#     Host mac
#         HostName <ip-or-hostname>
#         User <username>
#         IdentityFile ~/.ssh/id_ed25519
#         ForwardAgent yes
# ============================================================================

# SSH hostname (must match ~/.ssh/config or be a reachable address)
MAC_HOST="${MAC_HOST:-mac}"

# Directory on the Mac where the repo lives
REMOTE_DIR="${REMOTE_DIR:-~/ws_aic/src/aic}"

# Default policy class for evaluation
DEFAULT_POLICY="${DEFAULT_POLICY:-aic_example_policies.ros.BlindPush}"

# Whether to use remote eval (true) or local docker eval (false)
USE_REMOTE_EVAL="${USE_REMOTE_EVAL:-true}"

# Results directory (local side)
LOCAL_RESULTS_DIR="${LOCAL_RESULTS_DIR:-aic_results}"

# Results directory (remote side)
REMOTE_RESULTS_DIR="${REMOTE_RESULTS_DIR:-~/aic_results}"

# Rsync exclude patterns (space-separated, converted to --exclude args)
RSYNC_EXCLUDES="${RSYNC_EXCLUDES:-.pixi __pycache__ .git aic_results .beads .claude/worktrees}"

# SSH connection timeout in seconds
SSH_TIMEOUT="${SSH_TIMEOUT:-10}"
