#!/usr/bin/env bash
# Collect training data using Docker compose with randomized configs.
# Each run = 1 config × 3 trials = 3 episodes.
# Usage: scripts/collect_with_docker.sh [start_config] [end_config]
set -euo pipefail

WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_DIR="$WORKSPACE/aic_engine/config/random"
COLLECT_CONFIG_DIR="$WORKSPACE/docker/collect_config"
TRAINING_DATA_DIR="$HOME/training_data_new"

START=${1:-0}
END=${2:-19}

mkdir -p "$COLLECT_CONFIG_DIR" "$TRAINING_DATA_DIR"

for i in $(seq $START $END); do
    CONFIG=$(printf "config_%03d.yaml" $i)
    echo ""
    echo "=== Collecting config $i ($CONFIG) ==="
    echo "  Time: $(date)"

    # Copy config as sample_config.yaml (what the engine expects)
    cp "$CONFIG_DIR/$CONFIG" "$COLLECT_CONFIG_DIR/sample_config.yaml"

    # Run data collection
    cd "$WORKSPACE"
    docker compose -f docker/docker-compose.collect.yaml up --abort-on-container-exit 2>&1 | tail -5

    # Check for new episodes
    NEW_EPISODES=$(ls "$TRAINING_DATA_DIR" 2>/dev/null | wc -l)
    echo "  Episodes collected so far: $NEW_EPISODES"

    # Small delay between runs
    sleep 5
done

echo ""
echo "=== Collection Complete ==="
echo "  Total episodes: $(ls "$TRAINING_DATA_DIR" 2>/dev/null | wc -l)"
echo "  Data dir: $TRAINING_DATA_DIR"
