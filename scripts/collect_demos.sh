#!/bin/bash
# collect_demos.sh -- Collect CheatCode demonstrations for ACT training.
#
# Usage: scripts/collect_demos.sh [num_episodes]
#
# Runs the DataCollector policy which wraps CheatCode to record expert
# demonstrations. The engine runs 3 trials per run (one per cable task),
# so each run produces 3 episodes.
#
# To collect 100 episodes, run: scripts/collect_demos.sh 100
# This will execute ceil(100/3) = 34 engine runs.
#
# Prerequisites:
#   - ground_truth:=true must be set (CheatCode requires ground-truth TFs)
#   - Simulation environment must be running
#   - Data is saved to ~/training_data/episode_NNNN/

set -euo pipefail

NUM_EPISODES="${1:-100}"
EPISODES_PER_RUN=3
NUM_RUNS=$(( (NUM_EPISODES + EPISODES_PER_RUN - 1) / EPISODES_PER_RUN ))

POLICY="aic_example_policies.ros.DataCollector"
DATA_DIR="${HOME}/training_data"

echo "=== ACT Demo Collection ==="
echo "Target episodes: ${NUM_EPISODES}"
echo "Runs needed:     ${NUM_RUNS} (${EPISODES_PER_RUN} episodes/run)"
echo "Data directory:  ${DATA_DIR}"
echo ""

# Ensure data directory exists
mkdir -p "${DATA_DIR}"

# Count existing episodes
existing_episodes() {
    find "${DATA_DIR}" -maxdepth 1 -name 'episode_*' -type d 2>/dev/null | wc -l
}

START_EPISODES=$(existing_episodes)
echo "Existing episodes: ${START_EPISODES}"
echo ""

for run in $(seq 1 "${NUM_RUNS}"); do
    current=$(existing_episodes)
    collected=$((current - START_EPISODES))

    if [ "${collected}" -ge "${NUM_EPISODES}" ]; then
        echo "Reached target of ${NUM_EPISODES} episodes (collected ${collected}). Done."
        break
    fi

    echo "--- Run ${run}/${NUM_RUNS} (${collected}/${NUM_EPISODES} episodes collected) ---"

    # Run the evaluation with DataCollector policy and ground_truth enabled.
    # The docker compose setup handles launching the engine + sim.
    # For local eval:
    if command -v docker &>/dev/null && [ -f docker/docker-compose.yaml ]; then
        POLICY="${POLICY}" docker compose -f docker/docker-compose.yaml up \
            --abort-on-container-exit --exit-code-from model 2>&1 || {
            echo "WARNING: Run ${run} failed, continuing..."
            continue
        }
    else
        # Direct ROS 2 launch (e.g., on GPU instance with pixi)
        pixi run ros2 run aic_model aic_model \
            --ros-args \
            -p use_sim_time:=true \
            -p "policy:=${POLICY}" \
            -p ground_truth:=true 2>&1 || {
            echo "WARNING: Run ${run} failed, continuing..."
            continue
        }
    fi

    echo "Run ${run} complete."
    echo ""
done

FINAL_EPISODES=$(existing_episodes)
TOTAL_COLLECTED=$((FINAL_EPISODES - START_EPISODES))

echo ""
echo "=== Collection Complete ==="
echo "New episodes:   ${TOTAL_COLLECTED}"
echo "Total episodes: ${FINAL_EPISODES}"
echo "Data location:  ${DATA_DIR}"
