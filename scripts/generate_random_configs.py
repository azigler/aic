#!/usr/bin/env python3
"""Generate randomized trial configs for diverse data collection.

Creates N config files with randomized board poses and component positions,
matching the randomization ranges from the competition evaluation.
"""

import copy
import os
import random
import sys

import yaml


def load_base_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def randomize_config(base, seed):
    """Randomize board pose and component positions within competition limits."""
    rng = random.Random(seed)
    config = copy.deepcopy(base)

    # Task board limits (from sample_config)
    nic_trans_range = (-0.0215, 0.0234)
    sc_trans_range = (-0.06, 0.055)
    nic_yaw_range = (-0.1745, 0.1745)  # ±10 degrees in radians

    for trial_key in config.get("trials", {}):
        trial = config["trials"][trial_key]
        scene = trial.get("scene", {})
        tb = scene.get("task_board", {})
        pose = tb.get("pose", {})

        # Randomize board pose (small variations)
        pose["x"] = 0.15 + rng.uniform(-0.03, 0.03)
        pose["y"] = -0.2 + rng.uniform(-0.03, 0.03)
        pose["yaw"] = 3.1415 + rng.uniform(-0.2, 0.2)

        # Randomize NIC card positions
        for rail_idx in range(5):
            rail_key = f"nic_rail_{rail_idx}"
            if rail_key in tb and tb[rail_key].get("entity_present"):
                tb[rail_key]["entity_pose"]["translation"] = rng.uniform(
                    *nic_trans_range
                )
                tb[rail_key]["entity_pose"]["yaw"] = rng.uniform(*nic_yaw_range)

        # Randomize SC port positions
        for rail_idx in range(2):
            rail_key = f"sc_rail_{rail_idx}"
            if rail_key in tb and tb[rail_key].get("entity_present"):
                tb[rail_key]["entity_pose"]["translation"] = rng.uniform(
                    *sc_trans_range
                )

    return config


def main():
    n_configs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    base_config_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "aic_engine",
        "config",
        "sample_config.yaml",
    )
    output_dir = os.path.join(
        os.path.dirname(__file__), "..", "aic_engine", "config", "random"
    )
    os.makedirs(output_dir, exist_ok=True)

    base = load_base_config(base_config_path)

    for i in range(n_configs):
        config = randomize_config(base, seed=42 + i)
        output_path = os.path.join(output_dir, f"config_{i:03d}.yaml")
        with open(output_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Generated {output_path}")

    print(f"\n{n_configs} configs generated in {output_dir}")


if __name__ == "__main__":
    main()
