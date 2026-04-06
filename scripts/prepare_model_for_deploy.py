#!/usr/bin/env python3
"""Prepare a trained model directory for OurACT deployment.

Converts normalization_stats.pt -> normalization_stats.json
Creates best_model.pt symlink -> checkpoint_best.pt
"""

import json
import sys
from pathlib import Path

import torch


def main():
    if len(sys.argv) < 2:
        print("Usage: prepare_model_for_deploy.py <model_dir>")
        sys.exit(1)

    model_dir = Path(sys.argv[1])

    # Convert normalization stats from .pt to .json
    stats_pt = model_dir / "normalization_stats.pt"
    stats_json = model_dir / "normalization_stats.json"

    if stats_pt.exists():
        stats = torch.load(stats_pt, map_location="cpu", weights_only=True)
        json_stats = {}
        for k, v in stats.items():
            json_stats[k] = v.tolist()
        with open(stats_json, "w") as f:
            json.dump(json_stats, f, indent=2)
        print(f"Converted {stats_pt} -> {stats_json}")
    else:
        print(f"WARNING: {stats_pt} not found")

    # Create best_model.pt symlink
    best_model = model_dir / "best_model.pt"
    checkpoint = model_dir / "checkpoint_best.pt"

    if checkpoint.exists():
        if best_model.exists() or best_model.is_symlink():
            best_model.unlink()
        best_model.symlink_to(checkpoint.name)
        print(f"Created symlink {best_model} -> {checkpoint.name}")
    else:
        print(f"WARNING: {checkpoint} not found")

    # Verify config.json exists
    config = model_dir / "config.json"
    if config.exists():
        with open(config) as f:
            cfg = json.load(f)
        print(f"Config: {cfg}")
    else:
        print(f"WARNING: {config} not found")

    print("Done!")


if __name__ == "__main__":
    main()
