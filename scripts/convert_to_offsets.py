#!/usr/bin/env python3
"""Convert position-mode training data to offset-mode.

Reads episodes from --input-dir, computes:
  offset_xyz = action_xyz - state_xyz  (3D position delta)
  quaternion = kept absolute (barely changes across demos)

Saves to --output-dir with identical structure.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def convert_episode(ep_dir: Path, out_dir: Path):
    """Convert one episode from absolute to offset actions."""
    states = np.load(ep_dir / "states.npy")
    actions = np.load(ep_dir / "actions.npy")

    n = min(len(states), len(actions))
    states = states[:n]
    actions = actions[:n]

    # Compute position offsets: target_xyz - current_tcp_xyz
    offset_actions = actions.copy()
    offset_actions[:, :3] = actions[:, :3] - states[:, :3]  # XYZ offset
    # Keep quaternion (actions[:, 3:7]) absolute — it barely varies

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "states.npy", states)
    np.save(out_dir / "actions.npy", offset_actions)

    # Copy images (symlink to save disk space)
    for img_file in ep_dir.glob("images_*.npy"):
        dst = out_dir / img_file.name
        if not dst.exists():
            dst.symlink_to(img_file.resolve())

    # Copy and update metadata
    meta_path = ep_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        meta["action_format"] = "offset: tcp_delta_xyz(3) + tcp_quat_xyzw(4)"
        meta["offset_mode"] = True
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path.home() / "training_data_pos",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "training_data_offset",
    )
    args = parser.parse_args()

    episode_dirs = sorted(
        d
        for d in args.input_dir.iterdir()
        if d.is_dir() and d.name.startswith("episode_")
    )

    print(f"Converting {len(episode_dirs)} episodes from {args.input_dir}")
    print(f"Output: {args.output_dir}")

    # Copy episode counter
    counter = args.input_dir / ".episode_counter"
    if counter.exists():
        args.output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(counter, args.output_dir / ".episode_counter")

    for ep_dir in episode_dirs:
        out_ep = args.output_dir / ep_dir.name
        convert_episode(ep_dir, out_ep)
        print(f"  {ep_dir.name} -> offset actions")

    # Print stats on the converted data
    print("\nOffset action statistics:")
    all_offsets = []
    for ep_dir in episode_dirs:
        out_ep = args.output_dir / ep_dir.name
        a = np.load(out_ep / "actions.npy")
        all_offsets.append(a[:, :3])
    offsets = np.concatenate(all_offsets)
    print(f"  XYZ offset mean: {offsets.mean(axis=0)}")
    print(f"  XYZ offset std:  {offsets.std(axis=0)}")
    print(f"  XYZ offset range: [{offsets.min(axis=0)}, {offsets.max(axis=0)}]")


if __name__ == "__main__":
    main()
