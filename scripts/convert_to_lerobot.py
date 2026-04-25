#!/usr/bin/env python3
"""Convert collected numpy episodes to LeRobot v2 dataset format.

Usage:
    pixi run python scripts/convert_to_lerobot.py \
        --input ~/aic/data/velocity \
        --output ~/aic/data/velocity_lerobot \
        --repo-id local/aic_velocity_demos

The output dataset can be used directly with lerobot-train:
    pixi run lerobot-train \
        --dataset.repo_id=local/aic_velocity_demos \
        --dataset.root=~/aic/data/velocity_lerobot \
        ...
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def convert_episodes(input_dir: str, output_dir: str, repo_id: str):
    """Convert numpy episodes to LeRobot v2 parquet dataset."""
    input_path = Path(input_dir)
    output_path = Path(output_dir) / repo_id

    # Find all episodes
    episodes = sorted(
        [
            d
            for d in input_path.iterdir()
            if d.is_dir() and d.name.startswith("episode_")
        ]
    )
    print(f"Found {len(episodes)} episodes in {input_dir}")

    if not episodes:
        print("No episodes found!")
        return

    # Prepare output directory
    if output_path.exists():
        shutil.rmtree(output_path)
    data_dir = output_path / "data"
    data_dir.mkdir(parents=True)
    videos_dir = output_path / "videos"
    meta_dir = output_path / "meta"
    meta_dir.mkdir(parents=True)

    # Collect all data
    all_states = []
    all_actions = []
    all_episodes_info = []
    episode_indices = []
    frame_indices = []
    total_frames = 0

    for ep_idx, ep_dir in enumerate(episodes):
        states = np.load(ep_dir / "states.npy")
        actions = np.load(ep_dir / "actions.npy")
        n_steps = len(states)

        print(
            f"  Episode {ep_idx}: {n_steps} steps, "
            f"state={states.shape}, action={actions.shape}"
        )

        # Save images as PNG files
        for cam_name in ["left_camera", "center_camera", "right_camera"]:
            cam_dir = (
                videos_dir
                / f"observation.images.{cam_name}"
                / f"episode_{ep_idx:06d}"
            )
            cam_dir.mkdir(parents=True, exist_ok=True)

            short_name = cam_name.split("_")[0]  # left, center, right
            for step in range(n_steps):
                img_file = ep_dir / f"images_{short_name}_{step:04d}.npy"
                if img_file.exists():
                    img = np.load(img_file)
                    Image.fromarray(img).save(cam_dir / f"frame_{step:06d}.png")

        all_states.append(states)
        all_actions.append(actions)
        episode_indices.extend([ep_idx] * n_steps)
        frame_indices.extend(range(n_steps))

        all_episodes_info.append(
            {
                "episode_index": ep_idx,
                "tasks": ["Insert cable connector into port"],
                "length": n_steps,
            }
        )
        total_frames += n_steps

    # Concatenate all data
    all_states = np.concatenate(all_states, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    episode_indices = np.array(episode_indices)
    frame_indices = np.array(frame_indices)

    # Compute normalization statistics
    state_mean = all_states.mean(axis=0)
    state_std = all_states.std(axis=0)
    state_std[state_std < 1e-6] = 1.0  # Prevent division by zero

    action_mean = all_actions.mean(axis=0)
    action_std = all_actions.std(axis=0)
    action_std[action_std < 1e-6] = 1.0

    print("\nDataset statistics:")
    print(f"  Total frames: {total_frames}")
    print(f"  Episodes: {len(episodes)}")
    print(f"  State shape: {all_states.shape}")
    print(f"  Action shape: {all_actions.shape}")
    print(f"  Action mean: {action_mean}")
    print(f"  Action std: {action_std}")

    # Save as parquet (LeRobot v2 format)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        # Build the table
        table_data = {
            "episode_index": pa.array(episode_indices, type=pa.int64()),
            "frame_index": pa.array(frame_indices, type=pa.int64()),
            "timestamp": pa.array(
                np.arange(total_frames, dtype=np.float64) * 0.25
            ),
        }

        # Add state columns
        for i in range(all_states.shape[1]):
            table_data[f"observation.state.{i}"] = pa.array(
                all_states[:, i].astype(np.float32)
            )

        # Add action columns
        for i in range(all_actions.shape[1]):
            table_data[f"action.{i}"] = pa.array(
                all_actions[:, i].astype(np.float32)
            )

        # Add image path columns
        for cam_name in ["left_camera", "center_camera", "right_camera"]:
            paths = []
            for frame_idx in range(total_frames):
                ep = episode_indices[frame_idx]
                step = frame_indices[frame_idx]
                paths.append(
                    f"videos/observation.images.{cam_name}/episode_{ep:06d}/frame_{step:06d}.png"
                )
            table_data[f"observation.images.{cam_name}"] = pa.array(paths)

        table = pa.table(table_data)

        # Save as single parquet file
        pq.write_table(table, data_dir / "train-00000-of-00001.parquet")

    except ImportError:
        print("pyarrow not available, saving as numpy instead")
        np.save(data_dir / "states.npy", all_states)
        np.save(data_dir / "actions.npy", all_actions)
        np.save(data_dir / "episode_indices.npy", episode_indices)
        np.save(data_dir / "frame_indices.npy", frame_indices)

    # Save metadata
    info = {
        "codebase_version": "v2.1",
        "robot_type": "aic_controller",
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "fps": 4,
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [26],
                "names": [
                    "tcp_pose.position.x",
                    "tcp_pose.position.y",
                    "tcp_pose.position.z",
                    "tcp_pose.orientation.x",
                    "tcp_pose.orientation.y",
                    "tcp_pose.orientation.z",
                    "tcp_pose.orientation.w",
                    "tcp_velocity.linear.x",
                    "tcp_velocity.linear.y",
                    "tcp_velocity.linear.z",
                    "tcp_velocity.angular.x",
                    "tcp_velocity.angular.y",
                    "tcp_velocity.angular.z",
                    "tcp_error.0",
                    "tcp_error.1",
                    "tcp_error.2",
                    "tcp_error.3",
                    "tcp_error.4",
                    "tcp_error.5",
                    "joint_positions.0",
                    "joint_positions.1",
                    "joint_positions.2",
                    "joint_positions.3",
                    "joint_positions.4",
                    "joint_positions.5",
                    "joint_positions.6",
                ],
            },
            "action": {
                "dtype": "float32",
                "shape": [7],
                "names": [
                    "linear.x",
                    "linear.y",
                    "linear.z",
                    "angular.x",
                    "angular.y",
                    "angular.z",
                    "gripper",
                ],
            },
            "observation.images.left_camera": {
                "dtype": "image",
                "shape": [256, 288, 3],
            },
            "observation.images.center_camera": {
                "dtype": "image",
                "shape": [256, 288, 3],
            },
            "observation.images.right_camera": {
                "dtype": "image",
                "shape": [256, 288, 3],
            },
        },
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # Save stats for normalization
    stats = {
        "observation.state": {
            "mean": state_mean.tolist(),
            "std": state_std.tolist(),
        },
        "action": {
            "mean": action_mean.tolist(),
            "std": action_std.tolist(),
        },
    }
    with open(meta_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Save episodes info
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep_info in all_episodes_info:
            f.write(json.dumps(ep_info) + "\n")

    print(f"\nDataset saved to {output_path}")
    print(
        f"Use with: pixi run lerobot-train --dataset.repo_id={repo_id} --dataset.root={output_dir}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert numpy episodes to LeRobot format"
    )
    parser.add_argument(
        "--input", required=True, help="Input directory with numpy episodes"
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for LeRobot dataset"
    )
    parser.add_argument(
        "--repo-id", default="local/aic_velocity_demos", help="Dataset repo ID"
    )
    args = parser.parse_args()
    convert_episodes(args.input, args.output, args.repo_id)
