#!/usr/bin/env python3
"""Fine-tune ACTPolicy v2 - compute velocity actions from position differences.

The key change from v1: instead of using the observed TCP velocity directly,
we compute the action as position_delta / dt. This gives a stronger signal
near the insertion point where the robot is still moving but slowly.

Usage:
    pixi run python scripts/train_act_v2.py \
        --data-dir ~/aic/data/velocity \
        --output-dir ~/aic/models/act_velocity_v3 \
        --epochs 50 --batch-size 8 --lr 1e-5

NOTE: v2 (position-difference actions) was empirically worse than v1
(observed velocity). Kept for reference only — see exp-057 for the
3/300 score that proved this approach generates 10x too-large velocities.
"""

import argparse
import json
import time
from pathlib import Path

import draccus
import numpy as np
import torch
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset


class VelocityDemoDatasetV2(Dataset):
    """Dataset that computes velocity from position differences."""

    def __init__(
        self,
        data_dir: str,
        chunk_size: int = 100,
        min_episode_length: int = 20,
        dt: float = 0.25,  # 4Hz recording rate
    ):
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.dt = dt
        self.episodes = []
        self.cumulative_lengths = []

        total = 0
        for ep_dir in sorted(self.data_dir.iterdir()):
            if not ep_dir.is_dir() or not ep_dir.name.startswith("episode_"):
                continue
            states = np.load(ep_dir / "states.npy")
            if len(states) < min_episode_length:
                print(
                    f"  Skipping {ep_dir.name}: too short ({len(states)} steps)"
                )
                continue

            # Compute velocity actions from position differences
            # State: [0:3]=tcp_pos, [3:7]=tcp_quat, [7:10]=tcp_vel_lin, [10:13]=tcp_vel_ang
            tcp_pos = states[:, :3]  # (T, 3)

            # Linear velocity from position differences
            pos_diff = np.diff(tcp_pos, axis=0) / dt  # (T-1, 3)
            # Pad last step with zeros (or repeat last)
            lin_vel = np.vstack([pos_diff, pos_diff[-1:]])  # (T, 3)

            # For angular velocity, use the observed values from state
            # (computing from quaternion differences is complex and noisy)
            ang_vel = states[:, 10:13]  # (T, 3) - from observation

            # Gripper constant (matching pretrained model)
            gripper = np.full((len(states), 1), 0.012, dtype=np.float32)

            # Combine: 7D action = [lin_vel(3), ang_vel(3), gripper(1)]
            actions = np.concatenate(
                [lin_vel, ang_vel, gripper], axis=1
            ).astype(np.float32)

            self.episodes.append(
                {
                    "dir": ep_dir,
                    "states": states,
                    "actions": actions,
                    "length": len(states),
                }
            )
            total += len(states)
            self.cumulative_lengths.append(total)

        print(f"Loaded {len(self.episodes)} episodes, {total} total steps")

        # Compute normalization stats
        all_states = np.concatenate([ep["states"] for ep in self.episodes])
        all_actions = np.concatenate([ep["actions"] for ep in self.episodes])

        self.state_mean = torch.tensor(
            all_states.mean(axis=0), dtype=torch.float32
        )
        self.state_std = torch.tensor(
            all_states.std(axis=0), dtype=torch.float32
        )
        self.state_std[self.state_std < 1e-6] = 1.0

        self.action_mean = torch.tensor(
            all_actions.mean(axis=0), dtype=torch.float32
        )
        self.action_std = torch.tensor(
            all_actions.std(axis=0), dtype=torch.float32
        )
        self.action_std[self.action_std < 1e-6] = 1.0

        print(f"Action mean: {self.action_mean}")
        print(f"Action std: {self.action_std}")

        self._compute_image_stats()

    def _compute_image_stats(self):
        samples = []
        for ep in self.episodes[:5]:
            for i in range(0, ep["length"], 10):
                for cam in ["left", "center", "right"]:
                    img_path = ep["dir"] / f"images_{cam}_{i:04d}.npy"
                    if img_path.exists():
                        img = np.load(img_path).astype(np.float32) / 255.0
                        samples.append(img)
        samples = np.array(samples)
        self.img_mean = torch.tensor(
            samples.mean(axis=(0, 1, 2)), dtype=torch.float32
        ).view(1, 3, 1, 1)
        self.img_std = torch.tensor(
            samples.std(axis=(0, 1, 2)), dtype=torch.float32
        ).view(1, 3, 1, 1)
        self.img_std[self.img_std < 1e-6] = 1.0

    def __len__(self):
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0

    def _find_episode_and_step(self, idx):
        for ep_idx, cum_len in enumerate(self.cumulative_lengths):
            if idx < cum_len:
                prev = self.cumulative_lengths[ep_idx - 1] if ep_idx > 0 else 0
                return ep_idx, idx - prev
        raise IndexError(f"Index {idx} out of range")

    def __getitem__(self, idx):
        ep_idx, step_idx = self._find_episode_and_step(idx)
        ep = self.episodes[ep_idx]

        state = torch.tensor(ep["states"][step_idx], dtype=torch.float32)
        state_norm = (state - self.state_mean) / self.state_std

        actions = ep["actions"][step_idx : step_idx + self.chunk_size]
        n_real = len(actions)
        if n_real < self.chunk_size:
            pad = np.tile(actions[-1:], (self.chunk_size - n_real, 1))
            actions = np.concatenate([actions, pad])
        actions = torch.tensor(actions, dtype=torch.float32)
        actions_norm = (actions - self.action_mean) / self.action_std

        action_is_pad = torch.zeros(self.chunk_size, dtype=torch.bool)
        action_is_pad[n_real:] = True

        images = {}
        for cam_key, cam_name in [
            ("left_camera", "left"),
            ("center_camera", "center"),
            ("right_camera", "right"),
        ]:
            img_path = ep["dir"] / f"images_{cam_name}_{step_idx:04d}.npy"
            if img_path.exists():
                img = np.load(img_path)
            else:
                img = np.zeros((256, 288, 3), dtype=np.uint8)
            img_tensor = (
                torch.from_numpy(img)
                .permute(2, 0, 1)
                .float()
                .div(255.0)
                .unsqueeze(0)
            )
            img_norm = (img_tensor - self.img_mean) / self.img_std
            images[f"observation.images.{cam_key}"] = img_norm.squeeze(0)

        return {
            "observation.state": state_norm,
            "action": actions_norm,
            "action_is_pad": action_is_pad,
            **images,
        }


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from huggingface_hub import snapshot_download

    repo_id = "grkw/aic_act_policy"
    policy_path = Path(
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=[
                "config.json",
                "model.safetensors",
                "*.safetensors",
            ],
        )
    )

    with open(policy_path / "config.json") as f:
        config_dict = json.load(f)
        if "type" in config_dict:
            del config_dict["type"]

    config = draccus.decode(ACTConfig, config_dict)
    policy = ACTPolicy(config)
    policy.load_state_dict(load_file(policy_path / "model.safetensors"))
    policy.to(device)
    print(f"Loaded pretrained model from {policy_path}")

    dataset = VelocityDemoDatasetV2(
        args.data_dir,
        chunk_size=config.chunk_size,
        min_episode_length=20,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    policy.train()

    steps_per_epoch = len(dataloader)
    print(
        f"Training: {args.epochs} epochs, {steps_per_epoch} steps/epoch, "
        f"batch_size={args.batch_size}"
    )

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            loss, _loss_dict = policy.forward(batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 50 == 0:
                print(
                    f"  Epoch {epoch + 1}/{args.epochs}, "
                    f"Step {batch_idx}/{steps_per_epoch}, "
                    f"Loss: {loss.item():.6f}"
                )

        avg_loss = epoch_loss / max(steps_per_epoch, 1)
        elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch + 1}/{args.epochs}: avg_loss={avg_loss:.6f}, "
            f"time={elapsed:.1f}s"
        )

        if (epoch + 1) % args.save_every == 0 or avg_loss < best_loss:
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_path = output_dir / "best"
            else:
                save_path = output_dir / f"checkpoint_{epoch + 1}"

            save_path.mkdir(parents=True, exist_ok=True)
            save_file(policy.state_dict(), save_path / "model.safetensors")

            with open(save_path / "config.json", "w") as f:
                json.dump(config_dict, f, indent=2)

            img_mean = dataset.img_mean.squeeze()
            img_std = dataset.img_std.squeeze()
            norm_stats = {
                "observation.state.mean": dataset.state_mean,
                "observation.state.std": dataset.state_std,
                "action.mean": dataset.action_mean,
                "action.std": dataset.action_std,
                "observation.images.left_camera.mean": img_mean.clone(),
                "observation.images.left_camera.std": img_std.clone(),
                "observation.images.center_camera.mean": img_mean.clone(),
                "observation.images.center_camera.std": img_std.clone(),
                "observation.images.right_camera.mean": img_mean.clone(),
                "observation.images.right_camera.std": img_std.clone(),
            }
            save_file(
                norm_stats,
                save_path
                / "policy_preprocessor_step_3_normalizer_processor.safetensors",
            )

            print(f"  Saved checkpoint to {save_path}")

    print(f"Training complete! Best loss: {best_loss:.6f}")
    print(f"Best model saved to: {output_dir / 'best'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune ACTPolicy v2 (pos-diff actions)"
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--save-every", type=int, default=10)
    args = parser.parse_args()
    train(args)
