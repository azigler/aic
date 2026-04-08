#!/usr/bin/env python3
"""Fine-tune pretrained ACTPolicy on collected velocity-mode demos.

Usage:
    pixi run python scripts/train_act.py \
        --data-dir ~/training_data_velocity \
        --output-dir ~/models/act_finetuned \
        --epochs 50 --batch-size 8 --lr 1e-5

This loads the pretrained grkw/aic_act_policy weights and fine-tunes
on our collected demonstrations.
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


class VelocityDemoDataset(Dataset):
    """Dataset of velocity-mode demonstrations."""

    def __init__(
        self,
        data_dir: str,
        image_scaling: float = 0.25,
        chunk_size: int = 100,
        min_episode_length: int = 20,
    ):
        self.data_dir = Path(data_dir)
        self.image_scaling = image_scaling
        self.chunk_size = chunk_size
        self.episodes = []
        self.cumulative_lengths = []

        # Load all episodes
        total = 0
        for ep_dir in sorted(self.data_dir.iterdir()):
            if not ep_dir.is_dir() or not ep_dir.name.startswith("episode_"):
                continue
            states = np.load(ep_dir / "states.npy")
            actions = np.load(ep_dir / "actions.npy")
            if len(states) < min_episode_length:
                print(
                    f"  Skipping {ep_dir.name}: too short ({len(states)} steps)"
                )
                continue
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

        # Build weighted sample index (oversample insertion phase)
        self._build_sample_index(insertion_weight=3.0)

        # Compute image normalization stats from a sample
        self._compute_image_stats()

    def _compute_image_stats(self):
        """Compute per-channel image mean and std from a sample."""
        samples = []
        for ep in self.episodes[:5]:  # Sample from first 5 episodes
            for i in range(0, ep["length"], 10):  # Every 10th frame
                for cam in ["left", "center", "right"]:
                    img_path = ep["dir"] / f"images_{cam}_{i:04d}.npy"
                    if img_path.exists():
                        img = np.load(img_path).astype(np.float32) / 255.0
                        samples.append(img)
        samples = np.array(samples)  # (N, H, W, 3)
        self.img_mean = torch.tensor(
            samples.mean(axis=(0, 1, 2)), dtype=torch.float32
        ).view(1, 3, 1, 1)
        self.img_std = torch.tensor(
            samples.std(axis=(0, 1, 2)), dtype=torch.float32
        ).view(1, 3, 1, 1)
        self.img_std[self.img_std < 1e-6] = 1.0

    def __len__(self):
        return len(self._sample_index)

    def _build_sample_index(self, insertion_weight: float = 3.0):
        """Build weighted sample index that oversamples the insertion phase.

        The last 30% of each episode (insertion/descent) gets repeated
        insertion_weight times to give the model more examples of the
        fine insertion movements.
        """
        self._sample_index = []
        for ep_idx, ep in enumerate(self.episodes):
            insertion_start = int(ep["length"] * 0.7)
            for step in range(ep["length"]):
                self._sample_index.append((ep_idx, step))
                # Oversample the insertion phase
                if step >= insertion_start:
                    for _ in range(int(insertion_weight) - 1):
                        self._sample_index.append((ep_idx, step))
        print(
            f"Sample index: {len(self._sample_index)} entries "
            f"(insertion_weight={insertion_weight})"
        )

    def __getitem__(self, idx):
        ep_idx, step_idx = self._sample_index[idx]
        ep = self.episodes[ep_idx]

        # State
        state = torch.tensor(ep["states"][step_idx], dtype=torch.float32)
        state_norm = (state - self.state_mean) / self.state_std

        # Action chunk (pad with last action if needed)
        actions = ep["actions"][step_idx : step_idx + self.chunk_size]
        n_real = len(actions)
        if n_real < self.chunk_size:
            pad = np.tile(actions[-1:], (self.chunk_size - n_real, 1))
            actions = np.concatenate([actions, pad])
        actions = torch.tensor(actions, dtype=torch.float32)
        actions_norm = (actions - self.action_mean) / self.action_std

        # Padding mask: True for padded positions
        action_is_pad = torch.zeros(self.chunk_size, dtype=torch.bool)
        action_is_pad[n_real:] = True

        # Images (load, convert HWC->CHW, normalize)
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

    # Load pretrained model
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

    # Load dataset
    dataset = VelocityDemoDataset(
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

    # Optimizer
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Training loop
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    policy.train()

    steps_per_epoch = len(dataloader)
    total_steps = args.epochs * steps_per_epoch
    print(
        f"Training: {args.epochs} epochs, {steps_per_epoch} steps/epoch, "
        f"{total_steps} total steps, batch_size={args.batch_size}"
    )

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass (returns (loss, loss_dict))
            loss, _loss_dict = policy.forward(batch)

            # Backward pass
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

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or avg_loss < best_loss:
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_path = output_dir / "best"
            else:
                save_path = output_dir / f"checkpoint_{epoch + 1}"

            save_path.mkdir(parents=True, exist_ok=True)

            # Save model weights
            save_file(policy.state_dict(), save_path / "model.safetensors")

            # Save config
            with open(save_path / "config.json", "w") as f:
                json.dump(config_dict, f, indent=2)

            # Save our normalization stats as safetensors
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
    parser = argparse.ArgumentParser(description="Fine-tune ACTPolicy")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory with collected velocity demo episodes",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Where to save trained model",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--save-every", type=int, default=10)
    args = parser.parse_args()
    train(args)
