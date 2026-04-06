#!/usr/bin/env python3
"""Train ACT policy on collected CheatCode demonstrations.

Loads episodes from ~/training_data/, builds a PyTorch Dataset, configures
ACT for our observation/action spaces, and trains with L1 loss + Adam.

Observation space (matching RunACT.prepare_observations):
  - 3 cameras at 256x256 (left, center, right)
  - 26D robot state

Action space (position mode):
  - 7D: TCP position xyz (3) + quaternion xyzw (4)

Usage:
  pixi run python scripts/train_act.py --data-dir ~/training_data --output-dir ~/models/act_v1
  pixi run python scripts/train_act.py --resume ~/models/act_v1/checkpoint_latest.pt
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ACT policy on demo data"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / "training_data",
        help="Directory containing episode_NNNN folders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "models" / "act_v1",
        help="Directory for checkpoints and final model",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Initial learning rate"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="Number of future actions to predict per step",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of episodes for validation",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker count",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=256,
        help="Image resolution for training (default 256)",
    )
    parser.add_argument(
        "--offset-mode",
        action="store_true",
        help="Mark model as offset-mode (actions are position deltas)",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply image augmentation during training (brightness, contrast, noise)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class DemoDataset(Dataset):
    """PyTorch Dataset over collected demonstration episodes.

    Each sample is a (state, images, action_chunk) tuple at a single timestep.
    The action_chunk is a sequence of chunk_size future actions starting from
    that timestep (zero-padded at episode boundaries).

    Images are loaded lazily from individual .npy files to avoid OOM on large
    datasets.
    """

    def __init__(
        self,
        episode_dirs: list[Path],
        chunk_size: int = 50,
        img_size: int = 256,
        augment: bool = False,
    ):
        self.chunk_size = chunk_size
        self.img_size = img_size
        self.augment = augment
        self.samples: list[
            tuple[Path, int, int]
        ] = []  # (episode_dir, timestep, ep_len)

        for ep_dir in sorted(episode_dirs):
            actions_path = ep_dir / "actions.npy"
            if not actions_path.exists():
                continue
            actions = np.load(actions_path)
            ep_len = len(actions)
            for t in range(ep_len):
                self.samples.append((ep_dir, t, ep_len))

    def _augment_image(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Apply random augmentation to a CHW float [0,1] image tensor."""
        # Random brightness (±20%)
        if torch.rand(1).item() < 0.5:
            factor = 0.8 + 0.4 * torch.rand(1).item()
            img_tensor = (img_tensor * factor).clamp(0, 1)

        # Random contrast (±20%)
        if torch.rand(1).item() < 0.5:
            mean = img_tensor.mean()
            factor = 0.8 + 0.4 * torch.rand(1).item()
            img_tensor = ((img_tensor - mean) * factor + mean).clamp(0, 1)

        # Gaussian noise (small, std=0.02)
        if torch.rand(1).item() < 0.5:
            noise = torch.randn_like(img_tensor) * 0.02
            img_tensor = (img_tensor + noise).clamp(0, 1)

        return img_tensor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        ep_dir, t, ep_len = self.samples[idx]

        # State
        states = np.load(ep_dir / "states.npy")
        state = torch.from_numpy(states[t]).float()

        # Images (lazy load single frame)
        imgs = {}
        for cam in ("left", "center", "right"):
            img_path = ep_dir / f"images_{cam}_{t:04d}.npy"
            if img_path.exists():
                img_np = np.load(img_path)
            else:
                # Fallback: black image
                img_np = np.zeros(
                    (self.img_size, self.img_size, 3), dtype=np.uint8
                )
            # HWC uint8 -> CHW float [0,1]
            img_tensor = (
                torch.from_numpy(img_np).permute(2, 0, 1).float().div(255.0)
            )
            if self.augment:
                img_tensor = self._augment_image(img_tensor)
            imgs[cam] = img_tensor

        # Action chunk: actions[t:t+chunk_size], zero-padded if needed
        actions = np.load(ep_dir / "actions.npy")
        chunk_end = min(t + self.chunk_size, ep_len)
        action_chunk = actions[t:chunk_end]
        if len(action_chunk) < self.chunk_size:
            pad = np.zeros(
                (self.chunk_size - len(action_chunk), actions.shape[1]),
                dtype=np.float32,
            )
            action_chunk = np.concatenate([action_chunk, pad], axis=0)
        action_chunk = torch.from_numpy(action_chunk).float()

        return state, imgs["left"], imgs["center"], imgs["right"], action_chunk


# ---------------------------------------------------------------------------
# Normalization statistics
# ---------------------------------------------------------------------------


def compute_normalization_stats(
    episode_dirs: list[Path],
) -> dict[str, torch.Tensor]:
    """Compute mean/std for states and actions across all episodes.

    Image normalization is computed per-camera over a random subset of frames
    to avoid loading every image into memory.
    """
    all_states = []
    all_actions = []

    for ep_dir in episode_dirs:
        states_path = ep_dir / "states.npy"
        actions_path = ep_dir / "actions.npy"
        if states_path.exists():
            all_states.append(np.load(states_path))
        if actions_path.exists():
            all_actions.append(np.load(actions_path))

    if not all_states:
        raise ValueError("No episodes found with states.npy")

    states_cat = np.concatenate(all_states, axis=0)
    actions_cat = np.concatenate(all_actions, axis=0)

    stats = {
        "state_mean": torch.from_numpy(states_cat.mean(axis=0)).float(),
        "state_std": torch.from_numpy(
            states_cat.std(axis=0).clip(min=1e-6)
        ).float(),
        "action_mean": torch.from_numpy(actions_cat.mean(axis=0)).float(),
        "action_std": torch.from_numpy(
            actions_cat.std(axis=0).clip(min=1e-6)
        ).float(),
    }

    # Sample image stats from up to 200 random frames across episodes
    rng = np.random.default_rng(42)
    sample_frames: list[tuple[Path, int]] = []
    for ep_dir in episode_dirs:
        meta_path = ep_dir / "metadata.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        ep_len = meta.get("num_timesteps", 0)
        for t in range(ep_len):
            sample_frames.append((ep_dir, t))

    if len(sample_frames) > 200:
        indices = rng.choice(len(sample_frames), 200, replace=False)
        sample_frames = [sample_frames[i] for i in indices]

    for cam in ("left", "center", "right"):
        pixel_sums = np.zeros(3, dtype=np.float64)
        pixel_sq_sums = np.zeros(3, dtype=np.float64)
        pixel_count = 0

        for ep_dir, t in sample_frames:
            img_path = ep_dir / f"images_{cam}_{t:04d}.npy"
            if not img_path.exists():
                continue
            img = np.load(img_path).astype(np.float64) / 255.0
            pixel_sums += img.mean(axis=(0, 1))
            pixel_sq_sums += (img**2).mean(axis=(0, 1))
            pixel_count += 1

        if pixel_count > 0:
            mean = pixel_sums / pixel_count
            var = pixel_sq_sums / pixel_count - mean**2
            std = np.sqrt(np.clip(var, 1e-6, None))
        else:
            mean = np.array([0.5, 0.5, 0.5])
            std = np.array([0.25, 0.25, 0.25])

        stats[f"img_{cam}_mean"] = torch.from_numpy(mean).float()
        stats[f"img_{cam}_std"] = torch.from_numpy(std).float()

    return stats


# ---------------------------------------------------------------------------
# ACT Model (simplified standalone, compatible with LeRobot ACTConfig)
# ---------------------------------------------------------------------------

# Import SimpleACT from the package, or define inline for standalone training
try:
    from aic_example_policies.simple_act import SimpleACT
except ImportError:
    # Standalone mode (running script directly without package installed)
    import sys

    sys.path.insert(
        0,
        str(
            Path(__file__).resolve().parent.parent
            / "aic_example_policies"
            / "aic_example_policies"
        ),
    )
    from simple_act import SimpleACT

HAVE_LEROBOT = False


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Discover episodes
    data_dir = args.data_dir
    episode_dirs = sorted(
        [
            d
            for d in data_dir.iterdir()
            if d.is_dir() and d.name.startswith("episode_")
        ]
    )
    if not episode_dirs:
        print(f"ERROR: No episodes found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(episode_dirs)} episodes in {data_dir}")

    # Train/val split
    n_val = max(1, int(len(episode_dirs) * args.val_split))
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(episode_dirs))
    val_indices = set(perm[:n_val].tolist())
    train_dirs = [d for i, d in enumerate(episode_dirs) if i not in val_indices]
    val_dirs = [d for i, d in enumerate(episode_dirs) if i in val_indices]
    print(f"Train episodes: {len(train_dirs)}, Val episodes: {len(val_dirs)}")

    # Compute normalization stats
    print("Computing normalization statistics...")
    stats = compute_normalization_stats(train_dirs)
    for k, v in stats.items():
        print(f"  {k}: mean={v.mean().item():.4f}, shape={tuple(v.shape)}")

    # Save stats and config for inference
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(stats, args.output_dir / "normalization_stats.pt")

    # Save model config so OurACT can read img_size and offset_mode at inference time
    model_config = {
        "state_dim": 26,
        "action_dim": 7,
        "chunk_size": args.chunk_size,
        "img_size": args.img_size,
        "offset_mode": args.offset_mode,
    }
    with open(args.output_dir / "config.json", "w") as f:
        json.dump(model_config, f, indent=2)

    # Datasets (augment training data only, not validation)
    train_dataset = DemoDataset(
        train_dirs,
        chunk_size=args.chunk_size,
        img_size=args.img_size,
        augment=args.augment,
    )
    val_dataset = DemoDataset(
        val_dirs,
        chunk_size=args.chunk_size,
        img_size=args.img_size,
        augment=False,
    )
    print(
        f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}"
    )

    if len(train_dataset) == 0:
        print("ERROR: No training samples found.")
        sys.exit(1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Build model
    # Always use SimpleACT -- LeRobot ACTConfig API varies across versions
    if False:
        pass
    else:
        print("LeRobot not available, using SimpleACT model")
        model = SimpleACT(
            state_dim=26,
            action_dim=7,
            chunk_size=args.chunk_size,
            img_size=args.img_size,
        )
        model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(
        f"Model parameters: {total_params:,} total, {trainable_params:,} trainable"
    )

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    loss_fn = nn.L1Loss()

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume and args.resume.exists():
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(
            args.resume, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(
            f"Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.6f}"
        )

    # Move normalization stats to device
    state_mean = stats["state_mean"].to(device)
    state_std = stats["state_std"].to(device)
    action_mean = stats["action_mean"].to(device)
    action_std = stats["action_std"].to(device)
    img_stats_device = {}
    for cam in ("left", "center", "right"):
        img_stats_device[cam] = {
            "mean": stats[f"img_{cam}_mean"].to(device).view(1, 3, 1, 1),
            "std": stats[f"img_{cam}_std"].to(device).view(1, 3, 1, 1),
        }

    def normalize_batch(state, img_l, img_c, img_r, action_chunk):
        """Apply normalization to a batch."""
        state_norm = (state - state_mean) / state_std
        img_l_norm = (
            img_l - img_stats_device["left"]["mean"]
        ) / img_stats_device["left"]["std"]
        img_c_norm = (
            img_c - img_stats_device["center"]["mean"]
        ) / img_stats_device["center"]["std"]
        img_r_norm = (
            img_r - img_stats_device["right"]["mean"]
        ) / img_stats_device["right"]["std"]
        action_norm = (
            action_chunk - action_mean.unsqueeze(0)
        ) / action_std.unsqueeze(0)
        return state_norm, img_l_norm, img_c_norm, img_r_norm, action_norm

    # Training log
    log_path = args.output_dir / "training_log.jsonl"

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Checkpoints saved to {args.output_dir}")
    print()

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for batch in train_loader:
            state, img_l, img_c, img_r, action_chunk = [
                b.to(device) for b in batch
            ]

            state, img_l, img_c, img_r, action_chunk = normalize_batch(
                state, img_l, img_c, img_r, action_chunk
            )

            if HAVE_LEROBOT:
                # LeRobot ACT expects a specific dict format
                obs_dict = {
                    "observation.images.left_camera": img_l,
                    "observation.images.center_camera": img_c,
                    "observation.images.right_camera": img_r,
                    "observation.state": state,
                }
                loss_dict = model.forward(obs_dict, action_chunk)
                loss = loss_dict["loss"]
            else:
                pred_actions = model(state, img_l, img_c, img_r)
                loss = loss_fn(pred_actions, action_chunk)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

        avg_train_loss = train_loss_sum / max(train_batches, 1)

        # --- Validate ---
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                state, img_l, img_c, img_r, action_chunk = [
                    b.to(device) for b in batch
                ]

                state, img_l, img_c, img_r, action_chunk = normalize_batch(
                    state, img_l, img_c, img_r, action_chunk
                )

                if HAVE_LEROBOT:
                    obs_dict = {
                        "observation.images.left_camera": img_l,
                        "observation.images.center_camera": img_c,
                        "observation.images.right_camera": img_r,
                        "observation.state": state,
                    }
                    loss_dict = model.forward(obs_dict, action_chunk)
                    loss = loss_dict["loss"]
                else:
                    pred_actions = model(state, img_l, img_c, img_r)
                    loss = loss_fn(pred_actions, action_chunk)

                val_loss_sum += loss.item()
                val_batches += 1

        avg_val_loss = val_loss_sum / max(val_batches, 1)
        scheduler.step()
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        # Log
        log_entry = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "lr": current_lr,
            "epoch_time_s": epoch_time,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        improved = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save best model
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_loss,
                args.output_dir / "checkpoint_best.pt",
            )
            improved = " [BEST]"

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss: {avg_train_loss:.6f} | "
            f"val_loss: {avg_val_loss:.6f}{improved} | "
            f"lr: {current_lr:.2e} | "
            f"time: {epoch_time:.1f}s"
        )

        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_loss,
                args.output_dir / "checkpoint_latest.pt",
            )

    # Final save
    save_checkpoint(
        model,
        optimizer,
        scheduler,
        args.epochs - 1,
        best_val_loss,
        args.output_dir / "checkpoint_latest.pt",
    )

    # Export best model for deployment
    print(f"\nTraining complete. Best val_loss: {best_val_loss:.6f}")
    print(f"Checkpoints in {args.output_dir}")
    print(f"Training log: {log_path}")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_val_loss: float,
    path: Path,
):
    """Save a training checkpoint."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
        },
        path,
    )


if __name__ == "__main__":
    args = parse_args()
    train(args)
