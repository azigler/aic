#!/usr/bin/env python3
"""Fine-tune pretrained ACTPolicy on collected velocity-mode demos.

Usage:
    pixi run python scripts/train_act.py \
        --data-dir ~/aic-work/data/velocity \
        --output-dir ~/aic-work/models/act_finetuned \
        --epochs 100 --batch-size 8 --lr 5e-6 \
        --val-frac 0.2 --patience 10

This loads the pretrained grkw/aic_act_policy weights and fine-tunes
on our collected demonstrations with a validation split and early stopping.

Key design choices (see bd-1t7 and experiment-log.md):
  - Train/val split by EPISODE (not step) so val frames come from unseen
    trajectories, matching eval conditions where whole configs are new.
  - Best checkpoint selected by val loss (not train loss). Train loss goes
    to ~0 regardless of overfitting; val loss is the honest signal.
  - insertion_weight defaults to 1.0 (no re-weighting) because exp-047,
    exp-059, exp-060 all regressed with any value > 1.0. Kept as a
    CLI arg only so future experiments can be explicit about turning it
    back on for ablations.
"""

import argparse
import json
import random
import time
from pathlib import Path

import draccus
import numpy as np
import torch
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class VelocityDemoDataset(Dataset):
    """Dataset of velocity-mode demonstrations.

    Pass `episode_whitelist` to restrict the loaded episodes to a subset
    (used to build train/val splits that share preprocessing stats).
    """

    def __init__(
        self,
        data_dir: str,
        image_scaling: float = 0.25,
        chunk_size: int = 100,
        min_episode_length: int = 20,
        insertion_weight: float = 1.0,
        episode_whitelist: list[str] | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.image_scaling = image_scaling
        self.chunk_size = chunk_size
        self.episodes = []
        self.cumulative_lengths = []

        whitelist = set(episode_whitelist) if episode_whitelist else None

        total = 0
        for ep_dir in sorted(self.data_dir.iterdir()):
            if not ep_dir.is_dir() or not ep_dir.name.startswith("episode_"):
                continue
            if whitelist is not None and ep_dir.name not in whitelist:
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
                    "name": ep_dir.name,
                    "states": states,
                    "actions": actions,
                    "length": len(states),
                }
            )
            total += len(states)
            self.cumulative_lengths.append(total)

        print(f"Loaded {len(self.episodes)} episodes, {total} total steps")

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

        self._build_sample_index(insertion_weight=insertion_weight)
        self._compute_image_stats()

    def set_preprocess_stats(
        self,
        state_mean,
        state_std,
        action_mean,
        action_std,
        img_mean,
        img_std,
    ) -> None:
        """Overwrite stats, used by val split to share train-set stats."""
        self.state_mean = state_mean
        self.state_std = state_std
        self.action_mean = action_mean
        self.action_std = action_std
        self.img_mean = img_mean
        self.img_std = img_std

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
        return len(self._sample_index)

    def _build_sample_index(self, insertion_weight: float = 1.0):
        """Build sample index. insertion_weight=1.0 means no reweighting."""
        self._sample_index = []
        for ep_idx, ep in enumerate(self.episodes):
            insertion_start = int(ep["length"] * 0.7)
            for step in range(ep["length"]):
                self._sample_index.append((ep_idx, step))
                if insertion_weight > 1.0 and step >= insertion_start:
                    for _ in range(int(insertion_weight) - 1):
                        self._sample_index.append((ep_idx, step))
        print(
            f"Sample index: {len(self._sample_index)} entries "
            f"(insertion_weight={insertion_weight})"
        )

    def __getitem__(self, idx):
        ep_idx, step_idx = self._sample_index[idx]
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


def _split_episodes(
    data_dir: Path, val_frac: float, seed: int
) -> tuple[list[str], list[str]]:
    """Pick val episodes randomly by filename. Returns (train_names, val_names)."""
    all_names = sorted(
        p.name
        for p in data_dir.iterdir()
        if p.is_dir() and p.name.startswith("episode_")
    )
    rng = random.Random(seed)
    shuffled = all_names[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_frac))
    val = sorted(shuffled[:n_val])
    train = sorted(shuffled[n_val:])
    return train, val


def _save_checkpoint(
    policy: ACTPolicy,
    dataset: VelocityDemoDataset,
    config_dict: dict,
    save_path: Path,
) -> None:
    save_path.mkdir(parents=True, exist_ok=True)
    save_file(policy.state_dict(), save_path / "model.safetensors")
    with open(save_path / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    img_mean = dataset.img_mean.reshape(1, 3, 1, 1).contiguous()
    img_std = dataset.img_std.reshape(1, 3, 1, 1).contiguous()
    assert img_mean.numel() == 3 and img_std.numel() == 3, (
        f"image stats must have 3 elements; got {img_mean.shape}, {img_std.shape}"
    )
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


def _evaluate(policy: ACTPolicy, loader: DataLoader, device) -> float:
    policy.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, _ = policy.forward(batch)
            total += loss.item()
            n += 1
    policy.train()
    return total / max(n, 1)


def train(args):
    _seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, seed: {args.seed}")

    # Pretrained checkpoint
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

    # Episode-level train/val split
    data_dir = Path(args.data_dir).expanduser()
    train_names, val_names = _split_episodes(data_dir, args.val_frac, args.seed)
    print(f"Split: {len(train_names)} train, {len(val_names)} val episodes")

    train_dataset = VelocityDemoDataset(
        data_dir,
        chunk_size=config.chunk_size,
        min_episode_length=20,
        insertion_weight=args.insertion_weight,
        episode_whitelist=train_names,
    )
    # Val dataset: shares train stats so normalization is consistent
    val_dataset = VelocityDemoDataset(
        data_dir,
        chunk_size=config.chunk_size,
        min_episode_length=20,
        insertion_weight=1.0,
        episode_whitelist=val_names,
    )
    val_dataset.set_preprocess_stats(
        train_dataset.state_mean,
        train_dataset.state_std,
        train_dataset.action_mean,
        train_dataset.action_std,
        train_dataset.img_mean,
        train_dataset.img_std,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    steps_per_epoch = len(train_loader)
    print(
        f"Training: {args.epochs} epochs max, {steps_per_epoch} train steps/epoch, "
        f"{len(val_loader)} val batches, batch_size={args.batch_size}, "
        f"lr={args.lr}, patience={args.patience}"
    )

    policy.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, _loss_dict = policy.forward(batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 50 == 0:
                print(
                    f"  Epoch {epoch + 1}/{args.epochs}, "
                    f"Step {batch_idx}/{steps_per_epoch}, "
                    f"Loss: {loss.item():.6f}"
                )

        avg_train_loss = epoch_loss / max(steps_per_epoch, 1)
        val_loss = _evaluate(policy, val_loader, device)
        elapsed = time.time() - epoch_start
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "time_s": elapsed,
            }
        )
        print(
            f"Epoch {epoch + 1}/{args.epochs}: train={avg_train_loss:.6f}, "
            f"val={val_loss:.6f}, time={elapsed:.1f}s"
        )

        improved = val_loss < best_val_loss - args.min_delta
        if improved:
            best_val_loss = val_loss
            patience_counter = 0
            _save_checkpoint(
                policy, train_dataset, config_dict, output_dir / "best"
            )
            print(f"  ↓ new best val_loss={val_loss:.6f}, saved to best/")
        else:
            patience_counter += 1
            print(f"  no improvement ({patience_counter}/{args.patience})")

        if (epoch + 1) % args.save_every == 0:
            _save_checkpoint(
                policy,
                train_dataset,
                config_dict,
                output_dir / f"checkpoint_{epoch + 1}",
            )

        if patience_counter >= args.patience:
            print(
                f"Early stopping at epoch {epoch + 1} "
                f"(no val improvement for {args.patience} epochs)"
            )
            break

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"Training complete! Best val loss: {best_val_loss:.6f}")
    print(f"Best model saved to: {output_dir / 'best'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune ACTPolicy")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stop after this many epochs with no val improvement",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-5,
        help="Minimum val_loss drop to count as improvement",
    )
    parser.add_argument(
        "--insertion-weight",
        type=float,
        default=1.0,
        help="Oversample insertion phase (>1.0). Empirically hurts; kept for ablations.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)
