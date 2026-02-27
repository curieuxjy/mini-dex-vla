"""
Train VLA on DexMachina dataset

DexMachina 환경에서 수집된 데이터로 VLA 모델 학습

Usage:
    python -m scripts.train_dexmachina \
        --dataset-path data/dexmachina_box.npz \
        --epochs 100 \
        --batch-size 64 \
        --model-size base \
        --save-path checkpoints/dexmachina_model.pt
"""

import argparse
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.vla_dexmachina import VLADexMachinaPolicy, create_dexmachina_model
from utils.action_normalizer import AllegroActionNormalizer


class DexMachinaDataset(Dataset):
    """
    DexMachina 데이터셋 로더

    수집된 npz 파일에서 데이터를 로드

    Action normalization 방식:
    - "joint_limits": Joint limit 기반 [-1, 1] 정규화 (권장)
    - "statistical": Mean/std 기반 zero-mean unit-variance 정규화
    - "none": 정규화 없음
    """
    def __init__(self, path, resize_to=64, normalize_actions=True,
                 action_norm_mode="joint_limits"):
        data = np.load(path, allow_pickle=True)
        self.images = data["images"]             # (N, H, W, 3)
        self.states = data["states"]             # (N, state_dim)
        self.actions = data["actions"]           # (N, action_dim)
        self.text_ids = data["text_ids"]         # (N, T_text)
        self.vocab = data["vocab"].item() if data["vocab"].shape == () else data["vocab"]

        # 메타데이터 로드 (있는 경우)
        if "metadata" in data:
            self.metadata = data["metadata"].item()
        else:
            self.metadata = {}

        self.resize_to = resize_to
        self.normalize_actions = normalize_actions
        self.action_norm_mode = action_norm_mode if normalize_actions else "none"

        # Action normalization 설정
        action_dim = self.actions.shape[1]
        self.joint_normalizer = None

        if self.action_norm_mode == "joint_limits" and action_dim in (22, 44):
            self.joint_normalizer = AllegroActionNormalizer(action_dim=action_dim)
            # 전체 데이터를 joint limit 기반으로 정규화
            self.actions_normalized = self.joint_normalizer.normalize(self.actions)
            self.action_mean = np.zeros(action_dim, dtype=np.float32)
            self.action_std = np.ones(action_dim, dtype=np.float32)
            print(f"  Action norm: joint_limits ([-1, 1])")
        elif self.action_norm_mode == "statistical" or (
                self.action_norm_mode == "joint_limits" and action_dim not in (22, 44)):
            if self.action_norm_mode == "joint_limits":
                print(f"  Warning: joint_limits not available for action_dim={action_dim}, "
                      f"falling back to statistical")
                self.action_norm_mode = "statistical"
            self.action_mean = self.actions.mean(axis=0)
            self.action_std = self.actions.std(axis=0) + 1e-8
            self.actions_normalized = (self.actions - self.action_mean) / self.action_std
            print(f"  Action norm: statistical (zero-mean, unit-variance)")
        else:
            self.action_mean = np.zeros(action_dim, dtype=np.float32)
            self.action_std = np.ones(action_dim, dtype=np.float32)
            self.actions_normalized = self.actions.copy()
            print(f"  Action norm: none")

        try:
            import cv2
            self.cv2 = cv2
        except ImportError:
            self.cv2 = None

        print(f"Loaded dataset from {path}")
        print(f"  Images: {self.images.shape}")
        print(f"  States: {self.states.shape}")
        print(f"  Actions: {self.actions.shape}")
        print(f"  Text IDs: {self.text_ids.shape}")

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img = self.images[idx]  # (H, W, 3), uint8

        # Resize if needed
        if self.cv2 is not None and (img.shape[0] != self.resize_to or img.shape[1] != self.resize_to):
            img = self.cv2.resize(img, (self.resize_to, self.resize_to))

        # Convert to tensor and normalize
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (3, H, W)

        state = torch.from_numpy(self.states[idx]).float()

        action = torch.from_numpy(self.actions_normalized[idx].astype(np.float32))

        text_ids = torch.from_numpy(self.text_ids[idx]).long()

        return img, state, action, text_ids

    def get_action_stats(self):
        """Get action normalization statistics for checkpoint"""
        stats = {
            "mean": self.action_mean,
            "std": self.action_std,
            "norm_mode": self.action_norm_mode,
        }
        if self.joint_normalizer is not None:
            stats["joint_limits"] = self.joint_normalizer.get_limits()
        return stats


def parse_args():
    parser = argparse.ArgumentParser(description="Train DexMachina VLA model")

    # Data
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to npz dataset")
    parser.add_argument("--resize-to", type=int, default=64,
                        help="Resize images to this size")
    parser.add_argument("--normalize-actions", action="store_true", default=True,
                        help="Normalize actions (default: True)")
    parser.add_argument("--action-norm-mode", type=str, default="joint_limits",
                        choices=["joint_limits", "statistical", "none"],
                        help="Action normalization: joint_limits ([-1,1]), "
                             "statistical (zero-mean), or none")

    # Model
    parser.add_argument("--model-size", type=str, default="base",
                        choices=["small", "base", "large"],
                        help="Model size configuration")
    parser.add_argument("--d-model", type=int, default=None,
                        help="Override d_model (optional)")
    parser.add_argument("--diffusion-T", type=int, default=None,
                        help="Override diffusion steps (optional)")

    # Training
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping max norm")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="Number of warmup epochs")

    # Checkpointing
    parser.add_argument("--save-path", type=str,
                        default="checkpoints/dexmachina_model.pt")
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")

    # Misc
    parser.add_argument("--device", type=str, default="cuda",
                        help="'cuda' or 'cpu'")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)

    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Create directories
    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    dataset = DexMachinaDataset(
        args.dataset_path,
        resize_to=args.resize_to,
        normalize_actions=args.normalize_actions,
        action_norm_mode=args.action_norm_mode,
    )

    vocab_size = max(dataset.vocab.values()) + 1
    state_dim = dataset.states.shape[1]
    action_dim = dataset.actions.shape[1]

    print(f"\nDataset info:")
    print(f"  Vocab size: {vocab_size}")
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Samples: {len(dataset)}")

    # Create model
    model = create_dexmachina_model(
        vocab_size=vocab_size,
        state_dim=state_dim,
        action_dim=action_dim,
        model_size=args.model_size,
    ).to(device)

    # Override model config if specified
    if args.d_model is not None or args.diffusion_T is not None:
        print("Warning: Overriding model config with command line args")
        kwargs = {
            "vocab_size": vocab_size,
            "state_dim": state_dim,
            "action_dim": action_dim,
        }
        if args.d_model is not None:
            kwargs["d_model"] = args.d_model
        if args.diffusion_T is not None:
            kwargs["diffusion_T"] = args.diffusion_T

        model = VLADexMachinaPolicy(**kwargs).to(device)

    print(f"\nModel info:")
    print(f"  Total params: {model.get_num_params():,}")
    print(f"  Trainable params: {model.get_num_trainable_params():,}")

    # DataLoader
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs - args.warmup_epochs,
        eta_min=args.lr * 0.01,
    )

    # Resume from checkpoint
    start_epoch = 0
    if args.resume is not None:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]
        print(f"  Resumed at epoch {start_epoch}")

    # Training loop
    print(f"\nStarting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")

    best_loss = float('inf')

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (img, state, action, text_ids) in enumerate(loader):
            img = img.to(device)
            state = state.to(device)
            action = action.to(device)
            text_ids = text_ids.to(device)

            # Forward pass
            loss = model.loss(img, text_ids, state, action)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        # Update learning rate (after warmup)
        if epoch >= args.warmup_epochs:
            scheduler.step()

        avg_loss = total_loss / num_batches
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1:3d}/{args.epochs}  loss={avg_loss:.6f}  lr={current_lr:.2e}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab": dataset.vocab,
                    "state_dim": state_dim,
                    "action_dim": action_dim,
                    "model_size": args.model_size,
                    "action_stats": dataset.get_action_stats(),
                    "epoch": epoch + 1,
                    "loss": avg_loss,
                },
                args.save_path.replace(".pt", "_best.pt"),
            )

        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = args.save_path.replace(".pt", f"_epoch{epoch+1}.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "vocab": dataset.vocab,
                    "state_dim": state_dim,
                    "action_dim": action_dim,
                    "model_size": args.model_size,
                    "action_stats": dataset.get_action_stats(),
                    "epoch": epoch + 1,
                    "loss": avg_loss,
                },
                checkpoint_path,
            )
            print(f"  Saved checkpoint: {checkpoint_path}")

    # Save final model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": dataset.vocab,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "model_size": args.model_size,
            "action_stats": dataset.get_action_stats(),
            "epoch": args.epochs,
            "loss": avg_loss,
        },
        args.save_path,
    )

    print(f"\n{'='*50}")
    print(f"Training complete!")
    print(f"  Final loss: {avg_loss:.6f}")
    print(f"  Best loss: {best_loss:.6f}")
    print(f"  Saved to: {args.save_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
