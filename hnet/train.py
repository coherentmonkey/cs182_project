#!/usr/bin/env python3
"""
Training script for H-Net on the cs182 data directory.

Usage:
    python train.py --data-dir ../data/t-C_vs-16_sl-32_ntr-12800_nte-1280_km-1_vm-1_mq-1_fn-0#0_nvs-0_ntc-0_s-12345 \
                    --config-path configs/hnet_1stage_L.json \
                    --output-dir outputs/my_model \
                    --batch-size 32 \
                    --num-epochs 10
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from hnet.models.mixer_seq import HNetForCausalLM
from hnet.models.config_hnet import HNetConfig, AttnConfig, SSMConfig
from hnet.utils.train import load_balancing_loss, group_params


class SequenceDataset(Dataset):
    """Dataset for loading numpy arrays from the cs182 data directory."""

    def __init__(self, data_dir, split="train"):
        """
        Args:
            data_dir: Path to the data directory (e.g., t-C_vs-16_sl-32_ntr-12800_...)
            split: Either "train" or "test"
        """
        self.data_dir = Path(data_dir) / split

        # Load inputs and targets
        self.inputs = np.load(self.data_dir / "inputs.npy")
        self.targets = np.load(self.data_dir / "targets.npy")

        assert len(self.inputs) == len(self.targets), \
            f"Input and target lengths don't match: {len(self.inputs)} vs {len(self.targets)}"

        print(f"Loaded {split} dataset from {data_dir}")
        print(f"  Samples: {len(self.inputs)}")
        print(f"  Sequence length: {self.inputs.shape[1]}")
        print(f"  Input shape: {self.inputs.shape}")
        print(f"  Target shape: {self.targets.shape}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.from_numpy(self.inputs[idx]).long(),
            "labels": torch.from_numpy(self.targets[idx]).long(),
        }


def create_model_from_config(config_path, device, dtype=torch.float32):
    """Load model configuration and create model."""
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # Create config objects
    attn_cfg = AttnConfig(**config_dict.pop("attn_cfg"))
    ssm_cfg = SSMConfig(**config_dict.pop("ssm_cfg"))
    hnet_cfg = HNetConfig(**config_dict, attn_cfg=attn_cfg, ssm_cfg=ssm_cfg)

    # Create model
    model = HNetForCausalLM(hnet_cfg, device=device, dtype=dtype)
    model.init_weights(initializer_range=0.02)

    return model, hnet_cfg


def train_epoch(model, dataloader, optimizer, device, config, aux_loss_weight=0.01):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_aux_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        output = model(input_ids)

        # Compute cross-entropy loss
        # Shift so that tokens < n predict n
        shift_logits = output.logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        ce_loss = nn.functional.cross_entropy(
            shift_logits.view(-1, config.vocab_size),
            shift_labels.view(-1),
            reduction='mean'
        )

        # Add load balancing auxiliary loss if using dynamic chunking
        aux_loss = 0.0
        if output.bpred_output:
            for router_output in output.bpred_output:
                # Infer N from the router output
                N = 2.0  # Default downsampling factor
                aux_loss += load_balancing_loss(router_output, N=N)
            aux_loss = aux_loss / len(output.bpred_output)

        # Total loss
        loss = ce_loss + aux_loss_weight * aux_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        total_aux_loss += aux_loss if isinstance(aux_loss, float) else aux_loss.item()
        num_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ce': f'{ce_loss.item():.4f}',
            'aux': f'{aux_loss if isinstance(aux_loss, float) else aux_loss.item():.4f}'
        })

    return {
        'loss': total_loss / num_batches,
        'ce_loss': total_ce_loss / num_batches,
        'aux_loss': total_aux_loss / num_batches,
    }


def evaluate(model, dataloader, device, config):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    num_batches = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            output = model(input_ids)

            # Compute loss
            shift_logits = output.logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, config.vocab_size),
                shift_labels.view(-1),
                reduction='mean'
            )

            # Compute accuracy
            predictions = shift_logits.argmax(dim=-1)
            correct = (predictions == shift_labels).sum().item()
            total_correct += correct
            total_tokens += shift_labels.numel()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * total_correct / total_tokens:.2f}%'
            })

    return {
        'loss': total_loss / num_batches,
        'accuracy': total_correct / total_tokens,
    }


def main():
    parser = argparse.ArgumentParser(description="Train H-Net on cs182 data")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to data directory (e.g., ../data/t-C_vs-16_sl-32_...)",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="configs/hnet_1stage_L.json",
        help="Path to model configuration (.json file)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save checkpoints and logs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of epochs to train",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.1,
        help="Weight decay",
    )
    parser.add_argument(
        "--aux-loss-weight",
        type=float,
        default=0.01,
        help="Weight for auxiliary load balancing loss",
    )
    parser.add_argument(
        "--lr-multipliers",
        type=str,
        default=None,
        help="Comma-separated learning rate multipliers for each stage (e.g., '3.0,1.7,0.9')",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "bfloat16", "float16"],
        help="Data type for model parameters",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="Evaluate every N epochs",
    )

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set dtype
    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    dtype = dtype_map[args.dtype]
    print(f"Using dtype: {args.dtype}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    with open(output_dir / "train_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = SequenceDataset(args.data_dir, split="train")
    test_dataset = SequenceDataset(args.data_dir, split="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Create model
    print("\nCreating model...")
    model, config = create_model_from_config(args.config_path, device, dtype)

    # Apply learning rate multipliers if specified
    if args.lr_multipliers:
        lr_mults = [float(x) for x in args.lr_multipliers.split(",")]
        print(f"Applying learning rate multipliers: {lr_mults}")
        model.apply_lr_multiplier(lr_mults)

    # Create optimizer with parameter groups
    param_groups = group_params(model)

    # Apply base learning rate and weight decay to all groups
    for group in param_groups:
        if "lr_multiplier" in group:
            group["lr"] = args.lr * group["lr_multiplier"]
        else:
            group["lr"] = args.lr

        if "weight_decay" not in group:
            group["weight_decay"] = args.weight_decay

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {num_trainable_params:,}")

    # Training loop
    print("\nStarting training...")
    best_eval_loss = float('inf')

    for epoch in range(1, args.num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.num_epochs}")
        print(f"{'='*60}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, config, args.aux_loss_weight
        )

        print(f"\nTraining metrics:")
        print(f"  Loss: {train_metrics['loss']:.4f}")
        print(f"  CE Loss: {train_metrics['ce_loss']:.4f}")
        print(f"  Aux Loss: {train_metrics['aux_loss']:.4f}")

        # Evaluate
        if epoch % args.eval_every == 0:
            eval_metrics = evaluate(model, test_loader, device, config)
            print(f"\nEvaluation metrics:")
            print(f"  Loss: {eval_metrics['loss']:.4f}")
            print(f"  Accuracy: {eval_metrics['accuracy']*100:.2f}%")

            # Save best model
            if eval_metrics['loss'] < best_eval_loss:
                best_eval_loss = eval_metrics['loss']
                torch.save(
                    model.state_dict(),
                    output_dir / "best_model.pt"
                )
                print(f"  Saved best model (loss: {best_eval_loss:.4f})")

        # Save checkpoint
        if epoch % args.save_every == 0:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_metrics['loss'],
                },
                output_dir / f"checkpoint_epoch_{epoch}.pt"
            )
            print(f"  Saved checkpoint")

    # Save final model
    torch.save(model.state_dict(), output_dir / "final_model.pt")
    print(f"\nTraining complete! Models saved to {output_dir}")


if __name__ == "__main__":
    main()
