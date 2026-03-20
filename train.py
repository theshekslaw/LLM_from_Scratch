"""
Training script for the LLM.

Usage:
    python train.py                          # Train small model on The Verdict
    python train.py --epochs 20 --lr 5e-4    # Custom hyperparameters
    python train.py --checkpoint out/model.pt # Resume from checkpoint
"""

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

from tokenizer import BPETokenizer
from transformer import GPTModel, ModelConfig, generate
from utils import create_dataloaders, download_verdict


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cosine_lr(optimizer, step: int, warmup_steps: int, total_steps: int, max_lr: float, min_lr: float):
    """Apply cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        lr = max_lr * (step + 1) / warmup_steps
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


@torch.no_grad()
def evaluate(model, val_loader, device):
    """Compute average validation loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    loss_fn = nn.CrossEntropyLoss()

    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        total_loss += loss.item()
        num_batches += 1

    model.train()
    return total_loss / max(num_batches, 1)


def train(args):
    device = get_device()
    print(f"Using device: {device}")

    # ── Data ──
    tokenizer = BPETokenizer("gpt2")
    text = download_verdict()
    print(f"Training text: {len(text):,} characters")

    config = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(),
        embed_dim=args.embed_dim,
        max_seq_len=args.max_seq_len,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        drop_rate=args.dropout,
        qkv_bias=True,
        pos_encoding="learned",
        norm_type="layernorm",
        activation="gelu",
        ffn_type="standard",
    )

    train_loader, val_loader = create_dataloaders(
        text, tokenizer,
        max_seq_len=config.max_seq_len,
        stride=config.max_seq_len,
        batch_size=args.batch_size,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ── Model ──
    model = GPTModel(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)
    min_lr = args.lr * 0.1

    # ── Resume from checkpoint ──
    start_epoch = 0
    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from checkpoint at epoch {start_epoch}")

    # ── Training loop ──
    train_losses = []
    val_losses = []
    global_step = start_epoch * len(train_loader)

    print(f"\nTraining for {args.epochs} epochs ({total_steps} steps)...\n")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            lr = cosine_lr(optimizer, global_step, warmup_steps, total_steps, args.lr, min_lr)

            logits = model(inputs)
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_loss = evaluate(model, val_loader, device)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"Train loss: {avg_train_loss:.4f} | "
            f"Val loss: {avg_val_loss:.4f} | "
            f"LR: {lr:.2e} | "
            f"Time: {elapsed:.1f}s"
        )

    # ── Save checkpoint ──
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "model.pt"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "epoch": args.epochs - 1,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }, ckpt_path)
    print(f"\nCheckpoint saved to {ckpt_path}")

    # ── Generate sample ──
    print("\n--- Sample generation ---")
    prompt = "Every effort moves you"
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    output_ids = generate(model, prompt_ids, max_new_tokens=50, temperature=0.8, top_k=40)
    print(tokenizer.decode(output_ids[0].tolist()))

    return train_losses, val_losses


def main():
    parser = argparse.ArgumentParser(description="Train LLM from scratch")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="out")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
