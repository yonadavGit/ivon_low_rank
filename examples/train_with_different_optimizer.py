"""
cifar10_train_ivon.py

Train CIFAR-10 with one of: AdamW / IVON / IVONLR (from your uploaded _ivon.py).

Usage examples:
  # AdamW baseline
  python cifar10_train_ivon.py --opt adamw --epochs 20 --batch-size 128

  # IVON
  python cifar10_train_ivon.py --opt ivon --epochs 20 --batch-size 128

  # IVONLR (rank sweep)
  python cifar10_train_ivon.py --opt ivonlr --rank 0 --epochs 20
  python cifar10_train_ivon.py --opt ivonlr --rank 2 --epochs 20

Notes:
- This script downloads CIFAR-10 via torchvision if not present.
- It assumes _ivon.py is in the same folder as this script (your upload).
- IVON/IVONLR use optimizer.sampled_params(...) during training and (optionally) MC evaluation.
"""

import argparse
import os
import time
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T

# Import your IVON implementation (uploaded as /mnt/data/_ivon.py)
# Put this script in the same directory as _ivon.py, or adjust sys.path.
import ivon


def get_loaders(data_dir: str, batch_size: int, num_workers: int):
    # Common CIFAR-10 augmentation
    train_tf = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    test_tf = T.Compose(
        [
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    train_ds = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_tf
    )
    test_ds = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_tf
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, test_loader


def build_model(name: str, num_classes: int = 10):
    name = name.lower()
    if name == "resnet18":
        model = torchvision.models.resnet18(num_classes=num_classes)
        # For CIFAR-10, ResNet18 expects 224 by default but works on 32 too; to be nicer:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return model
    elif name == "cnn":
        # Small CNN baseline
        return nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(8 * 8 * 128, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )
    else:
        raise ValueError(f"Unknown model: {name}")


def accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


@torch.no_grad()
def evaluate(model, loader, device, optimizer=None, mc_samples: int = 1):
    """
    If optimizer is provided and mc_samples>1, we do MC evaluation:
      average predictive probabilities over sampled weights.
    Otherwise, deterministic eval.
    """
    model.eval()
    total_correct = 0
    total = 0
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        if optimizer is not None and mc_samples > 1:
            # MC: average probs across sampled parameter draws
            probs_sum = None
            for _ in range(mc_samples):
                with optimizer.sampled_params(train=False):
                    logits = model(x)
                    probs = F.softmax(logits, dim=1)
                probs_sum = probs if probs_sum is None else (probs_sum + probs)
            probs_mean = probs_sum / mc_samples
            loss = F.nll_loss(torch.log(probs_mean.clamp_min(1e-12)), y)
            preds = probs_mean.argmax(dim=1)
        else:
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            preds = logits.argmax(dim=1)

        total_loss += loss.item() * x.size(0)
        total_correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / total, total_correct / total


def make_optimizer(opt_name: str, model: nn.Module, args) -> tuple[object, object]:
    """
    Returns (optimizer, scheduler_or_None)
    """
    opt_name = opt_name.lower()

    if opt_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        return optimizer, scheduler

    if opt_name == "ivon":
        # Your IVON signature may differ; adjust if needed.
        optimizer = ivon.IVON(
            model.parameters(),
            lr=args.lr,
            ess=args.ess,
            beta1=args.beta1,
            beta2=args.beta2,
            weight_decay=args.weight_decay,
            hess_init=args.hess_init,
            mc_samples=args.mc_train,
        )
        scheduler = None
        return optimizer, scheduler

    if opt_name == "ivonlr":
        optimizer = ivon.IVONLR(
            model.parameters(),
            lr=args.lr,
            ess=args.ess,
            beta1=args.beta1,
            beta2=args.beta2,
            weight_decay=args.weight_decay,
            hess_init=args.hess_init,
            mc_samples=args.mc_train,
            rank=args.rank,
            low_rank_init=args.low_rank_init,
        )
        scheduler = None
        return optimizer, scheduler

    raise ValueError(f"Unknown optimizer: {opt_name}")


def train_one_epoch(model, loader, device, optimizer, opt_name: str, grad_clip: float | None):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # For IVON/IVONLR, sample parameters during forward/backward
        if opt_name in ("ivon", "ivonlr"):
            ctx = optimizer.sampled_params(train=True)
        else:
            ctx = nullcontext()

        with ctx:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()

            if grad_clip is not None and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, y) * bs
        n += bs

    return total_loss / n, total_acc / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "cnn"])
    p.add_argument("--opt", type=str, default="adamw", choices=["adamw", "ivon", "ivonlr"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Common optim params
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--grad-clip", type=float, default=0.0)

    # IVON params (tune as needed)
    p.add_argument("--ess", type=float, default=50_000.0)        # effective sample size
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.9999)
    p.add_argument("--hess-init", type=float, default=0.1)
    p.add_argument("--mc-train", type=int, default=1, help="MC samples per step inside IVON/IVONLR")

    # IVONLR params
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--low-rank-init", type=float, default=1e-2)

    # Eval MC
    p.add_argument("--mc-eval", type=int, default=1, help="MC samples at eval for IVON/IVONLR")

    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size, args.num_workers)
    model = build_model(args.model).to(device)

    optimizer, scheduler = make_optimizer(args.opt, model, args)

    print(f"Device: {device}")
    print(f"Model: {args.model}, Optimizer: {args.opt}")
    if args.opt == "ivonlr":
        print(f"  rank={args.rank}, low_rank_init={args.low_rank_init}")
    if args.opt in ("ivon", "ivonlr"):
        print(f"  ess={args.ess}, hess_init={args.hess_init}, mc_train={args.mc_train}, mc_eval={args.mc_eval}")

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model,
            train_loader,
            device,
            optimizer,
            opt_name=args.opt,
            grad_clip=args.grad_clip if args.grad_clip > 0 else None,
        )

        if scheduler is not None:
            scheduler.step()

        # If IVON/IVONLR, optionally do MC eval using sampled params
        opt_for_eval = optimizer if args.opt in ("ivon", "ivonlr") else None
        te_loss, te_acc = evaluate(model, test_loader, device, optimizer=opt_for_eval, mc_samples=args.mc_eval)

        best_acc = max(best_acc, te_acc)
        dt = time.time() - t0

        lr = optimizer.param_groups[0]["lr"] if hasattr(optimizer, "param_groups") else args.lr
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"lr={lr:.3g} | "
            f"train loss={tr_loss:.4f} acc={tr_acc*100:.2f}% | "
            f"test loss={te_loss:.4f} acc={te_acc*100:.2f}% | "
            f"best={best_acc*100:.2f}% | "
            f"time={dt:.1f}s"
        )


if __name__ == "__main__":
    main()
