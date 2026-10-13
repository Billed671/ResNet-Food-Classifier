"""Train ResNet-18 on the Food-101 subset for one experimental condition and seed.

The best epoch by validation accuracy is saved to ``<runs-dir>/<condition>/seed<N>/best.pt``
next to ``config.json`` and ``history.csv``.
"""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from food_classifier.data import CLASSES, Food101Subset, eval_transform, train_augmentation
from food_classifier.model import build_resnet18


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", action="store_true", help="start from ImageNet weights")
    parser.add_argument("--augment", action="store_true", help="augment training images")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    condition = "-".join(
        [
            "pretrained" if args.pretrained else "scratch",
            "augmented" if args.augment else "plain",
        ]
    )
    run_dir = args.runs_dir / condition / f"seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "condition": condition}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))

    train_set = Food101Subset(
        args.data_dir, "train", train_augmentation if args.augment else eval_transform
    )
    val_set = Food101Subset(args.data_dir, "val", eval_transform)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers)

    model = build_resnet18(args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_accuracy = 0.0
    with open(run_dir / "history.csv", "w", newline="") as history_file:
        history = csv.writer(history_file)
        history.writerow(["epoch", "lr", "train_loss", "val_loss", "val_accuracy"])

        for epoch in range(1, args.epochs + 1):
            lr = scheduler.get_last_lr()[0]
            model.train()
            train_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(labels)
            scheduler.step()

            model.eval()
            val_loss, correct = 0.0, 0
            with torch.inference_mode():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    logits = model(images)
                    val_loss += criterion(logits, labels).item() * len(labels)
                    correct += (logits.argmax(1) == labels).sum().item()

            train_loss /= len(train_set)
            val_loss /= len(val_set)
            val_accuracy = correct / len(val_set)
            history.writerow([epoch, lr, train_loss, val_loss, val_accuracy])
            history_file.flush()
            print(
                f"[{condition} seed {args.seed}] epoch {epoch}/{args.epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
            )

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "classes": CLASSES,
                        "condition": condition,
                        "seed": args.seed,
                        "epoch": epoch,
                        "val_accuracy": val_accuracy,
                    },
                    run_dir / "best.pt",
                )


if __name__ == "__main__":
    main()
