"""Evaluate every trained run on the held-out test set.

For each ``<runs-dir>/<condition>/seed<N>/best.pt`` this writes, into the same directory:
``metrics.json``, ``per_class_metrics.csv``, ``confusion_matrix.csv``,
``test_predictions.csv`` and ``misclassified.csv``.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from food_classifier.data import CLASSES, SIMILAR_GROUPS, Food101Subset, eval_transform
from food_classifier.model import build_resnet18


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_set = Food101Subset(args.data_dir, "test", eval_transform)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers)
    y_true = test_set.labels
    labels = list(range(len(CLASSES)))
    group_of = {CLASSES.index(name): i for i, group in enumerate(SIMILAR_GROUPS) for name in group}

    checkpoints = sorted(args.runs_dir.glob("*/seed*/best.pt"))
    if not checkpoints:
        raise SystemExit(f"No checkpoints found under {args.runs_dir}")

    for checkpoint_path in checkpoints:
        run_dir = checkpoint_path.parent
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model = build_resnet18(pretrained=False)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device).eval()

        with torch.inference_mode():
            probabilities = torch.cat(
                [model(images.to(device)).softmax(1).cpu() for images, _ in test_loader]
            )
        confidence, y_pred = probabilities.max(1)
        y_pred = y_pred.tolist()

        confusion = confusion_matrix(y_true, y_pred, labels=labels)
        errors = confusion.sum() - confusion.trace()
        similar_group_errors = sum(
            confusion[i, j]
            for i in labels
            for j in labels
            if i != j and i in group_of and group_of[i] == group_of.get(j)
        )
        metrics = {
            "condition": checkpoint["condition"],
            "seed": checkpoint["seed"],
            "best_epoch": checkpoint["epoch"],
            "val_accuracy": checkpoint["val_accuracy"],
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro"),
            "similar_group_error_share": float(similar_group_errors / errors) if errors else 0.0,
        }
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0
        )
        pd.DataFrame(
            {
                "class": CLASSES,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        ).to_csv(run_dir / "per_class_metrics.csv", index=False)
        pd.DataFrame(confusion, index=CLASSES, columns=CLASSES).to_csv(
            run_dir / "confusion_matrix.csv"
        )

        predictions = pd.DataFrame(
            {
                "image_id": test_set.image_ids,
                "true": [CLASSES[i] for i in y_true],
                "predicted": [CLASSES[i] for i in y_pred],
                "confidence": confidence.numpy(),
            }
        )
        predictions.to_csv(run_dir / "test_predictions.csv", index=False)
        predictions[predictions.true != predictions.predicted].sort_values(
            "confidence", ascending=False
        ).to_csv(run_dir / "misclassified.csv", index=False)

        print(
            f"{metrics['condition']} seed {metrics['seed']}: "
            f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
