"""Aggregate test results across seeds, produce result figures and export the final model.

Run after ``evaluate``. The final model is the single run with the best validation accuracy,
so the test set is never used for model selection.
"""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from food_classifier.data import CLASSES

CONDITIONS = ("scratch-plain", "pretrained-plain", "scratch-augmented", "pretrained-augmented")
SUMMARY_METRICS = ["accuracy", "macro_f1", "similar_group_error_share"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--export", type=Path, default=Path("models/resnet18_food12.pt"))
    args = parser.parse_args()
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    runs = pd.DataFrame(
        {**json.loads(path.read_text()), "run_dir": path.parent}
        for path in sorted(args.runs_dir.glob("*/seed*/metrics.json"))
    )
    if runs.empty:
        raise SystemExit(f"No evaluated runs under {args.runs_dir}; run food_classifier.evaluate")
    runs.to_csv(args.reports_dir / "runs.csv", index=False)
    conditions = [condition for condition in CONDITIONS if condition in set(runs.condition)]

    # Mean and standard deviation over seeds.
    summary = runs.groupby("condition")[SUMMARY_METRICS].agg(["mean", "std"]).reindex(conditions)
    summary.to_csv(args.reports_dir / "summary.csv")
    print(summary.round(4).to_string())

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(conditions))
    width = 0.38
    for offset, metric in ((-width / 2, "accuracy"), (width / 2, "macro_f1")):
        ax.bar(
            x + offset,
            summary[(metric, "mean")],
            width,
            yerr=summary[(metric, "std")].fillna(0),
            capsize=4,
            label=metric.replace("_", " "),
        )
    ax.set_xticks(x, conditions)
    ax.set_ylim(0, 1)
    ax.set_ylabel("test score, mean ± std over seeds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.reports_dir / "test_metrics.png", dpi=150)
    plt.close(fig)

    # Per-class precision, recall and F1, averaged over seeds.
    per_class = pd.concat(
        pd.read_csv(run.run_dir / "per_class_metrics.csv").assign(condition=run.condition)
        for run in runs.itertuples()
    )
    per_class.pivot_table(
        index="class", columns="condition", values=["precision", "recall", "f1"]
    ).reindex(CLASSES).reindex(columns=conditions, level=1).to_csv(
        args.reports_dir / "per_class_metrics.csv"
    )

    # Most common confusion pairs, summed over seeds.
    pair_counts = {}
    for condition, group in runs.groupby("condition"):
        total = sum(
            pd.read_csv(directory / "confusion_matrix.csv", index_col=0)
            for directory in group.run_dir
        )
        pairs = total.stack()
        pair_counts[condition] = pairs[[true != predicted for true, predicted in pairs.index]]
    confusion_pairs = pd.DataFrame(pair_counts)[conditions]
    confusion_pairs.index.names = ["true", "predicted"]
    top_pairs = confusion_pairs.sum(axis=1).sort_values(ascending=False).index[:15]
    confusion_pairs.loc[top_pairs].to_csv(args.reports_dir / "confusion_pairs.csv")

    # Final model.
    final = runs.loc[runs.val_accuracy.idxmax()]
    final_dir = final.run_dir
    args.export.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(final_dir / "best.pt", args.export)
    print(
        f"\nFinal model: {final.condition} seed {final.seed} "
        f"(val accuracy {final.val_accuracy:.4f}, test accuracy {final.accuracy:.4f}) "
        f"-> {args.export}"
    )

    confusion = pd.read_csv(final_dir / "confusion_matrix.csv", index_col=0)
    normalized = confusion.div(confusion.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(
                j,
                i,
                confusion.iat[i, j],
                ha="center",
                va="center",
                fontsize=8,
                color="white" if normalized.iat[i, j] > 0.5 else "black",
            )
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Test confusion matrix: {final.condition}, seed {final.seed}")
    fig.colorbar(image, ax=ax, fraction=0.046, label="share of true class")
    fig.tight_layout()
    fig.savefig(args.reports_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Most confident mistakes of the final model, as candidates for manual failure analysis.
    errors = pd.read_csv(final_dir / "misclassified.csv").head(12)
    fig, axes = plt.subplots(3, 4, figsize=(12, 9.5))
    for ax in axes.flat:
        ax.axis("off")
    for ax, error in zip(axes.flat, errors.itertuples(), strict=False):
        ax.imshow(Image.open(args.data_dir / "images" / f"{error.image_id}.jpg"))
        ax.set_title(f"{error.true} → {error.predicted}\n{error.confidence:.0%}", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.reports_dir / "misclassified_examples.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
