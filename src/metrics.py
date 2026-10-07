from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
from src.data_split import CLASSES

def compute_metrics(y_true, y_pred, class_names=CLASSES):
    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro', labels=labels),
        'precision_per_class': dict(zip(class_names, precision)),
        'recall_per_class':    dict(zip(class_names, recall)),
        'f1_per_class':        dict(zip(class_names, f1)),
        'support':             dict(zip(class_names, support)),
        'confusion_matrix':    confusion_matrix(y_true, y_pred, labels=labels),
        'report': classification_report(y_true, y_pred, labels=labels,
                                        target_names=class_names, zero_division=0,
                                        output_dict=True),
    }

def plot_confusion_matrix(cm, class_names=CLASSES, normalize=False, save_path=None, title=None):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)
        fmt = '.2f'
        vmin, vmax = 0.0, 1.0
    else:
        fmt = 'd'
        vmin, vmax = None, None

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap='Blues', vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title or ('Confusion Matrix' + (' (normalized)' if normalize else '')))

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=8)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)