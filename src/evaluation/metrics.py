"""Evaluation metrics for fake news detection."""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, List


def calibration_error(y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE).

    ECE measures how well predicted probabilities match actual outcomes.

    Args:
        y_true: True labels [N]
        y_probs: Predicted probabilities [N, num_classes]
        n_bins: Number of bins for calibration

    Returns:
        ECE score (lower is better)
    """
    # Get predicted class and confidence
    y_pred_class = np.argmax(y_probs, axis=1)
    y_pred_conf = np.max(y_probs, axis=1)

    # Create bins
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred_conf, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    for bin_idx in range(n_bins):
        # Get samples in this bin
        mask = bin_indices == bin_idx
        if np.sum(mask) == 0:
            continue

        # Accuracy in this bin
        bin_acc = np.mean(y_true[mask] == y_pred_class[mask])

        # Average confidence in this bin
        bin_conf = np.mean(y_pred_conf[mask])

        # Weighted difference
        bin_weight = np.sum(mask) / len(y_true)
        ece += bin_weight * np.abs(bin_acc - bin_conf)

    return ece


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_probs: List[List[float]]
) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_probs: Predicted probabilities

    Returns:
        Dictionary of metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    metrics = {}

    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')

    # ROC-AUC (one-vs-rest for multiclass)
    try:
        metrics['roc_auc_ovr'] = roc_auc_score(
            y_true,
            y_probs,
            multi_class='ovr',
            average='macro'
        )
    except ValueError:
        # Not all classes may be present
        metrics['roc_auc_ovr'] = 0.0

    # Calibration error
    metrics['calibration_error'] = calibration_error(y_true, y_probs)

    # Per-class F1 scores
    f1_per_class = f1_score(y_true, y_pred, average=None)
    metrics['f1_human'] = f1_per_class[0] if len(f1_per_class) > 0 else 0.0
    metrics['f1_ai'] = f1_per_class[1] if len(f1_per_class) > 1 else 0.0
    metrics['f1_inconclusive'] = f1_per_class[2] if len(f1_per_class) > 2 else 0.0

    return metrics


def print_evaluation_report(
    y_true: List[int],
    y_pred: List[int],
    y_probs: List[List[float]],
    class_names: List[str] = ['Human', 'AI', 'Inconclusive']
):
    """Print comprehensive evaluation report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_probs: Predicted probabilities
        class_names: Names of classes
    """
    print("=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_probs)

    print("\nOverall Metrics:")
    print(f"  Accuracy:           {metrics['accuracy']:.4f}")
    print(f"  F1 (macro):         {metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted):      {metrics['f1_weighted']:.4f}")
    print(f"  Precision (macro):  {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):     {metrics['recall_macro']:.4f}")
    print(f"  ROC-AUC (OvR):      {metrics['roc_auc_ovr']:.4f}")
    print(f"  Calibration Error:  {metrics['calibration_error']:.4f}")

    print("\nPer-Class F1 Scores:")
    print(f"  Human:              {metrics['f1_human']:.4f}")
    print(f"  AI:                 {metrics['f1_ai']:.4f}")
    print(f"  Inconclusive:       {metrics['f1_inconclusive']:.4f}")

    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(f"{'':12s} " + " ".join(f"{name:12s}" for name in class_names))
    for i, row in enumerate(cm):
        print(f"{class_names[i]:12s} " + " ".join(f"{val:12d}" for val in row))

    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    print("=" * 60)
