"""Evaluation metrics.

Reports accuracy plus the macro-averaged false positive and false negative
rates used in Table 3 of the paper.  Both are computed one-vs-rest from the
multiclass confusion matrix and then averaged over classes, so every class
counts equally regardless of support.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["ClassificationReport", "evaluate"]


@dataclass
class ClassificationReport:
    accuracy: float
    macro_fpr: float
    macro_fnr: float
    per_class_fpr: list[float] = field(default_factory=list)
    per_class_fnr: list[float] = field(default_factory=list)
    confusion_matrix: list[list[int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "macro_fpr": self.macro_fpr,
            "macro_fnr": self.macro_fnr,
            "per_class_fpr": self.per_class_fpr,
            "per_class_fnr": self.per_class_fnr,
            "confusion_matrix": self.confusion_matrix,
        }

    def __str__(self) -> str:
        return (
            f"accuracy={self.accuracy:.4f}  "
            f"macro FPR={self.macro_fpr:.4f}  "
            f"macro FNR={self.macro_fnr:.4f}"
        )


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (y_true.astype(int), y_pred.astype(int)), 1)
    return matrix


def evaluate(y_true, y_pred, n_classes: int) -> ClassificationReport:
    """Compute accuracy and one-vs-rest error rates."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")

    matrix = confusion_matrix(y_true, y_pred, n_classes)
    total = matrix.sum()

    fpr_per_class, fnr_per_class = [], []
    for i in range(n_classes):
        tp = matrix[i, i]
        fn = matrix[i, :].sum() - tp
        fp = matrix[:, i].sum() - tp
        tn = total - (tp + fn + fp)
        fnr_per_class.append(float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0)
        fpr_per_class.append(float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0)

    return ClassificationReport(
        accuracy=float(np.trace(matrix) / total) if total else 0.0,
        macro_fpr=float(np.mean(fpr_per_class)),
        macro_fnr=float(np.mean(fnr_per_class)),
        per_class_fpr=fpr_per_class,
        per_class_fnr=fnr_per_class,
        confusion_matrix=matrix.tolist(),
    )
