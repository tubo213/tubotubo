from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def visualize_confusion_matrix(
    y_true,
    pred_label,
    ax: Optional[plt.Axes] = None,
    labels: Optional[list] = None,
    conf_options: Optional[dict] = None,
    plot_options: Optional[dict] = None,
) -> Tuple[plt.Axes, np.ndarray]:
    """
    visualize confusion matrix
    Args:
        y_true:
            True Label. shape = (n_samples, )
        pred_label:
            Prediction Label. shape = (n_samples, )
        ax:
            matplotlib.pyplot.Axes object.
        labels:
            plot labels
        conf_options:
            option kwrgs when calculate confusion matrix.
            pass to `confusion_matrix` (defined at scikit-learn)
        plot_options:
            option key-words when plot seaborn heatmap
    Returns:
    """

    _conf_options = {
        "normalize": "true",
    }
    if conf_options is not None:
        _conf_options.update(conf_options)

    _plot_options = {"cmap": "Blues", "annot": True}
    if plot_options is not None:
        _plot_options.update(plot_options)

    conf = confusion_matrix(y_true=y_true, y_pred=pred_label, **_conf_options)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(conf, ax=ax, **_plot_options)
    ax.set_ylabel("Label")
    ax.set_xlabel("Predict")

    if labels is not None:
        ax.set_yticklabels(labels)
        ax.set_xticklabels(labels)
        ax.tick_params("y", labelrotation=0)
        ax.tick_params("x", labelrotation=90)

    return ax, conf


def check_y_and_pred(y_true, y_pred) -> Tuple[np.ndarray, np.ndarray, list]:
    y_true = np.asarray(y_true)

    le = LabelEncoder()
    y_true = le.fit_transform(y_true)

    classes = le.classes_
    if len(classes) == 2:
        classes = classes[:-1]
    n_classes = len(classes)

    ohe = OneHotEncoder()
    y_true = ohe.fit_transform(y_true.reshape(-1, 1)).toarray()
    y_true = y_true[:, -n_classes:]
    y_pred = np.asarray(y_pred).reshape(-1, n_classes)
    return y_true, y_pred, classes


def visualize_roc_auc_curve(
    y_true, y_pred, ax: Optional[plt.Axes] = None, label_prefix: Optional[str] = None
) -> Tuple[Optional[plt.Figure], plt.Axes]:
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_true, y_pred, classes = check_y_and_pred(y_true, y_pred)
    n_classes = len(classes)

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig, ax = None, ax

    for i in range(n_classes):
        label_i = f"label = {i} / area = {roc_auc[i]:.3f}"
        if label_prefix is not None:
            label_i = f"{label_prefix} {label_i}"
        ax.plot(fpr[i], tpr[i], label=label_i)
    ax.plot(np.linspace(0, 1), np.linspace(0, 1), "--", color="grey")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.grid()
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Auc Score")
    ax.legend(loc="lower right")
    return fig, ax


def visualize_pr_curve(
    y_true, y_pred, ax: Optional[plt.Axes] = None, label_prefix: Optional[str] = None
) -> Tuple[Optional[plt.Figure], plt.Axes]:
    precision = dict()
    recall = dict()
    pr_score = dict()
    y_true, y_pred, classes = check_y_and_pred(y_true, y_pred)
    n_classes = len(classes)

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
        pr_score[i] = average_precision_score(y_true[:, i], y_pred[:, i])

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig, ax = None, ax

    for i in range(n_classes):
        label_i = f"label = {i} / area = {pr_score[i]:.3f}"
        if label_prefix is not None:
            label_i = f"{label_prefix} {label_i}"
        ax.plot(recall[i], precision[i], label=label_i)

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.grid()
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision Recall Curve")
    ax.legend(loc="lower left")
    return fig, ax
