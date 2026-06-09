"""
Reusable matplotlib plotting functions for streaming machine learning metrics. This is usable across scripts, demo notebooks, and pipeline logs.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_metric_over_time(
    metric_values,
    title: str = "Metric over Time",
    ylabel: str = "Metric",
    xlabel: str = "Chunk",
    color: str = "steelblue",
    marker: str = "o",
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot a single metric across streaming chunks.
    Parameters:
    metric_values : array-like, shape (n_chunks,) where metric values recorded after each chunk.
    title : str
        Plot title.
    ylabel : str
        Y-axis label (e.g. 'Accuracy', 'F1').
    xlabel : str
        X-axis label (default 'Chunk').
    color : str, Line colour.
    marker : str, marker style for data points.
    save_path : str or None
    If provided, saves the figure to this path instead of displaying it, displaying would be an option.
    show : bool
        Whether to call plt.show(), Set False when running inside a notebook to let the notebook render the figure inline.

    Returns matplotlib.figure.Figure as fig

    """
    values = np.asarray(metric_values, dtype=float)
    chunks = np.arange(1, len(values) + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(chunks, values, marker=marker, color=color, linewidth=2,
            markersize=6, label=ylabel)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(chunks)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=11)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def compare_models(
    metric1,
    metric2,
    labels=("Model 1", "Model 2"),
    title: str = "Model Comparison",
    ylabel: str = "Metric",
    xlabel: str = "Chunk",
    colors=("steelblue", "tomato"),
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Compare two models on a streaming metric across chunks.
    Parameters:
    metric1 : array-like, shape (n_chunks,), metric values for the first model.
    metric2 : array-like, shape (n_chunks,), metric values for the second model.
    labels : tuple of str, legend labels for the two models.
    title : str, Plot title.
    ylabel : str, Y-axis label.
    xlabel : str, X-axis label.
    colors : tuple of str, Line colours for the two models.
    save_path : str or None
        If provided, saves the figure to this path.
    show : bool
        Whether to call plt.show().

    Returns matplotlib.figure.Figure as fig
    """
    v1 = np.asarray(metric1, dtype=float)
    v2 = np.asarray(metric2, dtype=float)

    if len(v1) != len(v2):
        raise ValueError(
            f"metric1 and metric2 must have the same length, "
            f"got {len(v1)} and {len(v2)}.")

    chunks = np.arange(1, len(v1) + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(chunks, v1, marker="o", color=colors[0], linewidth=2,
            markersize=6, label=labels[0])
    ax.plot(chunks, v2, marker="s", color=colors[1], linewidth=2,
            markersize=6, label=labels[1])
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(chunks)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=11)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def plot_predictions_vs_ground_truth(
    y_true,
    y_pred,
    title: str = "Predictions vs Ground Truth",
    xlabel: str = "Sample Index",
    ylabel: str = "Class Label",
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Visualise predicted labels against ground truth for the latest chunk, correct predictions are shown in green, incorrect ones in red.

    Parameters:
    y_true : array-like, shape (n_samples,), Ground truth class labels.
    y_pred : array-like, shape (n_samples,), Predicted class labels.
    title : str, Plot title.
    xlabel : str, X-axis label.
    ylabel : str, Y-axis label.
    save_path : str or None
        If provided, saves the figure to this path.
    show : bool
        Whether to call plt.show().

    Returns matplotlib.figure.Figure as fig
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"y_true and y_pred must have the same length, "
            f"got {y_true.shape[0]} and {y_pred.shape[0]}.")

    n = len(y_true)
    indices = np.arange(n)
    correct = y_true == y_pred

    fig, ax = plt.subplots(figsize=(max(8, n // 2), 4))

    ax.scatter(indices[correct], y_true[correct], color="green",
               label="Correct", zorder=3, s=60, marker="o")
    ax.scatter(indices[~correct], y_true[~correct], color="red",
               label="True (incorrect)", zorder=3, s=60, marker="o",
               facecolors="none", linewidths=1.5)
    ax.scatter(indices[~correct], y_pred[~correct], color="red",
               label="Predicted (incorrect)", zorder=3, s=60, marker="x",
               linewidths=1.5)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=10)
    fig.tight_layout()

    accuracy = float(np.mean(correct))
    ax.set_title(f"{title}  (accuracy={accuracy:.2%})",
                 fontsize=13, fontweight="bold")

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    labels=None,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Display a confusion matrix as a heatmap.
    Parameters:
    cm : np.ndarray, shape (n_classes, n_classes), Confusion matrix (rows = true, columns = predicted).
    labels : list of str or None, Class label names. Uses integer indices if None.
    title : str, Plot title.
    cmap : str, Matplotlib colormap name.
    save_path : str or None
        If provided, saves the figure to this path.
    show : bool
        Whether to call plt.show().

    Returns matplotlib.figure.Figure as fig
    """
    cm = np.asarray(cm, dtype=int)
    n = cm.shape[0]

    if labels is None:
        labels = [str(i) for i in range(n)]

    fig, ax = plt.subplots(figsize=(max(5, n), max(4, n)))
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    fig.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=12,
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig