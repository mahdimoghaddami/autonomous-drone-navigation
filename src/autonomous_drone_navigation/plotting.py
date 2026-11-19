"""Plotting utilities for DroNet training loss, steering histograms, and confusion matrices."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import confusion_matrix

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_loss(experiment_rootdir: str | Path, show: bool = False) -> Path:
    """Plot training and validation loss curves from `log.txt` into `log.png`."""
    exp_dir = Path(experiment_rootdir)
    log_file = exp_dir / "log.txt"
    if not log_file.exists():
        raise FileNotFoundError(f"Log file not found: {log_file}")

    log = np.genfromtxt(log_file, delimiter="\t", dtype=None, names=True)
    train_loss = np.atleast_1d(log["train_loss"])
    val_loss = np.atleast_1d(log["val_loss"])
    timesteps = list(range(train_loss.shape[0]))

    fig, ax = plt.subplots()
    ax.plot(timesteps, train_loss, "r--", label="Training loss")
    ax.plot(timesteps, val_loss, "b--", label="Validation loss")
    ax.legend()
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epochs")

    out_path = exp_dir / "log.png"
    fig.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def make_and_save_histograms(
    pred_steerings: list[float] | np.ndarray,
    real_steerings: list[float] | np.ndarray,
    img_name: str | Path = "histograms.png",
) -> Path:
    """Plot and save histograms of predicted vs. real steering angles."""
    pred_arr = np.asarray(pred_steerings, dtype=np.float32)
    real_arr = np.asarray(real_steerings, dtype=np.float32)
    max_h = float(np.maximum(np.max(pred_arr), np.max(real_arr)))
    min_h = float(np.minimum(np.min(pred_arr), np.min(real_arr)))
    if np.isclose(min_h, max_h):
        min_h -= 0.1
        max_h += 0.1
    bins = np.linspace(min_h, max_h, num=50)

    fig, ax = plt.subplots()
    ax.hist(pred_arr, bins=bins, alpha=0.5, label="Predicted", color="b")
    ax.hist(real_arr, bins=bins, alpha=0.5, label="Real", color="r")
    ax.set_title("Steering angle")
    ax.set_xlabel("Scaled Steerings")
    ax.legend(fontsize=10)

    out_path = Path(img_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_confusion_matrix(
    real_labels: list[float] | np.ndarray,
    pred_prob: list[float] | np.ndarray,
    classes: list[str],
    normalize: bool = False,
    img_name: str | Path = "confusion.png",
) -> Path:
    """Plot and save confusion matrix for binary collision prediction."""
    real_arr = np.asarray(real_labels)
    prob_arr = np.asarray(pred_prob)
    pred_labels = (prob_arr >= 0.5).astype(int)

    cm = confusion_matrix(real_arr, pred_labels, labels=list(range(len(classes))))
    if normalize:
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        cm_display = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    else:
        cm_display = cm

    fig, ax = plt.subplots()
    im = ax.imshow(cm_display, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion matrix")
    fig.colorbar(im, ax=ax)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, rotation=90, va="center")

    thresh = cm_display.max() / 2.0 if cm_display.size else 0.5
    for i, j in itertools.product(range(cm_display.shape[0]), range(cm_display.shape[1])):
        val = f"{cm_display[i, j]:.2f}" if normalize else str(cm_display[i, j])
        ax.text(
            j,
            i,
            val,
            horizontalalignment="center",
            color="white" if cm_display[i, j] > thresh else "black",
        )

    fig.tight_layout()
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")

    out_path = Path(img_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_results(experiment_rootdir: str | Path, normalize: bool = True) -> dict[str, Path]:
    """Generate `histograms.png` and `confusion.png` from evaluation outputs."""
    exp_dir = Path(experiment_rootdir)
    outputs: dict[str, Path] = {}

    fname_steer = exp_dir / "predicted_and_real_steerings.json"
    if fname_steer.exists():
        with open(fname_steer, "r", encoding="utf-8") as f:
            dict_steerings = json.load(f)
        outputs["histograms"] = make_and_save_histograms(
            dict_steerings["pred_steerings"],
            dict_steerings["real_steerings"],
            img_name=exp_dir / "histograms.png",
        )

    fname_labels = exp_dir / "predicted_and_real_labels.json"
    if fname_labels.exists():
        with open(fname_labels, "r", encoding="utf-8") as f:
            dict_labels = json.load(f)
        outputs["confusion"] = plot_confusion_matrix(
            dict_labels["real_labels"],
            dict_labels["pred_probabilities"],
            ["no collision", "collision"],
            normalize=normalize,
            img_name=exp_dir / "confusion.png",
        )

    return outputs
