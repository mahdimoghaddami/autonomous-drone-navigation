"""Steering regression and collision classification evaluation metrics and runner."""

from __future__ import annotations

import glob
import os
from pathlib import Path
from random import randint

import numpy as np
from sklearn import metrics

from autonomous_drone_navigation.config import DroneConfig
from autonomous_drone_navigation.data import DroneDataGenerator, compute_predictions_and_gt, write_to_file
from autonomous_drone_navigation.model import json_to_model, load_model_weights


def explained_variance_1d(ypred: np.ndarray, y: np.ndarray) -> float:
    """Compute proportion of variance explained: 1 - Var[y - ypred] / Var[y]."""
    assert y.ndim == 1 and ypred.ndim == 1
    vary = float(np.var(y))
    return float(np.nan) if vary == 0.0 else float(1.0 - np.var(y - ypred) / vary)


def compute_explained_variance(predictions: np.ndarray, real_values: np.ndarray) -> float:
    """Compute and print explained variance for steering regression."""
    assert np.all(predictions.shape == real_values.shape)
    ex_variance = explained_variance_1d(predictions, real_values)
    print(f"EVA = {ex_variance}")
    return ex_variance


def compute_sq_residuals(predictions: np.ndarray, real_values: np.ndarray) -> np.ndarray:
    """Compute squared residuals between predicted and real steering values."""
    assert np.all(predictions.shape == real_values.shape)
    sq_res = np.square(predictions - real_values)
    sr = np.mean(sq_res, axis=-1)
    print(f"MSE = {sr}")
    return sq_res


def compute_rmse(predictions: np.ndarray, real_values: np.ndarray) -> float:
    """Compute and print Root-Mean-Squared Error (RMSE)."""
    assert np.all(predictions.shape == real_values.shape)
    mse = np.mean(np.square(predictions - real_values))
    rmse = float(np.sqrt(mse))
    print(f"RMSE = {rmse}")
    return rmse


def compute_highest_regression_errors(
    predictions: np.ndarray, real_values: np.ndarray, n_errors: int = 20
) -> np.ndarray:
    """Return indices of the top `n_errors` highest squared-error predictions."""
    assert np.all(predictions.shape == real_values.shape)
    sq_res = np.square(predictions - real_values)
    return sq_res.argsort()[-n_errors:][::-1]


def random_regression_baseline(real_values: np.ndarray) -> np.ndarray:
    """Sample Gaussian steering baseline matching ground-truth mean and std."""
    mean = float(np.mean(real_values))
    std = float(np.std(real_values))
    if std == 0.0:
        return np.full_like(real_values, mean, dtype=np.float32)
    return np.random.normal(loc=mean, scale=abs(std), size=real_values.shape)


def constant_baseline(real_values: np.ndarray) -> np.ndarray:
    """Constant mean steering baseline."""
    mean = np.mean(real_values)
    return mean * np.ones_like(real_values)


def evaluate_regression(
    predictions: np.ndarray, real_values: np.ndarray, fname: str | Path
) -> dict:
    """Evaluate steering regression and write metrics JSON."""
    evas = compute_explained_variance(predictions, real_values)
    rmse = compute_rmse(predictions, real_values)
    highest_errors = compute_highest_regression_errors(predictions, real_values, n_errors=20)
    dictionary = {
        "evas": float(evas) if not np.isnan(evas) else None,
        "rmse": float(rmse),
        "highest_errors": highest_errors.tolist(),
    }
    write_to_file(dictionary, fname)
    return dictionary


def read_training_labels(file_name: str | Path) -> np.ndarray:
    """Load collision labels from a single `labels.txt` file."""
    try:
        labels = np.loadtxt(file_name, usecols=0)
        return np.atleast_1d(np.array(labels))
    except Exception:
        print(f"File {file_name} failed loading labels")
        return np.array([])


def count_samples_per_class(train_dir: str | Path) -> np.ndarray:
    """Count class 0 (no collision) and class 1 (collision) samples in `train_dir`."""
    experiments = glob.glob(os.path.join(str(train_dir), "*"))
    num_class0 = 0
    num_class1 = 0
    for exp in experiments:
        file_name = os.path.join(exp, "labels.txt")
        try:
            labels = np.atleast_1d(np.loadtxt(file_name, usecols=0))
            num_class1 += int(np.sum(labels == 1))
            num_class0 += int(np.sum(labels == 0))
        except Exception:
            continue
    return np.array([num_class0, num_class1])


def random_classification_baseline(real_values: np.ndarray) -> np.ndarray:
    """Random 50/50 binary classification baseline."""
    return np.array([randint(0, 1) for _ in range(real_values.shape[0])], dtype=np.float32)


def weighted_baseline(real_values: np.ndarray, samples_per_class: np.ndarray) -> np.ndarray:
    """Class-frequency weighted random binary classification baseline."""
    weights = samples_per_class / np.sum(samples_per_class)
    return np.random.choice(2, real_values.shape[0], p=weights)


def majority_class_baseline(real_values: np.ndarray, samples_per_class: np.ndarray) -> list[int]:
    """Always-predict-majority-class baseline."""
    major_class = int(np.argmax(samples_per_class))
    return [major_class for _ in real_values]


def compute_highest_classification_errors(
    predictions: np.ndarray, real_values: np.ndarray, n_errors: int = 20
) -> np.ndarray:
    """Return indices of the top `n_errors` largest probability errors."""
    assert np.all(predictions.shape == real_values.shape)
    dist = np.abs(predictions - real_values)
    return dist.argsort()[-n_errors:][::-1]


def evaluate_classification(
    pred_prob: np.ndarray,
    pred_labels: np.ndarray,
    real_labels: np.ndarray,
    fname: str | Path,
) -> dict:
    """Evaluate collision classification (accuracy, precision, recall, F1) and save JSON."""
    ave_accuracy = float(metrics.accuracy_score(real_labels, pred_labels))
    print("Average accuracy = ", ave_accuracy)
    precision = float(metrics.precision_score(real_labels, pred_labels, zero_division=0))
    print("Precision = ", precision)
    recall = float(metrics.recall_score(real_labels, pred_labels, zero_division=0))
    print("Recall = ", recall)
    f_score = float(metrics.f1_score(real_labels, pred_labels, zero_division=0))
    print("F1-score = ", f_score)
    highest_errors = compute_highest_classification_errors(pred_prob, real_labels, n_errors=20)
    dictionary = {
        "ave_accuracy": ave_accuracy,
        "precision": precision,
        "recall": recall,
        "f_score": f_score,
        "highest_errors": highest_errors.tolist(),
    }
    write_to_file(dictionary, fname)
    return dictionary


def run_evaluation(config: DroneConfig) -> dict[str, dict]:
    """Run steering and collision evaluation on `config.test_dir` and save JSON outputs."""
    exp_dir = Path(config.experiment_rootdir)
    test_datagen = DroneDataGenerator(rescale=1.0 / 255.0)
    test_generator = test_datagen.flow_from_directory(
        config.test_dir,
        shuffle=False,
        color_mode=config.img_mode,
        target_size=(config.img_width, config.img_height),
        crop_size=(config.crop_img_height, config.crop_img_width),
        batch_size=config.batch_size,
    )

    json_model_path = exp_dir / config.json_model_fname
    model = json_to_model(json_model_path)

    weights_load_path = exp_dir / config.weights_fname
    if weights_load_path.exists():
        load_model_weights(model, weights_load_path)
        print(f"Loaded model from {weights_load_path}")
    else:
        print("Impossible to find weight path. Returning untrained model")

    model.compile(loss="mse", optimizer="adam")

    nb_batches = len(test_generator)
    predictions, ground_truth, t = compute_predictions_and_gt(
        model, test_generator, nb_batches, verbose=1
    )

    results: dict[str, dict] = {}
    t_mask = t == 1

    if np.any(t_mask):
        pred_steerings = predictions[t_mask, 0]
        real_steerings = ground_truth[t_mask, 0]

        dict_fname = {
            "test_regression.json": pred_steerings,
            "random_regression.json": random_regression_baseline(real_steerings),
            "constant_regression.json": constant_baseline(real_steerings),
        }
        for fname, pred in dict_fname.items():
            results[fname] = evaluate_regression(pred, real_steerings, exp_dir / fname)

        write_to_file(
            {
                "pred_steerings": pred_steerings.tolist(),
                "real_steerings": real_steerings.tolist(),
            },
            exp_dir / "predicted_and_real_steerings.json",
        )

    if np.any(~t_mask):
        pred_prob = predictions[~t_mask, 1]
        pred_labels = (pred_prob >= 0.5).astype(np.float32)
        real_labels = ground_truth[~t_mask, 1]

        dict_fname = {
            "test_classification.json": pred_labels,
            "random_classification.json": random_classification_baseline(real_labels),
        }
        for fname, pred in dict_fname.items():
            results[fname] = evaluate_classification(pred_prob, pred, real_labels, exp_dir / fname)

        write_to_file(
            {
                "pred_probabilities": pred_prob.tolist(),
                "real_labels": real_labels.tolist(),
            },
            exp_dir / "predicted_and_real_labels.json",
        )

    return results
