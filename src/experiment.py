"""Reproducible experiments for the wine-quality course project.

The task is binary classification: quality >= 7 is treated as the positive
(high-quality) class. All preprocessing and threshold tuning are fitted inside
each outer training split.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FEATURE_NAMES_ZH = [
    "固定酸度",
    "挥发性酸度",
    "柠檬酸",
    "残糖",
    "氯化物",
    "游离二氧化硫",
    "总二氧化硫",
    "密度",
    "pH",
    "硫酸盐",
    "酒精度",
]


def load_wine_data(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle, delimiter=";"))
    header = rows[0]
    values = np.asarray([[float(item) for item in row] for row in rows[1:]], dtype=np.float64)
    quality = values[:, header.index("quality")].astype(int)
    features = values[:, :11]
    labels = (quality >= 7).astype(np.int64)
    return features, labels, quality


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Return average one-based ranks, including correct handling of ties."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks_sorted = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks_sorted[start:end] = 0.5 * (start + 1 + end)
        start = end
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = ranks_sorted
    return ranks


def roc_auc_score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    labels = labels.astype(int)
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = average_ranks(probabilities)
    positive_rank_sum = float(ranks[labels == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def precision_recall_curve(
    labels: np.ndarray, probabilities: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    labels = labels.astype(int)
    order = np.argsort(-probabilities, kind="mergesort")
    labels = labels[order]
    scores = probabilities[order]
    positives = int(labels.sum())

    distinct_ends = []
    index = 0
    while index < labels.size:
        end = index + 1
        while end < labels.size and scores[end] == scores[index]:
            end += 1
        distinct_ends.append(end - 1)
        index = end

    cumulative_tp = np.cumsum(labels)
    cumulative_fp = np.cumsum(1 - labels)
    precision_all = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1)
    recall_all = cumulative_tp / max(positives, 1)

    precision = precision_all[distinct_ends]
    recall = recall_all[distinct_ends]
    average_precision = 0.0
    previous_recall = 0.0
    for p_value, r_value in zip(precision, recall):
        average_precision += float((r_value - previous_recall) * p_value)
        previous_recall = float(r_value)
    return np.r_[1.0, precision, 0.0], np.r_[0.0, recall, 1.0], float(average_precision)


def binary_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float]:
    labels = labels.astype(int)
    predictions = (probabilities >= threshold).astype(int)
    tp = float(np.sum((labels == 1) & (predictions == 1)))
    tn = float(np.sum((labels == 0) & (predictions == 0)))
    fp = float(np.sum((labels == 0) & (predictions == 1)))
    fn = float(np.sum((labels == 1) & (predictions == 0)))

    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    specificity = tn / max(tn + fp, 1.0)
    negative_precision = tn / max(tn + fn, 1.0)
    f1_positive = 2.0 * precision * recall / max(precision + recall, 1e-12)
    f1_negative = (
        2.0 * negative_precision * specificity
        / max(negative_precision + specificity, 1e-12)
    )
    _, _, average_precision = precision_recall_curve(labels, probabilities)
    return {
        "accuracy": (tp + tn) / labels.size,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_pos": f1_positive,
        "macro_f1": 0.5 * (f1_positive + f1_negative),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "roc_auc": roc_auc_score(labels, probabilities),
        "pr_auc": average_precision,
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def choose_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.unique(np.r_[0.0, 0.5, probabilities])
    best_key = (-math.inf, -math.inf, -math.inf)
    best_threshold = 0.5
    for threshold in candidates:
        metrics = binary_metrics(labels, probabilities, float(threshold))
        key = (
            metrics["macro_f1"],
            metrics["f1_pos"],
            metrics["balanced_accuracy"],
        )
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def stratified_fold_ids(labels: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fold_ids = np.empty(labels.size, dtype=int)
    classes = np.unique(labels)
    offset = 0
    for class_index, class_value in enumerate(classes):
        indices = np.flatnonzero(labels == class_value)
        indices = rng.permutation(indices)
        fold_ids[indices] = (np.arange(indices.size) + offset) % n_splits
        offset = (offset + indices.size + class_index) % n_splits
    return fold_ids


def stratified_train_validation_split(
    labels: np.ndarray, validation_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts = []
    validation_parts = []
    for class_value in np.unique(labels):
        indices = np.flatnonzero(labels == class_value)
        indices = rng.permutation(indices)
        validation_size = max(1, int(round(indices.size * validation_fraction)))
        validation_size = min(validation_size, indices.size - 1)
        validation_parts.append(indices[:validation_size])
        train_parts.append(indices[validation_size:])
    return np.sort(np.concatenate(train_parts)), np.sort(np.concatenate(validation_parts))


def standardize(
    train_features: np.ndarray, other_features: list[np.ndarray]
) -> tuple[np.ndarray, list[np.ndarray]]:
    mean = train_features.mean(axis=0)
    scale = train_features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    standardized_train = (train_features - mean) / scale
    standardized_other = [(features - mean) / scale for features in other_features]
    return standardized_train, standardized_other


def weighted_bce(logits: np.ndarray, labels: np.ndarray, positive_weight: float) -> float:
    labels = labels.reshape(-1, 1)
    elementwise = np.maximum(logits, 0.0) - logits * labels + np.log1p(np.exp(-np.abs(logits)))
    sample_weights = labels * positive_weight + (1.0 - labels)
    return float(np.mean(sample_weights * elementwise))


def initialize_mlp(input_dim: int, hidden_dim: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {
        "W1": rng.normal(0.0, math.sqrt(2.0 / input_dim), (input_dim, hidden_dim)),
        "b1": np.zeros((1, hidden_dim)),
        "W2": rng.normal(0.0, math.sqrt(2.0 / hidden_dim), (hidden_dim, hidden_dim)),
        "b2": np.zeros((1, hidden_dim)),
        "W3": rng.normal(0.0, math.sqrt(1.0 / hidden_dim), (hidden_dim, 1)),
        "b3": np.zeros((1, 1)),
    }


def mlp_forward(
    features: np.ndarray, parameters: dict[str, np.ndarray]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    hidden_1 = np.maximum(features @ parameters["W1"] + parameters["b1"], 0.0)
    hidden_2 = np.maximum(hidden_1 @ parameters["W2"] + parameters["b2"], 0.0)
    logits = hidden_2 @ parameters["W3"] + parameters["b3"]
    cache = {
        "features": features,
        "hidden_1": hidden_1,
        "hidden_2": hidden_2,
        "logits": logits,
    }
    return logits, cache


def train_mlp(
    features: np.ndarray,
    labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    positive_weight: float,
    seed: int,
    hidden_dim: int = 32,
    batch_size: int = 128,
    max_epochs: int = 260,
    patience: int = 35,
    learning_rate: float = 0.003,
    l2: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    parameters = initialize_mlp(features.shape[1], hidden_dim, rng)
    first_moments = {key: np.zeros_like(value) for key, value in parameters.items()}
    second_moments = {key: np.zeros_like(value) for key, value in parameters.items()}
    best_ap = -math.inf
    best_parameters = {key: value.copy() for key, value in parameters.items()}
    stale_epochs = 0

    for epoch in range(1, max_epochs + 1):
        order = rng.permutation(features.shape[0])
        for start in range(0, features.shape[0], batch_size):
            batch_indices = order[start : start + batch_size]
            batch_features = features[batch_indices]
            batch_labels = labels[batch_indices].reshape(-1, 1)
            logits, cache = mlp_forward(batch_features, parameters)
            probabilities = sigmoid(logits)
            sample_weights = (
                batch_labels * positive_weight + (1.0 - batch_labels)
            )
            grad_logits = sample_weights * (probabilities - batch_labels) / batch_indices.size

            gradients: dict[str, np.ndarray] = {}
            gradients["W3"] = cache["hidden_2"].T @ grad_logits + l2 * parameters["W3"]
            gradients["b3"] = grad_logits.sum(axis=0, keepdims=True)
            grad_hidden_2 = (grad_logits @ parameters["W3"].T) * (
                cache["hidden_2"] > 0.0
            )
            gradients["W2"] = cache["hidden_1"].T @ grad_hidden_2 + l2 * parameters["W2"]
            gradients["b2"] = grad_hidden_2.sum(axis=0, keepdims=True)
            grad_hidden_1 = (grad_hidden_2 @ parameters["W2"].T) * (
                cache["hidden_1"] > 0.0
            )
            gradients["W1"] = cache["features"].T @ grad_hidden_1 + l2 * parameters["W1"]
            gradients["b1"] = grad_hidden_1.sum(axis=0, keepdims=True)

            for key, parameter in parameters.items():
                moment = first_moments[key]
                velocity = second_moments[key]
                moment *= 0.9
                moment += 0.1 * gradients[key]
                velocity *= 0.999
                velocity += 0.001 * gradients[key] ** 2
                corrected_moment = moment / (1.0 - 0.9**epoch)
                corrected_velocity = velocity / (1.0 - 0.999**epoch)
                parameter -= learning_rate * corrected_moment / (
                    np.sqrt(corrected_velocity) + 1e-8
                )

        validation_logits, _ = mlp_forward(validation_features, parameters)
        validation_probabilities = sigmoid(validation_logits).ravel()
        _, _, validation_ap = precision_recall_curve(
            validation_labels, validation_probabilities
        )
        if validation_ap > best_ap + 1e-5:
            best_ap = validation_ap
            best_parameters = {key: value.copy() for key, value in parameters.items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    validation_probabilities = sigmoid(
        mlp_forward(validation_features, best_parameters)[0]
    ).ravel()
    return validation_probabilities, validation_probabilities, best_parameters


def predict_mlp(
    parameters: dict[str, np.ndarray], features: np.ndarray
) -> np.ndarray:
    logits, _ = mlp_forward(features, parameters)
    return sigmoid(logits).ravel()


def train_one_split(
    model_name: str,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, float, np.ndarray, dict[str, object] | None]:
    inner_train_indices, validation_indices = stratified_train_validation_split(
        train_labels, validation_fraction=0.2, seed=seed + 10_000
    )
    inner_features, (validation_features, test_scaled) = standardize(
        train_features[inner_train_indices],
        [train_features[validation_indices], test_features],
    )
    inner_labels = train_labels[inner_train_indices]
    validation_labels = train_labels[validation_indices]
    negative_count = int(np.sum(inner_labels == 0))
    positive_count = int(np.sum(inner_labels == 1))
    full_ratio = negative_count / max(positive_count, 1)

    details: dict[str, object] | None = None
    if model_name == "logistic":
        positive_weight = full_ratio
        test_probabilities, logistic_details = fit_and_predict_logistic_for_test(
            inner_features,
            inner_labels,
            validation_features,
            validation_labels,
            test_scaled,
            positive_weight,
            seed,
        )
        validation_probabilities = np.asarray(
            logistic_details["validation_probabilities"]
        )
    else:
        if model_name == "mlp_ce":
            positive_weight = 1.0
        elif model_name == "mlp_cw_sqrt":
            positive_weight = math.sqrt(full_ratio)
        elif model_name == "mlp_cw":
            positive_weight = full_ratio
        else:
            raise ValueError(f"Unknown model: {model_name}")
        validation_probabilities, _, parameters = train_mlp(
            inner_features,
            inner_labels,
            validation_features,
            validation_labels,
            positive_weight,
            seed,
        )
        test_probabilities = predict_mlp(parameters, test_scaled)
        if model_name in {"mlp_ce", "mlp_cw_sqrt"}:
            details = {
                "parameters": parameters,
                "test_scaled": test_scaled,
            }

    threshold = choose_threshold(validation_labels, validation_probabilities)
    return test_probabilities, threshold, np.asarray([positive_weight]), details


def fit_and_predict_logistic_for_test(
    features: np.ndarray,
    labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    test_features: np.ndarray,
    positive_weight: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Train logistic regression and return both validation and test predictions."""
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.01, size=(features.shape[1], 1))
    bias = np.zeros((1, 1), dtype=np.float64)
    params = [weights, bias]
    first_moments = [np.zeros_like(item) for item in params]
    second_moments = [np.zeros_like(item) for item in params]
    best_ap = -math.inf
    best_params = [item.copy() for item in params]
    stale_epochs = 0
    max_epochs, patience, learning_rate, l2 = 400, 50, 0.03, 1e-4

    for epoch in range(1, max_epochs + 1):
        logits = features @ weights + bias
        probabilities = sigmoid(logits)
        sample_weights = labels.reshape(-1, 1) * positive_weight + (
            1.0 - labels.reshape(-1, 1)
        )
        grad_logits = sample_weights * (probabilities - labels.reshape(-1, 1)) / labels.size
        gradients = [features.T @ grad_logits + l2 * weights, grad_logits.sum(axis=0, keepdims=True)]

        for parameter, gradient, moment, velocity in zip(
            params, gradients, first_moments, second_moments
        ):
            moment *= 0.9
            moment += 0.1 * gradient
            velocity *= 0.999
            velocity += 0.001 * gradient**2
            corrected_moment = moment / (1.0 - 0.9**epoch)
            corrected_velocity = velocity / (1.0 - 0.999**epoch)
            parameter -= learning_rate * corrected_moment / (
                np.sqrt(corrected_velocity) + 1e-8
            )

        validation_probabilities = sigmoid(validation_features @ weights + bias).ravel()
        _, _, validation_ap = precision_recall_curve(
            validation_labels, validation_probabilities
        )
        if validation_ap > best_ap + 1e-5:
            best_ap = validation_ap
            best_params = [item.copy() for item in params]
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    weights, bias = best_params
    validation_probabilities = sigmoid(validation_features @ weights + bias).ravel()
    test_probabilities = sigmoid(test_features @ weights + bias).ravel()
    return test_probabilities, {
        "validation_probabilities": validation_probabilities,
        "weights": weights,
        "bias": bias,
    }


def run_experiments(
    features: np.ndarray,
    labels: np.ndarray,
    quality: np.ndarray,
    output_dir: Path,
    repeats: int = 3,
    folds: int = 5,
) -> None:
    model_names = ["logistic", "mlp_ce", "mlp_cw_sqrt", "mlp_cw"]
    seeds = [11, 22, 33][:repeats]
    metrics_rows: list[dict[str, float | int | str]] = []
    oof_rows: list[dict[str, float | int | str]] = []
    importance_rows: list[dict[str, float | int | str]] = []

    for seed in seeds:
        fold_ids = stratified_fold_ids(labels, folds, seed)
        for fold in range(folds):
            train_indices = np.flatnonzero(fold_ids != fold)
            test_indices = np.flatnonzero(fold_ids == fold)
            for model_name in model_names:
                # Share fold-wise seeds across models so paired comparisons use
                # the same data order and initialization stream.
                run_seed = seed + fold * 1_000
                test_probabilities, threshold, positive_weight_array, model_details = train_one_split(
                    model_name,
                    features[train_indices],
                    labels[train_indices],
                    features[test_indices],
                    labels[test_indices],
                    run_seed,
                )
                test_predictions = (test_probabilities >= threshold).astype(int)
                split_metrics = binary_metrics(
                    labels[test_indices], test_probabilities, threshold
                )
                fixed_threshold_metrics = binary_metrics(
                    labels[test_indices], test_probabilities, 0.5
                )
                split_metrics.update(
                    {
                        "seed": seed,
                        "fold": fold,
                        "model": model_name,
                        "threshold": threshold,
                        "positive_weight": float(positive_weight_array[0]),
                        "test_size": int(test_indices.size),
                        "fixed_recall": fixed_threshold_metrics["recall"],
                        "fixed_specificity": fixed_threshold_metrics["specificity"],
                        "fixed_f1_pos": fixed_threshold_metrics["f1_pos"],
                        "fixed_macro_f1": fixed_threshold_metrics["macro_f1"],
                    }
                )
                metrics_rows.append(split_metrics)
                for local_index, global_index in enumerate(test_indices):
                    oof_rows.append(
                        {
                            "seed": seed,
                            "fold": fold,
                            "model": model_name,
                            "sample_index": int(global_index),
                            "label": int(labels[global_index]),
                            "probability": float(test_probabilities[local_index]),
                            "threshold": float(threshold),
                            "prediction": int(test_predictions[local_index]),
                        }
                    )

                if model_name in {"mlp_ce", "mlp_cw_sqrt"} and model_details is not None:
                    base_curve_auc = split_metrics["pr_auc"]
                    importance_rng = np.random.default_rng(run_seed + 77)
                    parameters = model_details["parameters"]
                    test_scaled = np.asarray(model_details["test_scaled"])
                    for feature_index, feature_name in enumerate(FEATURE_NAMES_ZH):
                        drops = []
                        for _ in range(4):
                            permuted = test_scaled.copy()
                            permuted[:, feature_index] = importance_rng.permutation(
                                permuted[:, feature_index]
                            )
                            permuted_probabilities = predict_mlp(parameters, permuted)
                            _, _, permuted_auc = precision_recall_curve(
                                labels[test_indices], permuted_probabilities
                            )
                            drops.append(base_curve_auc - permuted_auc)
                        importance_rows.append(
                            {
                                "seed": seed,
                                "fold": fold,
                                "model": model_name,
                                "feature": feature_name,
                                "base_pr_auc": base_curve_auc,
                                "pr_auc_drop": float(np.mean(drops)),
                            }
                        )

    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / "metrics_by_run.csv"
    metric_fields = list(metrics_rows[0].keys())
    with metrics_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(metrics_rows)

    oof_path = results_dir / "oof_predictions.csv"
    with oof_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(oof_rows[0].keys()))
        writer.writeheader()
        writer.writerows(oof_rows)

    summary_rows = summarize_metrics(metrics_rows)
    summary_path = results_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    importance_path = results_dir / "feature_importance.csv"
    with importance_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(importance_rows[0].keys()))
        writer.writeheader()
        writer.writerows(importance_rows)

    dataset_summary = {
        "n_samples": int(labels.size),
        "n_features": int(features.shape[1]),
        "positive_count": int(labels.sum()),
        "negative_count": int(labels.size - labels.sum()),
        "positive_rate": float(labels.mean()),
        "imbalance_ratio": float((labels.size - labels.sum()) / labels.sum()),
        "quality_counts": {
            str(int(value)): int(np.sum(quality == value))
            for value in np.unique(quality)
        },
        "environment": {
            "python": "3.11.0",
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "evaluation": {
            "outer_folds": folds,
            "repeats": len(seeds),
            "inner_validation_fraction": 0.2,
            "seeds": seeds,
        },
    }
    (results_dir / "dataset_summary.json").write_text(
        json.dumps(dataset_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    make_experiment_figures(
        features, labels, quality, metrics_rows, oof_rows, output_dir / "figures"
    )


def summarize_metrics(
    metrics_rows: list[dict[str, float | int | str]]
) -> list[dict[str, float | int | str]]:
    model_names = []
    for row in metrics_rows:
        if row["model"] not in model_names:
            model_names.append(str(row["model"]))
    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1_pos",
        "macro_f1",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc",
        "brier",
        "threshold",
        "positive_weight",
        "fixed_recall",
        "fixed_specificity",
        "fixed_f1_pos",
        "fixed_macro_f1",
    ]
    summary = []
    for model_name in model_names:
        model_rows = [row for row in metrics_rows if row["model"] == model_name]
        result: dict[str, float | int | str] = {"model": model_name, "runs": len(model_rows)}
        for metric_name in metric_names:
            values = np.asarray([float(row[metric_name]) for row in model_rows])
            result[f"{metric_name}_mean"] = float(values.mean())
            result[f"{metric_name}_std"] = float(values.std(ddof=1))
        summary.append(result)
    return summary


def make_experiment_figures(
    features: np.ndarray,
    labels: np.ndarray,
    quality: np.ndarray,
    metrics_rows: list[dict[str, float | int | str]],
    oof_rows: list[dict[str, float | int | str]],
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    quality_values = np.unique(quality)
    counts = [int(np.sum(quality == value)) for value in quality_values]
    colors = ["#bdc3c7" if value < 7 else "#c0392b" for value in quality_values]
    figure, axes = plt.subplots(1, 2, figsize=(8.6, 3.2), gridspec_kw={"width_ratios": [1, 1.35]})
    axes[0].bar(
        [str(value) for value in quality_values],
        counts,
        color=colors,
        edgecolor="white",
    )
    for index, count in enumerate(counts):
        axes[0].text(index, count + 12, str(count), ha="center", va="bottom", fontsize=8)
    axes[0].set_xlabel("原始质量评分")
    axes[0].set_ylabel("样本数")
    axes[0].set_title("(a) 评分分布：红色为优质类", fontsize=10)
    axes[0].set_ylim(0, max(counts) * 1.16)

    correlation = np.asarray([np.corrcoef(features[:, index], labels)[0, 1] for index in range(features.shape[1])])
    order = np.argsort(correlation)
    axes[1].barh(
        [FEATURE_NAMES_ZH[index] for index in order],
        correlation[order],
        color=["#2c7fb8" if value >= 0 else "#d95f02" for value in correlation[order]],
    )
    axes[1].axvline(0.0, color="#555555", linewidth=0.8)
    axes[1].set_xlabel("与“质量≥7”的 Pearson 相关系数")
    axes[1].set_title("(b) 单变量线性关联", fontsize=10)
    axes[1].tick_params(axis="y", labelsize=7.5)
    figure.tight_layout()
    figure.savefig(figures_dir / "fig_dataset.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    model_names = []
    for row in metrics_rows:
        if row["model"] not in model_names:
            model_names.append(str(row["model"]))
    display_names = {
        "logistic": "加权逻辑回归",
        "mlp_ce": "MLP（无权重）",
        "mlp_cw_sqrt": "MLP（√类比权重）",
        "mlp_cw": "MLP（完整类比）",
    }
    macro_means = [
        float(np.mean([row["macro_f1"] for row in metrics_rows if row["model"] == model_name]))
        for model_name in model_names
    ]
    macro_stds = [
        float(np.std([row["macro_f1"] for row in metrics_rows if row["model"] == model_name], ddof=1))
        for model_name in model_names
    ]
    ap_means = [
        float(np.mean([row["pr_auc"] for row in metrics_rows if row["model"] == model_name]))
        for model_name in model_names
    ]
    ap_stds = [
        float(np.std([row["pr_auc"] for row in metrics_rows if row["model"] == model_name], ddof=1))
        for model_name in model_names
    ]
    x_positions = np.arange(len(model_names))
    width = 0.34
    figure, axis = plt.subplots(figsize=(7.1, 3.2))
    axis.bar(
        x_positions - width / 2,
        macro_means,
        width,
        yerr=macro_stds,
        capsize=3,
        label="Macro-F1",
        color="#2c7fb8",
    )
    axis.bar(
        x_positions + width / 2,
        ap_means,
        width,
        yerr=ap_stds,
        capsize=3,
        label="PR-AUC",
        color="#f39c12",
    )
    axis.set_xticks(x_positions)
    axis.set_xticklabels([display_names[name] for name in model_names], fontsize=8)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("交叉验证均值 ± 标准差")
    axis.set_title("模型性能对比（15 次外折评估）", fontsize=10)
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, ncol=2, loc="upper left")
    figure.tight_layout()
    figure.savefig(figures_dir / "fig_model_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    summary = summarize_metrics(metrics_rows)
    selected_model = max(summary, key=lambda row: float(row["macro_f1_mean"]))["model"]
    selected_oof = [row for row in oof_rows if row["model"] == selected_model and int(row["seed"]) == 11]
    selected_labels = np.asarray([int(row["label"]) for row in selected_oof])
    selected_probabilities = np.asarray([float(row["probability"]) for row in selected_oof])
    selected_predictions = np.asarray([int(row["prediction"]) for row in selected_oof])
    precision, recall, _ = precision_recall_curve(selected_labels, selected_probabilities)
    confusion = np.asarray(
        [
            [np.sum((selected_labels == 0) & (selected_predictions == 0)), np.sum((selected_labels == 0) & (selected_predictions == 1))],
            [np.sum((selected_labels == 1) & (selected_predictions == 0)), np.sum((selected_labels == 1) & (selected_predictions == 1))],
        ]
    )

    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.15), gridspec_kw={"width_ratios": [1.25, 1]})
    axes[0].plot(recall, precision, color="#2c7fb8", linewidth=2)
    axes[0].fill_between(recall, precision, alpha=0.16, color="#2c7fb8")
    axes[0].axhline(selected_labels.mean(), color="#a93226", linestyle="--", linewidth=1, label="正类率基线")
    axes[0].set_xlabel("召回率")
    axes[0].set_ylabel("精确率")
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_ylim(0.0, 1.03)
    axes[0].set_title(f"(a) {display_names[str(selected_model)]} 的 PR 曲线", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.18)

    image = axes[1].imshow(confusion, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axes[1].text(
                column,
                row,
                str(int(confusion[row, column])),
                ha="center",
                va="center",
                fontsize=13,
                color="white" if confusion[row, column] > confusion.max() * 0.55 else "#222222",
            )
    axes[1].set_xticks([0, 1], ["非优质", "优质"])
    axes[1].set_yticks([0, 1], ["非优质", "优质"])
    axes[1].set_xlabel("预测类别")
    axes[1].set_ylabel("真实类别")
    axes[1].set_title("(b) 单次 OOF 混淆矩阵", fontsize=10)
    figure.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(figures_dir / "fig_analysis.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/winequality-red.csv"))
    parser.add_argument("--output", type=Path, default=Path("."))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    features, labels, quality = load_wine_data(args.data)
    run_experiments(
        features,
        labels,
        quality,
        args.output,
        repeats=args.repeats,
        folds=args.folds,
    )
    print(f"Experiments finished. Results written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
