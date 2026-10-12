"""Discrimination, calibration, threshold and rating-grade analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             log_loss, roc_auc_score, roc_curve)


def ks_statistic(y: np.ndarray, p: np.ndarray) -> float:
    """Kolmogorov-Smirnov separation between defaulter and non-defaulter score distributions."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def classification_at(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"threshold": threshold, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "precision": precision, "recall": recall, "f1": f1,
            "flag_rate": float(pred.mean()), "specificity": tn / (tn + fp) if (tn + fp) else 0.0}


def discrimination_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    auc = roc_auc_score(y, p)
    return {"n": int(len(y)), "defaults": int(y.sum()), "default_rate": float(y.mean()),
            "mean_pd": float(p.mean()), "roc_auc": float(auc), "gini": float(2 * auc - 1),
            "ks": ks_statistic(y, p), "pr_auc": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1]))}


def threshold_table(y: np.ndarray, p: np.ndarray, cost_fn: float, cost_fp: float,
                    thresholds: np.ndarray | None = None) -> pd.DataFrame:
    """Precision/recall/F1 and expected cost across thresholds.

    Expected cost per firm = (cost_fn * FN + cost_fp * FP) / n, under the stated cost assumptions.
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.01, 0.91, 0.01), 2)
    rows = []
    for t in thresholds:
        r = classification_at(y, p, float(t))
        r["expected_cost_per_firm"] = (cost_fn * r["fn"] + cost_fp * r["fp"]) / len(y)
        rows.append(r)
    return pd.DataFrame(rows)


def choose_thresholds(tbl: pd.DataFrame) -> dict:
    """Pick thresholds from a table built on TRAINING out-of-fold predictions only."""
    cost_row = tbl.loc[tbl["expected_cost_per_firm"].idxmin()]
    f1_row = tbl.loc[tbl["f1"].idxmax()]
    return {"cost_optimal": float(cost_row["threshold"]), "f1_optimal": float(f1_row["threshold"])}


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Observed default rate vs mean predicted PD by predicted-PD decile."""
    df = pd.DataFrame({"y": np.asarray(y).astype(int), "p": np.asarray(p, dtype=float)})
    df["bucket"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False) + 1
    out = df.groupby("bucket").agg(n=("y", "size"), defaults=("y", "sum"),
                                   mean_pd=("p", "mean"), observed_dr=("y", "mean")).reset_index()
    return out


def assign_grade(p: np.ndarray, scale: list[dict]) -> np.ndarray:
    """Map PDs to rating grades using a master scale of upper PD bounds (exclusive)."""
    uppers = np.array([g["upper_pd"] for g in scale])
    names = np.array([g["grade"] for g in scale])
    idx = np.searchsorted(uppers, np.asarray(p, dtype=float), side="right")
    idx = np.clip(idx, 0, len(names) - 1)
    return names[idx]


def grade_table(y: np.ndarray | None, p: np.ndarray, scale: list[dict]) -> pd.DataFrame:
    """Per-grade counts, observed default rate and a two-sided binomial calibration test."""
    df = pd.DataFrame({"p": np.asarray(p, dtype=float), "grade": assign_grade(p, scale)})
    if y is not None:
        df["y"] = np.asarray(y).astype(int)
    rows = []
    for g in [s["grade"] for s in scale]:
        sub = df[df["grade"] == g]
        row = {"grade": g, "n": int(len(sub)), "share": len(sub) / len(df) if len(df) else 0.0,
               "mean_pd": float(sub["p"].mean()) if len(sub) else np.nan}
        if y is not None:
            d = int(sub["y"].sum()) if len(sub) else 0
            row["defaults"] = d
            row["observed_dr"] = d / len(sub) if len(sub) else np.nan
            row["binomial_p"] = (binomtest(d, len(sub), min(max(row["mean_pd"], 1e-6), 1 - 1e-6)).pvalue
                                 if len(sub) else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_auc(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray | None = None,
                  reps: int = 1000, seed: int = 42) -> dict:
    """Percentile bootstrap CI for AUC of model A and (optionally) the paired difference A - B."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    n = len(y)
    aucs_a, diffs = [], []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        a = roc_auc_score(y[idx], p_a[idx])
        aucs_a.append(a)
        if p_b is not None:
            diffs.append(a - roc_auc_score(y[idx], p_b[idx]))
    out = {"auc_ci_low": float(np.percentile(aucs_a, 2.5)), "auc_ci_high": float(np.percentile(aucs_a, 97.5))}
    if p_b is not None:
        out.update({"diff_mean": float(np.mean(diffs)), "diff_ci_low": float(np.percentile(diffs, 2.5)),
                    "diff_ci_high": float(np.percentile(diffs, 97.5))})
    return out
