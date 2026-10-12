"""Schema checks and data-quality validation.

Every check returns a row with status PASS / WARN / FAIL so the result can be
written to CSV and reviewed. FAIL stops the pipeline; WARN is reported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import FEATURES, TARGET


def _row(check: str, status: str, value, detail: str) -> dict:
    return {"check": check, "status": status, "value": value, "detail": detail}


def run_data_quality_checks(
    df: pd.DataFrame,
    expected_rows: int | None = None,
    expected_defaults: int | None = None,
    max_missing_rate: float = 0.5,
) -> pd.DataFrame:
    """Validate the development dataset. Returns one row per check."""
    rows: list[dict] = []

    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    rows.append(_row("schema_required_columns", "FAIL" if missing_cols else "PASS",
                     len(missing_cols), f"missing: {missing_cols}" if missing_cols else "X1..X64 and target present"))
    if missing_cols:
        return pd.DataFrame(rows)

    non_numeric = [c for c in FEATURES if not pd.api.types.is_numeric_dtype(df[c])]
    rows.append(_row("schema_numeric_features", "FAIL" if non_numeric else "PASS",
                     len(non_numeric), f"non-numeric: {non_numeric}" if non_numeric else "all 64 features numeric"))

    target_values = sorted(pd.unique(df[TARGET].dropna()).tolist())
    bad_target = df[TARGET].isna().any() or not set(target_values) <= {0, 1}
    rows.append(_row("target_binary", "FAIL" if bad_target else "PASS", str(target_values),
                     "target must be 0/1 with no missing values"))

    n = len(df)
    if expected_rows is not None:
        rows.append(_row("row_count", "PASS" if n == expected_rows else "WARN", n,
                         f"expected {expected_rows} (UCI documentation)"))
    n_def = int((df[TARGET] == 1).sum())
    if expected_defaults is not None:
        rows.append(_row("default_count", "PASS" if n_def == expected_defaults else "WARN", n_def,
                         f"expected {expected_defaults} (UCI documentation)"))
    rows.append(_row("default_rate", "PASS", round(n_def / n, 4) if n else np.nan,
                     "share of bankrupt firms; informs class-imbalance handling"))

    feat = df[FEATURES]
    dup_mask = df.duplicated(subset=FEATURES + [TARGET], keep="first")
    rows.append(_row("exact_duplicate_rows", "WARN" if dup_mask.any() else "PASS", int(dup_mask.sum()),
                     "removed before splitting so copies cannot sit in both train and test"))
    feat_dup = df[feat.duplicated(keep=False)]
    conflicting = 0
    if len(feat_dup):
        conflicting = int((feat_dup.groupby(FEATURES, dropna=False)[TARGET].nunique() > 1).sum())
    rows.append(_row("duplicates_with_conflicting_labels", "WARN" if conflicting else "PASS", conflicting,
                     "identical features but different labels (dropped entirely if present)"))

    n_inf = int(np.isinf(feat.to_numpy(dtype=float)).sum())
    rows.append(_row("infinite_values", "WARN" if n_inf else "PASS", n_inf, "inf values are converted to missing"))

    all_missing_rows = int(feat.isna().all(axis=1).sum())
    rows.append(_row("rows_all_features_missing", "FAIL" if all_missing_rows else "PASS", all_missing_rows,
                     "rows with no usable feature values"))

    miss = feat.isna().mean()
    high = miss[miss > max_missing_rate]
    rows.append(_row("high_missing_columns", "WARN" if len(high) else "PASS", len(high),
                     f"> {max_missing_rate:.0%} missing: {high.round(3).to_dict()}" if len(high)
                     else f"no column above {max_missing_rate:.0%} missing (max {miss.max():.1%}, {miss.idxmax()})"))
    rows.append(_row("rows_with_any_missing", "PASS", round(float(feat.isna().any(axis=1).mean()), 4),
                     "missing values are kept and handled inside the model pipelines"))

    constant = [c for c in FEATURES if feat[c].nunique(dropna=True) <= 1]
    rows.append(_row("constant_columns", "WARN" if constant else "PASS", len(constant), str(constant)))

    q01, q99 = feat.quantile(0.01), feat.quantile(0.99)
    extreme = int(((feat < q01 - 10 * (q99 - q01).abs()) | (feat > q99 + 10 * (q99 - q01).abs())).sum().sum())
    rows.append(_row("extreme_outlier_cells", "WARN" if extreme else "PASS", extreme,
                     "cells far outside the 1%-99% range; models use quantile bins / trees, which are robust to them"))
    return pd.DataFrame(rows)


def clean_development_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Convert inf to NaN, remove exact duplicates and any conflicting-label duplicates."""
    out = df.copy()
    out[FEATURES] = out[FEATURES].replace([np.inf, -np.inf], np.nan)
    before = len(out)
    out = out.drop_duplicates(subset=FEATURES + [TARGET], keep="first")
    after_exact = len(out)
    conflict_mask = out.duplicated(subset=FEATURES, keep=False)
    out = out[~conflict_mask]
    info = {"rows_in": before, "exact_duplicates_removed": before - after_exact,
            "conflicting_rows_removed": int(conflict_mask.sum()), "rows_out": len(out)}
    return out.reset_index(drop=True), info


def validate_batch_schema(batch: pd.DataFrame, required_features: list[str]) -> list[str]:
    """Return a list of schema problems for a monitoring batch (empty list = OK)."""
    problems = []
    missing = [c for c in required_features if c not in batch.columns]
    if missing:
        problems.append(f"missing required columns: {missing}")
    for c in required_features:
        if c in batch.columns and not pd.api.types.is_numeric_dtype(batch[c]):
            problems.append(f"column {c} is not numeric")
    if TARGET in batch.columns:
        vals = set(pd.unique(batch[TARGET].dropna()).tolist())
        if not vals <= {0, 1}:
            problems.append(f"target has values other than 0/1: {sorted(vals)}")
    if len(batch) == 0:
        problems.append("batch is empty")
    return problems
