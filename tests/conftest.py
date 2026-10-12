"""Shared fixtures: a small synthetic dataset in the same ARFF layout as the UCI file."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from creditrisk.config import load_config
from creditrisk.data import FEATURES, ROW_ID, TARGET


def make_synthetic(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    """Synthetic firms: default risk driven by X1, X2 and missing X21; other features are noise."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 64))
    X[:, 28] = rng.normal(8, 1, n)  # X29 ~ log(total assets)
    logit = -3.0 - 1.6 * X[:, 0] + 1.2 * X[:, 1]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame(X, columns=FEATURES)
    df.loc[(y == 1) & (rng.random(n) < 0.3), "X21"] = np.nan   # informative missingness
    df.loc[rng.random(n) < 0.05, "X37"] = np.nan                # random missingness
    df[TARGET] = y
    df.insert(0, ROW_ID, np.arange(n))
    return df


def write_arff(df: pd.DataFrame, path: Path) -> None:
    lines = ["@relation synthetic", ""]
    lines += [f"@attribute Attr{i} numeric" for i in range(1, 65)]
    lines += ["@attribute class {0,1}", "", "@data"]
    vals = df[FEATURES].to_numpy()
    for row, t in zip(vals, df[TARGET].to_numpy()):
        lines.append(",".join("?" if np.isnan(v) else f"{v:.6f}" for v in row) + f",{t}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def small_config(root: Path) -> dict:
    cfg = load_config(root=root)
    cfg["data"]["raw_path"] = "data/raw/synthetic.arff"
    cfg["data"]["expected_rows"] = None
    cfg["data"]["expected_defaults"] = None
    cfg["scorecard"]["C_grid"] = [0.1, 1.0]
    cfg["challenger"].update({"learning_rate": [0.1], "max_depth": [3], "max_iter": 50})
    cfg["cv_folds"] = 3
    cfg["bootstrap_reps"] = 100
    cfg["simulation"]["batch_size"] = 400
    return cfg


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    return make_synthetic()
