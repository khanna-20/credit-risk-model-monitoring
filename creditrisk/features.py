"""Leakage-safe preprocessing and model pipelines.

All transformers learn their parameters in ``fit`` from the data they are given.
Inside cross-validation and the final fit they only ever see training rows,
so bin edges, WoE values, Information Value and feature selection cannot leak
information from the test set or the monitoring batches.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class MissingRateFilter(BaseEstimator, TransformerMixin):
    """Drop columns whose missing rate in the training data exceeds ``max_missing_rate``."""

    def __init__(self, max_missing_rate: float = 0.5):
        self.max_missing_rate = max_missing_rate

    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        rates = X.isna().mean()
        self.kept_ = [c for c in X.columns if rates[c] <= self.max_missing_rate]
        self.dropped_ = [c for c in X.columns if c not in self.kept_]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(X)[self.kept_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.kept_, dtype=object)


class WoEScorecardTransformer(BaseEstimator, TransformerMixin):
    """Quantile binning + Weight of Evidence + Information-Value/correlation feature selection.

    WoE is "risk oriented": WoE = ln(share of defaulters in bin / share of non-defaulters in bin),
    so a positive WoE means higher-than-average risk. Missing values get their own bin.
    A smoothing constant of 0.5 avoids log(0) for bins without defaulters.
    """

    def __init__(self, n_bins: int = 5, min_iv: float = 0.02, max_corr: float = 0.7,
                 max_features: int = 12, smoothing: float = 0.5):
        self.n_bins = n_bins
        self.min_iv = min_iv
        self.max_corr = max_corr
        self.max_features = max_features
        self.smoothing = smoothing

    # -- binning helpers -------------------------------------------------
    def _bin_index(self, values: np.ndarray, edges: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(edges, values, side="right").astype(float)
        idx[np.isnan(values)] = -1  # missing bin
        return idx

    def fit(self, X: pd.DataFrame, y):
        X = pd.DataFrame(X)
        y = np.asarray(y).astype(int)
        n_bad_total, n_good_total = int(y.sum()), int((1 - y).sum())
        if n_bad_total == 0 or n_good_total == 0:
            raise ValueError("WoE needs both classes in the training data")
        s = self.smoothing
        self.edges_, self.woe_maps_, self.iv_ = {}, {}, {}
        tables = []
        for col in X.columns:
            v = X[col].to_numpy(dtype=float)
            nonmiss = v[~np.isnan(v)]
            if len(np.unique(nonmiss)) <= 1:
                edges = np.array([])
            else:
                qs = np.quantile(nonmiss, np.linspace(0, 1, self.n_bins + 1)[1:-1])
                edges = np.unique(qs)
            idx = self._bin_index(v, edges)
            bins = list(range(len(edges) + 1)) + [-1]
            woe_map, iv = {}, 0.0
            k = len([b for b in bins if (idx == b).any()])
            for b in bins:
                mask = idx == b
                n = int(mask.sum())
                if n == 0:
                    woe_map[b] = 0.0  # unseen bin (e.g. missing never seen in training) = neutral
                    continue
                bad = int(y[mask].sum())
                good = n - bad
                dist_bad = (bad + s) / (n_bad_total + s * k)
                dist_good = (good + s) / (n_good_total + s * k)
                woe = float(np.log(dist_bad / dist_good))
                contrib = (dist_bad - dist_good) * woe
                iv += contrib
                woe_map[b] = woe
                tables.append({"feature": col, "bin": int(b), "bin_label": self._label(b, edges),
                               "n": n, "defaults": bad, "non_defaults": good,
                               "default_rate": bad / n, "woe": woe, "iv_contribution": contrib})
            self.edges_[col] = edges
            self.woe_maps_[col] = woe_map
            self.iv_[col] = float(iv)
        self.bin_table_ = pd.DataFrame(tables)

        # Feature selection: IV threshold, then greedy correlation filter in IV order.
        woe_train = self._woe_frame(X, list(X.columns))
        ranked = sorted(self.iv_, key=self.iv_.get, reverse=True)
        selected: list[str] = []
        for col in ranked:
            if self.iv_[col] < self.min_iv or len(selected) >= self.max_features:
                continue
            if woe_train[col].std() == 0:
                continue
            if selected:
                corr = woe_train[selected].corrwith(woe_train[col]).abs().max()
                if corr > self.max_corr:
                    continue
            selected.append(col)
        if not selected:
            raise ValueError("No feature passed the IV threshold; lower scorecard.min_iv")
        self.selected_ = selected
        return self

    @staticmethod
    def _label(b: int, edges: np.ndarray) -> str:
        if b == -1:
            return "MISSING"
        lo = "-inf" if b == 0 else f"{edges[b - 1]:.4g}"
        hi = "+inf" if b == len(edges) else f"{edges[b]:.4g}"
        return f"[{lo}, {hi})"

    def _woe_frame(self, X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        out = {}
        for col in cols:
            idx = self._bin_index(X[col].to_numpy(dtype=float), self.edges_[col])
            m = self.woe_maps_[col]
            out[col] = np.array([m.get(int(i), 0.0) for i in idx])
        return pd.DataFrame(out, index=X.index)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._woe_frame(pd.DataFrame(X), self.selected_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_, dtype=object)


def build_scorecard(cfg: dict, C: float = 1.0) -> Pipeline:
    sc = cfg["scorecard"]
    return Pipeline([
        ("missing_filter", MissingRateFilter(cfg["features"]["max_missing_rate"])),
        ("woe", WoEScorecardTransformer(n_bins=sc["n_bins"], min_iv=sc["min_iv"],
                                        max_corr=sc["max_corr"], max_features=sc["max_features"])),
        ("lr", LogisticRegression(C=C, class_weight=cfg.get("class_weight"), max_iter=2000)),
    ])


def build_challenger(cfg: dict, learning_rate: float = 0.1, max_depth: int | None = 3) -> Pipeline:
    ch = cfg["challenger"]
    cw = cfg.get("class_weight")
    return Pipeline([
        ("missing_filter", MissingRateFilter(cfg["features"]["max_missing_rate"])),
        ("gbm", HistGradientBoostingClassifier(
            learning_rate=learning_rate, max_depth=max_depth, max_iter=ch["max_iter"],
            min_samples_leaf=ch["min_samples_leaf"], early_stopping=False,
            class_weight=cw, random_state=cfg["project"]["random_seed"])),
    ])


def model_features(pipe: Pipeline) -> list[str]:
    """Raw input features the fitted model actually uses (for drift monitoring)."""
    if "woe" in pipe.named_steps:
        return list(pipe.named_steps["woe"].selected_)
    return list(pipe.named_steps["missing_filter"].kept_)


def scorecard_points(pipe: Pipeline, pdo: float = 20.0, base_score: float = 600.0,
                     base_odds: float = 50.0) -> pd.DataFrame:
    """Translate the fitted WoE logistic model into a points-based scorecard table.

    Convention: score = offset - factor * logit(PD); higher score = lower risk.
    ``base_odds`` are good:bad odds at ``base_score``; ``pdo`` = points to double the odds.
    """
    woe = pipe.named_steps["woe"]
    lr = pipe.named_steps["lr"]
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)  # score = offset + factor * ln(good:bad odds)
    k = len(woe.selected_)
    rows = []
    for coef, col in zip(lr.coef_[0], woe.selected_):
        tbl = woe.bin_table_[woe.bin_table_["feature"] == col]
        for _, r in tbl.iterrows():
            pts = offset / k - factor * (coef * r["woe"] + lr.intercept_[0] / k)
            rows.append({"feature": col, "bin_label": r["bin_label"], "woe": r["woe"],
                         "coefficient": coef, "points": round(float(pts), 1),
                         "n_train": r["n"], "default_rate_train": r["default_rate"]})
    return pd.DataFrame(rows)
