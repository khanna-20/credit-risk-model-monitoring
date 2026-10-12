"""Create SIMULATED monitoring batches from the held-out monitoring pool.

IMPORTANT: these batches are synthetic test cases built by resampling (with
replacement) real UCI rows that were never used for model fitting, then
applying deliberate, documented changes. They are NOT real bank or production
observations. Every batch file carries ``is_simulated = True`` and a
``data_origin`` column saying so.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import resolve
from .data import TARGET

ORIGIN = "SIMULATED - resampled from held-out UCI monitoring pool; not real bank data"

SCENARIO_NOTES = {
    "baseline": "Random resample of the monitoring pool. Expected: stable.",
    "population_shift_small_firms": "Smaller firms (bottom 40% of X29 = log total assets) are 4x more likely "
                                    "to be drawn. Feature mix changes; the PD-outcome relationship does not.",
    "default_rate_increase": "Defaulters drawn at twice their pool share (prior shift). Ranking should hold; "
                             "calibration should break.",
    "missing_data_spike": "35% of values in the champion's 2nd-ranked feature set to missing (simulated upstream "
                          "feed failure).",
    "relationship_break": "Values of the champion's top-2 features shuffled within the batch: each feature's "
                          "distribution is unchanged but its link to the outcome is broken (simulated concept drift "
                          "or data-mapping error).",
    "baseline_labels_pending": "Random resample with outcome labels removed (1-year outcomes not yet observed).",
}


def _sample(pool: pd.DataFrame, size: int, rng, weights: np.ndarray | None = None) -> pd.DataFrame:
    p = None if weights is None else weights / weights.sum()
    idx = rng.choice(len(pool), size=size, replace=True, p=p)
    return pool.iloc[idx].reset_index(drop=True)


def build_batch(scenario: str, pool: pd.DataFrame, size: int, rng, model_feats: list[str]) -> pd.DataFrame:
    if scenario in ("baseline", "baseline_labels_pending"):
        out = _sample(pool, size, rng)
        if scenario == "baseline_labels_pending":
            out = out.drop(columns=[TARGET])
        return out
    if scenario == "population_shift_small_firms":
        size_var = pool["X29"].fillna(pool["X29"].median())
        w = np.where(size_var <= size_var.quantile(0.40), 4.0, 1.0)
        return _sample(pool, size, rng, w)
    if scenario == "default_rate_increase":
        defaults, healthy = pool[pool[TARGET] == 1], pool[pool[TARGET] == 0]
        n_def = int(round(size * min(2 * pool[TARGET].mean(), 0.9)))
        part1 = defaults.iloc[rng.choice(len(defaults), n_def, replace=True)]
        part2 = healthy.iloc[rng.choice(len(healthy), size - n_def, replace=True)]
        return pd.concat([part1, part2]).sample(frac=1, random_state=int(rng.integers(1e9))).reset_index(drop=True)
    if scenario == "missing_data_spike":
        out = _sample(pool, size, rng)
        col = model_feats[1] if len(model_feats) > 1 else model_feats[0]
        mask = rng.random(len(out)) < 0.35
        out.loc[mask, col] = np.nan
        return out
    if scenario == "relationship_break":
        out = _sample(pool, size, rng)
        for col in model_feats[:2]:
            out[col] = rng.permutation(out[col].to_numpy())
        return out
    raise ValueError(f"Unknown scenario '{scenario}'")


def simulate_batches(cfg: dict, bundle: dict) -> list[Path]:
    """Write one CSV per configured simulated batch and return the file paths."""
    sim = cfg["simulation"]
    rng = np.random.default_rng(cfg["project"]["random_seed"] + 1)
    pool = bundle["monitoring_pool"]
    feats = bundle["metadata"]["champion_features"]
    out_dir = resolve(cfg, cfg["monitoring"]["batch_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("simulated_batch_*.csv"):
        old.unlink()
    start = pd.Timestamp(sim["start_period"])
    paths = []
    for i, b in enumerate(sim["batches"]):
        batch = build_batch(b["scenario"], pool, sim["batch_size"], rng, feats)
        batch.insert(0, "batch_id", b["batch_id"])
        batch.insert(1, "period_start", (start + pd.DateOffset(months=i)).strftime("%Y-%m-%d"))
        batch.insert(2, "scenario", b["scenario"])
        batch.insert(3, "is_simulated", True)
        batch.insert(4, "data_origin", ORIGIN)
        path = out_dir / f"simulated_batch_{b['batch_id']}.csv"
        batch.to_csv(path, index=False)
        paths.append(path)
    print(f"[simulate] wrote {len(paths)} SIMULATED batches to {out_dir}")
    return paths
