"""Automated model-monitoring workflow.

For each batch file it:
  1. validates the schema,
  2. scores the batch with the champion model,
  3. measures score stability (PSI, KS), feature stability (CSI, KS) and missing-data changes,
  4. if outcome labels are present: default-rate shift, calibration and discrimination (Gini),
  5. assigns GREEN / AMBER / RED statuses from config thresholds,
  6. separates feature drift, target-distribution change and genuine performance deterioration,
  7. writes alerts, a Markdown report and Tableau-ready CSVs.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest, ks_2samp

from . import plots
from .config import output_dirs, resolve
from .data import FEATURE_DESCRIPTIONS, TARGET
from .evaluation import assign_grade, discrimination_metrics
from .utils import md_table
from .validation import validate_batch_schema

ORDER = {"GREEN": 0, "PENDING": 0, "N/A": 0, "AMBER": 1, "RED": 2}
ACTIONS = {
    "schema": "Batch rejected: fix the input file layout with the data owner before re-running.",
    "score_stability": "Review which features moved (CSI) and confirm with the business whether the portfolio "
                       "mix genuinely changed.",
    "feature_stability": "Check the feature against its source system; confirm whether the shift is genuine "
                         "or a data/mapping change.",
    "data_quality": "Raise a data incident with the source owner; treat scores for affected firms with caution.",
    "default_rate": "Confirm the outcome data; if the shift persists for 2+ periods, assess recalibration.",
    "calibration": "Predicted PDs do not match observed defaults: assess recalibration with the model owner.",
    "performance": "Escalate to the model owner: root-cause analysis and consider redevelopment or the challenger.",
}


# ------------------------------- statistics -----------------------------------------
def quantile_edges(ref: np.ndarray, n_bins: int) -> np.ndarray:
    ref = np.asarray(ref, dtype=float)
    ref = ref[~np.isnan(ref)]
    if len(np.unique(ref)) <= 1:
        return np.array([])
    return np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)[1:-1]))


def bin_shares(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Share of values per bin; the last element is the missing-value bin."""
    v = np.asarray(values, dtype=float)
    miss = np.isnan(v)
    idx = np.searchsorted(edges, v[~miss], side="right")
    counts = np.bincount(idx, minlength=len(edges) + 1).astype(float)
    counts = np.append(counts, miss.sum())
    return counts / max(len(v), 1)


def psi(ref: np.ndarray, cur: np.ndarray, n_bins: int = 10, eps: float = 1e-4) -> tuple[float, pd.DataFrame]:
    """Population Stability Index with bins from reference quantiles plus a missing bin.

    PSI = sum((cur% - ref%) * ln(cur% / ref%)). Empty bins are floored at ``eps``.
    """
    edges = quantile_edges(ref, n_bins)
    r, c = bin_shares(ref, edges), bin_shares(cur, edges)
    rf, cf = np.clip(r, eps, None), np.clip(c, eps, None)
    contrib = (cf - rf) * np.log(cf / rf)
    labels = [f"bin_{i + 1:02d}" for i in range(len(edges) + 1)] + ["missing"]
    lower = np.concatenate([[-np.inf], edges, [np.nan]])
    tbl = pd.DataFrame({"bin": labels, "lower_edge": lower, "ref_share": r, "cur_share": c,
                        "psi_contribution": contrib})
    return float(contrib.sum()), tbl


def rag(value: float, amber: float, red: float) -> str:
    """Higher = worse. NaN -> N/A."""
    if value is None or np.isnan(value):
        return "N/A"
    return "RED" if value >= red else "AMBER" if value >= amber else "GREEN"


def rag_two_sided_ratio(ratio: float, amber: float, red: float) -> str:
    """For ratios where both directions matter (e.g. 2.0x or 0.5x the reference)."""
    if ratio is None or np.isnan(ratio) or ratio <= 0:
        return "N/A"
    dev = max(ratio, 1 / ratio)
    return "RED" if dev >= red else "AMBER" if dev >= amber else "GREEN"


def rag_pvalue(p: float, amber: float, red: float) -> str:
    """Lower p-value = worse."""
    if p is None or np.isnan(p):
        return "N/A"
    return "RED" if p < red else "AMBER" if p < amber else "GREEN"


def worst(*statuses: str) -> str:
    return max(statuses, key=lambda s: ORDER.get(s, 0)) if statuses else "GREEN"


def diagnose(schema_ok: bool, labels: bool, score: str, feature: str, dq: str,
             default_rate: str, calibration: str, perf: str) -> str:
    """Translate component statuses into a plain-English diagnosis.

    Keeps three things separate: input drift (score/features), a change in the
    outcome distribution (default rate, calibration) and a fall in ranking
    power (Gini) - plus data-quality problems that can cause any of them.
    """
    if not schema_ok:
        return "Schema failure - batch not scored"
    bad = lambda s: s in ("AMBER", "RED")  # noqa: E731
    if bad(score) and bad(feature):
        drift_txt = "score and feature drift"
    elif bad(feature):
        drift_txt = "feature drift"
    elif bad(score):
        drift_txt = "score drift but no feature-level drift"
    else:
        drift_txt = ""
    parts = []
    if bad(dq):
        parts.append("Data-quality issue (missing-value spike)"
                     + (" - likely cause of the calibration breach" if bad(calibration) and not bad(default_rate)
                        else ""))
    if bad(perf):
        parts.append("Performance deterioration" + (f" with {drift_txt}" if drift_txt
                                                    else " without detectable input drift"))
    elif bad(default_rate):
        parts.append("Default-rate shift with ranking intact")
    elif bad(calibration) and not bad(dq):
        parts.append("Calibration shift with ranking intact")
    elif drift_txt and not bad(dq):
        parts.append(f"{drift_txt[0].upper()}{drift_txt[1:]} without performance deterioration" if labels
                     else f"{drift_txt[0].upper()}{drift_txt[1:]} - performance impact unknown until labels arrive")
    if not parts:
        return "Stable" if labels else "Stable on available signals (labels pending)"
    return "; ".join(parts)


# ------------------------------- per-batch monitoring ------------------------------
def monitor_batch(batch: pd.DataFrame, bundle: dict, cfg: dict) -> dict:
    mc = cfg["monitoring"]
    meta = bundle["metadata"]
    champ = bundle["models"][meta["champion"]]
    ref = bundle["reference"]
    feats = meta["champion_features"]
    extra = [f for f in mc.get("extra_features", []) if f not in feats]
    info = {
        "batch_id": str(batch["batch_id"].iloc[0]) if "batch_id" in batch else "unknown",
        "period_start": str(batch["period_start"].iloc[0]) if "period_start" in batch else "",
        "scenario": str(batch["scenario"].iloc[0]) if "scenario" in batch else "unspecified",
        "is_simulated": bool(batch["is_simulated"].iloc[0]) if "is_simulated" in batch else False,
    }
    labels = TARGET in batch.columns and batch[TARGET].notna().all()
    model_inputs = list(champ.named_steps["missing_filter"].feature_names_in_)
    problems = validate_batch_schema(batch, model_inputs)
    summary = {**info, "labels_available": labels, "n": len(batch), "schema_problems": "; ".join(problems)}
    out = {"summary": summary, "feature_drift": [], "grades": [], "score_dist": [], "metrics_long": []}
    if problems:
        summary.update({"overall_status": "RED", "diagnosis": diagnose(False, labels, "", "", "", "", "", "")})
        return out

    pdv = champ.predict_proba(batch[model_inputs])[:, 1]
    n_bins = mc["n_score_bins"]

    # Score stability
    score_psi, score_tbl = psi(ref["pd"].to_numpy(), pdv, n_bins)
    ks = ks_2samp(ref["pd"].to_numpy(), pdv)
    score_status = rag(score_psi, mc["psi"]["amber"], mc["psi"]["red"])
    out["score_dist"] = score_tbl.assign(**info).to_dict("records")

    # Feature stability and data quality
    ref_missing = meta["reference_missing_rates"]
    feat_statuses, miss_statuses = [], []
    for f in feats + extra:
        ref_vals = ref[f].to_numpy(dtype=float)
        cur_vals = batch[f].to_numpy(dtype=float)
        csi, _ = psi(ref_vals, cur_vals, n_bins)
        r_nm, c_nm = ref_vals[~np.isnan(ref_vals)], cur_vals[~np.isnan(cur_vals)]
        kst = ks_2samp(r_nm, c_nm) if len(r_nm) and len(c_nm) else None
        ref_m = ref_missing.get(f, float(np.isnan(ref_vals).mean()))
        cur_m = float(np.isnan(cur_vals).mean())
        csi_s = rag(csi, mc["psi"]["amber"], mc["psi"]["red"])
        miss_s = rag(cur_m - ref_m, mc["missing_rate_increase"]["amber"], mc["missing_rate_increase"]["red"])
        if f in feats:
            feat_statuses.append(csi_s)
            miss_statuses.append(miss_s)
        out["feature_drift"].append({
            **info, "feature": f, "description": FEATURE_DESCRIPTIONS.get(f, ""), "in_model": f in feats,
            "csi": csi, "csi_status": csi_s,
            "ks_stat": kst.statistic if kst else np.nan, "ks_pvalue": kst.pvalue if kst else np.nan,
            "ks_significant": bool(kst and kst.pvalue < mc["ks_alpha"]),
            "ref_missing_rate": ref_m, "cur_missing_rate": cur_m, "missing_increase": cur_m - ref_m,
            "missing_status": miss_s})
    feature_status = worst(*feat_statuses)
    dq_status = worst(*miss_statuses)

    # Grades
    grades_cur = assign_grade(pdv, cfg["rating_scale"])
    ref_share = ref["grade"].value_counts(normalize=True)
    y = batch[TARGET].to_numpy(dtype=int) if labels else None
    for g in [s["grade"] for s in cfg["rating_scale"]]:
        m = grades_cur == g
        row = {**info, "grade": g, "n": int(m.sum()), "share": float(m.mean()),
               "ref_share": float(ref_share.get(g, 0.0)), "mean_pd": float(pdv[m].mean()) if m.any() else np.nan}
        if labels:
            d = int(y[m].sum())
            row.update({"defaults": d, "observed_dr": d / m.sum() if m.any() else np.nan})
        out["grades"].append(row)

    # Outcome-based monitoring (only when labels exist)
    ref_dr = meta["reference_default_rate"]
    bench = meta["benchmark_gini"]
    mean_pd = float(pdv.mean())
    perf = {"auc": np.nan, "gini": np.nan, "ks": np.nan, "pr_auc": np.nan, "brier": np.nan}
    obs_dr = dr_ratio = calib_p = rel_drop = np.nan
    defaults = np.nan
    if labels:
        defaults = int(y.sum())
        obs_dr = float(y.mean())
        dr_ratio = obs_dr / ref_dr if ref_dr else np.nan
        calib_p = float(binomtest(defaults, len(y), min(max(mean_pd, 1e-6), 1 - 1e-6)).pvalue)
        dr_status = rag_two_sided_ratio(dr_ratio, mc["default_rate_ratio"]["amber"], mc["default_rate_ratio"]["red"])
        calib_status = rag_pvalue(calib_p, mc["calibration_pvalue"]["amber"], mc["calibration_pvalue"]["red"])
        if defaults >= mc["min_defaults_for_performance"] and defaults < len(y):
            d = discrimination_metrics(y, pdv)
            perf = {k: d[k] for k in ("roc_auc", "gini", "ks", "pr_auc", "brier")}
            perf["auc"] = perf.pop("roc_auc")
            rel_drop = (bench - perf["gini"]) / bench
            perf_status = rag(rel_drop, mc["gini_relative_drop"]["amber"], mc["gini_relative_drop"]["red"])
        else:
            perf_status = "N/A"
    else:
        dr_status = calib_status = perf_status = "PENDING"
    target_status = worst(dr_status, calib_status)

    overall = worst(score_status, feature_status, dq_status, target_status, perf_status)
    summary.update({
        "defaults": defaults, "observed_default_rate": obs_dr, "reference_default_rate": ref_dr,
        "default_rate_ratio": dr_ratio, "mean_pd": mean_pd, "calibration_pvalue": calib_p,
        "score_psi": score_psi, "score_ks_stat": ks.statistic, "score_ks_pvalue": ks.pvalue,
        "max_feature_csi": max(r["csi"] for r in out["feature_drift"] if r["in_model"]),
        "n_features_amber": feat_statuses.count("AMBER"), "n_features_red": feat_statuses.count("RED"),
        "max_missing_increase": max(r["missing_increase"] for r in out["feature_drift"] if r["in_model"]),
        **perf, "benchmark_gini": bench, "gini_relative_drop": rel_drop,
        "score_stability_status": score_status, "feature_stability_status": feature_status,
        "data_quality_status": dq_status, "default_rate_status": dr_status, "calibration_status": calib_status,
        "performance_status": perf_status, "overall_status": overall,
        "diagnosis": diagnose(True, labels, score_status, feature_status, dq_status, dr_status, calib_status,
                              perf_status),
    })
    p = mc
    out["metrics_long"] = [
        {**info, "component": c, "metric": m, "value": v, "amber_threshold": a, "red_threshold": r, "status": s}
        for c, m, v, a, r, s in [
            ("score_stability", "score_psi", score_psi, p["psi"]["amber"], p["psi"]["red"], score_status),
            ("feature_stability", "max_feature_csi", summary["max_feature_csi"], p["psi"]["amber"],
             p["psi"]["red"], feature_status),
            ("data_quality", "max_missing_increase", summary["max_missing_increase"],
             p["missing_rate_increase"]["amber"], p["missing_rate_increase"]["red"], dq_status),
            ("default_rate", "default_rate_ratio", dr_ratio, p["default_rate_ratio"]["amber"],
             p["default_rate_ratio"]["red"], dr_status),
            ("calibration", "calibration_pvalue", calib_p, p["calibration_pvalue"]["amber"],
             p["calibration_pvalue"]["red"], calib_status),
            ("performance", "gini", perf["gini"], bench * (1 - p["gini_relative_drop"]["amber"]),
             bench * (1 - p["gini_relative_drop"]["red"]), perf_status),
            ("performance", "gini_relative_drop", rel_drop, p["gini_relative_drop"]["amber"],
             p["gini_relative_drop"]["red"], perf_status),
        ]]
    return out


def build_alerts(summary: pd.DataFrame, feature_drift: pd.DataFrame) -> pd.DataFrame:
    rows = []
    comps = [("score_stability_status", "score_stability", "score_psi"),
             ("feature_stability_status", "feature_stability", "max_feature_csi"),
             ("data_quality_status", "data_quality", "max_missing_increase"),
             ("default_rate_status", "default_rate", "default_rate_ratio"),
             ("calibration_status", "calibration", "calibration_pvalue"),
             ("performance_status", "performance", "gini_relative_drop")]
    for _, s in summary.iterrows():
        base = {k: s[k] for k in ("batch_id", "period_start", "scenario", "is_simulated")}
        if s.get("schema_problems"):
            rows.append({**base, "severity": "RED", "component": "schema", "metric": "schema",
                         "value": np.nan, "detail": s["schema_problems"], "recommended_action": ACTIONS["schema"]})
            continue
        for col, comp, metric in comps:
            if s[col] in ("AMBER", "RED"):
                detail = ""
                if comp in ("feature_stability", "data_quality"):
                    key = "csi_status" if comp == "feature_stability" else "missing_status"
                    f = feature_drift[(feature_drift["batch_id"] == s["batch_id"]) & feature_drift["in_model"]
                                      & feature_drift[key].isin(["AMBER", "RED"])]
                    detail = "features: " + ", ".join(f["feature"])
                rows.append({**base, "severity": s[col], "component": comp, "metric": metric,
                             "value": s[metric], "detail": detail, "recommended_action": ACTIONS[comp]})
    alerts = pd.DataFrame(rows, columns=["batch_id", "period_start", "scenario", "is_simulated", "severity",
                                         "component", "metric", "value", "detail", "recommended_action"])
    alerts.insert(0, "alert_id", [f"ALERT_{i + 1:03d}" for i in range(len(alerts))])
    return alerts


def load_batches(batch_dir: Path) -> list[pd.DataFrame]:
    files = sorted(batch_dir.glob("*.csv"))
    batches = []
    for f in files:
        b = pd.read_csv(f)
        if "batch_id" not in b:
            b.insert(0, "batch_id", f.stem)
        batches.append(b)
    batches.sort(key=lambda b: (str(b["period_start"].iloc[0]) if "period_start" in b else "", str(b["batch_id"].iloc[0])))
    return batches


def run_monitoring(cfg: dict, batch_dir: str | Path | None = None) -> dict:
    dirs = output_dirs(cfg)
    bundle_path = dirs["model"] / "model_bundle.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError("No trained model found - run `python -m creditrisk develop` first.")
    bundle = joblib.load(bundle_path)
    bdir = Path(batch_dir) if batch_dir else resolve(cfg, cfg["monitoring"]["batch_dir"])
    batches = load_batches(bdir)
    if not batches:
        raise FileNotFoundError(f"No batch CSV files found in {bdir}")
    results = [monitor_batch(b, bundle, cfg) for b in batches]
    summary = pd.DataFrame([r["summary"] for r in results])
    feature_drift = pd.DataFrame([row for r in results for row in r["feature_drift"]])
    grades = pd.DataFrame([row for r in results for row in r["grades"]])
    score_dist = pd.DataFrame([row for r in results for row in r["score_dist"]])
    metrics_long = pd.DataFrame([row for r in results for row in r["metrics_long"]])
    alerts = build_alerts(summary, feature_drift)

    t = dirs["tableau"]
    summary.to_csv(t / "tableau_batch_summary.csv", index=False)
    feature_drift.to_csv(t / "tableau_feature_drift.csv", index=False)
    grades.to_csv(t / "tableau_grade_monitoring.csv", index=False)
    score_dist.to_csv(t / "tableau_score_distribution.csv", index=False)
    metrics_long.to_csv(t / "tableau_metrics_long.csv", index=False)
    alerts.to_csv(t / "tableau_alerts.csv", index=False)

    ok = summary[summary["schema_problems"] == ""]
    if len(ok):
        plots.monitoring_overview(ok.reset_index(drop=True), cfg, dirs["figures"] / "monitoring_overview.png")
        plots.csi_heatmap(feature_drift, cfg, dirs["figures"] / "monitoring_feature_csi.png")
    write_monitoring_report(cfg, dirs, bundle["metadata"], summary, alerts)
    print(f"[monitor] {len(summary)} batches monitored, {len(alerts)} alerts raised")
    for _, s in summary.iterrows():
        print(f"  {s['batch_id']:<10} {s['overall_status']:<6} {s['diagnosis']}")
    return {"summary": summary, "alerts": alerts, "feature_drift": feature_drift}


def write_monitoring_report(cfg, dirs, meta, summary: pd.DataFrame, alerts: pd.DataFrame) -> None:
    from .simulate import SCENARIO_NOTES
    simulated = bool(summary["is_simulated"].all())
    cols = ["batch_id", "period_start", "scenario", "labels_available", "score_psi", "max_feature_csi",
            "max_missing_increase", "observed_default_rate", "mean_pd", "calibration_pvalue", "gini",
            "gini_relative_drop", "overall_status"]
    status_cols = ["batch_id", "score_stability_status", "feature_stability_status", "data_quality_status",
                   "default_rate_status", "calibration_status", "performance_status", "overall_status", "diagnosis"]
    mc = cfg["monitoring"]
    lines = [
        "# Model Monitoring Report",
        "",
        "_Generated automatically by `python -m creditrisk monitor`._",
        "",
        ("> **All batches in this report are SIMULATED.** They were created by resampling the held-out UCI "
         "monitoring pool and applying documented scenario changes. They are not real bank or production data."
         if simulated else "> Batches include non-simulated input files."),
        "",
        f"- Monitored model: `{meta['champion']}` using features {meta['champion_features']}",
        f"- Reference (development) default rate: {meta['reference_default_rate']:.4f}; "
        f"benchmark Gini (test set): {meta['benchmark_gini']:.4f}",
        "",
        "## Thresholds (config/config.yaml)",
        "",
        f"| Check | Amber | Red |\n|---|---|---|\n"
        f"| Score PSI / feature CSI | >= {mc['psi']['amber']} | >= {mc['psi']['red']} |\n"
        f"| Missing-rate increase (abs.) | >= {mc['missing_rate_increase']['amber']} | >= {mc['missing_rate_increase']['red']} |\n"
        f"| Default-rate ratio (either direction) | >= {mc['default_rate_ratio']['amber']}x | >= {mc['default_rate_ratio']['red']}x |\n"
        f"| Calibration binomial p-value | < {mc['calibration_pvalue']['amber']} | < {mc['calibration_pvalue']['red']} |\n"
        f"| Relative Gini drop vs benchmark | >= {mc['gini_relative_drop']['amber']:.0%} | >= {mc['gini_relative_drop']['red']:.0%} |",
        "",
        f"Performance metrics need outcome labels and at least {mc['min_defaults_for_performance']} defaults.",
        "",
        "## Scenarios",
        "",
        md_table(pd.DataFrame([{"scenario": k, "what was simulated": v} for k, v in SCENARIO_NOTES.items()
                               if k in set(summary["scenario"])])),
        "",
        "## Batch results",
        "",
        md_table(summary.reindex(columns=cols)),
        "",
        "## Status and diagnosis",
        "",
        md_table(summary.reindex(columns=status_cols)),
        "",
        "**How to read the diagnosis.** *Feature drift* means inputs look different from development. "
        "*Default-rate / calibration shift* means the outcome rate moved and predicted PDs no longer match it. "
        "*Performance deterioration* means the model ranks defaulters worse (Gini falls). "
        "They can occur independently: a population shift can leave Gini intact, and a broken feature-outcome "
        "relationship can lower Gini while every feature distribution looks stable.",
        "",
        "![Monitoring overview](../figures/monitoring_overview.png)",
        "",
        "![Feature CSI](../figures/monitoring_feature_csi.png)",
        "",
        f"## Alerts ({len(alerts)})",
        "",
        md_table(alerts[["alert_id", "batch_id", "severity", "component", "value", "detail"]]) if len(alerts)
        else "No alerts.",
        "",
        "Recommended actions per component are in `outputs/tableau/tableau_alerts.csv` and `docs/RUNBOOK.md`.",
    ]
    (dirs["reports"] / "monitoring_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
