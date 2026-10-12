"""Model development: validation -> split -> tuning -> imbalance analysis -> evaluation -> artefacts."""
from __future__ import annotations

import hashlib
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split

from . import plots
from .config import output_dirs, resolve
from .data import FEATURE_DESCRIPTIONS, FEATURES, ROW_ID, TARGET, load_dataset
from .evaluation import (assign_grade, bootstrap_auc, calibration_table, choose_thresholds,
                         classification_at, discrimination_metrics, grade_table, threshold_table)
from .features import build_challenger, build_scorecard, model_features, scorecard_points
from .utils import md_table, write_json
from .validation import clean_development_data, run_data_quality_checks

MODEL_NAMES = {"scorecard_lr": "WoE logistic scorecard", "challenger_gbm": "Gradient boosting (HGB)"}


def split_data(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train / test / monitoring-pool split (done once, after de-duplication)."""
    seed = cfg["project"]["random_seed"]
    pool_share = cfg["split"]["monitoring_pool_share"]
    test_share = cfg["split"]["test_share"]
    dev, pool = train_test_split(df, test_size=pool_share, stratify=df[TARGET], random_state=seed)
    train, test = train_test_split(dev, test_size=test_share / (1 - pool_share),
                                   stratify=dev[TARGET], random_state=seed)
    return train.reset_index(drop=True), test.reset_index(drop=True), pool.reset_index(drop=True)


def model_inputs(cfg: dict) -> list[str]:
    excluded = set(cfg["features"]["excluded"])
    return [c for c in FEATURES if c not in excluded]


def tune(cfg: dict, X: pd.DataFrame, y: pd.Series) -> tuple[dict, pd.DataFrame]:
    """Small grid searches with stratified CV on the training data only (scoring = ROC-AUC)."""
    seed = cfg["project"]["random_seed"]
    cv = StratifiedKFold(cfg["cv_folds"], shuffle=True, random_state=seed)
    best, rows = {}, []
    sc = GridSearchCV(build_scorecard(cfg), {"lr__C": cfg["scorecard"]["C_grid"]},
                      scoring="roc_auc", cv=cv, n_jobs=-1).fit(X, y)
    best["scorecard_lr"] = sc.best_estimator_
    ch = GridSearchCV(build_challenger(cfg), {"gbm__learning_rate": cfg["challenger"]["learning_rate"],
                                              "gbm__max_depth": cfg["challenger"]["max_depth"]},
                      scoring="roc_auc", cv=cv, n_jobs=-1).fit(X, y)
    best["challenger_gbm"] = ch.best_estimator_
    for name, gs in (("scorecard_lr", sc), ("challenger_gbm", ch)):
        res = pd.DataFrame(gs.cv_results_)
        for _, r in res.iterrows():
            rows.append({"model": name, "params": str(r["params"]), "cv_auc_mean": r["mean_test_score"],
                         "cv_auc_std": r["std_test_score"], "selected": r["rank_test_score"] == 1})
    return best, pd.DataFrame(rows)


def imbalance_analysis(cfg: dict, models: dict, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Compare unweighted vs class_weight='balanced' training using out-of-fold predictions."""
    seed = cfg["project"]["random_seed"]
    cv = StratifiedKFold(cfg["cv_folds"], shuffle=True, random_state=seed)
    rows = []
    for name, est in models.items():
        step = "lr" if name == "scorecard_lr" else "gbm"
        for cw in (None, "balanced"):
            m = clone(est).set_params(**{f"{step}__class_weight": cw})
            p = cross_val_predict(m, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
            d = discrimination_metrics(y.to_numpy(), p)
            c = classification_at(y.to_numpy(), p, 0.5)
            rows.append({"model": name, "class_weight": cw or "none", "oof_roc_auc": d["roc_auc"],
                         "oof_pr_auc": d["pr_auc"], "brier": d["brier"], "observed_default_rate": d["default_rate"],
                         "mean_predicted_pd": d["mean_pd"], "recall_at_0.5": c["recall"],
                         "precision_at_0.5": c["precision"]})
    return pd.DataFrame(rows)


def develop(cfg: dict) -> dict:
    """Run the full model-development stage and save artefacts, reports, figures and CSVs."""
    dirs = output_dirs(cfg)
    seed = cfg["project"]["random_seed"]
    raw = load_dataset(cfg)
    raw_path = resolve(cfg, cfg["data"]["raw_path"])
    data_sha256 = hashlib.sha256(raw_path.read_bytes()).hexdigest() if raw_path.exists() else "n/a"

    # 1. Data-quality validation -------------------------------------------------
    dq = run_data_quality_checks(raw, cfg["data"].get("expected_rows"), cfg["data"].get("expected_defaults"),
                                 cfg["features"]["max_missing_rate"])
    dq.to_csv(dirs["reports"] / "data_quality_checks.csv", index=False)
    if (dq["status"] == "FAIL").any():
        raise RuntimeError("Data-quality FAIL - see outputs/reports/data_quality_checks.csv")
    df, clean_info = clean_development_data(raw)
    print(f"[develop] rows {clean_info['rows_in']} -> {clean_info['rows_out']} after de-duplication")

    # 2. Split ------------------------------------------------------------------------
    train, test, pool = split_data(df, cfg)
    inputs = model_inputs(cfg)
    X_tr, y_tr = train[inputs], train[TARGET]
    X_te, y_te = test[inputs], test[TARGET]
    split_summary = pd.DataFrame([{"split": n, "rows": len(d), "defaults": int(d[TARGET].sum()),
                                   "default_rate": d[TARGET].mean()}
                                  for n, d in (("train (development)", train), ("test", test),
                                               ("monitoring pool", pool))])

    # 3. Tuning + imbalance analysis (training data only) ---------------------------
    best, tuning = tune(cfg, X_tr, y_tr)
    tuning.to_csv(dirs["reports"] / "cv_tuning_results.csv", index=False)
    imbalance = imbalance_analysis(cfg, best, X_tr, y_tr)
    imbalance.to_csv(dirs["reports"] / "class_imbalance_analysis.csv", index=False)

    # 4. Threshold selection on out-of-fold TRAINING predictions of the champion -----
    champ = cfg["champion"]
    cv = StratifiedKFold(cfg["cv_folds"], shuffle=True, random_state=seed)
    oof = cross_val_predict(clone(best[champ]), X_tr, y_tr, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    tcfg = cfg["threshold"]
    oof_tbl = threshold_table(y_tr.to_numpy(), oof, tcfg["cost_false_negative"], tcfg["cost_false_positive"])
    chosen = choose_thresholds(oof_tbl)

    # 5. Final fit on all training rows; evaluate once on the untouched test set -----
    fitted = {name: est.fit(X_tr, y_tr) for name, est in best.items()}
    p_te = {name: m.predict_proba(X_te)[:, 1] for name, m in fitted.items()}
    p_tr = {name: m.predict_proba(X_tr)[:, 1] for name, m in fitted.items()}
    metrics_rows = []
    for name in fitted:
        for split_name, yy, pp in (("train", y_tr, p_tr[name]), ("test", y_te, p_te[name])):
            m = discrimination_metrics(yy.to_numpy(), pp)
            metrics_rows.append({"model": name, "split": split_name, **m})
    metrics = pd.DataFrame(metrics_rows)
    other = "challenger_gbm" if champ == "scorecard_lr" else "scorecard_lr"
    boot = bootstrap_auc(y_te.to_numpy(), p_te[champ], p_te[other], reps=cfg["bootstrap_reps"], seed=seed)
    boot_other = bootstrap_auc(y_te.to_numpy(), p_te[other], reps=cfg["bootstrap_reps"], seed=seed)
    metrics["auc_ci_low"] = np.nan
    metrics["auc_ci_high"] = np.nan
    for name, b in ((champ, boot), (other, boot_other)):
        mask = (metrics["model"] == name) & (metrics["split"] == "test")
        metrics.loc[mask, ["auc_ci_low", "auc_ci_high"]] = [b["auc_ci_low"], b["auc_ci_high"]]
    metrics.to_csv(dirs["reports"] / "model_metrics.csv", index=False)

    test_tbl = threshold_table(y_te.to_numpy(), p_te[champ], tcfg["cost_false_negative"], tcfg["cost_false_positive"])
    test_tbl.to_csv(dirs["reports"] / "threshold_analysis_test.csv", index=False)
    oof_tbl.to_csv(dirs["reports"] / "threshold_analysis_train_oof.csv", index=False)
    at_thresholds = pd.DataFrame([
        {"model": name, "rule": rule, **classification_at(y_te.to_numpy(), p_te[name], t)}
        for name in fitted for rule, t in (("cost_optimal", chosen["cost_optimal"]),
                                           ("f1_optimal", chosen["f1_optimal"]), ("fixed_0.5", 0.5))])
    at_thresholds.to_csv(dirs["reports"] / "classification_at_thresholds_test.csv", index=False)

    calib = pd.concat([calibration_table(y_te, p_te[n]).assign(model=n) for n in fitted])
    calib.to_csv(dirs["reports"] / "calibration_test.csv", index=False)
    grades = pd.concat([grade_table(y_te, p_te[n], cfg["rating_scale"]).assign(model=n) for n in fitted])
    grades.to_csv(dirs["reports"] / "rating_grades_test.csv", index=False)

    woe = fitted["scorecard_lr"].named_steps["woe"]
    iv = pd.DataFrame({"feature": list(woe.iv_), "iv": list(woe.iv_.values())})
    iv["description"] = iv["feature"].map(FEATURE_DESCRIPTIONS)
    iv["selected"] = iv["feature"].isin(woe.selected_)
    iv = iv.sort_values("iv", ascending=False)
    iv.to_csv(dirs["reports"] / "information_value.csv", index=False)
    woe.bin_table_.to_csv(dirs["reports"] / "woe_bins.csv", index=False)
    points = scorecard_points(fitted["scorecard_lr"])
    points.to_csv(dirs["reports"] / "scorecard_points.csv", index=False)
    lr = fitted["scorecard_lr"].named_steps["lr"]
    coefs = pd.DataFrame({"feature": woe.selected_, "coefficient": lr.coef_[0],
                          "iv": [woe.iv_[f] for f in woe.selected_]})
    coefs["description"] = coefs["feature"].map(FEATURE_DESCRIPTIONS)

    # 5b. Sensitivity: how much does the champion depend on individual high-IV features?
    base_auc = float(metrics[(metrics["model"] == champ) & (metrics["split"] == "test")]["roc_auc"].iloc[0])
    sens_rows = [{"variant": "champion (all features)", "excluded": "", "test_roc_auc": base_auc,
                  "test_gini": 2 * base_auc - 1, "auc_change": 0.0}]
    for feat in cfg.get("sensitivity_exclude", []):
        cols = [c for c in inputs if c != feat]
        m = clone(fitted[champ]).fit(X_tr[cols], y_tr)
        d = discrimination_metrics(y_te.to_numpy(), m.predict_proba(X_te[cols])[:, 1])
        sens_rows.append({"variant": f"champion without {feat}", "excluded": feat, "test_roc_auc": d["roc_auc"],
                          "test_gini": d["gini"], "auc_change": d["roc_auc"] - base_auc})
    sensitivity = pd.DataFrame(sens_rows)
    sensitivity.to_csv(dirs["reports"] / "sensitivity_analysis.csv", index=False)

    # 6. Reference sample for monitoring (development = training data) --------------
    feats = model_features(fitted[champ])
    reference = train[[ROW_ID] + inputs + [TARGET]].copy()
    reference["pd"] = p_tr[champ]
    reference["grade"] = assign_grade(reference["pd"], cfg["rating_scale"])
    test_m = metrics[(metrics["model"] == champ) & (metrics["split"] == "test")].iloc[0]
    metadata = {
        "champion": champ, "challenger": other,
        "champion_features": feats,
        "champion_params": str(best[champ].get_params()[("lr__C" if champ == "scorecard_lr" else "gbm__learning_rate")]),
        "excluded_features": cfg["features"]["excluded"],
        "dropped_high_missing": fitted[champ].named_steps["missing_filter"].dropped_,
        "thresholds": chosen,
        "benchmark_gini": float(test_m["gini"]), "benchmark_auc": float(test_m["roc_auc"]),
        "reference_default_rate": float(y_tr.mean()),
        "reference_missing_rates": train[feats].isna().mean().to_dict(),
        "data_file_sha256": data_sha256, "clean_info": clean_info,
        "rows": {"train": len(train), "test": len(test), "monitoring_pool": len(pool)},
        "random_seed": seed,
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
    }
    write_json(metadata, dirs["model"] / "model_metadata.json")
    joblib.dump({"models": fitted, "metadata": metadata, "reference": reference,
                 "monitoring_pool": pool[[ROW_ID] + inputs + [TARGET]]}, dirs["model"] / "model_bundle.joblib")

    # 7. Figures + report ---------------------------------------------------------------
    fig = dirs["figures"]
    plots.roc_pr_curves(y_te.to_numpy(), p_te, MODEL_NAMES, fig / "roc_pr_curves.png")
    plots.calibration_plot(calib, MODEL_NAMES, fig / "calibration_test.png")
    plots.threshold_plot(test_tbl, chosen, fig / "threshold_tradeoff_test.png")
    plots.grade_plot(grades[grades["model"] == champ], fig / "rating_grades_test.png")
    plots.iv_plot(iv.head(20), fig / "information_value.png")
    cm_row = at_thresholds[(at_thresholds["model"] == champ) & (at_thresholds["rule"] == "cost_optimal")].iloc[0]
    plots.confusion_plot(cm_row, fig / "confusion_matrix_test.png")

    write_development_report(cfg, dirs, dq, clean_info, split_summary, tuning, imbalance, metrics, boot,
                             chosen, at_thresholds, calib, grades, coefs, metadata, sensitivity,
                             woe.bin_table_)
    print(f"[develop] champion={champ}  test AUC={test_m['roc_auc']:.4f}  Gini={test_m['gini']:.4f}")
    return {"metrics": metrics, "metadata": metadata, "bootstrap": boot, "imbalance": imbalance,
            "thresholds": chosen, "sensitivity": sensitivity}


def write_development_report(cfg, dirs, dq, clean_info, split_summary, tuning, imbalance, metrics, boot,
                             chosen, at_thresholds, calib, grades, coefs, meta, sensitivity=None,
                             bins=None) -> None:
    champ, other = meta["champion"], meta["challenger"]
    t = metrics[metrics["split"] == "test"].set_index("model")
    tr = metrics[metrics["split"] == "train"].set_index("model")
    diff_sig = boot["diff_ci_low"] > 0 or boot["diff_ci_high"] < 0
    better = champ if boot["diff_mean"] > 0 else other
    if diff_sig:
        verdict = (f"The paired bootstrap 95% CI for AUC({champ}) - AUC({other}) is "
                   f"[{boot['diff_ci_low']:.4f}, {boot['diff_ci_high']:.4f}], which excludes zero: "
                   f"**{better}** ranks firms measurably better on this test set.")
    else:
        verdict = (f"The paired bootstrap 95% CI for AUC({champ}) - AUC({other}) is "
                   f"[{boot['diff_ci_low']:.4f}, {boot['diff_ci_high']:.4f}], which includes zero: "
                   "the difference is not statistically clear on this test set.")
    cw = imbalance.set_index(["model", "class_weight"])
    sc_none, sc_bal = cw.loc[("scorecard_lr", "none")], cw.loc[("scorecard_lr", "balanced")]
    cmc = at_thresholds[(at_thresholds["model"] == champ)]
    c = cfg["threshold"]
    lines = [
        "# Model Development Report",
        "",
        "_Generated automatically by `python -m creditrisk develop`. All numbers below come from this run._",
        "",
        "## 1. Data",
        "",
        "- Source: UCI Machine Learning Repository, *Polish Companies Bankruptcy* (Tomczak, 2016), "
        "DOI [10.24432/C5F600](https://doi.org/10.24432/C5F600), CC BY 4.0.",
        f"- File used: `{cfg['data']['horizon_file']}`: financial ratios from the 5th year of the forecasting "
        "period, label = bankruptcy status **after 1 year** (12-month horizon). The five horizon files are not merged.",
        f"- SHA-256 of the file used: `{meta['data_file_sha256']}`",
        f"- Rows: {clean_info['rows_in']} loaded; {clean_info['exact_duplicates_removed']} exact duplicates and "
        f"{clean_info['conflicting_rows_removed']} conflicting-label rows removed; {clean_info['rows_out']} used.",
        f"- Excluded as a leakage precaution: {meta['excluded_features']} (documented as gross profit *in 3 years* / TA).",
        "",
        "### Data-quality checks",
        "",
        md_table(dq),
        "",
        "### Split (stratified, seed {})".format(meta["random_seed"]),
        "",
        md_table(split_summary),
        "",
        "There is no date or company identifier in the file, so an out-of-time split is not possible. "
        "The monitoring pool is held out from all model fitting and is only used to build *simulated* batches.",
        "",
        "## 2. Models",
        "",
        f"- **Champion - {MODEL_NAMES['scorecard_lr']}**: quantile binning ({cfg['scorecard']['n_bins']} bins + a "
        "missing bin), Weight of Evidence, Information Value >= "
        f"{cfg['scorecard']['min_iv']}, correlation filter |r| <= {cfg['scorecard']['max_corr']}, "
        f"max {cfg['scorecard']['max_features']} features, logistic regression. Bins, WoE and selection are fitted "
        "inside each CV fold, so no test information is used.",
        f"- **Challenger - {MODEL_NAMES['challenger_gbm']}**: scikit-learn HistGradientBoosting on raw ratios "
        "(handles missing values natively; trees are insensitive to the extreme outliers in the ratios).",
        "",
        "### Cross-validated tuning (training data, ROC-AUC)",
        "",
        md_table(tuning),
        "",
        "### Scorecard features and coefficients",
        "",
        "WoE is risk-oriented (positive = riskier than average), so positive coefficients are expected.",
        "",
        md_table(coefs),
        "",
        "## 3. Class imbalance",
        "",
        f"The default rate is about {tr.loc[champ, 'default_rate']:.1%}. Out-of-fold results on the training data:",
        "",
        md_table(imbalance),
        "",
        f"For the scorecard, `class_weight='balanced'` changes out-of-fold ROC-AUC from {sc_none['oof_roc_auc']:.4f} "
        f"to {sc_bal['oof_roc_auc']:.4f}, but moves the mean predicted PD from {sc_none['mean_predicted_pd']:.4f} to "
        f"{sc_bal['mean_predicted_pd']:.4f} against an observed default rate of {sc_none['observed_default_rate']:.4f}. "
        "Re-weighting mainly shifts probabilities rather than improving ranking, and a PD model must stay calibrated, "
        "so the models are trained **unweighted** and imbalance is handled through the decision threshold instead. "
        "No resampling (e.g. SMOTE) is used.",
        "",
        "## 4. Test-set performance (untouched hold-out)",
        "",
        md_table(metrics[metrics["split"] == "test"].drop(columns=["split"])),
        "",
        f"Bootstrap ({cfg['bootstrap_reps']} reps) 95% CI for champion test AUC: "
        f"[{boot['auc_ci_low']:.4f}, {boot['auc_ci_high']:.4f}].",
        "",
        verdict,
        "",
        f"Train vs test AUC: champion {tr.loc[champ, 'roc_auc']:.4f} vs {t.loc[champ, 'roc_auc']:.4f}; "
        f"challenger {tr.loc[other, 'roc_auc']:.4f} vs {t.loc[other, 'roc_auc']:.4f} "
        "(a large gap indicates over-fitting).",
        "",
        "**Why the scorecard is the monitored champion:** each feature's effect can be read from its bins, points "
        "and coefficient, which is what credit analysts and model validators need to review. The challenger is "
        "kept as a benchmark; if it is materially better, that gap is itself a finding worth escalating "
        "(the scorecard may be missing non-linear effects).",
        "",
        "![ROC and PR curves](../figures/roc_pr_curves.png)",
        "",
        "## 5. Calibration",
        "",
        "Observed default rate vs mean predicted PD by predicted-PD decile (test set):",
        "",
        md_table(calib[["model", "bucket", "n", "defaults", "mean_pd", "observed_dr"]]),
        "",
        "![Calibration](../figures/calibration_test.png)",
        "",
        "## 6. Threshold analysis",
        "",
        f"**Assumptions (illustrative, not bank figures):** a missed defaulter (false negative) costs "
        f"{c['cost_false_negative']} units; an unnecessary review of a healthy firm (false positive) costs "
        f"{c['cost_false_positive']} unit. Thresholds were chosen on **out-of-fold training predictions**, "
        "then applied once to the test set.",
        "",
        f"- Cost-optimal threshold (train OOF): **{chosen['cost_optimal']:.2f}**",
        f"- F1-optimal threshold (train OOF): **{chosen['f1_optimal']:.2f}**",
        "",
        md_table(cmc[["rule", "threshold", "tp", "fp", "fn", "tn", "precision", "recall", "f1", "flag_rate"]]),
        "",
        "A lower threshold catches more defaulters (higher recall) at the price of more false alarms. "
        "With a 10:1 cost ratio, the cost-optimal threshold is far below 0.5.",
        "",
        "![Threshold trade-off](../figures/threshold_tradeoff_test.png)",
        "![Confusion matrix](../figures/confusion_matrix_test.png)",
        "",
        "## 7. Rating grades (illustrative master scale)",
        "",
        md_table(grades[grades["model"] == champ].drop(columns=["model"])),
        "",
        "Rank-ordering check: observed default rates should rise from G1 to G7. Grades with few firms give "
        "noisy rates, so read the binomial p-values together with the counts.",
        "",
        "![Rating grades](../figures/rating_grades_test.png)",
        "",
        "## 8. Sensitivity: dependence on high-IV features",
        "",
        "In scorecard practice an Information Value above about 0.5 is treated as 'suspiciously strong' and "
        "worth investigating. Several ratios here exceed that (see `information_value.csv`).",
        "",
        *_missing_bin_note(bins, "X21"),
        md_table(sensitivity) if sensitivity is not None else "",
        "",
        "If removing a single feature changes AUC only slightly, the model does not hinge on it. A large drop "
        "would mean its performance relies on that one (possibly sampling-driven) signal.",
        "",
        "## 9. Limitations",
        "",
        "- Polish companies, 2000-2013 data; bankrupt and surviving firms were sampled over different periods "
        "(2000-2012 vs 2007-2013), so some patterns (e.g. missing prior-year sales, X21) may partly reflect data "
        "collection rather than credit risk.",
        "- No dates or firm identifiers, so no out-of-time or out-of-sample-firm validation is possible.",
        "- Bankruptcy is used as the default definition; bank default definitions (e.g. 90 days past due) differ.",
        "- The cost ratio and rating master scale are illustrative assumptions.",
    ]
    (dirs["reports"] / "model_development_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _missing_bin_note(bins, feature: str) -> list[str]:
    """One plain-English sentence on how much of a feature's IV comes from its missing-value bin."""
    if bins is None or feature not in set(bins["feature"]):
        return []
    fb = bins[bins["feature"] == feature]
    mb = fb[fb["bin"] == -1]
    if mb.empty:
        return []
    r = mb.iloc[0]
    return [f"{feature}: in the training data {int(r['n'])} firms have a missing value and {int(r['defaults'])} of "
            f"them defaulted ({r['default_rate']:.1%}). That missing bin contributes {r['iv_contribution']:.2f} of "
            f"{feature}'s total IV of {fb['iv_contribution'].sum():.2f}. Because bankrupt and surviving firms were "
            "sampled over different periods, this may partly be a data-collection artefact rather than pure credit "
            "signal.", ""]
