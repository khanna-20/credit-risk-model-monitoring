"""End-to-end run of develop -> simulate -> monitor on a small synthetic dataset in a temp folder."""
from __future__ import annotations

import joblib
import pandas as pd
import pytest

from creditrisk.data import ROW_ID
from creditrisk.monitoring import run_monitoring
from creditrisk.simulate import simulate_batches
from creditrisk.train import develop

from .conftest import make_synthetic, small_config, write_arff

TABLEAU_FILES = ["tableau_batch_summary.csv", "tableau_feature_drift.csv", "tableau_grade_monitoring.csv",
                 "tableau_score_distribution.csv", "tableau_metrics_long.csv", "tableau_alerts.csv"]


def _run(root):
    (root / "data" / "raw").mkdir(parents=True)
    write_arff(make_synthetic(), root / "data" / "raw" / "synthetic.arff")
    cfg = small_config(root)
    dev = develop(cfg)
    bundle = joblib.load(root / "outputs" / "model" / "model_bundle.joblib")
    simulate_batches(cfg, bundle)
    mon = run_monitoring(cfg)
    return cfg, dev, bundle, mon


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("e2e"))


def test_expected_outputs_exist(run):
    cfg, *_ = run
    out = __import__("pathlib").Path(cfg["_root"]) / "outputs"
    for f in TABLEAU_FILES:
        assert (out / "tableau" / f).exists(), f
    for f in ["model_development_report.md", "monitoring_report.md", "data_quality_checks.csv",
              "model_metrics.csv", "class_imbalance_analysis.csv", "threshold_analysis_test.csv",
              "scorecard_points.csv", "sensitivity_analysis.csv"]:
        assert (out / "reports" / f).exists(), f
    for f in ["roc_pr_curves.png", "calibration_test.png", "monitoring_overview.png", "monitoring_feature_csi.png"]:
        assert (out / "figures" / f).exists(), f


def test_model_learns_signal(run):
    _, dev, *_ = run
    m = dev["metrics"]
    test_auc = m[(m["model"] == "scorecard_lr") & (m["split"] == "test")]["roc_auc"].iloc[0]
    assert test_auc > 0.75


def test_thresholds_chosen_from_training_predictions(run):
    _, dev, *_ = run
    t = dev["thresholds"]
    assert 0 < t["cost_optimal"] <= t["f1_optimal"] < 1  # 10:1 cost ratio pushes the threshold lower


def test_reference_and_monitoring_pool_do_not_overlap(run):
    _, _, bundle, _ = run
    ref_ids = set(bundle["reference"][ROW_ID])
    pool_ids = set(bundle["monitoring_pool"][ROW_ID])
    assert ref_ids and pool_ids and not ref_ids & pool_ids


def test_every_tableau_file_is_labelled_simulated(run):
    cfg, *_ = run
    tdir = __import__("pathlib").Path(cfg["_root"]) / "outputs" / "tableau"
    for f in TABLEAU_FILES:
        df = pd.read_csv(tdir / f)
        assert "is_simulated" in df.columns, f
        assert df["is_simulated"].all(), f


def test_labels_pending_batch_skips_performance(run):
    *_, mon = run
    s = mon["summary"].set_index("batch_id")
    assert not s.loc["SIM_M08", "labels_available"]
    assert s.loc["SIM_M08", "performance_status"] == "PENDING"
    assert pd.isna(s.loc["SIM_M08", "gini"])


def test_relationship_break_lowers_gini(run):
    *_, mon = run
    s = mon["summary"].set_index("batch_id")
    baseline_gini = s.loc[["SIM_M01", "SIM_M02", "SIM_M07"], "gini"].mean()
    assert s.loc["SIM_M06", "gini"] < baseline_gini - 0.2
    assert s.loc["SIM_M06", "performance_status"] in ("AMBER", "RED")


def test_missing_spike_raises_data_quality_alert(run):
    *_, mon = run
    s = mon["summary"].set_index("batch_id")
    assert s.loc["SIM_M05", "data_quality_status"] == "RED"
    alerts = mon["alerts"]
    assert ((alerts["batch_id"] == "SIM_M05") & (alerts["component"] == "data_quality")).any()


def test_default_rate_increase_flags_target_not_ranking(run):
    *_, mon = run
    s = mon["summary"].set_index("batch_id")
    assert s.loc["SIM_M04", "default_rate_status"] in ("AMBER", "RED")


def test_schema_failure_batch_is_rejected(run, tmp_path):
    cfg, *_ = run
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    batch = pd.read_csv(__import__("pathlib").Path(cfg["_root"]) / "data" / "simulated" / "simulated_batch_SIM_M01.csv")
    batch.drop(columns=["X1"]).assign(batch_id="BROKEN").to_csv(bad_dir / "broken.csv", index=False)
    res = run_monitoring(cfg, bad_dir)
    row = res["summary"].iloc[0]
    assert row["overall_status"] == "RED" and "Schema failure" in row["diagnosis"]
    assert (res["alerts"]["component"] == "schema").any()


def test_pipeline_is_reproducible(run, tmp_path):
    _, dev, *_ = run
    _, dev2, *_ = _run(tmp_path)
    cols = ["model", "split", "roc_auc", "pr_auc", "brier"]
    pd.testing.assert_frame_equal(dev["metrics"][cols], dev2["metrics"][cols])
