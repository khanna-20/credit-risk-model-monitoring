import numpy as np
import pytest

from creditrisk.evaluation import (assign_grade, calibration_table, choose_thresholds, classification_at,
                                   discrimination_metrics, grade_table, ks_statistic, threshold_table)

SCALE = [{"grade": "G1", "upper_pd": 0.005}, {"grade": "G2", "upper_pd": 0.01}, {"grade": "G3", "upper_pd": 0.02},
         {"grade": "G4", "upper_pd": 0.05}, {"grade": "G5", "upper_pd": 0.10}, {"grade": "G6", "upper_pd": 0.20},
         {"grade": "G7", "upper_pd": 1.01}]


def test_perfect_separation_gives_gini_and_ks_of_one():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    m = discrimination_metrics(y, p)
    assert m["gini"] == pytest.approx(1.0)
    assert ks_statistic(y, p) == pytest.approx(1.0)


def test_random_scores_give_gini_near_zero():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 5000)
    assert abs(discrimination_metrics(y, rng.random(5000))["gini"]) < 0.05


def test_confusion_counts_at_threshold():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.2, 0.6, 0.4, 0.9])
    r = classification_at(y, p, 0.5)
    assert (r["tn"], r["fp"], r["fn"], r["tp"]) == (1, 1, 1, 1)
    assert r["precision"] == pytest.approx(0.5) and r["recall"] == pytest.approx(0.5)


def test_threshold_table_expected_cost_and_choice():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.2, 0.6, 0.4, 0.9])
    tbl = threshold_table(y, p, cost_fn=10, cost_fp=1, thresholds=np.array([0.3, 0.5]))
    # t=0.3: FP=1, FN=0 -> cost 1/4 ; t=0.5: FP=1, FN=1 -> cost 11/4
    assert tbl["expected_cost_per_firm"].tolist() == pytest.approx([0.25, 2.75])
    assert choose_thresholds(tbl)["cost_optimal"] == pytest.approx(0.3)


def test_grade_boundaries():
    g = assign_grade(np.array([0.0, 0.0049, 0.005, 0.0199, 0.02, 0.5, 0.999]), SCALE)
    assert g.tolist() == ["G1", "G1", "G2", "G3", "G4", "G7", "G7"]


def test_grade_table_and_calibration_table_shapes():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 0.3, 2000)
    y = (rng.random(2000) < p).astype(int)
    gt = grade_table(y, p, SCALE)
    assert len(gt) == 7 and gt["n"].sum() == 2000
    assert {"observed_dr", "binomial_p"} <= set(gt.columns)
    ct = calibration_table(y, p, n_bins=10)
    assert len(ct) == 10 and ct["n"].sum() == 2000
    # data drawn from p itself, so it should be close to calibrated overall
    assert abs((ct["defaults"].sum() / 2000) - p.mean()) < 0.02
