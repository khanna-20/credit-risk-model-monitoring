import numpy as np
import pytest

from creditrisk.monitoring import diagnose, psi, rag, rag_pvalue, rag_two_sided_ratio, worst


def test_psi_identical_distributions_is_zero():
    x = np.random.default_rng(0).normal(size=5000)
    assert psi(x, x)[0] == pytest.approx(0.0, abs=1e-12)


def test_psi_same_distribution_new_sample_is_small():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(size=5000), rng.normal(size=5000))[0] < 0.02


def test_psi_large_shift_is_red():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(size=5000), rng.normal(1.0, 1, 5000))[0] > 0.25


def test_psi_counts_missing_values():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=5000)
    cur = rng.normal(size=5000)
    cur[:2500] = np.nan
    value, tbl = psi(ref, cur)
    assert value > 0.25
    assert tbl.loc[tbl["bin"] == "missing", "cur_share"].iloc[0] == pytest.approx(0.5)


def test_rag_boundaries():
    assert rag(0.0999, 0.1, 0.25) == "GREEN"
    assert rag(0.10, 0.1, 0.25) == "AMBER"
    assert rag(0.25, 0.1, 0.25) == "RED"
    assert rag(float("nan"), 0.1, 0.25) == "N/A"
    assert rag_two_sided_ratio(2.0, 1.5, 2.0) == "RED"
    assert rag_two_sided_ratio(0.5, 1.5, 2.0) == "RED"      # halving counts as much as doubling
    assert rag_two_sided_ratio(1.6, 1.5, 2.0) == "AMBER"
    assert rag_two_sided_ratio(1.1, 1.5, 2.0) == "GREEN"
    assert rag_pvalue(0.2, 0.05, 0.01) == "GREEN"
    assert rag_pvalue(0.03, 0.05, 0.01) == "AMBER"
    assert rag_pvalue(0.001, 0.05, 0.01) == "RED"
    assert worst("GREEN", "PENDING", "AMBER") == "AMBER"
    assert worst("GREEN", "RED", "AMBER") == "RED"


@pytest.mark.parametrize("args, expected", [
    ((True, True, "GREEN", "GREEN", "GREEN", "GREEN", "GREEN", "GREEN"), "Stable"),
    ((True, False, "GREEN", "GREEN", "GREEN", "PENDING", "PENDING", "PENDING"), "labels pending"),
    ((True, True, "GREEN", "AMBER", "GREEN", "GREEN", "GREEN", "GREEN"), "Feature drift without performance"),
    ((True, False, "AMBER", "GREEN", "GREEN", "PENDING", "PENDING", "PENDING"), "unknown until labels"),
    ((True, True, "GREEN", "GREEN", "GREEN", "RED", "RED", "GREEN"), "Default-rate shift with ranking intact"),
    ((True, True, "GREEN", "GREEN", "GREEN", "GREEN", "RED", "GREEN"), "Calibration shift"),
    ((True, True, "GREEN", "GREEN", "GREEN", "GREEN", "GREEN", "RED"), "without detectable input drift"),
    ((True, True, "AMBER", "GREEN", "GREEN", "GREEN", "AMBER", "RED"), "score drift but no feature-level drift"),
    ((True, True, "AMBER", "RED", "RED", "GREEN", "RED", "GREEN"), "likely cause of the calibration breach"),
    ((False, True, "", "", "", "", "", ""), "Schema failure"),
])
def test_diagnosis_separates_drift_target_and_performance(args, expected):
    assert expected in diagnose(*args)
