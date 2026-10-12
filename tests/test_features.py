import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from creditrisk.data import TARGET
from creditrisk.features import MissingRateFilter, WoEScorecardTransformer


@pytest.fixture
def split(synthetic_df):
    X = synthetic_df[["X1", "X2", "X21", "X40", "X50"]]
    y = synthetic_df[TARGET]
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)


def test_informative_features_have_higher_iv_than_noise(split):
    X_tr, _, y_tr, _ = split
    woe = WoEScorecardTransformer(min_iv=0.0).fit(X_tr, y_tr)
    assert woe.iv_["X1"] > 0.3 and woe.iv_["X2"] > 0.1
    assert woe.iv_["X40"] < 0.1 and woe.iv_["X50"] < 0.1


def test_woe_sign_is_risk_oriented(split):
    X_tr, _, y_tr, _ = split
    woe = WoEScorecardTransformer(min_iv=0.0).fit(X_tr, y_tr)
    tbl = woe.bin_table_[(woe.bin_table_["feature"] == "X1") & (woe.bin_table_["bin"] >= 0)]
    # X1 lowers risk, so the lowest-X1 bin must have the highest (riskiest) WoE
    assert tbl.sort_values("bin")["woe"].iloc[0] > tbl.sort_values("bin")["woe"].iloc[-1]


def test_missing_values_get_their_own_bin(split):
    X_tr, X_te, y_tr, _ = split
    woe = WoEScorecardTransformer(min_iv=0.0).fit(X_tr, y_tr)
    miss_bin = woe.bin_table_[(woe.bin_table_["feature"] == "X21") & (woe.bin_table_["bin"] == -1)]
    assert len(miss_bin) == 1 and miss_bin["woe"].iloc[0] > 1.0  # missing X21 is a strong risk signal here
    out = woe.transform(X_te)
    assert not out.isna().any().any()


def test_unseen_missing_maps_to_neutral_woe(split):
    X_tr, X_te, y_tr, _ = split
    X_tr = X_tr.copy()
    X_tr["X40"] = X_tr["X40"].fillna(0)
    woe = WoEScorecardTransformer(min_iv=0.0, max_corr=1.0).fit(X_tr, y_tr)
    X_te = X_te.copy()
    X_te.iloc[:10, X_te.columns.get_loc("X40")] = np.nan
    assert (woe.transform(X_te)["X40"].iloc[:10] == 0.0).all()


def test_transform_does_not_refit(split):
    """Bins and WoE learned on training data must not change when test data is transformed."""
    X_tr, X_te, y_tr, _ = split
    woe = WoEScorecardTransformer(min_iv=0.0).fit(X_tr, y_tr)
    before = woe.bin_table_.copy()
    edges_before = {k: v.copy() for k, v in woe.edges_.items()}
    woe.transform(X_te * 5)
    pd.testing.assert_frame_equal(before, woe.bin_table_)
    for k in edges_before:
        np.testing.assert_array_equal(edges_before[k], woe.edges_[k])


def test_correlation_filter_drops_duplicate_feature(split):
    X_tr, _, y_tr, _ = split
    X = X_tr.copy()
    X["X1_copy"] = X["X1"] * 2 + 1
    woe = WoEScorecardTransformer(min_iv=0.0, max_corr=0.7).fit(X, y_tr)
    assert not {"X1", "X1_copy"} <= set(woe.selected_)


def test_iv_threshold_and_max_features(split):
    X_tr, _, y_tr, _ = split
    woe = WoEScorecardTransformer(min_iv=0.1, max_features=2).fit(X_tr, y_tr)
    assert len(woe.selected_) <= 2
    assert all(woe.iv_[f] >= 0.1 for f in woe.selected_)


def test_missing_rate_filter_learns_from_training_only():
    tr = pd.DataFrame({"a": [1, np.nan, np.nan, np.nan], "b": [1, 2, 3, 4]})
    te = pd.DataFrame({"a": [1, 2, 3, 4], "b": [np.nan] * 4})
    f = MissingRateFilter(0.5).fit(tr)
    assert f.kept_ == ["b"]
    assert list(f.transform(te).columns) == ["b"]
