import numpy as np
import pandas as pd

from creditrisk.data import FEATURES, TARGET
from creditrisk.validation import clean_development_data, run_data_quality_checks, validate_batch_schema


def statuses(dq: pd.DataFrame) -> dict:
    return dict(zip(dq["check"], dq["status"]))


def test_clean_synthetic_data_has_no_failures(synthetic_df):
    dq = run_data_quality_checks(synthetic_df)
    assert not (dq["status"] == "FAIL").any()


def test_missing_required_column_fails(synthetic_df):
    dq = run_data_quality_checks(synthetic_df.drop(columns=["X5"]))
    assert statuses(dq)["schema_required_columns"] == "FAIL"


def test_non_binary_target_fails(synthetic_df):
    df = synthetic_df.copy()
    df.loc[0, TARGET] = 2
    assert statuses(run_data_quality_checks(df))["target_binary"] == "FAIL"


def test_row_count_mismatch_warns(synthetic_df):
    dq = run_data_quality_checks(synthetic_df, expected_rows=len(synthetic_df) + 1)
    assert statuses(dq)["row_count"] == "WARN"


def test_duplicates_and_conflicting_labels_are_removed(synthetic_df):
    df = synthetic_df.copy()
    exact = df.iloc[[0, 1]].copy()                   # 2 exact duplicates
    conflict = df.iloc[[2]].copy()
    conflict[TARGET] = 1 - conflict[TARGET]           # same features, opposite label
    df = pd.concat([df, exact, conflict], ignore_index=True)
    dq = run_data_quality_checks(df)
    assert statuses(dq)["exact_duplicate_rows"] == "WARN"
    assert statuses(dq)["duplicates_with_conflicting_labels"] == "WARN"
    clean, info = clean_development_data(df)
    assert info["exact_duplicates_removed"] == 2
    assert info["conflicting_rows_removed"] == 2      # both versions of the conflicting firm dropped
    assert not clean.duplicated(subset=FEATURES).any()


def test_infinite_values_become_missing(synthetic_df):
    df = synthetic_df.copy()
    df.loc[5, "X3"] = np.inf
    assert statuses(run_data_quality_checks(df))["infinite_values"] == "WARN"
    clean, _ = clean_development_data(df)
    assert not np.isinf(clean[FEATURES].to_numpy()).any()


def test_batch_schema_validation(synthetic_df):
    assert validate_batch_schema(synthetic_df, FEATURES) == []
    problems = validate_batch_schema(synthetic_df.drop(columns=["X1"]), FEATURES)
    assert any("missing required columns" in p for p in problems)
    bad = synthetic_df.copy()
    bad["X2"] = "text"
    assert any("not numeric" in p for p in validate_batch_schema(bad, FEATURES))
