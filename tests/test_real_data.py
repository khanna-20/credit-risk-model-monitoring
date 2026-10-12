"""Checks against the committed UCI file (skipped if it is not present)."""
import pytest

from creditrisk.config import load_config, resolve
from creditrisk.data import FEATURES, ROW_ID, TARGET, read_arff
from creditrisk.train import split_data
from creditrisk.validation import clean_development_data, run_data_quality_checks

CFG = load_config()
PATH = resolve(CFG, CFG["data"]["raw_path"])
pytestmark = pytest.mark.skipif(not PATH.exists(), reason="UCI 5year.arff not present")


@pytest.fixture(scope="module")
def raw():
    return read_arff(PATH)


def test_matches_uci_documentation(raw):
    assert len(raw) == 5910                 # UCI: 5th year, 5,910 instances
    assert int(raw[TARGET].sum()) == 410    # UCI: 410 bankrupt
    assert list(raw.columns) == [ROW_ID] + FEATURES + [TARGET]


def test_no_blocking_data_quality_failures(raw):
    dq = run_data_quality_checks(raw, 5910, 410, CFG["features"]["max_missing_rate"])
    assert not (dq["status"] == "FAIL").any()


def test_splits_are_disjoint_and_stratified(raw):
    clean, info = clean_development_data(raw)
    assert info["rows_out"] == 5850
    train, test, pool = split_data(clean, CFG)
    ids = [set(d[ROW_ID]) for d in (train, test, pool)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert sum(len(i) for i in ids) == 5850
    # no identical firm records across train and test after de-duplication
    tr_keys = set(map(tuple, train[FEATURES].fillna(-999).round(8).to_numpy()))
    te_keys = set(map(tuple, test[FEATURES].fillna(-999).round(8).to_numpy()))
    assert not tr_keys & te_keys
    rates = [d[TARGET].mean() for d in (train, test, pool)]
    assert max(rates) - min(rates) < 0.005
