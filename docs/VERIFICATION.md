# Verification log

This file records what was actually run and what the results were. It separates **verified** results from **not verified** items.

## Environment matrix (verified)
Each environment was created from scratch: the repository was copied without `outputs/` and without simulated batches, a fresh `venv` was created, then `pip install -r requirements.txt`, then `python -m creditrisk all`, then `pytest`.

| Python | OS | `python -m creditrisk all` | `pytest` |
|---|---|---|---|
| 3.11.17 | Linux x86-64 | completed | 50 passed |
| 3.12.3 | Linux x86-64 | completed | 50 passed |

`outputs/reports/model_metrics.csv` and `outputs/tableau/tableau_batch_summary.csv` were **byte-identical** across both supported Python versions and identical to the committed outputs, so results are reproducible with the pinned dependencies and seed 42.

**Supported Python versions: 3.11 and 3.12.** The pinned scikit-learn 1.5.2 declares support for Python 3.10-3.12 in its package metadata, so the GitHub Actions matrix tests 3.11 and 3.12 only. (An extra run on Python 3.13.16/Linux also completed with 50 tests passing, but 3.13 is outside scikit-learn 1.5.2's declared support and is not a supported target for this project.)

## Key results reproduced in every environment
- Champion (WoE scorecard): test ROC-AUC 0.8974, Gini 0.7948
- 8 simulated batches monitored, 10 alerts raised, diagnoses as listed in the README

## Not verified
- **Windows and macOS** were not tested. The code uses only cross-platform libraries and `pathlib`, but run `pytest` once after installing to confirm.
- **Automatic UCI download** (used only if `data/raw/5year.arff` is missing) could not be tested because UCI was unreachable from the build environment.
- **GitHub Actions workflow** (`.github/workflows/tests.yml`) has not run yet; it runs after the first push. Check the *Actions* tab.
- **Tableau dashboard:** not built. `docs/TABLEAU_GUIDE.md` gives the steps.

## Test results (pytest -v)
```
tests/test_evaluation.py::test_perfect_separation_gives_gini_and_ks_of_one PASSED
tests/test_evaluation.py::test_random_scores_give_gini_near_zero PASSED
tests/test_evaluation.py::test_confusion_counts_at_threshold PASSED
tests/test_evaluation.py::test_threshold_table_expected_cost_and_choice PASSED
tests/test_evaluation.py::test_grade_boundaries PASSED
tests/test_evaluation.py::test_grade_table_and_calibration_table_shapes PASSED
tests/test_features.py::test_informative_features_have_higher_iv_than_noise PASSED
tests/test_features.py::test_woe_sign_is_risk_oriented PASSED
tests/test_features.py::test_missing_values_get_their_own_bin PASSED
tests/test_features.py::test_unseen_missing_maps_to_neutral_woe PASSED
tests/test_features.py::test_transform_does_not_refit PASSED
tests/test_features.py::test_correlation_filter_drops_duplicate_feature PASSED
tests/test_features.py::test_iv_threshold_and_max_features PASSED
tests/test_features.py::test_missing_rate_filter_learns_from_training_only PASSED
tests/test_monitoring_stats.py::test_psi_identical_distributions_is_zero PASSED
tests/test_monitoring_stats.py::test_psi_same_distribution_new_sample_is_small PASSED
tests/test_monitoring_stats.py::test_psi_large_shift_is_red PASSED
tests/test_monitoring_stats.py::test_psi_counts_missing_values PASSED
tests/test_monitoring_stats.py::test_rag_boundaries PASSED
tests/test_monitoring_stats.py::test_diagnosis_separates_drift_target_and_performance[args0-Stable] PASSED
tests/test_monitoring_stats.py::test_diagnosis_separates_drift_target_and_performance[args1-labels pending] PASSED
tests/test_monitoring_stats.py::test_diagnosis_separates_drift_target_and_performance[args2-Feature drift without performance] PASSED
tests/test_monitoring_stats.py::test_diagnosis_separates_drift_target_and_performance[args3-unknown until labels] PASSED
tests/test_monitoring_stats.py::test_diagnosis_separates_drift_target_and_performance[args4-Default-rate shift with ranking intact] PASSED
tests/test_monitoring_stats.py::test_diagnosis_separates_drift_target_and_performance[args5-Calibration shift] PASSED
tests/test_monitoring_stats.py::test_diagnosis_separates_drift_target_and_performance[args6-without detectable input drift] PASSED
tests/test_monitoring_stats.py::test_diagnosis_separates_drift_target_and_performance[args7-score drift but no feature-level drift] PASSED
tests/test_monitoring_stats.py::test_diagnosis_separates_drift_target_and_performance[args8-likely cause of the calibration breach] PASSED
tests/test_monitoring_stats.py::test_diagnosis_separates_drift_target_and_performance[args9-Schema failure] PASSED
tests/test_pipeline_e2e.py::test_expected_outputs_exist PASSED
tests/test_pipeline_e2e.py::test_model_learns_signal PASSED
tests/test_pipeline_e2e.py::test_thresholds_chosen_from_training_predictions PASSED
tests/test_pipeline_e2e.py::test_reference_and_monitoring_pool_do_not_overlap PASSED
tests/test_pipeline_e2e.py::test_every_tableau_file_is_labelled_simulated PASSED
tests/test_pipeline_e2e.py::test_labels_pending_batch_skips_performance PASSED
tests/test_pipeline_e2e.py::test_relationship_break_lowers_gini PASSED
tests/test_pipeline_e2e.py::test_missing_spike_raises_data_quality_alert PASSED
tests/test_pipeline_e2e.py::test_default_rate_increase_flags_target_not_ranking PASSED
tests/test_pipeline_e2e.py::test_schema_failure_batch_is_rejected PASSED
tests/test_pipeline_e2e.py::test_pipeline_is_reproducible PASSED
tests/test_real_data.py::test_matches_uci_documentation PASSED
tests/test_real_data.py::test_no_blocking_data_quality_failures PASSED
tests/test_real_data.py::test_splits_are_disjoint_and_stratified PASSED
tests/test_validation.py::test_clean_synthetic_data_has_no_failures PASSED
tests/test_validation.py::test_missing_required_column_fails PASSED
tests/test_validation.py::test_non_binary_target_fails PASSED
tests/test_validation.py::test_row_count_mismatch_warns PASSED
tests/test_validation.py::test_duplicates_and_conflicting_labels_are_removed PASSED
tests/test_validation.py::test_infinite_values_become_missing PASSED
tests/test_validation.py::test_batch_schema_validation PASSED
```

To reproduce: `pip install -r requirements.txt`, then `python -m creditrisk all`, then `pytest -v`.
