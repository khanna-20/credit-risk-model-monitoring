# Model Monitoring Runbook

How to run the monitoring workflow each period, read the results and act on alerts.
This runbook describes this portfolio project. It is written in the style of a bank runbook, but the thresholds and roles are illustrative.

## 1. When to run
- **Every period** (e.g. monthly) when a new batch of scored firms is available.
- **Outcome-based checks** (default rate, calibration, Gini) only become possible once outcomes are observed, which takes 12 months for a 1-year PD. Until then, a batch is monitored on drift and data quality only, and those checks show `PENDING`.

## 2. Inputs
| Input | Location | Notes |
|---|---|---|
| Trained model bundle | `outputs/model/model_bundle.joblib` | created by `python -m creditrisk develop` |
| Development reference | inside the bundle | training rows, their PDs and grades |
| New batch file(s) | `data/simulated/` or any folder | CSV with `X1`-`X64`; optional `default_flag`, `batch_id`, `period_start`, `scenario`, `is_simulated` |
| Thresholds | `config/config.yaml` -> `monitoring` | change here, never in code |

## 3. Steps
1. Put the new batch CSV in the batch folder.
2. Run `python -m creditrisk monitor` (or `python -m creditrisk monitor --batch-dir <folder>`).
3. Read the console summary: one line per batch with its overall status and diagnosis.
4. Open `outputs/reports/monitoring_report.md` for the detail, and refresh the Tableau data sources (`outputs/tableau/*.csv`).
5. Triage every alert in `outputs/tableau/tableau_alerts.csv` using section 5.
6. Record decisions (accepted / investigating / escalated) in the team's issue log.

## 4. What each status means
| Status | Meaning |
|---|---|
| GREEN | Within tolerance |
| AMBER | Investigate and document; no immediate change to model use |
| RED | Escalate to the model owner within the agreed timeline |
| PENDING | Outcome labels not yet available |
| N/A | Not enough data (e.g. fewer than 15 defaults for a Gini estimate) |

The overall batch status is the worst of its component statuses.

## 5. Alert triage
| Component | Typical cause | First checks | Action |
|---|---|---|---|
| **schema** | File layout changed, column renamed, text in a numeric column | Compare columns with `X1`-`X64` | Batch is rejected; fix with the data owner and re-run |
| **data_quality** (missing rate up) | Upstream feed failure, mapping change | Which features (alert `detail`), since when, which source system | Raise a data incident; treat PDs for affected firms with caution. Missing values carry risk weight in this model, so they **inflate PDs**. |
| **feature_stability** (CSI) | Genuine portfolio change or a data change | KS test, missing rates, business context (new segment, new region) | If genuine and performance holds: document it. If it is a data error: treat as data_quality. |
| **score_stability** (PSI) | Mix shift, or feature interactions changed | Which features moved (CSI heatmap); grade migration (`tableau_grade_monitoring.csv`) | Investigate; a score shift with stable features can point to broken relationships (see SIM_M06) |
| **default_rate** | Economic cycle, portfolio change, outcome-data error | Confirm the outcome data; compare with the portfolio trend | If it persists for 2+ periods, assess recalibration |
| **calibration** | PDs too high or too low versus outcomes | Direction (`observed_default_rate` vs `mean_pd`), per-grade results | Assess recalibration with the model owner; check data_quality first |
| **performance** (Gini) | The model ranks defaulters worse | Rule out data issues; check the feature-outcome relationship | Escalate; root-cause analysis; consider redevelopment or the challenger |

**Key principle:** drift does not prove deterioration, and stability does not prove good performance. Confirm with outcome-based metrics whenever labels exist.

## 6. Changing thresholds
1. Edit `config/config.yaml` (`monitoring` section).
2. Re-run `python -m creditrisk monitor`. No retraining is needed.
3. Record the reason for the change. Typical starting points are PSI 0.10/0.25 and a relative Gini drop of 10%/20%.

## 7. Re-baselining after redevelopment
Run `python -m creditrisk develop`. This refits the models and **replaces the development reference and the benchmark Gini**. Earlier monitoring results stay valid only for the old model version, so archive `outputs/` before re-baselining.

## 8. Known limitations of this workflow
- PSI and KS on small batches are noisy. KS p-values flag even tiny shifts on large samples, so they are information only and do not drive the RAG status.
- Gini estimates on batches with few defaults have wide uncertainty. The minimum-defaults rule (15) only partly addresses this.
- The batches in this repository are **simulated**. Real monitoring would use genuinely new observations over time.
