# Building the Tableau monitoring dashboard

The pipeline writes six CSVs to `outputs/tableau/`. This guide builds a one-page monitoring dashboard from them in **Tableau Public** (free) and publishes it so it has a shareable link.

> **Status:** no workbook is included in this repository. The CSVs are generated and tested; the dashboard below is built by hand by following these steps. Menu names follow recent Tableau Public versions and may differ slightly in yours.

## 0. Install
Download Tableau Public Desktop from https://public.tableau.com (free; needs a Tableau Public account).

## 1. Connect the data
1. Open Tableau Public, then **Connect -> To a File -> Text file**, and pick `outputs/tableau/tableau_batch_summary.csv`.
2. On the Data Source page, use **Add** next to Connections (or drag more files from the left panel) to add `tableau_feature_drift.csv`, `tableau_metrics_long.csv` and `tableau_alerts.csv`.
3. Tableau creates **relationships** (the "noodles"). Relate each file to `tableau_batch_summary.csv` on **batch_id = batch_id**.
4. In `tableau_batch_summary`, check that `period_start` is a **Date** (click the type icon above the column if it is not).

## 2. Sheet "Status table"
1. Rows: `batch_id`, `scenario`, `diagnosis`.
2. Drag **Measure Names** to Columns and **Measure Values** to Text. Keep only the measures you need (e.g. `score_psi`, `max_feature_csi`, `gini`, `observed_default_rate`, `mean_pd`).
3. Create a calculated field **Status colour** = `[overall_status]` and drag it to **Color**. Assign GREEN = green, AMBER = amber, RED = red, PENDING = grey. Keep the status text visible as well, so meaning never depends on colour alone.

## 3. Sheet "Score PSI trend"
1. Columns: `period_start` (exact date, continuous). Rows: `score_psi`.
2. Use a line mark with circle markers.
3. Analytics pane -> **Reference Line** -> Constant 0.10 (label "Amber"), then again for 0.25 (label "Red").
4. Add `batch_id` and `diagnosis` to **Tooltip**.

## 4. Sheet "Gini vs benchmark"
1. Columns: `period_start`. Rows: `gini`.
2. Reference line: **Average of `benchmark_gini`**, labelled "Benchmark".
3. Calculated fields **Gini amber** = `[benchmark_gini]*0.9` and **Gini red** = `[benchmark_gini]*0.8`; add each as a reference line.
4. SIM_M08 has no Gini because its labels are pending, so the point is simply absent. Add a caption saying so.

## 5. Sheet "Default rate vs predicted PD"
1. Columns: `period_start`. Rows: `observed_default_rate` and `mean_pd` (use **Measure Values** for a single axis; never two separate y-axes).
2. Reference line: average of `reference_default_rate`.

## 6. Sheet "Feature CSI heatmap" (`tableau_feature_drift`)
1. Columns: `batch_id`. Rows: `feature`. Filter `in_model` = True (or keep X29 and label it "info only").
2. Mark type **Square**. Colour = `csi` with a sequential single-hue palette, and set the colour range end to 0.5.
3. Label = `csi` (format 0.00). Tooltip: `ks_pvalue`, `missing_increase`, `description`.

## 7. Sheet "Alert log" (`tableau_alerts`)
Rows: `alert_id`, `batch_id`, `severity`, `component`, `detail`, `recommended_action`. Colour by `severity`.

## 8. Dashboard
1. **New Dashboard**, size **Automatic** (or 1200 x 900).
2. Layout:
   - Top: a title text box, *"Model Monitoring - SIMULATED batches (not real bank data)"*.
   - Row 1: Status table.
   - Row 2: Score PSI trend | Gini vs benchmark.
   - Row 3: Default rate vs PD | Feature CSI heatmap.
   - Bottom: Alert log.
3. Add a `scenario` or `batch_id` filter and apply it to all sheets that use this data source (filter menu -> **Apply to Worksheets**).

## 9. Publish and get the link
**File -> Save to Tableau Public As...**, then name it *Credit Risk Model Monitoring*. Tableau uploads it and opens the public page; copy that URL into the README (under "Results") and onto the resume.

## 10. Refreshing after a new run
Re-run `python -m creditrisk monitor`, then in Tableau use **Data -> Refresh All Extracts** (or re-open the workbook), and save to Tableau Public again.
