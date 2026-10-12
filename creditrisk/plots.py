"""Static charts (PNG) generated from pipeline outputs."""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = {"scorecard_lr": "#2a78d6", "challenger_gbm": "#eb6834"}
SERIES_3 = "#1baf7a"
STATUS = {"GREEN": "#0ca30c", "AMBER": "#fab219", "RED": "#d03b3b", "PENDING": "#898781"}
AMBER_LINE = "#ec835a"   # "serious" step: readable on the light surface
RED_LINE = "#d03b3b"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.labelsize": 10, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.family": "DejaVu Sans", "legend.frameon": False,
    "legend.labelcolor": INK_2, "lines.linewidth": 2,
})


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def roc_pr_curves(y, probs: dict, names: dict, path: Path) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    for key, p in probs.items():
        fpr, tpr, _ = roc_curve(y, p)
        a1.plot(fpr, tpr, color=SERIES[key], label=f"{names[key]} (AUC {roc_auc_score(y, p):.3f})")
        pr, rc, _ = precision_recall_curve(y, p)
        a2.plot(rc, pr, color=SERIES[key], label=f"{names[key]} (AP {average_precision_score(y, p):.3f})")
    a1.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="Random")
    a2.axhline(np.mean(y), color=MUTED, lw=1, ls="--", label=f"Base rate ({np.mean(y):.3f})")
    a1.set(title="ROC curve - test set", xlabel="False positive rate", ylabel="True positive rate")
    a2.set(title="Precision-recall curve - test set", xlabel="Recall", ylabel="Precision")
    for a in (a1, a2):
        a.legend(loc="lower right" if a is a1 else "lower left", fontsize=9)
    _save(fig, path)


def calibration_plot(calib: pd.DataFrame, names: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 5))
    hi = max(calib["mean_pd"].max(), calib["observed_dr"].max()) * 1.05
    ax.plot([0, hi], [0, hi], color=MUTED, lw=1, ls="--", label="Perfect calibration")
    for key, sub in calib.groupby("model"):
        ax.plot(sub["mean_pd"], sub["observed_dr"], marker="o", ms=7, color=SERIES[key], label=names[key],
                markeredgecolor=SURFACE, markeredgewidth=2)
    ax.set(title="Calibration by PD decile - test set", xlabel="Mean predicted PD",
           ylabel="Observed default rate", xlim=(0, hi), ylim=(0, hi))
    ax.legend(fontsize=9, loc="upper left")
    _save(fig, path)


def threshold_plot(tbl: pd.DataFrame, chosen: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(tbl["threshold"], tbl["precision"], color=SERIES["scorecard_lr"], label="Precision")
    ax.plot(tbl["threshold"], tbl["recall"], color=SERIES["challenger_gbm"], label="Recall")
    ax.plot(tbl["threshold"], tbl["f1"], color=SERIES_3, label="F1")
    for key, ls in (("cost_optimal", "--"), ("f1_optimal", ":")):
        ax.axvline(chosen[key], color=INK_2, lw=1.2, ls=ls)
        ax.text(chosen[key], 0.04 if key == "cost_optimal" else 0.11,
                f" {key.replace('_', ' ')} = {chosen[key]:.2f}", color=INK_2, fontsize=8,
                transform=ax.get_xaxis_transform(), va="bottom")
    ax.set(title="Champion: precision / recall / F1 vs threshold (test set)", xlabel="PD threshold",
           ylabel="Value", ylim=(0, 1))
    ax.legend(fontsize=9, loc="center right")
    _save(fig, path)


def grade_plot(grades: pd.DataFrame, path: Path) -> None:
    g = grades[grades["n"] > 0]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(g))
    ax.bar(x, g["observed_dr"], width=0.6, color=SERIES["scorecard_lr"], label="Observed default rate")
    ax.plot(x, g["mean_pd"], color=SERIES["challenger_gbm"], marker="o", ms=8, lw=0, label="Mean predicted PD",
            markeredgecolor=SURFACE, markeredgewidth=2)
    for i, (_, r) in enumerate(g.iterrows()):
        ax.text(i, r["observed_dr"], f"n={int(r['n'])}", ha="center", va="bottom", fontsize=8, color=INK_2)
    ax.set_xticks(x, g["grade"])
    ax.set(title="Champion: default rate by rating grade (test set)", xlabel="Grade (G1 = lowest risk)",
           ylabel="Rate")
    ax.legend(fontsize=9, loc="upper left")
    _save(fig, path)


def iv_plot(iv: pd.DataFrame, path: Path) -> None:
    d = iv.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = [SERIES["scorecard_lr"] if s else AXIS for s in d["selected"]]
    ax.barh(d["feature"] + "  " + d["description"].str.slice(0, 38), d["iv"], color=colors, height=0.65)
    ax.set(title="Information Value - top 20 features (training data)", xlabel="Information Value")
    ax.tick_params(axis="y", labelsize=8, colors=INK_2)
    ax.text(0.98, 0.02, "blue = selected for scorecard\ngrey = not selected (low IV or correlated)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=INK_2)
    _save(fig, path)


def confusion_plot(row: pd.Series, path: Path) -> None:
    m = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(m, cmap="Blues", vmin=0, vmax=m.max() * 1.6)
    ax.grid(False)
    labels = [["True negative", "False positive"], ["False negative", "True positive"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{m[i, j]}", ha="center", va="center", color=INK, fontsize=10)
    ax.set_xticks([0, 1], ["Predicted: no default", "Predicted: default"])
    ax.set_yticks([0, 1], ["Actual: no default", "Actual: default"])
    ax.set_title(f"Champion confusion matrix (test)\nthreshold = {row['threshold']:.2f} (cost-optimal)")
    _save(fig, path)


# ------------------------------- monitoring -----------------------------------------
def _rag_lines(ax, amber: float, red: float, higher_is_worse: bool = True) -> None:
    ax.axhline(amber, color=AMBER_LINE, lw=1.2, ls="--")
    ax.axhline(red, color=RED_LINE, lw=1.2, ls="--")
    ax.text(1.0, amber, " amber", transform=ax.get_yaxis_transform(), color=INK_2, fontsize=8, va="center")
    ax.text(1.0, red, " red", transform=ax.get_yaxis_transform(), color=INK_2, fontsize=8, va="center")


def monitoring_overview(summary: pd.DataFrame, cfg: dict, path: Path) -> None:
    """Three stacked panels: score PSI, Gini vs benchmark, default rate vs mean PD."""
    mc = cfg["monitoring"]
    x = np.arange(len(summary))
    labels = [f"{b}\n" + "\n".join(textwrap.wrap(s.replace("_", " "), 16))
              for b, s in zip(summary["batch_id"], summary["scenario"])]
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    a = axes[0]
    a.plot(x, summary["score_psi"], color=SERIES["scorecard_lr"], marker="o", ms=8,
           markeredgecolor=SURFACE, markeredgewidth=2)
    _rag_lines(a, mc["psi"]["amber"], mc["psi"]["red"])
    a.set(title="Score stability (PSI vs development sample)", ylabel="PSI")

    a = axes[1]
    has = summary["gini"].notna()
    a.plot(x[has], summary.loc[has, "gini"], color=SERIES["scorecard_lr"], marker="o", ms=8,
           markeredgecolor=SURFACE, markeredgewidth=2, label="Batch Gini")
    bench = summary["benchmark_gini"].iloc[0]
    a.axhline(bench, color=INK_2, lw=1, label=f"Benchmark (test) {bench:.3f}")
    a.axhline(bench * (1 - mc["gini_relative_drop"]["amber"]), color=AMBER_LINE, lw=1.2, ls="--", label="Amber")
    a.axhline(bench * (1 - mc["gini_relative_drop"]["red"]), color=RED_LINE, lw=1.2, ls="--", label="Red")
    for i in x[~has]:
        a.text(i, bench * (1 - mc["gini_relative_drop"]["red"]) - 0.03, "labels\npending", ha="center",
               va="top", fontsize=8, color=INK_2)
    a.set(title="Discrimination (Gini) - only when outcome labels are available", ylabel="Gini")
    a.legend(fontsize=8, loc="lower left", ncol=4)

    a = axes[2]
    a.plot(x[has], summary.loc[has, "observed_default_rate"], color=SERIES["challenger_gbm"], marker="o", ms=8,
           markeredgecolor=SURFACE, markeredgewidth=2, label="Observed default rate")
    a.plot(x, summary["mean_pd"], color=SERIES["scorecard_lr"], marker="s", ms=7,
           markeredgecolor=SURFACE, markeredgewidth=2, label="Mean predicted PD")
    a.axhline(summary["reference_default_rate"].iloc[0], color=INK_2, lw=1, label="Development default rate")
    a.set(title="Default rate vs predicted PD", ylabel="Rate")
    a.legend(fontsize=8, loc="upper left")
    a.set_xticks(x, labels, fontsize=8)
    fig.suptitle("SIMULATED monitoring batches - not real bank or production data", color=RED_LINE,
                 fontsize=10, y=0.995)
    _save(fig, path)


def csi_heatmap(feature_drift: pd.DataFrame, cfg: dict, path: Path) -> None:
    fd = feature_drift.assign(label=np.where(feature_drift["in_model"], feature_drift["feature"],
                                             feature_drift["feature"] + " (info only)"))
    pv = fd.pivot(index="label", columns="batch_id", values="csi")
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(pv) + 2))
    red = cfg["monitoring"]["psi"]["red"]
    im = ax.imshow(np.clip(pv.to_numpy(), 0, red * 2), cmap="Blues", aspect="auto", vmin=0, vmax=red * 2)
    ax.grid(False)
    ax.set_xticks(range(len(pv.columns)), pv.columns, fontsize=8)
    ax.set_yticks(range(len(pv.index)), pv.index, fontsize=8)
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = pv.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color=SURFACE if v > red else INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label(f"CSI (capped at {red * 2:.2f})", color=INK_2)
    ax.set_title("Feature stability (CSI) by SIMULATED batch")
    _save(fig, path)
