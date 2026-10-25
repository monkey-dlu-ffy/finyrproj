"""Metrics tab of the review UI: the evaluation numbers, computed live from
experiments/fusion_scores.parquet with the functions scripts/verify_metrics.py
defines, so this page cannot show a number that script would not print.

Each section's "View code" opens the function behind it in VS Code. The
workspace's .vscode/settings.json points Code Runner at py -3.13 and its
OUTPUT tab, so Run Code (Ctrl+Alt+N) on the script prints the same numbers there.

Unlike the review queue, this page does show the labels (is_fraud, typology):
only as totals over the test split, never against a single case.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from config import EXPERIMENTS_DIR
from scripts.verify_metrics import (
    TYPOLOGIES,
    auc_mann_whitney,
    average_precision_by_hand,
    confusion_by_hand,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
VERIFY_SCRIPT = PROJECT_DIR / "scripts" / "verify_metrics.py"
EXPERIMENTS_SCRIPT = PROJECT_DIR / "scripts" / "run_experiments.py"
FUSION_SCRIPT = PROJECT_DIR / "scripts" / "run_fusion.py"
SCORES_FILE = EXPERIMENTS_DIR / "fusion_scores.parquet"
FUSION_RESULTS = EXPERIMENTS_DIR / "fusion_results.json"
EXPERIMENT_RESULTS = EXPERIMENTS_DIR / "experiment_results.json"

# Two series at most per chart: categorical slots 1 and 2 of the dataviz
# reference palette (blue, orange).
GENUINE_COLOR, FRAUD_COLOR = "#2a78d6", "#eb6834"
AGENTS = [
    ("transaction_score", "Transaction agent", "E1_transaction_only"),
    ("document_score", "Document agent", "E2_document_only"),
    ("reconciliation_score", "Reconciliation agent", None),
    ("fused_score", "Fused score", "E3_full_framework"),
]
ABLATIONS = [
    ("no_document", "Fused, without document agent"),
    ("no_transaction", "Fused, without transaction agent"),
    ("no_reconciliation", "Fused, without reconciliation agent"),
]


def _line_of(path: Path, prefix: str) -> int:
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.startswith(prefix):
            return number
    return 1


def _open_in_vscode(path: Path, prefix: str | None = None) -> None:
    """Open `path` in VS Code at the line starting with `prefix`. Found by
    text, not a hard-coded number, so it stays right as the script changes."""
    code = shutil.which("code")
    if code is None:
        st.error("VS Code's `code` command isn't on PATH. Open the file yourself: "
                 f"`{path.relative_to(PROJECT_DIR)}`")
        return
    line = _line_of(path, prefix) if prefix else 1
    # Opening the project folder as well, so the file lands in the window whose
    # .vscode/settings.json sends Run Code to the OUTPUT tab.
    subprocess.Popen([code, str(PROJECT_DIR), "--goto", f"{path}:{line}"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _section(title: str, path: Path, prefix: str, key: str) -> None:
    head, button = st.columns([5, 1])
    head.subheader(title)
    if button.button("View code", key=f"code_{key}", help=f"{path.name}, `{prefix}`", width="stretch"):
        _open_in_vscode(path, prefix)


def _mtimes(*paths: Path) -> tuple[float, ...]:
    return tuple(p.stat().st_mtime for p in paths)


@st.cache_data(show_spinner=False)
def _load(_mtimes: tuple[float, ...]) -> tuple[pd.DataFrame, dict, dict]:
    scores = pd.read_parquet(SCORES_FILE)
    test = scores.loc[scores["split"] == "test"].reset_index(drop=True)
    return test, json.loads(FUSION_RESULTS.read_text()), json.loads(EXPERIMENT_RESULTS.read_text())


@st.cache_data(show_spinner="Running scripts/verify_metrics.py ...")
def _run_verification(_mtimes: tuple[float, ...]) -> tuple[int, str]:
    """The script itself, run exactly as from a terminal. Keyed on the
    script's and its inputs' modification times, so editing either reruns it."""
    proc = subprocess.run(
        [sys.executable, "-X", "utf8", str(VERIFY_SCRIPT)],
        capture_output=True, text=True, encoding="utf-8", cwd=PROJECT_DIR,
    )
    return proc.returncode, proc.stdout + proc.stderr


def _set_threshold(value: float) -> None:
    st.session_state["metrics_threshold"] = value


def _ranking_rows(test: pd.DataFrame, experiments: dict) -> pd.DataFrame:
    y = test["is_fraud"].to_numpy()
    rows = []
    for col, label, stored_key in AGENTS:
        s = test[col].to_numpy()
        roc, ap = roc_auc_score(y, s), average_precision_score(y, s)
        # Same guard as verify_metrics.py section 3: the second, independent method must agree.
        assert abs(roc - auc_mann_whitney(y, s)) <= 1e-9 and abs(ap - average_precision_by_hand(y, s)) <= 1e-9
        stored = experiments.get(stored_key, {}) if stored_key else {}
        rows.append({"score": label, "ROC-AUC": roc, "ROC-AUC 95% CI": _ci(stored.get("roc_auc")),
                     "PR-AUC": ap, "PR-AUC 95% CI": _ci(stored.get("pr_auc"))})
    return pd.DataFrame(rows).set_index("score")


def _ci(entry: dict | None) -> str:
    return f"{entry['ci_low']:.3f} – {entry['ci_high']:.3f}" if entry else "—"


def _histogram(test: pd.DataFrame, threshold: float) -> alt.Chart:
    edges = np.linspace(0, 1, 51)
    frames = []
    for label, is_fraud in (("Genuine", 0), ("Fraud", 1)):
        counts, _ = np.histogram(test.loc[test["is_fraud"] == is_fraud, "fused_score"], bins=edges)
        frames.append(pd.DataFrame({"bin_start": edges[:-1], "bin_end": edges[1:], "entries": counts, "label": label}))
    bins = pd.concat(frames).query("entries > 0")
    color = alt.Color("label:N", title=None, legend=alt.Legend(orient="top"),
                      scale=alt.Scale(domain=["Genuine", "Fraud"], range=[GENUINE_COLOR, FRAUD_COLOR]))
    # bin="binned": the bins are already counted above, so Vega-Lite draws
    # them as given instead of treating bin_start as a continuous position.
    bars = alt.Chart(bins).mark_bar(binSpacing=2).encode(
        x=alt.X("bin_start:Q", bin="binned", title="fused_score", scale=alt.Scale(domain=[0, 1])),
        x2="bin_end:Q",
        y=alt.Y("entries:Q", title="test entries", stack="zero"),
        color=color,
        tooltip=[alt.Tooltip("label:N", title=""), alt.Tooltip("bin_start:Q", title="score from", format=".2f"),
                 alt.Tooltip("bin_end:Q", title="score to", format=".2f"), alt.Tooltip("entries:Q", format=",")],
    )
    rule = alt.Chart(pd.DataFrame({"t": [threshold]})).mark_rule(strokeDash=[4, 3], color="gray").encode(x="t:Q")
    return (bars + rule).properties(height=280)


def _curve(points: pd.DataFrame, x: str, y: str, x_title: str, y_title: str,
           op: dict, baseline: pd.DataFrame) -> alt.Chart:
    line = alt.Chart(points).mark_line(strokeWidth=2, color=GENUINE_COLOR, interpolate="step-after").encode(
        x=alt.X(f"{x}:Q", title=x_title, scale=alt.Scale(domain=[0, 1])),
        y=alt.Y(f"{y}:Q", title=y_title, scale=alt.Scale(domain=[0, 1])),
    )
    hover = alt.Chart(points).mark_point(size=60, opacity=0).encode(
        x=f"{x}:Q", y=f"{y}:Q",
        tooltip=[alt.Tooltip("threshold:Q", format=".3f"), alt.Tooltip(f"{x}:Q", title=x_title, format=".3f"),
                 alt.Tooltip(f"{y}:Q", title=y_title, format=".3f")],
    )
    chance = alt.Chart(baseline).mark_line(strokeDash=[4, 3], color="gray", strokeWidth=1).encode(x="x:Q", y="y:Q")
    marker_df = pd.DataFrame([op])
    marker = alt.Chart(marker_df).mark_point(size=110, filled=True, color=FRAUD_COLOR, stroke="white", strokeWidth=2).encode(
        x=f"{x}:Q", y=f"{y}:Q",
        tooltip=[alt.Tooltip("label:N", title=""), alt.Tooltip(f"{x}:Q", title=x_title, format=".3f"),
                 alt.Tooltip(f"{y}:Q", title=y_title, format=".3f")],
    )
    # Neutral ink readable on both themes; flipped to the marker's left near
    # the right edge, where the label would otherwise run off the chart.
    right = op[x] > 0.7
    text = alt.Chart(marker_df).mark_text(align="right" if right else "left", dx=-10 if right else 10, dy=14,
                                          color="#8b8b8b").encode(x=f"{x}:Q", y=f"{y}:Q", text="label:N")
    return (chance + line + hover + marker + text).properties(height=300)


def metrics_page() -> None:
    st.title("Evaluation metrics")
    missing = [p for p in (SCORES_FILE, FUSION_RESULTS, EXPERIMENT_RESULTS) if not p.exists()]
    if missing:
        st.error("Missing: " + ", ".join(f"`{p.relative_to(PROJECT_DIR)}`" for p in missing)
                 + ". Run scripts/run_fusion.py and scripts/run_experiments.py first.")
        return
    test, fusion, experiments = _load(_mtimes(SCORES_FILE, FUSION_RESULTS, EXPERIMENT_RESULTS))
    y = test["is_fraud"].to_numpy()
    s = test["fused_score"].to_numpy()
    st.caption(
        f"Test split: {len(test):,} entries with an invoice image, {int(y.sum()):,} of them fraud ({y.mean():.0%}). "
        "Every number on this page is computed live from `experiments/fusion_scores.parquet` "
        "with the functions in `scripts/verify_metrics.py`."
    )

    # --- the script itself ---------------------------------------------------
    with st.container(border=True):
        st.markdown("**The evaluation script** · `scripts/verify_metrics.py`")
        st.caption(
            "It recomputes every metric two independent ways (numpy and sklearn; the Mann-Whitney "
            "formula for ROC-AUC; a hand-written average precision for PR-AUC) and checks each one "
            "against the saved results. To run it yourself: open it in VS Code, then press "
            "**Ctrl+Alt+N** (Code Runner: Run Code). The results print in the **OUTPUT** tab."
        )
        open_col, rerun_col, _ = st.columns([1.4, 1, 2])
        if open_col.button("Open in VS Code", type="primary", width="stretch"):
            _open_in_vscode(VERIFY_SCRIPT, "def main(")
        if rerun_col.button("Run it again here", width="stretch"):
            _run_verification.clear()
        code, output = _run_verification(_mtimes(VERIFY_SCRIPT, SCORES_FILE, FUSION_RESULTS, EXPERIMENT_RESULTS))
        summary = re.search(r"(\d+) saved numbers reproduced exactly, (\d+) differ", output)
        if code == 0 and summary:
            st.success(f"Verified: {summary[1]} saved numbers reproduced exactly, {summary[2]} differ.")
        elif summary:
            st.error(f"{summary[2]} saved numbers differ from the recomputed ones. See the output below.")
        else:
            st.error(f"The script failed (exit code {code}). See the output below.")
        with st.expander("Script output (the same text VS Code shows)"):
            st.code(output, language="text")

    # --- threshold -----------------------------------------------------------
    f1_t = round(fusion["f1_threshold"], 2)
    r90_t = round(fusion["recall_90_threshold"], 2)
    st.session_state.setdefault("metrics_threshold", f1_t)
    _section("Operating point", FUSION_SCRIPT, "    # threshold selection on val only", "threshold")
    slider_col, b1, b2 = st.columns([4, 1, 1])
    threshold = slider_col.slider("Flag an entry when fused_score ≥", 0.01, 0.99, step=0.01, key="metrics_threshold")
    b1.button(f"Best F1 ({f1_t})", on_click=_set_threshold, args=(f1_t,), width="stretch")
    b2.button(f"Recall ≥ 90% ({r90_t})", on_click=_set_threshold, args=(r90_t,), width="stretch")
    st.caption(
        f"{f1_t} has the best F1 on the validation split, and the risk tiers use it. {r90_t} is the highest "
        "threshold that still catches 90% of validation fraud. Both were chosen on validation only, never on test."
    )

    m = confusion_by_hand(y, s, threshold)

    # --- threshold metrics ---------------------------------------------------
    _section("At this threshold", VERIFY_SCRIPT, "def confusion_by_hand(", "confusion")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Precision", f"{m['precision']:.3f}")
    k1.caption(f"TP / (TP + FP) = {m['tp']:,} / {m['tp'] + m['fp']:,}")
    k2.metric("Recall", f"{m['recall']:.3f}")
    k2.caption(f"TP / (TP + FN) = {m['tp']:,} / {m['tp'] + m['fn']:,}")
    k3.metric("F1", f"{m['f1']:.3f}")
    k3.caption("2 × P × R / (P + R)")
    k4.metric("False-positive rate", f"{m['false_positive_rate']:.3f}")
    k4.caption(f"FP / (FP + TN) = {m['fp']:,} / {m['fp'] + m['tn']:,}")

    table_col, _ = st.columns([1, 1])
    with table_col:
        st.table(pd.DataFrame(
            {"Actually fraud": [f"TP {m['tp']:,}", f"FN {m['fn']:,}"],
             "Actually genuine": [f"FP {m['fp']:,}", f"TN {m['tn']:,}"]},
            index=["Flagged", "Not flagged"],
        ))

    # --- threshold-free ------------------------------------------------------
    _section("Ranking quality (no threshold)", VERIFY_SCRIPT, "def auc_mann_whitney(", "ranking")
    st.caption(
        "ROC-AUC is the chance that a random fraud scores higher than a random genuine entry. "
        "PR-AUC is average precision: precision at each score cut-off, weighted by how much recall it adds. "
        "95% ranges are from 1,000 bootstrap resamples of the test split (`run_experiments.py`, `bootstrap_ci`)."
    )
    ranking = _ranking_rows(test, experiments)
    st.dataframe(ranking.style.format({"ROC-AUC": "{:.4f}", "PR-AUC": "{:.4f}"}), width="stretch")
    st.caption(
        f"A score that ranks at random gets ROC-AUC 0.5 and PR-AUC equal to the fraud rate ({y.mean():.3f}). "
        "The reconciliation agent alone is not a saved experiment, so it has no bootstrap range."
    )
    ablations = pd.DataFrame([
        {"fusion": label,
         "ROC-AUC": experiments["E4_ablations"][key]["roc_auc"]["point"],
         "ROC-AUC 95% CI": _ci(experiments["E4_ablations"][key]["roc_auc"]),
         "PR-AUC": experiments["E4_ablations"][key]["pr_auc"]["point"],
         "PR-AUC 95% CI": _ci(experiments["E4_ablations"][key]["pr_auc"])}
        for key, label in ABLATIONS if key in experiments.get("E4_ablations", {})
    ])
    if not ablations.empty:
        abl_head, abl_button = st.columns([5, 1])
        abl_head.markdown("**Leaving one agent out of the fusion** · saved by `run_experiments.py` (refits the fusion)")
        if abl_button.button("View code", key="code_ablations", width="stretch"):
            _open_in_vscode(EXPERIMENTS_SCRIPT, "def main(")
        st.dataframe(ablations.set_index("fusion").style.format({"ROC-AUC": "{:.4f}", "PR-AUC": "{:.4f}"}),
                     width="stretch")

    # --- charts --------------------------------------------------------------
    _section("Curves", VERIFY_SCRIPT, "def average_precision_by_hand(", "curves")
    fpr, tpr, roc_t = roc_curve(y, s)
    roc_points = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": np.minimum(roc_t, 1.0)})
    prec, rec, pr_t = precision_recall_curve(y, s)
    pr_points = pd.DataFrame({"recall": rec[:-1], "precision": prec[:-1], "threshold": pr_t})
    op_label = f"threshold {threshold:.2f}"
    roc_col, pr_col = st.columns(2)
    with roc_col:
        st.markdown("**ROC curve**")
        st.altair_chart(_curve(
            roc_points, "fpr", "tpr", "false-positive rate", "recall (true-positive rate)",
            {"fpr": m["false_positive_rate"], "tpr": m["recall"], "label": op_label},
            pd.DataFrame({"x": [0, 1], "y": [0, 1]}),
        ), width="stretch")
        st.caption("Dashed: a score that ranks at random.")
    with pr_col:
        st.markdown("**Precision–recall curve**")
        st.altair_chart(_curve(
            pr_points, "recall", "precision", "recall", "precision",
            {"recall": m["recall"], "precision": m["precision"], "label": op_label},
            pd.DataFrame({"x": [0, 1], "y": [y.mean(), y.mean()]}),
        ), width="stretch")
        st.caption(f"Dashed: a score that ranks at random (precision = fraud rate, {y.mean():.3f}).")

    st.markdown("**Where the scores fall**")
    st.altair_chart(_histogram(test, threshold), width="stretch")
    below, above = s[s < threshold], s[s >= threshold]
    gap = (f" No test entry scores between {below.max():.3f} and {above.min():.3f}."
           if len(below) and len(above) else "")
    st.caption(f"Dashed line: the threshold. {len(above):,} entries are flagged and {len(below):,} are not.{gap}")

    # --- by fraud type -------------------------------------------------------
    _section("Recall by fraud type", VERIFY_SCRIPT, '    heading("4. Recall by fraud typology', "typology")
    flagged = s >= threshold
    by_type = []
    for typ in TYPOLOGIES:
        mask = (test["typology"] == typ).to_numpy()
        caught, n = int((flagged & mask).sum()), int(mask.sum())
        by_type.append({"fraud type": typ, "frauds": n, "caught": caught, "missed": n - caught,
                        "recall": caught / n if n else float("nan")})
    st.dataframe(pd.DataFrame(by_type).set_index("fraud type").style.format({"recall": "{:.3f}"}),
                 width="stretch")
    st.caption("Totals over the test split only. The review queue still never shows a case's fraud type.")
