"""Head-to-head comparison against the base paper, AuditCopilot (Kadir et al.,
NeurIPS 2025 GenAI-in-Finance workshop, arXiv 2512.02726), on THIS corpus.

    py -3.13 scripts\\run_auditcopilot_comparison.py --no-llm      # fast: non-LLM rows + sample prompts
    py -3.13 scripts\\run_auditcopilot_comparison.py --limit 5     # smoke test: 5 sample rows through the LLM
    py -3.13 scripts\\run_auditcopilot_comparison.py               # full run (hours on CPU; resumable)

Rows of the comparison table, all scored on the SAME stratified sample of the
document-linked test subset (the only rows where every agent has a score):

  isolation_forest       AuditCopilot's own ML baseline. Label-free.
  auditcopilot           Their Figure 1 prompt (dataset stats + IF hint), one
                         LLM call per entry, ledger fields only. Label-free.
  auditcopilot_signals   The same prompt plus our document + reconciliation
                         agents' fired evidence. Label-free. If this beats
                         `auditcopilot`, our agents help ANY detector.
  agents_label_free      Document + reconciliation scores, noisy-OR combined,
                         baselines fit on ALL train rows (no is_fraud). The
                         fair, no-labels answer to "is it just supervision?".
  fusion                 Our full stacked system, read from
                         experiments/fusion_scores.parquet. Uses labels.

Deviations from the paper, each forced by this corpus, not chosen:
  * The paper's "user" fields (total users, user tx count, criterion 3) are
    mapped to vendor -- this ledger has no posting-user column.
  * The LLM is OLLAMA_MODEL from config (qwen2.5:3b-instruct by default), not
    Gemma/Llama/Mistral -- the only model that runs at a usable speed on this
    machine (see config.py). Override with $env:OLLAMA_MODEL.
  * Vendor ids are pseudonymised in every prompt. ghost_vendor rows carry
    `GHOST-<name>` ids (linked_pipeline/typologies.py), which would hand an
    LLM the label in plain text. A hard check refuses any prompt still
    containing "GHOST".
  * Isolation Forest runs on the transaction agent's engineered features
    (minus the synthetic hour_of_day draw, which is pure noise), with the
    vendor/account baselines fit on all train rows rather than train-genuine,
    so it stays label-free.

The sample over-represents each fraud typology so per-typology recall has
enough rows; every pooled metric is weighted back to the test subset's real
composition (weight = stratum size / sampled count). Note that subset is
~60% fraud (all fraud rows + a genuine sample, see generate_invoices.py), so
a random scorer's PR-AUC is ~0.60 -- the `base_rate` line is printed with
the table for exactly that reason.

LLM text is not reproducible across sessions (config.py, LLM_SEED caveat), so
every response is cached in experiments/auditcopilot_llm_cache.jsonl keyed by
model + prompt hash. A killed run resumes from the cache; the cache, not a
re-run, is the artifact the reported numbers come from.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.document.extract import DocumentAgent, fit_document_stats
from agents.reconciliation.rules import APPROVAL_LIMIT_QUANTILE, ReconciliationAgent
from agents.transaction.features_erp import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    add_features,
    fit_category_levels,
    fit_vendor_account_stats,
)
from config import (
    EXPERIMENTS_DIR,
    LINKED_DIR,
    LLM_NUM_CTX,
    LLM_SEED,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_S,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    RANDOM_SEED,
    seed_everything,
    stable_hash,
)
from linked_pipeline.invoices import resolve_vendor_name

TYPOLOGIES = ("duplicate_invoice", "ghost_vendor", "threshold_split", "desc_amount_mismatch", "altered_amount")
CACHE_PATH = EXPERIMENTS_DIR / "auditcopilot_llm_cache.jsonl"
N_BOOTSTRAP = 1000
MAX_TOKENS = 100  # the JSON answer + a one-sentence explanation; ~3 tok/s on CPU
IF_FEATURES = [f for f in NUMERIC_FEATURES if f != "hour_of_day"] + CATEGORICAL_FEATURES
# One strong reconciliation rule (amount_mismatch / duplicate_invoice_number /
# vendor_absent_from_master / split_purchase, weight 0.30 each) is enough to flag -- a
# label-free threshold, not tuned on val.
AGENTS_LABEL_FREE_THRESHOLD = 0.30
LEDGER_FIELDS = (
    "BELNR", "BUKRS", "WAERS", "KTOSL", "PRCTR", "BSCHL", "HKONT", "DMBTR", "WRBTR",
    "vendor", "posting_date", "doc_date", "invoice_number", "po_number", "description",
)

# Figure 1 of the paper, verbatim except user -> vendor (see module docstring).
# Everything above TRANSACTION DATA is identical across rows, so Ollama can
# reuse that prefix's KV cache instead of re-reading it every call.
AUDITCOPILOT_PROMPT = """You are an expert financial auditor specializing in journal entry testing (JET) and fraud detection.
DATASET CONTEXT:
- Total transactions in dataset: {total_transactions}
- Isolation Forest detected {total_if_anomalies} anomalies ({if_anomaly_rate})
- Amount statistics:
  - Mean: {amount_mean}
  - Median: {amount_median}
  - 95th percentile: {amount_q95}
  - 99th percentile: {amount_q99}
  - Range: {amount_min} to {amount_max}
- Payment period max: {payment_period_max} days
- Total vendors: {total_vendors}
- Total accounts: {total_accounts}
ANOMALY DETECTION CRITERIA:
1. Unusual amounts: extremely high/low compared to typical transactions
2. Round number bias: large round numbers (1000, 5000, 10000, ...)
3. Vendor behavior: low-volume vendors handling high-value transactions
4. Timing anomalies: very long or very short payment periods
5. Account patterns: unusual account relationships or concentrations
6. Statistical outliers: values far from normal distribution
7. Business logic violations: transactions that don't make business sense
RESPONSE FORMAT: Respond ONLY with valid JSON in this exact format:
{{"anomaly": 0, "explanation": "Normal transaction, with reasoning"}}
OR
{{"anomaly": 1, "explanation": "Specific reason why this transaction is anomalous"}}
Be conservative - only flag clear anomalies. Provide specific, actionable explanations.
TRANSACTION DATA: {transaction_data}
Isolation Forest Hint: {if_status} (score: {if_score})
ADDITIONAL CONTEXT:
- This vendor ({vendor}) has {vendor_tx_count} total transactions
- This amount ({abs_amount}) is at the {amount_percentile}th percentile
{signals_block}Evaluate for anomalies and respond with JSON only."""

SIGNALS_BLOCK = """CROSS-SOURCE EVIDENCE (invoice document and ledger reconciliation checks):
{lines}
"""


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_splits() -> dict[str, pd.DataFrame]:
    return {name: pd.read_parquet(LINKED_DIR / f"{name}.parquet") for name in ("train", "val", "test")}


def draw_sample(fusion_test: pd.DataFrame, n_per_typology: int, n_genuine: int) -> pd.DataFrame:
    """Stratified by typology ('none' = genuine), seeded, so a re-run draws
    the same rows and hits the LLM cache. Each row carries the weight that
    maps its stratum back to the test subset's real size."""
    rng = np.random.default_rng(RANDOM_SEED)
    parts = []
    for typology, group in fusion_test.groupby("typology"):
        n = min(len(group), n_genuine if typology == "none" else n_per_typology)
        picked = group.iloc[np.sort(rng.choice(len(group), size=n, replace=False))].copy()
        picked["weight"] = len(group) / n
        parts.append(picked)
    return pd.concat(parts).sort_values("row_id").reset_index(drop=True)


def vendor_pseudonym(vendor_id: str) -> str:
    return f"VND-{stable_hash(vendor_id) % 100_000:05d}"


def mask_vendors(text: str, vendor_ids) -> str:
    for vid in sorted(set(vendor_ids), key=len, reverse=True):
        text = text.replace(str(vid), vendor_pseudonym(vid))
    return text


# ---------------------------------------------------------------------------
# Label-free baselines: Isolation Forest + document/reconciliation agents
# ---------------------------------------------------------------------------


def isolation_forest_scores(splits: dict[str, pd.DataFrame], test_ids: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Fit on all train rows, label-free. Returns the test rows' IF status and
    score, plus the train-population counts the prompt's DATASET CONTEXT cites."""
    stats = fit_vendor_account_stats(splits["train"].assign(is_fraud=0))  # all rows = "genuine": no labels used
    levels = fit_category_levels(splits["train"], splits["val"], splits["test"])

    def features(df: pd.DataFrame) -> pd.DataFrame:
        X = add_features(df, stats, seed=RANDOM_SEED)[IF_FEATURES].copy()
        for col in CATEGORICAL_FEATURES:
            X[col] = pd.Categorical(X[col], categories=levels[col]).codes
        return X.astype(float)

    X_train = features(splits["train"])
    forest = IsolationForest(n_estimators=200, contamination="auto", random_state=RANDOM_SEED, n_jobs=-1)
    forest.fit(X_train)
    train_pred = forest.predict(X_train)

    X_test = features(splits["test"]).loc[test_ids]
    decision = forest.decision_function(X_test)  # < 0 means anomalous, sklearn's convention
    out = pd.DataFrame(
        {
            "if_score": -decision,  # flipped so higher = more anomalous, like every other score here
            "if_decision": decision,
            "if_flag": (forest.predict(X_test) == -1).astype(int),
        },
        index=test_ids,
    )
    return out, {"total_if_anomalies": int((train_pred == -1).sum()), "if_anomaly_rate": float((train_pred == -1).mean())}


def agent_evidence(splits: dict[str, pd.DataFrame], sample: pd.DataFrame) -> pd.DataFrame:
    """Document + reconciliation agents on the sampled test rows, with every
    baseline fit on all train rows -- no is_fraud anywhere, unlike
    run_fusion.py's train-genuine fits."""
    manifest = pd.read_parquet(LINKED_DIR / "images" / "manifest.parquet")
    manifest = manifest.loc[manifest["split"] == "test"].set_index("row_id")
    vendor_master = pd.read_json(LINKED_DIR / "vendor_master.json")
    vendor_name_by_id = dict(zip(vendor_master["vendor_id"], vendor_master["vendor_name"]))

    train = splits["train"]
    recon_agent = ReconciliationAgent(
        vendor_master_ids=set(vendor_master["vendor_id"]),
        high_amount_threshold=float(train["DMBTR"].quantile(0.9)),
        approval_limit=float(train["DMBTR"].quantile(APPROVAL_LIMIT_QUANTILE)),
    )
    doc_agent = DocumentAgent(stats=fit_document_stats(train.drop(columns=["is_fraud"])))

    ids = sample["row_id"].to_numpy()
    joined = splits["test"].loc[ids].copy()
    joined["vendor_name"] = joined["vendor_id"].map(lambda v: resolve_vendor_name(v, vendor_name_by_id))
    joined["extracted_amount"] = manifest.loc[ids, "invoice_amount"]
    joined["extracted_vendor_name"] = manifest.loc[ids, "vendor_name"]
    joined["extracted_invoice_number"] = manifest.loc[ids, "invoice_number"]

    full_ledger = pd.concat([df.reset_index(names="row_id").assign(split=name) for name, df in splits.items()])
    recon = {e.row_id: e for e in recon_agent.analyze_batch(joined, ledger=full_ledger)}
    doc = {e.row_id: e for e in doc_agent.analyze_batch(joined)}

    rows = []
    for row_id in ids:
        r, d = recon[row_id], doc[row_id]
        doc_rules = [e for e in d.evidence if e["type"] != "extracted_field"]
        rows.append(
            {
                "row_id": int(row_id),
                "lf_document_score": d.score,
                "lf_reconciliation_score": r.score,
                "agents_label_free_score": 1 - (1 - d.score) * (1 - r.score),  # noisy-OR: no fitted weights
                "evidence": r.evidence + doc_rules,
                "invoice_vendor": joined.at[row_id, "extracted_vendor_name"],
                "invoice_amount": joined.at[row_id, "extracted_amount"],
                "invoice_number_doc": joined.at[row_id, "extracted_invoice_number"],
            }
        )
    return pd.DataFrame(rows).set_index("row_id")


# ---------------------------------------------------------------------------
# AuditCopilot prompt + LLM
# ---------------------------------------------------------------------------


def dataset_context(train: pd.DataFrame, if_counts: dict) -> dict:
    amounts = train["DMBTR"].abs()
    lag = (pd.to_datetime(train["posting_date"]) - pd.to_datetime(train["doc_date"])).dt.days
    return {
        "total_transactions": f"{len(train):,}",
        "total_if_anomalies": f"{if_counts['total_if_anomalies']:,}",
        "if_anomaly_rate": f"{if_counts['if_anomaly_rate']:.2%}",
        "amount_mean": f"{amounts.mean():,.2f}",
        "amount_median": f"{amounts.median():,.2f}",
        "amount_q95": f"{amounts.quantile(0.95):,.2f}",
        "amount_q99": f"{amounts.quantile(0.99):,.2f}",
        "amount_min": f"{amounts.min():,.2f}",
        "amount_max": f"{amounts.max():,.2f}",
        "payment_period_max": int(lag.max()),
        "total_vendors": f"{train['vendor_id'].nunique():,}",
        "total_accounts": f"{train['HKONT'].nunique():,}",
    }


def build_prompt(
    row: pd.Series, ctx: dict, train_amounts_sorted: np.ndarray, vendor_counts: pd.Series,
    if_row: pd.Series, agents_row: pd.Series | None,
) -> str:
    record = {k: row[k] for k in LEDGER_FIELDS if k != "vendor"}
    record["vendor"] = row["vendor_id"]
    for k in ("posting_date", "doc_date"):
        record[k] = str(pd.Timestamp(record[k]).date())
    record["po_number"] = None if pd.isna(record["po_number"]) else record["po_number"]
    record = {k: record[k] for k in LEDGER_FIELDS}

    amount = abs(float(row["DMBTR"]))
    percentile = 100.0 * np.searchsorted(train_amounts_sorted, amount, side="right") / len(train_amounts_sorted)

    signals_block = ""
    vendor_ids = {row["vendor_id"]}
    if agents_row is not None:
        lines = [
            f"- Invoice document: vendor '{agents_row['invoice_vendor']}', amount {agents_row['invoice_amount']:,.2f},"
            f" invoice {agents_row['invoice_number_doc']}"
        ]
        for e in agents_row["evidence"]:
            lines.append(f"- {e['type']}: {e['detail']}")
            vendor_ids.update(m["vendor_id"] for m in e.get("matches", []))
        if len(lines) == 1:
            lines.append("- No document or reconciliation check fired.")
        signals_block = SIGNALS_BLOCK.format(lines="\n".join(lines))

    prompt = AUDITCOPILOT_PROMPT.format(
        **ctx,
        transaction_data=json.dumps(record, default=str),
        if_status="ANOMALY" if if_row["if_flag"] else "normal",
        if_score=f"{if_row['if_decision']:.4f}",
        vendor=row["vendor_id"],
        vendor_tx_count=int(vendor_counts.get(row["vendor_id"], 0)),
        abs_amount=f"{amount:,.2f}",
        amount_percentile=f"{percentile:.1f}",
        signals_block=signals_block,
    )
    prompt = mask_vendors(prompt, vendor_ids)
    if "GHOST" in prompt.upper():
        raise RuntimeError(f"row {row.name}: ghost-vendor label leaked into the prompt after masking")
    return prompt


def load_cache() -> dict[str, dict]:
    if not CACHE_PATH.exists():
        return {}
    cache = {}
    for line in CACHE_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entry = json.loads(line)
            cache[entry["key"]] = entry
    return cache


def cache_key(model: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}\x00{prompt}".encode("utf-8")).hexdigest()


ANOMALY_RE = re.compile(r'"anomaly"\s*:\s*"?([01])')


def parse_response(text: str) -> tuple[int | None, str]:
    try:
        obj = json.loads(text)
        return int(obj["anomaly"]), str(obj.get("explanation", ""))
    except (ValueError, KeyError, TypeError):
        m = ANOMALY_RE.search(text)  # truncated at MAX_TOKENS mid-explanation, verdict still readable
        return (int(m.group(1)) if m else None), text


def call_llm(prompt: str, model: str) -> str:
    resp = httpx.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": LLM_TEMPERATURE,
                "seed": LLM_SEED,
                "num_ctx": LLM_NUM_CTX,
                "num_predict": MAX_TOKENS,
            },
        },
        timeout=LLM_TIMEOUT_S,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def run_llm(prompts: dict[tuple[int, str], str], model: str) -> dict[tuple[int, str], dict]:
    cache = load_cache()
    todo = [(k, p) for k, p in prompts.items() if cache_key(model, p) not in cache]
    print(f"LLM: {len(prompts) - len(todo)} cached, {len(todo)} to run on {model}")
    if todo:
        try:
            httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=10).raise_for_status()
        except httpx.HTTPError as exc:
            raise SystemExit(f"Ollama not reachable at {OLLAMA_HOST} ({exc}). Start it with `ollama serve`.")

    started = time.monotonic()
    with CACHE_PATH.open("a", encoding="utf-8") as fh:
        for i, ((row_id, variant), prompt) in enumerate(todo, 1):
            text = call_llm(prompt, model)
            entry = {"key": cache_key(model, prompt), "model": model, "row_id": row_id, "variant": variant, "response": text}
            fh.write(json.dumps(entry) + "\n")
            fh.flush()  # a killed run keeps everything finished so far
            cache[entry["key"]] = entry
            per_call = (time.monotonic() - started) / i
            print(f"  [{i}/{len(todo)}] row {row_id} {variant}: {per_call:.0f}s/call, ~{per_call * (len(todo) - i) / 3600:.1f}h left")

    results = {}
    for k, prompt in prompts.items():
        anomaly, explanation = parse_response(cache[cache_key(model, prompt)]["response"])
        results[k] = {"anomaly": anomaly, "explanation": explanation}
    return results


# ---------------------------------------------------------------------------
# Metrics (weighted back to the test subset's composition)
# ---------------------------------------------------------------------------


def _binary(y, pred, w) -> dict:
    tp = float((w * ((pred == 1) & (y == 1))).sum())
    fp = float((w * ((pred == 1) & (y == 0))).sum())
    fn = float((w * ((pred == 0) & (y == 1))).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate(df: pd.DataFrame, score_col: str, pred_col: str) -> dict:
    y = df["is_fraud"].to_numpy()
    s = df[score_col].to_numpy(dtype=float)
    p = df[pred_col].to_numpy()
    w = df["weight"].to_numpy()
    strata = [np.flatnonzero(df["typology"].to_numpy() == t) for t in df["typology"].unique()]

    def point(idx):
        if len(set(y[idx])) < 2:  # tiny --limit smoke runs can draw one class only
            return {"pr_auc": float("nan"), "roc_auc": float("nan"), **_binary(y[idx], p[idx], w[idx])}
        return {
            "pr_auc": float(average_precision_score(y[idx], s[idx], sample_weight=w[idx])),
            "roc_auc": float(roc_auc_score(y[idx], s[idx], sample_weight=w[idx])),
            **_binary(y[idx], p[idx], w[idx]),
        }

    full = point(np.arange(len(df)))
    rng = np.random.default_rng(RANDOM_SEED)
    boots = [point(np.concatenate([rng.choice(ix, size=len(ix)) for ix in strata])) for _ in range(N_BOOTSTRAP)]
    out = {m: {"point": v, "ci_low": float(np.nanpercentile([b[m] for b in boots], 2.5)),
               "ci_high": float(np.nanpercentile([b[m] for b in boots], 97.5))} for m, v in full.items()}
    out["recall_by_typology"] = {
        t: float(df.loc[df["typology"] == t, pred_col].mean()) for t in TYPOLOGIES if (df["typology"] == t).any()
    }
    out["genuine_false_positive_rate"] = float(df.loc[df["typology"] == "none", pred_col].mean())
    return out


METHODS = {
    # name: (score column, prediction column, uses labels?)
    "isolation_forest": ("if_score", "if_flag", "no"),
    "auditcopilot": ("llm_auditcopilot", "llm_auditcopilot", "no"),
    "auditcopilot_signals": ("llm_auditcopilot_signals", "llm_auditcopilot_signals", "no"),
    "agents_label_free": ("agents_label_free_score", "agents_label_free_pred", "no"),
    "fusion": ("fused_score", "fusion_pred", "yes"),
}


def print_table(results: dict, n_rows: int, base_rate: float) -> None:
    short = {"duplicate_invoice": "dup", "ghost_vendor": "ghost", "threshold_split": "split",
             "desc_amount_mismatch": "desc", "altered_amount": "alter"}
    print(f"\nComparison on {n_rows} sampled test rows (weighted); base rate = {base_rate:.3f} "
          "(a random scorer's PR-AUC)\n")
    print("| Method | Labels | PR-AUC [95% CI] | ROC-AUC | P | R | F1 | "
          + " | ".join(f"R:{short[t]}" for t in TYPOLOGIES) + " | FPR genuine |")
    print("|" + "---|" * (8 + len(TYPOLOGIES)))
    for name, res in results.items():
        pr = res["pr_auc"]
        typ = " | ".join(f"{res['recall_by_typology'].get(t, float('nan')):.2f}" for t in TYPOLOGIES)
        print(f"| {name} | {METHODS[name][2]} | {pr['point']:.3f} [{pr['ci_low']:.3f}-{pr['ci_high']:.3f}] | "
              f"{res['roc_auc']['point']:.3f} | {res['precision']['point']:.2f} | {res['recall']['point']:.2f} | "
              f"{res['f1']['point']:.2f} | {typ} | {res['genuine_false_positive_rate']:.2f} |")
    print("\nLLM rows output a binary verdict, so their PR-AUC is a single operating point, not a ranking.")


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n-per-typology", type=int, default=40)
    parser.add_argument("--n-genuine", type=int, default=200)
    parser.add_argument("--limit", type=int, default=None, help="only send the first N sample rows to the LLM")
    parser.add_argument("--no-llm", action="store_true", help="skip both LLM rows; print one prompt per variant")
    parser.add_argument("--model", default=OLLAMA_MODEL)
    args = parser.parse_args()

    seed_everything()
    splits = load_splits()
    fusion = pd.read_parquet(EXPERIMENTS_DIR / "fusion_scores.parquet")
    fusion_threshold = json.loads((EXPERIMENTS_DIR / "fusion_results.json").read_text())["f1_threshold"]
    fusion_test = fusion.loc[fusion["split"] == "test"]
    base_rate = float(fusion_test["is_fraud"].mean())

    sample = draw_sample(fusion_test, args.n_per_typology, args.n_genuine)
    print(f"sample: {len(sample)} rows from {len(fusion_test)} test-subset rows "
          f"({sample.groupby('typology').size().to_dict()})")
    ids = sample["row_id"].to_numpy()

    print("fitting Isolation Forest on train (label-free) ...")
    if_scores, if_counts = isolation_forest_scores(splits, splits["test"].index.to_numpy())
    print("running document + reconciliation agents (label-free fits) ...")
    agents = agent_evidence(splits, sample)

    sample = sample.set_index("row_id")
    sample = sample.join(if_scores.loc[ids]).join(agents.drop(columns=["evidence"]))
    sample["agents_label_free_pred"] = (sample["agents_label_free_score"] >= AGENTS_LABEL_FREE_THRESHOLD).astype(int)
    sample["fusion_pred"] = (sample["fused_score"] >= fusion_threshold).astype(int)

    test = splits["test"]
    ctx = dataset_context(splits["train"], if_counts)
    train_amounts_sorted = np.sort(splits["train"]["DMBTR"].abs().to_numpy())
    vendor_counts = test["vendor_id"].value_counts()  # vendor-disjoint split: a test vendor's history is in test
    prompts = {}
    llm_ids = ids if args.limit is None else ids[: args.limit]
    for row_id in llm_ids:
        row = test.loc[row_id]
        prompts[(int(row_id), "auditcopilot")] = build_prompt(
            row, ctx, train_amounts_sorted, vendor_counts, if_scores.loc[row_id], None)
        prompts[(int(row_id), "auditcopilot_signals")] = build_prompt(
            row, ctx, train_amounts_sorted, vendor_counts, if_scores.loc[row_id], agents.loc[row_id])

    methods = dict(METHODS)
    if args.no_llm:
        for variant in ("auditcopilot", "auditcopilot_signals"):
            print(f"\n----- sample {variant} prompt (row {llm_ids[0]}) -----\n{prompts[(int(llm_ids[0]), variant)]}")
        methods = {k: v for k, v in methods.items() if not k.startswith("auditcopilot")}
        eval_df = sample
    else:
        llm = run_llm(prompts, args.model)
        parse_failures = {}
        for variant in ("auditcopilot", "auditcopilot_signals"):
            col = f"llm_{variant}"
            verdicts = pd.Series({rid: llm[(rid, v)]["anomaly"] for (rid, v) in llm if v == variant}, dtype=float)
            parse_failures[variant] = int(verdicts.isna().sum())
            sample[col] = verdicts.fillna(0)  # unparseable = not flagged
        print(f"unparseable LLM responses (counted as not flagged): {parse_failures}")
        eval_df = sample.loc[llm_ids]  # every method on the same rows, even mid-run

    results = {name: evaluate(eval_df, score, pred) for name, (score, pred, _) in methods.items()}
    print_table(results, len(eval_df), base_rate)

    out = {
        "base_paper": "AuditCopilot, arXiv 2512.02726",
        "model": None if args.no_llm else args.model,
        "n_rows_evaluated": len(eval_df),
        "sample_composition": eval_df.groupby("typology").size().to_dict(),
        "test_subset_base_rate": base_rate,
        "fusion_threshold": fusion_threshold,
        "agents_label_free_threshold": AGENTS_LABEL_FREE_THRESHOLD,
        "isolation_forest_train_anomaly_rate": if_counts["if_anomaly_rate"],
        "complete": args.limit is None and not args.no_llm,
        "results": results,
    }
    (EXPERIMENTS_DIR / "comparison_results.json").write_text(json.dumps(out, indent=2))
    eval_df.reset_index().to_parquet(EXPERIMENTS_DIR / "comparison_scores.parquet", index=False)
    print(f"\nwrote {EXPERIMENTS_DIR / 'comparison_results.json'}, comparison_scores.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
