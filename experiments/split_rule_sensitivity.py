"""How much does fusion depend on the split-purchase rule and on the
estimated approval limit it keys on? Also breaks down fusion's remaining test
errors by the rules that fired.

Reruns only the reconciliation agent under three approval limits and refits
the meta-classifier exactly as scripts/run_fusion.py does (class-balanced
logistic regression on train, F1-optimal threshold on val), reusing the
transaction/document scores saved in experiments/fusion_scores.parquet -- the
5-fold transaction retraining is the slow part and doesn't depend on the
limit. The "as_run" config must reproduce fusion_results.json; it is the
check that the refit matches.

  as_run              all-train 98th percentile (run_fusion.py's limit)
  no_split_rule       approval_limit=None, rule disabled
  genuine_train_limit genuine-train 98th percentile -- what a label-using fit
                      would estimate; lands below the largest split pieces

    py -3.13 experiments/split_rule_sensitivity.py

Writes experiments/split_rule_sensitivity.json.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.reconciliation.rules import APPROVAL_LIMIT_QUANTILE, ReconciliationAgent
from config import EXPERIMENTS_DIR, LINKED_DIR, RANDOM_SEED, seed_everything
from linked_pipeline.invoices import resolve_vendor_name

FEATURE_COLS = ["transaction_score", "document_score", "reconciliation_score"]


def main() -> int:
    seed_everything()
    splits = {s: pd.read_parquet(LINKED_DIR / f"{s}.parquet") for s in ("train", "val", "test")}
    manifest = pd.read_parquet(LINKED_DIR / "images" / "manifest.parquet").set_index("row_id")
    vendor_master = pd.read_json(LINKED_DIR / "vendor_master.json")
    vendor_name_by_id = dict(zip(vendor_master["vendor_id"], vendor_master["vendor_name"]))
    for df in splits.values():
        df["vendor_name"] = df["vendor_id"].map(lambda v: resolve_vendor_name(v, vendor_name_by_id))
    full_ledger = pd.concat([df.reset_index(names="row_id").assign(split=name) for name, df in splits.items()])
    saved = pd.read_parquet(EXPERIMENTS_DIR / "fusion_scores.parquet").set_index(["split", "row_id"])

    train = splits["train"]
    genuine_train = train.loc[train["is_fraud"] == 0]
    high_amount_threshold = float(genuine_train["DMBTR"].quantile(0.9))
    limits = {
        "as_run": float(train["DMBTR"].quantile(APPROVAL_LIMIT_QUANTILE)),
        "no_split_rule": None,
        "genuine_train_limit": float(genuine_train["DMBTR"].quantile(APPROVAL_LIMIT_QUANTILE)),
    }

    out = {}
    for name, limit in limits.items():
        agent = ReconciliationAgent(
            vendor_master_ids=set(vendor_master["vendor_id"]),
            high_amount_threshold=high_amount_threshold,
            approval_limit=limit,
        )
        rows, fired = [], {}
        for split_name, df in splits.items():
            split_manifest = manifest.loc[manifest["split"] == split_name]
            sub_ids = split_manifest.index.intersection(df.index)
            joined = df.loc[sub_ids].copy()
            joined["extracted_amount"] = split_manifest.loc[sub_ids, "invoice_amount"]
            joined["extracted_vendor_name"] = split_manifest.loc[sub_ids, "vendor_name"]
            for env in agent.analyze_batch(joined, ledger=full_ledger):
                s = saved.loc[(split_name, env.row_id)]
                rows.append(
                    {
                        "split": split_name,
                        "row_id": env.row_id,
                        "is_fraud": int(s["is_fraud"]),
                        "typology": s["typology"],
                        "transaction_score": s["transaction_score"],
                        "document_score": s["document_score"],
                        "reconciliation_score": env.score,
                    }
                )
                fired[(split_name, env.row_id)] = tuple(sorted(e["type"] for e in env.evidence))

        meta = pd.DataFrame(rows)
        meta_train = meta.loc[meta["split"] == "train"]
        clf = LogisticRegression(class_weight="balanced", random_state=RANDOM_SEED)
        clf.fit(meta_train[FEATURE_COLS], meta_train["is_fraud"])
        meta["fused_score"] = clf.predict_proba(meta[FEATURE_COLS])[:, 1]
        val = meta.loc[meta["split"] == "val"]
        test = meta.loc[meta["split"] == "test"]

        candidate_thresholds = np.linspace(0.01, 0.99, 99)
        f1s = [f1_score(val["is_fraud"], val["fused_score"] >= t, zero_division=0) for t in candidate_thresholds]
        threshold = float(candidate_thresholds[int(np.argmax(f1s))])

        y = test["is_fraud"].to_numpy()
        pred = test["fused_score"].to_numpy() >= threshold
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        is_split = (test["typology"] == "threshold_split").to_numpy()
        result = {
            "approval_limit": limit,
            "f1_threshold": threshold,
            "test_pr_auc": float(average_precision_score(y, test["fused_score"])),
            "test_roc_auc": float(roc_auc_score(y, test["fused_score"])),
            "test_precision": tp / (tp + fp) if (tp + fp) else 0.0,
            "test_recall": tp / (tp + fn) if (tp + fn) else 0.0,
            "tp": tp, "fp": fp, "fn": fn,
            "threshold_split_recall": float(pred[is_split].mean()),
            "threshold_split_n": int(is_split.sum()),
        }

        if name == "as_run":
            # Split rule over the FULL test ledger, not just the imaged subset:
            # how often does it fire on genuine postings?
            split_index = agent._split_index(full_ledger)
            test_ledger = full_ledger.loc[full_ledger["split"] == "test"]
            in_group = test_ledger.set_index(["vendor_id", "po_number"]).index.isin(list(split_index))
            genuine = (test_ledger["is_fraud"] == 0).to_numpy()
            pieces = (test_ledger["typology"] == "threshold_split").to_numpy()
            result["split_rule_full_test_ledger"] = {
                "genuine_fired": int((in_group & genuine).sum()),
                "genuine_total": int(genuine.sum()),
                "threshold_split_fired": int((in_group & pieces).sum()),
                "threshold_split_total": int(pieces.sum()),
            }
            fp_ids = test.loc[pred & (y == 0), "row_id"]
            fn_rows = test.loc[~pred & (y == 1)]
            result["false_positive_rules"] = {
                " + ".join(k) or "(none)": v for k, v in Counter(fired[("test", r)] for r in fp_ids).items()
            }
            result["false_negative_typologies"] = dict(Counter(fn_rows["typology"]))
            result["false_negative_rules"] = {
                " + ".join(k) or "(none)": v for k, v in Counter(fired[("test", r)] for r in fn_rows["row_id"]).items()
            }

        out[name] = result
        print(f"{name}: limit={limit} PR-AUC {result['test_pr_auc']:.4f} ROC-AUC {result['test_roc_auc']:.4f} "
              f"P {result['test_precision']:.4f} R {result['test_recall']:.4f} "
              f"threshold_split recall {result['threshold_split_recall']:.3f}")

    print(json.dumps(out, indent=2))
    out_path = EXPERIMENTS_DIR / "split_rule_sensitivity.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
