"""Benchmark local Ollama models so the size decision is measured, not guessed.

Phase 0 asks: does qwen2.5:7b-instruct actually run on this machine, or do we
drop to 3b now rather than in week 5? Answer it with numbers.

    python experiments/bench_llm.py
    python experiments/bench_llm.py --models qwen2.5:7b-instruct qwen2.5:3b-instruct

Reports load time, prompt-eval rate, and generation rate (tokens/sec) on a
prompt shaped like the coordinator's real workload: summarise structured
findings into a short auditor-facing narrative.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from config import LLM_NUM_CTX, LLM_SEED, LLM_TEMPERATURE, OLLAMA_HOST

# Representative of what the coordinator will actually ask for: fuse specialist
# findings into a short, evidence-grounded narrative.
PROMPT = """You are an audit coordinator. Three specialist agents reported on invoice INV-4471.

transaction_agent: amount 48,200 USD is 6.1 sigma above this vendor's 90-day mean; \
paid 03:14 local time; vendor added to master file 9 days ago.
document_agent: OCR confidence 0.71 on the total field; invoice template does not \
match this vendor's 14 prior submissions; no PO number present.
reconciliation_agent: no matching purchase order; counterparty bank account differs \
from the one on file for this vendor.

Write a 120-word finding for a human auditor. State the evidence, then the \
recommended action. Do not speculate beyond the evidence given."""

NS_PER_S = 1_000_000_000


def bench(model: str, host: str, repeats: int) -> dict | None:
    """Run `repeats` generations, return timing stats, or None if unavailable."""
    runs = []
    for i in range(repeats):
        try:
            resp = httpx.post(
                f"{host}/api/generate",
                json={
                    "model": model,
                    "prompt": PROMPT,
                    "stream": False,
                    "options": {
                        "temperature": LLM_TEMPERATURE,
                        "seed": LLM_SEED,
                        "num_ctx": LLM_NUM_CTX,
                    },
                },
                timeout=900.0,
            )
            resp.raise_for_status()
        except Exception as exc:
            print(f"  run {i + 1}: FAILED ({type(exc).__name__}: {exc})")
            return None

        d = resp.json()
        gen_tokens = d.get("eval_count", 0)
        gen_ns = d.get("eval_duration", 1)
        prompt_tokens = d.get("prompt_eval_count", 0)
        prompt_ns = d.get("prompt_eval_duration", 1)

        run = {
            "load_s": d.get("load_duration", 0) / NS_PER_S,
            "total_s": d.get("total_duration", 0) / NS_PER_S,
            "prompt_tokens": prompt_tokens,
            "prompt_tps": prompt_tokens / (prompt_ns / NS_PER_S) if prompt_ns else 0,
            "gen_tokens": gen_tokens,
            "gen_tps": gen_tokens / (gen_ns / NS_PER_S) if gen_ns else 0,
            "response": d.get("response", ""),
        }
        runs.append(run)
        print(
            f"  run {i + 1}: {run['total_s']:6.1f}s total | "
            f"{run['gen_tps']:5.2f} tok/s gen | {run['gen_tokens']} tokens"
        )

    stable = len({r["response"] for r in runs}) == 1 if len(runs) > 1 else None

    return {
        "model": model,
        "runs": len(runs),
        "median_total_s": statistics.median(r["total_s"] for r in runs),
        "median_gen_tps": statistics.median(r["gen_tps"] for r in runs),
        "median_prompt_tps": statistics.median(r["prompt_tps"] for r in runs),
        "first_load_s": runs[0]["load_s"],
        "identical_outputs": stable,
        "sample": runs[0]["response"][:400],
    }


def verdict(median_tps: float, total_s: float) -> str:
    """Usability call for an interactive reviewer dashboard."""
    if median_tps >= 15:
        return "GOOD -- comfortable for interactive use"
    if median_tps >= 7:
        return "OK -- usable interactively, batch preferred"
    if median_tps >= 3:
        return "SLOW -- batch/offline only; painful behind a dashboard"
    return "UNUSABLE -- drop to a smaller model"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        nargs="+",
        default=["qwen2.5:7b-instruct", "qwen2.5:3b-instruct"],
        help="models to benchmark (must already be pulled)",
    )
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--host", default=OLLAMA_HOST)
    ap.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "llm_bench.json"),
    )
    args = ap.parse_args()

    print(f"benchmarking against {args.host}")
    print(f"prompt: {len(PROMPT)} chars | temperature={LLM_TEMPERATURE} seed={LLM_SEED}")
    print("=" * 68)

    results = []
    for model in args.models:
        print(f"\n{model}")
        r = bench(model, args.host, args.repeats)
        if r is None:
            print(f"  skipped (pull it first: ollama pull {model})")
            continue
        r["verdict"] = verdict(r["median_gen_tps"], r["median_total_s"])
        results.append(r)

    if not results:
        print("\nno models benchmarked")
        return 1

    print("\n" + "=" * 68)
    print(f"{'model':<26}{'gen tok/s':>11}{'total s':>10}  verdict")
    print("-" * 68)
    for r in results:
        print(
            f"{r['model']:<26}{r['median_gen_tps']:>11.2f}"
            f"{r['median_total_s']:>10.1f}  {r['verdict']}"
        )

    for r in results:
        if r["identical_outputs"] is False:
            print(f"\nNOTE: {r['model']} produced non-identical output across runs")
            print("      at temperature 0 -- do not treat its text as reproducible.")

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
