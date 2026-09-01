"""Is local LLM output actually reproducible on this machine?

E5 (stability) requires that a rerun produces the same result. `bench_llm.py`
observed non-identical output at temperature 0 with a fixed seed, which would
be fatal if LLM text sat anywhere in the decision path. This isolates the
cause so the finding is evidence, not a guess.

Tests three configurations, N runs each:
  1. temperature=0, seed fixed                  (the intended config)
  2. temperature=0, seed fixed, num_thread=1    (removes parallel-reduction
                                                 float non-associativity)
  3. temperature=0, no seed                     (control)

    python experiments/probe_llm_determinism.py --model qwen2.5:3b-instruct
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from config import LLM_NUM_CTX, LLM_SEED, OLLAMA_HOST

# Short prompt: we are testing identity, not throughput. Keep generation small
# so the probe finishes on a slow CPU.
PROMPT = (
    "Invoice INV-4471 is 6.1 sigma above the vendor mean and has no matching "
    "purchase order. In exactly two sentences, state the finding for an auditor."
)


def generate(model: str, host: str, options: dict) -> str:
    resp = httpx.post(
        f"{host}/api/generate",
        json={
            "model": model,
            "prompt": PROMPT,
            "stream": False,
            "options": options,
        },
        timeout=900.0,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def unload(model: str, host: str) -> None:
    """Force Ollama to evict the model from memory (keep_alive=0).

    On a memory-starved machine this happens on its own between requests, so
    reproducibility must survive it to be worth anything.
    """
    try:
        httpx.post(
            f"{host}/api/generate",
            json={"model": model, "prompt": "", "keep_alive": 0},
            timeout=120.0,
        )
    except Exception:
        pass


def run_reload_config(name: str, model: str, host: str, options: dict,
                      n: int) -> tuple[bool, str]:
    """Same as run_config, but evicts the model before every generation."""
    print(f"\n{name}")
    print(f"  options: {options}  (model evicted between runs)")
    outs = []
    for i in range(n):
        unload(model, host)
        try:
            out = generate(model, host, options)
        except Exception as exc:
            print(f"  run {i + 1}: FAILED {type(exc).__name__}: {exc}")
            return False, ""
        outs.append(out)
        print(f"  run {i + 1}: sha={digest(out)} len={len(out)}")

    identical = len(set(outs)) == 1
    print(f"  -> {'IDENTICAL' if identical else 'DIVERGENT'} "
          f"({len(set(outs))} distinct of {n})")
    return identical, outs[0]


def run_config(name: str, model: str, host: str, options: dict,
               n: int) -> tuple[bool, str]:
    print(f"\n{name}")
    print(f"  options: {options}")
    outs = []
    for i in range(n):
        try:
            out = generate(model, host, options)
        except Exception as exc:
            print(f"  run {i + 1}: FAILED {type(exc).__name__}: {exc}")
            return False, ""
        outs.append(out)
        print(f"  run {i + 1}: sha={digest(out)} len={len(out)}")

    identical = len(set(outs)) == 1
    print(f"  -> {'IDENTICAL' if identical else 'DIVERGENT'} "
          f"({len(set(outs))} distinct of {n})")

    if not identical:
        a, b = outs[0], next(o for o in outs if o != outs[0])
        diff = list(difflib.unified_diff(
            a.split(), b.split(), lineterm="", n=0))[:14]
        print("  first divergence:")
        for line in diff:
            print(f"    {line}")

    return identical, outs[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:3b-instruct")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--host", default=OLLAMA_HOST)
    args = ap.parse_args()

    base = {"temperature": 0.0, "num_ctx": LLM_NUM_CTX, "num_predict": 80}

    print(f"determinism probe: {args.model} @ {args.host}")
    print("=" * 66)

    results = {
        "temp0 + seed": run_config(
            "1. temperature=0, seed fixed",
            args.model, args.host, {**base, "seed": LLM_SEED}, args.runs,
        ),
        "temp0 + seed + 1 thread": run_config(
            "2. temperature=0, seed fixed, num_thread=1",
            args.model, args.host,
            {**base, "seed": LLM_SEED, "num_thread": 1}, args.runs,
        ),
        "temp0 no seed": run_config(
            "3. temperature=0, no seed (control)",
            args.model, args.host, dict(base), args.runs,
        ),
        "temp0 + seed, across reloads": run_reload_config(
            "4. temperature=0, seed fixed, model evicted between runs",
            args.model, args.host, {**base, "seed": LLM_SEED}, args.runs,
        ),
    }

    print("\n" + "=" * 66)
    for name, (ok, _) in results.items():
        print(f"  {name:<28} {'reproducible' if ok else 'NOT reproducible'}")

    # Being self-consistent within a regime is not enough. If the resident and
    # the evicted regime each repeat themselves but disagree with each other,
    # output still changes whenever memory pressure evicts the model -- which
    # is the normal case on this box.
    resident_ok, resident_out = results["temp0 + seed"]
    reload_ok, reload_out = results["temp0 + seed, across reloads"]
    cross_regime_match = bool(resident_out) and resident_out == reload_out
    print(f"  {'resident vs evicted agree':<28} {cross_regime_match}")
    print("=" * 66)

    if resident_ok and reload_ok and cross_regime_match:
        print("Intended config is reproducible, and survives model eviction.")
    elif resident_ok and reload_ok and not cross_regime_match:
        print("Each memory regime repeats itself, but they DISAGREE with each")
        print("other: a model that stayed resident and one that was evicted")
        print("produce different text for identical input. With ~0.5 GB free")
        print("RAM, eviction happens on its own, so a rerun hours later can")
        print("differ. Cache LLM text keyed by input hash and treat the cache,")
        print("not the model, as the reproducible artifact.")
    elif results["temp0 + seed"][0]:
        print("Reproducible only while the model stays resident. This machine")
        print("has ~0.5 GB free RAM, so Ollama evicts and reloads routinely --")
        print("meaning a rerun hours later can differ. Cache LLM text keyed by")
        print("its input hash and treat the cache, not the model, as the")
        print("reproducible artifact.")
    elif results["temp0 + seed + 1 thread"][0]:
        print("Multi-threaded float reduction is the cause: identical output")
        print("only with num_thread=1. Single-threaded is far too slow to use,")
        print("so treat LLM text as NON-reproducible and keep it out of any")
        print("number that E5 is computed over.")
    else:
        print("Output is not reproducible even single-threaded. LLM text must")
        print("not appear in any reported metric; narrate evidence only.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
