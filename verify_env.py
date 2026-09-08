"""Phase 0 environment check.

Run after install. Verifies the interpreter, every declared dependency, that
seeding actually produces identical draws, and that the Ollama daemon is
reachable with the configured model.

    python verify_env.py

Exit code 0 = everything the project needs is present and reproducible.
"""

from __future__ import annotations

import importlib
import platform
import sys

from config import (
    OLLAMA_HOST,
    OLLAMA_MODEL,
    RANDOM_SEED,
    ensure_dirs,
    seed_everything,
)

# (import name, pip name) -- they differ often enough to be worth spelling out.
#
# Split required/optional because a missing OPTIONAL package used to fail the
# whole check, and `check_determinism()` is skipped whenever anything is
# missing -- so on a machine without paddleocr (which is every machine here;
# OCR runs on Kaggle) the reproducibility check silently never ran.
REQUIRED_PACKAGES = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("pyarrow", "pyarrow"),
    ("scipy", "scipy"),
    ("sklearn", "scikit-learn"),
    ("xgboost", "xgboost"),
    ("rapidfuzz", "rapidfuzz"),  # reconciliation agent's vendor fuzzy match
    ("shap", "shap"),
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("pdf2image", "pdf2image"),
    ("PIL", "pillow"),
    ("reportlab", "reportlab"),
    ("ollama", "ollama"),
    ("httpx", "httpx"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("streamlit", "streamlit"),
    ("jinja2", "jinja2"),
    ("matplotlib", "matplotlib"),
]

# Declared in requirements.txt but not imported by anything on the local
# pipeline path. Absence is reported, never fatal.
OPTIONAL_PACKAGES = [
    ("paddle", "paddlepaddle", "OCR runs on Kaggle; needed only by experiments/smoke_document.py"),
    ("paddleocr", "paddleocr", "same -- see README 'Document pipeline'"),
    ("lightgbm", "lightgbm", "declared, but the transaction agent is xgboost + IsolationForest"),
    ("sqlalchemy", "sqlalchemy", "declared, but the audit log uses stdlib sqlite3"),
    ("seaborn", "seaborn", "declared, not imported by any module"),
]

OK = "  ok  "
FAIL = " FAIL "
WARN = " warn "


# This project's environment is installed under 3.13. 3.11 is NOT present on
# this machine, and the default `python` on PATH is a bare 3.14 with none of
# the project's dependencies -- which surfaces as ModuleNotFoundError deep in
# a pipeline script rather than as an obvious wrong-interpreter error.
SUPPORTED_PYTHON = (3, 13)


def check_python() -> bool:
    version = sys.version_info[:2]
    good = version == SUPPORTED_PYTHON
    print(f"[{OK if good else WARN}] python {platform.python_version()} ({sys.executable})")
    if not good:
        want = ".".join(str(v) for v in SUPPORTED_PYTHON)
        print(f"         expected {want}; run scripts as `py -{want} <script>`")
    return good


def check_imports() -> list[str]:
    """Returns the missing REQUIRED packages; optional ones only warn."""
    missing: list[str] = []
    for module, pip_name in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(module)
        except Exception as exc:  # ImportError, but paddle can raise OSError
            missing.append(pip_name)
            print(f"[{FAIL}] {pip_name:<16} {type(exc).__name__}: {exc}")
            continue
        print(f"[{OK}] {pip_name:<16} {getattr(mod, '__version__', '?')}")

    for module, pip_name, why in OPTIONAL_PACKAGES:
        try:
            mod = importlib.import_module(module)
        except Exception:
            print(f"[{WARN}] {pip_name:<16} not installed -- {why}")
            continue
        print(f"[{OK}] {pip_name:<16} {getattr(mod, '__version__', '?')}")
    return missing


def check_determinism() -> bool:
    """Seeding must produce identical draws across independent calls."""
    import random

    import numpy as np

    seed_everything(RANDOM_SEED)
    a = (random.random(), np.random.rand(), hash_probe())

    seed_everything(RANDOM_SEED)
    b = (random.random(), np.random.rand(), hash_probe())

    good = a == b
    print(f"[{OK if good else FAIL}] determinism  seed={RANDOM_SEED} reproducible={good}")
    if not good:
        print(f"         run A: {a}")
        print(f"         run B: {b}")
    return good


def hash_probe() -> float:
    """A draw that depends on torch when present, numpy otherwise."""
    try:
        import torch

        return float(torch.rand(1).item())
    except ImportError:
        import numpy as np

        return float(np.random.rand())


def check_poppler() -> bool:
    """pdf2image imports cleanly without poppler, then fails at first use.

    Catch it here rather than in week 2 when the document agent hits its first
    real PDF.
    """
    import shutil

    exe = shutil.which("pdftoppm")
    if exe is None:
        print(f"[{FAIL}] poppler      pdftoppm not on PATH")
        print("         pdf2image imports fine but raises at first convert()")
        print("         winget install oschwartz10612.Poppler")
        return False

    print(f"[{OK}] poppler      {exe}")
    return True


def check_ollama() -> bool:
    try:
        import httpx
    except ImportError:
        print(f"[{WARN}] ollama       httpx missing, skipped")
        return False

    try:
        resp = httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=5.0)
        resp.raise_for_status()
    except Exception as exc:
        print(f"[{FAIL}] ollama       daemon unreachable at {OLLAMA_HOST} ({exc})")
        print("         start it with:  ollama serve")
        return False

    names = [m["name"] for m in resp.json().get("models", [])]
    have = OLLAMA_MODEL in names or f"{OLLAMA_MODEL}:latest" in names
    print(f"[{OK if have else FAIL}] ollama       {OLLAMA_HOST} models={names or '[]'}")
    if not have:
        print(f"         configured model {OLLAMA_MODEL!r} not pulled:")
        print(f"         ollama pull {OLLAMA_MODEL}")
    return have


def main() -> int:
    print("fraud-audit environment check")
    print("=" * 60)

    py_ok = check_python()
    print("-" * 60)
    missing = check_imports()
    print("-" * 60)
    det_ok = check_determinism() if not missing else False
    print("-" * 60)
    poppler_ok = check_poppler()
    llm_ok = check_ollama()
    print("=" * 60)

    ensure_dirs()

    if missing:
        print(f"missing required packages: pip install {' '.join(missing)}")
    if not (py_ok and not missing and det_ok and poppler_ok and llm_ok):
        print("environment INCOMPLETE")
        return 1

    print("environment OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
