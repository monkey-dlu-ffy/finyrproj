"""Central configuration for the fraud-audit project.

Every module imports from here. Nothing else defines a seed, a path, or an
LLM endpoint -- if you find yourself writing `random_state=42` inline, import
RANDOM_SEED instead.

Reproducibility contract
------------------------
Any number that appears in a report, a notebook, or the dashboard must be
regenerable by a fresh process that calls `seed_everything()` before touching
data. E5 (stability) is only meaningful if two runs of the same code on the
same input produce the same output; unseeded randomness anywhere in the
pipeline silently voids that claim.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Paths (absolute, so notebooks/ and experiments/ resolve identically)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"              # immutable source drops; read-only
LINKED_DIR = DATA_DIR / "linked"        # entity-resolved / joined records
PROCESSED_DIR = DATA_DIR / "processed"  # model-ready feature tables

MODELS_DIR = PROJECT_ROOT / "models"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
XAI_DIR = PROJECT_ROOT / "xai"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# ---------------------------------------------------------------------------
# Local LLM (Ollama)
# ---------------------------------------------------------------------------

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

# 3b, not 7b. Both pulled and measured on this machine, not assumed:
#   qwen2.5:3b-instruct -> 3.09 tok/s  ->  ~52-67 s per 160-token finding
#   qwen2.5:7b-instruct -> 0.12 tok/s  ->  ~24 MIN per 160-token finding
# 7b is ~26x slower. It is not a load-time artifact: with the model already
# resident (load_duration 0.0 s) generation still ran at 0.12 tok/s, and free
# RAM fell from 0.55 GB to 0.05 GB -- it thrashes the pagefile continuously on
# 7.76 GB total RAM with no GPU. 7b remains pulled for reference; do not use it
# here. Revisit only on hardware with materially more free RAM.
#
# Overridable per-machine without editing code:
#   PowerShell:  $env:OLLAMA_MODEL = "qwen2.5:7b-instruct"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b-instruct")

LLM_TEMPERATURE = 0.0   # greedy decoding -- agent output must be auditable
LLM_SEED = RANDOM_SEED  # necessary but NOT sufficient, see below

# Measured caveat (experiments/probe_llm_determinism.py): at temperature 0 with
# a fixed seed, the FIRST generation after a model load differs from every
# later one. Both regimes are internally repeatable but disagree with each
# other, and memory pressure evicts the model unpredictably. So LLM text is not
# reliably reproducible across sessions: cache it keyed by input hash and treat
# the cache as the artifact E5 is computed over.
LLM_NUM_CTX = 8192
LLM_TIMEOUT_S = 300     # CPU-only inference is slow; fail loudly, not silently

# ---------------------------------------------------------------------------
# Data splits
# ---------------------------------------------------------------------------

TEST_SIZE = 0.20
VAL_SIZE = 0.20         # fraction of the post-test remainder
N_FOLDS = 5

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def seed_everything(seed: int = RANDOM_SEED) -> int:
    """Seed every RNG this project can reach. Call once, at entry point.

    Seeds stdlib `random`, PYTHONHASHSEED, numpy, and torch (CPU + CUDA) when
    those are importable. torch is imported lazily so that lightweight scripts
    and the dashboard do not pay a multi-second import cost.

    Returns the seed, so callers can log exactly what was used.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Trade throughput for run-to-run identity. Required for E5.
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed


def ensure_dirs() -> None:
    """Create the data/model directories. Safe to call repeatedly.

    These are gitignored, so a fresh clone has no data/ or models/ tree until
    something calls this.
    """
    for path in (
        RAW_DIR,
        LINKED_DIR,
        PROCESSED_DIR,
        MODELS_DIR,
        EXPERIMENTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


__all__ = [
    "RANDOM_SEED",
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DIR",
    "LINKED_DIR",
    "PROCESSED_DIR",
    "MODELS_DIR",
    "EXPERIMENTS_DIR",
    "XAI_DIR",
    "DASHBOARD_DIR",
    "NOTEBOOKS_DIR",
    "OLLAMA_HOST",
    "OLLAMA_MODEL",
    "LLM_TEMPERATURE",
    "LLM_SEED",
    "LLM_NUM_CTX",
    "LLM_TIMEOUT_S",
    "TEST_SIZE",
    "VAL_SIZE",
    "N_FOLDS",
    "seed_everything",
    "ensure_dirs",
]
