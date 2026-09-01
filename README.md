# fraud-audit

Multi-agent fraud auditing over transactions and scanned documents, with an
explainability layer and a reviewer dashboard.

## Layout

```
data/raw/          immutable source drops (gitignored)
data/linked/       entity-resolved / joined records (gitignored)
data/processed/    model-ready feature tables (gitignored)
agents/
  transaction/     anomaly signals over ledger rows
  document/        OCR + field extraction from invoices/receipts
  reconciliation/  cross-source agreement checks
  coordinator/     fuses findings into a ranked case file (owns the LLM)
xai/               SHAP attributions, counterfactuals, evidence rendering
dashboard/         Streamlit review UI + FastAPI scoring service
experiments/       run configs and metrics
models/            trained artifacts (gitignored)
notebooks/         exploration
scripts/           operational utilities (model pulls, setup)
config.py          seed + paths + LLM settings; imported everywhere
```

## Setup

Python **3.11** (not 3.13 — several pinned deps lack 3.13 wheels).

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel

# torch must come from the CPU index on a machine with no NVIDIA GPU,
# otherwise pip pulls a multi-gigabyte CUDA build that cannot be used.
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

Verify:

```powershell
python verify_env.py
```

### System dependencies (not pip-installable)

| Dependency | Needed by | Install |
|---|---|---|
| **poppler** | `pdf2image` | `winget install oschwartz10612.Poppler`, then add its `Library\bin` to `PATH` |
| **Ollama** | coordinator agent | `winget install Ollama.Ollama` |

## Reproducibility contract

`config.RANDOM_SEED = 42` is the single source of truth. Import it; never write
`random_state=42` inline.

Call `seed_everything()` once at every entry point **before** touching data:

```python
from config import seed_everything, RANDOM_SEED
seed_everything()
```

It seeds stdlib `random`, `PYTHONHASHSEED`, numpy, and torch (including
deterministic algorithm selection).

Every reported number must be regenerable by a fresh process. Two runs of the
same code on the same input must agree — otherwise **E5 (stability) is
meaningless**, because a stability metric computed over unseeded randomness is
measuring the RNG, not the model.

Pin the environment alongside results:

```powershell
pip freeze > requirements.lock.txt
```

`requirements.txt` is intent (what we depend on); `requirements.lock.txt` is
the exact environment a result was produced in. Reproducing a past run means
installing from the lock file.

### LLM determinism caveat

The coordinator calls a local LLM. It runs at `temperature=0` with
`seed=RANDOM_SEED`, which makes output *stable across reruns on the same
machine and same model build* — but not identical across Ollama versions,
quantisations, or hardware. Treat LLM text as narration of evidence the
specialist agents already computed. Never let it be the component that decides
a fraud label, or E5 becomes undefendable.

## Local LLM

**Decision: qwen2.5:3b-instruct, not 7b.** Measured, not assumed.

This machine: 7.76 GB RAM total, ~0.5 GB free, 21.6 GB committed to pagefile,
4-core i5-1155G7, Intel Iris Xe (no discrete GPU).

Both models were pulled and benchmarked. This is measured, not extrapolated:

| model | size | gen rate | 160-token finding | verdict |
|---|---|---|---|---|
| qwen2.5:3b-instruct | 1.9 GB | **3.09 tok/s** | ~52–67 s | slow, usable for batch |
| qwen2.5:7b-instruct | 4.7 GB | **0.12 tok/s** | **~24 min** | unusable — thrashes |

**7b is ~26× slower**, and not because of load time: with the model already
resident (`load_duration` 0.0 s) generation still measured 0.12 tok/s, while
free RAM fell from 0.55 GB → 0.05 GB. It pages continuously against 7.76 GB
total RAM with no GPU. 7b stays pulled for reference but must not be used here.

3b is already in the "batch/offline only" band. Do not put a synchronous LLM
call behind a dashboard interaction; precompute narratives in a batch job and
serve them from cache.

```powershell
python experiments/bench_llm.py                  # tok/s per model
python experiments/probe_llm_determinism.py      # is output reproducible?
```

### Pulling a model on a flaky link

A plain `ollama pull` never completed here. Use the supervised wrapper — it is
how both Phase 0 models were actually obtained:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\pull_model.ps1
powershell -ExecutionPolicy Bypass -File scripts\pull_model.ps1 -Model qwen2.5:7b-instruct
```

It retries on TLS drops (Ollama resumes from the partial blob, so attempts make
forward progress) **and** kills a child that hangs without moving bytes — a
plain retry loop only handles the first case and blocks forever on the second.
Progress is measured by free-disk delta, not the partial file's length, because
Ollama pre-allocates the blob to full size up front.

### Reproducibility of LLM output — measured caveat

`probe_llm_determinism.py` found that at `temperature=0` with a fixed seed, the
**first generation after a model load differs from every subsequent one**. Each
regime repeats itself exactly, but the two disagree:

```
resident model    -> sha cae41f3b2b0d (167 chars), stable
after eviction    -> sha 1b83575e6424 (195 chars), stable
```

Seed and `num_thread` are not the cause; model residency is. With ~0.5 GB free
RAM, Ollama evicts and reloads unpredictably, so **a rerun hours later can
produce different text for identical input.**

Consequence: cache LLM output keyed by a hash of its input and treat the cache
as the reproducible artifact. Never let LLM text enter a number that E5 is
computed over.

## Document pipeline

Verified end-to-end by `experiments/smoke_document.py`
(reportlab -> PDF -> pdf2image/poppler -> PaddleOCR), all fields round-tripped:

```
reportlab    0.09s
pdf2image    0.22s   1654x2339 px at 200 dpi
paddleocr   71.11s   11 text lines     <-- the bottleneck
```

**OCR costs ~71 s/page on this CPU.** A 1,000-invoice batch is ~20 hours. Treat
OCR as an offline batch stage with cached, resumable output keyed by file hash;
never call it inside a request. Lower the DPI and revisit the detector size if
that is too slow.

### paddlepaddle oneDNN crash

paddlepaddle 3.3.1 aborts on this CPU with:

```
NotImplementedError: (Unimplemented) ConvertPirAttribute2RuntimeAttribute
not support [pir::ArrayAttribute<pir::DoubleAttribute>]
  (onednn_instruction.cc:118)
```

Fix applied in `smoke_document.py`: set `FLAGS_use_mkldnn=0` **before** paddle
is imported and pass `enable_mkldnn=False` to `PaddleOCR`. Any new module that
imports paddle must do the same, or it will crash the same way.

## Data policy

`data/` and `models/` are gitignored. **Never commit datasets.** The directory
skeleton is preserved with `.gitkeep` files; the ignore pattern
(`data/**` + `!data/**/` + `!data/**/.gitkeep`) keeps the structure while
excluding every file inside it.
