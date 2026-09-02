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

## The pipeline (SAP-ERP corpus)

This is the real system: a synthetic SAP-ERP journal-entry ledger
(`dummy/erp/fraud_dataset_v2.csv`, from GitiHubi/deepAD) linked to rendered
invoice images, with 5 fraud typologies injected (`duplicate_invoice`,
`ghost_vendor`, `threshold_split`, `desc_amount_mismatch`, `altered_amount`),
run through 4 agents, fused, and served on a review dashboard.

There's a separate, older reference track (`data_pipeline/`, the non-`_erp`
files in `agents/transaction/`) built on the Sparkov credit-card dataset —
kept around, not part of this pipeline, don't confuse the two.

**Currently runs on a ground-truth fast path for document fields, not real
OCR.** Two Kaggle attempts at real PaddleOCR extraction both failed (one too
slow — ~8.5hr for 792 images; one broke on a CUDA library version conflict).
Full story in `scripts/kaggle_ocr_extract.py`'s docstring. The document and
reconciliation agents currently read the linked corpus's own known
vendor/amount/invoice fields as if they'd been extracted — that's a
deliberate stand-in, not a bug, and it's why the reconciliation and fusion
numbers below are believable but not proof the OCR step works.

To reproduce from scratch:

```powershell
python scripts\build_linked_corpus.py       # ledger + synthesized vendor/date/invoice fields, typology injection, split
python scripts\generate_invoices.py         # renders + degrades invoice images for ~16% of the corpus (all fraud rows + a genuine sample)
python scripts\train_transaction_agent.py   # xgboost + IsolationForest + isotonic calibration, local (~1s, small corpus)
python scripts\run_reconciliation_agent.py  # rule-based checks; prints per-typology fire rate as a sanity check
python scripts\run_fusion.py                # 5-fold OOF stacking -> logistic regression meta-classifier
python scripts\run_xai.py                   # SHAP global/local, deletion faithfulness, seed stability
python scripts\run_experiments.py           # E1/E2/E3/E4/E7 + bootstrap CIs, from run_fusion.py's output
```

Each writes its result to `experiments/*.json` (metrics) or `experiments/*.parquet`
(scored rows) — read those directly rather than re-running everything to see
current numbers. Headline result: fusion test PR-AUC 0.92, recall 1.0 at 91%
precision; reconciliation alone carries almost the whole signal (E4 ablation:
dropping it collapses PR-AUC to 0.13).

To view the dashboard:

```powershell
# terminal 1
python -m uvicorn dashboard.api:app --port 8811
# terminal 2
python -m streamlit run dashboard\app.py
```

Then open `http://localhost:8501`. Sidebar filters by risk tier; click a
case in the left panel to see its invoice image, agent scores, fired
reconciliation rules, and an LLM-generated narrative (first view per case is
slow — ~30-60s to generate, then cached). Accept/Reject/Escalate buttons log
to an append-only SQLite audit log (`experiments/audit_log.sqlite3`).

**Known gaps, not fixed:**
- `threshold_split` typology has no dedicated reconciliation rule (it's a
  cross-invoice grouping pattern; no single-row check catches it)
- Document agent's text-risk is keyword/rule-based, not the spec's
  fine-tuned sentence-transformer classifier
- No LIME baseline (XAI has SHAP only), no LayoutLMv3 fine-tuning on DocILE,
  no auditor survey, no demo video, no formal write-up

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
