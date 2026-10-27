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

This is the real system: the full synthetic SAP-ERP journal-entry ledger
(`dummy/erp/fraud_dataset_v2.csv`, from GitiHubi/deepAD, 533,009 rows) linked
to rendered invoice images, with 5 fraud typologies injected
(`duplicate_invoice`, `ghost_vendor`, `threshold_split`,
`desc_amount_mismatch`, `altered_amount`, ~2% combined rate ->
~11,000 fraud rows), run through 4 agents, fused, and served on a review
dashboard. An earlier build used a 5,000-row subsample of the same source
for time-budget reasons; the full ledger is now used throughout.

Document images are capped at 20,000 (every fraud row + a matched genuine
sample), not one per ledger row -- at this corpus's ~213KB/image rendering
cost, imaging all 533,009 rows would need ~110GB, far beyond what a laptop's
free disk can hold. See `scripts/generate_invoices.py`'s docstring for the
disk-safety check that enforces this.

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
py -3.13 scripts\build_linked_corpus.py       # full 533,009-row ledger, synthesized vendor/date/invoice fields, typology injection, split
py -3.13 scripts\generate_invoices.py         # renders + degrades invoice images, capped at 20,000 (all fraud rows + a genuine sample), parallelized across 8 processes (~90 min)
```

`run_fusion.py` retrains the transaction agent 5x in-process for its OOF
scores; at full scale that ran locally in a few minutes with ~8GB RAM free
(the current results were produced that way). If RAM is short, the Kaggle
path still works — see `scripts/kaggle_train_transaction_erp.py`'s docstring
for the upload/push/download steps, and drop its output
(`transaction_scores.parquet`) into `experiments/`; `run_fusion.py` uses it
instead of retraining whenever that file exists, so delete a stale one after
a corpus rebuild.

```powershell
py -3.13 scripts\run_reconciliation_agent.py  # rule-based checks; prints per-typology fire rate as a sanity check
py -3.13 scripts\run_fusion.py                # 5-fold OOF stacking -> logistic regression meta-classifier
py -3.13 scripts\run_xai.py                   # SHAP global/local, deletion faithfulness, seed stability
py -3.13 scripts\run_experiments.py           # E1/E2/E3/E4/E7 + bootstrap CIs, from run_fusion.py's output
py -3.13 experiments\split_rule_sensitivity.py  # split-rule ablation, approval-limit sensitivity, remaining-error breakdown
```

Each writes its result to `experiments/*.json` (metrics) or `experiments/*.parquet`
(scored rows) — read those directly rather than re-running everything to see
current numbers. Headline result (full 533K-row corpus): fusion test PR-AUC
0.998 (95% CI 0.997-0.999), precision 0.994/recall 0.997 (F1 0.995; 2,883 TP,
18 FP, 10 FN) at the F1-optimal threshold -- read against a 0.599 base rate,
since the document-linked test subset is 60% fraud. Reconciliation carries
nearly all of the signal (E4 ablation: dropping it takes PR-AUC from 0.998 to
0.720, and ROC-AUC from 0.996 to 0.636; dropping the transaction or document
agent costs at most 0.003). Label-free agents reach 0.966 on the AuditCopilot
comparison sample, against 0.603 for AuditCopilot and 0.640 for Isolation
Forest.

These numbers are after the duplicate-label correction. Before it, fusion was
at PR-AUC 0.854, precision 0.84/recall 0.997 (F1 0.911), and 544 of its 556
test false positives were the genuine originals of duplicated invoices:
`inject_duplicate_invoice` picked which copy was "fraud" without regard to
posting order (the fraud copy was the later one in only 50.3% of pairs), so
the duplicate rule had to flag both. Now the injector always makes the later
posting (by `posting_date`, then `BELNR`) the duplicate, and the
`duplicate_invoice_number` rule only fires on a posting that has an earlier
one -- it fires on 0 genuine rows. This fix came from error analysis on the
test split, so the paper reports both sets of numbers. The pre-correction
metrics are in `experiments/*.json` at commit 8a9bf82; to regenerate that
corpus and its scores, check out 8a9bf82 and re-run the pipeline above.

The remaining 18 false positives are all genuine postings with a low-tier
description on an above-p90 amount (the 22% randomly redrawn tiers), which
look exactly like `desc_amount_mismatch`; all 10 false negatives are
`threshold_split` pieces the split-purchase rule misses. Near-perfect scores
here mostly mean the rules mirror the injectors, not that they would catch
real fraud this well.

Transaction-only (E1) is PR-AUC 0.683 -- most of the 5 typologies aren't
purely numeric-feature-visible. That 0.683 is on the 60%-fraud
document-linked subset, though: on the full 128,764-row test split (2.25%
fraud) the same model scores PR-AUC 0.038 against a 0.022 base rate (ROC-AUC
0.614), and no val threshold reaches 50% precision, so
`train_transaction_agent.py` falls back to the max score (threshold 0.30,
recall 0.02) -- see `experiments/transaction_erp_metrics.json`. On its own it
is not a usable detector at the natural fraud rate, and in fusion it adds
little once reconciliation is present (coefficient 0.29 vs 27.19).

### The document agent was blind because of the *generator*, not the agent

Worth recording, because the fix was one function away from where the
symptom appeared. The document agent used to score test ROC-AUC 0.5005 --
chance -- and the fusion meta-classifier gave it a **negative** coefficient
(-3.48), i.e. it was worse than useless; the `no_document` ablation beat the
full framework.

Rewriting the agent's scoring (grading the score instead of summing booleans,
replacing a `% 1000 == 0` round-amount test with a multi-scale digit check)
did not move the number at all, because the corpus contained no
single-document signal to find. `linked_pipeline/synth.py::_tier_for_ktosl`
assigned each line item's value tier from a *hash of its GL transaction type*,
so descriptions carried no information about amount: on 311K train genuine
rows, log10(DMBTR) class means spanned 5.616 to 5.644 against a 0.565
within-class std -- 0.05 sigma of separation, Spearman(tier, DMBTR) = -0.003.
"Office Supplies" for $3M was statistically indistinguishable from "Plant
Machinery" for $3M, so `amount_atypical_for_description` -- the agent's one
load-bearing feature -- measured a fraud/genuine lift of 1.17, and
`desc_amount_mismatch`, nominally the one typology visible from a single
document, was not actually an anomaly in the data. The reconciliation rule
that "caught" it (`tier == low and DMBTR > q90`) is the injector's literal
inverse, so its recall of 1.0 was measuring leakage, not detection.

`_amount_ranked_tiers` now assigns the tier from the row's amount tertile with
`TIER_NOISE = 0.22` of rows redrawn at random. The noise is the point: at 0.0
the tier would be an invertible function of the amount and any "anomaly" the
agent reported would just be that inverse. `DMBTR` is untouched, and no
typology injector reads descriptions, so the rebuild reproduced identical
splits, vendor assignments and typology counts -- only `description` and
`description_tier` moved.

| | before | after |
|---|---|---|
| `amount_atypical_for_description` lift | 1.17 | 5.47 |
| E2 document-only ROC-AUC | 0.500 | 0.594 (CI 0.580-0.608) |
| document coefficient in fusion | -3.48 | +2.96 |
| E4 `no_reconciliation` PR-AUC | 0.687 | 0.727 |
| E7 `threshold_split` recall | 0.070 | 0.204 |

Two honest caveats. Per-typology, the agent moved only on
`desc_amount_mismatch` (ROC-AUC 0.853) and, unfitted, `threshold_split`
(0.618); `altered_amount`, `duplicate_invoice` and `ghost_vendor` stay at
~0.50, which is correct -- they are ledger/cross-row disagreements and
invisible to a single document by construction. And the agent is now largely
*redundant* with reconciliation rather than additive: `no_document` still
edges the full framework on PR-AUC (0.792 vs 0.789, CIs heavily overlapping)
while losing on ROC-AUC (0.767 vs 0.770). The defensible claim is that the
document agent went from actively harmful to independently informative, not
that it lifted the ensemble.

**Not re-rendered:** the 20,000 invoice images predate this change, so the
description line printed on them is stale (every other field is unaffected).
`scripts/generate_invoices.py` is idempotent on row selection but reuses any
image file that already exists, so a refresh means deleting
`data/linked/images/{train,val,test}/` first.

**Gotcha found only at full scale:** `agents/transaction/features_erp.py`'s
categorical columns (`WAERS/BUKRS/KTOSL/PRCTR/BSCHL/HKONT`) used to infer
their category set independently per `.astype("category")` call. At 5,000
rows every split/fold happened to see every code; at 533K rows a rare code
landing only in one CV fold crashed xgboost at scoring time
(`XGBoostError: Found a category not in the training set`). Fixed by
`fit_category_levels()` pinning one canonical vocabulary across train+val+test,
threaded through every `build_xy()`/`TransactionAgent(...)` call — any new
caller of either must pass `category_levels` or risk hitting this again on a
large enough corpus.

To view the dashboard:

```powershell
# terminal 1
py -3.13 -m uvicorn dashboard.api:app --port 8811
# terminal 2
py -3.13 -m streamlit run dashboard\app.py
```

Then open `http://localhost:8501`. Sidebar filters by risk tier; click a
case in the left panel to see its invoice image, agent scores, its full
ledger row (BELNR included), fired reconciliation rules, and an LLM-generated
narrative (first view per case is slow — ~30-60s to generate, then cached).
A `duplicate_invoice` case additionally shows the entry it duplicates as an
aligned field-by-field diff plus both invoice images, so the reviewer can see
what matches (invoice no, vendor) against what doesn't (BELNR, posting date). Accept/Reject/Escalate buttons log
to an append-only SQLite audit log (`experiments/audit_log.sqlite3`).

### Live scoring: postings that were never in the corpus

The review queue shows the test split's *batch* scores, the ones the metrics
are computed from: `/case/{id}` and `/analyze` look them up in
`experiments/fusion_scores.parquet` and don't re-run anything. A new posting
is scored live instead:

- `run_fusion.py` saves everything it fits to `models/live_pipeline.joblib`:
  the fusion meta-classifier and its threshold, the full-train transaction
  agent, the vendor/account baselines, the document agent's per-description
  baselines, and the reconciliation agent's vendor master and limits.
- `agents/pipeline.py::LivePipeline` scores one posting plus its invoice
  fields with those objects. The transaction, document and reconciliation
  agents and the fusion model all run at request time. The duplicate and
  split-purchase rules and the vendor-velocity features check the posting
  against the full corpus ledger plus every earlier live submission.
- `POST /score` (API) and the dashboard's **Score a new entry** page call it.
  Each submission is stored in the `submissions` table of the audit-log
  database and replayed into the history on restart. Agent scores and the
  reviewer's decision are logged under `case_id = -submission_id`.

**Invoice fields are typed in, not OCR'd**, as an e-invoicing feed would
supply them. An uploaded invoice image is shown to the reviewer and never
read. `hour_of_day` comes from the posting's entered hour, defaulting to
10:00; the corpus has no posting time, so the batch run draws it at random.

To check it:

```powershell
py -3.13 scripts\run_fusion.py            # writes models\live_pipeline.joblib
py -3.13 scripts\smoke_live_scoring.py    # parity vs batch scores + 6 out-of-corpus scenarios
```

`smoke_live_scoring.py` re-scores 500 test postings one at a time and
compares them with their batch scores. It then runs six hand-built
postings, each of which must trip the expected rule: clean → no rules, low
tier; invoice 10% above ledger → `amount_mismatch`; unknown vendor →
`vendor_absent_from_master`; same invoice resubmitted → `duplicate_invoice_number`;
three pieces under the limit on one PO → `split_purchase` on the third;
low-tier description on a large amount → `description_tier_mismatch`.

Last run: document and reconciliation scores matched the batch run exactly on 500/500
postings. Transaction and fused scores matched exactly on 497/500; the
other 3 differ by at most 0.028 (transaction) and 0.0002 (fused). The risk
tier matched on 500/500, and all 6 scenarios pass. The median is ~0.4 s per
posting, after a ~3 s one-time load. When posting dates tie, a re-scored
corpus posting takes its original place in its split, as in the batch run;
a live posting goes after everything already on file.

**Known gaps, not fixed:**
- `threshold_split` is now caught by the reconciliation agent's
  `split_purchase` rule (2+ postings, same vendor + PO, within 7 days, each
  under the approval limit, summing over it): E7 recall 20.4% -> 98.4%, firing
  on 2 of 126,403 genuine test rows. It keys on the shared PO the injector
  writes, so it mirrors the typology's definition like the other rules do,
  and it is sensitive to the limit estimate -- see APPROVAL_LIMIT_QUANTILE in
  `agents/reconciliation/rules.py`
- Document agent's text-risk is keyword/rule-based, not the spec's
  fine-tuned sentence-transformer classifier. Its `vague_description`
  (lift 1.03) and `missing_po` (lift 0.88) features still carry no signal and
  are display-only; `low_extraction_confidence` cannot fire at all until real
  OCR replaces the ground-truth fast path
- No LIME baseline (XAI has SHAP only), no LayoutLMv3 fine-tuning on DocILE,
  no auditor survey, no demo video, no formal write-up

## Setup

Python **3.13**. Run every script as `py -3.13 <script>`, not `python`:

```powershell
py -3.13 scripts\run_fusion.py
```

**This matters more than it looks.** `python` on PATH resolves to a bare 3.14
install with none of the project's dependencies, so `run_fusion.py`,
`run_reconciliation_agent.py`, `generate_invoices.py` and `dashboard/api.py`
all die on `ModuleNotFoundError` for rapidfuzz / reportlab / fastapi /
pdf2image. That reads as broken project code rather than the wrong
interpreter, which is exactly how it wastes an afternoon. Confirm which one
you are on:

```powershell
py -0                                   # lists installed interpreters
py -3.13 -c "import rapidfuzz, fastapi, reportlab; print('ok')"
```

The dependencies are installed directly into the 3.13 interpreter here, not a
virtualenv — there is no `.venv` in this repo. For a clean install elsewhere:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel

# torch must come from the CPU index on a machine with no NVIDIA GPU,
# otherwise pip pulls a multi-gigabyte CUDA build that cannot be used.
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

Verify:

```powershell
py -3.13 verify_env.py
```

### Declared but not installed locally

`requirements.txt` is intent, and five of its entries are absent from the
working 3.13 environment. None are on the local pipeline path, so
`verify_env.py` reports them as warnings rather than failures:

| Package | Why it's absent |
|---|---|
| `paddlepaddle`, `paddleocr` | OCR runs on Kaggle (see "Document pipeline"); needed locally only by `experiments/smoke_document.py`, which is why that script does not run here |
| `lightgbm` | declared, but the transaction agent is xgboost + IsolationForest |
| `sqlalchemy` | declared, but the audit log uses stdlib `sqlite3` |
| `seaborn` | declared, not imported by any module |

### System dependencies (not pip-installable)

| Dependency | Needed by | Install |
|---|---|---|
| **poppler** | `pdf2image` | `winget install oschwartz10612.Poppler`, then add its `Library\bin` to `PATH` (already installed and on PATH here) |
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
py -3.13 experiments/bench_llm.py                  # tok/s per model
py -3.13 experiments/probe_llm_determinism.py      # is output reproducible?
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
