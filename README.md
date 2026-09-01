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

Configured in `config.py` via `OLLAMA_MODEL`, overridable per machine:

```powershell
$env:OLLAMA_MODEL = "qwen2.5:3b-instruct"
```

Benchmark before designing around a model size:

```powershell
python experiments/bench_llm.py
```

## Data policy

`data/` and `models/` are gitignored. **Never commit datasets.** The directory
skeleton is preserved with `.gitkeep` files; the ignore pattern
(`data/**` + `!data/**/` + `!data/**/.gitkeep`) keeps the structure while
excluding every file inside it.
