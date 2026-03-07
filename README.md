# HMITL: Manager-Governed LLM Iteration with Guardrails and Rollback for Reproducible Healthcare Machine Learning Pipelines

This repository accompanies the paper **“HMITL: Manager-Governed LLM Iteration with Guardrails and Rollback for Reproducible Healthcare Machine Learning Pipelines.”**

HMITL is a workflow protocol for human–LLM collaboration in multimodal healthcare machine learning. Instead of letting an LLM iterate autonomously on a pipeline, HMITL assigns the human the role of **workflow manager**: maintaining the task brief, enforcing data-integrity guardrails, evaluating every candidate under a deterministic harness, and rolling back regressions.

The experiments in this repository are built on **AgentDS-Healthcare**, a synthetic, privacy-free benchmark with three tasks spanning structured tables, short clinical text, PDF receipts, and time-series JSON.

---

## Reader Guide

If you only want the paper-relevant path through the repository:

1. Read the **Results at a Glance** section below.
2. Go directly to `agent_ds_healthcare/`.
3. Use the following canonical entry points:
   - **Challenge 1**: `Challenge1_Health_Final.ipynb`
   - **Challenge 2**: `Challenge2_baseline_ichi_v9.ipynb`
   - **Challenge 3 controlled study**: `Challenge3_ichi_v0.ipynb`, `Challenge3_ichi_v1.ipynb`, `Challenge3_ichi_v2.ipynb`, `Challenge3_ichi_v3.ipynb`
4. Treat `ch2_artifacts/` and `ch3_artifacts/` as **audit logs and reproducibility artifacts**. They are included for transparency, not because we expect readers to inspect every file line by line.

A number of files in this release are intentionally preserved in their working form (for example, raw prompt logs, iterative notes, or untranslated prompt fragments). These files are included to make the development process auditable. They are not required for understanding the main paper claims.

---

## Results at a Glance

| Benchmark task | Metric | Paper-reported result | Main entry point |
|---|---:|---:|---|
| Challenge 1: 30-day readmission prediction | Macro-F1 ↑ | **0.9014** | `agent_ds_healthcare/Challenge1_Health_Final.ipynb` |
| Challenge 2: ED cost forecasting | MAE ↓ | **447.9542** | `agent_ds_healthcare/Challenge2_baseline_ichi_v9.ipynb` |
| Challenge 3: controlled study, V2 (multi-window arbitration + rollback) | Macro-F1 ↑ | **0.8408** | `agent_ds_healthcare/Challenge3_ichi_v2.ipynb` |

### Controlled-study variants for Challenge 3

| Variant | Description | Notebook | Trace artifact |
|---|---|---|---|
| V0 | Single-window continuation without explicit rollback | `Challenge3_ichi_v0.ipynb` | `ch3_artifacts/clai_ch3_v0_iteration_detail.jsonl` |
| V1 | Single-window + rollback | `Challenge3_ichi_v1.ipynb` | `ch3_artifacts/clai_ch3_v1_iteration_detail.jsonl` |
| V2 | Multi-window arbitration + rollback | `Challenge3_ichi_v2.ipynb` | `ch3_artifacts/clai_ch3_v2_iteration_detail.jsonl` |
| V3 | V2 + consultant second opinion | `Challenge3_ichi_v3.ipynb` | `ch3_artifacts/clai_ch3_v3_iteration_detail.jsonl` |

---

## What Is in This Repository

### Paper-relevant content
- `agent_ds_healthcare/`
  - notebooks for all three healthcare challenges
  - Challenge 2 iteration history (`v2`–`v9`)
  - Challenge 3 controlled-study variants (`v0`–`v3`)
  - raw and summarized reproducibility artifacts
- `agent_ds_healthcare/ch2_artifacts/`
  - Challenge 2 human-manager records, prompt traces, iteration summaries, ablations, and optimization logs
- `agent_ds_healthcare/ch3_artifacts/`
  - machine-readable JSONL traces for the fixed-budget controlled study reported in the paper

### Additional non-core content
- `agent_ds_commerce/`
  - retained from broader AgentDS participation
  - **not required** to reproduce the healthcare paper results

---

## Repository Layout

```text
.
├── agent_ds_healthcare/
│   ├── Challenge1_Health_Final.ipynb
│   ├── Challenge2_baseline_ichi_v2.ipynb
│   ├── Challenge2_baseline_ichi_v3.ipynb
│   ├── Challenge2_baseline_ichi_v4.ipynb
│   ├── Challenge2_baseline_ichi_v5.ipynb
│   ├── Challenge2_baseline_ichi_v6.ipynb
│   ├── Challenge2_baseline_ichi_v7.ipynb
│   ├── Challenge2_baseline_ichi_v8.ipynb
│   ├── Challenge2_baseline_ichi_v9.ipynb
│   ├── Challenge3_ichi_v0.ipynb
│   ├── Challenge3_ichi_v1.ipynb
│   ├── Challenge3_ichi_v2.ipynb
│   ├── Challenge3_ichi_v3.ipynb
│   ├── ch2_artifacts/
│   └── ch3_artifacts/
├── agent_ds_commerce/                 # auxiliary; not needed for the healthcare paper
├── requirements.txt
├── environment.yml
└── README.md
```

---

## Dataset and Benchmark

The experiments use **AgentDS-Healthcare**, which provides three synthetic healthcare tasks:

1. **30-day readmission prediction** from admissions tables + a short discharge note.
2. **ED cost forecasting** from structured utilization history + one PDF billing receipt per patient.
3. **Discharge readiness prediction** from 10 days of vital-sign time series + short daily notes.

The upstream dataset repository groups all healthcare files under a single `Healthcare/` directory. The relevant files include:

```text
Healthcare/
├── patients.csv
├── admissions_train.csv
├── admissions_test.csv
├── discharge_notes.json
├── ed_cost_train.csv
├── ed_cost_test.csv
├── receipts_pdf/
│   └── receipt_<patient_id>.pdf
├── stays_train.csv
├── stays_test.csv
└── vitals_timeseries.json
```

---

## Environment Setup

We recommend using either the provided Conda environment or a clean virtual environment.

### Option A: Conda

```bash
conda env create -f environment.yml
conda activate hmitl
```

### Option B: venv

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Core dependencies

The notebooks rely on standard scientific Python tooling plus gradient-boosting and text-feature libraries. The main dependencies are:

- `agentds-bench`
- `pandas`, `numpy`, `scipy`, `scikit-learn`, `joblib`
- `lightgbm`, `catboost`, `xgboost`
- `sentence-transformers`, `torch`
- `jupyter` / `notebook`
- `huggingface_hub`

---

## AgentDS Credentials

To submit predictions to the official AgentDS benchmark, you need an **API key** and your **team name**.

Set them as environment variables before running the notebooks:

```bash
export AGENTDS_API_KEY=<your_agentds_api_key>
export AGENTDS_TEAM_NAME=<your_team_name>
```

On Windows PowerShell:

```powershell
$env:AGENTDS_API_KEY = "<your_agentds_api_key>"
$env:AGENTDS_TEAM_NAME = "<your_team_name>"
```

The notebooks are written so that local experimentation is possible without submission, but benchmark submission requires valid credentials.

---

## Downloading AgentDS-Healthcare

Because AgentDS-Healthcare mixes multiple challenge-specific tables and also includes a `receipts_pdf/` directory, we recommend downloading the **dataset repository snapshot** rather than relying on a single `datasets.load_dataset(...)` call.

### Recommended: Hugging Face CLI snapshot download

```bash
pip install -U huggingface_hub hf_xet
hf download lainmn/AgentDS-Healthcare \
  --repo-type dataset \
  --local-dir data/AgentDS-Healthcare
```

### Alternative: Git / Git Bash clone

This option is convenient on Windows, especially when working with the PDF folder directly.

```bash
git lfs install
git xet install
git clone https://huggingface.co/datasets/lainmn/AgentDS-Healthcare data/AgentDS-Healthcare
```

After download, point the repository to the healthcare folder:

```bash
export CLAI_BASE_DIR=$PWD/data/AgentDS-Healthcare/Healthcare
```

On Windows PowerShell:

```powershell
$env:CLAI_BASE_DIR = "$PWD/data/AgentDS-Healthcare/Healthcare"
```

If your local layout differs, update the path once and keep the folder contents unchanged.

---

## Challenge 2 Receipt Preparation

Challenge 2 uses one synthetic billing PDF per patient. For reproducibility and speed, the modeling notebook expects a cached parsed artifact:

```text
$CLAI_BASE_DIR/receipts_parsed.joblib
```

Generate it once with the receipt-preparation script:

```bash
python scripts/prepare_ch2_receipts.py \
  --data-dir "$CLAI_BASE_DIR" \
  --pdf-dir "$CLAI_BASE_DIR/receipts_pdf" \
  --output "$CLAI_BASE_DIR/receipts_parsed.joblib"
```

This preparation step performs PDF extraction, receipt-level sanity checks, and caching so that later modeling iterations operate on a fixed parsed representation.

If you are using the paper release cache instead of regenerating it, place the cached file directly at:

```text
$CLAI_BASE_DIR/receipts_parsed.joblib
```

---

## How to Run the Main Experiments

## 1) Challenge 1 — 30-day readmission

Canonical entry point:

```text
agent_ds_healthcare/Challenge1_Health_Final.ipynb
```

This notebook uses:
- `admissions_train.csv`
- `admissions_test.csv`
- `patients.csv`
- `discharge_notes.json`

Expected submission format:

```text
admission_id,readmit_30d
```

## 2) Challenge 2 — ED cost forecasting

Canonical entry point:

```text
agent_ds_healthcare/Challenge2_baseline_ichi_v9.ipynb
```

Required inputs:
- `ed_cost_train.csv`
- `ed_cost_test.csv`
- `patients.csv`
- `admissions_train.csv`
- `admissions_test.csv`
- `receipts_parsed.joblib`

Expected submission format:

```text
patient_id,ed_cost_next3y_usd
```

## 3) Challenge 3 — controlled study

Canonical entry points:

```text
agent_ds_healthcare/Challenge3_ichi_v0.ipynb
agent_ds_healthcare/Challenge3_ichi_v1.ipynb
agent_ds_healthcare/Challenge3_ichi_v2.ipynb
agent_ds_healthcare/Challenge3_ichi_v3.ipynb
```

Required inputs:
- `stays_train.csv`
- `stays_test.csv`
- `patients.csv`
- `vitals_timeseries.json`

Expected submission format:

```text
stay_id,discharge_ready_day11
```

### Practical note on the notebooks

These notebooks preserve real iteration history rather than only a single “clean-room final script.” That is deliberate. They are intended to document how the HMITL process evolved across trials, ablations, and branches.

For readers who want the shortest path:
- use the canonical entry-point notebooks listed above;
- jump to the final sections labeled **Submission** / **Submit Predictions** for the latest paper-relevant branch;
- consult the artifact folders only if you want the full audit trail.

---

## Reproducibility Artifacts

### `ch2_artifacts/`
Challenge 2 artifacts include:
- manager-governed iteration logs
- prompt + code delta records
- ablation records
- optimization traces
- raw working notes preserved for auditability

### `ch3_artifacts/`
Challenge 3 artifacts include:
- per-variant JSONL traces
- fixed-budget iteration records
- variant-level evidence for rollback, arbitration, and consultant effects

These files are included so that the **decision process** is inspectable, not just the final scores.

---

## Important Notes for Readers

1. **Paper-relevant content is under `agent_ds_healthcare/`.**
   The root repository also includes auxiliary material from broader AgentDS participation.

2. **Some artifacts are raw working records.**
   A subset of logs contain concise working notes or prompt fragments. They are kept for transparency and are not required for basic reproduction.

3. **The notebooks preserve multiple experimental branches.**
   This repository intentionally keeps more than a single final notebook so that reviewers can see the evolution of the pipeline and the role of rollback-aware selection.

4. **Challenge 2 depends on cached receipt parsing.**
   If `receipts_parsed.joblib` is absent, prepare it first from `receipts_pdf/`.

5. **Do not commit benchmark credentials.**
   Use environment variables for API keys and team identifiers.

---

## Citation

If you use this repository, please cite the accompanying paper:

```bibtex
@misc{chen2026hmitl,
  title={HMITL: Manager-Governed LLM Iteration with Guardrails and Rollback for Reproducible Healthcare Machine Learning Pipelines},
  author={Jialuo Chen and Haijing Wang and Siyu Shao},
  year={2026},
  note={Paper accompanying this repository}
}
```

Please also cite the upstream AgentDS benchmark and respect the dataset’s license and usage terms.

---

## Acknowledgements

We thank the AgentDS Benchmark team for releasing AgentDS-Healthcare and the challenge infrastructure that made this study possible.

---

## Upstream References

- AgentDS benchmark: <https://huggingface.co/datasets/lainmn/AgentDS>
- AgentDS-Healthcare dataset: <https://huggingface.co/datasets/lainmn/AgentDS-Healthcare>
- AgentDS website: <https://agentds.org>

