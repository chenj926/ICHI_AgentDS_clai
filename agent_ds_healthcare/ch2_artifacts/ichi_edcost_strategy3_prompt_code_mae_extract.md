# AgentDS Challenge 2 — ED Cost Forecasting
This Markdown is an *objective extraction* of **prompts**, **code cells**, and **MAE / submission logs** from the PDF:
- Source PDF: `IEEE ICHI - ED Cost Forecasting Strategy-3.pdf`
- Extraction time (UTC): 2026-02-20 04:38:19
- Notes:
  - Page numbers below refer to the PDF's 1-indexed pages.
  - Content is kept as verbatim as possible; formatting artifacts may exist due to PDF text extraction.
  - **No tables** are used (bullet lists and code/text blocks only).

---

## A) Prompt Blocks (verbatim)

### P0_master_prompt_system_role (pages 1-4)
```text
System Role & Persona:
You are a domain expert in Healthcare Economics and a 
Senior Scientist in Machine Learning/AI. You specialize in 
"ED Cost Forecasting" and predictive modeling for hospital 
readmissions. You possess deep knowledge of medical 
coding (CPT/HCPCS), chronic disease trajectories (Heart 
Failure, Diabetes, Pneumonia), and the socio-economic 
determinants of health.
The Challenge:
Our team is competing in the AgentDS Challenge 2: ED 
Cost Forecasting.
Goal: Forecast the total emergency department (ED) costs 
for patients over the next 3 years based on their historical 
ED utilization, chronic conditions, and prior costs.
Metric: Mean Absolute Error (MAE).
Current Status: We previously struggled with high MAE 
because our PDF data was incomplete. We have now fixed 
the pipeline so that every patient in the training set has a 
corresponding, complete Receipt PDF.
The Data Landscape:
We are working with a multimodal dataset consisting of 
CSVs and synthetic billing PDFs.
description.md
File
receipt_1.pdf
PDF
receipt_5.pdf
PDF
receipt_10.pdf
PDF

Tabular Data (ed_cost_*.csv):
patient_id: Join key.
primary_chronic: {HF (Heart Failure), DiabetesComp, 
Pneumonia}.
prior_ed_visits_5y: Integer count of visits in the prior 5 
years.
prior_ed_cost_5y_usd: Total cost (USD) in the prior 5 years.
ed_cost_next3y_usd (Target): The regression target for the 
next 3 years.
Note: There are other columns (like ZIP3) that need 
analysis.
Unstructured Data (receipt_<patient_id>.pdf):
These are synthetic "billing summaries."
Content: Line items containing CPT/HCPCS-like codes, 
quantities, and line totals.
Validation: The final "TOTAL" in the PDF matches the 
prior_ed_cost_5y_usd in the CSV.
Key Constraints & Environment:
Hardware: Local Windows environment with a single 
NVIDIA RTX 4060 (CUDA 12). The solution must be 
efficient enough to run here (e.g., careful selection of Local 
LLMs, efficient OCR/parsing, or feature extraction).
Search Constraint: You should search for relevant papers 
and challenges (e.g., similar Kaggle competitions, papers 
on ED cost regression) to inform your strategy, but do not 
over-search. We need practical, applicable strategies, not 
just a literature review.

Your Task:
Using your domain knowledge and the provided files (I will 
attach 3 sample PDFs, the description.md, and the CSV 
samples), please develop a comprehensive Strategy and 
Implementation Plan to win this challenge.
You must address the following specific domain inquiries 
in your plan:
The "Niche" Nature: This is not a generic regression 
problem. It is specific to Healthcare ED costs. How does 
the trajectory of specific conditions (HF vs. Diabetes) 
generally influence future cost spikes?
Data & Feature Engineering:
ZIP3 Analysis: What does the ZIP3 column represent in this 
context? Could it imply insurance type (Public vs. Private) 
or socioeconomic status? How should we encode this?
PDF Granularity: The PDF matches the prior cost, but the 
future cost is the target. How do we extract value from the 
specific line items (CPT codes) in the PDFs? Does your 
domain knowledge suggest that certain "cheap" codes in 
the past might indicate high-risk/high-cost recurrence in 
the future?
Technical Implementation (Local LLM vs. Code): given the 
RTX 4060 constraint, how should we process these PDFs? 
Should we use a quantized Local LLM (e.g., Llama 3, 
Mistral) to extract semantic meaning, or traditional 
NLP/Regex?
Correction of Past Errors: Since we fixed the missing PDF 
issue, how do we best align the tabular features with the 
extracted PDF features to lower MAE?
Immediate Next Step:
Please SEARCH for similar problems/papers first, then 

THINK deeply about the medical/economic implications of 
the data columns and PDF contents. Finally, output a 
detailed Step-by-Step Master Plan.
(I have attached the dataset samples and description 
below)
```

### P1_iterative_workflow_phase1_request (pages 15-17)
```text
Sources
Sources
System Role & Objective:
You are the Lead Healthcare AI Scientist and Domain 
Expert. We are competing in the AgentDS Healthcare 
Challenge 2: ED Cost Forecasting.
The Goal:
Forecast ed_cost_next3y_usd (Total ED costs for the next 3 
years) based on patient history, chronic conditions, and 
synthetic billing PDF receipts.
Operational Constraint - THE ITERATIVE WORKFLOW:
DO NOT generate the final training pipeline or model 
architecture yet.
Instead, we will adopt a strict "Research & Validation" 
workflow.
Phase 1 (Current): You will formulate hypotheses based on 
Domain Knowledge + Online Search. Then, you will 
independently design and write an Exploratory Data 
Analysis (EDA) script for me to run locally.
Phase 2 (User Feedback): I will run your code and provide 
the logs/outputs/charts back to you.
Phase 3 (Refinement): Based on the data evidence, you will 
refine the feature engineering strategy and then we will 
move to modeling.
Your Environment & Resources:
Hardware: Local Windows PC, NVIDIA RTX 4060 (CUDA 12). 
Efficiency is key.
Directory Structure (Python pathlib format):
Python

from pathlib import Path
BASE_DIR = Path(r"D:\AgentDs\agent_ds_healthcare")
TRAIN_CSV = BASE_DIR / "ed_cost_train.csv"
TEST_CSV  = BASE_DIR / "ed_cost_test.csv"
PATIENTS_CSV = BASE_DIR / "patients.csv"
PDF_DIR = BASE_DIR / "receipts_pdf"
SUBMISSION_PATH = BASE_DIR / 
"submission_ICHI_V1.csv"
# Cache for parsed PDF data to save time on re-runs
CACHE_DIR = BASE_DIR / "cache_iter10"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RECEIPT_CACHE = CACHE_DIR / "receipts_parsed.joblib"
Context & Domain Specifics (The "Niche"):
Problem: Regression of Future ED Costs.
Data:
ed_cost_*.csv: Tabular patient history and demographics.
PDF Receipts: Synthetic billing summaries where the total 
matches the prior 5-year cost.
The Core Challenge: We need to extract predictive signals 
from these PDFs and CSVs that standard models might 
miss. The relationships between past billing codes 
(CPT/HCPCS) and future risk are complex and non-linear.
Your Immediate Task:
Search & Hypothesize: Briefly search for "ED cost 
prediction features" or similar healthcare regression 
challenges to ground your thinking.
Design "Phase 1: Discovery & Validation":
I am not telling you what to check. As the Senior Scientist, 
YOU must decide what data quality checks, statistical 

distributions, and feature relationships are most critical to 
investigate before we touch any models.
Write a Python script that performs these specific analyses.
Your script should help us confirm the integrity of our data 
(especially the new PDFs) and reveal the underlying 
structure of the problem (e.g., cost distribution, feature 
relevance).
Output format:
Domain Analysis: Your high-level thoughts and 
hypotheses.
The Code: A clean, executable Python block that I can run 
immediately to generate the insights you need.
(I have attached the sample PDFs and CSV snippets for 
your reference in previous prompt)
```

### P2_bugfix_request_jupyter_one_cell (page 41)
```text
RECEIPT_CACHE=D:\AgentDs\agent_ds_healthcare\cache_i
ter10\receipts_parsed.joblib
=== BASIC INTEGRITY REPORT ===
{
  "n_train": 2000,
  "n_test": 2000,
  "train_dupe_patient_id": 0,
  "test_dupe_patient_id": 0,
  "train_missing_values": {
    "patient_id": 0,
    "primary_chronic": 0,
    "prior_ed_visits_5y": 0,
    "prior_ed_cost_5y_usd": 0,
    "ed_cost_next3y_usd": 0
  },
  "test_missing_values": {
    "patient_id": 0,
    "primary_chronic": 0,
    "prior_ed_visits_5y": 0,
    "prior_ed_cost_5y_usd": 0
  },
  "patients_loaded": true,
  "n_patients": 4000,
  "patients_dupe_patient_id": 0,
  "train_missing_in_patients_n": 0,
  "test_missing_in_patients_n": 0
}
Above is partial log.
Please fix the bug, and from now on, I am using jupyter 
notebook, so you should always give me the code I can 
copy and paste into 1 cell to run
Thought for 21m 42s
```

### P3_request_move_to_final_model (page 151)
```text
78                78
      V2_base_l1_ES regression_l1   False 452.955572 
16.870699    384.289108    77.677775         1317.6               
78                78
  V2_TE_quantile_ES      quantile    True 457.901519 
18.301663    389.915875    76.472322          695.8               
78                81
        V2_TE_l1_ES regression_l1    True 457.901519 
18.301663    389.915875    76.472322          695.8               
78                81
After many validation and baseline, should we now step 
into an actual, robust final training model that prepares for 
submission, I think we also need real MAE data to 
validatoin if our validataion is correct. Also my 
enviornment is pretty solid under this data size so less 
worrying about my local environment but focus on the 
quality of the final model.
Thought for 23m 57s
also, I noticed you are thinking about the same output for 
v2 TE l1 and quantile es, here is the full log for you:
[info] unique receipt codes (train): 18 -> ['31500', '36556',
```

### P4_request_improve_from_real_MAE_461 (page 216)
```text
OOF MAE=450.979 | Outlier OOF MAE=374.186
[calibration] NOT applied (did not improve OOF MAE).
[DONE] Wrote submission: 
D:\AgentDs\agent_ds_healthcare\submission_ICHI_V5.csv
 patient_id  ed_cost_next3y_usd
       3870         2469.876603
       3622         4402.300415
       1967         2001.719638
       1479         4987.029736
        362         2636.227230
       2578         4472.732691
       2451         2642.235981
       2358         1860.408412
       3555         3372.285811
       1581         3102.621342
now we have the real data to validation our solution. It is a 
good start, but we have a benchmark at 400 MAE, there is 
still a big gap in between us and the benchmark. Let's 
think about how to improve
Thought for 24m 26s
here are the additional files in the healthcare domain, I see 
in the thinking that you are considering cross check those 
informations, here you go.
discharge_notes.json
File
vitals_timeseries.json
```

### P5_request_restart_eda_cross_task (page 259)
```text
out_n=85
avg_best_iter=1787.2
[DONE] Wrote submission: 
D:\AgentDs\agent_ds_healthcare\submission_EDCost_cross
V1.csv
since we want to add in new data cross check, let's go 
back to few iterations of eda first, and refer to the eda to 
modify our model choice. this is the time we should start a 
new iteration, what do you think
Thought for 20m 21s
Yes — this is exactly the right moment to start a fresh EDA iteration before we commit to a
new “final” modeling direction.
Why the cross-task features likely hurt (hypotheses to validate)
```

### P6_request_next_step_after_score_460.9853 (page 537)
```text
D:\AgentDs\agent_ds_healthcare\cache_iter26_robust\sub
mission_iter26.csv
🚀 Submitting predictions...
✅ Prediction submitted successfully!
📊 Score: 460.9853 (MAE)
✅ Validation passed
✅ Submission successful!
   📊 Score: 460.9853
   📏 Metric: MAE
   ✔️  Validation: Passed
what should we do next
Thought for 22m 30s
你们现在最该做的下一步，不是继续“加特征/加模型堆参数”去追 OOF 445 → 430，而是先
解决一个更致命的问题：
你们的本地 CV（OOF=445）和线上得分（LB≈461）存在稳定且不小的 gap（+15~20
MAE）。
```

---

## B) Code Cells (verbatim)

### C01 — Code cell starting at page 19
```python
Phase 1: Discovery & Validation EDA for AgentDS Healthcare Challenge 2 (ED Cost F
What this script does (no modeling):
1) Loads ed_cost_train/test.csv and patients.csv
2) Validates joins, uniqueness, missingness, basic distributions
3) Parses receipt_<patient_id>.pdf with deterministic text parsing (PyMuPDF),
4) Runs integrity checks:
   - receipt TOTAL == prior_ed_cost_5y_usd
   - sum(line totals) == TOTAL
   - count(ED E/M lines 99281-99285) vs prior_ed_visits_5y
5) Explores structure:


# --- PAGE 20 ---

   - CPT/HCPCS code frequencies and "uplift" associations with the target (univar
   - receipt-derived composition features (cost shares, severity mix, complexity)
Dependencies:
- PyMuPDF (pymupdf) recommended for PDF text extraction:
from __future__ import annotations
from pathlib import Path
import re
import json
import math
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from joblib import dump, load
from tqdm import tqdm
import matplotlib.pyplot as plt
# -----------------------------
# USER PATHS (as provided)
# -----------------------------
BASE_DIR = Path(r"D:\AgentDs\agent_ds_healthcare")
TRAIN_CSV = BASE_DIR / "ed_cost_train.csv"
TEST_CSV  = BASE_DIR / "ed_cost_test.csv"
PATIENTS_CSV = BASE_DIR / "patients.csv"
PDF_DIR = BASE_DIR / "receipts_pdf"
SUBMISSION_PATH = BASE_DIR / "submission_ICHI_V1.csv"
# Cache for parsed PDF data to save time on re-runs


# --- PAGE 21 ---

CACHE_DIR = BASE_DIR / "cache_iter10"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RECEIPT_CACHE = CACHE_DIR / "receipts_parsed.joblib"
EDA_OUT_DIR = CACHE_DIR / "eda_outputs"
EDA_OUT_DIR.mkdir(parents=True, exist_ok=True)
# -----------------------------
# CONFIG KNOBS
# -----------------------------
FORCE_REPARSE = False
LIMIT_PDFS: Optional[int] = None  # e.g. 200 for quick smoke test; None = parse a
MIN_PATIENTS_WITH_CODE = 30       # for code-level uplift analysis (train only)
# plotting
plt.rcParams["figure.dpi"] = 120
# -----------------------------
# Helpers
# -----------------------------
def standardize_zip3(z) -> Optional[str]:
    if z is None or (isinstance(z, float) and np.isnan(z)):
        return None
    s = str(z).strip()
    s = re.sub(r"\D", "", s)  # keep digits only
    if not s:
        return None
    return s.zfill(3)
def parse_money(x: str) -> float:
    x = str(x).strip()
    x = re.sub(r"[^\d\.\-]", "", x)  # remove commas, $, etc
    return float(x) if x else np.nan


# --- PAGE 22 ---

def is_code_token(s: str) -> bool:
    """CPT/HCPCS-ish token: up to 2 letters + 4-5 digits (e.g., 99283, G0378)."""
    s = str(s).strip().upper()
    return bool(re.match(r"^[A-Z]{0,2}\d{4,5}$", s))
def categorize_code(code: str, desc: str = "") -> str:
    ED_EM: 99281-99285
    IMAGING: 70000-79999 (numeric starting with '7')
    LAB:     80000-89999 (numeric starting with '8')
    OBS:     G0378 or description contains 'observation'
    OTHER:   everything else
    code = str(code).strip().upper()
    desc_l = str(desc or "").lower()
    if re.match(r"^9928[1-5]$", code):
        return "ED_EM"
    if code == "G0378" or "observation" in desc_l:
        return "OBS"
    if code.isdigit() and len(code) == 5 and code.startswith("7"):
        return "IMAGING"
    if code.isdigit() and len(code) == 5 and code.startswith("8"):
        return "LAB"
    return "OTHER"
def extract_text_pymupdf(pdf_path: Path) -> str:
    """Extract text using PyMuPDF (fitz)."""
    try:
        import fitz  # PyMuPDF
    except Exception as e:
        raise RuntimeError(
            "PyMuPDF not available. Install with: pip install pymupdf\n"
            f"Original import error: {e}"
        )
    doc = fitz.open(str(pdf_path))


# --- PAGE 23 ---

    text = "\n".join(page.get_text() for page in doc)
    return text
def parse_receipt_text(text: str) -> Tuple[Dict, List[Dict]]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    header: Dict = {}
    m = re.search(r"Patient ID:\s*(\d+)", text)
    if m:
        header["patient_id"] = int(m.group(1))
    m = re.search(r"ZIP3:\s*([0-9]{1,3})", text)
    if m:
        header["zip3_receipt_raw"] = m.group(1)
    m = re.search(r"Insurance:\s*([A-Za-z_]+)", text)
    if m:
        header["insurance_receipt"] = m.group(1).lower()
    # TOTAL
    total = np.nan
    for i, ln in enumerate(lines):
        if ln.startswith("TOTAL"):
            m = re.search(r"TOTAL\s*([\d\.,\-]+)", ln)
            if m:
                total = parse_money(m.group(1))
            else:
                if i + 1 < len(lines):
                    total = parse_money(lines[i + 1])
    header["receipt_total"] = total
    # Table start: after "Line Total" header
    start = None


# --- PAGE 24 ---

    for i, ln in enumerate(lines):
        if "Line Total" in ln:
            start = i + 1
    if start is None:
        # fallback: first code
        for i, ln in enumerate(lines):
            if is_code_token(ln):
                start = i
    if start is None:
        return header, []
    line_items: List[Dict] = []
    i = start
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("TOTAL"):
        if not is_code_token(ln):
            i += 1
        code = ln.strip().upper()
        # description may be multi-line until qty line
        j = i + 1
        desc_parts: List[str] = []
        while j < len(lines) and not re.match(r"^\d+$", lines[j]) and not lines[j
            desc_parts.append(lines[j])
            j += 1
        if j >= len(lines) or lines[j].startswith("TOTAL"):
        desc = " ".join(desc_parts).strip()
        qty = int(lines[j])
        if j + 1 >= len(lines):


# --- PAGE 25 ---

        unit_price = parse_money(lines[j + 1])
        if j + 2 >= len(lines):
        line_total = parse_money(lines[j + 2])
        line_items.append(
            {
                "code": code,
                "description": desc,
                "qty": qty,
                "unit_price": unit_price,
                "line_total": line_total,
            }
        )
        i = j + 3
    return header, line_items
def parse_receipt_pdf(pdf_path: Path) -> Tuple[Dict, pd.DataFrame]:
    text = extract_text_pymupdf(pdf_path)
    header, items = parse_receipt_text(text)
    header["pdf_path"] = str(pdf_path)
    header["n_line_items"] = len(items)
    if items:
        li = pd.DataFrame(items)
        header["sum_line_total"] = float(np.nansum(li["line_total"].values))
        header["sum_unit_x_qty"] = float(np.nansum((li["unit_price"] * li["qty"])
    else:
        li = pd.DataFrame(columns=["code", "description", "qty", "unit_price", "l
        header["sum_line_total"] = 0.0
        header["sum_unit_x_qty"] = 0.0
    return header, li
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:


# --- PAGE 26 ---

    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    patients = None
    if PATIENTS_CSV.exists():
        patients = pd.read_csv(PATIENTS_CSV)
    # Standardize dtypes
    train["patient_id"] = train["patient_id"].astype(int)
    test["patient_id"] = test["patient_id"].astype(int)
    if patients is not None:
        patients["patient_id"] = patients["patient_id"].astype(int)
        # standardize zip3
        if "zip3" in patients.columns:
            patients["zip3"] = patients["zip3"].apply(standardize_zip3)
        if "insurance" in patients.columns:
            patients["insurance"] = patients["insurance"].astype(str).str.lower()
    return train, test, patients
def basic_integrity_checks(train: pd.DataFrame, test: pd.DataFrame, patients: Opt
    report = {}
    report["n_train"] = int(train.shape[0])
    report["n_test"] = int(test.shape[0])
    report["train_dupe_patient_id"] = int(train["patient_id"].duplicated().sum())
    report["test_dupe_patient_id"] = int(test["patient_id"].duplicated().sum())
    report["train_missing_values"] = train.isna().sum().to_dict()
    report["test_missing_values"] = test.isna().sum().to_dict()
    if patients is None:
        report["patients_loaded"] = False
    else:
        report["patients_loaded"] = True
        report["n_patients"] = int(patients.shape[0])
        report["patients_dupe_patient_id"] = int(patients["patient_id"].duplicate


# --- PAGE 27 ---

        train_missing_in_patients = sorted(set(train["patient_id"]) - set(patient
        test_missing_in_patients = sorted(set(test["patient_id"]) - set(patients[
        report["train_missing_in_patients_n"] = len(train_missing_in_patients)
        report["test_missing_in_patients_n"] = len(test_missing_in_patients)
    return report
def parse_all_receipts(patient_ids: List[int]) -> Tuple[pd.DataFrame, pd.DataFram
    Returns:
      headers_df: one row per patient_id
      lineitems_df: multiple rows per patient_id
    if RECEIPT_CACHE.exists() and not FORCE_REPARSE:
        cached = load(RECEIPT_CACHE)
        headers_df = cached["headers_df"]
        lineitems_df = cached["lineitems_df"]
        cached_ids = set(headers_df["patient_id"].astype(int).tolist())
        needed = set(patient_ids)
        missing = sorted(list(needed - cached_ids))
        if not missing:
            print(f"[cache] Loaded parsed receipts from {RECEIPT_CACHE} (n={len(c
            return headers_df, lineitems_df
        print(f"[cache] Found {len(missing)} missing receipts not in cache. Parsi
        to_parse = missing
        headers_new, items_new = _parse_receipts_no_cache(to_parse)
        headers_df = pd.concat([headers_df, headers_new], ignore_index=True)
        lineitems_df = pd.concat([lineitems_df, items_new], ignore_index=True)
        dump({"headers_df": headers_df, "lineitems_df": lineitems_df}, RECEIPT_CA
        return headers_df, lineitems_df
    # no cache or forced
    headers_df, lineitems_df = _parse_receipts_no_cache(patient_ids)


# --- PAGE 28 ---

    dump({"headers_df": headers_df, "lineitems_df": lineitems_df}, RECEIPT_CACHE)
    print(f"[cache] Wrote parsed receipts cache to {RECEIPT_CACHE}")
    return headers_df, lineitems_df
def _parse_receipts_no_cache(patient_ids: List[int]) -> Tuple[pd.DataFrame, pd.Da
    headers: List[Dict] = []
    all_items: List[pd.DataFrame] = []
    ids = patient_ids
    if LIMIT_PDFS is not None:
        ids = ids[:LIMIT_PDFS]
        print(f"[info] LIMIT_PDFS set: parsing only first {len(ids)} receipts (of
    missing_files = 0
    for pid in tqdm(ids, desc="Parsing receipts"):
        pdf_path = PDF_DIR / f"receipt_{pid}.pdf"
        if not pdf_path.exists():
            missing_files += 1
            headers.append(
                {
                    "patient_id": pid,
                    "pdf_path": str(pdf_path),
                    "parse_ok": False,
                    "parse_error": "missing_file",
                    "n_line_items": 0,
                    "receipt_total": np.nan,
                    "sum_line_total": np.nan,
                    "sum_unit_x_qty": np.nan,
                    "zip3_receipt_raw": None,
                    "insurance_receipt": None,
                }
            )
        try:
            h, li = parse_receipt_pdf(pdf_path)
            h["parse_ok"] = True
            h["parse_error"] = None


# --- PAGE 29 ---

            li = li.copy()
            li["patient_id"] = int(pid)
            all_items.append(li)
            headers.append(h)
        except Exception as e:
            headers.append(
                {
                    "patient_id": pid,
                    "pdf_path": str(pdf_path),
                    "parse_ok": False,
                    "parse_error": str(e)[:2000],
                    "n_line_items": 0,
                    "receipt_total": np.nan,
                    "sum_line_total": np.nan,
                    "sum_unit_x_qty": np.nan,
                    "zip3_receipt_raw": None,
                    "insurance_receipt": None,
                }
            )
    headers_df = pd.DataFrame(headers)
    lineitems_df = pd.concat(all_items, ignore_index=True) if all_items else pd.D
        columns=["patient_id", "code", "description", "qty", "unit_price", "line_
    )
    if missing_files:
        print(f"[warn] Missing receipt PDFs: {missing_files}")
    # standardize
    headers_df["patient_id"] = headers_df["patient_id"].astype(int)
    if "zip3_receipt_raw" in headers_df.columns:
        headers_df["zip3_receipt"] = headers_df["zip3_receipt_raw"].apply(standar
    if "insurance_receipt" in headers_df.columns:
        headers_df["insurance_receipt"] = headers_df["insurance_receipt"].astype(
    if not lineitems_df.empty:
        lineitems_df["code"] = lineitems_df["code"].astype(str).str.upper().str.st
        lineitems_df["description"] = lineitems_df["description"].astype(str)
        lineitems_df["qty"] = pd.to_numeric(lineitems_df["qty"], errors="coerce")


# --- PAGE 30 ---

        lineitems_df["unit_price"] = pd.to_numeric(lineitems_df["unit_price"], er
        lineitems_df["line_total"] = pd.to_numeric(lineitems_df["line_total"], er
    return headers_df, lineitems_df
def build_receipt_features(headers_df: pd.DataFrame, lineitems_df: pd.DataFrame)
    if lineitems_df.empty:
        return pd.DataFrame({"patient_id": headers_df["patient_id"].astype(int)})
    li = lineitems_df.copy()
    li["category"] = [categorize_code(c, d) for c, d in zip(li["code"], li["descr
    li["is_em"] = li["category"].eq("ED_EM")
    # ED E/M level from 9928X
    li["em_level"] = np.where(li["is_em"], li["code"].str[-1].astype(int), np.nan
    g = li.groupby("patient_id", observed=True)
    feat = pd.DataFrame(index=g.size().index)
    feat["pdf_n_line_items"] = g.size()
    feat["pdf_n_unique_codes"] = g["code"].nunique()
    feat["pdf_total_qty"] = g["qty"].sum()
    feat["pdf_total_line_cost"] = g["line_total"].sum()
    # category costs and counts
    for cat in ["ED_EM", "IMAGING", "LAB", "OBS", "OTHER"]:
        m = li["category"].eq(cat)
        feat[f"pdf_cost_{cat.lower()}"] = li.loc[m].groupby("patient_id")["line_t
        feat[f"pdf_cnt_{cat.lower()}"] = li.loc[m].groupby("patient_id")["code"].
    feat = feat.fillna(0)
    # cost shares
    denom = feat["pdf_total_line_cost"].replace(0, np.nan)
    for cat in ["ED_EM", "IMAGING", "LAB", "OBS", "OTHER"]:
        feat[f"pdf_share_{cat.lower()}"] = (feat[f"pdf_cost_{cat.lower()}"] / den
    # E/M severity mix


# --- PAGE 31 ---

    em = li[li["is_em"]].copy()
    if not em.empty:
        emg = em.groupby("patient_id", observed=True)
        feat["pdf_em_visit_lines"] = emg.size()
        feat["pdf_em_level_max"] = emg["em_level"].max()
        feat["pdf_em_level_mean"] = emg["em_level"].mean()
        feat["pdf_em_high_frac"] = (em[em["em_level"] >= 4].groupby("patient_id")
    else:
        feat["pdf_em_visit_lines"] = 0
        feat["pdf_em_level_max"] = 0
        feat["pdf_em_level_mean"] = 0
        feat["pdf_em_high_frac"] = 0
    # complexity: entropy of cost distribution over codes
    cost_by_code = li.groupby(["patient_id", "code"], observed=True)["line_total"
    def entropy_cost(series: pd.Series) -> float:
        total = series.sum()
        if total <= 0:
            return 0.0
        p = series / total
        return float(-(p * np.log(p + 1e-12)).sum())
    ent = cost_by_code.groupby(level=0).apply(entropy_cost)
    feat["pdf_code_cost_entropy"] = ent
    feat["pdf_code_cost_entropy_norm"] = feat["pdf_code_cost_entropy"] / np.log(
        feat["pdf_n_unique_codes"].replace(0, np.nan)
    ).fillna(1)
    # non-EM line density
    feat["pdf_nonem_per_em"] = (feat["pdf_n_line_items"] - feat["pdf_cnt_ed_em"])
        feat["pdf_cnt_ed_em"].replace(0, np.nan)
    )
    feat["pdf_nonem_per_em"] = feat["pdf_nonem_per_em"].fillna(0)
    feat = feat.reset_index()
    # attach key receipt header fields
    hdr = headers_df.set_index("patient_id")
    keep_cols = [c for c in ["receipt_total", "sum_line_total", "sum_unit_x_qty",


# --- PAGE 32 ---

    if keep_cols:
        feat = feat.merge(hdr[keep_cols].reset_index(), on="patient_id", how="left
    return feat
def save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
def plot_hist(series: pd.Series, title: str, path: Path, logx: bool = False) -> N
    s = series.dropna().astype(float)
    if s.empty:
        return
    plt.figure(figsize=(7, 4))
    if logx:
        s = s[s > 0]
        plt.hist(np.log1p(s), bins=60)
        plt.xlabel("log1p(value)")
    else:
        plt.hist(s, bins=60)
        plt.xlabel("value")
    plt.title(title)
    save_fig(path)
def plot_scatter(x: pd.Series, y: pd.Series, title: str, path: Path, logxy: bool
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if df.empty:
        return
    plt.figure(figsize=(6, 5))
    if logxy:
        df = df[(df["x"] > 0) & (df["y"] > 0)]
        plt.scatter(np.log1p(df["x"]), np.log1p(df["y"]), s=8, alpha=0.35)
        plt.xlabel("log1p(x)")
        plt.ylabel("log1p(y)")
    else:
        plt.scatter(df["x"], df["y"], s=8, alpha=0.35)


# --- PAGE 33 ---

        plt.xlabel("x")
        plt.ylabel("y")
    plt.title(title)
    save_fig(path)
def main():
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 160)
    print(f"[paths] BASE_DIR={BASE_DIR}")
    print(f"[paths] EDA_OUT_DIR={EDA_OUT_DIR}")
    print(f"[paths] RECEIPT_CACHE={RECEIPT_CACHE}")
    # 1) Load data
    train, test, patients = load_data()
    # 2) Basic checks
    report = basic_integrity_checks(train, test, patients)
    print("\n=== BASIC INTEGRITY REPORT ===")
    print(json.dumps(report, indent=2))
    # Write report
    with open(EDA_OUT_DIR / "basic_integrity_report.json", "w", encoding="utf-8")
        json.dump(report, f, indent=2)
    # 3) Merge with patients if available
    if patients is not None:
        train_full = train.merge(patients, on="patient_id", how="left", validate=
        test_full  = test.merge(patients, on="patient_id", how="left", validate="
    else:
        train_full, test_full = train.copy(), test.copy()
    # 4) Distribution plots (train)
    if "ed_cost_next3y_usd" in train_full.columns:
        plot_hist(train_full["ed_cost_next3y_usd"], "Target: ed_cost_next3y_usd (
        plot_hist(train_full["ed_cost_next3y_usd"], "Target: ed_cost_next3y_usd (
    plot_hist(train_full["prior_ed_cost_5y_usd"], "Prior 5y ED cost (train)", EDA_
    plot_hist(train_full["prior_ed_cost_5y_usd"], "Prior 5y ED cost (train) [log1


# --- PAGE 34 ---

    plot_hist(train_full["prior_ed_visits_5y"], "Prior 5y ED visits (train)", EDA_
    if "ed_cost_next3y_usd" in train_full.columns:
        plot_scatter(train_full["prior_ed_cost_5y_usd"], train_full["ed_cost_next
                     "Prior 5y cost vs Next 3y cost (train)", EDA_OUT_DIR / "scatt
        plot_scatter(train_full["prior_ed_cost_5y_usd"], train_full["ed_cost_next
                     "Prior 5y cost vs Next 3y cost (train) [log1p]", EDA_OUT_DIR
    # 5) Parse receipts
    all_ids = sorted(set(train_full["patient_id"].astype(int).tolist()) | set(test
    headers_df, lineitems_df = parse_all_receipts(all_ids)
    # Save raw parsed summaries
    headers_df.to_csv(EDA_OUT_DIR / "receipts_headers_parsed.csv", index=False)
    lineitems_df.head(2000).to_csv(EDA_OUT_DIR / "receipts_lineitems_head2000.csv
    # 6) Receipt integrity checks
    combined = pd.concat(
        [
            train_full[["patient_id", "prior_ed_visits_5y", "prior_ed_cost_5y_usd
            test_full[["patient_id", "prior_ed_visits_5y", "prior_ed_cost_5y_usd"
        ],
        ignore_index=True,
    ).merge(headers_df, on="patient_id", how="left")
    combined["diff_total_vs_prior"] = combined["receipt_total"] - combined["prior_
    combined["abs_diff_total_vs_prior"] = combined["diff_total_vs_prior"].abs()
    # sum(line totals) vs total
    combined["diff_sumlines_vs_total"] = combined["sum_line_total"] - combined["r
    combined["abs_diff_sumlines_vs_total"] = combined["diff_sumlines_vs_total"].a
    # E/M line count vs prior visits
    if not lineitems_df.empty:
        em_mask = lineitems_df["code"].astype(str).str.match(r"^9928[1-5]$")
        em_counts = lineitems_df.loc[em_mask].groupby("patient_id")["code"].size(
        combined = combined.merge(em_counts, on="patient_id", how="left")
        combined["em_line_count"] = combined["em_line_count"].fillna(0).astype(int
        combined["diff_emlines_vs_priorvisits"] = combined["em_line_count"] - com


# --- PAGE 35 ---

    else:
        combined["em_line_count"] = 0
        combined["diff_emlines_vs_priorvisits"] = np.nan
    # If patients.csv exists, compare insurance/zip3
    if patients is not None:
        if "insurance" in combined.columns and "insurance_receipt" in combined.co
            combined["insurance_match"] = combined["insurance"].astype(str).str.l
        if "zip3" in combined.columns and "zip3_receipt" in combined.columns:
            combined["zip3_match"] = combined["zip3"].apply(standardize_zip3) ==
    combined.to_csv(EDA_OUT_DIR / "receipt_integrity_joined.csv", index=False)
    print("\n=== RECEIPT INTEGRITY SUMMARY ===")
    ok_parse_rate = float(np.mean(combined["parse_ok"].fillna(False))) if "parse_
    print(f"Parse OK rate: {ok_parse_rate:.4f}")
    print("Abs(TOTAL - prior_ed_cost_5y_usd):")
    print(combined["abs_diff_total_vs_prior"].describe(percentiles=[0.5, 0.9, 0.9
    print("Abs(sum(line_totals) - TOTAL):")
    print(combined["abs_diff_sumlines_vs_total"].describe(percentiles=[0.5, 0.9,
    if "diff_emlines_vs_priorvisits" in combined.columns:
        print("E/M line count - prior_ed_visits_5y:")
        print(combined["diff_emlines_vs_priorvisits"].describe(percentiles=[0.5,
    plot_hist(combined["abs_diff_total_vs_prior"], "Abs(TOTAL - prior_cost) acros
    plot_hist(combined["abs_diff_total_vs_prior"], "Abs(TOTAL - prior_cost) acros
    # 7) Build receipt-derived per-patient features
    receipt_feat = build_receipt_features(headers_df, lineitems_df)
    receipt_feat.to_csv(EDA_OUT_DIR / "receipt_features_per_patient.csv", index=F
    # 8) Merge receipt features into train for correlation / group analysis
    train_aug = train_full.merge(receipt_feat, on="patient_id", how="left")
    train_aug.to_csv(EDA_OUT_DIR / "train_aug_with_receipt_features.csv", index=F
    # 9) Group summaries (train)
    if "ed_cost_next3y_usd" in train_aug.columns:


# --- PAGE 36 ---

        group_cols = ["primary_chronic"]
        if "insurance" in train_aug.columns:
            group_cols.append("insurance")
        summary = (
            .groupby(group_cols, dropna=False)
            .agg(
                n=("patient_id", "size"),
                target_mean=("ed_cost_next3y_usd", "mean"),
                target_median=("ed_cost_next3y_usd", "median"),
                prior_cost_mean=("prior_ed_cost_5y_usd", "mean"),
                prior_visits_mean=("prior_ed_visits_5y", "mean"),
                em_mean=("pdf_em_level_mean", "mean"),
                imaging_share_mean=("pdf_share_imaging", "mean"),
                obs_share_mean=("pdf_share_obs", "mean"),
            )
            .reset_index()
            .sort_values("target_mean", ascending=False)
        )
        summary.to_csv(EDA_OUT_DIR / "group_summary_chronic_insurance.csv", index
        print("\n=== GROUP SUMMARY (top rows) ===")
        print(summary.head(30).to_string(index=False))
    # ZIP3 analysis (train only, if present)
    if "zip3" in train_aug.columns and "ed_cost_next3y_usd" in train_aug.columns:
        zip_stats = (
            .groupby("zip3", dropna=False)
            .agg(
                n=("patient_id", "size"),
                target_mean=("ed_cost_next3y_usd", "mean"),
                target_median=("ed_cost_next3y_usd", "median"),
                prior_cost_mean=("prior_ed_cost_5y_usd", "mean"),
                prior_visits_mean=("prior_ed_visits_5y", "mean"),
            )
            .reset_index()
        )
        zip_stats = zip_stats[zip_stats["n"] >= 20].sort_values("target_mean", as
        zip_stats.to_csv(EDA_OUT_DIR / "zip3_stats_nge20.csv", index=False)
        print("\n=== ZIP3 STATS (n>=20, top 15 by target_mean) ===")


# --- PAGE 37 ---

        print(zip_stats.head(15).to_string(index=False))
    # 10) Correlation scan (train only)
    if "ed_cost_next3y_usd" in train_aug.columns:
        candidate_cols = [c for c in train_aug.columns if c.startswith("pdf_")]
        # remove trivial totals that replicate prior cost
        drop_like = {"receipt_total", "sum_line_total", "sum_unit_x_qty"}
        candidate_cols = [c for c in candidate_cols if c not in drop_like]
        corr = (
            train_aug[["ed_cost_next3y_usd"] + candidate_cols]
            .corr(method="spearman")["ed_cost_next3y_usd"]
            .drop("ed_cost_next3y_usd")
            .sort_values(key=lambda s: s.abs(), ascending=False)
        )
        corr.to_csv(EDA_OUT_DIR / "spearman_corr_receipt_features_vs_target.csv")
        print("\n=== TOP SPEARMAN CORRELATIONS (receipt features vs target) ===")
        print(corr.head(25).to_string())
    # 11) Code frequency + univariate "uplift" vs target (train only)
    if not lineitems_df.empty and "ed_cost_next3y_usd" in train_full.columns:
        train_ids = set(train_full["patient_id"].astype(int).tolist())
        li_train = lineitems_df[lineitems_df["patient_id"].isin(train_ids)].copy(
        # Basic frequency tables
        code_freq = (
            li_train.groupby("code", observed=True)
            .agg(
                n_lines=("code", "size"),
                n_patients=("patient_id", "nunique"),
                total_cost=("line_total", "sum"),
                mean_unit_price=("unit_price", "mean"),
            )
            .reset_index()
            .sort_values(["n_patients", "total_cost"], ascending=False)
        )
        code_freq.to_csv(EDA_OUT_DIR / "code_frequency_train.csv", index=False)
        # Uplift: mean target with code vs without code
        t = train_full[["patient_id", "ed_cost_next3y_usd"]].copy()


# --- PAGE 38 ---

        total_sum = float(t["ed_cost_next3y_usd"].sum())
        N = int(t.shape[0])
        presence = li_train[["patient_id", "code"]].drop_duplicates()
        merged = presence.merge(t, on="patient_id", how="left")
        mean_with = merged.groupby("code")["ed_cost_next3y_usd"].mean()
        n_with = merged.groupby("code")["patient_id"].nunique()
        sum_with = merged.groupby("code")["ed_cost_next3y_usd"].sum()
        # compute mean_without efficiently
        mean_without = (total_sum - sum_with) / (N - n_with).replace(0, np.nan)
        uplift = pd.DataFrame(
            {
                "n_patients_with_code": n_with,
                "mean_target_with_code": mean_with,
                "mean_target_without_code": mean_without,
            }
        ).reset_index()
        uplift["diff_with_minus_without"] = uplift["mean_target_with_code"] - upl
        uplift = uplift[uplift["n_patients_with_code"] >= MIN_PATIENTS_WITH_CODE]
            "diff_with_minus_without", ascending=False
        )
        uplift.to_csv(EDA_OUT_DIR / "code_uplift_train_nge_min.csv", index=False)
        print("\n=== TOP POSITIVE UPLIFT CODES (train, univariate; n>=threshold)
        print(uplift.head(25).to_string(index=False))
        # Insurance unit price comparison for common codes (if insurance exists)
        if patients is not None and "insurance" in patients.columns:
            li_train = li_train.merge(patients[["patient_id", "insurance"]], on="
            common_codes = code_freq.head(10)["code"].tolist()
            price_stats = (
                li_train[li_train["code"].isin(common_codes)]
                .groupby(["code", "insurance"], dropna=False)
                .agg(
                    n_lines=("unit_price", "size"),
                    unit_median=("unit_price", "median"),


# --- PAGE 39 ---
```

### C02 — Code cell starting at page 42
```python
# Phase 1 EDA (Notebook Cell Version) — Cache-safe + Robust PDF parsing
from __future__ import annotations
from pathlib import Path
import re
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from joblib import dump, load
from tqdm import tqdm


# --- PAGE 43 ---

import matplotlib.pyplot as plt
# -----------------------------
# PATHS (as you specified)
# -----------------------------
BASE_DIR = Path(r"D:\AgentDs\agent_ds_healthcare")
TRAIN_CSV = BASE_DIR / "ed_cost_train.csv"
TEST_CSV  = BASE_DIR / "ed_cost_test.csv"
PATIENTS_CSV = BASE_DIR / "patients.csv"
PDF_DIR = BASE_DIR / "receipts_pdf"
CACHE_DIR = BASE_DIR / "cache_iter10"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RECEIPT_CACHE = CACHE_DIR / "receipts_parsed.joblib"
EDA_OUT_DIR = CACHE_DIR / "eda_outputs"
EDA_OUT_DIR.mkdir(parents=True, exist_ok=True)
# -----------------------------
# CONFIG
# -----------------------------
FORCE_REPARSE = False          # set True to ignore cache
LIMIT_PDFS: Optional[int] = None  # set e.g. 200 for quick smoke test
MIN_PATIENTS_WITH_CODE = 30
CACHE_SCHEMA_VERSION = "receipts_parsed_v2"
plt.rcParams["figure.dpi"] = 120
# -----------------------------
# Helpers
# -----------------------------
def standardize_zip3(z) -> Optional[str]:
    if z is None or (isinstance(z, float) and np.isnan(z)):
        return None
    s = str(z).strip()
    s = re.sub(r"\D", "", s)
    if not s:
        return None


# --- PAGE 44 ---

    return s.zfill(3)
def parse_money(x: str) -> float:
    x = str(x).strip()
    x = re.sub(r"[^\d\.\-]", "", x)
    return float(x) if x else np.nan
def is_code_token(s: str) -> bool:
    s = str(s).strip().upper()
    return bool(re.match(r"^[A-Z]{0,2}\d{4,5}$", s))
_money_token_re = re.compile(r"^[\$\d,]+\.\d{2}$")
def try_parse_wide_line(line: str) -> Optional[Dict]:
    Parse a 'single-line' item like:
      99283 ED visit, moderate (level 3) 1 232.38 232.38
    toks = line.strip().split()
    if not toks:
        return None
    code = toks[0].upper()
    if not is_code_token(code):
        return None
    # Find last 2 money tokens scanning from end
    money_positions = []
    for idx in range(len(toks) - 1, 0, -1):
        if _money_token_re.match(toks[idx]):
            money_positions.append(idx)
            if len(money_positions) == 2:
    if len(money_positions) < 2:
        return None
    i_line = money_positions[0]  # last money token
    i_unit = money_positions[1]  # second last money token


# --- PAGE 45 ---

    # Qty is usually immediately before unit, but search backwards if needed
    qty_idx = i_unit - 1
    if qty_idx < 1:
        return None
    if not re.match(r"^\d+$", toks[qty_idx]):
        found = None
        for j in range(i_unit - 1, 0, -1):
            if re.match(r"^\d+$", toks[j]):
                found = j
        if found is None:
            return None
        qty_idx = found
    desc = " ".join(toks[1:qty_idx]).strip()
    qty = int(toks[qty_idx])
    unit_price = parse_money(toks[i_unit])
    line_total = parse_money(toks[i_line])
    return {
        "code": code,
        "description": desc,
        "qty": qty,
        "unit_price": unit_price,
        "line_total": line_total,
    }
def categorize_code(code: str, desc: str = "") -> str:
    code = str(code).strip().upper()
    desc_l = str(desc or "").lower()
    if re.match(r"^9928[1-5]$", code):
        return "ED_EM"
    if code == "G0378" or "observation" in desc_l:
        return "OBS"
    if code.isdigit() and len(code) == 5 and code.startswith("7"):
        return "IMAGING"
    if code.isdigit() and len(code) == 5 and code.startswith("8"):


# --- PAGE 46 ---

        return "LAB"
    return "OTHER"
def extract_text_pymupdf(pdf_path: Path) -> str:
    try:
        import fitz  # PyMuPDF
    except Exception as e:
        raise RuntimeError(
            "PyMuPDF not available. Install with:\n"
            f"Import error: {e}"
        )
    doc = fitz.open(str(pdf_path))
    return "\n".join(page.get_text() for page in doc)
def parse_receipt_text(text: str) -> Tuple[Dict, List[Dict]]:
    Robust parser:
    - Extract header fields (patient_id, zip3, insurance, TOTAL)
    - Extract line items either in:
        (a) stacked-column style (code on its own line, then description, qty, un
        (b) wide single-line style (code desc qty unit line_total)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    header: Dict = {}
    m = re.search(r"Patient ID:\s*(\d+)", text)
    if m:
        header["patient_id"] = int(m.group(1))
    m = re.search(r"ZIP3:\s*([0-9]{1,3})", text)
    if m:
        header["zip3_receipt_raw"] = m.group(1)
    m = re.search(r"Insurance:\s*([A-Za-z_]+)", text)
    if m:
        header["insurance_receipt"] = m.group(1).lower()
    # TOTAL


# --- PAGE 47 ---

    total = np.nan
    for i, ln in enumerate(lines):
        if ln.startswith("TOTAL"):
            m = re.search(r"TOTAL\s*([\d\.,\-]+)", ln)
            if m:
                total = parse_money(m.group(1))
            else:
                if i + 1 < len(lines):
                    total = parse_money(lines[i + 1])
    header["receipt_total"] = total
    # Identify table start
    start = None
    for i, ln in enumerate(lines):
        if "Line Total" in ln:
            start = i + 1
    if start is None:
        # fallback: first line whose first token is a code
        for i, ln in enumerate(lines):
            first = ln.split()[0] if ln.split() else ""
            if is_code_token(first):
                start = i
    if start is None:
        return header, []
    # Attempt WIDE parsing across table block first
    items: List[Dict] = []
    for i in range(start, len(lines)):
        ln = lines[i]
        if ln.startswith("TOTAL"):
        parsed = try_parse_wide_line(ln)
        if parsed:
            items.append(parsed)
    if items:
        return header, items


# --- PAGE 48 ---

    # Fallback: STACKED parsing (PyMuPDF commonly yields this)
    items = []
    i = start
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("TOTAL"):
        if not is_code_token(ln):
            i += 1
        code = ln.strip().upper()
        # description lines until qty integer
        j = i + 1
        desc_parts: List[str] = []
        while j < len(lines) and not re.match(r"^\d+$", lines[j]) and not lines[j
            desc_parts.append(lines[j])
            j += 1
        if j >= len(lines) or lines[j].startswith("TOTAL"):
        desc = " ".join(desc_parts).strip()
        qty = int(lines[j])
        if j + 1 >= len(lines):
        unit_price = parse_money(lines[j + 1])
        if j + 2 >= len(lines):
        line_total = parse_money(lines[j + 2])
        items.append(
            {"code": code, "description": desc, "qty": qty, "unit_price": unit_pr
        )
        i = j + 3


# --- PAGE 49 ---

    return header, items
def parse_receipt_pdf(pdf_path: Path) -> Tuple[Dict, pd.DataFrame]:
    text = extract_text_pymupdf(pdf_path)
    header, items = parse_receipt_text(text)
    header["pdf_path"] = str(pdf_path)
    header["n_line_items"] = len(items)
    if items:
        li = pd.DataFrame(items)
        header["sum_line_total"] = float(np.nansum(li["line_total"].values))
        header["sum_unit_x_qty"] = float(np.nansum((li["unit_price"] * li["qty"])
    else:
        li = pd.DataFrame(columns=["code", "description", "qty", "unit_price", "l
        header["sum_line_total"] = 0.0
        header["sum_unit_x_qty"] = 0.0
    return header, li
def backup_bad_cache(cache_path: Path, reason: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_path = cache_path.with_name(f"{cache_path.stem}.BAD_{reason}_{ts}{cache_p
    cache_path.rename(new_path)
    return new_path
def load_receipt_cache_safe(cache_path: Path) -> Optional[Tuple[pd.DataFrame, pd.
    if not cache_path.exists():
        return None
    try:
        obj = load(cache_path)
    except Exception as e:
        print(f"[cache] Failed to load cache ({cache_path.name}): {e}")
        bad = backup_bad_cache(cache_path, "load_error")
        print(f"[cache] Backed up bad cache to: {bad}")
        return None
    if not isinstance(obj, dict):


# --- PAGE 50 ---

        print(f"[cache] Cache schema mismatch: expected dict, got {type(obj)}")
        bad = backup_bad_cache(cache_path, "not_dict")
        print(f"[cache] Backed up bad cache to: {bad}")
        return None
    if obj.get("schema_version") != CACHE_SCHEMA_VERSION:
        # allow migration attempt only if it already has required keys
        if "headers_df" in obj and "lineitems_df" in obj:
            print("[cache] Cache schema_version mismatch but required keys exist.
        else:
            print(f"[cache] Cache schema_version mismatch or missing. Found schem
            print(f"[cache] Keys in cache: {list(obj.keys())}")
            bad = backup_bad_cache(cache_path, "schema_mismatch")
            print(f"[cache] Backed up bad cache to: {bad}")
            return None
    if "headers_df" not in obj or "lineitems_df" not in obj:
        print(f"[cache] Cache missing expected keys. Keys found: {list(obj.keys()
        bad = backup_bad_cache(cache_path, "missing_keys")
        print(f"[cache] Backed up bad cache to: {bad}")
        return None
    headers_df = obj["headers_df"]
    lineitems_df = obj["lineitems_df"]
    # minimal validation
    if not isinstance(headers_df, pd.DataFrame) or "patient_id" not in headers_df
        print("[cache] headers_df invalid in cache")
        bad = backup_bad_cache(cache_path, "bad_headers")
        print(f"[cache] Backed up bad cache to: {bad}")
        return None
    if not isinstance(lineitems_df, pd.DataFrame) or ("patient_id" not in lineite
        print("[cache] lineitems_df invalid in cache")
        bad = backup_bad_cache(cache_path, "bad_lineitems")
        print(f"[cache] Backed up bad cache to: {bad}")
        return None
    return headers_df, lineitems_df


# --- PAGE 51 ---

def save_receipt_cache(cache_path: Path, headers_df: pd.DataFrame, lineitems_df:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "headers_df": headers_df,
        "lineitems_df": lineitems_df,
    }
    dump(payload, cache_path, compress=3)
def parse_receipts(patient_ids: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Try safe load
    if not FORCE_REPARSE:
        cached = load_receipt_cache_safe(RECEIPT_CACHE)
        if cached is not None:
            headers_df, lineitems_df = cached
            cached_ids = set(headers_df["patient_id"].astype(int).tolist())
            needed = set(patient_ids)
            missing = sorted(list(needed - cached_ids))
            if not missing:
                print(f"[cache] Loaded cache OK ({RECEIPT_CACHE.name}): {len(cach
                return headers_df, lineitems_df
            print(f"[cache] Cache OK but missing {len(missing)} patients — parsin
            headers_new, items_new = _parse_receipts_no_cache(missing)
            headers_df = pd.concat([headers_df, headers_new], ignore_index=True)
            lineitems_df = pd.concat([lineitems_df, items_new], ignore_index=True
            save_receipt_cache(RECEIPT_CACHE, headers_df, lineitems_df)
            return headers_df, lineitems_df
    # Reparse all
    headers_df, lineitems_df = _parse_receipts_no_cache(patient_ids)
    save_receipt_cache(RECEIPT_CACHE, headers_df, lineitems_df)
    print(f"[cache] Wrote cache: {RECEIPT_CACHE}")
    return headers_df, lineitems_df
def _parse_receipts_no_cache(patient_ids: List[int]) -> Tuple[pd.DataFrame, pd.Da
    headers: List[Dict] = []
    all_items: List[pd.DataFrame] = []
    ids = patient_ids
    if LIMIT_PDFS is not None:


# --- PAGE 52 ---

        ids = ids[:LIMIT_PDFS]
        print(f"[info] LIMIT_PDFS={LIMIT_PDFS}: parsing {len(ids)} receipts (of {
    missing_files = 0
    for pid in tqdm(ids, desc="Parsing receipts"):
        pdf_path = PDF_DIR / f"receipt_{pid}.pdf"
        if not pdf_path.exists():
            missing_files += 1
            headers.append(
                {
                    "patient_id": pid,
                    "pdf_path": str(pdf_path),
                    "parse_ok": False,
                    "parse_error": "missing_file",
                    "n_line_items": 0,
                    "receipt_total": np.nan,
                    "sum_line_total": np.nan,
                    "sum_unit_x_qty": np.nan,
                    "zip3_receipt_raw": None,
                    "insurance_receipt": None,
                }
            )
        try:
            h, li = parse_receipt_pdf(pdf_path)
            h["parse_ok"] = True
            h["parse_error"] = None
            li = li.copy()
            li["patient_id"] = int(pid)
            all_items.append(li)
            headers.append(h)
        except Exception as e:
            headers.append(
                {
                    "patient_id": pid,
                    "pdf_path": str(pdf_path),
                    "parse_ok": False,


# --- PAGE 53 ---

                    "parse_error": str(e)[:2000],
                    "n_line_items": 0,
                    "receipt_total": np.nan,
                    "sum_line_total": np.nan,
                    "sum_unit_x_qty": np.nan,
                    "zip3_receipt_raw": None,
                    "insurance_receipt": None,
                }
            )
    headers_df = pd.DataFrame(headers)
    lineitems_df = (
        pd.concat(all_items, ignore_index=True)
        if all_items
        else pd.DataFrame(columns=["patient_id", "code", "description", "qty", "u
    )
    if missing_files:
        print(f"[warn] Missing receipt PDFs: {missing_files}")
    # Standardize
    headers_df["patient_id"] = headers_df["patient_id"].astype(int)
    headers_df["zip3_receipt"] = headers_df.get("zip3_receipt_raw", pd.Series([No
    headers_df["insurance_receipt"] = headers_df.get("insurance_receipt", pd.Seri
    if not lineitems_df.empty:
        lineitems_df["code"] = lineitems_df["code"].astype(str).str.upper().str.st
        lineitems_df["description"] = lineitems_df["description"].astype(str)
        lineitems_df["qty"] = pd.to_numeric(lineitems_df["qty"], errors="coerce")
        lineitems_df["unit_price"] = pd.to_numeric(lineitems_df["unit_price"], er
        lineitems_df["line_total"] = pd.to_numeric(lineitems_df["line_total"], er
    return headers_df, lineitems_df
def build_receipt_features(headers_df: pd.DataFrame, lineitems_df: pd.DataFrame)
    if lineitems_df.empty:
        base = pd.DataFrame({"patient_id": headers_df["patient_id"].astype(int)})
        return base.merge(headers_df[["patient_id","receipt_total","sum_line_tota
                          on="patient_id", how="left")


# --- PAGE 54 ---

    li = lineitems_df.copy()
    li["category"] = [categorize_code(c, d) for c, d in zip(li["code"], li["descr
    li["is_em"] = li["category"].eq("ED_EM")
    li["em_level"] = np.where(li["is_em"], li["code"].str[-1].astype(int), np.nan
    g = li.groupby("patient_id", observed=True)
    feat = pd.DataFrame(index=g.size().index)
    feat["pdf_n_line_items"] = g.size()
    feat["pdf_n_unique_codes"] = g["code"].nunique()
    feat["pdf_total_qty"] = g["qty"].sum()
    feat["pdf_total_line_cost"] = g["line_total"].sum()
    for cat in ["ED_EM", "IMAGING", "LAB", "OBS", "OTHER"]:
        m = li["category"].eq(cat)
        feat[f"pdf_cost_{cat.lower()}"] = li.loc[m].groupby("patient_id")["line_t
        feat[f"pdf_cnt_{cat.lower()}"] = li.loc[m].groupby("patient_id")["code"].
    feat = feat.fillna(0)
    denom = feat["pdf_total_line_cost"].replace(0, np.nan)
    for cat in ["ED_EM", "IMAGING", "LAB", "OBS", "OTHER"]:
        feat[f"pdf_share_{cat.lower()}"] = (feat[f"pdf_cost_{cat.lower()}"] / den
    em = li[li["is_em"]].copy()
    if not em.empty:
        emg = em.groupby("patient_id", observed=True)
        feat["pdf_em_visit_lines"] = emg.size()
        feat["pdf_em_level_max"] = emg["em_level"].max()
        feat["pdf_em_level_mean"] = emg["em_level"].mean()
        feat["pdf_em_high_frac"] = (em[em["em_level"] >= 4].groupby("patient_id")
    else:
        feat["pdf_em_visit_lines"] = 0
        feat["pdf_em_level_max"] = 0
        feat["pdf_em_level_mean"] = 0
        feat["pdf_em_high_frac"] = 0
    cost_by_code = li.groupby(["patient_id", "code"], observed=True)["line_total"
    def entropy_cost(series: pd.Series) -> float:


# --- PAGE 55 ---

        total = series.sum()
        if total <= 0:
            return 0.0
        p = series / total
        return float(-(p * np.log(p + 1e-12)).sum())
    ent = cost_by_code.groupby(level=0).apply(entropy_cost)
    feat["pdf_code_cost_entropy"] = ent
    feat["pdf_code_cost_entropy_norm"] = feat["pdf_code_cost_entropy"] / np.log(
        feat["pdf_n_unique_codes"].replace(0, np.nan)
    ).fillna(1)
    feat["pdf_nonem_per_em"] = (feat["pdf_n_line_items"] - feat["pdf_cnt_ed_em"])
        feat["pdf_cnt_ed_em"].replace(0, np.nan)
    )
    feat["pdf_nonem_per_em"] = feat["pdf_nonem_per_em"].fillna(0)
    feat = feat.reset_index()
    hdr_cols = ["patient_id","receipt_total","sum_line_total","sum_unit_x_qty","z
    feat = feat.merge(headers_df[hdr_cols], on="patient_id", how="left")
    return feat
def plot_hist(series: pd.Series, title: str, path: Path, logx: bool = False) -> N
    s = series.dropna().astype(float)
    if s.empty:
        return
    plt.figure(figsize=(7, 4))
    if logx:
        s = s[s > 0]
        plt.hist(np.log1p(s), bins=60)
        plt.xlabel("log1p(value)")
    else:
        plt.hist(s, bins=60)
        plt.xlabel("value")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# --- PAGE 56 ---

def plot_scatter(x: pd.Series, y: pd.Series, title: str, path: Path, logxy: bool
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if df.empty:
        return
    plt.figure(figsize=(6, 5))
    if logxy:
        df = df[(df["x"] > 0) & (df["y"] > 0)]
        plt.scatter(np.log1p(df["x"]), np.log1p(df["y"]), s=8, alpha=0.35)
        plt.xlabel("log1p(x)")
        plt.ylabel("log1p(y)")
    else:
        plt.scatter(df["x"], df["y"], s=8, alpha=0.35)
        plt.xlabel("x")
        plt.ylabel("y")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
# -----------------------------
# RUN EDA
# -----------------------------
print(f"[paths] BASE_DIR={BASE_DIR}")
print(f"[paths] PDF_DIR={PDF_DIR}")
print(f"[paths] EDA_OUT_DIR={EDA_OUT_DIR}")
print(f"[paths] RECEIPT_CACHE={RECEIPT_CACHE}")
# Load CSVs
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
patients = pd.read_csv(PATIENTS_CSV) if PATIENTS_CSV.exists() else None
train["patient_id"] = train["patient_id"].astype(int)
test["patient_id"] = test["patient_id"].astype(int)
if patients is not None:
    patients["patient_id"] = patients["patient_id"].astype(int)
    if "zip3" in patients.columns:
        patients["zip3"] = patients["zip3"].apply(standardize_zip3)


# --- PAGE 57 ---

    if "insurance" in patients.columns:
        patients["insurance"] = patients["insurance"].astype(str).str.lower().str
# Basic integrity report
report = {
    "n_train": int(train.shape[0]),
    "n_test": int(test.shape[0]),
    "train_dupe_patient_id": int(train["patient_id"].duplicated().sum()),
    "test_dupe_patient_id": int(test["patient_id"].duplicated().sum()),
    "train_missing_values": train.isna().sum().to_dict(),
    "test_missing_values": test.isna().sum().to_dict(),
    "patients_loaded": patients is not None,
}
if patients is not None:
    report["n_patients"] = int(patients.shape[0])
    report["patients_dupe_patient_id"] = int(patients["patient_id"].duplicated().
    report["train_missing_in_patients_n"] = len(set(train["patient_id"]) - set(pat
    report["test_missing_in_patients_n"] = len(set(test["patient_id"]) - set(pati
print("\n=== BASIC INTEGRITY REPORT ===")
print(json.dumps(report, indent=2))
(EDA_OUT_DIR / "basic_integrity_report.json").write_text(json.dumps(report, inden
# Merge with patients
if patients is not None:
    train_full = train.merge(patients, on="patient_id", how="left", validate="one_
    test_full  = test.merge(patients, on="patient_id", how="left", validate="one_t
else:
    train_full = train.copy()
    test_full = test.copy()
# Plots (train)
if "ed_cost_next3y_usd" in train_full.columns:
    plot_hist(train_full["ed_cost_next3y_usd"], "Target: ed_cost_next3y_usd (trai
    plot_hist(train_full["ed_cost_next3y_usd"], "Target: ed_cost_next3y_usd (trai
plot_hist(train_full["prior_ed_cost_5y_usd"], "Prior 5y ED cost (train)", EDA_OUT_
plot_hist(train_full["prior_ed_cost_5y_usd"], "Prior 5y ED cost (train) [log1p]",
plot_hist(train_full["prior_ed_visits_5y"], "Prior 5y ED visits (train)", EDA_OUT_


# --- PAGE 58 ---

if "ed_cost_next3y_usd" in train_full.columns:
    plot_scatter(train_full["prior_ed_cost_5y_usd"], train_full["ed_cost_next3y_u
    plot_scatter(train_full["prior_ed_cost_5y_usd"], train_full["ed_cost_next3y_u
                 "Prior 5y cost vs Next 3y cost [log1p]", EDA_OUT_DIR / "scatter_
# Parse receipts with cache safety
all_ids = sorted(set(train_full["patient_id"].tolist()) | set(test_full["patient_
headers_df, lineitems_df = parse_receipts(all_ids)
headers_df.to_csv(EDA_OUT_DIR / "receipts_headers_parsed.csv", index=False)
lineitems_df.head(3000).to_csv(EDA_OUT_DIR / "receipts_lineitems_head3000.csv", i
# Integrity checks vs CSV priors
combined = pd.concat(
    [
        train_full[["patient_id", "prior_ed_visits_5y", "prior_ed_cost_5y_usd"]].
        test_full[["patient_id", "prior_ed_visits_5y", "prior_ed_cost_5y_usd"]].a
    ],
    ignore_index=True,
).merge(headers_df, on="patient_id", how="left")
combined["diff_total_vs_prior"] = combined["receipt_total"] - combined["prior_ed_
combined["abs_diff_total_vs_prior"] = combined["diff_total_vs_prior"].abs()
combined["diff_sumlines_vs_total"] = combined["sum_line_total"] - combined["recei
combined["abs_diff_sumlines_vs_total"] = combined["diff_sumlines_vs_total"].abs()
# E/M lines vs prior visits
if not lineitems_df.empty:
    em_mask = lineitems_df["code"].astype(str).str.match(r"^9928[1-5]$")
    em_counts = lineitems_df.loc[em_mask].groupby("patient_id")["code"].size().re
    combined = combined.merge(em_counts, on="patient_id", how="left")
    combined["em_line_count"] = combined["em_line_count"].fillna(0).astype(int)
    combined["diff_emlines_vs_priorvisits"] = combined["em_line_count"] - combine
# Compare insurance/zip3 receipt vs patients
if patients is not None:
    if "insurance" in combined.columns:
        combined["insurance_match"] = (


# --- PAGE 59 ---

            combined["insurance"].astype(str).str.lower().str.strip()
            == combined["insurance_receipt"].astype(str).str.lower().str.strip()
        )
    if "zip3" in combined.columns:
        combined["zip3_match"] = combined["zip3"].apply(standardize_zip3) == comb
combined.to_csv(EDA_OUT_DIR / "receipt_integrity_joined.csv", index=False)
print("\n=== RECEIPT INTEGRITY SUMMARY ===")
ok_parse_rate = float(np.mean(combined["parse_ok"].fillna(False)))
print(f"Parse OK rate: {ok_parse_rate:.4f}")
print("\nAbs(TOTAL - prior_ed_cost_5y_usd):")
print(combined["abs_diff_total_vs_prior"].describe(percentiles=[0.5, 0.9, 0.99]).t
print("\nAbs(sum(line_totals) - TOTAL):")
print(combined["abs_diff_sumlines_vs_total"].describe(percentiles=[0.5, 0.9, 0.99
if "diff_emlines_vs_priorvisits" in combined.columns:
    print("\nE/M line count - prior_ed_visits_5y:")
    print(combined["diff_emlines_vs_priorvisits"].describe(percentiles=[0.5, 0.9,
plot_hist(combined["abs_diff_total_vs_prior"], "Abs(TOTAL - prior_cost)", EDA_OUT_
plot_hist(combined["abs_diff_total_vs_prior"], "Abs(TOTAL - prior_cost) [log1p]",
# Build receipt features and merge to train
receipt_feat = build_receipt_features(headers_df, lineitems_df)
receipt_feat.to_csv(EDA_OUT_DIR / "receipt_features_per_patient.csv", index=False
train_aug = train_full.merge(receipt_feat, on="patient_id", how="left")
train_aug.to_csv(EDA_OUT_DIR / "train_aug_with_receipt_features.csv", index=False
# Group summary
if "ed_cost_next3y_usd" in train_aug.columns:
    group_cols = ["primary_chronic"] + (["insurance"] if "insurance" in train_aug
    summary = (
        .groupby(group_cols, dropna=False)
        .agg(
            n=("patient_id", "size"),


# --- PAGE 60 ---

            target_mean=("ed_cost_next3y_usd", "mean"),
            target_median=("ed_cost_next3y_usd", "median"),
            prior_cost_mean=("prior_ed_cost_5y_usd", "mean"),
            prior_visits_mean=("prior_ed_visits_5y", "mean"),
            em_mean=("pdf_em_level_mean", "mean"),
            imaging_share_mean=("pdf_share_imaging", "mean"),
            obs_share_mean=("pdf_share_obs", "mean"),
        )
        .reset_index()
        .sort_values("target_mean", ascending=False)
    )
    summary.to_csv(EDA_OUT_DIR / "group_summary_chronic_insurance.csv", index=Fal
    print("\n=== GROUP SUMMARY (head) ===")
    print(summary.head(30).to_string(index=False))
# ZIP3 stats
if "zip3" in train_aug.columns and "ed_cost_next3y_usd" in train_aug.columns:
    zip_stats = (
        .groupby("zip3", dropna=False)
        .agg(
            n=("patient_id", "size"),
            target_mean=("ed_cost_next3y_usd", "mean"),
            target_median=("ed_cost_next3y_usd", "median"),
            prior_cost_mean=("prior_ed_cost_5y_usd", "mean"),
            prior_visits_mean=("prior_ed_visits_5y", "mean"),
        )
        .reset_index()
    )
    zip_stats = zip_stats[zip_stats["n"] >= 20].sort_values("target_mean", ascend
    zip_stats.to_csv(EDA_OUT_DIR / "zip3_stats_nge20.csv", index=False)
    print("\n=== ZIP3 STATS (n>=20, head 15) ===")
    print(zip_stats.head(15).to_string(index=False))
# Correlation scan (Spearman)
if "ed_cost_next3y_usd" in train_aug.columns:
    candidate_cols = [c for c in train_aug.columns if c.startswith("pdf_")]
    drop_like = {"receipt_total", "sum_line_total", "sum_unit_x_qty"}
    candidate_cols = [c for c in candidate_cols if c not in drop_like]


# --- PAGE 61 ---

    corr = (
        train_aug[["ed_cost_next3y_usd"] + candidate_cols]
        .corr(method="spearman")["ed_cost_next3y_usd"]
        .drop("ed_cost_next3y_usd")
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )
    corr.to_csv(EDA_OUT_DIR / "spearman_corr_receipt_features_vs_target.csv")
    print("\n=== TOP SPEARMAN CORRELATIONS (receipt features vs target) ===")
    print(corr.head(25).to_string())
# Code uplift (train only)
if not lineitems_df.empty and "ed_cost_next3y_usd" in train_full.columns:
    train_ids = set(train_full["patient_id"].astype(int).tolist())
    li_train = lineitems_df[lineitems_df["patient_id"].isin(train_ids)].copy()
    code_freq = (
        li_train.groupby("code", observed=True)
        .agg(
            n_lines=("code", "size"),
            n_patients=("patient_id", "nunique"),
            total_cost=("line_total", "sum"),
            mean_unit_price=("unit_price", "mean"),
        )
        .reset_index()
        .sort_values(["n_patients", "total_cost"], ascending=False)
    )
    code_freq.to_csv(EDA_OUT_DIR / "code_frequency_train.csv", index=False)
    t = train_full[["patient_id", "ed_cost_next3y_usd"]].copy()
    total_sum = float(t["ed_cost_next3y_usd"].sum())
    N = int(t.shape[0])
    presence = li_train[["patient_id", "code"]].drop_duplicates()
    merged = presence.merge(t, on="patient_id", how="left")
    mean_with = merged.groupby("code")["ed_cost_next3y_usd"].mean()
    n_with = merged.groupby("code")["patient_id"].nunique()
    sum_with = merged.groupby("code")["ed_cost_next3y_usd"].sum()
    mean_without = (total_sum - sum_with) / (N - n_with).replace(0, np.nan)


# --- PAGE 62 ---

    uplift = pd.DataFrame(
        {
            "n_patients_with_code": n_with,
            "mean_target_with_code": mean_with,
            "mean_target_without_code": mean_without,
        }
    ).reset_index()
    uplift["diff_with_minus_without"] = uplift["mean_target_with_code"] - uplift[
    uplift = uplift[uplift["n_patients_with_code"] >= MIN_PATIENTS_WITH_CODE].sor
        "diff_with_minus_without", ascending=False
    )
    uplift.to_csv(EDA_OUT_DIR / "code_uplift_train_nge_min.csv", index=False)
    print("\n=== TOP POSITIVE UPLIFT CODES (univariate; n>=threshold) ===")
    print(uplift.head(25).to_string(index=False))
    # Unit price by insurance for top codes (if insurance exists)
    if patients is not None and "insurance" in patients.columns:
        li_train2 = li_train.merge(patients[["patient_id", "insurance"]], on="pat
        common_codes = code_freq.head(10)["code"].tolist()
        price_stats = (
            li_train2[li_train2["code"].isin(common_codes)]
            .groupby(["code", "insurance"], dropna=False)
            .agg(
                n_lines=("unit_price", "size"),
                unit_median=("unit_price", "median"),
                unit_mean=("unit_price", "mean"),
            )
            .reset_index()
            .sort_values(["code", "insurance"])
        )
        price_stats.to_csv(EDA_OUT_DIR / "unit_price_by_insurance_common_codes.cs
        print("\n=== UNIT PRICE BY INSURANCE (common codes) ===")
        print(price_stats.head(60).to_string(index=False))


# --- PAGE 63 ---
```

### C03 — Code cell starting at page 73
```python
from pathlib import Path
import re
import numpy as np
import pandas as pd
from joblib import load
import matplotlib.pyplot as plt
# -----------------------------
# PATHS (your environment)
# -----------------------------
BASE_DIR = Path(r"D:\AgentDs\agent_ds_healthcare")
TRAIN_CSV = BASE_DIR / "ed_cost_train.csv"
TEST_CSV  = BASE_DIR / "ed_cost_test.csv"
PATIENTS_CSV = BASE_DIR / "patients.csv"
CACHE_DIR = BASE_DIR / "cache_iter10"


# --- PAGE 74 ---

RECEIPT_CACHE = CACHE_DIR / "receipts_parsed.joblib"
OUT_DIR = CACHE_DIR / "eda_outputs_phase15"
OUT_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams["figure.dpi"] = 120
def standardize_zip3(z):
    if z is None or (isinstance(z, float) and np.isnan(z)):
        return None
    s = re.sub(r"\D", "", str(z).strip())
    if not s:
        return None
    return s.zfill(3)
# -----------------------------
# Load tabular data
# -----------------------------
train = pd.read_csv(TRAIN_CSV)
test  = pd.read_csv(TEST_CSV)
patients = pd.read_csv(PATIENTS_CSV) if PATIENTS_CSV.exists() else None
train["patient_id"] = train["patient_id"].astype(int)
test["patient_id"] = test["patient_id"].astype(int)
if patients is not None:
    patients["patient_id"] = patients["patient_id"].astype(int)
    if "zip3" in patients.columns:
        patients["zip3"] = patients["zip3"].apply(standardize_zip3)
    if "insurance" in patients.columns:
        patients["insurance"] = patients["insurance"].astype(str).str.lower().str
train_full = train.merge(patients, on="patient_id", how="left", validate="one_to_
test_full  = test.merge(patients, on="patient_id", how="left", validate="one_to_o
print("[info] train_full:", train_full.shape, " test_full:", test_full.shape)
# -----------------------------
# Load cached parsed receipts
# -----------------------------
obj = load(RECEIPT_CACHE)


# --- PAGE 75 ---

if isinstance(obj, dict) and "headers_df" in obj and "lineitems_df" in obj:
    headers_df = obj["headers_df"].copy()
    li = obj["lineitems_df"].copy()
else:
    raise RuntimeError(f"Unexpected cache format in {RECEIPT_CACHE}. Keys: {list(
headers_df["patient_id"] = headers_df["patient_id"].astype(int)
li["patient_id"] = li["patient_id"].astype(int)
li["code"] = li["code"].astype(str).str.upper().str.strip()
li["qty"] = pd.to_numeric(li["qty"], errors="coerce").fillna(0).astype(int)
li["unit_price"] = pd.to_numeric(li["unit_price"], errors="coerce")
li["line_total"] = pd.to_numeric(li["line_total"], errors="coerce")
print("[info] Parsed lineitems rows:", li.shape[0], " unique patients:", li["pati
# -----------------------------
# Quick: qty distribution check (is qty always 1?)
# -----------------------------
qty_counts = li["qty"].value_counts(dropna=False).head(10)
qty_pct_1 = float((li["qty"] == 1).mean())
print("\n=== QTY DISTRIBUTION (top 10 values) ===")
print(qty_counts.to_string())
print(f"pct(qty==1): {qty_pct_1:.4f}")
# -----------------------------
# Extended categorization (separating 'OTHER' into critical buckets)
# -----------------------------
ED_EM = {f"9928{i}" for i in range(1,6)}              # 99281-99285
CRIT_CARE = {"99291", "99292"}                        # critical care
SEVERE_PROC = {"31500", "36556", "36620", "92950"}    # airway, central line, art
OBS = {"G0378"}
def cat_ext(code: str) -> str:
    if code in ED_EM: return "ED_EM"
    if code in CRIT_CARE: return "CRIT_CARE"
    if code in SEVERE_PROC: return "SEVERE_PROC"
    if code in OBS: return "OBS"
    if code.isdigit() and len(code) == 5 and code.startswith("7"): return "IMAGIN
    if code.isdigit() and len(code) == 5 and code.startswith("8"): return "LAB"
    return "OTHER"


# --- PAGE 76 ---

li["cat_ext"] = li["code"].map(cat_ext)
# -----------------------------
# Build per-patient features with extended categories
# -----------------------------
g = li.groupby("patient_id", observed=True)
feat = pd.DataFrame(index=g.size().index)
feat["li_n_lines"] = g.size()
feat["li_n_unique_codes"] = g["code"].nunique()
feat["li_total_cost"] = g["line_total"].sum()
feat["li_total_qty"] = g["qty"].sum()
for c in ["ED_EM","CRIT_CARE","SEVERE_PROC","LAB","IMAGING","OBS","OTHER"]:
    m = li["cat_ext"].eq(c)
    feat[f"cnt_{c.lower()}"] = li.loc[m].groupby("patient_id")["code"].size()
    feat[f"cost_{c.lower()}"] = li.loc[m].groupby("patient_id")["line_total"].sum
feat = feat.fillna(0)
# cost shares
den = feat["li_total_cost"].replace(0, np.nan)
for c in ["ED_EM","CRIT_CARE","SEVERE_PROC","LAB","IMAGING","OBS","OTHER"]:
    feat[f"share_{c.lower()}"] = (feat[f"cost_{c.lower()}"] / den).fillna(0)
# Encounter proxy: ED_EM + CRIT_CARE lines (candidate "visit-like" count)
feat["encounter_proxy_lines"] = feat["cnt_ed_em"] + feat["cnt_crit_care"]
feat = feat.reset_index()
# Merge into train for analysis
train_aug = train_full.merge(feat, on="patient_id", how="left")
train_aug.to_csv(OUT_DIR / "train_aug_phase15.csv", index=False)
# -----------------------------
# Visit proxy validation
# -----------------------------
d = train_aug[["prior_ed_visits_5y","cnt_ed_em","cnt_crit_care","encounter_proxy_
d["diff_em_only"] = d["cnt_ed_em"] - d["prior_ed_visits_5y"]


# --- PAGE 77 ---

d["diff_em_plus_crit"] = d["encounter_proxy_lines"] - d["prior_ed_visits_5y"]
print("\n=== VISIT PROXY VALIDATION (TRAIN) ===")
print("diff_em_only (cnt_ed_em - prior_visits):")
print(d["diff_em_only"].describe(percentiles=[0.5,0.9,0.99]).to_string())
print("\ndiff_em_plus_crit (cnt_ed_em + cnt_crit_care - prior_visits):")
print(d["diff_em_plus_crit"].describe(percentiles=[0.5,0.9,0.99]).to_string())
# Correlations with prior visits
corrs = d.corr(method="spearman")["prior_ed_visits_5y"].sort_values(ascending=Fal
print("\nSpearman corr with prior_ed_visits_5y:")
print(corrs.to_string())
# -----------------------------
# Target ratio analysis
# -----------------------------
train_aug["ratio_future_to_past"] = train_aug["ed_cost_next3y_usd"] / train_aug["
train_aug["ratio_future_to_past"] = train_aug["ratio_future_to_past"].replace([np
train_aug["critical_any"] = ((train_aug["cnt_crit_care"] > 0) | (train_aug["cnt_s
train_aug["critical_count"] = train_aug["cnt_crit_care"] + train_aug["cnt_severe_
print("\n=== FUTURE/PAST RATIO SUMMARY (TRAIN) ===")
print(train_aug["ratio_future_to_past"].describe(percentiles=[0.1,0.5,0.9,0.99]).t
grp = (
    .groupby(["primary_chronic","insurance","critical_any"], dropna=False)
    .agg(n=("patient_id","size"),
         ratio_mean=("ratio_future_to_past","mean"),
         ratio_med=("ratio_future_to_past","median"),
         target_mean=("ed_cost_next3y_usd","mean"),
         prior_mean=("prior_ed_cost_5y_usd","mean"),
         share_severe_mean=("share_severe_proc","mean"),
         share_crit_mean=("share_crit_care","mean"))
    .reset_index()
    .sort_values(["primary_chronic","insurance","critical_any"], ascending=[True,
)
grp.to_csv(OUT_DIR / "ratio_group_summary.csv", index=False)
print("\n=== RATIO GROUP SUMMARY (head 30) ===")


# --- PAGE 78 ---

print(grp.head(30).to_string(index=False))
# Simple plots
plt.figure(figsize=(7,4))
plt.hist(np.log1p(train_aug["ratio_future_to_past"].dropna()), bins=60)
plt.title("log1p(future/past ratio) distribution (train)")
plt.tight_layout()
plt.savefig(OUT_DIR / "hist_log_ratio_future_to_past.png")
plt.close()
plt.figure(figsize=(7,4))
box_df = train_aug.dropna(subset=["ratio_future_to_past"])
box_df = box_df[box_df["ratio_future_to_past"] > 0]
box_df.boxplot(column="ratio_future_to_past", by="primary_chronic")
plt.suptitle("")
plt.title("Future/Past ratio by primary_chronic (train)")
plt.ylabel("future/past ratio")
plt.tight_layout()
plt.savefig(OUT_DIR / "box_ratio_by_chronic.png")
plt.close()
# -----------------------------
# Adjusted uplift analysis (controls for confounding)
# We compute mean(target|code present) - mean(target|code absent) under:
#  - unadjusted
#  - stratified by chronic
#  - stratified by prior_cost decile
#  - stratified by chronic + cost decile
# -----------------------------
codes_to_test = [
]
# presence matrix (patient-level)
presence = li[li["patient_id"].isin(train_full["patient_id"])][["patient_id","cod
presence["present"] = 1
wide = presence.pivot_table(index="patient_id", columns="code", values="present",


# --- PAGE 79 ---

wide = wide.reindex(columns=sorted(set(codes_to_test)), fill_value=0).reset_index
df = train_full.merge(wide, on="patient_id", how="left").fillna({c:0 for c in cod
df["prior_cost_decile"] = pd.qcut(df["prior_ed_cost_5y_usd"], 10, labels=False, d
def weighted_stratified_diff(df_in: pd.DataFrame, present_col: str, strata: list)
    # weighted by stratum size; skips strata where one arm missing
    diffs = []
    weights = []
    for _, g in df_in.groupby(strata, dropna=False):
        n = g.shape[0]
        if n < 10:
        g1 = g[g[present_col] == 1]
        g0 = g[g[present_col] == 0]
        if g1.empty or g0.empty:
        diffs.append(g1["ed_cost_next3y_usd"].mean() - g0["ed_cost_next3y_usd"].m
        weights.append(n)
    if not diffs:
        return np.nan
    return float(np.average(diffs, weights=weights))
rows = []
for code in codes_to_test:
    col = code
    n_with = int(df[col].sum())
    n_total = int(df.shape[0])
    unadj = df.loc[df[col]==1, "ed_cost_next3y_usd"].mean() - df.loc[df[col]==0,
    adj_chronic = weighted_stratified_diff(df, col, ["primary_chronic"])
    adj_costdec = weighted_stratified_diff(df, col, ["prior_cost_decile"])
    adj_both = weighted_stratified_diff(df, col, ["primary_chronic","prior_cost_d
    rows.append({
        "code": code,
        "n_with": n_with,
        "pct_with": n_with / n_total,
        "diff_unadjusted": float(unadj),
        "diff_adj_chronic": adj_chronic,


# --- PAGE 80 ---

        "diff_adj_costdecile": adj_costdec,
        "diff_adj_chronic_costdecile": adj_both
    })
uplift_adj = pd.DataFrame(rows).sort_values("diff_adj_chronic_costdecile", ascend
uplift_adj.to_csv(OUT_DIR / "code_uplift_adjusted.csv", index=False)
print("\n=== ADJUSTED UPLIFT (sorted by chronic+cost-decile adjusted diff) ===")
print(uplift_adj.to_string(index=False))
# -----------------------------
# Patient-level "price index" (unit_price / code median) weighted by line_total
# -----------------------------
code_median_unit = li.groupby("code", dropna=False)["unit_price"].median()
li2 = li.copy()
li2["code_unit_median"] = li2["code"].map(code_median_unit)
li2["unit_price_ratio"] = li2["unit_price"] / li2["code_unit_median"].replace(0,
li2["unit_price_ratio"] = li2["unit_price_ratio"].replace([np.inf, -np.inf], np.n
# Weighted average by line_total
w = li2["line_total"].fillna(0)
li2["w_ratio"] = li2["unit_price_ratio"] * w
price_index = (
    li2.groupby("patient_id", observed=True)
       .agg(price_index=("w_ratio","sum"), wsum=("line_total","sum"))
       .reset_index()
)
price_index["price_index"] = price_index["price_index"] / price_index["wsum"].rep
price_index = price_index[["patient_id","price_index"]]
train_pi = train_full.merge(price_index, on="patient_id", how="left")
pi_grp = None
if "insurance" in train_pi.columns:
    pi_grp = train_pi.groupby("insurance", dropna=False)["price_index"].describe(
    pi_grp.to_csv(OUT_DIR / "price_index_by_insurance.csv", index=False)
print("\n=== PRICE INDEX CORRELATIONS (TRAIN) ===")
tmp = train_pi[["price_index","ed_cost_next3y_usd","prior_ed_cost_5y_usd"]].dropn


# --- PAGE 81 ---

6) What I need from you after running Phase 1.5 (so I can refine
strategy)
```

### C04 — Code cell starting at page 91
```python
from pathlib import Path
import re
import numpy as np
import pandas as pd
from joblib import load
import matplotlib.pyplot as plt
# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = Path(r"D:\AgentDs\agent_ds_healthcare")
TRAIN_CSV = BASE_DIR / "ed_cost_train.csv"
PATIENTS_CSV = BASE_DIR / "patients.csv"
CACHE_DIR = BASE_DIR / "cache_iter10"
RECEIPT_CACHE = CACHE_DIR / "receipts_parsed.joblib"
OUT_DIR = CACHE_DIR / "eda_outputs_phase16"
OUT_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams["figure.dpi"] = 120
def standardize_zip3(z):
    if z is None or (isinstance(z, float) and np.isnan(z)):
        return None
    s = re.sub(r"\D", "", str(z).strip())
    if not s:
        return None
    return s.zfill(3)
# -----------------------------
# Load tabular
# -----------------------------
train = pd.read_csv(TRAIN_CSV)
patients = pd.read_csv(PATIENTS_CSV) if PATIENTS_CSV.exists() else None
train["patient_id"] = train["patient_id"].astype(int)
if patients is not None:
    patients["patient_id"] = patients["patient_id"].astype(int)


# --- PAGE 92 ---

    if "zip3" in patients.columns:
        patients["zip3"] = patients["zip3"].apply(standardize_zip3)
    if "insurance" in patients.columns:
        patients["insurance"] = patients["insurance"].astype(str).str.lower().str
train_full = train.merge(patients, on="patient_id", how="left", validate="one_to_
print("[info] train_full shape:", train_full.shape)
# -----------------------------
# Load cached receipts
# -----------------------------
obj = load(RECEIPT_CACHE)
if not (isinstance(obj, dict) and "lineitems_df" in obj):
    raise RuntimeError("Unexpected receipt cache format; expected dict with linei
li = obj["lineitems_df"].copy()
li["patient_id"] = li["patient_id"].astype(int)
li["code"] = li["code"].astype(str).str.upper().str.strip()
li["unit_price"] = pd.to_numeric(li["unit_price"], errors="coerce")
li["line_total"] = pd.to_numeric(li["line_total"], errors="coerce")
# Keep train patients only
train_ids = set(train_full["patient_id"].tolist())
li = li[li["patient_id"].isin(train_ids)].copy()
print("[info] train lineitems rows:", li.shape[0], " unique patients:", li["patie
# -----------------------------
# Build patient-code presence matrix for ALL codes with n_patients>=MIN_N
# -----------------------------
MIN_N = 30
presence = li[["patient_id","code"]].drop_duplicates()
code_counts = presence["code"].value_counts()
codes = code_counts[code_counts >= MIN_N].index.tolist()
print("[info] #codes with n_patients>=MIN_N:", len(codes))
P = (presence[presence["code"].isin(codes)]
     .assign(val=1)
     .pivot_table(index="patient_id", columns="code", values="val", aggfunc="max"
# Align with train_full patient order


# --- PAGE 93 ---

P = P.reindex(train_full["patient_id"].values, fill_value=0)
y = train_full["ed_cost_next3y_usd"].astype(float).values
# -----------------------------
# Stratification variables
# -----------------------------
meta = train_full[["primary_chronic","prior_ed_cost_5y_usd","insurance","zip3"]].
meta["prior_cost_decile"] = pd.qcut(meta["prior_ed_cost_5y_usd"], 10, labels=Fals
meta["stratum"] = meta["primary_chronic"].astype(str) + "_D" + meta["prior_cost_d
# -----------------------------
# Vectorized adjusted uplift:
# For each stratum: diff_j = mean(y|code_j=1) - mean(y|code_j=0)
# Then weighted-average diffs across strata (weights=stratum size)
# -----------------------------
def adjusted_uplift_vectorized(P_df: pd.DataFrame, y_vec: np.ndarray, strata: pd.
    X = P_df.values.astype(float)  # n x m
    m = X.shape[1]
    total_weight = np.zeros(m, dtype=float)
    total_diff = np.zeros(m, dtype=float)
    for s, idx in strata.groupby(strata).groups.items():
        idx = np.array(list(idx), dtype=int)
        if idx.size < 20:
        y_s = y_vec[idx]
        X_s = X[idx, :]
        n = idx.size
        n_with = X_s.sum(axis=0)                   # m
        sum_with = (X_s.T @ y_s)                   # m
        sum_total = y_s.sum()
        n_without = n - n_with
        valid = (n_with > 0) & (n_without > 0)
        mean_with = np.zeros(m); mean_without = np.zeros(m); diff = np.zeros(m)
        mean_with[valid] = sum_with[valid] / n_with[valid]


# --- PAGE 94 ---

        mean_without[valid] = (sum_total - sum_with[valid]) / n_without[valid]
        diff[valid] = mean_with[valid] - mean_without[valid]
        # weight by stratum size
        total_diff[valid] += diff[valid] * n
        total_weight[valid] += n
    adj = np.full(m, np.nan)
    ok = total_weight > 0
    adj[ok] = total_diff[ok] / total_weight[ok]
    return adj
# Unadjusted diff (whole dataset)
X_all = P.values.astype(float)
n_all = X_all.shape[0]
n_with_all = X_all.sum(axis=0)
sum_with_all = X_all.T @ y
sum_total_all = y.sum()
n_wo_all = n_all - n_with_all
valid_all = (n_with_all > 0) & (n_wo_all > 0)
mean_with_all = np.full(X_all.shape[1], np.nan)
mean_wo_all = np.full(X_all.shape[1], np.nan)
mean_with_all[valid_all] = sum_with_all[valid_all] / n_with_all[valid_all]
mean_wo_all[valid_all] = (sum_total_all - sum_with_all[valid_all]) / n_wo_all[val
diff_unadj = mean_with_all - mean_wo_all
diff_adj = adjusted_uplift_vectorized(P, y, meta["stratum"])
uplift_all = pd.DataFrame({
    "code": P.columns,
    "n_with": n_with_all.astype(int),
    "pct_with": n_with_all / n_all,
    "diff_unadjusted": diff_unadj,
    "diff_adj_chronic_costdecile": diff_adj
}).sort_values("diff_adj_chronic_costdecile", ascending=False)
uplift_all.to_csv(OUT_DIR / "uplift_all_codes_adj.csv", index=False)
print("\n=== TOP 25 codes by ADJUSTED uplift (chronic+cost-decile controlled) ===
print(uplift_all.head(25).to_string(index=False))


# --- PAGE 95 ---

print("\n=== TOP 25 NEGATIVE codes by ADJUSTED uplift ===")
print(uplift_all.tail(25).sort_values("diff_adj_chronic_costdecile", ascending=Tr
# -----------------------------
# Build clinically grounded bucket features (shares) + price index
# -----------------------------
ED_EM = {f"9928{i}" for i in range(1,6)}          # 99281-99285 (ED E/M family)
CRIT_CARE = {"99291","99292"}                    # critical care
SEVERE_PROC = {"31500","36556","36620","92950"}  # intubation, central line, arte
OBS = {"G0378"}
def bucket(code: str) -> str:
    if code in CRIT_CARE: return "CRIT_CARE"
    if code in SEVERE_PROC: return "SEVERE_PROC"
    if code in ED_EM: return "ED_EM"
    if code in OBS: return "OBS"
    if code.isdigit() and len(code)==5 and code.startswith("7"): return "IMAGING"
    if code.isdigit() and len(code)==5 and code.startswith("8"): return "LAB"
    return "OTHER"
li["bucket"] = li["code"].map(bucket)
# cost share by bucket per patient
pt = li.groupby(["patient_id","bucket"], observed=True)["line_total"].sum().unsta
pt["total_cost_receipt"] = pt.sum(axis=1)
for b in ["CRIT_CARE","SEVERE_PROC","ED_EM","LAB","IMAGING","OBS","OTHER"]:
    if b not in pt.columns:
        pt[b] = 0.0
    pt[f"share_{b.lower()}"] = (pt[b] / pt["total_cost_receipt"].replace(0, np.na
# price index: weighted average(unit_price / median_unit_price(code)) weighted by
code_median_unit = li.groupby("code")["unit_price"].median()
li2 = li.copy()
li2["code_median_unit"] = li2["code"].map(code_median_unit)
li2["ratio"] = li2["unit_price"] / li2["code_median_unit"].replace(0, np.nan)
li2["ratio"] = li2["ratio"].replace([np.inf, -np.inf], np.nan)
li2["w_ratio"] = li2["ratio"] * li2["line_total"].fillna(0)
pi = li2.groupby("patient_id", observed=True).agg(w_ratio=("w_ratio","sum"), wsum


# --- PAGE 96 ---

pi["price_index"] = pi["w_ratio"] / pi["wsum"].replace(0, np.nan)
pi = pi[["price_index"]]
feat = pt[[c for c in pt.columns if c.startswith("share_")]].join(pi, how="left")
train_feat = train_full.merge(feat, on="patient_id", how="left").fillna(0)
# -----------------------------
# Residual association: remove chronic + prior-cost decile baseline
# (pure EDA; no model training)
# -----------------------------
baseline = (train_feat
            .groupby(["primary_chronic", pd.qcut(train_feat["prior_ed_cost_5y_usd
            ["ed_cost_next3y_usd"].mean())
tmp = train_feat.copy()
tmp["prior_cost_decile"] = pd.qcut(tmp["prior_ed_cost_5y_usd"], 10, labels=False,
tmp["baseline_mean"] = tmp.apply(lambda r: baseline.loc[(r["primary_chronic"], r[
tmp["residual"] = tmp["ed_cost_next3y_usd"] - tmp["baseline_mean"]
cols_check = ["residual","price_index","share_crit_care","share_severe_proc","sha
corr = tmp[cols_check].corr(method="spearman")["residual"].sort_values(key=lambda
print("\n=== Spearman correlation with RESIDUAL (after chronic+cost-decile baseli
print(corr.to_string())
# Correlation of residual with price_index within insurance (checks if PI adds wi
if "insurance" in tmp.columns:
    print("\n=== Residual vs price_index Spearman by insurance ===")
    for ins, g in tmp.groupby("insurance", dropna=False):
        g2 = g[["residual","price_index"]].dropna()
        if g2.shape[0] < 50:
        rho = g2.corr(method="spearman").iloc[0,1]
        print(f"{ins:>8}: n={g2.shape[0]:4d}  spearman(residual, price_index)={rh
# -----------------------------
# Outlier profiling: extreme future/past ratios
# -----------------------------
tmp["ratio_future_to_past"] = tmp["ed_cost_next3y_usd"] / tmp["prior_ed_cost_5y_u
tmp["ratio_future_to_past"] = tmp["ratio_future_to_past"].replace([np.inf, -np.in
out = tmp.sort_values("ratio_future_to_past", ascending=False).head(40)


# --- PAGE 97 ---

6) What to send me from Phase 1.6 (so we finalize the feature set
before modeling)
Paste back:
1. The printed:
2. The printed:
Spearman correlation with RESIDUAL (after chronic+prior-cost-decile baseline)
3. A glimpse (first ~10 rows) of:
That will tell us:
whether price_index remains predictive after baseline removal (the key validation),
and what drives the heavy-tail outliers (which dominate MAE).
keep = ["patient_id","primary_chronic","age","sex","insurance","zip3",
keep = [c for c in keep if c in out.columns]
out[keep].to_csv(OUT_DIR / "top_ratio_outliers_top40.csv", index=False)
print("\n=== Top 15 future/past ratio outliers (selected columns) ===")
print(out[keep].head(15).to_string(index=False))
# quick plot
plt.figure(figsize=(7,4))
plt.hist(np.log1p(tmp["ratio_future_to_past"].dropna()), bins=60)
plt.title("log1p(future/past ratio) (train)")
plt.tight_layout()
plt.savefig(OUT_DIR / "hist_log_ratio_future_to_past.png")
plt.close()
print(f"\n[DONE] Phase 1.6 outputs written to: {OUT_DIR}")


# --- PAGE 98 ---

7) Interim feature direction (still not modeling)
Based on what you already found, the strongest non-redundant receipt features are likely:
Critical history flags (presence of 99291/99292)
Severe procedure flags (31500, 36556, 36620, 92950)
Shares: share_crit_care , share_severe_proc  (composition beats totals)
Price environment: price_index  (validate residual correlation)
Routine phenotype markers: shares of ED_EM / LAB / IMAGING (often negative after
adjustment)
And we keep the tabular backbone ( primary_chronic , prior_ed_cost_5y_usd ,
prior_ed_visits_5y , plus age/sex/insurance/zip3) from the dataset spec.
Once Phase 1.6 confirms the residual signals, we’ll move to Phase 3 (modeling) with a
```

### C05 — Code cell starting at page 107
```python
from pathlib import Path
import re
import numpy as np
import pandas as pd
from joblib import load
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = Path(r"D:\AgentDs\agent_ds_healthcare")
TRAIN_CSV = BASE_DIR / "ed_cost_train.csv"
PATIENTS_CSV = BASE_DIR / "patients.csv"
CACHE_DIR = BASE_DIR / "cache_iter10"
RECEIPT_CACHE = CACHE_DIR / "receipts_parsed.joblib"
RANDOM_STATE = 42


# --- PAGE 108 ---

N_SPLITS = 5
def standardize_zip3(z):
    if z is None or (isinstance(z, float) and np.isnan(z)):
        return None
    s = re.sub(r"\D", "", str(z).strip())
    return s.zfill(3) if s else None
# -----------------------------
# Load tabular
# -----------------------------
train = pd.read_csv(TRAIN_CSV)
patients = pd.read_csv(PATIENTS_CSV) if PATIENTS_CSV.exists() else None
train["patient_id"] = train["patient_id"].astype(int)
if patients is not None:
    patients["patient_id"] = patients["patient_id"].astype(int)
    if "zip3" in patients.columns:
        patients["zip3"] = patients["zip3"].apply(standardize_zip3)
    if "insurance" in patients.columns:
        patients["insurance"] = patients["insurance"].astype(str).str.lower().str
    train = train.merge(patients, on="patient_id", how="left", validate="one_to_o
# -----------------------------
# Load cached receipts
# -----------------------------
obj = load(RECEIPT_CACHE)
if not (isinstance(obj, dict) and "lineitems_df" in obj):
    raise RuntimeError(f"Unexpected cache format in {RECEIPT_CACHE}")
li = obj["lineitems_df"].copy()
li["patient_id"] = li["patient_id"].astype(int)
li["code"] = li["code"].astype(str).str.upper().str.strip()
li["unit_price"] = pd.to_numeric(li["unit_price"], errors="coerce")
li["line_total"] = pd.to_numeric(li["line_total"], errors="coerce")
# Keep train patients only for feature build (no target leakage anyway, but clean
train_ids = set(train["patient_id"].tolist())
li = li[li["patient_id"].isin(train_ids)].copy()


# --- PAGE 109 ---

# -----------------------------
# Derive the global code list (should be 18)
# -----------------------------
codes = sorted(li["code"].unique().tolist())
print(f"[info] unique codes in train receipts: {len(codes)} -> {codes}")
# -----------------------------
# Per-patient code cost + shares + presence
# -----------------------------
cost_pc = li.groupby(["patient_id","code"], observed=True)["line_total"].sum().un
cost_pc = cost_pc.reindex(columns=codes, fill_value=0.0)
total_cost = cost_pc.sum(axis=1).replace(0, np.nan)
share_pc = cost_pc.div(total_cost, axis=0).fillna(0.0)
has_pc = (cost_pc > 0).astype(int)
# rename columns
cost_pc.columns = [f"cost_{c}" for c in cost_pc.columns]
share_pc.columns = [f"share_{c}" for c in share_pc.columns]
has_pc.columns  = [f"has_{c}" for c in has_pc.columns]
receipt_feat = pd.concat([has_pc, share_pc], axis=1).reset_index()
# -----------------------------
# Clinically grounded bundles (based on your EDA)
# -----------------------------
CRIT = ["99291","99292"]
SEV_PROC = ["31500","36556","36620","92950"]
LOW_EM = ["99281","99282","99283"]
HIGH_EM = ["99284","99285"]
BASIC_WORKUP = ["85025","71045","87070","84484"]  # labs/xray/culture/troponin
CT = ["70450","74177"]
OBS = ["G0378"]
def sum_cols(prefix, code_list):
    cols = [f"{prefix}_{c}" for c in code_list if f"{prefix}_{c}" in receipt_feat
    return receipt_feat[cols].sum(axis=1) if cols else 0.0
receipt_feat["has_crit"] = (sum_cols("has", CRIT) > 0).astype(int)


# --- PAGE 110 ---

receipt_feat["has_severe_proc"] = (sum_cols("has", SEV_PROC) > 0).astype(int)
receipt_feat["share_crit"] = sum_cols("share", CRIT)
receipt_feat["share_severe_proc"] = sum_cols("share", SEV_PROC)
receipt_feat["share_low_em"] = sum_cols("share", LOW_EM)
receipt_feat["share_high_em"] = sum_cols("share", HIGH_EM)
receipt_feat["share_basic_workup"] = sum_cols("share", BASIC_WORKUP)
receipt_feat["share_ct"] = sum_cols("share", CT)
receipt_feat["share_obs"] = sum_cols("share", OBS)
# -----------------------------
# Price index (weighted avg of unit_price / median_unit_price(code))
# -----------------------------
code_median_unit = li.groupby("code", observed=True)["unit_price"].median()
li2 = li.copy()
li2["code_median_unit"] = li2["code"].map(code_median_unit)
li2["unit_ratio"] = li2["unit_price"] / li2["code_median_unit"].replace(0, np.nan
li2["unit_ratio"] = li2["unit_ratio"].replace([np.inf, -np.inf], np.nan)
li2["w_ratio"] = li2["unit_ratio"] * li2["line_total"].fillna(0.0)
pi = li2.groupby("patient_id", observed=True).agg(w_ratio=("w_ratio","sum"), wsum
pi["price_index"] = pi["w_ratio"] / pi["wsum"].replace(0, np.nan)
pi = pi[["price_index"]].reset_index()
receipt_feat = receipt_feat.merge(pi, on="patient_id", how="left")
receipt_feat["price_index"] = receipt_feat["price_index"].fillna(1.0)
receipt_feat["log_price_index"] = np.log(receipt_feat["price_index"].astype(float
# -----------------------------
# Merge into train
# -----------------------------
df = train.merge(receipt_feat, on="patient_id", how="left")
y = df["ed_cost_next3y_usd"].astype(float).values
# tabular derived features
df["log_prior_cost"] = np.log1p(df["prior_ed_cost_5y_usd"].astype(float))
df["log_prior_visits"] = np.log1p(df["prior_ed_visits_5y"].astype(float))
df["prior_cost_per_visit"] = df["prior_ed_cost_5y_usd"].astype(float) / (df["prio
df["prior_visits_zero"] = (df["prior_ed_visits_5y"] == 0).astype(int)
df["prior_cost_floor"] = (df["prior_ed_cost_5y_usd"] <= 50).astype(int)


# --- PAGE 111 ---

# interactions capturing your outlier pattern
df["floor_x_crit"] = df["prior_cost_floor"] * df["has_crit"]
df["floor_x_severe"] = df["prior_cost_floor"] * df["has_severe_proc"]
df["vis0_x_crit"] = df["prior_visits_zero"] * df["has_crit"]
df["vis0_x_severe"] = df["prior_visits_zero"] * df["has_severe_proc"]
# -----------------------------
# Feature sets for ablation
# -----------------------------
base_num = ["age","prior_ed_visits_5y","prior_ed_cost_5y_usd","log_prior_cost","l
            "prior_visits_zero","prior_cost_floor"]
base_cat = ["primary_chronic","sex","insurance","zip3"]
# Keep only columns that exist (robustness)
base_num = [c for c in base_num if c in df.columns]
base_cat = [c for c in base_cat if c in df.columns]
# receipt presence only
receipt_presence_cols = [c for c in df.columns if c.startswith("has_") and c not
receipt_presence_cols += ["has_crit","has_severe_proc"]
receipt_presence_cols = [c for c in receipt_presence_cols if c in df.columns]
# full receipt features
receipt_full_cols = []
receipt_full_cols += receipt_presence_cols
receipt_full_cols += [c for c in df.columns if c.startswith("share_")]
receipt_full_cols += ["price_index","log_price_index","floor_x_crit","floor_x_sev
receipt_full_cols = [c for c in receipt_full_cols if c in df.columns]
FEATURESETS = {
    "A_tabular_only": base_num + base_cat,
    "B_tabular_plus_code_presence": base_num + base_cat + receipt_presence_cols,
    "C_tabular_plus_full_receipt": base_num + base_cat + receipt_full_cols,
}
# -----------------------------
# Model choice (LightGBM if available; else sklearn fallback)
# -----------------------------
MODEL_NAME = None


# --- PAGE 112 ---

model = None
try:
    import lightgbm as lgb
    MODEL_NAME = "LightGBM (objective=regression_l1)"
    model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=31,
        min_data_in_leaf=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
except Exception:
    from sklearn.ensemble import HistGradientBoostingRegressor
    MODEL_NAME = "HistGradientBoostingRegressor (squared_error fallback if needed
    # Try absolute_error if supported; otherwise squared_error (still OK for abla
    try:
        model = HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.05,
            max_depth=6,
            max_iter=600,
            random_state=RANDOM_STATE
        )
        MODEL_NAME = "HistGradientBoostingRegressor (loss=absolute_error)"
    except Exception:
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_depth=6,
            max_iter=600,
            random_state=RANDOM_STATE
        )
print(f"[info] Using model: {MODEL_NAME}")


# --- PAGE 113 ---

# -----------------------------
# Preprocessor (one-hot categoricals)
# -----------------------------
numeric_features = None
categorical_features = None
# -----------------------------
# CV evaluation
# -----------------------------
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
strata = df["primary_chronic"].astype(str).values
# outlier subset for diagnostics
outlier_mask = (df["prior_ed_cost_5y_usd"] <= 50) & (df["prior_ed_visits_5y"] ==
def run_cv(feature_cols):
    X = df[feature_cols].copy()
    num_cols = [c for c in feature_cols if c in base_num or c.startswith(("has_",
    cat_cols = [c for c in feature_cols if c in base_cat]
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imp", SimpleImputer(strategy="median")),
            ]), num_cols),
            ("cat", Pipeline(steps=[
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False)),
            ]), cat_cols),
        ],
        remainder="drop"
    )
    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    maes = []
    maes_out = []
    for fold, (tr, va) in enumerate(skf.split(X, strata), 1):


# --- PAGE 114 ---
```

### C06 — Code cell starting at page 118
```python
from pathlib import Path
import re
import numpy as np
import pandas as pd
from joblib import load
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = Path(r"D:\AgentDs\agent_ds_healthcare")
TRAIN_CSV = BASE_DIR / "ed_cost_train.csv"
PATIENTS_CSV = BASE_DIR / "patients.csv"
CACHE_DIR = BASE_DIR / "cache_iter10"
RECEIPT_CACHE = CACHE_DIR / "receipts_parsed.joblib"


# --- PAGE 119 ---

RANDOM_STATE = 42
N_SPLITS = 5
EPS = 1e-6
def standardize_zip3(z):
    if z is None or (isinstance(z, float) and np.isnan(z)):
        return None
    s = re.sub(r"\D", "", str(z).strip())
    return s.zfill(3) if s else None
# -----------------------------
# Load train + patients
# -----------------------------
train = pd.read_csv(TRAIN_CSV)
patients = pd.read_csv(PATIENTS_CSV) if PATIENTS_CSV.exists() else None
train["patient_id"] = train["patient_id"].astype(int)
if patients is not None:
    patients["patient_id"] = patients["patient_id"].astype(int)
    if "zip3" in patients.columns:
        patients["zip3"] = patients["zip3"].apply(standardize_zip3)
    if "insurance" in patients.columns:
        patients["insurance"] = patients["insurance"].astype(str).str.lower().str
    if "sex" in patients.columns:
        patients["sex"] = patients["sex"].astype(str).str.strip()
    train = train.merge(patients, on="patient_id", how="left", validate="one_to_o
# -----------------------------
# Load receipts cache
# -----------------------------
obj = load(RECEIPT_CACHE)
li = obj["lineitems_df"].copy()
li["patient_id"] = li["patient_id"].astype(int)
li["code"] = li["code"].astype(str).str.upper().str.strip()
li["unit_price"] = pd.to_numeric(li["unit_price"], errors="coerce")
li["line_total"] = pd.to_numeric(li["line_total"], errors="coerce")
# keep train patients
li = li[li["patient_id"].isin(set(train["patient_id"]))].copy()


# --- PAGE 120 ---

# code universe
codes = sorted(li["code"].unique().tolist())
print(f"[info] unique receipt codes (train): {len(codes)} -> {codes}")
# -----------------------------
# Receipt features: has_*, share_*, price_index
# -----------------------------
cost_pc = li.groupby(["patient_id","code"], observed=True)["line_total"].sum().un
cost_pc = cost_pc.reindex(columns=codes, fill_value=0.0)
total_cost = cost_pc.sum(axis=1).replace(0, np.nan)
share_pc = cost_pc.div(total_cost, axis=0).fillna(0.0)
has_pc = (cost_pc > 0).astype(int)
cost_pc.columns = [f"cost_{c}" for c in cost_pc.columns]
share_pc.columns = [f"share_{c}" for c in share_pc.columns]
has_pc.columns  = [f"has_{c}" for c in has_pc.columns]
receipt_feat = pd.concat([has_pc, share_pc], axis=1).reset_index()
# bundles
ED_EM = [f"9928{i}" for i in range(1,6)]
CRIT = ["99291","99292"]
SEV_PROC = ["31500","36556","36620","92950"]
def safe_sum(prefix, clist):
    cols = [f"{prefix}_{c}" for c in clist if f"{prefix}_{c}" in receipt_feat.col
    if not cols:
        return 0.0
    return receipt_feat[cols].sum(axis=1)
receipt_feat["has_any_em"] = (safe_sum("has", ED_EM) > 0).astype(int)
receipt_feat["has_crit"] = (safe_sum("has", CRIT) > 0).astype(int)
receipt_feat["has_severe_proc"] = (safe_sum("has", SEV_PROC) > 0).astype(int)
receipt_feat["crit_no_em"] = ((receipt_feat["has_any_em"] == 0) & ((receipt_feat[
receipt_feat["share_crit"] = safe_sum("share", CRIT)
receipt_feat["share_severe_proc"] = safe_sum("share", SEV_PROC)


# --- PAGE 121 ---

# price index: weighted avg (unit_price / median_unit_price(code)) weighted by li
code_median_unit = li.groupby("code", observed=True)["unit_price"].median()
li2 = li.copy()
li2["code_median_unit"] = li2["code"].map(code_median_unit)
li2["unit_ratio"] = li2["unit_price"] / li2["code_median_unit"].replace(0, np.nan
li2["unit_ratio"] = li2["unit_ratio"].replace([np.inf, -np.inf], np.nan)
li2["w_ratio"] = li2["unit_ratio"] * li2["line_total"].fillna(0.0)
pi = li2.groupby("patient_id", observed=True).agg(w_ratio=("w_ratio","sum"), wsum
pi["price_index"] = pi["w_ratio"] / pi["wsum"].replace(0, np.nan)
pi = pi[["price_index"]].reset_index()
receipt_feat = receipt_feat.merge(pi, on="patient_id", how="left")
receipt_feat["price_index"] = receipt_feat["price_index"].fillna(1.0)
receipt_feat["log_price_index"] = np.log(receipt_feat["price_index"].astype(float
# -----------------------------
# Merge all
# -----------------------------
df = train.merge(receipt_feat, on="patient_id", how="left").fillna(0.0)
# core engineered features
df["log_prior_cost"] = np.log1p(df["prior_ed_cost_5y_usd"].astype(float))
df["log_prior_visits"] = np.log1p(df["prior_ed_visits_5y"].astype(float))
df["prior_cost_per_visit"] = df["prior_ed_cost_5y_usd"].astype(float) / (df["prio
df["prior_visits_zero"] = (df["prior_ed_visits_5y"] == 0).astype(int)
df["prior_cost_floor"] = (df["prior_ed_cost_5y_usd"] <= 50).astype(int)
# NEW: price-deflated prior cost (intensity proxy)
df["prior_cost_deflated"] = df["prior_ed_cost_5y_usd"].astype(float) / (df["price_
df["log_prior_cost_deflated"] = np.log1p(df["prior_cost_deflated"])
df["deflated_cost_per_visit"] = df["prior_cost_deflated"] / (df["prior_ed_visits_
# helpful interactions for the weird floor+critical phenotype
df["floor_x_crit"] = df["prior_cost_floor"] * df["has_crit"]
df["floor_x_severe"] = df["prior_cost_floor"] * df["has_severe_proc"]
df["floor_x_crit_no_em"] = df["prior_cost_floor"] * df["crit_no_em"]
# categorical cleanup


# --- PAGE 122 ---

for c in ["primary_chronic","sex","insurance","zip3"]:
    if c in df.columns:
        df[c] = df[c].astype(str).replace({"nan":"missing", "None":"missing"}).fi
        df[c] = df[c].astype("category")
# bins for baseline strata
df["prior_cost_decile"] = pd.qcut(df["prior_ed_cost_5y_usd"], 10, labels=False, d
df["prior_cost_decile"] = df["prior_cost_decile"].fillna("missing").astype("categ
y = df["ed_cost_next3y_usd"].astype(float).values
strata = df["primary_chronic"].astype(str).values
outlier_mask = (df["prior_ed_cost_5y_usd"] <= 50) & (df["prior_ed_visits_5y"] ==
# -----------------------------
# Baseline median with backoff
# -----------------------------
def median_backoff_predict(df_tr, y_tr, df_va):
    tr = df_tr.copy()
    tr["_y"] = y_tr
    levels = [
        ["primary_chronic","insurance","zip3","prior_cost_decile"],
        ["primary_chronic","insurance","prior_cost_decile"],
        ["primary_chronic","insurance"],
        ["primary_chronic"],
    ]
    maps = []
    for cols in levels:
        cols = [c for c in cols if c in tr.columns and c in df_va.columns]
        med = tr.groupby(cols, dropna=False)["_y"].median()
        maps.append((cols, med))
    overall = float(np.median(y_tr))
    def map_level(cols, med_series, dfX):
        idx = pd.MultiIndex.from_frame(dfX[cols])
        return pd.Series(med_series.reindex(idx).to_numpy(), index=dfX.index)


# --- PAGE 123 ---

    pred = pd.Series(np.nan, index=df_va.index, dtype=float)
    for cols, med in maps:
        if len(cols) == 0:
        pred = pred.fillna(map_level(cols, med, df_va))
    pred = pred.fillna(overall)
    return pred.values
# -----------------------------
# Models
# -----------------------------
def make_lgbm():
    return lgb.LGBMRegressor(
        objective="quantile",  # median when alpha=0.5
        alpha=0.5,
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=63,
        min_data_in_leaf=20,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        force_col_wise=True,
        verbose=-1,
    )
# Feature sets: with vs without deflated prior cost
BASE_NUM = [
]
DEFLATED = ["prior_cost_deflated","log_prior_cost_deflated","deflated_cost_per_vi
# include all has_*/share_* code features (tiny universe, ok)


# --- PAGE 124 ---

ALL_CODE_HAS = [c for c in df.columns if c.startswith("has_") and c not in ["has_
ALL_CODE_SHARE = [c for c in df.columns if c.startswith("share_") and c not in ["
CAT = [c for c in ["primary_chronic","sex","insurance","zip3","prior_cost_decile"
FEATURES_NO_DEF = [c for c in (BASE_NUM + ALL_CODE_HAS + ALL_CODE_SHARE + CAT) if
FEATURES_WITH_DEF = [c for c in (BASE_NUM + DEFLATED + ALL_CODE_HAS + ALL_CODE_SHA
print(f"[info] n_features(no_deflated)={len(FEATURES_NO_DEF)}  n_features(with_de
# -----------------------------
# CV runner
# -----------------------------
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
def cv_eval(feature_cols, use_residual=False):
    maes = []
    maes_out = []
    out_counts = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(df, strata), 1):
        dtr = df.iloc[tr_idx].copy()
        dva = df.iloc[va_idx].copy()
        ytr = y[tr_idx]
        yva = y[va_idx]
        # baseline medians
        base_va = median_backoff_predict(dtr, ytr, dva)
        base_tr = median_backoff_predict(dtr, ytr, dtr)  # OK for training residu
        if use_residual:
            target_tr = ytr - base_tr
        else:
            target_tr = ytr
        Xtr = dtr[feature_cols]
        Xva = dva[feature_cols]
        # LightGBM expects categoricals as pandas 'category'
        model = make_lgbm()


# --- PAGE 125 ---

        model.fit(Xtr, target_tr)
        pred = model.predict(Xva)
        if use_residual:
            pred = pred + base_va
        mae = mean_absolute_error(yva, pred)
        maes.append(mae)
        m_out = outlier_mask.iloc[va_idx].values
        out_counts.append(int(m_out.sum()))
        if m_out.sum() > 0:
            maes_out.append(mean_absolute_error(yva[m_out], pred[m_out]))
        else:
            maes_out.append(np.nan)
        print(f"  fold {fold}: MAE={mae:,.3f} | out_n={out_counts[-1]:3d} | out_MA
    return (float(np.mean(maes)), float(np.std(maes)), float(np.nanmean(maes_out)
def baseline_only():
    maes = []
    maes_out = []
    out_counts = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(df, strata), 1):
        dtr = df.iloc[tr_idx]
        dva = df.iloc[va_idx]
        ytr = y[tr_idx]
        yva = y[va_idx]
        pred = median_backoff_predict(dtr, ytr, dva)
        mae = mean_absolute_error(yva, pred)
        maes.append(mae)
        m_out = outlier_mask.iloc[va_idx].values
        out_counts.append(int(m_out.sum()))
        if m_out.sum() > 0:
            maes_out.append(mean_absolute_error(yva[m_out], pred[m_out]))
        else:
            maes_out.append(np.nan)


# --- PAGE 126 ---
```

### C07 — Code cell starting at page 130
```python
from pathlib import Path
import re
import numpy as np
import pandas as pd
from joblib import load
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = Path(r"D:\AgentDs\agent_ds_healthcare")
TRAIN_CSV = BASE_DIR / "ed_cost_train.csv"


# --- PAGE 131 ---

PATIENTS_CSV = BASE_DIR / "patients.csv"
CACHE_DIR = BASE_DIR / "cache_iter10"
RECEIPT_CACHE = CACHE_DIR / "receipts_parsed.joblib"
RANDOM_STATE = 42
N_SPLITS = 5
EPS = 1e-6
def standardize_zip3(z):
    if z is None or (isinstance(z, float) and np.isnan(z)):
        return None
    s = re.sub(r"\D", "", str(z).strip())
    return s.zfill(3) if s else None
# -----------------------------
# Load train + patients
# -----------------------------
train = pd.read_csv(TRAIN_CSV)
patients = pd.read_csv(PATIENTS_CSV) if PATIENTS_CSV.exists() else None
train["patient_id"] = train["patient_id"].astype(int)
if patients is not None:
    patients["patient_id"] = patients["patient_id"].astype(int)
    if "zip3" in patients.columns:
        patients["zip3"] = patients["zip3"].apply(standardize_zip3)
    if "insurance" in patients.columns:
        patients["insurance"] = patients["insurance"].astype(str).str.lower().str
    if "sex" in patients.columns:
        patients["sex"] = patients["sex"].astype(str).str.strip()
    train = train.merge(patients, on="patient_id", how="left", validate="one_to_o
# -----------------------------
# Load receipts cache (train only)
# -----------------------------
obj = load(RECEIPT_CACHE)
li = obj["lineitems_df"].copy()
li["patient_id"] = li["patient_id"].astype(int)
li["code"] = li["code"].astype(str).str.upper().str.strip()
li["unit_price"] = pd.to_numeric(li["unit_price"], errors="coerce")
li["line_total"] = pd.to_numeric(li["line_total"], errors="coerce")


# --- PAGE 132 ---

li = li[li["patient_id"].isin(set(train["patient_id"]))].copy()
codes = sorted(li["code"].unique().tolist())
print(f"[info] unique receipt codes (train): {len(codes)} -> {codes}")
# -----------------------------
# Build per-patient: cost, share, has, cnt
# -----------------------------
cost_pc = li.groupby(["patient_id","code"], observed=True)["line_total"].sum().un
cnt_pc  = li.groupby(["patient_id","code"], observed=True).size().unstack(fill_va
cost_pc = cost_pc.reindex(columns=codes, fill_value=0.0)
cnt_pc  = cnt_pc.reindex(columns=codes, fill_value=0.0)
total_cost = cost_pc.sum(axis=1).replace(0, np.nan)
share_pc = cost_pc.div(total_cost, axis=0).fillna(0.0)
has_pc = (cnt_pc > 0).astype(float)
cost_pc.columns  = [f"cost_{c}" for c in cost_pc.columns]
share_pc.columns = [f"share_{c}" for c in share_pc.columns]
has_pc.columns   = [f"has_{c}" for c in has_pc.columns]
cnt_pc.columns   = [f"cnt_{c}" for c in cnt_pc.columns]
feat = pd.concat([cost_pc, share_pc, has_pc, cnt_pc], axis=1).reset_index()
# -----------------------------
# Price index (weighted avg of unit_price / median_unit_price(code))
# -----------------------------
code_median_unit = li.groupby("code", observed=True)["unit_price"].median()
li2 = li.copy()
li2["code_median_unit"] = li2["code"].map(code_median_unit)
li2["unit_ratio"] = li2["unit_price"] / li2["code_median_unit"].replace(0, np.nan
li2["unit_ratio"] = li2["unit_ratio"].replace([np.inf, -np.inf], np.nan)
li2["w_ratio"] = li2["unit_ratio"] * li2["line_total"].fillna(0.0)
pi = li2.groupby("patient_id", observed=True).agg(w_ratio=("w_ratio","sum"), wsum
pi["price_index"] = pi["w_ratio"] / pi["wsum"].replace(0, np.nan)
pi = pi[["price_index"]].reset_index()
feat = feat.merge(pi, on="patient_id", how="left")
feat["price_index"] = feat["price_index"].fillna(1.0)


# --- PAGE 133 ---

feat["log_price_index"] = np.log(feat["price_index"].astype(float) + EPS)
# -----------------------------
# Global receipt summary features
# -----------------------------
# counts
feat["n_line_items"] = cnt_pc.sum(axis=1).values
feat["n_unique_codes"] = (cnt_pc > 0).sum(axis=1).values
feat["total_receipt_cost_check"] = cost_pc.sum(axis=1).fillna(0.0).values
# ED E/M severity summaries
EM = ["99281","99282","99283","99284","99285"]
EM_LEVEL = {"99281":1, "99282":2, "99283":3, "99284":4, "99285":5}
em_cnt_cols = [f"cnt_{c}" for c in EM if f"cnt_{c}" in feat.columns]
feat["cnt_em_total"] = feat[em_cnt_cols].sum(axis=1) if em_cnt_cols else 0.0
# mean level (count-weighted)
num = 0.0
den = feat["cnt_em_total"].replace(0, np.nan)
for c, lvl in EM_LEVEL.items():
    col = f"cnt_{c}"
    if col in feat.columns:
        num = num + feat[col] * lvl
feat["em_level_mean"] = (num / den).fillna(0.0)
# max level
maxlvl = np.zeros(len(feat))
for c, lvl in EM_LEVEL.items():
    col = f"cnt_{c}"
    if col in feat.columns:
        maxlvl = np.maximum(maxlvl, (feat[col] > 0).astype(float) * lvl)
feat["em_level_max"] = maxlvl
# critical/severe bundles (counts + shares)
CRIT = ["99291","99292"]
SEV_PROC = ["31500","36556","36620","92950"]
crit_cnt_cols = [f"cnt_{c}" for c in CRIT if f"cnt_{c}" in feat.columns]
sev_cnt_cols  = [f"cnt_{c}" for c in SEV_PROC if f"cnt_{c}" in feat.columns]
crit_share_cols = [f"share_{c}" for c in CRIT if f"share_{c}" in feat.columns]
sev_share_cols  = [f"share_{c}" for c in SEV_PROC if f"share_{c}" in feat.columns
feat["cnt_crit_total"] = feat[crit_cnt_cols].sum(axis=1) if crit_cnt_cols else 0.
feat["cnt_severe_proc_total"] = feat[sev_cnt_cols].sum(axis=1) if sev_cnt_cols el


# --- PAGE 134 ---

feat["share_crit"] = feat[crit_share_cols].sum(axis=1) if crit_share_cols else 0.
feat["share_severe_proc"] = feat[sev_share_cols].sum(axis=1) if sev_share_cols el
feat["has_crit"] = (feat["cnt_crit_total"] > 0).astype(float)
feat["has_severe_proc"] = (feat["cnt_severe_proc_total"] > 0).astype(float)
# -----------------------------
# Merge features into training frame
# -----------------------------
df = train.merge(feat, on="patient_id", how="left")
# Fill numeric receipt features
receipt_num_cols = [c for c in df.columns if c.startswith(("cost_","share_","has_
df[receipt_num_cols] = df[receipt_num_cols].fillna(0.0)
# Tabular derived features
df["log_prior_cost"] = np.log1p(df["prior_ed_cost_5y_usd"].astype(float))
df["log_prior_visits"] = np.log1p(df["prior_ed_visits_5y"].astype(float))
df["prior_cost_per_visit"] = df["prior_ed_cost_5y_usd"].astype(float) / (df["prio
df["prior_visits_zero"] = (df["prior_ed_visits_5y"] == 0).astype(int)
df["prior_cost_floor"] = (df["prior_ed_cost_5y_usd"] <= 50).astype(int)
# Price-deflated prior cost
df["prior_cost_deflated"] = df["prior_ed_cost_5y_usd"].astype(float) / (df["price_
df["log_prior_cost_deflated"] = np.log1p(df["prior_cost_deflated"])
df["deflated_cost_per_visit"] = df["prior_cost_deflated"] / (df["prior_ed_visits_
# interactions (your outlier pattern)
df["floor_x_crit"] = df["prior_cost_floor"] * df["has_crit"]
df["floor_x_severe"] = df["prior_cost_floor"] * df["has_severe_proc"]
# categorical cleanup
for c in ["primary_chronic","sex","insurance","zip3"]:
    if c in df.columns:
        df[c] = df[c].astype(str).replace({"nan":"missing","None":"missing"}).fil
# decile (for baseline feature)
df["prior_cost_decile"] = pd.qcut(df["prior_ed_cost_5y_usd"], 10, labels=False, d
df["prior_cost_decile"] = df["prior_cost_decile"].fillna("missing").astype("categ
y = df["ed_cost_next3y_usd"].astype(float).values


# --- PAGE 135 ---

strata = df["primary_chronic"].astype(str).values
outlier_mask = (df["prior_ed_cost_5y_usd"] <= 50) & (df["prior_ed_visits_5y"] ==
# -----------------------------
# Baseline median feature (computed per fold)
# -----------------------------
BASE_GROUP = ["primary_chronic","insurance","prior_cost_decile"]
def add_baseline_feature(dtr, ytr, dva):
    tr = dtr.copy()
    tr["_y"] = ytr
    med = tr.groupby(BASE_GROUP, dropna=False)["_y"].median()
    overall = float(np.median(ytr))
    def map_med(dX):
        idx = pd.MultiIndex.from_frame(dX[BASE_GROUP])
        out = med.reindex(idx).to_numpy()
        out = np.where(np.isnan(out), overall, out)
        return out
    dtr = dtr.copy()
    dva = dva.copy()
    dtr["baseline_med"] = map_med(dtr)
    dva["baseline_med"] = map_med(dva)
    return dtr, dva
# -----------------------------
# LightGBM
# -----------------------------
def make_model(seed):
    return lgb.LGBMRegressor(
        objective="quantile",
        alpha=0.5,
        n_estimators=2200,
        learning_rate=0.03,
        num_leaves=63,
        min_data_in_leaf=20,
        subsample=0.9,
        colsample_bytree=0.9,


# --- PAGE 136 ---

        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        force_col_wise=True,
        verbose=-1,
    )
# -----------------------------
# Feature sets
# -----------------------------
tab_num = ["age","prior_ed_visits_5y","prior_ed_cost_5y_usd","log_prior_cost","lo
           "floor_x_crit","floor_x_severe"]
tab_cat = [c for c in ["primary_chronic","sex","insurance","zip3","prior_cost_dec
# V1: shares + has only (plus deflated + tabular)
v1 = tab_num + tab_cat + [c for c in df.columns if c.startswith("share_")] + [c f
# V2: V1 + counts + EM summaries
v2 = v1 + [c for c in df.columns if c.startswith("cnt_")] + ["n_line_items","n_un
# V3: V2 + per-code costs
v3 = v2 + [c for c in df.columns if c.startswith("cost_")]
# V4: V3 + baseline median feature
v4 = v3 + ["baseline_med"]
# de-dupe and keep only existing columns
def clean(cols):
    cols2 = []
    seen = set()
    for c in cols:
        if c in df.columns and c not in seen:
            cols2.append(c); seen.add(c)
    return cols2


# --- PAGE 137 ---

V = {
    "V1_current_style": clean(v1),
    "V2_plus_counts": clean(v2),
    "V3_plus_counts_costs": clean(v3),
    "V4_plus_counts_costs_baseline": clean(v4),
}
print({k: len(v) for k,v in V.items()})
# -----------------------------
# CV evaluation
# -----------------------------
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
def cv_run(name, feat_cols, seeds=(42,), use_baseline_feat=False):
    maes = []
    out_maes = []
    out_counts = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(df, strata), 1):
        dtr = df.iloc[tr_idx].copy()
        dva = df.iloc[va_idx].copy()
        ytr = y[tr_idx]
        yva = y[va_idx]
        if use_baseline_feat:
            dtr, dva = add_baseline_feature(dtr, ytr, dva)
        Xtr = dtr[feat_cols]
        Xva = dva[feat_cols]
        # ensemble over seeds
        preds = np.zeros(len(dva), dtype=float)
        for sd in seeds:
            m = make_model(sd)
            m.fit(Xtr, ytr)
            preds += m.predict(Xva)
        preds /= float(len(seeds))
        mae = mean_absolute_error(yva, preds)


# --- PAGE 138 ---
```

### C08 — Code cell starting at page 141
```python
# --- PAGE 142 ---

from pathlib import Path
import re
import numpy as np
import pandas as pd
from joblib import load
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
# -----------------------------
# PATHS / SETTINGS
# -----------------------------
BASE_DIR = Path(r"D:\AgentDs\agent_ds_healthcare")
TRAIN_CSV = BASE_DIR / "ed_cost_train.csv"
PATIENTS_CSV = BASE_DIR / "patients.csv"
CACHE_DIR = BASE_DIR / "cache_iter10"
RECEIPT_CACHE = CACHE_DIR / "receipts_parsed.joblib"
RANDOM_STATE = 42
N_SPLITS = 5
INNER_SPLITS = 5
EPS = 1e-6
def standardize_zip3(z):
    if z is None or (isinstance(z, float) and np.isnan(z)):
        return None
    s = re.sub(r"\D", "", str(z).strip())
    return s.zfill(3) if s else None
# -----------------------------
# Load train + patients
# -----------------------------
train = pd.read_csv(TRAIN_CSV)
patients = pd.read_csv(PATIENTS_CSV) if PATIENTS_CSV.exists() else None
train["patient_id"] = train["patient_id"].astype(int)
if patients is not None:
    patients["patient_id"] = patients["patient_id"].astype(int)


# --- PAGE 143 ---

    if "zip3" in patients.columns:
        patients["zip3"] = patients["zip3"].apply(standardize_zip3)
    if "insurance" in patients.columns:
        patients["insurance"] = patients["insurance"].astype(str).str.lower().str
    if "sex" in patients.columns:
        patients["sex"] = patients["sex"].astype(str).str.strip()
    train = train.merge(patients, on="patient_id", how="left", validate="one_to_o
# -----------------------------
# Load receipts cache (train only)
# -----------------------------
obj = load(RECEIPT_CACHE)
li = obj["lineitems_df"].copy()
li["patient_id"] = li["patient_id"].astype(int)
li["code"] = li["code"].astype(str).str.upper().str.strip()
li["unit_price"] = pd.to_numeric(li["unit_price"], errors="coerce")
li["line_total"] = pd.to_numeric(li["line_total"], errors="coerce")
li = li[li["patient_id"].isin(set(train["patient_id"]))].copy()
codes = sorted(li["code"].unique().tolist())
print(f"[info] unique receipt codes (train): {len(codes)} -> {codes}")
# -----------------------------
# Build per-patient: share_*, has_*, cnt_* + summaries + price_index
# -----------------------------
cost_pc = li.groupby(["patient_id","code"], observed=True)["line_total"].sum().un
cnt_pc  = li.groupby(["patient_id","code"], observed=True).size().unstack(fill_va
cost_pc = cost_pc.reindex(columns=codes, fill_value=0.0)
cnt_pc  = cnt_pc.reindex(columns=codes, fill_value=0.0)
total_cost = cost_pc.sum(axis=1).replace(0, np.nan)
share_pc = cost_pc.div(total_cost, axis=0).fillna(0.0)
has_pc = (cnt_pc > 0).astype(float)
share_pc.columns = [f"share_{c}" for c in share_pc.columns]
has_pc.columns   = [f"has_{c}" for c in has_pc.columns]
cnt_pc.columns   = [f"cnt_{c}" for c in cnt_pc.columns]
feat = pd.concat([share_pc, has_pc, cnt_pc], axis=1).reset_index()


# --- PAGE 144 ---

# receipt counts summaries
feat["n_line_items"] = cnt_pc.sum(axis=1).values
feat["n_unique_codes"] = (cnt_pc > 0).sum(axis=1).values
# EM summaries
EM_LEVEL = {"99281":1, "99282":2, "99283":3, "99284":4, "99285":5}
em_cnt_cols = [f"cnt_{c}" for c in EM_LEVEL if f"cnt_{c}" in feat.columns]
feat["cnt_em_total"] = feat[em_cnt_cols].sum(axis=1) if em_cnt_cols else 0.0
den = feat["cnt_em_total"].replace(0, np.nan)
num = 0.0
maxlvl = np.zeros(len(feat))
for c, lvl in EM_LEVEL.items():
    col = f"cnt_{c}"
    if col in feat.columns:
        num = num + feat[col] * lvl
        maxlvl = np.maximum(maxlvl, (feat[col] > 0).astype(float) * lvl)
feat["em_level_mean"] = (num / den).fillna(0.0)
feat["em_level_max"] = maxlvl
# price index: weighted avg(unit_price / median_unit_price(code)) weighted by lin
code_median_unit = li.groupby("code", observed=True)["unit_price"].median()
li2 = li.copy()
li2["code_median_unit"] = li2["code"].map(code_median_unit)
li2["unit_ratio"] = li2["unit_price"] / li2["code_median_unit"].replace(0, np.nan
li2["unit_ratio"] = li2["unit_ratio"].replace([np.inf, -np.inf], np.nan)
li2["w_ratio"] = li2["unit_ratio"] * li2["line_total"].fillna(0.0)
pi = li2.groupby("patient_id", observed=True).agg(w_ratio=("w_ratio","sum"), wsum
pi["price_index"] = pi["w_ratio"] / pi["wsum"].replace(0, np.nan)
pi = pi[["price_index"]].reset_index()
feat = feat.merge(pi, on="patient_id", how="left")
feat["price_index"] = feat["price_index"].fillna(1.0)
feat["log_price_index"] = np.log(feat["price_index"].astype(float) + EPS)
# -----------------------------
# Merge into df
# -----------------------------


# --- PAGE 145 ---

df = train.merge(feat, on="patient_id", how="left")
df = df.fillna(0.0)
# tabular derived
df["log_prior_cost"] = np.log1p(df["prior_ed_cost_5y_usd"].astype(float))
df["log_prior_visits"] = np.log1p(df["prior_ed_visits_5y"].astype(float))
df["prior_cost_per_visit"] = df["prior_ed_cost_5y_usd"].astype(float) / (df["prio
df["prior_visits_zero"] = (df["prior_ed_visits_5y"] == 0).astype(int)
df["prior_cost_floor"] = (df["prior_ed_cost_5y_usd"] <= 50).astype(int)
# price-deflated prior cost
df["prior_cost_deflated"] = df["prior_ed_cost_5y_usd"].astype(float) / (df["price_
df["log_prior_cost_deflated"] = np.log1p(df["prior_cost_deflated"])
df["deflated_cost_per_visit"] = df["prior_cost_deflated"] / (df["prior_ed_visits_
# interactions for the "floor weirdos"
df["floor_x_lowprice"] = df["prior_cost_floor"] * (df["price_index"] < df["price_
# categoricals
for c in ["primary_chronic","sex","insurance","zip3"]:
    if c in df.columns:
        df[c] = df[c].astype(str).replace({"nan":"missing","None":"missing"}).fil
df["prior_cost_decile"] = pd.qcut(df["prior_ed_cost_5y_usd"], 10, labels=False, d
df["prior_cost_decile"] = df["prior_cost_decile"].fillna("missing").astype("categ
# lock category levels globally (avoid fold mismatch)
for c in ["primary_chronic","sex","insurance","zip3","prior_cost_decile"]:
    if c in df.columns and str(df[c].dtype) == "category":
        df[c] = df[c].cat.set_categories(df[c].cat.categories)
y = df["ed_cost_next3y_usd"].astype(float).values
strata = df["primary_chronic"].astype(str).values
outlier_mask = (df["prior_ed_cost_5y_usd"] <= 50) & (df["prior_ed_visits_5y"] ==
# -----------------------------
# OOF Median Target Encoding (leakage-safe)
# -----------------------------
def te_median_oof(dtr, ytr, dva, group_cols, inner_splits=5):
    # validation encoding: group median in full training fold


# --- PAGE 146 ---

    tr_tmp = dtr[group_cols].copy()
    tr_tmp["_y"] = ytr
    med_map = tr_tmp.groupby(group_cols, dropna=False)["_y"].median()
    global_med = float(np.median(ytr))
    idx_va = pd.MultiIndex.from_frame(dva[group_cols])
    te_va = med_map.reindex(idx_va).to_numpy()
    te_va = np.where(np.isnan(te_va), global_med, te_va)
    # training encoding: OOF inside the training fold
    te_tr = np.full(len(dtr), np.nan, dtype=float)
    inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=RAN
    for itrn, ival in inner.split(dtr, dtr["primary_chronic"].astype(str).values)
        sub = dtr.iloc[itrn][group_cols].copy()
        sub["_y"] = ytr[itrn]
        med_sub = sub.groupby(group_cols, dropna=False)["_y"].median()
        idx_iv = pd.MultiIndex.from_frame(dtr.iloc[ival][group_cols])
        vals = med_sub.reindex(idx_iv).to_numpy()
        vals = np.where(np.isnan(vals), global_med, vals)
        te_tr[ival] = vals
    te_tr = np.where(np.isnan(te_tr), global_med, te_tr)
    return te_tr, te_va
# -----------------------------
# Model factory (early stopping)
# -----------------------------
def make_model(objective, seed):
    params = dict(
        objective=objective,
        n_estimators=20000,
        learning_rate=0.02,
        num_leaves=63,
        min_data_in_leaf=20,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,


# --- PAGE 147 ---

        force_col_wise=True,
        verbose=-1,
    )
    if objective == "quantile":
        params["alpha"] = 0.5
    return lgb.LGBMRegressor(**params)
def fit_predict_earlystop(model, Xtr, ytr, Xva, yva, metric):
    model.fit(
        eval_set=[(Xva, yva)],
        eval_metric=metric,
        callbacks=[lgb.early_stopping(stopping_rounds=250, verbose=False)]
    )
    return model.predict(Xva), getattr(model, "n_estimators_", None)
# -----------------------------
# Feature sets (V2-style) +/- TE
# -----------------------------
num_cols = [
]
num_cols += [c for c in df.columns if c.startswith("share_")]
num_cols += [c for c in df.columns if c.startswith("has_")]
num_cols += [c for c in df.columns if c.startswith("cnt_")]
cat_cols = [c for c in ["primary_chronic","sex","insurance","zip3","prior_cost_de
# Deduplicate and ensure existence
def uniq_exist(cols):
    out, seen = [], set()
    for c in cols:
        if c in df.columns and c not in seen:
            out.append(c); seen.add(c)


# --- PAGE 148 ---

    return out
BASE_FEATURES = uniq_exist(num_cols + cat_cols)
# TE definitions
TE_SPECS = {
    "te_zip3": ["zip3"],
    "te_zip3_ins": ["zip3","insurance"],
    "te_chronic_ins_dec": ["primary_chronic","insurance","prior_cost_decile"],
}
# -----------------------------
# CV evaluation
# -----------------------------
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
def cv_run(name, objective, use_te):
    maes, out_maes = [], []
    iters = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(df, strata), 1):
        dtr = df.iloc[tr_idx].copy()
        dva = df.iloc[va_idx].copy()
        ytr = y[tr_idx]
        yva = y[va_idx]
        # add TE features (OOF for training fold, mapped for validation)
        te_cols = []
        if use_te:
            for te_name, gcols in TE_SPECS.items():
                if all(gc in dtr.columns for gc in gcols):
                    te_tr, te_va = te_median_oof(dtr, ytr, dva, gcols, inner_spli
                    dtr[te_name] = te_tr
                    dva[te_name] = te_va
                    te_cols.append(te_name)
        feat_cols = BASE_FEATURES + te_cols
        Xtr = dtr[feat_cols]
        Xva = dva[feat_cols]


# --- PAGE 149 ---

        model = make_model(objective, seed=RANDOM_STATE + fold)
        # metric for early stopping
        metric = "l1" if objective == "regression_l1" else "quantile"
        pred, n_it = fit_predict_earlystop(model, Xtr, ytr, Xva, yva, metric=metr
        iters.append(n_it if n_it is not None else np.nan)
        mae = mean_absolute_error(yva, pred)
        maes.append(mae)
        m_out = outlier_mask.iloc[va_idx].values
        if m_out.sum() > 0:
            out_maes.append(mean_absolute_error(yva[m_out], pred[m_out]))
        else:
            out_maes.append(np.nan)
        print(f"  [{name}] fold {fold}: MAE={mae:,.3f} | out_n={int(m_out.sum()):
    return {
        "model": name,
        "objective": objective,
        "use_te": use_te,
        "mae_mean": float(np.mean(maes)),
        "mae_std": float(np.std(maes)),
        "out_mae_mean": float(np.nanmean(out_maes)),
        "out_mae_std": float(np.nanstd(out_maes)),
        "avg_best_iter": float(np.nanmean(iters)),
        "n_features_base": int(len(BASE_FEATURES)),
        "n_features_total": int(len(BASE_FEATURES) + (len(TE_SPECS) if use_te els
    }
runs = []
runs.append(cv_run("V2_base_quantile_ES", objective="quantile", use_te=False))
runs.append(cv_run("V2_TE_quantile_ES", objective="quantile", use_te=True))
runs.append(cv_run("V2_base_l1_ES", objective="regression_l1", use_te=False))
runs.append(cv_run("V2_TE_l1_ES", objective="regression_l1", use_te=True))
summary = pd.DataFrame(runs).sort_values("mae_mean")


# --- PAGE 150 ---
```

### C09 — Code cell starting at page 190
```python
from pathlib import Path
import re
import numpy as np
import pandas as pd
from joblib import load
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
# =========================
# PATHS (your structure)
# =========================
BASE_DIR = Path(r"D:\AgentDs\agent_ds_healthcare")
TRAIN_CSV = BASE_DIR / "ed_cost_train.csv"
TEST_CSV  = BASE_DIR / "ed_cost_test.csv"
PATIENTS_CSV = BASE_DIR / "patients.csv"
CACHE_DIR = BASE_DIR / "cache_iter10"
RECEIPT_CACHE = CACHE_DIR / "receipts_parsed.joblib"
SUBMISSION_PATH = BASE_DIR / "submission_ICHI_V1.csv"
# =========================


# --- PAGE 191 ---

# CONFIG
# =========================
RANDOM_STATE = 42
N_SPLITS = 5
EPS = 1e-6
# Try calibration (very cheap). We'll only apply it to test if it improves OOF MA
USE_GROUP_CALIBRATION = True
CAL_GROUP_COLS = ["primary_chronic", "insurance"]  # coarse + stable
# LightGBM params: close to your best V2_plus_counts-style settings
LGB_PARAMS = dict(
    objective="regression_l1",      # MAE-aligned; quantile(alpha=0.5) is effecti
    n_estimators=2200,
    learning_rate=0.03,
    num_leaves=63,
    min_data_in_leaf=20,
    subsample=0.90,
    subsample_freq=1,
    colsample_bytree=0.90,
    reg_lambda=1.0,
    n_jobs=-1,
    force_col_wise=True,
    verbose=-1,
)
def standardize_zip3(z):
    if z is None or (isinstance(z, float) and np.isnan(z)):
        return "missing"
    s = re.sub(r"\D", "", str(z).strip())
    return s.zfill(3) if s else "missing"
# =========================
# LOAD DATA
# =========================
train = pd.read_csv(TRAIN_CSV)
test  = pd.read_csv(TEST_CSV)
patients = pd.read_csv(PATIENTS_CSV)
for d in (train, test, patients):


# --- PAGE 192 ---

    d["patient_id"] = d["patient_id"].astype(int)
patients["zip3"] = patients["zip3"].apply(standardize_zip3)
patients["insurance"] = patients["insurance"].astype(str).str.lower().str.strip()
patients["sex"] = patients["sex"].astype(str).str.strip().replace({"nan":"missing
train = train.merge(patients, on="patient_id", how="left", validate="one_to_one")
test  = test.merge(patients, on="patient_id", how="left", validate="one_to_one")
train["__is_train__"] = 1
test["__is_train__"] = 0
test["ed_cost_next3y_usd"] = np.nan  # placeholder
df_all = pd.concat([train, test], axis=0, ignore_index=True)
# =========================
# LOAD RECEIPTS (cache)
# =========================
cache = load(RECEIPT_CACHE)
if "lineitems_df" not in cache:
    raise KeyError(f"RECEIPT_CACHE missing 'lineitems_df'. Keys: {list(cache.keys
li = cache["lineitems_df"].copy()
li["patient_id"] = li["patient_id"].astype(int)
li["code"] = li["code"].astype(str).str.upper().str.strip()
li["unit_price"] = pd.to_numeric(li["unit_price"], errors="coerce")
li["line_total"] = pd.to_numeric(li["line_total"], errors="coerce")
li = li[li["patient_id"].isin(set(df_all["patient_id"]))].copy()
CODES = sorted(li["code"].unique().tolist())
print(f"[info] receipt codes: {len(CODES)} -> {CODES}")
# =========================
# BUILD RECEIPT FEATURES
# =========================
all_ids = df_all["patient_id"].astype(int).tolist()
# per-patient per-code cost and count
cost_pc = li.groupby(["patient_id","code"], observed=True)["line_total"].sum().un
cnt_pc  = li.groupby(["patient_id","code"], observed=True).size().unstack(fill_va


# --- PAGE 193 ---

# ensure all patients exist
cost_pc = cost_pc.reindex(index=all_ids, fill_value=0.0).reindex(columns=CODES, f
cnt_pc  = cnt_pc.reindex(index=all_ids, fill_value=0.0).reindex(columns=CODES, fi
total_cost = cost_pc.sum(axis=1).replace(0, np.nan)
share_pc = cost_pc.div(total_cost, axis=0).fillna(0.0)
has_pc   = (cnt_pc > 0).astype(float)
share_pc.columns = [f"share_{c}" for c in share_pc.columns]
has_pc.columns   = [f"has_{c}" for c in has_pc.columns]
cnt_pc.columns   = [f"cnt_{c}" for c in cnt_pc.columns]
feat = pd.concat([share_pc, has_pc, cnt_pc], axis=1)
feat["n_line_items"] = cnt_pc.sum(axis=1).values
feat["n_unique_codes"] = (cnt_pc > 0).sum(axis=1).values
# EM summaries
EM_LEVEL = {"99281":1, "99282":2, "99283":3, "99284":4, "99285":5}
em_cnt_cols = [f"cnt_{c}" for c in EM_LEVEL if f"cnt_{c}" in feat.columns]
feat["cnt_em_total"] = feat[em_cnt_cols].sum(axis=1) if em_cnt_cols else 0.0
den = feat["cnt_em_total"].replace(0, np.nan)
num = 0.0
maxlvl = np.zeros(len(feat))
for c, lvl in EM_LEVEL.items():
    col = f"cnt_{c}"
    if col in feat.columns:
        num = num + feat[col] * lvl
        maxlvl = np.maximum(maxlvl, (feat[col] > 0).astype(float) * lvl)
feat["em_level_mean"] = (num / den).fillna(0.0)
feat["em_level_max"] = maxlvl
# critical + severe bundles
CRIT = ["99291","99292"]
SEV_PROC = ["31500","36556","36620","92950"]
crit_cnt_cols = [f"cnt_{c}" for c in CRIT if f"cnt_{c}" in feat.columns]
sev_cnt_cols  = [f"cnt_{c}" for c in SEV_PROC if f"cnt_{c}" in feat.columns]
crit_share_cols = [f"share_{c}" for c in CRIT if f"share_{c}" in feat.columns]
sev_share_cols  = [f"share_{c}" for c in SEV_PROC if f"share_{c}" in feat.columns


# --- PAGE 194 ---

feat["cnt_crit_total"] = feat[crit_cnt_cols].sum(axis=1) if crit_cnt_cols else 0.
feat["cnt_severe_proc_total"] = feat[sev_cnt_cols].sum(axis=1) if sev_cnt_cols el
feat["share_crit"] = feat[crit_share_cols].sum(axis=1) if crit_share_cols else 0.
feat["share_severe_proc"] = feat[sev_share_cols].sum(axis=1) if sev_share_cols el
feat["has_crit"] = (feat["cnt_crit_total"] > 0).astype(int)
feat["has_severe_proc"] = (feat["cnt_severe_proc_total"] > 0).astype(int)
# price_index (weighted average of unit_price / median_unit_price(code))
code_median_unit = li.groupby("code", observed=True)["unit_price"].median()
li2 = li.copy()
li2["code_median_unit"] = li2["code"].map(code_median_unit)
li2["unit_ratio"] = li2["unit_price"] / li2["code_median_unit"].replace(0, np.nan
li2["unit_ratio"] = li2["unit_ratio"].replace([np.inf, -np.inf], np.nan)
li2["w_ratio"] = li2["unit_ratio"] * li2["line_total"].fillna(0.0)
pi = li2.groupby("patient_id", observed=True).agg(w_ratio=("w_ratio","sum"), wsum
pi["price_index"] = pi["w_ratio"] / pi["wsum"].replace(0, np.nan)
pi = pi[["price_index"]].reindex(all_ids).fillna(1.0)
feat["price_index"] = pi["price_index"].values
feat["log_price_index"] = np.log(feat["price_index"].astype(float) + EPS)
feat = feat.reset_index().rename(columns={"index":"patient_id"})
feat["patient_id"] = feat["patient_id"].astype(int)
df_all = df_all.merge(feat, on="patient_id", how="left")
df_all = df_all.fillna(0.0)
# =========================
# TABULAR DERIVED FEATURES
# =========================
df_all["log_prior_cost"] = np.log1p(df_all["prior_ed_cost_5y_usd"].astype(float))
df_all["log_prior_visits"] = np.log1p(df_all["prior_ed_visits_5y"].astype(float))
df_all["prior_cost_per_visit"] = df_all["prior_ed_cost_5y_usd"].astype(float) / (
df_all["prior_visits_zero"] = (df_all["prior_ed_visits_5y"] == 0).astype(int)
df_all["prior_cost_floor"] = (df_all["prior_ed_cost_5y_usd"] <= 50).astype(int)
df_all["prior_cost_deflated"] = df_all["prior_ed_cost_5y_usd"].astype(float) / (d


# --- PAGE 195 ---

df_all["log_prior_cost_deflated"] = np.log1p(df_all["prior_cost_deflated"])
df_all["deflated_cost_per_visit"] = df_all["prior_cost_deflated"] / (df_all["prio
# low price threshold based on TRAIN only
p05 = np.quantile(df_all.loc[df_all["__is_train__"]==1, "price_index"].astype(flo
df_all["low_price_flag"] = (df_all["price_index"].astype(float) < p05).astype(int
df_all["floor_x_lowprice"] = df_all["prior_cost_floor"] * df_all["low_price_flag"
# prior_cost_decile from TRAIN bins, applied to both
train_cost = df_all.loc[df_all["__is_train__"]==1, "prior_ed_cost_5y_usd"].astype
edges = np.quantile(train_cost, np.linspace(0, 1, 11))
edges = np.unique(edges)
if len(edges) < 3:
    edges = np.array([train_cost.min() - 1e-9, train_cost.max() + 1e-9])
labels = list(range(len(edges) - 1))
df_all["prior_cost_decile"] = pd.cut(df_all["prior_ed_cost_5y_usd"].astype(float)
df_all["prior_cost_decile"] = df_all["prior_cost_decile"].astype("category")
# =========================
# CATEGORICALS
# =========================
CAT_COLS = [c for c in ["primary_chronic","sex","insurance","zip3","prior_cost_de
for c in CAT_COLS:
    df_all[c] = df_all[c].astype(str).replace({"nan":"missing","None":"missing"})
# =========================
# SPLIT BACK
# =========================
train_df = df_all[df_all["__is_train__"]==1].reset_index(drop=True)
test_df  = df_all[df_all["__is_train__"]==0].reset_index(drop=True)
y = train_df["ed_cost_next3y_usd"].astype(float).values
strata = train_df["primary_chronic"].astype(str).values
# Outlier definition you used
outlier_mask = (train_df["prior_ed_cost_5y_usd"] <= 50) & (train_df["prior_ed_vis
# =========================
# FEATURE LIST (V2-style + counts)
# =========================


# --- PAGE 196 ---

share_cols = [f"share_{c}" for c in CODES if f"share_{c}" in train_df.columns]
has_cols   = [f"has_{c}"   for c in CODES if f"has_{c}"   in train_df.columns]
cnt_cols   = [f"cnt_{c}"   for c in CODES if f"cnt_{c}"   in train_df.columns]
base_num = [
]
# keep only existing
base_num = [c for c in base_num if c in train_df.columns]
FEATURES = []
seen = set()
for c in base_num + share_cols + has_cols + cnt_cols + CAT_COLS:
    if c in train_df.columns and c not in seen:
        FEATURES.append(c); seen.add(c)
print(f"[info] n_features={len(FEATURES)} | n_cat={len(CAT_COLS)}")
X_train = train_df[FEATURES]
X_test  = test_df[FEATURES]
# =========================
# CV BAGGING (OOF + test preds)
# =========================
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
oof = np.zeros(len(train_df), dtype=float)
test_pred = np.zeros(len(test_df), dtype=float)
fold_id = np.zeros(len(train_df), dtype=int)
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, strata), 1):


# --- PAGE 197 ---

    Xtr, Xva = X_train.iloc[tr_idx], X_train.iloc[va_idx]
    ytr, yva = y[tr_idx], y[va_idx]
    model = lgb.LGBMRegressor(**LGB_PARAMS, random_state=RANDOM_STATE + fold)
    model.fit(Xtr, ytr, categorical_feature=CAT_COLS)
    p_va = model.predict(Xva)
    oof[va_idx] = p_va
    fold_id[va_idx] = fold
    test_pred += model.predict(X_test) / N_SPLITS
    mae = mean_absolute_error(yva, p_va)
    out_mask = outlier_mask.iloc[va_idx].values
    out_mae = mean_absolute_error(yva[out_mask], p_va[out_mask]) if out_mask.sum(
    print(f"fold {fold}: MAE={mae:,.3f} | out_n={int(out_mask.sum()):3d} | out_MA
raw_mae = mean_absolute_error(y, oof)
raw_out_mae = mean_absolute_error(y[outlier_mask.values], oof[outlier_mask.values
print("\n=== OOF SUMMARY (raw) ===")
print(f"OOF MAE={raw_mae:,.3f} | Outlier OOF MAE={raw_out_mae:,.3f} | out_n={int(
# =========================
# OPTIONAL: GROUP MEDIAN RESIDUAL CALIBRATION (fold-safe)
# =========================
final_test_pred = test_pred.copy()
if USE_GROUP_CALIBRATION:
    # fold-safe calibration on OOF
    resid = y - oof
    oof_cal = oof.copy()
    gdf = train_df[CAL_GROUP_COLS].copy()
    gdf["fold"] = fold_id
    gdf["resid"] = resid
    for fold in range(1, N_SPLITS+1):
        va_mask = (gdf["fold"].values == fold)
        tr_mask = ~va_mask
        tmp = gdf.loc[tr_mask, CAL_GROUP_COLS + ["resid"]].copy()


# --- PAGE 198 ---

        med = tmp.groupby(CAL_GROUP_COLS, observed=True)["resid"].median()
        global_med = float(np.median(tmp["resid"].values))
        idx = pd.MultiIndex.from_frame(gdf.loc[va_mask, CAL_GROUP_COLS])
        corr = med.reindex(idx).to_numpy()
        corr = np.where(pd.isna(corr), global_med, corr)
        oof_cal[va_mask] = oof_cal[va_mask] + corr
    cal_mae = mean_absolute_error(y, oof_cal)
    cal_out_mae = mean_absolute_error(y[outlier_mask.values], oof_cal[outlier_mas
    print("\n=== OOF SUMMARY (calibrated) ===")
    print(f"OOF MAE={cal_mae:,.3f} | Outlier OOF MAE={cal_out_mae:,.3f}")
    if cal_mae < raw_mae:
        # Apply calibration to test using all out-of-sample residuals
        tmp_all = gdf[CAL_GROUP_COLS + ["resid"]].copy()
        med_all = tmp_all.groupby(CAL_GROUP_COLS, observed=True)["resid"].median(
        global_all = float(np.median(tmp_all["resid"].values))
        idx_t = pd.MultiIndex.from_frame(test_df[CAL_GROUP_COLS])
        corr_t = med_all.reindex(idx_t).to_numpy()
        corr_t = np.where(pd.isna(corr_t), global_all, corr_t)
        final_test_pred = final_test_pred + corr_t
        print("[calibration] applied to test predictions (improved OOF MAE).")
    else:
        print("[calibration] NOT applied (did not improve OOF MAE).")
# clip to >= 0
final_test_pred = np.clip(final_test_pred, 0.0, None)
# =========================
# WRITE SUBMISSION
# =========================
sub = pd.DataFrame({
    "patient_id": test_df["patient_id"].astype(int).values,
    "ed_cost_next3y_usd": final_test_pred.astype(float),
})
sub.to_csv(SUBMISSION_PATH, index=False)


# --- PAGE 199 ---
```

### C10 — Code cell starting at page 219
```python
# ============================================================
# Phase 3.4: Cross-task feature mining + CV + submission writer
# ============================================================
from __future__ import annotations
import os, re, json, math, warnings
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
from joblib import dump, load
warnings.filterwarnings("ignore")
# ---------- CONFIG ----------
BASE_DIR = Path(r"D:\AgentDs\agent_ds_healthcare")
TRAIN_CSV = BASE_DIR / "ed_cost_train.csv"
TEST_CSV  = BASE_DIR / "ed_cost_test.csv"
PATIENTS_CSV = BASE_DIR / "patients.csv"
PDF_DIR = BASE_DIR / "receipts_pdf"
ADM_TRAIN_CSV = BASE_DIR / "admissions_train.csv"
ADM_TEST_CSV  = BASE_DIR / "admissions_test.csv"
DISCHARGE_NOTES_JSON = BASE_DIR / "discharge_notes.json"
STAYS_TRAIN_CSV = BASE_DIR / "stays_train.csv"
STAYS_TEST_CSV  = BASE_DIR / "stays_test.csv"
VITALS_JSON = BASE_DIR / "vitals_timeseries.json"
SUBMISSION_PATH = BASE_DIR / "submission_EDCost_crossV1.csv"
CACHE_DIR = BASE_DIR / "cache_cross_v1"


# --- PAGE 220 ---

CACHE_DIR.mkdir(parents=True, exist_ok=True)
RECEIPT_CACHE = CACHE_DIR / "receipts_parsed.joblib"
RECEIPT_FEAT_CACHE = CACHE_DIR / "receipt_features.joblib"
ADMISSIONS_FEAT_CACHE = CACHE_DIR / "admissions_patient_features.joblib"
STAYS_FEAT_CACHE = CACHE_DIR / "stays_patient_features.joblib"
FORCE_REPARSE_PDFS = False
FORCE_REBUILD_RECEIPT_FEATS = False
FORCE_REBUILD_ADMISSIONS_FEATS = False
FORCE_REBUILD_STAYS_FEATS = False
RANDOM_SEED = 42
N_FOLDS_ED = 5
N_FOLDS_ADM = 5
N_FOLDS_STAY = 5
# ---------- UTILS ----------
def norm_zip3(z) -> str:
    if pd.isna(z):
        return "UNK"
    z = str(z).strip()
    if z.isdigit():
        return z.zfill(3)
    return z
def safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, **kwargs)
def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))
# ---------- 1) LOAD BASE TABLES ----------
patients = safe_read_csv(PATIENTS_CSV, dtype={"zip3": str})
patients["zip3"] = patients["zip3"].apply(norm_zip3)
for c in ["sex", "insurance", "zip3"]:
    patients[c] = patients[c].astype("category")


# --- PAGE 221 ---

train = safe_read_csv(TRAIN_CSV)
test  = safe_read_csv(TEST_CSV)
train_full = train.merge(patients, on="patient_id", how="left")
test_full  = test.merge(patients, on="patient_id", how="left")
print(f"[info] train_full shape: {train_full.shape}  test_full shape: {test_full.
all_patient_ids = sorted(set(train_full["patient_id"].astype(int)) | set(test_ful
# ---------- 2) PARSE RECEIPTS ----------
def parse_receipt_text(text: str) -> Tuple[Dict, pd.DataFrame]:
    pid_re = re.compile(r"Patient ID:\s*(\d+)")
    zip_ins_re = re.compile(r"ZIP3:\s*([0-9A-Za-z]+)\s+Insurance:\s*([A-Za-z_]+)"
    line_re = re.compile(
        r"^(?P<code>\S+)\s+(?P<desc>.*?)\s+(?P<qty>\d+)\s+(?P<unit>[\d,]+\.\d{2})
    )
    total_re = re.compile(r"^TOTAL\s+(?P<total>[\d,]+\.\d{2})$")
    pid = None
    zip3 = None
    insurance = None
    total = None
    m = pid_re.search(text)
    if m:
        pid = int(m.group(1))
    m = zip_ins_re.search(text)
    if m:
        zip3 = norm_zip3(m.group(1))
        insurance = m.group(2)
    rows = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
        mt = total_re.match(raw)
        if mt:
            total = float(mt.group("total").replace(",", ""))


# --- PAGE 222 ---

        ml = line_re.match(raw)
        if ml:
            d = ml.groupdict()
            rows.append({
                "patient_id": pid,
                "code": d["code"],
                "desc": d["desc"],
                "qty": int(d["qty"]),
                "unit": float(d["unit"].replace(",", "")),
                "line_total": float(d["line_total"].replace(",", "")),
            })
    header = {"patient_id": pid, "zip3_pdf": zip3, "insurance_pdf": insurance, "p
    return header, pd.DataFrame(rows)
def parse_all_receipts(patient_ids: List[int]) -> Tuple[pd.DataFrame, pd.DataFram
    # Robust cache load (fixes old KeyError issue)
    if RECEIPT_CACHE.exists() and (not FORCE_REPARSE_PDFS):
        try:
            cached = load(RECEIPT_CACHE)
            if isinstance(cached, dict) and ("headers_df" in cached) and ("lineit
                headers_df = cached["headers_df"]
                lineitems_df = cached["lineitems_df"]
                # quick sanity
                if {"patient_id"}.issubset(headers_df.columns) and {"patient_id",
                    return headers_df, lineitems_df
            print("[cache] Receipt cache found but schema mismatch; reparsing.")
        except Exception as e:
            print(f"[cache] Failed to load receipt cache; reparsing. err={e}")
    import pdfplumber
    headers = []
    items = []
    missing = 0
    for pid in patient_ids:
        pdf_path = PDF_DIR / f"receipt_{pid}.pdf"
        if not pdf_path.exists():
            missing += 1


# --- PAGE 223 ---

        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = "\n".join((p.extract_text() or "") for p in pdf.pages)
            h, df = parse_receipt_text(text)
            headers.append(h)
            if df is not None and len(df):
                items.append(df)
        except Exception as e:
            headers.append({"patient_id": pid, "zip3_pdf": None, "insurance_pdf":
    headers_df = pd.DataFrame(headers)
    lineitems_df = pd.concat(items, ignore_index=True) if len(items) else pd.Data
    dump({"headers_df": headers_df, "lineitems_df": lineitems_df}, RECEIPT_CACHE)
    print(f"[cache] Wrote receipt cache: {RECEIPT_CACHE} | missing_pdfs={missing}
    return headers_df, lineitems_df
headers_df, lineitems_df = parse_all_receipts(all_patient_ids)
print(f"[info] Parsed lineitems rows: {len(lineitems_df)}  unique patients: {line
# ---------- 3) RECEIPT FEATURES ----------
def build_receipt_features(headers_df: pd.DataFrame, lineitems_df: pd.DataFrame)
    # Cache
    if RECEIPT_FEAT_CACHE.exists() and (not FORCE_REBUILD_RECEIPT_FEATS):
        try:
            df = load(RECEIPT_FEAT_CACHE)
            if isinstance(df, pd.DataFrame) and "patient_id" in df.columns:
                return df
        except Exception:
    li = lineitems_df.copy()
    li["code"] = li["code"].astype(str)
    # Identify code universe (use all codes observed)
    codes = sorted(li["code"].unique().tolist())
    print(f"[info] receipt codes: {len(codes)} -> {codes}")


# --- PAGE 224 ---

    # Per-code medians for deflation (to separate volume from price)
    med_unit_by_code = li.groupby("code")["unit"].median().to_dict()
    li["unit_med_code"] = li["code"].map(med_unit_by_code).astype(float)
    li["deflated_line_total"] = li["qty"].astype(float) * li["unit_med_code"]
    # Patient aggregates
    agg = li.groupby("patient_id").agg(
        pdf_total_line_cost=("line_total","sum"),
        pdf_deflated_total=("deflated_line_total","sum"),
        pdf_n_line_items=("code","size"),
        pdf_total_qty=("qty","sum"),
        pdf_n_unique_codes=("code","nunique"),
        pdf_unit_mean=("unit","mean"),
        pdf_unit_median=("unit","median"),
        pdf_line_max=("line_total","max"),
    ).reset_index()
    agg["pdf_price_index"] = agg["pdf_total_line_cost"] / (agg["pdf_deflated_tota
    agg["pdf_price_index"] = agg["pdf_price_index"].fillna(1.0)
    # Code-level wide features (counts/cost/shares)
    piv_cnt = li.pivot_table(index="patient_id", columns="code", values="line_tot
    piv_cost = li.pivot_table(index="patient_id", columns="code", values="line_tot
    piv_cnt.columns = [f"pdf_cnt_{c}" for c in piv_cnt.columns]
    piv_cost.columns = [f"pdf_cost_{c}" for c in piv_cost.columns]
    wide = piv_cnt.join(piv_cost, how="outer").reset_index()
    out = agg.merge(wide, on="patient_id", how="left").fillna(0)
    # Shares per code
    for c in codes:
        out[f"pdf_share_{c}"] = out.get(f"pdf_cost_{c}", 0.0) / out["pdf_total_li
        out[f"pdf_share_{c}"] = out[f"pdf_share_{c}"].fillna(0.0)
    # Clinical-ish category groupings for these synthetic CPTs
    ED_EM = {"99281","99282","99283","99284","99285"}
    CRIT = {"99291","99292"}
    LAB = {"85025","87070","84484"}


# --- PAGE 225 ---

    IMG = {"71045","70450","74177"}
    OBS = {"G0378"}
    SEVERE = {"31500","36556","36620","92950"}  # intubation/lines/CPR
    def sum_cols(prefix: str, code_set: set) -> pd.Series:
        cols = [f"{prefix}_{c}" for c in code_set if f"{prefix}_{c}" in out.colum
        if not cols:
            return pd.Series(np.zeros(len(out)), index=out.index)
        return out[cols].sum(axis=1)
    # Counts
    out["pdf_cnt_ed_em"] = sum_cols("pdf_cnt", ED_EM)
    out["pdf_cnt_crit_care"] = sum_cols("pdf_cnt", CRIT)
    out["pdf_cnt_lab"] = sum_cols("pdf_cnt", LAB)
    out["pdf_cnt_imaging"] = sum_cols("pdf_cnt", IMG)
    out["pdf_cnt_obs"] = sum_cols("pdf_cnt", OBS)
    out["pdf_cnt_severe_proc"] = sum_cols("pdf_cnt", SEVERE)
    # Costs
    out["pdf_cost_ed_em"] = sum_cols("pdf_cost", ED_EM)
    out["pdf_cost_crit_care"] = sum_cols("pdf_cost", CRIT)
    out["pdf_cost_lab"] = sum_cols("pdf_cost", LAB)
    out["pdf_cost_imaging"] = sum_cols("pdf_cost", IMG)
    out["pdf_cost_obs"] = sum_cols("pdf_cost", OBS)
    out["pdf_cost_severe_proc"] = sum_cols("pdf_cost", SEVERE)
    # Shares
    for k in ["ed_em","crit_care","lab","imaging","obs","severe_proc"]:
        out[f"pdf_share_{k}"] = out[f"pdf_cost_{k}"] / out["pdf_total_line_cost"]
        out[f"pdf_share_{k}"] = out[f"pdf_share_{k}"].fillna(0.0)
    # E/M severity stats
    em_level_map = {"99281":1, "99282":2, "99283":3, "99284":4, "99285":5}
    em = li[li["code"].isin(ED_EM)].copy()
    if len(em):
        em["em_level"] = em["code"].map(em_level_map).astype(float)
        em_agg = em.groupby("patient_id").agg(
            pdf_em_level_mean=("em_level","mean"),
            pdf_em_level_max=("em_level","max"),
            pdf_em_lines=("em_level","size"),


# --- PAGE 226 ---

            pdf_em_high_frac=("em_level", lambda x: float(np.mean(np.asarray(x) >
        ).reset_index()
        out = out.merge(em_agg, on="patient_id", how="left")
    else:
        out["pdf_em_level_mean"] = 0.0
        out["pdf_em_level_max"] = 0.0
        out["pdf_em_lines"] = 0.0
        out["pdf_em_high_frac"] = 0.0
    out[["pdf_em_level_mean","pdf_em_level_max","pdf_em_lines","pdf_em_high_frac"
    # Cost entropy over codes (diversity)
    cost_cols = [c for c in out.columns if c.startswith("pdf_cost_") and (c not i
    if cost_cols:
        cost_mat = out[cost_cols].to_numpy(dtype=float)
        row_sum = cost_mat.sum(axis=1, keepdims=True)
        p = np.divide(cost_mat, row_sum, out=np.zeros_like(cost_mat), where=row_s
        ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum(axis=1)
        ent_norm = ent / np.log(max(2, cost_mat.shape[1]))
        out["pdf_code_cost_entropy"] = ent
        out["pdf_code_cost_entropy_norm"] = ent_norm
    else:
        out["pdf_code_cost_entropy"] = 0.0
        out["pdf_code_cost_entropy_norm"] = 0.0
    dump(out, RECEIPT_FEAT_CACHE)
    print(f"[cache] Wrote receipt features: {RECEIPT_FEAT_CACHE}")
    return out
receipt_feat = build_receipt_features(headers_df, lineitems_df)
# ---------- 4) ADMISSIONS FEATURES + AUX READMISSION MODEL ----------
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
def build_admissions_patient_features(patients_df: pd.DataFrame) -> pd.DataFrame:
    if ADMISSIONS_FEAT_CACHE.exists() and (not FORCE_REBUILD_ADMISSIONS_FEATS):
        try:


# --- PAGE 227 ---

            df = load(ADMISSIONS_FEAT_CACHE)
            if isinstance(df, pd.DataFrame) and "patient_id" in df.columns:
                return df
        except Exception:
    if not (ADM_TRAIN_CSV.exists() and ADM_TEST_CSV.exists() and DISCHARGE_NOTES_
        print("[warn] Admissions files not found; skipping admissions features.")
        return patients_df[["patient_id"]].copy()
    adm_tr = safe_read_csv(ADM_TRAIN_CSV)
    adm_te = safe_read_csv(ADM_TEST_CSV)
    with open(DISCHARGE_NOTES_JSON, "r") as f:
        notes = json.load(f)
    notes_df = pd.DataFrame(notes)
    adm_tr = adm_tr.merge(notes_df, on="admission_id", how="left")
    adm_te = adm_te.merge(notes_df, on="admission_id", how="left")
    # Add patient demographics
    adm_tr = adm_tr.merge(patients_df, on="patient_id", how="left")
    adm_te = adm_te.merge(patients_df, on="patient_id", how="left")
    adm_tr["note"] = adm_tr["note"].fillna("")
    adm_te["note"] = adm_te["note"].fillna("")
    adm_tr["note_len"] = adm_tr["note"].astype(str).str.len()
    adm_te["note_len"] = adm_te["note"].astype(str).str.len()
    # TF-IDF + SVD on discharge notes (compact dense features)
    all_notes = pd.concat([adm_tr[["note"]], adm_te[["note"]]], ignore_index=True
    tfidf = TfidfVectorizer(max_features=4000, ngram_range=(1,2), min_df=2)
    X_tfidf = tfidf.fit_transform(all_notes["note"].astype(str).values)
    svd = TruncatedSVD(n_components=48, random_state=RANDOM_SEED)
    X_svd = svd.fit_transform(X_tfidf)
    svd_cols = [f"adm_note_svd_{i:02d}" for i in range(X_svd.shape[1])]
    X_svd_df = pd.DataFrame(X_svd, columns=svd_cols)
    adm_all = pd.concat([adm_tr.reset_index(drop=True), adm_te.reset_index(drop=T
    adm_all = pd.concat([adm_all.reset_index(drop=True), X_svd_df.reset_index(dro


# --- PAGE 228 ---

    # Aux readmission model (OOF for train admissions, averaged preds for test adm
    y = adm_all.loc[:len(adm_tr)-1, "readmit_30d"].astype(int).values
    feat_num = ["los_days","acuity_emergent","charlson_band","ed_visits_6m","disc
    feat_cat = ["primary_dx","sex","insurance","zip3"]
    for c in feat_cat:
        adm_all[c] = adm_all[c].astype("category")
    # Fill numeric
    for c in feat_num:
        adm_all[c] = adm_all[c].astype(float)
        adm_all[c] = adm_all[c].fillna(adm_all[c].median())
    X_train = adm_all.loc[:len(adm_tr)-1, feat_num + feat_cat]
    X_test  = adm_all.loc[len(adm_tr):, feat_num + feat_cat]
    skf = StratifiedKFold(n_splits=N_FOLDS_ADM, shuffle=True, random_state=RANDOM_
    oof = np.zeros(len(X_train), dtype=float)
    test_pred = np.zeros(len(X_test), dtype=float)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y), 1):
        Xtr, Xva = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]
        clf = lgb.LGBMClassifier(
            n_estimators=5000,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=0.0,
            objective="binary",
            random_state=RANDOM_SEED + fold,
            n_jobs=-1,
        )
        clf.fit(
            eval_set=[(Xva, yva)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(200, verbose=False)]


# --- PAGE 229 ---

        )
        oof[va_idx] = clf.predict_proba(Xva, num_iteration=clf.best_iteration_)[:
        test_pred += clf.predict_proba(X_test, num_iteration=clf.best_iteration_)
    # Attach predictions
    adm_all["adm_readmit_pred"] = np.nan
    adm_all.loc[:len(adm_tr)-1, "adm_readmit_pred"] = oof
    adm_all.loc[len(adm_tr):, "adm_readmit_pred"] = test_pred
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y, oof)
        print(f"[aux-readmit] OOF AUC={auc:.4f}")
    except Exception:
    # Patient-level aggregates
    base_agg = adm_all.groupby("patient_id").agg(
        adm_n=("admission_id","size"),
        adm_los_mean=("los_days","mean"),
        adm_los_max=("los_days","max"),
        adm_charlson_mean=("charlson_band","mean"),
        adm_charlson_max=("charlson_band","max"),
        adm_acuity_rate=("acuity_emergent","mean"),
        adm_edvis6m_mean=("ed_visits_6m","mean"),
        adm_edvis6m_max=("ed_visits_6m","max"),
        adm_readmit_pred_mean=("adm_readmit_pred","mean"),
        adm_readmit_pred_max=("adm_readmit_pred","max"),
        adm_readmit_pred_sum=("adm_readmit_pred","sum"),
        adm_note_len_mean=("note_len","mean"),
        adm_note_len_max=("note_len","max"),
    ).reset_index()
    # Dx mix counts
    dx_counts = pd.crosstab(adm_all["patient_id"], adm_all["primary_dx"])
    dx_counts.columns = [f"adm_dx_cnt_{c}" for c in dx_counts.columns.astype(str)
    dx_counts = dx_counts.reset_index()
    # Readmit label aggregates using admissions_train only (optional, but availab
    tr_only = adm_all.loc[:len(adm_tr)-1].copy()


# --- PAGE 230 ---

    lbl_agg = tr_only.groupby("patient_id").agg(
        adm_train_n=("admission_id","size"),
        adm_readmit_rate=("readmit_30d","mean"),
        adm_readmit_n=("readmit_30d","sum"),
    ).reset_index()
    out = base_agg.merge(dx_counts, on="patient_id", how="left").merge(lbl_agg, o
    out = out.fillna(0)
    dump(out, ADMISSIONS_FEAT_CACHE)
    print(f"[cache] Wrote admissions patient features: {ADMISSIONS_FEAT_CACHE}")
    return out
adm_feat = build_admissions_patient_features(patients)
# ---------- 5) STAYS/VITALS FEATURES (optional) ----------
def build_stays_patient_features(patients_df: pd.DataFrame) -> pd.DataFrame:
    if STAYS_FEAT_CACHE.exists() and (not FORCE_REBUILD_STAYS_FEATS):
        try:
            df = load(STAYS_FEAT_CACHE)
            if isinstance(df, pd.DataFrame) and "patient_id" in df.columns:
                return df
        except Exception:
    if not (STAYS_TRAIN_CSV.exists() and STAYS_TEST_CSV.exists() and VITALS_JSON.
        print("[warn] stays_train/test or vitals json not found; skipping stays/v
        return patients_df[["patient_id"]].copy()
    stays_tr = safe_read_csv(STAYS_TRAIN_CSV)
    stays_te = safe_read_csv(STAYS_TEST_CSV)
    stays_all = pd.concat([stays_tr, stays_te], ignore_index=True)
    with open(VITALS_JSON, "r") as f:
        vitals = json.load(f)
    # Per-stay vitals summary
    rows = []
    notes_concat = []
    for obj in vitals:


# --- PAGE 231 ---

        sid = obj.get("stay_id")
        days = obj.get("days", [])
        if not days:
        ddf = pd.DataFrame(days)
        ddf = ddf.sort_values("day")
        # slope helper
        def slope(x: np.ndarray) -> float:
            t = np.arange(1, len(x)+1, dtype=float)
            x = x.astype(float)
            vt = np.var(t)
            if vt == 0:
                return 0.0
            return float(np.cov(t, x, bias=True)[0,1] / vt)
        feat = {
            "stay_id": sid,
            "hr_mean": float(ddf["hr"].mean()),
            "hr_std": float(ddf["hr"].std(ddof=0)),
            "hr_slope": slope(ddf["hr"].to_numpy()),
            "sbp_mean": float(ddf["sbp"].mean()),
            "sbp_std": float(ddf["sbp"].std(ddof=0)),
            "sbp_slope": slope(ddf["sbp"].to_numpy()),
            "dbp_mean": float(ddf["dbp"].mean()),
            "dbp_std": float(ddf["dbp"].std(ddof=0)),
            "temp_mean": float(ddf["temp_c"].mean()),
            "temp_max": float(ddf["temp_c"].max()),
            "rr_mean": float(ddf["rr"].mean()),
            "rr_std": float(ddf["rr"].std(ddof=0)),
            "fever_days": float(np.sum(ddf["temp_c"].to_numpy(dtype=float) >= 37.
        }
        rows.append(feat)
        note_text = " ".join([str(x) for x in ddf.get("note", "").fillna("").toli
        notes_concat.append(note_text)
    vit_df = pd.DataFrame(rows)
    vit_df["stay_note"] = notes_concat[:len(vit_df)]
    vit_df["stay_note_len"] = vit_df["stay_note"].astype(str).str.len()
    # TF-IDF+SVD on stay notes (compact)


# --- PAGE 232 ---

    tfidf = TfidfVectorizer(max_features=6000, ngram_range=(1,2), min_df=2)
    X_tfidf = tfidf.fit_transform(vit_df["stay_note"].astype(str).values)
    svd = TruncatedSVD(n_components=32, random_state=RANDOM_SEED)
    X_svd = svd.fit_transform(X_tfidf)
    svd_cols = [f"stay_note_svd_{i:02d}" for i in range(X_svd.shape[1])]
    vit_df = pd.concat([vit_df.reset_index(drop=True), pd.DataFrame(X_svd, column
    # Merge into stays
    stays_all = stays_all.merge(vit_df.drop(columns=["stay_note"]), on="stay_id",
    stays_all = stays_all.merge(patients_df, on="patient_id", how="left")
    # Fill missing vitals (for stays without vitals rows)
    num_cols = [c for c in stays_all.columns if c.endswith(("_mean","_std","_slop
    for c in num_cols:
        stays_all[c] = stays_all[c].astype(float)
        stays_all[c] = stays_all[c].fillna(stays_all[c].median())
    # Aux discharge-ready model (OOF preds for train stays, avg preds for test st
    if "discharge_ready_day11" in stays_tr.columns:
        y = stays_tr["discharge_ready_day11"].astype(int).values
        feat_num = ["hr_mean","hr_std","hr_slope","sbp_mean","sbp_std","sbp_slope
        feat_cat = ["unit_type","admission_reason","sex","insurance","zip3"]
        for c in feat_cat:
            stays_all[c] = stays_all[c].astype("category")
        X_train = stays_all.loc[:len(stays_tr)-1, feat_num + feat_cat]
        X_test  = stays_all.loc[len(stays_tr):, feat_num + feat_cat]
        skf = StratifiedKFold(n_splits=N_FOLDS_STAY, shuffle=True, random_state=RA
        oof = np.zeros(len(X_train))
        test_pred = np.zeros(len(X_test))
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y), 1):
            Xtr, Xva = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            ytr, yva = y[tr_idx], y[va_idx]
            clf = lgb.LGBMClassifier(
                n_estimators=5000,
                learning_rate=0.03,
                num_leaves=63,
                subsample=0.8,


# --- PAGE 233 ---

                colsample_bytree=0.8,
                objective="binary",
                random_state=RANDOM_SEED + 100 + fold,
                n_jobs=-1,
            )
            clf.fit(
                eval_set=[(Xva, yva)],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(200, verbose=False)]
            )
            oof[va_idx] = clf.predict_proba(Xva, num_iteration=clf.best_iteration_
            test_pred += clf.predict_proba(X_test, num_iteration=clf.best_iterati
        stays_all["stay_discharge_ready_pred"] = np.nan
        stays_all.loc[:len(stays_tr)-1, "stay_discharge_ready_pred"] = oof
        stays_all.loc[len(stays_tr):, "stay_discharge_ready_pred"] = test_pred
    # Patient-level aggregates
    agg_cols = {
        "stay_n": ("stay_id","size"),
        "stay_hr_mean_mean": ("hr_mean","mean"),
        "stay_sbp_mean_mean": ("sbp_mean","mean"),
        "stay_temp_max_max": ("temp_max","max"),
        "stay_fever_days_mean": ("fever_days","mean"),
        "stay_note_len_mean": ("stay_note_len","mean"),
    }
    if "stay_discharge_ready_pred" in stays_all.columns:
        agg_cols.update({
            "stay_ready_pred_mean": ("stay_discharge_ready_pred","mean"),
            "stay_ready_pred_max": ("stay_discharge_ready_pred","max"),
            "stay_ready_pred_sum": ("stay_discharge_ready_pred","sum"),
        })
    base_agg = stays_all.groupby("patient_id").agg(**{k:v for k,v in agg_cols.ite
    # unit_type / admission_reason counts
    unit_ct = pd.crosstab(stays_all["patient_id"], stays_all["unit_type"])
    unit_ct.columns = [f"stay_unit_cnt_{c}" for c in unit_ct.columns.astype(str)]
    unit_ct = unit_ct.reset_index()


# --- PAGE 234 ---

    rsn_ct = pd.crosstab(stays_all["patient_id"], stays_all["admission_reason"])
    rsn_ct.columns = [f"stay_reason_cnt_{c}" for c in rsn_ct.columns.astype(str)]
    rsn_ct = rsn_ct.reset_index()
    out = base_agg.merge(unit_ct, on="patient_id", how="left").merge(rsn_ct, on="
    dump(out, STAYS_FEAT_CACHE)
    print(f"[cache] Wrote stays patient features: {STAYS_FEAT_CACHE}")
    return out
stays_feat = build_stays_patient_features(patients)
# ---------- 6) MERGE ALL FEATURES INTO ED COST TABLE ----------
def merge_features(ed_df: pd.DataFrame) -> pd.DataFrame:
    df = ed_df.copy()
    df = df.merge(receipt_feat, on="patient_id", how="left")
    df = df.merge(adm_feat, on="patient_id", how="left")
    df = df.merge(stays_feat, on="patient_id", how="left")
    return df
train_feat = merge_features(train_full)
test_feat  = merge_features(test_full)
# Basic engineered tabular features
for df in [train_feat, test_feat]:
    df["prior_cost_log1p"] = np.log1p(df["prior_ed_cost_5y_usd"])
    df["prior_visits_log1p"] = np.log1p(df["prior_ed_visits_5y"])
    df["prior_cost_per_visit"] = df["prior_ed_cost_5y_usd"] / (df["prior_ed_visit
# ---------- 7) TRAIN ED COST MODEL (CV + TEST PRED) ----------
from sklearn.model_selection import StratifiedKFold
target = train_feat["ed_cost_next3y_usd"].astype(float).values
# Define stratification key: chronic + cost decile
cost_dec = pd.qcut(train_feat["prior_ed_cost_5y_usd"], q=10, duplicates="drop")
strat_key = (train_feat["primary_chronic"].astype(str) + "_" + cost_dec.astype(st
cat_cols = ["primary_chronic","sex","insurance","zip3"]


# --- PAGE 235 ---

for c in cat_cols:
    train_feat[c] = train_feat[c].astype("category")
    test_feat[c] = test_feat[c].astype("category")
drop_cols = {"ed_cost_next3y_usd"}
feature_cols = [c for c in train_feat.columns if c not in drop_cols]
X = train_feat[feature_cols].copy()
X_test = test_feat[feature_cols].copy()
# Fill numeric NaNs
for c in feature_cols:
    if c in cat_cols:
    X[c] = pd.to_numeric(X[c], errors="coerce")
    X_test[c] = pd.to_numeric(X_test[c], errors="coerce")
    med = X[c].median()
    X[c] = X[c].fillna(med)
    X_test[c] = X_test[c].fillna(med)
skf = StratifiedKFold(n_splits=N_FOLDS_ED, shuffle=True, random_state=RANDOM_SEED
oof = np.zeros(len(X), dtype=float)
test_pred = np.zeros(len(X_test), dtype=float)
# outlier regime you already care about
outlier_mask = (train_feat["prior_ed_cost_5y_usd"] <= 50.0) & (train_feat["prior_
outlier_idx = np.where(outlier_mask.values)[0]
print(f"[info] n_features={len(feature_cols)} | n_cat={len(cat_cols)} | outlier_n
best_iters = []
for fold, (tr_idx, va_idx) in enumerate(skf.split(X, strat_key), 1):
    Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
    ytr, yva = target[tr_idx], target[va_idx]
    reg = lgb.LGBMRegressor(
        n_estimators=20000,
        learning_rate=0.01,
        num_leaves=127,


# --- PAGE 236 ---

        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        objective="regression_l1",
        random_state=RANDOM_SEED + fold,
        n_jobs=-1,
    )
    reg.fit(
        eval_set=[(Xva, yva)],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(500, verbose=False)]
    )
    best_iters.append(int(reg.best_iteration_ or 0))
    pred_va = reg.predict(Xva, num_iteration=reg.best_iteration_)
    oof[va_idx] = pred_va
    fold_mae = mae(yva, pred_va)
    # Outlier MAE within this fold
    fold_out = np.intersect1d(va_idx, outlier_idx)
    if len(fold_out) > 0:
        fold_out_mae = mae(target[fold_out], oof[fold_out])
    else:
        fold_out_mae = np.nan
    print(f"fold {fold}: MAE={fold_mae:.3f} | out_n={len(fold_out):3d} | out_MAE=
    # test preds
    test_pred += reg.predict(X_test, num_iteration=reg.best_iteration_) / N_FOLDS_
oof_mae = mae(target, oof)
out_mae = mae(target[outlier_idx], oof[outlier_idx]) if len(outlier_idx) else np.
print("\n=== OOF SUMMARY ===")
print(f"OOF MAE={oof_mae:.3f} | Outlier OOF MAE={out_mae:.3f} | out_n={len(outlie
print(f"avg_best_iter={np.mean(best_iters):.1f}")


# --- PAGE 237 ---
```

### C11 — Code cell starting at page 240
```python
from pathlib import Path
import re, json, math, warnings
import numpy as np
import pandas as pd
from joblib import load, dump
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
warnings.filterwarnings("ignore")
# =========================
# PATHS
# =========================
BASE_DIR = Path(r"D:\AgentDs\agent_ds_healthcare")
TRAIN_CSV = BASE_DIR / "ed_cost_train.csv"
TEST_CSV  = BASE_DIR / "ed_cost_test.csv"
PATIENTS_CSV = BASE_DIR / "patients.csv"
PDF_CACHE = BASE_DIR / "cache_iter10" / "receipts_parsed.joblib"
ADMISSIONS_TRAIN = BASE_DIR / "admissions_train.csv"
ADMISSIONS_TEST  = BASE_DIR / "admissions_test.csv"
DISCHARGE_NOTES  = BASE_DIR / "discharge_notes.json"
STAYS_TRAIN = BASE_DIR / "stays_train.csv"


# --- PAGE 241 ---

STAYS_TEST  = BASE_DIR / "stays_test.csv"
VITALS_JSON = BASE_DIR / "vitals_timeseries.json"
OUT_SUB = BASE_DIR / "submission_EDCost_V6pp.csv"
FEATURE_CACHE = BASE_DIR / "cache_iter10" / "features_v6pp.joblib"
# =========================
# CONFIG
# =========================
EPS = 1e-6
N_SPLITS = 5
REPEAT_SEEDS = [42, 2024, 999]   # reduces variance vs single CV
EARLY_STOP = 300
USE_TEXT_FEATURES = True         # discharge notes + vitals notes (if available)
TEXT_MAX_FEATURES = 3000
TEXT_SVD_DIM = 50
VITALS_NOTE_SVD_DIM = 30
TRY_CATBOOST = True              # if installed, will train + blend
# LightGBM: slightly more regularized than your earlier runs (often helps LB gene
LGB_PARAMS = dict(
    objective="regression_l1",
    n_estimators=20000,
    learning_rate=0.03,
    num_leaves=63,
    min_data_in_leaf=30,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=1.0,
    n_jobs=-1,
    force_col_wise=True,
    verbose=-1,
)
def standardize_zip3(z):
    if z is None or (isinstance(z, float) and np.isnan(z)):


# --- PAGE 242 ---

        return "missing"
    s = re.sub(r"\D", "", str(z).strip())
    return s.zfill(3) if s else "missing"
def safe_read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
def add_tfidf_svd(text_series, prefix, max_features=3000, svd_dim=50):
    text_series = text_series.fillna("").astype(str)
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1,2), min_df=
    X = tfidf.fit_transform(text_series.values)
    if svd_dim >= X.shape[1]:
        svd_dim = max(2, min(50, X.shape[1]-1))
    svd = TruncatedSVD(n_components=svd_dim, random_state=42)
    Z = svd.fit_transform(X)
    cols = [f"{prefix}_svd_{i}" for i in range(Z.shape[1])]
    return pd.DataFrame(Z, columns=cols, index=text_series.index)
# =========================
# LOAD TABULAR
# =========================
train = pd.read_csv(TRAIN_CSV)
test  = pd.read_csv(TEST_CSV)
patients = pd.read_csv(PATIENTS_CSV)
for d in (train, test, patients):
    d["patient_id"] = d["patient_id"].astype(int)
patients["zip3"] = patients["zip3"].apply(standardize_zip3)
patients["insurance"] = patients["insurance"].astype(str).str.lower().str.strip()
patients["sex"] = patients["sex"].astype(str).str.strip().replace({"nan":"missing
train = train.merge(patients, on="patient_id", how="left", validate="one_to_one")
test  = test.merge(patients, on="patient_id", how="left", validate="one_to_one")
# =========================
# FEATURE BUILD (cached)
# =========================


# --- PAGE 243 ---

if FEATURE_CACHE.exists():
    obj = load(FEATURE_CACHE)
    train_feat = obj["train_feat"]
    test_feat = obj["test_feat"]
    FEATURES = obj["FEATURES"]
    CAT_COLS = obj["CAT_COLS"]
    print(f"[cache] Loaded features from {FEATURE_CACHE}")
else:
    # ---------- Receipts ----------
    cache_obj = load(PDF_CACHE)
    li = cache_obj["lineitems_df"].copy() if isinstance(cache_obj, dict) else cac
    li["patient_id"] = li["patient_id"].astype(int)
    li["code"] = li["code"].astype(str).str.upper().str.strip()
    li["unit_price"] = pd.to_numeric(li.get("unit_price", np.nan), errors="coerce
    li["line_total"] = pd.to_numeric(li.get("line_total", np.nan), errors="coerce
    all_ids = sorted(set(train["patient_id"]).union(set(test["patient_id"])))
    li = li[li["patient_id"].isin(all_ids)].copy()
    codes = sorted(li["code"].unique().tolist())
    print(f"[info] receipt codes: {len(codes)} -> {codes}")
    # join insurance/zip3 so we can compute payer-adjusted price index
    pid_info = pd.concat([train[["patient_id","insurance","zip3"]], test[["patien
    li = li.merge(pid_info, on="patient_id", how="left")
    li["insurance"] = li["insurance"].fillna("missing").astype(str).str.lower().s
    li["zip3"] = li["zip3"].apply(standardize_zip3)
    # patient-code cost & counts
    cost_pc = li.groupby(["patient_id","code"], observed=True)["line_total"].sum(
    cnt_pc  = li.groupby(["patient_id","code"], observed=True).size().unstack(fil
    # ensure all patients exist
    cost_pc = cost_pc.reindex(index=all_ids, fill_value=0.0)
    cnt_pc  = cnt_pc.reindex(index=all_ids, fill_value=0.0)
    total_cost = cost_pc.sum(axis=1).replace(0, np.nan)
    share_pc = cost_pc.div(total_cost, axis=0).fillna(0.0)
    has_pc = (cnt_pc > 0).astype(float)


# --- PAGE 244 ---

    share_pc.columns = [f"share_{c}" for c in share_pc.columns]
    has_pc.columns   = [f"has_{c}" for c in has_pc.columns]
    cnt_pc.columns   = [f"cnt_{c}" for c in cnt_pc.columns]
    receipt_feat = pd.concat([share_pc, has_pc, cnt_pc], axis=1)
    receipt_feat.index.name = "patient_id"
    receipt_feat = receipt_feat.reset_index()
    # summaries
    receipt_feat["n_line_items"] = cnt_pc.sum(axis=1).values
    receipt_feat["n_unique_codes"] = (cnt_pc > 0).sum(axis=1).values
    # EM summaries
    EM_LEVEL = {"99281":1, "99282":2, "99283":3, "99284":4, "99285":5}
    em_cols = [f"cnt_{c}" for c in EM_LEVEL if f"cnt_{c}" in receipt_feat.columns
    receipt_feat["cnt_em_total"] = receipt_feat[em_cols].sum(axis=1) if em_cols e
    den = receipt_feat["cnt_em_total"].replace(0, np.nan)
    num = 0.0
    maxlvl = np.zeros(len(receipt_feat))
    for c, lvl in EM_LEVEL.items():
        col = f"cnt_{c}"
        if col in receipt_feat.columns:
            num = num + receipt_feat[col] * lvl
            maxlvl = np.maximum(maxlvl, (receipt_feat[col] > 0).astype(float) * l
    receipt_feat["em_level_mean"] = (num / den).fillna(0.0)
    receipt_feat["em_level_max"] = maxlvl
    # critical/severe bundles
    CRIT = ["99291","99292"]
    SEV_PROC = ["31500","36556","36620","92950"]
    crit_cnt_cols = [f"cnt_{c}" for c in CRIT if f"cnt_{c}" in receipt_feat.colum
    sev_cnt_cols  = [f"cnt_{c}" for c in SEV_PROC if f"cnt_{c}" in receipt_feat.c
    crit_share_cols = [f"share_{c}" for c in CRIT if f"share_{c}" in receipt_feat
    sev_share_cols  = [f"share_{c}" for c in SEV_PROC if f"share_{c}" in receipt_
    receipt_feat["cnt_crit_total"] = receipt_feat[crit_cnt_cols].sum(axis=1) if c
    receipt_feat["cnt_severe_proc_total"] = receipt_feat[sev_cnt_cols].sum(axis=1
    receipt_feat["share_crit"] = receipt_feat[crit_share_cols].sum(axis=1) if cri


# --- PAGE 245 ---

    receipt_feat["share_severe_proc"] = receipt_feat[sev_share_cols].sum(axis=1)
    receipt_feat["has_crit"] = (receipt_feat["cnt_crit_total"] > 0).astype(int)
    receipt_feat["has_severe_proc"] = (receipt_feat["cnt_severe_proc_total"] > 0)
    # price indices
    code_med_global = li.groupby("code", observed=True)["unit_price"].median()
    code_med_ins = li.groupby(["code","insurance"], observed=True)["unit_price"].
    li2 = li.copy()
    li2["code_med_global"] = li2["code"].map(code_med_global)
    li2["code_med_ins"] = li2.set_index(["code","insurance"]).index.map(code_med_
    li2["unit_ratio_global"] = li2["unit_price"] / li2["code_med_global"].replace
    li2["unit_ratio_ins"] = li2["unit_price"] / li2["code_med_ins"].replace(0, np
    for col in ["unit_ratio_global","unit_ratio_ins"]:
        li2[col] = li2[col].replace([np.inf, -np.inf], np.nan)
    # weighted averages + distribution stats
    def agg_ratio(df, col):
        w = df["line_total"].fillna(0.0).values
        x = df[col].fillna(np.nan).values
        good = np.isfinite(x) & np.isfinite(w) & (w > 0)
        if good.sum() == 0:
            return pd.Series({"wmean":1.0, "mean":1.0, "std":0.0, "min":1.0, "max
        ww = w[good]; xx = x[good]
        wmean = float(np.sum(ww*xx) / (np.sum(ww)+EPS))
        return pd.Series({
            "wmean": wmean,
            "mean": float(np.mean(xx)),
            "std": float(np.std(xx)),
            "min": float(np.min(xx)),
            "max": float(np.max(xx)),
            "n": int(len(xx)),
        })
    g1 = li2.groupby("patient_id", observed=True).apply(lambda d: agg_ratio(d, "u
    g1.columns = [f"pi_global_{c}" for c in g1.columns]
    g2 = li2.groupby("patient_id", observed=True).apply(lambda d: agg_ratio(d, "u
    g2.columns = [f"pi_ins_{c}" for c in g2.columns]


# --- PAGE 246 ---

    pi_feat = pd.concat([g1, g2], axis=1).reset_index().rename(columns={"patient_
    receipt_feat = receipt_feat.merge(pi_feat, on="patient_id", how="left")
    for c in ["pi_global_wmean","pi_ins_wmean","pi_global_mean","pi_ins_mean","pi
        if c in receipt_feat.columns:
            receipt_feat[c] = receipt_feat[c].fillna(1.0 if "wmean" in c or "mean
    receipt_feat["log_pi_global_wmean"] = np.log(receipt_feat["pi_global_wmean"].
    receipt_feat["log_pi_ins_wmean"] = np.log(receipt_feat["pi_ins_wmean"].astype
    # ---------- Start assembling all features ----------
    all_df = pd.concat([
        train.assign(__is_train__=1),
        test.assign(__is_train__=0, ed_cost_next3y_usd=np.nan)
    ], axis=0, ignore_index=True)
    all_df = all_df.merge(receipt_feat, on="patient_id", how="left")
    # fill missing numeric receipts
    for c in all_df.columns:
        if c.startswith(("share_","has_","cnt_","pi_","log_pi_")) or c in ["n_lin
            all_df[c] = pd.to_numeric(all_df[c], errors="coerce").fillna(0.0)
    # derived tabular
    all_df["log_prior_cost"] = np.log1p(all_df["prior_ed_cost_5y_usd"].astype(flo
    all_df["log_prior_visits"] = np.log1p(all_df["prior_ed_visits_5y"].astype(flo
    all_df["prior_cost_per_visit"] = all_df["prior_ed_cost_5y_usd"].astype(float)
    all_df["prior_visits_zero"] = (all_df["prior_ed_visits_5y"] == 0).astype(int)
    all_df["prior_cost_floor"] = (all_df["prior_ed_cost_5y_usd"] <= 50).astype(in
    # deflated prior costs (global + insurance-adjusted)
    all_df["prior_cost_deflated_global"] = all_df["prior_ed_cost_5y_usd"].astype(
    all_df["prior_cost_deflated_ins"] = all_df["prior_ed_cost_5y_usd"].astype(flo
    all_df["log_prior_cost_deflated_global"] = np.log1p(all_df["prior_cost_deflat
    all_df["log_prior_cost_deflated_ins"] = np.log1p(all_df["prior_cost_deflated_
    # low price flags learned on TRAIN distribution
    p05_ins = np.quantile(all_df.loc[all_df["__is_train__"]==1, "pi_ins_wmean"].a
    p05_glb = np.quantile(all_df.loc[all_df["__is_train__"]==1, "pi_global_wmean"


# --- PAGE 247 ---

    all_df["low_price_ins"] = (all_df["pi_ins_wmean"].astype(float) < p05_ins).as
    all_df["low_price_global"] = (all_df["pi_global_wmean"].astype(float) < p05_g
    # interactions capturing the "cheap past but severe receipt" phenotype
    all_df["floor_x_lowprice_ins"] = all_df["prior_cost_floor"] * all_df["low_pri
    all_df["floor_x_crit"] = all_df["prior_cost_floor"] * all_df["has_crit"]
    all_df["floor_x_severe"] = all_df["prior_cost_floor"] * all_df["has_severe_pr
    # ---------- Admissions features ----------
    if ADMISSIONS_TRAIN.exists() and ADMISSIONS_TEST.exists():
        adm_tr = pd.read_csv(ADMISSIONS_TRAIN)
        adm_te = pd.read_csv(ADMISSIONS_TEST)
        adm = pd.concat([adm_tr, adm_te], axis=0, ignore_index=True)
        adm["patient_id"] = adm["patient_id"].astype(int)
        # drop labels if present
        for col in ["readmit_30d"]:
            if col in adm.columns:
                adm = adm.drop(columns=[col])
        # numeric coercions
        for col in ["los_days","acuity_emergent","charlson_band","ed_visits_6m","
            if col in adm.columns:
                adm[col] = pd.to_numeric(adm[col], errors="coerce")
        agg = {
            "admission_id": "count",
            "los_days": ["mean","max","std"],
            "acuity_emergent": ["mean"],
            "charlson_band": ["mean","max"],
            "ed_visits_6m": ["mean","max"],
        }
        adm_agg = adm.groupby("patient_id").agg(agg)
        adm_agg.columns = ["adm_" + "_".join([a for a in col if a]) for col in ad
        adm_agg = adm_agg.reset_index()
        # dx counts
        if "primary_dx" in adm.columns:
            dx_ct = pd.crosstab(adm["patient_id"], adm["primary_dx"])
            dx_ct.columns = [f"adm_dx_{c}" for c in dx_ct.columns.astype(str)]


# --- PAGE 248 ---

            dx_ct = dx_ct.reset_index()
            adm_agg = adm_agg.merge(dx_ct, on="patient_id", how="left")
        all_df = all_df.merge(adm_agg, on="patient_id", how="left")
        print("[info] added admissions aggregates")
    else:
        print("[warn] admissions files not found; skipping admissions features")
    # ---------- Discharge note text ----------
    if USE_TEXT_FEATURES and DISCHARGE_NOTES.exists() and ADMISSIONS_TRAIN.exists
        # need admissions to map admission_id -> patient_id
        adm_tr = pd.read_csv(ADMISSIONS_TRAIN)
        adm_te = pd.read_csv(ADMISSIONS_TEST) if ADMISSIONS_TEST.exists() else pd
        adm_map = pd.concat([adm_tr[["admission_id","patient_id"]], adm_te[["admi
        adm_map["patient_id"] = adm_map["patient_id"].astype(int)
        notes = safe_read_json(DISCHARGE_NOTES)
        notes_df = pd.DataFrame(notes)
        if not notes_df.empty and "admission_id" in notes_df.columns:
            notes_df["admission_id"] = notes_df["admission_id"].astype(int)
            notes_df = notes_df.merge(adm_map, on="admission_id", how="left")
            notes_df["note"] = notes_df["note"].fillna("").astype(str)
            # patient-level concatenation
            pat_note = notes_df.groupby("patient_id")["note"].apply(lambda x: " "
            pat_note["note_len"] = pat_note["note"].str.len()
            # align to all_df patients
            pat_note = all_df[["patient_id"]].merge(pat_note, on="patient_id", ho
            pat_note["note"] = pat_note["note"].fillna("")
            pat_note["note_len"] = pat_note["note_len"].fillna(0.0)
            # TFIDF+SVD
            svd_df = add_tfidf_svd(pat_note["note"], prefix="disch", max_features
            svd_df["patient_id"] = pat_note["patient_id"].values
            svd_df["disch_note_len"] = pat_note["note_len"].values
            all_df = all_df.merge(svd_df, on="patient_id", how="left")
            print("[info] added discharge note TFIDF-SVD features")
        else:


# --- PAGE 249 ---

            print("[warn] discharge_notes.json empty or unexpected format; skippi
    else:
        print("[warn] discharge notes not used (missing file or USE_TEXT_FEATURES
    # ---------- Stays aggregates ----------
    if STAYS_TRAIN.exists() and STAYS_TEST.exists():
        st_tr = pd.read_csv(STAYS_TRAIN)
        st_te = pd.read_csv(STAYS_TEST)
        st = pd.concat([st_tr, st_te], axis=0, ignore_index=True)
        st["patient_id"] = st["patient_id"].astype(int)
        for col in ["discharge_ready_day11"]:
            if col in st.columns:
                st = st.drop(columns=[col])
        st_agg = st.groupby("patient_id").size().rename("stays_n").reset_index()
        all_df = all_df.merge(st_agg, on="patient_id", how="left")
        if "unit_type" in st.columns:
            u = pd.crosstab(st["patient_id"], st["unit_type"])
            u.columns = [f"stays_unit_{c}" for c in u.columns.astype(str)]
            all_df = all_df.merge(u.reset_index(), on="patient_id", how="left")
        if "admission_reason" in st.columns:
            r = pd.crosstab(st["patient_id"], st["admission_reason"])
            r.columns = [f"stays_reason_{c}" for c in r.columns.astype(str)]
            all_df = all_df.merge(r.reset_index(), on="patient_id", how="left")
        print("[info] added stays aggregates")
    else:
        print("[warn] stays files not found; skipping stays features")
    # ---------- Vitals timeseries aggregates ----------
    if VITALS_JSON.exists() and STAYS_TRAIN.exists():
        vit = safe_read_json(VITALS_JSON)
        # Build stay-level stats
        rows = []
        for obj in vit:
            sid = obj.get("stay_id", None)
            days = obj.get("days", [])


# --- PAGE 250 ---

            if sid is None or not isinstance(days, list) or len(days)==0:
            # extract arrays
            def arr(k):
                vals = [d.get(k, np.nan) for d in days]
                return np.array(pd.to_numeric(pd.Series(vals), errors="coerce").f
            hr = arr("hr"); sbp = arr("sbp"); dbp = arr("dbp"); temp = arr("temp_
            note = " ".join([str(d.get("note","")) for d in days if d.get("note")
            def stats(x):
                x = x[np.isfinite(x)]
                if len(x)==0:
                    return (np.nan,np.nan,np.nan,np.nan)
                return (float(np.mean(x)), float(np.std(x)), float(np.min(x)), fl
            hr_m, hr_s, hr_min, hr_max = stats(hr)
            sbp_m, sbp_s, sbp_min, sbp_max = stats(sbp)
            dbp_m, dbp_s, dbp_min, dbp_max = stats(dbp)
            t_m, t_s, t_min, t_max = stats(temp)
            rr_m, rr_s, rr_min, rr_max = stats(rr)
            # abnormal day counts (simple clinically meaningful thresholds)
            fever = int(np.sum(np.isfinite(temp) & (temp >= 38.0)))
            hypot = int(np.sum(np.isfinite(sbp) & (sbp < 90.0)))
            tachy = int(np.sum(np.isfinite(hr) & (hr > 100.0)))
            tachyp = int(np.sum(np.isfinite(rr) & (rr > 20.0)))
            rows.append({
                "stay_id": int(sid),
                "hr_mean": hr_m, "hr_std": hr_s, "hr_min": hr_min, "hr_max": hr_m
                "sbp_mean": sbp_m, "sbp_std": sbp_s, "sbp_min": sbp_min, "sbp_max
                "dbp_mean": dbp_m, "dbp_std": dbp_s, "dbp_min": dbp_min, "dbp_max
                "temp_mean": t_m, "temp_std": t_s, "temp_min": t_min, "temp_max":
                "rr_mean": rr_m, "rr_std": rr_s, "rr_min": rr_min, "rr_max": rr_m
                "days_fever": fever, "days_hypotension": hypot, "days_tachy": tac
                "vitals_note": note,
                "vitals_note_len": len(note)
            })


# --- PAGE 251 ---

        vit_df = pd.DataFrame(rows)
        # map stay_id -> patient_id
        st_tr = pd.read_csv(STAYS_TRAIN)[["stay_id","patient_id"]]
        st_te = pd.read_csv(STAYS_TEST)[["stay_id","patient_id"]] if STAYS_TEST.e
        st_map = pd.concat([st_tr, st_te], axis=0).drop_duplicates("stay_id")
        st_map["stay_id"] = st_map["stay_id"].astype(int)
        st_map["patient_id"] = st_map["patient_id"].astype(int)
        vit_df = vit_df.merge(st_map, on="stay_id", how="left")
        vit_df = vit_df.dropna(subset=["patient_id"])
        vit_df["patient_id"] = vit_df["patient_id"].astype(int)
        # aggregate to patient
        num_cols = [c for c in vit_df.columns if c not in ["stay_id","patient_id"
        agg = {}
        for c in num_cols:
            if c.startswith("days_"):
                agg[c] = "sum"
            else:
                agg[c] = "mean"
        vit_agg = vit_df.groupby("patient_id").agg(agg).reset_index()
        vit_agg.columns = ["patient_id"] + [f"v_{c}" for c in vit_agg.columns[1:]
        all_df = all_df.merge(vit_agg, on="patient_id", how="left")
        print("[info] added vitals numeric aggregates")
        # vitals note text
        if USE_TEXT_FEATURES:
            pat_vnote = vit_df.groupby("patient_id")["vitals_note"].apply(lambda
            pat_vnote = all_df[["patient_id"]].merge(pat_vnote, on="patient_id",
            pat_vnote["vitals_note"] = pat_vnote["vitals_note"].fillna("")
            svd_v = add_tfidf_svd(pat_vnote["vitals_note"], prefix="vnote", max_f
            svd_v["patient_id"] = pat_vnote["patient_id"].values
            all_df = all_df.merge(svd_v, on="patient_id", how="left")
            print("[info] added vitals note TFIDF-SVD features")
    else:
        print("[warn] vitals_timeseries.json or stays mapping not found; skipping
    # fill remaining NaNs numeric with 0
    for c in all_df.columns:


# --- PAGE 252 ---

        if c not in ["primary_chronic","sex","insurance","zip3","prior_cost_decil
            if all_df[c].dtype.kind in "biufc":
                all_df[c] = all_df[c].replace([np.inf, -np.inf], np.nan).fillna(0
    # handle rare zip3 (reduce overfit)
    zip_counts = all_df.loc[all_df["__is_train__"]==1, "zip3"].value_counts()
    rare = set(zip_counts[zip_counts < 10].index.tolist())
    all_df["zip3"] = all_df["zip3"].apply(lambda z: "RARE" if z in rare else z)
    # decile based on train only
    tr_cost = all_df.loc[all_df["__is_train__"]==1, "prior_ed_cost_5y_usd"].astyp
    qs = np.quantile(tr_cost, np.linspace(0,1,11))
    bins = np.unique(qs)
    if len(bins) < 3:
        bins = np.array([tr_cost.min()-1e-9, tr_cost.max()+1e-9])
    all_df["prior_cost_decile"] = pd.cut(all_df["prior_ed_cost_5y_usd"].astype(fl
    all_df["prior_cost_decile"] = all_df["prior_cost_decile"].fillna(-1).astype(i
    # categoricals
    CAT_COLS = ["primary_chronic","sex","insurance","zip3","prior_cost_decile"]
    for c in CAT_COLS:
        all_df[c] = all_df[c].astype(str).replace({"nan":"missing","None":"missin
        cats = sorted(all_df[c].unique().tolist())
        all_df[c] = pd.Categorical(all_df[c], categories=cats)
    # split back
    train_feat = all_df[all_df["__is_train__"]==1].reset_index(drop=True)
    test_feat  = all_df[all_df["__is_train__"]==0].reset_index(drop=True)
    # feature list
    drop_cols = {"ed_cost_next3y_usd","__is_train__"}
    FEATURES = [c for c in train_feat.columns if c not in drop_cols and c != "pat
    dump({"train_feat": train_feat, "test_feat": test_feat, "FEATURES": FEATURES,
    print(f"[cache] wrote features to {FEATURE_CACHE}")
# =========================
# MODELING HELPERS
# =========================
y = train_feat["ed_cost_next3y_usd"].astype(float).values


# --- PAGE 253 ---

X = train_feat[FEATURES]
X_test = test_feat[FEATURES]
strata = train_feat["primary_chronic"].astype(str).values
outlier_mask = (train_feat["prior_ed_cost_5y_usd"] <= 50) & (train_feat["prior_ed
def fit_lgb_repeated():
    oof_list = []
    test_list = []
    maes = []
    out_maes = []
    for seed in REPEAT_SEEDS:
        kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        oof = np.zeros(len(X), dtype=float)
        test_pred_folds = []
        for fold, (tr, va) in enumerate(kf.split(X, strata), 1):
            model = lgb.LGBMRegressor(**LGB_PARAMS, random_state=seed + fold)
            model.fit(
                X.iloc[tr], y[tr],
                eval_set=[(X.iloc[va], y[va])],
                eval_metric="l1",
                categorical_feature=CAT_COLS,
                callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)]
            )
            p_va = model.predict(X.iloc[va])
            oof[va] = p_va
            test_pred_folds.append(model.predict(X_test))
        test_pred = np.mean(np.vstack(test_pred_folds), axis=0)
        mae = mean_absolute_error(y, oof)
        out_mae = mean_absolute_error(y[outlier_mask.values], oof[outlier_mask.va
        print(f"[LGB seed={seed}] OOF MAE={mae:,.3f} | out_MAE={out_mae:,.3f}")
        oof_list.append(oof)
        test_list.append(test_pred)


# --- PAGE 254 ---

        maes.append(mae)
        out_maes.append(out_mae)
    oof_arr = np.vstack(oof_list)
    test_arr = np.vstack(test_list)
    # choose mean vs median aggregator by OOF
    oof_mean = oof_arr.mean(axis=0); oof_med = np.median(oof_arr, axis=0)
    mae_mean = mean_absolute_error(y, oof_mean); mae_med = mean_absolute_error(y,
    choose_med = mae_med < mae_mean
    test_pred = np.median(test_arr, axis=0) if choose_med else test_arr.mean(axis
    oof_pred  = oof_med if choose_med else oof_mean
    print(f"[LGB] repeats MAE list: {np.round(maes,3).tolist()}")
    print(f"[LGB] ensemble={'MEDIAN' if choose_med else 'MEAN'} | OOF MAE={min(ma
    return oof_pred, test_pred
def try_fit_catboost_repeated():
    if not TRY_CATBOOST:
        return None, None, False
    try:
        from catboost import CatBoostRegressor, Pool
    except Exception as e:
        print("[catboost] not installed; skipping. (pip install catboost)")
        return None, None, False
    # CatBoost wants raw strings for categoricals
    X_cb = X.copy()
    Xte_cb = X_test.copy()
    for c in CAT_COLS:
        X_cb[c] = X_cb[c].astype(str)
        Xte_cb[c] = Xte_cb[c].astype(str)
    cat_idx = [X_cb.columns.get_loc(c) for c in CAT_COLS]
    oof_list = []
    test_list = []
    maes = []


# --- PAGE 255 ---

    for seed in REPEAT_SEEDS:
        kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        oof = np.zeros(len(X_cb), dtype=float)
        test_pred_folds = []
        for fold, (tr, va) in enumerate(kf.split(X_cb, strata), 1):
            train_pool = Pool(X_cb.iloc[tr], y[tr], cat_features=cat_idx)
            val_pool   = Pool(X_cb.iloc[va], y[va], cat_features=cat_idx)
            model = CatBoostRegressor(
                loss_function="MAE",
                eval_metric="MAE",
                depth=8,
                learning_rate=0.05,
                iterations=20000,
                random_seed=seed + fold,
                l2_leaf_reg=6.0,
                subsample=0.85,
                rsm=0.85,
                verbose=False,
            )
            model.fit(train_pool, eval_set=val_pool, use_best_model=True, early_s
            p_va = model.predict(X_cb.iloc[va])
            oof[va] = p_va
            test_pred_folds.append(model.predict(Xte_cb))
        test_pred = np.mean(np.vstack(test_pred_folds), axis=0)
        mae = mean_absolute_error(y, oof)
        print(f"[CB seed={seed}] OOF MAE={mae:,.3f}")
        oof_list.append(oof)
        test_list.append(test_pred)
        maes.append(mae)
    oof_arr = np.vstack(oof_list)
    test_arr = np.vstack(test_list)
    oof_mean = oof_arr.mean(axis=0); oof_med = np.median(oof_arr, axis=0)


# --- PAGE 256 ---

    mae_mean = mean_absolute_error(y, oof_mean); mae_med = mean_absolute_error(y,
    choose_med = mae_med < mae_mean
    test_pred = np.median(test_arr, axis=0) if choose_med else test_arr.mean(axis
    oof_pred  = oof_med if choose_med else oof_mean
    print(f"[CB] repeats MAE list: {np.round(maes,3).tolist()}")
    print(f"[CB] ensemble={'MEDIAN' if choose_med else 'MEAN'} | OOF MAE={min(mae
    return oof_pred, test_pred, True
# =========================
# TRAIN + ENSEMBLE
# =========================
lgb_oof, lgb_test = fit_lgb_repeated()
cb_oof, cb_test, cb_ok = try_fit_catboost_repeated()
best_pred_test = lgb_test.copy()
best_name = "LGB_only"
best_oof = lgb_oof.copy()
best_mae = mean_absolute_error(y, best_oof)
if cb_ok:
    # weight search on OOF MAE
    best_w = None
    best_m = 1e18
    for w in np.linspace(0, 1, 51):  # step 0.02
        mix = w*lgb_oof + (1-w)*cb_oof
        m = mean_absolute_error(y, mix)
        if m < best_m:
            best_m = m
            best_w = w
    mix_test = best_w*lgb_test + (1-best_w)*cb_test
    print(f"[ensemble] best weight w(LGB)={best_w:.2f} -> OOF MAE={best_m:,.3f}")
    if best_m < best_mae:
        best_mae = best_m
        best_name = f"Ensemble_LGB_CB_w{best_w:.2f}"
        best_pred_test = mix_test
        best_oof = best_w*lgb_oof + (1-best_w)*cb_oof


# --- PAGE 257 ---

2. Compare:
distribution (feature shift was the main issue).
stress split (e.g., GroupKFold by ZIP3) and tune for robustness.
```

### C12 — Code cell starting at page 261
```python
# =========================
# EDA 2.0 — Cross-task alignment & incremental signal for ED Cost forecasting
# Single-cell, copy-paste runnable
# =========================
from pathlib import Path
import json
import re
import warnings
from collections import Counter
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)
# -------------------------
# PATHS (edit BASE_DIR if needed)
# -------------------------
BASE_DIR = Path(r"D:\AgentDs\agent_ds_healthcare")
TRAIN_ED = BASE_DIR / "ed_cost_train.csv"


# --- PAGE 262 ---

TEST_ED  = BASE_DIR / "ed_cost_test.csv"
PATIENTS = BASE_DIR / "patients.csv"
ADM_TRAIN = BASE_DIR / "admissions_train.csv"
ADM_TEST  = BASE_DIR / "admissions_test.csv"
DISCH_NOTES = BASE_DIR / "discharge_notes.json"
STAYS_TRAIN = BASE_DIR / "stays_train.csv"
STAYS_TEST  = BASE_DIR / "stays_test.csv"
VITALS_JSON = BASE_DIR / "vitals_timeseries.json"
CACHE_DIR = BASE_DIR / "cache_iter20_eda_cross"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = CACHE_DIR / "eda_outputs_cross2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_CROSS_PATIENT = CACHE_DIR / "cross_patient_features.joblib"
CACHE_VITALS_STAY   = CACHE_DIR / "vitals_stay_features.joblib"
print(f"[paths] BASE_DIR={BASE_DIR}")
print(f"[paths] OUT_DIR={OUT_DIR}")
# -------------------------
# Helpers
# -------------------------
def read_csv_safe(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        print(f"[WARN] Missing file: {path}")
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)
def load_json_list(path: Path):
    if not path.exists():
        print(f"[WARN] Missing file: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
def mode_or_nan(s: pd.Series):


# --- PAGE 263 ---

    s = s.dropna()
    if s.empty:
        return np.nan
    # if string-like, keep as string
    vc = s.astype(str).value_counts()
    return vc.index[0] if len(vc) else np.nan
def ks_statistic(x: np.ndarray, y: np.ndarray):
    # Train/test shift signal; uses scipy if available
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) < 20 or len(y) < 20:
        return np.nan
    try:
        from scipy.stats import ks_2samp
        return float(ks_2samp(x, y).statistic)
    except Exception:
        # fallback: max absolute quantile gap on coarse grid
        qs = np.linspace(0.05, 0.95, 19)
        qx = np.quantile(x, qs)
        qy = np.quantile(y, qs)
        denom = np.nanstd(np.concatenate([x, y])) + 1e-9
        return float(np.nanmax(np.abs(qx - qy)) / denom)
def spearman_corr(a: pd.Series, b: pd.Series):
    return float(a.corr(b, method="spearman"))
# -------------------------
# Load ED cost + patients
# -------------------------
patients = read_csv_safe(PATIENTS, dtype={"zip3": "string"})
ed_tr = read_csv_safe(TRAIN_ED)
ed_te = read_csv_safe(TEST_ED)
assert not ed_tr.empty and not ed_te.empty and not patients.empty, "Missing core
# Ensure types
patients["patient_id"] = patients["patient_id"].astype(int)
ed_tr["patient_id"] = ed_tr["patient_id"].astype(int)
ed_te["patient_id"] = ed_te["patient_id"].astype(int)


# --- PAGE 264 ---

# Join
train_full = ed_tr.merge(patients, on="patient_id", how="left")
test_full  = ed_te.merge(patients, on="patient_id", how="left")
print(f"[info] train_full shape: {train_full.shape}  test_full shape: {test_full.
assert train_full["age"].isna().sum() == 0, "Some train patients missing from pat
assert test_full["age"].isna().sum() == 0, "Some test patients missing from patie
all_patient_ids = set(pd.concat([train_full["patient_id"], test_full["patient_id"
# -------------------------
# Admissions patient-level features + diagnosis alignment
# -------------------------
adm_tr = read_csv_safe(ADM_TRAIN)
adm_te = read_csv_safe(ADM_TEST)
if not adm_tr.empty or not adm_te.empty:
    # Harmonize
    if not adm_tr.empty:
        adm_tr["split"] = "train"
    if not adm_te.empty:
        adm_te["split"] = "test"
    adm = pd.concat([adm_tr, adm_te], ignore_index=True)
    adm["patient_id"] = adm["patient_id"].astype(int)
    # Restrict to ED-cost patients for EDA (important: avoid mixing with extra pa
    adm = adm[adm["patient_id"].isin(all_patient_ids)].copy()
    # Basic patient aggregates
    g = adm.groupby("patient_id", dropna=False)
    adm_feat = pd.DataFrame({
        "patient_id": g.size().index.astype(int),
        "adm_n": g.size().values.astype(int),
        "adm_emergent_n": g["acuity_emergent"].sum().values if "acuity_emergent"
        "adm_emergent_rate": g["acuity_emergent"].mean().values if "acuity_emerge
        "adm_los_mean": g["los_days"].mean().values if "los_days" in adm.columns
        "adm_los_max": g["los_days"].max().values if "los_days" in adm.columns el
        "adm_los_sum": g["los_days"].sum().values if "los_days" in adm.columns el


# --- PAGE 265 ---

        "adm_charlson_mean": g["charlson_band"].mean().values if "charlson_band"
        "adm_charlson_max": g["charlson_band"].max().values if "charlson_band" in
        "adm_edvis6m_mean": g["ed_visits_6m"].mean().values if "ed_visits_6m" in
        "adm_edvis6m_max": g["ed_visits_6m"].max().values if "ed_visits_6m" in ad
        "adm_discharge_weekday_mode": g["discharge_weekday"].apply(mode_or_nan).v
        "adm_primary_dx_mode": g["primary_dx"].apply(mode_or_nan).values if "prim
        "adm_primary_dx_nunique": g["primary_dx"].nunique().values if "primary_dx
    })
    # dx mix counts (HF/Pneumonia/DiabetesComp)
    if "primary_dx" in adm.columns:
        dx_counts = pd.crosstab(adm["patient_id"], adm["primary_dx"])
        dx_counts.columns = [f"adm_dx_cnt_{c}" for c in dx_counts.columns]
        dx_counts = dx_counts.reset_index().rename(columns={"patient_id": "patient
        dx_counts["patient_id"] = dx_counts["patient_id"].astype(int)
        adm_feat = adm_feat.merge(dx_counts, on="patient_id", how="left")
else:
    adm_feat = pd.DataFrame({"patient_id": list(all_patient_ids)})
    print("[WARN] admissions_train/test not found or empty; skipping admissions f
# -------------------------
# Discharge note patient-level features (via admission_id)
# -------------------------
notes_list = load_json_list(DISCH_NOTES)
if notes_list and (not adm_tr.empty or not adm_te.empty):
    notes_df = pd.DataFrame(notes_list)
    # expected keys: admission_id, note
    if "admission_id" in notes_df.columns and "note" in notes_df.columns:
        notes_df["admission_id"] = notes_df["admission_id"].astype(int)
        # link to patient via admissions table (use full admissions table, then f
        adm_link = pd.concat([adm_tr, adm_te], ignore_index=True)
        adm_link["admission_id"] = adm_link["admission_id"].astype(int)
        adm_link["patient_id"] = adm_link["patient_id"].astype(int)
        adm_link = adm_link[adm_link["patient_id"].isin(all_patient_ids)][["admis
        notes_join = notes_df.merge(adm_link, on="admission_id", how="inner")
        # Simple text features
        txt = notes_join["note"].fillna("").astype(str)
        notes_join["note_len_chars"] = txt.str.len()
        notes_join["note_len_words"] = txt.str.split().str.len()


# --- PAGE 266 ---

        # Domain-ish keywords (lightweight)
        KEYWORDS = {
            "kw_followup": r"\bfollow[- ]?up\b|\bf/u\b",
            "kw_home": r"\bhome\b|\bdischarge home\b",
            "kw_snf_rehab": r"\bsnf\b|\bskilled nursing\b|\brehab\b",
            "kw_oxygen": r"\boxy(gen)?\b|\bO2\b",
            "kw_insulin": r"\binsulin\b",
            "kw_dialysis": r"\bdialysis\b|\bhd\b|\bhemodialysis\b",
            "kw_wound": r"\bwound\b|\bulcer\b",
            "kw_sob": r"\bshortness of breath\b|\bsob\b|\bdyspnea\b",
            "kw_chest_pain": r"\bchest pain\b",
            "kw_sepsis": r"\bsepsis\b|\bseptic\b",
            "kw_pneumonia": r"\bpneumonia\b",
            "kw_heart_failure": r"\bheart failure\b|\bCHF\b",
            "kw_diabetes": r"\bdiabetes\b|\bDKA\b|\bhypogly\b",
        }
        for col, pat in KEYWORDS.items():
            notes_join[col] = txt.str.contains(pat, flags=re.IGNORECASE, regex=Tr
        ng = notes_join.groupby("patient_id", dropna=False)
        note_feat = pd.DataFrame({
            "patient_id": ng.size().index.astype(int),
            "note_n": ng.size().values.astype(int),
            "note_len_chars_mean": ng["note_len_chars"].mean().values,
            "note_len_chars_max": ng["note_len_chars"].max().values,
            "note_len_words_mean": ng["note_len_words"].mean().values,
            "note_len_words_max": ng["note_len_words"].max().values,
        })
        for col in KEYWORDS.keys():
            note_feat[f"{col}_sum"] = ng[col].sum().values
            note_feat[f"{col}_rate"] = ng[col].mean().values
    else:
        note_feat = pd.DataFrame({"patient_id": list(all_patient_ids)})
        print("[WARN] discharge_notes.json missing expected keys; skipping.")
else:
    note_feat = pd.DataFrame({"patient_id": list(all_patient_ids)})
    if not notes_list:
        print("[WARN] discharge_notes.json missing/empty; skipping notes features
    else:


# --- PAGE 267 ---

        print("[WARN] admissions not available, cannot link notes by admission_id
# -------------------------
# Stays + vitals patient-level features
# -------------------------
st_tr = read_csv_safe(STAYS_TRAIN)
st_te = read_csv_safe(STAYS_TEST)
if not st_tr.empty or not st_te.empty:
    if not st_tr.empty:
        st_tr["split"] = "train"
    if not st_te.empty:
        st_te["split"] = "test"
    stays = pd.concat([st_tr, st_te], ignore_index=True)
    stays["patient_id"] = stays["patient_id"].astype(int)
    stays["stay_id"] = stays["stay_id"].astype(int)
    stays = stays[stays["patient_id"].isin(all_patient_ids)].copy()
    sg = stays.groupby("patient_id", dropna=False)
    stays_feat = pd.DataFrame({
        "patient_id": sg.size().index.astype(int),
        "stay_n": sg.size().values.astype(int),
        "stay_unit_mode": sg["unit_type"].apply(mode_or_nan).values if "unit_type
        "stay_reason_mode": sg["admission_reason"].apply(mode_or_nan).values if "
        "stay_reason_nunique": sg["admission_reason"].nunique().values if "admiss
    })
    # reason mix
    if "admission_reason" in stays.columns:
        reason_counts = pd.crosstab(stays["patient_id"], stays["admission_reason"
        reason_counts.columns = [f"stay_reason_cnt_{c}" for c in reason_counts.co
        reason_counts = reason_counts.reset_index().rename(columns={"patient_id":
        reason_counts["patient_id"] = reason_counts["patient_id"].astype(int)
        stays_feat = stays_feat.merge(reason_counts, on="patient_id", how="left")
else:
    stays = pd.DataFrame()
    stays_feat = pd.DataFrame({"patient_id": list(all_patient_ids)})
    print("[WARN] stays_train/test not found or empty; skipping stays/vitals feat
# Compute vitals stay-level features then aggregate to patient
if not stays.empty and VITALS_JSON.exists():
    try:


# --- PAGE 268 ---

        from joblib import dump, load
        if CACHE_VITALS_STAY.exists():
            vitals_stay = load(CACHE_VITALS_STAY)
            print(f"[cache] Loaded vitals stay features: {CACHE_VITALS_STAY}")
        else:
            needed_stay_ids = set(stays["stay_id"].astype(int).tolist())
            vitals_list = load_json_list(VITALS_JSON)
            rows = []
            for rec in vitals_list:
                sid = int(rec.get("stay_id", -1))
                if sid not in needed_stay_ids:
                days = rec.get("days", []) or []
                if not days:
                def arr(key):
                    return np.array([d.get(key, np.nan) for d in days], dtype=flo
                hr = arr("hr"); sbp = arr("sbp"); dbp = arr("dbp"); temp = arr("t
                note_lens = np.array([len(str(d.get("note",""))) for d in days],
                def nanmean(x): return float(np.nanmean(x)) if np.isfinite(np.nan
                def nanstd(x):  return float(np.nanstd(x)) if np.isfinite(np.nans
                def nanmin(x):  return float(np.nanmin(x)) if np.isfinite(np.nanm
                def nanmax(x):  return float(np.nanmax(x)) if np.isfinite(np.nanm
                def slope(x):
                    # (day10 - day1) / (n-1)
                    x = x.copy()
                    if len(x) < 2 or np.isnan(x[0]) or np.isnan(x[-1]):
                        return np.nan
                    return float((x[-1] - x[0]) / max(1, (len(x) - 1)))
                row = {
                    "stay_id": sid,
                    "v_days": len(days),
                    "v_hr_mean": nanmean(hr), "v_hr_std": nanstd(hr), "v_hr_min":


# --- PAGE 269 ---

                    "v_sbp_mean": nanmean(sbp), "v_sbp_std": nanstd(sbp), "v_sbp_
                    "v_dbp_mean": nanmean(dbp), "v_dbp_std": nanstd(dbp), "v_dbp_
                    "v_temp_mean": nanmean(temp), "v_temp_std": nanstd(temp), "v_t
                    "v_rr_mean": nanmean(rr), "v_rr_std": nanstd(rr), "v_rr_min":
                    "v_note_len_mean": nanmean(note_lens),
                    "v_note_len_max": nanmax(note_lens),
                    # instability flags (counts)
                    "v_fever_days": int(np.nansum(temp >= 38.0)),
                    "v_hypotension_days": int(np.nansum(sbp < 90.0)),
                    "v_tachy_days": int(np.nansum(hr > 100.0)),
                    "v_tachypnea_days": int(np.nansum(rr > 20.0)),
                }
                rows.append(row)
            vitals_stay = pd.DataFrame(rows)
            dump(vitals_stay, CACHE_VITALS_STAY)
            print(f"[cache] Wrote vitals stay features: {CACHE_VITALS_STAY} | n_st
        # Link stay->patient and aggregate to patient
        vitals_join = vitals_stay.merge(stays[["stay_id", "patient_id"]].drop_dup
        vg = vitals_join.groupby("patient_id", dropna=False)
        agg_cols_mean = [c for c in vitals_join.columns if c.startswith("v_") and
        vitals_patient = pd.DataFrame({"patient_id": vg.size().index.astype(int)}
        vitals_patient["vitals_stay_n"] = vg.size().values.astype(int)
        # Aggregate: mean of means, max of maxima, etc (simple, robust)
        for c in agg_cols_mean:
            vitals_patient[f"{c}_mean_over_stays"] = vg[c].mean().values
            vitals_patient[f"{c}_max_over_stays"] = vg[c].max().values
    except Exception as e:
        print(f"[WARN] vitals feature extraction failed: {e}")
        vitals_patient = pd.DataFrame({"patient_id": list(all_patient_ids)})
else:
    vitals_patient = pd.DataFrame({"patient_id": list(all_patient_ids)})
    if not VITALS_JSON.exists():
        print("[WARN] vitals_timeseries.json not found; skipping vitals.")


# --- PAGE 270 ---

    elif stays.empty:
        print("[WARN] stays not available; skipping vitals.")
# -------------------------
# Combine cross features (patient-level)
# -------------------------
# Ensure every patient_id exists at least once
base_pid = pd.DataFrame({"patient_id": sorted(list(all_patient_ids))})
cross = base_pid.merge(adm_feat, on="patient_id", how="left") \
                .merge(note_feat, on="patient_id", how="left") \
                .merge(stays_feat, on="patient_id", how="left") \
                .merge(vitals_patient, on="patient_id", how="left")
# Coverage flags
cross["has_adm"] = (cross.get("adm_n", 0).fillna(0) > 0).astype(int)
cross["has_note"] = (cross.get("note_n", 0).fillna(0) > 0).astype(int)
cross["has_stay"] = (cross.get("stay_n", 0).fillna(0) > 0).astype(int)
cross["has_vitals"] = (cross.get("vitals_stay_n", 0).fillna(0) > 0).astype(int)
# Cache it
try:
    from joblib import dump
    dump(cross, CACHE_CROSS_PATIENT)
    print(f"[cache] Wrote cross patient features: {CACHE_CROSS_PATIENT}")
except Exception as e:
    print(f"[WARN] Could not cache cross patient features: {e}")
print(f"[info] cross features shape: {cross.shape}")
print("[info] Coverage among ED-cost patients (all):")
print(cross[["has_adm","has_note","has_stay","has_vitals"]].mean().to_frame("rate
# -------------------------
# Merge into ED-cost train/test
# -------------------------
trX = train_full.merge(cross, on="patient_id", how="left")
teX = test_full.merge(cross, on="patient_id", how="left")
# -------------------------
# Diagnosis alignment checks


# --- PAGE 271 ---

# -------------------------
def norm_dx(x):
    if pd.isna(x): return np.nan
    x = str(x)
    # normalize HF label differences if any
    if x.lower() in ["hf", "heart failure", "chf"]:
        return "HF"
    return x
trX["primary_chronic_norm"] = trX["primary_chronic"].map(norm_dx)
trX["adm_dx_mode_norm"] = trX.get("adm_primary_dx_mode", np.nan).map(norm_dx)
trX["stay_reason_mode_norm"] = trX.get("stay_reason_mode", np.nan).map(norm_dx)
adm_match_rate = np.nan
stay_match_rate = np.nan
if "adm_primary_dx_mode" in trX.columns:
    adm_match_rate = float((trX["adm_dx_mode_norm"] == trX["primary_chronic_norm"
if "stay_reason_mode" in trX.columns:
    stay_match_rate = float((trX["stay_reason_mode_norm"] == trX["primary_chronic_
print("\n=== DX ALIGNMENT (TRAIN) ===")
print(f"Admissions dx_mode matches ED primary_chronic: {adm_match_rate:.3f} (NaNs
print(f"Stays reason_mode matches ED primary_chronic: {stay_match_rate:.3f} (NaNs
# -------------------------
# Baseline residual for incremental-signal analysis
# Use deflated prior cost to stabilize extreme ratios
# -------------------------
y = trX["ed_cost_next3y_usd"].astype(float)
prior = trX["prior_ed_cost_5y_usd"].astype(float)
prior_def = prior.clip(lower=50.0)
# Create deciles on train prior_def (save bins, apply to test if needed)
dec, bins = pd.qcut(prior_def, 10, labels=False, retbins=True, duplicates="drop")
trX["cost_decile_def"] = dec.astype(int)
# Deflated ratio
trX["ratio_def"] = y / prior_def
med1 = trX.groupby(["primary_chronic","insurance","cost_decile_def"], dropna=Fals
med2 = trX.groupby(["primary_chronic","insurance"], dropna=False)["ratio_def"].me


# --- PAGE 272 ---

med3 = trX.groupby(["primary_chronic"], dropna=False)["ratio_def"].median().renam
overall_med = float(trX["ratio_def"].median())
tmp = trX.merge(med1, on=["primary_chronic","insurance","cost_decile_def"], how="
         .merge(med2, on=["primary_chronic","insurance"], how="left") \
         .merge(med3, on=["primary_chronic"], how="left")
tmp["med_ratio"] = tmp["med_ratio_1"].fillna(tmp["med_ratio_2"]).fillna(tmp["med_
tmp["baseline_pred_def"] = prior_def * tmp["med_ratio"]
tmp["baseline_resid"] = y - tmp["baseline_pred_def"]
trX["baseline_pred_def"] = tmp["baseline_pred_def"].values
trX["baseline_resid"] = tmp["baseline_resid"].values
print("\n=== BASELINE (deflated-ratio) CHECK ===")
mae_base = float(np.mean(np.abs(trX["baseline_resid"])))
print(f"Baseline MAE (train, in-sample): {mae_base:.3f}  (note: in-sample baselin
# Outlier definition (your earlier one)
trX["is_outlier"] = ((trX["prior_ed_cost_5y_usd"] <= 50.0) & (trX["prior_ed_visit
out_n = int(trX["is_outlier"].sum())
print(f"Outlier count (train): {out_n}")
# -------------------------
# Incremental signal: Spearman corr with baseline_resid
# -------------------------
# Consider only numeric cross features
exclude_cols = set([
])
num_cols = [c for c in trX.columns if c not in exclude_cols and pd.api.types.is_n
corr_rows = []
for c in num_cols:
    s = trX[c]


# --- PAGE 273 ---

    if s.isna().mean() > 0.95:
    corr_t = spearman_corr(s, trX["ed_cost_next3y_usd"])
    corr_r = spearman_corr(s, trX["baseline_resid"])
    corr_p = spearman_corr(s, trX["prior_ed_cost_5y_usd"])
    corr_rows.append((c, corr_t, corr_r, corr_p, float(s.isna().mean()), float(s.
corr_df = pd.DataFrame(corr_rows, columns=[
]).sort_values("spearman_with_baseline_resid", ascending=False)
print("\n=== TOP 25 Cross/aux numeric features by Spearman(residual) ===")
print(corr_df.head(25).to_string(index=False))
print("\n=== BOTTOM 25 Cross/aux numeric features by Spearman(residual) ===")
print(corr_df.tail(25).to_string(index=False))
corr_path = OUT_DIR / "cross_feature_correlations.csv"
corr_df.to_csv(corr_path, index=False)
print(f"[write] {corr_path}")
# -------------------------
# Train/test shift for cross features (KS)
# -------------------------
shift_rows = []
# numeric shift
for c in [c for c in cross.columns if pd.api.types.is_numeric_dtype(cross[c]) and
    trv = trX[c].astype(float).values
    tev = teX[c].astype(float).values
    ks = ks_statistic(trv, tev)
    shift_rows.append((c,
                       float(np.mean(np.isnan(trv))), float(np.mean(np.isnan(tev)
                       float(np.nanmean(trv)), float(np.nanmean(tev)),
                       float(np.nanstd(trv)), float(np.nanstd(tev)),
                       ks))
shift_df = pd.DataFrame(shift_rows, columns=[


# --- PAGE 274 ---

]).sort_values("ks_stat", ascending=False)
print("\n=== TOP 25 numeric cross features by train/test shift (KS) ===")
print(shift_df.head(25).to_string(index=False))
shift_path = OUT_DIR / "cross_feature_shift_ks.csv"
shift_df.to_csv(shift_path, index=False)
print(f"[write] {shift_path}")
# -------------------------
# Coverage effect on target/residual
# -------------------------
def summarize_by_flag(flag_col: str):
    if flag_col not in trX.columns:
        return None
    gb = trX.groupby(flag_col, dropna=False)
    out = gb.agg(
        n=("patient_id","count"),
        target_mean=("ed_cost_next3y_usd","mean"),
        target_median=("ed_cost_next3y_usd","median"),
        resid_mean=("baseline_resid","mean"),
        resid_median=("baseline_resid","median"),
        prior_mean=("prior_ed_cost_5y_usd","mean"),
    ).reset_index()
    return out
print("\n=== COVERAGE EFFECT (TRAIN) ===")
for flag in ["has_adm","has_note","has_stay","has_vitals"]:
    tab = summarize_by_flag(flag)
    if tab is not None:
        print(f"\n[{flag}]")
        print(tab.to_string(index=False))
# -------------------------
# Outlier-focused: do cross features correlate with residual inside outliers?
# -------------------------
if out_n > 0:
    out_df = trX[trX["is_outlier"] == 1].copy()


# --- PAGE 275 ---

    non_df = trX[trX["is_outlier"] == 0].copy()
    # pick top 15 residual-correlated features from full data, then compute withi
    top_feats = corr_df.head(15)["feature"].tolist()
    rows = []
    for c in top_feats:
        if c in out_df.columns and pd.api.types.is_numeric_dtype(out_df[c]):
            cr_out = spearman_corr(out_df[c], out_df["baseline_resid"])
            cr_non = spearman_corr(non_df[c], non_df["baseline_resid"])
            rows.append((c, cr_out, cr_non, float(out_df[c].isna().mean()), float
    outcorr = pd.DataFrame(rows, columns=["feature","spearman_resid_outliers","sp
    print("\n=== OUTLIER vs NON-OUTLIER residual correlation (top residual-correl
    print(outcorr.to_string(index=False))
    outcorr_path = OUT_DIR / "cross_feature_outlier_vs_nonoutlier_corr.csv"
    outcorr.to_csv(outcorr_path, index=False)
    print(f"[write] {outcorr_path}")
# -------------------------
# Save merged train/test with cross features (optional, for inspection)
# -------------------------
small_cols = ["patient_id","primary_chronic","age","sex","insurance","zip3","prio
small_cols = [c for c in small_cols if c in trX.columns]
tr_small = trX[small_cols].copy()
te_small = teX[[c for c in small_cols if c != "ed_cost_next3y_usd"]].copy()
tr_small_path = OUT_DIR / "train_cross_small_view.csv"
te_small_path = OUT_DIR / "test_cross_small_view.csv"
tr_small.to_csv(tr_small_path, index=False)
te_small.to_csv(te_small_path, index=False)
print(f"\n[write] {tr_small_path}")
print(f"[write] {te_small_path}")
print("\n[DONE] EDA 2.0 cross-task analysis complete.")


# --- PAGE 276 ---
```

### C13 — Code cell starting at page 286
```python
# --- PAGE 287 ---

# Iteration 21 (EDA) — Deflation reconstruction from receipts + Discharge-note te
# Copy/paste into ONE Jupyter cell.
from __future__ import annotations
import json, re, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=FutureWarning)
# ---------- deps ----------
import joblib
import pdfplumber
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
# ---------- paths ----------
BASE_DIR = Path(r"D:\AgentDs\agent_ds_healthcare")
TRAIN_CSV = BASE_DIR / "ed_cost_train.csv"
TEST_CSV  = BASE_DIR / "ed_cost_test.csv"
PATIENTS_CSV = BASE_DIR / "patients.csv"
PDF_DIR = BASE_DIR / "receipts_pdf"
ADMISSIONS_TRAIN = BASE_DIR / "admissions_train.csv"
ADMISSIONS_TEST  = BASE_DIR / "admissions_test.csv"
DISCHARGE_NOTES_JSON = BASE_DIR / "discharge_notes.json"
CACHE_DIR = BASE_DIR / "cache_iter21_eda_deflation_text"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = CACHE_DIR / "eda_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RECEIPT_CACHE = CACHE_DIR / "receipts_parsed.joblib"
RECEIPT_FEATS_CACHE = CACHE_DIR / "receipt_features.joblib"


# --- PAGE 288 ---

NOTE_FEATS_CACHE = CACHE_DIR / "note_patient_svd.joblib"
FORCE_REPARSE = False
FORCE_REBUILD_RECEIPT_FEATS = False
FORCE_REBUILD_NOTE_FEATS = False
# ---------- constants ----------
TARGET = "ed_cost_next3y_usd"
CAT_COLS = ["primary_chronic","sex","insurance","zip3"]
DEFLATED_PRIOR_COST_FLOOR = 50
UNIT_REF_TOP_Q = 0.70
DEFLATION_THRESH = 0.35
# code categories
ED_EM = {"99281","99282","99283","99284","99285"}
CRIT = {"99291","99292"}
LAB  = {"85025","87070","84484"}
IMAGING = {"71045","70450","74177"}
OBS = {"G0378"}
SEVERE_PROC = {"31500","36556","36620","92950"}
def code_category(code: str) -> str:
    if code in ED_EM: return "ed_em"
    if code in CRIT: return "crit_care"
    if code in LAB: return "lab"
    if code in IMAGING: return "imaging"
    if code in OBS: return "obs"
    if code in SEVERE_PROC: return "severe_proc"
    return "other"
# ---------- load tabular ----------
train = pd.read_csv(TRAIN_CSV)
test  = pd.read_csv(TEST_CSV)
patients = pd.read_csv(PATIENTS_CSV)
patients["patient_id"] = patients["patient_id"].astype(int)
train["patient_id"] = train["patient_id"].astype(int)
test["patient_id"]  = test["patient_id"].astype(int)
patients["zip3"] = patients["zip3"].astype(str).str.replace(r"\.0$","",regex=True


# --- PAGE 289 ---

train_full = train.merge(patients, on="patient_id", how="left")
test_full  = test.merge(patients, on="patient_id", how="left")
print(f"[info] train_full: {train_full.shape}  test_full: {test_full.shape}")
ALL_PIDS = sorted(set(train_full.patient_id) | set(test_full.patient_id))
# ---------- parse receipts (cached) ----------
CODE_LINE_RE = re.compile(r"^([A-Z0-9]+)\s+(.*?)\s+(\d+)\s+([\d,]+\.\d{2})\s+([\d
TOTAL_RE = re.compile(r"^TOTAL\s+([\d,]+\.\d{2})\s*$")
PID_RE = re.compile(r"Patient ID:\s*(\d+)")
ZIP_INS_RE = re.compile(r"ZIP3:\s*([0-9]{1,3})\s+Insurance:\s*([A-Za-z_]+)")
def parse_receipt_pdf(pdf_path: Path):
    with pdfplumber.open(str(pdf_path)) as pdf:
        text = "\n".join((p.extract_text() or "") for p in pdf.pages)
    pid = zip3 = ins = None
    total = None
    items = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
        if pid is None:
            m = PID_RE.search(line)
            if m:
                pid = int(m.group(1))
        if zip3 is None or ins is None:
            m = ZIP_INS_RE.search(line)
            if m:
                zip3 = str(m.group(1)).zfill(3)
                ins = m.group(2).lower()
        m = TOTAL_RE.match(line)
        if m:
            total = float(m.group(1).replace(",",""))
        m = CODE_LINE_RE.match(line)
        if m:


# --- PAGE 290 ---

            code, desc, qty, unit, tot = (
                m.group(1),
                m.group(2).strip(),
                int(m.group(3)),
                float(m.group(4).replace(",","")),
                float(m.group(5).replace(",","")),
            )
            items.append(dict(patient_id=pid, code=code, qty=qty, unit_price=unit
    return dict(patient_id=pid, zip3_pdf=zip3, insurance_pdf=ins, total_pdf=total
def load_or_parse_receipts():
    if RECEIPT_CACHE.exists() and not FORCE_REPARSE:
        d = joblib.load(RECEIPT_CACHE)
        if isinstance(d, dict) and "headers_df" in d and "lineitems_df" in d:
            return d["headers_df"], d["lineitems_df"]
    headers, items = [], []
    bad = 0
    for pid in ALL_PIDS:
        p = PDF_DIR / f"receipt_{pid}.pdf"
        try:
            hdr, its = parse_receipt_pdf(p)
            if hdr["patient_id"] is None or hdr["total_pdf"] is None or len(its)=
                bad += 1
            headers.append(hdr)
            items.extend(its)
        except Exception:
            bad += 1
    headers_df = pd.DataFrame(headers)
    lineitems_df = pd.DataFrame(items)
    joblib.dump({"headers_df": headers_df, "lineitems_df": lineitems_df}, RECEIPT_
    print(f"[cache] wrote {RECEIPT_CACHE} | bad={bad}")
    return headers_df, lineitems_df
headers_df, lineitems_df = load_or_parse_receipts()
lineitems_df["code"] = lineitems_df["code"].astype(str)
print(f"[info] parsed lineitems: {len(lineitems_df)} rows | patients w/ lines: {l
# ---------- receipt features (cached) ----------
def build_receipt_features(lineitems_df: pd.DataFrame, patients_df: pd.DataFrame)


# --- PAGE 291 ---

    li = lineitems_df.merge(patients_df[["patient_id","insurance"]], on="patient_
    li["insurance"] = li["insurance"].astype(str).str.lower()
    li["category"] = li["code"].map(code_category)
    base = li.groupby("patient_id", as_index=False).agg(
        pdf_n_line_items=("code","size"),
        pdf_n_unique_codes=("code","nunique"),
        pdf_total_line_cost=("line_total","sum"),
        pdf_unit_median=("unit_price","median"),
    )
    cat_cost = li.pivot_table(index="patient_id", columns="category", values="lin
    cat_share = cat_cost.div(cat_cost.sum(axis=1).replace(0,np.nan), axis=0).fill
    cat_share.columns = [f"pdf_share_{c}" for c in cat_share.columns]
    cat_share = cat_share.reset_index()
    code_cnt = li.pivot_table(index="patient_id", columns="code", values="qty", a
    code_cnt.columns = [f"pdf_cnt_{c}" for c in code_cnt.columns]
    code_cnt = code_cnt.reset_index()
    feats = base.merge(cat_share, on="patient_id", how="left").merge(code_cnt, on
    # ---- expected cost reconstruction (deflation) ----
    grp = li.groupby(["code","insurance"])["unit_price"]
    q = grp.quantile(UNIT_REF_TOP_Q).rename("q").reset_index()
    hi = li.merge(q, on=["code","insurance"], how="left")
    hi = hi[hi["unit_price"] >= hi["q"]]
    unit_ref = hi.groupby(["code","insurance"])["unit_price"].median().rename("un
    code_med = li.groupby("code")["unit_price"].median().rename("code_med").reset_
    unit_ref = unit_ref.merge(code_med, on="code", how="left")
    unit_ref["unit_ref"] = unit_ref["unit_ref"].fillna(unit_ref["code_med"])
    unit_ref = unit_ref.drop(columns=["code_med"])
    li2 = li.merge(unit_ref, on=["code","insurance"], how="left")
    li2["expected_line_total"] = li2["qty"] * li2["unit_ref"]
    exp = li2.groupby("patient_id", as_index=False)["expected_line_total"].sum().
    feats = feats.merge(exp, on="patient_id", how="left")


# --- PAGE 292 ---

    feats["pdf_deflation_factor"] = feats["pdf_total_line_cost"] / feats["pdf_exp
    feats["pdf_deflation_factor"] = feats["pdf_deflation_factor"].replace([np.inf
    feats["pdf_log_deflation_factor"] = np.log(feats["pdf_deflation_factor"].clip
    li2["is_deflated_line"] = (li2["unit_price"] < 0.5*li2["unit_ref"]).astype(int
    defl = li2.groupby("patient_id", as_index=False)["is_deflated_line"].mean().r
    feats = feats.merge(defl, on="patient_id", how="left").fillna(0.0)
    feats["pdf_is_deflated"] = ((feats["pdf_deflation_factor"] > 0) & (feats["pdf_
    return feats.fillna(0.0)
if RECEIPT_FEATS_CACHE.exists() and not FORCE_REBUILD_RECEIPT_FEATS:
    receipt_feats = joblib.load(RECEIPT_FEATS_CACHE)
    print(f"[cache] loaded {RECEIPT_FEATS_CACHE} | shape={receipt_feats.shape}")
else:
    receipt_feats = build_receipt_features(lineitems_df, patients)
    joblib.dump(receipt_feats, RECEIPT_FEATS_CACHE)
    print(f"[cache] wrote {RECEIPT_FEATS_CACHE} | shape={receipt_feats.shape}")
train_full = train_full.merge(receipt_feats, on="patient_id", how="left").fillna(
test_full  = test_full.merge(receipt_feats, on="patient_id", how="left").fillna(0
print("\n[check] |pdf_total_line_cost - prior_ed_cost_5y_usd| (train)")
print((train_full["pdf_total_line_cost"] - train_full["prior_ed_cost_5y_usd"]).ab
# ---------- discharge notes -> patient SVD features (cached) ----------
def build_note_patient_features() -> pd.DataFrame:
    adm_tr = pd.read_csv(ADMISSIONS_TRAIN)
    adm_te = pd.read_csv(ADMISSIONS_TEST)
    adm = pd.concat([adm_tr, adm_te], ignore_index=True)
    adm["patient_id"] = adm["patient_id"].astype(int)
    notes = pd.DataFrame(json.loads(DISCHARGE_NOTES_JSON.read_text()))
    notes["admission_id"] = notes["admission_id"].astype(int)
    adm = adm.merge(notes, on="admission_id", how="left")
    adm["note"] = adm["note"].fillna("").astype(str)
    pt = adm.groupby("patient_id", as_index=False).agg(
        note_text=("note", lambda s: " ".join(s.tolist())),
        note_n=("note","size"),


# --- PAGE 293 ---

        note_len_mean=("note", lambda s: float(np.mean([len(x) for x in s.tolist(
        note_len_max=("note", lambda s: float(np.max([len(x) for x in s.tolist()]
        adm_n=("admission_id","size"),
        adm_los_sum=("los_days","sum"),
        adm_charlson_max=("charlson_band","max"),
        adm_emergent_rate=("acuity_emergent","mean"),
        adm_edvis6m_mean=("ed_visits_6m","mean"),
    ).fillna(0.0)
    corpus = pt["note_text"].tolist()
    vec = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1,2)
    X = vec.fit_transform(corpus)
    n_comp = min(32, max(2, X.shape[1]-1))
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    Xs = svd.fit_transform(X)
    Xs = StandardScaler().fit_transform(Xs)
    svd_cols = [f"note_svd_{i:02d}" for i in range(Xs.shape[1])]
    svd_df = pd.DataFrame(Xs, columns=svd_cols)
    out = pd.concat([pt.drop(columns=["note_text"]).reset_index(drop=True), svd_d
    return out
if NOTE_FEATS_CACHE.exists() and not FORCE_REBUILD_NOTE_FEATS:
    note_feats = joblib.load(NOTE_FEATS_CACHE)
    print(f"[cache] loaded {NOTE_FEATS_CACHE} | shape={note_feats.shape}")
else:
    note_feats = build_note_patient_features()
    joblib.dump(note_feats, NOTE_FEATS_CACHE)
    print(f"[cache] wrote {NOTE_FEATS_CACHE} | shape={note_feats.shape}")
train_full = train_full.merge(note_feats, on="patient_id", how="left").fillna(0.0
test_full  = test_full.merge(note_feats, on="patient_id", how="left").fillna(0.0)
# ---------- engineered baselines ----------
for df in (train_full, test_full):
    df["is_outlier"] = ((df["prior_ed_cost_5y_usd"] <= DEFLATED_PRIOR_COST_FLOOR)
    df["log_prior_cost"] = np.log1p(df["prior_ed_cost_5y_usd"])
    df["log_prior_visits"] = np.log1p(df["prior_ed_visits_5y"])
    df["prior_cost_per_visit"] = df["prior_ed_cost_5y_usd"] / (df["prior_ed_visit


# --- PAGE 294 ---

    df["log_prior_cost_per_visit"] = np.log1p(df["prior_cost_per_visit"])
    df["log_pdf_expected_total"] = np.log1p(df["pdf_expected_total"])
    for c in CAT_COLS:
        df[c] = df[c].astype("category")
# ---------- feature sets ----------
core_cols = ["prior_ed_cost_5y_usd","prior_ed_visits_5y","log_prior_cost","log_pr
receipt_cols_basic = [c for c in train_full.columns if c.startswith("pdf_") and c
defl_cols = ["pdf_expected_total","log_pdf_expected_total","pdf_deflation_factor"
note_cols = [c for c in train_full.columns if c.startswith("note_")] + [c for c i
def make_X(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    use = [c for c in cols if c in df.columns]
    return df[use + CAT_COLS + ["is_outlier"]].copy()
# ---------- quick CV ----------
LGB_PARAMS = dict(
    n_estimators=20000,
    learning_rate=0.02,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=25,
    objective="regression_l1",
    random_state=42,
    n_jobs=-1,
)
def cv_run(tag: str, cols: list[str]) -> dict:
    X = make_X(train_full, cols)
    y = train_full[TARGET].values
    strat = (train_full["primary_chronic"].astype(str)+"_"+train_full["insurance"
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(train_full))
    fold_mae = []
    for k,(tr,va) in enumerate(skf.split(X, strat), 1):
        m = LGBMRegressor(**LGB_PARAMS)
        m.fit(
            X.iloc[tr], y[tr],
            eval_set=[(X.iloc[va], y[va])],


# --- PAGE 295 ---

Use the ablation table to decide:
your “main” pipeline and tune around them (this is the most likely big win).
            eval_metric="mae",
            callbacks=[lgb.early_stopping(300, verbose=False)],
            categorical_feature=CAT_COLS + ["is_outlier"],
        )
        p = m.predict(X.iloc[va], num_iteration=m.best_iteration_)
        oof[va] = p
        mae = mean_absolute_error(y[va], p)
        fold_mae.append(mae)
        out_mask = ((train_full.iloc[va]["prior_ed_cost_5y_usd"] <= DEFLATED_PRIO
        out_mae = mean_absolute_error(y[va][out_mask], p[out_mask]) if out_mask.s
        print(f"  [{tag}] fold{k}: MAE={mae:.3f} | out_n={int(out_mask.sum()):3d}
    overall = mean_absolute_error(y, oof)
    out_all = ((train_full["prior_ed_cost_5y_usd"] <= DEFLATED_PRIOR_COST_FLOOR) &
    out_overall = mean_absolute_error(y[out_all], oof[out_all])
    return dict(model=tag, oof_mae=overall, mae_mean=float(np.mean(fold_mae)), ma
print("\n=== CV ABLATIONS (EDA) ===")
rows = []
rows.append(cv_run("A_core+receipt_basic", core_cols + receipt_cols_basic))
rows.append(cv_run("B_add_deflation", core_cols + receipt_cols_basic + defl_cols)
rows.append(cv_run("C_add_notes", core_cols + receipt_cols_basic + note_cols))
rows.append(cv_run("D_full", core_cols + receipt_cols_basic + defl_cols + note_co
res = pd.DataFrame(rows).sort_values("oof_mae")
print("\n=== SUMMARY (sorted by OOF MAE) ===")
print(res.to_string(index=False))
res.to_csv(OUT_DIR / "iter21_ablation_summary.csv", index=False)
print(f"\n[DONE] Outputs: {OUT_DIR}")


# --- PAGE 296 ---

If C_add_notes helps → keep the SVD dims small (16–64), and treat it as a compressed
If D_full is worse than B or C → it means you’re mixing two signals that are fighting (or
one is noisy); keep the winner only.
Then, for the next modeling iteration (not EDA):
Stratify folds by (primary_chronic, insurance, deflation_flag)  to stabilize and
Ensemble by median (MAE-optimal aggregation is closer to median than mean).
(optional).
If you paste the Iter21 ablation summary (the printed table), I can tell you which direction
```

### C14 — Code cell starting at page 301
```python
from pathlib import Path
import re
import json
import math
import numpy as np
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr, ks_2samp
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# --- PAGE 302 ---

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
import joblib
# =========================
# 0) Paths (EDIT if needed)
# =========================
BASE_DIR = Path(r"D:\AgentDs\agent_ds_healthcare")
TRAIN_CSV = BASE_DIR / "ed_cost_train.csv"
TEST_CSV  = BASE_DIR / "ed_cost_test.csv"
PATIENTS_CSV = BASE_DIR / "patients.csv"
ADMISSIONS_TRAIN = BASE_DIR / "admissions_train.csv"
ADMISSIONS_TEST  = BASE_DIR / "admissions_test.csv"
PDF_DIR = BASE_DIR / "receipts_pdf"
CACHE_DIR = BASE_DIR / "cache_iter22_eda_adm"
OUT_DIR = CACHE_DIR / "eda_outputs_iter22"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
RECEIPT_TOTAL_CACHE = CACHE_DIR / "receipt_totals_only.joblib"   # cache totals p
CHECK_ALL_PDFS = True   # set False to only sample
PDF_SAMPLE_N = None     # e.g., 500 (None -> all)
PDF_THREADS = 8         # threads for PDF total parsing
print(f"[paths] BASE_DIR={BASE_DIR}")
print(f"[paths] OUT_DIR={OUT_DIR}")
print(f"[paths] PDF_DIR={PDF_DIR}")
# =========================
# 1) Load core tables
# =========================
train = pd.read_csv(TRAIN_CSV)
test  = pd.read_csv(TEST_CSV)
patients = pd.read_csv(PATIENTS_CSV)


# --- PAGE 303 ---

train_full = train.merge(patients, on="patient_id", how="left")
test_full  = test.merge(patients, on="patient_id", how="left")
print(f"[info] train_full shape: {train_full.shape} | test_full shape: {test_full
# Outlier flag (same definition you used)
train_full["is_outlier"] = ((train_full["prior_ed_cost_5y_usd"] <= 50) & (train_f
out_n = int(train_full["is_outlier"].sum())
print(f"[info] outlier_n (prior_cost<=50 & visits==0): {out_n}")
# =========================
# 2) Admissions -> patient-level features
# =========================
adm_tr = pd.read_csv(ADMISSIONS_TRAIN)
adm_te = pd.read_csv(ADMISSIONS_TEST)
adm_tr["_split"] = "train"
adm_te["_split"] = "test"
adm_all = pd.concat([adm_tr, adm_te], ignore_index=True)
# Basic aggregates
adm_agg = adm_all.groupby("patient_id").agg(
    adm_n=("admission_id", "count"),
    adm_los_sum=("los_days", "sum"),
    adm_los_mean=("los_days", "mean"),
    adm_los_max=("los_days", "max"),
    adm_emergent_n=("acuity_emergent", "sum"),
    adm_emergent_rate=("acuity_emergent", "mean"),
    adm_charlson_mean=("charlson_band", "mean"),
    adm_charlson_max=("charlson_band", "max"),
    adm_edvis6m_sum=("ed_visits_6m", "sum"),
    adm_edvis6m_mean=("ed_visits_6m", "mean"),
    adm_edvis6m_max=("ed_visits_6m", "max"),
    adm_discharge_wd_mode=("discharge_weekday", lambda x: x.mode().iloc[0] if len
).reset_index()
# dx counts (HF / Pneumonia / DiabetesComp)
dx_ct = pd.crosstab(adm_all["patient_id"], adm_all["primary_dx"]).reset_index()
dx_ct = dx_ct.rename(columns={c: f"adm_dx_cnt_{c}" for c in dx_ct.columns if c !=


# --- PAGE 304 ---

adm_agg = adm_agg.merge(dx_ct, on="patient_id", how="left")
# Join onto ED cost tables
train_full = train_full.merge(adm_agg, on="patient_id", how="left")
test_full  = test_full.merge(adm_agg, on="patient_id", how="left")
print(f"[info] after admissions merge: train_full={train_full.shape} | test_full=
assert train_full["adm_n"].isna().mean() == 0.0, "Unexpected missing admissions f
assert test_full["adm_n"].isna().mean() == 0.0, "Unexpected missing admissions fe
# =========================
# 3) Build a simple baseline ratio (for residual analysis)
#    baseline_ratio = median(target/prior_cost) by (primary_chronic, insurance)
# =========================
train_full["_ratio"] = train_full["ed_cost_next3y_usd"] / train_full["prior_ed_co
global_med_ratio = float(train_full["_ratio"].median())
grp_cols = ["primary_chronic", "insurance"]
med_ratio_map = train_full.groupby(grp_cols)["_ratio"].median().to_dict()
keys = list(zip(train_full["primary_chronic"], train_full["insurance"]))
train_full["baseline_ratio"] = [med_ratio_map.get(k, global_med_ratio) for k in k
train_full["baseline_pred"] = train_full["baseline_ratio"] * train_full["prior_ed_
train_full["baseline_resid"] = train_full["ed_cost_next3y_usd"] - train_full["bas
print("\n=== BASELINE (group-median ratio) CHECK ===")
print("Baseline MAE (in-sample, for residual analysis only):",
      mean_absolute_error(train_full["ed_cost_next3y_usd"], train_full["baseline_
# =========================
# 4) Receipt TOTAL integrity check (fast totals-only parsing)
# =========================
TOTAL_RE = re.compile(r"TOTAL\s+([\d,]+\.\d{2})", re.IGNORECASE)
def _extract_text_any_backend(pdf_path: Path) -> str:
    # Try pypdf first
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        parts = []


# --- PAGE 305 ---

        for p in reader.pages:
            t = p.extract_text() or ""
            parts.append(t)
        return "\n".join(parts)
    except Exception:
    # Try PyPDF2
    try:
        import PyPDF2
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            parts = []
            for p in reader.pages:
                t = p.extract_text() or ""
                parts.append(t)
            return "\n".join(parts)
    except Exception:
    # Try PyMuPDF
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(pdf_path))
        return "\n".join([page.get_text("text") for page in doc])
    except Exception:
    # Try pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(str(pdf_path)) as pdf:
            return "\n".join([(page.extract_text() or "") for page in pdf.pages])
    except Exception:
    return ""
def parse_total_from_pdf(pdf_path: Path) -> float:
    txt = _extract_text_any_backend(pdf_path)


# --- PAGE 306 ---

    if not txt:
        return np.nan
    m = TOTAL_RE.search(txt)
    if not m:
        return np.nan
    return float(m.group(1).replace(",", ""))
def check_receipt_totals(ed_train_df: pd.DataFrame, ed_test_df: pd.DataFrame):
    df_all = pd.concat([
        ed_train_df[["patient_id", "prior_ed_cost_5y_usd"]].assign(_split="train"
        ed_test_df[["patient_id", "prior_ed_cost_5y_usd"]].assign(_split="test"),
    ], ignore_index=True)
    ids = df_all["patient_id"].astype(int).tolist()
    if (PDF_SAMPLE_N is not None) and (PDF_SAMPLE_N < len(ids)):
        rng = np.random.default_rng(42)
        ids = rng.choice(ids, size=PDF_SAMPLE_N, replace=False).tolist()
        df_all = df_all[df_all["patient_id"].isin(ids)].copy()
    # load cache
    cache = {}
    if RECEIPT_TOTAL_CACHE.exists():
        try:
            cache = joblib.load(RECEIPT_TOTAL_CACHE)
            if not isinstance(cache, dict):
                cache = {}
        except Exception:
            cache = {}
    to_parse = [pid for pid in ids if pid not in cache]
    print(f"\n[receipt-check] patients to check: {len(ids)} | new parse: {len(to_
    # threaded parse (safe in Jupyter on Windows)
    if len(to_parse) > 0:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        def _worker(pid: int):
            pdf_path = PDF_DIR / f"receipt_{pid}.pdf"
            if not pdf_path.exists():
                return pid, np.nan


# --- PAGE 307 ---

            return pid, parse_total_from_pdf(pdf_path)
        with ThreadPoolExecutor(max_workers=PDF_THREADS) as ex:
            futures = [ex.submit(_worker, pid) for pid in to_parse]
            for fut in as_completed(futures):
                pid, tot = fut.result()
                cache[pid] = tot
        joblib.dump(cache, RECEIPT_TOTAL_CACHE)
        print(f"[receipt-check] wrote cache: {RECEIPT_TOTAL_CACHE}")
    df_all["pdf_total"] = df_all["patient_id"].astype(int).map(cache)
    df_all["abs_diff"] = (df_all["pdf_total"] - df_all["prior_ed_cost_5y_usd"]).a
    # summarize
    ok_mask = df_all["abs_diff"].fillna(1e18) < 1e-6
    miss_mask = df_all["pdf_total"].isna()
    bad_mask = (~ok_mask) & (~miss_mask)
    print("\n=== RECEIPT TOTAL INTEGRITY (totals-only) ===")
    print(f"missing_total_rate: {miss_mask.mean():.4f}")
    print(f"bad_total_rate:     {bad_mask.mean():.4f}")
    print("\nabs_diff summary:")
    print(df_all["abs_diff"].describe())
    # write bad list
    bad_df = df_all.loc[bad_mask | miss_mask, ["patient_id", "_split", "prior_ed_
    bad_out = OUT_DIR / "receipt_total_mismatches.csv"
    bad_df.to_csv(bad_out, index=False)
    print(f"[write] {bad_out}  (n_bad_or_missing={len(bad_df)})")
    return df_all
# Run totals check
receipt_check_df = check_receipt_totals(train_full, test_full)
# =========================
# 5) Feature signal checks: admissions features vs target / residual
# =========================
def _safe_spearman(x, y):


# --- PAGE 308 ---

    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    m = x.notna() & y.notna()
    if m.sum() < 10:
        return np.nan
    return float(spearmanr(x[m], y[m]).correlation)
adm_num_cols = [
    if c.startswith("adm_") and c not in ["adm_discharge_wd_mode"]  # numeric only
]
# include dx count cols too
adm_num_cols += [c for c in train_full.columns if c.startswith("adm_dx_cnt_")]
adm_num_cols = sorted(list(dict.fromkeys(adm_num_cols)))
rows = []
for c in adm_num_cols:
    rows.append({
        "feature": c,
        "missing_rate": float(train_full[c].isna().mean()),
        "spearman_target": _safe_spearman(train_full[c], train_full["ed_cost_next
        "spearman_resid": _safe_spearman(train_full[c], train_full["baseline_resi
        "spearman_prior_cost": _safe_spearman(train_full[c], train_full["prior_ed_
        "spearman_prior_visits": _safe_spearman(train_full[c], train_full["prior_
        "mean": float(pd.to_numeric(train_full[c], errors="coerce").mean()),
        "std": float(pd.to_numeric(train_full[c], errors="coerce").std()),
    })
corr_df = pd.DataFrame(rows).sort_values("spearman_resid", ascending=False)
corr_out = OUT_DIR / "admissions_feature_correlations.csv"
corr_df.to_csv(corr_out, index=False)
print(f"\n[write] {corr_out}")
print("\n=== TOP 20 admissions features by Spearman(residual) ===")
print(corr_df.head(20).to_string(index=False))
# =========================
# 6) Outlier vs non-outlier: does admissions explain variation inside outliers?
# =========================
out = train_full[train_full["is_outlier"] == 1].copy()


# --- PAGE 309 ---

non = train_full[train_full["is_outlier"] == 0].copy()
rows = []
for c in adm_num_cols:
    rows.append({
        "feature": c,
        "spearman_target_outliers": _safe_spearman(out[c], out["ed_cost_next3y_us
        "spearman_target_nonoutliers": _safe_spearman(non[c], non["ed_cost_next3y_
        "miss_out": float(out[c].isna().mean()),
        "miss_non": float(non[c].isna().mean()),
        "mean_out": float(pd.to_numeric(out[c], errors="coerce").mean()),
        "mean_non": float(pd.to_numeric(non[c], errors="coerce").mean()),
    })
outcorr_df = pd.DataFrame(rows).sort_values("spearman_target_outliers", ascending
outcorr_out = OUT_DIR / "admissions_outlier_vs_nonoutlier_corr.csv"
outcorr_df.to_csv(outcorr_out, index=False)
print(f"\n[write] {outcorr_out}")
print("\n=== TOP 15 admissions features by Spearman(target) WITHIN outliers ===")
print(outcorr_df.head(15).to_string(index=False))
# =========================
# 7) Train/Test shift (KS) for admissions numeric features
# =========================
rows = []
for c in adm_num_cols:
    tr = pd.to_numeric(train_full[c], errors="coerce").dropna()
    te = pd.to_numeric(test_full[c], errors="coerce").dropna()
    if len(tr) < 50 or len(te) < 50:
        ks = np.nan
    else:
        ks = float(ks_2samp(tr, te).statistic)
    rows.append({
        "feature": c,
        "missing_rate_train": float(train_full[c].isna().mean()),
        "missing_rate_test": float(test_full[c].isna().mean()),
        "mean_train": float(tr.mean()) if len(tr) else np.nan,
        "mean_test": float(te.mean()) if len(te) else np.nan,
        "std_train": float(tr.std()) if len(tr) else np.nan,
        "std_test": float(te.std()) if len(te) else np.nan,


# --- PAGE 310 ---

        "ks_stat": ks
    })
ks_df = pd.DataFrame(rows).sort_values("ks_stat", ascending=False)
ks_out = OUT_DIR / "admissions_feature_shift_ks.csv"
ks_df.to_csv(ks_out, index=False)
print(f"\n[write] {ks_out}")
print("\n=== TOP 15 admissions features by train/test shift (KS) ===")
print(ks_df.head(15).to_string(index=False))
# =========================
# 8) Quick micro-validation: can admissions help outlier MAE at all?
#    (Ridge is just a sanity check; not a final model.)
# =========================
def ridge_cv_mae(df, feat_cols, y_col, n_splits=5, seed=42):
    X = df[feat_cols].copy()
    y = df[y_col].copy()
    cat_cols = [c for c in feat_cols if X[c].dtype == "object"]
    num_cols = [c for c in feat_cols if c not in cat_cols]
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
            ]), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_cols),
        ],
        remainder="drop"
    )
    model = Ridge(alpha=5.0)
    pipe = Pipeline([("pre", pre), ("model", model)])
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    maes = []
    for tr_idx, va_idx in kf.split(X):


# --- PAGE 311 ---

        pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        pred = pipe.predict(X.iloc[va_idx])
        maes.append(mean_absolute_error(y.iloc[va_idx], pred))
    return float(np.mean(maes)), float(np.std(maes))
# Features for this sanity test (keep it small + meaningful)
sanity_feats = [
]
sanity_feats = [c for c in sanity_feats if c in train_full.columns]
print("\n=== SANITY: OUTLIER SUBMODEL POTENTIAL (Ridge CV, admissions+demo only)
if len(out) >= 30:
    # baseline: outlier median by chronic+insurance
    out["_key"] = list(zip(out["primary_chronic"], out["insurance"]))
    med_map = out.groupby("_key")["ed_cost_next3y_usd"].median().to_dict()
    baseline = out["_key"].map(med_map).values
    base_mae = mean_absolute_error(out["ed_cost_next3y_usd"].values, baseline)
    print(f"Outlier baseline (median by chronic+insurance) MAE (in-sample): {base_
    cv_mae, cv_std = ridge_cv_mae(out, sanity_feats, "ed_cost_next3y_usd", n_split
    print(f"Outlier Ridge CV MAE: {cv_mae:.3f} ± {cv_std:.3f}")
else:
    print("Not enough outlier rows for CV sanity test.")
print("\n[DONE] Iter22 EDA outputs written to:", OUT_DIR)


# --- PAGE 312 ---
```

### C15 — Code cell starting at page 320
```python
# =========================
# Iter23 EDA (single cell)
# Price Standardization + Admissions Incremental Value
# =========================
from pathlib import Path
import re, os, gc, warnings, json
import numpy as np
import pandas as pd


# --- PAGE 321 ---

from joblib import dump, load
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# ---------- PATHS ----------
BASE_DIR = Path(r"D:\AgentDs\agent_ds_healthcare")
TRAIN_CSV = BASE_DIR / "ed_cost_train.csv"
TEST_CSV  = BASE_DIR / "ed_cost_test.csv"
PATIENTS_CSV = BASE_DIR / "patients.csv"
ADMISSIONS_TRAIN = BASE_DIR / "admissions_train.csv"
ADMISSIONS_TEST  = BASE_DIR / "admissions_test.csv"
PDF_DIR = BASE_DIR / "receipts_pdf"
CACHE_DIR = BASE_DIR / "cache_iter23_price_std_eda"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = CACHE_DIR / "eda_outputs_iter23"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RECEIPT_PARSE_CACHE = CACHE_DIR / "receipt_lineitems_parsed.joblib"
RECEIPT_FEATS_CACHE = CACHE_DIR / "receipt_features_price_std.joblib"
ADMISSIONS_FEATS_CACHE = CACHE_DIR / "admissions_patient_agg.joblib"
RANDOM_STATE = 42
N_SPLITS = 5
# ---------- HELPERS ----------
def _safe_float(x: str) -> float:
    return float(str(x).replace(",", "").strip())
def _safe_int(x: str) -> int:
    return int(str(x).strip())


# --- PAGE 322 ---

LINE_RE  = re.compile(r'^(\S+)\s+(.+?)\s+(\d+)\s+([\d,]+\.\d{2})\s+([\d,]+\.\d{2}
TOTAL_RE = re.compile(r'^TOTAL\s+([\d,]+\.\d{2})$')
def extract_text_pdf(pdf_path: Path) -> str | None:
    # pdfplumber first (often most robust for text PDFs)
    try:
        import pdfplumber
        with pdfplumber.open(str(pdf_path)) as pdf:
            txt = "\n".join((pg.extract_text() or "") for pg in pdf.pages)
        if txt and txt.strip():
            return txt
    except Exception:
    # fallback: pypdf
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        txt = "\n".join((pg.extract_text() or "") for pg in reader.pages)
        if txt and txt.strip():
            return txt
    except Exception:
    return None
def parse_receipt_text(text: str):
    lines = [ln.strip() for ln in text.splitlines() if ln and ln.strip()]
    items = []
    total = None
    for ln in lines:
        tm = TOTAL_RE.match(ln)
        if tm:
            total = _safe_float(tm.group(1))
        m = LINE_RE.match(ln)
        if m:
            code, desc, qty, unit, lt = m.groups()


# --- PAGE 323 ---

            items.append({
                "code": code.strip(),
                "desc": desc.strip(),
                "qty": _safe_int(qty),
                "unit": _safe_float(unit),
                "line_total": _safe_float(lt),
            })
    return items, total
def load_core_tables():
    train = pd.read_csv(TRAIN_CSV)
    test  = pd.read_csv(TEST_CSV)
    patients = pd.read_csv(PATIENTS_CSV)
    # normalize ids
    for df in (train, test, patients):
        df["patient_id"] = df["patient_id"].astype(int)
    # normalize zip3 as string with leading zeros
    if "zip3" in patients.columns:
        patients["zip3"] = patients["zip3"].astype(str).str.zfill(3)
    train = train.merge(patients, on="patient_id", how="left")
    test  = test.merge(patients, on="patient_id", how="left")
    return train, test, patients
def build_admissions_patient_agg() -> pd.DataFrame:
    if ADMISSIONS_FEATS_CACHE.exists():
        return load(ADMISSIONS_FEATS_CACHE)
    adm_tr = pd.read_csv(ADMISSIONS_TRAIN)
    adm_te = pd.read_csv(ADMISSIONS_TEST)
    # patient_id
    for df in (adm_tr, adm_te):
        df["patient_id"] = df["patient_id"].astype(int)
    # concat (drop label if exists)
    adm_all = pd.concat([adm_tr.drop(columns=["readmit_30d"], errors="ignore"), a


# --- PAGE 324 ---

    # expected columns (from description / your EDA logs)
    # primary_dx, acuity_emergent, charlson_band, ed_visits_6m, length_of_stay_day
    # make sure numeric
    for c in ["acuity_emergent", "charlson_band", "ed_visits_6m", "length_of_stay_
        if c in adm_all.columns:
            adm_all[c] = pd.to_numeric(adm_all[c], errors="coerce")
    # dx counts
    if "primary_dx" in adm_all.columns:
        dx_cnt = (adm_all
                  .assign(_n=1)
                  .pivot_table(index="patient_id", columns="primary_dx", values="_
        dx_cnt.columns = [f"adm_dx_cnt_{c}" for c in dx_cnt.columns]
    else:
        dx_cnt = pd.DataFrame(index=adm_all["patient_id"].unique())
    # core aggregates
    grp = adm_all.groupby("patient_id", observed=False)
    agg = pd.DataFrame({
        "adm_n": grp.size(),
    })
    if "acuity_emergent" in adm_all.columns:
        agg["adm_emergent_n"] = grp["acuity_emergent"].sum(min_count=1).fillna(0)
        agg["adm_emergent_rate"] = grp["acuity_emergent"].mean()
    if "ed_visits_6m" in adm_all.columns:
        agg["adm_edvis6m_sum"]  = grp["ed_visits_6m"].sum(min_count=1).fillna(0)
        agg["adm_edvis6m_mean"] = grp["ed_visits_6m"].mean()
        agg["adm_edvis6m_max"]  = grp["ed_visits_6m"].max()
    if "charlson_band" in adm_all.columns:
        agg["adm_charlson_mean"] = grp["charlson_band"].mean()
        agg["adm_charlson_max"]  = grp["charlson_band"].max()
    if "length_of_stay_days" in adm_all.columns:
        agg["adm_los_sum"]  = grp["length_of_stay_days"].sum(min_count=1).fillna(
        agg["adm_los_mean"] = grp["length_of_stay_days"].mean()
        agg["adm_los_max"]  = grp["length_of_stay_days"].max()


# --- PAGE 325 ---

    out = agg.join(dx_cnt, how="left").fillna(0).reset_index()
    dump(out, ADMISSIONS_FEATS_CACHE)
    return out
def parse_all_receipts(patient_ids, force=False):
    # robust cache loader
    if RECEIPT_PARSE_CACHE.exists() and not force:
        obj = load(RECEIPT_PARSE_CACHE)
        if isinstance(obj, dict) and ("lineitems_df" in obj) and ("headers_df" in
            return obj["headers_df"], obj["lineitems_df"]
    headers = []
    rows = []
    bad = []
    missing_pdf = []
    n = len(patient_ids)
    for i, pid in enumerate(patient_ids, 1):
        pdf_path = PDF_DIR / f"receipt_{pid}.pdf"
        if not pdf_path.exists():
            missing_pdf.append(pid)
        txt = extract_text_pdf(pdf_path)
        if not txt:
            bad.append(pid)
            headers.append({"patient_id": pid, "parse_ok": 0, "n_lines": 0, "pdf_
        items, total = parse_receipt_text(txt)
        parse_ok = int((total is not None) and (len(items) > 0))
        headers.append({"patient_id": pid, "parse_ok": parse_ok, "n_lines": len(it
        if not parse_ok:
            bad.append(pid)
        for it in items:
            rows.append({


# --- PAGE 326 ---

                "patient_id": pid,
                "code": it["code"],
                "qty": it["qty"],
                "unit": it["unit"],
                "line_total": it["line_total"],
            })
        if (i % 250) == 0 or i == n:
            print(f"[receipt-parse] {i}/{n} done | bad={len(bad)} missing_pdf={le
    headers_df = pd.DataFrame(headers)
    lineitems_df = pd.DataFrame(rows)
    dump({"headers_df": headers_df, "lineitems_df": lineitems_df, "bad_ids": bad,
    print(f"[cache] wrote {RECEIPT_PARSE_CACHE} | bad={len(bad)} missing_pdf={len
    return headers_df, lineitems_df
def build_receipt_features_price_std(lineitems_df: pd.DataFrame, patients_df: pd.
    if RECEIPT_FEATS_CACHE.exists():
        return load(RECEIPT_FEATS_CACHE)
    li = lineitems_df.copy()
    li["patient_id"] = li["patient_id"].astype(int)
    # merge insurance (used for standard price)
    use_cols = ["patient_id"]
    if "insurance" in patients_df.columns:
        use_cols.append("insurance")
    pat_ins = patients_df[use_cols].copy()
    li = li.merge(pat_ins, on="patient_id", how="left")
    # medians by (code, insurance) + global fallback
    med_ci = (li.groupby(["code", "insurance"], observed=False)["unit"]
                .median()
                .rename("unit_med_ci")
                .reset_index())
    med_c = (li.groupby("code", observed=False)["unit"]
               .median()
               .rename("unit_med_global")
               .reset_index())


# --- PAGE 327 ---

    li = li.merge(med_ci, on=["code","insurance"], how="left")
    li = li.merge(med_c, on=["code"], how="left")
    li["unit_med"] = li["unit_med_ci"].fillna(li["unit_med_global"])
    li["std_line_total"] = li["qty"] * li["unit_med"]
    # totals
    tot = li.groupby("patient_id", observed=False).agg(
        pdf_total_line_cost=("line_total","sum"),
        std_total=("std_line_total","sum"),
        n_line_items=("code","size"),
        n_unique_codes=("code","nunique"),
        total_qty=("qty","sum"),
    )
    tot["price_factor"] = tot["pdf_total_line_cost"] / tot["std_total"].replace(0
    tot["price_factor"] = tot["price_factor"].replace([np.inf, -np.inf], np.nan).
    # per-code features (counts + cost + std_cost)
    all_codes = sorted(li["code"].unique().tolist())
    cnt = li.pivot_table(index="patient_id", columns="code", values="qty", aggfun
    cnt.columns = [f"cnt_{c}" for c in cnt.columns]
    cost = li.pivot_table(index="patient_id", columns="code", values="line_total"
    cost.columns = [f"cost_{c}" for c in cost.columns]
    std_cost = li.pivot_table(index="patient_id", columns="code", values="std_lin
    std_cost.columns = [f"std_cost_{c}" for c in std_cost.columns]
    # cost shares (actual)
    cost_sum = cost.sum(axis=1).replace(0, np.nan)
    share = cost.div(cost_sum, axis=0).fillna(0)
    share.columns = [c.replace("cost_", "share_") for c in share.columns]
    # categories (based on your known 18 codes)
    ED_EM = {"99281","99282","99283","99284","99285"}
    CRIT  = {"99291","99292"}
    IMAG  = {"70450","74177","71045"}
    LAB   = {"85025","87070","84484"}
    OBS   = {"G0378"}
    SEVERE= {"31500","36556","36620","92950"}


# --- PAGE 328 ---

    def _cat(code):
        if code in ED_EM: return "ed_em"
        if code in CRIT:  return "crit"
        if code in IMAG:  return "imaging"
        if code in LAB:   return "lab"
        if code in OBS:   return "obs"
        if code in SEVERE:return "severe_proc"
        return "other"
    li["cat"] = li["code"].map(_cat)
    cat_cost = li.pivot_table(index="patient_id", columns="cat", values="line_tot
    cat_cost.columns = [f"cat_cost_{c}" for c in cat_cost.columns]
    cat_share = cat_cost.div(cat_cost.sum(axis=1).replace(0,np.nan), axis=0).fill
    cat_share.columns = [c.replace("cat_cost_", "cat_share_") for c in cat_cost.c
    # EM level stats
    em_level_map = {"99281":1, "99282":2, "99283":3, "99284":4, "99285":5}
    em_df = li[li["code"].isin(em_level_map)].copy()
    if len(em_df) > 0:
        em_df["em_level"] = em_df["code"].map(em_level_map).astype(float)
        em_stats = em_df.groupby("patient_id", observed=False).agg(
            em_level_mean=("em_level","mean"),
            em_level_max=("em_level","max"),
            em_high_frac=("em_level", lambda s: float(np.mean(np.array(s) >= 4.0)
            em_n=("em_level","size"),
        )
    else:
        em_stats = pd.DataFrame(index=tot.index, columns=["em_level_mean","em_lev
    feats = (tot
             .join(cnt, how="left")
             .join(cost, how="left")
             .join(std_cost, how="left")
             .join(share, how="left")
             .join(cat_cost, how="left")
             .join(cat_share, how="left")
             .join(em_stats, how="left")
             .fillna(0)


# --- PAGE 329 ---

             .reset_index())
    dump(feats, RECEIPT_FEATS_CACHE)
    print(f"[cache] wrote {RECEIPT_FEATS_CACHE} | shape={feats.shape} | codes={le
    return feats
def prepare_design_matrix(train_full: pd.DataFrame, feature_cols, cat_cols):
    X = train_full[feature_cols].copy()
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype(str).fillna("UNK").astype("category")
    return X
def run_cv_lgbm(train_full: pd.DataFrame, feature_cols, cat_cols, model_name: str
    import lightgbm as lgb
    y = train_full["ed_cost_next3y_usd"].values
    outlier_mask = ((train_full["prior_ed_cost_5y_usd"] <= 50) & (train_full["pri
    # stratify by chronic + target decile to stabilize folds
    dec = pd.qcut(train_full["ed_cost_next3y_usd"], q=10, duplicates="drop")
    strat = (train_full["primary_chronic"].astype(str) + "_" + dec.astype(str)).v
    strat = pd.factorize(strat)[0]
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STA
    oof = np.zeros(len(train_full), dtype=float)
    fold_mae = []
    fold_out_mae = []
    best_iters = []
    params = dict(
        objective="regression_l1",
        n_estimators=20000,
        learning_rate=0.01,
        num_leaves=64,
        min_data_in_leaf=30,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,


# --- PAGE 330 ---

        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_full, strat), 1):
        X_tr = prepare_design_matrix(train_full.iloc[tr_idx], feature_cols, cat_c
        y_tr = y[tr_idx]
        X_va = prepare_design_matrix(train_full.iloc[va_idx], feature_cols, cat_c
        y_va = y[va_idx]
        m = lgb.LGBMRegressor(**params)
        m.fit(
            eval_set=[(X_va, y_va)],
            eval_metric="mae",
            categorical_feature=[c for c in cat_cols if c in X_tr.columns],
            callbacks=[lgb.early_stopping(400, verbose=False)]
        )
        pred = m.predict(X_va, num_iteration=m.best_iteration_)
        oof[va_idx] = pred
        mae = mean_absolute_error(y_va, pred)
        fold_mae.append(mae)
        # outlier mae in fold
        va_out = outlier_mask[va_idx]
        out_mae = mean_absolute_error(y_va[va_out], pred[va_out]) if va_out.sum()
        fold_out_mae.append(out_mae)
        best_iters.append(getattr(m, "best_iteration_", np.nan))
        print(f"  [{model_name}] fold {fold}: MAE={mae:.3f} | out_n={int(va_out.s
    oof_mae = mean_absolute_error(y, oof)
    oof_out_mae = mean_absolute_error(y[outlier_mask], oof[outlier_mask]) if outl
    return {
        "model": model_name,


# --- PAGE 331 ---

        "oof_mae": oof_mae,
        "mae_mean": float(np.mean(fold_mae)),
        "mae_std": float(np.std(fold_mae)),
        "out_oof_mae": float(oof_out_mae),
        "out_mae_mean": float(np.nanmean(fold_out_mae)),
        "out_mae_std": float(np.nanstd(fold_out_mae)),
        "avg_best_iter": float(np.nanmean(best_iters)),
        "n_features": len(feature_cols),
    }
# ---------- MAIN ----------
print(f"[paths] BASE_DIR={BASE_DIR}")
print(f"[paths] OUT_DIR={OUT_DIR}")
print(f"[info] PDF_DIR exists={PDF_DIR.exists()}")
train_full, test_full, patients = load_core_tables()
print(f"[info] train_full: {train_full.shape}  test_full: {test_full.shape}")
# admissions agg
adm_feat = build_admissions_patient_agg()
train_full = train_full.merge(adm_feat, on="patient_id", how="left")
test_full  = test_full.merge(adm_feat, on="patient_id", how="left")
for c in adm_feat.columns:
    if c != "patient_id":
        train_full[c] = train_full[c].fillna(0)
        test_full[c] = test_full[c].fillna(0)
# receipt parse + features
all_ids = sorted(set(train_full["patient_id"].astype(int)) | set(test_full["patie
headers_df, lineitems_df = parse_all_receipts(all_ids, force=False)
print(f"[info] parsed lineitems rows: {len(lineitems_df)} | patients_w_lines: {li
receipt_feats = build_receipt_features_price_std(lineitems_df, patients)
train_full = train_full.merge(receipt_feats, on="patient_id", how="left")
test_full  = test_full.merge(receipt_feats, on="patient_id", how="left")
# fill NA receipts (should be rare; if many, parsing is broken)
receipt_cols = [c for c in receipt_feats.columns if c != "patient_id"]
train_full[receipt_cols] = train_full[receipt_cols].fillna(0)
test_full[receipt_cols] = test_full[receipt_cols].fillna(0)


# --- PAGE 332 ---

# integrity check: total vs prior_cost
abs_diff = (train_full["pdf_total_line_cost"] - train_full["prior_ed_cost_5y_usd"
print("\n[check] |pdf_total_line_cost - prior_ed_cost_5y_usd| (train)")
print(abs_diff.describe())
bad_mismatch = train_full.loc[abs_diff > 1e-6, ["patient_id","prior_ed_cost_5y_us
bad_mismatch.to_csv(OUT_DIR / "receipt_total_mismatches_iter23.csv", index=False)
print(f"[write] receipt_total_mismatches_iter23.csv | n_bad={len(bad_mismatch)}")
# core engineered features
train_full["log_prior_cost"] = np.log1p(train_full["prior_ed_cost_5y_usd"])
test_full["log_prior_cost"]  = np.log1p(test_full["prior_ed_cost_5y_usd"])
train_full["log_std_total"] = np.log1p(train_full["std_total"])
test_full["log_std_total"]  = np.log1p(test_full["std_total"])
train_full["avg_cost_per_visit"] = train_full["prior_ed_cost_5y_usd"] / (train_fu
test_full["avg_cost_per_visit"]  = test_full["prior_ed_cost_5y_usd"] / (test_full
train_full["std_total_per_visit"] = train_full["std_total"] / (train_full["prior_
test_full["std_total_per_visit"]  = test_full["std_total"] / (test_full["prior_ed_
# feature sets
cat_cols = ["primary_chronic", "sex", "insurance", "zip3"]
core_cols = [
] + cat_cols
# light receipt summary features + per-category shares + em stats + per-code shar
receipt_keep = [
]
receipt_keep += [c for c in train_full.columns if c.startswith("cat_share_")]
receipt_keep += [c for c in train_full.columns if c.startswith("share_")]  # per-
receipt_keep = [c for c in receipt_keep if c in train_full.columns]
adm_keep = [


# --- PAGE 333 ---

]
adm_keep += [c for c in train_full.columns if c.startswith("adm_dx_cnt_")]
adm_keep = [c for c in adm_keep if c in train_full.columns]
# Ablations
FEATS_A = core_cols + receipt_keep
FEATS_B = core_cols + receipt_keep + ["std_total","price_factor","std_total_per_v
FEATS_C = core_cols + receipt_keep + adm_keep
FEATS_D = core_cols + receipt_keep + ["std_total","price_factor","std_total_per_v
# de-dup while preserving order
def dedup(seq):
    seen=set()
    out=[]
    for x in seq:
        if x not in seen:
            out.append(x); seen.add(x)
    return out
FEATS_A = dedup(FEATS_A)
FEATS_B = dedup(FEATS_B)
FEATS_C = dedup(FEATS_C)
FEATS_D = dedup(FEATS_D)
print("\n[info] Feature counts:")
print("  A(core+receipt):", len(FEATS_A))
print("  B(+price_std):  ", len(FEATS_B))
print("  C(+admissions): ", len(FEATS_C))
print("  D(full):        ", len(FEATS_D))
# CV
results = []
print("\n=== CV ABLATIONS (Iter23) ===")
results.append(run_cv_lgbm(train_full, FEATS_A, cat_cols, "A_core+receipt"))
results.append(run_cv_lgbm(train_full, FEATS_B, cat_cols, "B_add_price_std"))
results.append(run_cv_lgbm(train_full, FEATS_C, cat_cols, "C_add_admissions"))
results.append(run_cv_lgbm(train_full, FEATS_D, cat_cols, "D_full"))


# --- PAGE 334 ---

5) 你跑完 Iter23 后，我希望看到哪几行输出（这样我能直接给 Iter24 路
6) 回答你中文问题：是不是“其他数据真的不行”？
高价值特征，而且你自己的 Spearman(target) 也证明它们有信息量。
若提升不显著：说明 benchmark=400 可能依赖更隐蔽的交互（比如
res_df = pd.DataFrame(results).sort_values("oof_mae")
print("\n=== SUMMARY (sorted by OOF MAE) ===")
display(res_df)
res_df.to_csv(OUT_DIR / "ablation_results_iter23.csv", index=False)
print(f"[write] {OUT_DIR / 'ablation_results_iter23.csv'}")
print("\n[DONE] Iter23 EDA finished.")
print(f"Outputs: {OUT_DIR}")


# --- PAGE 335 ---
```

### C16 — Code cell starting at page 339
```python
# --- PAGE 340 ---

# =========================
# Iter24 EDA + Ablation (one-cell)
# Focus: receipt parse integrity + feature injection audit + admissions (ed_visit
# =========================
from pathlib import Path
import re
import math
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import matplotlib.pyplot as plt
# -------------------------
# 0) Paths / switches
# -------------------------
BASE_DIR = Path(r"D:\AgentDs\agent_ds_healthcare")
TRAIN_CSV = BASE_DIR / "ed_cost_train.csv"
TEST_CSV  = BASE_DIR / "ed_cost_test.csv"
PATIENTS_CSV = BASE_DIR / "patients.csv"
PDF_DIR = BASE_DIR / "receipts_pdf"
ADM_TRAIN = BASE_DIR / "admissions_train.csv"
ADM_TEST  = BASE_DIR / "admissions_test.csv"
CACHE_DIR = BASE_DIR / "cache_iter24_eda_adm_receipt_v2"
OUT_DIR   = CACHE_DIR / "eda_outputs_iter24"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
RECEIPT_LINEITEMS_CACHE = CACHE_DIR / "receipt_lineitems_v2.joblib"
RECEIPT_FEATURES_CACHE  = CACHE_DIR / "receipt_features_v2.joblib"
ADMISSIONS_FEATURES_CACHE = CACHE_DIR / "admissions_patient_features_v2.joblib"
FORCE_REPARSE_RECEIPTS = False
RUN_CATBOOST = True   # will auto-skip if catboost not installed


# --- PAGE 341 ---

N_SPLITS = 5
SEED = 42
print(f"[paths] BASE_DIR={BASE_DIR}")
print(f"[paths] OUT_DIR={OUT_DIR}")
print(f"[paths] PDF_DIR exists={PDF_DIR.exists()}")
# -------------------------
# 1) Load core tables
# -------------------------
train = pd.read_csv(TRAIN_CSV)
test  = pd.read_csv(TEST_CSV)
patients = pd.read_csv(PATIENTS_CSV, dtype={"zip3": "string"})
train_full = train.merge(patients, on="patient_id", how="left")
test_full  = test.merge(patients, on="patient_id", how="left")
print(f"[info] train_full: {train_full.shape}  test_full: {test_full.shape}")
# Standardize dtypes for categoricals
for df in (train_full, test_full):
    for c in ["primary_chronic", "sex", "insurance", "zip3"]:
        if c in df.columns:
            df[c] = df[c].astype("string")
# Outlier definition you used
def is_outlier(df: pd.DataFrame) -> pd.Series:
    return (df["prior_ed_cost_5y_usd"] <= 50) & (df["prior_ed_visits_5y"] == 0)
train_full["is_outlier"] = is_outlier(train_full).astype(int)
test_full["is_outlier"]  = is_outlier(test_full).astype(int)
print(f"[info] outlier_n (train) = {train_full['is_outlier'].sum()} / {len(train_
# -------------------------
# 2) Robust receipt parsing (pdfplumber -> pypdf -> fitz)
# -------------------------
money_token_re = re.compile(r"^\$?[\d,]+\.\d{2}$")
code_re = re.compile(r"^[A-Z]?\d{4,5}$")   # 'G0378' or '99283' or '31500'


# --- PAGE 342 ---

def safe_money(x):
    if x is None:
        return None
    s = str(x).strip().replace("$", "").replace(",", "")
    try:
        return float(s)
    except Exception:
        return None
def extract_text_pdf(path: Path):
    # Return (text, backend, err)
    # 1) pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            text = "\n".join([(p.extract_text() or "") for p in pdf.pages])
        if text and text.strip():
            return text, "pdfplumber", None
    except Exception as e:
        err1 = repr(e)
    # 2) pypdf
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        text = "\n".join([(p.extract_text() or "") for p in reader.pages])
        if text and text.strip():
            return text, "pypdf", None
    except Exception as e:
        err2 = repr(e)
    # 3) fitz (PyMuPDF)
    try:
        import fitz
        doc = fitz.open(str(path))
        text = "\n".join([p.get_text() for p in doc])
        doc.close()
        if text and text.strip():
            return text, "fitz", None
    except Exception as e:


# --- PAGE 343 ---

        err3 = repr(e)
    return "", "fail", {"pdfplumber": err1 if "err1" in locals() else None,
                        "pypdf": err2 if "err2" in locals() else None,
                        "fitz": err3 if "err3" in locals() else None}
def parse_receipt_text(text: str):
    Parse:
      - TOTAL (float)
      - line items: code, qty, unit, line_total
    total = None
    items = []
    for ln in text.splitlines():
        ln = (ln or "").strip()
        if not ln:
        # TOTAL
        if ln.upper().startswith("TOTAL"):
            toks = ln.split()
            m = None
            if len(toks) >= 2:
                if money_token_re.match(toks[-1]):
                    total = safe_money(toks[-1])
            m = re.search(r"TOTAL\s+([\d,]+\.\d{2})", ln, flags=re.I)
            if m:
                total = safe_money(m.group(1))
        toks = ln.split()
        if len(toks) < 5:
        code = toks[0]
        if not code_re.match(code):


# --- PAGE 344 ---

        qty = toks[-3]
        unit = toks[-2]
        line_total = toks[-1]
        if not qty.isdigit():
        if not money_token_re.match(unit) or not money_token_re.match(line_total)
        unit_f = safe_money(unit)
        lt_f = safe_money(line_total)
        if unit_f is None or lt_f is None:
        items.append({
            "code": code,
            "qty": int(qty),
            "unit": unit_f,
            "line_total": lt_f,
        })
    return total, items
def load_or_parse_receipts(all_patient_ids, pdf_dir: Path, cache_path: Path, forc
    if cache_path.exists() and not force:
        obj = load(cache_path)
        if isinstance(obj, dict) and obj.get("version") == "v2" and "lineitems" i
            print(f"[cache] Loaded receipt parse cache: {cache_path}")
            return obj["lineitems"], obj["totals"], obj["status"]
        else:
            print(f"[cache] Cache format mismatch -> reparsing: {cache_path}")
    rows = []
    totals_rows = []
    status_rows = []
    bad = 0
    all_patient_ids = list(map(int, all_patient_ids))
    n = len(all_patient_ids)


# --- PAGE 345 ---

    for i, pid in enumerate(all_patient_ids, 1):
        pdf_path = pdf_dir / f"receipt_{pid}.pdf"
        if not pdf_path.exists():
            status_rows.append({"patient_id": pid, "parse_ok": 0, "backend": "mis
            bad += 1
        text, backend, err = extract_text_pdf(pdf_path)
        if not text.strip():
            status_rows.append({"patient_id": pid, "parse_ok": 0, "backend": back
            bad += 1
        total, items = parse_receipt_text(text)
        if total is None or len(items) == 0:
            status_rows.append({"patient_id": pid, "parse_ok": 0, "backend": back
            bad += 1
        totals_rows.append({"patient_id": pid, "pdf_total": total, "backend": bac
        for it in items:
            rows.append({"patient_id": pid, **it})
        status_rows.append({"patient_id": pid, "parse_ok": 1, "backend": backend,
        if i % 250 == 0 or i == n:
            print(f"[receipt-parse] {i}/{n} done | bad={bad}")
    lineitems = pd.DataFrame(rows)
    totals = pd.DataFrame(totals_rows)
    status = pd.DataFrame(status_rows)
    dump({"version": "v2", "lineitems": lineitems, "totals": totals, "status": st
    print(f"[cache] Wrote receipt parse cache: {cache_path} | bad={bad} | rows={l
    return lineitems, totals, status
# Parse all receipts for union(train,test)


# --- PAGE 346 ---

all_ids = sorted(set(train_full["patient_id"].astype(int)) | set(test_full["patie
lineitems_df, totals_df, receipt_status_df = load_or_parse_receipts(all_ids, PDF_
print(f"[info] parsed lineitems rows: {len(lineitems_df)} | patients_w_lines: {li
print(f"[info] receipt parse OK rate: {receipt_status_df['parse_ok'].mean():.4f}"
# Receipt integrity vs prior cost
prior_cost_map = pd.concat([
    train_full[["patient_id", "prior_ed_cost_5y_usd"]],
    test_full[["patient_id", "prior_ed_cost_5y_usd"]],
], axis=0).drop_duplicates("patient_id").set_index("patient_id")["prior_ed_cost_5y
totals_df = totals_df.copy()
totals_df["prior_cost"] = totals_df["patient_id"].map(prior_cost_map)
totals_df["abs_diff"] = (totals_df["pdf_total"] - totals_df["prior_cost"]).abs()
missing_total_ids = set(all_ids) - set(totals_df["patient_id"].astype(int).tolist
print("\n=== RECEIPT TOTAL INTEGRITY ===")
print(f"missing_total_rate: {len(missing_total_ids) / len(all_ids):.4f}  (missing
print("abs_diff (overall) summary:")
print(totals_df["abs_diff"].describe(percentiles=[0.5,0.9,0.99]))
mismatch = totals_df[(totals_df["abs_diff"] > 1e-6) | totals_df["prior_cost"].isn
mismatch.to_csv(OUT_DIR / "receipt_total_mismatches_iter24.csv", index=False)
if len(mismatch) or len(missing_total_ids):
    print(f"[write] {OUT_DIR/'receipt_total_mismatches_iter24.csv'}  mismatches={
# -------------------------
# 3) Build receipt features (per code + category + price deflation/dispersion)
# -------------------------
def build_receipt_features(lineitems: pd.DataFrame, patients_df: pd.DataFrame):
    if lineitems is None or len(lineitems) == 0:
        return pd.DataFrame({"patient_id": patients_df["patient_id"].unique(), "p
    li = lineitems.copy()
    li["patient_id"] = li["patient_id"].astype(int)
    li["code"] = li["code"].astype("string")
    li["qty"] = li["qty"].astype(int)
    li["unit"] = li["unit"].astype(float)
    li["line_total"] = li["line_total"].astype(float)


# --- PAGE 347 ---

    # attach insurance for deflation
    pat = patients_df[["patient_id", "insurance"]].copy()
    pat["patient_id"] = pat["patient_id"].astype(int)
    pat["insurance"] = pat["insurance"].astype("string")
    li = li.merge(pat, on="patient_id", how="left")
    # insurance+code median unit price
    med_unit = li.groupby(["insurance", "code"], dropna=False)["unit"].median()
    key = pd.MultiIndex.from_frame(li[["insurance", "code"]])
    li["unit_med_ins_code"] = med_unit.reindex(key).to_numpy()
    # fallback: overall code median
    med_code = li.groupby(["code"], dropna=False)["unit"].median()
    li.loc[li["unit_med_ins_code"].isna(), "unit_med_ins_code"] = li.loc[li["unit_
    li["expected_line_cost"] = li["unit_med_ins_code"] * li["qty"]
    li["unit_ratio"] = li["unit"] / (li["unit_med_ins_code"] + 1e-9)
    # Category map (tailored to your 18 codes)
    def code_to_cat(code: str) -> str:
        if code in {"99281","99282","99283","99284","99285"}:
            return "ed_em"
        if code in {"99291","99292"}:
            return "crit_care"
        if code in {"85025","87070","84484"}:
            return "lab"
        if code in {"71045","70450","74177"}:
            return "imaging"
        if code in {"G0378"}:
            return "obs"
        if code in {"31500","36556","36620","92950"}:
            return "severe_proc"
        return "other"
    li["cat"] = li["code"].astype(str).map(code_to_cat).astype("string")
    # patient-level base
    g = li.groupby("patient_id", dropna=False)
    base = pd.DataFrame({
        "patient_id": g.size().index,


# --- PAGE 348 ---

        "pdf_n_line_items": g.size().values,
        "pdf_total_qty": g["qty"].sum().values,
        "pdf_total_line_cost": g["line_total"].sum().values,
        "pdf_n_unique_codes": g["code"].nunique().values,
        "pdf_unit_mean": g["unit"].mean().values,
        "pdf_unit_std": g["unit"].std(ddof=0).fillna(0).values,
        "pdf_expected_total": g["expected_line_cost"].sum().values,
        "pdf_price_index_ins_code": (g["line_total"].sum() / (g["expected_line_co
        "pdf_unit_ratio_std": g["unit_ratio"].std(ddof=0).fillna(0).values,
    })
    # per-code counts/costs/shares
    cnt = li.pivot_table(index="patient_id", columns="code", values="qty", aggfun
    cnt.columns = [f"pdf_cnt_code_{c}" for c in cnt.columns.astype(str)]
    cost = li.pivot_table(index="patient_id", columns="code", values="line_total"
    cost.columns = [f"pdf_cost_code_{c}" for c in cost.columns.astype(str)]
    share = cost.div(cost.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    share.columns = [c.replace("pdf_cost_code_", "pdf_share_code_") for c in cost
    # per-code unit stats
    unit_mean = li.pivot_table(index="patient_id", columns="code", values="unit",
    unit_mean.columns = [f"pdf_unit_mean_{c}" for c in unit_mean.columns.astype(st
    unit_std = li.pivot_table(index="patient_id", columns="code", values="unit",
    unit_std.columns = [f"pdf_unit_std_{c}" for c in unit_std.columns.astype(str)
    unit_std = unit_std.fillna(0.0)
    # per-category costs/shares
    cat_cost = li.pivot_table(index="patient_id", columns="cat", values="line_tot
    cat_cost.columns = [f"pdf_cost_cat_{c}" for c in cat_cost.columns.astype(str)
    cat_share = cat_cost.div(cat_cost.sum(axis=1).replace(0, np.nan), axis=0).fil
    cat_share.columns = [c.replace("pdf_cost_cat_", "pdf_share_cat_") for c in cat
    # additional severity flags
    severe_codes = {"31500","36556","36620","92950"}
    crit_codes = {"99291","99292"}
    li["is_severe"] = li["code"].astype(str).isin(severe_codes).astype(int)
    li["is_crit"] = li["code"].astype(str).isin(crit_codes).astype(int)
    sev = li.groupby("patient_id")[["is_severe","is_crit"]].max().rename(columns=
    # ED E/M level features (99281-99285)


# --- PAGE 349 ---

    em_level_map = {"99281":1,"99282":2,"99283":3,"99284":4,"99285":5}
    li["em_level"] = li["code"].astype(str).map(em_level_map)
    em_g = li.dropna(subset=["em_level"]).groupby("patient_id")["em_level"]
    em_feat = pd.DataFrame({
        "pdf_em_level_mean": em_g.mean(),
        "pdf_em_level_max": em_g.max(),
        "pdf_em_level_min": em_g.min(),
        "pdf_em_lines": em_g.size(),
    }).fillna(0.0)
    feat = (base
            .set_index("patient_id")
            .join([cnt, cost, share, unit_mean, unit_std, cat_cost, cat_share, se
            .fillna(0.0)
            .reset_index())
    feat["pdf_parse_ok"] = 1
    return feat
if RECEIPT_FEATURES_CACHE.exists() and not FORCE_REPARSE_RECEIPTS:
    rf = load(RECEIPT_FEATURES_CACHE)
    if isinstance(rf, dict) and rf.get("version") == "v2" and "receipt_features"
        receipt_feat = rf["receipt_features"]
        print(f"[cache] Loaded receipt features: {RECEIPT_FEATURES_CACHE} | shape
    else:
        receipt_feat = build_receipt_features(lineitems_df, pd.concat([train_full
        dump({"version":"v2","receipt_features": receipt_feat}, RECEIPT_FEATURES_
        print(f"[cache] Wrote receipt features: {RECEIPT_FEATURES_CACHE} | shape=
else:
    receipt_feat = build_receipt_features(lineitems_df, pd.concat([train_full[["p
    dump({"version":"v2","receipt_features": receipt_feat}, RECEIPT_FEATURES_CACH
    print(f"[cache] Wrote receipt features: {RECEIPT_FEATURES_CACHE} | shape={rec
# Merge receipt features (outer to keep all)
train_full = train_full.merge(receipt_feat, on="patient_id", how="left")
test_full  = test_full.merge(receipt_feat, on="patient_id", how="left")
# For truly missing receipts (parse fail), fill parse_ok=0 and numeric=0
for df in (train_full, test_full):
    if "pdf_parse_ok" not in df.columns:


# --- PAGE 350 ---

        df["pdf_parse_ok"] = 0
    df["pdf_parse_ok"] = df["pdf_parse_ok"].fillna(0).astype(int)
# -------------------------
# 4) Admissions patient-level features (with "trend" features)
# -------------------------
def build_admissions_patient_features(adm_train_path: Path, adm_test_path: Path):
    adm_tr = pd.read_csv(adm_train_path)
    adm_te = pd.read_csv(adm_test_path)
    # remove target leak from train side for feature building (readmit_30d not us
    if "readmit_30d" in adm_tr.columns:
        adm_tr_feat = adm_tr.drop(columns=["readmit_30d"])
    else:
        adm_tr_feat = adm_tr
    adm = pd.concat([adm_tr_feat, adm_te], axis=0, ignore_index=True)
    adm["patient_id"] = adm["patient_id"].astype(int)
    adm["primary_dx"] = adm["primary_dx"].astype("string")
    # basic aggregations
    g = adm.groupby("patient_id", dropna=False)
    feat = pd.DataFrame({
        "patient_id": g.size().index,
        "adm_n": g.size().values,
        "adm_los_sum": g["los_days"].sum().values,
        "adm_los_mean": g["los_days"].mean().values,
        "adm_los_max": g["los_days"].max().values,
        "adm_charlson_mean": g["charlson_band"].mean().values,
        "adm_charlson_max": g["charlson_band"].max().values,
        "adm_edvis6m_sum": g["ed_visits_6m"].sum().values,
        "adm_edvis6m_mean": g["ed_visits_6m"].mean().values,
        "adm_edvis6m_max": g["ed_visits_6m"].max().values,
        "adm_emergent_n": g["acuity_emergent"].sum().values,
        "adm_emergent_rate": g["acuity_emergent"].mean().values,
        "adm_discharge_wd_mean": g["discharge_weekday"].mean().values,
    })
    # dx counts
    dx_cnt = adm.pivot_table(index="patient_id", columns="primary_dx", values="ad


# --- PAGE 351 ---

    dx_cnt.columns = [f"adm_dx_cnt_{c}" for c in dx_cnt.columns.astype(str)]
    # "last admission" features by admission_id (proxy of recency)
    last = adm.sort_values("admission_id").groupby("patient_id").tail(1).set_inde
    last_feat = last[["los_days","acuity_emergent","charlson_band","ed_visits_6m"
    last_feat.rename(columns={
        "los_days":"adm_last_los",
        "acuity_emergent":"adm_last_emergent",
        "charlson_band":"adm_last_charlson",
        "ed_visits_6m":"adm_last_edvis6m",
        "discharge_weekday":"adm_last_discharge_wd",
        "primary_dx":"adm_last_dx",
    }, inplace=True)
    last_feat["adm_last_dx"] = last_feat["adm_last_dx"].astype("string")
    # combine
    feat = feat.set_index("patient_id").join(dx_cnt, how="left").join(last_feat,
    return feat
if ADMISSIONS_FEATURES_CACHE.exists():
    obj = load(ADMISSIONS_FEATURES_CACHE)
    if isinstance(obj, dict) and obj.get("version") == "v2" and "adm_feat" in obj
        adm_feat = obj["adm_feat"]
        print(f"[cache] Loaded admissions patient features: {ADMISSIONS_FEATURES_
    else:
        adm_feat = build_admissions_patient_features(ADM_TRAIN, ADM_TEST)
        dump({"version":"v2","adm_feat": adm_feat}, ADMISSIONS_FEATURES_CACHE)
        print(f"[cache] Wrote admissions patient features: {ADMISSIONS_FEATURES_CA
else:
    adm_feat = build_admissions_patient_features(ADM_TRAIN, ADM_TEST)
    dump({"version":"v2","adm_feat": adm_feat}, ADMISSIONS_FEATURES_CACHE)
    print(f"[cache] Wrote admissions patient features: {ADMISSIONS_FEATURES_CACHE
# merge
train_full = train_full.merge(adm_feat, on="patient_id", how="left")
test_full  = test_full.merge(adm_feat, on="patient_id", how="left")
# Trend feature: recent 6m ED visits vs 5y average (expected 6m = prior_visits_5y/
for df in (train_full, test_full):


# --- PAGE 352 ---

    df["adm_edvis6m_rise"] = df.get("adm_edvis6m_max", 0) - (df["prior_ed_visits_
    df["adm_edvis6m_ratio"] = (df.get("adm_edvis6m_max", 0) + 1e-3) / ((df["prior_
# -------------------------
# 5) Quick EDA: residual correlation for admissions features
# -------------------------
y = train_full["ed_cost_next3y_usd"].astype(float).values
# baseline: group median by (primary_chronic, insurance, prior_cost decile)
tmp = train_full.copy()
tmp["cost_decile"] = pd.qcut(tmp["prior_ed_cost_5y_usd"].rank(method="first"), 10
grp = ["primary_chronic", "insurance", "cost_decile"]
med_map = tmp.groupby(grp, dropna=False)["ed_cost_next3y_usd"].median()
tmp["_baseline"] = pd.MultiIndex.from_frame(tmp[grp]).map(med_map).astype(float)
tmp["_resid"] = tmp["ed_cost_next3y_usd"] - tmp["_baseline"]
adm_cols = [c for c in train_full.columns if c.startswith("adm_") and c not in ["
corr_rows = []
for c in adm_cols:
    s = train_full[c]
    if s.isna().all():
    # Spearman via rank-corr
    x = s.fillna(s.median()).astype(float)
    rho_t = pd.Series(x).corr(pd.Series(y), method="spearman")
    rho_r = pd.Series(x).corr(tmp["_resid"], method="spearman")
    corr_rows.append([c, rho_t, rho_r, float(s.isna().mean()), float(s.mean()), f
corr_df = pd.DataFrame(corr_rows, columns=["feature","spearman_target","spearman_
corr_df.sort_values("spearman_resid", ascending=False).head(25).to_csv(OUT_DIR /
print("\n=== TOP 10 admissions features by Spearman(residual) ===")
print(corr_df.sort_values("spearman_resid", ascending=False).head(10).to_string(i
# -------------------------
# 6) Feature Injection Audit utilities
# -------------------------
CAT_COLS_CORE = ["primary_chronic","sex","insurance","zip3"]
NUM_COLS_CORE = ["age","prior_ed_visits_5y","prior_ed_cost_5y_usd","is_outlier"]
def prep_X(df: pd.DataFrame, use_cols, cat_cols):
    X = df[use_cols].copy()


# --- PAGE 353 ---

    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype("string").fillna("UNK").astype("category")
    return X
def feature_audit(name, cols, df_train):
    cols = list(cols)
    nunique = df_train[cols].nunique(dropna=False)
    const_cols = nunique[nunique <= 1].index.tolist()
    miss_rate = df_train[cols].isna().mean().sort_values(ascending=False)
    print(f"\n[feature-audit] {name}: n_cols={len(cols)} | const_cols={len(const_
    print(miss_rate.head(8).to_string())
    if const_cols:
        print("  const cols (first 15):", const_cols[:15])
    return const_cols
# Receipt columns bucket
receipt_cols = [c for c in train_full.columns if c.startswith("pdf_")]
# Admissions columns bucket (excluding categorical adm_last_dx handled separately
adm_num_cols = [c for c in train_full.columns if c.startswith("adm_") and c not i
adm_cat_cols = ["adm_last_dx"] if "adm_last_dx" in train_full.columns else []
# -------------------------
# 7) CV Ablations (LightGBM + optional CatBoost)
# -------------------------
def make_strata(df: pd.DataFrame):
    # keep it simple but stable: chronic + insurance + outlier + prior_cost decil
    x = df.copy()
    x["cost_dec"] = pd.qcut(x["prior_ed_cost_5y_usd"].rank(method="first"), 10, l
    return (x["primary_chronic"].astype(str) + "_" +
            x["insurance"].astype(str) + "_" +
            x["is_outlier"].astype(str) + "_" +
            x["cost_dec"].astype(str))
def cv_lgb(name, cols, cat_cols):
    X = prep_X(train_full, cols, cat_cols)
    y = train_full["ed_cost_next3y_usd"].astype(float).values
    out_mask = train_full["is_outlier"].values.astype(bool)
    strata = make_strata(train_full)


# --- PAGE 354 ---

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train_full), dtype=float)
    fold_mae = []
    fold_out_mae = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, strata), 1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        model = lgb.LGBMRegressor(
            n_estimators=12000,
            learning_rate=0.02,
            num_leaves=64,
            min_data_in_leaf=30,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            random_state=SEED + fold,
            objective="regression_l1",
            n_jobs=-1,
        )
        model.fit(
            eval_set=[(X_va, y_va)],
            eval_metric="mae",
            callbacks=[lgb.early_stopping(stopping_rounds=300, verbose=False)]
        )
        pred = model.predict(X_va, num_iteration=model.best_iteration_)
        oof[va_idx] = pred
        mae = mean_absolute_error(y_va, pred)
        fold_mae.append(mae)
        out_idx = va_idx[out_mask[va_idx]]
        out_mae = mean_absolute_error(y[out_idx], oof[out_idx]) if len(out_idx) e


# --- PAGE 355 ---

        fold_out_mae.append(out_mae)
        print(f"  [{name}] fold {fold}: MAE={mae:.3f} | out_n={len(out_idx):3d} |
    oof_mae = mean_absolute_error(y, oof)
    out_oof_mae = mean_absolute_error(y[out_mask], oof[out_mask]) if out_mask.any
    return {
        "model": name,
        "oof_mae": oof_mae,
        "mae_mean": float(np.mean(fold_mae)),
        "mae_std": float(np.std(fold_mae)),
        "out_oof_mae": out_oof_mae,
        "n_features": len(cols),
        "n_cat": len(cat_cols),
    }
def cv_catboost(name, cols, cat_cols):
    try:
        from catboost import CatBoostRegressor
    except Exception as e:
        print(f"[catboost] not available -> skip ({e})")
        return None
    X = train_full[cols].copy()
    y = train_full["ed_cost_next3y_usd"].astype(float).values
    out_mask = train_full["is_outlier"].values.astype(bool)
    # CatBoost wants string/object for categorical
    cat_idx = []
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype("string").fillna("UNK")
            cat_idx.append(X.columns.get_loc(c))
    strata = make_strata(train_full)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train_full), dtype=float)
    fold_mae, fold_out_mae = [], []


# --- PAGE 356 ---

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, strata), 1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        model = CatBoostRegressor(
            loss_function="MAE",
            eval_metric="MAE",
            iterations=20000,
            learning_rate=0.03,
            depth=8,
            l2_leaf_reg=6.0,
            random_seed=SEED + fold,
            od_type="Iter",
            od_wait=300,
            verbose=False,
        )
        model.fit(X_tr, y_tr, cat_features=cat_idx, eval_set=(X_va, y_va), use_be
        pred = model.predict(X_va)
        oof[va_idx] = pred
        mae = mean_absolute_error(y_va, pred)
        fold_mae.append(mae)
        out_idx = va_idx[out_mask[va_idx]]
        out_mae = mean_absolute_error(y[out_idx], oof[out_idx]) if len(out_idx) e
        fold_out_mae.append(out_mae)
        print(f"  [{name}] fold {fold}: MAE={mae:.3f} | out_n={len(out_idx):3d} |
    oof_mae = mean_absolute_error(y, oof)
    out_oof_mae = mean_absolute_error(y[out_mask], oof[out_mask]) if out_mask.any
    return {
        "model": name,
        "oof_mae": oof_mae,
        "mae_mean": float(np.mean(fold_mae)),
        "mae_std": float(np.std(fold_mae)),
        "out_oof_mae": out_oof_mae,
        "n_features": len(cols),


# --- PAGE 357 ---

        "n_cat": len(cat_cols),
    }
# Define ablation feature sets
core_cols = CAT_COLS_CORE + NUM_COLS_CORE
A_cols = core_cols + ["pdf_parse_ok"] + [c for c in receipt_cols if c in train_fu
B_cols = core_cols + ["pdf_parse_ok"] + [c for c in receipt_cols if c in train_fu
# "C" = B plus admissions trend features already in adm_num_cols (adm_edvis6m_ris
C_cols = B_cols
# Feature audit (to catch iter23-like "identical features" bugs)
const_A = feature_audit("A(core+receipt)", A_cols, train_full)
const_B = feature_audit("B(+admissions)", B_cols, train_full)
print("\n[feature-diff] B \\ A (first 30):", sorted(set(B_cols) - set(A_cols))[:3
print("[feature-diff] A \\ B (first 10):", sorted(set(A_cols) - set(B_cols))[:10]
# Run LightGBM CV
print("\n=== CV ABLATIONS (Iter24, LightGBM L1) ===")
res = []
res.append(cv_lgb("A_core+receipt_LGB", A_cols, CAT_COLS_CORE))
res.append(cv_lgb("B_add_admissions_LGB", B_cols, CAT_COLS_CORE + adm_cat_cols))
# Optional CatBoost
if RUN_CATBOOST:
    print("\n=== CV ABLATIONS (Iter24, CatBoost MAE) ===")
    cbA = cv_catboost("A_core+receipt_Cat", A_cols, CAT_COLS_CORE)
    if cbA: res.append(cbA)
    cbB = cv_catboost("B_add_admissions_Cat", B_cols, CAT_COLS_CORE + adm_cat_col
    if cbB: res.append(cbB)
res_df = pd.DataFrame(res).sort_values("oof_mae")
res_df.to_csv(OUT_DIR / "ablation_results_iter24.csv", index=False)
print("\n=== SUMMARY (sorted by OOF MAE) ===")
print(res_df.to_string(index=False))
print(f"\n[write] {OUT_DIR/'ablation_results_iter24.csv'}")
# -------------------------
# 8) Minimal plots (optional)


# --- PAGE 358 ---

5) 你跑完 Iter24 后，把这 4 个结果贴回来（我就能据此给你下一步“冲
6) 最后解释一下你提到的“Quantile vs L1 输出一样”现象
# -------------------------
# Scatter prior vs target for sanity
plt.figure()
plt.scatter(train_full["prior_ed_cost_5y_usd"], train_full["ed_cost_next3y_usd"],
plt.xlabel("prior_ed_cost_5y_usd")
plt.ylabel("ed_cost_next3y_usd")
plt.title("Train: prior cost vs future cost")
plt.show()
print("\n[DONE] Iter24 EDA + ablations complete.")
print(f"Outputs written to: {OUT_DIR}")


# --- PAGE 359 ---

objective=regression_l1  目标是 MAE
objective=quantile, alpha=0.5  本质也是“条件中位数”
```

### C17 — Code cell starting at page 373
```python
# Iter25: Sublinear-baseline EDA + Receipt utilization/price decomposition
# Copy-paste into ONE Jupyter cell and run.
import os
import re
import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
# Optional (won't crash if missing)
try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False
# -----------------------
# Paths (edit if needed)


# --- PAGE 374 ---

# -----------------------
BASE_DIR = Path(r"D:\AgentDs\agent_ds_healthcare")
TRAIN_CSV = BASE_DIR / "ed_cost_train.csv"
TEST_CSV  = BASE_DIR / "ed_cost_test.csv"
PATIENTS_CSV = BASE_DIR / "patients.csv"
ADM_TRAIN = BASE_DIR / "admissions_train.csv"
ADM_TEST  = BASE_DIR / "admissions_test.csv"
# Use your known-good receipt cache (Iter10)
RECEIPT_CACHE = BASE_DIR / "cache_iter10" / "receipts_parsed.joblib"
OUT_DIR = BASE_DIR / "cache_iter25_sublinear_baseline_eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
N_SPLITS = 5
TOL = 1e-2  # 1 cent tolerance for money comparisons
print(f"[paths] BASE_DIR={BASE_DIR}")
print(f"[paths] OUT_DIR={OUT_DIR}")
print(f"[paths] RECEIPT_CACHE={RECEIPT_CACHE}")
# -----------------------
# Load tables
# -----------------------
train = pd.read_csv(TRAIN_CSV)
test  = pd.read_csv(TEST_CSV)
patients = pd.read_csv(PATIENTS_CSV)
train_full = train.merge(patients, on="patient_id", how="left")
test_full  = test.merge(patients, on="patient_id", how="left")
print(f"[info] train_full={train_full.shape}  test_full={test_full.shape}")
# Outlier definition (your team's)
train_full["is_outlier"] = ((train_full["prior_ed_cost_5y_usd"] <= 50.0) & (train_
outlier_n = int(train_full["is_outlier"].sum())
print(f"[info] outlier_n={outlier_n} / {len(train_full)}")


# --- PAGE 375 ---

# -----------------------
# Load & featurize receipts (from cache)
# -----------------------
parsed = load(RECEIPT_CACHE)  # dict[patient_id] -> dict with "items", maybe "pdf_
# Known code sets (from your EDA)
ED_EM = {"99281","99282","99283","99284","99285"}
CRIT  = {"99291","99292"}
SEVERE_PROC = {"31500","92950","36556","36620"}  # intub, CPR, CVC, art line
IMAGING = {"70450","74177","71045"}              # CT head, CT A/P, CXR
LAB = {"85025","87070","84484"}                  # CBC, culture, troponin
OBS = {"G0378"}
ALL_CODES = sorted(list(ED_EM | CRIT | SEVERE_PROC | IMAGING | LAB | OBS))
print(f"[info] expected receipt code universe={len(ALL_CODES)} -> {ALL_CODES}")
def safe_items(obj):
    if not isinstance(obj, dict):
        return []
    items = obj.get("items", [])
    return items if isinstance(items, list) else []
def parse_amount(x):
    try:
        return float(x)
    except Exception:
        return np.nan
def build_receipt_features(parsed_dict):
    rows = []
    line_rows = []  # for global reference prices
    for pid, obj in parsed_dict.items():
        try:
            pid_int = int(pid)
        except Exception:
        items = safe_items(obj)
        codes = []


# --- PAGE 376 ---

        amounts = []
        for it in items:
            if not isinstance(it, dict):
            code = it.get("code", None)
            amt = parse_amount(it.get("amount", np.nan))
            if isinstance(code, str):
                code = code.strip()
            else:
                code = None
            if code:
                codes.append(code)
                amounts.append(amt)
                line_rows.append({"patient_id": pid_int, "code": code, "amount":
        total_sum = float(np.nansum(amounts)) if len(amounts) else 0.0
        n_lines = int(len(codes))
        n_unique = int(len(set(codes)))
        # Counts per code (unweighted)
        cc = {f"pdf_cnt_code_{c}": 0 for c in ALL_CODES}
        costc = {f"pdf_cost_code_{c}": 0.0 for c in ALL_CODES}
        for c, a in zip(codes, amounts):
            if c in cc:
                cc[f"pdf_cnt_code_{c}"] += 1
                if np.isfinite(a):
                    costc[f"pdf_cost_code_{c}"] += float(a)
        # Category costs
        def sum_cost(code_set):
            s = 0.0
            for c, a in zip(codes, amounts):
                if c in code_set and np.isfinite(a):
                    s += float(a)
            return s
        cost_em = sum_cost(ED_EM)
        cost_crit = sum_cost(CRIT)
        cost_severe = sum_cost(SEVERE_PROC)
        cost_img = sum_cost(IMAGING)


# --- PAGE 377 ---

        cost_lab = sum_cost(LAB)
        cost_obs = sum_cost(OBS)
        cost_other = max(0.0, total_sum - (cost_em + cost_crit + cost_severe + co
        # Shares (safe div)
        def share(x):
            return float(x / total_sum) if total_sum > 0 else 0.0
        # E/M level stats
        em_levels = []
        for c in codes:
            if c in ED_EM:
                try:
                    lvl = int(c[-1])  # 99281 -> 1 ... 99285 -> 5
                    em_levels.append(lvl)
                except Exception:
        em_levels = np.array(em_levels, dtype=float) if len(em_levels) else np.ar
        # Entropy of code cost shares
        code_costs = np.array([parse_amount(a) for a in amounts], dtype=float)
        code_costs = code_costs[np.isfinite(code_costs)]
        ent = 0.0
        if total_sum > 0 and len(code_costs) > 0:
            # by code
            tmp = {}
            for c, a in zip(codes, amounts):
                if not np.isfinite(a):
                tmp[c] = tmp.get(c, 0.0) + float(a)
            probs = np.array([v / total_sum for v in tmp.values() if v > 0], dtyp
            probs = probs[probs > 0]
            ent = float(-(probs * np.log(probs)).sum()) if len(probs) else 0.0
        row = {
            "patient_id": pid_int,
            "pdf_total_sum_items": total_sum,
            "pdf_n_line_items": n_lines,
            "pdf_n_unique_codes": n_unique,
            "pdf_cost_ed_em": cost_em,


# --- PAGE 378 ---

            "pdf_cost_crit": cost_crit,
            "pdf_cost_severe_proc": cost_severe,
            "pdf_cost_imaging": cost_img,
            "pdf_cost_lab": cost_lab,
            "pdf_cost_obs": cost_obs,
            "pdf_cost_other": cost_other,
            "pdf_share_ed_em": share(cost_em),
            "pdf_share_crit": share(cost_crit),
            "pdf_share_severe_proc": share(cost_severe),
            "pdf_share_imaging": share(cost_img),
            "pdf_share_lab": share(cost_lab),
            "pdf_share_obs": share(cost_obs),
            "pdf_share_other": share(cost_other),
            "pdf_has_crit": int(any(c in CRIT for c in codes)),
            "pdf_has_severe_proc": int(any(c in SEVERE_PROC for c in codes)),
            "pdf_em_level_mean": float(np.mean(em_levels)) if len(em_levels) else
            "pdf_em_level_max": float(np.max(em_levels)) if len(em_levels) else 0
            "pdf_em_high_frac": float(np.mean(em_levels >= 4)) if len(em_levels)
            "pdf_code_cost_entropy": ent,
        }
        row.update(cc)
        row.update(costc)
        rows.append(row)
    feat = pd.DataFrame(rows)
    lines = pd.DataFrame(line_rows)
    return feat, lines
receipt_feat, receipt_lines = build_receipt_features(parsed)
print(f"[info] receipt_feat shape={receipt_feat.shape} | receipt_lines shape={rec
# Receipt integrity check vs prior cost (use sum_items as truth)
tmp_chk = train_full[["patient_id","prior_ed_cost_5y_usd"]].merge(
    receipt_feat[["patient_id","pdf_total_sum_items"]],
    on="patient_id", how="left"
)
absdiff = (tmp_chk["prior_ed_cost_5y_usd"] - tmp_chk["pdf_total_sum_items"]).abs(
match_rate = float((absdiff <= TOL).mean())
print(f"[receipt-check] match_rate(sum_items_total vs prior_cost)={match_rate:.4f
print(absdiff.describe())


# --- PAGE 379 ---

# Build reference (median) unit price per code for standardized utilization total
# Here qty==1 in your dataset, but we treat amount as per-line total.
ref_price = receipt_lines.groupby("code")["amount"].median().to_dict()
def standardized_total_for_patient(row):
    # sum of median prices for each line item code (utilization proxy)
    pid = row["patient_id"]
    # For speed: use patient's per-code counts from receipt_feat
    total = 0.0
    for c in ALL_CODES:
        cnt = row.get(f"pdf_cnt_code_{c}", 0)
        if cnt and c in ref_price and np.isfinite(ref_price[c]):
            total += float(cnt) * float(ref_price[c])
    return total
receipt_feat["pdf_standardized_total"] = receipt_feat.apply(standardized_total_fo
receipt_feat["pdf_price_index"] = receipt_feat["pdf_total_sum_items"] / receipt_f
receipt_feat["pdf_price_index"] = receipt_feat["pdf_price_index"].replace([np.inf
# -----------------------
# Admissions patient-level features (hard features only)
# -----------------------
adm_tr = pd.read_csv(ADM_TRAIN)
adm_te = pd.read_csv(ADM_TEST)
adm_all = pd.concat([adm_tr, adm_te], axis=0, ignore_index=True)
# numeric cast
for col in ["los_days", "acuity_emergent", "charlson_band", "ed_visits_6m", "disc
    if col in adm_all.columns:
        adm_all[col] = pd.to_numeric(adm_all[col], errors="coerce")
# Aggregate
g = adm_all.groupby("patient_id", dropna=False)
adm_feat = pd.DataFrame({
    "patient_id": g.size().index.astype(int),
    "adm_n": g.size().values.astype(int),
    "adm_los_mean": g["los_days"].mean().values,
    "adm_los_max": g["los_days"].max().values,
    "adm_charlson_mean": g["charlson_band"].mean().values,


# --- PAGE 380 ---

    "adm_charlson_max": g["charlson_band"].max().values,
    "adm_emergent_rate": g["acuity_emergent"].mean().values,
    "adm_edvis6m_mean": g["ed_visits_6m"].mean().values,
    "adm_edvis6m_max": g["ed_visits_6m"].max().values,
    "adm_edvis6m_sum": g["ed_visits_6m"].sum().values,
})
# dx counts
if "primary_dx" in adm_all.columns:
    dx_ct = pd.crosstab(adm_all["patient_id"], adm_all["primary_dx"])
    dx_ct.columns = [f"adm_dx_cnt_{c}" for c in dx_ct.columns]
    dx_ct = dx_ct.reset_index().rename(columns={"patient_id":"patient_id"})
    dx_ct["patient_id"] = dx_ct["patient_id"].astype(int)
    adm_feat = adm_feat.merge(dx_ct, on="patient_id", how="left")
# -----------------------
# Merge all features
# -----------------------
all_train = train_full.merge(receipt_feat, on="patient_id", how="left").merge(adm_
all_test  = test_full.merge(receipt_feat, on="patient_id", how="left").merge(adm_
# Fill some numeric NaNs
num_cols = all_train.select_dtypes(include=[np.number]).columns.tolist()
for c in num_cols:
    if c in ["ed_cost_next3y_usd"]:
    med = all_train[c].median()
    all_train[c] = all_train[c].fillna(med)
    if c in all_test.columns:
        all_test[c] = all_test[c].fillna(med)
# -----------------------
# CV setup
# -----------------------
def strat_label(df):
    # Keep chronic+insurance+outlier balanced
    return (df["primary_chronic"].astype(str) + "|" + df["insurance"].astype(str)
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


# --- PAGE 381 ---

y = all_train["ed_cost_next3y_usd"].values
prior_cost = all_train["prior_ed_cost_5y_usd"].values
prior_vis = all_train["prior_ed_visits_5y"].values
groups = list(zip(all_train["primary_chronic"].astype(str), all_train["insurance"
is_out = all_train["is_outlier"].values.astype(int)
# -----------------------
# Baseline 1: group-median ratio (old)
# -----------------------
def cv_group_ratio(df):
    oof = np.zeros(len(df), dtype=float)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(df, strat_label(df)) , star
        tr = df.iloc[tr_idx].copy()
        va = df.iloc[va_idx].copy()
        denom = tr["prior_ed_cost_5y_usd"].values + 1.0
        tr["ratio"] = tr["ed_cost_next3y_usd"].values / denom
        med_ratio = tr.groupby(["primary_chronic","insurance"])["ratio"].median()
        # predict
        pred = []
        for _, r in va.iterrows():
            key = (r["primary_chronic"], r["insurance"])
            rr = float(med_ratio.get(key, tr["ratio"].median()))
            pred.append(rr * (r["prior_ed_cost_5y_usd"] + 1.0))
        pred = np.array(pred, dtype=float)
        oof[va_idx] = pred
        mae = mean_absolute_error(va["ed_cost_next3y_usd"].values, pred)
        out_mask = (va["is_outlier"].values.astype(int)==1)
        out_mae = mean_absolute_error(va.loc[out_mask,"ed_cost_next3y_usd"].value
        print(f"[ratio] fold{fold}: MAE={mae:.3f} | out_MAE={out_mae:.3f} | out_n
    return oof
# -----------------------
# Baseline 2: power-law/log baseline per group
# log1p(y)=a+b*log1p(cost)+c*log1p(visits)
# -----------------------


# --- PAGE 382 ---

def cv_powerlaw(df):
    oof = np.zeros(len(df), dtype=float)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(df, strat_label(df)) , star
        tr = df.iloc[tr_idx].copy()
        va = df.iloc[va_idx].copy()
        # fit per group
        models = {}
        for (ch, ins), sub in tr.groupby(["primary_chronic","insurance"], dropna=
            X = np.column_stack([
                np.log1p(sub["prior_ed_cost_5y_usd"].values),
                np.log1p(sub["prior_ed_visits_5y"].values),
            ])
            ylog = np.log1p(sub["ed_cost_next3y_usd"].values)
            lr = LinearRegression()
            lr.fit(X, ylog)
            models[(ch, ins)] = lr
        # predict
        pred = np.zeros(len(va), dtype=float)
        for i, r in enumerate(va.itertuples(index=False)):
            key = (getattr(r, "primary_chronic"), getattr(r, "insurance"))
            lr = models.get(key, None)
            if lr is None:
                # fallback global
                lr = LinearRegression().fit(
                    np.column_stack([np.log1p(tr["prior_ed_cost_5y_usd"].values),
                    np.log1p(tr["ed_cost_next3y_usd"].values)
                )
            Xv = np.array([[np.log1p(getattr(r, "prior_ed_cost_5y_usd")),
                            np.log1p(getattr(r, "prior_ed_visits_5y"))]], dtype=f
            yhat = float(np.expm1(lr.predict(Xv)[0]))
            pred[i] = max(0.0, yhat)
        oof[va_idx] = pred
        mae = mean_absolute_error(va["ed_cost_next3y_usd"].values, pred)
        out_mask = (va["is_outlier"].values.astype(int)==1)
        out_mae = mean_absolute_error(va.loc[out_mask,"ed_cost_next3y_usd"].value
        print(f"[powerlaw] fold{fold}: MAE={mae:.3f} | out_MAE={out_mae:.3f} | ou
    return oof


# --- PAGE 383 ---

# -----------------------
# Baseline 3: isotonic per group (monotone, flexible)
# -----------------------
def cv_isotonic(df):
    oof = np.zeros(len(df), dtype=float)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(df, strat_label(df)) , star
        tr = df.iloc[tr_idx].copy()
        va = df.iloc[va_idx].copy()
        iso_models = {}
        for (ch, ins), sub in tr.groupby(["primary_chronic","insurance"], dropna=
            x = sub["prior_ed_cost_5y_usd"].values.astype(float)
            yv = sub["ed_cost_next3y_usd"].values.astype(float)
            # If not enough unique x, fallback later
            if len(np.unique(x)) < 5:
            iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
            iso.fit(x, yv)
            iso_models[(ch, ins)] = iso
        pred = np.zeros(len(va), dtype=float)
        global_iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        global_iso.fit(tr["prior_ed_cost_5y_usd"].values.astype(float), tr["ed_co
        for i, r in enumerate(va.itertuples(index=False)):
            key = (getattr(r, "primary_chronic"), getattr(r, "insurance"))
            iso = iso_models.get(key, global_iso)
            yhat = float(iso.predict([float(getattr(r, "prior_ed_cost_5y_usd"))])
            pred[i] = max(0.0, yhat)
        oof[va_idx] = pred
        mae = mean_absolute_error(va["ed_cost_next3y_usd"].values, pred)
        out_mask = (va["is_outlier"].values.astype(int)==1)
        out_mae = mean_absolute_error(va.loc[out_mask,"ed_cost_next3y_usd"].value
        print(f"[isotonic] fold{fold}: MAE={mae:.3f} | out_MAE={out_mae:.3f} | ou
    return oof
# -----------------------
# Baseline 4: very-light LightGBM (optional)


# --- PAGE 384 ---

# Only a few engineered transforms + key receipt decompositions + small admission
# -----------------------
def cv_light_lgb(df):
    if not HAS_LGB:
        print("[lgb] LightGBM not available, skipping.")
        return None
    oof = np.zeros(len(df), dtype=float)
    # define features
    df2 = df.copy()
    df2["log_prior_cost"] = np.log1p(df2["prior_ed_cost_5y_usd"].values)
    df2["sqrt_prior_cost"] = np.sqrt(np.clip(df2["prior_ed_cost_5y_usd"].values,
    df2["log_prior_vis"] = np.log1p(df2["prior_ed_visits_5y"].values)
    df2["prior_cost_per_visit"] = df2["prior_ed_cost_5y_usd"] / (df2["prior_ed_vi
    feat_cols = [
        # receipt decomposition
        # admissions "hard" features
    ]
    feat_cols = [c for c in feat_cols if c in df2.columns]
    # categorical encoding: use simple category codes (fast) for chronic/insuranc
    cat_cols = ["primary_chronic","insurance","sex","zip3"]
    for c in cat_cols:
        df2[c] = df2[c].astype("category")
        df2[c] = df2[c].cat.codes.astype(int)
    feat_cols += cat_cols
    for fold, (tr_idx, va_idx) in enumerate(skf.split(df2, strat_label(df2)) , st
        tr = df2.iloc[tr_idx]
        va = df2.iloc[va_idx]


# --- PAGE 385 ---

        X_tr = tr[feat_cols]
        y_tr = tr["ed_cost_next3y_usd"].values
        X_va = va[feat_cols]
        y_va = va["ed_cost_next3y_usd"].values
        model = lgb.LGBMRegressor(
            objective="regression_l1",
            n_estimators=5000,
            learning_rate=0.02,
            num_leaves=64,
            min_data_in_leaf=30,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            lambda_l2=2.0,
            random_state=RANDOM_STATE + fold,
            n_jobs=-1,
        )
        model.fit(
            eval_set=[(X_va, y_va)],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(stopping_rounds=300, verbose=False)]
        )
        pred = model.predict(X_va, num_iteration=model.best_iteration_)
        pred = np.clip(pred, 0, None)
        oof[va_idx] = pred
        mae = mean_absolute_error(y_va, pred)
        out_mask = (va["is_outlier"].values.astype(int)==1)
        out_mae = mean_absolute_error(y_va[out_mask], pred[out_mask]) if out_mask
        print(f"[lgb_light] fold{fold}: MAE={mae:.3f} | out_MAE={out_mae:.3f} | o
    return oof, feat_cols
# -----------------------
# Run baselines
# -----------------------
print("\n=== CV: baseline comparison ===")


# --- PAGE 386 ---

oof_ratio = cv_group_ratio(all_train)
oof_pow   = cv_powerlaw(all_train)
oof_iso   = cv_isotonic(all_train)
res = []
def summarize(name, oof):
    mae = mean_absolute_error(y, oof)
    out_mae = mean_absolute_error(y[is_out==1], oof[is_out==1]) if (is_out==1).any
    res.append({"model": name, "oof_mae": mae, "out_oof_mae": out_mae})
summarize("group_ratio", oof_ratio)
summarize("powerlaw_log", oof_pow)
summarize("isotonic_group", oof_iso)
lgb_out = None
if HAS_LGB:
    lgb_out = cv_light_lgb(all_train)
    if lgb_out is not None:
        oof_lgb, feat_cols = lgb_out
        summarize("lgb_light_sublinear", oof_lgb)
summary_df = pd.DataFrame(res).sort_values("oof_mae")
print("\n=== SUMMARY (lower is better) ===")
print(summary_df.to_string(index=False))
summary_df.to_csv(OUT_DIR / "iter25_baseline_summary.csv", index=False)
print(f"[write] {OUT_DIR / 'iter25_baseline_summary.csv'}")
# -----------------------
# Plot: prior cost vs target + fitted curves (global)
# -----------------------
plt.figure(figsize=(8,6))
plt.scatter(all_train["prior_ed_cost_5y_usd"], all_train["ed_cost_next3y_usd"], s
plt.xlabel("prior_ed_cost_5y_usd")
plt.ylabel("ed_cost_next3y_usd")
plt.title("Train: prior cost vs future cost")
plt.tight_layout()
plt.savefig(OUT_DIR / "scatter_prior_vs_future.png", dpi=160)
plt.show()


# --- PAGE 387 ---

6) 你跑完这段 Iter25 之后，我们要看什么（决定下一步路线）
1. SUMMARY (lower is better)  表里四个 baseline 的 OOF MAE
7) 最后：ZIP3 的定位（你最初问的 domain point）
# Plot predicted vs actual for the best baseline
best_name = summary_df.iloc[0]["model"]
best_oof = {"group_ratio":oof_ratio, "powerlaw_log":oof_pow, "isotonic_group":oof_
if best_name == "lgb_light_sublinear" and lgb_out is not None:
    best_oof = oof_lgb
if best_oof is not None:
    plt.figure(figsize=(6,6))
    plt.scatter(y, best_oof, s=10)
    plt.xlabel("actual")
    plt.ylabel("oof_pred")
    plt.title(f"OOF pred vs actual ({best_name})")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"oof_pred_vs_actual_{best_name}.png", dpi=160)
    plt.show()
print("\n[DONE] Iter25 EDA finished.")


# --- PAGE 388 ---
```

### C18 — Code cell starting at page 408
```python
# Iter25 (FIXED): Sublinear-baseline EDA + robust receipt-cache handling
# Copy-paste into ONE Jupyter cell and run.
import re
import math
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
# Optional LightGBM (won't crash if missing)
try:
    import lightgbm as lgb
    HAS_LGB = True
    try:
        from lightgbm.callback import early_stopping as lgb_early_stopping
    except Exception:
        lgb_early_stopping = None
except Exception:
    HAS_LGB = False
    lgb = None
    lgb_early_stopping = None
# -----------------------
# Paths (edit if needed)
# -----------------------
BASE_DIR = Path(r"D:\AgentDs\agent_ds_healthcare")
TRAIN_CSV = BASE_DIR / "ed_cost_train.csv"
TEST_CSV  = BASE_DIR / "ed_cost_test.csv"
PATIENTS_CSV = BASE_DIR / "patients.csv"


# --- PAGE 409 ---

ADM_TRAIN = BASE_DIR / "admissions_train.csv"
ADM_TEST  = BASE_DIR / "admissions_test.csv"
# Receipt cache (your iter10)
RECEIPT_CACHE = BASE_DIR / "cache_iter10" / "receipts_parsed.joblib"
OUT_DIR = BASE_DIR / "cache_iter25_sublinear_baseline_eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
N_SPLITS = 5
TOL = 1e-2  # 1 cent tolerance for money comparisons
print(f"[paths] BASE_DIR={BASE_DIR}")
print(f"[paths] OUT_DIR={OUT_DIR}")
print(f"[paths] RECEIPT_CACHE={RECEIPT_CACHE}")
# -----------------------
# Helpers
# -----------------------
def safe_read_csv(path: Path):
    if path is None:
        return None
    if not path.exists():
        print(f"[warn] file not found: {path} (skipping)")
        return None
    return pd.read_csv(path)
def can_int(x) -> bool:
    try:
        int(str(x).strip())
        return True
    except Exception:
        return False
def normalize_code(x):
    if x is None:
        return None
    try:


# --- PAGE 410 ---

        if isinstance(x, (int, np.integer)):
            return str(int(x))
        if isinstance(x, float) and np.isfinite(x) and float(x).is_integer():
            return str(int(x))
    except Exception:
    s = str(x).strip()
    # remove trailing ".0" if came from float-like rendering
    if re.fullmatch(r"\d+\.0", s):
        s = s.split(".")[0]
    return s if s else None
def parse_amount(x):
    if x is None:
        return np.nan
    try:
        # strip currency symbols/commas if any
        if isinstance(x, str):
            xs = x.strip().replace(",", "").replace("$", "")
            return float(xs)
        return float(x)
    except Exception:
        return np.nan
def infer_col(df: pd.DataFrame, candidates):
    """Find a column by exact match (case-insensitive) or substring match."""
    if df is None or df.empty:
        return None
    cols = list(df.columns)
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in low:
            return low[cand.lower()]
    # substring heuristic
    for cand in candidates:
        for c in cols:
            if cand.lower() in str(c).lower():
                return c


# --- PAGE 411 ---

    return None
# -----------------------
# Known code sets (from your EDA)
# -----------------------
ED_EM = {"99281","99282","99283","99284","99285"}
CRIT  = {"99291","99292"}
SEVERE_PROC = {"31500","92950","36556","36620"}  # intub, CPR, CVC, art line
IMAGING = {"70450","74177","71045"}              # CT head, CT A/P, CXR
LAB = {"85025","87070","84484"}                  # CBC, culture, troponin
OBS = {"G0378"}
ALL_CODES = sorted(list(ED_EM | CRIT | SEVERE_PROC | IMAGING | LAB | OBS))
print(f"[info] expected receipt code universe={len(ALL_CODES)} -> {ALL_CODES}")
# -----------------------
# Load base tables
# -----------------------
train = safe_read_csv(TRAIN_CSV)
test  = safe_read_csv(TEST_CSV)
patients = safe_read_csv(PATIENTS_CSV)
assert train is not None and test is not None and patients is not None, "Missing
# standardize dtypes
for df in (train, test, patients):
    df["patient_id"] = pd.to_numeric(df["patient_id"], errors="coerce").astype(int
# zip3: keep leading zeros if present
if "zip3" in patients.columns:
    patients["zip3"] = patients["zip3"].astype(str)
    patients["zip3"] = patients["zip3"].str.replace(r"\.0$", "", regex=True)
    patients.loc[patients["zip3"].str.lower().isin(["nan","none"]), "zip3"] = "UN
    # if numeric-like, pad
    patients["zip3"] = patients["zip3"].apply(lambda s: s.zfill(3) if s.isdigit()
train_full = train.merge(patients, on="patient_id", how="left")
test_full  = test.merge(patients, on="patient_id", how="left")
print(f"[info] train_full={train_full.shape}  test_full={test_full.shape}")


# --- PAGE 412 ---

# Outlier definition (your team's)
train_full["is_outlier"] = ((train_full["prior_ed_cost_5y_usd"] <= 50.0) & (train_
outlier_n = int(train_full["is_outlier"].sum())
print(f"[info] outlier_n={outlier_n} / {len(train_full)}")
patient_ids_all = pd.concat([train_full["patient_id"], test_full["patient_id"]]).
patient_ids_all = np.sort(patient_ids_all)
# -----------------------
# Receipt cache loading (robust to multiple formats)
# -----------------------
def build_receipt_features_from_patient_dict(patient_dict: dict, patient_ids_all:
    rows = []
    line_rows = []
    CODE_KEYS = ["code", "cpt", "hcpcs", "billing_code", "service_code"]
    AMT_KEYS  = ["amount", "line_total", "line_total_usd", "total", "charge", "li
    def safe_items(obj):
        if not isinstance(obj, dict):
            return []
        items = obj.get("items", [])
        return items if isinstance(items, list) else []
    for pid, obj in patient_dict.items():
        if not can_int(pid):
        pid_int = int(str(pid).strip())
        items = safe_items(obj)
        codes, amounts = [], []
        # also grab header total if present (for debugging only)
        header_total = np.nan
        if isinstance(obj, dict):
            header_total = parse_amount(obj.get("pdf_total", np.nan))
        for it in items:
            if not isinstance(it, dict):


# --- PAGE 413 ---

            code = None
            for k in CODE_KEYS:
                if k in it:
                    code = normalize_code(it.get(k))
            amt = np.nan
            for k in AMT_KEYS:
                if k in it:
                    amt = parse_amount(it.get(k))
            if code is None:
            codes.append(code)
            amounts.append(amt)
            line_rows.append({"patient_id": pid_int, "code": code, "amount": amt}
        total_sum = float(np.nansum(amounts)) if len(amounts) else 0.0
        n_lines = int(len(codes))
        n_unique = int(len(set(codes)))
        # counts and costs per known code
        cnt = {f"pdf_cnt_code_{c}": 0 for c in ALL_CODES}
        cost = {f"pdf_cost_code_{c}": 0.0 for c in ALL_CODES}
        for c, a in zip(codes, amounts):
            if c in ALL_CODES:
                cnt[f"pdf_cnt_code_{c}"] += 1
                if np.isfinite(a):
                    cost[f"pdf_cost_code_{c}"] += float(a)
        def sum_cost(code_set):
            s = 0.0
            for c, a in zip(codes, amounts):
                if c in code_set and np.isfinite(a):
                    s += float(a)
            return s
        cost_em = sum_cost(ED_EM)


# --- PAGE 414 ---

        cost_crit = sum_cost(CRIT)
        cost_sev = sum_cost(SEVERE_PROC)
        cost_img = sum_cost(IMAGING)
        cost_lab = sum_cost(LAB)
        cost_obs = sum_cost(OBS)
        cost_other = max(0.0, total_sum - (cost_em + cost_crit + cost_sev + cost_
        def share(x):
            return float(x / total_sum) if total_sum > 0 else 0.0
        # E/M levels
        em_levels = []
        for c in codes:
            if c in ED_EM and len(c) == 5 and c[-1].isdigit():
                em_levels.append(int(c[-1]))
        em_levels = np.array(em_levels, dtype=float)
        # entropy over all codes in this receipt (not only known codes)
        ent = 0.0
        if total_sum > 0:
            tmp = {}
            for c, a in zip(codes, amounts):
                if not np.isfinite(a):
                tmp[c] = tmp.get(c, 0.0) + float(a)
            probs = np.array([v / total_sum for v in tmp.values() if v > 0], dtyp
            probs = probs[probs > 0]
            if len(probs):
                ent = float(-(probs * np.log(probs)).sum())
        row = {
            "patient_id": pid_int,
            "pdf_total_sum_items": total_sum,
            "pdf_total_header": header_total,
            "pdf_n_line_items": n_lines,
            "pdf_n_unique_codes": n_unique,
            "pdf_cost_ed_em": cost_em,
            "pdf_cost_crit": cost_crit,
            "pdf_cost_severe_proc": cost_sev,
            "pdf_cost_imaging": cost_img,


# --- PAGE 415 ---

            "pdf_cost_lab": cost_lab,
            "pdf_cost_obs": cost_obs,
            "pdf_cost_other": cost_other,
            "pdf_share_ed_em": share(cost_em),
            "pdf_share_crit": share(cost_crit),
            "pdf_share_severe_proc": share(cost_sev),
            "pdf_share_imaging": share(cost_img),
            "pdf_share_lab": share(cost_lab),
            "pdf_share_obs": share(cost_obs),
            "pdf_share_other": share(cost_other),
            "pdf_has_crit": int(any(c in CRIT for c in codes)),
            "pdf_has_severe_proc": int(any(c in SEVERE_PROC for c in codes)),
            "pdf_em_level_mean": float(np.mean(em_levels)) if len(em_levels) else
            "pdf_em_level_max": float(np.max(em_levels)) if len(em_levels) else 0
            "pdf_em_high_frac": float(np.mean(em_levels >= 4)) if len(em_levels)
            "pdf_code_cost_entropy": ent,
        }
        row.update(cnt)
        row.update(cost)
        rows.append(row)
    feat = pd.DataFrame(rows)
    lines = pd.DataFrame(line_rows)
    # guarantee required columns exist (even if empty)
    required = ["patient_id", "pdf_total_sum_items"]
    for c in required:
        if c not in feat.columns:
            feat[c] = pd.Series(dtype=float)
    # ensure all patients represented
    base = pd.DataFrame({"patient_id": patient_ids_all.astype(int)})
    feat = base.merge(feat, on="patient_id", how="left")
    return feat, lines
def build_receipt_features_from_lineitems_df(lineitems: pd.DataFrame, patient_ids_
    df = lineitems.copy()
    pid_col = infer_col(df, ["patient_id", "patient"])
    code_col = infer_col(df, ["code", "cpt", "hcpcs"])
    amt_col  = infer_col(df, ["amount", "line_total", "line total", "line_total_u


# --- PAGE 416 ---

    if pid_col is None or code_col is None or amt_col is None:
        print("[warn] Could not infer columns in lineitems_df. Columns are:")
        print(df.columns.tolist())
        # fallback empty
        base = pd.DataFrame({"patient_id": patient_ids_all.astype(int)})
        base["pdf_total_sum_items"] = np.nan
        return base, pd.DataFrame(columns=["patient_id","code","amount"])
    df = df[[pid_col, code_col, amt_col]].rename(columns={pid_col:"patient_id", c
    df["patient_id"] = pd.to_numeric(df["patient_id"], errors="coerce")
    df = df.dropna(subset=["patient_id"])
    df["patient_id"] = df["patient_id"].astype(int)
    df["code"] = df["code"].apply(normalize_code)
    df["amount"] = df["amount"].apply(parse_amount)
    df = df.dropna(subset=["code"])
    # keep amounts even if NaN for counting; but for sum we ignore NaN
    lines = df.copy()
    # totals
    g = df.groupby("patient_id", dropna=False)
    feat = pd.DataFrame({"patient_id": g.size().index.astype(int)})
    feat["pdf_total_sum_items"] = g["amount"].sum(min_count=1).fillna(0.0).values
    feat["pdf_n_line_items"] = g.size().values.astype(int)
    feat["pdf_n_unique_codes"] = g["code"].nunique().values.astype(int)
    # per-code counts/costs (known codes only)
    ct = pd.crosstab(df["patient_id"], df["code"])
    ct = ct.reindex(columns=ALL_CODES, fill_value=0)
    ct.columns = [f"pdf_cnt_code_{c}" for c in ct.columns]
    cost = df.pivot_table(index="patient_id", columns="code", values="amount", ag
    cost = cost.reindex(columns=ALL_CODES, fill_value=0.0)
    cost.columns = [f"pdf_cost_code_{c}" for c in cost.columns]
    feat = feat.set_index("patient_id").join(ct, how="left").join(cost, how="left
    # category costs
    def sum_by_codes(code_set, out_col):


# --- PAGE 417 ---

        sub = df[df["code"].isin(code_set)].groupby("patient_id")["amount"].sum(m
        return sub.rename(out_col)
    cost_em   = sum_by_codes(ED_EM, "pdf_cost_ed_em")
    cost_crit = sum_by_codes(CRIT, "pdf_cost_crit")
    cost_sev  = sum_by_codes(SEVERE_PROC, "pdf_cost_severe_proc")
    cost_img  = sum_by_codes(IMAGING, "pdf_cost_imaging")
    cost_lab  = sum_by_codes(LAB, "pdf_cost_lab")
    cost_obs  = sum_by_codes(OBS, "pdf_cost_obs")
    feat = feat.set_index("patient_id")
    for s in [cost_em, cost_crit, cost_sev, cost_img, cost_lab, cost_obs]:
        feat = feat.join(s, how="left")
    feat = feat.fillna(0.0)
    total = feat["pdf_total_sum_items"].values.astype(float)
    feat["pdf_cost_other"] = np.maximum(0.0, total - (
        feat["pdf_cost_ed_em"] + feat["pdf_cost_crit"] + feat["pdf_cost_severe_pr
        feat["pdf_cost_imaging"] + feat["pdf_cost_lab"] + feat["pdf_cost_obs"]
    ))
    def share(series):
        return np.where(total > 0, series.values.astype(float) / total, 0.0)
    feat["pdf_share_ed_em"] = share(feat["pdf_cost_ed_em"])
    feat["pdf_share_crit"]  = share(feat["pdf_cost_crit"])
    feat["pdf_share_severe_proc"] = share(feat["pdf_cost_severe_proc"])
    feat["pdf_share_imaging"] = share(feat["pdf_cost_imaging"])
    feat["pdf_share_lab"] = share(feat["pdf_cost_lab"])
    feat["pdf_share_obs"] = share(feat["pdf_cost_obs"])
    feat["pdf_share_other"] = share(feat["pdf_cost_other"])
    # flags
    feat["pdf_has_crit"] = (feat[[f"pdf_cnt_code_{c}" for c in CRIT if f"pdf_cnt_
    feat["pdf_has_severe_proc"] = (feat[[f"pdf_cnt_code_{c}" for c in SEVERE_PROC
    # E/M levels stats
    em_df = df[df["code"].isin(ED_EM)].copy()
    if len(em_df):
        em_df["lvl"] = em_df["code"].apply(lambda s: int(s[-1]) if isinstance(s,


# --- PAGE 418 ---

        em_g = em_df.groupby("patient_id")["lvl"]
        feat["pdf_em_level_mean"] = em_g.mean().reindex(feat.index).fillna(0.0).v
        feat["pdf_em_level_max"] = em_g.max().reindex(feat.index).fillna(0.0).val
        feat["pdf_em_high_frac"] = em_g.apply(lambda x: float(np.mean(np.array(x)
    else:
        feat["pdf_em_level_mean"] = 0.0
        feat["pdf_em_level_max"] = 0.0
        feat["pdf_em_high_frac"] = 0.0
    # entropy over known-code costs (works fine if only 18 codes exist)
    cost_cols = [f"pdf_cost_code_{c}" for c in ALL_CODES if f"pdf_cost_code_{c}"
    if len(cost_cols):
        mat = feat[cost_cols].values.astype(float)
        row_sum = mat.sum(axis=1, keepdims=True)
        p = np.divide(mat, row_sum, out=np.zeros_like(mat), where=row_sum>0)
        ent = -np.sum(np.where(p>0, p*np.log(p), 0.0), axis=1)
        feat["pdf_code_cost_entropy"] = ent
    else:
        feat["pdf_code_cost_entropy"] = 0.0
    feat = feat.reset_index()
    # ensure all patients represented
    base = pd.DataFrame({"patient_id": patient_ids_all.astype(int)})
    feat = base.merge(feat, on="patient_id", how="left")
    return feat, lines
def load_receipts_flexible(cache_obj, patient_ids_all: np.ndarray):
    Returns (receipt_feat, receipt_lines) always with at least:
      - receipt_feat['patient_id']
      - receipt_feat['pdf_total_sum_items']
      - receipt_lines columns: patient_id, code, amount (possibly empty)
    # Case 1: direct patient dict
    if isinstance(cache_obj, dict):
        # If it looks like a patient_id dict (most keys int-like)
        keys = list(cache_obj.keys())
        sample = keys[: min(50, len(keys))]
        int_like = sum(can_int(k) for k in sample) / max(1, len(sample))


# --- PAGE 419 ---

        # Also handle nested common wrappers
        for k in ["by_pid", "by_patient", "parsed_by_patient", "receipts_by_patie
            if k in cache_obj and isinstance(cache_obj[k], dict):
                cache_obj = cache_obj[k]
                keys = list(cache_obj.keys())
                sample = keys[: min(50, len(keys))]
                int_like = sum(can_int(k) for k in sample) / max(1, len(sample))
        if int_like >= 0.6:
            print(f"[receipt-cache] treating as patient_dict (int_like={int_like:
            return build_receipt_features_from_patient_dict(cache_obj, patient_id
        # Case 2: dict container that holds a lineitems DataFrame
        dfs = [(k,v) for k,v in cache_obj.items() if isinstance(v, pd.DataFrame)]
        if dfs:
            # pick best-scoring candidate
            def score_df(df):
                cols = [str(c).lower() for c in df.columns]
                score = 0
                if any("patient" in c for c in cols): score += 1
                if any("code" in c or "cpt" in c or "hcpcs" in c for c in cols):
                if any("amount" in c or "line" in c or "total" in c or "charge" i
                return score
            best_k, best_df = None, None
            best_score = -1
            for k,v in dfs:
                s = score_df(v)
                if s > best_score:
                    best_score = s
                    best_k, best_df = k, v
            print(f"[receipt-cache] treating as df_container; using df key={best_
            return build_receipt_features_from_lineitems_df(best_df, patient_ids_
    # Case 3: cache itself is a DataFrame of lineitems
    if isinstance(cache_obj, pd.DataFrame):
        print("[receipt-cache] treating as lineitems_df directly")
        return build_receipt_features_from_lineitems_df(cache_obj, patient_ids_al


# --- PAGE 420 ---

    # Fallback: empty receipts
    print("[warn] Unknown receipt cache format; falling back to empty receipt fea
    base = pd.DataFrame({"patient_id": patient_ids_all.astype(int)})
    base["pdf_total_sum_items"] = np.nan
    return base, pd.DataFrame(columns=["patient_id","code","amount"])
# Load receipt cache
if not RECEIPT_CACHE.exists():
    print(f"[warn] Receipt cache missing: {RECEIPT_CACHE}. Falling back to empty
    receipt_feat = pd.DataFrame({"patient_id": patient_ids_all.astype(int), "pdf_t
    receipt_lines = pd.DataFrame(columns=["patient_id","code","amount"])
else:
    cache_obj = load(RECEIPT_CACHE)
    print(f"[debug] receipt cache type={type(cache_obj)}")
    if isinstance(cache_obj, dict):
        print(f"[debug] receipt cache dict keys sample={list(cache_obj.keys())[:5
    receipt_feat, receipt_lines = load_receipts_flexible(cache_obj, patient_ids_a
# Safety: guarantee columns exist
if "patient_id" not in receipt_feat.columns:
    receipt_feat["patient_id"] = patient_ids_all.astype(int)
if "pdf_total_sum_items" not in receipt_feat.columns:
    receipt_feat["pdf_total_sum_items"] = np.nan
# De-dup
receipt_feat = receipt_feat.drop_duplicates("patient_id", keep="first")
print(f"[info] receipt_feat shape={receipt_feat.shape} | receipt_lines shape={rec
# -----------------------
# Receipt integrity check vs prior cost (use sum_items as truth)
# -----------------------
tmp_chk = train_full[["patient_id","prior_ed_cost_5y_usd"]].merge(
    receipt_feat[["patient_id","pdf_total_sum_items"]],
    on="patient_id", how="left"
)
missing_rate = float(tmp_chk["pdf_total_sum_items"].isna().mean())
absdiff = (tmp_chk["prior_ed_cost_5y_usd"] - tmp_chk["pdf_total_sum_items"]).abs(
mask_valid = tmp_chk["pdf_total_sum_items"].notna()


# --- PAGE 421 ---

match_rate_valid = float((absdiff[mask_valid] <= TOL).mean()) if mask_valid.any()
match_rate_all   = float((absdiff.fillna(np.inf) <= TOL).mean())
print(f"[receipt-check] missing_rate(sum_items_total)={missing_rate:.4f}")
print(f"[receipt-check] match_rate(valid only)={match_rate_valid:.4f} | match_rat
print("[receipt-check] absdiff (valid only) summary:")
print(absdiff[mask_valid].describe())
# -----------------------
# Build reference median unit price per code for standardized utilization
# -----------------------
ref_price = {}
if not receipt_lines.empty and "code" in receipt_lines.columns and "amount" in re
    tmp_lines = receipt_lines.copy()
    tmp_lines["code"] = tmp_lines["code"].apply(normalize_code)
    tmp_lines["amount"] = tmp_lines["amount"].apply(parse_amount)
    tmp_lines = tmp_lines.dropna(subset=["code"])
    ref_price = tmp_lines.groupby("code")["amount"].median().to_dict()
# Ensure all per-code count columns exist
for c in ALL_CODES:
    cc = f"pdf_cnt_code_{c}"
    if cc not in receipt_feat.columns:
        receipt_feat[cc] = 0
# Vectorized standardized total
count_cols = [f"pdf_cnt_code_{c}" for c in ALL_CODES]
price_vec = np.array([ref_price.get(c, np.nan) for c in ALL_CODES], dtype=float)
price_vec = np.nan_to_num(price_vec, nan=0.0)
receipt_feat["pdf_standardized_total"] = receipt_feat[count_cols].values.astype(f
# price index
num = pd.to_numeric(receipt_feat["pdf_total_sum_items"], errors="coerce").astype(
den = pd.to_numeric(receipt_feat["pdf_standardized_total"], errors="coerce").asty
pi = np.where(den > 0, num / den, 1.0)
pi = np.where(np.isfinite(pi), pi, 1.0)
receipt_feat["pdf_price_index"] = pi.astype(np.float32)


# --- PAGE 422 ---

# -----------------------
# Admissions patient-level features (robust)
# -----------------------
adm_tr = safe_read_csv(ADM_TRAIN)
adm_te = safe_read_csv(ADM_TEST)
if adm_tr is None or adm_te is None:
    print("[warn] admissions files missing -> skipping admissions features.")
    adm_feat = pd.DataFrame({"patient_id": patient_ids_all.astype(int)})
else:
    adm_all = pd.concat([adm_tr, adm_te], axis=0, ignore_index=True)
    if "patient_id" not in adm_all.columns:
        print("[warn] admissions missing patient_id col -> skipping admissions fe
        adm_feat = pd.DataFrame({"patient_id": patient_ids_all.astype(int)})
    else:
        adm_all["patient_id"] = pd.to_numeric(adm_all["patient_id"], errors="coer
        adm_all = adm_all.dropna(subset=["patient_id"])
        adm_all["patient_id"] = adm_all["patient_id"].astype(int)
        # numeric cast if present
        for col in ["los_days", "acuity_emergent", "charlson_band", "ed_visits_6m
            if col in adm_all.columns:
                adm_all[col] = pd.to_numeric(adm_all[col], errors="coerce")
        g = adm_all.groupby("patient_id", dropna=False)
        adm_feat = pd.DataFrame({"patient_id": g.size().index.astype(int)})
        adm_feat["adm_n"] = g.size().values.astype(int)
        # safe aggregations
        if "los_days" in adm_all.columns:
            adm_feat["adm_los_mean"] = g["los_days"].mean().values
            adm_feat["adm_los_max"]  = g["los_days"].max().values
            adm_feat["adm_los_sum"]  = g["los_days"].sum(min_count=1).values
        else:
            adm_feat["adm_los_mean"] = np.nan
            adm_feat["adm_los_max"]  = np.nan
            adm_feat["adm_los_sum"]  = np.nan
        if "charlson_band" in adm_all.columns:


# --- PAGE 423 ---

            adm_feat["adm_charlson_mean"] = g["charlson_band"].mean().values
            adm_feat["adm_charlson_max"]  = g["charlson_band"].max().values
        else:
            adm_feat["adm_charlson_mean"] = np.nan
            adm_feat["adm_charlson_max"]  = np.nan
        if "acuity_emergent" in adm_all.columns:
            adm_feat["adm_emergent_rate"] = g["acuity_emergent"].mean().values
            adm_feat["adm_emergent_n"] = g["acuity_emergent"].sum(min_count=1).va
        else:
            adm_feat["adm_emergent_rate"] = np.nan
            adm_feat["adm_emergent_n"] = np.nan
        if "ed_visits_6m" in adm_all.columns:
            adm_feat["adm_edvis6m_mean"] = g["ed_visits_6m"].mean().values
            adm_feat["adm_edvis6m_max"]  = g["ed_visits_6m"].max().values
            adm_feat["adm_edvis6m_sum"]  = g["ed_visits_6m"].sum(min_count=1).val
        else:
            adm_feat["adm_edvis6m_mean"] = np.nan
            adm_feat["adm_edvis6m_max"]  = np.nan
            adm_feat["adm_edvis6m_sum"]  = np.nan
        # dx counts if present
        if "primary_dx" in adm_all.columns:
            dx_ct = pd.crosstab(adm_all["patient_id"], adm_all["primary_dx"])
            dx_ct.columns = [f"adm_dx_cnt_{c}" for c in dx_ct.columns]
            dx_ct = dx_ct.reset_index().rename(columns={"patient_id":"patient_id"
            dx_ct["patient_id"] = dx_ct["patient_id"].astype(int)
            adm_feat = adm_feat.merge(dx_ct, on="patient_id", how="left")
        # ensure all patients represented
        base = pd.DataFrame({"patient_id": patient_ids_all.astype(int)})
        adm_feat = base.merge(adm_feat, on="patient_id", how="left")
print(f"[info] adm_feat shape={adm_feat.shape}")
# -----------------------
# Merge all features
# -----------------------
all_train = train_full.merge(receipt_feat, on="patient_id", how="left").merge(adm_


# --- PAGE 424 ---

all_test  = test_full.merge(receipt_feat, on="patient_id", how="left").merge(adm_
# If receipt totals missing, impute from prior cost (safe given dataset design)
all_train["pdf_total_sum_items"] = all_train["pdf_total_sum_items"].fillna(all_tr
all_test["pdf_total_sum_items"]  = all_test["pdf_total_sum_items"].fillna(all_test
# Fill remaining numeric NaNs robustly
num_cols = all_train.select_dtypes(include=[np.number]).columns.tolist()
for c in num_cols:
    if c == "ed_cost_next3y_usd":
    med = float(all_train[c].median()) if c in all_train.columns else 0.0
    if not np.isfinite(med):
        med = 0.0
    all_train[c] = all_train[c].fillna(med)
    if c in all_test.columns:
        all_test[c] = all_test[c].fillna(med)
# Fill critical categoricals
for c in ["primary_chronic","insurance","sex","zip3"]:
    if c in all_train.columns:
        all_train[c] = all_train[c].astype(str).fillna("UNK")
    if c in all_test.columns:
        all_test[c] = all_test[c].astype(str).fillna("UNK")
# -----------------------
# CV setup
# -----------------------
def strat_label(df):
    return (df["primary_chronic"].astype(str) + "|" +
            df["insurance"].astype(str) + "|out=" +
            df["is_outlier"].astype(int).astype(str))
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
y = all_train["ed_cost_next3y_usd"].values.astype(float)
is_out = all_train["is_outlier"].values.astype(int)
# -----------------------
# Baseline 1: group-median ratio


# --- PAGE 425 ---

# -----------------------
def cv_group_ratio(df):
    oof = np.zeros(len(df), dtype=float)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(df, strat_label(df)), start
        tr = df.iloc[tr_idx].copy()
        va = df.iloc[va_idx].copy()
        denom = tr["prior_ed_cost_5y_usd"].values.astype(float) + 1.0
        tr["ratio"] = tr["ed_cost_next3y_usd"].values.astype(float) / denom
        med_ratio = tr.groupby(["primary_chronic","insurance"], dropna=False)["rat
        global_med = float(tr["ratio"].median())
        preds = []
        for _, r in va.iterrows():
            key = (r["primary_chronic"], r["insurance"])
            rr = med_ratio.get(key, global_med)
            rr = float(rr) if np.isfinite(rr) else global_med
            preds.append(rr * (float(r["prior_ed_cost_5y_usd"]) + 1.0))
        pred = np.array(preds, dtype=float)
        oof[va_idx] = pred
        mae = mean_absolute_error(va["ed_cost_next3y_usd"].values, pred)
        out_mask = (va["is_outlier"].values.astype(int)==1)
        out_mae = mean_absolute_error(va.loc[out_mask,"ed_cost_next3y_usd"].value
        print(f"[ratio] fold{fold}: MAE={mae:.3f} | out_MAE={out_mae:.3f} | out_n
    return oof
# -----------------------
# Baseline 2: power-law/log per group
# log1p(y)=a+b*log1p(cost)+c*log1p(visits)
# -----------------------
def cv_powerlaw(df):
    oof = np.zeros(len(df), dtype=float)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(df, strat_label(df)), start
        tr = df.iloc[tr_idx].copy()
        va = df.iloc[va_idx].copy()
        global_lr = LinearRegression()
        global_X = np.column_stack([np.log1p(tr["prior_ed_cost_5y_usd"].values.ast


# --- PAGE 426 ---

                                    np.log1p(tr["prior_ed_visits_5y"].values.asty
        global_lr.fit(global_X, np.log1p(tr["ed_cost_next3y_usd"].values.astype(f
        models = {}
        for (ch, ins), sub in tr.groupby(["primary_chronic","insurance"], dropna=
            X = np.column_stack([np.log1p(sub["prior_ed_cost_5y_usd"].values.asty
                                 np.log1p(sub["prior_ed_visits_5y"].values.astype
            ylog = np.log1p(sub["ed_cost_next3y_usd"].values.astype(float))
            if len(sub) < 30:
            lr = LinearRegression()
            lr.fit(X, ylog)
            models[(ch, ins)] = lr
        pred = np.zeros(len(va), dtype=float)
        for i, r in enumerate(va.itertuples(index=False)):
            key = (getattr(r, "primary_chronic"), getattr(r, "insurance"))
            lr = models.get(key, global_lr)
            Xv = np.array([[np.log1p(float(getattr(r, "prior_ed_cost_5y_usd"))),
                            np.log1p(float(getattr(r, "prior_ed_visits_5y")))]],
            yhat = float(np.expm1(lr.predict(Xv)[0]))
            pred[i] = max(0.0, yhat)
        oof[va_idx] = pred
        mae = mean_absolute_error(va["ed_cost_next3y_usd"].values, pred)
        out_mask = (va["is_outlier"].values.astype(int)==1)
        out_mae = mean_absolute_error(va.loc[out_mask,"ed_cost_next3y_usd"].value
        print(f"[powerlaw] fold{fold}: MAE={mae:.3f} | out_MAE={out_mae:.3f} | ou
    return oof
# -----------------------
# Baseline 3: isotonic per group (monotone, flexible)
# -----------------------
def cv_isotonic(df):
    oof = np.zeros(len(df), dtype=float)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(df, strat_label(df)), start
        tr = df.iloc[tr_idx].copy()
        va = df.iloc[va_idx].copy()
        global_iso = IsotonicRegression(increasing=True, out_of_bounds="clip")


# --- PAGE 427 ---

        global_iso.fit(tr["prior_ed_cost_5y_usd"].values.astype(float),
                       tr["ed_cost_next3y_usd"].values.astype(float))
        iso_models = {}
        for (ch, ins), sub in tr.groupby(["primary_chronic","insurance"], dropna=
            x = sub["prior_ed_cost_5y_usd"].values.astype(float)
            yv = sub["ed_cost_next3y_usd"].values.astype(float)
            if len(np.unique(x)) < 5 or len(sub) < 50:
            iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
            iso.fit(x, yv)
            iso_models[(ch, ins)] = iso
        pred = np.zeros(len(va), dtype=float)
        for i, r in enumerate(va.itertuples(index=False)):
            key = (getattr(r, "primary_chronic"), getattr(r, "insurance"))
            iso = iso_models.get(key, global_iso)
            yhat = float(iso.predict([float(getattr(r, "prior_ed_cost_5y_usd"))])
            pred[i] = max(0.0, yhat)
        oof[va_idx] = pred
        mae = mean_absolute_error(va["ed_cost_next3y_usd"].values, pred)
        out_mask = (va["is_outlier"].values.astype(int)==1)
        out_mae = mean_absolute_error(va.loc[out_mask,"ed_cost_next3y_usd"].value
        print(f"[isotonic] fold{fold}: MAE={mae:.3f} | out_MAE={out_mae:.3f} | ou
    return oof
# -----------------------
# Baseline 4: very-light LightGBM (optional)
# -----------------------
def cv_light_lgb(df):
    if not HAS_LGB:
        print("[lgb] LightGBM not available, skipping.")
        return None
    oof = np.zeros(len(df), dtype=float)
    df2 = df.copy()
    # transforms capturing sublinear relationship
    df2["log_prior_cost"] = np.log1p(df2["prior_ed_cost_5y_usd"].values.astype(fl


# --- PAGE 428 ---

    df2["sqrt_prior_cost"] = np.sqrt(np.clip(df2["prior_ed_cost_5y_usd"].values.a
    df2["log_prior_vis"] = np.log1p(df2["prior_ed_visits_5y"].values.astype(float
    df2["prior_cost_per_visit"] = df2["prior_ed_cost_5y_usd"].values.astype(float
    feat_cols = [
    ]
    feat_cols = [c for c in feat_cols if c in df2.columns]
    # categorical encoding (fast)
    cat_cols = [c for c in ["primary_chronic","insurance","sex","zip3"] if c in d
    for c in cat_cols:
        df2[c] = df2[c].astype("category")
        df2[c] = df2[c].cat.codes.astype(int)
    feat_cols += cat_cols
    for fold, (tr_idx, va_idx) in enumerate(skf.split(df2, strat_label(df2)), sta
        tr = df2.iloc[tr_idx]
        va = df2.iloc[va_idx]
        X_tr = tr[feat_cols]
        y_tr = tr["ed_cost_next3y_usd"].values.astype(float)
        X_va = va[feat_cols]
        y_va = va["ed_cost_next3y_usd"].values.astype(float)
        model = lgb.LGBMRegressor(
            objective="regression_l1",
            n_estimators=8000,
            learning_rate=0.02,
            num_leaves=64,
            min_data_in_leaf=30,
            feature_fraction=0.85,
            bagging_fraction=0.85,


# --- PAGE 429 ---

            bagging_freq=1,
            lambda_l2=2.0,
            random_state=RANDOM_STATE + fold,
            n_jobs=-1,
            force_col_wise=True,
        )
        callbacks = []
        if lgb_early_stopping is not None:
            callbacks.append(lgb_early_stopping(stopping_rounds=300, verbose=Fals
        elif hasattr(lgb, "early_stopping"):
            callbacks.append(lgb.early_stopping(stopping_rounds=300, verbose=Fals
        model.fit(
            eval_set=[(X_va, y_va)],
            eval_metric="l1",
            callbacks=callbacks if callbacks else None
        )
        best_iter = getattr(model, "best_iteration_", None)
        pred = model.predict(X_va, num_iteration=best_iter)
        pred = np.clip(pred, 0, None)
        oof[va_idx] = pred
        mae = mean_absolute_error(y_va, pred)
        out_mask = (va["is_outlier"].values.astype(int)==1)
        out_mae = mean_absolute_error(y_va[out_mask], pred[out_mask]) if out_mask
        print(f"[lgb_light] fold{fold}: MAE={mae:.3f} | out_MAE={out_mae:.3f} | o
    return oof, feat_cols
# -----------------------
# Run baselines
# -----------------------
print("\n=== CV: baseline comparison ===")
oof_ratio = cv_group_ratio(all_train)
oof_pow   = cv_powerlaw(all_train)
oof_iso   = cv_isotonic(all_train)
res = []


# --- PAGE 430 ---

def summarize(name, oof):
    mae = mean_absolute_error(y, oof)
    out_mae = mean_absolute_error(y[is_out==1], oof[is_out==1]) if (is_out==1).any
    res.append({"model": name, "oof_mae": mae, "out_oof_mae": out_mae})
summarize("group_ratio", oof_ratio)
summarize("powerlaw_log", oof_pow)
summarize("isotonic_group", oof_iso)
lgb_out = None
oof_lgb = None
feat_cols_lgb = None
if HAS_LGB:
    lgb_out = cv_light_lgb(all_train)
    if lgb_out is not None:
        oof_lgb, feat_cols_lgb = lgb_out
        summarize("lgb_light_sublinear", oof_lgb)
summary_df = pd.DataFrame(res).sort_values("oof_mae")
print("\n=== SUMMARY (lower is better) ===")
print(summary_df.to_string(index=False))
summary_path = OUT_DIR / "iter25_baseline_summary.csv"
summary_df.to_csv(summary_path, index=False)
print(f"[write] {summary_path}")
# -----------------------
# Plot: prior cost vs target
# -----------------------
plt.figure(figsize=(8,6))
plt.scatter(all_train["prior_ed_cost_5y_usd"], all_train["ed_cost_next3y_usd"], s
plt.xlabel("prior_ed_cost_5y_usd")
plt.ylabel("ed_cost_next3y_usd")
plt.title("Train: prior cost vs future cost")
plt.tight_layout()
plt.savefig(OUT_DIR / "scatter_prior_vs_future.png", dpi=160)
plt.show()
# Plot predicted vs actual for best baseline
best_name = summary_df.iloc[0]["model"]


# --- PAGE 431 ---

mismatch。你现在用 TOL=0.01  是对的，建议以后所有 money 对齐都走
absdiff<=0.01  或 np.isclose(..., atol=0.01)  这一套。
1. === SUMMARY (lower is better) ===  那个表（四个 baseline 的 OOF MAE）
2. [receipt-check] missing_rate + match_rate （确认 receipt_total 真没掉）
best_oof = {"group_ratio":oof_ratio, "powerlaw_log":oof_pow, "isotonic_group":oof_
if best_name == "lgb_light_sublinear" and oof_lgb is not None:
    best_oof = oof_lgb
if best_oof is not None:
    plt.figure(figsize=(6,6))
    plt.scatter(y, best_oof, s=10)
    plt.xlabel("actual")
    plt.ylabel("oof_pred")
    plt.title(f"OOF pred vs actual ({best_name})")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"oof_pred_vs_actual_{best_name}.png", dpi=160)
    plt.show()
print("\n[DONE] Iter25 EDA finished.")


# --- PAGE 432 ---
```

### C19 — Code cell starting at page 434
```python
# --- PAGE 435 ---

自动识别 receipt cache 的结构（patient_dict / dict含lineitems_df / tuple(DataFrame,
DataFrame) / 直接 DataFrame 都能处理）
IsotonicRegression(out_of_bounds="clip")  这样设置会把训练范围外的 x “截断”到端
lightgbm.early_stopping()  的 callback 用法也和你现在一致（需要 eval_set 才能生
if c in cc:
    cc[f"pdf_cnt_code_{c}"] += 1
# Iter25 (FIXED): Sublinear-baseline EDA + robust receipt cache loader + correct
# Copy-paste into ONE Jupyter cell and run.
import os
import re
import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.isotonic import IsotonicRegression
```

### C20 — Code cell starting at page 435
```python
# Iter25 (FIXED): Sublinear-baseline EDA + robust receipt cache loader + correct
# Copy-paste into ONE Jupyter cell and run.
import os
import re
import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.isotonic import IsotonicRegression


# --- PAGE 436 ---

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
# Optional (won't crash if missing)
try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False
# -----------------------
# Paths (edit if needed)
# -----------------------
BASE_DIR = Path(r"D:\AgentDs\agent_ds_healthcare")
TRAIN_CSV = BASE_DIR / "ed_cost_train.csv"
TEST_CSV  = BASE_DIR / "ed_cost_test.csv"
PATIENTS_CSV = BASE_DIR / "patients.csv"
ADM_TRAIN = BASE_DIR / "admissions_train.csv"
ADM_TEST  = BASE_DIR / "admissions_test.csv"
# Use your known-good receipt cache (Iter10)
RECEIPT_CACHE = BASE_DIR / "cache_iter10" / "receipts_parsed.joblib"
OUT_DIR = BASE_DIR / "cache_iter25_sublinear_baseline_eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
N_SPLITS = 5
TOL = 1e-2  # 1 cent tolerance for money comparisons
print(f"[paths] BASE_DIR={BASE_DIR}")
print(f"[paths] OUT_DIR={OUT_DIR}")
print(f"[paths] RECEIPT_CACHE={RECEIPT_CACHE}")
# -----------------------
# Helpers
# -----------------------
def to_int_patient_id(x):


# --- PAGE 437 ---

    if x is None:
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    s = str(x).strip()
    try:
        return int(s)
    except Exception:
        m = re.search(r"\d+", s)
        return int(m.group()) if m else None
def parse_amount(x):
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x)
    s = s.replace(",", "")
    s = s.replace("$", "")
    # keep digits, dot, minus
    s = re.sub(r"[^0-9\.\-]+", "", s)
    try:
        return float(s)
    except Exception:
        return np.nan
def guess_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None
# -----------------------
# Load tables
# -----------------------
train = pd.read_csv(TRAIN_CSV)
test  = pd.read_csv(TEST_CSV)
patients = pd.read_csv(PATIENTS_CSV)


# --- PAGE 438 ---

# Ensure patient_id numeric
for df in (train, test, patients):
    if "patient_id" in df.columns:
        df["patient_id"] = pd.to_numeric(df["patient_id"], errors="coerce")
        df["patient_id"] = df["patient_id"].astype("Int64")
# zip3 as zero-padded string (helps categorical stability)
if "zip3" in patients.columns:
    patients["zip3"] = patients["zip3"].astype(str).str.replace(".0", "", regex=F
train_full = train.merge(patients, on="patient_id", how="left")
test_full  = test.merge(patients, on="patient_id", how="left")
# Convert patient_id back to int (after merge)
train_full["patient_id"] = train_full["patient_id"].astype(int)
test_full["patient_id"]  = test_full["patient_id"].astype(int)
print(f"[info] train_full={train_full.shape}  test_full={test_full.shape}")
# Outlier definition (your team's)
train_full["is_outlier"] = ((train_full["prior_ed_cost_5y_usd"] <= 50.0) & (train_
outlier_n = int(train_full["is_outlier"].sum())
print(f"[info] outlier_n={outlier_n} / {len(train_full)}")
# -----------------------
# Receipt code universe (known from your EDA)
# -----------------------
ED_EM = {"99281","99282","99283","99284","99285"}
CRIT  = {"99291","99292"}
SEVERE_PROC = {"31500","92950","36556","36620"}  # intub, CPR, CVC, art line
IMAGING = {"70450","74177","71045"}              # CT head, CT A/P, CXR
LAB = {"85025","87070","84484"}                  # CBC, culture, troponin
OBS = {"G0378"}
ALL_CODES = sorted(list(ED_EM | CRIT | SEVERE_PROC | IMAGING | LAB | OBS))
print(f"[info] expected receipt code universe={len(ALL_CODES)} -> {ALL_CODES}")
# -----------------------
# Robust receipt cache -> receipt_lines (patient_id, code, amount)


# --- PAGE 439 ---

# -----------------------
cache_obj = load(RECEIPT_CACHE)
print(f"[debug] receipt cache type={type(cache_obj)}")
def receipt_lines_from_cache(obj):
    Supports:
      - dict(patient_id -> {'items':[...]})
      - dict with {'lineitems_df': DataFrame} / {'headers_df':..., 'lineitems_df'
      - tuple/list of (headers_df, lineitems_df)
    Returns cleaned DataFrame with columns: patient_id(int), code(str), amount(fl
    # Case 1: DataFrame directly
    if isinstance(obj, pd.DataFrame):
        line_df = obj.copy()
        mode = "dataframe"
    # Case 2: tuple/list of dataframes
    elif isinstance(obj, (tuple, list)) and len(obj) == 2 and all(isinstance(x, p
        line_df = obj[1].copy()
        mode = "tuple(lineitems_df)"
    # Case 3: dict
    elif isinstance(obj, dict):
        # dict has explicit lineitems_df
        if "lineitems_df" in obj and isinstance(obj["lineitems_df"], pd.DataFrame
            line_df = obj["lineitems_df"].copy()
            mode = "dict(lineitems_df)"
        else:
            # heuristically decide if keys are mostly patient_ids
            keys = list(obj.keys())
            int_like = [to_int_patient_id(k) is not None for k in keys[: min(len(
            int_like_rate = float(np.mean(int_like)) if len(int_like) else 0.0
            if int_like_rate >= 0.5:
                # patient_dict style
                mode = f"patient_dict(int_like_rate={int_like_rate:.2f})"
                rows = []
                for k, v in obj.items():
                    pid = to_int_patient_id(k)
                    if pid is None:


# --- PAGE 440 ---

                    items = None
                    if isinstance(v, dict):
                        items = v.get("items", None)
                        if items is None:
                            items = v.get("lineitems", None)
                        if items is None:
                            items = v.get("lines", None)
                    elif isinstance(v, list):
                        items = v
                    else:
                        items = None
                    if items is None:
                    # If items is a DataFrame somehow, convert it
                    if isinstance(items, pd.DataFrame):
                        df_items = items.copy()
                        c_code = guess_col(df_items, ["code","cpt_code","cpt","bi
                        c_amt  = guess_col(df_items, ["amount","line_total","line_
                        if c_code is None or c_amt is None:
                        for _, rr in df_items.iterrows():
                            code = str(rr[c_code]).strip()
                            amt = parse_amount(rr[c_amt])
                            if code and np.isfinite(amt):
                                rows.append({"patient_id": pid, "code": code, "am
                    # Normal list[dict]
                    if isinstance(items, list):
                        for it in items:
                            if not isinstance(it, dict):
                            code = it.get("code", it.get("cpt_code", it.get("cpt"
                            amt = it.get("amount", it.get("line_total", it.get("l
                            code = str(code).strip() if code is not None else ""
                            amt = parse_amount(amt)


# --- PAGE 441 ---

                            if code and np.isfinite(amt):
                                rows.append({"patient_id": pid, "code": code, "am
                line_df = pd.DataFrame(rows)
            else:
                # unknown dict structure -> best effort: try to locate any DataFr
                dfs = [v for v in obj.values() if isinstance(v, pd.DataFrame)]
                if len(dfs) > 0:
                    line_df = dfs[0].copy()
                    mode = "dict(first_dataframe_value)"
                else:
                    print("[warn] Receipt cache dict structure not recognized. Us
                    return pd.DataFrame(columns=["patient_id","code","amount"]),
    else:
        print("[warn] Receipt cache type not recognized. Using empty receipt_line
        return pd.DataFrame(columns=["patient_id","code","amount"]), "unknown/empt
    # Normalize DataFrame case (line_df -> patient_id, code, amount)
    if line_df is None or len(line_df) == 0:
        return pd.DataFrame(columns=["patient_id","code","amount"]), mode
    # Guess columns
    pid_col  = guess_col(line_df, ["patient_id","patient","pat_id","pid"])
    code_col = guess_col(line_df, ["code","cpt_code","cpt","billing_code","proced
    amt_col  = guess_col(line_df, ["amount","line_total","line_total_usd","line_c
    if pid_col is None or code_col is None or amt_col is None:
        # maybe already normalized
        if set(["patient_id","code","amount"]).issubset(set(line_df.columns)):
            pid_col, code_col, amt_col = "patient_id","code","amount"
        else:
            print(f"[warn] Could not find required columns in lineitems df. "
                  f"pid_col={pid_col}, code_col={code_col}, amt_col={amt_col}. Us
            return pd.DataFrame(columns=["patient_id","code","amount"]), mode
    out = line_df[[pid_col, code_col, amt_col]].copy()
    out.columns = ["patient_id","code","amount"]
    # Clean types
    out["patient_id"] = pd.to_numeric(out["patient_id"], errors="coerce")


# --- PAGE 442 ---

    out["code"] = out["code"].astype(str).str.strip()
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
    out = out.dropna(subset=["patient_id","code","amount"])
    out["patient_id"] = out["patient_id"].astype(int)
    out["amount"] = out["amount"].astype(float)
    out = out[np.isfinite(out["amount"])]
    out = out[out["amount"] >= 0.0]
    return out, mode
receipt_lines, receipt_mode = receipt_lines_from_cache(cache_obj)
print(f"[receipt-cache] mode={receipt_mode}")
print(f"[info] receipt_lines shape={receipt_lines.shape} | patients_w_lines={rece
# -----------------------
# Build receipt features from receipt_lines (never empty columns)
# -----------------------
def build_receipt_features_from_lines(lines, patient_ids):
    Returns receipt_feat with guaranteed columns:
    base = pd.DataFrame({"patient_id": sorted(set(int(x) for x in patient_ids))})
    # Minimal columns always exist
    feat = base.copy()
    feat["pdf_total_sum_items"] = 0.0
    feat["pdf_n_line_items"] = 0
    feat["pdf_n_unique_codes"] = 0
    feat["pdf_unit_mean"] = 0.0
    feat["pdf_unit_std"] = 0.0
    feat["pdf_unit_cv"] = 0.0
    # Code-level columns always exist
    for c in ALL_CODES:
        feat[f"pdf_cnt_code_{c}"] = 0
        feat[f"pdf_cost_code_{c}"] = 0.0
    # Category costs


# --- PAGE 443 ---

    cat_cols = ["ed_em","crit","severe_proc","imaging","lab","obs","other"]
    for nm in cat_cols:
        feat[f"pdf_cost_{nm}"] = 0.0
        feat[f"pdf_share_{nm}"] = 0.0
    # Other engineered
    feat["pdf_has_crit"] = 0
    feat["pdf_has_severe_proc"] = 0
    feat["pdf_em_level_mean"] = 0.0
    feat["pdf_em_level_max"] = 0.0
    feat["pdf_em_high_frac"] = 0.0
    feat["pdf_code_cost_entropy"] = 0.0
    feat["pdf_standardized_total"] = 0.0
    feat["pdf_price_index"] = 1.0
    if lines is None or len(lines) == 0:
        return feat
    # Only keep rows for our patient universe
    lines = lines[lines["patient_id"].isin(set(feat["patient_id"]))].copy()
    if len(lines) == 0:
        return feat
    # Totals
    tot = lines.groupby("patient_id")["amount"].sum().rename("pdf_total_sum_items
    n_lines = lines.groupby("patient_id").size().rename("pdf_n_line_items")
    n_unique = lines.groupby("patient_id")["code"].nunique().rename("pdf_n_unique_
    unit_mean = lines.groupby("patient_id")["amount"].mean().rename("pdf_unit_mea
    unit_std = lines.groupby("patient_id")["amount"].std(ddof=0).fillna(0.0).rena
    feat = feat.merge(tot.reset_index(), on="patient_id", how="left", suffixes=("
    feat["pdf_total_sum_items"] = feat["pdf_total_sum_items_new"].fillna(feat["pd
    feat = feat.drop(columns=["pdf_total_sum_items_new"])
    feat = feat.merge(n_lines.reset_index(), on="patient_id", how="left", suffixe
    feat["pdf_n_line_items"] = feat["pdf_n_line_items_new"].fillna(feat["pdf_n_li
    feat = feat.drop(columns=["pdf_n_line_items_new"])
    feat = feat.merge(n_unique.reset_index(), on="patient_id", how="left", suffix
    feat["pdf_n_unique_codes"] = feat["pdf_n_unique_codes_new"].fillna(feat["pdf_


# --- PAGE 444 ---

    feat = feat.drop(columns=["pdf_n_unique_codes_new"])
    feat = feat.merge(unit_mean.reset_index(), on="patient_id", how="left", suffi
    feat["pdf_unit_mean"] = feat["pdf_unit_mean_new"].fillna(feat["pdf_unit_mean"
    feat = feat.drop(columns=["pdf_unit_mean_new"])
    feat = feat.merge(unit_std.reset_index(), on="patient_id", how="left", suffix
    feat["pdf_unit_std"] = feat["pdf_unit_std_new"].fillna(feat["pdf_unit_std"])
    feat = feat.drop(columns=["pdf_unit_std_new"])
    feat["pdf_unit_cv"] = np.where(
        feat["pdf_unit_mean"] > 0,
        feat["pdf_unit_std"] / feat["pdf_unit_mean"],
    ).astype(float)
    # Code counts & costs (FIXED logic: groupby pivot, not "if c in cc")
    lines_u = lines[lines["code"].isin(ALL_CODES)].copy()
    if len(lines_u) > 0:
        cnt = lines_u.groupby(["patient_id","code"]).size().unstack(fill_value=0)
        cnt = cnt.reindex(columns=ALL_CODES, fill_value=0)
        cnt.columns = [f"pdf_cnt_code_{c}" for c in cnt.columns]
        cnt = cnt.reset_index()
        cost = lines_u.groupby(["patient_id","code"])["amount"].sum().unstack(fil
        cost = cost.reindex(columns=ALL_CODES, fill_value=0.0)
        cost.columns = [f"pdf_cost_code_{c}" for c in cost.columns]
        cost = cost.reset_index()
        feat = feat.merge(cnt, on="patient_id", how="left")
        feat = feat.merge(cost, on="patient_id", how="left")
        # fill NaNs from merge
        for c in ALL_CODES:
            feat[f"pdf_cnt_code_{c}"] = feat[f"pdf_cnt_code_{c}"].fillna(0).astyp
            feat[f"pdf_cost_code_{c}"] = feat[f"pdf_cost_code_{c}"].fillna(0.0).a
    # Category mapping
    cat_map = {}
    for c in ED_EM: cat_map[c] = "ed_em"


# --- PAGE 445 ---

    for c in CRIT: cat_map[c] = "crit"
    for c in SEVERE_PROC: cat_map[c] = "severe_proc"
    for c in IMAGING: cat_map[c] = "imaging"
    for c in LAB: cat_map[c] = "lab"
    for c in OBS: cat_map[c] = "obs"
    lines_cat = lines.copy()
    lines_cat["cat"] = lines_cat["code"].map(cat_map).fillna("other")
    cat_sum = lines_cat.groupby(["patient_id","cat"])["amount"].sum().unstack(fil
    for nm in cat_cols:
        if nm not in cat_sum.columns:
            cat_sum[nm] = 0.0
    cat_sum = cat_sum[cat_cols].reset_index()
    cat_sum.columns = ["patient_id"] + [f"pdf_cost_{nm}" for nm in cat_cols]
    feat = feat.merge(cat_sum, on="patient_id", how="left")
    for nm in cat_cols:
        feat[f"pdf_cost_{nm}"] = feat[f"pdf_cost_{nm}"].fillna(0.0).astype(float)
    # Shares
    total = feat["pdf_total_sum_items"].values
    for nm in cat_cols:
        feat[f"pdf_share_{nm}"] = np.where(total > 0, feat[f"pdf_cost_{nm}"].valu
    # Flags
    has_crit = lines["code"].isin(CRIT).groupby(lines["patient_id"]).any().astype
    has_sev  = lines["code"].isin(SEVERE_PROC).groupby(lines["patient_id"]).any()
    feat = feat.merge(has_crit.reset_index(), on="patient_id", how="left")
    feat = feat.merge(has_sev.reset_index(), on="patient_id", how="left")
    feat["pdf_has_crit"] = feat["pdf_has_crit"].fillna(0).astype(int)
    feat["pdf_has_severe_proc"] = feat["pdf_has_severe_proc"].fillna(0).astype(int
    # E/M level stats
    em = lines[lines["code"].isin(ED_EM)].copy()
    if len(em) > 0:
        em["lvl"] = pd.to_numeric(em["code"].str[-1], errors="coerce")
        em = em.dropna(subset=["lvl"])
        em_mean = em.groupby("patient_id")["lvl"].mean().rename("pdf_em_level_mea
        em_max  = em.groupby("patient_id")["lvl"].max().rename("pdf_em_level_max"


# --- PAGE 446 ---

        em_hi   = (em["lvl"] >= 4).groupby(em["patient_id"]).mean().rename("pdf_e
        feat = feat.merge(em_mean.reset_index(), on="patient_id", how="left")
        feat = feat.merge(em_max.reset_index(), on="patient_id", how="left")
        feat = feat.merge(em_hi.reset_index(), on="patient_id", how="left")
        feat["pdf_em_level_mean"] = feat["pdf_em_level_mean"].fillna(0.0)
        feat["pdf_em_level_max"]  = feat["pdf_em_level_max"].fillna(0.0)
        feat["pdf_em_high_frac"]  = feat["pdf_em_high_frac"].fillna(0.0)
    # Entropy of per-code cost distribution
    ccost = lines.groupby(["patient_id","code"])["amount"].sum().reset_index()
    ccost = ccost.merge(tot.reset_index(), on="patient_id", how="left")
    ccost["p"] = ccost["amount"] / ccost["pdf_total_sum_items"].replace(0.0, np.n
    ccost = ccost.replace([np.inf, -np.inf], np.nan).dropna(subset=["p"])
    ccost = ccost[ccost["p"] > 0]
    if len(ccost) > 0:
        ent = (-(ccost["p"] * np.log(ccost["p"]))).groupby(ccost["patient_id"]).s
        feat = feat.merge(ent.reset_index(), on="patient_id", how="left")
        feat["pdf_code_cost_entropy"] = feat["pdf_code_cost_entropy"].fillna(0.0)
    # Standardized total (utilization proxy) using global median price per code
    # If your codes universe is complete (18 codes), this is stable.
    ref_price = lines.groupby("code")["amount"].median()
    lines_std = lines.copy()
    lines_std["ref_price"] = lines_std["code"].map(ref_price)
    # If some code missing median (shouldn't happen), fallback to its own amount
    lines_std["ref_price"] = lines_std["ref_price"].fillna(lines_std["amount"])
    std_total = lines_std.groupby("patient_id")["ref_price"].sum().rename("pdf_st
    feat = feat.merge(std_total.reset_index(), on="patient_id", how="left")
    feat["pdf_standardized_total"] = feat["pdf_standardized_total"].fillna(0.0)
    denom = feat["pdf_standardized_total"].replace(0.0, np.nan)
    feat["pdf_price_index"] = (feat["pdf_total_sum_items"] / denom).replace([np.i
    return feat
patient_ids_all = pd.concat([train_full["patient_id"], test_full["patient_id"]],
receipt_feat = build_receipt_features_from_lines(receipt_lines, patient_ids_all)
print(f"[info] receipt_feat shape={receipt_feat.shape}")
# Receipt integrity check vs prior cost


# --- PAGE 447 ---

tmp_chk = train_full[["patient_id","prior_ed_cost_5y_usd"]].merge(
    receipt_feat[["patient_id","pdf_total_sum_items"]],
    on="patient_id", how="left"
)
absdiff = (tmp_chk["prior_ed_cost_5y_usd"] - tmp_chk["pdf_total_sum_items"]).abs(
missing_rate = float(tmp_chk["pdf_total_sum_items"].isna().mean())
valid_mask = tmp_chk["pdf_total_sum_items"].notna().values
match_rate_valid = float((absdiff[valid_mask] <= TOL).mean()) if valid_mask.any()
match_rate_all = float((absdiff.fillna(np.inf) <= TOL).mean())
print(f"[receipt-check] missing_rate(sum_items_total)={missing_rate:.4f}")
print(f"[receipt-check] match_rate(valid only)={match_rate_valid:.4f} | match_rat
if valid_mask.any():
    print("[receipt-check] absdiff (valid only) summary:")
    print(absdiff[valid_mask].describe())
# Quick sanity: make sure per-code counts are NOT all zeros
nz = {}
for c in ["99285","99291","31500","G0378"]:
    col = f"pdf_cnt_code_{c}"
    if col in receipt_feat.columns:
        nz[c] = float((receipt_feat[col] > 0).mean())
print(f"[receipt-sanity] nonzero rate: {nz}")
# -----------------------
# Admissions patient-level features (hard features only)
# -----------------------
adm_tr = pd.read_csv(ADM_TRAIN)
adm_te = pd.read_csv(ADM_TEST)
adm_all = pd.concat([adm_tr, adm_te], axis=0, ignore_index=True)
# Ensure patient_id int
adm_all["patient_id"] = pd.to_numeric(adm_all["patient_id"], errors="coerce").asty
# Ensure required columns exist (robust)
for col in ["los_days", "acuity_emergent", "charlson_band", "ed_visits_6m", "disc
    if col not in adm_all.columns:
        adm_all[col] = np.nan
    adm_all[col] = pd.to_numeric(adm_all[col], errors="coerce")
g = adm_all.groupby("patient_id", dropna=False)


# --- PAGE 448 ---

adm_feat = pd.DataFrame({
    "patient_id": g.size().index.astype(int),
    "adm_n": g.size().values.astype(int),
    "adm_los_mean": g["los_days"].mean().values,
    "adm_los_max": g["los_days"].max().values,
    "adm_charlson_mean": g["charlson_band"].mean().values,
    "adm_charlson_max": g["charlson_band"].max().values,
    "adm_emergent_rate": g["acuity_emergent"].mean().values,
    "adm_edvis6m_mean": g["ed_visits_6m"].mean().values,
    "adm_edvis6m_max": g["ed_visits_6m"].max().values,
    "adm_edvis6m_sum": g["ed_visits_6m"].sum().values,
    "adm_discharge_wd_mean": g["discharge_weekday"].mean().values,
})
# dx counts (optional)
if "primary_dx" in adm_all.columns:
    dx_ct = pd.crosstab(adm_all["patient_id"], adm_all["primary_dx"])
    dx_ct.columns = [f"adm_dx_cnt_{c}" for c in dx_ct.columns]
    dx_ct = dx_ct.reset_index().rename(columns={"patient_id":"patient_id"})
    dx_ct["patient_id"] = pd.to_numeric(dx_ct["patient_id"], errors="coerce").asty
    adm_feat = adm_feat.merge(dx_ct, on="patient_id", how="left")
print(f"[info] adm_feat shape={adm_feat.shape}")
# -----------------------
# Merge all features
# -----------------------
all_train = train_full.merge(receipt_feat, on="patient_id", how="left").merge(adm_
all_test  = test_full.merge(receipt_feat, on="patient_id", how="left").merge(adm_
# Fill numeric NaNs with train median (robust)
num_cols = all_train.select_dtypes(include=[np.number]).columns.tolist()
for c in num_cols:
    if c == "ed_cost_next3y_usd":
    med = all_train[c].median()
    all_train[c] = all_train[c].fillna(med)
    if c in all_test.columns:
        all_test[c] = all_test[c].fillna(med)


# --- PAGE 449 ---

# Fill categorical NaNs with string token (robust)
for c in ["primary_chronic","insurance","sex","zip3"]:
    if c in all_train.columns:
        all_train[c] = all_train[c].astype(str).fillna("UNK")
    if c in all_test.columns:
        all_test[c] = all_test[c].astype(str).fillna("UNK")
# -----------------------
# CV setup
# -----------------------
def strat_label(df):
    # Keep chronic+insurance+outlier balanced
    pc = df["primary_chronic"].astype(str) if "primary_chronic" in df.columns els
    ins = df["insurance"].astype(str) if "insurance" in df.columns else "NA"
    out = df["is_outlier"].astype(int).astype(str) if "is_outlier" in df.columns
    return (pc + "|" + ins + "|out=" + out)
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
y = all_train["ed_cost_next3y_usd"].values
is_out = all_train["is_outlier"].values.astype(int)
# -----------------------
# Baseline 1: group-median ratio (old)
# -----------------------
def cv_group_ratio(df):
    oof = np.zeros(len(df), dtype=float)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(df, strat_label(df)), start
        tr = df.iloc[tr_idx].copy()
        va = df.iloc[va_idx].copy()
        denom = tr["prior_ed_cost_5y_usd"].values + 1.0
        tr["ratio"] = tr["ed_cost_next3y_usd"].values / denom
        med_ratio = tr.groupby(["primary_chronic","insurance"], dropna=False)["rat
        global_med = float(tr["ratio"].median())
        pred = []
        for _, r in va.iterrows():
            key = (r["primary_chronic"], r["insurance"])


# --- PAGE 450 ---

            rr = float(med_ratio.get(key, global_med))
            pred.append(rr * (float(r["prior_ed_cost_5y_usd"]) + 1.0))
        pred = np.array(pred, dtype=float)
        oof[va_idx] = pred
        mae = mean_absolute_error(va["ed_cost_next3y_usd"].values, pred)
        out_mask = (va["is_outlier"].values.astype(int)==1)
        out_mae = mean_absolute_error(va.loc[out_mask,"ed_cost_next3y_usd"].value
        print(f"[ratio] fold{fold}: MAE={mae:.3f} | out_MAE={out_mae:.3f} | out_n
    return oof
# -----------------------
# Baseline 2: power-law/log baseline per group
# log1p(y)=a+b*log1p(cost)+c*log1p(visits)
# -----------------------
def cv_powerlaw(df):
    oof = np.zeros(len(df), dtype=float)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(df, strat_label(df)), start
        tr = df.iloc[tr_idx].copy()
        va = df.iloc[va_idx].copy()
        # global fallback fit ONCE per fold
        Xg = np.column_stack([np.log1p(tr["prior_ed_cost_5y_usd"].values),
                              np.log1p(tr["prior_ed_visits_5y"].values)])
        yg = np.log1p(tr["ed_cost_next3y_usd"].values)
        global_lr = LinearRegression().fit(Xg, yg)
        models = {}
        for (ch, ins), sub in tr.groupby(["primary_chronic","insurance"], dropna=
            X = np.column_stack([
                np.log1p(sub["prior_ed_cost_5y_usd"].values),
                np.log1p(sub["prior_ed_visits_5y"].values),
            ])
            ylog = np.log1p(sub["ed_cost_next3y_usd"].values)
            lr = LinearRegression()
            lr.fit(X, ylog)
            models[(ch, ins)] = lr
        pred = np.zeros(len(va), dtype=float)
        for i, r in enumerate(va.itertuples(index=False)):


# --- PAGE 451 ---

            key = (getattr(r, "primary_chronic"), getattr(r, "insurance"))
            lr = models.get(key, global_lr)
            Xv = np.array([[np.log1p(getattr(r, "prior_ed_cost_5y_usd")),
                            np.log1p(getattr(r, "prior_ed_visits_5y"))]], dtype=f
            yhat = float(np.expm1(lr.predict(Xv)[0]))
            pred[i] = max(0.0, yhat)
        oof[va_idx] = pred
        mae = mean_absolute_error(va["ed_cost_next3y_usd"].values, pred)
        out_mask = (va["is_outlier"].values.astype(int)==1)
        out_mae = mean_absolute_error(va.loc[out_mask,"ed_cost_next3y_usd"].value
        print(f"[powerlaw] fold{fold}: MAE={mae:.3f} | out_MAE={out_mae:.3f} | ou
    return oof
# -----------------------
# Baseline 3: isotonic per group (monotone, flexible)
# -----------------------
def cv_isotonic(df):
    oof = np.zeros(len(df), dtype=float)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(df, strat_label(df)), start
        tr = df.iloc[tr_idx].copy()
        va = df.iloc[va_idx].copy()
        iso_models = {}
        for (ch, ins), sub in tr.groupby(["primary_chronic","insurance"], dropna=
            x = sub["prior_ed_cost_5y_usd"].values.astype(float)
            yv = sub["ed_cost_next3y_usd"].values.astype(float)
            if len(np.unique(x)) < 5:
            iso = IsotonicRegression(increasing=True, out_of_bounds="clip")  # av
            iso.fit(x, yv)
            iso_models[(ch, ins)] = iso
        global_iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        global_iso.fit(tr["prior_ed_cost_5y_usd"].values.astype(float), tr["ed_co
        pred = np.zeros(len(va), dtype=float)
        for i, r in enumerate(va.itertuples(index=False)):
            key = (getattr(r, "primary_chronic"), getattr(r, "insurance"))
            iso = iso_models.get(key, global_iso)


# --- PAGE 452 ---

            yhat = float(iso.predict([float(getattr(r, "prior_ed_cost_5y_usd"))])
            pred[i] = max(0.0, yhat)
        oof[va_idx] = pred
        mae = mean_absolute_error(va["ed_cost_next3y_usd"].values, pred)
        out_mask = (va["is_outlier"].values.astype(int)==1)
        out_mae = mean_absolute_error(va.loc[out_mask,"ed_cost_next3y_usd"].value
        print(f"[isotonic] fold{fold}: MAE={mae:.3f} | out_MAE={out_mae:.3f} | ou
    return oof
# -----------------------
# Baseline 4: very-light LightGBM (optional)
# -----------------------
def cv_light_lgb(df):
    if not HAS_LGB:
        print("[lgb] LightGBM not available, skipping.")
        return None
    oof = np.zeros(len(df), dtype=float)
    df2 = df.copy()
    df2["log_prior_cost"] = np.log1p(df2["prior_ed_cost_5y_usd"].values)
    df2["sqrt_prior_cost"] = np.sqrt(np.clip(df2["prior_ed_cost_5y_usd"].values,
    df2["log_prior_vis"] = np.log1p(df2["prior_ed_visits_5y"].values)
    df2["prior_cost_per_visit"] = df2["prior_ed_cost_5y_usd"] / (df2["prior_ed_vi
    feat_cols = [
        # receipt decomposition
        # admissions
    ]
    feat_cols = [c for c in feat_cols if c in df2.columns]
    # Categorical -> codes


# --- PAGE 453 ---

    cat_cols = [c for c in ["primary_chronic","insurance","sex","zip3"] if c in d
    for c in cat_cols:
        df2[c] = df2[c].astype("category")
        df2[c] = df2[c].cat.codes.astype(int)
    feat_cols += cat_cols
    for fold, (tr_idx, va_idx) in enumerate(skf.split(df2, strat_label(df2)), sta
        tr = df2.iloc[tr_idx]
        va = df2.iloc[va_idx]
        X_tr = tr[feat_cols]
        y_tr = tr["ed_cost_next3y_usd"].values
        X_va = va[feat_cols]
        y_va = va["ed_cost_next3y_usd"].values
        model = lgb.LGBMRegressor(
            objective="regression_l1",
            n_estimators=5000,
            learning_rate=0.02,
            num_leaves=64,
            min_data_in_leaf=30,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            lambda_l2=2.0,
            random_state=RANDOM_STATE + fold,
            n_jobs=-1,
        )
        model.fit(
            eval_set=[(X_va, y_va)],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(stopping_rounds=300, verbose=False)]
        )
        pred = model.predict(X_va, num_iteration=model.best_iteration_)
        pred = np.clip(pred, 0, None)
        oof[va_idx] = pred
        mae = mean_absolute_error(y_va, pred)


# --- PAGE 454 ---

        out_mask = (va["is_outlier"].values.astype(int)==1)
        out_mae = mean_absolute_error(y_va[out_mask], pred[out_mask]) if out_mask
        print(f"[lgb_light] fold{fold}: MAE={mae:.3f} | out_MAE={out_mae:.3f} | o
    return oof, feat_cols
# -----------------------
# Run baselines
# -----------------------
print("\n=== CV: baseline comparison ===")
oof_ratio = cv_group_ratio(all_train)
oof_pow   = cv_powerlaw(all_train)
oof_iso   = cv_isotonic(all_train)
res = []
def summarize(name, oof):
    mae = mean_absolute_error(y, oof)
    out_mae = mean_absolute_error(y[is_out==1], oof[is_out==1]) if (is_out==1).any
    res.append({"model": name, "oof_mae": mae, "out_oof_mae": out_mae})
summarize("group_ratio", oof_ratio)
summarize("powerlaw_log", oof_pow)
summarize("isotonic_group", oof_iso)
lgb_out = None
if HAS_LGB:
    lgb_out = cv_light_lgb(all_train)
    if lgb_out is not None:
        oof_lgb, feat_cols = lgb_out
        summarize("lgb_light_sublinear", oof_lgb)
summary_df = pd.DataFrame(res).sort_values("oof_mae")
print("\n=== SUMMARY (lower is better) ===")
print(summary_df.to_string(index=False))
summary_df.to_csv(OUT_DIR / "iter25_baseline_summary.csv", index=False)
print(f"[write] {OUT_DIR / 'iter25_baseline_summary.csv'}")
# -----------------------
# Plot: prior cost vs target


# --- PAGE 455 ---

1. "[receipt-cache] mode=..."
应该是 patient_dict(...)  或 dict(lineitems_df)  这类
2. receipt_feat shape=(4000, ...)
# -----------------------
plt.figure(figsize=(8,6))
plt.scatter(all_train["prior_ed_cost_5y_usd"], all_train["ed_cost_next3y_usd"], s
plt.xlabel("prior_ed_cost_5y_usd")
plt.ylabel("ed_cost_next3y_usd")
plt.title("Train: prior cost vs future cost")
plt.tight_layout()
plt.savefig(OUT_DIR / "scatter_prior_vs_future.png", dpi=160)
plt.show()
# Plot predicted vs actual for the best baseline
best_name = summary_df.iloc[0]["model"]
best_oof = {"group_ratio":oof_ratio, "powerlaw_log":oof_pow, "isotonic_group":oof_
if best_name == "lgb_light_sublinear" and HAS_LGB and lgb_out is not None:
    best_oof = oof_lgb
if best_oof is not None:
    plt.figure(figsize=(6,6))
    plt.scatter(y, best_oof, s=10)
    plt.xlabel("actual")
    plt.ylabel("oof_pred")
    plt.title(f"OOF pred vs actual ({best_name})")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"oof_pred_vs_actual_{best_name}.png", dpi=160)
    plt.show()
print("\n[DONE] Iter25 EDA finished.")


# --- PAGE 456 ---

3. [receipt-sanity] nonzero rate: {...}
```

### C21 — Code cell starting at page 457
```python
# -----------------------
# Receipt feature schema guard (drop-in patch)
# -----------------------
def _ensure_cols(df, required, df_name="df"):
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"[{df_name}] expected DataFrame, got {type(df)}")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"[{df_name}] missing columns={missing}. "
            f"Available columns (first 50)={list(df.columns)[:50]}"
        )
# 1) Ensure receipt_feat is a DataFrame and has patient_id
if not isinstance(receipt_feat, pd.DataFrame):
    raise TypeError(f"[receipt_feat] not a DataFrame: {type(receipt_feat)}")
# Sometimes patient_id is an index in older pipelines
if "patient_id" not in receipt_feat.columns:
    if receipt_feat.index.name == "patient_id":
        receipt_feat = receipt_feat.reset_index()
    else:
        raise KeyError(


# --- PAGE 458 ---

            "[receipt_feat] missing 'patient_id'. "
            f"Columns (first 50)={list(receipt_feat.columns)[:50]} | index_name={
        )
# 2) Unify the total column name across versions
if "pdf_total_sum_items" not in receipt_feat.columns:
    # try common historical names
    rename_candidates = [
    ]
    found = next((c for c in rename_candidates if c in receipt_feat.columns), Non
    if found is not None:
        receipt_feat = receipt_feat.rename(columns={found: "pdf_total_sum_items"}
        print(f"[receipt-guard] renamed {found} -> pdf_total_sum_items")
    else:
        # last resort: compute from receipt_lines if present
        if isinstance(receipt_lines, pd.DataFrame) and (not receipt_lines.empty):
            # make sure receipt_lines has patient_id & amount
            if "patient_id" not in receipt_lines.columns:
                raise KeyError("[receipt_lines] missing patient_id; cannot recons
            if "amount" not in receipt_lines.columns:
                raise KeyError("[receipt_lines] missing amount; cannot reconstruc
            tmp_total = (
                receipt_lines.assign(amount=pd.to_numeric(receipt_lines["amount"]
                .groupby("patient_id", as_index=False)["amount"]
                .sum()
                .rename(columns={"amount": "pdf_total_sum_items"})
            )
            receipt_feat = receipt_feat.merge(tmp_total, on="patient_id", how="le
            print("[receipt-guard] reconstructed pdf_total_sum_items from receipt_
        else:
            raise KeyError(
                "[receipt_feat] cannot find or reconstruct pdf_total_sum_items. "
                f"Available columns (first 50)={list(receipt_feat.columns)[:50]}"
            )


# --- PAGE 459 ---

merge(receipt_feat[["patient_id","pdf_total_sum_items"]])  永不炸。
2) 你们之前的分析/建模哪里可能“偏了”，以及怎么更接近 400 MAE
A) 你们之前 “receipt mismatch / diff 很大” 的 EDA 偏差点
B) 为什么 quantile(α=0.5) 和 L1 输出几乎一样？
quantile  目标当 alpha=0.5  时本质上就是在学中位数；regression_l1  也是 MAE/中位
# 3) Final assert (now safe to select)
_ensure_cols(receipt_feat, ["patient_id", "pdf_total_sum_items"], "receipt_feat")


# --- PAGE 460 ---

3) 真正可能把你从 460+ 拉向 400 的改进优先级
log1p(prior_cost)
sqrt(prior_cost)
log1p(prior_visits)
prior_cost_per_visit = prior_cost/(visits+1)
receipt_signature = "31500:1|36556:1|...|99291:1" （按 code 排序 + 计数）


# --- PAGE 461 ---

adm_recent_util = adm_edvis6m_max （最大近期 ED 利用）
adm_burden = adm_charlson_max （最高共病负担）
adm_recent_util / (prior_ed_visits_5y+1) （近期是否突然升高）
4) 你下一轮 Iter26/Iter27 我建议怎么跑（不浪费算力版）
1. CatBoost 主模型（loss=MAE），特征：
log1p(prior_cost) , sqrt(prior_cost) , log1p(prior_visits) ,


# --- PAGE 462 ---

5) 最后：你这份数据集结构确认一下
```

### C22 — Code cell starting at page 502
```python
# ============================
# Iter26 (ROBUST): Sublinear + Receipt(Utilization/Price) + Optional CatBoost/LGB
# Copy-paste into ONE Jupyter cell and run.
# ============================
import os
import re
import math
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load, dump
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression


# --- PAGE 503 ---

import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
# -----------------------
# Optional packages
# -----------------------
try:
    import lightgbm as lgb
    HAS_LGB = True
    try:
        from lightgbm.callback import early_stopping as lgb_early_stopping
    except Exception:
        lgb_early_stopping = None
except Exception:
    HAS_LGB = False
    lgb = None
    lgb_early_stopping = None
try:
    from catboost import CatBoostRegressor, Pool
    HAS_CAT = True
except Exception:
    HAS_CAT = False
    CatBoostRegressor = None
    Pool = None
# -----------------------
# Config
# -----------------------
BASE_DIR = Path(r"D:\AgentDs\agent_ds_healthcare")
TRAIN_CSV    = BASE_DIR / "ed_cost_train.csv"
TEST_CSV     = BASE_DIR / "ed_cost_test.csv"
PATIENTS_CSV = BASE_DIR / "patients.csv"
ADM_TRAIN = BASE_DIR / "admissions_train.csv"
ADM_TEST  = BASE_DIR / "admissions_test.csv"
# receipt cache you trust (iter10)


# --- PAGE 504 ---

RECEIPT_CACHE = BASE_DIR / "cache_iter10" / "receipts_parsed.joblib"
OUT_DIR = BASE_DIR / "cache_iter26_robust"
OUT_DIR.mkdir(parents=True, exist_ok=True)
# Toggle switches
USE_ADMISSIONS   = True         # admissions historically "弱提升/有时变差"，但这里保
USE_CATBOOST     = True         # if installed
USE_LIGHTGBM     = True         # if installed
USE_MONOTONE_LGB = True         # monotone constraints for key sublinear drivers
USE_LOG_TARGET_LGB = True       # second LGB model on log1p(y) (ensemble option)
# Training controls
RANDOM_STATE = 42
N_SPLITS = 5
SEEDS = [42, 2024, 3407]        # small seed ensemble
EARLY_STOP = 300
N_ESTIMATORS = 20000
TOL = 1e-2                      # money comparison tolerance
# group residual correction (post-hoc bias adjust)
USE_GROUP_RESID_CORR = True
GROUP_SHRINK_K = 80.0           # larger => more shrink for small groups
print(f"[paths] BASE_DIR={BASE_DIR}")
print(f"[paths] OUT_DIR={OUT_DIR}")
print(f"[paths] RECEIPT_CACHE={RECEIPT_CACHE}")
print(f"[flags] HAS_LGB={HAS_LGB} | HAS_CAT={HAS_CAT}")
# -----------------------
# Helpers
# -----------------------
def safe_read_csv(path: Path):
    if path is None or (not path.exists()):
        print(f"[warn] missing file: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[warn] failed reading {path}: {e}")


# --- PAGE 505 ---

        return None
def can_int(x) -> bool:
    try:
        int(str(x).strip())
        return True
    except Exception:
        return False
def normalize_code(x):
    if x is None:
        return None
    try:
        if isinstance(x, (int, np.integer)):
            return str(int(x))
        if isinstance(x, float) and np.isfinite(x) and float(x).is_integer():
            return str(int(x))
    except Exception:
    s = str(x).strip()
    if re.fullmatch(r"\d+\.0", s):
        s = s.split(".")[0]
    return s if s else None
def parse_amount(x):
    if x is None:
        return np.nan
    try:
        if isinstance(x, str):
            xs = x.strip().replace(",", "").replace("$", "")
            return float(xs)
        return float(x)
    except Exception:
        return np.nan
def infer_col(df: pd.DataFrame, candidates):
    if df is None or df.empty:
        return None
    cols = list(df.columns)
    low = {c.lower(): c for c in cols}


# --- PAGE 506 ---

    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in low:
            return low[cand.lower()]
    for cand in candidates:
        for c in cols:
            if cand.lower() in str(c).lower():
                return c
    return None
def safe_median(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    if x.notna().any():
        return float(x.median())
    return 0.0
# -----------------------
# Receipt code universe
# -----------------------
ED_EM = {"99281","99282","99283","99284","99285"}
CRIT  = {"99291","99292"}
SEVERE_PROC = {"31500","92950","36556","36620"}  # intub, CPR, CVC, art line
IMAGING = {"70450","74177","71045"}              # CT head, CT A/P, CXR
LAB = {"85025","87070","84484"}                  # CBC, culture, troponin
OBS = {"G0378"}
ALL_CODES = sorted(list(ED_EM | CRIT | SEVERE_PROC | IMAGING | LAB | OBS))
print(f"[info] expected receipt code universe={len(ALL_CODES)} -> {ALL_CODES}")
# -----------------------
# Load core tables
# -----------------------
train = safe_read_csv(TRAIN_CSV)
test  = safe_read_csv(TEST_CSV)
patients = safe_read_csv(PATIENTS_CSV)
if train is None or test is None or patients is None:
    raise RuntimeError("Missing required CSVs (train/test/patients). Fix paths fi


# --- PAGE 507 ---

for df in (train, test, patients):
    if "patient_id" not in df.columns:
        raise RuntimeError("Missing patient_id column.")
    df["patient_id"] = pd.to_numeric(df["patient_id"], errors="coerce")
    df.dropna(subset=["patient_id"], inplace=True)
    df["patient_id"] = df["patient_id"].astype(int)
# zip3 keep leading zeros
if "zip3" in patients.columns:
    patients["zip3"] = patients["zip3"].astype(str).str.replace(r"\.0$", "", rege
    patients.loc[patients["zip3"].str.lower().isin(["nan","none"]), "zip3"] = "UN
    patients["zip3"] = patients["zip3"].apply(lambda s: s.zfill(3) if s.isdigit()
train_full = train.merge(patients, on="patient_id", how="left")
test_full  = test.merge(patients, on="patient_id", how="left")
print(f"[info] train_full={train_full.shape}  test_full={test_full.shape}")
# outlier
train_full["is_outlier"] = ((train_full["prior_ed_cost_5y_usd"] <= 50.0) & (train_
outlier_n = int(train_full["is_outlier"].sum())
print(f"[info] outlier_n={outlier_n} / {len(train_full)}")
patient_ids_all = np.sort(pd.concat([train_full["patient_id"], test_full["patient_
# -----------------------
# Receipt cache -> robust features
# -----------------------
def build_receipt_features_from_patient_dict(patient_dict: dict, patient_ids_all:
    rows = []
    line_rows = []
    CODE_KEYS = ["code", "cpt", "hcpcs", "billing_code", "service_code"]
    AMT_KEYS  = ["amount", "line_total", "line_total_usd", "total", "charge", "li
    def safe_items(obj):
        if not isinstance(obj, dict):
            return []
        items = obj.get("items", [])
        return items if isinstance(items, list) else []


# --- PAGE 508 ---

    for pid, obj in patient_dict.items():
        if not can_int(pid):
        pid_int = int(str(pid).strip())
        items = safe_items(obj)
        codes, amounts = [], []
        header_total = np.nan
        if isinstance(obj, dict):
            header_total = parse_amount(obj.get("pdf_total", np.nan))
        for it in items:
            if not isinstance(it, dict):
            code = None
            for k in CODE_KEYS:
                if k in it:
                    code = normalize_code(it.get(k))
            amt = np.nan
            for k in AMT_KEYS:
                if k in it:
                    amt = parse_amount(it.get(k))
            if code is None:
            codes.append(code)
            amounts.append(amt)
            line_rows.append({"patient_id": pid_int, "code": code, "amount": amt}
        total_sum = float(np.nansum(amounts)) if len(amounts) else 0.0
        n_lines = int(len(codes))
        n_unique = int(len(set(codes)))
        # per-code counts/costs (known codes only)
        cnt = {f"pdf_cnt_code_{c}": 0 for c in ALL_CODES}
        cost = {f"pdf_cost_code_{c}": 0.0 for c in ALL_CODES}
        for c, a in zip(codes, amounts):


# --- PAGE 509 ---

            if c in ALL_CODES:
                cnt[f"pdf_cnt_code_{c}"] += 1
                if np.isfinite(a):
                    cost[f"pdf_cost_code_{c}"] += float(a)
        def sum_cost(code_set):
            s = 0.0
            for c, a in zip(codes, amounts):
                if c in code_set and np.isfinite(a):
                    s += float(a)
            return s
        cost_em = sum_cost(ED_EM)
        cost_crit = sum_cost(CRIT)
        cost_sev = sum_cost(SEVERE_PROC)
        cost_img = sum_cost(IMAGING)
        cost_lab = sum_cost(LAB)
        cost_obs = sum_cost(OBS)
        cost_other = max(0.0, total_sum - (cost_em + cost_crit + cost_sev + cost_
        def share(x):
            return float(x / total_sum) if total_sum > 0 else 0.0
        em_levels = []
        for c in codes:
            if c in ED_EM and len(c) == 5 and c[-1].isdigit():
                em_levels.append(int(c[-1]))
        em_levels = np.array(em_levels, dtype=float)
        ent = 0.0
        if total_sum > 0:
            tmp = {}
            for c, a in zip(codes, amounts):
                if not np.isfinite(a):
                tmp[c] = tmp.get(c, 0.0) + float(a)
            probs = np.array([v / total_sum for v in tmp.values() if v > 0], dtyp
            probs = probs[probs > 0]
            if len(probs):
                ent = float(-(probs * np.log(probs)).sum())


# --- PAGE 510 ---

        amt_arr = np.array([a for a in amounts if np.isfinite(a)], dtype=float)
        unit_mean = float(amt_arr.mean()) if len(amt_arr) else 0.0
        unit_std  = float(amt_arr.std()) if len(amt_arr) else 0.0
        unit_min  = float(amt_arr.min()) if len(amt_arr) else 0.0
        unit_max  = float(amt_arr.max()) if len(amt_arr) else 0.0
        # simple severity counts (non-leaky)
        sev_cnt = sum(1 for c in codes if c in SEVERE_PROC)
        crit_cnt = sum(1 for c in codes if c in CRIT)
        em_cnt = sum(1 for c in codes if c in ED_EM)
        high_acuity_score = 2.0 * crit_cnt + 1.5 * sev_cnt + 0.2 * em_cnt
        row = {
            "patient_id": pid_int,
            "pdf_total_sum_items": total_sum,
            "pdf_total_header": header_total,
            "pdf_n_line_items": n_lines,
            "pdf_n_unique_codes": n_unique,
            "pdf_unit_mean": unit_mean,
            "pdf_unit_std": unit_std,
            "pdf_unit_min": unit_min,
            "pdf_unit_max": unit_max,
            "pdf_cost_ed_em": cost_em,
            "pdf_cost_crit": cost_crit,
            "pdf_cost_severe_proc": cost_sev,
            "pdf_cost_imaging": cost_img,
            "pdf_cost_lab": cost_lab,
            "pdf_cost_obs": cost_obs,
            "pdf_cost_other": cost_other,
            "pdf_share_ed_em": share(cost_em),
            "pdf_share_crit": share(cost_crit),
            "pdf_share_severe_proc": share(cost_sev),
            "pdf_share_imaging": share(cost_img),
            "pdf_share_lab": share(cost_lab),
            "pdf_share_obs": share(cost_obs),
            "pdf_share_other": share(cost_other),
            "pdf_has_crit": int(any(c in CRIT for c in codes)),
            "pdf_has_severe_proc": int(any(c in SEVERE_PROC for c in codes)),
            "pdf_em_level_mean": float(np.mean(em_levels)) if len(em_levels) else


# --- PAGE 511 ---

            "pdf_em_level_max": float(np.max(em_levels)) if len(em_levels) else 0
            "pdf_em_high_frac": float(np.mean(em_levels >= 4)) if len(em_levels)
            "pdf_code_cost_entropy": ent,
            "pdf_sev_cnt": float(sev_cnt),
            "pdf_crit_cnt": float(crit_cnt),
            "pdf_em_cnt": float(em_cnt),
            "pdf_high_acuity_score": float(high_acuity_score),
        }
        row.update(cnt)
        row.update(cost)
        rows.append(row)
    feat = pd.DataFrame(rows)
    lines = pd.DataFrame(line_rows)
    # guarantee key columns
    base = pd.DataFrame({"patient_id": patient_ids_all.astype(int)})
    if feat.empty:
        feat = base.copy()
        feat["pdf_total_sum_items"] = np.nan
    else:
        feat = base.merge(feat, on="patient_id", how="left")
    return feat, lines
def build_receipt_features_from_lineitems_df(lineitems: pd.DataFrame, patient_ids_
    df = lineitems.copy()
    pid_col = infer_col(df, ["patient_id", "patient"])
    code_col = infer_col(df, ["code", "cpt", "hcpcs"])
    amt_col  = infer_col(df, ["amount", "line_total", "line total", "line_total_u
    if pid_col is None or code_col is None or amt_col is None:
        base = pd.DataFrame({"patient_id": patient_ids_all.astype(int)})
        base["pdf_total_sum_items"] = np.nan
        return base, pd.DataFrame(columns=["patient_id","code","amount"])
    df = df[[pid_col, code_col, amt_col]].rename(columns={pid_col:"patient_id", c
    df["patient_id"] = pd.to_numeric(df["patient_id"], errors="coerce")
    df.dropna(subset=["patient_id"], inplace=True)
    df["patient_id"] = df["patient_id"].astype(int)
    df["code"] = df["code"].apply(normalize_code)
    df["amount"] = df["amount"].apply(parse_amount)


# --- PAGE 512 ---

    df.dropna(subset=["code"], inplace=True)
    lines = df[["patient_id","code","amount"]].copy()
    g = df.groupby("patient_id", dropna=False)
    feat = pd.DataFrame({"patient_id": g.size().index.astype(int)})
    feat["pdf_total_sum_items"] = g["amount"].sum(min_count=1).fillna(0.0).values
    feat["pdf_n_line_items"] = g.size().values.astype(int)
    feat["pdf_n_unique_codes"] = g["code"].nunique().values.astype(int)
    # create empty known-code columns
    for c in ALL_CODES:
        feat[f"pdf_cnt_code_{c}"] = 0
        feat[f"pdf_cost_code_{c}"] = 0.0
    # fill via crosstab/pivot
    ct = pd.crosstab(df["patient_id"], df["code"])
    for c in ALL_CODES:
        if c in ct.columns:
            feat.loc[feat["patient_id"].isin(ct.index), f"pdf_cnt_code_{c}"] = ct
    cost_pt = df.pivot_table(index="patient_id", columns="code", values="amount",
    for c in ALL_CODES:
        if c in cost_pt.columns:
            feat.loc[feat["patient_id"].isin(cost_pt.index), f"pdf_cost_code_{c}"
    base = pd.DataFrame({"patient_id": patient_ids_all.astype(int)})
    feat = base.merge(feat, on="patient_id", how="left")
    return feat, lines
def load_receipts_flexible(cache_obj, patient_ids_all: np.ndarray):
    if isinstance(cache_obj, dict):
        # unwrap common wrappers
        for k in ["by_pid", "by_patient", "parsed_by_patient", "receipts_by_patie
            if k in cache_obj and isinstance(cache_obj[k], dict):
                cache_obj = cache_obj[k]
        keys = list(cache_obj.keys())
        sample = keys[: min(50, len(keys))]


# --- PAGE 513 ---

        int_like = sum(can_int(k) for k in sample) / max(1, len(sample))
        if int_like >= 0.6:
            print(f"[receipt-cache] treating as patient_dict (int_like={int_like:
            return build_receipt_features_from_patient_dict(cache_obj, patient_id
        # dict container with a df
        dfs = [(k,v) for k,v in cache_obj.items() if isinstance(v, pd.DataFrame)]
        if dfs:
            best_k, best_df = dfs[0]
            print(f"[receipt-cache] treating as df_container; using df key={best_
            return build_receipt_features_from_lineitems_df(best_df, patient_ids_
    if isinstance(cache_obj, pd.DataFrame):
        print("[receipt-cache] treating as lineitems_df")
        return build_receipt_features_from_lineitems_df(cache_obj, patient_ids_al
    # fallback empty
    print("[warn] Unknown receipt cache format; using empty receipts.")
    base = pd.DataFrame({"patient_id": patient_ids_all.astype(int)})
    base["pdf_total_sum_items"] = np.nan
    return base, pd.DataFrame(columns=["patient_id","code","amount"])
# cache processed receipt features for speed
RECEIPT_FEAT_CACHE = OUT_DIR / "receipt_feat_iter26.joblib"
RECEIPT_LINES_CACHE = OUT_DIR / "receipt_lines_iter26.joblib"
if RECEIPT_FEAT_CACHE.exists() and RECEIPT_LINES_CACHE.exists():
    receipt_feat = load(RECEIPT_FEAT_CACHE)
    receipt_lines = load(RECEIPT_LINES_CACHE)
    print(f"[cache] loaded receipt_feat={receipt_feat.shape} receipt_lines={recei
else:
    if RECEIPT_CACHE.exists():
        cache_obj = load(RECEIPT_CACHE)
        print(f"[debug] receipt cache type={type(cache_obj)}")
        receipt_feat, receipt_lines = load_receipts_flexible(cache_obj, patient_i
    else:
        print(f"[warn] missing receipt cache: {RECEIPT_CACHE}")
        receipt_feat = pd.DataFrame({"patient_id": patient_ids_all.astype(int), "
        receipt_lines = pd.DataFrame(columns=["patient_id","code","amount"])


# --- PAGE 514 ---

    receipt_feat = receipt_feat.drop_duplicates("patient_id", keep="first")
    for c in ["patient_id","pdf_total_sum_items"]:
        if c not in receipt_feat.columns:
            receipt_feat[c] = np.nan
    dump(receipt_feat, RECEIPT_FEAT_CACHE)
    dump(receipt_lines, RECEIPT_LINES_CACHE)
    print(f"[cache] wrote receipt_feat/lines to {OUT_DIR}")
print(f"[info] receipt_feat={receipt_feat.shape} receipt_lines={receipt_lines.sha
# receipt integrity check (tolerant)
tmp_chk = train_full[["patient_id","prior_ed_cost_5y_usd"]].merge(
    receipt_feat[["patient_id","pdf_total_sum_items"]],
    on="patient_id", how="left"
)
absdiff = (pd.to_numeric(tmp_chk["prior_ed_cost_5y_usd"], errors="coerce") -
           pd.to_numeric(tmp_chk["pdf_total_sum_items"], errors="coerce")).abs()
mask_valid = tmp_chk["pdf_total_sum_items"].notna()
miss_rate = float((~mask_valid).mean())
match_rate = float((absdiff[mask_valid] <= TOL).mean()) if mask_valid.any() else
print(f"[receipt-check] missing_rate={miss_rate:.4f} | match_rate(valid)={match_r
print(absdiff[mask_valid].describe())
# ---- price decomposition from receipt_lines
# global medians by code
if receipt_lines is not None and (not receipt_lines.empty):
    rl = receipt_lines.copy()
    rl["code"] = rl["code"].apply(normalize_code)
    rl["amount"] = rl["amount"].apply(parse_amount)
    rl = rl.dropna(subset=["code"])
    med_price = rl.groupby("code")["amount"].median().to_dict()
    # per-line log price ratio vs global median(code)
    def line_log_ratio(r):
        c = r["code"]
        a = r["amount"]
        m = med_price.get(c, np.nan)
        if (not np.isfinite(a)) or (not np.isfinite(m)) or m <= 0:


# --- PAGE 515 ---

            return np.nan
        return float(np.log(a / m))
    rl["log_price_ratio"] = rl.apply(line_log_ratio, axis=1)
    pr_g = rl.groupby("patient_id", dropna=False)["log_price_ratio"]
    price_feat = pr_g.agg(["mean","std","count"]).reset_index().rename(
        columns={"mean":"pdf_log_price_ratio_mean", "std":"pdf_log_price_ratio_st
    )
    price_feat["pdf_log_price_ratio_mean"] = price_feat["pdf_log_price_ratio_mean
    price_feat["pdf_log_price_ratio_std"] = price_feat["pdf_log_price_ratio_std"]
    # fraction above/below thresholds
    rl["is_hi_price"] = (rl["log_price_ratio"] > np.log(1.2)).astype(float)
    rl["is_lo_price"] = (rl["log_price_ratio"] < np.log(0.8)).astype(float)
    frac_hi = rl.groupby("patient_id")["is_hi_price"].mean().reset_index().rename
    frac_lo = rl.groupby("patient_id")["is_lo_price"].mean().reset_index().rename
    price_feat = price_feat.merge(frac_hi, on="patient_id", how="left").merge(fra
    price_feat["pdf_hi_price_frac"] = price_feat["pdf_hi_price_frac"].fillna(0.0)
    price_feat["pdf_lo_price_frac"] = price_feat["pdf_lo_price_frac"].fillna(0.0)
else:
    med_price = {}
    price_feat = pd.DataFrame({"patient_id": patient_ids_all.astype(int),
                               "pdf_log_price_ratio_mean": 0.0,
                               "pdf_log_price_ratio_std": 0.0,
                               "pdf_price_ratio_n": 0.0,
                               "pdf_hi_price_frac": 0.0,
                               "pdf_lo_price_frac": 0.0})
# merge price_feat into receipt_feat
receipt_feat = receipt_feat.merge(price_feat, on="patient_id", how="left")
# standardized utilization total = sum(count(code) * global_median_price(code))
for c in ALL_CODES:
    cc = f"pdf_cnt_code_{c}"
    if cc not in receipt_feat.columns:
        receipt_feat[cc] = 0
price_vec = np.array([med_price.get(c, np.nan) for c in ALL_CODES], dtype=float)


# --- PAGE 516 ---

price_vec = np.nan_to_num(price_vec, nan=0.0)
count_cols = [f"pdf_cnt_code_{c}" for c in ALL_CODES]
receipt_feat["pdf_standardized_total"] = receipt_feat[count_cols].values.astype(f
num = pd.to_numeric(receipt_feat["pdf_total_sum_items"], errors="coerce").astype(
den = pd.to_numeric(receipt_feat["pdf_standardized_total"], errors="coerce").asty
pi = np.where(den > 0, num / den, 1.0)
pi = np.where(np.isfinite(pi), pi, 1.0)
receipt_feat["pdf_price_index"] = pi.astype(np.float32)
# -----------------------
# Admissions features (robust)
# -----------------------
def build_admissions_features(adm_tr, adm_te, patient_ids_all):
    base = pd.DataFrame({"patient_id": patient_ids_all.astype(int)})
    if adm_tr is None or adm_te is None:
        return base.copy()
    adm_all = pd.concat([adm_tr, adm_te], axis=0, ignore_index=True)
    if "patient_id" not in adm_all.columns:
        return base.copy()
    adm_all["patient_id"] = pd.to_numeric(adm_all["patient_id"], errors="coerce")
    adm_all = adm_all.dropna(subset=["patient_id"])
    adm_all["patient_id"] = adm_all["patient_id"].astype(int)
    # numeric cast candidates
    for col in ["los_days", "acuity_emergent", "charlson_band", "ed_visits_6m", "
        if col in adm_all.columns:
            adm_all[col] = pd.to_numeric(adm_all[col], errors="coerce")
    # try find a datetime column for recency
    dt_col = infer_col(adm_all, ["admit_date", "admission_date", "admit_time", "d
    if dt_col is not None:
        tmp_dt = pd.to_datetime(adm_all[dt_col], errors="coerce")
        ok_rate = float(tmp_dt.notna().mean())
        if ok_rate >= 0.3:
            adm_all["_dt"] = tmp_dt
        else:


# --- PAGE 517 ---

            dt_col = None
    g = adm_all.groupby("patient_id", dropna=False)
    feat = pd.DataFrame({"patient_id": g.size().index.astype(int)})
    feat["adm_n"] = g.size().values.astype(int)
    def agg_if(col, fn, default=np.nan):
        if col not in adm_all.columns:
            return np.full(len(feat), default, dtype=float)
        return getattr(g[col], fn)().values
    feat["adm_los_mean"] = agg_if("los_days", "mean")
    feat["adm_los_max"]  = agg_if("los_days", "max")
    feat["adm_los_sum"]  = g["los_days"].sum(min_count=1).reindex(feat["patient_i
    feat["adm_charlson_mean"] = agg_if("charlson_band", "mean")
    feat["adm_charlson_max"]  = agg_if("charlson_band", "max")
    feat["adm_emergent_rate"] = agg_if("acuity_emergent", "mean")
    feat["adm_emergent_n"]    = g["acuity_emergent"].sum(min_count=1).reindex(feat
    feat["adm_edvis6m_mean"] = agg_if("ed_visits_6m", "mean")
    feat["adm_edvis6m_max"]  = agg_if("ed_visits_6m", "max")
    feat["adm_edvis6m_sum"]  = g["ed_visits_6m"].sum(min_count=1).reindex(feat["p
    # last-row features (recency-based if datetime exists)
    if dt_col is not None and "_dt" in adm_all.columns:
        adm_all = adm_all.sort_values(["patient_id","_dt"])
    else:
        adm_all = adm_all.sort_values(["patient_id"])
    last = adm_all.groupby("patient_id").tail(1).copy()
    last_cols = []
    for col in ["los_days", "charlson_band", "acuity_emergent", "ed_visits_6m", "
        if col in last.columns:
            last_cols.append(col)
    last = last[["patient_id"] + last_cols]
    rename = {}
    for col in last_cols:


# --- PAGE 518 ---

        rename[col] = f"adm_last_{col}"
    last = last.rename(columns=rename)
    feat = feat.merge(last, on="patient_id", how="left")
    # dx crosstab counts
    if "primary_dx" in adm_all.columns:
        dx_ct = pd.crosstab(adm_all["patient_id"], adm_all["primary_dx"])
        dx_ct.columns = [f"adm_dx_cnt_{c}" for c in dx_ct.columns]
        dx_ct = dx_ct.reset_index().rename(columns={"patient_id":"patient_id"})
        dx_ct["patient_id"] = dx_ct["patient_id"].astype(int)
        feat = feat.merge(dx_ct, on="patient_id", how="left")
    # derived trend
    if "adm_last_ed_visits_6m" in feat.columns and "adm_edvis6m_mean" in feat.col
        feat["adm_edvis6m_rise"] = pd.to_numeric(feat["adm_last_ed_visits_6m"], e
    else:
        feat["adm_edvis6m_rise"] = np.nan
    # ensure all patients
    feat = base.merge(feat, on="patient_id", how="left")
    return feat
adm_feat = pd.DataFrame({"patient_id": patient_ids_all.astype(int)})
if USE_ADMISSIONS:
    adm_tr = safe_read_csv(ADM_TRAIN)
    adm_te = safe_read_csv(ADM_TEST)
    adm_feat = build_admissions_features(adm_tr, adm_te, patient_ids_all)
print(f"[info] adm_feat={adm_feat.shape}")
# -----------------------
# Merge all
# -----------------------
all_train = train_full.merge(receipt_feat, on="patient_id", how="left").merge(adm_
all_test  = test_full.merge(receipt_feat, on="patient_id", how="left").merge(adm_
# If receipt totals missing, impute from prior cost (dataset design: sum_items≈pr
all_train["pdf_total_sum_items"] = all_train["pdf_total_sum_items"].fillna(all_tr
all_test["pdf_total_sum_items"]  = all_test["pdf_total_sum_items"].fillna(all_test


# --- PAGE 519 ---

# -----------------------
# Sublinear transforms
# -----------------------
def add_sublinear(df):
    pc = pd.to_numeric(df["prior_ed_cost_5y_usd"], errors="coerce").fillna(0.0).a
    pv = pd.to_numeric(df["prior_ed_visits_5y"], errors="coerce").fillna(0.0).asty
    df["prior_cost_deflated"] = np.maximum(pc, 50.0)      # key fix: avoid 0-cost
    df["log_prior_cost"] = np.log1p(pc)
    df["sqrt_prior_cost"] = np.sqrt(np.clip(pc, 0, None))
    df["log_prior_visits"] = np.log1p(pv)
    df["prior_cost_per_visit"] = pc / (pv + 1.0)
    df["log_prior_cost_per_visit"] = np.log1p(df["prior_cost_per_visit"].values.a
    # optional cap features (robust)
    df["prior_cost_cap20k"] = np.minimum(pc, 20000.0)
    df["prior_cost_cap10k"] = np.minimum(pc, 10000.0)
    return df
all_train = add_sublinear(all_train)
all_test  = add_sublinear(all_test)
# -----------------------
# Baseline stacking features (OOF) – powerlaw + isotonic on deflated prior cost
# -----------------------
def strat_label(df):
    # keep chronic|insurance|outlier balanced
    return (df["primary_chronic"].astype(str) + "|" +
            df["insurance"].astype(str) + "|out=" +
            df["is_outlier"].astype(int).astype(str))
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
def oof_powerlaw_baseline(train_df, test_df, min_group_n=80):
    oof = np.zeros(len(train_df), dtype=float)
    test_pred = np.zeros(len(test_df), dtype=float)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, strat_label(train_
        tr = train_df.iloc[tr_idx].copy()


# --- PAGE 520 ---

        va = train_df.iloc[va_idx].copy()
        # global model
        Xg = np.column_stack([
            np.log1p(tr["prior_cost_deflated"].values.astype(float)),
            np.log1p(tr["prior_ed_visits_5y"].values.astype(float)),
        ])
        yg = np.log1p(tr["ed_cost_next3y_usd"].values.astype(float))
        global_lr = LinearRegression().fit(Xg, yg)
        # per group models (only if enough samples)
        models = {}
        for (ch, ins), sub in tr.groupby(["primary_chronic","insurance"], dropna=
            if len(sub) < min_group_n:
            X = np.column_stack([
                np.log1p(sub["prior_cost_deflated"].values.astype(float)),
                np.log1p(sub["prior_ed_visits_5y"].values.astype(float)),
            ])
            ylog = np.log1p(sub["ed_cost_next3y_usd"].values.astype(float))
            lr = LinearRegression().fit(X, ylog)
            models[(ch, ins)] = lr
        # predict val
        pv = np.zeros(len(va), dtype=float)
        for i, r in enumerate(va.itertuples(index=False)):
            key = (getattr(r,"primary_chronic"), getattr(r,"insurance"))
            lr = models.get(key, global_lr)
            Xv = np.array([[np.log1p(float(getattr(r,"prior_cost_deflated"))),
                            np.log1p(float(getattr(r,"prior_ed_visits_5y")))]], dt
            pv[i] = max(0.0, float(np.expm1(lr.predict(Xv)[0])))
        oof[va_idx] = pv
        mae = mean_absolute_error(va["ed_cost_next3y_usd"].values, pv)
        print(f"[baseline_powerlaw] fold{fold}: MAE={mae:.3f}")
    # fit full for test
    tr = train_df.copy()
    Xg = np.column_stack([
        np.log1p(tr["prior_cost_deflated"].values.astype(float)),


# --- PAGE 521 ---

        np.log1p(tr["prior_ed_visits_5y"].values.astype(float)),
    ])
    yg = np.log1p(tr["ed_cost_next3y_usd"].values.astype(float))
    global_lr = LinearRegression().fit(Xg, yg)
    models = {}
    for (ch, ins), sub in tr.groupby(["primary_chronic","insurance"], dropna=Fals
        if len(sub) < min_group_n:
        X = np.column_stack([
            np.log1p(sub["prior_cost_deflated"].values.astype(float)),
            np.log1p(sub["prior_ed_visits_5y"].values.astype(float)),
        ])
        ylog = np.log1p(sub["ed_cost_next3y_usd"].values.astype(float))
        lr = LinearRegression().fit(X, ylog)
        models[(ch, ins)] = lr
    for i, r in enumerate(test_df.itertuples(index=False)):
        key = (getattr(r,"primary_chronic"), getattr(r,"insurance"))
        lr = models.get(key, global_lr)
        Xv = np.array([[np.log1p(float(getattr(r,"prior_cost_deflated"))),
                        np.log1p(float(getattr(r,"prior_ed_visits_5y")))]], dtype
        test_pred[i] = max(0.0, float(np.expm1(lr.predict(Xv)[0])))
    return oof, test_pred
def oof_isotonic_baseline(train_df, test_df):
    oof = np.zeros(len(train_df), dtype=float)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, strat_label(train_
        tr = train_df.iloc[tr_idx]
        va = train_df.iloc[va_idx]
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(tr["prior_cost_deflated"].values.astype(float), tr["ed_cost_next3y
        pv = iso.predict(va["prior_cost_deflated"].values.astype(float))
        oof[va_idx] = np.clip(pv, 0, None)
        mae = mean_absolute_error(va["ed_cost_next3y_usd"].values, pv)
        print(f"[baseline_isotonic] fold{fold}: MAE={mae:.3f}")


# --- PAGE 522 ---

    iso_full = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso_full.fit(train_df["prior_cost_deflated"].values.astype(float), train_df["
    test_pred = np.clip(iso_full.predict(test_df["prior_cost_deflated"].values.ast
    return oof, test_pred
# add baseline feats (these are not the final pred; they are extra signals)
y = all_train["ed_cost_next3y_usd"].values.astype(float)
is_out = all_train["is_outlier"].values.astype(int)
print("\n[iter26] building baseline stacking features...")
oof_bl_pow, te_bl_pow = oof_powerlaw_baseline(all_train, all_test, min_group_n=80
oof_bl_iso, te_bl_iso = oof_isotonic_baseline(all_train, all_test)
all_train["bl_powerlaw"] = oof_bl_pow
all_test["bl_powerlaw"]  = te_bl_pow
all_train["bl_isotonic"] = oof_bl_iso
all_test["bl_isotonic"]  = te_bl_iso
# -----------------------
# Fill missing values (robust)
# -----------------------
# categoricals
CAT_COLS = [c for c in ["primary_chronic","insurance","sex","zip3"] if c in all_t
for c in CAT_COLS:
    all_train[c] = all_train[c].astype(str).fillna("UNK")
    all_test[c]  = all_test[c].astype(str).fillna("UNK")
# if admissions last dx exists, keep as cat
if "adm_last_primary_dx" in all_train.columns:
    all_train["adm_last_primary_dx"] = all_train["adm_last_primary_dx"].astype(st
    all_test["adm_last_primary_dx"]  = all_test["adm_last_primary_dx"].astype(str
    CAT_COLS.append("adm_last_primary_dx")
# numeric
num_cols = all_train.select_dtypes(include=[np.number]).columns.tolist()
for c in num_cols:
    if c == "ed_cost_next3y_usd":
    med = safe_median(all_train[c])
    all_train[c] = pd.to_numeric(all_train[c], errors="coerce").fillna(med)


# --- PAGE 523 ---

    if c in all_test.columns:
        all_test[c] = pd.to_numeric(all_test[c], errors="coerce").fillna(med)
# -----------------------
# Feature list (drop obvious leakage / ids)
# -----------------------
DROP_COLS = {"ed_cost_next3y_usd","patient_id"}
feat_cols = [c for c in all_train.columns if c not in DROP_COLS]
# Drop high-cardinality raw strings except chosen cats
for c in feat_cols:
    if all_train[c].dtype == "object" and c not in CAT_COLS:
        # probably some leftover text columns -> drop
feat_cols = [c for c in feat_cols if not (all_train[c].dtype == "object" and c not
# drop constant columns
const_cols = [c for c in feat_cols if all_train[c].nunique(dropna=False) <= 1]
if const_cols:
    print(f"[feature-audit] dropping const cols: {const_cols[:20]}{'...' if len(c
feat_cols = [c for c in feat_cols if c not in const_cols]
print(f"[info] n_features_total={len(feat_cols)} | n_cat={len(CAT_COLS)}")
# -----------------------
# LightGBM CV
# -----------------------
def lgb_make_monotone_constraints(feature_names):
    # 1 = increasing; 0 = no constraint
    inc = {


# --- PAGE 524 ---

    }
    cons = []
    for f in feature_names:
        if f in inc:
            cons.append(1)
        else:
            cons.append(0)
    return cons
def train_lgb_cv(train_df, test_df, feat_cols, cat_cols, seeds, log_target=False)
    if not HAS_LGB:
        print("[lgb] not installed; skip")
        return None
    oof = np.zeros(len(train_df), dtype=float)
    test_pred = np.zeros(len(test_df), dtype=float)
    # prepare copies: keep cat as category for LGB
    trn = train_df.copy()
    tes = test_df.copy()
    for c in cat_cols:
        trn[c] = trn[c].astype("category")
        tes[c] = tes[c].astype("category")
    # monotone constraints
    mono = None
    if USE_MONOTONE_LGB and (not log_target):
        mono = lgb_make_monotone_constraints(feat_cols)
    for seed in seeds:
        fold_oof = np.zeros(len(trn), dtype=float)
        fold_test = np.zeros(len(tes), dtype=float)
        for fold, (tr_idx, va_idx) in enumerate(skf.split(trn, strat_label(trn)),
            tr = trn.iloc[tr_idx]
            va = trn.iloc[va_idx]
            X_tr = tr[feat_cols]
            X_va = va[feat_cols]


# --- PAGE 525 ---

            X_te = tes[feat_cols]
            y_tr = tr["ed_cost_next3y_usd"].values.astype(float)
            y_va = va["ed_cost_next3y_usd"].values.astype(float)
            if log_target:
                y_tr_fit = np.log1p(y_tr)
                y_va_fit = np.log1p(y_va)
                objective = "regression"
                metric = "l2"
            else:
                y_tr_fit = y_tr
                y_va_fit = y_va
                objective = "regression_l1"
                metric = "l1"
            params = dict(
                objective=objective,
                n_estimators=N_ESTIMATORS,
                learning_rate=0.02 if not log_target else 0.03,
                num_leaves=128,
                min_data_in_leaf=25,
                feature_fraction=0.80,
                bagging_fraction=0.85,
                bagging_freq=1,
                lambda_l2=5.0,
                random_state=int(seed) + fold,
                n_jobs=-1,
                force_col_wise=True,
            )
            if mono is not None:
                params["monotone_constraints"] = mono
            model = lgb.LGBMRegressor(**params)
            callbacks = []
            if lgb_early_stopping is not None:
                callbacks.append(lgb_early_stopping(stopping_rounds=EARLY_STOP, v
            elif hasattr(lgb, "early_stopping"):
                callbacks.append(lgb.early_stopping(stopping_rounds=EARLY_STOP, v


# --- PAGE 526 ---

            # robust fit across versions
            try:
                model.fit(
                    eval_set=[(X_va, y_va_fit)],
                    eval_metric=metric,
                    callbacks=callbacks if callbacks else None,
                )
            except TypeError:
                # fallback: early_stopping_rounds
                model.fit(
                    eval_set=[(X_va, y_va_fit)],
                    eval_metric=metric,
                    early_stopping_rounds=EARLY_STOP,
                    verbose=False
                )
            best_iter = getattr(model, "best_iteration_", None)
            pv = model.predict(X_va, num_iteration=best_iter)
            pt = model.predict(X_te, num_iteration=best_iter)
            if log_target:
                pv = np.expm1(pv)
                pt = np.expm1(pt)
            pv = np.clip(pv, 0, None)
            pt = np.clip(pt, 0, None)
            fold_oof[va_idx] = pv
            fold_test += pt / N_SPLITS
            mae = mean_absolute_error(y_va, pv)
            out_mask = (va["is_outlier"].values.astype(int) == 1)
            out_mae = mean_absolute_error(y_va[out_mask], pv[out_mask]) if out_ma
            tag = "logY" if log_target else "L1"
            print(f"[lgb_{tag}] seed={seed} fold{fold}: MAE={mae:.3f} | out_MAE={
        test_pred += fold_test / len(seeds)


# --- PAGE 527 ---

        oof += fold_oof / len(seeds)
    return oof, test_pred
# -----------------------
# CatBoost CV
# -----------------------
def train_cat_cv(train_df, test_df, feat_cols, cat_cols, seeds):
    if not HAS_CAT:
        print("[cat] not installed; skip")
        return None
    oof = np.zeros(len(train_df), dtype=float)
    test_pred = np.zeros(len(test_df), dtype=float)
    # cat feature indices
    cat_idx = [feat_cols.index(c) for c in cat_cols if c in feat_cols]
    X = train_df[feat_cols].copy()
    X_test = test_df[feat_cols].copy()
    y = train_df["ed_cost_next3y_usd"].values.astype(float)
    for seed in seeds:
        fold_oof = np.zeros(len(train_df), dtype=float)
        fold_test = np.zeros(len(test_df), dtype=float)
        for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, strat_label(t
            X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
            X_va, y_va = X.iloc[va_idx], y[va_idx]
            tr_pool = Pool(X_tr, y_tr, cat_features=cat_idx) if len(cat_idx) else
            va_pool = Pool(X_va, y_va, cat_features=cat_idx) if len(cat_idx) else
            te_pool = Pool(X_test, cat_features=cat_idx) if len(cat_idx) else Poo
            model = CatBoostRegressor(
                loss_function="MAE",
                iterations=12000,
                learning_rate=0.03,
                depth=6,
                l2_leaf_reg=6.0,


# --- PAGE 528 ---

                random_seed=int(seed) + fold,
                subsample=0.85,
                rsm=0.85,
                bootstrap_type="Bernoulli",
                od_type="Iter",
                od_wait=EARLY_STOP,
                allow_writing_files=False,
                verbose=False,
            )
            model.fit(tr_pool, eval_set=va_pool, use_best_model=True)
            pv = model.predict(va_pool)
            pt = model.predict(te_pool)
            pv = np.clip(pv, 0, None)
            pt = np.clip(pt, 0, None)
            fold_oof[va_idx] = pv
            fold_test += pt / N_SPLITS
            mae = mean_absolute_error(y_va, pv)
            out_mask = (train_df.iloc[va_idx]["is_outlier"].values.astype(int) ==
            out_mae = mean_absolute_error(y_va[out_mask], pv[out_mask]) if out_ma
            print(f"[cat_MAE] seed={seed} fold{fold}: MAE={mae:.3f} | out_MAE={ou
        oof += fold_oof / len(seeds)
        test_pred += fold_test / len(seeds)
    return oof, test_pred
# -----------------------
# Train models
# -----------------------
oof_preds = {}
test_preds = {}
if USE_LIGHTGBM and HAS_LGB:
    print("\n=== Train LightGBM (L1 on y) ===")
    out = train_lgb_cv(all_train, all_test, feat_cols, CAT_COLS, SEEDS, log_targe
    if out is not None:
        oof_lgb, te_lgb = out


# --- PAGE 529 ---

        oof_preds["lgb_l1"] = oof_lgb
        test_preds["lgb_l1"] = te_lgb
if USE_LOG_TARGET_LGB and USE_LIGHTGBM and HAS_LGB:
    print("\n=== Train LightGBM (L2 on log1p(y)) ===")
    out = train_lgb_cv(all_train, all_test, feat_cols, CAT_COLS, SEEDS, log_targe
    if out is not None:
        oof_lgb_log, te_lgb_log = out
        oof_preds["lgb_logy"] = oof_lgb_log
        test_preds["lgb_logy"] = te_lgb_log
if USE_CATBOOST and HAS_CAT:
    print("\n=== Train CatBoost (MAE) ===")
    out = train_cat_cv(all_train, all_test, feat_cols, CAT_COLS, SEEDS)
    if out is not None:
        oof_cat, te_cat = out
        oof_preds["cat_mae"] = oof_cat
        test_preds["cat_mae"] = te_cat
# -----------------------
# Evaluate & ensemble
# -----------------------
def eval_pred(name, oof):
    mae = mean_absolute_error(y, oof)
    out_mae = mean_absolute_error(y[is_out==1], oof[is_out==1]) if (is_out==1).any
    return mae, out_mae
rows = []
for name, oof in oof_preds.items():
    mae, out_mae = eval_pred(name, oof)
    rows.append((name, mae, out_mae))
summary = pd.DataFrame(rows, columns=["model","oof_mae","out_oof_mae"]).sort_valu
print("\n=== OOF SUMMARY ===")
print(summary.to_string(index=False))
summary.to_csv(OUT_DIR / "iter26_model_oof_summary.csv", index=False)
# weighted ensemble: inverse MAE
if len(oof_preds) == 0:
    raise RuntimeError("No model trained. Check LightGBM/CatBoost installation to


# --- PAGE 530 ---

inv = {k: 1.0 / max(1e-9, mean_absolute_error(y, v)) for k, v in oof_preds.items(
s = sum(inv.values())
w = {k: inv[k]/s for k in inv}
print("\n[ensemble] weights=", w)
oof_ens = np.zeros_like(y)
te_ens = np.zeros(len(all_test), dtype=float)
for k in oof_preds:
    oof_ens += w[k] * oof_preds[k]
    te_ens  += w[k] * test_preds[k]
# -----------------------
# Group residual correction (optional, shrunk)
# -----------------------
def apply_group_resid_corr(train_df, oof_pred, test_df, test_pred, shrink_k=80.0)
    key_tr = train_df["primary_chronic"].astype(str) + "|" + train_df["insurance"
    resid = train_df["ed_cost_next3y_usd"].values.astype(float) - oof_pred.astype
    tmp = pd.DataFrame({"key": key_tr.values, "resid": resid})
    grp = tmp.groupby("key")["resid"].agg(["median","count"]).reset_index()
    grp["shrink"] = grp["count"] / (grp["count"] + shrink_k)
    grp["adj"] = grp["median"] * grp["shrink"]
    global_adj = float(np.median(resid))
    key_te = test_df["primary_chronic"].astype(str) + "|" + test_df["insurance"].
    adj_map = dict(zip(grp["key"], grp["adj"]))
    te_adj = key_te.map(adj_map).fillna(global_adj).values.astype(float)
    te_corr = np.clip(test_pred + te_adj, 0, None)
    # (for reporting only) "corrected oof" using same group adj (small leakage bu
    tr_adj = pd.Series(key_tr).map(adj_map).fillna(global_adj).values.astype(float
    oof_corr = np.clip(oof_pred + tr_adj, 0, None)
    return oof_corr, te_corr
if USE_GROUP_RESID_CORR:
    oof_ens_corr, te_ens_corr = apply_group_resid_corr(all_train, oof_ens, all_te
    mae0 = mean_absolute_error(y, oof_ens)
    mae1 = mean_absolute_error(y, oof_ens_corr)


# --- PAGE 531 ---

    print(f"\n[group-corr] OOF MAE: before={mae0:.3f} after={mae1:.3f}  (shrink_k
    oof_final = oof_ens_corr
    te_final = te_ens_corr
else:
    oof_final = oof_ens
    te_final = te_ens
# final metrics
mae_final = mean_absolute_error(y, oof_final)
out_mae_final = mean_absolute_error(y[is_out==1], oof_final[is_out==1]) if (is_out
print(f"\n=== FINAL ENSEMBLE OOF ===\nOOF MAE={mae_final:.3f} | Outlier OOF MAE={
# -----------------------
# Save artifacts
# -----------------------
oof_df = pd.DataFrame({
    "patient_id": all_train["patient_id"].values,
    "y_true": y,
    "oof_pred": oof_final
})
oof_df.to_csv(OUT_DIR / "iter26_oof_predictions.csv", index=False)
print(f"[write] {OUT_DIR / 'iter26_oof_predictions.csv'}")
# submission
sub = pd.DataFrame({
    "patient_id": all_test["patient_id"].values,
    "ed_cost_next3y_usd": te_final.astype(float)
})
sub_path = OUT_DIR / "submission_iter26.csv"
sub.to_csv(sub_path, index=False)
print(f"[write] {sub_path}")
# plots
plt.figure(figsize=(7,6))
plt.scatter(all_train["prior_ed_cost_5y_usd"].values, y, s=8)
plt.xlabel("prior_ed_cost_5y_usd")
plt.ylabel("ed_cost_next3y_usd")
plt.title("Train: prior cost vs future cost (Iter26)")
plt.tight_layout()
plt.savefig(OUT_DIR / "scatter_prior_vs_future_iter26.png", dpi=160)


# --- PAGE 532 ---

看 CatBoost / LGB(L1) / LGB(logY) 哪个更稳，然后看 ensemble 权重是不是“被某个模
StratifiedKFold  + GroupKFold(zip3)  两套 CV，并用两套 OOF 的一致性来决定特征/模
plt.show()
plt.figure(figsize=(6,6))
plt.scatter(y, oof_final, s=8)
plt.xlabel("actual")
plt.ylabel("oof_pred")
plt.title("OOF pred vs actual (Iter26 ensemble)")
plt.tight_layout()
plt.savefig(OUT_DIR / "oof_pred_vs_actual_iter26.png", dpi=160)
plt.show()
print("\n[DONE] Iter26 finished.")
```

### C23 — Code cell starting at page 539
```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
import numpy as np
# 1) 准备 adv 数据：train=0, test=1
adv_cols = [c for c in feat_cols if c != "ed_cost_next3y_usd"]  # 只用特征
X_tr = all_train[adv_cols].copy()


# --- PAGE 540 ---

B) LightGBM 加 monotone_constraints
X_te = all_test[adv_cols].copy()
# 简单处理：把category先转成字符串再做 factorize（或你用 LGB 分类器也行）
for c in cat_cols:
    X_tr[c] = X_tr[c].astype(str)
    X_te[c] = X_te[c].astype(str)
# 对齐编码
for c in cat_cols:
    all_vals = pd.concat([X_tr[c], X_te[c]], axis=0)
    codes, _ = pd.factorize(all_vals)
    X_tr[c] = codes[:len(X_tr)]
    X_te[c] = codes[len(X_tr):]
X_adv = pd.concat([X_tr, X_te], axis=0).reset_index(drop=True)
y_adv = np.concatenate([np.zeros(len(X_tr)), np.ones(len(X_te))]).astype(int)
# 2) OOF 预测 p(test|x)，避免泄露
skf_adv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = LogisticRegression(max_iter=2000, n_jobs=-1)
p_test_oof = cross_val_predict(clf, X_adv, y_adv, cv=skf_adv, method="predict_pro
auc = roc_auc_score(y_adv, p_test_oof)
print(f"[adv] train-vs-test AUC={auc:.4f}")
# 3) importance weights (只取 train 部分)
p_tr = np.clip(p_test_oof[:len(X_tr)], 1e-3, 1-1e-3)
w = p_tr / (1.0 - p_tr)
# 关键：clip 防止极端权重把训练搞炸
w = np.clip(w, 0.25, 4.0)
all_train["sample_weight"] = w
```

### C24 — Code cell starting at page 540
```python
# --- PAGE 541 ---

C) 训练时把 weight 传进去
# 你要约束的特征集合（按你真实 feat_cols 改）
mono_pos = {
}
mono = [1 if f in mono_pos else 0 for f in feat_cols]
lgb_params = dict(
    objective="regression_l1",
    n_estimators=8000,
    learning_rate=0.02,
    num_leaves=64,
    min_data_in_leaf=30,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=1,
    lambda_l2=2.0,
    random_state=42,
    n_jobs=-1,
    force_col_wise=True,
    monotone_constraints=mono,   # <= 这行就是关键
)
model.fit(
    sample_weight=tr_df["sample_weight"].values,
    eval_set=[(X_va, y_va)],
    # eval_sample_weight=[va_df["sample_weight"].values],  # 如果你想更严格
    eval_metric="l1",
```

### C25 — Code cell starting at page 541
```python
model.fit(
    sample_weight=tr_df["sample_weight"].values,
    eval_set=[(X_va, y_va)],
    # eval_sample_weight=[va_df["sample_weight"].values],  # 如果你想更严格
    eval_metric="l1",


# --- PAGE 542 ---

    callbacks=[...]
)
```

---

## C) MAE / Submission / OOF Logs (verbatim extracts)

### C1) Server submission scores found in PDF

#### Page 215
```text
- or add a CatBoost second model for a final “winning” blend.
And yes — at this point, you should absolutely start submitting models built from 
::contentReference[oaicite:11]{index=11}
Sources
Sources
Sources
Here is my real MAE output:
🚀 Submitting predictions...
✅ Prediction submitted successfully!
📊 Score: 461.0468 (MAE)
✅ Validation passed
✅ Submission successful!
   📊 Score: 461.0468
   📏 Metric: MAE
   ✔️  Validation: Passed
And here is our own log:
[info] receipt codes: 18 -> ['31500', '36556', '36620', 
'70450', '71045', '74177', '84484', '85025', '87070', '92950', 
'99281', '99282', '99283', '99284', '99285', '99291', '99292', 
'G0378']
[info] n_features=85 | n_cat=5
fold 1: MAE=448.159 | out_n= 16 | out_MAE=509.772
```

#### Page 258
```text
Then we’ll decide the next highest-yield move:
(A) tune the new cross-task feature blocks (admissions vs vitals vs text)
(B) add a second LightGBM parameter “family” for a more diverse ensemble
(C) do a ZIP3-group stress validation to reduce leaderboard gap
This is the first approach that plausibly has enough missing signal to make a ~60 MAE
jump, because it uses the rest of the dataset that’s explicitly provided and joinable for this
challenge. 
description
Sources
🚀 Submitting predictions...
✅ Prediction submitted successfully!
📊 Score: 473.8055 (MAE)
✅ Validation passed
✅ Submission successful!
   📊 Score: 473.8055
   📏 Metric: MAE
   ✔️  Validation: Passed
[info] train_full shape: (2000, 9)  test_full shape: (2000, 8)
[cache] Wrote receipt cache: 
D:\AgentDs\agent_ds_healthcare\cache_cross_v1\receipts_
parsed.joblib | missing_pdfs=0
[info] Parsed lineitems rows: 27182  unique patients: 3900
[info] receipt codes: 18 -> ['31500', '36556', '36620', 
'70450', '71045', '74177', '84484', '85025', '87070', '92950',
```

#### Page 537
```text
D:\AgentDs\agent_ds_healthcare\cache_iter26_robust\sub
mission_iter26.csv
🚀 Submitting predictions...
✅ Prediction submitted successfully!
📊 Score: 460.9853 (MAE)
✅ Validation passed
✅ Submission successful!
   📊 Score: 460.9853
   📏 Metric: MAE
   ✔️  Validation: Passed
what should we do next
Thought for 22m 30s
你们现在最该做的下一步，不是继续“加特征/加模型堆参数”去追 OOF 445 → 430，而是先
解决一个更致命的问题：
你们的本地 CV（OOF=445）和线上得分（LB≈461）存在稳定且不小的 gap（+15~20
MAE）。
只要这个 gap 不被压缩，你再怎么把 OOF 做到 430，线上也可能还是 450+ —— 这就是你
```

### C2) Page-level log line extracts

#### Page 115
```text
fold 4: MAE=451.742 | outlier_MAE=261.653
=> MAE: 456.190 ± 14.898
=> Outlier MAE (prior_cost<=50 & visits==0): 384.886 ± 
CV MAE ≈ 456.19 ± 14.90
Outlier MAE (prior_cost≤50 & visits==0) ≈ 384.89 ± 108.34
```

#### Page 127
```text
fold 5: MAE=605.015 | out_n= 21 | out_MAE=514.845
=== SUMMARY ===
```

#### Page 139
```text
=== SUMMARY (sorted by MAE) ===
```

#### Page 151
```text
=== SUMMARY (sorted by MAE) ===
[info] unique receipt codes (train): 18 -> ['31500', '36556',
```

#### Page 197
```text
print(f"OOF MAE={raw_mae:,.3f} | Outlier OOF MAE={raw_out_mae:,.3f} | out_n={int(
```

#### Page 198
```text
print(f"OOF MAE={cal_mae:,.3f} | Outlier OOF MAE={cal_out_mae:,.3f}")
        print("[calibration] applied to test predictions (improved OOF MAE).")
        print("[calibration] NOT applied (did not improve OOF MAE).")
```

#### Page 215
```text
🚀 Submitting predictions...
✅ Prediction submitted successfully!
📊 Score: 461.0468 (MAE)
✅ Validation passed
✅ Submission successful!
[info] receipt codes: 18 -> ['31500', '36556', '36620', 
[info] n_features=85 | n_cat=5
fold 1: MAE=448.159 | out_n= 16 | out_MAE=509.772
fold 2: MAE=429.263 | out_n= 19 | out_MAE=322.445
fold 3: MAE=478.537 | out_n= 18 | out_MAE=306.441
fold 4: MAE=453.732 | out_n= 11 | out_MAE=301.995
fold 5: MAE=443.378 | out_n= 21 | out_MAE=400.582
=== OOF SUMMARY (raw) ===
OOF MAE=450.614 | Outlier OOF MAE=370.975 |
```

#### Page 216
```text
=== OOF SUMMARY (calibrated) ===
OOF MAE=450.979 | Outlier OOF MAE=374.186
[calibration] NOT applied (did not improve OOF MAE).
[DONE] Wrote submission:
```

#### Page 236
```text
print(f"OOF MAE={oof_mae:.3f} | Outlier OOF MAE={out_mae:.3f} | out_n={len(outlie
```

#### Page 258
```text
🚀 Submitting predictions...
✅ Prediction submitted successfully!
📊 Score: 473.8055 (MAE)
✅ Validation passed
✅ Submission successful!
[info] train_full shape: (2000, 9)  test_full shape: (2000, 8)
[cache] Wrote receipt cache: 
[info] Parsed lineitems rows: 27182  unique patients: 3900
[info] receipt codes: 18 -> ['31500', '36556', '36620', 
[cache] Wrote receipt features: 
[aux-readmit] OOF AUC=0.8795
```

#### Page 259
```text
[cache] Wrote admissions patient features: 
[cache] Wrote stays patient features: 
[info] n_features=133 | n_cat=4 | outlier_n=85
fold 1: MAE=455.560 | out_n= 21 | out_MAE=401.515 | 
fold 2: MAE=448.357 | out_n= 15 | out_MAE=583.819 | 
fold 3: MAE=465.416 | out_n= 18 | out_MAE=318.744 | 
fold 4: MAE=474.749 | out_n= 15 | out_MAE=374.070 | 
fold 5: MAE=459.143 | out_n= 16 | out_MAE=415.048 | 
=== OOF SUMMARY ===
OOF MAE=460.645 | Outlier OOF MAE=413.863 | 
[DONE] Wrote submission:
```

#### Page 295
```text
print("\n=== SUMMARY (sorted by OOF MAE) ===")
```

#### Page 298
```text
=== SUMMARY (sorted by OOF MAE) ===
[DONE] Outputs:
```

#### Page 334
```text
2. ablation_results_iter23.csv  里四个模型的 OOF MAE
print("\n=== SUMMARY (sorted by OOF MAE) ===")
```

#### Page 337
```text
=== SUMMARY (sorted by OOF MAE) ===
```

#### Page 357
```text
print("\n=== SUMMARY (sorted by OOF MAE) ===")
```

#### Page 369
```text
=== SUMMARY (sorted by OOF MAE) ===
```

#### Page 431
```text
1. === SUMMARY (lower is better) ===  那个表（四个 baseline 的 OOF MAE）
```

#### Page 434
```text
=== SUMMARY (lower is better) ===
```

#### Page 495
```text
=== CV ABLATIONS (Iter24, CatBoost MAE) ===
=== SUMMARY (sorted by OOF MAE) ===
```

#### Page 536
```text
=== OOF SUMMARY ===
[group-corr] OOF MAE: before=448.809 after=445.446  
=== FINAL ENSEMBLE OOF ===
OOF MAE=445.446 | Outlier OOF MAE=407.128
```

#### Page 537
```text
🚀 Submitting predictions...
✅ Prediction submitted successfully!
📊 Score: 460.9853 (MAE)
✅ Validation passed
✅ Submission successful!
```

### C3) Parsed MAE/OOF/Score events (machine-friendly list)
Each bullet is `(page)  event`.
- (p115) FOLD_MAE  fold=4  mae=451.742  | fold 4: MAE=451.742
- (p115) FOLD_MAE  fold=5  mae=443.042  | fold 5: MAE=443.042
- (p115) MAE_LINE  mae=456.190  | MAE: 456.190
- (p127) FOLD_MAE  fold=5  mae=605.015  | fold 5: MAE=605.015
- (p128) MAE_LINE  mae=401.580  | MAE: 401.58
- (p128) MAE_LINE  mae=454.910  | MAE: 454.91
- (p139) FOLD_MAE  fold=5  mae=439.052  | fold 5: MAE=439.052
- (p151) FOLD_MAE  fold=1  mae=451.170  | fold 1: MAE=451.170
- (p152) FOLD_MAE  fold=2  mae=436.316  | fold 2: MAE=436.316
- (p152) FOLD_MAE  fold=3  mae=484.490  | fold 3: MAE=484.490
- (p152) FOLD_MAE  fold=4  mae=451.998  | fold 4: MAE=451.998
- (p152) FOLD_MAE  fold=5  mae=440.803  | fold 5: MAE=440.803
- (p155) FOLD_MAE  fold=1  mae=449.359  | fold 1: MAE=449.359
- (p159) FOLD_MAE  fold=2  mae=439.188  | fold 2: MAE=439.188
- (p163) FOLD_MAE  fold=3  mae=487.206  | fold 3: MAE=487.206
- (p166) FOLD_MAE  fold=4  mae=470.855  | fold 4: MAE=470.855
- (p170) FOLD_MAE  fold=1  mae=451.170  | fold 1: MAE=451.170
- (p170) FOLD_MAE  fold=2  mae=436.316  | fold 2: MAE=436.316
- (p170) FOLD_MAE  fold=3  mae=484.490  | fold 3: MAE=484.490
- (p170) FOLD_MAE  fold=4  mae=451.998  | fold 4: MAE=451.998
- (p170) FOLD_MAE  fold=5  mae=440.803  | fold 5: MAE=440.803
- (p170) FOLD_MAE  fold=5  mae=442.901  | fold 5: MAE=442.901
- (p174) FOLD_MAE  fold=1  mae=449.359  | fold 1: MAE=449.359
- (p177) FOLD_MAE  fold=2  mae=439.188  | fold 2: MAE=439.188
- (p181) FOLD_MAE  fold=3  mae=487.206  | fold 3: MAE=487.206
- (p185) FOLD_MAE  fold=4  mae=470.855  | fold 4: MAE=470.855
- (p188) FOLD_MAE  fold=5  mae=442.901  | fold 5: MAE=442.901
- (p215) FOLD_MAE  fold=1  mae=448.159  | fold 1: MAE=448.159
- (p215) FOLD_MAE  fold=2  mae=429.263  | fold 2: MAE=429.263
- (p215) FOLD_MAE  fold=3  mae=478.537  | fold 3: MAE=478.537
- (p215) FOLD_MAE  fold=4  mae=453.732  | fold 4: MAE=453.732
- (p215) FOLD_MAE  fold=5  mae=443.378  | fold 5: MAE=443.378
- (p215) OOF_MAE  mae=370.975  | OOF MAE=370.975
- (p215) OOF_MAE  mae=450.614  | OOF MAE=450.614
- (p215) SERVER_SCORE  461.0468  | Score: 461.0468 (MAE)
- (p216) OOF_MAE  mae=374.186  | OOF MAE=374.186
- (p216) OOF_MAE  mae=450.979  | OOF MAE=450.979
- (p258) SERVER_SCORE  473.8055  | Score: 473.8055 (MAE)
- (p259) FOLD_MAE  fold=1  mae=455.560  | fold 1: MAE=455.560
- (p259) FOLD_MAE  fold=2  mae=448.357  | fold 2: MAE=448.357
- (p259) FOLD_MAE  fold=3  mae=465.416  | fold 3: MAE=465.416
- (p259) FOLD_MAE  fold=4  mae=474.749  | fold 4: MAE=474.749
- (p259) FOLD_MAE  fold=5  mae=459.143  | fold 5: MAE=459.143
- (p259) OOF_MAE  mae=413.863  | OOF MAE=413.863
- (p259) OOF_MAE  mae=460.645  | OOF MAE=460.645
- (p317) MAE_LINE  mae=445.370  | MAE: 445.370
- (p336) FOLD_MAE  fold=1  mae=434.173  | fold 1: MAE=434.173
- (p336) FOLD_MAE  fold=1  mae=434.173  | fold 1: MAE=434.173
- (p336) FOLD_MAE  fold=2  mae=449.486  | fold 2: MAE=449.486
- (p336) FOLD_MAE  fold=2  mae=449.486  | fold 2: MAE=449.486
- (p336) FOLD_MAE  fold=3  mae=485.181  | fold 3: MAE=485.181
- (p336) FOLD_MAE  fold=3  mae=485.181  | fold 3: MAE=485.181
- (p336) FOLD_MAE  fold=4  mae=438.882  | fold 4: MAE=438.882
- (p336) FOLD_MAE  fold=4  mae=438.882  | fold 4: MAE=438.882
- (p336) FOLD_MAE  fold=5  mae=450.883  | fold 5: MAE=450.883
- (p337) FOLD_MAE  fold=1  mae=436.756  | fold 1: MAE=436.756
- (p337) FOLD_MAE  fold=1  mae=436.756  | fold 1: MAE=436.756
- (p337) FOLD_MAE  fold=2  mae=448.278  | fold 2: MAE=448.278
- (p337) FOLD_MAE  fold=2  mae=448.278  | fold 2: MAE=448.278
- (p337) FOLD_MAE  fold=3  mae=489.302  | fold 3: MAE=489.302
- (p337) FOLD_MAE  fold=3  mae=489.302  | fold 3: MAE=489.302
- (p337) FOLD_MAE  fold=4  mae=443.527  | fold 4: MAE=443.527
- (p337) FOLD_MAE  fold=4  mae=443.527  | fold 4: MAE=443.527
- (p337) FOLD_MAE  fold=5  mae=450.883  | fold 5: MAE=450.883
- (p337) FOLD_MAE  fold=5  mae=451.589  | fold 5: MAE=451.589
- (p337) FOLD_MAE  fold=5  mae=451.589  | fold 5: MAE=451.589
- (p368) FOLD_MAE  fold=1  mae=454.158  | fold 1: MAE=454.158
- (p368) FOLD_MAE  fold=1  mae=457.064  | fold 1: MAE=457.064
- (p368) FOLD_MAE  fold=1  mae=462.380  | fold 1: MAE=462.380
- (p368) FOLD_MAE  fold=2  mae=434.120  | fold 2: MAE=434.120
- (p368) FOLD_MAE  fold=2  mae=436.374  | fold 2: MAE=436.374
- (p368) FOLD_MAE  fold=2  mae=439.938  | fold 2: MAE=439.938
- (p368) FOLD_MAE  fold=3  mae=454.308  | fold 3: MAE=454.308
- (p368) FOLD_MAE  fold=3  mae=468.511  | fold 3: MAE=468.511
- (p368) FOLD_MAE  fold=3  mae=468.850  | fold 3: MAE=468.850
- (p368) FOLD_MAE  fold=4  mae=447.567  | fold 4: MAE=447.567
- (p368) FOLD_MAE  fold=4  mae=449.924  | fold 4: MAE=449.924
- (p368) FOLD_MAE  fold=4  mae=452.629  | fold 4: MAE=452.629
- (p368) FOLD_MAE  fold=5  mae=444.706  | fold 5: MAE=444.706
- (p368) FOLD_MAE  fold=5  mae=444.934  | fold 5: MAE=444.934
- (p368) FOLD_MAE  fold=5  mae=448.666  | fold 5: MAE=448.666
- (p369) FOLD_MAE  fold=1  mae=459.864  | fold 1: MAE=459.864
- (p369) FOLD_MAE  fold=2  mae=442.014  | fold 2: MAE=442.014
- (p369) FOLD_MAE  fold=3  mae=460.954  | fold 3: MAE=460.954
- (p369) FOLD_MAE  fold=4  mae=456.448  | fold 4: MAE=456.448
- (p369) FOLD_MAE  fold=5  mae=446.461  | fold 5: MAE=446.461
- (p494) FOLD_MAE  fold=1  mae=454.158  | fold 1: MAE=454.158
- (p494) FOLD_MAE  fold=1  mae=462.380  | fold 1: MAE=462.380
- (p494) FOLD_MAE  fold=2  mae=434.120  | fold 2: MAE=434.120
- (p494) FOLD_MAE  fold=2  mae=439.938  | fold 2: MAE=439.938
- (p494) FOLD_MAE  fold=3  mae=468.511  | fold 3: MAE=468.511
- (p494) FOLD_MAE  fold=4  mae=452.629  | fold 4: MAE=452.629
- (p494) FOLD_MAE  fold=5  mae=444.706  | fold 5: MAE=444.706
- (p495) FOLD_MAE  fold=1  mae=457.064  | fold 1: MAE=457.064
- (p495) FOLD_MAE  fold=1  mae=459.864  | fold 1: MAE=459.864
- (p495) FOLD_MAE  fold=2  mae=436.374  | fold 2: MAE=436.374
- (p495) FOLD_MAE  fold=2  mae=442.014  | fold 2: MAE=442.014
- (p495) FOLD_MAE  fold=3  mae=454.308  | fold 3: MAE=454.308
- (p495) FOLD_MAE  fold=3  mae=460.954  | fold 3: MAE=460.954
- (p495) FOLD_MAE  fold=3  mae=468.850  | fold 3: MAE=468.850
- (p495) FOLD_MAE  fold=4  mae=447.567  | fold 4: MAE=447.567
- (p495) FOLD_MAE  fold=4  mae=449.924  | fold 4: MAE=449.924
- (p495) FOLD_MAE  fold=4  mae=456.448  | fold 4: MAE=456.448
- (p495) FOLD_MAE  fold=5  mae=444.934  | fold 5: MAE=444.934
- (p495) FOLD_MAE  fold=5  mae=446.461  | fold 5: MAE=446.461
- (p495) FOLD_MAE  fold=5  mae=448.666  | fold 5: MAE=448.666
- (p536) OOF_MAE  mae=407.128  | OOF MAE=407.128
- (p536) OOF_MAE  mae=445.446  | OOF MAE=445.446
- (p537) SERVER_SCORE  460.9853  | Score: 460.9853 (MAE)
