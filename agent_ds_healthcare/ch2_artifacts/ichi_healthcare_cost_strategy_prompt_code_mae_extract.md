# AgentDS Healthcare Domain — Challenge 2: ED Cost Forecasting

This Markdown is an *objective extraction* of **prompts**, **code scripts**, and **MAE / submission / OOF logs** from the PDF:
- Source PDF: `IEEE ICHI - Healthcare Cost Forecasting Strategy.pdf`
- Extraction time (UTC): 2026-02-20 04:52:59
- Notes:
  - Page numbers below refer to the PDF's **1-indexed** pages.
  - Content is kept as verbatim as possible; formatting artifacts may exist due to PDF text extraction (line wrapping, broken indentation).
  - **No tables** are used (bullet lists and code/text blocks only).

---

## Contents
- A) Prompt Blocks (verbatim)
- B) Code Scripts (verbatim)
- C) Key MAE / Score / Submission Logs (verbatim)

---

## A) Prompt Blocks (verbatim)

### P01 — You are a World-Class AI Researcher and Kaggle (pages 1-4)
```text
[System Role]
You are a World-Class AI Researcher and Kaggle 
Grandmaster. You are famous for winning "Data-Centric 
AI" challenges where feature engineering and creative data 
processing outweigh raw model size. You operate with 
strict scientific rigor.
[The Objective]
I am competing in AgentDS Challenge 2: ED Cost 
Forecasting.
My goal is to minimize Mean Absolute Error (MAE) on a 
regression task predicting total Emergency Department 
costs over the next 3 years.
[The Data Ecosystem]
I have two distinct data modalities for each patient:
Structured Tabular Data (ed_cost_train.csv): Contains 
patient_id, primary_chronic condition, prior_ed_visits_5y, 
and prior_ed_cost_5y_usd. 
Unstructured PDF Receipts (receipts_pdf/): A collection of 
PDF files (one per patient). These are synthetic "billing 
summaries" containing line items (CPT/HCPCS codes), 
quantities, and subtotals.
Critical Note: The detailed billing codes (CPT) inside the 
PDFs are likely highly predictive of future risk, but they are 
description.md
File
receipt_2.pdf
PDF
receipt_3.pdf
PDF
trapped in PDF format.
I have attached the sample files and full description of the 
task for your reference.
[The Constraints]
Metric: MAE (Mean Absolute Error).
Hardware: Local Windows PC with NVIDIA RTX 4060 (8GB 
VRAM).
Time: I need a solution that is feasible to run/train locally.
[Your Task: Tree of Thoughts Reasoning]
Do not rush to code or a single solution. I want you to 
perform a Deep Strategic Search to find the optimal 
winning path.
Step 1: The Literature & Context Search (Mental Sandbox)
Search for similar "Healthcare Cost Prediction" or 
"Insurance Claims Forecasting" competitions (e.g., Kaggle, 
drivendata).
Search for "Zero-Inflated Regression" techniques, as 
healthcare cost data is notoriously sparse and right-
skewed.
Search for lightweight "PDF to Structured Data" pipelines 
suitable for an RTX 4060 (e.g., OCR vs. text extraction vs. 
layout models).
Step 2: Generate 3 Distinct Strategic Approaches
Propose three fundamentally different architectures to 
solve this.
Strategy A (The Baseline): A robust, low-compute approach 
focusing purely on tabular aggregates.
Strategy B (The Multimodal Specialist): A strategy that
heavily exploits the PDF receipts to extract granular CPT 
codes, using efficient NLP/embedding techniques.
Strategy C (The Deep Regressor): A deep learning 
approach (e.g., TabTransformer or similar) that learns 
representations jointly.
Step 3: The Critique & Selection
Critique each strategy based on:
Feasibility: Can it run on 8GB VRAM?
MAE Suitability: Does it handle outliers/skewed data well? 
(MAE is sensitive to the median, not the mean).
Performance Ceiling: Which has the highest probability of 
winning?
Step 4: The Final Execution Plan
Select the single best strategy from Step 3 and detail the 
execution plan:
Data Preprocessing: Specifically, how to handle the skew in 
cost data without losing the information needed for MAE.
PDF Pipeline: A concrete method to turn PDFs into model 
features on my local GPU.
Model Architecture: The specific algorithm(s) and why they 
fit the data distribution.
Validation: A validation scheme that mimics the test set 
distribution (handling potential time/patient leakage).
[Action]
Start by searching the web for "healthcare cost regression 
kaggle winning solutions" and "handling zero-inflated cost 
data mae". Then, present your Tree of Thoughts analysis.
```

### P02 — You are a Kaggle Grandmaster and Senior Machine (pages 14-16)
```text
[System Role]
You are a Kaggle Grandmaster and Senior Machine 
Learning Engineer. You are tasked with generating a 
winning solution for AgentDS Healthcare Challenge 2. You 
have full autonomy over the technical implementation 
details (preprocessing, feature engineering, model 
selection) provided they fit the hardware constraints.
[The Objective]
Minimize Mean Absolute Error (MAE) on the "ED Cost 
Forecasting" task.
Input: Tabular patient history + PDF Receipts (containing 
granular CPT/HCPCS billing codes).
Output: A submission.csv file predicting 
ed_cost_next3y_usd.
[Hard Constraints & Environment]
Hardware: Local Windows PC with NVIDIA RTX 4060 (8GB 
VRAM).
Constraint: You must prioritize efficiency. Do not use heavy 
OCR (like Tesseract) unless necessary; prefer fast text 
extraction (e.g., pymupdf).
File Paths (Local):
Train Data: 
D:\AgentDs\agent_ds_healthcare\ed_cost_train.csv
Test Data: 
D:\AgentDs\agent_ds_healthcare\ed_cost_test.csv
PDF Directory: 
D:\AgentDs\agent_ds_healthcare\receipts_pdf\ (Files are 
named receipt_{patient_id}.pdf)
Sample PDF: 
D:\AgentDs\agent_ds_healthcare\receipts_pdf\receipt_2.pd
f
Submission Format:
File name: submission_ICHI_V1.csv
Columns: patient_id, ed_cost_next3y_usd (Header 
required).
The patient_id must match ed_cost_test.csv exactly.
[Strategic Direction: "The Multimodal Specialist"]
I want you to implement Strategy B (Multimodal Gradient 
Boosting) because it balances performance with my 
hardware limits.
PDF Intelligence: You must extract meaningful signals from 
the PDF receipts (e.g., specific CPT codes, line item counts, 
or text embeddings) because the tabular data alone is 
insufficient.
Metric Alignment: The competition metric is MAE. Your 
loss function, target transformations, and model objectives 
must be mathematically aligned to minimize MAE (not 
MSE).
Robustness: Healthcare cost data is skewed. Your code 
must handle outliers and zero-inflation risks appropriate 
for an MAE target.
[Deliverable]
Write a single, complete, production-grade Python script 
that I can run in Jupyter or VS Code.
Include a comment block at the top with necessary pip 
install commands.
[Action]
Think deeply about how to translate PDF text into 
regression features for an MAE target. Then, generate the 
code.
```

### P03 — You are a Kaggle Grandmaster and Lead AI Scientist. You (pages 45-47)
```text
[System Role]
You are a Kaggle Grandmaster and Lead AI Scientist. You 
are in an autonomous iteration loop to solve AgentDS 
Challenge 2: ED Cost Forecasting.
[Current Status Report]
Iteration 1 Score: 493.9 MAE.
Leaderboard Target: ~448.7 MAE.
Gap Analysis: We are trailing the best solution by ~45 
points (approx. 10%). This is a massive gap in a regression 
challenge.
Diagnosis: Our previous approach likely under-utilized the 
unstructured PDF data or failed to capture the non-linear 
relationships in the cost data.
[The Challenge: "Black Box" Improvement]
I have NO domain knowledge of healthcare or CPT codes. I 
cannot help you feature engineer.
Your Task: You must analyze the problem abstractly and 
design a "Level 2" Solution that closes this gap.
The Core Question: What specific mathematical or 
architectural change is required to squeeze an extra 10% 
accuracy out of this dataset?
Is it better text processing? (e.g., embeddings vs counts?)
Is it a different loss function? (e.g., Tweedie vs MAE vs 
Quantile?)
Is it a stronger ensemble? (e.g., Stacking vs Blending?)
[Directives for Iteration 2]
Write a complete, self-contained Python script for Iteration 
2 that implements your best hypothesis for closing this 
gap.
Hypothesize: Briefly comment at the top of the code why 
you believe this new approach will beat the 493.9 baseline.
Execute: Write the full pipeline (Parsing -> Engineering -> 
Modeling -> Submission).
Constraint: You must use LightGBM, XGBoost, or CatBoost 
(or a mix). Do not use deep learning (transformers) due to 
the small dataset size (~2k rows) and instability risks.
Robustness: Ensure your local validation strategy is 
rigorous (e.g., Repeated K-Fold) so we know if the score is 
actually improving before we submit.
[Environment & Paths]
GPU: RTX 4060 (8GB).
Input: D:\AgentDs\agent_ds_healthcare\ed_cost_train.csv, 
test.csv, receipts_pdf/
Output: submission.csv
[Action]
Analyze the gap. Formulate a superior strategy. Write the 
code.
```

### P04 — You are a Senior ML Engineer focusing on MLOps and (pages 135-136)
```text
[System Role]
You are a Senior ML Engineer focusing on MLOps and 
Observability.
[The Problem]
I am running the "Iteration 2" script you provided. It runs, 
but it is too silent.
It hangs at CV folds: 0% for a long time. I cannot tell if the 
GPU is frozen, if the kernel died, or if it is just training 
slowly.
I am getting SyntaxWarning: invalid escape sequence 
errors regarding the Regex patterns at the top of the file.
[The Request]
Please provide the corrected, full Python script with the 
following specific improvements:
Fix Regex Warnings: Convert all Regex pattern strings (e.g., 
"\d...") to Python raw strings (e.g., r"\d...") to silence the 
SyntaxWarnings.
Enable Training Heartbeats (Crucial):
Change XGBoost verbosity to 1 (or 2) and set 
verbose_eval=100 (print every 100 rounds) inside the .fit() 
call.
Change CatBoost verbose to 100 (print every 100
iterations) in the parameters.
Add explicit logging.info messages before and after 
model.fit() for each fold (e.g., INFO | Fold 0 | Starting 
XGBoost training...).
Preserve Logic: Do not change the model architecture, 
hyperparameters, or seed logic. Only improve the logging 
and syntax safety.
[Output]
Provide the full, runnable code block. I want to copy-paste 
it and see immediate output as the models train.
```

### P05 — You are a Dual-Expert: a Clinical Data Scientist (specializing (pages 159-160)
```text
[System Role]
You are a Dual-Expert: a Clinical Data Scientist (specializing 
in medical coding, CPT/HCPCS hierarchies, and chronic 
disease burden) and a Kaggle Grandmaster (specializing in 
feature engineering for tabular regression).
[Situation Report]
Current Score: 478.38 MAE (Iteration 2).
Previous Score: 493.90 MAE.
Improvement: ~15.5 points gained by using "Blind" NLP 
(TF-IDF on billing codes).
The Problem: We are still trailing the leader (~448 MAE) by 
30 points. The "Blind NLP" approach has likely hit a ceiling 
because it treats medical codes (e.g., 99214 vs 99285) as 
arbitrary text tokens, ignoring their inherent clinical 
hierarchy and severity.
[The Objective: Iteration 3]
We need to "Unlock Domain Knowledge."
I want you to revisit the data (ed_cost_train.csv + PDF 
receipts) and apply your internal clinical domain 
knowledge to engineer superior features.
Do not ask me for directions. You are the expert.
Think: How does a "Level 5 ED Visit" (CPT 99285) differ 
from a "Level 1" (99281) in terms of cost risk? How do 
"Surgery" codes (10000-60000 range) differ from "Lab" 
codes (80000 range)?
Hypothesize: What specific clinical interactions (e.g., 
Chronic Condition + Recent Utilization Intensity) are likely 
driving the highest costs?
[Execution Directive]
Write the complete Iteration 3 Python Script.
Refined Strategy: Replace or augment the "Blind TF-IDF" 
with Structured Clinical Features. (e.g., grouping codes by 
CPT ranges, calculating "Severity Scores", or interaction 
terms between chronic conditions and utilization).
Rich Logging: The previous code was good, but ensure this 
version continues to have verbose, timestamped logs so I 
know exactly where the training is (e.g., INFO | Fold 1 | 
Feature Engineering..., INFO | XGBoost Epoch 50...).
Robustness: Keep the try-except parsing safety.
Hardware: Optimize for RTX 4060 (GPU enabled for 
XGB/CatBoost).
[Environment & Paths]
Input: D:\AgentDs\agent_ds_healthcare\ed_cost_train.csv, 
test.csv, receipts_pdf/
Output: submission_ICHI_V1.csv
[Action]
Analyze the clinical nature of the data. Formulate the 
"Domain Expert" feature set. Write the code.
```

### P06 — You are a Lead AI Scientist and Kaggle Grandmaster. We (pages 193-195)
```text
[System Role]
You are a Lead AI Scientist and Kaggle Grandmaster. We 
are conducting a "Failure Analysis" and strategic pivot.
[Performance Review]
Iteration 2 (Blind Statistical NLP): 478.4 MAE.
Iteration 3 (Clinical Domain Rules): 480.9 MAE.
The Verdict: The "Domain Expert" approach failed. The 
score degraded. This indicates that imposing manual 
clinical hierarchies (e.g., grouping codes by range) likely 
introduced bias or noise that the model couldn't handle 
given the small dataset size. The data "prefers" the raw, 
unbiased statistical signal over our human-defined rules.
[The Pivot: "Return to the Math"]
We are abandoning the "Manual Clinical Feature" path 
immediately. Do not use CPT ranges or manual groupings.
[The Challenge: Iteration 4]
The Problem: We know "simple counts" (Iter 1) are too 
weak. We know "manual clinical rules" (Iter 3) are too 
biased.
The Goal: You need to find a "Third Path"—a method to 
capture the structure and interaction of the medical codes 
without hard-coding their meaning.
Your Task: Think like a Mathematician, not a Doctor.
How do you represent high-dimensional, sparse data 
(thousands of unique billing codes) in a way that a tree-
based model (XGBoost/CatBoost) can digest efficiently?
If you can't group them by "medical meaning," how do 
you group them by "statistical similarity"?
Is there a better way to transform the target (Cost) or the 
loss function to align with the MAE metric?
[Execution Directive]
Write the complete Iteration 4 Python Script.
Discard: Remove the "Clinical Grouping" logic from the 
previous script.
Innovate: Implement a new mathematical representation 
technique for the PDF code data. (e.g., Embeddings? 
Dimensionality Reduction? Clustering? You decide).
Refine: Stick to the robust "Rich Logging" style from the 
previous iteration. I want to see the training heartbeat.
Constraint: RTX 4060 compatible.
[Action]
The data rejected our hypothesis. Listen to the data. 
Formulate the mathematical solution. Write the code.
```

### P07 — You are a Principal Data Scientist and Kaggle Grandmaster. (pages 225-226)
```text
[System Role]
You are a Principal Data Scientist and Kaggle Grandmaster. 
You are strictly evaluated on results.
[Status Report]
Baseline (Iteration 2): 478.4 MAE (Best so far).
Attempt 3 (Clinical Grouping): 480.9 MAE (Failed).
Attempt 4 (Latent/SVD Features): 483.9 MAE (Failed Badly).
[The Problem]
We are regressing. Every time we add "smart" complexity, 
the model gets worse. We need to beat 478.4, and we are 
currently moving in the wrong direction.
[Directives: WHAT TO AVOID]
You have full autonomy to design Iteration 5, but you must 
strictly adhere to these Negative Constraints based on our 
failures:
DO NOT use Dimensionality Reduction: Do not use SVD, 
PCA, NMF, or Learned Embeddings on the text data. 
Iteration 4 proved that compressing the sparse signal 
destroys information we need for this specific metric.
DO NOT use Manual Domain Groupings: Do not try to 
group CPT codes by range or "clinical meaning." Iteration 
3 proved that our manual buckets introduce bias and 
noise.
DO NOT ignore the Hardware: Do not build a solution that 
runs out of memory on an 8GB GPU.
DO NOT hide the training progress: Do not produce silent 
code. If the model is training, I need to see logs.
[The Objective]
Your goal is simply to minimize MAE below 478.4.
You know that "Blind NLP" (Iter 2) worked best.
You know that "Complexity" (Iter 3 & 4) hurt us.
Figure out how to scale the "simple" approach without 
hitting the "complexity" trap.
[Action]
Write the complete Iteration 5 Python Script. I am not 
telling you what features to build. I am telling you that 
Complexity = Failure on this dataset. Solve it.
```

### P08 — You are a Kaggle Grandmaster specializing in XGBoost (pages 246-248)
```text
[System Role]
You are a Kaggle Grandmaster specializing in XGBoost 
Optimization.
[Critical Error Analysis]
Look at the logs from our last run:
XGBoost MAE: 477.79 (Excellent, competitive).
CatBoost MAE: 1307.71 (Catastrophic failure).
Blend Weight: 100% XGBoost.
[The Diagnosis]
We have been wasting compute on a CatBoost model that 
is completely broken (likely due to a loss function 
mismatch or unscaled target). The "ensemble" approach is 
hurting us because we are blending a race car with a 
broken bicycle.
[The Strategy: "Monolithic XGBoost"]
We are reverting to the Iteration 2 logic (TF-IDF on CPT 
codes), but we are going to go "All-In" on the single 
working model.
Remove CatBoost: Delete it entirely. It is dead weight.
Scale the NLP: Since XGBoost is working well with the text 
features, give it more. Increase TF-IDF max_features from 
100 to 1000. Let the tree find the signal.
Target Calibration: The logs showed that 
Calibration(by_chronic) improved the MAE by ~2 points 
(477 -> 475). Keep this logic. It is a free win.
Tuning: Since we are only training one model, increase 
n_estimators to 8000 and lower the learning_rate to 0.01 
or 0.02 for a more precise convergence.
[The Task: Iteration 6]
Write the complete Python Script for the "Monolithic 
XGBoost" solution.
Features: Tabular + TF-IDF (1-3 grams, 1000 features).
Model: Single XGBoost (GPU enabled).
Post-Processing: Apply the "Chronic Condition Bias 
Correction" (subtracting the mean error per chronic group)
as seen in the logs.
Output: submission.csv.
[Action]
Cut the noise. Give me the single, highly-tuned XGBoost 
script that maximizes the "Blind NLP" signal.
```

---

## B) Code Scripts (verbatim)

### C01 — Script starting page 17 (pages 17-47)
```python
# -*- coding: utf-8 -*-
"""
AgentDS Healthcare Challenge 2 — ED Cost Forecasting (MAE)
Multimodal Specialist (PDF receipts + tabular) — LightGBM Quantile (Median) Regre
===========================================================
INSTALL (run once)
====:contentReference[oaicite:4]{index=4}==================================
pip install -U pandas numpy scipy scikit-learn lightgbm pymupdf joblib tqdm
Notes:
- This script avoids heavy OCR and uses fast text extraction with PyMuPDF.
- It parses CPT/HCPCS-like codes, quantities, and line totals from receipts to cr
- It trains MAE-aligned models using median (quantile=0.5) regression.
===========================================================

"""
from __future__ import annotations
import os
import re
import math
import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import fitz  # PyMuPDF
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb
from lightgbm import LGBMRegressor
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =========================
# CONFIG (EDIT THESE)
# =========================
BASE_DIR = Path(r"D:\AgentDs\agent_ds_healthcare")
TRAIN_CSV = BASE_DIR / "ed_cost_train.csv"
TEST_CSV = BASE_DIR / "ed_cost_test.csv"
PATIENTS_CSV = BASE_DIR / "patients.csv"          # optional (if present)
PDF_DIR = BASE_DIR / "receipts_pdf"               # receipt_{patient_id}.pdf
SUBMISSION_PATH = BASE_DIR / "submission_ICHI_V1.csv"
# Caching receipts is strongly recommended (parsing 4k PDFs repeatedly is wastefu
CACHE_PATH = BASE_DIR / "cache_receipts_multimodal_v1.joblib"
CACHE_VERSION = 1
# PDF parsing behavior
PARSE_WORKERS = 1  # keep 1 for maximum safety; increase if you want (read-only PD
PDF_Y_TOL = 1.5    # y-axis tolerance when clustering words into lines
# Feature blocks
USE_PATIENTS_TABLE_IF_AVAILABLE = True
USE_TOP_CODE_FEATURES = True
TOP_K_CODES = 250
USE_HASHED_CODE_FEATURES = True
HASH_DIM_COUNT = 2**12
HASH_DIM_SPEND = 2**13
HASH_DIM_QTY = 2**12
USE_TFIDF_SVD = True
TFIDF_MAX_FEATURES = 20000
TFIDF_NGRAM_RANGE = (1, 2)
SVD_COMPONENTS = 64
# Modeling / CV
N_FOLDS = 5
SEEDS = [2024, 2025, 2026]  # small bagging ensemble
EARLY_STOPPING_ROUNDS = 200
LGBM_BASE_PARAMS = dict(
    objective="quantile",
    alpha=0.5,                 # median regression -> MAE-aligned
    boosting_type="gbdt",

    learning_rate=0.03,
    n_estimators=6000,         # large upper bound; early stopping will pick best 
    num_leaves=64,
    min_child_samples=30,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    max_bin=255,
    n_jobs=-1,
    verbose=-1,
)
# Blend raw-target + log-target models (often robust on skewed costs)
TRAIN_BOTH_RAW_AND_LOG = True
# Final prediction post-processing
CLIP_PREDICTIONS_AT_ZERO = True
# =========================
# LOGGING
# =========================
def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
# =========================
# UTIL
# =========================
_money_re = re.compile(r"^\d{1,3}(?:,\d{3})*(?:\.\d{2})$|^\d+(?:\.\d{2})$")
_code_re = re.compile(r"^(?:[A-Z]\d{4}|\d{4,5})$")  # examples: G0378, 99284, 8502
def is_money(token: str) -> bool:
    return bool(_money_re.match(token))

def money_to_float(token: str) -> float:
    return float(token.replace(",", ""))
def is_code(token: str) -> bool:
    return bool(_code_re.match(token))
def safe_log1p(x: float) -> float:
    # avoids tiny negative due to float noise
    return math.log1p(max(0.0, float(x)))
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))
# =========================
# PDF PARSING
# =========================
def parse_header_fields(text: str) -> Tuple[Optional[int], Optional[str], Optiona
    """
    Extract patient_id, zip3, insurance from receipt header if present.
    """
    pid = None
    zip3 = None
    insurance = None
    m = re.search(r"Patient ID:\s*(\d+)", text)
    if m:
        pid = int(m.group(1))
    m = re.search(r"ZIP3:\s*([0-9]{3})\s+Insurance:\s*([A-Za-z_]+)", text)
    if m:
        zip3 = str(m.group(1))
        insurance = str(m.group(2)).lower()
    return pid, zip3, insurance
def cluster_words_into_lines(words: List[Tuple], y_tol: float = PDF_Y_TOL) -> List
    """
    Cluster PyMuPDF 'words' into lines by y coordinate, then sort by x coordinate

    Each word tuple: (x0, y0, x1, y1, word, block_no, line_no, word_no)
    Returns lines as list of tokens (strings) in reading order.
    """
    if not words:
        return []
    words_sorted = sorted(words, key=lambda w: (w[1], w[0]))  # sort by y0 then x
    lines: List[List[str]] = []
    current: List[Tuple[float, str]] = []
    current_y: Optional[float] = None
    for w in words_sorted:
        x0, y0, *_rest = w
        token = str(w[4])
        if current_y is None or abs(y0 - current_y) <= y_tol:
            current.append((float(x0), token))
            # smooth y to reduce micro-jitter
            current_y = y0 if current_y is None else (0.8 * current_y + 0.2 * y0)
        else:
            lines.append([t for _, t in sorted(current, key=lambda z: z[0])])
            current = [(float(x0), token)]
            current_y = y0
    if current:
        lines.append([t for _, t in sorted(current, key=lambda z: z[0])])
    return lines
def parse_line_items_from_words(words: List[Tuple]) -> Tuple[List[Dict[str, Any]]
    """
    Primary parser: uses word positions -> lines -> row parsing.
    Returns list of items: {code, desc, qty, unit, line_total}, and receipt total 
    """
    items: List[Dict[str, Any]] = []
    total: Optional[float] = None
    lines = cluster_words_into_lines(words, y_tol=PDF_Y_TOL)
    if not lines:

        return items, total
    # Find the header line (contains Code / Description / Qty)
    header_idx = None
    for i, toks in enumerate(lines):
        low = [t.lower() for t in toks]
        if ("code" in low) and ("description" in low) and ("qty" in low):
            header_idx = i
            break
    if header_idx is None:
        return items, total
    for toks in lines[header_idx + 1 :]:
        if not toks:
            continue
        # TOTAL line can appear as ["TOTAL", "7,978.28"] or similar
        if toks[0].upper() == "TOTAL":
            # find last money token
            for t in reversed(toks[1:]):
                if is_money(t):
                    total = money_to_float(t)
                    break
            break
        # Typical row pattern: [code, desc..., qty, unit, line_total]
        if len(toks) >= 5 and is_code(toks[0]) and toks[-3].isdigit() and is_money
            code = toks[0]
            qty = int(toks[-3])
            unit = money_to_float(toks[-2])
            line_total = money_to_float(toks[-1])
            desc = " ".join(toks[1:-3])
            items.append({"code": code, "desc": desc, "qty": qty, "unit": unit, "
    return items, total
def parse_line_items_fallback_text(text: str) -> Tuple[List[Dict[str, Any]], Opti
    """

    Fallback parser: works if PyMuPDF text extraction returns one cell per line i
    Example sequence after header:
        CODE
        DESC
        QTY
        UNIT
        LINE_TOTAL
        ...
        TOTAL
        50.00
    """
    items: List[Dict[str, Any]] = []
    total: Optional[float] = None
    lines = [ln.strip() for ln in (text.splitlines() if text else []) if ln.strip
    if not lines:
        return items, total
    # Find where the table header begins
    # Look for a line that is exactly "Code" and next lines contain "Description"
    start_idx = None
    for i in range(len(lines) - 4):
        if lines[i].lower() == "code" and lines[i + 1].lower() == "description" a
            start_idx = i + 5  # data begins after 5 header lines
            break
        # Alternate: "Code Description Qty Unit ($) Line Total ($)" in one line
        if "code" in lines[i].lower() and "description" in lines[i].lower() and "
            start_idx = i + 1
            break
    if start_idx is None:
        return items, total
    i = start_idx
    while i < len(lines):
        if lines[i].upper().startswith("TOTAL"):
            # TOTAL may be in same line ("TOTAL 7,978.28") or next line ("TOTAL" 
            parts = lines[i].split()
            if len(parts) >= 2 and is_money(parts[-1]):
                total = money_to_float(parts[-1])

            elif i + 1 < len(lines) and is_money(lines[i + 1]):
                total = money_to_float(lines[i + 1])
            break
        # Row-major 5-line pattern
        if i + 4 < len(lines) and is_code(lines[i]) and lines[i + 2].isdigit() an
            code = lines[i]
            desc = lines[i + 1]
            qty = int(lines[i + 2])
            unit = money_to_float(lines[i + 3])
            line_total = money_to_float(lines[i + 4])
            items.append({"code": code, "desc": desc, "qty": qty, "unit": unit, "
            i += 5
        else:
            # If not matching, advance by 1 to re-sync
            i += 1
    return items, total
@dataclass
class ReceiptEntry:
    meta: Dict[str, Any]
    tok_count: Dict[str, float]
    tok_spend: Dict[str, float]
    tok_qty: Dict[str, float]
    doc_text: str
    items: List[Dict[str, Any]]
def featurize_receipt(patient_id: int, pdf_dir: Path) -> ReceiptEntry:
    """
    Read receipt_{patient_id}.pdf and produce:
      - meta numeric/categorical features
      - hashed-token dicts (count/spend/qty) for long-tail codes
      - doc_text for TF-IDF
      - items list for explicit top-code pivots
    """
    pdf_path = pdf_dir / f"receipt_{patient_id}.pdf"

    # Defaults for missing / parse failure
    empty_meta = dict(
        pdf_ok=0,
        pdf_n_lines=0,
        pdf_n_unique_codes=0,
        pdf_sum_qty=0.0,
        pdf_mean_qty=0.0,
        pdf_max_qty=0.0,
        pdf_sum_line_total=0.0,
        pdf_mean_line_total=0.0,
        pdf_max_line_total=0.0,
        pdf_std_line_total=0.0,
        pdf_top1_cost_share=0.0,
        pdf_top3_cost_share=0.0,
        pdf_repeat_cost_share=0.0,
        pdf_sum_unit=0.0,
        pdf_mean_unit=0.0,
        pdf_max_unit=0.0,
        pdf_std_unit=0.0,
        pdf_total=0.0,
        pdf_total_diff=0.0,
        pdf_insurance=None,
        pdf_zip3=None,
        pdf_parse_notes="missing_pdf",
    )
    if not pdf_path.exists():
        return ReceiptEntry(meta=empty_meta, tok_count={}, tok_spend={}, tok_qty=
    try:
        with fitz.open(pdf_path) as doc:
            if doc.page_count < 1:
                empty_meta["pdf_parse_notes"] = "empty_pdf"
                return ReceiptEntry(meta=empty_meta, tok_count={}, tok_spend={}, 
            page = doc[0]
            text = page.get_text("text") or ""
            words = page.get_text("words") or []
    except Exception as e:

        empty_meta["pdf_parse_notes"] = f"open_error:{type(e).__name__}"
        return ReceiptEntry(meta=empty_meta, tok_count={}, tok_spend={}, tok_qty=
    header_pid, zip3, insurance = parse_header_fields(text)
    # Primary parse via words; fallback via text
    items, total = parse_line_items_from_words(words)
    if not items:
        items, total = parse_line_items_fallback_text(text)
    if not items:
        empty_meta["pdf_parse_notes"] = "no_items_parsed"
        empty_meta["pdf_zip3"] = zip3
        empty_meta["pdf_insurance"] = insurance
        return ReceiptEntry(meta=empty_meta, tok_count={}, tok_spend={}, tok_qty=
    # Aggregate stats
    line_totals = np.array([float(it["line_total"]) for it in items], dtype=float
    qtys = np.array([float(it["qty"]) for it in items], dtype=float)
    units = np.array([float(it["unit"]) for it in items], dtype=float)
    sum_line_total = float(line_totals.sum())
    pdf_total = float(total) if total is not None else sum_line_total
    codes = [str(it["code"]) for it in items]
    n_lines = len(items)
    n_unique = len(set(codes))
    sorted_cost = np.sort(line_totals)[::-1]
    top1 = float(sorted_cost[0] / sum_line_total) if sum_line_total > 0 else 0.0
    top3 = float(sorted_cost[:3].sum() / sum_line_total) if sum_line_total > 0 el
    # repeat spend share
    spend_by_code: Dict[str, float] = {}
    cnt_by_code: Dict[str, int] = {}
    for it in items:
        c = str(it["code"])
        spend_by_code[c] = spend_by_code.get(c, 0.0) + float(it["line_total"])
        cnt_by_code[c] = cnt_by_code.get(c, 0) + 1
    repeat_spend = sum(v for c, v in spend_by_code.items() if cnt_by_code.get(c, 

    repeat_share = float(repeat_spend / sum_line_total) if sum_line_total > 0 els
    meta = dict(
        pdf_ok=1,
        pdf_n_lines=n_lines,
        pdf_n_unique_codes=n_unique,
        pdf_sum_qty=float(qtys.sum()),
        pdf_mean_qty=float(qtys.mean()) if n_lines else 0.0,
        pdf_max_qty=float(qtys.max()) if n_lines else 0.0,
        pdf_sum_line_total=sum_line_total,
        pdf_mean_line_total=float(line_totals.mean()) if n_lines else 0.0,
        pdf_max_line_total=float(line_totals.max()) if n_lines else 0.0,
        pdf_std_line_total=float(line_totals.std()) if n_lines else 0.0,
        pdf_top1_cost_share=top1,
        pdf_top3_cost_share=top3,
        pdf_repeat_cost_share=repeat_share,
        pdf_sum_unit=float(units.sum()),
        pdf_mean_unit=float(units.mean()) if n_lines else 0.0,
        pdf_max_unit=float(units.max()) if n_lines else 0.0,
        pdf_std_unit=float(units.std()) if n_lines else 0.0,
        pdf_total=pdf_total,
        pdf_total_diff=float(abs(sum_line_total - pdf_total)),
        pdf_insurance=(insurance.lower() if isinstance(insurance, str) else None)
        pdf_zip3=(str(zip3) if zip3 is not None else None),
        pdf_parse_notes="ok",
    )
    # Token dicts for hashing (long-tail code signals)
    tok_count: Dict[str, float] = {}
    tok_spend: Dict[str, float] = {}
    tok_qty: Dict[str, float] = {}
    doc_tokens: List[str] = []
    for it in items:
        code = str(it["code"])
        qty = float(it["qty"])
        spend = float(it["line_total"])
        # tokens: explicit code + prefixes + type

        toks = [
            f"code={code}",
            f"p1={code[0]}",
            f"p2={code[:2]}",
            f"p3={code[:3]}",
            f"type={'hcpcs' if code[0].isalpha() else 'cpt'}",
        ]
        for t in toks:
            tok_count[t] = tok_count.get(t, 0.0) + 1.0
            tok_qty[t] = tok_qty.get(t, 0.0) + qty
            tok_spend[t] = tok_spend.get(t, 0.0) + spend
        # doc text: include code tokens + cleaned description words
        doc_tokens.append(f"code_{code}")
        desc = str(it.get("desc", ""))
        desc_clean = re.sub(r"[^a-z0-9]+", " ", desc.lower()).strip()
        if desc_clean:
            doc_tokens.extend(desc_clean.split())
    # compress values (helps stability with trees + hashing)
    tok_count = {k: safe_log1p(v) for k, v in tok_count.items()}
    tok_qty = {k: safe_log1p(v) for k, v in tok_qty.items()}
    tok_spend = {k: safe_log1p(v) for k, v in tok_spend.items()}
    doc_text = " ".join(doc_tokens)
    return ReceiptEntry(meta=meta, tok_count=tok_count, tok_spend=tok_spend, tok_
def load_receipt_cache(cache_path: Path) -> Dict[str, Any]:
    if cache_path.exists():
        try:
            obj = joblib.load(cache_path)
            if isinstance(obj, dict) and obj.get("version") == CACHE_VERSION and 
                return obj
        except Exception:
            pass
    return {"version": CACHE_VERSION, "data": {}}

def save_receipt_cache(cache_path: Path, cache_obj: Dict[str, Any]) -> None:
    try:
        joblib.dump(cache_obj, cache_path)
    except Exception as e:
        logging.warning(f"Failed to save cache to {cache_path}: {e}")
def build_receipt_artifacts(patient_ids: List[int], pdf_dir: Path, cache_path: Pa
    """
    Returns:
      - receipt_meta_df: one row per patient_id with meta features + pdf_zip3/pdf_
      - tok_count_map: patient_id -> token-count dict
      - tok_spend_map: patient_id -> token-spend dict
      - tok_qty_map: patient_id -> token-qty dict
      - doc_text_map: patient_id -> doc_text
      - line_items_df: long table of (patient_id, code, qty, unit, line_total, de
    """
    cache_obj = load_receipt_cache(cache_path)
    cache_data: Dict[str, Any] = cache_obj.get("data", {})
    # Parse missing entries
    to_parse = [pid for pid in patient_ids if str(pid) not in cache_data]
    if to_parse:
        logging.info(f"Parsing {len(to_parse):,} receipts (cache miss).")
    else:
        logging.info("All receipts loaded from cache.")
    for pid in tqdm(to_parse, desc="Parsing receipts", ncols=100):
        entry = featurize_receipt(pid, pdf_dir)
        cache_data[str(pid)] = {
            "meta": entry.meta,
            "tok_count": entry.tok_count,
            "tok_spend": entry.tok_spend,
            "tok_qty": entry.tok_qty,
            "doc_text": entry.doc_text,
            "items": entry.items,
        }
    cache_obj["data"] = cache_data
    if to_parse:

        save_receipt_cache(cache_path, cache_obj)
    # Build artifacts
    meta_rows: List[Dict[str, Any]] = []
    tok_count_map: Dict[int, Dict[str, float]] = {}
    tok_spend_map: Dict[int, Dict[str, float]] = {}
    tok_qty_map: Dict[int, Dict[str, float]] = {}
    doc_text_map: Dict[int, str] = {}
    item_rows: List[Dict[str, Any]] = []
    for pid in patient_ids:
        obj = cache_data.get(str(pid), None)
        if obj is None:
            entry = featurize_receipt(pid, pdf_dir)
            obj = {
                "meta": entry.meta,
                "tok_count": entry.tok_count,
                "tok_spend": entry.tok_spend,
                "tok_qty": entry.tok_qty,
                "doc_text": entry.doc_text,
                "items": entry.items,
            }
        meta = dict(patient_id=pid, **obj["meta"])
        meta_rows.append(meta)
        tok_count_map[pid] = obj.get("tok_count", {}) or {}
        tok_spend_map[pid] = obj.get("tok_spend", {}) or {}
        tok_qty_map[pid] = obj.get("tok_qty", {}) or {}
        doc_text_map[pid] = obj.get("doc_text", "") or ""
        for it in (obj.get("items", []) or []):
            item_rows.append(
                {
                    "patient_id": pid,
                    "code": str(it.get("code", "")),
                    "qty": float(it.get("qty", 0.0)),
                    "unit": float(it.get("unit", 0.0)),
                    "line_total": float(it.get("line_total", 0.0)),
                    "desc": str(it.get("desc", "")),

                }
            )
    receipt_meta_df = pd.DataFrame(meta_rows)
    line_items_df = pd.DataFrame(item_rows)
    return receipt_meta_df, tok_count_map, tok_spend_map, tok_qty_map, doc_text_m
# =========================
# FEATURE ENGINEERING
# =========================
def make_ohe() -> OneHotEncoder:
    """
    Compatible across sklearn versions:
      - newer: sparse_output
      - older: sparse
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_out
    except TypeError:
        try:
            return OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=True)
def add_top_code_features(
    df_all: pd.DataFrame,
    line_items_df: pd.DataFrame,
    train_patient_ids: np.ndarray,
    top_k: int = TOP_K_CODES,
) -> pd.DataFrame:
    """
    Adds explicit (non-hashed) CPT/HCPCS composition features:
      - cpt_cnt_{code}
      - cpt_share_{code}  (spend share relative to prior_ed_cost_5y_usd)
    Uses top codes by frequency in TRAIN receipts only.
    """
    if line_items_df.empty:

        logging.warning("line_items_df is empty: skipping top-code features.")
        return df_all
    train_items = line_items_df[line_items_df["patient_id"].isin(train_patient_id
    vc = train_items["code"].value_counts()
    top_codes = vc.head(top_k).index.tolist()
    if not top_codes:
        logging.warning("No codes found in receipts: skipping top-code features."
        return df_all
    logging.info(f"Adding top-code features for TOP_K_CODES={len(top_codes)}")
    # Aggregate per patient+code
    agg = (
        line_items_df[line_items_df["code"].isin(top_codes)]
        .groupby(["patient_id", "code"], as_index=False)
        .agg(cnt=("code", "size"), spend=("line_total", "sum"))
    )
    cnt_wide = agg.pivot(index="patient_id", columns="code", values="cnt").fillna
    spend_wide = agg.pivot(index="patient_id", columns="code", values="spend").fi
    # Compute spend share relative to prior cost
    denom = df_all.set_index("patient_id")["prior_ed_cost_5y_usd"].astype(float)
    # Align denom index to spend_wide rows
    spend_share = spend_wide.div(denom.reindex(spend_wide.index).values.reshape(-
    cnt_wide.columns = [f"cpt_cnt_{c}" for c in cnt_wide.columns]
    spend_share.columns = [f"cpt_share_{c}" for c in spend_share.columns]
    # Merge
    df_all = df_all.merge(cnt_wide.reset_index(), on="patient_id", how="left")
    df_all = df_all.merge(spend_share.reset_index(), on="patient_id", how="left")
    # Missing -> 0
    new_cols = [c for c in df_all.columns if c.startswith("cpt_cnt_") or c.startsw
    df_all[new_cols] = df_all[new_cols].fillna(0.0)
    return df_all

def build_tfidf_svd(doc_texts: List[str], max_features: int, ngram_range: Tuple[i
    """
    Fit TF-IDF on all docs (train+test) and then SVD to get dense embedding featu
    (Unsupervised on test is usually acceptable in competitions.)
    """
    vectorizer = TfidfVectorizer(
        min_df=2,
        max_features=max_features,
        ngram_range=ngram_range,
        token_pattern=r"(?u)\b\w+\b",
    )
    X_tfidf = vectorizer.fit_transform(doc_texts)
    if X_tfidf.shape[1] <= 2:
        # not enough features for SVD
        X_dense = X_tfidf.toarray().astype(np.float32)
        return X_dense, vectorizer, None
    k = int(min(n_components, X_tfidf.shape[1] - 1))
    svd = TruncatedSVD(n_components=k, random_state=seed)
    X_svd = svd.fit_transform(X_tfidf).astype(np.float32)
    return X_svd, vectorizer, svd
def build_feature_matrix(
    df_all: pd.DataFrame,
    tok_count_map: Dict[int, Dict[str, float]],
    tok_spend_map: Dict[int, Dict[str, float]],
    tok_qty_map: Dict[int, Dict[str, float]],
    doc_text_map: Dict[int, str],
) -> Tuple[csr_matrix, Dict[str, Any]]:
    """
    Build a combined sparse feature matrix:
      - tabular numeric + categorical (OHE)
      - hashed long-tail code dicts
      - optional TF-IDF+SVD dense embeddings
    """
    df_all = df_all.copy()

    # Basic derived features (tabular)
    df_all["prior_cost_per_visit"] = df_all["prior_ed_cost_5y_usd"] / (df_all["pr
    df_all["log_prior_cost"] = np.log1p(df_all["prior_ed_cost_5y_usd"].astype(flo
    df_all["log_prior_visits"] = np.log1p(df_all["prior_ed_visits_5y"].astype(flo
    # Normalize pdf categoricals
    if "pdf_insurance" in df_all.columns:
        df_all["pdf_insurance"] = df_all["pdf_insurance"].fillna("missing").astyp
    if "pdf_zip3" in df_all.columns:
        df_all["pdf_zip3"] = df_all["pdf_zip3"].fillna("missing").astype(str)
    # Decide categorical columns present
    categorical_cols = []
    for c in ["primary_chronic", "sex", "insurance", "zip3", "pdf_insurance", "pd
        if c in df_all.columns:
            categorical_cols.append(c)
    # Numeric columns: everything numeric excluding ids/targets; keep receipt met
    exclude = {"patient_id", "ed_cost_next3y_usd", "is_train"}
    numeric_cols = [
        c for c in df_all.columns
        if (c not in exclude)
        and (c not in categorical_cols)
        and (pd.api.types.is_numeric_dtype(df_all[c]))
    ]
    # Tabular preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), nu
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")
        ],
        remainder="drop",
        sparse_threshold=0.5,
    )
    X_tab = preprocessor.fit_transform(df_all)
    if not hasattr(X_tab, "tocsr"):
        X_tab = csr_matrix(X_tab)

    else:
        X_tab = X_tab.tocsr()
    blocks = [X_tab]
    artifacts: Dict[str, Any] = {"preprocessor": preprocessor, "numeric_cols": nu
    # Hashed code features
    if USE_HASHED_CODE_FEATURES:
        ids = df_all["patient_id"].astype(int).tolist()
        tok_counts = [tok_count_map.get(pid, {}) for pid in ids]
        tok_spends = [tok_spend_map.get(pid, {}) for pid in ids]
        tok_qtys = [tok_qty_map.get(pid, {}) for pid in ids]
        hasher_count = FeatureHasher(n_features=HASH_DIM_COUNT, input_type="dict"
        hasher_spend = FeatureHasher(n_features=HASH_DIM_SPEND, input_type="dict"
        hasher_qty = FeatureHasher(n_features=HASH_DIM_QTY, input_type="dict", dty
        X_count = hasher_count.transform(tok_counts)
        X_spend = hasher_spend.transform(tok_spends)
        X_qty = hasher_qty.transform(tok_qtys)
        blocks.extend([X_count, X_spend, X_qty])
        artifacts.update({"hasher_count": hasher_count, "hasher_spend": hasher_sp
    # TF-IDF + SVD embeddings
    if USE_TFIDF_SVD:
        ids = df_all["patient_id"].astype(int).tolist()
        doc_texts = [doc_text_map.get(pid, "") for pid in ids]
        X_svd, vectorizer, svd = build_tfidf_svd(
            doc_texts=doc_texts,
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            n_components=SVD_COMPONENTS,
            seed=SEEDS[0],
        )
        blocks.append(csr_matrix(X_svd))
        artifacts.update({"tfidf": vectorizer, "svd": svd, "svd_dim": X_svd.shape
    X = hstack(blocks, format="csr")
    artifacts["n_features"] = X.shape[1]

    return X, artifacts
# =========================
# MODELING
# =========================
def mae_original_from_log(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> Tupl
    """
    Custom eval metric for log1p(target) training:
      - y_true_log, y_pred_log are log1p values
      - compute MAE in original USD scale
    """
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return "mae_usd", float(np.mean(np.abs(y_true - y_pred))), False
def make_strat_labels(train_df: pd.DataFrame) -> np.ndarray:
    """
    Stratify by (primary_chronic + prior_cost decile) to keep severity mix stable 
    """
    chronic = train_df["primary_chronic"].astype(str)
    # qcut returns categorical bins; duplicates drop if ties
    cost_bin = pd.qcut(train_df["prior_ed_cost_5y_usd"], q=10, duplicates="drop")
    return (chronic + "_" + cost_bin).values
def cv_oof_and_best_iters(
    X_train: csr_matrix,
    y: np.ndarray,
    strat_labels: np.ndarray,
    params: Dict[str, Any],
    seeds: List[int],
    use_log_target: bool,
) -> Tuple[np.ndarray, List[int], float]:
    """
    CV loop to:
      - produce out-of-fold predictions (OOF) for MAE estimation
      - collect best_iteration_ for later full-data fitting
    """

    oof_preds_per_seed: List[np.ndarray] = []
    best_iters: List[int] = []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        oof_seed = np.zeros(len(y), dtype=np.float32)
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, strat_labels))
            X_tr, X_va = X_train[tr_idx], X_train[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]
            model_params = dict(params)
            model_params["random_state"] = seed * 100 + fold
            model = LGBMRegressor(**model_params)
            if use_log_target:
                y_tr_fit = np.log1p(y_tr)
                y_va_fit = np.log1p(y_va)
                model.fit(
                    X_tr, y_tr_fit,
                    eval_set=[(X_va, y_va_fit)],
                    eval_metric=mae_original_from_log,
                    callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=
                )
                best_it = int(getattr(model, "best_iteration_", model_params["n_e
                pred_va = np.expm1(model.predict(X_va, num_iteration=best_it))
            else:
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    eval_metric="mae",
                    callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=
                )
                best_it = int(getattr(model, "best_iteration_", model_params["n_e
                pred_va = model.predict(X_va, num_iteration=best_it)
            if CLIP_PREDICTIONS_AT_ZERO:
                pred_va = np.maximum(pred_va, 0.0)

            oof_seed[va_idx] = pred_va.astype(np.float32)
            best_iters.append(best_it)
        oof_preds_per_seed.append(oof_seed)
    # MAE likes medians; median across seeds tends to be robust
    oof = np.median(np.vstack(oof_preds_per_seed), axis=0)
    score = mae(y, oof)
    return oof, best_iters, score
def fit_full_and_predict(
    X_train: csr_matrix,
    y: np.ndarray,
    X_test: csr_matrix,
    params: Dict[str, Any],
    seeds: List[int],
    n_estimators: int,
    use_log_target: bool,
) -> np.ndarray:
    """
    Fit on full training data for each seed, then take median prediction across s
    """
    preds_seed: List[np.ndarray] = []
    for seed in seeds:
        model_params = dict(params)
        model_params["random_state"] = seed
        model_params["n_estimators"] = int(n_estimators)
        model = LGBMRegressor(**model_params)
        if use_log_target:
            y_fit = np.log1p(y)
            model.fit(X_train, y_fit)
            pred = np.expm1(model.predict(X_test))
        else:
            model.fit(X_train, y)
            pred = model.predict(X_test)
        if CLIP_PREDICTIONS_AT_ZERO:

            pred = np.maximum(pred, 0.0)
        preds_seed.append(pred.astype(np.float32))
    return np.median(np.vstack(preds_seed), axis=0)
def optimize_blend_weight(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndar
    """
    Find w in [0,1] minimizing MAE of blend = w*pred_a + (1-w)*pred_b.
    Simple grid search is fine for 2k rows.
    """
    best_w = 0.5
    best_mae = float("inf")
    for w in np.linspace(0, 1, 101):
        blend = w * pred_a + (1.0 - w) * pred_b
        m = mae(y_true, blend)
        if m < best_mae:
            best_mae = m
            best_w = float(w)
    return best_w, best_mae
# =========================
# MAIN
# =========================
def main() -> None:
    setup_logging()
    logging.info("=== AgentDS Challenge 2: ED Cost Forecasting (Multimodal Specia
    # Load train/test
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"Train CSV not found: {TRAIN_CSV}")
    if not TEST_CSV.exists():
        raise FileNotFoundError(f"Test CSV not found: {TEST_CSV}")
    if not PDF_DIR.exists():
        raise FileNotFoundError(f"PDF directory not found: {PDF_DIR}")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    required_train_cols = {"patient_id", "primary_chronic", "prior_ed_visits_5y", 
    required_test_cols = {"patient_id", "primary_chronic", "prior_ed_visits_5y", 
    if not required_train_cols.issubset(train_df.columns):
        raise ValueError(f"Train CSV missing columns: {required_train_cols - set(t
    if not required_test_cols.issubset(test_df.columns):
        raise ValueError(f"Test CSV missing columns: {required_test_cols - set(te
    logging.info(f"Train shape: {train_df.shape} | Test shape: {test_df.shape}")
    # Optional patients.csv
    if USE_PATIENTS_TABLE_IF_AVAILABLE and PATIENTS_CSV.exists():
        patients = pd.read_csv(PATIENTS_CSV)
        if "patient_id" in patients.columns:
            logging.info("Merging patients.csv features (age/sex/insurance/zip3).
            train_df = train_df.merge(patients, on="patient_id", how="left")
            test_df = test_df.merge(patients, on="patient_id", how="left")
        else:
            logging.warning("patients.csv exists but has no patient_id column; sk
    else:
        logging.info("patients.csv not used (not found or disabled). Will rely on 
    # Build receipt cache/artifacts for all patients
    all_patient_ids = pd.concat([train_df["patient_id"], test_df["patient_id"]]).
    all_patient_ids.sort()
    receipt_meta_df, tok_count_map, tok_spend_map, tok_qty_map, doc_text_map, lin
        patient_ids=all_patient_ids,
        pdf_dir=PDF_DIR,
        cache_path=CACHE_PATH,
    )
    # Merge receipt meta
    train_df = train_df.merge(receipt_meta_df, on="patient_id", how="left")
    test_df = test_df.merge(receipt_meta_df, on="patient_id", how="left")
    # If patients.csv missing zip3/insurance, receipt header might provide pdf_zip
    # (We keep both; model will decide. If both exist, they're usually identical.
    for col in ["pdf_zip3", "pdf_insurance"]:
        if col in train_df.columns:

            train_df[col] = train_df[col].astype("string")
            test_df[col] = test_df[col].astype("string")
    # Combine train+test for feature building
    train_df["is_train"] = 1
    test_df["is_train"] = 0
    df_all = pd.concat([train_df, test_df], ignore_index=True)
    # Add explicit top-code features (counts + spend shares)
    if USE_TOP_CODE_FEATURES:
        df_all = add_top_code_features(
            df_all=df_all,
            line_items_df=line_items_df,
            train_patient_ids=train_df["patient_id"].astype(int).values,
            top_k=TOP_K_CODES,
        )
    # Build combined feature matrix
    logging.info("Building feature matrix...")
    X_all, artifacts = build_feature_matrix(
        df_all=df_all,
        tok_count_map=tok_count_map,
        tok_spend_map=tok_spend_map,
        tok_qty_map=tok_qty_map,
        doc_text_map=doc_text_map,
    )
    logging.info(f"Feature matrix built: shape={X_all.shape} | n_features={artifa
    # Split back
    is_train_mask = df_all["is_train"].values.astype(int) == 1
    X_train = X_all[is_train_mask]
    X_test = X_all[~is_train_mask]
    y = train_df["ed_cost_next3y_usd"].values.astype(np.float32)
    # Strat labels for CV
    strat_labels = make_strat_labels(train_df)
    # CV OOF + best iterations
    logging.info("Running CV (raw target)...")
    oof_raw, best_iters_raw, mae_raw = cv_oof_and_best_iters(

        X_train=X_train,
        y=y,
        strat_labels=strat_labels,
        params=LGBM_BASE_PARAMS,
        seeds=SEEDS,
        use_log_target=False,
    )
    logging.info(f"OOF MAE (raw target): {mae_raw:.4f}")
    if TRAIN_BOTH_RAW_AND_LOG:
        logging.info("Running CV (log1p target)...")
        oof_log, best_iters_log, mae_log = cv_oof_and_best_iters(
            X_train=X_train,
            y=y,
            strat_labels=strat_labels,
            params=LGBM_BASE_PARAMS,
            seeds=SEEDS,
            use_log_target=True,
        )
        logging.info(f"OOF MAE (log1p target, evaluated in USD): {mae_log:.4f}")
        # Blend OOF
        w, mae_blend = optimize_blend_weight(y_true=y, pred_a=oof_raw, pred_b=oof_
        oof_blend = w * oof_raw + (1.0 - w) * oof_log
        # Bias calibration (MAE-optimal constant shift is median residual)
        bias = float(np.median(y - oof_blend))
        oof_blend_cal = oof_blend + bias
        mae_blend_cal = mae(y, oof_blend_cal)
        logging.info(f"Blend weight w(raw)={w:.2f} | OOF MAE (blend): {mae_blend:
        # Full-data training iterations
        n_est_raw = int(np.median(best_iters_raw)) if best_iters_raw else int(LGBM
        n_est_log = int(np.median(best_iters_log)) if best_iters_log else int(LGBM
        logging.info(f"Full fit n_estimators: raw={n_est_raw} | log={n_est_log}")
        # Fit full and predict test
        logging.info("Fitting full-data models and predicting test (raw)...")
        test_pred_raw = fit_full_and_predict(

            X_train=X_train, y=y, X_test=X_test,
            params=LGBM_BASE_PARAMS, seeds=SEEDS,
            n_estimators=n_est_raw, use_log_target=False,
        )
        logging.info("Fitting full-data models and predicting test (log1p)...")
        test_pred_log = fit_full_and_predict(
            X_train=X_train, y=y, X_test=X_test,
            params=LGBM_BASE_PARAMS, seeds=SEEDS,
            n_estimators=n_est_log, use_log_target=True,
        )
        test_pred = w * test_pred_raw + (1.0 - w) * test_pred_log + bias
    else:
        # Use raw only
        bias = float(np.median(y - oof_raw))
        logging.info(f"Bias calibration (raw-only): bias={bias:.4f}")
        n_est_raw = int(np.median(best_iters_raw)) if best_iters_raw else int(LGBM
        logging.info(f"Full fit n_estimators (raw-only): {n_est_raw}")
        logging.info("Fitting full-data models and predicting test (raw-only)..."
        test_pred_raw = fit_full_and_predict(
            X_train=X_train, y=y, X_test=X_test,
            params=LGBM_BASE_PARAMS, seeds=SEEDS,
            n_estimators=n_est_raw, use_log_target=False,
        )
        test_pred = test_pred_raw + bias
    if CLIP_PREDICTIONS_AT_ZERO:
        test_pred = np.maximum(test_pred, 0.0)
    # Write submission
    submission = pd.DataFrame(
        {
            "patient_id": test_df["patient_id"].values,
            "ed_cost_next3y_usd": test_pred.astype(np.float32),
        }
    )

If you want a quick “speed mode” (for iteration), set:
USE_TFIDF_SVD = False
USE_TOP_CODE_FEATURES = False
SEEDS = [2024]
N_FOLDS = 3
and you’ll still keep the hashed CPT long-tail + receipt aggregates, which are usually
the biggest lift.
    # Ensure exact patient_id order as test
    submission["patient_id"] = submission["patient_id"].astype(int)
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    logging.info(f"Saved submission to: {SUBMISSION_PATH}")
    logging.info("Done.")
if __name__ == "__main__":
    main()
```

### C02 — ITERATION 2 (Level-2 Upgrade) — Why this should beat ~493.9 MAE (pages 48-81)
```python
# -*- coding: utf-8 -*-
"""
ITERATION 2 (Level-2 Upgrade) — Why this should beat ~493.9 MAE
===============================================================
Hypothesis:
1) The missing 10% accuracy is primarily from NOT learning "how risky each CPT/HC
   Sparse hashing / TF-IDF can be too noisy/weak on ~2k rows.
2) We add CROSS-FITTED TARGET ENCODING (CV-safe) for:
     - exact CPT/HCPCS codes
     - 2-char code prefixes (shares signal for rare codes)
   Then aggregate per patient with spend-weights (from the receipt line totals).
3) We train an MAE-aligned ensemble:
     - LightGBM regression_l1 (direct MAE)
     - LightGBM quantile alpha=0.5 on log1p(y) (median / MAE-aligned under monoto
     - XGBoost reg:absoluteerror (L1)
     - CatBoost MAE on dense features (handles categorical well)
   Finally we blend via OOF MAE-optimized weights + median-residual bias correcti
This typically closes large gaps when the key information is high-cardinality "co
===========================================================
INSTALL (run once)
===========================================================

pip install -U pandas numpy scipy scikit-learn lightgbm xgboost catboost pymupdf 
===========================================================
Paths expected on your machine:
D:\AgentDs\agent_ds_healthcare\ed_cost_train.csv
D:\AgentDs\agent_ds_healthcare\ed_cost_test.csv
D:\AgentDs\agent_ds_healthcare\patients.csv          (optional but recommended)
D:\AgentDs\agent_ds_healthcare\receipts_pdf\receipt_{patient_id}.pdf
Output:
D:\AgentDs\agent_ds_healthcare\submission.csv
===========================================================
"""
from __future__ import annotations
import os
import re
import math
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Iterable
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import fitz  # PyMuPDF
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.feature_extraction import FeatureHasher
import lightgbm as lgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor, Pool
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# =========================
# CONFIG
# =========================
BASE_DIR = Path(r"D:\AgentDs\agent_ds_healthcare")
TRAIN_CSV = BASE_DIR / "ed_cost_train.csv"
TEST_CSV = BASE_DIR / "ed_cost_test.csv"
PATIENTS_CSV = BASE_DIR / "patients.csv"          # optional
PDF_DIR = BASE_DIR / "receipts_pdf"
SUBMISSION_PATH = BASE_DIR / "submission.csv"
CACHE_PATH = BASE_DIR / "cache_receipts_iter2.joblib"
CACHE_VERSION = 2
# CV
N_FOLDS = 5
REPEAT_SEEDS = [2024, 2025, 2026]   # 3 repeats => 15 folds total (rigorous + sta
EARLY_STOPPING = 250
# Target encoding hyperparams
TE_SMOOTH = 20.0          # smoothing for mean(log y) encoding
TE_MIN_COUNT_MEDIAN = 5   # below this, median encoding falls back to global medi
TE_FEATURE_PREFIX = "te_"
# Top-code pivot features (non-target)
USE_TOP_CODE_PIVOTS = True
TOP_K_CODES = 180

# Hashed long-tail code dict features (non-target)
USE_HASHED_DICT_FEATURES = True
HASH_DIM_COUNT = 2**12
HASH_DIM_SPEND = 2**12
HASH_DIM_QTY = 2**11
# Modeling toggles (you can switch off models for speed ablation)
USE_LGB_L1 = True
USE_LGB_QLOG = True
USE_XGB_L1 = True
USE_CAT_MAE = True
# CatBoost: use GPU if available
CAT_TASK_TYPE = "GPU"  # change to "CPU" if GPU drivers are problematic on your s
# Prediction post-process
CLIP_AT_ZERO = True
# =========================
# LOGGING
# =========================
def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
# =========================
# HELPERS
# =========================
_money_re = re.compile(r"^\d{1,3}(?:,\d{3})*(?:\.\d{2})$|^\d+(?:\.\d{2})$")
_code_re = re.compile(r"^(?:[A-Z]\d{4}|\d{4,5})$")  # e.g., G0378, 99284, 85025
def is_money(tok: str) -> bool:
    return bool(_money_re.match(tok))
def money_to_float(tok: str) -> float:
    return float(tok.replace(",", ""))

def is_code(tok: str) -> bool:
    return bool(_code_re.match(tok))
def safe_log1p(x: float) -> float:
    return math.log1p(max(0.0, float(x)))
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))
def make_ohe() -> OneHotEncoder:
    # sklearn compat: sparse_output vs sparse
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_out
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)
# =========================
# PDF PARSING
# =========================
def cluster_words_into_lines(words: List[Tuple], y_tol: float = 1.5) -> List[List
    """
    Cluster PyMuPDF 'words' into lines using y coordinate, then sort by x.
    """
    if not words:
        return []
    words_sorted = sorted(words, key=lambda w: (w[1], w[0]))  # y0 then x0
    lines: List[List[str]] = []
    cur: List[Tuple[float, str]] = []
    cur_y: Optional[float] = None
    for w in words_sorted:
        x0, y0 = float(w[0]), float(w[1])
        tok = str(w[4])
        if cur_y is None or abs(y0 - cur_y) <= y_tol:
            cur.append((x0, tok))
            cur_y = y0 if cur_y is None else (0.85 * cur_y + 0.15 * y0)

        else:
            lines.append([t for _, t in sorted(cur, key=lambda z: z[0])])
            cur = [(x0, tok)]
            cur_y = y0
    if cur:
        lines.append([t for _, t in sorted(cur, key=lambda z: z[0])])
    return lines
def parse_header_fields(text: str) -> Tuple[Optional[int], Optional[str], Optiona
    pid = None
    zip3 = None
    insurance = None
    m = re.search(r"Patient ID:\s*(\d+)", text)
    if m:
        pid = int(m.group(1))
    m = re.search(r"ZIP3:\s*([0-9]{3})\s+Insurance:\s*([A-Za-z_]+)", text)
    if m:
        zip3 = str(m.group(1))
        insurance = str(m.group(2)).lower()
    return pid, zip3, insurance
def parse_items_from_words(words: List[Tuple]) -> Tuple[List[Dict[str, Any]], Opt
    items: List[Dict[str, Any]] = []
    total: Optional[float] = None
    lines = cluster_words_into_lines(words, y_tol=1.5)
    if not lines:
        return items, total
    header_idx = None
    for i, toks in enumerate(lines):
        low = [t.lower() for t in toks]
        if ("code" in low) and ("description" in low) and ("qty" in low):

            header_idx = i
            break
    if header_idx is None:
        return items, total
    for toks in lines[header_idx + 1 :]:
        if not toks:
            continue
        if toks[0].upper() == "TOTAL":
            for t in reversed(toks[1:]):
                if is_money(t):
                    total = money_to_float(t)
                    break
            break
        if len(toks) >= 5 and is_code(toks[0]) and toks[-3].isdigit() and is_money
            code = toks[0]
            qty = int(toks[-3])
            unit = money_to_float(toks[-2])
            line_total = money_to_float(toks[-1])
            desc = " ".join(toks[1:-3])
            items.append({"code": code, "qty": qty, "unit": unit, "line_total": l
    return items, total
def parse_items_fallback_text(text: str) -> Tuple[List[Dict[str, Any]], Optional[
    items: List[Dict[str, Any]] = []
    total: Optional[float] = None
    lines = [ln.strip() for ln in (text.splitlines() if text else []) if ln.strip
    if not lines:
        return items, total
    start_idx = None
    for i in range(len(lines) - 4):
        if lines[i].lower() == "code" and lines[i + 1].lower() == "description" a
            start_idx = i + 5
            break

        if "code" in lines[i].lower() and "description" in lines[i].lower() and "
            start_idx = i + 1
            break
    if start_idx is None:
        return items, total
    i = start_idx
    while i < len(lines):
        if lines[i].upper().startswith("TOTAL"):
            parts = lines[i].split()
            if len(parts) >= 2 and is_money(parts[-1]):
                total = money_to_float(parts[-1])
            elif i + 1 < len(lines) and is_money(lines[i + 1]):
                total = money_to_float(lines[i + 1])
            break
        if i + 4 < len(lines) and is_code(lines[i]) and lines[i + 2].isdigit() an
            code = lines[i]
            desc = lines[i + 1]
            qty = int(lines[i + 2])
            unit = money_to_float(lines[i + 3])
            line_total = money_to_float(lines[i + 4])
            items.append({"code": code, "qty": qty, "unit": unit, "line_total": l
            i += 5
        else:
            i += 1
    return items, total
@dataclass
class ReceiptProfile:
    meta: Dict[str, Any]
    code_spend: Dict[str, float]  # aggregated spend per code
    code_qty: Dict[str, float]    # aggregated qty per code
    tok_count: Dict[str, float]   # dict for FeatureHasher
    tok_spend: Dict[str, float]
    tok_qty: Dict[str, float]

def featurize_receipt(patient_id: int, pdf_dir: Path) -> ReceiptProfile:
    """
    Parse receipt_{patient_id}.pdf -> meta + per-code aggregates + hashed token d
    """
    pdf_path = pdf_dir / f"receipt_{patient_id}.pdf"
    empty_meta = dict(
        pdf_ok=0,
        pdf_n_lines=0,
        pdf_n_unique_codes=0,
        pdf_sum_qty=0.0,
        pdf_mean_qty=0.0,
        pdf_max_qty=0.0,
        pdf_sum_line_total=0.0,
        pdf_mean_line_total=0.0,
        pdf_max_line_total=0.0,
        pdf_std_line_total=0.0,
        pdf_top1_cost_share=0.0,
        pdf_top3_cost_share=0.0,
        pdf_repeat_cost_share=0.0,
        pdf_total=0.0,
        pdf_total_diff=0.0,
        pdf_insurance=None,
        pdf_zip3=None,
        pdf_parse_notes="missing_pdf",
    )
    if not pdf_path.exists():
        return ReceiptProfile(meta=empty_meta, code_spend={}, code_qty={}, tok_co
    try:
        with fitz.open(pdf_path) as doc:
            full_text = ""
            header_pid = None
            zip3 = None
            insurance = None
            items: List[Dict[str, Any]] = []
            total: Optional[float] = None

            for page in doc:
                text = page.get_text("text") or ""
                full_text += text + "\n"
                words = page.get_text("words") or []
                items_p, total_p = parse_items_from_words(words)
                if not items_p:
                    items_p, total_p = parse_items_fallback_text(text)
                if items_p:
                    items.extend(items_p)
                if total_p is not None:
                    total = total_p
            header_pid, zip3, insurance = parse_header_fields(full_text)
    except Exception as e:
        empty_meta["pdf_parse_notes"] = f"open_error:{type(e).__name__}"
        return ReceiptProfile(meta=empty_meta, code_spend={}, code_qty={}, tok_co
    if not items:
        empty_meta["pdf_parse_notes"] = "no_items_parsed"
        empty_meta["pdf_zip3"] = zip3
        empty_meta["pdf_insurance"] = insurance
        return ReceiptProfile(meta=empty_meta, code_spend={}, code_qty={}, tok_co
    # Aggregate by code
    code_spend: Dict[str, float] = {}
    code_qty: Dict[str, float] = {}
    for it in items:
        c = str(it["code"])
        code_spend[c] = code_spend.get(c, 0.0) + float(it["line_total"])
        code_qty[c] = code_qty.get(c, 0.0) + float(it["qty"])
    line_totals = np.array([float(it["line_total"]) for it in items], dtype=float
    qtys = np.array([float(it["qty"]) for it in items], dtype=float)
    sum_line_total = float(line_totals.sum())
    pdf_total = float(total) if total is not None else sum_line_total

    codes = list(code_spend.keys())
    n_lines = len(items)
    n_unique = len(codes)
    sorted_cost = np.sort(line_totals)[::-1]
    top1 = float(sorted_cost[0] / sum_line_total) if sum_line_total > 0 else 0.0
    top3 = float(sorted_cost[:3].sum() / sum_line_total) if sum_line_total > 0 el
    # repeat spend share (code seen >=2 line items)
    cnt_by_code: Dict[str, int] = {}
    spend_by_code_line: Dict[str, float] = {}
    for it in items:
        c = str(it["code"])
        cnt_by_code[c] = cnt_by_code.get(c, 0) + 1
        spend_by_code_line[c] = spend_by_code_line.get(c, 0.0) + float(it["line_t
    repeat_spend = sum(v for c, v in spend_by_code_line.items() if cnt_by_code.get
    repeat_share = float(repeat_spend / sum_line_total) if sum_line_total > 0 els
    meta = dict(
        pdf_ok=1,
        pdf_n_lines=n_lines,
        pdf_n_unique_codes=n_unique,
        pdf_sum_qty=float(qtys.sum()),
        pdf_mean_qty=float(qtys.mean()) if n_lines else 0.0,
        pdf_max_qty=float(qtys.max()) if n_lines else 0.0,
        pdf_sum_line_total=sum_line_total,
        pdf_mean_line_total=float(line_totals.mean()) if n_lines else 0.0,
        pdf_max_line_total=float(line_totals.max()) if n_lines else 0.0,
        pdf_std_line_total=float(line_totals.std()) if n_lines else 0.0,
        pdf_top1_cost_share=top1,
        pdf_top3_cost_share=top3,
        pdf_repeat_cost_share=repeat_share,
        pdf_total=pdf_total,
        pdf_total_diff=float(abs(sum_line_total - pdf_total)),
        pdf_insurance=(insurance.lower() if isinstance(insurance, str) else None)
        pdf_zip3=(str(zip3) if zip3 is not None else None),
        pdf_parse_notes="ok",
    )
    # Hashed token dicts (long-tail code features)

    tok_count: Dict[str, float] = {}
    tok_spend: Dict[str, float] = {}
    tok_qty: Dict[str, float] = {}
    for code, spend in code_spend.items():
        qty = float(code_qty.get(code, 0.0))
        toks = [
            f"code={code}",
            f"p1={code[0]}",
            f"p2={code[:2]}",
            f"p3={code[:3]}",
            f"type={'hcpcs' if code[0].isalpha() else 'cpt'}",
        ]
        for t in toks:
            tok_count[t] = tok_count.get(t, 0.0) + 1.0
            tok_spend[t] = tok_spend.get(t, 0.0) + float(spend)
            tok_qty[t] = tok_qty.get(t, 0.0) + qty
    # compress magnitudes
    tok_count = {k: safe_log1p(v) for k, v in tok_count.items()}
    tok_spend = {k: safe_log1p(v) for k, v in tok_spend.items()}
    tok_qty = {k: safe_log1p(v) for k, v in tok_qty.items()}
    return ReceiptProfile(meta=meta, code_spend=code_spend, code_qty=code_qty,
                          tok_count=tok_count, tok_spend=tok_spend, tok_qty=tok_q
def load_cache(cache_path: Path) -> Dict[str, Any]:
    if cache_path.exists():
        try:
            obj = joblib.load(cache_path)
            if isinstance(obj, dict) and obj.get("version") == CACHE_VERSION and 
                return obj
        except Exception:
            pass
    return {"version": CACHE_VERSION, "data": {}}
def save_cache(cache_path: Path, cache_obj: Dict[str, Any]) -> None:
    try:

        joblib.dump(cache_obj, cache_path)
    except Exception as e:
        logging.warning(f"Cache save failed: {e}")
def build_receipt_artifacts(patient_ids: List[int], pdf_dir: Path, cache_path: Pa
    """
    Parse all receipts (with caching) and return:
      - receipt_meta_df: per patient meta
      - profiles: dict patient_id -> ReceiptProfile-like dict (code_spend, code_q
    """
    cache_obj = load_cache(cache_path)
    data: Dict[str, Any] = cache_obj.get("data", {})
    to_parse = [pid for pid in patient_ids if str(pid) not in data]
    if to_parse:
        logging.info(f"Parsing {len(to_parse):,} receipts (cache miss).")
    else:
        logging.info("All receipts loaded from cache.")
    for pid in tqdm(to_parse, desc="Parsing receipts", ncols=100):
        prof = featurize_receipt(pid, pdf_dir)
        data[str(pid)] = {
            "meta": prof.meta,
            "code_spend": prof.code_spend,
            "code_qty": prof.code_qty,
            "tok_count": prof.tok_count,
            "tok_spend": prof.tok_spend,
            "tok_qty": prof.tok_qty,
        }
    cache_obj["data"] = data
    if to_parse:
        save_cache(cache_path, cache_obj)
    meta_rows = []
    profiles: Dict[int, Dict[str, Any]] = {}
    for pid in patient_ids:
        obj = data.get(str(pid), None)
        if obj is None:

            prof = featurize_receipt(pid, pdf_dir)
            obj = {
                "meta": prof.meta,
                "code_spend": prof.code_spend,
                "code_qty": prof.code_qty,
                "tok_count": prof.tok_count,
                "tok_spend": prof.tok_spend,
                "tok_qty": prof.tok_qty,
            }
        meta_rows.append({"patient_id": pid, **obj["meta"]})
        profiles[pid] = obj
    receipt_meta_df = pd.DataFrame(meta_rows)
    return receipt_meta_df, profiles
# =========================
# NON-TARGET FEATURES
# =========================
def add_top_code_pivots(df_all: pd.DataFrame, profiles: Dict[int, Dict[str, Any]]
                        train_patient_ids: np.ndarray, top_k: int) -> pd.DataFram
    """
    Create pivot features for top-K codes by frequency (TRAIN only):
      - cpt_cnt_{code}: how many distinct occurrences (here: presence => 1, but w
      - cpt_share_{code}: spend share relative to prior_ed_cost_5y_usd
    Uses per-patient code_spend from profiles.
    """
    # count code frequency across train
    code_freq: Dict[str, int] = {}
    for pid in train_patient_ids.tolist():
        code_spend = profiles.get(int(pid), {}).get("code_spend", {}) or {}
        for c in code_spend.keys():
            code_freq[c] = code_freq.get(c, 0) + 1
    if not code_freq:
        logging.warning("No codes found for top-code pivots; skipping.")
        return df_all
    top_codes = [c for c, _ in sorted(code_freq.items(), key=lambda kv: kv[1], rev
    top_set = set(top_codes)

    logging.info(f"Adding top-code pivots: K={len(top_codes)}")
    # build per patient row dict
    cnt_cols = {f"cpt_cnt_{c}": [] for c in top_codes}
    share_cols = {f"cpt_share_{c}": [] for c in top_codes}
    prior_cost = df_all.set_index("patient_id")["prior_ed_cost_5y_usd"].astype(fl
    for pid in df_all["patient_id"].astype(int).tolist():
        code_spend = profiles.get(pid, {}).get("code_spend", {}) or {}
        denom = float(prior_cost.get(pid, 0.0)) + 1e-6
        for c in top_codes:
            spend = float(code_spend.get(c, 0.0))
            # presence count: 1 if present else 0 (robust, low-noise); can change 
            cnt_cols[f"cpt_cnt_{c}"].append(1.0 if spend > 0 else 0.0)
            share_cols[f"cpt_share_{c}"].append(spend / denom if spend > 0 else 0
    for c in top_codes:
        df_all[f"cpt_cnt_{c}"] = cnt_cols[f"cpt_cnt_{c}"]
        df_all[f"cpt_share_{c}"] = share_cols[f"cpt_share_{c}"]
    return df_all
def build_base_sparse_matrix(df_all: pd.DataFrame,
                             profiles: Dict[int, Dict[str, Any]]) -> Tuple[csr_ma
    """
    Base sparse feature matrix (no target encoding):
      - numeric + categorical OHE
      - optional hashed token dicts (count/spend/qty)
    """
    df = df_all.copy()
    # derived tabular
    df["prior_cost_per_visit"] = df["prior_ed_cost_5y_usd"] / (df["prior_ed_visit
    df["log_prior_cost"] = np.log1p(df["prior_ed_cost_5y_usd"].astype(float))
    df["log_prior_visits"] = np.log1p(df["prior_ed_visits_5y"].astype(float))
    # normalize receipt header categoricals
    for c in ["pdf_insurance", "pdf_zip3", "insurance", "zip3", "sex", "primary_c

        if c in df.columns:
            df[c] = df[c].astype("string").fillna("missing")
    # choose cols
    exclude = {"patient_id", "ed_cost_next3y_usd", "is_train"}
    cat_cols = [c for c in ["primary_chronic", "sex", "insurance", "zip3", "pdf_i
                if c in df.columns]
    num_cols = [c for c in df.columns
                if c not in exclude and c not in cat_cols and pd.api.types.is_num
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), nu
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    X_tab = preprocessor.fit_transform(df)
    X_tab = X_tab.tocsr() if hasattr(X_tab, "tocsr") else csr_matrix(X_tab)
    blocks = [X_tab]
    artifacts: Dict[str, Any] = {"preprocessor": preprocessor, "num_cols": num_co
    if USE_HASHED_DICT_FEATURES:
        ids = df["patient_id"].astype(int).tolist()
        dict_count = [profiles.get(pid, {}).get("tok_count", {}) or {} for pid in 
        dict_spend = [profiles.get(pid, {}).get("tok_spend", {}) or {} for pid in 
        dict_qty = [profiles.get(pid, {}).get("tok_qty", {}) or {} for pid in ids
        hasher_count = FeatureHasher(n_features=HASH_DIM_COUNT, input_type="dict"
        hasher_spend = FeatureHasher(n_features=HASH_DIM_SPEND, input_type="dict"
        hasher_qty = FeatureHasher(n_features=HASH_DIM_QTY, input_type="dict", dty
        X_count = hasher_count.transform(dict_count)
        X_spend = hasher_spend.transform(dict_spend)
        X_qty = hasher_qty.transform(dict_qty)
        blocks.extend([X_count, X_spend, X_qty])

        artifacts.update({"hasher_count": hasher_count, "hasher_spend": hasher_sp
    X = hstack(blocks, format="csr")
    artifacts["n_features"] = X.shape[1]
    return X, artifacts
# =========================
# TARGET ENCODING (CV-SAFE)
# =========================
@dataclass
class TokenStats:
    cnt: int
    mean: float
    median: float
    mean_smooth: float
    median_adj: float
def code_to_prefix2(code: str) -> str:
    return code[:2] if len(code) >= 2 else code
def build_token_stats_for_fold(
    train_pids: List[int],
    y_log: np.ndarray,
    pid_to_row: Dict[int, int],
    profiles: Dict[int, Dict[str, Any]],
) -> Tuple[Dict[str, TokenStats], Dict[str, TokenStats], float, float]:
    """
    Build stats on log1p(y) for:
      - exact codes
      - prefix2 codes
    using ONLY training fold patients.
    """
    values_code: Dict[str, List[float]] = {}
    values_p2: Dict[str, List[float]] = {}
    ys = []
    for pid in train_pids:

        r = pid_to_row.get(pid, None)
        if r is None:
            continue
        v = float(y_log[r])
        ys.append(v)
        code_spend = profiles.get(pid, {}).get("code_spend", {}) or {}
        codes = list(code_spend.keys())
        # use presence (unique codes per patient) for mapping
        for c in codes:
            values_code.setdefault(c, []).append(v)
            p2 = code_to_prefix2(c)
            values_p2.setdefault(p2, []).append(v)
    if not ys:
        global_mean = 0.0
        global_median = 0.0
    else:
        global_mean = float(np.mean(ys))
        global_median = float(np.median(ys))
    def to_stats(values_map: Dict[str, List[float]]) -> Dict[str, TokenStats]:
        out: Dict[str, TokenStats] = {}
        for tok, arr in values_map.items():
            a = np.asarray(arr, dtype=float)
            cnt = int(a.size)
            m = float(np.mean(a))
            med = float(np.median(a))
            mean_smooth = float((cnt * m + TE_SMOOTH * global_mean) / (cnt + TE_SM
            median_adj = med if cnt >= TE_MIN_COUNT_MEDIAN else global_median
            out[tok] = TokenStats(cnt=cnt, mean=m, median=med, mean_smooth=mean_s
        return out
    return to_stats(values_code), to_stats(values_p2), global_mean, global_median
def te_features_for_patients(
    pids: List[int],

    profiles: Dict[int, Dict[str, Any]],
    stats_code: Dict[str, TokenStats],
    stats_p2: Dict[str, TokenStats],
    global_mean: float,
    global_median: float,
) -> np.ndarray:
    """
    For each patient, aggregate token stats with spend weights.
    Returns features (float32) with columns:
      codes: wmean_mean, wmean_median, unw_mean, max_mean, min_mean, top_spend_me
      p2:    same 8 features
    Total: 16 features
    """
    feats = np.zeros((len(pids), 16), dtype=np.float32)
    for i, pid in enumerate(pids):
        code_spend = profiles.get(pid, {}).get("code_spend", {}) or {}
        if not code_spend:
            # no codes -> global
            base = np.array([global_mean, global_median, global_mean, global_mean
            feats[i, 0:8] = base
            feats[i, 8:16] = base
            continue
        # ----- exact codes -----
        codes = list(code_spend.keys())
        spends = np.array([float(code_spend[c]) for c in codes], dtype=float)
        ssum = float(spends.sum())
        weights = spends / (ssum + 1e-9)
        vals_mean = []
        vals_med = []
        cnts = []
        known = 0
        for c in codes:
            st = stats_code.get(c, None)
            if st is None:
                vals_mean.append(global_mean)

                vals_med.append(global_median)
                cnts.append(0)
            else:
                vals_mean.append(st.mean_smooth)
                vals_med.append(st.median_adj)
                cnts.append(st.cnt)
                known += 1
        vals_mean = np.array(vals_mean, dtype=float)
        vals_med = np.array(vals_med, dtype=float)
        cnts = np.array(cnts, dtype=float)
        top_idx = int(np.argmax(weights))
        code_feat = np.array([
            float(np.sum(weights * vals_mean)),                 # wmean mean_smoo
            float(np.sum(weights * vals_med)),                  # wmean median_adj
            float(np.mean(vals_mean)),                          # unweighted mean
            float(np.max(vals_mean)),
            float(np.min(vals_mean)),
            float(vals_med[top_idx]),                           # top-spend median
            float(known / max(1, len(codes))),                  # known fraction
            float(np.mean(np.log1p(cnts))),                     # avg log count
        ], dtype=np.float32)
        feats[i, 0:8] = code_feat
        # ----- prefix2 -----
        p2_map: Dict[str, float] = {}
        for c, sp in code_spend.items():
            p2 = code_to_prefix2(c)
            p2_map[p2] = p2_map.get(p2, 0.0) + float(sp)
        p2s = list(p2_map.keys())
        p2sp = np.array([float(p2_map[p]) for p in p2s], dtype=float)
        p2w = p2sp / (float(p2sp.sum()) + 1e-9)
        p2_mean = []
        p2_med = []
        p2_cnt = []
        known2 = 0

        for p in p2s:
            st = stats_p2.get(p, None)
            if st is None:
                p2_mean.append(global_mean)
                p2_med.append(global_median)
                p2_cnt.append(0)
            else:
                p2_mean.append(st.mean_smooth)
                p2_med.append(st.median_adj)
                p2_cnt.append(st.cnt)
                known2 += 1
        p2_mean = np.array(p2_mean, dtype=float)
        p2_med = np.array(p2_med, dtype=float)
        p2_cnt = np.array(p2_cnt, dtype=float)
        top2 = int(np.argmax(p2w))
        p2_feat = np.array([
            float(np.sum(p2w * p2_mean)),
            float(np.sum(p2w * p2_med)),
            float(np.mean(p2_mean)),
            float(np.max(p2_mean)),
            float(np.min(p2_mean)),
            float(p2_med[top2]),
            float(known2 / max(1, len(p2s))),
            float(np.mean(np.log1p(p2_cnt))),
        ], dtype=np.float32)
        feats[i, 8:16] = p2_feat
    return feats
def make_strat_labels(train_df: pd.DataFrame) -> np.ndarray:
    chronic = train_df["primary_chronic"].astype(str)
    cost_bin = pd.qcut(train_df["prior_ed_cost_5y_usd"], q=10, duplicates="drop")
    return (chronic + "_" + cost_bin).values

# =========================
# MODELS
# =========================
def mae_original_from_log(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> Tupl
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return "mae_usd", float(np.mean(np.abs(y_true - y_pred))), False
def predict_xgb(model: XGBRegressor, X) -> np.ndarray:
    """
    Robustly predict using the best iteration if early stopping happened.
    """
    best_iter = getattr(model, "best_iteration", None)
    if best_iter is None:
        try:
            best_iter = model.get_booster().best_iteration
        except Exception:
            best_iter = None
    if best_iter is None:
        return model.predict(X)
    # xgboost sklearn supports iteration_range in newer versions
    try:
        return model.predict(X, iteration_range=(0, int(best_iter) + 1))
    except TypeError:
        # fallback
        return model.predict(X)
def optimize_weights_mae(y_true: np.ndarray, preds: Dict[str, np.ndarray], step: 
    """
    Grid search weights on simplex to minimize MAE.
    Works for 2-4 models; step=0.05 is usually enough.
    """
    names = list(preds.keys())
    m = len(names)
    if m == 1:
        return {names[0]: 1.0}, mae(y_true, preds[names[0]])

    best_w: Dict[str, float] = {}
    best_mae = float("inf")
    grid = np.arange(0.0, 1.0 + 1e-9, step)
    if m == 2:
        a, b = names
        for w in grid:
            p = w * preds[a] + (1 - w) * preds[b]
            score = mae(y_true, p)
            if score < best_mae:
                best_mae = score
                best_w = {a: float(w), b: float(1 - w)}
        return best_w, best_mae
    if m == 3:
        a, b, c = names
        for w1 in grid:
            for w2 in grid:
                w3 = 1.0 - w1 - w2
                if w3 < -1e-9:
                    continue
                if w3 < 0:
                    w3 = 0.0
                p = w1 * preds[a] + w2 * preds[b] + w3 * preds[c]
                score = mae(y_true, p)
                if score < best_mae:
                    best_mae = score
                    best_w = {a: float(w1), b: float(w2), c: float(w3)}
        return best_w, best_mae
    if m == 4:
        a, b, c, d = names
        for w1 in grid:
            for w2 in grid:
                for w3 in grid:
                    w4 = 1.0 - w1 - w2 - w3
                    if w4 < -1e-9:
                        continue

                    if w4 < 0:
                        w4 = 0.0
                    p = w1 * preds[a] + w2 * preds[b] + w3 * preds[c] + w4 * pred
                    score = mae(y_true, p)
                    if score < best_mae:
                        best_mae = score
                        best_w = {a: float(w1), b: float(w2), c: float(w3), d: fl
        return best_w, best_mae
    raise ValueError("Too many models for grid search")
# =========================
# MAIN TRAINING LOOP (CV + TE)
# =========================
def main() -> None:
    setup_logging()
    logging.info("=== Iteration 2: Target-Encoded CPT Risk + Ensemble (MAE) ===")
    # I/O checks
    for p in [TRAIN_CSV, TEST_CSV]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")
    if not PDF_DIR.exists():
        raise FileNotFoundError(f"Missing PDF directory: {PDF_DIR}")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    # Optional patients.csv
    if PATIENTS_CSV.exists():
        patients = pd.read_csv(PATIENTS_CSV)
        if "patient_id" in patients.columns:
            train_df = train_df.merge(patients, on="patient_id", how="left")
            test_df = test_df.merge(patients, on="patient_id", how="left")
            logging.info("Merged patients.csv into train/test.")
        else:
            logging.warning("patients.csv exists but lacks patient_id; skipping."
    else:
        logging.info("patients.csv not found; proceeding without it.")

    train_df["is_train"] = 1
    test_df["is_train"] = 0
    df_all = pd.concat([train_df, test_df], ignore_index=True)
    # Parse receipts (cached)
    all_pids = df_all["patient_id"].astype(int).unique().tolist()
    all_pids.sort()
    receipt_meta_df, profiles = build_receipt_artifacts(all_pids, PDF_DIR, CACHE_
    df_all = df_all.merge(receipt_meta_df, on="patient_id", how="left")
    # Top-code pivots (optional; non-target)
    if USE_TOP_CODE_PIVOTS:
        df_all = add_top_code_pivots(
            df_all=df_all,
            profiles=profiles,
            train_patient_ids=train_df["patient_id"].astype(int).values,
            top_k=TOP_K_CODES,
        )
    # Base sparse matrix (no TE)
    logging.info("Building base sparse matrix (no target-encoding)...")
    X_all_base, base_artifacts = build_base_sparse_matrix(df_all, profiles)
    logging.info(f"Base X: {X_all_base.shape}")
    # Split back
    is_train = df_all["is_train"].values.astype(int) == 1
    X_train_base = X_all_base[is_train]
    X_test_base = X_all_base[~is_train]
    y = train_df["ed_cost_next3y_usd"].values.astype(np.float32)
    y_log = np.log1p(y)
    # For TE building
    train_pids = train_df["patient_id"].astype(int).tolist()
    pid_to_row = {pid: i for i, pid in enumerate(train_pids)}
    # Dense df for CatBoost (no giant sparse; keep raw cats + numerics + top pivo
    # We'll just use df_all directly (excluding target), and CatBoost will handle 
    cb_cols_exclude = {"ed_cost_next3y_usd"}

    df_cb_all = df_all.drop(columns=[c for c in cb_cols_exclude if c in df_all.co
    # Ensure categorical columns are strings
    cb_cat_cols = [c for c in ["primary_chronic", "sex", "insurance", "zip3", "pd
                   if c in df_cb_all.columns]
    for c in cb_cat_cols:
        df_cb_all[c] = df_cb_all[c].astype("string").fillna("missing")
    # Derived numeric features for catboost too
    df_cb_all["prior_cost_per_visit"] = df_cb_all["prior_ed_cost_5y_usd"] / (df_c
    df_cb_all["log_prior_cost"] = np.log1p(df_cb_all["prior_ed_cost_5y_usd"].asty
    df_cb_all["log_prior_visits"] = np.log1p(df_cb_all["prior_ed_visits_5y"].asty
    # Strat labels
    strat_labels = make_strat_labels(train_df)
    # Storage for OOF + TEST predictions
    model_names: List[str] = []
    if USE_LGB_L1: model_names.append("lgb_l1")
    if USE_LGB_QLOG: model_names.append("lgb_qlog")
    if USE_XGB_L1: model_names.append("xgb_l1")
    if USE_CAT_MAE: model_names.append("cat_mae")
    oof_sum = {name: np.zeros(len(y), dtype=np.float64) for name in model_names}
    oof_cnt = {name: np.zeros(len(y), dtype=np.float64) for name in model_names}
    test_sum = {name: np.zeros(len(test_df), dtype=np.float64) for name in model_
    test_cnt = {name: 0 for name in model_names}
    # CV generator (repeated stratified k-fold)
    splits: List[Tuple[int, int, np.ndarray, np.ndarray]] = []
    for rep, seed in enumerate(REPEAT_SEEDS):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), strat_
            splits.append((rep, fold, tr_idx, va_idx))
    logging.info(f"Running CV with TE: repeats={len(REPEAT_SEEDS)}, folds={N_FOLD
    # Model hyperparams
    lgb_l1_params = dict(
        objective="regression_l1",

        learning_rate=0.03,
        n_estimators=12000,
        num_leaves=96,
        min_child_samples=25,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_lambda=1.5,
        reg_alpha=0.0,
        max_bin=255,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_qlog_params = dict(
        objective="quantile",
        alpha=0.5,
        learning_rate=0.03,
        n_estimators=16000,
        num_leaves=128,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.70,
        reg_lambda=1.2,
        reg_alpha=0.0,
        max_bin=255,
        n_jobs=-1,
        verbose=-1,
    )
    xgb_params = dict(
        objective="reg:absoluteerror",
        n_estimators=14000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        reg_alpha=0.0,

        min_child_weight=1.0,
        tree_method="hist",
        eval_metric="mae",
        n_jobs=-1,
        verbosity=0,
    )
    cat_params = dict(
        loss_function="MAE",
        eval_metric="MAE",
        iterations=20000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=6.0,
        random_strength=0.5,
        bagging_temperature=0.8,
        od_type="Iter",
        od_wait=EARLY_STOPPING,
        task_type=CAT_TASK_TYPE,
        verbose=False,
    )
    # Main CV loop
    for rep, fold, tr_idx, va_idx in tqdm(splits, desc="CV folds", ncols=110):
        # Build TE stats from training fold only
        tr_pids = [train_pids[i] for i in tr_idx.tolist()]
        stats_code, stats_p2, gmean, gmed = build_token_stats_for_fold(
            train_pids=tr_pids,
            y_log=y_log,
            pid_to_row=pid_to_row,
            profiles=profiles,
        )
        # TE features for train/val/test (log-space stats, but features are numer
        te_tr = te_features_for_patients(tr_pids, profiles, stats_code, stats_p2, 
        va_pids = [train_pids[i] for i in va_idx.tolist()]
        te_va = te_features_for_patients(va_pids, profiles, stats_code, stats_p2, 
        test_pids = test_df["patient_id"].astype(int).tolist()
        te_te = te_features_for_patients(test_pids, profiles, stats_code, stats_p

        # Sparse matrices for LGB/XGB
        X_tr = hstack([X_train_base[tr_idx], csr_matrix(te_tr)], format="csr")
        X_va = hstack([X_train_base[va_idx], csr_matrix(te_va)], format="csr")
        X_te = hstack([X_test_base, csr_matrix(te_te)], format="csr")
        y_tr = y[tr_idx]
        y_va = y[va_idx]
        # ----------------------
        # LightGBM L1 on raw target
        # ----------------------
        if USE_LGB_L1:
            mdl = LGBMRegressor(**lgb_l1_params, random_state=1000 * rep + fold)
            mdl.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric="mae",
                callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
            )
            best_it = int(getattr(mdl, "best_iteration_", lgb_l1_params["n_estimat
            p_va = mdl.predict(X_va, num_iteration=best_it)
            p_te = mdl.predict(X_te, num_iteration=best_it)
            if CLIP_AT_ZERO:
                p_va = np.maximum(p_va, 0.0)
                p_te = np.maximum(p_te, 0.0)
            oof_sum["lgb_l1"][va_idx] += p_va
            oof_cnt["lgb_l1"][va_idx] += 1.0
            test_sum["lgb_l1"] += p_te
            test_cnt["lgb_l1"] += 1
        # ----------------------
        # LightGBM Quantile (median) on log1p(target)
        # ----------------------
        if USE_LGB_QLOG:
            mdl = LGBMRegressor(**lgb_qlog_params, random_state=2000 * rep + fold
            y_tr_log = np.log1p(y_tr)
            y_va_log = np.log1p(y_va)
            mdl.fit(

                X_tr, y_tr_log,
                eval_set=[(X_va, y_va_log)],
                eval_metric=mae_original_from_log,  # early stop aligned to USD MA
                callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
            )
            best_it = int(getattr(mdl, "best_iteration_", lgb_qlog_params["n_esti
            p_va = np.expm1(mdl.predict(X_va, num_iteration=best_it))
            p_te = np.expm1(mdl.predict(X_te, num_iteration=best_it))
            if CLIP_AT_ZERO:
                p_va = np.maximum(p_va, 0.0)
                p_te = np.maximum(p_te, 0.0)
            oof_sum["lgb_qlog"][va_idx] += p_va
            oof_cnt["lgb_qlog"][va_idx] += 1.0
            test_sum["lgb_qlog"] += p_te
            test_cnt["lgb_qlog"] += 1
        # ----------------------
        # XGBoost L1 on raw target
        # ----------------------
        if USE_XGB_L1:
            mdl = XGBRegressor(**xgb_params, random_state=3000 * rep + fold)
            mdl.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                verbose=False,
                early_stopping_rounds=EARLY_STOPPING,
            )
            p_va = predict_xgb(mdl, X_va)
            p_te = predict_xgb(mdl, X_te)
            if CLIP_AT_ZERO:
                p_va = np.maximum(p_va, 0.0)
                p_te = np.maximum(p_te, 0.0)
            oof_sum["xgb_l1"][va_idx] += p_va
            oof_cnt["xgb_l1"][va_idx] += 1.0
            test_sum["xgb_l1"] += p_te
            test_cnt["xgb_l1"] += 1

        # ----------------------
        # CatBoost MAE on dense data (+ TE columns)
        # ----------------------
        if USE_CAT_MAE:
            # Build fold-specific dense frames
            df_train_cb = df_cb_all[is_train].reset_index(drop=True)
            df_test_cb = df_cb_all[~is_train].reset_index(drop=True)
            # add TE columns
            te_cols = [
                f"{TE_FEATURE_PREFIX}code_wmean_mean",
                f"{TE_FEATURE_PREFIX}code_wmean_median",
                f"{TE_FEATURE_PREFIX}code_unw_mean",
                f"{TE_FEATURE_PREFIX}code_max_mean",
                f"{TE_FEATURE_PREFIX}code_min_mean",
                f"{TE_FEATURE_PREFIX}code_top_median",
                f"{TE_FEATURE_PREFIX}code_known_frac",
                f"{TE_FEATURE_PREFIX}code_avg_log_cnt",
                f"{TE_FEATURE_PREFIX}p2_wmean_mean",
                f"{TE_FEATURE_PREFIX}p2_wmean_median",
                f"{TE_FEATURE_PREFIX}p2_unw_mean",
                f"{TE_FEATURE_PREFIX}p2_max_mean",
                f"{TE_FEATURE_PREFIX}p2_min_mean",
                f"{TE_FEATURE_PREFIX}p2_top_median",
                f"{TE_FEATURE_PREFIX}p2_known_frac",
                f"{TE_FEATURE_PREFIX}p2_avg_log_cnt",
            ]
            # assign TE for this fold (train has two subsets)
            # build full arrays aligned to train_df rows
            te_train_full = np.zeros((len(train_df), 16), dtype=np.float32)
            te_train_full[tr_idx, :] = te_tr
            te_train_full[va_idx, :] = te_va
            for j, col in enumerate(te_cols):
                df_train_cb[col] = te_train_full[:, j]
                df_test_cb[col] = te_te[:, j]
            # Determine cat feature indices for this DF

            # (must recompute since we added columns)
            cat_cols = [c for c in cb_cat_cols if c in df_train_cb.columns]
            cat_idx = [df_train_cb.columns.get_loc(c) for c in cat_cols]
            # Pools
            tr_pool = Pool(df_train_cb.iloc[tr_idx], label=y_tr, cat_features=cat_
            va_pool = Pool(df_train_cb.iloc[va_idx], label=y_va, cat_features=cat_
            te_pool = Pool(df_test_cb, cat_features=cat_idx)
            mdl = CatBoostRegressor(**cat_params, random_seed=4000 * rep + fold)
            mdl.fit(tr_pool, eval_set=va_pool, use_best_model=True)
            p_va = mdl.predict(va_pool)
            p_te = mdl.predict(te_pool)
            if CLIP_AT_ZERO:
                p_va = np.maximum(p_va, 0.0)
                p_te = np.maximum(p_te, 0.0)
            oof_sum["cat_mae"][va_idx] += p_va
            oof_cnt["cat_mae"][va_idx] += 1.0
            test_sum["cat_mae"] += p_te
            test_cnt["cat_mae"] += 1
    # Finalize OOF and TEST preds per model
    oof_preds: Dict[str, np.ndarray] = {}
    test_preds: Dict[str, np.ndarray] = {}
    for name in model_names:
        denom = np.maximum(oof_cnt[name], 1.0)
        oof_preds[name] = (oof_sum[name] / denom).astype(np.float32)
        test_preds[name] = (test_sum[name] / max(1, test_cnt[name])).astype(np.fl
        score = mae(y, oof_preds[name])
        logging.info(f"OOF MAE | {name}: {score:.4f}")
    # Optimize blend on OOF (MAE)
    weights, best_mae = optimize_weights_mae(y, oof_preds, step=0.05)
    logging.info(f"Best OOF blend MAE (pre-bias): {best_mae:.4f} | weights={weight

What to look for in your local results
If this idea is correct, you should see:
OOF MAE drops materially vs your previous 493.9 baseline before submitting.
    # Blend + MAE-optimal constant bias (median residual)
    oof_blend = np.zeros(len(y), dtype=np.float64)
    test_blend = np.zeros(len(test_df), dtype=np.float64)
    for name, w in weights.items():
        oof_blend += w * oof_preds[name]
        test_blend += w * test_preds[name]
    bias = float(np.median(y - oof_blend))
    oof_final = oof_blend + bias
    test_final = test_blend + bias
    if CLIP_AT_ZERO:
        oof_final = np.maximum(oof_final, 0.0)
        test_final = np.maximum(test_final, 0.0)
    final_mae = mae(y, oof_final)
    logging.info(f"Final OOF MAE (blend + median-bias): {final_mae:.4f} | bias={b
    # Save submission
    sub = pd.DataFrame({
        "patient_id": test_df["patient_id"].astype(int).values,
        "ed_cost_next3y_usd": test_final.astype(np.float32),
    })
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(SUBMISSION_PATH, index=False)
    logging.info(f"Saved: {SUBMISSION_PATH}")
if __name__ == "__main__":
    main()

The TE + ensemble blend gives a bigger gain than any single model.
If OOF improves but LB doesn’t, it’s almost always a CV mismatch: in that case, we’ll
tighten the split strategy (e.g., stratify also by receipt complexity bins, or use repeated
CV with different strat rules).
If you paste your OOF MAE per model and final OOF MAE from this script, I can tell you
exactly which component is carrying and what to ablate next for Iteration 3.
```

### C03 — ITERATION 2 (Fixed) — Target-Encoded CPT Risk + Ensemble (MAE) (pages 82-113)
```python
# -*- coding: utf-8 -*-
"""
ITERATION 2 (Fixed) — Target-Encoded CPT Risk + Ensemble (MAE)
Why this should beat Iteration 1:
- Biggest missing lift is learning "code -> future-cost risk". We do CV-safe targ
  (exact code + 2-char prefix), then spend-weight aggregate per patient.
- MAE-aligned objectives: LightGBM L1, LightGBM quantile-median on log1p(y), XGBo
- Rigorous repeated stratified CV + OOF-optimized blend + median-residual bias ca
Fix for your error:
- Newer XGBoost versions moved `early_stopping_rounds` out of `.fit()` and into t
  (or use callbacks). This script uses a compatibility wrapper so it works across 

===========================================================
INSTALL (run once)
===========================================================
pip install -U pandas numpy scipy scikit-learn lightgbm xgboost catboost pymupdf 
"""
from __future__ import annotations
import os
import re
import math
import logging
import warnings
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import fitz  # PyMuPDF
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction import FeatureHasher
import lightgbm as lgb
from lightgbm import LGBMRegressor
import xgboost as xgb

from xgboost import XGBRegressor
from catboost import CatBoostRegressor, Pool
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# =========================
# CONFIG
# =========================
BASE_DIR = Path(r"D:\AgentDs\agent_ds_healthcare")
TRAIN_CSV = BASE_DIR / "ed_cost_train.csv"
TEST_CSV = BASE_DIR / "ed_cost_test.csv"
PATIENTS_CSV = BASE_DIR / "patients.csv"          # optional
PDF_DIR = BASE_DIR / "receipts_pdf"
SUBMISSION_PATH = BASE_DIR / "submission.csv"
CACHE_PATH = BASE_DIR / "cache_receipts_iter2.joblib"
CACHE_VERSION = 2
# CV
N_FOLDS = 5
REPEAT_SEEDS = [2024, 2025, 2026]   # 3 repeats => 15 folds total
EARLY_STOPPING = 250
# Target encoding hyperparams
TE_SMOOTH = 20.0
TE_MIN_COUNT_MEDIAN = 5
TE_FEATURE_PREFIX = "te_"
# Top-code pivot features (non-target)
USE_TOP_CODE_PIVOTS = True
TOP_K_CODES = 180
# Hashed long-tail dict features (non-target)
USE_HASHED_DICT_FEATURES = True
HASH_DIM_COUNT = 2**12
HASH_DIM_SPEND = 2**12
HASH_DIM_QTY = 2**11

# Models
USE_LGB_L1 = True
USE_LGB_QLOG = True
USE_XGB_L1 = True
USE_CAT_MAE = True
# CatBoost: try GPU, auto-fallback to CPU
CAT_TASK_TYPE = "GPU"
# Prediction post-process
CLIP_AT_ZERO = True
# =========================
# LOGGING
# =========================
def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
# =========================
# HELPERS
# =========================
_money_re = re.compile(r"^\d{1,3}(?:,\d{3})*(?:\.\d{2})$|^\d+(?:\.\d{2})$")
_code_re = re.compile(r"^(?:[A-Z]\d{4}|\d{4,5})$")  # e.g., G0378, 99284
def is_money(tok: str) -> bool:
    return bool(_money_re.match(tok))
def money_to_float(tok: str) -> float:
    return float(tok.replace(",", ""))
def is_code(tok: str) -> bool:
    return bool(_code_re.match(tok))
def safe_log1p(x: float) -> float:

    return math.log1p(max(0.0, float(x)))
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))
def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_out
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)
# =========================
# PDF PARSING
# =========================
def cluster_words_into_lines(words: List[Tuple], y_tol: float = 1.5) -> List[List
    """
    Cluster PyMuPDF 'words' into lines using y coordinate, then sort by x.
    """
    if not words:
        return []
    words_sorted = sorted(words, key=lambda w: (w[1], w[0]))  # y0 then x0
    lines: List[List[str]] = []
    cur: List[Tuple[float, str]] = []
    cur_y: Optional[float] = None
    for w in words_sorted:
        x0, y0 = float(w[0]), float(w[1])
        tok = str(w[4])
        if cur_y is None or abs(y0 - cur_y) <= y_tol:
            cur.append((x0, tok))
            cur_y = y0 if cur_y is None else (0.85 * cur_y + 0.15 * y0)
        else:
            lines.append([t for _, t in sorted(cur, key=lambda z: z[0])])
            cur = [(x0, tok)]
            cur_y = y0
    if cur:

        lines.append([t for _, t in sorted(cur, key=lambda z: z[0])])
    return lines
def parse_header_fields(text: str) -> Tuple[Optional[int], Optional[str], Optiona
    pid = None
    zip3 = None
    insurance = None
    m = re.search(r"Patient ID:\s*(\d+)", text)
    if m:
        pid = int(m.group(1))
    m = re.search(r"ZIP3:\s*([0-9]{3})\s+Insurance:\s*([A-Za-z_]+)", text)
    if m:
        zip3 = str(m.group(1))
        insurance = str(m.group(2)).lower()
    return pid, zip3, insurance
def parse_items_from_words(words: List[Tuple]) -> Tuple[List[Dict[str, Any]], Opt
    items: List[Dict[str, Any]] = []
    total: Optional[float] = None
    lines = cluster_words_into_lines(words, y_tol=1.5)
    if not lines:
        return items, total
    header_idx = None
    for i, toks in enumerate(lines):
        low = [t.lower() for t in toks]
        if ("code" in low) and ("description" in low) and ("qty" in low):
            header_idx = i
            break
    if header_idx is None:
        return items, total
    for toks in lines[header_idx + 1 :]:

        if not toks:
            continue
        if toks[0].upper() == "TOTAL":
            for t in reversed(toks[1:]):
                if is_money(t):
                    total = money_to_float(t)
                    break
            break
        if len(toks) >= 5 and is_code(toks[0]) and toks[-3].isdigit() and is_money
            code = toks[0]
            qty = int(toks[-3])
            unit = money_to_float(toks[-2])
            line_total = money_to_float(toks[-1])
            desc = " ".join(toks[1:-3])
            items.append({"code": code, "qty": qty, "unit": unit, "line_total": l
    return items, total
def parse_items_fallback_text(text: str) -> Tuple[List[Dict[str, Any]], Optional[
    items: List[Dict[str, Any]] = []
    total: Optional[float] = None
    lines = [ln.strip() for ln in (text.splitlines() if text else []) if ln.strip
    if not lines:
        return items, total
    start_idx = None
    for i in range(len(lines) - 4):
        if lines[i].lower() == "code" and lines[i + 1].lower() == "description" a
            start_idx = i + 5
            break
        if "code" in lines[i].lower() and "description" in lines[i].lower() and "
            start_idx = i + 1
            break
    if start_idx is None:
        return items, total

    i = start_idx
    while i < len(lines):
        if lines[i].upper().startswith("TOTAL"):
            parts = lines[i].split()
            if len(parts) >= 2 and is_money(parts[-1]):
                total = money_to_float(parts[-1])
            elif i + 1 < len(lines) and is_money(lines[i + 1]):
                total = money_to_float(lines[i + 1])
            break
        if i + 4 < len(lines) and is_code(lines[i]) and lines[i + 2].isdigit() an
            code = lines[i]
            desc = lines[i + 1]
            qty = int(lines[i + 2])
            unit = money_to_float(lines[i + 3])
            line_total = money_to_float(lines[i + 4])
            items.append({"code": code, "qty": qty, "unit": unit, "line_total": l
            i += 5
        else:
            i += 1
    return items, total
@dataclass
class ReceiptProfile:
    meta: Dict[str, Any]
    code_spend: Dict[str, float]
    code_qty: Dict[str, float]
    tok_count: Dict[str, float]
    tok_spend: Dict[str, float]
    tok_qty: Dict[str, float]
def featurize_receipt(patient_id: int, pdf_dir: Path) -> ReceiptProfile:
    """
    Parse receipt_{patient_id}.pdf -> meta + per-code aggregates + hashed token d
    """
    pdf_path = pdf_dir / f"receipt_{patient_id}.pdf"

    empty_meta = dict(
        pdf_ok=0, pdf_n_lines=0, pdf_n_unique_codes=0,
        pdf_sum_qty=0.0, pdf_mean_qty=0.0, pdf_max_qty=0.0,
        pdf_sum_line_total=0.0, pdf_mean_line_total=0.0, pdf_max_line_total=0.0, 
        pdf_top1_cost_share=0.0, pdf_top3_cost_share=0.0, pdf_repeat_cost_share=0
        pdf_total=0.0, pdf_total_diff=0.0,
        pdf_insurance=None, pdf_zip3=None, pdf_parse_notes="missing_pdf",
    )
    if not pdf_path.exists():
        return ReceiptProfile(meta=empty_meta, code_spend={}, code_qty={}, tok_co
    try:
        with fitz.open(pdf_path) as doc:
            full_text = ""
            items: List[Dict[str, Any]] = []
            total: Optional[float] = None
            zip3 = None
            insurance = None
            for page in doc:
                text = page.get_text("text") or ""
                full_text += text + "\n"
                words = page.get_text("words") or []
                items_p, total_p = parse_items_from_words(words)
                if not items_p:
                    items_p, total_p = parse_items_fallback_text(text)
                if items_p:
                    items.extend(items_p)
                if total_p is not None:
                    total = total_p
            _, zip3, insurance = parse_header_fields(full_text)
    except Exception as e:
        empty_meta["pdf_parse_notes"] = f"open_error:{type(e).__name__}"
        return ReceiptProfile(meta=empty_meta, code_spend={}, code_qty={}, tok_co

    if not items:
        empty_meta["pdf_parse_notes"] = "no_items_parsed"
        empty_meta["pdf_zip3"] = zip3
        empty_meta["pdf_insurance"] = insurance
        return ReceiptProfile(meta=empty_meta, code_spend={}, code_qty={}, tok_co
    # Aggregate by code
    code_spend: Dict[str, float] = {}
    code_qty: Dict[str, float] = {}
    cnt_by_code: Dict[str, int] = {}
    for it in items:
        c = str(it["code"])
        code_spend[c] = code_spend.get(c, 0.0) + float(it["line_total"])
        code_qty[c] = code_qty.get(c, 0.0) + float(it["qty"])
        cnt_by_code[c] = cnt_by_code.get(c, 0) + 1
    line_totals = np.array([float(it["line_total"]) for it in items], dtype=float
    qtys = np.array([float(it["qty"]) for it in items], dtype=float)
    sum_line_total = float(line_totals.sum())
    pdf_total = float(total) if total is not None else sum_line_total
    sorted_cost = np.sort(line_totals)[::-1]
    top1 = float(sorted_cost[0] / sum_line_total) if sum_line_total > 0 else 0.0
    top3 = float(sorted_cost[:3].sum() / sum_line_total) if sum_line_total > 0 el
    repeat_spend = sum(v for c, v in code_spend.items() if cnt_by_code.get(c, 0) 
    repeat_share = float(repeat_spend / sum_line_total) if sum_line_total > 0 els
    meta = dict(
        pdf_ok=1,
        pdf_n_lines=len(items),
        pdf_n_unique_codes=len(code_spend),
        pdf_sum_qty=float(qtys.sum()),
        pdf_mean_qty=float(qtys.mean()) if len(items) else 0.0,
        pdf_max_qty=float(qtys.max()) if len(items) else 0.0,
        pdf_sum_line_total=sum_line_total,
        pdf_mean_line_total=float(line_totals.mean()) if len(items) else 0.0,

        pdf_max_line_total=float(line_totals.max()) if len(items) else 0.0,
        pdf_std_line_total=float(line_totals.std()) if len(items) else 0.0,
        pdf_top1_cost_share=top1,
        pdf_top3_cost_share=top3,
        pdf_repeat_cost_share=repeat_share,
        pdf_total=pdf_total,
        pdf_total_diff=float(abs(sum_line_total - pdf_total)),
        pdf_insurance=(insurance.lower() if isinstance(insurance, str) else None)
        pdf_zip3=(str(zip3) if zip3 is not None else None),
        pdf_parse_notes="ok",
    )
    # Hashed token dicts (long-tail code features)
    tok_count: Dict[str, float] = {}
    tok_spend: Dict[str, float] = {}
    tok_qty: Dict[str, float] = {}
    for code, spend in code_spend.items():
        qty = float(code_qty.get(code, 0.0))
        toks = [
            f"code={code}",
            f"p1={code[0]}",
            f"p2={code[:2]}",
            f"p3={code[:3]}",
            f"type={'hcpcs' if code[0].isalpha() else 'cpt'}",
        ]
        for t in toks:
            tok_count[t] = tok_count.get(t, 0.0) + 1.0
            tok_spend[t] = tok_spend.get(t, 0.0) + float(spend)
            tok_qty[t] = tok_qty.get(t, 0.0) + qty
    # compress
    tok_count = {k: safe_log1p(v) for k, v in tok_count.items()}
    tok_spend = {k: safe_log1p(v) for k, v in tok_spend.items()}
    tok_qty = {k: safe_log1p(v) for k, v in tok_qty.items()}
    return ReceiptProfile(meta=meta, code_spend=code_spend, code_qty=code_qty,
                          tok_count=tok_count, tok_spend=tok_spend, tok_qty=tok_q

def load_cache(cache_path: Path) -> Dict[str, Any]:
    if cache_path.exists():
        try:
            obj = joblib.load(cache_path)
            if isinstance(obj, dict) and obj.get("version") == CACHE_VERSION and 
                return obj
        except Exception:
            pass
    return {"version": CACHE_VERSION, "data": {}}
def save_cache(cache_path: Path, cache_obj: Dict[str, Any]) -> None:
    try:
        joblib.dump(cache_obj, cache_path)
    except Exception as e:
        logging.warning(f"Cache save failed: {e}")
def build_receipt_artifacts(patient_ids: List[int], pdf_dir: Path, cache_path: Pa
    cache_obj = load_cache(cache_path)
    data: Dict[str, Any] = cache_obj.get("data", {})
    to_parse = [pid for pid in patient_ids if str(pid) not in data]
    if to_parse:
        logging.info(f"Parsing {len(to_parse):,} receipts (cache miss).")
    for pid in tqdm(to_parse, desc="Parsing receipts", ncols=100):
        prof = featurize_receipt(pid, pdf_dir)
        data[str(pid)] = {
            "meta": prof.meta,
            "code_spend": prof.code_spend,
            "code_qty": prof.code_qty,
            "tok_count": prof.tok_count,
            "tok_spend": prof.tok_spend,
            "tok_qty": prof.tok_qty,
        }
    cache_obj["data"] = data
    if to_parse:
        save_cache(cache_path, cache_obj)

    meta_rows = []
    profiles: Dict[int, Dict[str, Any]] = {}
    for pid in patient_ids:
        obj = data.get(str(pid), None)
        if obj is None:
            prof = featurize_receipt(pid, pdf_dir)
            obj = {
                "meta": prof.meta,
                "code_spend": prof.code_spend,
                "code_qty": prof.code_qty,
                "tok_count": prof.tok_count,
                "tok_spend": prof.tok_spend,
                "tok_qty": prof.tok_qty,
            }
        meta_rows.append({"patient_id": pid, **obj["meta"]})
        profiles[pid] = obj
    return pd.DataFrame(meta_rows), profiles
# =========================
# BASE FEATURES (NON-TE)
# =========================
def add_top_code_pivots(
    df_all: pd.DataFrame,
    profiles: Dict[int, Dict[str, Any]],
    train_patient_ids: np.ndarray,
    top_k: int,
) -> pd.DataFrame:
    """
    Simple pivot features for top-K codes by TRAIN frequency:
      - cpt_cnt_{code}   (presence)
      - cpt_share_{code} (spend share vs prior_ed_cost_5y_usd)
    """
    code_freq: Dict[str, int] = {}
    for pid in train_patient_ids.tolist():
        code_spend = profiles.get(int(pid), {}).get("code_spend", {}) or {}
        for c in code_spend.keys():
            code_freq[c] = code_freq.get(c, 0) + 1

    if not code_freq:
        logging.warning("No codes found for top-code pivots; skipping.")
        return df_all
    top_codes = [c for c, _ in sorted(code_freq.items(), key=lambda kv: kv[1], rev
    logging.info(f"Adding top-code pivots: K={len(top_codes)}")
    prior_cost = df_all.set_index("patient_id")["prior_ed_cost_5y_usd"].astype(fl
    # initialize cols
    for c in top_codes:
        df_all[f"cpt_cnt_{c}"] = 0.0
        df_all[f"cpt_share_{c}"] = 0.0
    # fill
    pids = df_all["patient_id"].astype(int).tolist()
    for i, pid in enumerate(pids):
        code_spend = profiles.get(pid, {}).get("code_spend", {}) or {}
        denom = float(prior_cost.get(pid, 0.0)) + 1e-6
        for c in top_codes:
            sp = float(code_spend.get(c, 0.0))
            if sp > 0:
                df_all.at[i, f"cpt_cnt_{c}"] = 1.0
                df_all.at[i, f"cpt_share_{c}"] = sp / denom
    return df_all
def build_base_sparse_matrix(df_all: pd.DataFrame, profiles: Dict[int, Dict[str, A
    df = df_all.copy()
    # derived tabular
    df["prior_cost_per_visit"] = df["prior_ed_cost_5y_usd"] / (df["prior_ed_visit
    df["log_prior_cost"] = np.log1p(df["prior_ed_cost_5y_usd"].astype(float))
    df["log_prior_visits"] = np.log1p(df["prior_ed_visits_5y"].astype(float))
    # normalize cats
    for c in ["pdf_insurance", "pdf_zip3", "insurance", "zip3", "sex", "primary_c
        if c in df.columns:

            df[c] = df[c].astype("string").fillna("missing")
    exclude = {"patient_id", "ed_cost_next3y_usd", "is_train"}
    cat_cols = [c for c in ["primary_chronic", "sex", "insurance", "zip3", "pdf_i
                if c in df.columns]
    num_cols = [c for c in df.columns
                if c not in exclude and c not in cat_cols and pd.api.types.is_num
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), nu
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    X_tab = preprocessor.fit_transform(df)
    X_tab = X_tab.tocsr() if hasattr(X_tab, "tocsr") else csr_matrix(X_tab)
    blocks = [X_tab]
    artifacts: Dict[str, Any] = {"preprocessor": preprocessor, "num_cols": num_co
    if USE_HASHED_DICT_FEATURES:
        ids = df["patient_id"].astype(int).tolist()
        dict_count = [profiles.get(pid, {}).get("tok_count", {}) or {} for pid in 
        dict_spend = [profiles.get(pid, {}).get("tok_spend", {}) or {} for pid in 
        dict_qty = [profiles.get(pid, {}).get("tok_qty", {}) or {} for pid in ids
        hasher_count = FeatureHasher(n_features=HASH_DIM_COUNT, input_type="dict"
        hasher_spend = FeatureHasher(n_features=HASH_DIM_SPEND, input_type="dict"
        hasher_qty = FeatureHasher(n_features=HASH_DIM_QTY, input_type="dict", dty
        blocks.extend([
            hasher_count.transform(dict_count),
            hasher_spend.transform(dict_spend),
            hasher_qty.transform(dict_qty),
        ])
        artifacts.update({"hasher_count": hasher_count, "hasher_spend": hasher_sp

    X = hstack(blocks, format="csr")
    artifacts["n_features"] = X.shape[1]
    return X, artifacts
# =========================
# TARGET ENCODING (CV-SAFE)
# =========================
@dataclass
class TokenStats:
    cnt: int
    mean: float
    median: float
    mean_smooth: float
    median_adj: float
def code_to_prefix2(code: str) -> str:
    return code[:2] if len(code) >= 2 else code
def build_token_stats_for_fold(
    train_pids: List[int],
    y_log: np.ndarray,
    pid_to_row: Dict[int, int],
    profiles: Dict[int, Dict[str, Any]],
) -> Tuple[Dict[str, TokenStats], Dict[str, TokenStats], float, float]:
    values_code: Dict[str, List[float]] = {}
    values_p2: Dict[str, List[float]] = {}
    ys = []
    for pid in train_pids:
        r = pid_to_row.get(pid, None)
        if r is None:
            continue
        v = float(y_log[r])
        ys.append(v)
        code_spend = profiles.get(pid, {}).get("code_spend", {}) or {}
        for c in code_spend.keys():

            values_code.setdefault(c, []).append(v)
            values_p2.setdefault(code_to_prefix2(c), []).append(v)
    global_mean = float(np.mean(ys)) if ys else 0.0
    global_median = float(np.median(ys)) if ys else 0.0
    def to_stats(values_map: Dict[str, List[float]]) -> Dict[str, TokenStats]:
        out: Dict[str, TokenStats] = {}
        for tok, arr in values_map.items():
            a = np.asarray(arr, dtype=float)
            cnt = int(a.size)
            m = float(np.mean(a))
            med = float(np.median(a))
            mean_smooth = float((cnt * m + TE_SMOOTH * global_mean) / (cnt + TE_SM
            median_adj = med if cnt >= TE_MIN_COUNT_MEDIAN else global_median
            out[tok] = TokenStats(cnt=cnt, mean=m, median=med, mean_smooth=mean_s
        return out
    return to_stats(values_code), to_stats(values_p2), global_mean, global_median
def te_features_for_patients(
    pids: List[int],
    profiles: Dict[int, Dict[str, Any]],
    stats_code: Dict[str, TokenStats],
    stats_p2: Dict[str, TokenStats],
    global_mean: float,
    global_median: float,
) -> np.ndarray:
    """
    16 numeric TE features per patient (exact codes + prefix2), spend-weight aggr
    """
    feats = np.zeros((len(pids), 16), dtype=np.float32)
    for i, pid in enumerate(pids):
        code_spend = profiles.get(pid, {}).get("code_spend", {}) or {}
        if not code_spend:
            base = np.array([global_mean, global_median, global_mean, global_mean
            feats[i, 0:8] = base
            feats[i, 8:16] = base

            continue
        # ---- exact codes ----
        codes = list(code_spend.keys())
        spends = np.array([float(code_spend[c]) for c in codes], dtype=float)
        ssum = float(spends.sum())
        weights = (np.ones_like(spends) / max(1, spends.size)) if ssum <= 0 else 
        vals_mean, vals_med, cnts = [], [], []
        known = 0
        for c in codes:
            st = stats_code.get(c, None)
            if st is None:
                vals_mean.append(global_mean); vals_med.append(global_median); cn
            else:
                vals_mean.append(st.mean_smooth); vals_med.append(st.median_adj); 
        vals_mean = np.array(vals_mean, dtype=float)
        vals_med = np.array(vals_med, dtype=float)
        cnts = np.array(cnts, dtype=float)
        top_idx = int(np.argmax(weights)) if weights.size else 0
        feats[i, 0:8] = np.array([
            float(np.sum(weights * vals_mean)),
            float(np.sum(weights * vals_med)),
            float(np.mean(vals_mean)),
            float(np.max(vals_mean)),
            float(np.min(vals_mean)),
            float(vals_med[top_idx] if vals_med.size else global_median),
            float(known / max(1, len(codes))),
            float(np.mean(np.log1p(cnts))) if cnts.size else 0.0,
        ], dtype=np.float32)
        # ---- prefix2 ----
        p2_map: Dict[str, float] = {}
        for c, sp in code_spend.items():
            p = code_to_prefix2(c)
            p2_map[p] = p2_map.get(p, 0.0) + float(sp)
        p2s = list(p2_map.keys())

        p2sp = np.array([float(p2_map[p]) for p in p2s], dtype=float)
        p2sum = float(p2sp.sum())
        p2w = (np.ones_like(p2sp) / max(1, p2sp.size)) if p2sum <= 0 else (p2sp / 
        p2_mean, p2_med, p2_cnt = [], [], []
        known2 = 0
        for p in p2s:
            st = stats_p2.get(p, None)
            if st is None:
                p2_mean.append(global_mean); p2_med.append(global_median); p2_cnt
            else:
                p2_mean.append(st.mean_smooth); p2_med.append(st.median_adj); p2_
        p2_mean = np.array(p2_mean, dtype=float)
        p2_med = np.array(p2_med, dtype=float)
        p2_cnt = np.array(p2_cnt, dtype=float)
        top2 = int(np.argmax(p2w)) if p2w.size else 0
        feats[i, 8:16] = np.array([
            float(np.sum(p2w * p2_mean)),
            float(np.sum(p2w * p2_med)),
            float(np.mean(p2_mean)),
            float(np.max(p2_mean)),
            float(np.min(p2_mean)),
            float(p2_med[top2] if p2_med.size else global_median),
            float(known2 / max(1, len(p2s))),
            float(np.mean(np.log1p(p2_cnt))) if p2_cnt.size else 0.0,
        ], dtype=np.float32)
    return feats
def make_strat_labels(train_df: pd.DataFrame) -> np.ndarray:
    chronic = train_df["primary_chronic"].astype(str)
    cost_bin = pd.qcut(train_df["prior_ed_cost_5y_usd"], q=10, duplicates="drop")
    return (chronic + "_" + cost_bin).values
# =========================
# MODELS & COMPAT

# =========================
def mae_original_from_log(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> Tupl
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return "mae_usd", float(np.mean(np.abs(y_true - y_pred))), False
def predict_xgb(model: XGBRegressor, X) -> np.ndarray:
    """
    Predict using best_iteration if available, otherwise normal predict.
    """
    best_iter = getattr(model, "best_iteration", None)
    if best_iter is None:
        try:
            best_iter = model.get_booster().best_iteration
        except Exception:
            best_iter = None
    if best_iter is None:
        return model.predict(X)
    try:
        return model.predict(X, iteration_range=(0, int(best_iter) + 1))
    except TypeError:
        return model.predict(X)
def fit_xgb_compat(
    X_tr, y_tr, X_va, y_va,
    base_params: Dict[str, Any],
    random_state: int,
    early_stopping_rounds: int,
) -> XGBRegressor:
    """
    XGBoost compatibility wrapper:
    - Some versions want early_stopping_rounds in constructor (newer).
    - Some accept it in fit() (older).
    - If neither, try callbacks.
    """
    params = dict(base_params)

    params.pop("random_state", None)
    es_in_ctor = False
    try:
        mdl = XGBRegressor(**params, random_state=random_state, early_stopping_ro
        es_in_ctor = True
    except TypeError:
        mdl = XGBRegressor(**params, random_state=random_state)
    fit_sig = inspect.signature(mdl.fit).parameters
    fit_kwargs = dict(eval_set=[(X_va, y_va)], verbose=False)
    if not es_in_ctor:
        if "early_stopping_rounds" in fit_sig:
            fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
        elif "callbacks" in fit_sig:
            try:
                early_stop = xgb.callback.EarlyStopping(
                    rounds=early_stopping_rounds,
                    metric_name="mae",
                    data_name="validation_0",
                    save_best=True,
                )
                fit_kwargs["callbacks"] = [early_stop]
            except Exception:
                pass
    mdl.fit(X_tr, y_tr, **fit_kwargs)
    return mdl
def optimize_weights_mae(y_true: np.ndarray, preds: Dict[str, np.ndarray], step: 
    """
    Grid search weights on simplex to minimize MAE.
    """
    names = list(preds.keys())
    m = len(names)
    if m == 1:
        return {names[0]: 1.0}, mae(y_true, preds[names[0]])

    best_w: Dict[str, float] = {}
    best_mae = float("inf")
    grid = np.arange(0.0, 1.0 + 1e-9, step)
    if m == 2:
        a, b = names
        for w in grid:
            p = w * preds[a] + (1 - w) * preds[b]
            score = mae(y_true, p)
            if score < best_mae:
                best_mae = score
                best_w = {a: float(w), b: float(1 - w)}
        return best_w, best_mae
    if m == 3:
        a, b, c = names
        for w1 in grid:
            for w2 in grid:
                w3 = 1.0 - w1 - w2
                if w3 < -1e-9:
                    continue
                w3 = max(0.0, w3)
                p = w1 * preds[a] + w2 * preds[b] + w3 * preds[c]
                score = mae(y_true, p)
                if score < best_mae:
                    best_mae = score
                    best_w = {a: float(w1), b: float(w2), c: float(w3)}
        return best_w, best_mae
    if m == 4:
        a, b, c, d = names
        for w1 in grid:
            for w2 in grid:
                for w3 in grid:
                    w4 = 1.0 - w1 - w2 - w3
                    if w4 < -1e-9:
                        continue
                    w4 = max(0.0, w4)
                    p = w1 * preds[a] + w2 * preds[b] + w3 * preds[c] + w4 * pred
                    score = mae(y_true, p)

                    if score < best_mae:
                        best_mae = score
                        best_w = {a: float(w1), b: float(w2), c: float(w3), d: fl
        return best_w, best_mae
    raise ValueError("Too many models for this simple grid search")
# =========================
# MAIN
# =========================
def main() -> None:
    setup_logging()
    logging.info(f"XGBoost version: {getattr(xgb, '__version__', 'unknown')}")
    logging.info("=== Iteration 2 (Fixed): TE CPT Risk + Ensemble ===")
    # I/O checks
    for p in [TRAIN_CSV, TEST_CSV]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")
    if not PDF_DIR.exists():
        raise FileNotFoundError(f"Missing PDF directory: {PDF_DIR}")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    # Optional patients.csv
    if PATIENTS_CSV.exists():
        patients = pd.read_csv(PATIENTS_CSV)
        if "patient_id" in patients.columns:
            train_df = train_df.merge(patients, on="patient_id", how="left")
            test_df = test_df.merge(patients, on="patient_id", how="left")
            logging.info("Merged patients.csv")
    train_df["is_train"] = 1
    test_df["is_train"] = 0
    df_all = pd.concat([train_df, test_df], ignore_index=True)
    # Parse receipts (cached)
    all_pids = df_all["patient_id"].astype(int).unique().tolist()

    all_pids.sort()
    receipt_meta_df, profiles = build_receipt_artifacts(all_pids, PDF_DIR, CACHE_
    df_all = df_all.merge(receipt_meta_df, on="patient_id", how="left")
    # Optional top-code pivots
    if USE_TOP_CODE_PIVOTS:
        df_all = add_top_code_pivots(
            df_all=df_all,
            profiles=profiles,
            train_patient_ids=train_df["patient_id"].astype(int).values,
            top_k=TOP_K_CODES,
        )
    # Base sparse matrix (no TE)
    X_all_base, _ = build_base_sparse_matrix(df_all, profiles)
    is_train = df_all["is_train"].values.astype(int) == 1
    X_train_base = X_all_base[is_train]
    X_test_base = X_all_base[~is_train]
    y = train_df["ed_cost_next3y_usd"].values.astype(np.float32)
    y_log = np.log1p(y)
    train_pids = train_df["patient_id"].astype(int).tolist()
    pid_to_row = {pid: i for i, pid in enumerate(train_pids)}
    test_pids = test_df["patient_id"].astype(int).tolist()
    # CatBoost base frames
    df_cb_all = df_all.drop(columns=["ed_cost_next3y_usd"], errors="ignore").copy
    for c in ["primary_chronic", "sex", "insurance", "zip3", "pdf_insurance", "pd
        if c in df_cb_all.columns:
            df_cb_all[c] = df_cb_all[c].astype(str).fillna("missing")
    df_cb_all["prior_cost_per_visit"] = df_cb_all["prior_ed_cost_5y_usd"] / (df_c
    df_cb_all["log_prior_cost"] = np.log1p(df_cb_all["prior_ed_cost_5y_usd"].asty
    df_cb_all["log_prior_visits"] = np.log1p(df_cb_all["prior_ed_visits_5y"].asty
    df_train_base = df_cb_all[is_train].reset_index(drop=True)
    df_test_base = df_cb_all[~is_train].reset_index(drop=True)

    cb_cat_cols = [c for c in ["primary_chronic", "sex", "insurance", "zip3", "pd
                   if c in df_train_base.columns]
    cb_cat_idx = [df_train_base.columns.get_loc(c) for c in cb_cat_cols]
    # CV splits
    strat_labels = make_strat_labels(train_df)
    splits: List[Tuple[int, int, np.ndarray, np.ndarray]] = []
    for rep, seed in enumerate(REPEAT_SEEDS):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), strat_
            splits.append((rep, fold, tr_idx, va_idx))
    logging.info(f"Total CV splits: {len(splits)}")
    # Model toggles
    model_names: List[str] = []
    if USE_LGB_L1: model_names.append("lgb_l1")
    if USE_LGB_QLOG: model_names.append("lgb_qlog")
    if USE_XGB_L1: model_names.append("xgb_l1")
    if USE_CAT_MAE: model_names.append("cat_mae")
    oof_sum = {n: np.zeros(len(y), dtype=np.float64) for n in model_names}
    oof_cnt = {n: np.zeros(len(y), dtype=np.float64) for n in model_names}
    test_sum = {n: np.zeros(len(test_df), dtype=np.float64) for n in model_names}
    test_cnt = {n: 0 for n in model_names}
    # Hyperparams
    lgb_l1_params = dict(
        objective="regression_l1",
        learning_rate=0.03,
        n_estimators=12000,
        num_leaves=96,
        min_child_samples=25,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_lambda=1.5,
        max_bin=255,
        n_jobs=-1,
        verbose=-1,
    )

    lgb_qlog_params = dict(
        objective="quantile",
        alpha=0.5,
        learning_rate=0.03,
        n_estimators=16000,
        num_leaves=128,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.70,
        reg_lambda=1.2,
        max_bin=255,
        n_jobs=-1,
        verbose=-1,
    )
    xgb_base_params = dict(
        objective="reg:absoluteerror",
        n_estimators=14000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        min_child_weight=1.0,
        tree_method="hist",
        eval_metric="mae",
        n_jobs=-1,
        verbosity=0,
    )
    cat_params = dict(
        loss_function="MAE",
        eval_metric="MAE",
        iterations=20000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=6.0,
        random_strength=0.5,

        bagging_temperature=0.8,
        od_type="Iter",
        od_wait=EARLY_STOPPING,
        task_type=CAT_TASK_TYPE,
        verbose=False,
    )
    te_cols = [
        f"{TE_FEATURE_PREFIX}code_wmean_mean",
        f"{TE_FEATURE_PREFIX}code_wmean_median",
        f"{TE_FEATURE_PREFIX}code_unw_mean",
        f"{TE_FEATURE_PREFIX}code_max_mean",
        f"{TE_FEATURE_PREFIX}code_min_mean",
        f"{TE_FEATURE_PREFIX}code_top_median",
        f"{TE_FEATURE_PREFIX}code_known_frac",
        f"{TE_FEATURE_PREFIX}code_avg_log_cnt",
        f"{TE_FEATURE_PREFIX}p2_wmean_mean",
        f"{TE_FEATURE_PREFIX}p2_wmean_median",
        f"{TE_FEATURE_PREFIX}p2_unw_mean",
        f"{TE_FEATURE_PREFIX}p2_max_mean",
        f"{TE_FEATURE_PREFIX}p2_min_mean",
        f"{TE_FEATURE_PREFIX}p2_top_median",
        f"{TE_FEATURE_PREFIX}p2_known_frac",
        f"{TE_FEATURE_PREFIX}p2_avg_log_cnt",
    ]
    # CV loop
    for rep, fold, tr_idx, va_idx in tqdm(splits, desc="CV folds", ncols=110):
        tr_pids = [train_pids[i] for i in tr_idx.tolist()]
        va_pids = [train_pids[i] for i in va_idx.tolist()]
        stats_code, stats_p2, gmean, gmed = build_token_stats_for_fold(tr_pids, y_
        te_tr = te_features_for_patients(tr_pids, profiles, stats_code, stats_p2, 
        te_va = te_features_for_patients(va_pids, profiles, stats_code, stats_p2, 
        te_te = te_features_for_patients(test_pids, profiles, stats_code, stats_p
        X_tr = hstack([X_train_base[tr_idx], csr_matrix(te_tr)], format="csr")
        X_va = hstack([X_train_base[va_idx], csr_matrix(te_va)], format="csr")
        X_te = hstack([X_test_base, csr_matrix(te_te)], format="csr")

        y_tr = y[tr_idx]
        y_va = y[va_idx]
        # LightGBM L1
        if USE_LGB_L1:
            mdl = LGBMRegressor(**lgb_l1_params, random_state=1000 * rep + fold)
            mdl.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric="mae",
                callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
            )
            best_it = int(getattr(mdl, "best_iteration_", lgb_l1_params["n_estimat
            p_va = mdl.predict(X_va, num_iteration=best_it)
            p_te = mdl.predict(X_te, num_iteration=best_it)
            if CLIP_AT_ZERO:
                p_va = np.maximum(p_va, 0.0); p_te = np.maximum(p_te, 0.0)
            oof_sum["lgb_l1"][va_idx] += p_va; oof_cnt["lgb_l1"][va_idx] += 1.0
            test_sum["lgb_l1"] += p_te; test_cnt["lgb_l1"] += 1
        # LightGBM Quantile on log1p(y)
        if USE_LGB_QLOG:
            mdl = LGBMRegressor(**lgb_qlog_params, random_state=2000 * rep + fold
            mdl.fit(
                X_tr, np.log1p(y_tr),
                eval_set=[(X_va, np.log1p(y_va))],
                eval_metric=mae_original_from_log,
                callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
            )
            best_it = int(getattr(mdl, "best_iteration_", lgb_qlog_params["n_esti
            p_va = np.expm1(mdl.predict(X_va, num_iteration=best_it))
            p_te = np.expm1(mdl.predict(X_te, num_iteration=best_it))
            if CLIP_AT_ZERO:
                p_va = np.maximum(p_va, 0.0); p_te = np.maximum(p_te, 0.0)
            oof_sum["lgb_qlog"][va_idx] += p_va; oof_cnt["lgb_qlog"][va_idx] += 1
            test_sum["lgb_qlog"] += p_te; test_cnt["lgb_qlog"] += 1
        # XGBoost L1 (fixed early stopping)
        if USE_XGB_L1:

            mdl = fit_xgb_compat(
                X_tr, y_tr, X_va, y_va,
                base_params=xgb_base_params,
                random_state=3000 * rep + fold,
                early_stopping_rounds=EARLY_STOPPING,
            )
            p_va = predict_xgb(mdl, X_va)
            p_te = predict_xgb(mdl, X_te)
            if CLIP_AT_ZERO:
                p_va = np.maximum(p_va, 0.0); p_te = np.maximum(p_te, 0.0)
            oof_sum["xgb_l1"][va_idx] += p_va; oof_cnt["xgb_l1"][va_idx] += 1.0
            test_sum["xgb_l1"] += p_te; test_cnt["xgb_l1"] += 1
        # CatBoost MAE (GPU -> CPU fallback)
        if USE_CAT_MAE:
            te_train_full = np.zeros((len(train_df), 16), dtype=np.float32)
            te_train_full[tr_idx, :] = te_tr
            te_train_full[va_idx, :] = te_va
            df_train_cb = df_train_base.copy()
            df_test_cb = df_test_base.copy()
            for j, col in enumerate(te_cols):
                df_train_cb[col] = te_train_full[:, j]
                df_test_cb[col] = te_te[:, j]
            tr_pool = Pool(df_train_cb.iloc[tr_idx], label=y_tr, cat_features=cb_
            va_pool = Pool(df_train_cb.iloc[va_idx], label=y_va, cat_features=cb_
            te_pool = Pool(df_test_cb, cat_features=cb_cat_idx)
            try:
                mdl = CatBoostRegressor(**cat_params, random_seed=4000 * rep + fo
                mdl.fit(tr_pool, eval_set=va_pool, use_best_model=True)
            except Exception as e:
                logging.warning(f"CatBoost({CAT_TASK_TYPE}) failed on rep={rep}, 
                cat_cpu = dict(cat_params); cat_cpu["task_type"] = "CPU"
                mdl = CatBoostRegressor(**cat_cpu, random_seed=4000 * rep + fold)
                mdl.fit(tr_pool, eval_set=va_pool, use_best_model=True)
            p_va = mdl.predict(va_pool)
            p_te = mdl.predict(te_pool)

            if CLIP_AT_ZERO:
                p_va = np.maximum(p_va, 0.0); p_te = np.maximum(p_te, 0.0)
            oof_sum["cat_mae"][va_idx] += p_va; oof_cnt["cat_mae"][va_idx] += 1.0
            test_sum["cat_mae"] += p_te; test_cnt["cat_mae"] += 1
    # finalize per-model oof/test
    oof_preds: Dict[str, np.ndarray] = {}
    test_preds: Dict[str, np.ndarray] = {}
    for name in model_names:
        denom = np.maximum(oof_cnt[name], 1.0)
        oof_preds[name] = (oof_sum[name] / denom).astype(np.float32)
        test_preds[name] = (test_sum[name] / max(1, test_cnt[name])).astype(np.fl
        logging.info(f"OOF MAE | {name}: {mae(y, oof_preds[name]):.4f}")
    # blend weights (OOF)
    weights, best_mae = optimize_weights_mae(y, oof_preds, step=0.05)
    logging.info(f"Best OOF blend MAE (pre-bias): {best_mae:.4f} | weights={weight
    # blend + MAE-optimal bias (median residual)
    oof_blend = np.zeros(len(y), dtype=np.float64)
    test_blend = np.zeros(len(test_df), dtype=np.float64)
    for name, w in weights.items():
        oof_blend += w * oof_preds[name]
        test_blend += w * test_preds[name]
    bias = float(np.median(y - oof_blend))
    oof_final = oof_blend + bias
    test_final = test_blend + bias
    if CLIP_AT_ZERO:
        oof_final = np.maximum(oof_final, 0.0)
        test_final = np.maximum(test_final, 0.0)
    logging.info(f"Final OOF MAE (blend + median-bias): {mae(y, oof_final):.4f} | 
    # save submission
    sub = pd.DataFrame({
        "patient_id": test_df["patient_id"].astype(int).values,
        "ed_cost_next3y_usd": test_final.astype(np.float32),

If you hit a different XGBoost constructor error next (older versions), this script already
handles it by moving early stopping into .fit()  or callbacks automatically. The behavior
change (constructor vs fit) is documented in XGBoost’s API notes. 
    })
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(SUBMISSION_PATH, index=False)
    logging.info(f"Saved: {SUBMISSION_PATH}")
if __name__ == "__main__":
    main()
XGBoost Docum… +2
```

### C04 — ITERATION 2 (GPU-Fast) — Fixes GPU usage + reduces runtime while keeping Level-2 (pages 114-135)
```python
# -*- coding: utf-8 -*-
"""
ITERATION 2 (GPU-Fast) — Fixes GPU usage + reduces runtime while keeping Level-2 
Key changes vs your slow run:
- XGBoost now uses GPU by setting device="cuda" (tree_method can remain "hist").  
- Default CV uses 2 repeats (10 folds total) instead of 3 repeats (15 folds).
- Default ensemble is GPU-only: XGBoost GPU + CatBoost GPU.
- Keeps CV-safe Target Encoding (TE) of CPT/HCPCS code risk, aggregated by spend.
===========================================================
INSTALL (run once)
===========================================================
pip install -U pandas numpy scipy scikit-learn xgboost catboost pymupdf joblib tq
===========================================================
Paths:
D:\AgentDs\agent_ds_healthcare\ed_cost_train.csv
D:\AgentDs\agent_ds_healthcare\ed_cost_test.csv
D:\AgentDs\agent_ds_healthcare\patients.csv (optional)
D:\AgentDs\agent_ds_healthcare\receipts_pdf\receipt_{patient_id}.pdf
Output:
D:\AgentDs\agent_ds_healthcare\submission.csv
===========================================================
"""
from __future__ import annotations
import re
import math
import time
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import fitz  # PyMuPDF
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import xgboost as xgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor, Pool
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# =========================
# CONFIG
# =========================
BASE_DIR = Path(r"D:\AgentDs\agent_ds_healthcare")
TRAIN_CSV = BASE_DIR / "ed_cost_train.csv"
TEST_CSV = BASE_DIR / "ed_cost_test.csv"
PATIENTS_CSV = BASE_DIR / "patients.csv"          # optional
PDF_DIR = BASE_DIR / "receipts_pdf"
SUBMISSION_PATH = BASE_DIR / "submission.csv"
CACHE_PATH = BASE_DIR / "cache_receipts_iter2_gpu_fast.joblib"
CACHE_VERSION = 21
# Speed / rigor tradeoff
FAST_MODE = True   # True -> 2 repeats (10 folds). False -> 3 repeats (15 folds)
N_FOLDS = 5

REPEAT_SEEDS = [2025, 2026] if FAST_MODE else [2024, 2025, 2026]
EARLY_STOPPING = 200
# Target encoding
TE_SMOOTH = 20.0
TE_MIN_COUNT_MEDIAN = 5
# Use GPU
XGB_DEVICE = "cuda"       # use "cuda:0" if you want explicit device
CAT_TASK_TYPE = "GPU"     # will auto-fallback to CPU if GPU fails
# Prediction post-process
CLIP_AT_ZERO = True
# Blending
BLEND_STEP = 0.02         # grid step for weight search on OOF
# =========================
# LOGGING
# =========================
def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | 
# =========================
# HELPERS
# =========================
_money_re = re.compile(r"^\d{1,3}(?:,\d{3})*(?:\.\d{2})$|^\d+(?:\.\d{2})$")
_code_re = re.compile(r"^(?:[A-Z]\d{4}|\d{4,5})$")
def is_money(tok: str) -> bool:
    return bool(_money_re.match(tok))
def money_to_float(tok: str) -> float:
    return float(tok.replace(",", ""))
def is_code(tok: str) -> bool:
    return bool(_code_re.match(tok))
def safe_log1p(x: float) -> float:

    return math.log1p(max(0.0, float(x)))
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))
def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_out
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)
# =========================
# PDF PARSING
# =========================
def cluster_words_into_lines(words: List[Tuple], y_tol: float = 1.5) -> List[List
    if not words:
        return []
    words_sorted = sorted(words, key=lambda w: (w[1], w[0]))
    lines: List[List[str]] = []
    cur: List[Tuple[float, str]] = []
    cur_y: Optional[float] = None
    for w in words_sorted:
        x0, y0 = float(w[0]), float(w[1])
        tok = str(w[4])
        if cur_y is None or abs(y0 - cur_y) <= y_tol:
            cur.append((x0, tok))
            cur_y = y0 if cur_y is None else (0.85 * cur_y + 0.15 * y0)
        else:
            lines.append([t for _, t in sorted(cur, key=lambda z: z[0])])
            cur = [(x0, tok)]
            cur_y = y0
    if cur:
        lines.append([t for _, t in sorted(cur, key=lambda z: z[0])])
    return lines
def parse_header_fields(text: str) -> Tuple[Optional[int], Optional[str], Optiona
    pid = None

    zip3 = None
    insurance = None
    m = re.search(r"Patient ID:\s*(\d+)", text)
    if m:
        pid = int(m.group(1))
    m = re.search(r"ZIP3:\s*([0-9]{3})\s+Insurance:\s*([A-Za-z_]+)", text)
    if m:
        zip3 = str(m.group(1))
        insurance = str(m.group(2)).lower()
    return pid, zip3, insurance
def parse_items_from_words(words: List[Tuple]) -> Tuple[List[Dict[str, Any]], Opt
    items: List[Dict[str, Any]] = []
    total: Optional[float] = None
    lines = cluster_words_into_lines(words, y_tol=1.5)
    if not lines:
        return items, total
    header_idx = None
    for i, toks in enumerate(lines):
        low = [t.lower() for t in toks]
        if ("code" in low) and ("description" in low) and ("qty" in low):
            header_idx = i
            break
    if header_idx is None:
        return items, total
    for toks in lines[header_idx + 1:]:
        if not toks:
            continue
        if toks[0].upper() == "TOTAL":
            for t in reversed(toks[1:]):
                if is_money(t):
                    total = money_to_float(t)
                    break
            break
        if len(toks) >= 5 and is_code(toks[0]) and toks[-3].isdigit() and is_money
            items.append({

                "code": toks[0],
                "qty": int(toks[-3]),
                "unit": money_to_float(toks[-2]),
                "line_total": money_to_float(toks[-1]),
                "desc": " ".join(toks[1:-3]),
            })
    return items, total
def parse_items_fallback_text(text: str) -> Tuple[List[Dict[str, Any]], Optional[
    items: List[Dict[str, Any]] = []
    total: Optional[float] = None
    lines = [ln.strip() for ln in (text.splitlines() if text else []) if ln.strip
    if not lines:
        return items, total
    start_idx = None
    for i in range(len(lines) - 4):
        if lines[i].lower() == "code" and lines[i+1].lower() == "description" and 
            start_idx = i + 5
            break
        if "code" in lines[i].lower() and "description" in lines[i].lower() and "
            start_idx = i + 1
            break
    if start_idx is None:
        return items, total
    i = start_idx
    while i < len(lines):
        if lines[i].upper().startswith("TOTAL"):
            parts = lines[i].split()
            if len(parts) >= 2 and is_money(parts[-1]):
                total = money_to_float(parts[-1])
            elif i+1 < len(lines) and is_money(lines[i+1]):
                total = money_to_float(lines[i+1])
            break
        if i+4 < len(lines) and is_code(lines[i]) and lines[i+2].isdigit() and is_
            items.append({
                "code": lines[i],

                "desc": lines[i+1],
                "qty": int(lines[i+2]),
                "unit": money_to_float(lines[i+3]),
                "line_total": money_to_float(lines[i+4]),
            })
            i += 5
        else:
            i += 1
    return items, total
@dataclass
class ReceiptProfile:
    meta: Dict[str, Any]
    code_spend: Dict[str, float]
def featurize_receipt(patient_id: int, pdf_dir: Path) -> ReceiptProfile:
    pdf_path = pdf_dir / f"receipt_{patient_id}.pdf"
    empty_meta = dict(
        pdf_ok=0, pdf_n_lines=0, pdf_n_unique_codes=0,
        pdf_sum_line_total=0.0, pdf_top1_cost_share=0.0, pdf_top3_cost_share=0.0,
        pdf_repeat_cost_share=0.0, pdf_total=0.0, pdf_total_diff=0.0,
        pdf_insurance=None, pdf_zip3=None, pdf_parse_notes="missing_pdf",
    )
    if not pdf_path.exists():
        return ReceiptProfile(meta=empty_meta, code_spend={})
    try:
        with fitz.open(pdf_path) as doc:
            full_text = ""
            items: List[Dict[str, Any]] = []
            total: Optional[float] = None
            zip3 = None
            insurance = None
            for page in doc:
                text = page.get_text("text") or ""
                full_text += text + "\n"

                words = page.get_text("words") or []
                items_p, total_p = parse_items_from_words(words)
                if not items_p:
                    items_p, total_p = parse_items_fallback_text(text)
                if items_p:
                    items.extend(items_p)
                if total_p is not None:
                    total = total_p
            _, zip3, insurance = parse_header_fields(full_text)
    except Exception as e:
        empty_meta["pdf_parse_notes"] = f"open_error:{type(e).__name__}"
        return ReceiptProfile(meta=empty_meta, code_spend={})
    if not items:
        empty_meta["pdf_parse_notes"] = "no_items_parsed"
        empty_meta["pdf_zip3"] = zip3
        empty_meta["pdf_insurance"] = insurance
        return ReceiptProfile(meta=empty_meta, code_spend={})
    code_spend: Dict[str, float] = {}
    cnt_by_code: Dict[str, int] = {}
    for it in items:
        c = str(it["code"])
        code_spend[c] = code_spend.get(c, 0.0) + float(it["line_total"])
        cnt_by_code[c] = cnt_by_code.get(c, 0) + 1
    line_totals = np.array([float(it["line_total"]) for it in items], dtype=float
    sum_line_total = float(line_totals.sum())
    pdf_total = float(total) if total is not None else sum_line_total
    sorted_cost = np.sort(line_totals)[::-1]
    top1 = float(sorted_cost[0] / sum_line_total) if sum_line_total > 0 else 0.0
    top3 = float(sorted_cost[:3].sum() / sum_line_total) if sum_line_total > 0 el
    repeat_spend = sum(v for c, v in code_spend.items() if cnt_by_code.get(c, 0) 
    repeat_share = float(repeat_spend / sum_line_total) if sum_line_total > 0 els

    meta = dict(
        pdf_ok=1,
        pdf_n_lines=len(items),
        pdf_n_unique_codes=len(code_spend),
        pdf_sum_line_total=sum_line_total,
        pdf_top1_cost_share=top1,
        pdf_top3_cost_share=top3,
        pdf_repeat_cost_share=repeat_share,
        pdf_total=pdf_total,
        pdf_total_diff=float(abs(sum_line_total - pdf_total)),
        pdf_insurance=(insurance.lower() if isinstance(insurance, str) else None)
        pdf_zip3=(str(zip3) if zip3 is not None else None),
        pdf_parse_notes="ok",
    )
    return ReceiptProfile(meta=meta, code_spend=code_spend)
def load_cache(cache_path: Path) -> Dict[str, Any]:
    if cache_path.exists():
        try:
            obj = joblib.load(cache_path)
            if isinstance(obj, dict) and obj.get("version") == CACHE_VERSION and 
                return obj
        except Exception:
            pass
    return {"version": CACHE_VERSION, "data": {}}
def save_cache(cache_path: Path, cache_obj: Dict[str, Any]) -> None:
    try:
        joblib.dump(cache_obj, cache_path)
    except Exception as e:
        logging.warning(f"Cache save failed: {e}")
def build_receipt_artifacts(patient_ids: List[int], pdf_dir: Path, cache_path: Pa
    cache_obj = load_cache(cache_path)
    data: Dict[str, Any] = cache_obj.get("data", {})

    to_parse = [pid for pid in patient_ids if str(pid) not in data]
    if to_parse:
        logging.info(f"Parsing {len(to_parse):,} receipts (cache miss).")
    for pid in tqdm(to_parse, desc="Parsing receipts", ncols=100):
        prof = featurize_receipt(pid, pdf_dir)
        data[str(pid)] = {"meta": prof.meta, "code_spend": prof.code_spend}
    cache_obj["data"] = data
    if to_parse:
        save_cache(cache_path, cache_obj)
    meta_rows = []
    profiles: Dict[int, Dict[str, Any]] = {}
    for pid in patient_ids:
        obj = data.get(str(pid), None)
        if obj is None:
            prof = featurize_receipt(pid, pdf_dir)
            obj = {"meta": prof.meta, "code_spend": prof.code_spend}
        meta_rows.append({"patient_id": pid, **obj["meta"]})
        profiles[pid] = obj
    return pd.DataFrame(meta_rows), profiles
# =========================
# BASE FEATURES (OHE for XGB)
# =========================
def build_base_sparse_matrix(df_all: pd.DataFrame) -> Tuple[csr_matrix, Dict[str, 
    df = df_all.copy()
    # derived tabular
    df["prior_cost_per_visit"] = df["prior_ed_cost_5y_usd"] / (df["prior_ed_visit
    df["log_prior_cost"] = np.log1p(df["prior_ed_cost_5y_usd"].astype(float))
    df["log_prior_visits"] = np.log1p(df["prior_ed_visits_5y"].astype(float))
    for c in ["primary_chronic", "sex", "insurance", "zip3", "pdf_insurance", "pd
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("missing")

    exclude = {"patient_id", "ed_cost_next3y_usd", "is_train"}
    cat_cols = [c for c in ["primary_chronic", "sex", "insurance", "zip3", "pdf_i
                if c in df.columns]
    num_cols = [c for c in df.columns if c not in exclude and c not in cat_cols a
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), nu
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    X = preprocessor.fit_transform(df)
    X = X.tocsr() if hasattr(X, "tocsr") else csr_matrix(X)
    return X, {"preprocessor": preprocessor, "num_cols": num_cols, "cat_cols": cat
# =========================
# TARGET ENCODING (CV-SAFE)
# =========================
@dataclass
class TokenStats:
    cnt: int
    mean_smooth: float
    median_adj: float
def code_to_prefix2(code: str) -> str:
    return code[:2] if len(code) >= 2 else code
def build_token_stats_for_fold(
    train_pids: List[int],
    y_log: np.ndarray,
    pid_to_row: Dict[int, int],
    profiles: Dict[int, Dict[str, Any]],
) -> Tuple[Dict[str, TokenStats], Dict[str, TokenStats], float, float]:
    vals_code: Dict[str, List[float]] = {}
    vals_p2: Dict[str, List[float]] = {}

    ys = []
    for pid in train_pids:
        r = pid_to_row.get(pid, None)
        if r is None:
            continue
        v = float(y_log[r])
        ys.append(v)
        code_spend = profiles.get(pid, {}).get("code_spend", {}) or {}
        for c in code_spend.keys():
            vals_code.setdefault(c, []).append(v)
            vals_p2.setdefault(code_to_prefix2(c), []).append(v)
    gmean = float(np.mean(ys)) if ys else 0.0
    gmed = float(np.median(ys)) if ys else 0.0
    def to_stats(m: Dict[str, List[float]]) -> Dict[str, TokenStats]:
        out = {}
        for tok, arr in m.items():
            a = np.asarray(arr, dtype=float)
            cnt = int(a.size)
            mean = float(np.mean(a))
            med = float(np.median(a))
            mean_smooth = float((cnt * mean + TE_SMOOTH * gmean) / (cnt + TE_SMOO
            median_adj = med if cnt >= TE_MIN_COUNT_MEDIAN else gmed
            out[tok] = TokenStats(cnt=cnt, mean_smooth=mean_smooth, median_adj=me
        return out
    return to_stats(vals_code), to_stats(vals_p2), gmean, gmed
def te_features_for_patients(
    pids: List[int],
    profiles: Dict[int, Dict[str, Any]],
    stats_code: Dict[str, TokenStats],
    stats_p2: Dict[str, TokenStats],
    gmean: float,
    gmed: float,
) -> np.ndarray:
    """

    16 TE features: exact code (8) + prefix2 (8), spend-weighted.
    """
    feats = np.zeros((len(pids), 16), dtype=np.float32)
    for i, pid in enumerate(pids):
        code_spend = profiles.get(pid, {}).get("code_spend", {}) or {}
        if not code_spend:
            base = np.array([gmean, gmed, gmean, gmean, gmean, gmed, 0.0, 0.0], dt
            feats[i, 0:8] = base
            feats[i, 8:16] = base
            continue
        codes = list(code_spend.keys())
        spends = np.array([float(code_spend[c]) for c in codes], dtype=float)
        ssum = float(spends.sum())
        w = (np.ones_like(spends) / max(1, spends.size)) if ssum <= 0 else (spend
        # exact
        means = []
        meds = []
        cnts = []
        known = 0
        for c in codes:
            st = stats_code.get(c)
            if st is None:
                means.append(gmean); meds.append(gmed); cnts.append(0)
            else:
                means.append(st.mean_smooth); meds.append(st.median_adj); cnts.ap
        means = np.asarray(means, dtype=float)
        meds = np.asarray(meds, dtype=float)
        cnts = np.asarray(cnts, dtype=float)
        top = int(np.argmax(w)) if w.size else 0
        feats[i, 0:8] = np.array([
            float(np.sum(w * means)),
            float(np.sum(w * meds)),
            float(np.mean(means)),
            float(np.max(means)),
            float(np.min(means)),
            float(meds[top] if meds.size else gmed),

            float(known / max(1, len(codes))),
            float(np.mean(np.log1p(cnts))) if cnts.size else 0.0,
        ], dtype=np.float32)
        # prefix2
        p2_map: Dict[str, float] = {}
        for c, sp in code_spend.items():
            p2 = code_to_prefix2(c)
            p2_map[p2] = p2_map.get(p2, 0.0) + float(sp)
        p2s = list(p2_map.keys())
        p2sp = np.asarray([p2_map[p] for p in p2s], dtype=float)
        p2sum = float(p2sp.sum())
        w2 = (np.ones_like(p2sp) / max(1, p2sp.size)) if p2sum <= 0 else (p2sp / 
        means2 = []
        meds2 = []
        cnts2 = []
        known2 = 0
        for p in p2s:
            st = stats_p2.get(p)
            if st is None:
                means2.append(gmean); meds2.append(gmed); cnts2.append(0)
            else:
                means2.append(st.mean_smooth); meds2.append(st.median_adj); cnts2
        means2 = np.asarray(means2, dtype=float)
        meds2 = np.asarray(meds2, dtype=float)
        cnts2 = np.asarray(cnts2, dtype=float)
        top2 = int(np.argmax(w2)) if w2.size else 0
        feats[i, 8:16] = np.array([
            float(np.sum(w2 * means2)),
            float(np.sum(w2 * meds2)),
            float(np.mean(means2)),
            float(np.max(means2)),
            float(np.min(means2)),
            float(meds2[top2] if meds2.size else gmed),
            float(known2 / max(1, len(p2s))),
            float(np.mean(np.log1p(cnts2))) if cnts2.size else 0.0,
        ], dtype=np.float32)

    return feats
def make_strat_labels(train_df: pd.DataFrame) -> np.ndarray:
    chronic = train_df["primary_chronic"].astype(str)
    cost_bin = pd.qcut(train_df["prior_ed_cost_5y_usd"], q=10, duplicates="drop")
    return (chronic + "_" + cost_bin).values
# =========================
# BLEND
# =========================
def best_blend_weight(y_true: np.ndarray, p1: np.ndarray, p2: np.ndarray, step: f
    best_w = 0.5
    best_m = 1e18
    for w in np.arange(0.0, 1.0 + 1e-9, step):
        blend = w * p1 + (1.0 - w) * p2
        m = mae(y_true, blend)
        if m < best_m:
            best_m = m
            best_w = float(w)
    return best_w, best_m
# =========================
# MAIN
# =========================
def main() -> None:
    setup_logging()
    logging.info(f"XGBoost version: {getattr(xgb, '__version__', 'unknown')}")
    logging.info(f"FAST_MODE={FAST_MODE} | repeats={len(REPEAT_SEEDS)} | folds={N_
    # load
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    # optional patients
    if PATIENTS_CSV.exists():
        patients = pd.read_csv(PATIENTS_CSV)
        if "patient_id" in patients.columns:

            train_df = train_df.merge(patients, on="patient_id", how="left")
            test_df = test_df.merge(patients, on="patient_id", how="left")
            logging.info("Merged patients.csv")
    train_df["is_train"] = 1
    test_df["is_train"] = 0
    df_all = pd.concat([train_df, test_df], ignore_index=True)
    # receipts (cached)
    all_pids = df_all["patient_id"].astype(int).unique().tolist()
    all_pids.sort()
    receipt_meta_df, profiles = build_receipt_artifacts(all_pids, PDF_DIR, CACHE_
    df_all = df_all.merge(receipt_meta_df, on="patient_id", how="left")
    # base sparse (OHE) for XGB
    X_all_base, _ = build_base_sparse_matrix(df_all)
    is_train = df_all["is_train"].values.astype(int) == 1
    X_train_base = X_all_base[is_train]
    X_test_base = X_all_base[~is_train]
    y = train_df["ed_cost_next3y_usd"].values.astype(np.float32)
    y_log = np.log1p(y)
    train_pids = train_df["patient_id"].astype(int).tolist()
    test_pids = test_df["patient_id"].astype(int).tolist()
    pid_to_row = {pid: i for i, pid in enumerate(train_pids)}
    # CatBoost base frames
    df_cb_all = df_all.drop(columns=["ed_cost_next3y_usd"], errors="ignore").copy
    for c in ["primary_chronic", "sex", "insurance", "zip3", "pdf_insurance", "pd
        if c in df_cb_all.columns:
            df_cb_all[c] = df_cb_all[c].astype(str).fillna("missing")
    df_cb_all["prior_cost_per_visit"] = df_cb_all["prior_ed_cost_5y_usd"] / (df_c
    df_cb_all["log_prior_cost"] = np.log1p(df_cb_all["prior_ed_cost_5y_usd"].asty
    df_cb_all["log_prior_visits"] = np.log1p(df_cb_all["prior_ed_visits_5y"].asty
    df_train_cb = df_cb_all[is_train].reset_index(drop=True)
    df_test_cb = df_cb_all[~is_train].reset_index(drop=True)

    cb_cat_cols = [c for c in ["primary_chronic", "sex", "insurance", "zip3", "pd
                   if c in df_train_cb.columns]
    cb_cat_idx = [df_train_cb.columns.get_loc(c) for c in cb_cat_cols]
    # CV splits
    strat = make_strat_labels(train_df)
    splits = []
    for rep, seed in enumerate(REPEAT_SEEDS):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), strat
            splits.append((rep, fold, tr_idx, va_idx))
    logging.info(f"Total splits: {len(splits)}")
    # storage
    oof_xgb = np.zeros(len(y), dtype=np.float64)
    oof_cat = np.zeros(len(y), dtype=np.float64)
    cnt_xgb = np.zeros(len(y), dtype=np.float64)
    cnt_cat = np.zeros(len(y), dtype=np.float64)
    test_xgb_sum = np.zeros(len(test_df), dtype=np.float64)
    test_cat_sum = np.zeros(len(test_df), dtype=np.float64)
    n_xgb = 0
    n_cat = 0
    # model params
    xgb_params = dict(
        objective="reg:absoluteerror",
        eval_metric="mae",
        tree_method="hist",
        device=XGB_DEVICE,     # <-- GPU ON
        n_estimators=6000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_lambda=1.0,
        min_child_weight=1.0,
        n_jobs=-1,
        verbosity=0,
        early_stopping_rounds=EARLY_STOPPING,  # XGBoost >=2.x wants this in cons

    )
    cat_params = dict(
        loss_function="MAE",
        eval_metric="MAE",
        iterations=8000,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=6.0,
        bagging_temperature=0.8,
        random_strength=0.5,
        od_type="Iter",
        od_wait=EARLY_STOPPING,
        task_type=CAT_TASK_TYPE,
        devices="0",
        verbose=False,
    )
    te_cols = [f"te_{i:02d}" for i in range(16)]
    # CV loop
    for rep, fold, tr_idx, va_idx in tqdm(splits, desc="CV folds", ncols=110):
        t0 = time.time()
        tr_pids = [train_pids[i] for i in tr_idx.tolist()]
        va_pids = [train_pids[i] for i in va_idx.tolist()]
        stats_code, stats_p2, gmean, gmed = build_token_stats_for_fold(tr_pids, y_
        te_tr = te_features_for_patients(tr_pids, profiles, stats_code, stats_p2, 
        te_va = te_features_for_patients(va_pids, profiles, stats_code, stats_p2, 
        te_te = te_features_for_patients(test_pids, profiles, stats_code, stats_p
        # XGB matrices (sparse)
        X_tr = hstack([X_train_base[tr_idx], csr_matrix(te_tr)], format="csr")
        X_va = hstack([X_train_base[va_idx], csr_matrix(te_va)], format="csr")
        X_te = hstack([X_test_base, csr_matrix(te_te)], format="csr")
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        # ---- XGBoost GPU ----
        try:
            xgb_model = XGBRegressor(**xgb_params, random_state=1000 * rep + fold
            xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            p_va_x = xgb_model.predict(X_va)
            p_te_x = xgb_model.predict(X_te)
        except Exception as e:
            logging.warning(f"XGBoost GPU failed (rep={rep}, fold={fold}): {e} -> 
            xgb_cpu = dict(xgb_params)
            xgb_cpu["device"] = "cpu"
            xgb_model = XGBRegressor(**xgb_cpu, random_state=1000 * rep + fold)
            xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            p_va_x = xgb_model.predict(X_va)
            p_te_x = xgb_model.predict(X_te)
        if CLIP_AT_ZERO:
            p_va_x = np.maximum(p_va_x, 0.0)
            p_te_x = np.maximum(p_te_x, 0.0)
        oof_xgb[va_idx] += p_va_x
        cnt_xgb[va_idx] += 1.0
        test_xgb_sum += p_te_x
        n_xgb += 1
        # ---- CatBoost GPU ----
        te_train_full = np.zeros((len(train_df), 16), dtype=np.float32)
        te_train_full[tr_idx, :] = te_tr
        te_train_full[va_idx, :] = te_va
        df_tr = df_train_cb.copy()
        df_te = df_test_cb.copy()
        for j, col in enumerate(te_cols):
            df_tr[col] = te_train_full[:, j]
            df_te[col] = te_te[:, j]
        tr_pool = Pool(df_tr.iloc[tr_idx], label=y_tr, cat_features=cb_cat_idx)
        va_pool = Pool(df_tr.iloc[va_idx], label=y_va, cat_features=cb_cat_idx)
        te_pool = Pool(df_te, cat_features=cb_cat_idx)
        try:

            cat_model = CatBoostRegressor(**cat_params, random_seed=2000 * rep + 
            cat_model.fit(tr_pool, eval_set=va_pool, use_best_model=True)
        except Exception as e:
            logging.warning(f"CatBoost GPU failed (rep={rep}, fold={fold}): {e} -
            cat_cpu = dict(cat_params)
            cat_cpu["task_type"] = "CPU"
            cat_cpu.pop("devices", None)
            cat_model = CatBoostRegressor(**cat_cpu, random_seed=2000 * rep + fol
            cat_model.fit(tr_pool, eval_set=va_pool, use_best_model=True)
        p_va_c = cat_model.predict(va_pool)
        p_te_c = cat_model.predict(te_pool)
        if CLIP_AT_ZERO:
            p_va_c = np.maximum(p_va_c, 0.0)
            p_te_c = np.maximum(p_te_c, 0.0)
        oof_cat[va_idx] += p_va_c
        cnt_cat[va_idx] += 1.0
        test_cat_sum += p_te_c
        n_cat += 1
        dt = time.time() - t0
        if (fold == 0) or ((fold + 1) % 2 == 0):
            logging.info(f"rep={rep} fold={fold} done in {dt:.1f}s")
    # finalize oof/test
    oof_x = (oof_xgb / np.maximum(cnt_xgb, 1.0)).astype(np.float32)
    oof_c = (oof_cat / np.maximum(cnt_cat, 1.0)).astype(np.float32)
    te_x = (test_xgb_sum / max(1, n_xgb)).astype(np.float32)
    te_c = (test_cat_sum / max(1, n_cat)).astype(np.float32)
    mae_x = mae(y, oof_x)
    mae_c = mae(y, oof_c)
    logging.info(f"OOF MAE | XGB: {mae_x:.4f}")
    logging.info(f"OOF MAE | CAT: {mae_c:.4f}")
    # blend weight
    w, blend_mae = best_blend_weight(y, oof_x, oof_c, step=BLEND_STEP)
    oof_blend = w * oof_x + (1.0 - w) * oof_c

What you should see now
XGBoost should stop being CPU-bound. With device="cuda" , XGBoost uses GPU
acceleration. 
CV should move past 0%  quickly (first fold should finish in seconds to ~tens of
seconds on 2k rows, depending on your feature width).
If it’s still slow
The next bottleneck is too many boosting rounds. Drop:
n_estimators=6000 → 3000
iterations=8000 → 4000
and keep early_stopping_rounds=200 . CatBoost’s OD settings are designed for
exactly this. 
If you paste the log lines:
    bias = float(np.median(y - oof_blend))  # MAE-optimal constant shift
    oof_final = oof_blend + bias
    test_final = w * te_x + (1.0 - w) * te_c + bias
    if CLIP_AT_ZERO:
        oof_final = np.maximum(oof_final, 0.0)
        test_final = np.maximum(test_final, 0.0)
    logging.info(f"Blend w(XGB)={w:.2f} | OOF MAE pre-bias={blend_mae:.4f} | bias
    # save submission
    sub = pd.DataFrame({
        "patient_id": test_df["patient_id"].astype(int).values,
        "ed_cost_next3y_usd": test_final.astype(np.float32),
    })
    sub.to_csv(SUBMISSION_PATH, index=False)
    logging.info(f"Saved submission: {SUBMISSION_PATH}")
if __name__ == "__main__":
    main()
XGBoost Docum…
CatBoost

OOF MAE | XGB: ...
OOF MAE | CAT: ...
Blend ... OOF MAE final= ...
I can tell you whether the remaining gap is model bias (needs different objective /
calibration) or signal extraction (TE features need a stronger aggregation).
```

### C05 — ITERATION 2 (GPU-Fast) — Observability / Heartbeats Edition (pages 136-160)
```python
# -*- coding: utf-8 -*-
r"""
ITERATION 2 (GPU-Fast) — Observability / Heartbeats Edition
===========================================================
Changes requested:
1) Fix Regex Warnings: all regex patterns use raw strings r"..."

2) Training Heartbeats:
   - XGBoost: verbosity=1 and print every 100 rounds via fit(verbose=100) (or ver
   - CatBoost: verbose=100 in params (prints every 100 iterations)
   - logging.info before/after each model.fit() per fold
3) Preserve Logic: NO changes to model architecture, hyperparameters (except verb
===========================================================
INSTALL (run once)
===========================================================
pip install -U pandas numpy scipy scikit-learn xgboost catboost pymupdf joblib tq
Paths:
D:\AgentDs\agent_ds_healthcare\ed_cost_train.csv
D:\AgentDs\agent_ds_healthcare\ed_cost_test.csv
D:\AgentDs\agent_ds_healthcare\patients.csv (optional)
D:\AgentDs\agent_ds_healthcare\receipts_pdf\receipt_{patient_id}.pdf
Output:
D:\AgentDs\agent_ds_healthcare\submission.csv
===========================================================
"""
from __future__ import annotations
import re
import math
import time
import logging
import warnings
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import fitz  # PyMuPDF

from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import xgboost as xgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor, Pool
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# =========================
# CONFIG (PRESERVED)
# =========================
BASE_DIR = Path(r"D:\AgentDs\agent_ds_healthcare")
TRAIN_CSV = BASE_DIR / "ed_cost_train.csv"
TEST_CSV = BASE_DIR / "ed_cost_test.csv"
PATIENTS_CSV = BASE_DIR / "patients.csv"          # optional
PDF_DIR = BASE_DIR / "receipts_pdf"
SUBMISSION_PATH = BASE_DIR / "submission.csv"
CACHE_PATH = BASE_DIR / "cache_receipts_iter2_gpu_fast.joblib"
CACHE_VERSION = 21
# Speed / rigor tradeoff (PRESERVED)
FAST_MODE = True   # True -> 2 repeats (10 folds). False -> 3 repeats (15 folds)
N_FOLDS = 5
REPEAT_SEEDS = [2025, 2026] if FAST_MODE else [2024, 2025, 2026]
EARLY_STOPPING = 200
# Target encoding (PRESERVED)
TE_SMOOTH = 20.0
TE_MIN_COUNT_MEDIAN = 5

# Use GPU (PRESERVED)
XGB_DEVICE = "cuda"       # use "cuda:0" if you want explicit device
CAT_TASK_TYPE = "GPU"     # will auto-fallback to CPU if GPU fails
# Prediction post-process (PRESERVED)
CLIP_AT_ZERO = True
# Blending (PRESERVED)
BLEND_STEP = 0.02
# =========================
# LOGGING
# =========================
def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | 
# =========================
# HELPERS
# =========================
_money_re = re.compile(r"^\d{1,3}(?:,\d{3})*(?:\.\d{2})$|^\d+(?:\.\d{2})$")
_code_re = re.compile(r"^(?:[A-Z]\d{4}|\d{4,5})$")
def is_money(tok: str) -> bool:
    return bool(_money_re.match(tok))
def money_to_float(tok: str) -> float:
    return float(tok.replace(",", ""))
def is_code(tok: str) -> bool:
    return bool(_code_re.match(tok))
def safe_log1p(x: float) -> float:
    return math.log1p(max(0.0, float(x)))
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_out
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)
# =========================
# PDF PARSING
# =========================
def cluster_words_into_lines(words: List[Tuple], y_tol: float = 1.5) -> List[List
    if not words:
        return []
    words_sorted = sorted(words, key=lambda w: (w[1], w[0]))
    lines: List[List[str]] = []
    cur: List[Tuple[float, str]] = []
    cur_y: Optional[float] = None
    for w in words_sorted:
        x0, y0 = float(w[0]), float(w[1])
        tok = str(w[4])
        if cur_y is None or abs(y0 - cur_y) <= y_tol:
            cur.append((x0, tok))
            cur_y = y0 if cur_y is None else (0.85 * cur_y + 0.15 * y0)
        else:
            lines.append([t for _, t in sorted(cur, key=lambda z: z[0])])
            cur = [(x0, tok)]
            cur_y = y0
    if cur:
        lines.append([t for _, t in sorted(cur, key=lambda z: z[0])])
    return lines
def parse_header_fields(text: str) -> Tuple[Optional[int], Optional[str], Optiona
    pid = None
    zip3 = None
    insurance = None
    m = re.search(r"Patient ID:\s*(\d+)", text)
    if m:

        pid = int(m.group(1))
    m = re.search(r"ZIP3:\s*([0-9]{3})\s+Insurance:\s*([A-Za-z_]+)", text)
    if m:
        zip3 = str(m.group(1))
        insurance = str(m.group(2)).lower()
    return pid, zip3, insurance
def parse_items_from_words(words: List[Tuple]) -> Tuple[List[Dict[str, Any]], Opt
    items: List[Dict[str, Any]] = []
    total: Optional[float] = None
    lines = cluster_words_into_lines(words, y_tol=1.5)
    if not lines:
        return items, total
    header_idx = None
    for i, toks in enumerate(lines):
        low = [t.lower() for t in toks]
        if ("code" in low) and ("description" in low) and ("qty" in low):
            header_idx = i
            break
    if header_idx is None:
        return items, total
    for toks in lines[header_idx + 1:]:
        if not toks:
            continue
        if toks[0].upper() == "TOTAL":
            for t in reversed(toks[1:]):
                if is_money(t):
                    total = money_to_float(t)
                    break
            break
        if len(toks) >= 5 and is_code(toks[0]) and toks[-3].isdigit() and is_money
            items.append({
                "code": toks[0],
                "qty": int(toks[-3]),

                "unit": money_to_float(toks[-2]),
                "line_total": money_to_float(toks[-1]),
                "desc": " ".join(toks[1:-3]),
            })
    return items, total
def parse_items_fallback_text(text: str) -> Tuple[List[Dict[str, Any]], Optional[
    items: List[Dict[str, Any]] = []
    total: Optional[float] = None
    lines = [ln.strip() for ln in (text.splitlines() if text else []) if ln.strip
    if not lines:
        return items, total
    start_idx = None
    for i in range(len(lines) - 4):
        if lines[i].lower() == "code" and lines[i+1].lower() == "description" and 
            start_idx = i + 5
            break
        if "code" in lines[i].lower() and "description" in lines[i].lower() and "
            start_idx = i + 1
            break
    if start_idx is None:
        return items, total
    i = start_idx
    while i < len(lines):
        if lines[i].upper().startswith("TOTAL"):
            parts = lines[i].split()
            if len(parts) >= 2 and is_money(parts[-1]):
                total = money_to_float(parts[-1])
            elif i+1 < len(lines) and is_money(lines[i+1]):
                total = money_to_float(lines[i+1])
            break
        if i+4 < len(lines) and is_code(lines[i]) and lines[i+2].isdigit() and is_
            items.append({
                "code": lines[i],
                "desc": lines[i+1],
                "qty": int(lines[i+2]),

                "unit": money_to_float(lines[i+3]),
                "line_total": money_to_float(lines[i+4]),
            })
            i += 5
        else:
            i += 1
    return items, total
@dataclass
class ReceiptProfile:
    meta: Dict[str, Any]
    code_spend: Dict[str, float]
def featurize_receipt(patient_id: int, pdf_dir: Path) -> ReceiptProfile:
    pdf_path = pdf_dir / f"receipt_{patient_id}.pdf"
    empty_meta = dict(
        pdf_ok=0, pdf_n_lines=0, pdf_n_unique_codes=0,
        pdf_sum_line_total=0.0, pdf_top1_cost_share=0.0, pdf_top3_cost_share=0.0,
        pdf_repeat_cost_share=0.0, pdf_total=0.0, pdf_total_diff=0.0,
        pdf_insurance=None, pdf_zip3=None, pdf_parse_notes="missing_pdf",
    )
    if not pdf_path.exists():
        return ReceiptProfile(meta=empty_meta, code_spend={})
    try:
        with fitz.open(pdf_path) as doc:
            full_text = ""
            items: List[Dict[str, Any]] = []
            total: Optional[float] = None
            zip3 = None
            insurance = None
            for page in doc:
                text = page.get_text("text") or ""
                full_text += text + "\n"
                words = page.get_text("words") or []

                items_p, total_p = parse_items_from_words(words)
                if not items_p:
                    items_p, total_p = parse_items_fallback_text(text)
                if items_p:
                    items.extend(items_p)
                if total_p is not None:
                    total = total_p
            _, zip3, insurance = parse_header_fields(full_text)
    except Exception as e:
        empty_meta["pdf_parse_notes"] = f"open_error:{type(e).__name__}"
        return ReceiptProfile(meta=empty_meta, code_spend={})
    if not items:
        empty_meta["pdf_parse_notes"] = "no_items_parsed"
        empty_meta["pdf_zip3"] = zip3
        empty_meta["pdf_insurance"] = insurance
        return ReceiptProfile(meta=empty_meta, code_spend={})
    code_spend: Dict[str, float] = {}
    cnt_by_code: Dict[str, int] = {}
    for it in items:
        c = str(it["code"])
        code_spend[c] = code_spend.get(c, 0.0) + float(it["line_total"])
        cnt_by_code[c] = cnt_by_code.get(c, 0) + 1
    line_totals = np.array([float(it["line_total"]) for it in items], dtype=float
    sum_line_total = float(line_totals.sum())
    pdf_total = float(total) if total is not None else sum_line_total
    sorted_cost = np.sort(line_totals)[::-1]
    top1 = float(sorted_cost[0] / sum_line_total) if sum_line_total > 0 else 0.0
    top3 = float(sorted_cost[:3].sum() / sum_line_total) if sum_line_total > 0 el
    repeat_spend = sum(v for c, v in code_spend.items() if cnt_by_code.get(c, 0) 
    repeat_share = float(repeat_spend / sum_line_total) if sum_line_total > 0 els
    meta = dict(

        pdf_ok=1,
        pdf_n_lines=len(items),
        pdf_n_unique_codes=len(code_spend),
        pdf_sum_line_total=sum_line_total,
        pdf_top1_cost_share=top1,
        pdf_top3_cost_share=top3,
        pdf_repeat_cost_share=repeat_share,
        pdf_total=pdf_total,
        pdf_total_diff=float(abs(sum_line_total - pdf_total)),
        pdf_insurance=(insurance.lower() if isinstance(insurance, str) else None)
        pdf_zip3=(str(zip3) if zip3 is not None else None),
        pdf_parse_notes="ok",
    )
    return ReceiptProfile(meta=meta, code_spend=code_spend)
def load_cache(cache_path: Path) -> Dict[str, Any]:
    if cache_path.exists():
        try:
            obj = joblib.load(cache_path)
            if isinstance(obj, dict) and obj.get("version") == CACHE_VERSION and 
                return obj
        except Exception:
            pass
    return {"version": CACHE_VERSION, "data": {}}
def save_cache(cache_path: Path, cache_obj: Dict[str, Any]) -> None:
    try:
        joblib.dump(cache_obj, cache_path)
    except Exception as e:
        logging.warning(f"Cache save failed: {e}")
def build_receipt_artifacts(patient_ids: List[int], pdf_dir: Path, cache_path: Pa
    cache_obj = load_cache(cache_path)
    data: Dict[str, Any] = cache_obj.get("data", {})
    to_parse = [pid for pid in patient_ids if str(pid) not in data]

    if to_parse:
        logging.info(f"Parsing {len(to_parse):,} receipts (cache miss).")
    for pid in tqdm(to_parse, desc="Parsing receipts", ncols=100):
        prof = featurize_receipt(pid, pdf_dir)
        data[str(pid)] = {"meta": prof.meta, "code_spend": prof.code_spend}
    cache_obj["data"] = data
    if to_parse:
        save_cache(cache_path, cache_obj)
    meta_rows = []
    profiles: Dict[int, Dict[str, Any]] = {}
    for pid in patient_ids:
        obj = data.get(str(pid), None)
        if obj is None:
            prof = featurize_receipt(pid, pdf_dir)
            obj = {"meta": prof.meta, "code_spend": prof.code_spend}
        meta_rows.append({"patient_id": pid, **obj["meta"]})
        profiles[pid] = obj
    return pd.DataFrame(meta_rows), profiles
# =========================
# BASE FEATURES (OHE for XGB)
# =========================
def build_base_sparse_matrix(df_all: pd.DataFrame) -> Tuple[csr_matrix, Dict[str, 
    df = df_all.copy()
    # derived tabular
    df["prior_cost_per_visit"] = df["prior_ed_cost_5y_usd"] / (df["prior_ed_visit
    df["log_prior_cost"] = np.log1p(df["prior_ed_cost_5y_usd"].astype(float))
    df["log_prior_visits"] = np.log1p(df["prior_ed_visits_5y"].astype(float))
    for c in ["primary_chronic", "sex", "insurance", "zip3", "pdf_insurance", "pd
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("missing")
    exclude = {"patient_id", "ed_cost_next3y_usd", "is_train"}

    cat_cols = [c for c in ["primary_chronic", "sex", "insurance", "zip3", "pdf_i
                if c in df.columns]
    num_cols = [c for c in df.columns if c not in exclude and c not in cat_cols a
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), nu
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    X = preprocessor.fit_transform(df)
    X = X.tocsr() if hasattr(X, "tocsr") else csr_matrix(X)
    return X, {"preprocessor": preprocessor, "num_cols": num_cols, "cat_cols": cat
# =========================
# TARGET ENCODING (CV-SAFE)
# =========================
@dataclass
class TokenStats:
    cnt: int
    mean_smooth: float
    median_adj: float
def code_to_prefix2(code: str) -> str:
    return code[:2] if len(code) >= 2 else code
def build_token_stats_for_fold(
    train_pids: List[int],
    y_log: np.ndarray,
    pid_to_row: Dict[int, int],
    profiles: Dict[int, Dict[str, Any]],
) -> Tuple[Dict[str, TokenStats], Dict[str, TokenStats], float, float]:
    vals_code: Dict[str, List[float]] = {}
    vals_p2: Dict[str, List[float]] = {}
    ys = []

    for pid in train_pids:
        r = pid_to_row.get(pid, None)
        if r is None:
            continue
        v = float(y_log[r])
        ys.append(v)
        code_spend = profiles.get(pid, {}).get("code_spend", {}) or {}
        for c in code_spend.keys():
            vals_code.setdefault(c, []).append(v)
            vals_p2.setdefault(code_to_prefix2(c), []).append(v)
    gmean = float(np.mean(ys)) if ys else 0.0
    gmed = float(np.median(ys)) if ys else 0.0
    def to_stats(m: Dict[str, List[float]]) -> Dict[str, TokenStats]:
        out = {}
        for tok, arr in m.items():
            a = np.asarray(arr, dtype=float)
            cnt = int(a.size)
            mean = float(np.mean(a))
            med = float(np.median(a))
            mean_smooth = float((cnt * mean + TE_SMOOTH * gmean) / (cnt + TE_SMOO
            median_adj = med if cnt >= TE_MIN_COUNT_MEDIAN else gmed
            out[tok] = TokenStats(cnt=cnt, mean_smooth=mean_smooth, median_adj=me
        return out
    return to_stats(vals_code), to_stats(vals_p2), gmean, gmed
def te_features_for_patients(
    pids: List[int],
    profiles: Dict[int, Dict[str, Any]],
    stats_code: Dict[str, TokenStats],
    stats_p2: Dict[str, TokenStats],
    gmean: float,
    gmed: float,
) -> np.ndarray:
    """
    16 TE features: exact code (8) + prefix2 (8), spend-weighted.
    """

    feats = np.zeros((len(pids), 16), dtype=np.float32)
    for i, pid in enumerate(pids):
        code_spend = profiles.get(pid, {}).get("code_spend", {}) or {}
        if not code_spend:
            base = np.array([gmean, gmed, gmean, gmean, gmean, gmed, 0.0, 0.0], dt
            feats[i, 0:8] = base
            feats[i, 8:16] = base
            continue
        codes = list(code_spend.keys())
        spends = np.array([float(code_spend[c]) for c in codes], dtype=float)
        ssum = float(spends.sum())
        w = (np.ones_like(spends) / max(1, spends.size)) if ssum <= 0 else (spend
        # exact
        means = []
        meds = []
        cnts = []
        known = 0
        for c in codes:
            st = stats_code.get(c)
            if st is None:
                means.append(gmean); meds.append(gmed); cnts.append(0)
            else:
                means.append(st.mean_smooth); meds.append(st.median_adj); cnts.ap
        means = np.asarray(means, dtype=float)
        meds = np.asarray(meds, dtype=float)
        cnts = np.asarray(cnts, dtype=float)
        top = int(np.argmax(w)) if w.size else 0
        feats[i, 0:8] = np.array([
            float(np.sum(w * means)),
            float(np.sum(w * meds)),
            float(np.mean(means)),
            float(np.max(means)),
            float(np.min(means)),
            float(meds[top] if meds.size else gmed),
            float(known / max(1, len(codes))),
            float(np.mean(np.log1p(cnts))) if cnts.size else 0.0,

        ], dtype=np.float32)
        # prefix2
        p2_map: Dict[str, float] = {}
        for c, sp in code_spend.items():
            p2 = code_to_prefix2(c)
            p2_map[p2] = p2_map.get(p2, 0.0) + float(sp)
        p2s = list(p2_map.keys())
        p2sp = np.asarray([p2_map[p] for p in p2s], dtype=float)
        p2sum = float(p2sp.sum())
        w2 = (np.ones_like(p2sp) / max(1, p2sp.size)) if p2sum <= 0 else (p2sp / 
        means2 = []
        meds2 = []
        cnts2 = []
        known2 = 0
        for p in p2s:
            st = stats_p2.get(p)
            if st is None:
                means2.append(gmean); meds2.append(gmed); cnts2.append(0)
            else:
                means2.append(st.mean_smooth); meds2.append(st.median_adj); cnts2
        means2 = np.asarray(means2, dtype=float)
        meds2 = np.asarray(meds2, dtype=float)
        cnts2 = np.asarray(cnts2, dtype=float)
        top2 = int(np.argmax(w2)) if w2.size else 0
        feats[i, 8:16] = np.array([
            float(np.sum(w2 * means2)),
            float(np.sum(w2 * meds2)),
            float(np.mean(means2)),
            float(np.max(means2)),
            float(np.min(means2)),
            float(meds2[top2] if meds2.size else gmed),
            float(known2 / max(1, len(p2s))),
            float(np.mean(np.log1p(cnts2))) if cnts2.size else 0.0,
        ], dtype=np.float32)
    return feats

def make_strat_labels(train_df: pd.DataFrame) -> np.ndarray:
    chronic = train_df["primary_chronic"].astype(str)
    cost_bin = pd.qcut(train_df["prior_ed_cost_5y_usd"], q=10, duplicates="drop")
    return (chronic + "_" + cost_bin).values
# =========================
# BLEND
# =========================
def best_blend_weight(y_true: np.ndarray, p1: np.ndarray, p2: np.ndarray, step: f
    best_w = 0.5
    best_m = 1e18
    for w in np.arange(0.0, 1.0 + 1e-9, step):
        blend = w * p1 + (1.0 - w) * p2
        m = mae(y_true, blend)
        if m < best_m:
            best_m = m
            best_w = float(w)
    return best_w, best_m
# =========================
# MAIN
# =========================
def main() -> None:
    setup_logging()
    logging.info(f"XGBoost version: {getattr(xgb, '__version__', 'unknown')}")
    logging.info(f"FAST_MODE={FAST_MODE} | repeats={len(REPEAT_SEEDS)} | folds={N_
    # load
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    # optional patients
    if PATIENTS_CSV.exists():
        patients = pd.read_csv(PATIENTS_CSV)
        if "patient_id" in patients.columns:
            train_df = train_df.merge(patients, on="patient_id", how="left")
            test_df = test_df.merge(patients, on="patient_id", how="left")

            logging.info("Merged patients.csv")
    train_df["is_train"] = 1
    test_df["is_train"] = 0
    df_all = pd.concat([train_df, test_df], ignore_index=True)
    # receipts (cached)
    all_pids = df_all["patient_id"].astype(int).unique().tolist()
    all_pids.sort()
    receipt_meta_df, profiles = build_receipt_artifacts(all_pids, PDF_DIR, CACHE_
    df_all = df_all.merge(receipt_meta_df, on="patient_id", how="left")
    # base sparse (OHE) for XGB
    X_all_base, _ = build_base_sparse_matrix(df_all)
    is_train = df_all["is_train"].values.astype(int) == 1
    X_train_base = X_all_base[is_train]
    X_test_base = X_all_base[~is_train]
    y = train_df["ed_cost_next3y_usd"].values.astype(np.float32)
    y_log = np.log1p(y)
    train_pids = train_df["patient_id"].astype(int).tolist()
    test_pids = test_df["patient_id"].astype(int).tolist()
    pid_to_row = {pid: i for i, pid in enumerate(train_pids)}
    # CatBoost base frames
    df_cb_all = df_all.drop(columns=["ed_cost_next3y_usd"], errors="ignore").copy
    for c in ["primary_chronic", "sex", "insurance", "zip3", "pdf_insurance", "pd
        if c in df_cb_all.columns:
            df_cb_all[c] = df_cb_all[c].astype(str).fillna("missing")
    df_cb_all["prior_cost_per_visit"] = df_cb_all["prior_ed_cost_5y_usd"] / (df_c
    df_cb_all["log_prior_cost"] = np.log1p(df_cb_all["prior_ed_cost_5y_usd"].asty
    df_cb_all["log_prior_visits"] = np.log1p(df_cb_all["prior_ed_visits_5y"].asty
    df_train_cb = df_cb_all[is_train].reset_index(drop=True)
    df_test_cb = df_cb_all[~is_train].reset_index(drop=True)
    cb_cat_cols = [c for c in ["primary_chronic", "sex", "insurance", "zip3", "pd
                   if c in df_train_cb.columns]

    cb_cat_idx = [df_train_cb.columns.get_loc(c) for c in cb_cat_cols]
    # CV splits
    strat = make_strat_labels(train_df)
    splits = []
    for rep, seed in enumerate(REPEAT_SEEDS):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), strat
            splits.append((rep, fold, tr_idx, va_idx))
    logging.info(f"Total splits: {len(splits)}")
    # storage
    oof_xgb = np.zeros(len(y), dtype=np.float64)
    oof_cat = np.zeros(len(y), dtype=np.float64)
    cnt_xgb = np.zeros(len(y), dtype=np.float64)
    cnt_cat = np.zeros(len(y), dtype=np.float64)
    test_xgb_sum = np.zeros(len(test_df), dtype=np.float64)
    test_cat_sum = np.zeros(len(test_df), dtype=np.float64)
    n_xgb = 0
    n_cat = 0
    # ===== Model params (PRESERVED except verbosity/verbose) =====
    xgb_params = dict(
        objective="reg:absoluteerror",
        eval_metric="mae",
        tree_method="hist",
        device=XGB_DEVICE,     # GPU ON
        n_estimators=6000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_lambda=1.0,
        min_child_weight=1.0,
        n_jobs=-1,
        verbosity=1,  # CHANGED for heartbeat / observability (was 0)
        early_stopping_rounds=EARLY_STOPPING,
    )

    cat_params = dict(
        loss_function="MAE",
        eval_metric="MAE",
        iterations=8000,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=6.0,
        bagging_temperature=0.8,
        random_strength=0.5,
        od_type="Iter",
        od_wait=EARLY_STOPPING,
        task_type=CAT_TASK_TYPE,
        devices="0",
        verbose=100,  # CHANGED for heartbeat / observability (was False)
    )
    te_cols = [f"te_{i:02d}" for i in range(16)]
    verbose_eval = 100  # requested "print every 100 rounds"
    # CV loop
    for rep, fold, tr_idx, va_idx in tqdm(splits, desc="CV folds", ncols=110):
        t0 = time.time()
        fold_tag = f"rep={rep} fold={fold}"
        logging.info(f"{fold_tag} | Starting fold | train={len(tr_idx)} val={len(v
        tr_pids = [train_pids[i] for i in tr_idx.tolist()]
        va_pids = [train_pids[i] for i in va_idx.tolist()]
        logging.info(f"{fold_tag} | Building TE stats...")
        stats_code, stats_p2, gmean, gmed = build_token_stats_for_fold(tr_pids, y_
        logging.info(f"{fold_tag} | Building TE feature blocks (train/val/test)..
        te_tr = te_features_for_patients(tr_pids, profiles, stats_code, stats_p2, 
        te_va = te_features_for_patients(va_pids, profiles, stats_code, stats_p2, 
        te_te = te_features_for_patients(test_pids, profiles, stats_code, stats_p
        # XGB matrices (sparse)
        X_tr = hstack([X_train_base[tr_idx], csr_matrix(te_tr)], format="csr")
        X_va = hstack([X_train_base[va_idx], csr_matrix(te_va)], format="csr")

        X_te = hstack([X_test_base, csr_matrix(te_te)], format="csr")
        y_tr = y[tr_idx]
        y_va = y[va_idx]
        # ---- XGBoost GPU ----
        logging.info(f"{fold_tag} | Starting XGBoost training | device={xgb_param
        try:
            xgb_model = XGBRegressor(**xgb_params, random_state=1000 * rep + fold
            # "verbose_eval" is not always a sklearn-fit arg; use it if present, 
            fit_params = inspect.signature(xgb_model.fit).parameters
            fit_kwargs = {"eval_set": [(X_va, y_va)]}
            if "verbose_eval" in fit_params:
                fit_kwargs["verbose_eval"] = verbose_eval
            else:
                # XGBoost sklearn API: verbose=int prints evaluation metric every 
                fit_kwargs["verbose"] = verbose_eval
            xgb_model.fit(X_tr, y_tr, **fit_kwargs)
            p_va_x = xgb_model.predict(X_va)
            p_te_x = xgb_model.predict(X_te)
            best_it = getattr(xgb_model, "best_iteration", None)
            logging.info(f"{fold_tag} | Finished XGBoost training | best_iteratio
        except Exception as e:
            logging.warning(f"{fold_tag} | XGBoost GPU failed: {e} -> fallback CP
            xgb_cpu = dict(xgb_params)
            xgb_cpu["device"] = "cpu"
            xgb_model = XGBRegressor(**xgb_cpu, random_state=1000 * rep + fold)
            fit_params = inspect.signature(xgb_model.fit).parameters
            fit_kwargs = {"eval_set": [(X_va, y_va)]}
            if "verbose_eval" in fit_params:
                fit_kwargs["verbose_eval"] = verbose_eval
            else:
                fit_kwargs["verbose"] = verbose_eval

            logging.info(f"{fold_tag} | Starting XGBoost training (CPU fallback) 
            xgb_model.fit(X_tr, y_tr, **fit_kwargs)
            logging.info(f"{fold_tag} | Finished XGBoost training (CPU fallback)"
            p_va_x = xgb_model.predict(X_va)
            p_te_x = xgb_model.predict(X_te)
        if CLIP_AT_ZERO:
            p_va_x = np.maximum(p_va_x, 0.0)
            p_te_x = np.maximum(p_te_x, 0.0)
        oof_xgb[va_idx] += p_va_x
        cnt_xgb[va_idx] += 1.0
        test_xgb_sum += p_te_x
        n_xgb += 1
        # ---- CatBoost GPU ----
        logging.info(f"{fold_tag} | Preparing CatBoost pools...")
        te_train_full = np.zeros((len(train_df), 16), dtype=np.float32)
        te_train_full[tr_idx, :] = te_tr
        te_train_full[va_idx, :] = te_va
        df_tr = df_train_cb.copy()
        df_te = df_test_cb.copy()
        for j, col in enumerate(te_cols):
            df_tr[col] = te_train_full[:, j]
            df_te[col] = te_te[:, j]
        tr_pool = Pool(df_tr.iloc[tr_idx], label=y_tr, cat_features=cb_cat_idx)
        va_pool = Pool(df_tr.iloc[va_idx], label=y_va, cat_features=cb_cat_idx)
        te_pool = Pool(df_te, cat_features=cb_cat_idx)
        logging.info(f"{fold_tag} | Starting CatBoost training | task_type={cat_p
        try:
            cat_model = CatBoostRegressor(**cat_params, random_seed=2000 * rep + 
            cat_model.fit(tr_pool, eval_set=va_pool, use_best_model=True)
            best_it = None
            try:

                best_it = cat_model.get_best_iteration()
            except Exception:
                best_it = None
            logging.info(f"{fold_tag} | Finished CatBoost training | best_iterati
        except Exception as e:
            logging.warning(f"{fold_tag} | CatBoost GPU failed: {e} -> fallback C
            cat_cpu = dict(cat_params)
            cat_cpu["task_type"] = "CPU"
            cat_cpu.pop("devices", None)
            cat_model = CatBoostRegressor(**cat_cpu, random_seed=2000 * rep + fol
            logging.info(f"{fold_tag} | Starting CatBoost training (CPU fallback) 
            cat_model.fit(tr_pool, eval_set=va_pool, use_best_model=True)
            logging.info(f"{fold_tag} | Finished CatBoost training (CPU fallback)
        p_va_c = cat_model.predict(va_pool)
        p_te_c = cat_model.predict(te_pool)
        if CLIP_AT_ZERO:
            p_va_c = np.maximum(p_va_c, 0.0)
            p_te_c = np.maximum(p_te_c, 0.0)
        oof_cat[va_idx] += p_va_c
        cnt_cat[va_idx] += 1.0
        test_cat_sum += p_te_c
        n_cat += 1
        dt = time.time() - t0
        logging.info(f"{fold_tag} | Fold complete in {dt:.1f}s")
    # finalize oof/test
    oof_x = (oof_xgb / np.maximum(cnt_xgb, 1.0)).astype(np.float32)
    oof_c = (oof_cat / np.maximum(cnt_cat, 1.0)).astype(np.float32)
    te_x = (test_xgb_sum / max(1, n_xgb)).astype(np.float32)
    te_c = (test_cat_sum / max(1, n_cat)).astype(np.float32)
    mae_x = mae(y, oof_x)
    mae_c = mae(y, oof_c)
    logging.info(f"OOF MAE | XGB: {mae_x:.4f}")

If you still see “CV folds: 0%” for a while, you should now still see:
INFO | rep=0 fold=0 | Starting XGBoost training ...
XGBoost’s metric lines every 100 rounds
CatBoost’s lines every 100 iterations
So you can distinguish “training is working” vs “kernel is stuck”.
    logging.info(f"OOF MAE | CAT: {mae_c:.4f}")
    # blend weight
    w, blend_mae = best_blend_weight(y, oof_x, oof_c, step=BLEND_STEP)
    oof_blend = w * oof_x + (1.0 - w) * oof_c
    bias = float(np.median(y - oof_blend))  # MAE-optimal constant shift
    oof_final = oof_blend + bias
    test_final = w * te_x + (1.0 - w) * te_c + bias
    if CLIP_AT_ZERO:
        oof_final = np.maximum(oof_final, 0.0)
        test_final = np.maximum(test_final, 0.0)
    logging.info(f"Blend w(XGB)={w:.2f} | OOF MAE pre-bias={blend_mae:.4f} | bias
    # save submission
    sub = pd.DataFrame({
        "patient_id": test_df["patient_id"].astype(int).values,
        "ed_cost_next3y_usd": test_final.astype(np.float32),
    })
    sub.to_csv(SUBMISSION_PATH, index=False)
    logging.info(f"Saved submission: {SUBMISSION_PATH}")
if __name__ == "__main__":
    main()
```

### C06 — ITERATION 3 — Domain Expert CPT/HCPCS Feature Engineering (ED Cost Forecasting) (pages 161-194)
```python
# -*- coding: utf-8 -*-
r"""
ITERATION 3 — Domain Expert CPT/HCPCS Feature Engineering (ED Cost Forecasting)
===============================================================================
What’s new vs Iteration 2:
- Replaces “blind TF-IDF” with structured clinical features:
  * ED E/M visit acuity from CPT 99281–99285 (levels 1–5)
  * Critical care intensity from CPT 99291/99292
  * Observation hours from HCPCS G0378
  * CPT major section spend mix using standard CPT range groupings
  * Surgical subrange buckets (10000–69999 broken into coarse 10k bands)
  * Keyword-derived procedure signals from the line-item descriptions (rule-based
- Keeps GPU XGBoost + GPU CatBoost ensemble with rich logging.
- Keeps CV-safe target encoding but upgrades hierarchy backoff: exact code + 3-ch
  (e.g., 992xx family, 850xx family, G03x family).
===========================================================
INSTALL (run once)
===========================================================
pip install -U pandas numpy scipy scikit-learn xgboost catboost pymupdf joblib tq
===========================================================
PATHS (Windows)
===========================================================
Train: D:\AgentDs\agent_ds_healthcare\ed_cost_train.csv
Test : D:\AgentDs\agent_ds_healthcare\ed_cost_test.csv
PDFs : D:\AgentDs\agent_ds_healthcare\receipts_pdf\receipt_{patient_id}.pdf

Optional: D:\AgentDs\agent_ds_healthcare\patients.csv
Output:
submission_ICHI_V1.csv (in the base dir)
===========================================================
"""
from __future__ import annotations
import re
import math
import time
import logging
import warnings
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import fitz  # PyMuPDF
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import xgboost as xgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor, Pool
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =========================
# CONFIG
# =========================
BASE_DIR = Path(r"D:\AgentDs\agent_ds_healthcare")
TRAIN_CSV = BASE_DIR / "ed_cost_train.csv"
TEST_CSV = BASE_DIR / "ed_cost_test.csv"
PATIENTS_CSV = BASE_DIR / "patients.csv"          # optional
PDF_DIR = BASE_DIR / "receipts_pdf"
SUBMISSION_PATH = BASE_DIR / "submission_ICHI_V1.csv"
CACHE_PATH = BASE_DIR / "cache_receipts_iter3_domain.joblib"
CACHE_VERSION = 31
# CV (keep practical; set FAST_MODE=False if you want more rigor)
FAST_MODE = True
N_FOLDS = 5
REPEAT_SEEDS = [2025, 2026] if FAST_MODE else [2024, 2025, 2026]
EARLY_STOPPING = 200
# Target encoding
TE_SMOOTH = 20.0
TE_MIN_COUNT_MEDIAN = 5
# GPU
XGB_DEVICE = "cuda"       # "cuda" or "cuda:0"
CAT_TASK_TYPE = "GPU"     # will auto-fallback to CPU if GPU fails
# Post-process
CLIP_AT_ZERO = True
# Blending
BLEND_STEP = 0.02
# Verbose heartbeats
XGB_VERBOSE_EVERY = 100
CAT_VERBOSE_EVERY = 100

# =========================
# LOGGING
# =========================
def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | 
# =========================
# REGEX (RAW STRINGS)
# =========================
_money_re = re.compile(r"^\d{1,3}(?:,\d{3})*(?:\.\d{2})$|^\d+(?:\.\d{2})$")
_code_re = re.compile(r"^(?:[A-Z]\d{4}|\d{4,5})$")
def is_money(tok: str) -> bool:
    return bool(_money_re.match(tok))
def money_to_float(tok: str) -> float:
    return float(tok.replace(",", ""))
def is_code(tok: str) -> bool:
    return bool(_code_re.match(tok))
def safe_log1p(x: float) -> float:
    return math.log1p(max(0.0, float(x)))
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))
def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_out
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)

# =========================
# CPT/HCPCS DOMAIN HELPERS
# =========================
def code_prefix3(code: str) -> str:
    # Hierarchical family (992xx, 850xx, G03x, etc.)
    return code[:3] if len(code) >= 3 else code
def parse_cpt_int(code: str) -> Optional[int]:
    # numeric CPT codes may have leading zeros; int() will drop them which is fin
    if code.isdigit():
        try:
            return int(code)
        except Exception:
            return None
    return None
def cpt_major_section(code: str) -> Tuple[str, str]:
    """
    Returns (section, detail)
    section in:
      em, anesthesia, surgery, radiology, lab, medicine, hcpcs, other
    detail: for hcpcs -> letter, for numeric -> ""
    CPT major range groupings used:
      - Anesthesia: 00100–01999 and 99100–99140
      - Surgery: 10021–69990
      - Radiology: 70010–79999
      - Path/Lab: 80047–89398
      - Medicine: 90281–99607
      - E/M: 99201–99499
    """
    if not code:
        return ("other", "")
    if code[0].isalpha():
        return ("hcpcs", code[0])

    n = parse_cpt_int(code)
    if n is None:
        return ("other", "")
    # anesthesia (00100-01999) => int 100-1999 after stripping leading zeros
    if (100 <= n <= 1999) or (99100 <= n <= 99140):
        return ("anesthesia", "")
    if 99201 <= n <= 99499:
        return ("em", "")
    if 10021 <= n <= 69990:
        return ("surgery", "")
    if 70010 <= n <= 79999:
        return ("radiology", "")
    if 80047 <= n <= 89398:
        return ("lab", "")
    if 90281 <= n <= 99607:
        return ("medicine", "")
    return ("other", "")
def ed_em_level(code: str) -> int:
    """
    ED E/M visit levels: 99281–99285 => level 1..5
    """
    n = parse_cpt_int(code) if code and code[0].isdigit() else None
    if n is None:
        return 0
    if 99281 <= n <= 99285:
        return int(n - 99280)
    return 0
def is_critical_care(code: str) -> bool:
    # 99291, 99292

    n = parse_cpt_int(code) if code and code[0].isdigit() else None
    return bool(n in (99291, 99292))
# =========================
# PDF PARSING
# =========================
def cluster_words_into_lines(words: List[Tuple], y_tol: float = 1.5) -> List[List
    if not words:
        return []
    words_sorted = sorted(words, key=lambda w: (w[1], w[0]))
    lines: List[List[str]] = []
    cur: List[Tuple[float, str]] = []
    cur_y: Optional[float] = None
    for w in words_sorted:
        x0, y0 = float(w[0]), float(w[1])
        tok = str(w[4])
        if cur_y is None or abs(y0 - cur_y) <= y_tol:
            cur.append((x0, tok))
            cur_y = y0 if cur_y is None else (0.85 * cur_y + 0.15 * y0)
        else:
            lines.append([t for _, t in sorted(cur, key=lambda z: z[0])])
            cur = [(x0, tok)]
            cur_y = y0
    if cur:
        lines.append([t for _, t in sorted(cur, key=lambda z: z[0])])
    return lines
def parse_header_fields(text: str) -> Tuple[Optional[int], Optional[str], Optiona
    pid = None
    zip3 = None
    insurance = None
    m = re.search(r"Patient ID:\s*(\d+)", text)
    if m:

        pid = int(m.group(1))
    m = re.search(r"ZIP3:\s*([0-9]{3})\s+Insurance:\s*([A-Za-z_]+)", text)
    if m:
        zip3 = str(m.group(1))
        insurance = str(m.group(2)).lower()
    return pid, zip3, insurance
def parse_items_from_words(words: List[Tuple]) -> Tuple[List[Dict[str, Any]], Opt
    items: List[Dict[str, Any]] = []
    total: Optional[float] = None
    lines = cluster_words_into_lines(words, y_tol=1.5)
    if not lines:
        return items, total
    header_idx = None
    for i, toks in enumerate(lines):
        low = [t.lower() for t in toks]
        if ("code" in low) and ("description" in low) and ("qty" in low):
            header_idx = i
            break
    if header_idx is None:
        return items, total
    for toks in lines[header_idx + 1:]:
        if not toks:
            continue
        if toks[0].upper() == "TOTAL":
            for t in reversed(toks[1:]):
                if is_money(t):
                    total = money_to_float(t)
                    break
            break
        if len(toks) >= 5 and is_code(toks[0]) and toks[-3].isdigit() and is_money
            items.append({

                "code": toks[0],
                "qty": int(toks[-3]),
                "unit": money_to_float(toks[-2]),
                "line_total": money_to_float(toks[-1]),
                "desc": " ".join(toks[1:-3]),
            })
    return items, total
def parse_items_fallback_text(text: str) -> Tuple[List[Dict[str, Any]], Optional[
    items: List[Dict[str, Any]] = []
    total: Optional[float] = None
    lines = [ln.strip() for ln in (text.splitlines() if text else []) if ln.strip
    if not lines:
        return items, total
    start_idx = None
    for i in range(len(lines) - 4):
        if lines[i].lower() == "code" and lines[i + 1].lower() == "description" a
            start_idx = i + 5
            break
        if "code" in lines[i].lower() and "description" in lines[i].lower() and "
            start_idx = i + 1
            break
    if start_idx is None:
        return items, total
    i = start_idx
    while i < len(lines):
        if lines[i].upper().startswith("TOTAL"):
            parts = lines[i].split()
            if len(parts) >= 2 and is_money(parts[-1]):
                total = money_to_float(parts[-1])
            elif i + 1 < len(lines) and is_money(lines[i + 1]):
                total = money_to_float(lines[i + 1])
            break

        if i + 4 < len(lines) and is_code(lines[i]) and lines[i + 2].isdigit() an
            items.append({
                "code": lines[i],
                "desc": lines[i + 1],
                "qty": int(lines[i + 2]),
                "unit": money_to_float(lines[i + 3]),
                "line_total": money_to_float(lines[i + 4]),
            })
            i += 5
        else:
            i += 1
    return items, total
@dataclass
class ReceiptProfile:
    meta: Dict[str, Any]
    code_spend: Dict[str, float]
    code_qty: Dict[str, float]
    code_lines: Dict[str, int]
    kw_spend: Dict[str, float]
    kw_lines: Dict[str, int]
KW_GROUPS = {
    "kw_ed_visit": [r"ed visit"],
    "kw_critical": [r"critical care"],
    "kw_observation": [r"observation"],
    "kw_intubation": [r"intubation"],
    "kw_catheter": [r"catheter"],
    "kw_lab": [r"\bcvc\b", r"\bcbc\b", r"metabolic", r"panel", r"differential", r
    "kw_imaging": [r"\bct\b", r"\bmri\b", r"x-?ray", r"ultrasound", r"radiograph"
}
def featurize_receipt(patient_id: int, pdf_dir: Path) -> ReceiptProfile:
    pdf_path = pdf_dir / f"receipt_{patient_id}.pdf"
    empty_meta = dict(

        pdf_ok=0, pdf_n_lines=0, pdf_n_unique_codes=0,
        pdf_sum_qty=0.0,
        pdf_sum_line_total=0.0, pdf_mean_line_total=0.0, pdf_max_line_total=0.0, 
        pdf_top1_cost_share=0.0, pdf_top3_cost_share=0.0,
        pdf_total=0.0, pdf_total_diff=0.0,
        pdf_insurance=None, pdf_zip3=None, pdf_parse_notes="missing_pdf",
    )
    if not pdf_path.exists():
        return ReceiptProfile(meta=empty_meta, code_spend={}, code_qty={}, code_l
    try:
        with fitz.open(pdf_path) as doc:
            full_text = ""
            items: List[Dict[str, Any]] = []
            total: Optional[float] = None
            zip3 = None
            insurance = None
            for page in doc:
                text = page.get_text("text") or ""
                full_text += text + "\n"
                words = page.get_text("words") or []
                items_p, total_p = parse_items_from_words(words)
                if not items_p:
                    items_p, total_p = parse_items_fallback_text(text)
                if items_p:
                    items.extend(items_p)
                if total_p is not None:
                    total = total_p
            _, zip3, insurance = parse_header_fields(full_text)
    except Exception as e:
        empty_meta["pdf_parse_notes"] = f"open_error:{type(e).__name__}"
        return ReceiptProfile(meta=empty_meta, code_spend={}, code_qty={}, code_l
    if not items:

        empty_meta["pdf_parse_notes"] = "no_items_parsed"
        empty_meta["pdf_zip3"] = zip3
        empty_meta["pdf_insurance"] = insurance
        return ReceiptProfile(meta=empty_meta, code_spend={}, code_qty={}, code_l
    code_spend: Dict[str, float] = {}
    code_qty: Dict[str, float] = {}
    code_lines: Dict[str, int] = {}
    kw_spend: Dict[str, float] = {k: 0.0 for k in KW_GROUPS.keys()}
    kw_lines: Dict[str, int] = {k: 0 for k in KW_GROUPS.keys()}
    for it in items:
        c = str(it["code"])
        q = float(it.get("qty", 0))
        lt = float(it.get("line_total", 0.0))
        desc = str(it.get("desc", "")).lower()
        code_spend[c] = code_spend.get(c, 0.0) + lt
        code_qty[c] = code_qty.get(c, 0.0) + q
        code_lines[c] = code_lines.get(c, 0) + 1
        for group, pats in KW_GROUPS.items():
            for pat in pats:
                if re.search(pat, desc):
                    kw_spend[group] += lt
                    kw_lines[group] += 1
                    break
    line_totals = np.array([float(it.get("line_total", 0.0)) for it in items], dty
    qtys = np.array([float(it.get("qty", 0.0)) for it in items], dtype=float)
    sum_line_total = float(line_totals.sum())
    pdf_total = float(total) if total is not None else sum_line_total
    sorted_cost = np.sort(line_totals)[::-1]
    top1 = float(sorted_cost[0] / sum_line_total) if sum_line_total > 0 else 0.0
    top3 = float(sorted_cost[:3].sum() / sum_line_total) if sum_line_total > 0 el
    meta = dict(

        pdf_ok=1,
        pdf_n_lines=int(len(items)),
        pdf_n_unique_codes=int(len(code_spend)),
        pdf_sum_qty=float(qtys.sum()),
        pdf_sum_line_total=float(sum_line_total),
        pdf_mean_line_total=float(line_totals.mean()) if len(items) else 0.0,
        pdf_max_line_total=float(line_totals.max()) if len(items) else 0.0,
        pdf_std_line_total=float(line_totals.std()) if len(items) else 0.0,
        pdf_top1_cost_share=float(top1),
        pdf_top3_cost_share=float(top3),
        pdf_total=float(pdf_total),
        pdf_total_diff=float(abs(sum_line_total - pdf_total)),
        pdf_insurance=(insurance.lower() if isinstance(insurance, str) else None)
        pdf_zip3=(str(zip3) if zip3 is not None else None),
        pdf_parse_notes="ok",
    )
    return ReceiptProfile(meta=meta, code_spend=code_spend, code_qty=code_qty, co
                          kw_spend=kw_spend, kw_lines=kw_lines)
def load_cache(cache_path: Path) -> Dict[str, Any]:
    if cache_path.exists():
        try:
            obj = joblib.load(cache_path)
            if isinstance(obj, dict) and obj.get("version") == CACHE_VERSION and 
                return obj
        except Exception:
            pass
    return {"version": CACHE_VERSION, "data": {}}
def save_cache(cache_path: Path, cache_obj: Dict[str, Any]) -> None:
    try:
        joblib.dump(cache_obj, cache_path)
    except Exception as e:
        logging.warning(f"Cache save failed: {e}")
def build_receipt_artifacts(patient_ids: List[int], pdf_dir: Path, cache_path: Pa

    cache_obj = load_cache(cache_path)
    data: Dict[str, Any] = cache_obj.get("data", {})
    to_parse = [pid for pid in patient_ids if str(pid) not in data]
    if to_parse:
        logging.info(f"Receipt parsing: {len(to_parse):,} cache-miss PDFs to pars
    else:
        logging.info("Receipt parsing: all PDFs loaded from cache.")
    for pid in tqdm(to_parse, desc="Parsing receipts", ncols=100):
        prof = featurize_receipt(pid, pdf_dir)
        data[str(pid)] = {
            "meta": prof.meta,
            "code_spend": prof.code_spend,
            "code_qty": prof.code_qty,
            "code_lines": prof.code_lines,
            "kw_spend": prof.kw_spend,
            "kw_lines": prof.kw_lines,
        }
    cache_obj["data"] = data
    if to_parse:
        save_cache(cache_path, cache_obj)
    meta_rows = []
    profiles: Dict[int, Dict[str, Any]] = {}
    for pid in patient_ids:
        obj = data.get(str(pid), None)
        if obj is None:
            prof = featurize_receipt(pid, pdf_dir)
            obj = {
                "meta": prof.meta,
                "code_spend": prof.code_spend,
                "code_qty": prof.code_qty,
                "code_lines": prof.code_lines,
                "kw_spend": prof.kw_spend,
                "kw_lines": prof.kw_lines,
            }
        meta_rows.append({"patient_id": pid, **obj["meta"]})
        profiles[pid] = obj

    receipt_meta_df = pd.DataFrame(meta_rows)
    return receipt_meta_df, profiles
# =========================
# DOMAIN FEATURE ENGINEERING
# =========================
def compute_domain_features_for_patient(pid: int, prof: Dict[str, Any]) -> Dict[st
    meta = prof.get("meta", {}) or {}
    code_spend: Dict[str, float] = prof.get("code_spend", {}) or {}
    code_qty: Dict[str, float] = prof.get("code_qty", {}) or {}
    code_lines: Dict[str, int] = prof.get("code_lines", {}) or {}
    kw_spend: Dict[str, float] = prof.get("kw_spend", {}) or {}
    kw_lines: Dict[str, int] = prof.get("kw_lines", {}) or {}
    total_spend = float(meta.get("pdf_total", 0.0)) if meta.get("pdf_ok", 0) == 1 
    if total_spend <= 0:
        total_spend = float(sum(code_spend.values()))
    feats: Dict[str, Any] = {}
    # Overall
    feats["clin_total_spend_pdf"] = float(total_spend)
    feats["clin_total_unique_codes"] = float(len(code_spend))
    feats["clin_total_lines"] = float(sum(code_lines.values())) if code_lines els
    feats["clin_total_qty"] = float(sum(code_qty.values())) if code_qty else 0.0
    # Major sections
    sections = ["em", "anesthesia", "surgery", "radiology", "lab", "medicine", "h
    sec_spend = {s: 0.0 for s in sections}
    sec_lines = {s: 0.0 for s in sections}
    sec_unique = {s: 0.0 for s in sections}
    sec_qty = {s: 0.0 for s in sections}
    # HCPCS letter breakout (small fixed set)
    hcpcs_letters = ["G", "J", "A", "Q", "S", "T", "V", "C", "K", "L", "M", "P"]
    hcpcs_spend = {f"hcpcs_{L}_spend": 0.0 for L in hcpcs_letters}
    hcpcs_lines = {f"hcpcs_{L}_lines": 0.0 for L in hcpcs_letters}

    # ED E/M acuity
    ed_cnt = {lvl: 0.0 for lvl in range(1, 6)}
    ed_sp = {lvl: 0.0 for lvl in range(1, 6)}
    # Critical care
    crit_99291_cnt = 0.0
    crit_99292_cnt = 0.0
    crit_spend = 0.0
    # Observation
    obs_hours = 0.0
    obs_spend = 0.0
    obs_lines = 0.0
    # Surgery coarse 10k buckets (domain-ish hierarchy)
    surg_buckets = [
        (10000, 19999, "surg_10_19"),
        (20000, 29999, "surg_20_29"),
        (30000, 39999, "surg_30_39"),
        (40000, 49999, "surg_40_49"),
        (50000, 59999, "surg_50_59"),
        (60000, 69999, "surg_60_69"),
    ]
    for _, _, name in surg_buckets:
        feats[f"{name}_spend"] = 0.0
        feats[f"{name}_lines"] = 0.0
        feats[f"{name}_unique"] = 0.0
    # Iterate codes
    for code, spend in code_spend.items():
        spend_f = float(spend)
        lines_f = float(code_lines.get(code, 1))
        qty_f = float(code_qty.get(code, 0.0))
        sec, detail = cpt_major_section(code)
        if sec not in sec_spend:
            sec = "other"
        sec_spend[sec] += spend_f
        sec_lines[sec] += lines_f
        sec_qty[sec] += qty_f

        sec_unique[sec] += 1.0
        # HCPCS letter detail
        if sec == "hcpcs" and detail in hcpcs_letters:
            hcpcs_spend[f"hcpcs_{detail}_spend"] += spend_f
            hcpcs_lines[f"hcpcs_{detail}_lines"] += lines_f
        # ED E/M levels
        lvl = ed_em_level(code)
        if lvl in ed_cnt:
            ed_cnt[lvl] += lines_f
            ed_sp[lvl] += spend_f
        # Critical care
        if code == "99291":
            crit_99291_cnt += lines_f
            crit_spend += spend_f
        elif code == "99292":
            crit_99292_cnt += lines_f
            crit_spend += spend_f
        # Observation (facility, per hour)
        if code == "G0378":
            obs_lines += lines_f
            obs_spend += spend_f
            obs_hours += qty_f
        # Surgery buckets
        n = parse_cpt_int(code)
        if n is not None and 10021 <= n <= 69990:
            for lo, hi, name in surg_buckets:
                if lo <= n <= hi:
                    feats[f"{name}_spend"] += spend_f
                    feats[f"{name}_lines"] += lines_f
                    feats[f"{name}_unique"] += 1.0
                    break
    # Major section features
    denom = total_spend + 1e-9
    for s in sections:
ChatGPT 5.2 Pro

        feats[f"sec_{s}_spend"] = float(sec_spend[s])
        feats[f"sec_{s}_share"] = float(sec_spend[s] / denom)
        feats[f"sec_{s}_lines"] = float(sec_lines[s])
        feats[f"sec_{s}_unique"] = float(sec_unique[s])
        feats[f"sec_{s}_qty"] = float(sec_qty[s])
    # HCPCS letter spend/lines + shares
    for k, v in hcpcs_spend.items():
        feats[k] = float(v)
        feats[k.replace("_spend", "_share")] = float(v / denom)
    for k, v in hcpcs_lines.items():
        feats[k] = float(v)
    # ED E/M derived stats
    ed_total_cnt = float(sum(ed_cnt.values()))
    ed_total_spend = float(sum(ed_sp.values()))
    feats["ed_total_cnt"] = ed_total_cnt
    feats["ed_total_spend"] = ed_total_spend
    feats["ed_total_share"] = float(ed_total_spend / denom)
    max_lvl = 0
    for lvl in range(5, 0, -1):
        if ed_cnt[lvl] > 0 or ed_sp[lvl] > 0:
            max_lvl = lvl
            break
    feats["ed_max_level"] = float(max_lvl)
    # mean level weighted by count and by spend
    if ed_total_cnt > 0:
        feats["ed_mean_level_cntw"] = float(sum(lvl * ed_cnt[lvl] for lvl in rang
    else:
        feats["ed_mean_level_cntw"] = 0.0
    if ed_total_spend > 0:
        feats["ed_mean_level_spw"] = float(sum(lvl * ed_sp[lvl] for lvl in range(
    else:
        feats["ed_mean_level_spw"] = 0.0
    # per-level features
    for lvl in range(1, 6):

        feats[f"ed_lvl{lvl}_cnt"] = float(ed_cnt[lvl])
        feats[f"ed_lvl{lvl}_spend"] = float(ed_sp[lvl])
        feats[f"ed_lvl{lvl}_share"] = float(ed_sp[lvl] / denom)
    feats["ed_high_cnt_45"] = float(ed_cnt[4] + ed_cnt[5])
    feats["ed_high_share_45"] = float((ed_sp[4] + ed_sp[5]) / denom)
    # Critical care features
    feats["crit_99291_cnt"] = float(crit_99291_cnt)
    feats["crit_99292_cnt"] = float(crit_99292_cnt)
    feats["crit_blocks_cnt"] = float(crit_99291_cnt + crit_99292_cnt)
    feats["crit_spend"] = float(crit_spend)
    feats["crit_share"] = float(crit_spend / denom)
    # Observation features
    feats["obs_g0378_lines"] = float(obs_lines)
    feats["obs_g0378_spend"] = float(obs_spend)
    feats["obs_g0378_share"] = float(obs_spend / denom)
    feats["obs_g0378_hours"] = float(obs_hours)
    # Keyword-rule features (spend + share + line counts)
    for k in KW_GROUPS.keys():
        s = float(kw_spend.get(k, 0.0))
        c = float(kw_lines.get(k, 0))
        feats[f"{k}_spend"] = s
        feats[f"{k}_share"] = float(s / denom)
        feats[f"{k}_lines"] = c
    # Ratios capturing “E/M vs Procedures” intensity
    proc_spend = sec_spend["surgery"] + sec_spend["radiology"] + sec_spend["lab"] 
    feats["em_to_proc_spend_ratio"] = float(sec_spend["em"] / (proc_spend + 1e-6)
    feats["lab_to_total_ratio"] = float(sec_spend["lab"] / denom)
    feats["surg_to_total_ratio"] = float(sec_spend["surgery"] / denom)
    # Simple composite acuity score (let model learn; this is just a compressed s
    feats["acuity_score"] = float(
        feats["ed_mean_level_spw"]
        + 1.5 * math.log1p(feats["crit_blocks_cnt"])
        + 0.05 * feats["obs_g0378_hours"]
        + 0.25 * math.log1p(sec_lines["surgery"])

        + 0.10 * feats["sec_radiology_share"] * 10.0
    )
    return feats
def build_domain_feature_df(patient_ids: List[int], profiles: Dict[int, Dict[str, 
    rows = []
    for pid in tqdm(patient_ids, desc="Domain features", ncols=100):
        prof = profiles.get(pid, {}) or {}
        feats = compute_domain_features_for_patient(pid, prof)
        rows.append({"patient_id": pid, **feats})
    return pd.DataFrame(rows)
# =========================
# BASE MATRIX (XGB)
# =========================
def build_base_sparse_matrix(df_all: pd.DataFrame) -> Tuple[csr_matrix, Dict[str, 
    df = df_all.copy()
    # derived tabular
    df["prior_cost_per_visit"] = df["prior_ed_cost_5y_usd"] / (df["prior_ed_visit
    df["log_prior_cost"] = np.log1p(df["prior_ed_cost_5y_usd"].astype(float))
    df["log_prior_visits"] = np.log1p(df["prior_ed_visits_5y"].astype(float))
    # normalize cats
    for c in ["primary_chronic", "sex", "insurance", "zip3", "pdf_insurance", "pd
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("missing")
    exclude = {"patient_id", "ed_cost_next3y_usd", "is_train"}
    cat_cols = [c for c in ["primary_chronic", "sex", "insurance", "zip3", "pdf_i
                if c in df.columns]
    num_cols = [c for c in df.columns
                if c not in exclude and c not in cat_cols and pd.api.types.is_num
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), nu

            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    X = preprocessor.fit_transform(df)
    X = X.tocsr() if hasattr(X, "tocsr") else csr_matrix(X)
    return X, {"preprocessor": preprocessor, "num_cols": num_cols, "cat_cols": cat
# =========================
# CV-SAFE TARGET ENCODING (exact code + prefix3)
# =========================
@dataclass
class TokenStats:
    cnt: int
    mean_smooth: float
    median_adj: float
def build_token_stats_for_fold(
    train_pids: List[int],
    y_log: np.ndarray,
    pid_to_row: Dict[int, int],
    profiles: Dict[int, Dict[str, Any]],
) -> Tuple[Dict[str, TokenStats], Dict[str, TokenStats], float, float]:
    vals_code: Dict[str, List[float]] = {}
    vals_p3: Dict[str, List[float]] = {}
    ys = []
    for pid in train_pids:
        r = pid_to_row.get(pid, None)
        if r is None:
            continue
        v = float(y_log[r])
        ys.append(v)
        code_spend = profiles.get(pid, {}).get("code_spend", {}) or {}
        for c in code_spend.keys():

            vals_code.setdefault(c, []).append(v)
            vals_p3.setdefault(code_prefix3(c), []).append(v)
    gmean = float(np.mean(ys)) if ys else 0.0
    gmed = float(np.median(ys)) if ys else 0.0
    def to_stats(m: Dict[str, List[float]]) -> Dict[str, TokenStats]:
        out: Dict[str, TokenStats] = {}
        for tok, arr in m.items():
            a = np.asarray(arr, dtype=float)
            cnt = int(a.size)
            mean = float(np.mean(a))
            med = float(np.median(a))
            mean_smooth = float((cnt * mean + TE_SMOOTH * gmean) / (cnt + TE_SMOO
            median_adj = med if cnt >= TE_MIN_COUNT_MEDIAN else gmed
            out[tok] = TokenStats(cnt=cnt, mean_smooth=mean_smooth, median_adj=me
        return out
    return to_stats(vals_code), to_stats(vals_p3), gmean, gmed
def te_features_for_patients(
    pids: List[int],
    profiles: Dict[int, Dict[str, Any]],
    stats_code: Dict[str, TokenStats],
    stats_p3: Dict[str, TokenStats],
    gmean: float,
    gmed: float,
) -> np.ndarray:
    """
    16 TE features:
      exact-code (8) + prefix3 (8), spend-weighted.
    """
    feats = np.zeros((len(pids), 16), dtype=np.float32)
    for i, pid in enumerate(pids):
        code_spend = profiles.get(pid, {}).get("code_spend", {}) or {}
        if not code_spend:
            base = np.array([gmean, gmed, gmean, gmean, gmean, gmed, 0.0, 0.0], dt
            feats[i, 0:8] = base

            feats[i, 8:16] = base
            continue
        codes = list(code_spend.keys())
        spends = np.array([float(code_spend[c]) for c in codes], dtype=float)
        ssum = float(spends.sum())
        w = (np.ones_like(spends) / max(1, spends.size)) if ssum <= 0 else (spend
        # exact code
        means, meds, cnts = [], [], []
        known = 0
        for c in codes:
            st = stats_code.get(c)
            if st is None:
                means.append(gmean); meds.append(gmed); cnts.append(0)
            else:
                means.append(st.mean_smooth); meds.append(st.median_adj); cnts.ap
        means = np.asarray(means, dtype=float)
        meds = np.asarray(meds, dtype=float)
        cnts = np.asarray(cnts, dtype=float)
        top = int(np.argmax(w)) if w.size else 0
        feats[i, 0:8] = np.array([
            float(np.sum(w * means)),
            float(np.sum(w * meds)),
            float(np.mean(means)),
            float(np.max(means)),
            float(np.min(means)),
            float(meds[top] if meds.size else gmed),
            float(known / max(1, len(codes))),
            float(np.mean(np.log1p(cnts))) if cnts.size else 0.0,
        ], dtype=np.float32)
        # prefix3
        p3_map: Dict[str, float] = {}
        for c, sp in code_spend.items():
            p3 = code_prefix3(c)
            p3_map[p3] = p3_map.get(p3, 0.0) + float(sp)
        p3s = list(p3_map.keys())
        p3sp = np.asarray([p3_map[p] for p in p3s], dtype=float)

        p3sum = float(p3sp.sum())
        w2 = (np.ones_like(p3sp) / max(1, p3sp.size)) if p3sum <= 0 else (p3sp / 
        means2, meds2, cnts2 = [], [], []
        known2 = 0
        for p in p3s:
            st = stats_p3.get(p)
            if st is None:
                means2.append(gmean); meds2.append(gmed); cnts2.append(0)
            else:
                means2.append(st.mean_smooth); meds2.append(st.median_adj); cnts2
        means2 = np.asarray(means2, dtype=float)
        meds2 = np.asarray(meds2, dtype=float)
        cnts2 = np.asarray(cnts2, dtype=float)
        top2 = int(np.argmax(w2)) if w2.size else 0
        feats[i, 8:16] = np.array([
            float(np.sum(w2 * means2)),
            float(np.sum(w2 * meds2)),
            float(np.mean(means2)),
            float(np.max(means2)),
            float(np.min(means2)),
            float(meds2[top2] if meds2.size else gmed),
            float(known2 / max(1, len(p3s))),
            float(np.mean(np.log1p(cnts2))) if cnts2.size else 0.0,
        ], dtype=np.float32)
    return feats
def make_strat_labels(train_df: pd.DataFrame) -> np.ndarray:
    chronic = train_df["primary_chronic"].astype(str)
    cost_bin = pd.qcut(train_df["prior_ed_cost_5y_usd"], q=10, duplicates="drop")
    return (chronic + "_" + cost_bin).values
# =========================
# BLEND
# =========================
def best_blend_weight(y_true: np.ndarray, p1: np.ndarray, p2: np.ndarray, step: f

    best_w = 0.5
    best_m = 1e18
    for w in np.arange(0.0, 1.0 + 1e-9, step):
        blend = w * p1 + (1.0 - w) * p2
        m = mae(y_true, blend)
        if m < best_m:
            best_m = m
            best_w = float(w)
    return best_w, best_m
# =========================
# MAIN
# =========================
def main() -> None:
    setup_logging()
    logging.info(f"XGBoost version: {getattr(xgb, '__version__', 'unknown')}")
    logging.info(f"FAST_MODE={FAST_MODE} | repeats={len(REPEAT_SEEDS)} | folds={N_
    logging.info(f"GPU config: XGB_DEVICE={XGB_DEVICE}, CAT_TASK_TYPE={CAT_TASK_TY
    # load
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"Missing: {TRAIN_CSV}")
    if not TEST_CSV.exists():
        raise FileNotFoundError(f"Missing: {TEST_CSV}")
    if not PDF_DIR.exists():
        raise FileNotFoundError(f"Missing PDF dir: {PDF_DIR}")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    logging.info(f"Loaded train={train_df.shape}, test={test_df.shape}")
    # optional patients.csv
    if PATIENTS_CSV.exists():
        patients = pd.read_csv(PATIENTS_CSV)
        if "patient_id" in patients.columns:
            train_df = train_df.merge(patients, on="patient_id", how="left")
            test_df = test_df.merge(patients, on="patient_id", how="left")
            logging.info(f"Merged patients.csv: train={train_df.shape}, test={test
        else:

            logging.warning("patients.csv exists but has no patient_id; skipping 
    else:
        logging.info("patients.csv not found; continuing without it.")
    train_df["is_train"] = 1
    test_df["is_train"] = 0
    df_all = pd.concat([train_df, test_df], ignore_index=True)
    # receipts parsing (cached)
    all_pids = df_all["patient_id"].astype(int).unique().tolist()
    all_pids.sort()
    receipt_meta_df, profiles = build_receipt_artifacts(all_pids, PDF_DIR, CACHE_
    df_all = df_all.merge(receipt_meta_df, on="patient_id", how="left")
    # domain features
    logging.info("Building structured clinical domain features from parsed codes.
    domain_df = build_domain_feature_df(all_pids, profiles)
    df_all = df_all.merge(domain_df, on="patient_id", how="left")
    # Tabular interactions (domain-driven)
    logging.info("Adding chronic/utilization interaction features...")
    df_all["prior_cost_per_visit"] = df_all["prior_ed_cost_5y_usd"] / (df_all["pr
    df_all["log_prior_cost"] = np.log1p(df_all["prior_ed_cost_5y_usd"].astype(flo
    df_all["log_prior_visits"] = np.log1p(df_all["prior_ed_visits_5y"].astype(flo
    # Chronic flags + interactions with acuity
    for chronic in ["HF", "DiabetesComp", "Pneumonia"]:
        col = f"is_{chronic}"
        df_all[col] = (df_all["primary_chronic"].astype(str) == chronic).astype(n
        df_all[f"acuity_x_{chronic}"] = df_all["acuity_score"].astype(float) * df_
        df_all[f"edmax_x_{chronic}"] = df_all["ed_max_level"].astype(float) * df_
        df_all[f"crit_x_{chronic}"] = df_all["crit_blocks_cnt"].astype(float) * d
    # Utilization x severity
    df_all["edmax_x_logvis"] = df_all["ed_max_level"].astype(float) * df_all["log_
    df_all["edmean_spw_x_logcost"] = df_all["ed_mean_level_spw"].astype(float) * 
    df_all["critblocks_x_logcost"] = df_all["crit_blocks_cnt"].astype(float) * df_
    df_all["obshrs_x_logvis"] = df_all["obs_g0378_hours"].astype(float) * df_all[
    # Base sparse for XGB

    logging.info("Building XGB base sparse matrix (OHE cats + numeric clinical fe
    X_all_base, base_art = build_base_sparse_matrix(df_all)
    logging.info(f"X_all_base shape: {X_all_base.shape}")
    is_train = df_all["is_train"].values.astype(int) == 1
    X_train_base = X_all_base[is_train]
    X_test_base = X_all_base[~is_train]
    y = train_df["ed_cost_next3y_usd"].values.astype(np.float32)
    y_log = np.log1p(y)
    train_pids = train_df["patient_id"].astype(int).tolist()
    test_pids = test_df["patient_id"].astype(int).tolist()
    pid_to_row = {pid: i for i, pid in enumerate(train_pids)}
    # CatBoost frames: use df_all but drop target & keys
    logging.info("Preparing CatBoost base frames...")
    df_cb_all = df_all.drop(columns=["ed_cost_next3y_usd"], errors="ignore").copy
    for c in ["primary_chronic", "sex", "insurance", "zip3", "pdf_insurance", "pd
        if c in df_cb_all.columns:
            df_cb_all[c] = df_cb_all[c].astype(str).fillna("missing")
    # Remove keys from CB features
    drop_cols = [c for c in ["patient_id", "is_train"] if c in df_cb_all.columns]
    df_cb_all = df_cb_all.drop(columns=drop_cols, errors="ignore")
    df_train_cb = df_cb_all[is_train].reset_index(drop=True)
    df_test_cb = df_cb_all[~is_train].reset_index(drop=True)
    cb_cat_cols = [c for c in ["primary_chronic", "sex", "insurance", "zip3", "pd
                   if c in df_train_cb.columns]
    cb_cat_idx = [df_train_cb.columns.get_loc(c) for c in cb_cat_cols]
    logging.info(f"CatBoost cat_features={cb_cat_cols}")
    # CV splits
    strat = make_strat_labels(train_df)
    splits = []
    for rep, seed in enumerate(REPEAT_SEEDS):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), strat

            splits.append((rep, fold, tr_idx, va_idx))
    logging.info(f"Total CV splits: {len(splits)}")
    # Storage
    oof_xgb = np.zeros(len(y), dtype=np.float64)
    oof_cat = np.zeros(len(y), dtype=np.float64)
    cnt_xgb = np.zeros(len(y), dtype=np.float64)
    cnt_cat = np.zeros(len(y), dtype=np.float64)
    test_xgb_sum = np.zeros(len(test_df), dtype=np.float64)
    test_cat_sum = np.zeros(len(test_df), dtype=np.float64)
    n_xgb = 0
    n_cat = 0
    # Model params (same core as Iter2; only verbosity increased)
    xgb_params = dict(
        objective="reg:absoluteerror",
        eval_metric="mae",
        tree_method="hist",
        device=XGB_DEVICE,
        n_estimators=6000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_lambda=1.0,
        min_child_weight=1.0,
        n_jobs=-1,
        verbosity=1,
        early_stopping_rounds=EARLY_STOPPING,
    )
    cat_params = dict(
        loss_function="MAE",
        eval_metric="MAE",
        iterations=8000,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=6.0,
        bagging_temperature=0.8,

        random_strength=0.5,
        od_type="Iter",
        od_wait=EARLY_STOPPING,
        task_type=CAT_TASK_TYPE,
        devices="0",
        verbose=CAT_VERBOSE_EVERY,
    )
    te_cols = [f"te_{i:02d}" for i in range(16)]
    fit_sig_xgb = None
    # CV loop
    for rep, fold, tr_idx, va_idx in tqdm(splits, desc="CV folds", ncols=110):
        t0 = time.time()
        fold_tag = f"rep={rep} fold={fold}"
        logging.info(f"{fold_tag} | START | train={len(tr_idx)} val={len(va_idx)}
        # Build TE stats for fold
        tr_pids = [train_pids[i] for i in tr_idx.tolist()]
        va_pids = [train_pids[i] for i in va_idx.tolist()]
        logging.info(f"{fold_tag} | TE stats build...")
        stats_code, stats_p3, gmean, gmed = build_token_stats_for_fold(tr_pids, y_
        logging.info(f"{fold_tag} | TE feature build (train/val/test)...")
        te_tr = te_features_for_patients(tr_pids, profiles, stats_code, stats_p3, 
        te_va = te_features_for_patients(va_pids, profiles, stats_code, stats_p3, 
        te_te = te_features_for_patients(test_pids, profiles, stats_code, stats_p
        # XGB matrices
        X_tr = hstack([X_train_base[tr_idx], csr_matrix(te_tr)], format="csr")
        X_va = hstack([X_train_base[va_idx], csr_matrix(te_va)], format="csr")
        X_te = hstack([X_test_base, csr_matrix(te_te)], format="csr")
        y_tr = y[tr_idx]
        y_va = y[va_idx]
        # ----- XGBoost -----
        logging.info(f"{fold_tag} | XGB fit START | device={xgb_params['device']} 
        try:

            xgb_model = XGBRegressor(**xgb_params, random_state=1000 * rep + fold
            # Use verbose_eval if supported; else use verbose=int (prints every n 
            if fit_sig_xgb is None:
                fit_sig_xgb = inspect.signature(xgb_model.fit).parameters
            fit_kwargs = {"eval_set": [(X_va, y_va)]}
            if "verbose_eval" in fit_sig_xgb:
                fit_kwargs["verbose_eval"] = XGB_VERBOSE_EVERY
            else:
                fit_kwargs["verbose"] = XGB_VERBOSE_EVERY
            xgb_model.fit(X_tr, y_tr, **fit_kwargs)
            p_va_x = xgb_model.predict(X_va)
            p_te_x = xgb_model.predict(X_te)
            best_it = getattr(xgb_model, "best_iteration", None)
            logging.info(f"{fold_tag} | XGB fit END | best_iteration={best_it}")
        except Exception as e:
            logging.warning(f"{fold_tag} | XGB GPU failed: {e} -> fallback CPU")
            xgb_cpu = dict(xgb_params)
            xgb_cpu["device"] = "cpu"
            xgb_model = XGBRegressor(**xgb_cpu, random_state=1000 * rep + fold)
            fit_sig_xgb = inspect.signature(xgb_model.fit).parameters
            fit_kwargs = {"eval_set": [(X_va, y_va)]}
            if "verbose_eval" in fit_sig_xgb:
                fit_kwargs["verbose_eval"] = XGB_VERBOSE_EVERY
            else:
                fit_kwargs["verbose"] = XGB_VERBOSE_EVERY
            logging.info(f"{fold_tag} | XGB (CPU fallback) fit START")
            xgb_model.fit(X_tr, y_tr, **fit_kwargs)
            logging.info(f"{fold_tag} | XGB (CPU fallback) fit END")
            p_va_x = xgb_model.predict(X_va)
            p_te_x = xgb_model.predict(X_te)
        if CLIP_AT_ZERO:
            p_va_x = np.maximum(p_va_x, 0.0)

            p_te_x = np.maximum(p_te_x, 0.0)
        oof_xgb[va_idx] += p_va_x
        cnt_xgb[va_idx] += 1.0
        test_xgb_sum += p_te_x
        n_xgb += 1
        # ----- CatBoost -----
        logging.info(f"{fold_tag} | CatBoost pools build...")
        te_train_full = np.zeros((len(train_df), 16), dtype=np.float32)
        te_train_full[tr_idx, :] = te_tr
        te_train_full[va_idx, :] = te_va
        df_tr = df_train_cb.copy()
        df_te = df_test_cb.copy()
        for j, col in enumerate(te_cols):
            df_tr[col] = te_train_full[:, j]
            df_te[col] = te_te[:, j]
        tr_pool = Pool(df_tr.iloc[tr_idx], label=y_tr, cat_features=cb_cat_idx)
        va_pool = Pool(df_tr.iloc[va_idx], label=y_va, cat_features=cb_cat_idx)
        te_pool = Pool(df_te, cat_features=cb_cat_idx)
        logging.info(f"{fold_tag} | CatBoost fit START | task_type={cat_params['t
        try:
            cat_model = CatBoostRegressor(**cat_params, random_seed=2000 * rep + 
            cat_model.fit(tr_pool, eval_set=va_pool, use_best_model=True)
            best_it = None
            try:
                best_it = cat_model.get_best_iteration()
            except Exception:
                best_it = None
            logging.info(f"{fold_tag} | CatBoost fit END | best_iteration={best_it
        except Exception as e:
            logging.warning(f"{fold_tag} | CatBoost GPU failed: {e} -> fallback C
            cat_cpu = dict(cat_params)
            cat_cpu["task_type"] = "CPU"
            cat_cpu.pop("devices", None)
            cat_model = CatBoostRegressor(**cat_cpu, random_seed=2000 * rep + fol

            logging.info(f"{fold_tag} | CatBoost (CPU fallback) fit START | verbo
            cat_model.fit(tr_pool, eval_set=va_pool, use_best_model=True)
            logging.info(f"{fold_tag} | CatBoost (CPU fallback) fit END")
        p_va_c = cat_model.predict(va_pool)
        p_te_c = cat_model.predict(te_pool)
        if CLIP_AT_ZERO:
            p_va_c = np.maximum(p_va_c, 0.0)
            p_te_c = np.maximum(p_te_c, 0.0)
        oof_cat[va_idx] += p_va_c
        cnt_cat[va_idx] += 1.0
        test_cat_sum += p_te_c
        n_cat += 1
        dt = time.time() - t0
        logging.info(f"{fold_tag} | END | fold_time={dt:.1f}s")
    # finalize
    oof_x = (oof_xgb / np.maximum(cnt_xgb, 1.0)).astype(np.float32)
    oof_c = (oof_cat / np.maximum(cnt_cat, 1.0)).astype(np.float32)
    te_x = (test_xgb_sum / max(1, n_xgb)).astype(np.float32)
    te_c = (test_cat_sum / max(1, n_cat)).astype(np.float32)
    logging.info(f"OOF MAE | XGB: {mae(y, oof_x):.4f}")
    logging.info(f"OOF MAE | CAT: {mae(y, oof_c):.4f}")
    # blend + median bias
    w, blend_mae = best_blend_weight(y, oof_x, oof_c, step=BLEND_STEP)
    oof_blend = w * oof_x + (1.0 - w) * oof_c
    bias = float(np.median(y - oof_blend))  # MAE-optimal constant shift
    oof_final = oof_blend + bias
    test_final = w * te_x + (1.0 - w) * te_c + bias
    if CLIP_AT_ZERO:
        oof_final = np.maximum(oof_final, 0.0)
        test_final = np.maximum(test_final, 0.0)

If you run this and paste the log lines:
OOF MAE | XGB: ...
OOF MAE | CAT: ...
Blend ... OOF MAE final= ...
…I can tell you whether your remaining ~30 MAE gap is primarily:
feature ceiling (need sharper clinical grouping or better TE hierarchy), or
modeling ceiling (need a different MAE-aligned objective strategy like quantile
stacking), or
CV mismatch (split needs to mimic hidden test distribution).
    logging.info(f"Blend w(XGB)={w:.2f} | OOF MAE pre-bias={blend_mae:.4f} | bias
    # Save submission
    sub = pd.DataFrame({
        "patient_id": test_df["patient_id"].astype(int).values,
        "ed_cost_next3y_usd": test_final.astype(np.float32),
    })
    sub.to_csv(SUBMISSION_PATH, index=False)
    logging.info(f"Saved submission: {SUBMISSION_PATH}")
if __name__ == "__main__":
    main()
```

### C07 — ITERATION 4 — "Return to the Math" (No manual CPT grouping) (pages 195-225)
```python
# -*- coding: utf-8 -*-
r"""
ITERATION 4 — "Return to the Math" (No manual CPT grouping)
==========================================================
Goal:

- Represent high-dimensional sparse code data (thousands of CPT/HCPCS-like codes) 
  XGBoost/CatBoost can digest efficiently, WITHOUT manual clinical hierarchies.
Core idea (Third Path):
1) Build sparse patient×code matrices from receipts (binary presence + spend weig
2) Learn dense embeddings via TruncatedSVD (LSA) on the sparse matrix (unsupervis
3) Cluster code embeddings (MiniBatchKMeans) => statistically similar code groups
4) Create patient "bag-of-clusters" features (counts + spend shares + entropy).
5) Add CV-safe target encoding (OOF) for exact codes + learned clusters (no manua
Models:
- XGBoost GPU (device=cuda, reg:absoluteerror)
- CatBoost GPU (MAE)
Output:
- submission_ICHI_V1.csv with columns: patient_id, ed_cost_next3y_usd
===========================================================
INSTALL (run once)
===========================================================
pip install -U pandas numpy scipy scikit-learn xgboost catboost pymupdf joblib tq
"""
from __future__ import annotations
import re
import math
import time
import logging
import warnings
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import fitz  # PyMuPDF

from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans
import xgboost as xgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor, Pool
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# =========================
# CONFIG
# =========================
BASE_DIR = Path(r"D:\AgentDs\agent_ds_healthcare")
TRAIN_CSV = BASE_DIR / "ed_cost_train.csv"
TEST_CSV = BASE_DIR / "ed_cost_test.csv"
PATIENTS_CSV = BASE_DIR / "patients.csv"          # optional
PDF_DIR = BASE_DIR / "receipts_pdf"
SUBMISSION_PATH = BASE_DIR / "submission_ICHI_V1.csv"
CACHE_PATH = BASE_DIR / "cache_receipts_iter4_math.joblib"
CACHE_VERSION = 41
# CV
FAST_MODE = True
N_FOLDS = 5
REPEAT_SEEDS = [2025, 2026] if FAST_MODE else [2024, 2025, 2026]

EARLY_STOPPING = 200
# Target encoding
TE_SMOOTH = 20.0
TE_MIN_COUNT_MEDIAN = 5
# Unsupervised representation
N_SVD_TFIDF = 64
N_SVD_SPEND = 32
N_CODE_CLUSTERS = 48
# GPU
XGB_DEVICE = "cuda"
CAT_TASK_TYPE = "GPU"
# Post-process
CLIP_AT_ZERO = True
# Blending
BLEND_STEP = 0.02
# Verbose heartbeats
XGB_VERBOSE_EVERY = 100
CAT_VERBOSE_EVERY = 100
# =========================
# LOGGING
# =========================
def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | 
# =========================
# REGEX (RAW STRINGS)
# =========================
_money_re = re.compile(r"^\d{1,3}(?:,\d{3})*(?:\.\d{2})$|^\d+(?:\.\d{2})$")
_code_re = re.compile(r"^(?:[A-Z]\d{4}|\d{4,5})$")

def is_money(tok: str) -> bool:
    return bool(_money_re.match(tok))
def money_to_float(tok: str) -> float:
    return float(tok.replace(",", ""))
def is_code(tok: str) -> bool:
    return bool(_code_re.match(tok))
def safe_log1p(x: float) -> float:
    return math.log1p(max(0.0, float(x)))
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))
def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_out
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)
# =========================
# PDF PARSING
# =========================
def cluster_words_into_lines(words: List[Tuple], y_tol: float = 1.5) -> List[List
    if not words:
        return []
    words_sorted = sorted(words, key=lambda w: (w[1], w[0]))
    lines: List[List[str]] = []
    cur: List[Tuple[float, str]] = []
    cur_y: Optional[float] = None
    for w in words_sorted:
        x0, y0 = float(w[0]), float(w[1])

        tok = str(w[4])
        if cur_y is None or abs(y0 - cur_y) <= y_tol:
            cur.append((x0, tok))
            cur_y = y0 if cur_y is None else (0.85 * cur_y + 0.15 * y0)
        else:
            lines.append([t for _, t in sorted(cur, key=lambda z: z[0])])
            cur = [(x0, tok)]
            cur_y = y0
    if cur:
        lines.append([t for _, t in sorted(cur, key=lambda z: z[0])])
    return lines
def parse_header_fields(text: str) -> Tuple[Optional[int], Optional[str], Optiona
    pid = None
    zip3 = None
    insurance = None
    m = re.search(r"Patient ID:\s*(\d+)", text)
    if m:
        pid = int(m.group(1))
    m = re.search(r"ZIP3:\s*([0-9]{3})\s+Insurance:\s*([A-Za-z_]+)", text)
    if m:
        zip3 = str(m.group(1))
        insurance = str(m.group(2)).lower()
    return pid, zip3, insurance
def parse_items_from_words(words: List[Tuple]) -> Tuple[List[Dict[str, Any]], Opt
    items: List[Dict[str, Any]] = []
    total: Optional[float] = None
    lines = cluster_words_into_lines(words, y_tol=1.5)
    if not lines:
        return items, total

    header_idx = None
    for i, toks in enumerate(lines):
        low = [t.lower() for t in toks]
        if ("code" in low) and ("description" in low) and ("qty" in low):
            header_idx = i
            break
    if header_idx is None:
        return items, total
    for toks in lines[header_idx + 1:]:
        if not toks:
            continue
        if toks[0].upper() == "TOTAL":
            for t in reversed(toks[1:]):
                if is_money(t):
                    total = money_to_float(t)
                    break
            break
        if len(toks) >= 5 and is_code(toks[0]) and toks[-3].isdigit() and is_money
            items.append({
                "code": toks[0],
                "qty": int(toks[-3]),
                "unit": money_to_float(toks[-2]),
                "line_total": money_to_float(toks[-1]),
                "desc": " ".join(toks[1:-3]),
            })
    return items, total
def parse_items_fallback_text(text: str) -> Tuple[List[Dict[str, Any]], Optional[
    items: List[Dict[str, Any]] = []
    total: Optional[float] = None
    lines = [ln.strip() for ln in (text.splitlines() if text else []) if ln.strip
    if not lines:
        return items, total

    start_idx = None
    for i in range(len(lines) - 4):
        if lines[i].lower() == "code" and lines[i + 1].lower() == "description" a
            start_idx = i + 5
            break
        if "code" in lines[i].lower() and "description" in lines[i].lower() and "
            start_idx = i + 1
            break
    if start_idx is None:
        return items, total
    i = start_idx
    while i < len(lines):
        if lines[i].upper().startswith("TOTAL"):
            parts = lines[i].split()
            if len(parts) >= 2 and is_money(parts[-1]):
                total = money_to_float(parts[-1])
            elif i + 1 < len(lines) and is_money(lines[i + 1]):
                total = money_to_float(lines[i + 1])
            break
        if i + 4 < len(lines) and is_code(lines[i]) and lines[i + 2].isdigit() an
            items.append({
                "code": lines[i],
                "desc": lines[i + 1],
                "qty": int(lines[i + 2]),
                "unit": money_to_float(lines[i + 3]),
                "line_total": money_to_float(lines[i + 4]),
            })
            i += 5
        else:
            i += 1
    return items, total
@dataclass
class ReceiptProfile:

    meta: Dict[str, Any]
    code_spend: Dict[str, float]
    code_qty: Dict[str, float]
    code_lines: Dict[str, int]
def featurize_receipt(patient_id: int, pdf_dir: Path) -> ReceiptProfile:
    pdf_path = pdf_dir / f"receipt_{patient_id}.pdf"
    empty_meta = dict(
        pdf_ok=0, pdf_n_lines=0, pdf_n_unique_codes=0,
        pdf_sum_qty=0.0,
        pdf_sum_line_total=0.0, pdf_mean_line_total=0.0, pdf_max_line_total=0.0, 
        pdf_top1_cost_share=0.0, pdf_top3_cost_share=0.0,
        pdf_total=0.0, pdf_total_diff=0.0,
        pdf_insurance=None, pdf_zip3=None, pdf_parse_notes="missing_pdf",
    )
    if not pdf_path.exists():
        return ReceiptProfile(meta=empty_meta, code_spend={}, code_qty={}, code_l
    try:
        with fitz.open(pdf_path) as doc:
            full_text = ""
            items: List[Dict[str, Any]] = []
            total: Optional[float] = None
            zip3 = None
            insurance = None
            for page in doc:
                text = page.get_text("text") or ""
                full_text += text + "\n"
                words = page.get_text("words") or []
                items_p, total_p = parse_items_from_words(words)
                if not items_p:
                    items_p, total_p = parse_items_fallback_text(text)
                if items_p:
                    items.extend(items_p)

                if total_p is not None:
                    total = total_p
            _, zip3, insurance = parse_header_fields(full_text)
    except Exception as e:
        empty_meta["pdf_parse_notes"] = f"open_error:{type(e).__name__}"
        return ReceiptProfile(meta=empty_meta, code_spend={}, code_qty={}, code_l
    if not items:
        empty_meta["pdf_parse_notes"] = "no_items_parsed"
        empty_meta["pdf_zip3"] = zip3
        empty_meta["pdf_insurance"] = insurance
        return ReceiptProfile(meta=empty_meta, code_spend={}, code_qty={}, code_l
    code_spend: Dict[str, float] = {}
    code_qty: Dict[str, float] = {}
    code_lines: Dict[str, int] = {}
    for it in items:
        c = str(it["code"])
        q = float(it.get("qty", 0.0))
        lt = float(it.get("line_total", 0.0))
        code_spend[c] = code_spend.get(c, 0.0) + lt
        code_qty[c] = code_qty.get(c, 0.0) + q
        code_lines[c] = code_lines.get(c, 0) + 1
    line_totals = np.array([float(it.get("line_total", 0.0)) for it in items], dty
    qtys = np.array([float(it.get("qty", 0.0)) for it in items], dtype=float)
    sum_line_total = float(line_totals.sum())
    pdf_total = float(total) if total is not None else sum_line_total
    sorted_cost = np.sort(line_totals)[::-1]
    top1 = float(sorted_cost[0] / sum_line_total) if sum_line_total > 0 else 0.0
    top3 = float(sorted_cost[:3].sum() / sum_line_total) if sum_line_total > 0 el
    meta = dict(
        pdf_ok=1,
        pdf_n_lines=int(len(items)),
        pdf_n_unique_codes=int(len(code_spend)),

        pdf_sum_qty=float(qtys.sum()),
        pdf_sum_line_total=float(sum_line_total),
        pdf_mean_line_total=float(line_totals.mean()) if len(items) else 0.0,
        pdf_max_line_total=float(line_totals.max()) if len(items) else 0.0,
        pdf_std_line_total=float(line_totals.std()) if len(items) else 0.0,
        pdf_top1_cost_share=float(top1),
        pdf_top3_cost_share=float(top3),
        pdf_total=float(pdf_total),
        pdf_total_diff=float(abs(sum_line_total - pdf_total)),
        pdf_insurance=(insurance.lower() if isinstance(insurance, str) else None)
        pdf_zip3=(str(zip3) if zip3 is not None else None),
        pdf_parse_notes="ok",
    )
    return ReceiptProfile(meta=meta, code_spend=code_spend, code_qty=code_qty, co
def load_cache(cache_path: Path) -> Dict[str, Any]:
    if cache_path.exists():
        try:
            obj = joblib.load(cache_path)
            if isinstance(obj, dict) and obj.get("version") == CACHE_VERSION and 
                return obj
        except Exception:
            pass
    return {"version": CACHE_VERSION, "data": {}}
def save_cache(cache_path: Path, cache_obj: Dict[str, Any]) -> None:
    try:
        joblib.dump(cache_obj, cache_path)
    except Exception as e:
        logging.warning(f"Cache save failed: {e}")
def build_receipt_artifacts(patient_ids: List[int], pdf_dir: Path, cache_path: Pa
    cache_obj = load_cache(cache_path)
    data: Dict[str, Any] = cache_obj.get("data", {})
    to_parse = [pid for pid in patient_ids if str(pid) not in data]

    if to_parse:
        logging.info(f"Receipt parsing: {len(to_parse):,} cache-miss PDFs to pars
    else:
        logging.info("Receipt parsing: all PDFs loaded from cache.")
    for pid in tqdm(to_parse, desc="Parsing receipts", ncols=100):
        prof = featurize_receipt(pid, pdf_dir)
        data[str(pid)] = {
            "meta": prof.meta,
            "code_spend": prof.code_spend,
            "code_qty": prof.code_qty,
            "code_lines": prof.code_lines,
        }
    cache_obj["data"] = data
    if to_parse:
        save_cache(cache_path, cache_obj)
    meta_rows = []
    profiles: Dict[int, Dict[str, Any]] = {}
    for pid in patient_ids:
        obj = data.get(str(pid), None)
        if obj is None:
            prof = featurize_receipt(pid, pdf_dir)
            obj = {
                "meta": prof.meta,
                "code_spend": prof.code_spend,
                "code_qty": prof.code_qty,
                "code_lines": prof.code_lines,
            }
        meta_rows.append({"patient_id": pid, **(obj["meta"] or {})})
        profiles[pid] = obj
    receipt_meta_df = pd.DataFrame(meta_rows)
    return receipt_meta_df, profiles
# =========================
# UNSUPERVISED REPRESENTATION (SVD + clustering)
# =========================

def build_code_vocab(profiles: Dict[int, Dict[str, Any]], patient_ids: List[int]) 
    vocab = set()
    for pid in patient_ids:
        d = profiles.get(pid, {}).get("code_spend", {}) or {}
        for c in d.keys():
            vocab.add(str(c))
    return sorted(vocab)
def build_sparse_code_matrices(
    patient_ids: List[int],
    profiles: Dict[int, Dict[str, Any]],
    code_to_idx: Dict[str, int],
) -> Tuple[csr_matrix, csr_matrix, csr_matrix]:
    """
    Returns:
      X_bin   (rows=patients, cols=codes): 1 if code present
      X_spend (rows=patients, cols=codes): raw spend
      X_qty   (rows=patients, cols=codes): raw qty
    """
    n_pat = len(patient_ids)
    n_code = len(code_to_idx)
    rb, cb, db = [], [], []
    rs, cs, ds = [], [], []
    rq, cq, dq = [], [], []
    for i, pid in enumerate(patient_ids):
        obj = profiles.get(pid, {}) or {}
        spend = obj.get("code_spend", {}) or {}
        qty = obj.get("code_qty", {}) or {}
        for c, sp in spend.items():
            c = str(c)
            j = code_to_idx.get(c, None)
            if j is None:
                continue
            rb.append(i); cb.append(j); db.append(1.0)
            rs.append(i); cs.append(j); ds.append(float(sp))
            rq.append(i); cq.append(j); dq.append(float(qty.get(c, 0.0)))

    X_bin = csr_matrix((np.asarray(db, dtype=np.float32), (np.asarray(rb), np.asa
    X_spend = csr_matrix((np.asarray(ds, dtype=np.float32), (np.asarray(rs), np.a
    X_qty = csr_matrix((np.asarray(dq, dtype=np.float32), (np.asarray(rq), np.asa
    return X_bin, X_spend, X_qty
def build_unsupervised_features(
    patient_ids: List[int],
    profiles: Dict[int, Dict[str, Any]],
    n_svd_tfidf: int,
    n_svd_spend: int,
    n_clusters: int,
    random_state: int = 2027,
) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int]]:
    """
    Produces:
      - patient-level dense SVD features (TFIDF-LSA and SPEND-SVD)
      - patient-level bag-of-clusters features (counts + spend shares + entropy)
      - code_to_cluster mapping for TE/cluster encoding
    """
    logging.info("Building code vocabulary from receipts...")
    vocab = build_code_vocab(profiles, patient_ids)
    code_to_idx = {c: i for i, c in enumerate(vocab)}
    n_pat = len(patient_ids)
    n_code = len(vocab)
    logging.info(f"Vocabulary size: {n_code:,} codes | patients: {n_pat:,}")
    if n_code == 0:
        # No codes at all -> return empty features
        df = pd.DataFrame({"patient_id": patient_ids})
        return df, code_to_idx, {}
    logging.info("Building sparse code matrices (binary/spend/qty)...")
    X_bin, X_spend, X_qty = build_sparse_code_matrices(patient_ids, profiles, cod
    # TF-IDF on binary presence
    logging.info("TF-IDF transform on binary matrix...")
    tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True, sublinear_
    X_tfidf = tfidf.fit_transform(X_bin)

    # SVD on TF-IDF (LSA)
    k1 = int(max(2, min(n_svd_tfidf, n_pat - 1, n_code - 1)))
    logging.info(f"Fitting TruncatedSVD on TF-IDF: n_components={k1} ...")
    svd_tfidf = TruncatedSVD(n_components=k1, random_state=random_state, algorith
    Z_tfidf = svd_tfidf.fit_transform(X_tfidf).astype(np.float32)
    # Spend composition SVD (normalize rows to shares, then SVD)
    logging.info("Building spend-composition matrix (L1 normalize raw spend)...")
    X_sp_share = normalize(X_spend, norm="l1", axis=1, copy=True)
    k2 = int(max(2, min(n_svd_spend, n_pat - 1, n_code - 1)))
    logging.info(f"Fitting TruncatedSVD on spend-share: n_components={k2} ...")
    svd_spend = TruncatedSVD(n_components=k2, random_state=random_state + 1, algo
    Z_spend = svd_spend.fit_transform(X_sp_share).astype(np.float32)
    # Cluster codes by their TF-IDF SVD embedding (statistical similarity)
    code_emb = svd_tfidf.components_.T  # (n_code, k1)
    k_clu = int(max(2, min(n_clusters, n_code)))
    logging.info(f"Clustering codes in embedding space: n_clusters={k_clu} ...")
    try:
        kmeans = MiniBatchKMeans(
            n_clusters=k_clu,
            random_state=random_state + 2,
            batch_size=2048,
            n_init="auto",
            max_iter=200,
        )
    except TypeError:
        kmeans = MiniBatchKMeans(
            n_clusters=k_clu,
            random_state=random_state + 2,
            batch_size=2048,
            n_init=10,
            max_iter=200,
        )
    code_cluster = kmeans.fit_predict(code_emb)
    idx_to_code = {i: c for c, i in code_to_idx.items()}
    code_to_cluster = {idx_to_code[i]: int(code_cluster[i]) for i in range(n_code

    # Patient bag-of-clusters features
    logging.info("Building patient bag-of-clusters features (count/share/entropy)
    clu_cnt = np.zeros((n_pat, k_clu), dtype=np.float32)
    clu_sp = np.zeros((n_pat, k_clu), dtype=np.float32)
    for i, pid in enumerate(patient_ids):
        obj = profiles.get(pid, {}) or {}
        spend = obj.get("code_spend", {}) or {}
        if not spend:
            continue
        total = float(sum(float(v) for v in spend.values()))
        total = max(total, 1e-9)
        for c, sp in spend.items():
            c = str(c)
            clu = code_to_cluster.get(c, None)
            if clu is None:
                continue
            clu_cnt[i, clu] += 1.0
            clu_sp[i, clu] += float(sp) / total  # spend share
    # Entropy + active clusters
    eps = 1e-12
    sp_p = np.clip(clu_sp, eps, 1.0)
    sp_p = sp_p / np.sum(sp_p, axis=1, keepdims=True)
    sp_entropy = -np.sum(sp_p * np.log(sp_p), axis=1).astype(np.float32)
    active_sp = (clu_sp > 0).sum(axis=1).astype(np.float32)
    cnt_sum = np.sum(clu_cnt, axis=1, keepdims=True)
    cnt_p = np.where(cnt_sum > 0, clu_cnt / (cnt_sum + eps), 0.0)
    cnt_p2 = np.clip(cnt_p, eps, 1.0)
    cnt_entropy = -np.sum(cnt_p2 * np.log(cnt_p2), axis=1).astype(np.float32)
    active_cnt = (clu_cnt > 0).sum(axis=1).astype(np.float32)
    # Build dataframe
    rows = {"patient_id": patient_ids}
    for j in range(k1):
        rows[f"lsa_tfidf_{j:03d}"] = Z_tfidf[:, j]
    for j in range(k2):
        rows[f"svd_spend_{j:03d}"] = Z_spend[:, j]

    for k in range(k_clu):
        rows[f"clu_cnt_{k:02d}"] = clu_cnt[:, k]
        rows[f"clu_sp_share_{k:02d}"] = clu_sp[:, k]
    rows["clu_sp_entropy"] = sp_entropy
    rows["clu_cnt_entropy"] = cnt_entropy
    rows["clu_active_sp"] = active_sp
    rows["clu_active_cnt"] = active_cnt
    feat_df = pd.DataFrame(rows)
    return feat_df, code_to_idx, code_to_cluster
# =========================
# BASE MATRIX (XGB)
# =========================
def build_base_sparse_matrix(df_all: pd.DataFrame) -> Tuple[csr_matrix, Dict[str, 
    df = df_all.copy()
    # derived tabular
    df["prior_cost_per_visit"] = df["prior_ed_cost_5y_usd"] / (df["prior_ed_visit
    df["log_prior_cost"] = np.log1p(df["prior_ed_cost_5y_usd"].astype(float))
    df["log_prior_visits"] = np.log1p(df["prior_ed_visits_5y"].astype(float))
    for c in ["primary_chronic", "sex", "insurance", "zip3", "pdf_insurance", "pd
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("missing")
    exclude = {"patient_id", "ed_cost_next3y_usd", "is_train"}
    cat_cols = [c for c in ["primary_chronic", "sex", "insurance", "zip3", "pdf_i
                if c in df.columns]
    num_cols = [c for c in df.columns
                if c not in exclude and c not in cat_cols and pd.api.types.is_num
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), nu
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")
        ],

        remainder="drop",
        sparse_threshold=0.3,
    )
    X = preprocessor.fit_transform(df)
    X = X.tocsr() if hasattr(X, "tocsr") else csr_matrix(X)
    return X, {"preprocessor": preprocessor, "num_cols": num_cols, "cat_cols": cat
# =========================
# CV-SAFE TARGET ENCODING (exact code + learned cluster)
# =========================
@dataclass
class TokenStats:
    cnt: int
    mean_smooth: float
    median_adj: float
def build_te_stats_for_fold(
    train_pids: List[int],
    y_log: np.ndarray,
    pid_to_row: Dict[int, int],
    profiles: Dict[int, Dict[str, Any]],
    code_to_cluster: Dict[str, int],
) -> Tuple[Dict[str, TokenStats], Dict[int, TokenStats], float, float]:
    """
    Build stats for:
      - exact code tokens (string) => TokenStats
      - code clusters (int) => TokenStats
    """
    vals_code: Dict[str, List[float]] = {}
    vals_clu: Dict[int, List[float]] = {}
    ys = []
    for pid in train_pids:
        r = pid_to_row.get(pid, None)
        if r is None:
            continue
        v = float(y_log[r])

        ys.append(v)
        spend = profiles.get(pid, {}).get("code_spend", {}) or {}
        for c in spend.keys():
            c = str(c)
            vals_code.setdefault(c, []).append(v)
            clu = code_to_cluster.get(c, None)
            if clu is not None:
                vals_clu.setdefault(int(clu), []).append(v)
    gmean = float(np.mean(ys)) if ys else 0.0
    gmed = float(np.median(ys)) if ys else 0.0
    def to_stats_str(m: Dict[Any, List[float]]) -> Dict[Any, TokenStats]:
        out: Dict[Any, TokenStats] = {}
        for tok, arr in m.items():
            a = np.asarray(arr, dtype=float)
            cnt = int(a.size)
            mean = float(np.mean(a))
            med = float(np.median(a))
            mean_smooth = float((cnt * mean + TE_SMOOTH * gmean) / (cnt + TE_SMOO
            median_adj = med if cnt >= TE_MIN_COUNT_MEDIAN else gmed
            out[tok] = TokenStats(cnt=cnt, mean_smooth=mean_smooth, median_adj=me
        return out
    return to_stats_str(vals_code), to_stats_str(vals_clu), gmean, gmed
def te_features_for_patients(
    pids: List[int],
    profiles: Dict[int, Dict[str, Any]],
    stats_code: Dict[str, TokenStats],
    stats_clu: Dict[int, TokenStats],
    code_to_cluster: Dict[str, int],
    gmean: float,
    gmed: float,
) -> np.ndarray:
    """
    16 TE features:
      exact-code (8) + cluster (8), spend-weighted.

    """
    feats = np.zeros((len(pids), 16), dtype=np.float32)
    for i, pid in enumerate(pids):
        spend = profiles.get(pid, {}).get("code_spend", {}) or {}
        if not spend:
            base = np.array([gmean, gmed, gmean, gmean, gmean, gmed, 0.0, 0.0], dt
            feats[i, 0:8] = base
            feats[i, 8:16] = base
            continue
        codes = list(spend.keys())
        sp = np.asarray([float(spend[c]) for c in codes], dtype=float)
        ssum = float(sp.sum())
        w = (np.ones_like(sp) / max(1, sp.size)) if ssum <= 0 else (sp / (ssum + 
        top = int(np.argmax(w)) if w.size else 0
        # ----- exact-code -----
        means, meds, cnts = [], [], []
        known = 0
        for c in codes:
            st = stats_code.get(str(c))
            if st is None:
                means.append(gmean); meds.append(gmed); cnts.append(0)
            else:
                means.append(st.mean_smooth); meds.append(st.median_adj); cnts.ap
        means = np.asarray(means, dtype=float)
        meds = np.asarray(meds, dtype=float)
        cnts = np.asarray(cnts, dtype=float)
        feats[i, 0:8] = np.array([
            float(np.sum(w * means)),
            float(np.sum(w * meds)),
            float(np.mean(means)),
            float(np.max(means)),
            float(np.min(means)),
            float(meds[top] if meds.size else gmed),
            float(known / max(1, len(codes))),
            float(np.mean(np.log1p(cnts))) if cnts.size else 0.0,
        ], dtype=np.float32)

        # ----- cluster -----
        clu_spend: Dict[int, float] = {}
        for c, v in spend.items():
            clu = code_to_cluster.get(str(c), None)
            if clu is None:
                continue
            clu_spend[int(clu)] = clu_spend.get(int(clu), 0.0) + float(v)
        if not clu_spend:
            base = np.array([gmean, gmed, gmean, gmean, gmean, gmed, 0.0, 0.0], dt
            feats[i, 8:16] = base
            continue
        clus = list(clu_spend.keys())
        sp2 = np.asarray([float(clu_spend[k]) for k in clus], dtype=float)
        ssum2 = float(sp2.sum())
        w2 = (np.ones_like(sp2) / max(1, sp2.size)) if ssum2 <= 0 else (sp2 / (ss
        top2 = int(np.argmax(w2)) if w2.size else 0
        means2, meds2, cnts2 = [], [], []
        known2 = 0
        for k in clus:
            st = stats_clu.get(int(k))
            if st is None:
                means2.append(gmean); meds2.append(gmed); cnts2.append(0)
            else:
                means2.append(st.mean_smooth); meds2.append(st.median_adj); cnts2
        means2 = np.asarray(means2, dtype=float)
        meds2 = np.asarray(meds2, dtype=float)
        cnts2 = np.asarray(cnts2, dtype=float)
        feats[i, 8:16] = np.array([
            float(np.sum(w2 * means2)),
            float(np.sum(w2 * meds2)),
            float(np.mean(means2)),
            float(np.max(means2)),
            float(np.min(means2)),
            float(meds2[top2] if meds2.size else gmed),

            float(known2 / max(1, len(clus))),
            float(np.mean(np.log1p(cnts2))) if cnts2.size else 0.0,
        ], dtype=np.float32)
    return feats
def make_strat_labels(train_df: pd.DataFrame) -> np.ndarray:
    chronic = train_df["primary_chronic"].astype(str)
    cost_bin = pd.qcut(train_df["prior_ed_cost_5y_usd"], q=10, duplicates="drop")
    return (chronic + "_" + cost_bin).values
def best_blend_weight(y_true: np.ndarray, p1: np.ndarray, p2: np.ndarray, step: f
    best_w = 0.5
    best_m = 1e18
    for w in np.arange(0.0, 1.0 + 1e-9, step):
        blend = w * p1 + (1.0 - w) * p2
        m = mae(y_true, blend)
        if m < best_m:
            best_m = m
            best_w = float(w)
    return best_w, best_m
# =========================
# MAIN
# =========================
def main() -> None:
    setup_logging()
    logging.info(f"XGBoost version: {getattr(xgb, '__version__', 'unknown')}")
    logging.info(f"FAST_MODE={FAST_MODE} | repeats={len(REPEAT_SEEDS)} | folds={N_
    logging.info(f"GPU config: XGB_DEVICE={XGB_DEVICE}, CAT_TASK_TYPE={CAT_TASK_TY
    # load
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"Missing: {TRAIN_CSV}")
    if not TEST_CSV.exists():
        raise FileNotFoundError(f"Missing: {TEST_CSV}")
    if not PDF_DIR.exists():

        raise FileNotFoundError(f"Missing PDF dir: {PDF_DIR}")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    logging.info(f"Loaded train={train_df.shape}, test={test_df.shape}")
    # optional patients.csv
    if PATIENTS_CSV.exists():
        patients = pd.read_csv(PATIENTS_CSV)
        if "patient_id" in patients.columns:
            train_df = train_df.merge(patients, on="patient_id", how="left")
            test_df = test_df.merge(patients, on="patient_id", how="left")
            logging.info(f"Merged patients.csv: train={train_df.shape}, test={test
        else:
            logging.warning("patients.csv exists but has no patient_id; skipping 
    else:
        logging.info("patients.csv not found; continuing without it.")
    train_df["is_train"] = 1
    test_df["is_train"] = 0
    df_all = pd.concat([train_df, test_df], ignore_index=True)
    # receipts parsing (cached)
    all_pids = df_all["patient_id"].astype(int).unique().tolist()
    all_pids.sort()
    receipt_meta_df, profiles = build_receipt_artifacts(all_pids, PDF_DIR, CACHE_
    df_all = df_all.merge(receipt_meta_df, on="patient_id", how="left")
    logging.info(f"Receipt meta merged: df_all={df_all.shape}")
    # unsupervised features
    logging.info("=== Unsupervised code representation: SVD + clustering ===")
    unsup_df, code_to_idx, code_to_cluster = build_unsupervised_features(
        patient_ids=all_pids,
        profiles=profiles,
        n_svd_tfidf=N_SVD_TFIDF,
        n_svd_spend=N_SVD_SPEND,
        n_clusters=N_CODE_CLUSTERS,
        random_state=2027,
    )
    df_all = df_all.merge(unsup_df, on="patient_id", how="left")

    logging.info(f"Unsupervised features merged: df_all={df_all.shape} | clusters
    # Build XGB base sparse matrix
    logging.info("Building XGB base sparse matrix (OHE cats + numeric)...")
    X_all_base, _ = build_base_sparse_matrix(df_all)
    logging.info(f"X_all_base shape: {X_all_base.shape}")
    is_train_mask = df_all["is_train"].values.astype(int) == 1
    X_train_base = X_all_base[is_train_mask]
    X_test_base = X_all_base[~is_train_mask]
    y = train_df["ed_cost_next3y_usd"].values.astype(np.float32)
    y_log = np.log1p(y)
    train_pids = train_df["patient_id"].astype(int).tolist()
    test_pids = test_df["patient_id"].astype(int).tolist()
    pid_to_row = {pid: i for i, pid in enumerate(train_pids)}
    # CatBoost base frames
    logging.info("Preparing CatBoost base frames...")
    df_cb_all = df_all.drop(columns=["ed_cost_next3y_usd"], errors="ignore").copy
    for c in ["primary_chronic", "sex", "insurance", "zip3", "pdf_insurance", "pd
        if c in df_cb_all.columns:
            df_cb_all[c] = df_cb_all[c].astype(str).fillna("missing")
    drop_cols = [c for c in ["patient_id", "is_train"] if c in df_cb_all.columns]
    df_cb_all = df_cb_all.drop(columns=drop_cols, errors="ignore")
    df_train_cb = df_cb_all[is_train_mask].reset_index(drop=True)
    df_test_cb = df_cb_all[~is_train_mask].reset_index(drop=True)
    cb_cat_cols = [c for c in ["primary_chronic", "sex", "insurance", "zip3", "pd
                   if c in df_train_cb.columns]
    cb_cat_idx = [df_train_cb.columns.get_loc(c) for c in cb_cat_cols]
    logging.info(f"CatBoost cat_features={cb_cat_cols}")
    # CV splits
    strat = make_strat_labels(train_df)
    splits = []
    for rep, seed in enumerate(REPEAT_SEEDS):

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), strat
            splits.append((rep, fold, tr_idx, va_idx))
    logging.info(f"Total CV splits: {len(splits)}")
    # storage
    oof_xgb = np.zeros(len(y), dtype=np.float64)
    oof_cat = np.zeros(len(y), dtype=np.float64)
    cnt_xgb = np.zeros(len(y), dtype=np.float64)
    cnt_cat = np.zeros(len(y), dtype=np.float64)
    test_xgb_sum = np.zeros(len(test_df), dtype=np.float64)
    test_cat_sum = np.zeros(len(test_df), dtype=np.float64)
    n_xgb = 0
    n_cat = 0
    # Model params (GPU + heartbeats)
    xgb_params = dict(
        objective="reg:absoluteerror",
        eval_metric="mae",
        tree_method="hist",
        device=XGB_DEVICE,
        n_estimators=6000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_lambda=1.0,
        min_child_weight=1.0,
        n_jobs=-1,
        verbosity=1,
        early_stopping_rounds=EARLY_STOPPING,
    )
    cat_params = dict(
        loss_function="MAE",
        eval_metric="MAE",
        iterations=8000,
        learning_rate=0.05,
        depth=8,

        l2_leaf_reg=6.0,
        bagging_temperature=0.8,
        random_strength=0.5,
        od_type="Iter",
        od_wait=EARLY_STOPPING,
        task_type=CAT_TASK_TYPE,
        devices="0",
        verbose=CAT_VERBOSE_EVERY,
    )
    te_cols = [f"te_{i:02d}" for i in range(16)]
    fit_sig_xgb = None
    # CV loop
    for rep, fold, tr_idx, va_idx in tqdm(splits, desc="CV folds", ncols=110):
        t0 = time.time()
        fold_tag = f"rep={rep} fold={fold}"
        logging.info(f"{fold_tag} | START | train={len(tr_idx)} val={len(va_idx)}
        tr_pids = [train_pids[i] for i in tr_idx.tolist()]
        va_pids = [train_pids[i] for i in va_idx.tolist()]
        # TE maps from TRAIN fold only (no leakage)
        logging.info(f"{fold_tag} | TE stats build (codes + learned clusters)..."
        stats_code, stats_clu, gmean, gmed = build_te_stats_for_fold(
            train_pids=tr_pids,
            y_log=y_log,
            pid_to_row=pid_to_row,
            profiles=profiles,
            code_to_cluster=code_to_cluster,
        )
        logging.info(f"{fold_tag} | TE feature build (train/val/test)...")
        te_tr = te_features_for_patients(tr_pids, profiles, stats_code, stats_clu
        te_va = te_features_for_patients(va_pids, profiles, stats_code, stats_clu
        te_te = te_features_for_patients(test_pids, profiles, stats_code, stats_c
        # XGB matrices
        X_tr = hstack([X_train_base[tr_idx], csr_matrix(te_tr)], format="csr")
        X_va = hstack([X_train_base[va_idx], csr_matrix(te_va)], format="csr")

        X_te = hstack([X_test_base, csr_matrix(te_te)], format="csr")
        y_tr = y[tr_idx]
        y_va = y[va_idx]
        # ----- XGBoost -----
        logging.info(f"{fold_tag} | XGB fit START | device={xgb_params['device']} 
        try:
            xgb_model = XGBRegressor(**xgb_params, random_state=1000 * rep + fold
            if fit_sig_xgb is None:
                fit_sig_xgb = inspect.signature(xgb_model.fit).parameters
            fit_kwargs = {"eval_set": [(X_va, y_va)]}
            if "verbose_eval" in fit_sig_xgb:
                fit_kwargs["verbose_eval"] = XGB_VERBOSE_EVERY
            else:
                fit_kwargs["verbose"] = XGB_VERBOSE_EVERY
            xgb_model.fit(X_tr, y_tr, **fit_kwargs)
            p_va_x = xgb_model.predict(X_va)
            p_te_x = xgb_model.predict(X_te)
            best_it = getattr(xgb_model, "best_iteration", None)
            logging.info(f"{fold_tag} | XGB fit END | best_iteration={best_it}")
        except Exception as e:
            logging.warning(f"{fold_tag} | XGB GPU failed: {e} -> fallback CPU")
            xgb_cpu = dict(xgb_params)
            xgb_cpu["device"] = "cpu"
            xgb_model = XGBRegressor(**xgb_cpu, random_state=1000 * rep + fold)
            fit_sig_xgb = inspect.signature(xgb_model.fit).parameters
            fit_kwargs = {"eval_set": [(X_va, y_va)]}
            if "verbose_eval" in fit_sig_xgb:
                fit_kwargs["verbose_eval"] = XGB_VERBOSE_EVERY
            else:
                fit_kwargs["verbose"] = XGB_VERBOSE_EVERY
            logging.info(f"{fold_tag} | XGB (CPU fallback) fit START")

            xgb_model.fit(X_tr, y_tr, **fit_kwargs)
            logging.info(f"{fold_tag} | XGB (CPU fallback) fit END")
            p_va_x = xgb_model.predict(X_va)
            p_te_x = xgb_model.predict(X_te)
        if CLIP_AT_ZERO:
            p_va_x = np.maximum(p_va_x, 0.0)
            p_te_x = np.maximum(p_te_x, 0.0)
        oof_xgb[va_idx] += p_va_x
        cnt_xgb[va_idx] += 1.0
        test_xgb_sum += p_te_x
        n_xgb += 1
        # ----- CatBoost -----
        logging.info(f"{fold_tag} | CatBoost pools build...")
        te_train_full = np.zeros((len(train_df), 16), dtype=np.float32)
        te_train_full[tr_idx, :] = te_tr
        te_train_full[va_idx, :] = te_va
        df_tr = df_train_cb.copy()
        df_te = df_test_cb.copy()
        for j, col in enumerate(te_cols):
            df_tr[col] = te_train_full[:, j]
            df_te[col] = te_te[:, j]
        tr_pool = Pool(df_tr.iloc[tr_idx], label=y_tr, cat_features=cb_cat_idx)
        va_pool = Pool(df_tr.iloc[va_idx], label=y_va, cat_features=cb_cat_idx)
        te_pool = Pool(df_te, cat_features=cb_cat_idx)
        logging.info(f"{fold_tag} | CatBoost fit START | task_type={cat_params['t
        try:
            cat_model = CatBoostRegressor(**cat_params, random_seed=2000 * rep + 
            cat_model.fit(tr_pool, eval_set=va_pool, use_best_model=True)
            best_it = None
            try:
                best_it = cat_model.get_best_iteration()
            except Exception:
                best_it = None
            logging.info(f"{fold_tag} | CatBoost fit END | best_iteration={best_it

        except Exception as e:
            logging.warning(f"{fold_tag} | CatBoost GPU failed: {e} -> fallback C
            cat_cpu = dict(cat_params)
            cat_cpu["task_type"] = "CPU"
            cat_cpu.pop("devices", None)
            cat_model = CatBoostRegressor(**cat_cpu, random_seed=2000 * rep + fol
            logging.info(f"{fold_tag} | CatBoost (CPU fallback) fit START | verbo
            cat_model.fit(tr_pool, eval_set=va_pool, use_best_model=True)
            logging.info(f"{fold_tag} | CatBoost (CPU fallback) fit END")
        p_va_c = cat_model.predict(va_pool)
        p_te_c = cat_model.predict(te_pool)
        if CLIP_AT_ZERO:
            p_va_c = np.maximum(p_va_c, 0.0)
            p_te_c = np.maximum(p_te_c, 0.0)
        oof_cat[va_idx] += p_va_c
        cnt_cat[va_idx] += 1.0
        test_cat_sum += p_te_c
        n_cat += 1
        dt = time.time() - t0
        logging.info(f"{fold_tag} | END | fold_time={dt:.1f}s")
    # finalize
    oof_x = (oof_xgb / np.maximum(cnt_xgb, 1.0)).astype(np.float32)
    oof_c = (oof_cat / np.maximum(cnt_cat, 1.0)).astype(np.float32)
    te_x = (test_xgb_sum / max(1, n_xgb)).astype(np.float32)
    te_c = (test_cat_sum / max(1, n_cat)).astype(np.float32)
    logging.info(f"OOF MAE | XGB: {mae(y, oof_x):.4f}")
    logging.info(f"OOF MAE | CAT: {mae(y, oof_c):.4f}")
    # blend + median bias (MAE-optimal shift)
    w, blend_mae = best_blend_weight(y, oof_x, oof_c, step=BLEND_STEP)
    oof_blend = w * oof_x + (1.0 - w) * oof_c
    bias = float(np.median(y - oof_blend))

What to watch in your logs
If this “third path” is working, you should see:
The SVD + clustering stage completes quickly (tiny data).
During CV, XGBoost prints MAE every 100 rounds; CatBoost prints every 100
iterations.
OOF MAE should ideally beat 478.4; if it doesn’t, the next pivot is to make the
unsupervised representation supervised (e.g., supervised SVD via target-weighted
matrix factorization / stacking), but this script is the cleanest “no manual grouping”
leap that still stays stable and Kaggle‑practical on ~2k rows.
If you paste your resulting:
OOF MAE | XGB: ...
OOF MAE | CAT: ...
OOF MAE final= ...
    oof_final = oof_blend + bias
    test_final = w * te_x + (1.0 - w) * te_c + bias
    if CLIP_AT_ZERO:
        oof_final = np.maximum(oof_final, 0.0)
        test_final = np.maximum(test_final, 0.0)
    logging.info(f"Blend w(XGB)={w:.2f} | OOF MAE pre-bias={blend_mae:.4f} | bias
    # Save submission
    sub = pd.DataFrame({
        "patient_id": test_df["patient_id"].astype(int).values,
        "ed_cost_next3y_usd": test_final.astype(np.float32),
    })
    sub.to_csv(SUBMISSION_PATH, index=False)
    logging.info(f"Saved submission: {SUBMISSION_PATH}")
if __name__ == "__main__":
    main()

…I’ll tell you whether the bottleneck is now (a) embedding dimensions/clusters, (b) TE
strength vs leakage/noise, or (c) CV mismatch to hidden test.
```

### C08 — ITERATION 5 — "Scale the Simple" (Sparse TF-IDF + Sparse Spend-Share, No SVD, No (pages 226-247)
```python
# -*- coding: utf-8 -*-
r"""
ITERATION 5 — "Scale the Simple" (Sparse TF-IDF + Sparse Spend-Share, No SVD, No 
=================================================================================
Why this should beat Iteration 2 (478.4) without falling into the complexity trap
- Keeps the proven "blind" sparse token signal (TF-IDF on raw billing codes).

- Adds ONE extra sparse view that TF-IDF ignores: per-code spend composition (spe
  still high-dimensional + sparse (no embeddings, no dimensionality reduction).
- Adds a tiny MAE-aligned calibration: optional per-primary_chronic median bias c
  (constant shift per group), which often improves MAE for biased residuals.
Hard constraints respected:
- NO SVD/PCA/NMF/embeddings.
- NO manual CPT range buckets or clinical rules.
- GPU enabled for XGBoost; CatBoost GPU attempted but safely falls back if needed
- Rich, timestamped logs + training heartbeats.
===========================================================
INSTALL (run once)
===========================================================
pip install -U pandas numpy scipy scikit-learn xgboost catboost pymupdf joblib tq
"""
from __future__ import annotations
import re
import math
import time
import logging
import warnings
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import fitz  # PyMuPDF
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfTransformer
import xgboost as xgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor, Pool
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# =========================
# PATHS
# =========================
BASE_DIR = Path(r"D:\AgentDs\agent_ds_healthcare")
TRAIN_CSV = BASE_DIR / "ed_cost_train.csv"
TEST_CSV = BASE_DIR / "ed_cost_test.csv"
PATIENTS_CSV = BASE_DIR / "patients.csv"          # optional
PDF_DIR = BASE_DIR / "receipts_pdf"
SUBMISSION_PATH = BASE_DIR / "submission_ICHI_V1.csv"
CACHE_PATH = BASE_DIR / "cache_receipts_iter5_sparse.joblib"
CACHE_VERSION = 51
# =========================
# TRAINING CONFIG
# =========================
FAST_MODE = True
N_FOLDS = 5
REPEAT_SEEDS = [2025, 2026] if FAST_MODE else [2024, 2025, 2026]
EARLY_STOPPING = 200
# Text / sparse code settings
CODE_MIN_DF = 1         # keep as 1 to match "don't throw info away"; raise to 2 
# GPU

XGB_DEVICE = "cuda"
CAT_TASK_TYPE = "GPU"
# Heartbeats
XGB_VERBOSE_EVERY = 100
CAT_VERBOSE_EVERY = 100
# Post-process
CLIP_AT_ZERO = True
# Blend
BLEND_STEP = 0.02
# Calibration
USE_GROUP_BIAS_BY_CHRONIC = True
# =========================
# LOGGING
# =========================
def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | 
# =========================
# UTIL
# =========================
_money_re = re.compile(r"^\d{1,3}(?:,\d{3})*(?:\.\d{2})$|^\d+(?:\.\d{2})$")
_code_re = re.compile(r"^(?:[A-Z]\d{4}|\d{4,5})$")
LINE_ITEM_RE = re.compile(r"^([A-Z]\d{4}|\d{4,5})\s+(.+?)\s+(\d+)\s+([\d,]+\.\d{2
TOTAL_RE = re.compile(r"^TOTAL\s+([\d,]+\.\d{2})$")
def is_money(tok: str) -> bool:
    return bool(_money_re.match(tok))
def money_to_float(tok: str) -> float:
    return float(tok.replace(",", ""))
def is_code(tok: str) -> bool:

    return bool(_code_re.match(tok))
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))
def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_out
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)
def ensure_csr_float32_int32(X: csr_matrix) -> csr_matrix:
    X = X.tocsr()
    if X.data.dtype != np.float32:
        X.data = X.data.astype(np.float32, copy=False)
    # CatBoost sometimes dislikes int64 indices; keep int32 to be safe
    if X.indices.dtype != np.int32:
        X.indices = X.indices.astype(np.int32, copy=False)
    if X.indptr.dtype != np.int32:
        X.indptr = X.indptr.astype(np.int32, copy=False)
    return X
# =========================
# PDF PARSING (simple + robust)
# =========================
def parse_header_fields(text: str) -> Tuple[Optional[int], Optional[str], Optiona
    pid = None
    zip3 = None
    insurance = None
    m = re.search(r"Patient ID:\s*(\d+)", text)
    if m:
        pid = int(m.group(1))
    m = re.search(r"ZIP3:\s*([0-9]{3})\s+Insurance:\s*([A-Za-z_]+)", text)
    if m:
        zip3 = str(m.group(1))
        insurance = str(m.group(2)).lower()

    return pid, zip3, insurance
@dataclass
class ReceiptProfile:
    meta: Dict[str, Any]
    code_cnt: Dict[str, float]
    code_spend: Dict[str, float]
    code_qty: Dict[str, float]
def featurize_receipt(patient_id: int, pdf_dir: Path) -> ReceiptProfile:
    pdf_path = pdf_dir / f"receipt_{patient_id}.pdf"
    empty_meta = dict(
        pdf_ok=0,
        pdf_n_lines=0,
        pdf_n_unique_codes=0,
        pdf_sum_qty=0.0,
        pdf_sum_line_total=0.0,
        pdf_total=0.0,
        pdf_total_diff=0.0,
        pdf_insurance=None,
        pdf_zip3=None,
        pdf_parse_notes="missing_pdf",
    )
    if not pdf_path.exists():
        return ReceiptProfile(meta=empty_meta, code_cnt={}, code_spend={}, code_q
    try:
        with fitz.open(pdf_path) as doc:
            full_text = ""
            for page in doc:
                full_text += (page.get_text("text") or "") + "\n"
    except Exception as e:
        empty_meta["pdf_parse_notes"] = f"open_error:{type(e).__name__}"
        return ReceiptProfile(meta=empty_meta, code_cnt={}, code_spend={}, code_q
    # header fields

    _, zip3, insurance = parse_header_fields(full_text)
    # parse line items from text lines
    lines = [ln.strip() for ln in full_text.splitlines() if ln.strip()]
    if not lines:
        empty_meta["pdf_parse_notes"] = "empty_text"
        empty_meta["pdf_zip3"] = zip3
        empty_meta["pdf_insurance"] = insurance
        return ReceiptProfile(meta=empty_meta, code_cnt={}, code_spend={}, code_q
    # find table header
    start = None
    for i, ln in enumerate(lines):
        if ln.lower().startswith("code description qty"):
            start = i + 1
            break
    if start is None:
        # fallback: try the exact header fragment
        for i, ln in enumerate(lines):
            if "code description" in ln.lower() and "line total" in ln.lower():
                start = i + 1
                break
    if start is None:
        empty_meta["pdf_parse_notes"] = "no_table_header"
        empty_meta["pdf_zip3"] = zip3
        empty_meta["pdf_insurance"] = insurance
        return ReceiptProfile(meta=empty_meta, code_cnt={}, code_spend={}, code_q
    code_cnt: Dict[str, float] = {}
    code_spend: Dict[str, float] = {}
    code_qty: Dict[str, float] = {}
    total_val: Optional[float] = None
    sum_line_total = 0.0
    sum_qty = 0.0
    for ln in lines[start:]:
        m_tot = TOTAL_RE.match(ln)
        if m_tot:
            total_val = money_to_float(m_tot.group(1))

            break
        m = LINE_ITEM_RE.match(ln)
        if m:
            code = m.group(1)
            qty = float(m.group(3))
            line_total = money_to_float(m.group(5))
            code_cnt[code] = code_cnt.get(code, 0.0) + 1.0
            code_qty[code] = code_qty.get(code, 0.0) + qty
            code_spend[code] = code_spend.get(code, 0.0) + line_total
            sum_line_total += line_total
            sum_qty += qty
        else:
            # very light fallback: split-based heuristic
            parts = ln.split()
            if len(parts) >= 5 and is_code(parts[0]) and parts[-3].isdigit() and 
                code = parts[0]
                qty = float(parts[-3])
                line_total = money_to_float(parts[-1])
                code_cnt[code] = code_cnt.get(code, 0.0) + 1.0
                code_qty[code] = code_qty.get(code, 0.0) + qty
                code_spend[code] = code_spend.get(code, 0.0) + line_total
                sum_line_total += line_total
                sum_qty += qty
    if not code_cnt:
        empty_meta["pdf_parse_notes"] = "no_items_parsed"
        empty_meta["pdf_zip3"] = zip3
        empty_meta["pdf_insurance"] = insurance
        return ReceiptProfile(meta=empty_meta, code_cnt={}, code_spend={}, code_q
    pdf_total = float(total_val) if total_val is not None else float(sum_line_tot
    meta = dict(
        pdf_ok=1,
        pdf_n_lines=int(sum(code_cnt.values())),
        pdf_n_unique_codes=int(len(code_cnt)),

        pdf_sum_qty=float(sum_qty),
        pdf_sum_line_total=float(sum_line_total),
        pdf_total=float(pdf_total),
        pdf_total_diff=float(abs(pdf_total - sum_line_total)),
        pdf_insurance=(insurance.lower() if isinstance(insurance, str) else None)
        pdf_zip3=(str(zip3) if zip3 is not None else None),
        pdf_parse_notes="ok" if total_val is not None else "ok_no_total",
    )
    return ReceiptProfile(meta=meta, code_cnt=code_cnt, code_spend=code_spend, co
def load_cache(cache_path: Path) -> Dict[str, Any]:
    if cache_path.exists():
        try:
            obj = joblib.load(cache_path)
            if isinstance(obj, dict) and obj.get("version") == CACHE_VERSION and 
                return obj
        except Exception:
            pass
    return {"version": CACHE_VERSION, "data": {}}
def save_cache(cache_path: Path, cache_obj: Dict[str, Any]) -> None:
    try:
        joblib.dump(cache_obj, cache_path)
    except Exception as e:
        logging.warning(f"Cache save failed: {e}")
def build_receipt_artifacts(patient_ids: List[int], pdf_dir: Path, cache_path: Pa
    cache_obj = load_cache(cache_path)
    data: Dict[str, Any] = cache_obj.get("data", {})
    to_parse = [pid for pid in patient_ids if str(pid) not in data]
    logging.info(f"Receipt cache: {len(patient_ids)-len(to_parse):,} hit | {len(t
    for pid in tqdm(to_parse, desc="Parsing receipts", ncols=100):
        prof = featurize_receipt(pid, pdf_dir)
        data[str(pid)] = {

            "meta": prof.meta,
            "code_cnt": prof.code_cnt,
            "code_spend": prof.code_spend,
            "code_qty": prof.code_qty,
        }
    cache_obj["data"] = data
    if to_parse:
        save_cache(cache_path, cache_obj)
    meta_rows = []
    profiles: Dict[int, Dict[str, Any]] = {}
    for pid in patient_ids:
        obj = data.get(str(pid), None)
        if obj is None:
            prof = featurize_receipt(pid, pdf_dir)
            obj = {"meta": prof.meta, "code_cnt": prof.code_cnt, "code_spend": pr
        meta_rows.append({"patient_id": pid, **(obj.get("meta", {}) or {})})
        profiles[pid] = obj
    receipt_meta_df = pd.DataFrame(meta_rows)
    return receipt_meta_df, profiles
# =========================
# SPARSE CODE MATRICES (NO SVD, NO GROUPING)
# =========================
def build_code_vocab_and_df(profiles: Dict[int, Dict[str, Any]], patient_ids: Lis
    df = {}
    for pid in patient_ids:
        codes = (profiles.get(pid, {}) or {}).get("code_cnt", {}) or {}
        for c in codes.keys():
            df[c] = df.get(c, 0) + 1
    # filter by min df
    vocab = [c for c, cdf in df.items() if cdf >= CODE_MIN_DF]
    vocab.sort()
    code_to_idx = {c: i for i, c in enumerate(vocab)}
    return vocab, code_to_idx

def build_sparse_code_views(
    patient_ids: List[int],
    profiles: Dict[int, Dict[str, Any]],
    code_to_idx: Dict[str, int],
) -> Tuple[csr_matrix, csr_matrix]:
    """
    Returns:
      X_bin   : 1 if code appears for patient
      X_share : spend share for code (sum(code_spend)/total_spend_patient)
    """
    n_pat = len(patient_ids)
    n_code = len(code_to_idx)
    rb, cb, db = [], [], []
    rs, cs, ds = [], [], []
    for i, pid in enumerate(patient_ids):
        obj = profiles.get(pid, {}) or {}
        cnt = obj.get("code_cnt", {}) or {}
        spend = obj.get("code_spend", {}) or {}
        total_spend = float(sum(float(v) for v in spend.values()))
        total_spend = max(total_spend, 1e-9)
        for c in cnt.keys():
            j = code_to_idx.get(str(c), None)
            if j is None:
                continue
            rb.append(i); cb.append(j); db.append(1.0)
        for c, sp in spend.items():
            j = code_to_idx.get(str(c), None)
            if j is None:
                continue
            rs.append(i); cs.append(j); ds.append(float(sp) / total_spend)
    X_bin = csr_matrix((np.asarray(db, dtype=np.float32), (np.asarray(rb), np.asa
    X_share = csr_matrix((np.asarray(ds, dtype=np.float32), (np.asarray(rs), np.a
    return X_bin, X_share

# =========================
# TABULAR FEATURES
# =========================
def build_tabular_sparse_matrix(df_all: pd.DataFrame) -> Tuple[csr_matrix, Dict[st
    df = df_all.copy()
    # basic derived features (simple + safe)
    df["prior_cost_per_visit"] = df["prior_ed_cost_5y_usd"] / (df["prior_ed_visit
    df["log_prior_cost"] = np.log1p(df["prior_ed_cost_5y_usd"].astype(float))
    df["log_prior_visits"] = np.log1p(df["prior_ed_visits_5y"].astype(float))
    # normalize likely categorical columns if present
    for c in ["primary_chronic", "sex", "insurance", "zip3", "pdf_insurance", "pd
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("missing")
    exclude = {"patient_id", "ed_cost_next3y_usd", "is_train"}
    cat_cols = [c for c in ["primary_chronic", "sex", "insurance", "zip3", "pdf_i
                if c in df.columns]
    num_cols = [c for c in df.columns
                if c not in exclude and c not in cat_cols and pd.api.types.is_num
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), nu
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    X = preprocessor.fit_transform(df)
    X = X.tocsr() if hasattr(X, "tocsr") else csr_matrix(X)
    return X, {"preprocessor": preprocessor, "num_cols": num_cols, "cat_cols": cat
def make_strat_labels(train_df: pd.DataFrame) -> np.ndarray:
    chronic = train_df["primary_chronic"].astype(str)
    cost_bin = pd.qcut(train_df["prior_ed_cost_5y_usd"], q=10, duplicates="drop")

    return (chronic + "_" + cost_bin).values
def best_blend_weight(y_true: np.ndarray, p1: np.ndarray, p2: np.ndarray, step: f
    best_w = 0.5
    best_m = 1e18
    for w in np.arange(0.0, 1.0 + 1e-9, step):
        blend = w * p1 + (1.0 - w) * p2
        m = mae(y_true, blend)
        if m < best_m:
            best_m = m
            best_w = float(w)
    return best_w, best_m
# =========================
# MAIN
# =========================
def main() -> None:
    setup_logging()
    logging.info(f"XGBoost version: {getattr(xgb, '__version__', 'unknown')}")
    logging.info(f"FAST_MODE={FAST_MODE} | repeats={len(REPEAT_SEEDS)} | folds={N_
    logging.info(f"GPU config: XGB_DEVICE={XGB_DEVICE}, CAT_TASK_TYPE={CAT_TASK_TY
    # load
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    logging.info(f"Loaded train={train_df.shape}, test={test_df.shape}")
    # optional patients
    if PATIENTS_CSV.exists():
        patients = pd.read_csv(PATIENTS_CSV)
        if "patient_id" in patients.columns:
            train_df = train_df.merge(patients, on="patient_id", how="left")
            test_df = test_df.merge(patients, on="patient_id", how="left")
            logging.info(f"Merged patients.csv: train={train_df.shape}, test={test
        else:
            logging.warning("patients.csv exists but no patient_id; skipping.")
    train_df["is_train"] = 1

    test_df["is_train"] = 0
    df_all = pd.concat([train_df, test_df], ignore_index=True)
    # parse receipts
    all_pids = df_all["patient_id"].astype(int).unique().tolist()
    all_pids.sort()
    receipt_meta_df, profiles = build_receipt_artifacts(all_pids, PDF_DIR, CACHE_
    df_all = df_all.merge(receipt_meta_df, on="patient_id", how="left")
    logging.info(f"Merged receipt meta: df_all={df_all.shape}")
    # build code vocab + sparse code matrices
    logging.info("Building sparse code views (TF-IDF codes + spend-share codes)..
    vocab, code_to_idx = build_code_vocab_and_df(profiles, all_pids)
    logging.info(f"Code vocab size={len(vocab):,} (CODE_MIN_DF={CODE_MIN_DF})")
    if len(vocab) == 0:
        logging.warning("No codes parsed at all. Falling back to tabular-only.")
        X_code_all = None
    else:
        X_bin, X_share = build_sparse_code_views(all_pids, profiles, code_to_idx)
        # TF-IDF on binary presence
        tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True, sublin
        X_tfidf = tfidf.fit_transform(X_bin)
        # Keep spend-share raw (already normalized per patient)
        X_code_all = hstack([X_tfidf, X_share], format="csr")
        X_code_all = ensure_csr_float32_int32(X_code_all)
        logging.info(f"X_code_all shape={X_code_all.shape} | nnz={X_code_all.nnz:
    # tabular sparse
    logging.info("Building tabular sparse matrix (OHE + numeric)...")
    X_tab_all, _ = build_tabular_sparse_matrix(df_all)
    X_tab_all = ensure_csr_float32_int32(X_tab_all)
    logging.info(f"X_tab_all shape={X_tab_all.shape} | nnz={X_tab_all.nnz:,}")
    # combine
    if X_code_all is None:
        X_all = X_tab_all

    else:
        X_all = hstack([X_tab_all, X_code_all], format="csr")
        X_all = ensure_csr_float32_int32(X_all)
    logging.info(f"FINAL X_all shape={X_all.shape} | nnz={X_all.nnz:,}")
    is_train_mask = df_all["is_train"].values.astype(int) == 1
    X_train = X_all[is_train_mask]
    X_test = X_all[~is_train_mask]
    y = train_df["ed_cost_next3y_usd"].values.astype(np.float32)
    # CV splits
    strat = make_strat_labels(train_df)
    splits = []
    for rep, seed in enumerate(REPEAT_SEEDS):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), strat
            splits.append((rep, fold, tr_idx, va_idx))
    logging.info(f"Total CV splits: {len(splits)}")
    # storage
    oof_xgb = np.zeros(len(y), dtype=np.float64)
    oof_cat = np.zeros(len(y), dtype=np.float64)
    cnt_xgb = np.zeros(len(y), dtype=np.float64)
    cnt_cat = np.zeros(len(y), dtype=np.float64)
    test_xgb_sum = np.zeros(X_test.shape[0], dtype=np.float64)
    test_cat_sum = np.zeros(X_test.shape[0], dtype=np.float64)
    n_xgb = 0
    n_cat = 0
    # model params (kept conservative; aligned to MAE)
    xgb_params = dict(
        objective="reg:absoluteerror",
        eval_metric="mae",
        tree_method="hist",
        device=XGB_DEVICE,
        n_estimators=6000,
        learning_rate=0.05,

        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_lambda=1.0,
        min_child_weight=1.0,
        n_jobs=-1,
        verbosity=1,
        early_stopping_rounds=EARLY_STOPPING,
    )
    cat_params = dict(
        loss_function="MAE",
        eval_metric="MAE",
        iterations=8000,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=6.0,
        bagging_temperature=0.8,
        random_strength=0.5,
        od_type="Iter",
        od_wait=EARLY_STOPPING,
        task_type=CAT_TASK_TYPE,
        devices="0",
        verbose=CAT_VERBOSE_EVERY,
    )
    fit_sig_xgb = None
    # CV loop
    for rep, fold, tr_idx, va_idx in tqdm(splits, desc="CV folds", ncols=110):
        t0 = time.time()
        fold_tag = f"rep={rep} fold={fold}"
        logging.info(f"{fold_tag} | START | train={len(tr_idx)} val={len(va_idx)}
        X_tr = X_train[tr_idx]
        X_va = X_train[va_idx]
        y_tr = y[tr_idx]
        y_va = y[va_idx]
        # ----- XGBoost -----

        logging.info(f"{fold_tag} | XGB fit START | device={xgb_params['device']} 
        try:
            xgb_model = XGBRegressor(**xgb_params, random_state=1000 * rep + fold
            if fit_sig_xgb is None:
                fit_sig_xgb = inspect.signature(xgb_model.fit).parameters
            fit_kwargs = {"eval_set": [(X_va, y_va)]}
            # For sklearn API, verbose=int prints metrics every N boosting stages 
            if "verbose_eval" in fit_sig_xgb:
                fit_kwargs["verbose_eval"] = XGB_VERBOSE_EVERY
            else:
                fit_kwargs["verbose"] = XGB_VERBOSE_EVERY
            xgb_model.fit(X_tr, y_tr, **fit_kwargs)
            p_va_x = xgb_model.predict(X_va)
            p_te_x = xgb_model.predict(X_test)
            best_it = getattr(xgb_model, "best_iteration", None)
            logging.info(f"{fold_tag} | XGB fit END | best_iteration={best_it}")
        except Exception as e:
            logging.warning(f"{fold_tag} | XGB GPU failed: {e} -> fallback CPU")
            xgb_cpu = dict(xgb_params)
            xgb_cpu["device"] = "cpu"
            xgb_model = XGBRegressor(**xgb_cpu, random_state=1000 * rep + fold)
            fit_sig_xgb = inspect.signature(xgb_model.fit).parameters
            fit_kwargs = {"eval_set": [(X_va, y_va)]}
            if "verbose_eval" in fit_sig_xgb:
                fit_kwargs["verbose_eval"] = XGB_VERBOSE_EVERY
            else:
                fit_kwargs["verbose"] = XGB_VERBOSE_EVERY
            logging.info(f"{fold_tag} | XGB (CPU fallback) fit START")
            xgb_model.fit(X_tr, y_tr, **fit_kwargs)
            logging.info(f"{fold_tag} | XGB (CPU fallback) fit END")
            p_va_x = xgb_model.predict(X_va)
            p_te_x = xgb_model.predict(X_test)

        if CLIP_AT_ZERO:
            p_va_x = np.maximum(p_va_x, 0.0)
            p_te_x = np.maximum(p_te_x, 0.0)
        oof_xgb[va_idx] += p_va_x
        cnt_xgb[va_idx] += 1.0
        test_xgb_sum += p_te_x
        n_xgb += 1
        # ----- CatBoost (sparse numeric) -----
        logging.info(f"{fold_tag} | CatBoost fit START | task_type={cat_params['t
        try:
            tr_pool = Pool(X_tr, label=y_tr)
            va_pool = Pool(X_va, label=y_va)
            te_pool = Pool(X_test)
            cat_model = CatBoostRegressor(**cat_params, random_seed=2000 * rep + 
            cat_model.fit(tr_pool, eval_set=va_pool, use_best_model=True)
            p_va_c = cat_model.predict(va_pool)
            p_te_c = cat_model.predict(te_pool)
            best_it = None
            try:
                best_it = cat_model.get_best_iteration()
            except Exception:
                best_it = None
            logging.info(f"{fold_tag} | CatBoost fit END | best_iteration={best_it
        except Exception as e:
            logging.warning(f"{fold_tag} | CatBoost GPU failed: {e} -> fallback C
            cat_cpu = dict(cat_params)
            cat_cpu["task_type"] = "CPU"
            cat_cpu.pop("devices", None)
            cat_model = CatBoostRegressor(**cat_cpu, random_seed=2000 * rep + fol
            tr_pool = Pool(X_tr, label=y_tr)
            va_pool = Pool(X_va, label=y_va)
            te_pool = Pool(X_test)

            logging.info(f"{fold_tag} | CatBoost (CPU fallback) fit START | verbo
            cat_model.fit(tr_pool, eval_set=va_pool, use_best_model=True)
            logging.info(f"{fold_tag} | CatBoost (CPU fallback) fit END")
            p_va_c = cat_model.predict(va_pool)
            p_te_c = cat_model.predict(te_pool)
        if CLIP_AT_ZERO:
            p_va_c = np.maximum(p_va_c, 0.0)
            p_te_c = np.maximum(p_te_c, 0.0)
        oof_cat[va_idx] += p_va_c
        cnt_cat[va_idx] += 1.0
        test_cat_sum += p_te_c
        n_cat += 1
        dt = time.time() - t0
        # quick fold-level MAE snapshots
        fold_mae_x = mae(y_va, p_va_x)
        fold_mae_c = mae(y_va, p_va_c)
        logging.info(f"{fold_tag} | END | fold_time={dt:.1f}s | fold_MAE(XGB)={fo
    # finalize OOF/test
    oof_x = (oof_xgb / np.maximum(cnt_xgb, 1.0)).astype(np.float32)
    oof_c = (oof_cat / np.maximum(cnt_cat, 1.0)).astype(np.float32)
    te_x = (test_xgb_sum / max(1, n_xgb)).astype(np.float32)
    te_c = (test_cat_sum / max(1, n_cat)).astype(np.float32)
    mae_x = mae(y, oof_x)
    mae_c = mae(y, oof_c)
    logging.info(f"OOF MAE | XGB: {mae_x:.4f}")
    logging.info(f"OOF MAE | CAT: {mae_c:.4f}")
    # blend weight on OOF
    w, blend_mae = best_blend_weight(y, oof_x, oof_c, step=BLEND_STEP)
    oof_blend = w * oof_x + (1.0 - w) * oof_c
    test_blend = w * te_x + (1.0 - w) * te_c
    logging.info(f"Blend search | best w(XGB)={w:.2f} | OOF MAE={blend_mae:.4f}")
    # MAE-aligned bias correction (global and optional per-chronic)

    train_chronic = train_df["primary_chronic"].astype(str).values
    test_chronic = test_df["primary_chronic"].astype(str).values
    # global bias
    bias_global = float(np.median(y - oof_blend))
    oof_global = oof_blend + bias_global
    test_global = test_blend + bias_global
    mae_global = mae(y, oof_global)
    logging.info(f"Calibration(global) | bias={bias_global:.4f} | OOF MAE={mae_gl
    # per-chronic bias (very low complexity; 3 groups)
    if USE_GROUP_BIAS_BY_CHRONIC:
        biases = {}
        for g in np.unique(train_chronic):
            mask = (train_chronic == g)
            if mask.sum() == 0:
                continue
            biases[g] = float(np.median(y[mask] - oof_blend[mask]))
        oof_group = oof_blend.copy()
        for g, b in biases.items():
            oof_group[train_chronic == g] += b
        test_group = test_blend.copy()
        for i in range(len(test_group)):
            b = biases.get(test_chronic[i], 0.0)
            test_group[i] += b
        mae_group = mae(y, oof_group)
        logging.info(f"Calibration(by_chronic) | biases={biases} | OOF MAE={mae_g
        if mae_group <= mae_global:
            oof_final = oof_group
            test_final = test_group
            logging.info("Using per-chronic bias correction (better or equal OOF M
        else:
            oof_final = oof_global
            test_final = test_global
            logging.info("Using global bias correction (better OOF MAE).")
    else:

If this still doesn’t beat 478.4, the next minimal, “non-complexity” lever I’d pull (still within
your constraints) is very small vocabulary pruning ( CODE_MIN_DF=2 ) + slightly higher
regularization ( min_child_weight=2–5 ) to curb rare-code overfit—without changing the
representation paradigm.
        oof_final = oof_global
        test_final = test_global
    if CLIP_AT_ZERO:
        oof_final = np.maximum(oof_final, 0.0)
        test_final = np.maximum(test_final, 0.0)
    logging.info(f"FINAL OOF MAE (post-calibration): {mae(y, oof_final):.4f}")
    # save submission
    sub = pd.DataFrame({
        "patient_id": test_df["patient_id"].astype(int).values,
        "ed_cost_next3y_usd": test_final.astype(np.float32),
    })
    sub.to_csv(SUBMISSION_PATH, index=False)
    logging.info(f"Saved submission: {SUBMISSION_PATH}")
if __name__ == "__main__":
    main()
```

### C09 — ITERATION 6 — Monolithic XGBoost (GPU) + TF-IDF (1–3 grams, 1000 features) + Chro (pages 248-265)
```python
# -*- coding: utf-8 -*-
r"""
ITERATION 6 — Monolithic XGBoost (GPU) + TF-IDF (1–3 grams, 1000 features) + Chro
=================================================================================
What this does (strictly aligned to your directives):
- Deletes CatBoost entirely.
- Uses ONLY: Tabular features + TF-IDF over raw CPT/HCPCS code tokens extracted f
- TF-IDF uses ngram_range=(1,3) and max_features=1000.
- Single XGBoost model (GPU enabled) with n_estimators=8000 and learning_rate=0.0

- Post-processing: Chronic condition bias correction using MEAN residual per chro
===========================================================
INSTALL (run once)
===========================================================
pip install -U pandas numpy scipy scikit-learn xgboost pymupdf joblib tqdm
===========================================================
PATHS (Windows)
===========================================================
Train: D:\AgentDs\agent_ds_healthcare\ed_cost_train.csv
Test : D:\AgentDs\agent_ds_healthcare\ed_cost_test.csv
PDFs : D:\AgentDs\agent_ds_healthcare\receipts_pdf\receipt_{patient_id}.pdf
Optional patients.csv: D:\AgentDs\agent_ds_healthcare\patients.csv
Output:
- D:\AgentDs\agent_ds_healthcare\submission.csv
===========================================================
"""
from __future__ import annotations
import re
import time
import json
import logging
import warnings
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import fitz  # PyMuPDF
from scipy.sparse import csr_matrix, hstack

from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
from xgboost import XGBRegressor
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# =========================
# CONFIG
# =========================
BASE_DIR = Path(r"D:\AgentDs\agent_ds_healthcare")
TRAIN_CSV = BASE_DIR / "ed_cost_train.csv"
TEST_CSV = BASE_DIR / "ed_cost_test.csv"
PATIENTS_CSV = BASE_DIR / "patients.csv"          # optional
PDF_DIR = BASE_DIR / "receipts_pdf"
SUBMISSION_PATH = BASE_DIR / "submission.csv"
CACHE_PATH = BASE_DIR / "cache_receipts_iter6_codes.joblib"
CACHE_VERSION = 61
# CV
FAST_MODE = True
N_FOLDS = 5
REPEAT_SEEDS = [2025, 2026] if FAST_MODE else [2024, 2025, 2026]
EARLY_STOPPING = 200
# TF-IDF
TFIDF_MAX_FEATURES = 1000
TFIDF_NGRAM_RANGE = (1, 3)
# XGB (monolithic)
XGB_DEVICE = "cuda"  # "cuda" or "cuda:0"

N_ESTIMATORS = 8000
LEARNING_RATE = 0.02
# Logging heartbeats
XGB_VERBOSE_EVERY = 100
# Post-process
CLIP_AT_ZERO = True
# =========================
# LOGGING
# =========================
def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | 
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))
def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_out
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)
def ensure_csr_float32_int32(X: csr_matrix) -> csr_matrix:
    X = X.tocsr()
    if X.data.dtype != np.float32:
        X.data = X.data.astype(np.float32, copy=False)
    if X.indices.dtype != np.int32:
        X.indices = X.indices.astype(np.int32, copy=False)
    if X.indptr.dtype != np.int32:
        X.indptr = X.indptr.astype(np.int32, copy=False)
    return X
# =========================

# REGEX (RAW STRINGS — no SyntaxWarnings)
# =========================
LINE_ITEM_RE = re.compile(r"^([A-Z]\d{4}|\d{4,5})\s+(.+?)\s+(\d+)\s+([\d,]+\.\d{2
TOTAL_RE = re.compile(r"^TOTAL\s+([\d,]+\.\d{2})$")
PID_RE = re.compile(r"Patient ID:\s*(\d+)")
ZIP_INS_RE = re.compile(r"ZIP3:\s*([0-9]{3})\s+Insurance:\s*([A-Za-z_]+)")
def parse_header_fields(text: str) -> Tuple[Optional[int], Optional[str], Optiona
    pid = None
    zip3 = None
    insurance = None
    m = PID_RE.search(text)
    if m:
        pid = int(m.group(1))
    m = ZIP_INS_RE.search(text)
    if m:
        zip3 = str(m.group(1))
        insurance = str(m.group(2)).lower()
    return pid, zip3, insurance
def money_to_float(tok: str) -> float:
    return float(tok.replace(",", ""))
@dataclass
class ReceiptProfile:
    meta: Dict[str, Any]
    codes_doc: str
def extract_codes_doc_from_pdf_text(text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Extract a whitespace-separated code sequence "doc" for TF-IDF.
    We intentionally keep it simple: each parsed line item contributes one code t
    """

    _, zip3, insurance = parse_header_fields(text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "", dict(pdf_ok=0, pdf_parse_notes="empty_text", pdf_zip3=zip3, pd
                        pdf_n_lines=0, pdf_n_unique_codes=0, pdf_total=0.0, pdf_s
    # find header line
    start = None
    for i, ln in enumerate(lines):
        if ln.lower().startswith("code description qty"):
            start = i + 1
            break
    if start is None:
        for i, ln in enumerate(lines):
            low = ln.lower()
            if ("code" in low) and ("description" in low) and ("qty" in low) and 
                start = i + 1
                break
    if start is None:
        return "", dict(pdf_ok=0, pdf_parse_notes="no_table_header", pdf_zip3=zip
                        pdf_n_lines=0, pdf_n_unique_codes=0, pdf_total=0.0, pdf_s
    codes: List[str] = []
    sum_line_total = 0.0
    total_val = None
    for ln in lines[start:]:
        m_tot = TOTAL_RE.match(ln)
        if m_tot:
            total_val = money_to_float(m_tot.group(1))
            break
        m = LINE_ITEM_RE.match(ln)
        if m:
            code = m.group(1)
            line_total = money_to_float(m.group(5))
            codes.append(code)
            sum_line_total += float(line_total)
            continue

        # fallback heuristic (still simple)
        parts = ln.split()
        if len(parts) >= 5:
            code = parts[0]
            qty = parts[-3]
            unit = parts[-2]
            lt = parts[-1]
            if (re.fullmatch(r"(?:[A-Z]\d{4}|\d{4,5})", code) and qty.isdigit() a
                    and re.fullmatch(r"[\d,]+\.\d{2}", lt)):
                codes.append(code)
                sum_line_total += money_to_float(lt)
    if not codes:
        return "", dict(pdf_ok=0, pdf_parse_notes="no_items_parsed", pdf_zip3=zip
                        pdf_n_lines=0, pdf_n_unique_codes=0, pdf_total=float(tota
    codes_doc = " ".join(codes)
    meta = dict(
        pdf_ok=1,
        pdf_parse_notes="ok" if total_val is not None else "ok_no_total",
        pdf_zip3=zip3,
        pdf_insurance=insurance,
        pdf_n_lines=int(len(codes)),
        pdf_n_unique_codes=int(len(set(codes))),
        pdf_total=float(total_val) if total_val is not None else float(sum_line_t
        pdf_sum_line_total=float(sum_line_total),
    )
    return codes_doc, meta
def featurize_receipt(patient_id: int, pdf_dir: Path) -> ReceiptProfile:
    pdf_path = pdf_dir / f"receipt_{patient_id}.pdf"
    empty = ReceiptProfile(
        meta=dict(pdf_ok=0, pdf_parse_notes="missing_pdf", pdf_zip3=None, pdf_ins
                  pdf_n_lines=0, pdf_n_unique_codes=0, pdf_total=0.0, pdf_sum_lin
        codes_doc=""
    )

    if not pdf_path.exists():
        return empty
    try:
        with fitz.open(pdf_path) as doc:
            text = ""
            for page in doc:
                text += (page.get_text("text") or "") + "\n"
    except Exception as e:
        empty.meta["pdf_parse_notes"] = f"open_error:{type(e).__name__}"
        return empty
    codes_doc, meta = extract_codes_doc_from_pdf_text(text)
    return ReceiptProfile(meta=meta, codes_doc=codes_doc)
def load_cache(cache_path: Path) -> Dict[str, Any]:
    if cache_path.exists():
        try:
            obj = joblib.load(cache_path)
            if isinstance(obj, dict) and obj.get("version") == CACHE_VERSION and 
                return obj
        except Exception:
            pass
    return {"version": CACHE_VERSION, "data": {}}
def save_cache(cache_path: Path, cache_obj: Dict[str, Any]) -> None:
    try:
        joblib.dump(cache_obj, cache_path)
    except Exception as e:
        logging.warning(f"Cache save failed: {e}")
def build_receipt_tables(patient_ids: List[int], pdf_dir: Path, cache_path: Path) 
    """
    Returns:
      receipt_meta_df: patient_id + pdf_* meta columns
      receipt_doc_df : patient_id + codes_doc
    """

    cache_obj = load_cache(cache_path)
    data: Dict[str, Any] = cache_obj.get("data", {})
    to_parse = [pid for pid in patient_ids if str(pid) not in data]
    logging.info(f"Receipt cache: {len(patient_ids) - len(to_parse):,} hit | {len
    for pid in tqdm(to_parse, desc="Parsing receipts", ncols=100):
        prof = featurize_receipt(pid, pdf_dir)
        data[str(pid)] = {"meta": prof.meta, "codes_doc": prof.codes_doc}
    cache_obj["data"] = data
    if to_parse:
        save_cache(cache_path, cache_obj)
    meta_rows = []
    doc_rows = []
    for pid in patient_ids:
        obj = data.get(str(pid))
        if obj is None:
            prof = featurize_receipt(pid, pdf_dir)
            obj = {"meta": prof.meta, "codes_doc": prof.codes_doc}
        meta_rows.append({"patient_id": pid, **(obj.get("meta", {}) or {})})
        doc_rows.append({"patient_id": pid, "codes_doc": obj.get("codes_doc", "") 
    return pd.DataFrame(meta_rows), pd.DataFrame(doc_rows)
# =========================
# TABULAR SPARSE MATRIX
# =========================
def build_tabular_sparse_matrix(df_all: pd.DataFrame) -> csr_matrix:
    df = df_all.copy()
    # simple derived features
    df["prior_cost_per_visit"] = df["prior_ed_cost_5y_usd"] / (df["prior_ed_visit
    df["log_prior_cost"] = np.log1p(df["prior_ed_cost_5y_usd"].astype(float))
    df["log_prior_visits"] = np.log1p(df["prior_ed_visits_5y"].astype(float))
    # categorical normalization

    for c in ["primary_chronic", "sex", "insurance", "zip3", "pdf_insurance", "pd
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("missing")
    exclude = {"patient_id", "ed_cost_next3y_usd", "is_train", "codes_doc"}
    cat_cols = [c for c in ["primary_chronic", "sex", "insurance", "zip3", "pdf_i
                if c in df.columns]
    num_cols = [c for c in df.columns
                if c not in exclude and c not in cat_cols and pd.api.types.is_num
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), nu
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    X = preprocessor.fit_transform(df)
    X = X.tocsr() if hasattr(X, "tocsr") else csr_matrix(X)
    return ensure_csr_float32_int32(X)
def make_strat_labels(train_df: pd.DataFrame) -> np.ndarray:
    chronic = train_df["primary_chronic"].astype(str)
    cost_bin = pd.qcut(train_df["prior_ed_cost_5y_usd"], q=10, duplicates="drop")
    return (chronic + "_" + cost_bin).values
def chronic_bias_correction_mean(
    y_true: np.ndarray,
    oof_pred: np.ndarray,
    train_group: np.ndarray,
    test_pred: np.ndarray,
    test_group: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], float]:
    """
    Apply mean residual correction per group:
      bias[g] = mean(y - pred) over OOF

      pred_adj = pred + bias[group]
    """
    biases: Dict[str, float] = {}
    global_bias = float(np.mean(y_true - oof_pred))
    for g in np.unique(train_group):
        m = (train_group == g)
        if m.sum() == 0:
            continue
        biases[str(g)] = float(np.mean(y_true[m] - oof_pred[m]))
    oof_adj = oof_pred.copy()
    for g, b in biases.items():
        oof_adj[train_group == g] += b
    test_adj = test_pred.copy()
    for i in range(len(test_adj)):
        g = str(test_group[i])
        test_adj[i] += biases.get(g, global_bias)
    return oof_adj, test_adj, biases, global_bias
# =========================
# MAIN
# =========================
def main() -> None:
    setup_logging()
    logging.info(f"XGBoost version: {getattr(xgb, '__version__', 'unknown')}")
    logging.info("=== Iteration 6: Monolithic XGBoost (GPU) + TF-IDF(1-3 grams, 1
    logging.info(f"CV: FAST_MODE={FAST_MODE} | repeats={len(REPEAT_SEEDS)} | fold
    logging.info(f"XGB: n_estimators={N_ESTIMATORS} | lr={LEARNING_RATE} | device
    # Load
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    logging.info(f"Loaded train={train_df.shape}, test={test_df.shape}")
    # Optional patients.csv
    if PATIENTS_CSV.exists():

        patients = pd.read_csv(PATIENTS_CSV)
        if "patient_id" in patients.columns:
            train_df = train_df.merge(patients, on="patient_id", how="left")
            test_df = test_df.merge(patients, on="patient_id", how="left")
            logging.info(f"Merged patients.csv: train={train_df.shape}, test={test
        else:
            logging.warning("patients.csv exists but has no patient_id; skipping 
    else:
        logging.info("patients.csv not found; continuing without it.")
    train_df["is_train"] = 1
    test_df["is_train"] = 0
    df_all = pd.concat([train_df, test_df], ignore_index=True)
    # Parse receipts -> codes_doc + meta
    all_pids = df_all["patient_id"].astype(int).unique().tolist()
    all_pids.sort()
    meta_df, doc_df = build_receipt_tables(all_pids, PDF_DIR, CACHE_PATH)
    df_all = df_all.merge(meta_df, on="patient_id", how="left").merge(doc_df, on=
    df_all["codes_doc"] = df_all["codes_doc"].fillna("").astype(str)
    logging.info(f"Merged receipt meta+doc: df_all={df_all.shape}")
    # Build TF-IDF (1–3 grams, max_features=1000)
    logging.info("Building TF-IDF on code documents...")
    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b[A-Za-z0-9]+\b",
        lowercase=False,
        ngram_range=TFIDF_NGRAM_RANGE,
        max_features=TFIDF_MAX_FEATURES,
        min_df=1,
        max_df=1.0,
        sublinear_tf=True,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        dtype=np.float32,
    )
    X_tfidf_all = vectorizer.fit_transform(df_all["codes_doc"].values)
    X_tfidf_all = ensure_csr_float32_int32(X_tfidf_all)

    logging.info(f"TF-IDF matrix: shape={X_tfidf_all.shape} | nnz={X_tfidf_all.nn
    # Build tabular sparse
    logging.info("Building tabular sparse matrix (OHE + numeric)...")
    X_tab_all = build_tabular_sparse_matrix(df_all)
    logging.info(f"Tabular matrix: shape={X_tab_all.shape} | nnz={X_tab_all.nnz:,
    # Combine
    X_all = hstack([X_tab_all, X_tfidf_all], format="csr")
    X_all = ensure_csr_float32_int32(X_all)
    logging.info(f"FINAL X_all: shape={X_all.shape} | nnz={X_all.nnz:,}")
    # Split
    is_train_mask = df_all["is_train"].values.astype(int) == 1
    X_train = X_all[is_train_mask]
    X_test = X_all[~is_train_mask]
    y = train_df["ed_cost_next3y_usd"].values.astype(np.float32)
    train_chronic = train_df["primary_chronic"].astype(str).values
    test_chronic = test_df["primary_chronic"].astype(str).values
    # CV splits
    strat = make_strat_labels(train_df)
    splits = []
    for rep, seed in enumerate(REPEAT_SEEDS):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), strat
            splits.append((rep, fold, tr_idx, va_idx))
    logging.info(f"Total CV splits: {len(splits)}")
    # Storage
    oof_sum = np.zeros(len(y), dtype=np.float64)
    oof_cnt = np.zeros(len(y), dtype=np.float64)
    test_sum = np.zeros(X_test.shape[0], dtype=np.float64)
    n_models = 0
    # XGB params (single model, tuned for finer convergence)
    xgb_params = dict(
        objective="reg:absoluteerror",
        eval_metric="mae",

        tree_method="hist",
        device=XGB_DEVICE,
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_lambda=1.0,
        min_child_weight=1.0,
        n_jobs=-1,
        verbosity=1,
        early_stopping_rounds=EARLY_STOPPING,  # must be in __init__ for xgboost>
    )
    fit_sig = None
    # Train
    for rep, fold, tr_idx, va_idx in tqdm(splits, desc="CV folds", ncols=110):
        t0 = time.time()
        tag = f"rep={rep} fold={fold}"
        logging.info(f"{tag} | START | train={len(tr_idx)} val={len(va_idx)}")
        X_tr = X_train[tr_idx]
        X_va = X_train[va_idx]
        y_tr = y[tr_idx]
        y_va = y[va_idx]
        logging.info(f"{tag} | XGB fit START | verbose_every={XGB_VERBOSE_EVERY}"
        model = XGBRegressor(**xgb_params, random_state=1000 * rep + fold)
        # heartbeat compatibility across versions
        if fit_sig is None:
            fit_sig = inspect.signature(model.fit).parameters
        fit_kwargs = {"eval_set": [(X_va, y_va)]}
        if "verbose_eval" in fit_sig:
            fit_kwargs["verbose_eval"] = XGB_VERBOSE_EVERY
        else:
            fit_kwargs["verbose"] = XGB_VERBOSE_EVERY

        model.fit(X_tr, y_tr, **fit_kwargs)
        best_it = getattr(model, "best_iteration", None)
        logging.info(f"{tag} | XGB fit END | best_iteration={best_it}")
        p_va = model.predict(X_va)
        p_te = model.predict(X_test)
        if CLIP_AT_ZERO:
            p_va = np.maximum(p_va, 0.0)
            p_te = np.maximum(p_te, 0.0)
        oof_sum[va_idx] += p_va
        oof_cnt[va_idx] += 1.0
        test_sum += p_te
        n_models += 1
        fold_mae = mae(y_va, p_va)
        dt = time.time() - t0
        logging.info(f"{tag} | END | fold_time={dt:.1f}s | fold_MAE={fold_mae:.3f
    # Aggregate predictions
    oof = (oof_sum / np.maximum(oof_cnt, 1.0)).astype(np.float32)
    test_pred = (test_sum / max(1, n_models)).astype(np.float32)
    base_oof_mae = mae(y, oof)
    logging.info(f"OOF MAE (raw): {base_oof_mae:.4f}")
    # Chronic condition bias correction (mean residual per group)
    logging.info("Applying chronic condition mean-residual bias correction (OOF-b
    oof_cal, test_cal, biases, global_bias = chronic_bias_correction_mean(
        y_true=y,
        oof_pred=oof,
        train_group=train_chronic,
        test_pred=test_pred,
        test_group=test_chronic,
    )
    if CLIP_AT_ZERO:
        oof_cal = np.maximum(oof_cal, 0.0)
        test_cal = np.maximum(test_cal, 0.0)

    cal_mae = mae(y, oof_cal)
    logging.info(f"Biases by chronic (mean residual): {biases} | global_bias={glo
    logging.info(f"OOF MAE (after chronic bias correction): {cal_mae:.4f}")
    # Save submission
    sub = pd.DataFrame({
        "patient_id": test_df["patient_id"].astype(int).values,
        "ed_cost_next3y_usd": test_cal.astype(np.float32),
    })
    sub.to_csv(SUBMISSION_PATH, index=False)
    logging.info(f"Saved submission: {SUBMISSION_PATH}")
if __name__ == "__main__":
    main()
description
```

### C10 — ITERATION 6 (FIXED) — Monolithic XGBoost (GPU) + TF-IDF (1–3 grams, 1000 features (pages 266-289)
```python
# -*- coding: utf-8 -*-
r"""
ITERATION 6 (FIXED) — Monolithic XGBoost (GPU) + TF-IDF (1–3 grams, 1000 features
=================================================================================
Fixes vs previous Iter 6:
- Robust code extraction: regex-find all CPT/HCPCS-like codes from full PDF text 
- TF-IDF guardrails: if vocabulary is empty, logs + falls back to tabular-only in
- Cache version bumped so a previously-bad cache won't poison current run.
===========================================================
INSTALL (run once)
===========================================================
pip install -U pandas numpy scipy scikit-learn xgboost pymupdf joblib tqdm
"""
from __future__ import annotations
import re
import time

import logging
import warnings
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import fitz  # PyMuPDF
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
from xgboost import XGBRegressor
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# =========================
# CONFIG
# =========================
BASE_DIR = Path(r"D:\AgentDs\agent_ds_healthcare")
TRAIN_CSV = BASE_DIR / "ed_cost_train.csv"
TEST_CSV = BASE_DIR / "ed_cost_test.csv"
PATIENTS_CSV = BASE_DIR / "patients.csv"  # optional
PDF_DIR = BASE_DIR / "receipts_pdf"
SUBMISSION_PATH = BASE_DIR / "submission.csv"

# Cache (bumped to invalidate any previous "empty codes_doc" cache)
CACHE_PATH = BASE_DIR / "cache_receipts_iter6_codes.joblib"
CACHE_VERSION = 62
# CV
FAST_MODE = True
N_FOLDS = 5
REPEAT_SEEDS = [2025, 2026] if FAST_MODE else [2024, 2025, 2026]
EARLY_STOPPING = 200
# TF-IDF
TFIDF_MAX_FEATURES = 1000
TFIDF_NGRAM_RANGE = (1, 3)
# XGB
XGB_DEVICE = "cuda"     # "cuda" or "cuda:0"
N_ESTIMATORS = 8000
LEARNING_RATE = 0.02
# Heartbeats
XGB_VERBOSE_EVERY = 100
# Post-process
CLIP_AT_ZERO = True
# =========================
# LOGGING
# =========================
def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | 
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))
def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_out

    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)
def ensure_csr_float32_int32(X: csr_matrix) -> csr_matrix:
    X = X.tocsr()
    if X.data.dtype != np.float32:
        X.data = X.data.astype(np.float32, copy=False)
    if X.indices.dtype != np.int32:
        X.indices = X.indices.astype(np.int32, copy=False)
    if X.indptr.dtype != np.int32:
        X.indptr = X.indptr.astype(np.int32, copy=False)
    return X
# =========================
# REGEX (RAW STRINGS)
# =========================
PID_RE = re.compile(r"Patient ID:\s*(\d+)")
ZIP_INS_RE = re.compile(r"ZIP3:\s*([0-9]{3})\s+Insurance:\s*([A-Za-z_]+)")
TOTAL_ANY_RE = re.compile(r"TOTAL\s+([\d,]+\.\d{2})")
# IMPORTANT:
# Use 5-digit numeric CPTs + HCPCS like "G0378" (letter + 4 digits).
# This avoids accidentally capturing 3-digit ZIPs or small patient IDs.
CODE_TOKEN_RE = re.compile(r"\b(?:[A-Z]\d{4}|\d{5})\b")
def parse_header_fields(text: str) -> Tuple[Optional[int], Optional[str], Optiona
    pid = None
    zip3 = None
    insurance = None
    m = PID_RE.search(text)
    if m:
        pid = int(m.group(1))
    m = ZIP_INS_RE.search(text)
    if m:
        zip3 = str(m.group(1))

        insurance = str(m.group(2)).lower()
    return pid, zip3, insurance
def money_to_float(tok: str) -> float:
    return float(tok.replace(",", ""))
@dataclass
class ReceiptProfile:
    meta: Dict[str, Any]
    codes_doc: str
def extract_codes_doc_from_pdf_text(text: str, patient_id: int) -> Tuple[str, Dict
    """
    Robustly build a whitespace-separated "document" consisting of CPT/HCPCS-like 
    Key design:
    - Do NOT rely on line structure (PyMuPDF can break tables into column-cells).
    - Just regex-extract all codes from the full text.
    """
    _, zip3, insurance = parse_header_fields(text)
    codes = CODE_TOKEN_RE.findall(text or "")
    # Defensive: if patient_id happens to be 5 digits and appears in text, drop i
    pid_str = str(patient_id)
    if len(pid_str) == 5:
        codes = [c for c in codes if c != pid_str]
    codes_doc = " ".join(codes)
    # total (optional meta feature)
    total_val = None
    m = TOTAL_ANY_RE.search(text or "")
    if m:
        try:
            total_val = money_to_float(m.group(1))
        except Exception:

            total_val = None
    meta = dict(
        pdf_ok=1 if len(codes) > 0 else 0,
        pdf_parse_notes="ok" if len(codes) > 0 else "no_codes_found",
        pdf_zip3=zip3,
        pdf_insurance=insurance,
        pdf_n_lines=int(len(codes)),                 # "lines" here = number of c
        pdf_n_unique_codes=int(len(set(codes))),
        pdf_total=float(total_val) if total_val is not None else 0.0,
    )
    return codes_doc, meta
def featurize_receipt(patient_id: int, pdf_dir: Path) -> ReceiptProfile:
    pdf_path = pdf_dir / f"receipt_{patient_id}.pdf"
    empty = ReceiptProfile(
        meta=dict(pdf_ok=0, pdf_parse_notes="missing_pdf", pdf_zip3=None, pdf_ins
                  pdf_n_lines=0, pdf_n_unique_codes=0, pdf_total=0.0),
        codes_doc=""
    )
    if not pdf_path.exists():
        return empty
    try:
        with fitz.open(pdf_path) as doc:
            text = ""
            for page in doc:
                text += (page.get_text("text") or "") + "\n"
    except Exception as e:
        empty.meta["pdf_parse_notes"] = f"open_error:{type(e).__name__}"
        return empty
    codes_doc, meta = extract_codes_doc_from_pdf_text(text, patient_id)
    return ReceiptProfile(meta=meta, codes_doc=codes_doc)
def load_cache(cache_path: Path) -> Dict[str, Any]:

    if cache_path.exists():
        try:
            obj = joblib.load(cache_path)
            if isinstance(obj, dict) and obj.get("version") == CACHE_VERSION and 
                return obj
        except Exception:
            pass
    return {"version": CACHE_VERSION, "data": {}}
def save_cache(cache_path: Path, cache_obj: Dict[str, Any]) -> None:
    try:
        joblib.dump(cache_obj, cache_path)
    except Exception as e:
        logging.warning(f"Cache save failed: {e}")
def build_receipt_tables(patient_ids: List[int], pdf_dir: Path, cache_path: Path) 
    """
    Returns:
      receipt_meta_df: patient_id + pdf_* meta columns
      receipt_doc_df : patient_id + codes_doc
    """
    cache_obj = load_cache(cache_path)
    data: Dict[str, Any] = cache_obj.get("data", {})
    to_parse = [pid for pid in patient_ids if str(pid) not in data]
    logging.info(f"Receipt cache: {len(patient_ids) - len(to_parse):,} hit | {len
    for pid in tqdm(to_parse, desc="Parsing receipts", ncols=100):
        prof = featurize_receipt(pid, pdf_dir)
        data[str(pid)] = {"meta": prof.meta, "codes_doc": prof.codes_doc}
    cache_obj["data"] = data
    if to_parse:
        save_cache(cache_path, cache_obj)
    meta_rows = []
    doc_rows = []
    for pid in patient_ids:

        obj = data.get(str(pid))
        if obj is None:
            prof = featurize_receipt(pid, pdf_dir)
            obj = {"meta": prof.meta, "codes_doc": prof.codes_doc}
        meta_rows.append({"patient_id": pid, **(obj.get("meta", {}) or {})})
        doc_rows.append({"patient_id": pid, "codes_doc": obj.get("codes_doc", "") 
    return pd.DataFrame(meta_rows), pd.DataFrame(doc_rows)
# =========================
# TABULAR SPARSE MATRIX
# =========================
def build_tabular_sparse_matrix(df_all: pd.DataFrame) -> csr_matrix:
    df = df_all.copy()
    # simple derived features
    df["prior_cost_per_visit"] = df["prior_ed_cost_5y_usd"] / (df["prior_ed_visit
    df["log_prior_cost"] = np.log1p(df["prior_ed_cost_5y_usd"].astype(float))
    df["log_prior_visits"] = np.log1p(df["prior_ed_visits_5y"].astype(float))
    # categorical normalization
    for c in ["primary_chronic", "sex", "insurance", "zip3", "pdf_insurance", "pd
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("missing")
    exclude = {"patient_id", "ed_cost_next3y_usd", "is_train", "codes_doc"}
    cat_cols = [c for c in ["primary_chronic", "sex", "insurance", "zip3", "pdf_i
                if c in df.columns]
    num_cols = [c for c in df.columns
                if c not in exclude and c not in cat_cols and pd.api.types.is_num
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), nu
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")
        ],
        remainder="drop",
        sparse_threshold=0.3,

    )
    X = preprocessor.fit_transform(df)
    X = X.tocsr() if hasattr(X, "tocsr") else csr_matrix(X)
    return ensure_csr_float32_int32(X)
def make_strat_labels(train_df: pd.DataFrame) -> np.ndarray:
    chronic = train_df["primary_chronic"].astype(str)
    cost_bin = pd.qcut(train_df["prior_ed_cost_5y_usd"], q=10, duplicates="drop")
    return (chronic + "_" + cost_bin).values
def chronic_bias_correction_mean(
    y_true: np.ndarray,
    oof_pred: np.ndarray,
    train_group: np.ndarray,
    test_pred: np.ndarray,
    test_group: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], float]:
    """
    Mean residual correction per group:
      bias[g] = mean(y - pred) over OOF
      pred_adj = pred + bias[group]
    """
    biases: Dict[str, float] = {}
    global_bias = float(np.mean(y_true - oof_pred))
    for g in np.unique(train_group):
        m = (train_group == g)
        if m.sum() == 0:
            continue
        biases[str(g)] = float(np.mean(y_true[m] - oof_pred[m]))
    oof_adj = oof_pred.copy()
    for g, b in biases.items():
        oof_adj[train_group == g] += b
    test_adj = test_pred.copy()
    for i in range(len(test_adj)):

        g = str(test_group[i])
        test_adj[i] += biases.get(g, global_bias)
    return oof_adj, test_adj, biases, global_bias
# =========================
# MAIN
# =========================
def main() -> None:
    setup_logging()
    logging.info(f"XGBoost version: {getattr(xgb, '__version__', 'unknown')}")
    logging.info("=== Iteration 6 (Fixed): Monolithic XGBoost (GPU) + TF-IDF code
    logging.info(f"CV: FAST_MODE={FAST_MODE} | repeats={len(REPEAT_SEEDS)} | fold
    logging.info(f"XGB: n_estimators={N_ESTIMATORS} | lr={LEARNING_RATE} | device
    # Load
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    logging.info(f"Loaded train={train_df.shape}, test={test_df.shape}")
    # Optional patients.csv
    if PATIENTS_CSV.exists():
        patients = pd.read_csv(PATIENTS_CSV)
        if "patient_id" in patients.columns:
            train_df = train_df.merge(patients, on="patient_id", how="left")
            test_df = test_df.merge(patients, on="patient_id", how="left")
            logging.info(f"Merged patients.csv: train={train_df.shape}, test={test
        else:
            logging.warning("patients.csv exists but has no patient_id; skipping 
    else:
        logging.info("patients.csv not found; continuing without it.")
    train_df["is_train"] = 1
    test_df["is_train"] = 0
    df_all = pd.concat([train_df, test_df], ignore_index=True)
    # Parse receipts -> codes_doc + meta
    all_pids = df_all["patient_id"].astype(int).unique().tolist()
    all_pids.sort()

    meta_df, doc_df = build_receipt_tables(all_pids, PDF_DIR, CACHE_PATH)
    df_all = df_all.merge(meta_df, on="patient_id", how="left").merge(doc_df, on=
    df_all["codes_doc"] = df_all["codes_doc"].fillna("").astype(str)
    nonempty_docs = int((df_all["codes_doc"].str.len() > 0).sum())
    logging.info(f"codes_doc non-empty: {nonempty_docs}/{len(df_all)}")
    # Build TF-IDF
    logging.info("Building TF-IDF on code documents...")
    if nonempty_docs == 0:
        logging.warning("No code tokens found in any PDF -> TF-IDF disabled (tabu
        X_tfidf_all = csr_matrix((len(df_all), 0), dtype=np.float32)
    else:
        vectorizer = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"(?u)\b[A-Za-z0-9]+\b",
            lowercase=False,
            ngram_range=TFIDF_NGRAM_RANGE,
            max_features=TFIDF_MAX_FEATURES,
            min_df=1,
            max_df=1.0,
            sublinear_tf=True,
            norm="l2",
            use_idf=True,
            smooth_idf=True,
            dtype=np.float32,
        )
        try:
            X_tfidf_all = vectorizer.fit_transform(df_all["codes_doc"].values)
            X_tfidf_all = ensure_csr_float32_int32(X_tfidf_all)
            logging.info(f"TF-IDF OK: shape={X_tfidf_all.shape} | nnz={X_tfidf_al
        except ValueError as e:
            # Robust fallback: restrict token_pattern to code-like tokens
            logging.warning(f"TF-IDF failed ({e}). Retrying with stricter code to
            vectorizer2 = TfidfVectorizer(
                analyzer="word",
                token_pattern=r"(?u)\b(?:[A-Z]\d{4}|\d{5})\b",
                lowercase=False,
                ngram_range=TFIDF_NGRAM_RANGE,

                max_features=TFIDF_MAX_FEATURES,
                min_df=1,
                max_df=1.0,
                sublinear_tf=True,
                norm="l2",
                use_idf=True,
                smooth_idf=True,
                dtype=np.float32,
            )
            try:
                X_tfidf_all = vectorizer2.fit_transform(df_all["codes_doc"].value
                X_tfidf_all = ensure_csr_float32_int32(X_tfidf_all)
                logging.info(f"TF-IDF fallback OK: shape={X_tfidf_all.shape} | nn
            except ValueError as e2:
                logging.warning(f"TF-IDF fallback still failed ({e2}). Disabling 
                X_tfidf_all = csr_matrix((len(df_all), 0), dtype=np.float32)
    # Build tabular sparse
    logging.info("Building tabular sparse matrix (OHE + numeric)...")
    X_tab_all = build_tabular_sparse_matrix(df_all)
    logging.info(f"Tabular OK: shape={X_tab_all.shape} | nnz={X_tab_all.nnz:,}")
    # Combine
    X_all = hstack([X_tab_all, X_tfidf_all], format="csr")
    X_all = ensure_csr_float32_int32(X_all)
    logging.info(f"FINAL X_all: shape={X_all.shape} | nnz={X_all.nnz:,}")
    # Split
    is_train_mask = df_all["is_train"].values.astype(int) == 1
    X_train = X_all[is_train_mask]
    X_test = X_all[~is_train_mask]
    y = train_df["ed_cost_next3y_usd"].values.astype(np.float32)
    train_chronic = train_df["primary_chronic"].astype(str).values
    test_chronic = test_df["primary_chronic"].astype(str).values
    # CV splits (stratified; fallback to KFold if needed)
    strat = make_strat_labels(train_df)
    splits = []
    for rep, seed in enumerate(REPEAT_SEEDS):

        try:
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=se
            for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), st
                splits.append((rep, fold, tr_idx, va_idx))
        except ValueError as e:
            logging.warning(f"StratifiedKFold failed ({e}); falling back to KFold 
            kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
            for fold, (tr_idx, va_idx) in enumerate(kf.split(np.zeros(len(y)))):
                splits.append((rep, fold, tr_idx, va_idx))
    logging.info(f"Total CV splits: {len(splits)}")
    # Storage
    oof_sum = np.zeros(len(y), dtype=np.float64)
    oof_cnt = np.zeros(len(y), dtype=np.float64)
    test_sum = np.zeros(X_test.shape[0], dtype=np.float64)
    n_models = 0
    # XGB params
    xgb_params = dict(
        objective="reg:absoluteerror",
        eval_metric="mae",
        tree_method="hist",
        device=XGB_DEVICE,
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_lambda=1.0,
        min_child_weight=1.0,
        n_jobs=-1,
        verbosity=1,
        early_stopping_rounds=EARLY_STOPPING,  # required in __init__ for newer xg
    )
    fit_sig = None
    # Train
    for rep, fold, tr_idx, va_idx in tqdm(splits, desc="CV folds", ncols=110):

        t0 = time.time()
        tag = f"rep={rep} fold={fold}"
        logging.info(f"{tag} | START | train={len(tr_idx)} val={len(va_idx)}")
        X_tr = X_train[tr_idx]
        X_va = X_train[va_idx]
        y_tr = y[tr_idx]
        y_va = y[va_idx]
        logging.info(f"{tag} | XGB fit START | verbose_every={XGB_VERBOSE_EVERY}"
        model = XGBRegressor(**xgb_params, random_state=1000 * rep + fold)
        if fit_sig is None:
            fit_sig = inspect.signature(model.fit).parameters
        fit_kwargs = {"eval_set": [(X_va, y_va)]}
        if "verbose_eval" in fit_sig:
            fit_kwargs["verbose_eval"] = XGB_VERBOSE_EVERY
        else:
            fit_kwargs["verbose"] = XGB_VERBOSE_EVERY
        model.fit(X_tr, y_tr, **fit_kwargs)
        best_it = getattr(model, "best_iteration", None)
        logging.info(f"{tag} | XGB fit END | best_iteration={best_it}")
        p_va = model.predict(X_va)
        p_te = model.predict(X_test)
        if CLIP_AT_ZERO:
            p_va = np.maximum(p_va, 0.0)
            p_te = np.maximum(p_te, 0.0)
        oof_sum[va_idx] += p_va
        oof_cnt[va_idx] += 1.0
        test_sum += p_te
        n_models += 1
        fold_mae = mae(y_va, p_va)
        dt = time.time() - t0

        logging.info(f"{tag} | END | fold_time={dt:.1f}s | fold_MAE={fold_mae:.3f
    # Aggregate predictions
    oof = (oof_sum / np.maximum(oof_cnt, 1.0)).astype(np.float32)
    test_pred = (test_sum / max(1, n_models)).astype(np.float32)
    base_oof_mae = mae(y, oof)
    logging.info(f"OOF MAE (raw): {base_oof_mae:.4f}")
    # Chronic condition bias correction (mean residual per chronic group)
    logging.info("Applying chronic condition mean-residual bias correction (OOF-b
    oof_cal, test_cal, biases, global_bias = chronic_bias_correction_mean(
        y_true=y,
        oof_pred=oof,
        train_group=train_chronic,
        test_pred=test_pred,
        test_group=test_chronic,
    )
    if CLIP_AT_ZERO:
        oof_cal = np.maximum(oof_cal, 0.0)
        test_cal = np.maximum(test_cal, 0.0)
    cal_mae = mae(y, oof_cal)
    logging.info(f"Biases by chronic (mean residual): {biases} | global_bias={glo
    logging.info(f"OOF MAE (after chronic bias correction): {cal_mae:.4f}")
    # Safety: if correction unexpectedly worsens OOF, revert
    if cal_mae > base_oof_mae + 1e-6:
        logging.warning("Chronic bias correction worsened OOF MAE. Reverting to r
        test_out = test_pred
    else:
        test_out = test_cal
    if CLIP_AT_ZERO:
        test_out = np.maximum(test_out, 0.0)
    # Save submission
    sub = pd.DataFrame({
        "patient_id": test_df["patient_id"].astype(int).values,

If you still see codes_doc non-empty: 0/XXXX , that means you’re not reading any PDFs
(wrong folder / filenames). In that case this script will still run (tabular-only), but you’ll
want to check:
PDF_DIR  points to the right folder
files are named exactly receipt_<patient_id>.pdf
Otherwise, this should eliminate the crash and keep your monolithic XGBoost + TF‑IDF
pipeline intact.
        "ed_cost_next3y_usd": test_out.astype(np.float32),
    })
    sub.to_csv(SUBMISSION_PATH, index=False)
    logging.info(f"Saved submission: {SUBMISSION_PATH}")
if __name__ == "__main__":
    main()
```

### C11 — ITERATION 7 — Drift-Aware XGBoost (GPU) + TF-IDF Codes + Importance Weighting + MA (pages 290-313)
```python
# -*- coding: utf-8 -*-
r"""
ITERATION 7 — Drift-Aware XGBoost (GPU) + TF-IDF Codes + Importance Weighting + MA
=================================================================================
Why this should beat the 478–486 plateau:
- If train/test are distribution-shifted, random CV is optimistic. Adversarial va
  drift and build a validation scheme closer to the hidden test distribution. (Tr
- Under covariate shift, standard CV/model selection can be biased; importance we
  technique to adjust training toward the test distribution (via density-ratio pr
  ratio using odds p/(1-p) from the adversarial classifier.
- MAE is minimized by the median; we calibrate using a shrunk (regularized) media
  group for stability (avoids huge unstable mean offsets).
===========================================================
INSTALL (run once)
===========================================================
pip install -U pandas numpy scipy scikit-learn xgboost pymupdf joblib tqdm
"""
from __future__ import annotations
import re
import time
import math
import argparse
import logging
import warnings
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import fitz  # PyMuPDF
from scipy.sparse import csr_matrix, hstack

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from xgboost import XGBRegressor
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# =========================
# PATHS / CONFIG
# =========================
BASE_DIR = Path(r"D:\AgentDs\agent_ds_healthcare")
TRAIN_CSV = BASE_DIR / "ed_cost_train.csv"
TEST_CSV = BASE_DIR / "ed_cost_test.csv"
PATIENTS_CSV = BASE_DIR / "patients.csv"
ADMISSIONS_TRAIN = BASE_DIR / "admissions_train.csv"
ADMISSIONS_TEST = BASE_DIR / "admissions_test.csv"
STAYS_TRAIN = BASE_DIR / "stays_train.csv"
STAYS_TEST = BASE_DIR / "stays_test.csv"
PDF_DIR = BASE_DIR / "receipts_pdf"
SUBMISSION_PATH = BASE_DIR / "submission.csv"
# Receipt cache
CACHE_PATH = BASE_DIR / "cache_receipts_iter7_codes.joblib"
CACHE_VERSION = 71
# TF-IDF (keep "simple", no SVD/PCA)
TFIDF_MAX_FEATURES = 1000
TFIDF_NGRAM_RANGE = (1, 3)

# Adversarial validation
ADV_N_SPLITS = 5
ADV_RANDOM_STATE = 2027
# Drift-aware folds for regression
REG_N_FOLDS = 5
REG_REPEAT_SEEDS = [2025, 2026]  # keep small for speed; increase if you want mor
EARLY_STOPPING = 200
# Importance weighting
USE_IMPORTANCE_WEIGHTS_DEFAULT = True
WEIGHT_EPS = 1e-4
WEIGHT_BETA = 0.50          # sqrt-odds to reduce variance
WEIGHT_CLIP_MIN = 0.25
WEIGHT_CLIP_MAX = 4.0
# Calibration (MAE-aligned)
CALIB_TOP_QUANTILE = 0.80   # judge calibration on most test-like 20% of train
CALIB_ALPHA = 50.0          # shrinkage strength (bigger => more shrink to global
# XGBoost
N_ESTIMATORS = 8000
LEARNING_RATE = 0.02
XGB_MAX_DEPTH = 6
XGB_SUBSAMPLE = 0.85
XGB_COLSAMPLE = 0.75
XGB_LAMBDA = 1.0
XGB_MIN_CHILD_WEIGHT = 1.0
# Heartbeats
XGB_VERBOSE_EVERY = 100
# Post
CLIP_AT_ZERO = True
# =========================
# LOGGING
# =========================

def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | 
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))
def make_ohe() -> OneHotEncoder:
    # sklearn version compat
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_out
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)
def ensure_csr_float32_int32(X: csr_matrix) -> csr_matrix:
    X = X.tocsr()
    if X.data.dtype != np.float32:
        X.data = X.data.astype(np.float32, copy=False)
    if X.indices.dtype != np.int32:
        X.indices = X.indices.astype(np.int32, copy=False)
    if X.indptr.dtype != np.int32:
        X.indptr = X.indptr.astype(np.int32, copy=False)
    return X
# =========================
# REGEX (RAW STRINGS)
# =========================
PID_RE = re.compile(r"Patient ID:\s*(\d+)")
ZIP_INS_RE = re.compile(r"ZIP3:\s*([0-9]{3})\s+Insurance:\s*([A-Za-z_]+)")
TOTAL_ANY_RE = re.compile(r"TOTAL\s+([\d,]+\.\d{2})")
# CPT 5-digit numeric + HCPCS like G0378
CODE_TOKEN_RE = re.compile(r"\b(?:[A-Z]\d{4}|\d{5})\b")
def money_to_float(tok: str) -> float:
    return float(tok.replace(",", ""))

def parse_header_fields(text: str) -> Tuple[Optional[int], Optional[str], Optiona
    pid = None
    zip3 = None
    insurance = None
    m = PID_RE.search(text)
    if m:
        pid = int(m.group(1))
    m = ZIP_INS_RE.search(text)
    if m:
        zip3 = str(m.group(1))
        insurance = str(m.group(2)).lower()
    return pid, zip3, insurance
@dataclass
class ReceiptProfile:
    meta: Dict[str, Any]
    codes_doc: str
def extract_codes_doc_from_pdf_text(text: str, patient_id: int) -> ReceiptProfile
    _, zip3, insurance = parse_header_fields(text or "")
    codes = CODE_TOKEN_RE.findall(text or "")
    pid_str = str(patient_id)
    if len(pid_str) == 5:
        codes = [c for c in codes if c != pid_str]
    codes_doc = " ".join(codes)
    total_val = None
    m = TOTAL_ANY_RE.search(text or "")
    if m:
        try:
            total_val = money_to_float(m.group(1))

        except Exception:
            total_val = None
    meta = dict(
        pdf_ok=1 if len(codes) > 0 else 0,
        pdf_parse_notes="ok" if len(codes) > 0 else "no_codes_found",
        pdf_zip3=zip3,
        pdf_insurance=insurance,
        pdf_n_codes=int(len(codes)),
        pdf_n_unique_codes=int(len(set(codes))),
        pdf_total=float(total_val) if total_val is not None else 0.0,
    )
    return ReceiptProfile(meta=meta, codes_doc=codes_doc)
def featurize_receipt(patient_id: int, pdf_dir: Path) -> ReceiptProfile:
    pdf_path = pdf_dir / f"receipt_{patient_id}.pdf"
    empty = ReceiptProfile(
        meta=dict(
            pdf_ok=0,
            pdf_parse_notes="missing_pdf",
            pdf_zip3=None,
            pdf_insurance=None,
            pdf_n_codes=0,
            pdf_n_unique_codes=0,
            pdf_total=0.0,
        ),
        codes_doc="",
    )
    if not pdf_path.exists():
        return empty
    try:
        with fitz.open(pdf_path) as doc:
            text = ""
            for page in doc:
                text += (page.get_text("text") or "") + "\n"
    except Exception as e:

        empty.meta["pdf_parse_notes"] = f"open_error:{type(e).__name__}"
        return empty
    return extract_codes_doc_from_pdf_text(text, patient_id)
def load_cache(cache_path: Path) -> Dict[str, Any]:
    if cache_path.exists():
        try:
            obj = joblib.load(cache_path)
            if isinstance(obj, dict) and obj.get("version") == CACHE_VERSION and 
                return obj
        except Exception:
            pass
    return {"version": CACHE_VERSION, "data": {}}
def save_cache(cache_path: Path, cache_obj: Dict[str, Any]) -> None:
    try:
        joblib.dump(cache_obj, cache_path)
    except Exception as e:
        logging.warning(f"Cache save failed: {e}")
def build_receipt_tables(patient_ids: List[int], pdf_dir: Path, cache_path: Path) 
    cache_obj = load_cache(cache_path)
    data: Dict[str, Any] = cache_obj.get("data", {})
    to_parse = [pid for pid in patient_ids if str(pid) not in data]
    logging.info(f"Receipt cache: {len(patient_ids) - len(to_parse):,} hit | {len
    for pid in tqdm(to_parse, desc="Parsing receipts", ncols=100):
        prof = featurize_receipt(pid, pdf_dir)
        data[str(pid)] = {"meta": prof.meta, "codes_doc": prof.codes_doc}
    cache_obj["data"] = data
    if to_parse:
        save_cache(cache_path, cache_obj)
    meta_rows = []

    doc_rows = []
    for pid in patient_ids:
        obj = data.get(str(pid))
        if obj is None:
            prof = featurize_receipt(pid, pdf_dir)
            obj = {"meta": prof.meta, "codes_doc": prof.codes_doc}
        meta_rows.append({"patient_id": pid, **(obj.get("meta", {}) or {})})
        doc_rows.append({"patient_id": pid, "codes_doc": obj.get("codes_doc", "") 
    return pd.DataFrame(meta_rows), pd.DataFrame(doc_rows)
# =========================
# OPTIONAL RELATIONAL AGGS
# =========================
def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        logging.warning(f"Failed reading {path}: {e}")
        return None
def add_admissions_aggs(df_all: pd.DataFrame) -> pd.DataFrame:
    tr = safe_read_csv(ADMISSIONS_TRAIN)
    te = safe_read_csv(ADMISSIONS_TEST)
    if tr is None or te is None:
        logging.info("Admissions files not found (or unreadable). Skipping admiss
        return df_all
    adm = pd.concat([tr, te], ignore_index=True)
    if "readmit_30d" in adm.columns:
        adm = adm.drop(columns=["readmit_30d"])
    if "patient_id" not in adm.columns:
        logging.info("Admissions table has no patient_id. Skipping admissions agg
        return df_all

    logging.info(f"Admissions rows: {adm.shape}")
    # Numeric aggregations
    num_cols = [c for c in ["los_days", "acuity_emergent", "charlson_band", "ed_v
    agg_dict = {}
    if "admission_id" in adm.columns:
        agg_dict["adm_n"] = ("admission_id", "count")
    else:
        agg_dict["adm_n"] = ("patient_id", "size")
    for c in num_cols:
        agg_dict[f"adm_{c}_mean"] = (c, "mean")
        agg_dict[f"adm_{c}_max"] = (c, "max")
    agg = adm.groupby("patient_id").agg(**agg_dict).reset_index()
    # Categorical counts: primary_dx
    if "primary_dx" in adm.columns:
        dx_ct = pd.crosstab(adm["patient_id"], adm["primary_dx"])
        dx_ct.columns = [f"adm_primary_dx_{str(c)}_count" for c in dx_ct.columns]
        dx_ct = dx_ct.reset_index()
        agg = agg.merge(dx_ct, on="patient_id", how="left")
    # discharge_weekday counts
    if "discharge_weekday" in adm.columns:
        wd_ct = pd.crosstab(adm["patient_id"], adm["discharge_weekday"])
        wd_ct.columns = [f"adm_discharge_wd_{str(c)}_count" for c in wd_ct.column
        wd_ct = wd_ct.reset_index()
        agg = agg.merge(wd_ct, on="patient_id", how="left")
    logging.info(f"Admissions aggregates shape: {agg.shape}")
    out = df_all.merge(agg, on="patient_id", how="left")
    return out
def add_stays_aggs(df_all: pd.DataFrame) -> pd.DataFrame:
    tr = safe_read_csv(STAYS_TRAIN)
    te = safe_read_csv(STAYS_TEST)
    if tr is None or te is None:
        logging.info("Stays files not found (or unreadable). Skipping stays aggre

        return df_all
    st = pd.concat([tr, te], ignore_index=True)
    if "discharge_ready_day11" in st.columns:
        st = st.drop(columns=["discharge_ready_day11"])
    if "patient_id" not in st.columns:
        logging.info("Stays table has no patient_id. Skipping stays aggregates.")
        return df_all
    logging.info(f"Stays rows: {st.shape}")
    # Base count
    if "stay_id" in st.columns:
        agg = st.groupby("patient_id").agg(stay_n=("stay_id", "count")).reset_ind
    else:
        agg = st.groupby("patient_id").agg(stay_n=("patient_id", "size")).reset_i
    # unit_type counts
    if "unit_type" in st.columns:
        ut_ct = pd.crosstab(st["patient_id"], st["unit_type"])
        ut_ct.columns = [f"stay_unit_{str(c)}_count" for c in ut_ct.columns]
        ut_ct = ut_ct.reset_index()
        agg = agg.merge(ut_ct, on="patient_id", how="left")
    # admission_reason counts
    if "admission_reason" in st.columns:
        ar_ct = pd.crosstab(st["patient_id"], st["admission_reason"])
        ar_ct.columns = [f"stay_reason_{str(c)}_count" for c in ar_ct.columns]
        ar_ct = ar_ct.reset_index()
        agg = agg.merge(ar_ct, on="patient_id", how="left")
    logging.info(f"Stays aggregates shape: {agg.shape}")
    out = df_all.merge(agg, on="patient_id", how="left")
    return out
# =========================
# FEATURE BUILDING
# =========================

def build_tabular_sparse(df_all: pd.DataFrame) -> Tuple[csr_matrix, Optional[np.n
    df = df_all.copy()
    # derived
    if "prior_ed_cost_5y_usd" in df.columns and "prior_ed_visits_5y" in df.column
        df["prior_cost_per_visit"] = df["prior_ed_cost_5y_usd"] / (df["prior_ed_v
        df["log_prior_cost"] = np.log1p(df["prior_ed_cost_5y_usd"].astype(float))
        df["log_prior_visits"] = np.log1p(df["prior_ed_visits_5y"].astype(float))
    # receipt consistency feature
    if "pdf_total" in df.columns and "prior_ed_cost_5y_usd" in df.columns:
        df["pdf_total_minus_prior_cost"] = df["pdf_total"].astype(float) - df["pr
        df["pdf_total_absdiff_prior_cost"] = np.abs(df["pdf_total_minus_prior_cos
    # categorical normalization
    cat_candidates = ["primary_chronic", "sex", "insurance", "zip3", "pdf_insuran
    for c in cat_candidates:
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("missing")
    exclude = {"patient_id", "ed_cost_next3y_usd", "is_train", "codes_doc"}
    cat_cols = [c for c in cat_candidates if c in df.columns]
    num_cols = [c for c in df.columns if c not in exclude and c not in cat_cols a
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), nu
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")
                              ("ohe", make_ohe())]), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    X = preprocessor.fit_transform(df)
    X = X.tocsr() if hasattr(X, "tocsr") else csr_matrix(X)
    X = ensure_csr_float32_int32(X)
    # feature names (best effort)
    feat_names = None

    try:
        feat_names = preprocessor.get_feature_names_out()
    except Exception:
        feat_names = None
    return X, feat_names, preprocessor
def build_tfidf_sparse(codes_doc: np.ndarray) -> Tuple[csr_matrix, Optional[np.nd
    nonempty = int(np.sum(pd.Series(codes_doc).astype(str).str.len().values > 0))
    if nonempty == 0:
        logging.warning("No non-empty codes_doc found. TF-IDF disabled.")
        X = csr_matrix((len(codes_doc), 0), dtype=np.float32)
        return X, None, None
    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b(?:[A-Z]\d{4}|\d{5})\b",
        lowercase=False,
        ngram_range=TFIDF_NGRAM_RANGE,
        max_features=TFIDF_MAX_FEATURES,
        min_df=1,
        max_df=1.0,
        sublinear_tf=True,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        dtype=np.float32,
    )
    try:
        X = vectorizer.fit_transform(codes_doc)
        X = ensure_csr_float32_int32(X)
        names = vectorizer.get_feature_names_out()
        return X, names, vectorizer
    except ValueError as e:
        logging.warning(f"TF-IDF failed ({e}). Disabling TF-IDF.")
        X = csr_matrix((len(codes_doc), 0), dtype=np.float32)
        return X, None, None

def build_full_features(df_all: pd.DataFrame) -> Tuple[csr_matrix, Optional[np.nd
    X_tab, tab_names, _ = build_tabular_sparse(df_all)
    X_txt, txt_names, _ = build_tfidf_sparse(df_all["codes_doc"].values.astype(st
    X_all = hstack([X_tab, X_txt], format="csr")
    X_all = ensure_csr_float32_int32(X_all)
    if tab_names is not None and txt_names is not None:
        feat_names = np.concatenate([tab_names, txt_names])
    else:
        feat_names = None
    return X_all, feat_names
# =========================
# ADVERSARIAL VALIDATION
# =========================
def adversarial_oof_proba(X_all: csr_matrix, is_test: np.ndarray, feat_names: Opt
    y_adv = is_test.astype(int)
    if len(np.unique(y_adv)) < 2:
        raise RuntimeError("Adversarial labels have <2 classes. Check train/test 
    skf = StratifiedKFold(n_splits=ADV_N_SPLITS, shuffle=True, random_state=ADV_RA
    p = np.zeros(X_all.shape[0], dtype=np.float32)
    fold_aucs = []
    logging.info("Adversarial validation: training LogisticRegression(train=0 vs 
    for k, (tr, va) in enumerate(skf.split(np.zeros(len(y_adv)), y_adv)):
        t0 = time.time()
        clf = LogisticRegression(
            solver="liblinear",
            C=1.0,
            class_weight="balanced",
            max_iter=200,
        )
        clf.fit(X_all[tr], y_adv[tr])
        p_va = clf.predict_proba(X_all[va])[:, 1].astype(np.float32)
        p[va] = p_va
        auc = roc_auc_score(y_adv[va], p_va)

        fold_aucs.append(float(auc))
        logging.info(f"ADV fold {k} | AUC={auc:.4f} | time={time.time()-t0:.1f}s"
    auc_all = roc_auc_score(y_adv, p)
    logging.info(f"Adversarial OOF AUC (overall): {auc_all:.4f} | fold_AUCs={fold_
    if diagnostic:
        # Fit on full to display top drift features (best-effort)
        try:
            clf_full = LogisticRegression(
                solver="liblinear",
                C=1.0,
                class_weight="balanced",
                max_iter=200,
            )
            clf_full.fit(X_all, y_adv)
            coef = clf_full.coef_.ravel()
            if feat_names is not None and len(feat_names) == len(coef):
                top_pos = np.argsort(coef)[-20:][::-1]
                top_neg = np.argsort(coef)[:20]
                logging.info("Top features pushing 'TEST' (positive coef):")
                for i in top_pos:
                    logging.info(f"  + {feat_names[i]} | coef={coef[i]:.4f}")
                logging.info("Top features pushing 'TRAIN' (negative coef):")
                for i in top_neg:
                    logging.info(f"  - {feat_names[i]} | coef={coef[i]:.4f}")
            else:
                logging.info("Feature names unavailable; skipping drift feature p
        except Exception as e:
            logging.warning(f"Diagnostic coefficient dump failed: {e}")
    return p
def make_drift_folds(p_train: np.ndarray, n_folds: int) -> np.ndarray:
    n = len(p_train)
    order = np.argsort(p_train)  # low -> high test-likeness
    fold_id = np.empty(n, dtype=np.int32)
    bins = (np.arange(n) * n_folds) // n
    bins = np.clip(bins, 0, n_folds - 1)

    fold_id[order] = bins.astype(np.int32)
    return fold_id
def compute_importance_weights(p_train: np.ndarray) -> np.ndarray:
    # odds = p/(1-p) = (pi_test/pi_train) * (p_test(x)/p_train(x))  (constant fac
    p = np.clip(p_train.astype(np.float64), WEIGHT_EPS, 1.0 - WEIGHT_EPS)
    odds = p / (1.0 - p)
    w = np.power(odds, WEIGHT_BETA)
    w = np.clip(w, WEIGHT_CLIP_MIN, WEIGHT_CLIP_MAX)
    w = w / np.mean(w)  # normalize to mean 1
    return w.astype(np.float32)
# =========================
# MAE CALIBRATION (SHRUNK MEDIAN RESIDUAL)
# =========================
def shrunk_median_bias(residuals: np.ndarray, groups: np.ndarray, alpha: float) -
    g = groups.astype(str)
    global_med = float(np.median(residuals))
    biases: Dict[str, float] = {}
    for key in np.unique(g):
        m = (g == key)
        rg = residuals[m]
        if rg.size == 0:
            continue
        med_g = float(np.median(rg))
        n = float(rg.size)
        b = (n * med_g + alpha * global_med) / (n + alpha)
        biases[str(key)] = float(b)
    return biases, global_med
def apply_group_bias(pred: np.ndarray, groups: np.ndarray, biases: Dict[str, float
    out = pred.astype(np.float64).copy()
    g = groups.astype(str)
    for i in range(len(out)):

        out[i] += biases.get(str(g[i]), default_bias)
    return out.astype(np.float32)
# =========================
# MAIN
# =========================
def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic", action="store_true", help="Run adversaria
    parser.add_argument("--no_relational", action="store_true", help="Disable adm
    parser.add_argument("--no_weights", action="store_true", help="Disable import
    parser.add_argument("--cpu", action="store_true", help="Force CPU for XGBoost
    args = parser.parse_args()
    use_relational = not args.no_relational
    use_weights = USE_IMPORTANCE_WEIGHTS_DEFAULT and (not args.no_weights)
    logging.info(f"XGBoost version: {getattr(xgb, '__version__', 'unknown')}")
    logging.info("=== Iteration 7: Drift-Aware XGB + TF-IDF + Importance Weightin
    logging.info(f"diagnostic_only={args.diagnostic} | use_relational={use_relati
    # Load base
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    logging.info(f"Loaded train={train_df.shape}, test={test_df.shape}")
    # patients.csv
    if PATIENTS_CSV.exists():
        pat = pd.read_csv(PATIENTS_CSV)
        if "patient_id" in pat.columns:
            train_df = train_df.merge(pat, on="patient_id", how="left")
            test_df = test_df.merge(pat, on="patient_id", how="left")
            logging.info(f"Merged patients.csv: train={train_df.shape}, test={test
        else:
            logging.warning("patients.csv exists but missing patient_id; skipping
    else:
        logging.info("patients.csv not found; continuing without it.")

    train_df["is_train"] = 1
    test_df["is_train"] = 0
    df_all = pd.concat([train_df, test_df], ignore_index=True)
    # Optional relational aggregates
    if use_relational:
        df_all = add_admissions_aggs(df_all)
        df_all = add_stays_aggs(df_all)
        logging.info(f"After relational merges: df_all={df_all.shape}")
    else:
        logging.info("Relational aggregates disabled.")
    # Receipts parsing
    all_pids = df_all["patient_id"].astype(int).unique().tolist()
    all_pids.sort()
    meta_df, doc_df = build_receipt_tables(all_pids, PDF_DIR, CACHE_PATH)
    df_all = df_all.merge(meta_df, on="patient_id", how="left").merge(doc_df, on=
    df_all["codes_doc"] = df_all["codes_doc"].fillna("").astype(str)
    nonempty_docs = int((df_all["codes_doc"].str.len() > 0).sum())
    logging.info(f"codes_doc non-empty: {nonempty_docs}/{len(df_all)}")
    # Build features (transductive: fit on train+test to capture test vocab/categ
    logging.info("Building full sparse feature matrix: tabular(OHE) + TF-IDF(code
    X_all, feat_names = build_full_features(df_all)
    logging.info(f"X_all: shape={X_all.shape} | nnz={X_all.nnz:,}")
    is_train_mask = df_all["is_train"].values.astype(int) == 1
    is_test_mask = ~is_train_mask
    # Adversarial validation scores
    p_adv = adversarial_oof_proba(X_all, is_test_mask.astype(int), feat_names, di
    p_train = p_adv[is_train_mask]
    p_test = p_adv[is_test_mask]
    def q(x, qs=(0.01, 0.05, 0.50, 0.95, 0.99)):
        return {str(qq): float(np.quantile(x, qq)) for qq in qs}

    logging.info(f"Adv proba summary | TRAIN: mean={p_train.mean():.4f} qs={q(p_t
    logging.info(f"Adv proba summary | TEST : mean={p_test.mean():.4f} qs={q(p_te
    # If diagnostic-only, stop here
    if args.diagnostic:
        logging.info("Diagnostic-only mode: exiting before regression training.")
        return
    # Split matrices
    X_train = X_all[is_train_mask]
    X_test = X_all[is_test_mask]
    y = train_df["ed_cost_next3y_usd"].values.astype(np.float32)
    train_chronic = train_df["primary_chronic"].astype(str).values
    test_chronic = test_df["primary_chronic"].astype(str).values
    # Drift-aware folds
    fold_id = make_drift_folds(p_train, REG_N_FOLDS)
    for k in range(REG_N_FOLDS):
        m = (fold_id == k)
        logging.info(
            f"Fold {k}: n={int(m.sum())} | p_range=[{p_train[m].min():.4f},{p_tra
        )
    # Importance weights
    if use_weights:
        w_train = compute_importance_weights(p_train)
        logging.info(
            f"Importance weights: mean={w_train.mean():.3f} min={w_train.min():.3
            f"p50={np.quantile(w_train,0.5):.3f} p90={np.quantile(w_train,0.9):.3
        )
    else:
        w_train = np.ones_like(p_train, dtype=np.float32)
        logging.info("Importance weights disabled (all ones).")
    # XGB device
    device = "cpu" if args.cpu else "cuda"
    xgb_params = dict(
        objective="reg:absoluteerror",

        eval_metric="mae",
        tree_method="hist",
        device=device,
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=XGB_MAX_DEPTH,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE,
        reg_lambda=XGB_LAMBDA,
        min_child_weight=XGB_MIN_CHILD_WEIGHT,
        n_jobs=-1,
        verbosity=1,
        early_stopping_rounds=EARLY_STOPPING,  # put here for xgboost>=2/3 sklearn
    )
    fit_sig = None
    # CV bagging over drift folds
    oof_sum = np.zeros(len(y), dtype=np.float64)
    oof_cnt = np.zeros(len(y), dtype=np.float64)
    test_sum = np.zeros(X_test.shape[0], dtype=np.float64)
    n_models = 0
    splits = []
    for rep, seed in enumerate(REG_REPEAT_SEEDS):
        for k in range(REG_N_FOLDS):
            va_idx = np.where(fold_id == k)[0]
            tr_idx = np.where(fold_id != k)[0]
            splits.append((rep, k, seed, tr_idx, va_idx))
    logging.info(f"Total regression fits: {len(splits)} (repeats={len(REG_REPEAT_
    for rep, k, seed, tr_idx, va_idx in tqdm(splits, desc="REG folds", ncols=110)
        tag = f"rep={rep} fold={k}"
        t0 = time.time()
        X_tr = X_train[tr_idx]
        y_tr = y[tr_idx]
        X_va = X_train[va_idx]
        y_va = y[va_idx]
        sw_tr = w_train[tr_idx]

        logging.info(f"{tag} | START | train={len(tr_idx)} val={len(va_idx)} | de
        model = XGBRegressor(**xgb_params, random_state=1000 * rep + seed + k)
        if fit_sig is None:
            fit_sig = inspect.signature(model.fit).parameters
        fit_kwargs = {
            "eval_set": [(X_va, y_va)],
            "sample_weight": sw_tr,
        }
        # verbose heartbeat (xgboost sklearn API varies)
        if "verbose_eval" in fit_sig:
            fit_kwargs["verbose_eval"] = XGB_VERBOSE_EVERY
        else:
            fit_kwargs["verbose"] = XGB_VERBOSE_EVERY
        logging.info(f"{tag} | XGB fit START | verbose_every={XGB_VERBOSE_EVERY}"
        model.fit(X_tr, y_tr, **fit_kwargs)
        best_it = getattr(model, "best_iteration", None)
        logging.info(f"{tag} | XGB fit END | best_iteration={best_it}")
        p_va = model.predict(X_va)
        p_te = model.predict(X_test)
        if CLIP_AT_ZERO:
            p_va = np.maximum(p_va, 0.0)
            p_te = np.maximum(p_te, 0.0)
        oof_sum[va_idx] += p_va
        oof_cnt[va_idx] += 1.0
        test_sum += p_te
        n_models += 1
        fold_mae = mae(y_va, p_va)
        logging.info(f"{tag} | END | fold_MAE={fold_mae:.3f} | time={time.time()-
    oof = (oof_sum / np.maximum(oof_cnt, 1.0)).astype(np.float32)
    test_pred = (test_sum / max(1, n_models)).astype(np.float32)

    raw_mae_all = mae(y, oof)
    logging.info(f"OOF MAE (raw, drift-fold CV bagging): {raw_mae_all:.4f}")
    # Report MAE on most test-like subset (proxy for hidden test behavior)
    thr = float(np.quantile(p_train, CALIB_TOP_QUANTILE))
    idx_top = p_train >= thr
    raw_mae_top = mae(y[idx_top], oof[idx_top]) if idx_top.any() else raw_mae_all
    logging.info(f"OOF MAE on top-{int((1-CALIB_TOP_QUANTILE)*100)}% test-like tr
    # Calibration: shrunk median residual per chronic group
    residuals = (y.astype(np.float64) - oof.astype(np.float64)).astype(np.float64
    biases, global_med = shrunk_median_bias(residuals, train_chronic, alpha=CALIB_
    oof_cal = apply_group_bias(oof, train_chronic, biases, default_bias=global_me
    test_cal = apply_group_bias(test_pred, test_chronic, biases, default_bias=glo
    if CLIP_AT_ZERO:
        oof_cal = np.maximum(oof_cal, 0.0)
        test_cal = np.maximum(test_cal, 0.0)
    cal_mae_all = mae(y, oof_cal)
    cal_mae_top = mae(y[idx_top], oof_cal[idx_top]) if idx_top.any() else cal_mae_
    logging.info(f"Calibration biases (shrunk median residual) by chronic: {biase
    logging.info(f"OOF MAE (calibrated): all={cal_mae_all:.4f} | top-testlike={ca
    # Choose based on TOP test-like proxy (more aligned with hidden test)
    if cal_mae_top <= raw_mae_top + 1e-9:
        logging.info("Using CALIBRATED predictions (better on test-like subset)."
        final_test = test_cal
    else:
        logging.info("Using RAW predictions (calibration hurt test-like subset)."
        final_test = test_pred
    if CLIP_AT_ZERO:
        final_test = np.maximum(final_test, 0.0)
    sub = pd.DataFrame({
        "patient_id": test_df["patient_id"].astype(int).values,
        "ed_cost_next3y_usd": final_test.astype(np.float32),

What to look for in the diagnostic output
Adversarial AUC: if it’s noticeably above 0.5, you truly have drift and drift-aware
CV/weights are justified. 
p(train) vs p(test) distributions: if test probabilities are much higher, then the “top
test-like train subset” is a good proxy validation slice.
Top drift features: if you see that drift is dominated by something like zip3 ,
insurance , or a handful of TF‑IDF tokens, that helps interpret why LB differs.
If you run --diagnostic  and paste only these three lines back:
1. Adversarial OOF AUC (overall): ...
2. Adv proba summary | TRAIN ...
3. Adv proba summary | TEST ...
…I can tell you whether to (a) increase/decrease WEIGHT_BETA , (b) change fold
construction to “hold out the most test-like fold only”, or (c) drop relational aggregates
because they’re creating drift.
    })
    sub.to_csv(SUBMISSION_PATH, index=False)
    logging.info(f"Saved submission: {SUBMISSION_PATH}")
if __name__ == "__main__":
    main()
Analytics Vidhya
```

### C12 — ITERATION 7 — Drift-Aware XGBoost (GPU) + TF-IDF Codes + Importance Weighting + MA (pages 314-337)
```python
# -*- coding: utf-8 -*-
r"""
ITERATION 7 — Drift-Aware XGBoost (GPU) + TF-IDF Codes + Importance Weighting + MA
=================================================================================
Notebook-safe update:
- No argparse crash in Jupyter (ipykernel injects --f=... which breaks parse_args
- In notebooks, this file does NOT auto-run training. You call run_iteration7(...
- If run as a .py script, it supports CLI flags via parse_known_args().
Core idea:
1) Adversarial validation: classify train vs test to estimate test-likeness p(tes
2) Drift-aware folds: build validation folds by sorting train rows by p(test|x).
3) Importance weighting: weight regression training by odds p/(1-p) (stabilized).
4) MAE-aligned calibration: shrunk median residual per chronic group, selected by 
INSTALL:
pip install -U pandas numpy scipy scikit-learn xgboost pymupdf joblib tqdm
"""
from __future__ import annotations

import re
import time
import math
import sys
import argparse
import logging
import warnings
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import fitz  # PyMuPDF
from scipy.sparse import csr_matrix, hstack
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from xgboost import XGBRegressor
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# =========================
# PATHS / CONFIG
# =========================

BASE_DIR = Path(r"D:\AgentDs\agent_ds_healthcare")
TRAIN_CSV = BASE_DIR / "ed_cost_train.csv"
TEST_CSV = BASE_DIR / "ed_cost_test.csv"
PATIENTS_CSV = BASE_DIR / "patients.csv"
ADMISSIONS_TRAIN = BASE_DIR / "admissions_train.csv"
ADMISSIONS_TEST = BASE_DIR / "admissions_test.csv"
STAYS_TRAIN = BASE_DIR / "stays_train.csv"
STAYS_TEST = BASE_DIR / "stays_test.csv"
PDF_DIR = BASE_DIR / "receipts_pdf"
SUBMISSION_PATH = BASE_DIR / "submission.csv"
# Receipt cache
CACHE_PATH = BASE_DIR / "cache_receipts_iter7_codes.joblib"
CACHE_VERSION = 72  # bump if you change parsing logic
# TF-IDF (NO SVD/PCA)
TFIDF_MAX_FEATURES = 1000
TFIDF_NGRAM_RANGE = (1, 3)
# Adversarial validation
ADV_N_SPLITS = 5
ADV_RANDOM_STATE = 2027
# Drift-aware folds for regression
REG_N_FOLDS = 5
REG_REPEAT_SEEDS = [2025, 2026]
EARLY_STOPPING = 200
# Importance weighting
WEIGHT_EPS = 1e-4
WEIGHT_BETA = 0.50          # sqrt-odds to reduce variance
WEIGHT_CLIP_MIN = 0.25
WEIGHT_CLIP_MAX = 4.0
# Calibration (MAE-aligned)
CALIB_TOP_QUANTILE = 0.80   # use most test-like 20% as proxy
CALIB_ALPHA = 50.0          # shrinkage strength

# XGBoost
N_ESTIMATORS = 8000
LEARNING_RATE = 0.02
XGB_MAX_DEPTH = 6
XGB_SUBSAMPLE = 0.85
XGB_COLSAMPLE = 0.75
XGB_LAMBDA = 1.0
XGB_MIN_CHILD_WEIGHT = 1.0
# Heartbeats
XGB_VERBOSE_EVERY = 100
# Post
CLIP_AT_ZERO = True
# =========================
# LOGGING
# =========================
def setup_logging() -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname
    else:
        root.setLevel(logging.INFO)
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))
def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_out
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)
def ensure_csr_float32_int32(X: csr_matrix) -> csr_matrix:
    X = X.tocsr()

    if X.data.dtype != np.float32:
        X.data = X.data.astype(np.float32, copy=False)
    if X.indices.dtype != np.int32:
        X.indices = X.indices.astype(np.int32, copy=False)
    if X.indptr.dtype != np.int32:
        X.indptr = X.indptr.astype(np.int32, copy=False)
    return X
# =========================
# REGEX (RAW STRINGS)
# =========================
PID_RE = re.compile(r"Patient ID:\s*(\d+)")
ZIP_INS_RE = re.compile(r"ZIP3:\s*([0-9]{3})\s+Insurance:\s*([A-Za-z_]+)")
TOTAL_ANY_RE = re.compile(r"TOTAL\s+([\d,]+\.\d{2})")
CODE_TOKEN_RE = re.compile(r"\b(?:[A-Z]\d{4}|\d{5})\b")  # HCPCS or 5-digit CPT
def money_to_float(tok: str) -> float:
    return float(tok.replace(",", ""))
def parse_header_fields(text: str) -> Tuple[Optional[int], Optional[str], Optiona
    pid = None
    zip3 = None
    insurance = None
    m = PID_RE.search(text or "")
    if m:
        pid = int(m.group(1))
    m = ZIP_INS_RE.search(text or "")
    if m:
        zip3 = str(m.group(1))
        insurance = str(m.group(2)).lower()
    return pid, zip3, insurance
@dataclass

class ReceiptProfile:
    meta: Dict[str, Any]
    codes_doc: str
def extract_codes_doc_from_pdf_text(text: str, patient_id: int) -> ReceiptProfile
    _, zip3, insurance = parse_header_fields(text or "")
    codes = CODE_TOKEN_RE.findall(text or "")
    pid_str = str(patient_id)
    if len(pid_str) == 5:
        codes = [c for c in codes if c != pid_str]
    codes_doc = " ".join(codes)
    total_val = None
    m = TOTAL_ANY_RE.search(text or "")
    if m:
        try:
            total_val = money_to_float(m.group(1))
        except Exception:
            total_val = None
    meta = dict(
        pdf_ok=1 if len(codes) > 0 else 0,
        pdf_parse_notes="ok" if len(codes) > 0 else "no_codes_found",
        pdf_zip3=zip3,
        pdf_insurance=insurance,
        pdf_n_codes=int(len(codes)),
        pdf_n_unique_codes=int(len(set(codes))),
        pdf_total=float(total_val) if total_val is not None else 0.0,
    )
    return ReceiptProfile(meta=meta, codes_doc=codes_doc)
def featurize_receipt(patient_id: int, pdf_dir: Path) -> ReceiptProfile:
    pdf_path = pdf_dir / f"receipt_{patient_id}.pdf"
    empty = ReceiptProfile(
        meta=dict(

            pdf_ok=0,
            pdf_parse_notes="missing_pdf",
            pdf_zip3=None,
            pdf_insurance=None,
            pdf_n_codes=0,
            pdf_n_unique_codes=0,
            pdf_total=0.0,
        ),
        codes_doc="",
    )
    if not pdf_path.exists():
        return empty
    try:
        with fitz.open(pdf_path) as doc:
            text = ""
            for page in doc:
                text += (page.get_text("text") or "") + "\n"
    except Exception as e:
        empty.meta["pdf_parse_notes"] = f"open_error:{type(e).__name__}"
        return empty
    return extract_codes_doc_from_pdf_text(text, patient_id)
def load_cache(cache_path: Path) -> Dict[str, Any]:
    if cache_path.exists():
        try:
            obj = joblib.load(cache_path)
            if isinstance(obj, dict) and obj.get("version") == CACHE_VERSION and 
                return obj
        except Exception:
            pass
    return {"version": CACHE_VERSION, "data": {}}
def save_cache(cache_path: Path, cache_obj: Dict[str, Any]) -> None:
    try:
        joblib.dump(cache_obj, cache_path)

    except Exception as e:
        logging.warning(f"Cache save failed: {e}")
def build_receipt_tables(patient_ids: List[int], pdf_dir: Path, cache_path: Path) 
    cache_obj = load_cache(cache_path)
    data: Dict[str, Any] = cache_obj.get("data", {})
    to_parse = [pid for pid in patient_ids if str(pid) not in data]
    logging.info(f"Receipt cache: {len(patient_ids) - len(to_parse):,} hit | {len
    for pid in tqdm(to_parse, desc="Parsing receipts", ncols=100):
        prof = featurize_receipt(pid, pdf_dir)
        data[str(pid)] = {"meta": prof.meta, "codes_doc": prof.codes_doc}
    cache_obj["data"] = data
    if to_parse:
        save_cache(cache_path, cache_obj)
    meta_rows = []
    doc_rows = []
    for pid in patient_ids:
        obj = data.get(str(pid))
        if obj is None:
            prof = featurize_receipt(pid, pdf_dir)
            obj = {"meta": prof.meta, "codes_doc": prof.codes_doc}
        meta_rows.append({"patient_id": pid, **(obj.get("meta", {}) or {})})
        doc_rows.append({"patient_id": pid, "codes_doc": obj.get("codes_doc", "") 
    return pd.DataFrame(meta_rows), pd.DataFrame(doc_rows)
# =========================
# OPTIONAL RELATIONAL AGGS
# =========================
def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)

    except Exception as e:
        logging.warning(f"Failed reading {path}: {e}")
        return None
def add_admissions_aggs(df_all: pd.DataFrame) -> pd.DataFrame:
    tr = safe_read_csv(ADMISSIONS_TRAIN)
    te = safe_read_csv(ADMISSIONS_TEST)
    if tr is None or te is None:
        logging.info("Admissions files not found. Skipping admissions aggregates.
        return df_all
    adm = pd.concat([tr, te], ignore_index=True)
    if "readmit_30d" in adm.columns:
        adm = adm.drop(columns=["readmit_30d"])
    if "patient_id" not in adm.columns:
        logging.info("Admissions table missing patient_id. Skipping.")
        return df_all
    logging.info(f"Admissions rows: {adm.shape}")
    num_cols = [c for c in ["los_days", "acuity_emergent", "charlson_band", "ed_v
    agg_dict = {}
    agg_dict["adm_n"] = ("admission_id", "count") if "admission_id" in adm.column
    for c in num_cols:
        agg_dict[f"adm_{c}_mean"] = (c, "mean")
        agg_dict[f"adm_{c}_max"] = (c, "max")
    agg = adm.groupby("patient_id").agg(**agg_dict).reset_index()
    if "primary_dx" in adm.columns:
        dx_ct = pd.crosstab(adm["patient_id"], adm["primary_dx"]).reset_index()
        dx_ct.columns = ["patient_id"] + [f"adm_primary_dx_{str(c)}_count" for c 
        agg = agg.merge(dx_ct, on="patient_id", how="left")
    if "discharge_weekday" in adm.columns:
        wd_ct = pd.crosstab(adm["patient_id"], adm["discharge_weekday"]).reset_in
        wd_ct.columns = ["patient_id"] + [f"adm_discharge_wd_{str(c)}_count" for 

        agg = agg.merge(wd_ct, on="patient_id", how="left")
    logging.info(f"Admissions aggregates shape: {agg.shape}")
    return df_all.merge(agg, on="patient_id", how="left")
def add_stays_aggs(df_all: pd.DataFrame) -> pd.DataFrame:
    tr = safe_read_csv(STAYS_TRAIN)
    te = safe_read_csv(STAYS_TEST)
    if tr is None or te is None:
        logging.info("Stays files not found. Skipping stays aggregates.")
        return df_all
    st = pd.concat([tr, te], ignore_index=True)
    if "discharge_ready_day11" in st.columns:
        st = st.drop(columns=["discharge_ready_day11"])
    if "patient_id" not in st.columns:
        logging.info("Stays table missing patient_id. Skipping.")
        return df_all
    logging.info(f"Stays rows: {st.shape}")
    agg = st.groupby("patient_id").agg(stay_n=("stay_id", "count") if "stay_id" i
    if "unit_type" in st.columns:
        ut = pd.crosstab(st["patient_id"], st["unit_type"]).reset_index()
        ut.columns = ["patient_id"] + [f"stay_unit_{str(c)}_count" for c in ut.co
        agg = agg.merge(ut, on="patient_id", how="left")
    if "admission_reason" in st.columns:
        ar = pd.crosstab(st["patient_id"], st["admission_reason"]).reset_index()
        ar.columns = ["patient_id"] + [f"stay_reason_{str(c)}_count" for c in ar.
        agg = agg.merge(ar, on="patient_id", how="left")
    logging.info(f"Stays aggregates shape: {agg.shape}")
    return df_all.merge(agg, on="patient_id", how="left")
# =========================

# FEATURE BUILDING
# =========================
def build_tabular_sparse(df_all: pd.DataFrame) -> Tuple[csr_matrix, Optional[np.n
    df = df_all.copy()
    if "prior_ed_cost_5y_usd" in df.columns and "prior_ed_visits_5y" in df.column
        df["prior_cost_per_visit"] = df["prior_ed_cost_5y_usd"] / (df["prior_ed_v
        df["log_prior_cost"] = np.log1p(df["prior_ed_cost_5y_usd"].astype(float))
        df["log_prior_visits"] = np.log1p(df["prior_ed_visits_5y"].astype(float))
    if "pdf_total" in df.columns and "prior_ed_cost_5y_usd" in df.columns:
        df["pdf_total_minus_prior_cost"] = df["pdf_total"].astype(float) - df["pr
        df["pdf_total_absdiff_prior_cost"] = np.abs(df["pdf_total_minus_prior_cos
    cat_candidates = ["primary_chronic", "sex", "insurance", "zip3", "pdf_insuran
    for c in cat_candidates:
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("missing")
    exclude = {"patient_id", "ed_cost_next3y_usd", "is_train", "codes_doc"}
    cat_cols = [c for c in cat_candidates if c in df.columns]
    num_cols = [c for c in df.columns if c not in exclude and c not in cat_cols a
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), nu
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")
                              ("ohe", make_ohe())]), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    X = pre.fit_transform(df)
    X = X.tocsr() if hasattr(X, "tocsr") else csr_matrix(X)
    X = ensure_csr_float32_int32(X)
    feat_names = None
    try:
        feat_names = pre.get_feature_names_out()

    except Exception:
        pass
    return X, feat_names
def build_tfidf_sparse(codes_doc: np.ndarray) -> Tuple[csr_matrix, Optional[np.nd
    nonempty = int(np.sum(pd.Series(codes_doc).astype(str).str.len().values > 0))
    if nonempty == 0:
        logging.warning("No non-empty codes_doc found. TF-IDF disabled.")
        X = csr_matrix((len(codes_doc), 0), dtype=np.float32)
        return X, None
    vec = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b(?:[A-Z]\d{4}|\d{5})\b",
        lowercase=False,
        ngram_range=TFIDF_NGRAM_RANGE,
        max_features=TFIDF_MAX_FEATURES,
        min_df=1,
        max_df=1.0,
        sublinear_tf=True,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        dtype=np.float32,
    )
    X = vec.fit_transform(codes_doc.astype(str))
    X = ensure_csr_float32_int32(X)
    names = vec.get_feature_names_out()
    return X, names
def build_full_features(df_all: pd.DataFrame) -> Tuple[csr_matrix, Optional[np.nd
    X_tab, n_tab = build_tabular_sparse(df_all)
    X_txt, n_txt = build_tfidf_sparse(df_all["codes_doc"].values.astype(str))
    X_all = hstack([X_tab, X_txt], format="csr")
    X_all = ensure_csr_float32_int32(X_all)
    if (n_tab is not None) and (n_txt is not None):

        feat_names = np.concatenate([n_tab, n_txt])
    else:
        feat_names = None
    return X_all, feat_names
# =========================
# ADVERSARIAL VALIDATION
# =========================
def adversarial_oof_proba(X_all: csr_matrix, is_test: np.ndarray, diagnostic: boo
    y_adv = is_test.astype(int)
    skf = StratifiedKFold(n_splits=ADV_N_SPLITS, shuffle=True, random_state=ADV_RA
    p = np.zeros(X_all.shape[0], dtype=np.float32)
    fold_aucs = []
    logging.info("Adversarial validation: LogisticRegression(train=0 vs test=1)..
    for k, (tr, va) in enumerate(skf.split(np.zeros(len(y_adv)), y_adv)):
        t0 = time.time()
        clf = LogisticRegression(
            solver="liblinear",
            C=1.0,
            class_weight="balanced",
            max_iter=200,
        )
        clf.fit(X_all[tr], y_adv[tr])
        p_va = clf.predict_proba(X_all[va])[:, 1].astype(np.float32)
        p[va] = p_va
        auc = roc_auc_score(y_adv[va], p_va)
        fold_aucs.append(float(auc))
        logging.info(f"ADV fold {k} | AUC={auc:.4f} | time={time.time()-t0:.1f}s"
    auc_all = float(roc_auc_score(y_adv, p))
    logging.info(f"Adversarial OOF AUC: {auc_all:.4f} | fold_AUCs={fold_aucs}")
    return p, auc_all
def make_drift_folds(p_train: np.ndarray, n_folds: int) -> np.ndarray:
    n = len(p_train)
    order = np.argsort(p_train)  # low -> high test-likeness

    fold_id = np.empty(n, dtype=np.int32)
    bins = (np.arange(n) * n_folds) // n
    fold_id[order] = np.clip(bins, 0, n_folds - 1).astype(np.int32)
    return fold_id
def compute_importance_weights(p_train: np.ndarray) -> np.ndarray:
    p = np.clip(p_train.astype(np.float64), WEIGHT_EPS, 1.0 - WEIGHT_EPS)
    odds = p / (1.0 - p)
    w = np.power(odds, WEIGHT_BETA)
    w = np.clip(w, WEIGHT_CLIP_MIN, WEIGHT_CLIP_MAX)
    w = w / np.mean(w)
    return w.astype(np.float32)
# =========================
# MAE CALIBRATION (SHRUNK MEDIAN RESIDUAL)
# =========================
def shrunk_median_bias(residuals: np.ndarray, groups: np.ndarray, alpha: float) -
    g = groups.astype(str)
    global_med = float(np.median(residuals))
    biases: Dict[str, float] = {}
    for key in np.unique(g):
        m = (g == key)
        rg = residuals[m]
        if rg.size == 0:
            continue
        med_g = float(np.median(rg))
        n = float(rg.size)
        b = (n * med_g + alpha * global_med) / (n + alpha)
        biases[str(key)] = float(b)
    return biases, global_med
def apply_group_bias(pred: np.ndarray, groups: np.ndarray, biases: Dict[str, float
    out = pred.astype(np.float64).copy()
    g = groups.astype(str)
    for i in range(len(out)):
        out[i] += biases.get(str(g[i]), default_bias)
    return out.astype(np.float32)

# =========================
# DRIVER (NOTEBOOK FRIENDLY)
# =========================
def run_iteration7(
    diagnostic: bool = False,
    use_relational: bool = True,
    use_weights: bool = True,
    force_cpu: bool = False,
) -> Dict[str, Any]:
    """
    Notebook-friendly entrypoint.
    Examples:
      run_iteration7(diagnostic=True)
      run_iteration7()
      run_iteration7(use_weights=False)
      run_iteration7(use_relational=False)
    """
    setup_logging()
    logging.info(f"XGBoost version: {getattr(xgb, '__version__', 'unknown')}")
    logging.info("=== Iteration 7 (Notebook-safe) ===")
    logging.info(f"diagnostic={diagnostic} | use_relational={use_relational} | us
    # Load base
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    logging.info(f"Loaded train={train_df.shape}, test={test_df.shape}")
    # patients.csv
    if PATIENTS_CSV.exists():
        pat = pd.read_csv(PATIENTS_CSV)
        if "patient_id" in pat.columns:
            train_df = train_df.merge(pat, on="patient_id", how="left")
            test_df = test_df.merge(pat, on="patient_id", how="left")
            logging.info(f"Merged patients.csv: train={train_df.shape}, test={test
    train_df["is_train"] = 1

    test_df["is_train"] = 0
    df_all = pd.concat([train_df, test_df], ignore_index=True)
    # Optional relational aggregates
    if use_relational:
        df_all = add_admissions_aggs(df_all)
        df_all = add_stays_aggs(df_all)
        logging.info(f"After relational merges: df_all={df_all.shape}")
    # Receipts parsing
    all_pids = df_all["patient_id"].astype(int).unique().tolist()
    all_pids.sort()
    meta_df, doc_df = build_receipt_tables(all_pids, PDF_DIR, CACHE_PATH)
    df_all = df_all.merge(meta_df, on="patient_id", how="left").merge(doc_df, on=
    df_all["codes_doc"] = df_all["codes_doc"].fillna("").astype(str)
    nonempty_docs = int((df_all["codes_doc"].str.len() > 0).sum())
    logging.info(f"codes_doc non-empty: {nonempty_docs}/{len(df_all)}")
    # Feature build
    logging.info("Building full sparse feature matrix: tabular(OHE) + TF-IDF(code
    X_all, feat_names = build_full_features(df_all)
    logging.info(f"X_all: shape={X_all.shape} | nnz={X_all.nnz:,}")
    is_train_mask = (df_all["is_train"].values.astype(int) == 1)
    is_test_mask = ~is_train_mask
    # Adversarial validation scores
    p_adv, adv_auc = adversarial_oof_proba(X_all, is_test_mask.astype(int), diagn
    p_train = p_adv[is_train_mask]
    p_test = p_adv[is_test_mask]
    def q(x, qs=(0.01, 0.05, 0.50, 0.95, 0.99)):
        return {str(qq): float(np.quantile(x, qq)) for qq in qs}
    logging.info(f"Adv proba TRAIN | mean={p_train.mean():.4f} qs={q(p_train)}")
    logging.info(f"Adv proba TEST  | mean={p_test.mean():.4f} qs={q(p_test)}")
    out = {
        "adv_auc": adv_auc,

        "p_train_mean": float(p_train.mean()),
        "p_test_mean": float(p_test.mean()),
    }
    if diagnostic:
        logging.info("Diagnostic mode: stopping after adversarial validation.")
        return out
    # Regression data
    X_train = X_all[is_train_mask]
    X_test = X_all[is_test_mask]
    y = train_df["ed_cost_next3y_usd"].values.astype(np.float32)
    train_chronic = train_df["primary_chronic"].astype(str).values
    test_chronic = test_df["primary_chronic"].astype(str).values
    # Drift-aware folds
    fold_id = make_drift_folds(p_train, REG_N_FOLDS)
    for k in range(REG_N_FOLDS):
        m = (fold_id == k)
        logging.info(f"Fold {k}: n={int(m.sum())} | p_mean={p_train[m].mean():.4f
    # Weights
    if use_weights:
        w_train = compute_importance_weights(p_train)
        logging.info(f"Weights: mean={w_train.mean():.3f} min={w_train.min():.3f} 
    else:
        w_train = np.ones_like(p_train, dtype=np.float32)
        logging.info("Weights disabled.")
    device = "cpu" if force_cpu else "cuda"
    xgb_params = dict(
        objective="reg:absoluteerror",
        eval_metric="mae",
        tree_method="hist",
        device=device,
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=XGB_MAX_DEPTH,

        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE,
        reg_lambda=XGB_LAMBDA,
        min_child_weight=XGB_MIN_CHILD_WEIGHT,
        n_jobs=-1,
        verbosity=1,
        early_stopping_rounds=EARLY_STOPPING,
    )
    fit_sig = None
    # Bagging over drift folds and repeats
    oof_sum = np.zeros(len(y), dtype=np.float64)
    oof_cnt = np.zeros(len(y), dtype=np.float64)
    test_sum = np.zeros(X_test.shape[0], dtype=np.float64)
    n_models = 0
    splits = []
    for rep, seed in enumerate(REG_REPEAT_SEEDS):
        for k in range(REG_N_FOLDS):
            va_idx = np.where(fold_id == k)[0]
            tr_idx = np.where(fold_id != k)[0]
            splits.append((rep, k, seed, tr_idx, va_idx))
    logging.info(f"Total regression fits: {len(splits)}")
    for rep, k, seed, tr_idx, va_idx in tqdm(splits, desc="REG folds", ncols=110)
        tag = f"rep={rep} fold={k}"
        t0 = time.time()
        X_tr = X_train[tr_idx]
        y_tr = y[tr_idx]
        X_va = X_train[va_idx]
        y_va = y[va_idx]
        sw_tr = w_train[tr_idx]
        logging.info(f"{tag} | START | train={len(tr_idx)} val={len(va_idx)} | de
        model = XGBRegressor(**xgb_params, random_state=1000 * rep + seed + k)
        if fit_sig is None:

            fit_sig = inspect.signature(model.fit).parameters
        fit_kwargs = {
            "eval_set": [(X_va, y_va)],
            "sample_weight": sw_tr,
        }
        if "verbose_eval" in fit_sig:
            fit_kwargs["verbose_eval"] = XGB_VERBOSE_EVERY
        else:
            fit_kwargs["verbose"] = XGB_VERBOSE_EVERY
        logging.info(f"{tag} | XGB fit START | verbose_every={XGB_VERBOSE_EVERY}"
        model.fit(X_tr, y_tr, **fit_kwargs)
        best_it = getattr(model, "best_iteration", None)
        logging.info(f"{tag} | XGB fit END | best_iteration={best_it}")
        p_va = model.predict(X_va)
        p_te = model.predict(X_test)
        if CLIP_AT_ZERO:
            p_va = np.maximum(p_va, 0.0)
            p_te = np.maximum(p_te, 0.0)
        oof_sum[va_idx] += p_va
        oof_cnt[va_idx] += 1.0
        test_sum += p_te
        n_models += 1
        fold_mae = mae(y_va, p_va)
        logging.info(f"{tag} | END | fold_MAE={fold_mae:.3f} | time={time.time()-
    oof = (oof_sum / np.maximum(oof_cnt, 1.0)).astype(np.float32)
    test_pred = (test_sum / max(1, n_models)).astype(np.float32)
    raw_mae_all = mae(y, oof)
    logging.info(f"OOF MAE (raw, drift-fold): {raw_mae_all:.4f}")
    thr = float(np.quantile(p_train, CALIB_TOP_QUANTILE))
    idx_top = (p_train >= thr)
    raw_mae_top = mae(y[idx_top], oof[idx_top]) if idx_top.any() else raw_mae_all

    logging.info(f"OOF MAE on top-testlike (p>={thr:.4f}): {raw_mae_top:.4f}")
    # Calibration (shrunk median residual)
    residuals = (y.astype(np.float64) - oof.astype(np.float64)).astype(np.float64
    biases, global_med = shrunk_median_bias(residuals, train_chronic, alpha=CALIB_
    oof_cal = apply_group_bias(oof, train_chronic, biases, default_bias=global_me
    test_cal = apply_group_bias(test_pred, test_chronic, biases, default_bias=glo
    if CLIP_AT_ZERO:
        oof_cal = np.maximum(oof_cal, 0.0)
        test_cal = np.maximum(test_cal, 0.0)
    cal_mae_all = mae(y, oof_cal)
    cal_mae_top = mae(y[idx_top], oof_cal[idx_top]) if idx_top.any() else cal_mae_
    logging.info(f"Calibration biases (shrunk median) by chronic: {biases} | glob
    logging.info(f"OOF MAE (calibrated): all={cal_mae_all:.4f} | top-testlike={ca
    # Choose based on top-testlike proxy
    if cal_mae_top <= raw_mae_top + 1e-9:
        logging.info("Using CALIBRATED predictions (better on test-like subset)."
        final_test = test_cal
        chosen = "calibrated"
    else:
        logging.info("Using RAW predictions (calibration hurt test-like subset)."
        final_test = test_pred
        chosen = "raw"
    if CLIP_AT_ZERO:
        final_test = np.maximum(final_test, 0.0)
    sub = pd.DataFrame({
        "patient_id": test_df["patient_id"].astype(int).values,
        "ed_cost_next3y_usd": final_test.astype(np.float32),
    })
    sub.to_csv(SUBMISSION_PATH, index=False)
    logging.info(f"Saved submission: {SUBMISSION_PATH}")
    out.update({

        "oof_mae_raw_all": float(raw_mae_all),
        "oof_mae_raw_top": float(raw_mae_top),
        "oof_mae_cal_all": float(cal_mae_all),
        "oof_mae_cal_top": float(cal_mae_top),
        "chosen": chosen,
        "submission_path": str(SUBMISSION_PATH),
    })
    return out
# =========================
# CLI SUPPORT (SAFE IN NOTEBOOK TOO)
# =========================
def parse_cli_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--no_relational", action="store_true")
    parser.add_argument("--no_weights", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    # parse_known_args avoids crashing on ipykernel's extra args (e.g., --f=...js
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        logging.info(f"Ignoring unknown args: {unknown}")
    return args
def _in_notebook() -> bool:
    # robust notebook check
    try:
        import IPython
        ip = IPython.get_ipython()
        if ip is None:
            return False
        return ("IPKernelApp" in ip.config) or ("ipykernel" in sys.modules)
    except Exception:
        return ("ipykernel" in sys.modules)
if __name__ == "__main__":
    setup_logging()

Why this fixes your current notebook error
We don’t require CLI args in notebooks; you call the function.
Even if you do run as a script, we use parse_known_args()  which ignores ipykernel’s
injected args instead of crashing. 
If you run run_iteration7(diagnostic=True)  and paste:
Adversarial OOF AUC: ...
Adv proba TRAIN mean/quantiles
Adv proba TEST mean/quantiles
…I can tell you whether weighting should be stronger/weaker (tune WEIGHT_BETA ) and
whether relational aggregates are helping or increasing drift.
    if _in_notebook():
        logging.info("Notebook detected: not auto-running. Call run_iteration7(di
    else:
        a = parse_cli_args()
        run_iteration7(
            diagnostic=a.diagnostic,
            use_relational=not a.no_relational,
            use_weights=not a.no_weights,
            force_cpu=a.cpu,
        )
Stack Overflow +1
```

### C13 — ITERATION 8 — Multi-Source Patient Features (Receipts + Notes + Vitals) + XGBoost (pages 338-367)
```python
# -*- coding: utf-8 -*-
r"""
ITERATION 8 — Multi-Source Patient Features (Receipts + Notes + Vitals) + XGBoost 
=================================================================================
:contentReference[oaicite:7]{index=7}hould beat the stuck 478~480 plateau):
1) Your adversarial AUC ~0.51 => train/test covariate shift is basically absent.
   So "drift folds" and importance weighting are the wrong framework.
2) CV looks great but leaderboard/test gets worse — classic symptom of preprocess
   (vectorizers/encoders fitted globally) or overly-rich n-grams overfitting on s
   This script makes CV leakage-safe by fitting ALL encoders/vectorizers strictly 
3) Add new modalities that exist in this dataset package (joinable via patient_id
   admissions + discharge notes + stays + vitals time series notes (auto-skip if 

4) Receipts: not "codes only" — parse the table structure and inject Description w
   This learns code similarity statistically without manual clinical grouping.
MODEL:
- Two XGBoost models (GPU):
  (A) reg:squarederror (often generalizes better even when metric is MAE)
  (B) reg:quantileerror with quantile_alpha=0.5 (median reg; MAE-aligned)
- Blend weight optimized on leakage-safe OOF MAE.
- Optional MAE-aligned calibration (global median residual / shrunk median residu
USAGE (Jupyter):
    res = run_iteration8(diagnostic=True)      # check coverage of extra modaliti
    res = run_iteration8(diagnostic=False)     # train + write submission.csv
INSTALL:
pip install -U pandas numpy scipy scikit-learn xgboost pymupdf joblib tqdm
"""
from __future__ import annotations
import re
import sys
import json
import time
import argparse
import logging
import warnings
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import fitz  # PyMuPDF
from scipy.sparse import csr_matrix, hstack

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
from xgboost import XGBRegressor
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# =========================
# PATHS (edit if needed)
# =========================
BASE_DIR = Path(r"D:\AgentDs\agent_ds_healthcare")
TRAIN_CSV = BASE_DIR / "ed_cost_train.csv"
TEST_CSV = BASE_DIR / "ed_cost_test.csv"
PATIENTS_CSV = BASE_DIR / "patients.csv"
ADMISSIONS_TRAIN = BASE_DIR / "admissions_train.csv"
ADMISSIONS_TEST = BASE_DIR / "admissions_test.csv"
DISCHARGE_NOTES_JSON = BASE_DIR / "discharge_notes.json"
STAYS_TRAIN = BASE_DIR / "stays_train.csv"
STAYS_TEST = BASE_DIR / "stays_test.csv"
VITALS_JSON = BASE_DIR / "vitals_timeseries.json"
PDF_DIR = BASE_DIR / "receipts_pdf"
SUBMISSION_PATH = BASE_DIR / "submission.csv"
CACHE_DIR = BASE_DIR / "cache_iter8"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RECEIPT_CACHE = CACHE_DIR / "receipts.joblib"
DISCHARGE_CACHE = CACHE_DIR / "discharge_docs.joblib"

VITALS_CACHE = CACHE_DIR / "vitals_features.joblib"
RELATIONAL_CACHE = CACHE_DIR / "relational_aggs.joblib"
CACHE_VERSION = 82  # bump if you change parsing logic
# =========================
# GLOBAL CONFIG
# =========================
# Text feature caps (keep bounded to avoid overfitting)
MAXF_RECEIPT = 4000
MAXF_DISCHARGE = 2500
MAXF_VITALS_NOTE = 2500
# For small datasets: filter 1-off tokens to reduce noise
MIN_DF_RECEIPT = 2
MIN_DF_NOTES = 2
# CV
N_FOLDS = 5
REPEAT_SEEDS = [2025, 2026]   # 10 fits total
EARLY_STOPPING = 200
# XGB
N_ESTIMATORS = 8000
LR = 0.02
MAX_DEPTH = 6
SUBSAMPLE = 0.85
COLSAMPLE = 0.75
REG_LAMBDA = 1.0
MIN_CHILD_WEIGHT = 1.0
# Objectives to try
USE_L2 = True
USE_Q50 = True
# Blend grid
BLEND_GRID_STEP = 0.02
# Calibration

USE_CALIBRATION = True
CALIB_ALPHA = 50.0  # shrinkage for group median residual
CLIP_AT_ZERO = True
# Heartbeats
VERBOSE_EVERY = 100
# =========================
# LOGGING
# =========================
def setup_logging() -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname
    else:
        root.setLevel(logging.INFO)
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))
def ensure_csr_float32_int32(X: csr_matrix) -> csr_matrix:
    X = X.tocsr()
    if X.data.dtype != np.float32:
        X.data = X.data.astype(np.float32, copy=False)
    if X.indices.dtype != np.int32:
        X.indices = X.indices.astype(np.int32, copy=False)
    if X.indptr.dtype != np.int32:
        X.indptr = X.indptr.astype(np.int32, copy=False)
    return X
def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_out
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)

# =========================
# RECEIPT PARSING (table-structure aware)
# =========================
ZIP_INS_RE = re.compile(r"ZIP3:\s*([0-9]{3})\s+Insurance:\s*([A-Za-z_]+)")
TOTAL_RE = re.compile(r"TOTAL\s+([\d,]+\.\d{2})")
CODE_FULL_RE = re.compile(r"^(?:[A-Z]\d{4}|\d{5})$")
MONEY_RE = re.compile(r"^[\d,]+\.\d{2}$")
def money_to_float(s: str) -> float:
    return float(s.replace(",", ""))
@dataclass
class ReceiptProfile:
    meta: Dict[str, Any]
    receipt_doc: str
def parse_receipt_table(lines: List[str]) -> List[Tuple[str, int, float, float, s
    """
    Parse PyMuPDF-extracted receipt table lines.
    Many PDFs come out "columnar": code, description, qty, unit, line_total each 
    This parser reconstructs line items by:
      code_line -> (one or more desc lines) -> qty_line -> unit_line -> line_tota
    """
    items: List[Tuple[str, int, float, float, str]] = []
    i = 0
    n = len(lines)
    while i < n:
        s = lines[i].strip()
        if CODE_FULL_RE.match(s):
            code = s
            j = i + 1
            desc_parts: List[str] = []

            # accumulate description (can be multi-line) until we hit qty
            while j < n:
                sj = lines[j].strip()
                if sj.isdigit():
                    break
                if sj.startswith("TOTAL"):
                    break
                # guard against header lines repeating
                if sj in {"Code", "Description", "Qty", "Unit ($)", "Line Total (
                    j += 1
                    continue
                if sj:
                    desc_parts.append(sj)
                j += 1
            # need qty + unit + line
            if j + 2 < n:
                qty_s = lines[j].strip()
                unit_s = lines[j + 1].strip()
                line_s = lines[j + 2].strip()
                if qty_s.isdigit() and MONEY_RE.match(unit_s) and MONEY_RE.match(
                    qty = int(qty_s)
                    qty = max(1, min(qty, 50))
                    unit = money_to_float(unit_s)
                    line_total = money_to_float(line_s)
                    desc = " ".join(desc_parts)
                    items.append((code, qty, unit, line_total, desc))
                    i = j + 3
                    continue
        i += 1
    return items
def featurize_receipt(patient_id: int, pdf_dir: Path) -> ReceiptProfile:
    pdf_path = pdf_dir / f"receipt_{patient_id}.pdf"
    empty = ReceiptProfile(
        meta=dict(
            pdf_ok=0,

            pdf_zip3=None,
            pdf_insurance=None,
            pdf_n_items=0,
            pdf_n_codes=0,
            pdf_n_unique_codes=0,
            pdf_qty_sum=0,
            pdf_unit_mean=np.nan,
            pdf_unit_max=np.nan,
            pdf_line_mean=np.nan,
            pdf_line_max=np.nan,
            pdf_cost_concentration=np.nan,  # max(line_total)/total
            pdf_total=0.0,
        ),
        receipt_doc="",
    )
    if not pdf_path.exists():
        return empty
    try:
        with fitz.open(pdf_path) as doc:
            text = ""
            for page in doc:
                text += (page.get_text("text") or "") + "\n"
    except Exception:
        return empty
    text = text or ""
    lines = text.splitlines()
    # Header info (zip/insurance)
    zip3 = None
    ins = None
    m = ZIP_INS_RE.search(text)
    if m:
        zip3 = str(m.group(1))
        ins = str(m.group(2)).lower()
    # Total (should match prior cost)
    total_val = 0.0

    m = TOTAL_RE.search(text)
    if m:
        try:
            total_val = money_to_float(m.group(1))
        except Exception:
            total_val = 0.0
    items = parse_receipt_table(lines)
    # Build receipt_doc = code + description words (repeat by qty to encode inten
    tokens: List[str] = []
    units = []
    lines_total = []
    qty_sum = 0
    for code, qty, unit, line_total, desc in items:
        qty_sum += qty
        units.append(unit)
        lines_total.append(line_total)
        # repeat by qty, but cap repetition to avoid huge docs
        rep = max(1, min(qty, 10))
        for _ in range(rep):
            if desc:
                tokens.append(f"{code} {desc}")
            else:
                tokens.append(f"{code}")
    receipt_doc = " ".join(tokens)
    # If parsing failed, fallback to codes only (still useful)
    if len(items) == 0:
        code_tokens = re.findall(r"\b(?:[A-Z]\d{4}|\d{5})\b", text)
        receipt_doc = " ".join(code_tokens)
    n_codes = len(re.findall(r"\b(?:[A-Z]\d{4}|\d{5})\b", receipt_doc))
    uniq_codes = len(set(re.findall(r"\b(?:[A-Z]\d{4}|\d{5})\b", receipt_doc)))
    unit_mean = float(np.mean(units)) if units else np.nan
    unit_max = float(np.max(units)) if units else np.nan
    line_mean = float(np.mean(lines_total)) if lines_total else np.nan

    line_max = float(np.max(lines_total)) if lines_total else np.nan
    conc = float(line_max / total_val) if (lines_total and total_val > 0) else np
    meta = dict(
        pdf_ok=1 if len(receipt_doc) > 0 else 0,
        pdf_zip3=zip3,
        pdf_insurance=ins,
        pdf_n_items=int(len(items)),
        pdf_n_codes=int(n_codes),
        pdf_n_unique_codes=int(uniq_codes),
        pdf_qty_sum=int(qty_sum),
        pdf_unit_mean=unit_mean,
        pdf_unit_max=unit_max,
        pdf_line_mean=line_mean,
        pdf_line_max=line_max,
        pdf_cost_concentration=conc,
        pdf_total=float(total_val),
    )
    return ReceiptProfile(meta=meta, receipt_doc=receipt_doc)
def load_or_build_receipts(patient_ids: List[int]) -> pd.DataFrame:
    cache = {"version": CACHE_VERSION, "data": {}}
    if RECEIPT_CACHE.exists():
        try:
            cache = joblib.load(RECEIPT_CACHE)
        except Exception:
            cache = {"version": CACHE_VERSION, "data": {}}
    if not isinstance(cache, dict) or cache.get("version") != CACHE_VERSION:
        cache = {"version": CACHE_VERSION, "data": {}}
    data = cache.get("data", {})
    missing = [pid for pid in patient_ids if str(pid) not in data]
    logging.info(f"Receipts cache: hit={len(patient_ids)-len(missing):,} miss={le
    for pid in tqdm(missing, desc="Parsing receipts", ncols=100):
        prof = featurize_receipt(pid, PDF_DIR)
        data[str(pid)] = {"meta": prof.meta, "receipt_doc": prof.receipt_doc}

    cache["data"] = data
    if missing:
        try:
            joblib.dump(cache, RECEIPT_CACHE)
        except Exception as e:
            logging.warning(f"Receipt cache save failed: {e}")
    rows = []
    for pid in patient_ids:
        obj = data.get(str(pid), {"meta": {}, "receipt_doc": ""})
        row = {"patient_id": pid, "receipt_doc": obj.get("receipt_doc", "") or ""
        row.update(obj.get("meta") or {})
        rows.append(row)
    return pd.DataFrame(rows)
# =========================
# DISCHARGE NOTES -> patient docs
# =========================
def load_discharge_docs() -> pd.DataFrame:
    if DISCHARGE_CACHE.exists():
        try:
            obj = joblib.load(DISCHARGE_CACHE)
            if isinstance(obj, dict) and obj.get("version") == CACHE_VERSION and 
                return obj["df"]
        except Exception:
            pass
    if (not ADMISSIONS_TRAIN.exists()) or (not ADMISSIONS_TEST.exists()) or (not 
        logging.info("Discharge notes files missing. Skipping discharge_doc.")
        df = pd.DataFrame({"patient_id": [], "discharge_doc": [], "discharge_n_not
        joblib.dump({"version": CACHE_VERSION, "df": df}, DISCHARGE_CACHE)
        return df
    adm = pd.concat([pd.read_csv(ADMISSIONS_TRAIN), pd.read_csv(ADMISSIONS_TEST)]
    if "readmit_30d" in adm.columns:
        adm = adm.drop(columns=["readmit_30d"])

    if "admission_id" not in adm.columns or "patient_id" not in adm.columns:
        logging.info("Admissions schema unexpected. Skipping discharge_doc.")
        df = pd.DataFrame({"patient_id": [], "discharge_doc": [], "discharge_n_not
        joblib.dump({"version": CACHE_VERSION, "df": df}, DISCHARGE_CACHE)
        return df
    with open(DISCHARGE_NOTES_JSON, "r", encoding="utf-8") as f:
        notes_list = json.load(f)
    notes = pd.DataFrame(notes_list)
    if "admission_id" not in notes.columns or "note" not in notes.columns:
        logging.info("discharge_notes.json schema unexpected. Skipping discharge_
        df = pd.DataFrame({"patient_id": [], "discharge_doc": [], "discharge_n_not
        joblib.dump({"version": CACHE_VERSION, "df": df}, DISCHARGE_CACHE)
        return df
    merged = notes.merge(adm[["admission_id", "patient_id"]], on="admission_id", 
    merged["note"] = merged["note"].astype(str).fillna("")
    agg = merged.groupby("patient_id").agg(
        discharge_doc=("note", lambda s: " ".join(s.tolist())),
        discharge_n_notes=("note", "size"),
    ).reset_index()
    joblib.dump({"version": CACHE_VERSION, "df": agg}, DISCHARGE_CACHE)
    logging.info(f"Built discharge docs: {agg.shape}")
    return agg
# =========================
# VITALS -> numeric stats + patient doc
# =========================
def load_vitals_features() -> pd.DataFrame:
    if VITALS_CACHE.exists():
        try:
            obj = joblib.load(VITALS_CACHE)
            if isinstance(obj, dict) and obj.get("version") == CACHE_VERSION and 
                return obj["df"]
        except Exception:
            pass

    if (not STAYS_TRAIN.exists()) or (not STAYS_TEST.exists()) or (not VITALS_JSO
        logging.info("Vitals files missing. Skipping vitals features.")
        df = pd.DataFrame({"patient_id": [], "vitals_doc": [], "vitals_n_stays": 
        joblib.dump({"version": CACHE_VERSION, "df": df}, VITALS_CACHE)
        return df
    st = pd.concat([pd.read_csv(STAYS_TRAIN), pd.read_csv(STAYS_TEST)], ignore_in
    if "discharge_ready_day11" in st.columns:
        st = st.drop(columns=["discharge_ready_day11"])
    if "stay_id" not in st.columns or "patient_id" not in st.columns:
        logging.info("Stays schema unexpected. Skipping vitals.")
        df = pd.DataFrame({"patient_id": [], "vitals_doc": [], "vitals_n_stays": 
        joblib.dump({"version": CACHE_VERSION, "df": df}, VITALS_CACHE)
        return df
    stay2pid = dict(zip(st["stay_id"].astype(int).values, st["patient_id"].astype
    with open(VITALS_JSON, "r", encoding="utf-8") as f:
        vit_list = json.load(f)
    rows = []
    for item in vit_list:
        stay_id = int(item.get("stay_id"))
        pid = stay2pid.get(stay_id)
        if pid is None:
            continue
        days = item.get("days", []) or []
        if len(days) == 0:
            continue
        def arr(key: str) -> np.ndarray:
            vals = []
            for d in days:
                v = d.get(key)
                if v is None:
                    continue
                try:
                    vals.append(float(v))
                except Exception:

                    pass
            return np.array(vals, dtype=np.float32)
        notes = []
        for d in days:
            n = d.get("note")
            if n:
                notes.append(str(n))
        feat = {"patient_id": pid, "stay_id": stay_id}
        for key in ["hr", "sbp", "dbp", "temp_c", "rr"]:
            a = arr(key)
            if a.size > 0:
                feat[f"v_{key}_mean"] = float(np.mean(a))
                feat[f"v_{key}_std"] = float(np.std(a))
                feat[f"v_{key}_min"] = float(np.min(a))
                feat[f"v_{key}_max"] = float(np.max(a))
                feat[f"v_{key}_delta"] = float(a[-1] - a[0])
            else:
                feat[f"v_{key}_mean"] = np.nan
                feat[f"v_{key}_std"] = np.nan
                feat[f"v_{key}_min"] = np.nan
                feat[f"v_{key}_max"] = np.nan
                feat[f"v_{key}_delta"] = np.nan
        feat["vitals_doc"] = " ".join(notes)
        feat["v_n_days"] = int(len(days))
        rows.append(feat)
    if len(rows) == 0:
        logging.info("Vitals JSON parsed but no matching stay_id→patient_id. Skip
        df = pd.DataFrame({"patient_id": [], "vitals_doc": [], "vitals_n_stays": 
        joblib.dump({"version": CACHE_VERSION, "df": df}, VITALS_CACHE)
        return df
    stay_df = pd.DataFrame(rows)
    num_cols = [c for c in stay_df.columns if c not in ["patient_id", "stay_id", 
    agg_num = stay_df.groupby("patient_id")[num_cols].agg(["mean", "max"])
    agg_num.columns = [f"{c}_{stat}" for c, stat in agg_num.columns]

    agg_num = agg_num.reset_index()
    agg_txt = stay_df.groupby("patient_id").agg(
        vitals_doc=("vitals_doc", lambda s: " ".join([x for x in s.tolist() if is
        vitals_n_stays=("stay_id", "nunique"),
    ).reset_index()
    agg = agg_num.merge(agg_txt, on="patient_id", how="left")
    joblib.dump({"version": CACHE_VERSION, "df": agg}, VITALS_CACHE)
    logging.info(f"Built vitals features: {agg.shape}")
    return agg
# =========================
# RELATIONAL AGGS (admissions + stays meta counts)
# =========================
def load_relational_aggs() -> pd.DataFrame:
    if RELATIONAL_CACHE.exists():
        try:
            obj = joblib.load(RELATIONAL_CACHE)
            if isinstance(obj, dict) and obj.get("version") == CACHE_VERSION and 
                return obj["df"]
        except Exception:
            pass
    parts = []
    if ADMISSIONS_TRAIN.exists() and ADMISSIONS_TEST.exists():
        adm = pd.concat([pd.read_csv(ADMISSIONS_TRAIN), pd.read_csv(ADMISSIONS_TE
        if "readmit_30d" in adm.columns:
            adm = adm.drop(columns=["readmit_30d"])
        if "patient_id" in adm.columns:
            agg = adm.groupby("patient_id").agg(adm_n=("admission_id", "count") i
            for c in ["los_days", "acuity_emergent", "charlson_band", "ed_visits_
                if c in adm.columns:
                    g = adm.groupby("patient_id")[c]
                    agg[f"adm_{c}_mean"] = g.mean().values
                    agg[f"adm_{c}_max"] = g.max().values
            parts.append(agg)

    if STAYS_TRAIN.exists() and STAYS_TEST.exists():
        st = pd.concat([pd.read_csv(STAYS_TRAIN), pd.read_csv(STAYS_TEST)], ignor
        if "discharge_ready_day11" in st.columns:
            st = st.drop(columns=["discharge_ready_day11"])
        if "patient_id" in st.columns:
            agg = st.groupby("patient_id").agg(stay_n=("stay_id", "count") if "st
            parts.append(agg)
    if len(parts) == 0:
        df = pd.DataFrame({"patient_id": []})
        joblib.dump({"version": CACHE_VERSION, "df": df}, RELATIONAL_CACHE)
        return df
    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on="patient_id", how="outer")
    joblib.dump({"version": CACHE_VERSION, "df": out}, RELATIONAL_CACHE)
    logging.info(f"Built relational aggs: {out.shape}")
    return out
# =========================
# DATA ASSEMBLY
# =========================
def assemble_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    if PATIENTS_CSV.exists():
        pat = pd.read_csv(PATIENTS_CSV)
        if "patient_id" in pat.columns:
            train = train.merge(pat, on="patient_id", how="left")
            test = test.merge(pat, on="patient_id", how="left")
            logging.info(f"Merged patients.csv: train={train.shape}, test={test.s
    train["is_train"] = 1
    test["is_train"] = 0
    all_df = pd.concat([train, test], ignore_index=True)

    pids = all_df["patient_id"].astype(int).tolist()
    rec = load_or_build_receipts(pids)
    all_df = all_df.merge(rec, on="patient_id", how="left")
    rel = load_relational_aggs()
    if "patient_id" in rel.columns and len(rel) > 0:
        all_df = all_df.merge(rel, on="patient_id", how="left")
    dis = load_discharge_docs()
    if "patient_id" in dis.columns and len(dis) > 0:
        all_df = all_df.merge(dis, on="patient_id", how="left")
    else:
        all_df["discharge_doc"] = ""
        all_df["discharge_n_notes"] = 0
    vit = load_vitals_features()
    if "patient_id" in vit.columns and len(vit) > 0:
        all_df = all_df.merge(vit, on="patient_id", how="left")
    else:
        all_df["vitals_doc"] = ""
        all_df["vitals_n_stays"] = 0
    for c in ["receipt_doc", "discharge_doc", "vitals_doc"]:
        if c in all_df.columns:
            all_df[c] = all_df[c].fillna("").astype(str)
    if "prior_ed_cost_5y_usd" in all_df.columns and "prior_ed_visits_5y" in all_d
        all_df["prior_cost_per_visit"] = all_df["prior_ed_cost_5y_usd"] / (all_df
        all_df["log_prior_cost"] = np.log1p(all_df["prior_ed_cost_5y_usd"].astype
        all_df["log_prior_visits"] = np.log1p(all_df["prior_ed_visits_5y"].astype
    train_out = all_df[all_df["is_train"] == 1].reset_index(drop=True)
    test_out = all_df[all_df["is_train"] == 0].reset_index(drop=True)
    return train_out, test_out
# =========================
# FEATURE BUILDERS (fold-safe)
# =========================
def build_tab_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTra

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_co
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("ohe", make_ohe())]), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
def safe_tfidf_fit_transform(vec: TfidfVectorizer, tr_text: np.ndarray, va_text: 
    try:
        Xtr = vec.fit_transform(tr_text)
        Xva = vec.transform(va_text)
        Xte = vec.transform(te_text)
        return ensure_csr_float32_int32(Xtr), ensure_csr_float32_int32(Xva), ensu
    except ValueError:
        n_tr = len(tr_text); n_va = len(va_text); n_te = len(te_text)
        Ztr = csr_matrix((n_tr, 0), dtype=np.float32)
        Zva = csr_matrix((n_va, 0), dtype=np.float32)
        Zte = csr_matrix((n_te, 0), dtype=np.float32)
        return Ztr, Zva, Zte
def build_fold_matrices(
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    df_te: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
) -> Tuple[csr_matrix, csr_matrix, csr_matrix]:
    pre = build_tab_preprocessor(num_cols, cat_cols)
    Xtr_tab = ensure_csr_float32_int32(pre.fit_transform(df_tr).tocsr())
    Xva_tab = ensure_csr_float32_int32(pre.transform(df_va).tocsr())
    Xte_tab = ensure_csr_float32_int32(pre.transform(df_te).tocsr())
    vec_rec = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b[A-Za-z0-9]+\b",

        lowercase=True,
        ngram_range=(1, 2),
        max_features=MAXF_RECEIPT,
        min_df=MIN_DF_RECEIPT,
        max_df=1.0,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    Xtr_r, Xva_r, Xte_r = safe_tfidf_fit_transform(
        vec_rec,
        df_tr["receipt_doc"].values.astype(str),
        df_va["receipt_doc"].values.astype(str),
        df_te["receipt_doc"].values.astype(str),
    )
    vec_dis = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
        lowercase=True,
        ngram_range=(1, 2),
        max_features=MAXF_DISCHARGE,
        min_df=MIN_DF_NOTES,
        max_df=1.0,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    Xtr_d, Xva_d, Xte_d = safe_tfidf_fit_transform(
        vec_dis,
        df_tr["discharge_doc"].values.astype(str),
        df_va["discharge_doc"].values.astype(str),
        df_te["discharge_doc"].values.astype(str),
    )
    vec_vn = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
        lowercase=True,
        ngram_range=(1, 2),

        max_features=MAXF_VITALS_NOTE,
        min_df=MIN_DF_NOTES,
        max_df=1.0,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    Xtr_vt, Xva_vt, Xte_vt = safe_tfidf_fit_transform(
        vec_vn,
        df_tr["vitals_doc"].values.astype(str),
        df_va["vitals_doc"].values.astype(str),
        df_te["vitals_doc"].values.astype(str),
    )
    Xtr = ensure_csr_float32_int32(hstack([Xtr_tab, Xtr_r, Xtr_d, Xtr_vt], format
    Xva = ensure_csr_float32_int32(hstack([Xva_tab, Xva_r, Xva_d, Xva_vt], format
    Xte = ensure_csr_float32_int32(hstack([Xte_tab, Xte_r, Xte_d, Xte_vt], format
    return Xtr, Xva, Xte
# =========================
# TRAINING
# =========================
def fit_xgb(
    X_tr: csr_matrix, y_tr: np.ndarray,
    X_va: csr_matrix, y_va: np.ndarray,
    params: Dict[str, Any],
    seed: int,
    tag: str,
) -> XGBRegressor:
    model = XGBRegressor(**params, random_state=seed)
    fit_sig = inspect.signature(model.fit).parameters
    fit_kwargs = {"eval_set": [(X_va, y_va)]}
    if "verbose_eval" in fit_sig:
        fit_kwargs["verbose_eval"] = VERBOSE_EVERY
    else:
        fit_kwargs["verbose"] = VERBOSE_EVERY
    logging.info(f"{tag} | fit START | obj={params.get('objective')} | device={pa
    model.fit(X_tr, y_tr, **fit_kwargs)
    logging.info(f"{tag} | fit END | best_iteration={getattr(model,'best_iteratio

    return model
def shrunk_group_median_bias(residuals: np.ndarray, groups: np.ndarray, alpha: fl
    g = groups.astype(str)
    global_med = float(np.median(residuals))
    biases: Dict[str, float] = {}
    for key in np.unique(g):
        m = (g == key)
        rg = residuals[m]
        if rg.size == 0:
            continue
        med_g = float(np.median(rg))
        n = float(rg.size)
        b = (n * med_g + alpha * global_med) / (n + alpha)
        biases[str(key)] = float(b)
    return biases, global_med
def apply_bias(pred: np.ndarray, groups: np.ndarray, biases: Dict[str, float], de
    out = pred.astype(np.float64).copy()
    g = groups.astype(str)
    for i in range(len(out)):
        out[i] += biases.get(str(g[i]), default_bias)
    return out.astype(np.float32)
def run_iteration8(diagnostic: bool = False, force_cpu: bool = False) -> Dict[str
    setup_logging()
    logging.info(f"XGBoost version: {getattr(xgb,'__version__','unknown')}")
    logging.info("=== Iteration 8: multi-source + leakage-safe CV + receipt seman
    train_df, test_df = assemble_dataset()
    logging.info(f"train_df={train_df.shape} test_df={test_df.shape}")
    coverage = {
        "receipt_nonempty": int((train_df["receipt_doc"].str.len() > 0).sum()),
        "discharge_nonempty": int((train_df["discharge_doc"].str.len() > 0).sum()
        "vitals_nonempty": int((train_df["vitals_doc"].str.len() > 0).sum()) if "
    }

    logging.info(f"Coverage (train): {coverage}")
    if diagnostic:
        return {"coverage": coverage}
    y = train_df["ed_cost_next3y_usd"].values.astype(np.float32)
    doc_cols = {"receipt_doc", "discharge_doc", "vitals_doc"}
    exclude = {"patient_id", "ed_cost_next3y_usd", "is_train"} | doc_cols
    preferred_cats = ["primary_chronic", "sex", "insurance", "zip3", "pdf_insuran
    cat_cols = [c for c in preferred_cats if c in train_df.columns]
    for c in train_df.columns:
        if c in exclude or c in cat_cols:
            continue
        if train_df[c].dtype == "object" or str(train_df[c].dtype).startswith("st
            cat_cols.append(c)
    num_cols = []
    for c in train_df.columns:
        if c in exclude or c in cat_cols:
            continue
        if pd.api.types.is_numeric_dtype(train_df[c]):
            num_cols.append(c)
    for c in cat_cols:
        train_df[c] = train_df[c].astype("string").fillna("missing")
        test_df[c] = test_df[c].astype("string").fillna("missing")
    logging.info(f"Num cols={len(num_cols)} | Cat cols={len(cat_cols)} | Text col
    device = "cpu" if force_cpu else "cuda"
    common_params = dict(
        eval_metric="mae",
        tree_method="hist",
        device=device,
        n_estimators=N_ESTIMATORS,
        learning_rate=LR,
        max_depth=MAX_DEPTH,
        subsample=SUBSAMPLE,

        colsample_bytree=COLSAMPLE,
        reg_lambda=REG_LAMBDA,
        min_child_weight=MIN_CHILD_WEIGHT,
        n_jobs=-1,
        verbosity=1,
        early_stopping_rounds=EARLY_STOPPING,
    )
    params_l2 = dict(**common_params, objective="reg:squarederror")
    params_q50 = dict(**common_params, objective="reg:quantileerror", quantile_al
    if "prior_ed_cost_5y_usd" in train_df.columns:
        bins = pd.qcut(train_df["prior_ed_cost_5y_usd"], q=10, duplicates="drop")
        strat = train_df["primary_chronic"].astype(str) + "_" + bins.astype(str)
    else:
        strat = train_df["primary_chronic"].astype(str)
    oof_l2 = np.zeros(len(train_df), dtype=np.float64)
    oof_q = np.zeros(len(train_df), dtype=np.float64)
    cnt = np.zeros(len(train_df), dtype=np.float64)
    te_l2 = np.zeros(len(test_df), dtype=np.float64)
    te_q = np.zeros(len(test_df), dtype=np.float64)
    n_fit = 0
    splits = []
    for rep, seed in enumerate(REPEAT_SEEDS):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(train_df))
            splits.append((rep, fold, seed, tr_idx, va_idx))
    logging.info(f"Total CV folds: {len(splits)} | device={device}")
    for rep, fold, seed, tr_idx, va_idx in tqdm(splits, desc="CV", ncols=110):
        tag = f"rep={rep} fold={fold}"
        t0 = time.time()
        df_tr = train_df.iloc[tr_idx].copy()
        df_va = train_df.iloc[va_idx].copy()
        df_te = test_df.copy()
        logging.info(f"{tag} | FE START")

        X_tr, X_va, X_te = build_fold_matrices(df_tr, df_va, df_te, num_cols, cat_
        logging.info(f"{tag} | FE END | X_tr={X_tr.shape} nnz={X_tr.nnz:,}")
        if USE_L2:
            mdl_l2 = fit_xgb(X_tr, y[tr_idx], X_va, y[va_idx], params_l2, seed=10
            p_va = mdl_l2.predict(X_va)
            p_te = mdl_l2.predict(X_te)
            oof_l2[va_idx] += p_va
            te_l2 += p_te
            logging.info(f"{tag} | L2 fold MAE={mae(y[va_idx], p_va):.3f}")
        if USE_Q50:
            mdl_q = fit_xgb(X_tr, y[tr_idx], X_va, y[va_idx], params_q50, seed=20
            p_va = mdl_q.predict(X_va)
            p_te = mdl_q.predict(X_te)
            oof_q[va_idx] += p_va
            te_q += p_te
            logging.info(f"{tag} | Q50 fold MAE={mae(y[va_idx], p_va):.3f}")
        cnt[va_idx] += 1.0
        n_fit += 1
        logging.info(f"{tag} | DONE | fold_time={time.time()-t0:.1f}s")
    oof_l2 = (oof_l2 / np.maximum(cnt, 1.0)).astype(np.float32)
    oof_q = (oof_q / np.maximum(cnt, 1.0)).astype(np.float32)
    te_l2 = (te_l2 / max(1, n_fit)).astype(np.float32)
    te_q = (te_q / max(1, n_fit)).astype(np.float32)
    if USE_L2 and not USE_Q50:
        w_best = 1.0
    elif USE_Q50 and not USE_L2:
        w_best = 0.0
    else:
        grid = np.arange(0.0, 1.0 + 1e-9, BLEND_GRID_STEP)
        best_mae = 1e18
        w_best = 0.5
        for w in grid:
            pred = w * oof_l2 + (1.0 - w) * oof_q
            s = mae(y, pred)

            if s < best_mae:
                best_mae = s
                w_best = float(w)
        logging.info(f"Blend best: w(L2)={w_best:.2f} | OOF MAE={best_mae:.4f}")
    oof = (w_best * oof_l2 + (1.0 - w_best) * oof_q).astype(np.float32)
    te = (w_best * te_l2 + (1.0 - w_best) * te_q).astype(np.float32)
    if CLIP_AT_ZERO:
        oof = np.maximum(oof, 0.0)
        te = np.maximum(te, 0.0)
    mae_raw = mae(y, oof)
    logging.info(f"OOF MAE (blend raw): {mae_raw:.4f}")
    chosen = "raw"
    final_te = te
    if USE_CALIBRATION:
        resid = (y.astype(np.float64) - oof.astype(np.float64))
        b0 = float(np.median(resid))
        oof_g = oof + b0
        te_g = te + b0
        if CLIP_AT_ZERO:
            oof_g = np.maximum(oof_g, 0.0)
            te_g = np.maximum(te_g, 0.0)
        mae_g = mae(y, oof_g)
        grp = train_df["primary_chronic"].astype(str).values
        biases, default_bias = shrunk_group_median_bias(resid, grp, alpha=CALIB_A
        oof_c = apply_bias(oof, grp, biases, default_bias)
        te_c = apply_bias(te, test_df["primary_chronic"].astype(str).values, bias
        if CLIP_AT_ZERO:
            oof_c = np.maximum(oof_c, 0.0)
            te_c = np.maximum(te_c, 0.0)
        mae_c = mae(y, oof_c)
        logging.info(f"Calibration OOF MAE: raw={mae_raw:.4f} | global_median={ma

        best = (mae_raw, "raw", te)
        if mae_g < best[0] - 1e-9:
            best = (mae_g, "global_median", te_g)
        if mae_c < best[0] - 1e-9:
            best = (mae_c, "chronic_shrunk_median", te_c)
        chosen = best[1]
        final_te = best[2].astype(np.float32)
        logging.info(f"Chosen postprocess: {chosen} | OOF MAE={best[0]:.4f}")
    sub = pd.DataFrame({
        "patient_id": test_df["patient_id"].astype(int).values,
        "ed_cost_next3y_usd": final_te.astype(np.float32),
    })
    sub.to_csv(SUBMISSION_PATH, index=False)
    logging.info(f"Saved submission: {SUBMISSION_PATH}")
    return {
        "oof_mae_raw": float(mae_raw),
        "blend_w_l2": float(w_best),
        "postprocess": chosen,
        "submission_path": str(SUBMISSION_PATH),
        "coverage": coverage,
    }
# =========================
# Notebook/CLI glue
# =========================
def _in_notebook() -> bool:
    try:
        import IPython  # noqa
        ip = IPython.get_ipython()
        if ip is None:
            return False
        return ("IPKernelApp" in ip.config) or ("ipykernel" in sys.modules)
    except Exception:
        return ("ipykernel" in sys.modules)

你在 Notebook 里怎么跑（建议流程）
1. 第一次先跑覆盖率（确认“新模态”真的存在）：
你要重点看：
discharge_nonempty  是否明显 > 0
vitals_nonempty  是否明显 > 0
如果这两个基本为 0，说明你本地目录里没有这些文件/或 patient_id 没对上，那这条
路就不会给你带来 30 分的提升（那就需要另一种“跳出框架”的方案）。
2. 真正训练并生成 submission：
def _parse_cli(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        logging.info(f"Ignoring unknown args: {unknown}")
    return args
if __name__ == "__main__":
    setup_logging()
    if _in_notebook():
        logging.info("Notebook detected: call run_iteration8(diagnostic=True) or 
    else:
        a = _parse_cli()
        run_iteration8(diagnostic=a.diagnostic, force_cpu=a.cpu)
python
res = run_iteration8(diagnostic=True)
res
Copy code
python

3. 如果你修改了解析逻辑但缓存没更新：删掉
D:\AgentDs\agent_ds_healthcare\cache_iter8\
重新跑即可。
如果你跑完 Iter8 后把两行信息贴我：
Coverage (train): {...}
最终你提交得到的 MAE
我就能立刻判断：到底是“新模态确实存在但没被有效利用”，还是 “你这份数据其实没有其
它模态可用”（那我们就要走另一条完全不同的“数学路径”，例如更强的 outlier-aware 训练/
分段回归/极端值校准，但仍然保持不做 SVD、不做人为分组）。
res = run_iteration8(diagnostic=False)
res
Copy code
```

### C14 — ITERATION 9 — Outlier-Aware Segmented Regression + Extreme Calibration (XGBoost G (pages 368-395)
```python
# -*- coding: utf-8 -*-
r"""
ITERATION 9 — Outlier-Aware Segmented Regression + Extreme Calibration (XGBoost G
=================================================================================
Why this is a "different mathematical path":
- Your adversarial AUC ~ 0.51 means train/test feature distributions are similar; 
- We're focusing on heavy-tail/outliers: a hurdle/segmented model:
    y = body (<= cap) + tail_excess (only if y > cap)
- Body model: quantile regression alpha=0.5 (MAE-aligned).  :contentReference[oai
- Tail model: (1) classifier for tail event (2) regressor for excess magnitude (l
- Then do "extreme calibration": tune (k, gamma) on OOF without retraining, and d
Constraints honored:
- NO SVD/PCA/NMF/embeddings
- NO manual clinical CPT grouping

- GPU-friendly XGBoost only
- Rich logs, notebook-safe (no argparse crash)
INSTALL:
pip install -U pandas numpy scipy scikit-learn xgboost pymupdf joblib tqdm
"""
from __future__ import annotations
import re
import time
import sys
import logging
import warnings
import inspect
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import fitz  # PyMuPDF
from scipy.sparse import csr_matrix, hstack
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from xgboost import XGBRegressor, XGBClassifier
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =========================
# PATHS
# =========================
BASE_DIR = Path(r"D:\AgentDs\agent_ds_healthcare")
TRAIN_CSV = BASE_DIR / "ed_cost_train.csv"
TEST_CSV = BASE_DIR / "ed_cost_test.csv"
PATIENTS_CSV = BASE_DIR / "patients.csv"
PDF_DIR = BASE_DIR / "receipts_pdf"
SUBMISSION_PATH = BASE_DIR / "submission.csv"
CACHE_DIR = BASE_DIR / "cache_iter9"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RECEIPT_CACHE = CACHE_DIR / "receipts_codes_meta.joblib"
CACHE_VERSION = 91
# =========================
# CONFIG
# =========================
# TF-IDF (simple, sparse, no DR)
TFIDF_MAX_FEATURES = 1500      # you can try 1000/1500/2000; 1500 is a safe middl
TFIDF_NGRAM_RANGE = (1, 3)
TFIDF_MIN_DF = 1
TFIDF_MAX_DF = 1.0
# CV
N_FOLDS = 5
REPEAT_SEEDS = [2025, 2026]     # total 10 folds
EARLY_STOPPING = 200
# Outlier segmentation
TAIL_Q = 0.90                  # 0.90 => ~200 tail samples if n=2000
CLIP_AT_ZERO = True
# Tail mixing calibration search
GAMMA_GRID = [0.6, 0.8, 1.0, 1.2, 1.5]
K_GRID = [0.6, 0.8, 1.0, 1.2, 1.5]

# Piecewise median residual calibration
CALIB_BINS = 10
CALIB_SHRINK_ALPHA = 80.0      # larger => less overfit
CALIB_CORR_CLIP = 500.0        # clip residual correction per bin for safety
# XGB common
N_ESTIMATORS = 8000
LR = 0.02
SUBSAMPLE = 0.85
COLSAMPLE = 0.75
REG_LAMBDA = 1.0
MIN_CHILD_WEIGHT = 1.0
# Separate depths for stability
DEPTH_BODY = 6
DEPTH_TAIL_CLS = 4
DEPTH_TAIL_REG = 4
# Heartbeats
VERBOSE_EVERY = 100
# =========================
# LOGGING
# =========================
def setup_logging() -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname
    else:
        root.setLevel(logging.INFO)
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))
def ensure_csr_float32_int32(X: csr_matrix) -> csr_matrix:
    X = X.tocsr()
    if X.data.dtype != np.float32:

        X.data = X.data.astype(np.float32, copy=False)
    if X.indices.dtype != np.int32:
        X.indices = X.indices.astype(np.int32, copy=False)
    if X.indptr.dtype != np.int32:
        X.indptr = X.indptr.astype(np.int32, copy=False)
    return X
def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_out
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)
# =========================
# RECEIPT PARSING (FAST, ROBUST)
# =========================
ZIP_INS_RE = re.compile(r"ZIP3:\s*([0-9]{3})\s+Insurance:\s*([A-Za-z_]+)")
TOTAL_RE = re.compile(r"TOTAL\s+([\d,]+\.\d{2})")
LINE_ITEM_RE = re.compile(
    r"^([A-Z]\d{4}|\d{5})\s+(.+?)\s+(\d+)\s+([\d,]+\.\d{2})\s+([\d,]+\.\d{2})\s*$
)
CODE_TOKEN_RE = re.compile(r"\b(?:[A-Z]\d{4}|\d{5})\b")
def _money_to_float(s: str) -> float:
    return float(s.replace(",", ""))
def featurize_receipt(patient_id: int) -> Dict[str, Any]:
    """
    Output dict:
      - codes_doc: string of repeated CPT/HCPCS tokens (repeated by qty capped)
      - meta numeric/cat features
    """
    pdf_path = PDF_DIR / f"receipt_{patient_id}.pdf"

    out = dict(
        patient_id=int(patient_id),
        codes_doc="",
        pdf_ok=0,
        pdf_zip3="missing",
        pdf_insurance="missing",
        pdf_n_items=0,
        pdf_qty_sum=0,
        pdf_n_unique_codes=0,
        pdf_line_max=0.0,
        pdf_line_mean=0.0,
        pdf_total=0.0,
        pdf_cost_concentration=0.0,
        pdf_parse_notes="missing_pdf",
    )
    if not pdf_path.exists():
        return out
    try:
        with fitz.open(pdf_path) as doc:
            text = ""
            for page in doc:
                text += (page.get_text("text") or "") + "\n"
    except Exception as e:
        out["pdf_parse_notes"] = f"open_error:{type(e).__name__}"
        return out
    text = text or ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # header
    m = ZIP_INS_RE.search(text)
    if m:
        out["pdf_zip3"] = str(m.group(1))
        out["pdf_insurance"] = str(m.group(2)).lower()
    m = TOTAL_RE.search(text)
    if m:
        try:

            out["pdf_total"] = float(_money_to_float(m.group(1)))
        except Exception:
            out["pdf_total"] = 0.0
    # line items
    tokens: List[str] = []
    line_totals: List[float] = []
    qty_sum = 0
    n_items = 0
    for ln in lines:
        mm = LINE_ITEM_RE.match(ln)
        if not mm:
            continue
        code, desc, qty_s, unit_s, line_s = mm.groups()
        try:
            qty = int(qty_s)
        except Exception:
            qty = 1
        qty = max(1, min(qty, 20))
        qty_sum += qty
        n_items += 1
        try:
            line_total = float(_money_to_float(line_s))
            line_totals.append(line_total)
        except Exception:
            pass
        # repeat by qty (capped) to preserve intensity without manual grouping
        rep = max(1, min(qty, 10))
        tokens.extend([code] * rep)
    if len(tokens) == 0:
        # fallback: just extract any code tokens from whole text
        tokens = CODE_TOKEN_RE.findall(text)
    out["codes_doc"] = " ".join(tokens)
    out["pdf_ok"] = 1 if len(out["codes_doc"]) > 0 else 0
    out["pdf_n_items"] = int(n_items)

    out["pdf_qty_sum"] = int(qty_sum)
    out["pdf_n_unique_codes"] = int(len(set(tokens)))
    if len(line_totals) > 0:
        out["pdf_line_max"] = float(np.max(line_totals))
        out["pdf_line_mean"] = float(np.mean(line_totals))
        if out["pdf_total"] > 0:
            out["pdf_cost_concentration"] = float(out["pdf_line_max"] / out["pdf_t
    else:
        out["pdf_line_max"] = 0.0
        out["pdf_line_mean"] = 0.0
        out["pdf_cost_concentration"] = 0.0
    out["pdf_parse_notes"] = "ok"
    return out
def load_receipts(patient_ids: List[int]) -> pd.DataFrame:
    cache = {"version": CACHE_VERSION, "data": {}}
    if RECEIPT_CACHE.exists():
        try:
            cache = joblib.load(RECEIPT_CACHE)
        except Exception:
            cache = {"version": CACHE_VERSION, "data": {}}
    if not isinstance(cache, dict) or cache.get("version") != CACHE_VERSION:
        cache = {"version": CACHE_VERSION, "data": {}}
    data = cache.get("data", {})
    missing = [pid for pid in patient_ids if str(pid) not in data]
    logging.info(f"Receipt cache | hit={len(patient_ids)-len(missing):,} miss={le
    for pid in tqdm(missing, desc="Parsing receipts", ncols=100):
        data[str(pid)] = featurize_receipt(int(pid))
    cache["data"] = data
    if missing:
        try:
            joblib.dump(cache, RECEIPT_CACHE)
        except Exception as e:

            logging.warning(f"Receipt cache save failed: {e}")
    rows = []
    for pid in patient_ids:
        obj = data.get(str(pid))
        if obj is None:
            obj = featurize_receipt(int(pid))
        rows.append(obj)
    return pd.DataFrame(rows)
# =========================
# FEATURE BUILDING (fold-safe)
# =========================
def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransfo
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_co
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("ohe", make_ohe())]), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
def safe_tfidf_fit_transform(tr_text: np.ndarray, va_text: np.ndarray, te_text: n
    vec = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b(?:[A-Z]\d{4}|\d{5})\b",
        lowercase=False,
        ngram_range=TFIDF_NGRAM_RANGE,
        max_features=TFIDF_MAX_FEATURES,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    try:

        Xtr = vec.fit_transform(tr_text.astype(str))
        Xva = vec.transform(va_text.astype(str))
        Xte = vec.transform(te_text.astype(str))
        return ensure_csr_float32_int32(Xtr), ensure_csr_float32_int32(Xva), ensu
    except ValueError:
        # empty vocabulary etc.
        n_tr, n_va, n_te = len(tr_text), len(va_text), len(te_text)
        Ztr = csr_matrix((n_tr, 0), dtype=np.float32)
        Zva = csr_matrix((n_va, 0), dtype=np.float32)
        Zte = csr_matrix((n_te, 0), dtype=np.float32)
        return Ztr, Zva, Zte
def build_fold_matrices(
    df_tr: pd.DataFrame, df_va: pd.DataFrame, df_te: pd.DataFrame,
    num_cols: List[str], cat_cols: List[str],
) -> Tuple[csr_matrix, csr_matrix, csr_matrix]:
    pre = build_preprocessor(num_cols, cat_cols)
    Xtr_tab = ensure_csr_float32_int32(pre.fit_transform(df_tr).tocsr())
    Xva_tab = ensure_csr_float32_int32(pre.transform(df_va).tocsr())
    Xte_tab = ensure_csr_float32_int32(pre.transform(df_te).tocsr())
    Xtr_txt, Xva_txt, Xte_txt = safe_tfidf_fit_transform(
        df_tr["codes_doc"].values,
        df_va["codes_doc"].values,
        df_te["codes_doc"].values,
    )
    Xtr = ensure_csr_float32_int32(hstack([Xtr_tab, Xtr_txt], format="csr"))
    Xva = ensure_csr_float32_int32(hstack([Xva_tab, Xva_txt], format="csr"))
    Xte = ensure_csr_float32_int32(hstack([Xte_tab, Xte_txt], format="csr"))
    return Xtr, Xva, Xte
# =========================
# MODEL HELPERS
# =========================
def _fit_with_verbose(model, X_tr, y_tr, X_va, y_va, tag: str):
    fit_sig = inspect.signature(model.fit).parameters
    fit_kwargs = {"eval_set": [(X_va, y_va)]}

    if "verbose_eval" in fit_sig:
        fit_kwargs["verbose_eval"] = VERBOSE_EVERY
    else:
        fit_kwargs["verbose"] = VERBOSE_EVERY
    logging.info(f"{tag} | fit START")
    model.fit(X_tr, y_tr, **fit_kwargs)
    logging.info(f"{tag} | fit END | best_iteration={getattr(model,'best_iteratio
    return model
# =========================
# CALIBRATION: bin-wise median residual (shrunk)
# =========================
def build_bin_median_correction(oof_pred: np.ndarray, y: np.ndarray, n_bins: int, 
    """
    Returns:
      edges: quantile bin edges (len n_bins+1)
      corr: per-bin correction (len n_bins)
      global_med: global median residual
    """
    oof = oof_pred.astype(np.float64)
    resid = (y.astype(np.float64) - oof)
    # quantile edges; ensure strictly increasing
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(oof, qs)
    # fix duplicates
    for i in range(1, len(edges)):
        if edges[i] <= edges[i-1]:
            edges[i] = edges[i-1] + 1e-6
    bin_id = np.clip(np.digitize(oof, edges[1:-1], right=False), 0, n_bins - 1)
    global_med = float(np.median(resid))
    corr = np.zeros(n_bins, dtype=np.float64)
    for b in range(n_bins):
        m = (bin_id == b)
        rb = resid[m]

        if rb.size == 0:
            corr[b] = global_med
            continue
        med_b = float(np.median(rb))
        n = float(rb.size)
        c = (n * med_b + alpha * global_med) / (n + alpha)
        c = float(np.clip(c, -CALIB_CORR_CLIP, CALIB_CORR_CLIP))
        corr[b] = c
    return edges.astype(np.float64), corr.astype(np.float64), global_med
def apply_bin_correction(pred: np.ndarray, edges: np.ndarray, corr: np.ndarray) -
    p = pred.astype(np.float64)
    n_bins = len(corr)
    bin_id = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    p2 = p + corr[bin_id]
    return p2.astype(np.float32)
# =========================
# MAIN RUNNER
# =========================
def run_iteration9(diagnostic: bool = False, force_cpu: bool = False) -> Dict[str
    setup_logging()
    logging.info(f"XGBoost version: {getattr(xgb,'__version__','unknown')}")
    logging.info("=== Iteration 9: Outlier-aware segmented regression + extreme c
    logging.info(f"diagnostic={diagnostic} | force_cpu={force_cpu}")
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    logging.info(f"Loaded train={train.shape} test={test.shape}")
    if PATIENTS_CSV.exists():
        pat = pd.read_csv(PATIENTS_CSV)
        if "patient_id" in pat.columns:
            train = train.merge(pat, on="patient_id", how="left")
            test = test.merge(pat, on="patient_id", how="left")
            logging.info(f"Merged patients.csv: train={train.shape} test={test.sh

    # receipts
    all_ids = pd.concat([train["patient_id"], test["patient_id"]], ignore_index=T
    rec = load_receipts(all_ids)
    train = train.merge(rec, on="patient_id", how="left")
    test = test.merge(rec, on="patient_id", how="left")
    for df in [train, test]:
        df["codes_doc"] = df["codes_doc"].fillna("").astype(str)
    # basic feature derivations (simple, numeric)
    for df in [train, test]:
        if "prior_ed_cost_5y_usd" in df.columns and "prior_ed_visits_5y" in df.co
            df["prior_cost_per_visit"] = df["prior_ed_cost_5y_usd"] / (df["prior_
            df["log_prior_cost"] = np.log1p(df["prior_ed_cost_5y_usd"].astype(flo
            df["log_prior_visits"] = np.log1p(df["prior_ed_visits_5y"].astype(flo
        # sanity: pdf_total should match prior_ed_cost_5y_usd per dataset descrip
        if "pdf_total" in df.columns and "prior_ed_cost_5y_usd" in df.columns:
            df["pdf_total_minus_prior"] = df["pdf_total"].astype(float) - df["pri
            df["pdf_total_absdiff_prior"] = np.abs(df["pdf_total_minus_prior"])
    y = train["ed_cost_next3y_usd"].values.astype(np.float32)
    cap = float(np.quantile(y, TAIL_Q))
    y_tail = (y > cap).astype(np.int32)
    y_body = np.minimum(y, cap).astype(np.float32)
    y_excess = np.maximum(0.0, y - cap).astype(np.float32)
    logging.info(f"Tail threshold: cap=q{TAIL_Q:.2f}={cap:.2f} | tail_rate={y_tai
    if diagnostic:
        return {"cap": cap, "tail_rate": float(y_tail.mean()), "n_tail": int(y_ta
    # columns
    doc_cols = {"codes_doc"}
    exclude = {"patient_id", "ed_cost_next3y_usd"} | doc_cols
    # categorical columns
    base_cat = []
    for c in ["primary_chronic", "sex", "insurance", "zip3", "pdf_insurance", "pd
        if c in train.columns:

            base_cat.append(c)
    # any other object/string => cat
    cat_cols = list(dict.fromkeys(base_cat))
    for c in train.columns:
        if c in exclude or c in cat_cols:
            continue
        if train[c].dtype == "object" or str(train[c].dtype).startswith("string")
            cat_cols.append(c)
    num_cols = []
    for c in train.columns:
        if c in exclude or c in cat_cols:
            continue
        if pd.api.types.is_numeric_dtype(train[c]):
            num_cols.append(c)
    # fill cat
    for c in cat_cols:
        train[c] = train[c].astype("string").fillna("missing")
        test[c] = test[c].astype("string").fillna("missing")
    logging.info(f"Features: num={len(num_cols)} cat={len(cat_cols)} tfidf_max={T
    # stratification: ensure tail appears in each fold + chronic + prior cost dec
    if "prior_ed_cost_5y_usd" in train.columns:
        cost_bin = pd.qcut(train["prior_ed_cost_5y_usd"], q=10, duplicates="drop"
    else:
        cost_bin = pd.Series(["bin"] * len(train))
    strat = train["primary_chronic"].astype(str) + "_t" + pd.Series(y_tail).astyp
    # storage for components
    oof_body = np.zeros(len(train), dtype=np.float64)
    oof_p = np.zeros(len(train), dtype=np.float64)
    oof_ex = np.zeros(len(train), dtype=np.float64)
    oof_cnt = np.zeros(len(train), dtype=np.float64)
    te_body = np.zeros(len(test), dtype=np.float64)
    te_p = np.zeros(len(test), dtype=np.float64)
    te_ex = np.zeros(len(test), dtype=np.float64)

    n_models = 0
    device = "cpu" if force_cpu else "cuda"
    params_body = dict(
        objective="reg:quantileerror",
        quantile_alpha=0.5,
        eval_metric="mae",
        tree_method="hist",
        device=device,
        n_estimators=N_ESTIMATORS,
        learning_rate=LR,
        max_depth=DEPTH_BODY,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE,
        reg_lambda=REG_LAMBDA,
        min_child_weight=MIN_CHILD_WEIGHT,
        n_jobs=-1,
        verbosity=1,
        early_stopping_rounds=EARLY_STOPPING,
    )
    params_tail_cls = dict(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        device=device,
        n_estimators=6000,
        learning_rate=LR,
        max_depth=DEPTH_TAIL_CLS,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=REG_LAMBDA,
        min_child_weight=5.0,
        n_jobs=-1,
        verbosity=1,
        early_stopping_rounds=EARLY_STOPPING,
    )
    params_tail_reg = dict(

        objective="reg:squarederror",
        eval_metric="mae",
        tree_method="hist",
        device=device,
        n_estimators=6000,
        learning_rate=LR,
        max_depth=DEPTH_TAIL_REG,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=REG_LAMBDA,
        min_child_weight=5.0,
        n_jobs=-1,
        verbosity=1,
        early_stopping_rounds=EARLY_STOPPING,
    )
    splits = []
    for rep, seed in enumerate(REPEAT_SEEDS):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(train)), st
            splits.append((rep, fold, seed, tr_idx, va_idx))
    logging.info(f"Total folds: {len(splits)} | models per fold=3 | device={devic
    for rep, fold, seed, tr_idx, va_idx in tqdm(splits, desc="CV", ncols=110):
        tag = f"rep={rep} fold={fold}"
        t0 = time.time()
        df_tr = train.iloc[tr_idx].copy()
        df_va = train.iloc[va_idx].copy()
        logging.info(f"{tag} | FE START")
        X_tr, X_va, X_te = build_fold_matrices(df_tr, df_va, test, num_cols, cat_
        logging.info(f"{tag} | FE END | X_tr={X_tr.shape} nnz={X_tr.nnz:,}")
        # -------- Body model (capped target) --------
        body = XGBRegressor(**params_body, random_state=1000 * rep + seed + fold)
        _fit_with_verbose(body, X_tr, y_body[tr_idx], X_va, y_body[va_idx], f"{ta
        p_body_va = body.predict(X_va).astype(np.float32)
        p_body_te = body.predict(X_te).astype(np.float32)

        # keep body within [0, cap] to avoid double-counting tail
        p_body_va = np.clip(p_body_va, 0.0, cap)
        p_body_te = np.clip(p_body_te, 0.0, cap)
        # -------- Tail classifier --------
        # class imbalance weight
        pos = float(np.sum(y_tail[tr_idx] == 1))
        neg = float(np.sum(y_tail[tr_idx] == 0))
        spw = (neg / max(pos, 1.0))
        cls = XGBClassifier(**params_tail_cls, scale_pos_weight=spw, random_state
        _fit_with_verbose(cls, X_tr, y_tail[tr_idx], X_va, y_tail[va_idx], f"{tag
        p_tail_va = cls.predict_proba(X_va)[:, 1].astype(np.float32)
        p_tail_te = cls.predict_proba(X_te)[:, 1].astype(np.float32)
        try:
            auc_va = roc_auc_score(y_tail[va_idx], p_tail_va)
            logging.info(f"{tag} | Tail AUC={auc_va:.4f}")
        except Exception:
            pass
        # -------- Tail excess regressor (only tail rows) --------
        tail_tr = np.where(y_tail[tr_idx] == 1)[0]
        tail_va = np.where(y_tail[va_idx] == 1)[0]  # for eval
        # handle rare case: no tail rows in train fold (shouldn't happen due stra
        if tail_tr.size >= 20:
            X_tr_tail = X_tr[tail_tr]
            y_tr_tail = np.log1p(y_excess[tr_idx][tail_tr]).astype(np.float32)
            if tail_va.size >= 5:
                X_va_tail = X_va[tail_va]
                y_va_tail = np.log1p(y_excess[va_idx][tail_va]).astype(np.float32
            else:
                # fallback eval
                X_va_tail = X_va
                y_va_tail = np.log1p(y_excess[va_idx]).astype(np.float32)
            ex = XGBRegressor(**params_tail_reg, random_state=3000 * rep + seed + 
            _fit_with_verbose(ex, X_tr_tail, y_tr_tail, X_va_tail, y_va_tail, f"{t

            p_ex_va_log = ex.predict(X_va).astype(np.float32)
            p_ex_te_log = ex.predict(X_te).astype(np.float32)
            p_ex_va = np.expm1(np.maximum(p_ex_va_log, 0.0)).astype(np.float32)
            p_ex_te = np.expm1(np.maximum(p_ex_te_log, 0.0)).astype(np.float32)
        else:
            logging.info(f"{tag} | TAIL_REG skipped (tail_tr={tail_tr.size}) -> e
            p_ex_va = np.zeros(len(va_idx), dtype=np.float32)
            p_ex_te = np.zeros(len(test), dtype=np.float32)
        # store OOF components
        oof_body[va_idx] += p_body_va
        oof_p[va_idx] += p_tail_va
        oof_ex[va_idx] += p_ex_va
        oof_cnt[va_idx] += 1.0
        # store TEST components
        te_body += p_body_te
        te_p += p_tail_te
        te_ex += p_ex_te
        n_models += 1
        logging.info(f"{tag} | fold_time={time.time()-t0:.1f}s")
    # average components
    oof_body = (oof_body / np.maximum(oof_cnt, 1.0)).astype(np.float32)
    oof_p = (oof_p / np.maximum(oof_cnt, 1.0)).astype(np.float32)
    oof_ex = (oof_ex / np.maximum(oof_cnt, 1.0)).astype(np.float32)
    te_body = (te_body / max(1, n_models)).astype(np.float32)
    te_p = (te_p / max(1, n_models)).astype(np.float32)
    te_ex = (te_ex / max(1, n_models)).astype(np.float32)
    # search best (k, gamma) on OOF to minimize MAE
    best = (1e18, 1.0, 1.0)  # (mae, k, gamma)
    for gamma in GAMMA_GRID:
        p_g = np.power(np.clip(oof_p, 0.0, 1.0), gamma).astype(np.float32)
        for k in K_GRID:
            pred = oof_body + float(k) * p_g * oof_ex
            if CLIP_AT_ZERO:

                pred = np.maximum(pred, 0.0)
            s = mae(y, pred)
            if s < best[0]:
                best = (s, float(k), float(gamma))
    best_mae, best_k, best_gamma = best
    logging.info(f"Best tail mix on OOF: MAE={best_mae:.4f} | k={best_k:.2f} gamm
    # build final OOF / TEST pred
    oof_pred = oof_body + best_k * np.power(np.clip(oof_p, 0.0, 1.0), best_gamma) 
    te_pred = te_body + best_k * np.power(np.clip(te_p, 0.0, 1.0), best_gamma) * t
    if CLIP_AT_ZERO:
        oof_pred = np.maximum(oof_pred, 0.0)
        te_pred = np.maximum(te_pred, 0.0)
    mae_raw = mae(y, oof_pred)
    logging.info(f"OOF MAE (segmented raw): {mae_raw:.4f}")
    # piecewise median residual calibration
    edges, corr, global_med = build_bin_median_correction(oof_pred, y, n_bins=CAL
    oof_cal = apply_bin_correction(oof_pred, edges, corr)
    te_cal = apply_bin_correction(te_pred, edges, corr)
    if CLIP_AT_ZERO:
        oof_cal = np.maximum(oof_cal, 0.0)
        te_cal = np.maximum(te_cal, 0.0)
    mae_cal = mae(y, oof_cal)
    logging.info(f"OOF MAE (after bin-median calibration): {mae_cal:.4f} | global_
    # choose by OOF
    if mae_cal <= mae_raw + 1e-9:
        final = te_cal
        chosen = "bin_median_calibrated"
    else:
        final = te_pred
        chosen = "raw"
    # submission

你接下来怎么验证这条“新数学路”有没有希望（不用等上榜）
你跑完 run_iteration9()  后，把这几行日志贴我就够了：
1. Tail threshold: cap=... | tail_rate=... （看尾部样本是否足够）
2. Best tail mix on OOF: ... | k=... gamma=... （如果最优 k  明显 > 1 或 gamma  明
显 < 1，说明模型确实需要加强尾部）
3. OOF MAE (segmented raw)  和 after bin-median calibration （校准是否真的在 OOF
上稳定改善，而不是瞎抖）
如果你发现：
tail_rate  很低（比如 < 3%），那 TAIL_Q=0.90  太高了，改成 0.85  让 tail 有更多样
本。
最优 k  总是 < 0.8，说明 excess 在过拟合/噪声大，需要把 DEPTH_TAIL_REG  再降，或
者 TFIDF_MAX_FEATURES  下降一点。
    sub = pd.DataFrame({
        "patient_id": test["patient_id"].astype(int).values,
        "ed_cost_next3y_usd": final.astype(np.float32),
    })
    sub.to_csv(SUBMISSION_PATH, index=False)
    logging.info(f"Saved submission: {SUBMISSION_PATH} | postprocess={chosen}")
    return {
        "cap": cap,
        "tail_rate": float(y_tail.mean()),
        "best_k": best_k,
        "best_gamma": best_gamma,
        "oof_mae_raw": float(mae_raw),
        "oof_mae_cal": float(mae_cal),
        "chosen": chosen,
        "submission_path": str(SUBMISSION_PATH),
    }

我知道你很火大，但这次我们做的是数学上真正不同的分解：
不是“加更多乱七八糟特征”，而是把问题拆成主体 + 尾部，然后用 MAE 对齐的中位数回归
+ 分段残差校准去专门处理极端值。
你先把 Iter9 跑起来上榜（哪怕只看一次结果），我们再决定下一步是否要把阈值/校准桶数
做一个小网格搜索（那也仍然是“简单数学”，不是复杂特征工程）。
```

### C15 — ITERATION 10 — Claims-aware ED cost forecasting (Domain+Data-Driven) (pages 396-427)
```python
# -*- coding: utf-8 -*-
r"""
ITERATION 10 — Claims-aware ED cost forecasting (Domain+Data-Driven)
===================================================================
What changed vs your previous attempts:
1) Treat receipts as claims: build "structure features" from line items (qty, lin
2) Keep sparse TF-IDF on CPT/HCPCS codes (no SVD, no embeddings).
3) Add OOF target-encoding risk scores per code (smoothed median) -> patient aggr
4) Minimal, high-signal domain markers (ED level codes 99281-99285, critical care 
   (NOT CPT range bucketing.)
5) Strong QC: verify pdf coverage and TOTAL==prior_ed_cost_5y_usd.
INSTALL:
pip install -U pandas numpy scipy scikit-learn xgboost pymupdf joblib tqdm

"""
from __future__ import annotations
import re
import sys
import time
import math
import logging
import warnings
import inspect
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import fitz  # PyMuPDF
from scipy.sparse import csr_matrix, hstack
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from xgboost import XGBRegressor
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# -----------------------
# PATHS
# -----------------------
BASE_DIR = Path(r"D:\AgentDs\agent_ds_healthcare")

TRAIN_CSV = BASE_DIR / "ed_cost_train.csv"
TEST_CSV  = BASE_DIR / "ed_cost_test.csv"
PATIENTS_CSV = BASE_DIR / "patients.csv"
PDF_DIR = BASE_DIR / "receipts_pdf"
SUBMISSION_PATH = BASE_DIR / "submission.csv"
CACHE_DIR = BASE_DIR / "cache_iter10"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RECEIPT_CACHE = CACHE_DIR / "receipts_parsed.joblib"
CACHE_VERSION = 101  # bump when you change parsing logic
# -----------------------
# CONFIG
# -----------------------
# TF-IDF
TFIDF_MAX_CODE = 2500
TFIDF_MIN_DF_CODE = 2
TFIDF_NGRAM = (1, 3)
TFIDF_MAX_DESC = 800
TFIDF_MIN_DF_DESC = 2
DESC_NGRAM = (1, 2)
# OOF Target Encoding (code risk)
TE_ALPHA = 50.0   # shrinkage to global median
# CV + Model
N_FOLDS = 5
REPEAT_SEEDS = [2025, 2026]  # 10 fits
EARLY_STOPPING = 200
N_ESTIMATORS = 8000
LR = 0.02
MAX_DEPTH = 6
SUBSAMPLE = 0.85
COLSAMPLE = 0.75
REG_LAMBDA = 1.0
MIN_CHILD_WEIGHT = 1.0

USE_PSEUDOHUBER = True
HUBER_SLOPE = 50.0  # smoother than pure L1, more stable for dollar-scale residua
VERBOSE_EVERY = 100
CLIP_AT_ZERO = True
# -----------------------
# Logging
# -----------------------
def setup_logging():
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname
    else:
        root.setLevel(logging.INFO)
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))
def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_out
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)
def ensure_csr_float32_int32(X: csr_matrix) -> csr_matrix:
    X = X.tocsr()
    if X.data.dtype != np.float32:
        X.data = X.data.astype(np.float32, copy=False)
    if X.indices.dtype != np.int32:
        X.indices = X.indices.astype(np.int32, copy=False)
    if X.indptr.dtype != np.int32:
        X.indptr = X.indptr.astype(np.int32, copy=False)
    return X

# -----------------------
# Receipt parsing
# -----------------------
ZIP_INS_RE = re.compile(r"ZIP3:\s*([0-9]{3})\s+Insurance:\s*([A-Za-z_]+)")
TOTAL_RE   = re.compile(r"TOTAL\s+([\d,]+\.\d{2})")
# matches: code + desc + qty + unit + line_total (synthetic PDFs are consistent)
LINE_RE = re.compile(
    r"^([A-Z]\d{4}|\d{5})\s+(.+?)\s+(\d+)\s+([\d,]+\.\d{2})\s+([\d,]+\.\d{2})\s*$
)
CODE_TOKEN_RE = re.compile(r"\b(?:[A-Z]\d{4}|\d{5})\b")
LEVEL_RE = re.compile(r"\(level\s*([1-5])\)", re.IGNORECASE)
ED_LEVEL_CODES = {"99281":1,"99282":2,"99283":3,"99284":4,"99285":5}
SENTINEL_CODES = {
    "99291":"critical_care_99291",
    "99292":"critical_care_99292",
    "G0378":"obs_G0378",
    "31500":"intub_31500",
    "36556":"central_line_36556",
    "85025":"cbc_85025",
}
def _money(s: str) -> float:
    return float(s.replace(",", ""))
def parse_receipt(patient_id: int) -> Dict[str, Any]:
    pdf_path = PDF_DIR / f"receipt_{patient_id}.pdf"
    out = {
        "patient_id": int(patient_id),
        "pdf_ok": 0,
        "pdf_zip3": "missing",
        "pdf_insurance": "missing",
        "pdf_total": 0.0,
        "pdf_parse_notes": "missing_pdf",
        # numeric structure features
        "n_items": 0,
        "n_unique_codes": 0,
        "qty_sum": 0,

        "line_mean": 0.0,
        "line_max": 0.0,
        "line_std": 0.0,
        "cost_concentration": 0.0,  # max_line / total
        "hhi": 0.0,  # sum((line/total)^2)
        # domain markers
        "ed_em_count": 0,
        "ed_em_level_avg": 0.0,
        "ed_em_high_ratio": 0.0,
    }
    # sentinel counts + cost shares
    for k in SENTINEL_CODES.values():
        out[f"{k}_count"] = 0
        out[f"{k}_cost_share"] = 0.0
    # for downstream
    out["codes_doc_qty"] = ""
    out["codes_doc_cost"] = ""
    out["desc_doc"] = ""
    out["_code_qty_dict"] = {}   # code -> qty_sum
    out["_code_cost_dict"] = {}  # code -> line_total_sum
    if not pdf_path.exists():
        return out
    try:
        with fitz.open(pdf_path) as doc:
            text = ""
            for page in doc:
                text += (page.get_text("text") or "") + "\n"
    except Exception as e:
        out["pdf_parse_notes"] = f"open_error:{type(e).__name__}"
        return out
    text = text or ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    m = ZIP_INS_RE.search(text)
    if m:
        out["pdf_zip3"] = str(m.group(1))

        out["pdf_insurance"] = str(m.group(2)).lower()
    m = TOTAL_RE.search(text)
    if m:
        try:
            out["pdf_total"] = _money(m.group(1))
        except Exception:
            out["pdf_total"] = 0.0
    code_qty = defaultdict(int)
    code_cost = defaultdict(float)
    items = []
    desc_tokens = []
    line_totals = []
    for ln in lines:
        mm = LINE_RE.match(ln)
        if not mm:
            continue
        code, desc, qty_s, unit_s, line_s = mm.groups()
        qty = int(qty_s)
        qty = max(1, min(qty, 50))
        try:
            line_total = _money(line_s)
        except Exception:
            line_total = 0.0
        code_qty[code] += qty
        code_cost[code] += line_total
        items.append((code, desc, qty, line_total))
        # description doc
        if desc:
            desc_tokens.append(desc)
        if line_total > 0:
            line_totals.append(line_total)
    # fallback if table parse failed: extract code tokens only
    if len(items) == 0:

        codes = CODE_TOKEN_RE.findall(text)
        for c in codes:
            code_qty[c] += 1
        out["pdf_parse_notes"] = "fallback_codes_only"
    else:
        out["pdf_parse_notes"] = "ok"
    # build docs
    tokens_qty = []
    tokens_cost = []
    for code, q in code_qty.items():
        rep_q = max(1, min(q, 10))
        tokens_qty.extend([code] * rep_q)
    # cost-based repetition (log bucket)
    for code, csum in code_cost.items():
        rep_c = int(np.clip(np.round(np.log1p(max(csum, 0.0))), 1, 12))
        tokens_cost.extend([code] * rep_c)
    out["codes_doc_qty"] = " ".join(tokens_qty)
    out["codes_doc_cost"] = " ".join(tokens_cost)
    out["desc_doc"] = " ".join(desc_tokens).lower()
    out["_code_qty_dict"] = dict(code_qty)
    out["_code_cost_dict"] = dict(code_cost)
    out["pdf_ok"] = 1 if (len(out["codes_doc_qty"]) > 0) else 0
    # structure features
    out["n_items"] = int(len(items))
    out["n_unique_codes"] = int(len(code_qty))
    out["qty_sum"] = int(sum(code_qty.values()))
    if len(line_totals) > 0:
        arr = np.array(line_totals, dtype=np.float32)
        out["line_mean"] = float(arr.mean())
        out["line_max"] = float(arr.max())
        out["line_std"] = float(arr.std())
        tot = float(out["pdf_total"])

        if tot > 0:
            out["cost_concentration"] = float(out["line_max"] / tot)
            share = arr / tot
            out["hhi"] = float(np.sum(share * share))
    # ED E/M level features
    em_counts = 0
    level_sum = 0
    high = 0
    for c, q in code_qty.items():
        if c in ED_LEVEL_CODES:
            em_counts += q
            lv = ED_LEVEL_CODES[c]
            level_sum += lv * q
            if lv >= 4:
                high += q
    out["ed_em_count"] = int(em_counts)
    out["ed_em_level_avg"] = float(level_sum / em_counts) if em_counts > 0 else 0
    out["ed_em_high_ratio"] = float(high / em_counts) if em_counts > 0 else 0.0
    # sentinel features (count + cost share)
    tot = float(out["pdf_total"])
    for code, name in SENTINEL_CODES.items():
        q = int(code_qty.get(code, 0))
        csum = float(code_cost.get(code, 0.0))
        out[f"{name}_count"] = q
        out[f"{name}_cost_share"] = float(csum / tot) if tot > 0 else 0.0
    return out
def load_receipts(patient_ids: List[int]) -> pd.DataFrame:
    cache = {"version": CACHE_VERSION, "data": {}}
    if RECEIPT_CACHE.exists():
        try:
            cache = joblib.load(RECEIPT_CACHE)
        except Exception:
            cache = {"version": CACHE_VERSION, "data": {}}
    if not isinstance(cache, dict) or cache.get("version") != CACHE_VERSION:
        cache = {"version": CACHE_VERSION, "data": {}}

    data = cache.get("data", {})
    missing = [pid for pid in patient_ids if str(pid) not in data]
    logging.info(f"Receipt cache hit={len(patient_ids)-len(missing):,} miss={len(
    for pid in tqdm(missing, desc="Parsing receipts", ncols=100):
        data[str(pid)] = parse_receipt(int(pid))
    cache["data"] = data
    if missing:
        try:
            joblib.dump(cache, RECEIPT_CACHE)
        except Exception as e:
            logging.warning(f"Receipt cache save failed: {e}")
    rows = []
    for pid in patient_ids:
        rows.append(data.get(str(pid), parse_receipt(int(pid))))
    return pd.DataFrame(rows)
# -----------------------
# OOF Target Encoding per code (smoothed median)
# -----------------------
def build_strat_labels(df_train: pd.DataFrame) -> np.ndarray:
    # chronic + prior_cost decile is usually a decent stratification
    if "prior_ed_cost_5y_usd" in df_train.columns:
        b = pd.qcut(df_train["prior_ed_cost_5y_usd"], q=10, duplicates="drop").ast
    else:
        b = pd.Series(["bin"] * len(df_train))
    return (df_train["primary_chronic"].astype(str) + "_" + b).values
def compute_code_te_mapping(code_sets: List[List[str]], y: np.ndarray, alpha: flo
    global_med = float(np.median(y))
    bucket = defaultdict(list)
    for i, codes in enumerate(code_sets):
        for c in codes:
            bucket[c].append(float(y[i]))
    te = {}
    for c, vals in bucket.items():

        n = len(vals)
        med = float(np.median(vals))
        te[c] = (n * med + alpha * global_med) / (n + alpha)
    return te, global_med
def apply_patient_te_features(
    te: Dict[str,float],
    global_med: float,
    code_qty_dicts: List[Dict[str,int]],
    code_cost_dicts: List[Dict[str,float]],
    totals: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for d_qty, d_cost, tot in zip(code_qty_dicts, code_cost_dicts, totals):
        if not d_qty:
            rows.append({
                "te_sum_qty": 0.0, "te_mean_qty": 0.0, "te_max": global_med,
                "te_top3_mean": global_med, "te_sum_costshare": 0.0
            })
            continue
        vals = []
        sum_qty = 0.0
        sum_w = 0.0
        sum_costshare = 0.0
        for c, q in d_qty.items():
            v = float(te.get(c, global_med))
            vals.append(v)
            sum_qty += v * q
            sum_w += q
        # cost-share weighted sum (uses line totals per code)
        if tot and tot > 0:
            for c, cs in d_cost.items():
                v = float(te.get(c, global_med))
                sum_costshare += v * (cs / tot)
        vals_sorted = sorted(vals, reverse=True)
        top3 = vals_sorted[:3] if len(vals_sorted) >= 3 else vals_sorted

        rows.append({
            "te_sum_qty": float(sum_qty),
            "te_mean_qty": float(sum_qty / max(sum_w, 1.0)),
            "te_max": float(vals_sorted[0]) if vals_sorted else global_med,
            "te_top3_mean": float(np.mean(top3)) if top3 else global_med,
            "te_sum_costshare": float(sum_costshare),
        })
    return pd.DataFrame(rows)
def build_oof_te_features(df_train: pd.DataFrame, df_test: pd.DataFrame, alpha: f
    y = df_train["ed_cost_next3y_usd"].values.astype(np.float32)
    # prepare code sets (unique codes per patient) from cached dicts
    train_sets = [list(d.keys()) for d in df_train["_code_qty_dict"].tolist()]
    test_sets  = [list(d.keys()) for d in df_test["_code_qty_dict"].tolist()]
    strat = build_strat_labels(df_train)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=2025)
    oof_feats = pd.DataFrame(index=df_train.index, columns=["te_sum_qty","te_mean_
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(df_train)), st
        te_map, gmed = compute_code_te_mapping([train_sets[i] for i in tr_idx], y
        f_va = apply_patient_te_features(
            te_map, gmed,
            [df_train["_code_qty_dict"].iloc[i] for i in va_idx],
            [df_train["_code_cost_dict"].iloc[i] for i in va_idx],
            df_train["pdf_total"].iloc[va_idx].values.astype(float),
        )
        oof_feats.iloc[va_idx] = f_va.values
        logging.info(f"TE fold {fold} | built mapping codes={len(te_map):,}")
    # test feats from full train mapping
    te_map_full, gmed_full = compute_code_te_mapping(train_sets, y, alpha=alpha)
    test_feats = apply_patient_te_features(
        te_map_full, gmed_full,
        df_test["_code_qty_dict"].tolist(),
        df_test["_code_cost_dict"].tolist(),
        df_test["pdf_total"].values.astype(float),
    )

    return oof_feats.reset_index(drop=True), test_feats.reset_index(drop=True)
# -----------------------
# Feature matrix build
# -----------------------
def safe_tfidf_fit_transform(vec, texts: np.ndarray):
    try:
        X = vec.fit_transform(texts.astype(str))
        return ensure_csr_float32_int32(X)
    except ValueError:
        return csr_matrix((len(texts), 0), dtype=np.float32)
def safe_tfidf_transform(vec, texts: np.ndarray):
    try:
        X = vec.transform(texts.astype(str))
        return ensure_csr_float32_int32(X)
    except Exception:
        return csr_matrix((len(texts), 0), dtype=np.float32)
def build_feature_matrix(df_all: pd.DataFrame) -> Tuple[csr_matrix, ColumnTransfo
    # numeric derivations
    df = df_all.copy()
    if "prior_ed_cost_5y_usd" in df.columns and "prior_ed_visits_5y" in df.column
        df["prior_cost_per_visit"] = df["prior_ed_cost_5y_usd"] / (df["prior_ed_v
        df["log_prior_cost"] = np.log1p(df["prior_ed_cost_5y_usd"].astype(float))
        df["log_prior_visits"] = np.log1p(df["prior_ed_visits_5y"].astype(float))
    # mismatch checks if patients.csv fields exist
    if "insurance" in df.columns:
        df["ins_mismatch"] = ((df["pdf_insurance"].astype(str) != df["insurance"]
    else:
        df["ins_mismatch"] = 0
    if "zip3" in df.columns:
        df["zip_mismatch"] = ((df["pdf_zip3"].astype(str) != df["zip3"].astype(st
    else:
        df["zip_mismatch"] = 0

    df["total_absdiff_prior"] = np.abs(df["pdf_total"].astype(float) - df.get("pr
    # define cat + num
    doc_cols = {"codes_doc_qty","codes_doc_cost","desc_doc"}
    exclude = {"patient_id","ed_cost_next3y_usd","is_train","_code_qty_dict","_co
    cat_cols = []
    for c in ["primary_chronic","sex","insurance","zip3","pdf_insurance","pdf_zip
        if c in df.columns:
            cat_cols.append(c)
    for c in df.columns:
        if c in exclude or c in cat_cols:
            continue
        if df[c].dtype == "object" or str(df[c].dtype).startswith("string"):
            cat_cols.append(c)
    num_cols = []
    for c in df.columns:
        if c in exclude or c in cat_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
    # fill cats
    for c in cat_cols:
        df[c] = df[c].astype("string").fillna("missing")
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_co
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("ohe", make_ohe())]), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    X_tab = pre.fit_transform(df)
    X_tab = ensure_csr_float32_int32(X_tab.tocsr() if hasattr(X_tab, "tocsr") els

    # TF-IDF code docs
    vec_code = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b(?:[A-Z]\d{4}|\d{5})\b",
        lowercase=False,
        ngram_range=TFIDF_NGRAM,
        max_features=TFIDF_MAX_CODE,
        min_df=TFIDF_MIN_DF_CODE,
        max_df=1.0,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    # fit on both qty and cost docs separately but same vectorizer? we use two ve
    vec_cost = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b(?:[A-Z]\d{4}|\d{5})\b",
        lowercase=False,
        ngram_range=(1,2),
        max_features=int(TFIDF_MAX_CODE * 0.7),
        min_df=TFIDF_MIN_DF_CODE,
        max_df=1.0,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    X_code = safe_tfidf_fit_transform(vec_code, df["codes_doc_qty"].values)
    X_cost = safe_tfidf_fit_transform(vec_cost, df["codes_doc_cost"].values)
    # TF-IDF on descriptions (optional small)
    vec_desc = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
        lowercase=True,
        ngram_range=DESC_NGRAM,
        max_features=TFIDF_MAX_DESC,
        min_df=TFIDF_MIN_DF_DESC,
        max_df=1.0,
        sublinear_tf=True,

        norm="l2",
        dtype=np.float32,
    )
    X_desc = safe_tfidf_fit_transform(vec_desc, df["desc_doc"].values)
    X = ensure_csr_float32_int32(hstack([X_tab, X_code, X_cost, X_desc], format="
    meta = {"num_cols": num_cols, "cat_cols": cat_cols}
    vecs = {"vec_code": vec_code, "vec_cost": vec_cost, "vec_desc": vec_desc}
    return X, pre, vecs, meta
# -----------------------
# Calibration (median residual, shrink)
# -----------------------
def shrunk_group_median_bias(resid: np.ndarray, group: np.ndarray, alpha: float) 
    g = group.astype(str)
    global_med = float(np.median(resid))
    out = {}
    for key in np.unique(g):
        m = (g == key)
        r = resid[m]
        if r.size == 0:
            continue
        med = float(np.median(r))
        n = float(r.size)
        out[str(key)] = (n * med + alpha * global_med) / (n + alpha)
    return out, global_med
def apply_bias(pred: np.ndarray, group: np.ndarray, bias: Dict[str,float], default
    g = group.astype(str)
    p = pred.astype(np.float64).copy()
    for i in range(len(p)):
        p[i] += bias.get(str(g[i]), default_bias)
    return p.astype(np.float32)
# -----------------------
# Runner
# -----------------------
def run_iteration10(diagnostic: bool = False, force_cpu: bool = False) -> Dict[st

    setup_logging()
    logging.info(f"XGBoost version: {getattr(xgb,'__version__','unknown')}")
    logging.info("=== Iteration 10: claims-aware + OOF code risk + sparse TFIDF + 
    logging.info(f"diagnostic={diagnostic} | force_cpu={force_cpu}")
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    logging.info(f"Loaded train={train.shape} test={test.shape}")
    if PATIENTS_CSV.exists():
        pat = pd.read_csv(PATIENTS_CSV)
        if "patient_id" in pat.columns:
            train = train.merge(pat, on="patient_id", how="left")
            test  = test.merge(pat, on="patient_id", how="left")
            logging.info(f"Merged patients.csv: train={train.shape} test={test.sh
    # PDF coverage QC
    train_ids = train["patient_id"].astype(int).tolist()
    test_ids  = test["patient_id"].astype(int).tolist()
    def pdf_exists(pid): return (PDF_DIR / f"receipt_{pid}.pdf").exists()
    cov_tr = np.mean([pdf_exists(pid) for pid in train_ids])
    cov_te = np.mean([pdf_exists(pid) for pid in test_ids])
    logging.info(f"PDF coverage | train={cov_tr:.3f} test={cov_te:.3f}")
    if cov_te < 0.98:
        logging.warning("TEST PDF coverage is not ~100%. Receipt-based models may 
    # parse receipts (train+test)
    all_ids = list(map(int, pd.concat([train["patient_id"], test["patient_id"]]).v
    rec = load_receipts(all_ids)
    train = train.merge(rec, on="patient_id", how="left")
    test  = test.merge(rec, on="patient_id", how="left")
    # basic TOTAL sanity (sample)
    sample = train.sample(min(200, len(train)), random_state=7)
    absdiff = np.abs(sample["pdf_total"].astype(float) - sample["prior_ed_cost_5y_
    logging.info(f"TOTAL sanity (sample n={len(sample)}): absdiff median={absdiff
    # add OOF target encoding features

    train_te, test_te = build_oof_te_features(train, test, alpha=TE_ALPHA)
    for c in train_te.columns:
        train[c] = train_te[c].values
        test[c] = test_te[c].values
    # build docs fill
    for df in [train, test]:
        for c in ["codes_doc_qty","codes_doc_cost","desc_doc"]:
            df[c] = df[c].fillna("").astype(str)
    if diagnostic:
        return {
            "pdf_cov_train": float(cov_tr),
            "pdf_cov_test": float(cov_te),
            "total_absdiff_sample_median": float(absdiff.median()),
            "total_absdiff_sample_max": float(absdiff.max()),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "nonempty_codes_doc_qty_train": int((train["codes_doc_qty"].str.len() 
            "nonempty_codes_doc_qty_test": int((test["codes_doc_qty"].str.len() > 
        }
    # build matrix on all (transductive, unsupervised)
    train["is_train"] = 1
    test["is_train"] = 0
    df_all = pd.concat([train, test], ignore_index=True)
    logging.info("Building sparse feature matrix (tabular OHE + TFIDF codes + TFI
    X_all, pre, vecs, meta = build_feature_matrix(df_all)
    logging.info(f"X_all shape={X_all.shape} nnz={X_all.nnz:,}")
    is_train = (df_all["is_train"].values.astype(int) == 1)
    X_tr_all = X_all[is_train]
    X_te_all = X_all[~is_train]
    y = train["ed_cost_next3y_usd"].values.astype(np.float32)
    # stratification: chronic + cost decile
    if "prior_ed_cost_5y_usd" in train.columns:
        b = pd.qcut(train["prior_ed_cost_5y_usd"], q=10, duplicates="drop").astyp

    else:
        b = pd.Series(["bin"] * len(train))
    strat = (train["primary_chronic"].astype(str) + "_" + b).values
    device = "cpu" if force_cpu else "cuda"
    params = dict(
        objective="reg:absoluteerror",
        eval_metric="mae",
        tree_method="hist",
        device=device,
        n_estimators=N_ESTIMATORS,
        learning_rate=LR,
        max_depth=MAX_DEPTH,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE,
        reg_lambda=REG_LAMBDA,
        min_child_weight=MIN_CHILD_WEIGHT,
        n_jobs=-1,
        verbosity=1,
        early_stopping_rounds=EARLY_STOPPING,
    )
    if USE_PSEUDOHUBER:
        params["objective"] = "reg:pseudohubererror"
        params["huber_slope"] = HUBER_SLOPE
    oof_sum = np.zeros(len(train), dtype=np.float64)
    oof_cnt = np.zeros(len(train), dtype=np.float64)
    te_sum  = np.zeros(len(test), dtype=np.float64)
    n_models = 0
    splits = []
    for rep, seed in enumerate(REPEAT_SEEDS):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(train)), st
            splits.append((rep, fold, seed, tr_idx, va_idx))
    logging.info(f"Total fits: {len(splits)} | device={device} | objective={param
    # fit loop

    for rep, fold, seed, tr_idx, va_idx in tqdm(splits, desc="CV", ncols=110):
        tag = f"rep={rep} fold={fold}"
        t0 = time.time()
        X_tr = X_tr_all[tr_idx]
        y_tr = y[tr_idx]
        X_va = X_tr_all[va_idx]
        y_va = y[va_idx]
        model = XGBRegressor(**params, random_state=10000 * rep + seed + fold)
        fit_sig = inspect.signature(model.fit).parameters
        fit_kwargs = {"eval_set": [(X_va, y_va)]}
        if "verbose" in fit_sig:
            fit_kwargs["verbose"] = VERBOSE_EVERY
        elif "verbose_eval" in fit_sig:
            fit_kwargs["verbose_eval"] = VERBOSE_EVERY
        logging.info(f"{tag} | fit START")
        try:
            model.fit(X_tr, y_tr, **fit_kwargs)
        except Exception as e:
            # fallback if pseudohuber not supported
            if params.get("objective") == "reg:pseudohubererror":
                logging.warning(f"{tag} | pseudohuber failed ({type(e).__name__})
                params2 = dict(params)
                params2["objective"] = "reg:absoluteerror"
                params2.pop("huber_slope", None)
                model = XGBRegressor(**params2, random_state=10000 * rep + seed + 
                model.fit(X_tr, y_tr, **fit_kwargs)
            else:
                raise
        logging.info(f"{tag} | fit END | best_iteration={getattr(model,'best_iter
        p_va = model.predict(X_va).astype(np.float32)
        p_te = model.predict(X_te_all).astype(np.float32)
        if CLIP_AT_ZERO:
            p_va = np.maximum(p_va, 0.0)

            p_te = np.maximum(p_te, 0.0)
        oof_sum[va_idx] += p_va
        oof_cnt[va_idx] += 1.0
        te_sum += p_te
        n_models += 1
        fold_mae = mae(y_va, p_va)
        logging.info(f"{tag} | fold_MAE={fold_mae:.3f} | fold_time={time.time()-t
    oof = (oof_sum / np.maximum(oof_cnt, 1.0)).astype(np.float32)
    pred_test = (te_sum / max(1, n_models)).astype(np.float32)
    raw_mae = mae(y, oof)
    logging.info(f"OOF MAE (raw): {raw_mae:.4f}")
    # Calibration candidates (median residual shrink)
    resid = (y.astype(np.float64) - oof.astype(np.float64))
    # (1) none
    best = (raw_mae, "none", pred_test)
    # (2) chronic
    bias_c, def_c = shrunk_group_median_bias(resid, train["primary_chronic"].asty
    oof_c = apply_bias(oof, train["primary_chronic"].astype(str).values, bias_c, 
    if CLIP_AT_ZERO: oof_c = np.maximum(oof_c, 0.0)
    mae_c = mae(y, oof_c)
    logging.info(f"OOF MAE (chronic calib): {mae_c:.4f}")
    if mae_c < best[0]:
        pred_c = apply_bias(pred_test, test["primary_chronic"].astype(str).values
        if CLIP_AT_ZERO: pred_c = np.maximum(pred_c, 0.0)
        best = (mae_c, "chronic", pred_c)
    # (3) chronic+insurance (if available)
    if "insurance" in train.columns and "insurance" in test.columns:
        key_tr = (train["primary_chronic"].astype(str) + "_" + train["insurance"]
        key_te = (test["primary_chronic"].astype(str) + "_" + test["insurance"].a
        bias_ci, def_ci = shrunk_group_median_bias(resid, key_tr, alpha=TE_ALPHA)
        oof_ci = apply_bias(oof, key_tr, bias_ci, def_ci)
        if CLIP_AT_ZERO: oof_ci = np.maximum(oof_ci, 0.0)

你现在应该怎么做（一步步，不再瞎试）
1. 先跑诊断：
你要重点看：
pdf_cov_test  是否接近 1.0
nonempty_codes_doc_qty_test  是否接近 test 行数
        mae_ci = mae(y, oof_ci)
        logging.info(f"OOF MAE (chronic+ins calib): {mae_ci:.4f}")
        if mae_ci < best[0]:
            pred_ci = apply_bias(pred_test, key_te, bias_ci, def_ci)
            if CLIP_AT_ZERO: pred_ci = np.maximum(pred_ci, 0.0)
            best = (mae_ci, "chronic+insurance", pred_ci)
    chosen_mae, chosen_name, final_test = best
    logging.info(f"Chosen calibration: {chosen_name} | OOF MAE={chosen_mae:.4f}")
    sub = pd.DataFrame({
        "patient_id": test["patient_id"].astype(int).values,
        "ed_cost_next3y_usd": final_test.astype(np.float32),
    })
    sub.to_csv(SUBMISSION_PATH, index=False)
    logging.info(f"Saved submission: {SUBMISSION_PATH}")
    return {
        "oof_mae_raw": float(raw_mae),
        "oof_mae_best": float(chosen_mae),
        "calibration": chosen_name,
        "n_models": int(n_models),
        "submission_path": str(SUBMISSION_PATH),
    }
python
res = run_iteration10(diagnostic=True)
res
Copy code

total_absdiff_sample_max  是否≈0
2. 再训练：
3. 如果 pdf_cov_test < 0.98  或 nonempty_codes_doc_qty_test  很低：
你现在所有“吃 receipt” 的模型都会线上崩，这就是 Iter9 OOF 好但榜单差的最常见解
释。
先把 test receipts 补齐/放对位置，再谈模型。
关于“要不要本地 LLM？”
就你这个数据形态（synthetic billing summary、短描述、强结构表格），不需要 LLM。
LLM 适合的是：长自由文本、强语义推断、或需要抽取复杂实体关系。
这里最值钱的是：
code 本身（稀疏高维）
line_total/qty 的结构统计
极少数强信号码（E/M level、critical care、插管等）
这些用规则+统计学习就够了，而且更稳、更快。
如果你愿意，把 Iteration10 diagnostic 的输出（尤其 pdf_cov_test ）贴出来，我可以立刻
判断你们现在的最大瓶颈到底是 数据覆盖问题 还是 模型/特征问题。
python
res = run_iteration10(diagnostic=False)
res
Copy code
```

### C16 — ITERATION 11 — Leakage-safe code risk + robust PDF parsing + monolithic XGBoost ( (pages 428-453)
```python
# -*- coding: utf-8 -*-
r"""
ITERATION 11 — Leakage-safe code risk + robust PDF parsing + monolithic XGBoost (
=================================================================================
Why this should beat / stabilize vs your current best:

1) Fold-safe target encoding: any target-derived features must be built INSIDE ea
   otherwise CV becomes overly optimistic (classic target encoding leakage issue)
2) Robust PDF parsing: PyMuPDF text for these synthetic receipts is column-wise (
   so naive single-line regex loses qty/line totals. We reconstruct rows from ext
3) Add claim-structure features + per-code pivot counts/shares (NO manual CPT ran
4) Keep single XGBoost reg:absoluteerror on GPU + chronic(+insurance) median-resi
Run in notebook:
    res = run_iteration11(diagnostic=True)   # quick checks
    res = run_iteration11(diagnostic=False)  # train + write submission
INSTALL:
pip install -U pandas numpy scipy scikit-learn xgboost pymupdf joblib tqdm
"""
from __future__ import annotations
import time
import logging
import warnings
import inspect
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import fitz  # PyMuPDF
from scipy.sparse import csr_matrix, hstack
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold

import xgboost as xgb
from xgboost import XGBRegressor
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# =========================
# PATHS (edit if needed)
# =========================
BASE_DIR = Path(r"D:\AgentDs\agent_ds_healthcare")
TRAIN_CSV = BASE_DIR / "ed_cost_train.csv"
TEST_CSV  = BASE_DIR / "ed_cost_test.csv"
PATIENTS_CSV = BASE_DIR / "patients.csv"
PDF_DIR = BASE_DIR / "receipts_pdf"
SUBMISSION_PATH = BASE_DIR / "submission_ICHI_V1.csv"
CACHE_DIR = BASE_DIR / "cache_iter11"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RECEIPT_CACHE = CACHE_DIR / "receipts_parsed_iter11.joblib"
CACHE_VERSION = 111
# =========================
# CONFIG
# =========================
# TF-IDF
TFIDF_MAX_QTY = 1600
TFIDF_MAX_COST = 900
TFIDF_NGRAM_QTY = (1, 3)
TFIDF_NGRAM_COST = (1, 2)
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 1.0
# Per-code pivots
TOP_CODE_PIVOTS = 40  # safe upper bound; actual unique codes ~ small in this syn

# Target encoding
TE_ALPHA = 80.0              # smoothing strength (higher => less overfit)
TE_USE_GROUP = True          # group-conditioned TE by chronic+insurance (still d
# CV
N_FOLDS = 5
REPEAT_SEEDS = [2025, 2026]  # 10 models total
EARLY_STOPPING = 200
# XGB
N_ESTIMATORS = 8000
LR = 0.02
MAX_DEPTH = 6
SUBSAMPLE = 0.85
COLSAMPLE = 0.75
REG_LAMBDA = 1.0
MIN_CHILD_WEIGHT = 1.0
GAMMA = 0.0
# Runtime / safety
DEVICE = "cuda"              # set to "cpu" if needed
CLIP_AT_ZERO = True
VERBOSE_EVERY = 100          # fit heartbeat
# =========================
# LOGGING
# =========================
def setup_logging() -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname
    else:
        root.setLevel(logging.INFO)
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_out
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)
def ensure_csr_float32_int32(X: csr_matrix) -> csr_matrix:
    X = X.tocsr()
    if X.data.dtype != np.float32:
        X.data = X.data.astype(np.float32, copy=False)
    if X.indices.dtype != np.int32:
        X.indices = X.indices.astype(np.int32, copy=False)
    if X.indptr.dtype != np.int32:
        X.indptr = X.indptr.astype(np.int32, copy=False)
    return X
# =========================
# PDF PARSING (robust for synthetic layout)
# =========================
ZIP_INS_RE = re.compile(r"ZIP3:\s*([0-9]{3})\s+Insurance:\s*([A-Za-z_]+)", re.IGNO
TOTAL_RE   = re.compile(r"TOTAL\s*([\d,]+\.\d{2})", re.IGNORECASE)
CODE_FULL_RE = re.compile(r"^(?:[A-Z]\d{4}|\d{5})$")
CODE_TOKEN_RE = re.compile(r"\b(?:[A-Z]\d{4}|\d{5})\b")
MONEY_RE = re.compile(r"^[\d,]+\.\d{2}$")
LEVEL_RE = re.compile(r"\(level\s*([1-5])\)", re.IGNORECASE)
def _money_to_float(s: str) -> float:
    return float(s.replace(",", ""))
def parse_receipt_table(lines: List[str]) -> List[Tuple[str, int, float, float, s
    """
    Extract table rows from column-wise PDF text:
      code, description (may span lines), qty, unit, line_total
    Returns list of tuples: (code, qty, unit, line_total, desc)
    """

    items: List[Tuple[str, int, float, float, str]] = []
    i, n = 0, len(lines)
    while i < n:
        s = lines[i].strip()
        if CODE_FULL_RE.match(s):
            code = s
            j = i + 1
            desc_parts: List[str] = []
            # gather description lines until we hit qty or TOTAL
            while j < n:
                sj = lines[j].strip()
                if sj.isdigit() or sj.upper().startswith("TOTAL"):
                    break
                # skip headers
                if sj in {"Code", "Description", "Qty", "Unit ($)", "Line Total (
                    j += 1
                    continue
                if sj:
                    desc_parts.append(sj)
                j += 1
            # expect qty/unit/line_total next
            if j + 2 < n:
                qty_s = lines[j].strip()
                unit_s = lines[j + 1].strip()
                line_s = lines[j + 2].strip()
                if qty_s.isdigit() and MONEY_RE.match(unit_s) and MONEY_RE.match(
                    qty = int(qty_s)
                    qty = max(1, min(qty, 50))
                    unit = _money_to_float(unit_s)
                    line = _money_to_float(line_s)
                    desc = " ".join(desc_parts).strip()
                    items.append((code, qty, unit, line, desc))
                    i = j + 3
                    continue

        i += 1
    return items
def parse_receipt_pdf(patient_id: int) -> Dict[str, Any]:
    pdf_path = PDF_DIR / f"receipt_{patient_id}.pdf"
    out: Dict[str, Any] = {
        "patient_id": int(patient_id),
        "pdf_ok": 0,
        "pdf_zip3": "missing",
        "pdf_insurance": "missing",
        "pdf_total_parsed": np.nan,
        "parse_ok": 0,
        "n_items": 0,
        "n_unique_codes": 0,
        "qty_sum": 0,
        "line_sum": 0.0,
        "line_mean": 0.0,
        "line_max": 0.0,
        "line_std": 0.0,
        "cost_concentration": 0.0,
        "hhi": 0.0,
        "ed_level_avg": 0.0,
        "ed_level_high_ratio": 0.0,
        "codes_doc_qty": "",
        "codes_doc_cost": "",
        "desc_doc": "",
        "code_qty_dict": {},
        "code_cost_dict": {},
    }
    if not pdf_path.exists():
        return out
    try:
        with fitz.open(pdf_path) as doc:
            text = ""
            for page in doc:

                text += (page.get_text("text") or "") + "\n"
    except Exception:
        return out
    text = text or ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    m = ZIP_INS_RE.search(text)
    if m:
        out["pdf_zip3"] = str(m.group(1))
        out["pdf_insurance"] = str(m.group(2)).lower()
    m = TOTAL_RE.search(text)
    if m:
        try:
            out["pdf_total_parsed"] = _money_to_float(m.group(1))
        except Exception:
            out["pdf_total_parsed"] = np.nan
    items = parse_receipt_table(lines)
    # If parsing failed, fallback to just code tokens (still helpful for TF-IDF)
    code_qty: Dict[str, int] = defaultdict(int)
    code_cost: Dict[str, float] = defaultdict(float)
    if items:
        out["parse_ok"] = 1
        out["n_items"] = int(len(items))
        qty_tokens: List[str] = []
        cost_tokens: List[str] = []
        desc_tokens: List[str] = []
        line_totals: List[float] = []
        lvl_list: List[int] = []
        for code, qty, unit, line, desc in items:
            code_qty[code] += qty
            code_cost[code] += line
            rep_q = max(1, min(qty, 10))

            qty_tokens.extend([code] * rep_q)
            rep_c = int(np.clip(round(np.log1p(max(line, 0.0))), 1, 12))
            cost_tokens.extend([code] * rep_c)
            if desc:
                desc_tokens.append(desc.lower())
                mm = LEVEL_RE.search(desc)
                if mm:
                    lv = int(mm.group(1))
                    lvl_list.extend([lv] * qty)
            if line > 0:
                line_totals.append(line)
        out["codes_doc_qty"] = " ".join(qty_tokens)
        out["codes_doc_cost"] = " ".join(cost_tokens) if cost_tokens else out["co
        out["desc_doc"] = " ".join(desc_tokens)
        out["qty_sum"] = int(sum(code_qty.values()))
        out["n_unique_codes"] = int(len(code_qty))
        if line_totals:
            arr = np.array(line_totals, dtype=np.float32)
            out["line_sum"] = float(arr.sum())
            out["line_mean"] = float(arr.mean())
            out["line_max"] = float(arr.max())
            out["line_std"] = float(arr.std())
            tot = float(out["pdf_total_parsed"]) if (np.isfinite(out["pdf_total_p
            if tot > 0:
                out["cost_concentration"] = float(out["line_max"] / tot)
                share = arr / tot
                out["hhi"] = float(np.sum(share * share))
        if lvl_list:
            out["ed_level_avg"] = float(np.mean(lvl_list))
            out["ed_level_high_ratio"] = float(np.mean([1.0 if lv >= 4 else 0.0 f
    else:
        codes = CODE_TOKEN_RE.findall(text)

        for c in codes:
            code_qty[c] += 1
        out["codes_doc_qty"] = " ".join(codes)
        out["codes_doc_cost"] = out["codes_doc_qty"]
        out["desc_doc"] = ""
        out["qty_sum"] = int(sum(code_qty.values()))
        out["n_unique_codes"] = int(len(code_qty))
    out["code_qty_dict"] = dict(code_qty)
    out["code_cost_dict"] = dict(code_cost)
    out["pdf_ok"] = 1 if len(out["codes_doc_qty"]) > 0 else 0
    return out
def load_receipts(patient_ids: List[int]) -> pd.DataFrame:
    cache = {"version": CACHE_VERSION, "data": {}}
    if RECEIPT_CACHE.exists():
        try:
            cache = joblib.load(RECEIPT_CACHE)
        except Exception:
            cache = {"version": CACHE_VERSION, "data": {}}
    if not isinstance(cache, dict) or cache.get("version") != CACHE_VERSION:
        cache = {"version": CACHE_VERSION, "data": {}}
    data = cache.get("data", {})
    missing = [pid for pid in patient_ids if str(pid) not in data]
    logging.info(f"Receipt cache | hit={len(patient_ids)-len(missing):,} miss={le
    for pid in tqdm(missing, desc="Parsing receipts", ncols=100):
        data[str(pid)] = parse_receipt_pdf(int(pid))
    cache["data"] = data
    if missing:
        try:
            joblib.dump(cache, RECEIPT_CACHE, compress=3)
        except Exception as e:
            logging.warning(f"Receipt cache save failed: {e}")
    rows = [data.get(str(pid), parse_receipt_pdf(int(pid))) for pid in patient_id
    return pd.DataFrame(rows)

# =========================
# CODE PIVOT FEATURES
# =========================
def _sanitize_code(code: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(code))
def add_code_pivots(df: pd.DataFrame, codes: List[str], total_col: str) -> pd.Dat
    dict_qty = df["code_qty_dict"].tolist()
    dict_cost = df["code_cost_dict"].tolist()
    total = df[total_col].astype(float).values
    n = len(df)
    for code in codes:
        col = _sanitize_code(code)
        qty_arr = np.fromiter((d.get(code, 0) if isinstance(d, dict) else 0 for d 
        cost_arr = np.fromiter((d.get(code, 0.0) if isinstance(d, dict) else 0.0 
        df[f"qty_{col}"] = qty_arr
        df[f"cost_{col}"] = cost_arr
        df[f"share_{col}"] = cost_arr / (total + 1e-6)
    return df
# =========================
# TARGET ENCODING (fold-safe)
# =========================
def build_code_te_maps(
    df: pd.DataFrame,
    idx_tr: np.ndarray,
    y: np.ndarray,
    group_key: Optional[np.ndarray],
    alpha: float,
) -> Tuple[Dict[str, float], float, Optional[Dict[Tuple[str, str], float]], Optio
    y_tr = y[idx_tr]
    global_med = float(np.median(y_tr))
    bucket = defaultdict(list)
    for i in idx_tr:

        d = df.at[i, "code_qty_dict"]
        if not isinstance(d, dict):
            continue
        for c in d.keys():
            bucket[c].append(float(y[i]))
    code_map = {}
    for c, vals in bucket.items():
        n = len(vals)
        med = float(np.median(vals))
        code_map[c] = (n * med + alpha * global_med) / (n + alpha)
    grp_code_map = None
    grp_med = None
    if group_key is not None:
        grp_code_map = {}
        grp_med = {}
        g_tr = group_key[idx_tr].astype(str)
        for g in np.unique(g_tr):
            m = (g_tr == g)
            grp_med[g] = float(np.median(y_tr[m])) if np.any(m) else global_med
        bucket_gc = defaultdict(list)
        for i in idx_tr:
            g = str(group_key[i])
            d = df.at[i, "code_qty_dict"]
            if not isinstance(d, dict):
                continue
            for c in d.keys():
                bucket_gc[(g, c)].append(float(y[i]))
        for (g, c), vals in bucket_gc.items():
            n = len(vals)
            med = float(np.median(vals))
            base = grp_med.get(g, global_med)
            grp_code_map[(g, c)] = (n * med + alpha * base) / (n + alpha)
    return code_map, global_med, grp_code_map, grp_med

def patient_te_features(
    df: pd.DataFrame,
    indices: np.ndarray,
    code_map: Dict[str, float],
    global_med: float,
    grp_code_map: Optional[Dict[Tuple[str, str], float]],
    grp_med: Optional[Dict[str, float]],
    group_key: Optional[np.ndarray],
    total_col: str,
) -> np.ndarray:
    totals = df[total_col].astype(float).values
    use_group = (TE_USE_GROUP and grp_code_map is not None and group_key is not N
    out = np.zeros((len(indices), 10 if use_group else 5), dtype=np.float32)
    for r, i in enumerate(indices):
        d_qty = df.at[i, "code_qty_dict"]
        d_cost = df.at[i, "code_cost_dict"]
        if not isinstance(d_qty, dict):
            d_qty = {}
        if not isinstance(d_cost, dict):
            d_cost = {}
        tot = float(totals[i])
        codes = list(d_qty.keys())
        if not codes:
            base = global_med
            out[r, :5] = base
            if out.shape[1] > 5:
                out[r, 5:] = base
            continue
        vals = []
        w_qty = []
        w_cost = []
        for c in codes:
            v = float(code_map.get(c, global_med))
            vals.append(v)
            w_qty.append(float(d_qty.get(c, 0)))

            w_cost.append(float(d_cost.get(c, 0.0)) / tot if tot > 0 else 0.0)
        vals_arr = np.array(vals, dtype=np.float32)
        w_qty_arr = np.array(w_qty, dtype=np.float32)
        w_cost_arr = np.array(w_cost, dtype=np.float32)
        vals_sorted = np.sort(vals_arr)[::-1]
        top3 = vals_sorted[:3]
        mean_v = float(np.mean(vals_arr))
        max_v = float(vals_sorted[0])
        top3_mean = float(np.mean(top3))
        wq = float(np.average(vals_arr, weights=w_qty_arr)) if np.sum(w_qty_arr) 
        wc = float(np.average(vals_arr, weights=w_cost_arr)) if np.sum(w_cost_arr
        out[r, 0] = mean_v
        out[r, 1] = max_v
        out[r, 2] = top3_mean
        out[r, 3] = wq
        out[r, 4] = wc
        if use_group:
            g = str(group_key[i])
            base_g = float(grp_med.get(g, global_med))
            vals_g = [float(grp_code_map.get((g, c), base_g)) for c in codes]
            vals_g = np.array(vals_g, dtype=np.float32)
            vals_g_sorted = np.sort(vals_g)[::-1]
            top3_g = vals_g_sorted[:3]
            out[r, 5] = float(np.mean(vals_g))
            out[r, 6] = float(vals_g_sorted[0])
            out[r, 7] = float(np.mean(top3_g))
            out[r, 8] = out[r, 5] - out[r, 0]
            out[r, 9] = out[r, 6] - out[r, 1]
    return out
# =========================
# PREPROCESSORS
# =========================
def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransfo

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_co
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("ohe", make_ohe())]), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
def fit_vectorizers_and_transform(
    tr_text_qty: np.ndarray, va_text_qty: np.ndarray, te_text_qty: np.ndarray,
    tr_text_cost: np.ndarray, va_text_cost: np.ndarray, te_text_cost: np.ndarray,
) -> Tuple[csr_matrix, csr_matrix, csr_matrix]:
    vec_qty = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b(?:[A-Z]\d{4}|\d{5})\b",
        lowercase=False,
        ngram_range=TFIDF_NGRAM_QTY,
        max_features=TFIDF_MAX_QTY,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    vec_cost = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b(?:[A-Z]\d{4}|\d{5})\b",
        lowercase=False,
        ngram_range=TFIDF_NGRAM_COST,
        max_features=TFIDF_MAX_COST,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )

    def _safe_fit_transform(vec, tr, va, te):
        try:
            Xtr = vec.fit_transform(tr.astype(str))
            Xva = vec.transform(va.astype(str))
            Xte = vec.transform(te.astype(str))
            return ensure_csr_float32_int32(Xtr), ensure_csr_float32_int32(Xva), 
        except ValueError:
            return csr_matrix((len(tr), 0), dtype=np.float32), csr_matrix((len(va
    Xtr_q, Xva_q, Xte_q = _safe_fit_transform(vec_qty, tr_text_qty, va_text_qty, 
    Xtr_c, Xva_c, Xte_c = _safe_fit_transform(vec_cost, tr_text_cost, va_text_cos
    Xtr = ensure_csr_float32_int32(hstack([Xtr_q, Xtr_c], format="csr"))
    Xva = ensure_csr_float32_int32(hstack([Xva_q, Xva_c], format="csr"))
    Xte = ensure_csr_float32_int32(hstack([Xte_q, Xte_c], format="csr"))
    return Xtr, Xva, Xte
# =========================
# CALIBRATION
# =========================
def shrunk_group_median(resid: np.ndarray, group: np.ndarray, alpha: float) -> Tu
    g = group.astype(str)
    global_med = float(np.median(resid))
    out = {}
    for key in np.unique(g):
        m = (g == key)
        r = resid[m]
        if r.size == 0:
            continue
        med = float(np.median(r))
        n = float(r.size)
        out[str(key)] = (n * med + alpha * global_med) / (n + alpha)
    return out, global_med
def apply_bias(pred: np.ndarray, group: np.ndarray, bias: Dict[str, float], defau
    g = group.astype(str)
    p = pred.astype(np.float64).copy()
    for i in range(len(p)):

        p[i] += bias.get(str(g[i]), default_bias)
    return p.astype(np.float32)
# =========================
# MAIN
# =========================
def run_iteration11(diagnostic: bool = False) -> Dict[str, Any]:
    setup_logging()
    logging.info(f"XGBoost version: {getattr(xgb,'__version__','unknown')}")
    logging.info("=== Iteration 11: Leakage-safe TE + robust receipt parsing + XG
    logging.info(f"diagnostic={diagnostic} | DEVICE={DEVICE}")
    train = pd.read_csv(TRAIN_CSV)
    test  = pd.read_csv(TEST_CSV)
    logging.info(f"Loaded train={train.shape} test={test.shape}")
    if PATIENTS_CSV.exists():
        pat = pd.read_csv(PATIENTS_CSV)
        if "patient_id" in pat.columns:
            train = train.merge(pat, on="patient_id", how="left")
            test  = test.merge(pat, on="patient_id", how="left")
            logging.info(f"Merged patients.csv: train={train.shape} test={test.sh
    else:
        logging.warning("patients.csv not found; continuing with ed_cost_*.csv on
    # receipts parse
    all_ids = pd.concat([train["patient_id"], test["patient_id"]], ignore_index=T
    rec = load_receipts(all_ids)
    train = train.merge(rec, on="patient_id", how="left")
    test  = test.merge(rec, on="patient_id", how="left")
    # ensure docs strings
    for df in [train, test]:
        for c in ["codes_doc_qty", "codes_doc_cost", "desc_doc"]:
            df[c] = df[c].fillna("").astype(str)
        df["pdf_insurance"] = df["pdf_insurance"].fillna("missing").astype(str)
        df["pdf_zip3"] = df["pdf_zip3"].fillna("missing").astype(str)
    # totals sanity: spec says parsed TOTAL matches prior cost; use prior cost as 

    for df in [train, test]:
        df["pdf_total_used"] = df["prior_ed_cost_5y_usd"].astype(float)
        df["total_absdiff_prior"] = np.abs(df["pdf_total_parsed"].astype(float) - 
        df["total_parse_ok"] = ((df["total_absdiff_prior"] <= 0.01) & np.isfinite
        df["line_absdiff_total"] = np.abs(df["line_sum"].astype(float) - df["prio
    logging.info(
        "Receipt sanity | parse_ok_rate(train)={:.3f} | total_parse_ok_rate(train
        "line_absdiff_total_med(train)={:.3f} | max_absdiff_total(train)={:.3f}".
            float(train["parse_ok"].mean()),
            float(train["total_parse_ok"].mean()),
            float(train["line_absdiff_total"].median()),
            float(train["line_absdiff_total"].max()),
        )
    )
    # choose pivot codes (from TRAIN frequency)
    code_freq = defaultdict(int)
    for d in train["code_qty_dict"].tolist():
        if isinstance(d, dict):
            for c in d.keys():
                code_freq[c] += 1
    codes_sorted = sorted(code_freq.items(), key=lambda x: (-x[1], x[0]))
    pivot_codes = [c for c, _ in codes_sorted[:TOP_CODE_PIVOTS]]
    logging.info(f"Top code pivots: K={len(pivot_codes)} | unique_codes_in_train=
    train = add_code_pivots(train, pivot_codes, total_col="pdf_total_used")
    test  = add_code_pivots(test,  pivot_codes, total_col="pdf_total_used")
    # derived tabular
    for df in [train, test]:
        df["prior_cost_per_visit"] = df["prior_ed_cost_5y_usd"] / (df["prior_ed_v
        df["log_prior_cost"] = np.log1p(df["prior_ed_cost_5y_usd"].astype(float))
        df["log_prior_visits"] = np.log1p(df["prior_ed_visits_5y"].astype(float))
        df["log_pdf_total_used"] = np.log1p(df["pdf_total_used"].astype(float))
        if "insurance" in df.columns:
            df["ins_mismatch"] = ((df["insurance"].astype(str) != df["pdf_insuran
        else:
            df["ins_mismatch"] = 0

        if "zip3" in df.columns:
            df["zip_mismatch"] = ((df["zip3"].astype(str) != df["pdf_zip3"].astyp
        else:
            df["zip_mismatch"] = 0
    if diagnostic:
        return {
            "train_parse_ok_rate": float(train["parse_ok"].mean()),
            "train_total_parse_ok_rate": float(train["total_parse_ok"].mean()),
            "train_line_absdiff_total_max": float(train["line_absdiff_total"].max
            "unique_codes_in_train": int(len(code_freq)),
            "pivot_codes": pivot_codes[:20],
            "nonempty_codes_doc_qty_train": int((train["codes_doc_qty"].str.len() 
            "nonempty_codes_doc_qty_test": int((test["codes_doc_qty"].str.len() > 
        }
    y = train["ed_cost_next3y_usd"].values.astype(np.float32)
    # stratification: chronic + insurance + cost/visit bins
    if "insurance" in train.columns:
        ins = train["insurance"].astype(str)
    else:
        ins = train["pdf_insurance"].astype(str)
    cost_bin = pd.qcut(train["prior_ed_cost_5y_usd"], q=6, duplicates="drop").asty
    vis_bin  = pd.qcut(train["prior_ed_visits_5y"], q=3, duplicates="drop").astyp
    strat = (train["primary_chronic"].astype(str) + "_" + ins + "_" + cost_bin + 
    # identify columns for tabular preprocessor
    text_cols = {"codes_doc_qty", "codes_doc_cost", "desc_doc"}
    dict_cols = {"code_qty_dict", "code_cost_dict"}
    exclude = {"patient_id", "ed_cost_next3y_usd"} | text_cols | dict_cols
    cat_cols: List[str] = []
    base_cat = ["primary_chronic", "sex", "insurance", "zip3", "pdf_insurance", "
    for c in base_cat:
        if c in train.columns:
            cat_cols.append(c)
    for c in train.columns:

        if c in exclude or c in cat_cols:
            continue
        if train[c].dtype == "object" or str(train[c].dtype).startswith("string")
            cat_cols.append(c)
    num_cols: List[str] = []
    for c in train.columns:
        if c in exclude or c in cat_cols:
            continue
        if pd.api.types.is_numeric_dtype(train[c]):
            num_cols.append(c)
    for c in cat_cols:
        train[c] = train[c].astype("string").fillna("missing")
        test[c] = test[c].astype("string").fillna("missing")
    logging.info(f"Tabular cols | num={len(num_cols)} cat={len(cat_cols)}")
    # CV splits
    splits = []
    for rep, seed in enumerate(REPEAT_SEEDS):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(train)), st
            splits.append((rep, fold, seed, tr_idx, va_idx))
    logging.info(f"Total folds: {len(splits)} | device={DEVICE}")
    oof_sum = np.zeros(len(train), dtype=np.float64)
    oof_cnt = np.zeros(len(train), dtype=np.float64)
    te_sum  = np.zeros(len(test), dtype=np.float64)
    n_models = 0
    params = dict(
        objective="reg:absoluteerror",
        eval_metric="mae",
        tree_method="hist",
        device=DEVICE,
        n_estimators=N_ESTIMATORS,
        learning_rate=LR,
        max_depth=MAX_DEPTH,
        subsample=SUBSAMPLE,

        colsample_bytree=COLSAMPLE,
        reg_lambda=REG_LAMBDA,
        min_child_weight=MIN_CHILD_WEIGHT,
        gamma=GAMMA,
        n_jobs=-1,
        verbosity=1,
        early_stopping_rounds=EARLY_STOPPING,
    )
    for rep, fold, seed, tr_idx, va_idx in tqdm(splits, desc="CV", ncols=110):
        tag = f"rep={rep} fold={fold}"
        t0 = time.time()
        df_tr = train.iloc[tr_idx].reset_index(drop=True)
        df_va = train.iloc[va_idx].reset_index(drop=True)
        df_te = test.reset_index(drop=True)
        y_tr = y[tr_idx]
        y_va = y[va_idx]
        # ------------------
        # Fold-safe Target Encoding
        # ------------------
        idx_tr_local = np.arange(len(df_tr))
        group_tr = (df_tr["primary_chronic"].astype(str) + "_" +
                    (df_tr["insurance"].astype(str) if "insurance" in df_tr.colum
        group_va = (df_va["primary_chronic"].astype(str) + "_" +
                    (df_va["insurance"].astype(str) if "insurance" in df_va.colum
        group_te = (df_te["primary_chronic"].astype(str) + "_" +
                    (df_te["insurance"].astype(str) if "insurance" in df_te.colum
        code_map, global_med, grp_code_map, grp_med = build_code_te_maps(
            df_tr, idx_tr_local, y_tr, group_key=group_tr, alpha=TE_ALPHA
        )
        te_tr = patient_te_features(df_tr, np.arange(len(df_tr)), code_map, globa
        te_va = patient_te_features(df_va, np.arange(len(df_va)), code_map, globa
        te_te = patient_te_features(df_te, np.arange(len(df_te)), code_map, globa
        X_te_tr = ensure_csr_float32_int32(csr_matrix(te_tr.astype(np.float32)))

        X_te_va = ensure_csr_float32_int32(csr_matrix(te_va.astype(np.float32)))
        X_te_te = ensure_csr_float32_int32(csr_matrix(te_te.astype(np.float32)))
        # ------------------
        # Tabular preprocessor (fold fit)
        # ------------------
        pre = build_preprocessor(num_cols, cat_cols)
        X_tr_tab = pre.fit_transform(df_tr)
        X_va_tab = pre.transform(df_va)
        X_te_tab = pre.transform(df_te)
        X_tr_tab = ensure_csr_float32_int32(X_tr_tab.tocsr() if hasattr(X_tr_tab, 
        X_va_tab = ensure_csr_float32_int32(X_va_tab.tocsr() if hasattr(X_va_tab, 
        X_te_tab = ensure_csr_float32_int32(X_te_tab.tocsr() if hasattr(X_te_tab, 
        # ------------------
        # TF-IDF (fold fit)
        # ------------------
        X_tr_txt, X_va_txt, X_te_txt = fit_vectorizers_and_transform(
            df_tr["codes_doc_qty"].values, df_va["codes_doc_qty"].values, df_te["
            df_tr["codes_doc_cost"].values, df_va["codes_doc_cost"].values, df_te
        )
        X_tr = ensure_csr_float32_int32(hstack([X_tr_tab, X_te_tr, X_tr_txt], for
        X_va = ensure_csr_float32_int32(hstack([X_va_tab, X_te_va, X_va_txt], for
        X_te = ensure_csr_float32_int32(hstack([X_te_tab, X_te_te, X_te_txt], for
        logging.info(f"{tag} | X shapes | tr={X_tr.shape} va={X_va.shape} nnz(tr)
        model = XGBRegressor(**params, random_state=10000 * rep + seed + fold)
        fit_sig = inspect.signature(model.fit).parameters
        fit_kwargs = {"eval_set": [(X_va, y_va)]}
        if "verbose" in fit_sig:
            fit_kwargs["verbose"] = VERBOSE_EVERY
        elif "verbose_eval" in fit_sig:
            fit_kwargs["verbose_eval"] = VERBOSE_EVERY
        logging.info(f"{tag} | fit START")

        model.fit(X_tr, y_tr, **fit_kwargs)
        logging.info(f"{tag} | fit END | best_iteration={getattr(model,'best_iter
        p_va = model.predict(X_va).astype(np.float32)
        p_te = model.predict(X_te).astype(np.float32)
        if CLIP_AT_ZERO:
            p_va = np.maximum(p_va, 0.0)
            p_te = np.maximum(p_te, 0.0)
        oof_sum[va_idx] += p_va
        oof_cnt[va_idx] += 1.0
        te_sum += p_te
        n_models += 1
        fold_mae = mae(y_va, p_va)
        logging.info(f"{tag} | fold_MAE={fold_mae:.3f} | fold_time={time.time()-t
    oof = (oof_sum / np.maximum(oof_cnt, 1.0)).astype(np.float32)
    pred_test = (te_sum / max(1, n_models)).astype(np.float32)
    raw_mae = mae(y, oof)
    logging.info(f"OOF MAE (raw): {raw_mae:.4f}")
    resid = (y.astype(np.float64) - oof.astype(np.float64))
    best = (raw_mae, "none", pred_test)
    # chronic calibration
    bias_c, def_c = shrunk_group_median(resid, train["primary_chronic"].astype(st
    oof_c = apply_bias(oof, train["primary_chronic"].astype(str).values, bias_c, 
    if CLIP_AT_ZERO:
        oof_c = np.maximum(oof_c, 0.0)
    mae_c = mae(y, oof_c)
    logging.info(f"OOF MAE (chronic calib): {mae_c:.4f}")
    if mae_c < best[0]:
        pred_c = apply_bias(pred_test, test["primary_chronic"].astype(str).values
        if CLIP_AT_ZERO:
            pred_c = np.maximum(pred_c, 0.0)
        best = (mae_c, "chronic", pred_c)

跑完 Iter11 后，你应该看什么（判断有没有继续提分的潜力）
    # chronic+insurance calibration
    if "insurance" in train.columns and "insurance" in test.columns:
        key_tr = (train["primary_chronic"].astype(str) + "_" + train["insurance"]
        key_te = (test["primary_chronic"].astype(str) + "_" + test["insurance"].a
        bias_ci, def_ci = shrunk_group_median(resid, key_tr, alpha=TE_ALPHA)
        oof_ci = apply_bias(oof, key_tr, bias_ci, def_ci)
        if CLIP_AT_ZERO:
            oof_ci = np.maximum(oof_ci, 0.0)
        mae_ci = mae(y, oof_ci)
        logging.info(f"OOF MAE (chronic+ins calib): {mae_ci:.4f}")
        if mae_ci < best[0]:
            pred_ci = apply_bias(pred_test, key_te, bias_ci, def_ci)
            if CLIP_AT_ZERO:
                pred_ci = np.maximum(pred_ci, 0.0)
            best = (mae_ci, "chronic+insurance", pred_ci)
    chosen_mae, chosen_name, final_pred = best
    logging.info(f"Chosen calibration: {chosen_name} | OOF MAE={chosen_mae:.4f}")
    sub = pd.DataFrame({
        "patient_id": test["patient_id"].astype(int).values,
        "ed_cost_next3y_usd": final_pred.astype(np.float32),
    })
    sub.to_csv(SUBMISSION_PATH, index=False)
    logging.info(f"Saved submission: {SUBMISSION_PATH}")
    return {
        "oof_mae_raw": float(raw_mae),
        "oof_mae_best": float(chosen_mae),
        "calibration": chosen_name,
        "n_models": int(n_models),
        "submission_path": str(SUBMISSION_PATH),
    }

跑 diagnostic=True  时重点看：
train_parse_ok_rate ：如果明显 < 0.95，说明仍有不少 PDF 行重建失败，特征损失
大
train_line_absdiff_total_max ：如果很大，说明有 PDF 行解析不完整（line_sum 不
等于 TOTAL），要进一步修 parser 或做更强 fallback
unique_codes_in_train ：如果确实只有十几二十个 code，那么“per-code pivot + 费
用结构特征”通常会比纯 TF‑IDF 更稳
下一步（如果 Iter11 还没逼近 448）
你现在已经能证明：receipt 本身就能把模型推到 465。要继续逼近 448，我认为最可能的
“最后 10–15 分”来自 跨 Challenge 的辅助监督信号：用 Challenge 1/3 的 train label 训练
risk models，把 risk score 汇总成 patient severity proxy，再喂给 cost 模型。数据包明确包
含 admissions/stays/vitals/notes。
这一步我可以给你 Iter12（stacking 版）——但建议你先把 Iter11 跑一遍，因为
F 更可信，调参方向更不容易跑偏**。
你把 Iter11 跑出来的：
OOF raw / after calib
LB score
diagnostic 输出（尤其 parse_ok_rate / line_absdiff_total_max）
贴给我，我就能基于“真实 parser 质量 + 真实泛化差距”给你 Iter12 的最优设计（只做
最小增量，不乱加复杂模块）。
description
```

---

## C) Key MAE / Score / Submission Logs (verbatim)

### C1 — Key iteration score statements (verbatim lines + page refs)

- Page 46: `Iteration 1 Score: 493.9 MAE.`
- Page 159: `Current Score: 478.38 MAE (Iteration 2).`
- Page 159: `Previous Score: 493.90 MAE.`
- Page 225: `Baseline (Iteration 2): 478.4 MAE (Best so far).`
- Page 282: `📊 Score: 486.4260 (MAE)`
- Page 282: `📊 Score: 486.4260`
- Page 366: `📊 Score: 496.6276 (MAE)`
- Page 366: `📊 Score: 496.6276`
- Page 388: `📊 Score: 490.2396 (MAE)`
- Page 388: `📊 Score: 490.2396`
- Page 418: `📊 Score: 118393.0109 (MAE)`
- Page 419: `📊 Score: 118393.0109`
- Page 426: `📊 Score: 465.1965 (MAE)`
- Page 426: `📊 Score: 465.1965`

### C2 — Submission result blocks (full page extracts)

#### S01 (pages 282-282)
```text
2026-02-13 15:46:34,716 | INFO | Biases by chronic (mean 
residual): {'DiabetesComp': -35.36746597290039, 'HF': 
55.52805709838867, 'Pneumonia': -151.41448974609375} | 
global_bias=-19.0170
2026-02-13 15:46:34,717 | INFO | OOF MAE (after chronic 
bias correction): 465.6756
2026-02-13 15:46:34,722 | INFO | Saved submission: 
D:\AgentDs\agent_ds_healthcare\submission.csv
🚀 Submitting predictions...
✅ Prediction submitted successfully!
📊 Score: 486.4260 (MAE)
✅ Validation passed
✅ Submission successful!
   📊 Score: 486.4260
   📏 Metric: MAE
   ✔️  Validation: Passed
🎯 Next steps:
   1. Try incorporating relevant information outside this 
table!
   2. Move on to Healthcare Challenge 3!
I have attached log and score, we are getting stucked in 
478~480 range, so I want you to think from another angel,
```

#### S02 (pages 366-366)
```text
2026-02-13 18:24:48,507 | INFO | Chosen postprocess: 
chronic_shrunk_median | OOF MAE=478.1611
2026-02-13 18:24:48,532 | INFO | Saved submission: 
D:\AgentDs\agent_ds_healthcare\submission.csv
{'oof_mae_raw': 488.22125244140625,
 'blend_w_l2': 0.04,
 'postprocess': 'chronic_shrunk_median',
 'submission_path': 
'D:\\AgentDs\\agent_ds_healthcare\\submission.csv',
 'coverage': {'receipt_nonempty': 1154,
  'discharge_nonempty': 2000,
  'vitals_nonempty': 972}}🚀 Submitting predictions...
✅ Prediction submitted successfully!
📊 Score: 496.6276 (MAE)
✅ Validation passed
✅ Submission successful!
   📊 Score: 496.6276
   📏 Metric: MAE
   ✔️  Validation: Passed
🎯 Next steps:
   1. Try incorporating relevant information outside this 
table!
   2. Move on to Healthcare Challenge 3!
傻逼!!!!!!!!! 越来越烂了!!! 为什么让你跳出框架思维还变烂
了, 尝试:走另一条完全不同的“数学路径”，例如更强的 
outlier-aware 训练/分段回归/极端值校准
Thought for 26m 13s
```

#### S03 (pages 388-388)
```text
我知道你很火大，但这次我们做的是数学上真正不同的分解：
不是“加更多乱七八糟特征”，而是把问题拆成主体 + 尾部，然后用 MAE 对齐的中位数回归
+ 分段残差校准去专门处理极端值。
你先把 Iter9 跑起来上榜（哪怕只看一次结果），我们再决定下一步是否要把阈值/校准桶数
做一个小网格搜索（那也仍然是“简单数学”，不是复杂特征工程）。
Sources
Sources
Sources
🚀 Submitting predictions...
✅ Prediction submitted successfully!
📊 Score: 490.2396 (MAE)
✅ Validation passed
✅ Submission successful!
   📊 Score: 490.2396
   📏 Metric: MAE
   ✔️  Validation: Passed
🎯 Next steps:
   1. Try incorporating relevant information outside this 
table!
   2. Move on to Healthcare Challenge 3!
2026-02-13 19:23:27,266 | INFO | rep=1 fold=4 | TAIL_REG | 
fit END | best_iteration=114
2026-02-13 19:23:27,276 | INFO | rep=1 fold=4 | 
fold_time=5.1s
CV: 100%|
████████████████████████████████████
█████████████████████████████████| 
10/10 [00:49<00:00,  4.99s/it]
2026-02-13 19:23:27,280 | INFO | Best tail mix on OOF: 
MAE=474.5739 | k=1.50 gamma=1.50
2026-02-13 19:23:27,281 | INFO | OOF MAE (segmented 
receipt_7.pdf
PDF
```

#### S04 (pages 418-419)
```text
total_absdiff_sample_max  是否≈0
2. 再训练：
3. 如果 pdf_cov_test < 0.98  或 nonempty_codes_doc_qty_test  很低：
你现在所有“吃 receipt” 的模型都会线上崩，这就是 Iter9 OOF 好但榜单差的最常见解
释。
先把 test receipts 补齐/放对位置，再谈模型。
关于“要不要本地 LLM？”
就你这个数据形态（synthetic billing summary、短描述、强结构表格），不需要 LLM。
LLM 适合的是：长自由文本、强语义推断、或需要抽取复杂实体关系。
这里最值钱的是：
code 本身（稀疏高维）
line_total/qty 的结构统计
极少数强信号码（E/M level、critical care、插管等）
这些用规则+统计学习就够了，而且更稳、更快。
如果你愿意，把 Iteration10 diagnostic 的输出（尤其 pdf_cov_test ）贴出来，我可以立刻
判断你们现在的最大瓶颈到底是 数据覆盖问题 还是 模型/特征问题。
python
res = run_iteration10(diagnostic=False)
res
Copy code
Sources
Sources
Sources
🚀 Submitting predictions...
✅ Prediction submitted successfully!
📊 Score: 118393.0109 (MAE)
✅ Validation passed

---

✅ Submission successful!
   📊 Score: 118393.0109
   📏 Metric: MAE
   ✔️  Validation: Passed
🎯 Next steps:
   1. Try incorporating relevant information outside this 
table!
   2. Move on to Healthcare Challenge 3!
2026-02-14 04:01:37,124 | INFO | XGBoost version: 3.0.0
2026-02-14 04:01:37,126 | INFO | === Iteration 10: claims-
aware + OOF code risk + sparse TFIDF + XGB GPU ===
2026-02-14 04:01:37,129 | INFO | diagnostic=True | 
force_cpu=False
2026-02-14 04:01:37,135 | INFO | Loaded train=(2000, 5) 
test=(2000, 4)
2026-02-14 04:01:37,167 | INFO | Merged patients.csv: 
train=(2000, 9) test=(2000, 8)
2026-02-14 04:01:37,247 | INFO | PDF coverage | 
train=1.000 test=1.000
2026-02-14 04:01:37,251 | INFO | Receipt cache hit=0 
miss=4,000 version=101
Parsing receipts: 100%|
████████████████████████████████████
█████| 4000/4000 [00:55<00:00, 72.56it/s]
2026-02-14 04:02:40,587 | INFO | TOTAL sanity (sample 
n=200): absdiff median=0.000000, max=25937.300000
2026-02-14 04:02:40,610 | INFO | TE fold 0 | built mapping 
codes=18
2026-02-14 04:02:40,619 | INFO | TE fold 1 | built mapping 
codes=18
2026-02-14 04:02:40,630 | INFO | TE fold 2 | built mapping 
codes=18
2026-02-14 04:02:40,641 | INFO | TE fold 3 | built mapping 
codes=18
2026-02-14 04:02:40,651 | INFO | TE fold 4 | built mapping 
codes=18
[0]
validation_0-mae:3899225.74313
```

#### S05 (pages 426-426)
```text
我就按你选的方向把单脚本给出来（Jupyter 直接跑、GPU、全日志、带缓存、带 sanity
checks）。
Sources
Sources
Sources
🚀 Submitting predictions...
✅ Prediction submitted successfully!
📊 Score: 465.1965 (MAE)
✅ Validation passed
✅ Submission successful!
   📊 Score: 465.1965
   📏 Metric: MAE
   ✔️  Validation: Passed
🎯 Next steps:
   1. Try incorporating relevant information outside this 
table!
   2. Move on to Healthcare Challenge 3!
{'oof_mae_raw': 448.10430908203125,
 'oof_mae_best': 440.1262512207031,
 'calibration': 'chronic+insurance',
 'n_models': 10,
 'submission_path': 
'D:\\AgentDs\\agent_ds_healthcare\\submission_ICHI_V1.c
sv'}
2026-02-14 04:33:26,598 | INFO | rep=1 fold=4 | fit END | 
best_iteration=540
2026-02-14 04:33:26,649 | INFO | rep=1 fold=4 | 
fold_MAE=439.919 | fold_time=3.8s
CV: 100%|
████████████████████████████████████
█████████████████████████████████| 
10/10 [01:13<00:00,  7.30s/it]
2026-02-14 04:33:26,652 | INFO | OOF MAE (raw): 448.1043
```

### C3 — Performance review / failure analysis pages (full page extracts)

#### Page 46
```text
Iteration 1 Score: 493.9 MAE.
Leaderboard Target: ~448.7 MAE.
Gap Analysis: We are trailing the best solution by ~45 
points (approx. 10%). This is a massive gap in a regression 
challenge.
Diagnosis: Our previous approach likely under-utilized the 
unstructured PDF data or failed to capture the non-linear 
relationships in the cost data.
[The Challenge: "Black Box" Improvement]
I have NO domain knowledge of healthcare or CPT codes. I 
cannot help you feature engineer.
Your Task: You must analyze the problem abstractly and 
design a "Level 2" Solution that closes this gap.
The Core Question: What specific mathematical or 
architectural change is required to squeeze an extra 10% 
accuracy out of this dataset?
Is it better text processing? (e.g., embeddings vs counts?)
Is it a different loss function? (e.g., Tweedie vs MAE vs 
Quantile?)
Is it a stronger ensemble? (e.g., Stacking vs Blending?)
[Directives for Iteration 2]
Write a complete, self-contained Python script for Iteration 
2 that implements your best hypothesis for closing this 
gap.
Hypothesize: Briefly comment at the top of the code why 
you believe this new approach will beat the 493.9 baseline.
Execute: Write the full pipeline (Parsing -> Engineering -> 
Modeling -> Submission).
```

#### Page 159
```text
[System Role]
You are a Dual-Expert: a Clinical Data Scientist (specializing 
in medical coding, CPT/HCPCS hierarchies, and chronic 
disease burden) and a Kaggle Grandmaster (specializing in 
feature engineering for tabular regression).
[Situation Report]
Current Score: 478.38 MAE (Iteration 2).
Previous Score: 493.90 MAE.
Improvement: ~15.5 points gained by using "Blind" NLP 
(TF-IDF on billing codes).
The Problem: We are still trailing the leader (~448 MAE) by 
30 points. The "Blind NLP" approach has likely hit a ceiling 
because it treats medical codes (e.g., 99214 vs 99285) as 
arbitrary text tokens, ignoring their inherent clinical 
hierarchy and severity.
[The Objective: Iteration 3]
We need to "Unlock Domain Knowledge."
I want you to revisit the data (ed_cost_train.csv + PDF 
receipts) and apply your internal clinical domain 
knowledge to engineer superior features.
Do not ask me for directions. You are the expert.
Think: How does a "Level 5 ED Visit" (CPT 99285) differ 
from a "Level 1" (99281) in terms of cost risk? How do 
"Surgery" codes (10000-60000 range) differ from "Lab" 
codes (80000 range)?
Hypothesize: What specific clinical interactions (e.g., 
Chronic Condition + Recent Utilization Intensity) are likely 
driving the highest costs?
```

#### Page 193
```text
If you run this and paste the log lines:
OOF MAE | XGB: ...
OOF MAE | CAT: ...
Blend ... OOF MAE final= ...
…I can tell you whether your remaining ~30 MAE gap is primarily:
feature ceiling (need sharper clinical grouping or better TE hierarchy), or
modeling ceiling (need a different MAE-aligned objective strategy like quantile
stacking), or
CV mismatch (split needs to mimic hidden test distribution).
    logging.info(f"Blend w(XGB)={w:.2f} | OOF MAE pre-bias={blend_mae:.4f} | bias
    # Save submission
    sub = pd.DataFrame({
        "patient_id": test_df["patient_id"].astype(int).values,
        "ed_cost_next3y_usd": test_final.astype(np.float32),
    })
    sub.to_csv(SUBMISSION_PATH, index=False)
    logging.info(f"Saved submission: {SUBMISSION_PATH}")
if __name__ == "__main__":
    main()
Sources
Sources
[System Role]
You are a Lead AI Scientist and Kaggle Grandmaster. We 
are conducting a "Failure Analysis" and strategic pivot.
[Performance Review]
Iteration 2 (Blind Statistical NLP): 478.4 MAE.
Iteration 3 (Clinical Domain Rules): 480.9 MAE.
```

#### Page 194
```text
The Verdict: The "Domain Expert" approach failed. The 
score degraded. This indicates that imposing manual 
clinical hierarchies (e.g., grouping codes by range) likely 
introduced bias or noise that the model couldn't handle 
given the small dataset size. The data "prefers" the raw, 
unbiased statistical signal over our human-defined rules.
[The Pivot: "Return to the Math"]
We are abandoning the "Manual Clinical Feature" path 
immediately. Do not use CPT ranges or manual groupings.
[The Challenge: Iteration 4]
The Problem: We know "simple counts" (Iter 1) are too 
weak. We know "manual clinical rules" (Iter 3) are too 
biased.
The Goal: You need to find a "Third Path"—a method to 
capture the structure and interaction of the medical codes 
without hard-coding their meaning.
Your Task: Think like a Mathematician, not a Doctor.
How do you represent high-dimensional, sparse data 
(thousands of unique billing codes) in a way that a tree-
based model (XGBoost/CatBoost) can digest efficiently?
If you can't group them by "medical meaning," how do 
you group them by "statistical similarity"?
Is there a better way to transform the target (Cost) or the 
loss function to align with the MAE metric?
[Execution Directive]
Write the complete Iteration 4 Python Script.
Discard: Remove the "Clinical Grouping" logic from the 
previous script.
```

#### Page 225
```text
…I’ll tell you whether the bottleneck is now (a) embedding dimensions/clusters, (b) TE
strength vs leakage/noise, or (c) CV mismatch to hidden test.
Sources
Sources
Copy & Paste This Prompt
[System Role]
You are a Principal Data Scientist and Kaggle Grandmaster. 
You are strictly evaluated on results.
[Status Report]
Baseline (Iteration 2): 478.4 MAE (Best so far).
Attempt 3 (Clinical Grouping): 480.9 MAE (Failed).
Attempt 4 (Latent/SVD Features): 483.9 MAE (Failed Badly).
[The Problem]
We are regressing. Every time we add "smart" complexity, 
the model gets worse. We need to beat 478.4, and we are 
currently moving in the wrong direction.
[Directives: WHAT TO AVOID]
You have full autonomy to design Iteration 5, but you must 
strictly adhere to these Negative Constraints based on our 
failures:
DO NOT use Dimensionality Reduction: Do not use SVD, 
PCA, NMF, or Learned Embeddings on the text data. 
Iteration 4 proved that compressing the sparse signal 
destroys information we need for this specific metric.
DO NOT use Manual Domain Groupings: Do not try to 
group CPT codes by range or "clinical meaning." Iteration 
3 proved that our manual buckets introduce bias and 
noise.
```

#### Page 246
```text
If this still doesn’t beat 478.4, the next minimal, “non-complexity” lever I’d pull (still within
your constraints) is very small vocabulary pruning ( CODE_MIN_DF=2 ) + slightly higher
regularization ( min_child_weight=2–5 ) to curb rare-code overfit—without changing the
representation paradigm.
        oof_final = oof_global
        test_final = test_global
    if CLIP_AT_ZERO:
        oof_final = np.maximum(oof_final, 0.0)
        test_final = np.maximum(test_final, 0.0)
    logging.info(f"FINAL OOF MAE (post-calibration): {mae(y, oof_final):.4f}")
    # save submission
    sub = pd.DataFrame({
        "patient_id": test_df["patient_id"].astype(int).values,
        "ed_cost_next3y_usd": test_final.astype(np.float32),
    })
    sub.to_csv(SUBMISSION_PATH, index=False)
    logging.info(f"Saved submission: {SUBMISSION_PATH}")
if __name__ == "__main__":
    main()
Sources
Sources
[System Role]
You are a Kaggle Grandmaster specializing in XGBoost 
Optimization.
[Critical Error Analysis]
Look at the logs from our last run:
XGBoost MAE: 477.79 (Excellent, competitive).
```

#### Page 247
```text
CatBoost MAE: 1307.71 (Catastrophic failure).
Blend Weight: 100% XGBoost.
[The Diagnosis]
We have been wasting compute on a CatBoost model that 
is completely broken (likely due to a loss function 
mismatch or unscaled target). The "ensemble" approach is 
hurting us because we are blending a race car with a 
broken bicycle.
[The Strategy: "Monolithic XGBoost"]
We are reverting to the Iteration 2 logic (TF-IDF on CPT 
codes), but we are going to go "All-In" on the single 
working model.
Remove CatBoost: Delete it entirely. It is dead weight.
Scale the NLP: Since XGBoost is working well with the text 
features, give it more. Increase TF-IDF max_features from 
100 to 1000. Let the tree find the signal.
Target Calibration: The logs showed that 
Calibration(by_chronic) improved the MAE by ~2 points 
(477 -> 475). Keep this logic. It is a free win.
Tuning: Since we are only training one model, increase 
n_estimators to 8000 and lower the learning_rate to 0.01 
or 0.02 for a more precise convergence.
[The Task: Iteration 6]
Write the complete Python Script for the "Monolithic 
XGBoost" solution.
Features: Tabular + TF-IDF (1-3 grams, 1000 features).
Model: Single XGBoost (GPU enabled).
Post-Processing: Apply the "Chronic Condition Bias 
Correction" (subtracting the mean error per chronic group)
```

#### Page 283
```text
maybe thinking of refering at similar situation in other 
domain. 
Thought for 12m 16s
What you just observed is the classic “CV looks amazing → leaderboard doesn’t move (or
gets worse)” failure mode. Your local model isn’t necessarily “bad”; it’s likely that your
validation protocol is no longer measuring the same problem the hidden test set is
measuring.
The key tell is this:
OOF MAE ~469 → 465 after correction
Hidden test MAE ~486
That ~20 MAE gap is too large to be explained by “random variance” on ~2k rows. This
usually means distribution shift / concept drift / validation leakage—a pattern that
shows up in fraud detection, malware, speech recognition, recommender systems, etc.
Below is the “another angle” view, grounded in tactics that win when feature engineering
hits a ceiling.
1) Reframe the problem: this looks like a domain shift issue, not a
feature issue
In other domains, when train/test differ (time period, geography, platform, user cohort),
you see exactly what you saw: strong in-sample/CV performance but weak real-
world/test performance.
Adversarial validation exists for this exact reason: treat “train vs test” as a classification
task; if a classifier can separate them with AUC >> 0.5, then your model is learning
patterns that don’t transfer. 
Medium +1
```

#### Page 284
```text
Analytics Vidhya even gives the typical causes (different time periods, different feature
values) and explicitly notes CV won’t fix it because CV only resamples within train.
Medium’s explanation is similar: create dummy labels (train=0, test=1) and measure
separability; high AUC implies drift. 
So the “new angle” is: stop asking “how do I extract more from PDFs?” and start asking
“how do I make train look like test (or validate like test)?”
2) Why your current CV is probably lying
Even if everything is “correct,” there are three common ways your CV becomes over-
optimistic here:
A) Transductive preprocessing inflates OOF
If TF‑IDF is fit on train+test (or even on full train before folds) then your validation fold is
not truly unseen w.r.t. IDF statistics. That’s usually “mild leakage,” but with small data +
sparse codes it can be non-trivial.
Result: OOF MAE drops, but the hidden set doesn’t follow.
B) Your bias correction is overfitting the train residual structure
Your chronic mean residuals were huge (e.g., Pneumonia ≈ −151). That magnitude
screams “unstable correction” — mean residual is very sensitive to a few outliers. MAE is
not.
For MAE, the optimal constant shift is a median residual, not mean. Mean can look great
in OOF and still wreck test if the outliers don’t repeat.
C) Fold strategy doesn’t mimic how the hidden test was formed
If the hidden test is shifted by zip3/insurance/age cohort/etc., then stratifying by
(primary_chronic + prior_cost_bin)  won’t replicate it. You’ll “win CV” and lose LB.
Analytics Vidhya
Medium
```

#### Page 285
```text
3) The playbook from other domains: Adversarial validation → fix CV
→ optionally importance-weight training
This is the most direct way to get unstuck without “smart complexity.”
Step 1 — Run adversarial validation (1 hour work, huge insight)
Create a dataset:
Features = whatever you feed the model (tabular + simple receipt meta + a very
cheap sparse code count / TF‑IDF)
Label = 0 for train rows, 1 for test rows
Train a light classifier (logreg / lightGBM / xgb) and measure AUC:
AUC ≈ 0.5 → train/test similar, focus back on modeling
AUC >> 0.6 → meaningful shift (your situation is likely here) 
Also inspect feature importance: it tells you what is drifting.
Step 2 — Build folds using “test-likeness”
Once you have adversarial probabilities p(test|x)  for each train row:
Sort train rows by p(test|x)
Create folds by slicing this sorted list (fold 0 = most train-like, fold K = most test-like)
Now, your CV is measuring performance across the same drift axis that matters for the
hidden test.
This single change often “makes OOF worse but leaderboard better” — which is exactly
what you need.
Step 3 — Importance weighting (optional but often wins)
If you confirm covariate shift, you can weight regression training by something like:
Medium +1
w(x) =
​
1 −p(test∣x)
p(test∣x)
```

#### Page 286
```text
Then train XGBoost on train with sample_weight=w .
This is a standard domain adaptation trick: push the regressor to care more about train
examples that look like test. The adversarial validation articles explicitly suggest
selecting/weighting samples that are more test-like. 
4) A minimal “fix” you should do immediately: replace mean residual
bias with shrunk median residual bias
You got an OOF boost from chronic correction, but it likely didn’t transfer.
Do this instead (still “free win,” but stable):
Let r = y - oof_pred
For each chronic group g , compute med_g = median(r_g)  and n_g
Global med_all = median(r)
Shrink:
Pick α  like 20–50.
Why this works:
Median aligns with MAE (robust)
Shrinkage prevents massive corrections like −151 unless it’s strongly supported by
lots of data
This is borrowed from the same principle as smoothing target encodings: don’t trust
small-group estimates.
5) The biggest “other angle” you may be missing: use the other tables
in the dataset
Medium +1
b
​ =
g
​
n
​ + α
g
n
​ ⋅med
​ + α ⋅med_all
g
g
```

#### Page 287
```text
Your dataset package includes admissions and stays tasks and they are joinable by
patient_id .
If the competition platform doesn’t forbid it, this is the single most likely reason the leader
is ~448 while you’re stuck ~478.
Why this is the Home Credit / Instacart / relational-Kaggle pattern:
Your current feature set is “application table + noisy text”
The missing lift often comes from “bureau table / previous applications / transactions”
Here, that is admissions + stays + vitals summaries (even without deep NLP)
Eve
-level aggregates can be high signal:
# admissions , mean los_days , max charlson_band , % acuity_emergent ,
ed_visits_6m  stats
# stays , unit_type counts, admission_reason counts
All are directly described in the dataset overview.
This is not “smart clinical rules.” It’s just “more rows per patient → aggregate.”
6) What I’d do next (high probability of breaking the plateau)
Order matters; do these in sequence:
1. Run adversarial validation to quantify drift (AUC). 
2. Change CV to adversarial-score folds (so OOF becomes meaningful).
3. Replace chronic mean correction with shrunk chronic median correction (MA
rift is real, try importance-weighted XGB (still one model).
4. Add cross-task patient aggregates from admissions/stays (if allowed).
If you want, I can write Iteration 7 as a single script that:
builds adversarial scores
uses them to define folds
trains monolithic GPU XGBoost with weights
applies shrunk median bias correction
optionally joins admissions/stays aggregates (guarded by file existence)
description
Medium +1
description
```

#### Page 288
```text
…but the main “another angle” takeaway is:
You are likely not feature-limited; you are validation-limited (distribution shift).
description
Sources
Sources
Sources
follow what you proposed above, give me the iteration 7 
as single script. If you need me to help you do some pre-
data analysis validation work before you decide how to 
right the code, give me those code and step by step 
producer.
Thought for 26m 56s
You’re hitting the classic “CV ≠ hidden test” wall. Iteration 7 below implements the exact
pivot we discussed:
Adversarial validation (train=0 vs test=1) to measure and model train↔test drift and
produce a test-likeness score. 
Drift-aware folds built by sorting train rows by that score (so validation is “more like
test”), instead of random KFold.
Importance weighting using odds p/(1-p)  (a density-ratio proxy under covariate
shift) to bias training toward test-like samples. 
MAE-aligned calibration using a shrunk median residual per chronic group (median
is the MAE minimizer; shrinkage prevents huge unstable offsets). 
XGBoost GPU regression with sample_weight  supported by the sklearn API.
Optional: patient-level aggregates from admissions/stays if those CSVs exist, since
the dataset is explicitly joinable by patient_id .
Analytics Vidhya
Department of S… +1
Wikipedia
XGBoost Docum… +1
```

### C4 — Critical error blocks (selected, verbatim)

#### Page 81
```text
The TE + ensemble blend gives a bigger gain than any single model.
If OOF improves but LB doesn’t, it’s almost always a CV mismatch: in that case, we’ll
tighten the split strategy (e.g., stratify also by receipt complexity bins, or use repeated
CV with different strat rules).
If you paste your OOF MAE per model and final OOF MAE from this script, I can tell you
exactly which component is carrying and what to ablate next for Iteration 3.
Sources
Sources
encountered error: -------------------------------------------
--------------------------------
TypeError                                 Traceback (most recent call 
last)
Cell In[5], line 1297
   1293     logging.info(f"Saved: {SUBMISSION_PATH}")
   1296 if __name__ == "__main__":
-> 1297     main()
Cell In[5], line 1169, in main()
   1167 if USE_XGB_L1:
   1168     mdl = XGBRegressor(**xgb_params, 
random_state=3000 * rep + fold)
-> 1169     mdl.fit(
   1170         X_tr, y_tr,
   1171         eval_set=[(X_va, y_va)],
   1172         verbose=False,
   1173         early_stopping_rounds=EARLY_STOPPING,
   1174     )
   1175     p_va = predict_xgb(mdl, X_va)
   1176     p_te = predict_xgb(mdl, X_te)
File 
c:\Users\shend\AppData\Local\Programs\Python\Python3
12\Lib\site-packages\xgboost\core.py:729, in 
require_keyword_args.<locals>.throw_if.
<locals>.inner_f(*args, **kwargs)
    727 for k, arg in zip(sig.parameters, args):
```

#### Page 82
```text
728     kwargs[k] = arg
--> 729 return func(**kwargs)
TypeError: XGBModel.fit() got an unexpected keyword 
argument 'early_stopping_rounds'
fix this and iterate to investigate other potential errors
Thought for 23m 28s
You’re hitting an XGBoost API shift: in newer XGBoost releases, early_stopping_rounds  is
no longer accepted in .fit()  and is expected in the estimator constructor (or via
callbacks). That’s why you see TypeError: ...fit() got an unexpected keyword argument
'early_stopping_rounds' . 
Below is a fixed Iteration‑2 script that:
Adds an XGBoost early-stopping compatibility wrapper (constructor vs fit vs
callbacks).
Adds CatBoost GPU → CPU fallback so GPU issues don’t crash the run.
Adds a couple of safety guards (e.g., spend-sum=0 in TE weighting).
Run it as-is in Jupyter/VS Code.
XGBoost Docum… +2
python
# -*- coding: utf-8 -*-
"""
ITERATION 2 (Fixed) — Target-Encoded CPT Risk + Ensemble (MAE)
Why this should beat Iteration 1:
- Biggest missing lift is learning "code -> future-cost risk". We do CV-safe targ
  (exact code + 2-char prefix), then spend-weight aggregate per patient.
- MAE-aligned objectives: LightGBM L1, LightGBM quantile-median on log1p(y), XGBo
- Rigorous repeated stratified CV + OOF-optimized blend + median-residual bias ca
Fix for your error:
- Newer XGBoost versions moved `early_stopping_rounds` out of `.fit()` and into t
  (or use callbacks). This script uses a compatibility wrapper so it works across
```

#### Page 264
```text
435 vectorizer = TfidfVectorizer(
    436     analyzer="word",
    437     token_pattern=r"(?u)\b[A-Za-z0-9]+\b",
   (...)    447     dtype=np.float32,
    448 )
--> 449 X_tfidf_all = 
vectorizer.fit_transform(df_all["codes_doc"].values)
    450 X_tfidf_all = ensure_csr_float32_int32(X_tfidf_all)
    451 logging.info(f"TF-IDF matrix: shape=
{X_tfidf_all.shape} | nnz={X_tfidf_all.nnz:,}")
File 
c:\Users\shend\AppData\Local\Programs\Python\Python3
12\Lib\site-
packages\sklearn\feature_extraction\text.py:2105, in 
TfidfVectorizer.fit_transform(self, raw_documents, y)
   2098 self._check_params()
   2099 self._tfidf = TfidfTransformer(
   2100     norm=self.norm,
   2101     use_idf=self.use_idf,
   2102     smooth_idf=self.smooth_idf,
   2103     sublinear_tf=self.sublinear_tf,
   2104 )
-> 2105 X = super().fit_transform(raw_documents)
   2106 self._tfidf.fit(X)
   2107 # X is already a transformed view of 
raw_documents so
   2108 # we set copy to False
File 
c:\Users\shend\AppData\Local\Programs\Python\Python3
12\Lib\site-packages\sklearn\base.py:1365, in _fit_context.
<locals>.decorator.<locals>.wrapper(estimator, *args, 
**kwargs)
   1358     estimator._validate_params()
   1360 with config_context(
   1361     skip_parameter_validation=(
   1362         prefer_skip_nested_validation or 
global_skip_validation
   1363     )
```

#### Page 265
```text
1364 ):
-> 1365     return fit_method(estimator, *args, **kwargs)
File 
c:\Users\shend\AppData\Local\Programs\Python\Python3
12\Lib\site-
packages\sklearn\feature_extraction\text.py:1377, in 
CountVectorizer.fit_transform(self, raw_documents, y)
   1369             warnings.warn(
   1370                 "Upper case characters found in"
   1371                 " vocabulary while 'lowercase'"
   1372                 " is True. These entries will not"
   1373                 " be matched with any documents"
   1374             )
   1375             break
-> 1377 vocabulary, X = self._count_vocab(raw_documents, 
self.fixed_vocabulary_)
   1379 if self.binary:
   1380     X.data.fill(1)
File 
c:\Users\shend\AppData\Local\Programs\Python\Python3
12\Lib\site-
packages\sklearn\feature_extraction\text.py:1283, in 
CountVectorizer._count_vocab(self, raw_documents, 
fixed_vocab)
   1281     vocabulary = dict(vocabulary)
   1282     if not vocabulary:
-> 1283         raise ValueError(
   1284             "empty vocabulary; perhaps the documents 
only contain stop words"
   1285         )
   1287 if indptr[-1] > np.iinfo(np.int32).max:  # = 2**31 - 1
   1288     if _IS_32BIT:
ValueError: empty vocabulary; perhaps the documents only 
contain stop words
fix and iterate to make sure no more error and robust
```

#### Page 266
```text
Thought for 26m 53s
Your ValueError: empty vocabulary  happened because PyMuPDF’s
page.get_text("text")  often outputs the table as “one cell per line”, so my earlier
“single-line line-item regex” parsed zero codes, leaving every codes_doc  empty → TF‑IDF
sees no tokens → empty vocabulary. That error is exactly what sklearn raises when it can’t
build any terms from the corpus. 
Fix in Iteration 6 (robust + minimal):
1. Extract codes by regex from the full PDF text (works regardless of how the table is
line-broken).
2. Add a hard guard + fallback: if TF‑IDF still fails (or all docs empty), it logs and falls
back to tabular-only instead of crashing.
3. Bump cache version so you don’t reuse a “bad cache” that stored empty docs.
Here is the full corrected script (copy‑paste runnable):
Stack Overflow
python
# -*- coding: utf-8 -*-
r"""
ITERATION 6 (FIXED) — Monolithic XGBoost (GPU) + TF-IDF (1–3 grams, 1000 features
=================================================================================
Fixes vs previous Iter 6:
- Robust code extraction: regex-find all CPT/HCPCS-like codes from full PDF text 
- TF-IDF guardrails: if vocabulary is empty, logs + falls back to tabular-only in
- Cache version bumped so a previously-bad cache won't poison current run.
===========================================================
INSTALL (run once)
===========================================================
pip install -U pandas numpy scipy scikit-learn xgboost pymupdf joblib tqdm
"""
from __future__ import annotations
import re
import time
```
