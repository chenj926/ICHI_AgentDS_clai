# AgentDS Healthcare Ch2 — 模型调参与特征选择：Prompt + Iteration Record

**Source PDF:** `IEEE ICHI - 模型调参与特征选择.pdf`

**Extraction date:** 2026-02-19

**Goal:** Provide a *detailed*, machine-readable record of the prompts, iteration actions, code artifacts, errors, and MAE outcomes, so a stronger AI can later synthesize a reproducible workflow.

> Note: The PDF text has line-wrapping and may contain minor formatting artifacts from extraction.


## 1) Key user prompts (chronological)

### Prompt A — Iter7/Iter8/Code17 discrepancy

> 为什么iteration 7能达到mae 451, iteration 8 时间长反而 mae 480+ code 17时间最长也只有mae 456?


### Prompt B — User synthesizes Iteration 9

> 根据这个3个结果我总结并合成了一个code:


### Prompt C — Report leaderboard MAE for Iter9

> 得到real mae:
📊 Score: 450.5367 (MAE)


### Prompt D — Request for a runnable full-cell optimization code

> 给我一个full code 优化我可以直接past in jupyter cell to run


### Prompt E — Debug request after CatBoost error

> TypeError: CatBoostRegressor.fit() got an unexpected keyword argument 'eval_sample_weight'  / debug


## 2) Iteration artifacts (what changed, what ran, what it produced)

This PDF contains several major runnable code artifacts and the conversation around them:


- **ITERATION 7**: Anti-overfit++ low-dim receipts features + shallow CatBoost + multi-seed bagging + stability-aware ensemble selection.

- **ITERATION 8**: Adds receipts *line distribution stats* + replaces one model with LOG-target RMSE + tail-aware objective + optional severity-bin correction.

- **Code 17**: “Code16→Code17” path: stronger regularization + stability pruning + log/deltalog dual-mode + multi-seed.

- **ITERATION 9**: User’s “Iter7-Refined” with per-chronic stratified shift optimization (Nelder-Mead) + ensemble weights search.

- **ITERATION 10**: Assistant-proposed Iter7-core + adversarial importance weighting + robust stacking + shrunk chronic shifts; initially hit CatBoost `eval_sample_weight` API mismatch, then fixed using `Pool(weight=...)`.


## 3) Timeline & metrics (as recorded in the PDF)

### Iteration 7

- **Intent:** reduce generalization gap; keep model practical, fast, LB-oriented.

- **Core:** LOW-DIM receipts features (buckets/shares/HHI/EM stats/procedure flags), shallow trees (depth 4–5), strong RSM=0.8, multi-seed bagging.

- **Ensemble:** choose weights by stability across seeds (mean + std penalty), optional baseline blend (`lam`) and small global shift.


### Iteration 8

- **Delta vs Iter7:**

  - Receipts: add line distribution features (n_line_items, line_mean/median/max/std, top1/top2 share).

  - Models: 2 RMSE (full/pruned) + **LOG-RMSE** pruned (tail-robust idea).

  - Ensemble objective adds tail penalty: `(Top20MAE - OverallMAE)`.

  - Optional severity-bin residual correction.


### Code 17 (Code16→Code17)

- **Intent:** emulate Code16 “anti-overfit” philosophy: strong regularization + ordered boosting + stability pruning.

- **Receipts features:** counts/flags/small top-codes; avoid huge sparse.

- **Modeling:** compare log1p(y) vs deltalog(y/baseline); choose by CV; multi-seed final.


### Iteration 9 (User synthesized)

- **Intent:** return to Iter7 “sweet spot” and squeeze gains via automated stratified shift optimization per chronic group.

- **Logs recorded:** OOF ~427.428 after shift; LB **450.5367 (MAE)**.


### Iteration 10 (Assistant proposed)

- **Intent:** improve LB stability and reduce OOF→LB optimism gap by:

  1) adversarial train-vs-test importance weights,

  2) robust stacking (Huber) on OOF predictions + baseline,

  3) shrunk per-chronic median residual shifts (hierarchical shrink + cap).

- **Issue:** CatBoost version mismatch for `eval_sample_weight` → fix via `catboost.Pool(..., weight=...)`.


## 4) Diagnostic reasoning captured in the PDF (Iter7 vs Iter8 vs Code17)

The PDF contains a diagnostic explanation for why Iter7 looked strong, Iter8 collapsed, and Code17 was slower but not better. Key hypotheses captured:


- **OOF vs LB mismatch:** Iter7’s ~451 likely an optimistic OOF due to post-hoc tuning on OOF residuals; LB may not improve correspondingly.

- **Receipts amount-column risk in Iter8:** line distribution stats are *extremely sensitive* to selecting the correct per-line amount; choosing a per-patient total column as “line_total” can poison features and blow up MAE.

- **LOG target mismatch:** optimizing RMSE in log space doesn’t align with MAE in original space; can induce systematic underprediction in high tail (Top20).

- **Code17 underfitting:** very strong regularization + pruning yields stability but may cap performance.


## 5) Raw extracts (for downstream AI)

Below are raw extracted blocks from the PDF. Use them as ground truth for any later summarization.


<details>
<summary>RAW — ITERATION 7 code + comments</summary>

```text
# === ITERATION 7: Anti-overfit++ on the v3(Iter15/16) 
path (keep it practical, fast, and LB-oriented) ===
# Core ideas:
#   - Still LOW-DIM receipts features (buckets + shares + 
HHI + EM stats + key procedure flags)
#   - Shallow trees (depth 4-5), strong RSM (0.8), multi-
seed bagging
#   - Stronger "generalization" ensemble selection: choose 
weights by stability across seeds (mean + std), not just 
best OOF
#   - Add ONE diverse model: MAE-loss on pruned features 
(metric-aligned but still regularized)
#   - Tiny baseline shrink option (toward baseline_next3y) 
considered in ensemble search (helps LB sometimes)
# Output: D:\AgentDs\agent_ds_healthcare\submission.csv 
with correct format.
import os, re, sys, gc, math, warnings, random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
# -----------------------------
# Paths (must match prompt)
# -----------------------------
DATA_DIR = r"D:\AgentDs\agent_ds_healthcare"
TRAIN_PATH = 

r"D:\AgentDs\agent_ds_healthcare\ed_cost_train.csv"
TEST_PATH  = 
r"D:\AgentDs\agent_ds_healthcare\ed_cost_test.csv"
PATIENTS_PATH = 
r"D:\AgentDs\agent_ds_healthcare\patients.csv"
ADM_TRAIN_PATH = 
r"D:\AgentDs\agent_ds_healthcare\admissions_train.csv"
ADM_TEST_PATH  = 
r"D:\AgentDs\agent_ds_healthcare\admissions_test.csv"
RECEIPTS_JOBLIB_PATH = 
r"D:\AgentDs\agent_ds_healthcare\receipts_parsed.joblib"
RECEIPTS_PDF_DIR = 
r"D:\AgentDs\agent_ds_healthcare\receipts_pdf"  # last 
resort only (we will NOT parse)
OUT_SUB_PATH = 
r"D:\AgentDs\agent_ds_healthcare\submission.csv"
TARGET = "ed_cost_next3y_usd"
print("="*90)
print("ITERATION 7 | v3-spirit anti-overfit++: shallow trees 
+ RSM + pruning + multi-seed + STABLE ensemble 
selection")
print("Goal: improve LB beyond ~452 by reducing 
generalization gap (avoid overfitting weights/features).")
print("="*90)
# -----------------------------
# Minimal deps
# -----------------------------
def _pip_install(pkg: str):
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", 
"install", "-q", pkg])
try:
    from joblib import load as joblib_load
except Exception:
    _pip_install("joblib")
    from joblib import load as joblib_load

try:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import mean_absolute_error
except Exception:
    _pip_install("scikit-learn")
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import mean_absolute_error
try:
    from catboost import CatBoostRegressor
except Exception:
    _pip_install("catboost")
    from catboost import CatBoostRegressor
# -----------------------------
# Config (keep fast)
# -----------------------------
class CFG:
    N_FOLDS = 7
    N_SEEDS = 5
    # iterations are upper bounds; early stopping typically 
stops earlier
    ITERS = 3000
    ES_ROUNDS = 120
    LR = 0.03
    RSM = 0.80
    # ensemble search
    W_STEP = 0.05
    LAM_GRID = [0.00, 0.03, 0.05, 0.08, 0.10, 0.15]  # 
baseline blend (small)
    SHIFT_GRID = [0.0, 0.5, 1.0]  # global median shift 
multiplier
    # small complexity penalties (LB-oriented)
    STD_PEN = 0.20    # penalize variance across seeds 
(stability)
    LAM_PEN = 8.0     # penalize baseline blend (discourage 
too large)
    SHIFT_PEN = 0.002 # penalize big shift slightly

# -----------------------------
# Utilities
# -----------------------------
def log_shape(name: str, df: pd.DataFrame):
    mem = df.memory_usage(deep=True).sum() / (1024**2)
    print(f"[{name}] shape={df.shape} | cols={df.shape[1]} | 
mem={mem:.2f} MB")
def qdict(x, qs=
(0,0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99,1.0)):
    x = np.asarray(x, dtype=float)
    return {q: float(np.quantile(x, q)) for q in qs}
def standardize_zip3(z):
    if z is None or (isinstance(z, float) and np.isnan(z)):
        return None
    s = re.sub(r"\D", "", str(z).strip())
    if not s:
        return None
    return s.zfill(3)
def norm_code(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().upper()
    if s == "" or s.lower() == "nan":
        return None
    # strip .0
    if re.fullmatch(r"\d+\.0+", s):
        s = s.split(".")[0]
    s = re.sub(r"\s+", "", s)
    return s
def safe_num(s, default=0.0):
    try:
        v = float(s)
        if np.isnan(v) or np.isinf(v): return default
        return v
    except Exception:

        return default
# -----------------------------
# Admissions features (keep simple, robust)
# -----------------------------
def load_admissions_features(adm_train_path: str, 
adm_test_path: str) -> Optional[pd.DataFrame]:
    dfs = []
    for path in [adm_train_path, adm_test_path]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "readmit_30d" in df.columns:
                df = df.drop(columns=["readmit_30d"])
            dfs.append(df)
    if not dfs:
        return None
    adm = pd.concat(dfs, ignore_index=True)
    need = ["patient_id","charlson_band","acuity_emergent"]
    if not all(c in adm.columns for c in need):
        return None
    adm["patient_id"] = pd.to_numeric(adm["patient_id"], 
errors="coerce")
    adm["charlson_band"] = 
pd.to_numeric(adm["charlson_band"], errors="coerce")
    adm["acuity_emergent"] = 
pd.to_numeric(adm["acuity_emergent"], errors="coerce")
    out = adm.groupby("patient_id").agg(
        charlson_max=("charlson_band","max"),
        charlson_mean=("charlson_band","mean"),
        pct_emergent=("acuity_emergent","mean"),
    ).reset_index()
    # fill numeric
    for c in 
["charlson_max","charlson_mean","pct_emergent"]:
        out[c] = pd.to_numeric(out[c], 
errors="coerce").fillna(0.0)
    return out
# -----------------------------
# Low-dim receipts features from parsed lineitems (v3-

style, but slightly expanded buckets)
# -----------------------------
def build_pdf_features_from_lineitems(li: pd.DataFrame) -> 
pd.DataFrame:
    li = li.copy()
    # locate columns
    code_col = None
    for c in ["code","cpt","cpt_code","hcpcs","proc_code"]:
        if c in li.columns:
            code_col = c
            break
    total_col = None
    for c in 
["line_total","line_total_usd","total","amount","line_cost","su
m_items","item_total","extended_price"]:
        if c in li.columns:
            total_col = c
            break
    if code_col is None or total_col is None or "patient_id" 
not in li.columns:
        raise ValueError("Lineitems DF missing required 
columns.")
    li["patient_id"] = pd.to_numeric(li["patient_id"], 
errors="coerce").astype("Int64")
    li = li.dropna(subset=["patient_id"]).copy()
    li["patient_id"] = li["patient_id"].astype(int)
    li["code"] = li[code_col].map(norm_code)
    li = li.dropna(subset=["code"]).copy()
    li["amt"] = pd.to_numeric(li[total_col], 
errors="coerce").fillna(0.0).astype(float)
    li.loc[li["amt"] < 0, "amt"] = 0.0
    # totals
    total = li.groupby("patient_id")
["amt"].sum().rename("receipt_total")
    li = li.join(total, on="patient_id")

    denom = li["receipt_total"].replace(0.0, np.nan)
    # code numeric where possible
    code_num = 
pd.to_numeric(li["code"].where(li["code"].str.fullmatch(r"\d
+"), None), errors="coerce")
    # buckets
    is_em = 
li["code"].isin(["99281","99282","99283","99284","99285"])
    em_map = 
{"99281":1,"99282":2,"99283":3,"99284":4,"99285":5}
    em_level = li["code"].map(em_map).fillna(0).astype(float)
    is_crit = li["code"].isin(["99291","99292"])
    is_obs = li["code"].str.startswith("G037", na=False)
    is_high = 
li["code"].isin(["31500","36556","32551","36620","92950"]) 
 # airway/lines/chest tube/cpr
    is_lab = code_num.between(80000, 89999)
    is_imaging = code_num.between(70000, 79999)
    is_proc_general = code_num.between(10000, 69999)
    is_proc_any = is_high | (is_proc_general & (~is_high) & 
(~is_em) & (~is_crit))
    # per-line shares
    share = (li["amt"] / denom).fillna(0.0)
    cost_hhi = (share * 
share).groupby(li["patient_id"]).sum().rename("cost_hhi")
    # basic counts
    n_unique_codes = li.groupby("patient_id")
["code"].nunique().rename("n_unique_codes")
    n_categories = li.groupby("patient_id").apply(
        lambda g: int(len(set(
            np.select(
                [
                    
g["code"].isin(["99281","99282","99283","99284","99285"]),

                    g["code"].isin(["99291","99292"]),
                    g["code"].str.startswith("G037", na=False),
                    
g["code"].isin(["31500","36556","32551","36620","92950"]),
                    
pd.to_numeric(g["code"].where(g["code"].str.fullmatch(r"\d
+"), None), errors="coerce").between(80000, 89999),
                    
pd.to_numeric(g["code"].where(g["code"].str.fullmatch(r"\d
+"), None), errors="coerce").between(70000, 79999),
                    
pd.to_numeric(g["code"].where(g["code"].str.fullmatch(r"\d
+"), None), errors="coerce").between(10000, 69999),
                ],
                
["em","critical","obs","high","lab","imaging","procedure"],
                default="other"
            )
        )))
    ).rename("n_categories")
    # EM stats
    n_em_codes = 
is_em.astype(int).groupby(li["patient_id"]).sum().rename("n_
em_codes")
    max_em_level = 
em_level.groupby(li["patient_id"]).max().rename("max_em_l
evel")
    sum_em_level = (em_level * 
is_em.astype(int)).groupby(li["patient_id"]).sum().rename("s
um_em_level")
    avg_em_level = (sum_em_level / n_em_codes.replace(0, 
np.nan)).fillna(0.0).rename("avg_em_level")
    n_high_em = ((em_level >= 4) & 
is_em).astype(int).groupby(li["patient_id"]).sum().rename("n
_high_em")
    # totals by bucket
    em_total = li.loc[is_em].groupby("patient_id")
["amt"].sum().rename("em_total")

    crit_total = li.loc[is_crit].groupby("patient_id")
["amt"].sum().rename("crit_total")
    proc_total = li.loc[is_proc_any].groupby("patient_id")
["amt"].sum().rename("proc_total")
    img_total = li.loc[is_imaging].groupby("patient_id")
["amt"].sum().rename("img_total")
    lab_total = li.loc[is_lab].groupby("patient_id")
["amt"].sum().rename("lab_total")
    high_total = li.loc[is_high].groupby("patient_id")
["amt"].sum().rename("high_total")
    # counts by bucket
    n_procedures = 
is_proc_any.astype(int).groupby(li["patient_id"]).sum().rena
me("n_procedures")
    n_imaging = 
is_imaging.astype(int).groupby(li["patient_id"]).sum().renam
e("n_imaging")
    n_lab = 
is_lab.astype(int).groupby(li["patient_id"]).sum().rename("n_
lab")
    # flags
    has_critical_care = 
is_crit.astype(int).groupby(li["patient_id"]).max().rename("h
as_critical_care")
    has_high_acuity = 
is_high.astype(int).groupby(li["patient_id"]).max().rename("
has_high_acuity")
    has_observation = 
is_obs.astype(int).groupby(li["patient_id"]).max().rename("h
as_observation")
    has_imaging = 
is_imaging.astype(int).groupby(li["patient_id"]).max().renam
e("has_imaging")
    def has_code(code: str, name: str):
        return 
(li["code"].eq(code).astype(int).groupby(li["patient_id"]).ma
x()).rename(name)

    has_intub_31500 = 
has_code("31500","has_intub_31500")
    has_cvc_36556 = has_code("36556","has_cvc_36556")
    has_cpr_92950 = has_code("92950","has_cpr_92950")
    has_artline_36620 = 
has_code("36620","has_artline_36620")
    has_ct_head_70450 = 
has_code("70450","has_ct_head_70450")
    has_99285 = has_code("99285","has_99285")
    has_ct_abdpel_74177 = 
has_code("74177","has_ct_abdpel_74177")
    has_troponin_84484 = 
has_code("84484","has_troponin_84484")
    has_obs_G0378 = has_code("G0378","has_obs_G0378")
    out = pd.concat([
        n_unique_codes, cost_hhi,
        n_em_codes, max_em_level, avg_em_level, n_high_em,
        has_critical_care, has_high_acuity, has_observation, 
has_imaging,
        has_intub_31500, has_cvc_36556, has_cpr_92950, 
has_artline_36620,
        has_ct_head_70450, has_99285, has_ct_abdpel_74177, 
has_troponin_84484, has_obs_G0378,
        n_categories, n_procedures, n_imaging, n_lab,
        total
    ], axis=1).reset_index()
    # merge totals (may be missing)
    for s in [em_total, crit_total, proc_total, img_total, 
lab_total, high_total]:
        out = out.merge(s.reset_index(), on="patient_id", 
how="left")
    for c in 
["em_total","crit_total","proc_total","img_total","lab_total","
high_total","receipt_total"]:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], 

errors="coerce").fillna(0.0)
    denom2 = out["receipt_total"].replace(0.0, np.nan)
    # pct shares
    out["pct_cost_em"] = (out["em_total"] / 
denom2).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    out["pct_cost_procedure"] = (out["proc_total"] / 
denom2).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    out["pct_cost_critical"] = (out["crit_total"] / 
denom2).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    out["pct_cost_imaging"] = (out["img_total"] / 
denom2).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    out["pct_cost_lab"] = (out["lab_total"] / 
denom2).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    out["pct_cost_high_acuity"] = (out["high_total"] / 
denom2).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    # cost_per_em uses total/ n_em (if none -> total)
    out["cost_per_em"] = np.where(out["n_em_codes"] > 0, 
out["receipt_total"] / out["n_em_codes"].clip(lower=1), 
out["receipt_total"])
    # composite high acuity count
    out["n_high_acuity_total"] = (
        out["has_intub_31500"].fillna(0).astype(int)
        + out["has_cvc_36556"].fillna(0).astype(int)
        + out["has_cpr_92950"].fillna(0).astype(int)
        + out["has_artline_36620"].fillna(0).astype(int)
        + out["has_critical_care"].fillna(0).astype(int)
    ).astype(int)
    # cleanup helper totals
    out.drop(columns=[c for c in 
["em_total","crit_total","proc_total","img_total","lab_total","
high_total","receipt_total"] if c in out.columns], 
inplace=True)
    # fill numeric
    for c in out.columns:

        if c == "patient_id":
            continue
        out[c] = pd.to_numeric(out[c], 
errors="coerce").fillna(0.0)
    return out
def load_receipts_joblib(joblib_path: str) -> 
Optional[pd.DataFrame]:
    if not os.path.exists(joblib_path):
        return None
    data = joblib_load(joblib_path)
    # if dict contains lineitems_df
    if isinstance(data, dict):
        for k in 
["lineitems_df","lineitems","items_df","items","line_items_df
","line_items"]:
            if k in data and isinstance(data[k], pd.DataFrame):
                li = data[k]
                return build_pdf_features_from_lineitems(li)
        # else: maybe patient_id->dict
        try:
            df = pd.DataFrame.from_dict(data, orient="index")
            df.index.name = "patient_id"
            df = df.reset_index()
            return df
        except Exception:
            return None
    # if direct df
    if isinstance(data, pd.DataFrame):
        df = data
        if "patient_id" in df.columns and any(c in df.columns 
for c in ["code","cpt","cpt_code","hcpcs","proc_code"]):
            return build_pdf_features_from_lineitems(df)
        return df
    # if list/tuple

    if isinstance(data, (list, tuple)):
        dfs = [x for x in data if isinstance(x, pd.DataFrame)]
        for df in dfs:
            if "patient_id" in df.columns and any(c in 
df.columns for c in 
["code","cpt","cpt_code","hcpcs","proc_code"]):
                return build_pdf_features_from_lineitems(df)
        for df in dfs:
            if "patient_id" in df.columns:
                return df
    return None
# -----------------------------
# Feature engineering (numeric-only, v3 style)
# -----------------------------
def build_features(ed_df: pd.DataFrame,
                   patients_df: pd.DataFrame,
                   adm_df: Optional[pd.DataFrame],
                   pdf_df: Optional[pd.DataFrame]) -> 
pd.DataFrame:
    feat = ed_df.copy()
    # chronic encoding
    chronic_map = {"PNEUMONIA":0, "DIABETESCOMP":1, 
"HF":2}
    feat["primary_chronic"] = 
feat["primary_chronic"].astype(str)
    feat["chronic_encoded"] = 
feat["primary_chronic"].str.upper().map(chronic_map).fillna(
-1).astype(float)
    # base priors
    feat["prior_ed_visits_5y"] = 
pd.to_numeric(feat["prior_ed_visits_5y"], 
errors="coerce").fillna(0.0)
    feat["prior_ed_cost_5y_usd"] = 
pd.to_numeric(feat["prior_ed_cost_5y_usd"], 
errors="coerce").fillna(0.0)

    # transformations (anti-tail)
    feat["prior_cost_cap20k"] = 
feat["prior_ed_cost_5y_usd"].clip(upper=20000.0)
    feat["sqrt_prior_cost"] = 
np.sqrt(feat["prior_ed_cost_5y_usd"].clip(lower=0.0))
    feat["log_prior_cost"] = 
np.log1p(feat["prior_ed_cost_5y_usd"].clip(lower=0.0))
    feat["log_prior_cost_cap20k"] = 
np.log1p(feat["prior_cost_cap20k"].clip(lower=0.0))
    feat["log_visits"] = 
np.log1p(feat["prior_ed_visits_5y"].clip(lower=0.0))
    feat["cost_per_visit"] = feat["prior_ed_cost_5y_usd"] / 
feat["prior_ed_visits_5y"].clip(lower=1.0)
    # baseline for LB-safe blending
    feat["baseline_next3y"] = feat["prior_ed_cost_5y_usd"] * 
(3.0/5.0)
    # patients encodings
    p = patients_df.copy()
    p["patient_id"] = pd.to_numeric(p["patient_id"], 
errors="coerce").astype(int)
    p["age"] = pd.to_numeric(p["age"], 
errors="coerce").fillna(p["age"].median())
    p["sex_encoded"] = (p["sex"].astype(str).str.upper() == 
"M").astype(int)
    ins = p["insurance"].astype(str).str.lower()
    ins_map = {"private":2, "public":1, "self_pay":0, 
"selfpay":0}
    p["insurance_encoded"] = 
ins.map(ins_map).fillna(-1).astype(float)
    z3 = p["zip3"].apply(standardize_zip3).astype("string")
    zr = z3.fillna("000").str.replace(r"\D","", 
regex=True).str.zfill(3).str[0]
    p["zip_region"] = pd.to_numeric(zr, 
errors="coerce").fillna(-1).astype(float)
    feat = 

feat.merge(p[["patient_id","age","sex_encoded","insurance_
encoded","zip_region"]], on="patient_id", how="left")
    feat["ins_x_chronic"] = 
feat["insurance_encoded"].fillna(-1) * 
feat["chronic_encoded"].fillna(-1)
    # admissions aggregates
    if adm_df is not None:
        feat = feat.merge(adm_df, on="patient_id", 
how="left")
        for c in 
["charlson_max","charlson_mean","pct_emergent"]:
            if c in feat.columns:
                feat[c] = pd.to_numeric(feat[c], 
errors="coerce").fillna(feat[c].median())
    # receipts features
    if pdf_df is not None:
        feat = feat.merge(pdf_df, on="patient_id", how="left")
        for c in pdf_df.columns:
            if c == "patient_id": continue
            feat[c] = pd.to_numeric(feat[c], 
errors="coerce").fillna(feat[c].median())
    # a couple of LIGHT interactions (still low-risk)
    if "pct_cost_critical" in feat.columns:
        feat["logprior_x_pctcritical"] = feat["log_prior_cost"] * 
feat["pct_cost_critical"]
    if "n_high_acuity_total" in feat.columns:
        feat["logprior_x_highacu"] = feat["log_prior_cost"] * 
feat["n_high_acuity_total"]
    # derived stable ratios
    if "n_unique_codes" in feat.columns:
        feat["cost_per_code"] = feat["prior_ed_cost_5y_usd"] / 
feat["n_unique_codes"].clip(lower=1.0)
    # numeric safety
    feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    for c in feat.columns:

        if c in ["patient_id", "primary_chronic", TARGET]:
            continue
        feat[c] = pd.to_numeric(feat[c], errors="coerce")
        if feat[c].isna().any():
            feat[c] = feat[c].fillna(feat[c].median())
    return feat
def get_numeric_feature_cols(df: pd.DataFrame) -> 
List[str]:
    exclude = 
{"patient_id","primary_chronic",TARGET,"sex","insurance","zi
p3"}
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols
# -----------------------------
# Training (multi-seed + fold bagging)
# -----------------------------
def train_models(train_feat: pd.DataFrame, test_feat: 
pd.DataFrame,
                 feat_full: List[str], feat_pruned: List[str]) -> 
Tuple[Dict[str, List[np.ndarray]], Dict[str, List[np.ndarray]], 
Dict[str, List[int]]]:
    y = train_feat[TARGET].values.astype(float)
    # stratify: chronic + target bin (v3)
    tmp = train_feat[["primary_chronic", TARGET]].copy()
    tmp["cost_bin"] = pd.qcut(tmp[TARGET], q=5, 
labels=False, duplicates="drop")
    tmp["strat"] = tmp["primary_chronic"].astype(str) + "_" + 
tmp["cost_bin"].astype(str)
    strat = LabelEncoder().fit_transform(tmp["strat"].values)
    # 3 models (2 RMSE + 1 MAE)

    # NOTE: keep shallow + RSM + regularization to 
preserve generalization.
    model_specs = {
        "A_RMSE_full_d5": dict(
            loss_function="RMSE", eval_metric="MAE",
            depth=5, l2_leaf_reg=5, min_data_in_leaf=28,
            learning_rate=CFG.LR, iterations=CFG.ITERS,
            rsm=CFG.RSM, random_strength=1.0,
        ),
        "B_RMSE_pruned_d4": dict(
            loss_function="RMSE", eval_metric="MAE",
            depth=4, l2_leaf_reg=4, min_data_in_leaf=32,
            learning_rate=CFG.LR, iterations=CFG.ITERS,
            rsm=CFG.RSM, random_strength=2.0,
        ),
        "C_MAE_pruned_d4": dict(
            loss_function="MAE", eval_metric="MAE",
            depth=4, l2_leaf_reg=7, min_data_in_leaf=36,
            learning_rate=CFG.LR, iterations=CFG.ITERS,
            rsm=CFG.RSM, random_strength=1.5,
        ),
    }
    model_featcols = {
        "A_RMSE_full_d5": feat_full,
        "B_RMSE_pruned_d4": feat_pruned,
        "C_MAE_pruned_d4": feat_pruned,
    }
    oof_by_seed = {m: [] for m in model_specs.keys()}
    test_by_seed = {m: [] for m in model_specs.keys()}
    best_iters = {m: [] for m in model_specs.keys()}
    print("\n[training] CatBoost CPU | shallow trees | 
rsm=0.8 | multi-seed bagging")
    print("Models:", list(model_specs.keys()))
    print(f"Seeds={CFG.N_SEEDS}, Folds={CFG.N_FOLDS}\n")
    for seed_idx in range(CFG.N_SEEDS):
        rs = SEED + seed_idx * 7
        kf = StratifiedKFold(n_splits=CFG.N_FOLDS, 

shuffle=True, random_state=rs)
        oof_seed = {m: np.zeros(len(train_feat), dtype=float) 
for m in model_specs.keys()}
        test_seed = {m: np.zeros(len(test_feat), dtype=float) 
for m in model_specs.keys()}
        for fold, (ti, vi) in enumerate(kf.split(train_feat, strat), 
1):
            for mname, params in model_specs.items():
                cols = model_featcols[mname]
                X_tr = train_feat[cols].iloc[ti]
                y_tr = y[ti]
                X_va = train_feat[cols].iloc[vi]
                y_va = y[vi]
                X_te = test_feat[cols]
                cb = CatBoostRegressor(
                    **params,
                    task_type="CPU",
                    thread_count=-1,
                    verbose=0,
                    allow_writing_files=False,
                    random_seed=rs + fold * 31 + (hash(mname) 
% 997),
                    early_stopping_rounds=CFG.ES_ROUNDS,
                )
                cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0)
                try:
                    
best_iters[mname].append(int(cb.get_best_iteration()))
                except Exception:
                    pass
                pred_va = cb.predict(X_va)
                pred_te = cb.predict(X_te)
                oof_seed[mname][vi] = pred_va
                test_seed[mname] += pred_te / CFG.N_FOLDS

                del cb
                gc.collect()
        # per-seed MAE
        seed_maes = {m: float(mean_absolute_error(y, 
oof_seed[m])) for m in model_specs.keys()}
        print(f"  Seed {seed_idx+1}/{CFG.N_SEEDS} OOF MAE: 
" + " | ".join([f"{m}={seed_maes[m]:.2f}" for m in 
model_specs.keys()]))
        for m in model_specs.keys():
            oof_by_seed[m].append(oof_seed[m])
            test_by_seed[m].append(test_seed[m])
    print("\n[seed-averaged OOF MAE per model]")
    for m in oof_by_seed.keys():
        avg_oof = np.mean(np.vstack(oof_by_seed[m]), 
axis=0)
        print(f"  {m:18s}: {mean_absolute_error(y, 
avg_oof):.2f}")
    # print median best iters
    print("\n[median best_iteration per model] (for 
reference)")
    for m in best_iters.keys():
        if best_iters[m]:
            print(f"  {m:18s}: {int(np.median(best_iters[m]))}")
        else:
            print(f"  {m:18s}: (n/a)")
    return oof_by_seed, test_by_seed, best_iters
# -----------------------------
# Ensemble selection (stability across seeds)
# -----------------------------
def stable_ensemble_search(train_feat: pd.DataFrame,
                           oof_by_seed: Dict[str, List[np.ndarray]],
                           test_by_seed: Dict[str, List[np.ndarray]],
                           baseline_vec: np.ndarray) -> 
Tuple[np.ndarray, np.ndarray, Dict]:

    y = train_feat[TARGET].values.astype(float)
    model_names = list(oof_by_seed.keys())
    assert len(model_names) == 3, "This search expects 
exactly 3 models."
    # precompute seed-averaged oof/test per model
    oof_avg = {m: np.mean(np.vstack(oof_by_seed[m]), 
axis=0) for m in model_names}
    test_avg = {m: np.mean(np.vstack(test_by_seed[m]), 
axis=0) for m in model_names}
    # weights grid
    step = CFG.W_STEP
    grid = np.round(np.arange(0.0, 1.0 + 1e-9, step), 10)
    best = None
    top_list = []
    def eval_combo(wA, wB, wC, lam, shift_mult):
        # per-seed MAE for stability
        maes = []
        for s in range(CFG.N_SEEDS):
            pred = wA*oof_by_seed[model_names[0]][s] + 
wB*oof_by_seed[model_names[1]][s] + 
wC*oof_by_seed[model_names[2]][s]
            pred = (1.0-lam)*pred + lam*baseline_vec
            # global shift derived from avg oof (not per seed), 
scaled
            pred_avg = wA*oof_avg[model_names[0]] + 
wB*oof_avg[model_names[1]] + 
wC*oof_avg[model_names[2]]
            pred_avg = (1.0-lam)*pred_avg + lam*baseline_vec
            shift = float(np.median(y - pred_avg)) * shift_mult
            pred = pred + shift
            maes.append(float(mean_absolute_error(y, pred)))
        mean_m = float(np.mean(maes))
        std_m = float(np.std(maes, ddof=0))
        # objective prefers stability + simplicity
        obj = mean_m + CFG.STD_PEN*std_m + 
CFG.LAM_PEN*lam + CFG.SHIFT_PEN*abs(shift_mult)

        return obj, mean_m, std_m
    # enumerate
    for wA in grid:
        for wB in grid:
            wC = 1.0 - wA - wB
            if wC < -1e-9:
                continue
            wC = float(max(0.0, wC))
            if abs((wA+wB+wC) - 1.0) > 1e-6:
                continue
            for lam in CFG.LAM_GRID:
                for sm in CFG.SHIFT_GRID:
                    obj, mean_m, std_m = eval_combo(wA, wB, 
wC, lam, sm)
                    rec = (obj, mean_m, std_m, wA, wB, wC, lam, 
sm)
                    # keep top 12
                    top_list.append(rec)
                    if best is None or obj < best[0]:
                        best = rec
    top_list.sort(key=lambda x: x[0])
    print("\n[ensemble search] Top candidates by robust 
objective (mean + std + simplicity penalties):")
    for i, rec in enumerate(top_list[:10], 1):
        obj, mean_m, std_m, wA, wB, wC, lam, sm = rec
        print(f"  #{i:02d} obj={obj:.3f} meanMAE={mean_m:.3f} 
std={std_m:.3f} | w=({wA:.2f},{wB:.2f},{wC:.2f}) | lam=
{lam:.2f} | shift_mult={sm:.1f}")
    # finalize with best config
    _, mean_m, std_m, wA, wB, wC, lam, sm = best
    mA, mB, mC = model_names
    oof_final = wA*oof_avg[mA] + wB*oof_avg[mB] + 
wC*oof_avg[mC]
    test_final = wA*test_avg[mA] + wB*test_avg[mB] + 
wC*test_avg[mC]

    oof_final = (1.0-lam)*oof_final + lam*baseline_vec
    test_final = (1.0-lam)*test_final + lam*
(baseline_vec[:len(test_final)]*0 + (baseline_vec.mean()))  # 
placeholder; will be overwritten outside
    # we need baseline for test separately; will handle 
outside
    # compute shift from OOF (avg)
    shift = float(np.median(y - oof_final)) * sm
    oof_final = oof_final + shift
    meta = {
        "models_order": model_names,
        "weights": (float(wA), float(wB), float(wC)),
        "lam_baseline": float(lam),
        "shift_mult": float(sm),
        "shift_value": float(shift),
        "oof_mean_mae_across_seeds": float(mean_m),
        "oof_std_mae_across_seeds": float(std_m),
    }
    return oof_final, test_final, meta
# -----------------------------
# Optional small group shift (very conservative)
# -----------------------------
def apply_chronic_group_shift(train_feat: pd.DataFrame, 
pred_oof: np.ndarray, shrink: float = 0.3) -> 
Tuple[np.ndarray, Dict]:
    y = train_feat[TARGET].values.astype(float)
    chronic = train_feat["primary_chronic"].astype(str).values
    resid = y - pred_oof
    shifts = {}
    for g in np.unique(chronic):
        med = float(np.median(resid[chronic == g]))
        shifts[g] = shrink * med
    pred2 = pred_oof.copy()
    for g, s in shifts.items():
        pred2[chronic == g] += s
    return pred2, shifts

# -----------------------------
# Main load
# -----------------------------
def must_exist(path: str, name: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {name}: {path}")
must_exist(TRAIN_PATH, "TRAIN")
must_exist(TEST_PATH, "TEST")
must_exist(PATIENTS_PATH, "patients")
must_exist(ADM_TRAIN_PATH, "admissions_train")
must_exist(ADM_TEST_PATH, "admissions_test")
# receipts joblib may exist; if missing we will run but warn
if not os.path.exists(RECEIPTS_JOBLIB_PATH):
    print("[warn] receipts_parsed.joblib missing -> receipts 
features will be empty (likely worse).")
print("\n[load] reading CSVs...")
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
patients = pd.read_csv(PATIENTS_PATH)
adm_tr = pd.read_csv(ADM_TRAIN_PATH)
adm_te = pd.read_csv(ADM_TEST_PATH)
log_shape("ed_cost_train", train)
log_shape("ed_cost_test", test)
log_shape("patients", patients)
log_shape("admissions_train", adm_tr)
log_shape("admissions_test", adm_te)
print("\n[target stats]")
print(train[TARGET].describe().to_string())
# ids
for df in [train, test, patients]:
    df["patient_id"] = pd.to_numeric(df["patient_id"], 
errors="coerce").astype(int)
# admissions
print("\n[admissions] building robust aggregates...")

adm_df = load_admissions_features(ADM_TRAIN_PATH, 
ADM_TEST_PATH)
if adm_df is None:
    print("  admissions features: None")
else:
    print(f"  admissions features: {adm_df.shape} | cols=
{list(adm_df.columns)}")
# receipts
print("\n[receipts] loading receipts_parsed.joblib and 
building low-dim receipt features...")
pdf_df = None
if os.path.exists(RECEIPTS_JOBLIB_PATH):
    try:
        pdf_df = load_receipts_joblib(RECEIPTS_JOBLIB_PATH)
        if pdf_df is not None:
            # normalize patient_id
            pdf_df["patient_id"] = 
pd.to_numeric(pdf_df["patient_id"], 
errors="coerce").astype("Int64")
            pdf_df = pdf_df.dropna(subset=
["patient_id"]).copy()
            pdf_df["patient_id"] = 
pdf_df["patient_id"].astype(int)
            pdf_df = pdf_df.drop_duplicates("patient_id", 
keep="last")
            print(f"  receipt_feat shape: {pdf_df.shape}")
            print(f"  receipt_feat cols ({len(pdf_df.columns)-1}): 
{[c for c in pdf_df.columns if c!='patient_id']}")
        else:
            print("  [warn] could not build receipt features from 
joblib structure.")
    except Exception as e:
        print(f"  [warn] receipts joblib load/build failed: {e}")
        pdf_df = None
else:
    print("  [warn] receipts joblib missing; skipping receipts 
features.")
# sanity invariant: sum(line totals) == prior_ed_cost_5y_usd 

(should match); check using available receipt feature proxy 
if possible
# Our low-dim features do not include sum_items; use n/a.
# build features
print("\n[features] building train/test feature frames...")
train_feat = build_features(train, patients, adm_df, pdf_df)
test_feat  = build_features(test,  patients, adm_df, pdf_df)
# choose features
feat_full = get_numeric_feature_cols(train_feat)
# remove any accidental target leakage
feat_full = [c for c in feat_full if c != TARGET]
# PRUNED set: hand-picked stable features (from your 
v3/v16 + latest importances)
pruned_candidates = [
    # priors + transforms
    
"prior_ed_visits_5y","prior_ed_cost_5y_usd","sqrt_prior_cost
","log_prior_cost","log_prior_cost_cap20k","cost_per_visit","l
og_visits",
    "baseline_next3y",
    # demographics
    
"chronic_encoded","age","sex_encoded","insurance_encod
ed","zip_region","ins_x_chronic",
    # admissions
    "charlson_max","charlson_mean","pct_emergent",
    # receipt robust
    
"cost_per_em","cost_hhi","pct_cost_procedure","pct_cost_cr
itical","pct_cost_high_acuity",
    
"n_high_acuity_total","has_critical_care","has_99285","max_
em_level","n_unique_codes",
    # light interactions
    "logprior_x_pctcritical","logprior_x_highacu",
    # stable ratio
    "cost_per_code",

]
feat_pruned = [c for c in pruned_candidates if c in 
train_feat.columns]
# median fill + numeric conversion safety for selected 
columns
for c in set(feat_full + feat_pruned):
    if c in train_feat.columns:
        med = train_feat[c].median() if not 
train_feat[c].isna().all() else 0.0
        train_feat[c] = pd.to_numeric(train_feat[c], 
errors="coerce").fillna(med)
        test_feat[c]  = pd.to_numeric(test_feat[c], 
errors="coerce").fillna(med)
# drop constant columns (anti-overfit)
def drop_constants(cols: List[str], df: pd.DataFrame) -> 
List[str]:
    out = []
    for c in cols:
        if c not in df.columns:
            continue
        if df[c].nunique(dropna=False) <= 1:
            continue
        out.append(c)
    return out
feat_full = drop_constants(feat_full, train_feat)
feat_pruned = drop_constants(feat_pruned, train_feat)
print(f"  FULL feature count:   {len(feat_full)}")
print(f"  PRUNED feature count: {len(feat_pruned)}")
print(f"  PRUNED features: {feat_pruned}")
miss_train = int(train_feat[feat_full].isna().sum().sum())
miss_test  = int(test_feat[feat_full].isna().sum().sum())
print(f"  Missing cells after fill (full): train={miss_train} test=
{miss_test}")
# train

oof_by_seed, test_by_seed, best_iters = 
train_models(train_feat, test_feat, feat_full, feat_pruned)
# baseline vectors for blending
baseline_oof = 
train_feat["baseline_next3y"].values.astype(float)
baseline_test = 
test_feat["baseline_next3y"].values.astype(float)
# ensemble search (stable)
# NOTE: stable_ensemble_search expects baseline_vec 
length == train length; test baseline handled outside.
oof_ens, test_ens_placeholder, meta = 
stable_ensemble_search(train_feat, oof_by_seed, 
test_by_seed, baseline_oof)
# build final test ensemble using chosen weights
mA, mB, mC = meta["models_order"]
wA, wB, wC = meta["weights"]
lam = meta["lam_baseline"]
shift = meta["shift_value"]
test_avg = {m: np.mean(np.vstack(test_by_seed[m]), 
axis=0) for m in meta["models_order"]}
test_ens = wA*test_avg[mA] + wB*test_avg[mB] + 
wC*test_avg[mC]
test_ens = (1.0-lam)*test_ens + lam*baseline_test
test_ens = test_ens + shift
# optional chronic shift (very conservative; only apply if 
meaningful OOF gain)
y = train_feat[TARGET].values.astype(float)
base_mae = float(mean_absolute_error(y, oof_ens))
best_oof = oof_ens
best_shift = {"type": "none"}
for shrink in [0.0, 0.3, 0.5]:
    if shrink <= 0:
        continue
    oof2, shifts = apply_chronic_group_shift(train_feat, 

oof_ens, shrink=shrink)
    m2 = float(mean_absolute_error(y, oof2))
    # require noticeable improvement to accept (avoid LB 
overfit)
    if m2 < base_mae - 0.15:
        base_mae = m2
        best_oof = oof2
        best_shift = {"type": "chronic_group", "shrink": shrink, 
"shifts": shifts}
# apply chosen chronic shift to test
if best_shift["type"] == "chronic_group":
    test_chronic = 
test_feat["primary_chronic"].astype(str).values
    for g, s in best_shift["shifts"].items():
        test_ens[test_chronic == g] += s
# clip predictions to reasonable range (LB-safe; small)
y_max = float(np.max(y))
test_ens = np.clip(test_ens, 0.0, y_max * 1.5)
# feature importance from a full-fit Model A (compliance + 
insight)
print("\n[full-train] fitting Model A on full train for feature 
importance...")
A_params = dict(
    loss_function="RMSE", eval_metric="MAE",
    depth=5, l2_leaf_reg=5, min_data_in_leaf=28,
    learning_rate=CFG.LR, iterations=CFG.ITERS,
    rsm=CFG.RSM, random_strength=1.0,
    task_type="CPU", thread_count=-1,
    verbose=0, allow_writing_files=False,
    random_seed=SEED,
)
mA_full = CatBoostRegressor(**A_params)
mA_full.fit(train_feat[feat_full], y, verbose=0)
try:
    imp = mA_full.get_feature_importance()
    imp_df = pd.DataFrame({"feature": feat_full, 
"importance": imp}).sort_values("importance", 

ascending=False).head(10)
    print("\n[Top 10 feature importances] (Model A full)")
    print(imp_df.to_string(index=False))
except Exception as e:
    print(f"[warn] feature importance failed: {e}")
del mA_full
gc.collect()
# final logs
final_oof_mae = float(mean_absolute_error(y, best_oof))
print("\n" + "="*70)
print("[FINAL OOF]")
print(f"  ensemble OOF MAE (stable search + optional 
chronic shift): {final_oof_mae:.2f}")
print("  ensemble meta:", meta)
print("  extra shift:", best_shift["type"], 
("shrink="+str(best_shift.get("shrink")) if 
best_shift["type"]!="none" else ""))
print("  OOF pred quantiles:", qdict(best_oof, qs=
(0,0.01,0.05,0.1,0.5,0.9,0.95,0.99,1.0)))
print("="*70)
# write submission
sub = pd.DataFrame({
    "patient_id": test["patient_id"].values.astype(int),
    "ed_cost_next3y_usd": np.round(test_ens.astype(float), 2)
})
sub = sub[["patient_id", "ed_cost_next3y_usd"]]
out_path = Path(OUT_SUB_PATH)
out_path.parent.mkdir(parents=True, exist_ok=True)
sub.to_csv(out_path, index=False)
# sanity checks
print("\n[SUBMISSION sanity checks]")
print("submission shape:", sub.shape)
print("submission columns exactly:", list(sub.columns))
print("any NaNs in preds:", 
bool(np.isnan(sub["ed_cost_next3y_usd"]).any()))
print("pred min/median/max:", 

float(sub["ed_cost_next3y_usd"].min()), 
float(sub["ed_cost_next3y_usd"].median()), 
float(sub["ed_cost_next3y_usd"].max()))
print("pred quantiles:", 
qdict(sub["ed_cost_next3y_usd"].values))
print("\nSaved submission to:", str(out_path))
print("\nPaste back: (1) leaderboard MAE, (2) FINAL OOF 
MAE, (3) ensemble meta + extra shift, (4) pred quantiles.")
```

</details>


<details>
<summary>RAW — ITERATION 8 code + comments</summary>

```text
# === ITERATION 8: v3-spirit++ (keep low-dim + stronger 
anti-overfit + add LOG-target model + add line-
distribution stats) ===
# What changes vs Iteration 7:
#   1) Receipts: add *line distribution* features 
(n_line_items, line_max/median/std, top1/top2 share) — 
low-dim but informative.
#      (You previously saw pdf_line_median/mean/max 
dominate in earlier attempts; now we add them safely in 
the low-dim setup.)
#   2) Models: keep 2 RMSE models (full/pruned) + replace 
the unused MAE model with a LOG-RMSE pruned model 
(tail-robust).
#   3) Ensemble: still stability-aware across seeds, but also 
penalize *tail underperformance* via (Top20MAE - 
OverallMAE).
#   4) Optional severity-bin residual correction (0/1/2) with 
small shrink, only if it improves OOF.
#
# Output: D:\AgentDs\agent_ds_healthcare\submission.csv 
(patient_id, ed_cost_next3y_usd)
import os, re, sys, gc, warnings, random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
# -----------------------------
# Paths (MUST match prompt)
# -----------------------------
DATA_DIR = r"D:\AgentDs\agent_ds_healthcare"
TRAIN_PATH = 
r"D:\AgentDs\agent_ds_healthcare\ed_cost_train.csv"
TEST_PATH  = 
r"D:\AgentDs\agent_ds_healthcare\ed_cost_test.csv"
PATIENTS_PATH = 
r"D:\AgentDs\agent_ds_healthcare\patients.csv"
ADM_TRAIN_PATH = 
r"D:\AgentDs\agent_ds_healthcare\admissions_train.csv"
ADM_TEST_PATH  = 
r"D:\AgentDs\agent_ds_healthcare\admissions_test.csv"
RECEIPTS_JOBLIB_PATH = 
r"D:\AgentDs\agent_ds_healthcare\receipts_parsed.joblib"
RECEIPTS_PDF_DIR = 
r"D:\AgentDs\agent_ds_healthcare\receipts_pdf"  # last 
resort only (we will NOT parse)
OUT_SUB_PATH = 
r"D:\AgentDs\agent_ds_healthcare\submission.csv"
TARGET = "ed_cost_next3y_usd"
print("="*100)
print("ITERATION 8 | v3-spirit++: low-dim + line-
distribution stats + add LOG-RMSE model + tail-aware 
stable ensemble")
print("Goal: push LB below ~451 by improving 
generalization & tail handling without overfitting.")
print("="*100)
# -----------------------------
# Minimal deps
# -----------------------------

def _pip_install(pkg: str):
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", 
"install", "-q", pkg])
try:
    from joblib import load as joblib_load
except Exception:
    _pip_install("joblib")
    from joblib import load as joblib_load
try:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import mean_absolute_error
except Exception:
    _pip_install("scikit-learn")
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import mean_absolute_error
try:
    from catboost import CatBoostRegressor
except Exception:
    _pip_install("catboost")
    from catboost import CatBoostRegressor
# -----------------------------
# Config
# -----------------------------
class CFG:
    N_FOLDS = 7
    N_SEEDS = 5
    ITERS = 3500
    ES_ROUNDS = 130
    LR = 0.03
    RSM = 0.80
    # ensemble search
    W_STEP = 0.05

    LAM_GRID = [0.00, 0.03, 0.05, 0.08, 0.10, 0.15]   # 
baseline shrink/blend
    SHIFT_GRID = [0.0, 0.5, 1.0]                       # global 
median shift multiplier
    # robust objective weights (LB-oriented)
    STD_PEN = 0.20
    TAIL_PEN = 0.15        # penalize (Top20MAE - 
OverallMAE)
    LAM_PEN = 2.5          # discourage too much baseline 
blending
    SHIFT_ABS_PEN = 0.0005 # discourage huge global shift 
(usually tiny anyway)
# -----------------------------
# Utilities
# -----------------------------
def log_shape(name: str, df: pd.DataFrame):
    mem = df.memory_usage(deep=True).sum() / (1024**2)
    print(f"[{name}] shape={df.shape} | cols={df.shape[1]} | 
mem={mem:.2f} MB")
def qdict(x, qs=
(0,0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99,1.0)):
    x = np.asarray(x, dtype=float)
    return {q: float(np.quantile(x, q)) for q in qs}
def standardize_zip3(z):
    if z is None or (isinstance(z, float) and np.isnan(z)):
        return None
    s = re.sub(r"\D", "", str(z).strip())
    if not s:
        return None
    return s.zfill(3)
def norm_code(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().upper()
    if s == "" or s.lower() == "nan":

        return None
    if re.fullmatch(r"\d+\.0+", s):
        s = s.split(".")[0]
    s = re.sub(r"\s+", "", s)
    return s
# -----------------------------
# Admissions features
# -----------------------------
def load_admissions_features(adm_train_path: str, 
adm_test_path: str) -> Optional[pd.DataFrame]:
    dfs = []
    for path in [adm_train_path, adm_test_path]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "readmit_30d" in df.columns:
                df = df.drop(columns=["readmit_30d"])
            dfs.append(df)
    if not dfs:
        return None
    adm = pd.concat(dfs, ignore_index=True)
    need = ["patient_id","charlson_band","acuity_emergent"]
    if not all(c in adm.columns for c in need):
        return None
    adm["patient_id"] = pd.to_numeric(adm["patient_id"], 
errors="coerce")
    adm["charlson_band"] = 
pd.to_numeric(adm["charlson_band"], errors="coerce")
    adm["acuity_emergent"] = 
pd.to_numeric(adm["acuity_emergent"], errors="coerce")
    out = adm.groupby("patient_id").agg(
        charlson_max=("charlson_band","max"),
        charlson_mean=("charlson_band","mean"),
        pct_emergent=("acuity_emergent","mean"),
    ).reset_index()
    for c in 
["charlson_max","charlson_mean","pct_emergent"]:
        out[c] = pd.to_numeric(out[c], 
errors="coerce").fillna(0.0)
    return out

# -----------------------------
# Receipts: low-dim features + NEW line distribution stats 
(vectorized)
# -----------------------------
def build_pdf_features_from_lineitems(li: pd.DataFrame) -> 
pd.DataFrame:
    li = li.copy()
    # locate columns
    code_col = None
    for c in ["code","cpt","cpt_code","hcpcs","proc_code"]:
        if c in li.columns:
            code_col = c
            break
    total_col = None
    for c in 
["line_total","line_total_usd","total","amount","line_cost","su
m_items","item_total","extended_price"]:
        if c in li.columns:
            total_col = c
            break
    if code_col is None or total_col is None or "patient_id" 
not in li.columns:
        raise ValueError("Lineitems DF missing required 
columns.")
    li["patient_id"] = pd.to_numeric(li["patient_id"], 
errors="coerce").astype("Int64")
    li = li.dropna(subset=["patient_id"]).copy()
    li["patient_id"] = li["patient_id"].astype(int)
    li["code"] = li[code_col].map(norm_code)
    li = li.dropna(subset=["code"]).copy()
    li["amt"] = pd.to_numeric(li[total_col], 
errors="coerce").fillna(0.0).astype(float)
    li.loc[li["amt"] < 0, "amt"] = 0.0
    # per-patient total + line stats

    g = li.groupby("patient_id")
    receipt_total = g["amt"].sum().rename("receipt_total")
    n_line_items = g.size().rename("n_line_items")
    line_mean = g["amt"].mean().rename("line_mean")
    line_median = g["amt"].median().rename("line_median")
    line_max = g["amt"].max().rename("line_max")
    line_std = 
g["amt"].std(ddof=0).fillna(0.0).rename("line_std")
    # top1/top2 shares (sorted within patient)
    li_sorted = li.sort_values(["patient_id","amt"], ascending=
[True, False])
    top1_amt = li_sorted.groupby("patient_id")
["amt"].nth(0).fillna(0.0).rename("line_top1_amt")
    top2_amt = li_sorted.groupby("patient_id")
["amt"].nth(1).fillna(0.0).rename("line_top2_amt")
    tmp = pd.concat([receipt_total, n_line_items, line_mean, 
line_median, line_max, line_std, top1_amt, top2_amt], 
axis=1).reset_index()
    denom = tmp["receipt_total"].replace(0.0, np.nan)
    tmp["line_top1_share"] = (tmp["line_top1_amt"] / 
denom).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    tmp["line_top2_share"] = ((tmp["line_top1_amt"] + 
tmp["line_top2_amt"]) / denom).replace([np.inf,-np.inf], 
np.nan).fillna(0.0)
    tmp.drop(columns=
["line_top1_amt","line_top2_amt","receipt_total"], 
inplace=True)
    # bucket flags
    code = li["code"]
    code_num = 
pd.to_numeric(code.where(code.str.fullmatch(r"\d+"), 
None), errors="coerce")
    is_em = 
code.isin(["99281","99282","99283","99284","99285"])
    em_map = 
{"99281":1,"99282":2,"99283":3,"99284":4,"99285":5}

    em_level = code.map(em_map).fillna(0).astype(float)
    is_crit = code.isin(["99291","99292"])
    is_obs = code.str.startswith("G037", na=False)
    is_high = 
code.isin(["31500","36556","32551","36620","92950"])
    is_lab = code_num.between(80000, 89999)
    is_imaging = code_num.between(70000, 79999)
    is_proc_general = code_num.between(10000, 69999)
    is_proc_any = is_high | (is_proc_general & (~is_high) & 
(~is_em) & (~is_crit))
    # per-line shares for HHI
    li = li.join(receipt_total, on="patient_id")
    share = (li["amt"] / li["receipt_total"].replace(0.0, 
np.nan)).fillna(0.0)
    cost_hhi = (share * 
share).groupby(li["patient_id"]).sum().rename("cost_hhi")
    # counts and EM stats
    n_unique_codes = li.groupby("patient_id")
["code"].nunique().rename("n_unique_codes")
    n_em_codes = 
is_em.astype(int).groupby(li["patient_id"]).sum().rename("n_
em_codes")
    max_em_level = 
em_level.groupby(li["patient_id"]).max().rename("max_em_l
evel")
    sum_em_level = (em_level * 
is_em.astype(int)).groupby(li["patient_id"]).sum().rename("s
um_em_level")
    avg_em_level = (sum_em_level / n_em_codes.replace(0, 
np.nan)).fillna(0.0).rename("avg_em_level")
    n_high_em = ((em_level >= 4) & 
is_em).astype(int).groupby(li["patient_id"]).sum().rename("n
_high_em")
    # totals by bucket (for pct shares)
    em_total = li.loc[is_em].groupby("patient_id")

["amt"].sum().rename("em_total")
    crit_total = li.loc[is_crit].groupby("patient_id")
["amt"].sum().rename("crit_total")
    proc_total = li.loc[is_proc_any].groupby("patient_id")
["amt"].sum().rename("proc_total")
    img_total = li.loc[is_imaging].groupby("patient_id")
["amt"].sum().rename("img_total")
    lab_total = li.loc[is_lab].groupby("patient_id")
["amt"].sum().rename("lab_total")
    high_total = li.loc[is_high].groupby("patient_id")
["amt"].sum().rename("high_total")
    # counts by bucket
    n_procedures = 
is_proc_any.astype(int).groupby(li["patient_id"]).sum().rena
me("n_procedures")
    n_imaging = 
is_imaging.astype(int).groupby(li["patient_id"]).sum().renam
e("n_imaging")
    n_lab = 
is_lab.astype(int).groupby(li["patient_id"]).sum().rename("n_
lab")
    # flags
    has_critical_care = 
is_crit.astype(int).groupby(li["patient_id"]).max().rename("h
as_critical_care")
    has_high_acuity = 
is_high.astype(int).groupby(li["patient_id"]).max().rename("
has_high_acuity")
    has_observation = 
is_obs.astype(int).groupby(li["patient_id"]).max().rename("h
as_observation")
    has_imaging = 
is_imaging.astype(int).groupby(li["patient_id"]).max().renam
e("has_imaging")
    def has_code(code_str: str, name: str):
        return 
(li["code"].eq(code_str).astype(int).groupby(li["patient_id"]).

max()).rename(name)
    has_intub_31500 = 
has_code("31500","has_intub_31500")
    has_cvc_36556 = has_code("36556","has_cvc_36556")
    has_cpr_92950 = has_code("92950","has_cpr_92950")
    has_artline_36620 = 
has_code("36620","has_artline_36620")
    has_ct_head_70450 = 
has_code("70450","has_ct_head_70450")
    has_99285 = has_code("99285","has_99285")
    has_ct_abdpel_74177 = 
has_code("74177","has_ct_abdpel_74177")
    has_troponin_84484 = 
has_code("84484","has_troponin_84484")
    has_obs_G0378 = has_code("G0378","has_obs_G0378")
    # category count (vectorized)
    cat = np.select(
        [is_em, is_crit, is_obs, is_high, is_lab, is_imaging, 
(is_proc_general & (~is_high))],
        
["em","critical","obs","high","lab","imaging","procedure"],
        default="other"
    )
    n_categories = pd.Series(cat, 
index=li.index).groupby(li["patient_id"]).nunique().rename("
n_categories")
    base = pd.concat([
        n_unique_codes, cost_hhi,
        n_em_codes, max_em_level, avg_em_level, n_high_em,
        has_critical_care, has_high_acuity, has_observation, 
has_imaging,
        has_intub_31500, has_cvc_36556, has_cpr_92950, 
has_artline_36620,
        has_ct_head_70450, has_99285, has_ct_abdpel_74177, 
has_troponin_84484, has_obs_G0378,
        n_categories, n_procedures, n_imaging, n_lab,
    ], axis=1).reset_index()

    # merge totals and compute pct shares
    out = base
    for s in [em_total, crit_total, proc_total, img_total, 
lab_total, high_total, receipt_total]:
        out = out.merge(s.reset_index(), on="patient_id", 
how="left")
    for c in 
["em_total","crit_total","proc_total","img_total","lab_total","
high_total","receipt_total"]:
        out[c] = pd.to_numeric(out[c], 
errors="coerce").fillna(0.0)
    denom2 = out["receipt_total"].replace(0.0, np.nan)
    out["pct_cost_em"] = (out["em_total"] / 
denom2).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    out["pct_cost_procedure"] = (out["proc_total"] / 
denom2).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    out["pct_cost_critical"] = (out["crit_total"] / 
denom2).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    out["pct_cost_imaging"] = (out["img_total"] / 
denom2).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    out["pct_cost_lab"] = (out["lab_total"] / 
denom2).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    out["pct_cost_high_acuity"] = (out["high_total"] / 
denom2).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    out["n_em_codes"] = pd.to_numeric(out["n_em_codes"], 
errors="coerce").fillna(0.0)
    out["cost_per_em"] = np.where(out["n_em_codes"] > 0, 
out["receipt_total"] / out["n_em_codes"].clip(lower=1), 
out["receipt_total"])
    out["n_high_acuity_total"] = (
        out["has_intub_31500"].fillna(0).astype(int)
        + out["has_cvc_36556"].fillna(0).astype(int)
        + out["has_cpr_92950"].fillna(0).astype(int)
        + out["has_artline_36620"].fillna(0).astype(int)
        + out["has_critical_care"].fillna(0).astype(int)

    ).astype(int)
    # drop helper totals
    out.drop(columns=
["em_total","crit_total","proc_total","img_total","lab_total","
high_total","receipt_total"], inplace=True)
    # merge NEW line distribution stats
    out = out.merge(tmp, on="patient_id", how="left")
    for c in tmp.columns:
        if c == "patient_id": continue
        out[c] = pd.to_numeric(out[c], 
errors="coerce").fillna(out[c].median())
    # numeric fill
    for c in out.columns:
        if c == "patient_id": continue
        out[c] = pd.to_numeric(out[c], 
errors="coerce").fillna(0.0)
    return out
def load_receipts_joblib(joblib_path: str) -> 
Optional[pd.DataFrame]:
    if not os.path.exists(joblib_path):
        return None
    data = joblib_load(joblib_path)
    if isinstance(data, dict):
        for k in 
["lineitems_df","lineitems","items_df","items","line_items_df
","line_items"]:
            if k in data and isinstance(data[k], pd.DataFrame):
                return build_pdf_features_from_lineitems(data[k])
        # else: try patient_id->dict
        try:
            df = pd.DataFrame.from_dict(data, orient="index")
            df.index.name = "patient_id"
            return df.reset_index()
        except Exception:

            return None
    if isinstance(data, pd.DataFrame):
        df = data
        if "patient_id" in df.columns and any(c in df.columns 
for c in ["code","cpt","cpt_code","hcpcs","proc_code"]):
            return build_pdf_features_from_lineitems(df)
        return df
    if isinstance(data, (list, tuple)):
        dfs = [x for x in data if isinstance(x, pd.DataFrame)]
        for df in dfs:
            if "patient_id" in df.columns and any(c in 
df.columns for c in 
["code","cpt","cpt_code","hcpcs","proc_code"]):
                return build_pdf_features_from_lineitems(df)
        for df in dfs:
            if "patient_id" in df.columns:
                return df
    return None
# -----------------------------
# Feature engineering
# -----------------------------
def build_features(ed_df: pd.DataFrame,
                   patients_df: pd.DataFrame,
                   adm_df: Optional[pd.DataFrame],
                   pdf_df: Optional[pd.DataFrame]) -> 
pd.DataFrame:
    feat = ed_df.copy()
    # chronic encoding (keep v3-style)
    chronic_map = {"PNEUMONIA":0, "DIABETESCOMP":1, 
"HF":2}
    feat["primary_chronic"] = 
feat["primary_chronic"].astype(str)
    feat["chronic_encoded"] = 
feat["primary_chronic"].str.upper().map(chronic_map).fillna(
-1).astype(float)

    # priors
    feat["prior_ed_visits_5y"] = 
pd.to_numeric(feat["prior_ed_visits_5y"], 
errors="coerce").fillna(0.0)
    feat["prior_ed_cost_5y_usd"] = 
pd.to_numeric(feat["prior_ed_cost_5y_usd"], 
errors="coerce").fillna(0.0)
    feat["prior_cost_cap20k"] = 
feat["prior_ed_cost_5y_usd"].clip(upper=20000.0)
    feat["sqrt_prior_cost"] = 
np.sqrt(feat["prior_ed_cost_5y_usd"].clip(lower=0.0))
    feat["log_prior_cost"] = 
np.log1p(feat["prior_ed_cost_5y_usd"].clip(lower=0.0))
    feat["log_prior_cost_cap20k"] = 
np.log1p(feat["prior_cost_cap20k"].clip(lower=0.0))
    feat["log_visits"] = 
np.log1p(feat["prior_ed_visits_5y"].clip(lower=0.0))
    feat["cost_per_visit"] = feat["prior_ed_cost_5y_usd"] / 
feat["prior_ed_visits_5y"].clip(lower=1.0)
    # baseline anchor
    feat["baseline_next3y"] = feat["prior_ed_cost_5y_usd"] * 
(3.0/5.0)
    # patient encodings
    p = patients_df.copy()
    p["patient_id"] = pd.to_numeric(p["patient_id"], 
errors="coerce").astype(int)
    p["age"] = pd.to_numeric(p["age"], 
errors="coerce").fillna(p["age"].median())
    p["sex_encoded"] = (p["sex"].astype(str).str.upper() == 
"M").astype(int)
    ins = p["insurance"].astype(str).str.lower()
    ins_map = {"private":2, "public":1, "self_pay":0, 
"selfpay":0}
    p["insurance_encoded"] = 
ins.map(ins_map).fillna(-1).astype(float)
    z3 = p["zip3"].apply(standardize_zip3).astype("string")

    zr = z3.fillna("000").str.replace(r"\D","", 
regex=True).str.zfill(3).str[0]
    p["zip_region"] = pd.to_numeric(zr, 
errors="coerce").fillna(-1).astype(float)
    feat = 
feat.merge(p[["patient_id","age","sex_encoded","insurance_
encoded","zip_region"]], on="patient_id", how="left")
    feat["ins_x_chronic"] = 
feat["insurance_encoded"].fillna(-1) * 
feat["chronic_encoded"].fillna(-1)
    # admissions
    if adm_df is not None:
        feat = feat.merge(adm_df, on="patient_id", 
how="left")
        for c in 
["charlson_max","charlson_mean","pct_emergent"]:
            if c in feat.columns:
                feat[c] = pd.to_numeric(feat[c], 
errors="coerce").fillna(feat[c].median())
    # receipts
    if pdf_df is not None:
        feat = feat.merge(pdf_df, on="patient_id", how="left")
        for c in pdf_df.columns:
            if c == "patient_id": continue
            feat[c] = pd.to_numeric(feat[c], 
errors="coerce").fillna(feat[c].median())
    # light interactions (low-risk)
    if "pct_cost_critical" in feat.columns:
        feat["logprior_x_pctcritical"] = feat["log_prior_cost"] * 
feat["pct_cost_critical"]
    if "n_high_acuity_total" in feat.columns:
        feat["logprior_x_highacu"] = feat["log_prior_cost"] * 
feat["n_high_acuity_total"]
    # NEW robust ratios (low-dim)
    if "n_unique_codes" in feat.columns:

        feat["cost_per_code"] = feat["prior_ed_cost_5y_usd"] / 
feat["n_unique_codes"].clip(lower=1.0)
        feat["highacu_per_code"] = 
feat.get("n_high_acuity_total", 0.0) / 
(feat["n_unique_codes"].clip(lower=1.0))
        feat["em_per_code"] = feat.get("n_em_codes", 0.0) / 
(feat["n_unique_codes"].clip(lower=1.0))
    if "n_line_items" in feat.columns:
        feat["highacu_per_line"] = 
feat.get("n_high_acuity_total", 0.0) / 
feat["n_line_items"].clip(lower=1.0)
    # numeric safety fill
    feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    for c in feat.columns:
        if c in 
["patient_id","primary_chronic",TARGET,"sex","insurance","zi
p3"]:
            continue
        feat[c] = pd.to_numeric(feat[c], errors="coerce")
        if feat[c].isna().any():
            feat[c] = feat[c].fillna(feat[c].median())
    return feat
def get_numeric_feature_cols(df: pd.DataFrame) -> 
List[str]:
    exclude = 
{"patient_id","primary_chronic",TARGET,"sex","insurance","zi
p3"}
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols
def drop_constants(cols: List[str], df: pd.DataFrame) -> 
List[str]:

    out = []
    for c in cols:
        if c not in df.columns:
            continue
        if df[c].nunique(dropna=False) <= 1:
            continue
        out.append(c)
    return out
# -----------------------------
# Training: 3 models (2 RMSE + 1 LOG-RMSE)
# -----------------------------
def train_models(train_feat: pd.DataFrame, test_feat: 
pd.DataFrame,
                 feat_full: List[str], feat_pruned: List[str]) -> 
Tuple[Dict[str, List[np.ndarray]], Dict[str, List[np.ndarray]], 
Dict[str, List[int]]]:
    y = train_feat[TARGET].values.astype(float)
    y_log = np.log1p(y)
    # stratify: chronic + target bins
    tmp = train_feat[["primary_chronic", TARGET]].copy()
    tmp["cost_bin"] = pd.qcut(tmp[TARGET], q=5, 
labels=False, duplicates="drop")
    tmp["strat"] = tmp["primary_chronic"].astype(str) + "_" + 
tmp["cost_bin"].astype(str)
    strat = LabelEncoder().fit_transform(tmp["strat"].values)
    model_specs = {
        # slightly more regularized than Iter7 (LB-oriented)
        "A_RMSE_full_d5": dict(
            loss_function="RMSE", eval_metric="MAE",
            depth=5, l2_leaf_reg=6, min_data_in_leaf=34,
            learning_rate=CFG.LR, iterations=CFG.ITERS,
            rsm=CFG.RSM, random_strength=1.2,
            bootstrap_type="Bayesian", 
bagging_temperature=1.2,
        ),
        "B_RMSE_pruned_d4": dict(
            loss_function="RMSE", eval_metric="MAE",

            depth=4, l2_leaf_reg=6, min_data_in_leaf=42,
            learning_rate=CFG.LR, iterations=CFG.ITERS,
            rsm=CFG.RSM, random_strength=2.0,
            bootstrap_type="Bayesian", 
bagging_temperature=1.5,
        ),
        # NEW: log-target RMSE (tail robust)
        "C_LOGRMSE_pruned_d4": dict(
            loss_function="RMSE", eval_metric="RMSE",
            depth=4, l2_leaf_reg=5, min_data_in_leaf=42,
            learning_rate=CFG.LR, iterations=CFG.ITERS,
            rsm=CFG.RSM, random_strength=1.0,
            bootstrap_type="Bayesian", 
bagging_temperature=1.0,
        ),
    }
    model_featcols = {
        "A_RMSE_full_d5": feat_full,
        "B_RMSE_pruned_d4": feat_pruned,
        "C_LOGRMSE_pruned_d4": feat_pruned,
    }
    oof_by_seed = {m: [] for m in model_specs.keys()}
    test_by_seed = {m: [] for m in model_specs.keys()}
    best_iters = {m: [] for m in model_specs.keys()}
    y_q80 = float(np.quantile(y, 0.80))
    top_mask = (y >= y_q80)
    print("\n[training] CatBoost CPU | shallow trees | strong 
RSM | 5 seeds x 7 folds")
    print("Models:", list(model_specs.keys()))
    print(f"Top20 threshold (y>=q80): {y_q80:.2f}\n")
    for seed_idx in range(CFG.N_SEEDS):
        rs = SEED + seed_idx * 7
        kf = StratifiedKFold(n_splits=CFG.N_FOLDS, 
shuffle=True, random_state=rs)
        oof_seed = {m: np.zeros(len(train_feat), dtype=float) 

for m in model_specs.keys()}
        test_seed = {m: np.zeros(len(test_feat), dtype=float) 
for m in model_specs.keys()}
        for fold, (ti, vi) in enumerate(kf.split(train_feat, strat), 
1):
            for mname, params in model_specs.items():
                cols = model_featcols[mname]
                X_tr = train_feat[cols].iloc[ti]
                X_va = train_feat[cols].iloc[vi]
                X_te = test_feat[cols]
                if mname.startswith("C_LOG"):
                    y_tr = y_log[ti]
                    y_va = y_log[vi]
                else:
                    y_tr = y[ti]
                    y_va = y[vi]
                cb = CatBoostRegressor(
                    **params,
                    task_type="CPU",
                    thread_count=-1,
                    verbose=0,
                    allow_writing_files=False,
                    random_seed=rs + fold * 31 + (hash(mname) 
% 997),
                    early_stopping_rounds=CFG.ES_ROUNDS,
                )
                cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0)
                try:
                    
best_iters[mname].append(int(cb.get_best_iteration()))
                except Exception:
                    pass
                pred_va = cb.predict(X_va)
                pred_te = cb.predict(X_te)

                if mname.startswith("C_LOG"):
                    pred_va = np.expm1(pred_va)
                    pred_te = np.expm1(pred_te)
                pred_va = np.clip(pred_va, 0.0, None)
                pred_te = np.clip(pred_te, 0.0, None)
                oof_seed[mname][vi] = pred_va
                test_seed[mname] += pred_te / CFG.N_FOLDS
                del cb
                gc.collect()
        # per-seed logs: overall + top20
        parts = []
        for m in model_specs.keys():
            mae_all = float(mean_absolute_error(y, 
oof_seed[m]))
            mae_top = float(mean_absolute_error(y[top_mask], 
oof_seed[m][top_mask]))
            parts.append(f"
{m}=all{mae_all:.2f}/top{mae_top:.2f}")
        print(f"  Seed {seed_idx+1}/{CFG.N_SEEDS} OOF: " + " 
| ".join(parts))
        for m in model_specs.keys():
            oof_by_seed[m].append(oof_seed[m])
            test_by_seed[m].append(test_seed[m])
    print("\n[seed-averaged OOF MAE per model] (overall / 
top20)")
    for m in model_specs.keys():
        oof_avg = np.mean(np.vstack(oof_by_seed[m]), 
axis=0)
        mae_all = float(mean_absolute_error(y, oof_avg))
        mae_top = float(mean_absolute_error(y[top_mask], 
oof_avg[top_mask]))
        print(f"  {m:20s}: all={mae_all:.2f} | top20=
{mae_top:.2f}")

    print("\n[median best_iteration per model] (reference)")
    for m in model_specs.keys():
        med = int(np.median(best_iters[m])) if best_iters[m] 
else None
        print(f"  {m:20s}: {med}")
    return oof_by_seed, test_by_seed, best_iters
# -----------------------------
# Ensemble selection (stable + tail-aware)
# -----------------------------
def ensemble_search(train_feat: pd.DataFrame,
                    oof_by_seed: Dict[str, List[np.ndarray]],
                    test_by_seed: Dict[str, List[np.ndarray]],
                    baseline_oof: np.ndarray,
                    baseline_test: np.ndarray) -> Tuple[np.ndarray, 
np.ndarray, Dict]:
    y = train_feat[TARGET].values.astype(float)
    y_q80 = float(np.quantile(y, 0.80))
    top_mask = (y >= y_q80)
    models = list(oof_by_seed.keys())
    assert len(models) == 3, "This ensemble search assumes 
3 models."
    oof_avg = {m: np.mean(np.vstack(oof_by_seed[m]), 
axis=0) for m in models}
    test_avg = {m: np.mean(np.vstack(test_by_seed[m]), 
axis=0) for m in models}
    grid = np.round(np.arange(0.0, 1.0 + 1e-9, CFG.W_STEP), 
10)
    best = None
    records = []
    def eval_combo(wA, wB, wC, lam, shift_mult):
        # avg pred for shift estimation (stable)

        pred_avg = wA*oof_avg[models[0]] + 
wB*oof_avg[models[1]] + wC*oof_avg[models[2]]
        pred_avg = (1.0-lam)*pred_avg + lam*baseline_oof
        shift_val = float(np.median(y - pred_avg)) * shift_mult
        maes = []
        maes_top = []
        for s in range(CFG.N_SEEDS):
            pred = wA*oof_by_seed[models[0]][s] + 
wB*oof_by_seed[models[1]][s] + 
wC*oof_by_seed[models[2]][s]
            pred = (1.0-lam)*pred + lam*baseline_oof
            pred = pred + shift_val
            maes.append(float(mean_absolute_error(y, pred)))
            
maes_top.append(float(mean_absolute_error(y[top_mask], 
pred[top_mask])))
        mean_all = float(np.mean(maes))
        std_all = float(np.std(maes, ddof=0))
        mean_top = float(np.mean(maes_top))
        obj = (
            mean_all
            + CFG.STD_PEN*std_all
            + CFG.TAIL_PEN*max(0.0, (mean_top - mean_all))
            + CFG.LAM_PEN*lam
            + CFG.SHIFT_ABS_PEN*abs(shift_val)
        )
        return obj, mean_all, std_all, mean_top, shift_val
    for wA in grid:
        for wB in grid:
            wC = 1.0 - wA - wB
            if wC < -1e-9:
                continue
            wC = float(max(0.0, wC))
            if abs((wA+wB+wC) - 1.0) > 1e-6:
                continue
            for lam in CFG.LAM_GRID:

                for sm in CFG.SHIFT_GRID:
                    obj, mean_all, std_all, mean_top, shift_val = 
eval_combo(wA, wB, wC, lam, sm)
                    rec = (obj, mean_all, std_all, mean_top, wA, 
wB, wC, lam, sm, shift_val)
                    records.append(rec)
                    if best is None or obj < best[0]:
                        best = rec
    records.sort(key=lambda x: x[0])
    print("\n[ensemble search] Top 12 candidates (robust 
objective):")
    for i, rec in enumerate(records[:12], 1):
        obj, mean_all, std_all, mean_top, wA, wB, wC, lam, sm, 
shift_val = rec
        print(f"  #{i:02d} obj={obj:.3f} | all={mean_all:.3f} std=
{std_all:.3f} top20={mean_top:.3f} | "
              f"w=({wA:.2f},{wB:.2f},{wC:.2f}) lam={lam:.2f} 
shift_mult={sm:.1f} shift={shift_val:+.2f}")
    obj, mean_all, std_all, mean_top, wA, wB, wC, lam, sm, 
shift_val = best
    meta = {
        "models_order": models,
        "weights": (float(wA), float(wB), float(wC)),
        "lam_baseline": float(lam),
        "shift_mult": float(sm),
        "shift_value": float(shift_val),
        "oof_mean_mae_across_seeds": float(mean_all),
        "oof_std_mae_across_seeds": float(std_all),
        "oof_top20_mae_across_seeds": float(mean_top),
        "top20_threshold": float(y_q80),
    }
    # final OOF avg
    oof_final = wA*oof_avg[models[0]] + 
wB*oof_avg[models[1]] + wC*oof_avg[models[2]]
    oof_final = (1.0-lam)*oof_final + lam*baseline_oof
    oof_final = oof_final + shift_val

    # final TEST avg
    test_final = wA*test_avg[models[0]] + 
wB*test_avg[models[1]] + wC*test_avg[models[2]]
    test_final = (1.0-lam)*test_final + lam*baseline_test
    test_final = test_final + shift_val
    return oof_final, test_final, meta
# -----------------------------
# Optional severity-bin correction (0/1/2) using OOF 
residual medians (shrunk)
# -----------------------------
def severity_bin(train_feat: pd.DataFrame) -> np.ndarray:
    # Safely get 'n_high_acuity_total', defaulting to zeros if 
missing
    if "n_high_acuity_total" in train_feat.columns:
        hi = pd.to_numeric(train_feat["n_high_acuity_total"], 
errors="coerce").fillna(0.0).values
    else:
        hi = np.zeros(len(train_feat))
    # Safely get 'has_critical_care', defaulting to zeros if 
missing
    if "has_critical_care" in train_feat.columns:
        crit = pd.to_numeric(train_feat["has_critical_care"], 
errors="coerce").fillna(0.0).values
    else:
        crit = np.zeros(len(train_feat))
    # 0: no high acuity and no critical
    # 1: mild (exactly 1 high acuity, no critical)
    # 2: severe (critical OR >=2 high acuity)
    sev = np.where((crit >= 1) | (hi >= 2), 2, np.where(hi >= 
1, 1, 0)).astype(int)
    return sev
def apply_bin_shift(y: np.ndarray, pred: np.ndarray, bins: 
np.ndarray, shrink: float) -> Tuple[np.ndarray, Dict[int, 
float]]:

    resid = y - pred
    shifts = {}
    for b in [0,1,2]:
        r = resid[bins == b]
        shifts[b] = float(np.median(r)) if len(r) else 0.0
    pred2 = pred.copy()
    for b in [0,1,2]:
        pred2[bins == b] += shrink * shifts[b]
    return pred2, {b: float(shrink*shifts[b]) for b in shifts}
# -----------------------------
# Main
# -----------------------------
def must_exist(path: str, name: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {name}: {path}")
must_exist(TRAIN_PATH, "TRAIN")
must_exist(TEST_PATH, "TEST")
must_exist(PATIENTS_PATH, "patients")
must_exist(ADM_TRAIN_PATH, "admissions_train")
must_exist(ADM_TEST_PATH, "admissions_test")
if not os.path.exists(RECEIPTS_JOBLIB_PATH):
    print("[warn] receipts_parsed.joblib missing -> receipts 
features will be absent (likely worse).")
print("\n[load] reading CSVs...")
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
patients = pd.read_csv(PATIENTS_PATH)
adm_tr = pd.read_csv(ADM_TRAIN_PATH)
adm_te = pd.read_csv(ADM_TEST_PATH)
log_shape("ed_cost_train", train)
log_shape("ed_cost_test", test)
log_shape("patients", patients)
log_shape("admissions_train", adm_tr)
log_shape("admissions_test", adm_te)
print("\n[target stats]")

print(train[TARGET].describe().to_string())
# ids
for df in [train, test, patients]:
    df["patient_id"] = pd.to_numeric(df["patient_id"], 
errors="coerce").astype(int)
# admissions
print("\n[admissions] building aggregates...")
adm_df = load_admissions_features(ADM_TRAIN_PATH, 
ADM_TEST_PATH)
print("  admissions features:", None if adm_df is None else 
adm_df.shape)
# receipts
print("\n[receipts] loading receipts_parsed.joblib -> low-
dim + line distribution features...")
pdf_df = None
if os.path.exists(RECEIPTS_JOBLIB_PATH):
    try:
        pdf_df = load_receipts_joblib(RECEIPTS_JOBLIB_PATH)
        if pdf_df is not None:
            pdf_df["patient_id"] = 
pd.to_numeric(pdf_df["patient_id"], 
errors="coerce").astype("Int64")
            pdf_df = pdf_df.dropna(subset=
["patient_id"]).copy()
            pdf_df["patient_id"] = 
pdf_df["patient_id"].astype(int)
            pdf_df = pdf_df.drop_duplicates("patient_id", 
keep="last")
            print(f"  receipt_feat shape: {pdf_df.shape}")
            feat_cols = [c for c in pdf_df.columns if c != 
"patient_id"]
            print(f"  receipt_feat cols ({len(feat_cols)}): 
{feat_cols}")
        else:
            print("  [warn] Could not build receipt features from 
joblib structure.")
    except Exception as e:

        print(f"  [warn] receipts joblib failed: {e}")
        pdf_df = None
else:
    print("  [warn] receipts joblib missing; skipping receipts 
features.")
# build features
print("\n[features] building model frames...")
train_feat = build_features(train, patients, adm_df, pdf_df)
test_feat  = build_features(test,  patients, adm_df, pdf_df)
feat_full = 
drop_constants(get_numeric_feature_cols(train_feat), 
train_feat)
# PRUNED set: stable hand-picked features (plus new line 
stats)
pruned_candidates = [
    
"prior_ed_visits_5y","prior_ed_cost_5y_usd","prior_cost_cap
20k","sqrt_prior_cost","log_prior_cost","log_prior_cost_cap2
0k","cost_per_visit","log_visits",
    "baseline_next3y",
    
"chronic_encoded","age","sex_encoded","insurance_encod
ed","zip_region","ins_x_chronic",
    "charlson_max","charlson_mean","pct_emergent",
    
"cost_per_em","cost_hhi","pct_cost_procedure","pct_cost_cr
itical","pct_cost_high_acuity",
    
"n_high_acuity_total","has_critical_care","has_99285","max_
em_level","n_unique_codes",
    "logprior_x_pctcritical","logprior_x_highacu",
    "cost_per_code","highacu_per_code","em_per_code",
    # NEW line distribution stats
    
"n_line_items","line_median","line_max","line_std","line_top
1_share","line_top2_share",
]

feat_pruned = drop_constants([c for c in 
pruned_candidates if c in train_feat.columns], train_feat)
# fill numeric for selected cols
for c in set(feat_full + feat_pruned + [TARGET]):
    if c in train_feat.columns:
        med = train_feat[c].median() if not 
train_feat[c].isna().all() else 0.0
        train_feat[c] = pd.to_numeric(train_feat[c], 
errors="coerce").fillna(med)
    if c in test_feat.columns:
        med = train_feat[c].median() if (c in train_feat.columns 
and not train_feat[c].isna().all()) else 0.0
        test_feat[c] = pd.to_numeric(test_feat[c], 
errors="coerce").fillna(med)
print(f"  FULL feature count:   {len(feat_full)}")
print(f"  PRUNED feature count: {len(feat_pruned)}")
print(f"  PRUNED features: {feat_pruned}")
print(f"  Missing cells after fill: train=
{int(train_feat[feat_full].isna().sum().sum())} test=
{int(test_feat[feat_full].isna().sum().sum())}")
# train CV bagging
oof_by_seed, test_by_seed, best_iters = 
train_models(train_feat, test_feat, feat_full, feat_pruned)
# baseline vectors
baseline_oof = 
train_feat["baseline_next3y"].values.astype(float)
baseline_test = 
test_feat["baseline_next3y"].values.astype(float)
# ensemble
oof_ens, test_ens, meta = ensemble_search(train_feat, 
oof_by_seed, test_by_seed, baseline_oof, baseline_test)
y = train_feat[TARGET].values.astype(float)
base_mae = float(mean_absolute_error(y, oof_ens))
y_q80 = meta["top20_threshold"]

top_mask = (y >= y_q80)
base_top = float(mean_absolute_error(y[top_mask], 
oof_ens[top_mask]))
print("\n[ENS result before severity correction]")
print("  meta:", meta)
print(f"  OOF MAE: {base_mae:.3f} | Top20 MAE: 
{base_top:.3f}")
print("  OOF pred quantiles:", qdict(oof_ens, qs=
(0,0.01,0.05,0.1,0.5,0.9,0.95,0.99,1.0)))
# severity-bin correction (optional)
bins_tr = severity_bin(train_feat)
bins_te = severity_bin(test_feat)  # same function works 
(uses available cols)
best_oof = oof_ens.copy()
best_test = test_ens.copy()
best_corr = {"type":"none"}
for shrink in [0.0, 0.20, 0.30, 0.40]:
    if shrink <= 0:
        continue
    oof2, shifts = apply_bin_shift(y, oof_ens, bins_tr, 
shrink=shrink)
    mae2 = float(mean_absolute_error(y, oof2))
    top2 = float(mean_absolute_error(y[top_mask], 
oof2[top_mask]))
    # accept only if improves overall by a meaningful 
margin (avoid LB overfit)
    if mae2 < base_mae - 0.08:
        base_mae, base_top = mae2, top2
        best_oof = oof2
        # apply same shifts to test
        test2 = test_ens.copy()
        for b, s in shifts.items():
            test2[bins_te == b] += s
        best_test = test2
        best_corr = {"type":"severity_bin_shift", "shrink": 
shrink, "shifts": shifts}

print("\n[FINAL OOF after optional severity correction]")
print(f"  OOF MAE: {base_mae:.3f} | Top20 MAE: 
{base_top:.3f}")
print("  correction:", best_corr)
print("  OOF pred quantiles:", qdict(best_oof, qs=
(0,0.01,0.05,0.1,0.5,0.9,0.95,0.99,1.0)))
# clip test predictions
y_max = float(np.max(y))
best_test = np.clip(best_test, 0.0, y_max*1.5)
# -----------------------------
# Full-train fits (compliance) + feature importance
# -----------------------------
print("\n[full-train] fitting final models on FULL train 
(compliance) ...")
y_log = np.log1p(y)
def fit_full_model(name: str, params: dict, cols: List[str], 
y_fit: np.ndarray) -> CatBoostRegressor:
    it_med = int(np.median(best_iters[name])) if 
best_iters.get(name) else 800
    it_use = int(max(300, it_med + 80))
    p = dict(params)
    p.pop("early_stopping_rounds", None)
    p["iterations"] = it_use
    cb = CatBoostRegressor(
        **p,
        task_type="CPU",
        thread_count=-1,
        verbose=0,
        allow_writing_files=False,
        random_seed=SEED + 2025 + (hash(name) % 1000),
    )
    cb.fit(train_feat[cols], y_fit, verbose=0)
    print(f"  fitted {name:20s} | iterations={it_use} | features=
{len(cols)}")
    return cb

# reconstruct model params used (same as in 
train_models)
PARAMS = {
    "A_RMSE_full_d5": dict(
        loss_function="RMSE", eval_metric="MAE",
        depth=5, l2_leaf_reg=6, min_data_in_leaf=34,
        learning_rate=CFG.LR, iterations=CFG.ITERS,
        rsm=CFG.RSM, random_strength=1.2,
        bootstrap_type="Bayesian", 
bagging_temperature=1.2,
    ),
    "B_RMSE_pruned_d4": dict(
        loss_function="RMSE", eval_metric="MAE",
        depth=4, l2_leaf_reg=6, min_data_in_leaf=42,
        learning_rate=CFG.LR, iterations=CFG.ITERS,
        rsm=CFG.RSM, random_strength=2.0,
        bootstrap_type="Bayesian", 
bagging_temperature=1.5,
    ),
    "C_LOGRMSE_pruned_d4": dict(
        loss_function="RMSE", eval_metric="RMSE",
        depth=4, l2_leaf_reg=5, min_data_in_leaf=42,
        learning_rate=CFG.LR, iterations=CFG.ITERS,
        rsm=CFG.RSM, random_strength=1.0,
        bootstrap_type="Bayesian", 
bagging_temperature=1.0,
    ),
}
COLS = {"A_RMSE_full_d5": feat_full, "B_RMSE_pruned_d4": 
feat_pruned, "C_LOGRMSE_pruned_d4": feat_pruned}
YFIT = {"A_RMSE_full_d5": y, "B_RMSE_pruned_d4": y, 
"C_LOGRMSE_pruned_d4": y_log}
full_models = {}
for m in ["A_RMSE_full_d5", "B_RMSE_pruned_d4", 
"C_LOGRMSE_pruned_d4"]:
    full_models[m] = fit_full_model(m, PARAMS[m], 
COLS[m], YFIT[m])
# feature importance from B (pruned) tends to be more 

stable
try:
    imp = 
full_models["B_RMSE_pruned_d4"].get_feature_importance(
)
    imp_df = pd.DataFrame({"feature": feat_pruned, 
"importance": imp}).sort_values("importance", 
ascending=False).head(10)
    print("\n[Top 10 feature importances] 
(B_RMSE_pruned_d4 full)")
    print(imp_df.to_string(index=False))
except Exception as e:
    print(f"[warn] feature importance failed: {e}")
# -----------------------------
# Write submission
# -----------------------------
sub = pd.DataFrame({
    "patient_id": test["patient_id"].values.astype(int),
    "ed_cost_next3y_usd": np.round(best_test.astype(float), 
2),
})[["patient_id","ed_cost_next3y_usd"]]
out_path = Path(OUT_SUB_PATH)
out_path.parent.mkdir(parents=True, exist_ok=True)
sub.to_csv(out_path, index=False)
# sanity
print("\n[SUBMISSION sanity checks]")
print("overall CV MAE (ensemble OOF, after correction):", 
f"{base_mae:.3f}")
print("ensemble meta:", meta)
print("correction used:", best_corr)
print("submission shape:", sub.shape)
print("submission columns exactly:", list(sub.columns))
print("any NaNs in preds:", 
bool(np.isnan(sub["ed_cost_next3y_usd"]).any()))
print("pred min/median/max:", 
float(sub["ed_cost_next3y_usd"].min()), 
float(sub["ed_cost_next3y_usd"].median()), 

float(sub["ed_cost_next3y_usd"].max()))
print("pred quantiles:", 
qdict(sub["ed_cost_next3y_usd"].values))
print("\nSaved submission to:", str(out_path))
print("\nPaste back: (1) leaderboard MAE, (2) OOF 
MAE+Top20 MAE, (3) ensemble meta, (4) correction used, 
(5) pred quantiles.")
```

</details>


<details>
<summary>RAW — Code 17 (Code16→Code17) code + comments</summary>

```text
# Code 17 (based on Code16 philosophy): shallow 
CatBoost + strong regularization 
(RSM/subsample/L2/min_leaf) + stability pruning + multi-
seed
# Single-cell, end-to-end, Windows paths fixed as 
requested.
import os, re, gc, random, warnings
from pathlib import Path
import numpy as np
import pandas as pd
SEED = 42
def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
set_seed(SEED)
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 250)
pd.set_option("display.width", 200)
# -------------------- paths --------------------
DATA_DIR = r"D:\AgentDs\agent_ds_healthcare"
TRAIN_PATH = 
r"D:\AgentDs\agent_ds_healthcare\ed_cost_train.csv"
TEST_PATH  = 
r"D:\AgentDs\agent_ds_healthcare\ed_cost_test.csv"
PATIENTS_PATH = 
r"D:\AgentDs\agent_ds_healthcare\patients.csv"
ADM_TRAIN_PATH = 

r"D:\AgentDs\agent_ds_healthcare\admissions_train.csv"
ADM_TEST_PATH  = 
r"D:\AgentDs\agent_ds_healthcare\admissions_test.csv"
RECEIPTS_JOBLIB_PATH = 
r"D:\AgentDs\agent_ds_healthcare\receipts_parsed.joblib"
RECEIPTS_PDF_DIR = 
r"D:\AgentDs\agent_ds_healthcare\receipts_pdf"  # last 
resort
SUBMISSION_OUT_PATH = 
r"D:\AgentDs\agent_ds_healthcare\submission.csv"
ID_COL = "patient_id"
TARGET_COL = "ed_cost_next3y_usd"
print("Plan: (Code16->Code17) shallow+strong-reg 
CatBoost (depth 4/5, rsm=0.8, subsample=0.8) + stability 
pruning + multi-seed; model space: log1p(y) and delta-
log(y/baseline), choose best by CV; train full; save 
submission.csv")
# -------------------- helpers --------------------
def ensure_paths_exist(paths):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("Missing required file(s):\n" + 
"\n".join(missing))
def canon(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())
def find_col(df, candidates):
    cmap = {canon(c): c for c in df.columns}
    for cand in candidates:
        cc = canon(cand)
        if cc in cmap:
            return cmap[cc]
        for k, v in cmap.items():
            if cc and cc in k and len(cc) >= 4:
                return v
    return None

def make_unique_columns(cols):
    seen = {}
    out = []
    for c in list(cols):
        if c not in seen:
            seen[c] = 0
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}__dup{seen[c]}")
    return out
def collapse_duplicate_columns(df: pd.DataFrame) -> 
pd.DataFrame:
    idx = pd.Index(df.columns)
    if not idx.duplicated().any():
        return df
    df2 = df.copy()
    dup_names = idx[idx.duplicated()].unique()
    for name in dup_names:
        cols = df2.loc[:, name]
        if isinstance(cols, pd.DataFrame):
            combined = cols.bfill(axis=1).iloc[:, 0]
        else:
            combined = cols
        df2 = df2.drop(columns=[name])
        df2[name] = combined
    if pd.Index(df2.columns).duplicated().any():
        df2.columns = make_unique_columns(df2.columns)
    return df2
def code_to_str(x):
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    s = str(x).strip()
    if s == "":
        return None

    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]
    return s.upper()
def code_category(code_str):
    if code_str is None:
        return "missing"
    s = str(code_str).strip().upper()
    if s == "" or s == "NAN":
        return "missing"
    if s[0].isalpha():
        if s.startswith("G"): return "hcpcs_G"
        if s.startswith("J"): return "hcpcs_J"
        if s.startswith("A"): return "hcpcs_A"
        return "alpha_" + s[0]
    try:
        num = int(float(s))
    except Exception:
        return "other"
    if 99200 <= num <= 99499: return "cpt_em"
    if 70000 <= num <= 79999: return "cpt_radiology"
    if 80000 <= num <= 89999: return "cpt_labpath"
    if 90000 <= num <= 99999: return "cpt_medicine"
    if 10000 <= num <= 69999: return "cpt_surgery"
    return "cpt_other"
def safe_div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), 
where=np.abs(b) > 0)
def load_joblib_df(path):
    import joblib
    obj = joblib.load(path)
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if isinstance(obj, dict):
        for k in ["df", "data", "items", "lines", "line_items", 
"lineitems"]:

            if k in obj and isinstance(obj[k], pd.DataFrame):
                return obj[k].copy()
        # try dict-of-rows
        try:
            return pd.DataFrame(obj)
        except Exception:
            rows = []
            for pid, val in obj.items():
                if isinstance(val, pd.DataFrame):
                    d = val.copy()
                    if ID_COL not in d.columns:
                        d[ID_COL] = pid
                    rows.append(d)
                elif isinstance(val, dict):
                    inner = None
                    for k2 in ["df", "data", "items", "lines", 
"line_items", "lineitems"]:
                        if k2 in val and isinstance(val[k2], 
pd.DataFrame):
                            inner = val[k2].copy()
                            break
                    if inner is None and "items" in val and 
isinstance(val["items"], list):
                        inner = pd.DataFrame(val["items"])
                    if inner is not None:
                        if ID_COL not in inner.columns:
                            inner[ID_COL] = pid
                        rows.append(inner)
            if rows:
                return pd.concat(rows, ignore_index=True)
    if isinstance(obj, list):
        return pd.DataFrame(obj)
    raise ValueError("Unsupported receipts_parsed.joblib 
content type: " + str(type(obj)))
def parse_receipts_pdfs_to_long(pdf_dir, 
patient_ids=None):
    # last resort only
    try:
        import pdfplumber

    except Exception as e:
        raise ImportError("pdfplumber not installed; cannot 
parse PDFs. Please ensure receipts_parsed.joblib exists.") 
from e
    rows = []
    pdf_path = Path(pdf_dir)
    pdf_files = list(pdf_path.glob("receipt_*.pdf"))
    if patient_ids is not None:
        want = set(patient_ids)
        filtered = []
        for p in pdf_files:
            m = re.findall(r"receipt_(\d+)\.pdf", p.name)
            if m and int(m[0]) in want:
                filtered.append(p)
        pdf_files = filtered
    for p in pdf_files:
        m = re.findall(r"receipt_(\d+)\.pdf", p.name)
        if not m:
            continue
        pid = int(m[0])
        with pdfplumber.open(str(p)) as pdf:
            text = "\n".join([(page.extract_text() or "") for page 
in pdf.pages])
        # naive pattern: CODE ... QTY UNIT LINE
        for line in text.splitlines():
            line = line.strip()
            mm = re.match(r"^([A-Z0-9]{4,6})\s+.*?\s+(\d+)\s+
([\d,]+\.\d{2})\s+([\d,]+\.\d{2})$", line)
            if mm:
                code, qty, unit, lt = mm.group(1), 
float(mm.group(2)), float(mm.group(3).replace(",", "")), 
float(mm.group(4).replace(",", ""))
                rows.append({ID_COL: pid, "code": code, "qty": 
qty, "unit_price": unit, "line_total": lt})
    if not rows:
        raise ValueError("No receipts parsed from PDFs; 
please check pdf directory or receipts_parsed.joblib.")
    return pd.DataFrame(rows)

# -------------------- load base tables --------------------
ensure_paths_exist([TRAIN_PATH, TEST_PATH, 
PATIENTS_PATH, ADM_TRAIN_PATH, ADM_TEST_PATH])
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
patients = pd.read_csv(PATIENTS_PATH)
adm_train = pd.read_csv(ADM_TRAIN_PATH)
adm_test  = pd.read_csv(ADM_TEST_PATH)
assert ID_COL in train.columns and ID_COL in test.columns, 
"patient_id missing"
assert TARGET_COL in train.columns, "target missing in 
train"
for df in (train, test, patients, adm_train, adm_test):
    if ID_COL in df.columns:
        df[ID_COL] = pd.to_numeric(df[ID_COL], 
errors="coerce").astype("Int64")
train = train[train[ID_COL].notna()].copy()
test  = test[test[ID_COL].notna()].copy()
train[ID_COL] = train[ID_COL].astype(int)
test[ID_COL]  = test[ID_COL].astype(int)
patients = patients[patients[ID_COL].notna()].copy()
patients[ID_COL] = patients[ID_COL].astype(int)
adm_train = adm_train[adm_train[ID_COL].notna()].copy()
adm_test  = adm_test[adm_test[ID_COL].notna()].copy()
adm_train[ID_COL] = adm_train[ID_COL].astype(int)
adm_test[ID_COL]  = adm_test[ID_COL].astype(int)
# normalize likely ED cols
pc = find_col(train, ["prior_ed_cost_5y_usd", 
"prior_ed_cost_5y"])
pv = find_col(train, ["prior_ed_visits_5y", 
"prior_ed_visits_5y_count"])
if pc and pc != "prior_ed_cost_5y_usd":

    train = train.rename(columns={pc: 
"prior_ed_cost_5y_usd"})
    test  = test.rename(columns={pc: 
"prior_ed_cost_5y_usd"}) if pc in test.columns else test
if pv and pv != "prior_ed_visits_5y":
    train = train.rename(columns={pv: "prior_ed_visits_5y"})
    test  = test.rename(columns={pv: "prior_ed_visits_5y"}) if 
pv in test.columns else test
# -------------------- patients merge (minimal) -------------
-------
if "zip3" in patients.columns:
    patients["zip3"] = patients["zip3"].astype(str).str.zfill(3)
train = train.merge(patients, on=ID_COL, how="left")
test  = test.merge(patients, on=ID_COL, how="left")
# -------------------- admissions aggregates (minimal, low-
noise) --------------------
for df in (adm_train, adm_test):
    if "readmit_30d" in df.columns:
        df.drop(columns=["readmit_30d"], inplace=True, 
errors="ignore")
def mode_or_unknown(s):
    s = s.dropna()
    if len(s) == 0:
        return "Unknown"
    m = s.mode()
    return str(m.iloc[0]) if len(m) else "Unknown"
def build_adm_features(adm):
    adm = adm.copy()
    # drop admission_id-like columns if present (avoid 
unique ids)
    for c in list(adm.columns):
        if "admission_id" in canon(c):
            adm.drop(columns=[c], inplace=True, 
errors="ignore")
    num_cols = []

    for c in ["los_days", "acuity_emergent", "charlson_band", 
"ed_visits_6m", "discharge_weekday"]:
        if c in adm.columns:
            adm[c] = pd.to_numeric(adm[c], errors="coerce")
            num_cols.append(c)
    if "primary_dx" in adm.columns:
        adm["primary_dx"] = adm["primary_dx"].astype(str)
    g = adm.groupby(ID_COL)
    feat = pd.DataFrame({ID_COL: g.size().index.astype(int)})
    feat["adm_n"] = g.size().astype(float).values
    for c in ["los_days", "acuity_emergent", "charlson_band", 
"ed_visits_6m"]:
        if c in adm.columns:
            agg = g[c].agg(["mean", "max", 
"sum"]).rename(columns=
{"mean":f"adm_{c}_mean","max":f"adm_{c}_max","sum":f"ad
m_{c}_sum"})
            feat = feat.merge(agg.reset_index(), on=ID_COL, 
how="left")
    if "discharge_weekday" in adm.columns:
        feat = 
feat.merge(g["discharge_weekday"].apply(mode_or_unkno
wn).rename("adm_discharge_weekday_mode").reset_index(
), on=ID_COL, how="left")
    if "primary_dx" in adm.columns:
        feat = 
feat.merge(g["primary_dx"].apply(mode_or_unknown).rena
me("adm_primary_dx_mode").reset_index(), on=ID_COL, 
how="left")
        feat = 
feat.merge(g["primary_dx"].nunique().rename("adm_primar
y_dx_nunique").reset_index(), on=ID_COL, how="left")
    # log transforms (stable)

    for c in [c for c in feat.columns if c.endswith("_sum") or c 
in ["adm_n", "adm_primary_dx_nunique"]]:
        if c != ID_COL:
            feat[c + "_log1p"] = 
np.log1p(pd.to_numeric(feat[c], 
errors="coerce").fillna(0).clip(lower=0))
    return feat
adm_feat_train = build_adm_features(adm_train)
adm_feat_test  = build_adm_features(adm_test)
train = train.merge(adm_feat_train, on=ID_COL, 
how="left")
test  = test.merge(adm_feat_test, on=ID_COL, how="left")
# -------------------- receipts features (Code16-style: 
counts + flags + small top-codes; avoid huge sparse) -----
---------------
SPECIAL_CODES = ["31500", "36556", "92950", "36620", 
"99291", "99292", "70450", "74177", "84484", "G0378", 
"99285"]
ED_LEVEL_MAP = {"99281": 1, "99282": 2, "99283": 3, 
"99284": 4, "99285": 5}
if os.path.exists(RECEIPTS_JOBLIB_PATH):
    receipts = load_joblib_df(RECEIPTS_JOBLIB_PATH)
else:
    print("WARNING: receipts_parsed.joblib missing. Parsing 
PDFs (slow).")
    pid_all = pd.concat([train[ID_COL], 
test[ID_COL]]).astype(int).tolist()
    receipts = 
parse_receipts_pdfs_to_long(RECEIPTS_PDF_DIR, 
patient_ids=pid_all)
receipts = collapse_duplicate_columns(receipts)
if ID_COL not in receipts.columns:
    pid_col = find_col(receipts, [ID_COL, "patientid", "pid"])
    if pid_col is None:
        raise ValueError("patient_id not found in receipts")

    receipts = receipts.rename(columns={pid_col: ID_COL})
code_col = find_col(receipts, ["code", "cpt", "hcpcs", 
"procedurecode", "proccode", "billingcode"])
if code_col is None:
    raise ValueError("Receipts must contain a code column 
(found cols: {})".format(list(receipts.columns)[:30]))
if code_col != "code":
    receipts = receipts.rename(columns={code_col: "code"})
receipts[ID_COL] = pd.to_numeric(receipts[ID_COL], 
errors="coerce").astype("Int64")
receipts = receipts[receipts[ID_COL].notna()].copy()
receipts[ID_COL] = receipts[ID_COL].astype(int)
# sanitize code
receipts["code_str"] = receipts["code"].apply(code_to_str)
receipts["cat"] = receipts["code_str"].apply(code_category)
print(f"Receipts loaded shape: {receipts.shape} | cols: 
{list(receipts.columns)}")
# meta columns (patient-level totals repeated per row)
meta_map = {}
for src, dst in [
    ("sum_line_total", "rcpt_sum_items"),
    ("sum_unit_x_qty", "rcpt_sum_unit_x_qty"),
    ("n_line_items", "rcpt_n_line_items_meta"),
    ("receipt_total", "rcpt_pdf_total"),
    ("parse_ok", "rcpt_parse_ok"),
    ("zip3_receipt", "rcpt_zip3_receipt"),
    ("insurance_receipt", "rcpt_insurance_receipt"),
    ("zip3_receipt_raw", "rcpt_zip3_receipt_raw"),
]:
    if src in receipts.columns:
        meta_map[src] = dst
meta_cols = list(meta_map.keys())
if meta_cols:
    meta = receipts.groupby(ID_COL)

[meta_cols].first().rename(columns=meta_map).reset_index
()
else:
    meta = pd.DataFrame({ID_COL: 
receipts[ID_COL].unique()})
# ensure zip3 formatting
for zc in ["rcpt_zip3_receipt", "rcpt_zip3_receipt_raw"]:
    if zc in meta.columns:
        meta[zc] = meta[zc].astype(str).str.extract(r"(\d+)", 
expand=False)
        meta[zc] = meta[zc].fillna("").apply(lambda x: x.zfill(3) 
if x != "" else "Unknown")
# core receipt counts
g = receipts.groupby(ID_COL)
rcpt_core = pd.DataFrame({ID_COL: 
g.size().index.astype(int)})
rcpt_core["rcpt_n_lines"] = g.size().astype(float).values
rcpt_core["rcpt_n_unique_codes"] = 
g["code_str"].nunique().astype(float).values
rcpt_core = rcpt_core.merge(meta, on=ID_COL, 
how="left")
# categories (small)
cat_ct = pd.crosstab(receipts[ID_COL], receipts["cat"])
cat_ct = cat_ct.reindex(index=rcpt_core[ID_COL].values, 
fill_value=0)
cat_ct.columns = [f"rcpt_cat_cnt_{c}" for c in 
cat_ct.columns]
cat_ct = cat_ct.reset_index().rename(columns={"index": 
ID_COL})
rcpt_core = rcpt_core.merge(cat_ct, on=ID_COL, 
how="left")
# high-acuity flags + counts
for c in SPECIAL_CODES:
    receipts[f"_is_{c}"] = (receipts["code_str"] == 
c).astype(int)
flag_cols = [f"_is_{c}" for c in SPECIAL_CODES]

flag_agg = receipts.groupby(ID_COL)
[flag_cols].sum().reset_index()
for c in SPECIAL_CODES:
    flag_agg[f"rcpt_cnt_{c}"] = 
flag_agg[f"_is_{c}"].astype(float)
    flag_agg[f"rcpt_has_{c}"] = (flag_agg[f"_is_{c}"] > 
0).astype(int)
flag_agg["rcpt_high_acuity_cnt_sum"] = 
flag_agg[flag_cols].sum(axis=1).astype(float)
flag_agg["rcpt_high_acuity_n_present"] = 
(flag_agg[flag_cols] > 0).sum(axis=1).astype(float)
flag_agg = flag_agg.drop(columns=flag_cols)
rcpt_core = rcpt_core.merge(flag_agg, on=ID_COL, 
how="left")
# ED level max + counts
ed_rows = 
receipts[receipts["code_str"].isin(list(ED_LEVEL_MAP.keys()))
].copy()
if len(ed_rows):
    ed_rows["ed_level"] = 
ed_rows["code_str"].map(ED_LEVEL_MAP).astype(float)
    ed_max = ed_rows.groupby(ID_COL)
["ed_level"].max().rename("rcpt_ed_level_max").reset_index(
)
    ed_cnt = pd.crosstab(ed_rows[ID_COL], 
ed_rows["code_str"]).reset_index()
    for code, lvl in ED_LEVEL_MAP.items():
        if code in ed_cnt.columns:
            ed_cnt.rename(columns={code: f"rcpt_cnt_{code}"}, 
inplace=True)
        else:
            ed_cnt[f"rcpt_cnt_{code}"] = 0
    # ensure all
    for code in ED_LEVEL_MAP.keys():
        if f"rcpt_cnt_{code}" not in ed_cnt.columns:
            ed_cnt[f"rcpt_cnt_{code}"] = 0
    rcpt_core = rcpt_core.merge(ed_max, on=ID_COL, 
how="left").merge(ed_cnt[[ID_COL] + [f"rcpt_cnt_{c}" for c 
in ED_LEVEL_MAP.keys()]], on=ID_COL, how="left")

else:
    rcpt_core["rcpt_ed_level_max"] = 0.0
    for code in ED_LEVEL_MAP.keys():
        rcpt_core[f"rcpt_cnt_{code}"] = 0.0
# top codes by patient coverage (small K) -> counts
code_pat = receipts.dropna(subset=
["code_str"]).groupby("code_str")
[ID_COL].nunique().sort_values(ascending=False)
top_codes = [c for c in code_pat.head(25).index.tolist() if c 
not in set(SPECIAL_CODES) and c not in 
set(ED_LEVEL_MAP.keys())]
dfk = receipts[receipts["code_str"].isin(top_codes)]
if len(dfk) and len(top_codes):
    top_ct = pd.crosstab(dfk[ID_COL], 
dfk["code_str"]).reindex(columns=top_codes, fill_value=0)
    top_ct.columns = [f"rcpt_topcode_cnt_{c}" for c in 
top_ct.columns]
    top_ct = top_ct.reset_index()
    rcpt_core = rcpt_core.merge(top_ct, on=ID_COL, 
how="left")
# cleanup fill
rcpt_core = collapse_duplicate_columns(rcpt_core)
rcpt_core_cols = [c for c in rcpt_core.columns if c != 
ID_COL]
for c in rcpt_core_cols:
    if rcpt_core[c].dtype == "object":
        rcpt_core[c] = 
rcpt_core[c].fillna("Unknown").astype(str)
    else:
        rcpt_core[c] = pd.to_numeric(rcpt_core[c], 
errors="coerce")
# merge receipts features
train = train.merge(rcpt_core, on=ID_COL, how="left")
test  = test.merge(rcpt_core, on=ID_COL, how="left")
# receipts invariant sanity (sum_line_total vs prior cost)
if "rcpt_sum_items" in train.columns and 

"prior_ed_cost_5y_usd" in train.columns:
    diff = (pd.to_numeric(train["rcpt_sum_items"], 
errors="coerce") - 
pd.to_numeric(train["prior_ed_cost_5y_usd"], 
errors="coerce")).abs()
    print(f"Receipt rcpt_sum_items vs prior_ed_cost_5y_usd 
match_rate(train): {(diff < 1e-2).mean():.4f}")
del receipts, rcpt_core, adm_feat_train, adm_feat_test
gc.collect()
# -------------------- base engineered features (minimal, 
stable) --------------------
def add_base_feats(df):
    df = df.copy()
    df["prior_ed_cost_5y_usd"] = 
pd.to_numeric(df.get("prior_ed_cost_5y_usd", 0), 
errors="coerce").fillna(0).clip(lower=0)
    df["prior_ed_visits_5y"] = 
pd.to_numeric(df.get("prior_ed_visits_5y", 0), 
errors="coerce").fillna(0).clip(lower=0)
    df["prior_cost_per_visit"] = df["prior_ed_cost_5y_usd"] / 
np.maximum(df["prior_ed_visits_5y"], 1)
    df["prior_cost_log1p"] = 
np.log1p(df["prior_ed_cost_5y_usd"])
    df["prior_visits_log1p"] = 
np.log1p(df["prior_ed_visits_5y"])
    df["prior_cost_per_visit_log1p"] = 
np.log1p(df["prior_cost_per_visit"])
    df["baseline_next3y"] = df["prior_ed_cost_5y_usd"] * (3.0 
/ 5.0)
    df["baseline_next3y_log1p"] = 
np.log1p(df["baseline_next3y"])
    # interactions that are clinically meaningful + low 
dimensional
    if "rcpt_high_acuity_n_present" in df.columns:
        df["baseline_x_high_acuity_n"] = df["baseline_next3y"] 

* pd.to_numeric(df["rcpt_high_acuity_n_present"], 
errors="coerce").fillna(0)
    if "rcpt_has_99291" in df.columns and "rcpt_has_99292" 
in df.columns:
        df["rcpt_has_critical_care"] = 
((df["rcpt_has_99291"].fillna(0).astype(int) + 
df["rcpt_has_99292"].fillna(0).astype(int)) > 0).astype(int)
        df["baseline_x_critical_care"] = df["baseline_next3y"] * 
df["rcpt_has_critical_care"].astype(float)
    # zip_region (noise-reduced geography)
    if "zip3" in df.columns:
        z = df["zip3"].astype(str).str.extract(r"(\d{3})", 
expand=False)
        df["zip3"] = z.fillna("Unknown")
        df["zip_region"] = 
df["zip3"].str[0].where(df["zip3"].str.len() > 0, "U")
        df["zip_region"] = df["zip_region"].fillna("U")
    return df
train = add_base_feats(train)
test  = add_base_feats(test)
# prior cost bins for stratification
all_cost = pd.concat([train["prior_ed_cost_5y_usd"], 
test["prior_ed_cost_5y_usd"]], axis=0)
edges = np.unique(np.quantile(all_cost.replace([np.inf, -
np.inf], np.nan).dropna().values, np.linspace(0,1,11)))
if len(edges) < 3:
    edges = np.array([all_cost.min() - 1e-9, all_cost.max() + 
1e-9])
train["prior_cost_bin"] = 
pd.cut(train["prior_ed_cost_5y_usd"], bins=edges, 
include_lowest=True).astype(str)
test["prior_cost_bin"]  = 
pd.cut(test["prior_ed_cost_5y_usd"], bins=edges, 
include_lowest=True).astype(str)
# -------------------- dataset build --------------------
y = pd.to_numeric(train[TARGET_COL], 

errors="coerce").astype(float).values
assert np.isfinite(y).all(), "Non-finite targets found"
X_train = train.drop(columns=[TARGET_COL]).copy()
X_test  = test.copy()
X_train = collapse_duplicate_columns(X_train)
X_test  = collapse_duplicate_columns(X_test)
# align cols
X_train, X_test = X_train.align(X_test, join="left", axis=1)
# remove obvious useless/leaky unique path-like columns 
if present
drop_like = []
for c in X_train.columns:
    cc = canon(c)
    if "pdfpath" in cc or (("path" in cc) and (X_train[c].dtype 
== "object")):
        drop_like.append(c)
if drop_like:
    X_train.drop(columns=drop_like, inplace=True, 
errors="ignore")
    X_test.drop(columns=drop_like, inplace=True, 
errors="ignore")
# cat cols
cat_cols = [c for c in X_train.columns if (X_train[c].dtype == 
"object" or str(X_train[c].dtype).startswith("category"))]
# ensure key bins are categorical
for c in ["prior_cost_bin", "primary_chronic", "sex", 
"insurance", "zip3", "zip_region", 
"adm_discharge_weekday_mode", 
"adm_primary_dx_mode",
          "rcpt_insurance_receipt", "rcpt_zip3_receipt", 
"rcpt_zip3_receipt_raw"]:
    if c in X_train.columns and c not in cat_cols:
        cat_cols.append(c)
def fix_types(df):

    df = df.copy()
    for c in df.columns:
        if c in cat_cols:
            df[c] = df[c].astype(object).where(~pd.isna(df[c]), 
"Unknown").astype(str)
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # fill numerics with 0 (counts & robust)
    num_cols = [c for c in df.columns if c not in cat_cols and 
c != ID_COL]
    df[num_cols] = df[num_cols].fillna(0.0)
    # keep patient_id as int
    df[ID_COL] = pd.to_numeric(df[ID_COL], 
errors="coerce").astype("Int64")
    return df
X_train = fix_types(X_train)
X_test  = fix_types(X_test)
# model matrices
Xtr_full = X_train.drop(columns=[ID_COL])
Xte_full = X_test.drop(columns=[ID_COL])
# -------------------- CatBoost setup --------------------
try:
    from catboost import CatBoostRegressor
except Exception as e:
    raise ImportError("CatBoost is required for this Code17. 
Please pip install catboost . Original error: " + str(e))
def cb_params(seed, depth=4, lr=0.035, iters=8000):
    # Code16-style: strong regularization + column/row 
subsampling + ordered boosting (reduces overfit on small 
data)
    return dict(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=iters,
        learning_rate=lr,
        depth=depth,

        l2_leaf_reg=18.0,
        min_data_in_leaf=60,
        random_strength=0.6,
        bootstrap_type="Bernoulli",
        subsample=0.80,
        rsm=0.80,
        boosting_type="Ordered",
        random_seed=int(seed),
        od_type="Iter",
        od_wait=300,
        allow_writing_files=False,
        verbose=False,
        task_type="CPU",
        thread_count=-1,
    )
# -------------------- stability pruning (Code16 key) --------
------------
# Start from "small but not tiny" candidate set, then prune 
by stability across seeds.
mandatory = []
for c in [
    "primary_chronic",
    "prior_ed_cost_5y_usd", "prior_ed_visits_5y",
    "baseline_next3y", "baseline_next3y_log1p",
    "prior_cost_per_visit", "prior_cost_log1p", 
"prior_visits_log1p",
    "rcpt_n_lines", "rcpt_n_unique_codes",
    "rcpt_sum_items",
    "rcpt_high_acuity_n_present", 
"rcpt_high_acuity_cnt_sum",
    "rcpt_ed_level_max",
    "baseline_x_high_acuity_n",
    "zip_region",
    "age", "sex", "insurance",
]:
    if c in Xtr_full.columns:
        mandatory.append(c)
# ensure uniqueness

mandatory = list(dict.fromkeys(mandatory))
# Candidate feature list: everything except patient_id; but 
we will cap later.
all_feats = Xtr_full.columns.tolist()
# For pruning importance, use log1p(y) (most stable per 
your logs)
y_log = np.log1p(y)
prune_seeds = [42, 202, 777]
TOPK = 60
freq = {f: 0 for f in all_feats}
rank_sum = {f: 0.0 for f in all_feats}
cat_idx_all = [Xtr_full.columns.get_loc(c) for c in cat_cols if c 
in Xtr_full.columns]
# Fit a few small-ish models to get stable importance 
quickly
for s in prune_seeds:
    m = CatBoostRegressor(**cb_params(seed=s, depth=4, 
lr=0.04, iters=2500))
    m.fit(Xtr_full, y_log, cat_features=cat_idx_all)
    imp = m.get_feature_importance()
    order = np.argsort(-imp)
    top = order[:min(TOPK, len(order))]
    for r, idx in enumerate(top):
        f = Xtr_full.columns[idx]
        freq[f] += 1
        rank_sum[f] += (r + 1)
# selection rule: keep mandatory + features that appear in 
topK at least 2/3 seeds
selected = set(mandatory)
for f, c in freq.items():
    if c >= 2:
        selected.add(f)
# cap to MAX_FEATS by best average rank (keep 

mandatory no matter what)
MAX_FEATS = 70  # Code16 "pruning": keep compact
if len(selected) > MAX_FEATS:
    # sort by (is_mandatory desc, freq desc, avg_rank asc)
    def score(f):
        is_m = 1 if f in mandatory else 0
        fr = freq.get(f, 0)
        avg_rank = rank_sum.get(f, 1e9) / max(fr, 1)
        return (-is_m, -fr, avg_rank)
    selected_sorted = sorted(list(selected), key=score)
    selected = set(selected_sorted[:MAX_FEATS])
selected_feats = [f for f in all_feats if f in selected]
print(f"Pruning summary: total_feats={len(all_feats)} | 
mandatory={len(mandatory)} | selected=
{len(selected_feats)}")
print("Selected head:", selected_feats[:25])
Xtr = Xtr_full[selected_feats].copy()
Xte = Xte_full[selected_feats].copy()
cat_idx = [Xtr.columns.get_loc(c) for c in cat_cols if c in 
Xtr.columns]
# -------------------- CV (robust-ish, no target bins; Code16 
speed) --------------------
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_absolute_error
if "primary_chronic" in train.columns:
    chronic = 
train["primary_chronic"].astype(str).reset_index(drop=True)
else:
    chronic = pd.Series(["all"] * len(train))
prior_bin = 
train["prior_cost_bin"].astype(str).reset_index(drop=True) if 
"prior_cost_bin" in train.columns else pd.Series(["bin"] * 
len(train))
strat = (chronic.astype(str) + "_" + 
prior_bin.astype(str)).values

n_splits = 5
vals, counts = np.unique(strat, return_counts=True)
if len(counts) == 0 or counts.min() < n_splits:
    print(f"WARNING: strat too granular (min count {0 if 
len(counts)==0 else int(counts.min())}). Falling back to 
stratify by primary_chronic.")
    strat = chronic.astype(str).values
    vals, counts = np.unique(strat, return_counts=True)
    if len(counts) == 0 or counts.min() < n_splits:
        print("WARNING: chronic strat still sparse. Using plain 
KFold.")
        splitter = KFold(n_splits=n_splits, shuffle=True, 
random_state=SEED)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, 
shuffle=True, random_state=SEED)
else:
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, 
random_state=SEED)
baseline_train = pd.to_numeric(train.get("baseline_next3y", 
np.nan), 
errors="coerce").fillna(np.median(y)).astype(float).values
def fit_predict_fold(X_tr, y_tr, X_va, y_va, seed, mode, 
base_va):
    """
    mode:
      - "log": train on log1p(y)
      - "deltalog": train on log1p(y) - log1p(baseline)
    """
    p = cb_params(seed=seed, depth=4, lr=0.035, 
iters=8000)
    m = CatBoostRegressor(**p)
    if mode == "log":
        yt = np.log1p(y_tr)
        yv = np.log1p(y_va)
        m.fit(X_tr, yt, eval_set=(X_va, yv), cat_features=cat_idx, 
use_best_model=True)

        pred = np.expm1(m.predict(X_va))
        return np.clip(pred, 0, None), m.get_best_iteration() or 
p["iterations"]
    else:
        bt = 
np.log1p(np.clip(pd.to_numeric(train.loc[X_tr.index, 
"baseline_next3y"], errors="coerce").fillna(0).values, 0, 
None))
        bv = np.log1p(np.clip(base_va, 0, None))
        yt = np.log1p(y_tr) - bt
        yv = np.log1p(y_va) - bv
        m.fit(X_tr, yt, eval_set=(X_va, yv), cat_features=cat_idx, 
use_best_model=True)
        pred_delta = m.predict(X_va)
        pred = np.expm1(pred_delta + bv)
        return np.clip(pred, 0, None), m.get_best_iteration() or 
p["iterations"]
oof_log = np.zeros(len(train), dtype=float)
oof_dlt = np.zeros(len(train), dtype=float)
iters_log, iters_dlt = [], []
fold_mae_log, fold_mae_dlt, fold_mae_ens = [], [], []
for fold, (tr_idx, va_idx) in enumerate(splitter.split(Xtr, strat), 
1):
    X_tr, X_va = Xtr.iloc[tr_idx], Xtr.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    base_va = baseline_train[va_idx]
    pred_log, it_log = fit_predict_fold(X_tr, y_tr, X_va, y_va, 
seed=SEED, mode="log", base_va=base_va)
    pred_dlt, it_dlt = fit_predict_fold(X_tr, y_tr, X_va, y_va, 
seed=SEED, mode="deltalog", base_va=base_va)
    oof_log[va_idx] = pred_log
    oof_dlt[va_idx] = pred_dlt
    iters_log.append(int(it_log))
    iters_dlt.append(int(it_dlt))
    mae_log = mean_absolute_error(y_va, pred_log)

    mae_dlt = mean_absolute_error(y_va, pred_dlt)
    pred_ens = 0.5 * pred_log + 0.5 * pred_dlt
    mae_ens = mean_absolute_error(y_va, pred_ens)
    fold_mae_log.append(mae_log); 
fold_mae_dlt.append(mae_dlt); 
fold_mae_ens.append(mae_ens)
    print(f"Fold {fold} MAE | log={mae_log:.3f} | deltalog=
{mae_dlt:.3f} | ens(0.5/0.5)={mae_ens:.3f} | best_it log=
{it_log} dlt={it_dlt}")
cv_log = mean_absolute_error(y, oof_log)
cv_dlt = mean_absolute_error(y, oof_dlt)
cv_ens = mean_absolute_error(y, 0.5*oof_log + 0.5*oof_dlt)
cv_base = mean_absolute_error(y, np.clip(baseline_train, 0, 
None))
print(f"Overall CV MAE | log={cv_log:.3f} | deltalog=
{cv_dlt:.3f} | ens={cv_ens:.3f} | baseline={cv_base:.3f}")
# choose approach (simple, avoids overfitting via weight 
search)
scores = {"log": cv_log, "deltalog": cv_dlt, "ens": cv_ens}
best_mode = min(scores, key=scores.get)
print("Chosen approach for final training:", best_mode, " | 
CV MAE:", round(scores[best_mode], 4))
# -------------------- train final multi-seed ensemble -------
-------------
final_seeds = [42, 202, 777, 1001, 2024]
best_it_log = int(np.median(iters_log)) if len(iters_log) else 
1200
best_it_dlt = int(np.median(iters_dlt)) if len(iters_dlt) else 
1200
def train_full_and_predict(mode, seeds):
    preds = []
    for s in seeds:
        if mode == "log":
            p = cb_params(seed=s, depth=4, lr=0.035, 

iters=max(1200, int(best_it_log * 1.15)))
            p.pop("od_type", None); p.pop("od_wait", None)
            m = CatBoostRegressor(**p)
            m.fit(Xtr, np.log1p(y), cat_features=cat_idx)
            pred = np.expm1(m.predict(Xte))
            preds.append(pred.astype(float))
        elif mode == "deltalog":
            base_tr = 
np.log1p(np.clip(pd.to_numeric(train["baseline_next3y"], 
errors="coerce").fillna(0).values, 0, None))
            base_te = 
np.log1p(np.clip(pd.to_numeric(test["baseline_next3y"], 
errors="coerce").fillna(0).values, 0, None))
            target = np.log1p(y) - base_tr
            p = cb_params(seed=s, depth=4, lr=0.035, 
iters=max(1200, int(best_it_dlt * 1.15)))
            p.pop("od_type", None); p.pop("od_wait", None)
            m = CatBoostRegressor(**p)
            m.fit(Xtr, target, cat_features=cat_idx)
            pred = np.expm1(m.predict(Xte) + base_te)
            preds.append(pred.astype(float))
        else:
            raise ValueError("unknown mode")
    pred_mean = np.mean(np.column_stack(preds), axis=1)
    return np.clip(pred_mean, 0, None)
if best_mode == "log":
    pred_test = train_full_and_predict("log", final_seeds)
    models_used = 
"CatBoost(depth=4,rsm=0.8,subsample=0.8,l2=18,min_leaf
=60) on log1p(y), multi-seed avg"
elif best_mode == "deltalog":
    pred_test = train_full_and_predict("deltalog", 
final_seeds)
    models_used = 
"CatBoost(depth=4,rsm=0.8,subsample=0.8,l2=18,min_leaf
=60) on log1p(y)-log1p(baseline), multi-seed avg"
else:
    pred_log = train_full_and_predict("log", final_seeds)
    pred_dlt = train_full_and_predict("deltalog", final_seeds)

    pred_test = 0.5 * pred_log + 0.5 * pred_dlt
    pred_test = np.clip(pred_test, 0, None)
    models_used = "Ensemble: 0.5*log + 0.5*deltalog, each 
multi-seed avg (depth=4,rsm=0.8,subsample=0.8)"
# -------------------- submission --------------------
sub = pd.DataFrame({ID_COL: 
test[ID_COL].astype(int).values, TARGET_COL: 
pred_test.astype(float)})
sub = sub[[ID_COL, TARGET_COL]]
# sanity checks
print("Final model(s):", models_used)
print("Selected feature count:", len(selected_feats))
print("Submission sanity:")
print(" shape:", sub.shape)
print(" columns:", list(sub.columns))
print(" pred NaNs:", 
int(np.isnan(sub[TARGET_COL].values).sum()))
print(" pred min/median/max:", 
float(np.min(sub[TARGET_COL])), 
float(np.median(sub[TARGET_COL])), 
float(np.max(sub[TARGET_COL])))
# save
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
sub.to_csv(SUBMISSION_OUT_PATH, index=False)
print("Saved submission to:", SUBMISSION_OUT_PATH)
print("Paste back CV MAE + these logs + new leaderboard 
MAE for next iteration.")
```

</details>


<details>
<summary>RAW — Diagnostic prompt & response (Iter7 vs Iter8 vs Code17)</summary>

```text
为什么iteration 7能达到mae 451, iteration 8 时间长反而
mae 480+ code 17时间最长也只有mae 456?
Thought for 3m 12s
你这三个结果（Iter7 ≈451、Iter8 ≈480+、Code17 ≈456）最常见不是“训练久不久”的问
题，而是下面三类问题叠加导致的：

1) 你拿 Iter7 的“451”在跟谁比？（OOF vs LB 的错觉）
Iter7 的 451 大概率是 OOF（CV）MAE，而不是 LB。
你 Iter7/8 都做了“在同一份 OOF 上继续调参”的操作：网格搜权重、baseline 混合系数
lam、全局 median shift、再加 group shift / severity shift。
这会产生典型现象：
OOF 会被“二次拟合”压得很低（因为你在用整份训练集的 OOF 残差来调 shift / 权重，
本质是在拟合噪声）
LB 不一定跟着变好，甚至变差（因为这些“微调”对 test 的分布偏移不稳）
所以你看到：
Iter7：OOF 很好看（451），LB 可能并没对应下降（你之前真实 LB 459.x 就是这个典型
差距）
Iter8：你加了更多旋钮（tail-pen、log 模型、line stats、severity shift），二次拟合风
险更大，LB 更容易翻车
一句话：Iter7 的 451 很可能是“乐观的 OOF”，不是可泛化的提升。
2) Iter8 变成 480+：最可疑的是 receipts 的“行金额列”用错了（直接把
特征搞坏）
你 Iter8 新增的 line distribution 特征（n_line_items / line_mean / line_median /
line_max / top1/top2 share ）非常敏感：前提是你拿到的 amt  必须是每一行的
line_total。
但你 receipts_parsed.joblib 在你日志里长这样（类似）：
有 sum_line_total （病人级总额，重复在每行）
有 receipt_total （病人级总额）
未必有真正的 line_total
而你 Iter8 的列选择逻辑是：


python
total_col candidates = ["line_total","line_total_usd","total","amount","line_cost

这里有两个坑会把特征“毒化”：
1. 如果真实的行金额列叫 sum_line_total ，你根本选不到它 → receipts 特征直接缺失
（pdf_df=None 或大量 0/median），模型会退化到“只有
priors+patients+admissions”，结果可能直接跳到 470~490。
2. 更糟：如果你误选到的是“病人级总额列”并把它当 line_total（比如某列名碰巧命中
total/amount ，但其实是重复的总额），那会导致：
receipt_total = g["amt"].sum()  变成 总额×行数
line_mean/median/max  变成 “总额”本身或被放大
em_total/proc_total/...  变成 把总额按“出现某类code的行数”重复累计
pct_cost_*  全都扭曲
这种特征对模型是“强噪声 + 错方向信号”，OOF/LB 都会崩，480+ 很常见。
这条是 Iter8 最优先怀疑点：你新增的 line stats 让“金额列一旦错了”影响成倍放大。
3) Iter8 里“LOGRMSE 模型”也容易把 MAE 搞差（指标不对齐 + 系统性
偏低）
你 Iter8 的 log 模型：
训练目标：y_log = log1p(y)
loss：RMSE
early stopping eval：RMSE（在 log 空间）
然后你把预测 expm1  回原空间，再拿来做 MAE/ensemble。
问题是：
在 log 空间优化 RMSE ≠ 在原空间优化 MAE
log 模型通常会对高值更“保守”（偏低），MAE 对高尾很敏感 → Top20 会拉爆
你虽然加了 tail penalty，但 tail penalty 是在 OOF 上调权重/shift，依然可能对 test 不
稳
所以 Iter8 即便训练更久、模型更多，也可能在你关心的 MAE 上更差。
Copy code
你不需要猜：最简单的自检是——在 build_pdf_features_from_lineitems 里打印
code_col 、total_col ，并抽 3 个 patient 看 amt  的分布（是不是每行都一样/是不是
等于 sum_line_total）。

4) Code17 为什么更久但只有 456：它“更稳但更容易欠拟合”
Code17 的核心风格是：
极强正则：l2_leaf_reg=18 , min_data_in_leaf=60 , depth=4
subsample=0.8, rsm=0.8
先做 stability pruning（要多训几次拿稳定重要性）→ 时间长
最后用固定迭代数训练多 seed → 稳，但很可能欠拟合
所以你看到：
OOF 不会像 Iter7 那样“被二次拟合压很低”
但因为正则太强，模型容量不足 → 卡在 456 左右很正常
你现在最该做的 3 个动作（能最快解释“为啥会这样”）
A) 先把 Iter8 的 receipts 金额列彻底确认（99% 的时候问题在这）
在 build_pdf_features_from_lineitems  里加一次性打印/断言：
打印选中的 code_col 、total_col
随机抽一个 patient：
看该 patient 的 amt  是否每行都一样
看 g["amt"].sum()  是否≈ prior_ed_cost_5y_usd （如果变成约 prior_cost *
n_lines ，你就中毒了）
只要这里确认错列，Iter8 的 480+ 就解释通了。
B) 别再用“全量 OOF 调权重/shift”来对比 Iter7/8
你 Iter7 的 451 很可能是 OOF 被你自己“二次拟合”压出来的。
要公平比较：先把这些东西关掉，只比“模型本体”：
固定权重（比如简单平均）
shift_mult=0
不做 chronic/severity residual shift
lam 固定一个很小值或直接 0
这一步通常会让 Iter7 的 OOF 回到更真实的水平（接近 455~465），你就不会被“451幻觉”
误导。
C) Code17 想突破 456：把正则稍微放松一点点（别动太多）

只改两处就够明显：
l2_leaf_reg: 18 → 10~12
min_data_in_leaf: 60 → 35~45
depth 可以试 4→5 （如果你 GPU/CPU 还能扛住）
这通常会把欠拟合拉回来一点，同时仍保持稳定性。
如果你愿意，我也可以直接帮你把 Iter8 receipts 金额列选择改成“更鲁棒”的：
优先找真正的行金额列（例如 line_total  / extended_price ），找不到就完全跳过 line
distribution 特征，只用你已经验证稳定的低维 bucket/share/flag（这样 Iter8 至少不会炸
到 480+）。
根据这个3个结果我总结并合成了一个code:
```

</details>


<details>
<summary>RAW — ITERATION 9 (user code) + logs + LB score 450.5367 + analysis</summary>

```text
# === ITERATION 9: The "Iter7-Refined" (Hierarchical Shift 
Optimization) ===
# Back to Iteration 7 codebase exactly (proven best 
features).
# IMPROVEMENT: Replace manual "chronic shift" with an 
automated, stratified shift optimization.
# Rationale: Iter 8 failed due to feature noise; Code 17 
underfitted. Iter 7 was the sweet spot.
#            We squeeze the last juice by optimally shifting 
predictions per chronic group.
import os, re, sys, gc, math, warnings, random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from scipy.optimize import minimize
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
# -----------------------------
# Paths
# -----------------------------
DATA_DIR = r"D:\AgentDs\agent_ds_healthcare"
TRAIN_PATH = 
r"D:\AgentDs\agent_ds_healthcare\ed_cost_train.csv"
TEST_PATH  = 
r"D:\AgentDs\agent_ds_healthcare\ed_cost_test.csv"
PATIENTS_PATH = 
r"D:\AgentDs\agent_ds_healthcare\patients.csv"
ADM_TRAIN_PATH = 
r"D:\AgentDs\agent_ds_healthcare\admissions_train.csv"
ADM_TEST_PATH  = 
r"D:\AgentDs\agent_ds_healthcare\admissions_test.csv"
RECEIPTS_JOBLIB_PATH = 
r"D:\AgentDs\agent_ds_healthcare\receipts_parsed.joblib"
OUT_SUB_PATH = 
r"D:\AgentDs\agent_ds_healthcare\submission.csv"
TARGET = "ed_cost_next3y_usd"
print("="*90)
print("ITERATION 9 | Back to Iter7 (Winner) + Automated 
Stratified Shift Optimization")
print("Logic: Iter7 (MAE~451) was best. We refine it by 
optimizing shifts per chronic group.")
print("="*90)
# -----------------------------
# Deps
# -----------------------------
def _pip_install(pkg: str):
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", 
"install", "-q", pkg])

try:
    from joblib import load as joblib_load
except Exception:
    _pip_install("joblib")
    from joblib import load as joblib_load
try:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import mean_absolute_error
except Exception:
    _pip_install("scikit-learn")
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import mean_absolute_error
try:
    from catboost import CatBoostRegressor
except Exception:
    _pip_install("catboost")
    from catboost import CatBoostRegressor
# -----------------------------
# Config (Iter7 defaults)
# -----------------------------
class CFG:
    N_FOLDS = 7
    N_SEEDS = 5
    ITERS = 3000
    ES_ROUNDS = 120
    LR = 0.03
    RSM = 0.80
    
    # Ensemble search
    W_STEP = 0.05
    LAM_GRID = [0.00, 0.03, 0.05, 0.08, 0.10, 0.15]
    # We will remove global SHIFT_GRID from ensemble 
search and move it to post-processing optimization
    

    # Penalties (Iter7)
    STD_PEN = 0.20
    LAM_PEN = 8.0
# -----------------------------
# Utilities
# -----------------------------
def norm_code(x):
    if x is None or (isinstance(x, float) and np.isnan(x)): 
return None
    s = str(x).strip().upper()
    if s == "" or s.lower() == "nan": return None
    if re.fullmatch(r"\d+\.0+", s): s = s.split(".")[0]
    s = re.sub(r"\s+", "", s)
    return s
def standardize_zip3(z):
    if z is None or (isinstance(z, float) and np.isnan(z)): 
return None
    s = re.sub(r"\D", "", str(z).strip())
    if not s: return None
    return s.zfill(3)
def qdict(x, qs=(0,0.01,0.05,0.1,0.5,0.9,0.95,0.99,1.0)):
    x = np.asarray(x, dtype=float)
    return {q: float(np.quantile(x, q)) for q in qs}
# -----------------------------
# Data Loading & Feats (Iter7 Exact)
# -----------------------------
def load_admissions_features(adm_train_path, 
adm_test_path):
    dfs = []
    for path in [adm_train_path, adm_test_path]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "readmit_30d" in df.columns: df = 
df.drop(columns=["readmit_30d"])
            dfs.append(df)
    if not dfs: return None

    adm = pd.concat(dfs, ignore_index=True)
    adm["patient_id"] = pd.to_numeric(adm["patient_id"], 
errors="coerce")
    adm["charlson_band"] = 
pd.to_numeric(adm["charlson_band"], errors="coerce")
    adm["acuity_emergent"] = 
pd.to_numeric(adm["acuity_emergent"], errors="coerce")
    out = adm.groupby("patient_id").agg(
        charlson_max=("charlson_band","max"),
        charlson_mean=("charlson_band","mean"),
        pct_emergent=("acuity_emergent","mean"),
    ).reset_index()
    for c in 
["charlson_max","charlson_mean","pct_emergent"]:
        out[c] = pd.to_numeric(out[c], 
errors="coerce").fillna(0.0)
    return out
def build_pdf_features_from_lineitems(li):
    li = li.copy()
    # Locate cols
    code_col = next((c for c in 
["code","cpt","cpt_code","hcpcs","proc_code"] if c in 
li.columns), None)
    total_col = next((c for c in 
["line_total","line_total_usd","total","amount"] if c in 
li.columns), None)
    if not code_col or not total_col or "patient_id" not in 
li.columns: return None
    li["patient_id"] = pd.to_numeric(li["patient_id"], 
errors="coerce").astype("Int64")
    li = li.dropna(subset=["patient_id"]).copy()
    li["patient_id"] = li["patient_id"].astype(int)
    li["code"] = li[code_col].map(norm_code)
    li = li.dropna(subset=["code"]).copy()
    li["amt"] = pd.to_numeric(li[total_col], 
errors="coerce").fillna(0.0).clip(lower=0.0)
    # Base totals

    total = li.groupby("patient_id")
["amt"].sum().rename("receipt_total")
    li = li.join(total, on="patient_id")
    denom = li["receipt_total"].replace(0.0, np.nan)
    # Buckets (Iter7 definition)
    is_em = 
li["code"].isin(["99281","99282","99283","99284","99285"])
    em_map = 
{"99281":1,"99282":2,"99283":3,"99284":4,"99285":5}
    em_level = li["code"].map(em_map).fillna(0).astype(float)
    
    is_crit = li["code"].isin(["99291","99292"])
    is_obs = li["code"].str.startswith("G037", na=False)
    is_high = 
li["code"].isin(["31500","36556","32551","36620","92950"])
    
    code_num = 
pd.to_numeric(li["code"].where(li["code"].str.fullmatch(r"\d
+"), None), errors="coerce")
    is_lab = code_num.between(80000, 89999)
    is_imaging = code_num.between(70000, 79999)
    is_proc_general = code_num.between(10000, 69999)
    is_proc_any = is_high | (is_proc_general & (~is_high) & 
(~is_em) & (~is_crit))
    # Features
    share = (li["amt"] / denom).fillna(0.0)
    cost_hhi = (share * 
share).groupby(li["patient_id"]).sum().rename("cost_hhi")
    n_unique_codes = li.groupby("patient_id")
["code"].nunique().rename("n_unique_codes")
    
    n_em_codes = 
is_em.astype(int).groupby(li["patient_id"]).sum().rename("n_
em_codes")
    max_em_level = 
em_level.groupby(li["patient_id"]).max().rename("max_em_l
evel")
    

    # Flags
    has_critical_care = 
is_crit.astype(int).groupby(li["patient_id"]).max().rename("h
as_critical_care")
    has_high_acuity = 
is_high.astype(int).groupby(li["patient_id"]).max().rename("
has_high_acuity")
    has_imaging = 
is_imaging.astype(int).groupby(li["patient_id"]).max().renam
e("has_imaging")
    has_obs = 
is_obs.astype(int).groupby(li["patient_id"]).max().rename("h
as_observation")
    
    # Specifics
    def has_c(c, n): return 
(li["code"].eq(c).astype(int).groupby(li["patient_id"]).max()).r
ename(n)
    has_99285 = has_c("99285","has_99285")
    
    # Bucket totals
    em_total = li.loc[is_em].groupby("patient_id")
["amt"].sum().rename("em_total")
    crit_total = li.loc[is_crit].groupby("patient_id")
["amt"].sum().rename("crit_total")
    proc_total = li.loc[is_proc_any].groupby("patient_id")
["amt"].sum().rename("proc_total")
    high_total = li.loc[is_high].groupby("patient_id")
["amt"].sum().rename("high_total")
    out = pd.concat([
        n_unique_codes, cost_hhi, n_em_codes, max_em_level,
        has_critical_care, has_high_acuity, has_imaging, 
has_obs, has_99285,
        total
    ], axis=1).reset_index()
    for s in [em_total, crit_total, proc_total, high_total]:
        out = out.merge(s.reset_index(), on="patient_id", 
how="left")

    
    for c in 
["em_total","crit_total","proc_total","high_total","receipt_tot
al"]:
        if c not in out.columns: out[c] = 0.0
        out[c] = pd.to_numeric(out[c], 
errors="coerce").fillna(0.0)
        
    d2 = out["receipt_total"].replace(0.0, np.nan)
    out["pct_cost_critical"] = (out["crit_total"] / d2).fillna(0.0)
    out["pct_cost_procedure"] = (out["proc_total"] / 
d2).fillna(0.0)
    out["pct_cost_high_acuity"] = (out["high_total"] / 
d2).fillna(0.0)
    out["cost_per_em"] = np.where(out["n_em_codes"]>0, 
out["receipt_total"]/out["n_em_codes"].clip(lower=1), 
out["receipt_total"])
    
    # Composite
    out["n_high_acuity_total"] = (
        out["has_high_acuity"].fillna(0) + 
out["has_critical_care"].fillna(0)
    ).astype(int) # Simplified proxy from Iter7 logic
    return out
def load_receipts(path):
    if not os.path.exists(path): return None
    try:
        d = joblib_load(path)
        if isinstance(d, dict):
            for k in ["lineitems_df","lineitems","items_df"]:
                if k in d and isinstance(d[k], pd.DataFrame): 
return build_pdf_features_from_lineitems(d[k])
        if isinstance(d, pd.DataFrame): return 
build_pdf_features_from_lineitems(d)
    except: pass
    return None
def build_features(ed, pts, adm, pdf):

    f = ed.copy()
    # Chronic
    cmap = {"PNEUMONIA":0, "DIABETESCOMP":1, "HF":2}
    f["primary_chronic"] = f["primary_chronic"].astype(str)
    f["chronic_encoded"] = 
f["primary_chronic"].str.upper().map(cmap).fillna(-1).astype(
float)
    
    # Prior
    f["prior_ed_visits_5y"] = 
pd.to_numeric(f["prior_ed_visits_5y"], 
errors="coerce").fillna(0.0)
    f["prior_ed_cost_5y_usd"] = 
pd.to_numeric(f["prior_ed_cost_5y_usd"], 
errors="coerce").fillna(0.0)
    
    f["sqrt_prior_cost"] = 
np.sqrt(f["prior_ed_cost_5y_usd"].clip(lower=0))
    f["log_prior_cost"] = 
np.log1p(f["prior_ed_cost_5y_usd"].clip(lower=0))
    f["log_visits"] = 
np.log1p(f["prior_ed_visits_5y"].clip(lower=0))
    f["cost_per_visit"] = f["prior_ed_cost_5y_usd"] / 
f["prior_ed_visits_5y"].clip(lower=1)
    f["prior_cost_cap20k"] = 
f["prior_ed_cost_5y_usd"].clip(upper=20000)
    f["log_prior_cost_cap20k"] = 
np.log1p(f["prior_cost_cap20k"])
    
    # Baseline
    f["baseline_next3y"] = f["prior_ed_cost_5y_usd"] * 0.6
    
    # Patients
    p = pts.copy()
    p["patient_id"] = pd.to_numeric(p["patient_id"], 
errors="coerce").astype(int)
    p["age"] = pd.to_numeric(p["age"], 
errors="coerce").fillna(p["age"].median())
    p["sex_encoded"] = 
(p["sex"].astype(str).str.upper()=="M").astype(int)

    ins_map = {"private":2,"public":1,"self_pay":0,"selfpay":0}
    p["insurance_encoded"] = 
p["insurance"].astype(str).str.lower().map(ins_map).fillna(-1)
    z = p["zip3"].apply(standardize_zip3).astype(str).str[0]
    p["zip_region"] = pd.to_numeric(z, 
errors="coerce").fillna(-1)
    
    f = 
f.merge(p[["patient_id","age","sex_encoded","insurance_en
coded","zip_region"]], on="patient_id", how="left")
    f["ins_x_chronic"] = f["insurance_encoded"] * 
f["chronic_encoded"]
    
    # Adm
    if adm is not None:
        f = f.merge(adm, on="patient_id", how="left")
        for c in 
["charlson_max","charlson_mean","pct_emergent"]:
            if c in f.columns: f[c] = f[c].fillna(f[c].median())
            
    # PDF
    if pdf is not None:
        f = f.merge(pdf, on="patient_id", how="left")
        for c in pdf.columns:
            if c!="patient_id" and 
pd.api.types.is_numeric_dtype(pdf[c]):
                f[c] = f[c].fillna(f[c].median())
                
    # Interactions
    if "pct_cost_critical" in f.columns:
        f["logprior_x_pctcritical"] = f["log_prior_cost"] * 
f["pct_cost_critical"]
    if "n_high_acuity_total" in f.columns:
        f["logprior_x_highacu"] = f["log_prior_cost"] * 
f["n_high_acuity_total"]
    if "n_unique_codes" in f.columns:
        f["cost_per_code"] = f["prior_ed_cost_5y_usd"] / 
f["n_unique_codes"].clip(lower=1)
    # Clean

    for c in f.columns:
        if c not in ["patient_id","primary_chronic",TARGET]:
            f[c] = pd.to_numeric(f[c], errors="coerce").fillna(0.0)
            
    return f
# -----------------------------
# Main Execution
# -----------------------------
print("\n[load] Reading data...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
pts_df = pd.read_csv(PATIENTS_PATH)
adm_df = load_admissions_features(ADM_TRAIN_PATH, 
ADM_TEST_PATH)
pdf_df = load_receipts(RECEIPTS_JOBLIB_PATH)
for d in [train_df, test_df, pts_df]: d["patient_id"] = 
pd.to_numeric(d["patient_id"], errors="coerce").astype(int)
train_feat = build_features(train_df, pts_df, adm_df, pdf_df)
test_feat = build_features(test_df, pts_df, adm_df, pdf_df)
# Feature Selection (Iter7 Pruned Set)
pruned_cols = [
    
"prior_ed_visits_5y","prior_ed_cost_5y_usd","sqrt_prior_cost
","log_prior_cost","log_prior_cost_cap20k","cost_per_visit","l
og_visits",
    "baseline_next3y",
    
"chronic_encoded","age","sex_encoded","insurance_encod
ed","zip_region","ins_x_chronic",
    "charlson_max","charlson_mean","pct_emergent",
    
"cost_per_em","cost_hhi","pct_cost_procedure","pct_cost_cr
itical","pct_cost_high_acuity",
    
"n_high_acuity_total","has_critical_care","has_99285","max_
em_level","n_unique_codes",

    
"logprior_x_pctcritical","logprior_x_highacu","cost_per_cod
e"
]
# Full numeric set for Model A
all_cols = [c for c in train_feat.columns if c not in 
["patient_id","primary_chronic",TARGET,"sex","insurance","zi
p3"] and pd.api.types.is_numeric_dtype(train_feat[c])]
# Remove constants
feat_full = [c for c in all_cols if train_feat[c].nunique() > 1]
feat_pruned = [c for c in pruned_cols if c in 
train_feat.columns and train_feat[c].nunique() > 1]
print(f"Features: Full={len(feat_full)}, Pruned=
{len(feat_pruned)}")
# -----------------------------
# Training (Iter7 Specs)
# -----------------------------
y = train_feat[TARGET].values.astype(float)
tmp = train_feat[["primary_chronic", TARGET]].copy()
tmp["bin"] = pd.qcut(tmp[TARGET], 5, labels=False)
tmp["strat"] = tmp["primary_chronic"].astype(str) + "_" + 
tmp["bin"].astype(str)
strat = LabelEncoder().fit_transform(tmp["strat"])
model_specs = {
    "A_RMSE_full_d5": dict(
        loss_function="RMSE", eval_metric="MAE", depth=5, 
l2_leaf_reg=5, min_data_in_leaf=28,
        learning_rate=CFG.LR, iterations=CFG.ITERS, 
rsm=CFG.RSM, random_strength=1.0
    ),
    "B_RMSE_pruned_d4": dict(
        loss_function="RMSE", eval_metric="MAE", depth=4, 
l2_leaf_reg=4, min_data_in_leaf=32,
        learning_rate=CFG.LR, iterations=CFG.ITERS, 
rsm=CFG.RSM, random_strength=2.0
    ),
    "C_MAE_pruned_d4": dict(

        loss_function="MAE", eval_metric="MAE", depth=4, 
l2_leaf_reg=7, min_data_in_leaf=36,
        learning_rate=CFG.LR, iterations=CFG.ITERS, 
rsm=CFG.RSM, random_strength=1.5
    )
}
cols_map = {"A_RMSE_full_d5": feat_full, 
"B_RMSE_pruned_d4": feat_pruned, "C_MAE_pruned_d4": 
feat_pruned}
oof_by_seed = {m: [] for m in model_specs}
test_by_seed = {m: [] for m in model_specs}
print("\n[Training] 5 Seeds x 7 Folds...")
for seed_idx in range(CFG.N_SEEDS):
    rs = SEED + seed_idx * 7
    kf = StratifiedKFold(CFG.N_FOLDS, shuffle=True, 
random_state=rs)
    
    s_oof = {m: np.zeros(len(y)) for m in model_specs}
    s_test = {m: np.zeros(len(test_feat)) for m in 
model_specs}
    
    for fold, (ti, vi) in enumerate(kf.split(train_feat, strat)):
        for m, p in model_specs.items():
            cols = cols_map[m]
            X_tr, y_tr = train_feat[cols].iloc[ti], y[ti]
            X_va, y_va = train_feat[cols].iloc[vi], y[vi]
            
            cb = CatBoostRegressor(**p, task_type="CPU", 
thread_count=-1, verbose=0, random_seed=rs+fold*99, 
early_stopping_rounds=CFG.ES_ROUNDS)
            cb.fit(X_tr, y_tr, eval_set=(X_va, y_va))
            
            s_oof[m][vi] = cb.predict(X_va)
            s_test[m] += cb.predict(test_feat[cols]) / 
CFG.N_FOLDS
            
    for m in model_specs:
        oof_by_seed[m].append(s_oof[m])

        test_by_seed[m].append(s_test[m])
        
    print(f"  Seed {seed_idx+1} MAE: " + " | ".join([f"{m}=
{mean_absolute_error(y, s_oof[m]):.2f}" for m in 
model_specs]))
# Averaging seeds
oof_avg = {m: np.mean(np.vstack(oof_by_seed[m]), axis=0) 
for m in model_specs}
test_avg = {m: np.mean(np.vstack(test_by_seed[m]), 
axis=0) for m in model_specs}
# -----------------------------
# Ensemble Selection (Iter7 Stable)
# -----------------------------
# We do stable search WITHOUT global shift first, then do 
refined shift later
baseline_tr = train_feat["baseline_next3y"].values
baseline_te = test_feat["baseline_next3y"].values
models = list(model_specs.keys())
best_ens = None
best_score = float("inf")
# Grid search for weights & lambda only (no shift yet)
print("\n[Ensemble] Searching weights...")
grid = np.arange(0.0, 1.01, CFG.W_STEP)
for wA in grid:
    for wB in grid:
        wC = 1.0 - wA - wB
        if wC < -1e-9: continue
        wC = max(0.0, wC)
        if abs(wA+wB+wC-1.0) > 1e-5: continue
        
        for lam in CFG.LAM_GRID:
            # Calc objective
            maes = []
            for s in range(CFG.N_SEEDS):
                p = wA*oof_by_seed[models[0]][s] + 
wB*oof_by_seed[models[1]][s] + 

wC*oof_by_seed[models[2]][s]
                p = (1-lam)*p + lam*baseline_tr
                maes.append(mean_absolute_error(y, p))
            
            mean_m = np.mean(maes)
            std_m = np.std(maes)
            obj = mean_m + CFG.STD_PEN*std_m + 
CFG.LAM_PEN*lam
            
            if obj < best_score:
                best_score = obj
                best_ens = (wA, wB, wC, lam)
wA, wB, wC, lam = best_ens
print(f"  Best Weights: {models[0]}={wA:.2f}, {models[1]}=
{wB:.2f}, {models[2]}={wC:.2f}")
print(f"  Best Lambda: {lam:.2f}")
# Construct raw ensemble
oof_raw = wA*oof_avg[models[0]] + 
wB*oof_avg[models[1]] + wC*oof_avg[models[2]]
oof_raw = (1-lam)*oof_raw + lam*baseline_tr
test_raw = wA*test_avg[models[0]] + 
wB*test_avg[models[1]] + wC*test_avg[models[2]]
test_raw = (1-lam)*test_raw + lam*baseline_te
print(f"  Ensemble MAE (Raw): {mean_absolute_error(y, 
oof_raw):.3f}")
# -----------------------------
# Post-Processing: Stratified Shift Optimization
# -----------------------------
# Instead of one global shift, we optimize 3 shifts for 
chronic conditions 0, 1, 2
# Objective: Minimize MAE on OOF
chronic_tr = 
train_feat["chronic_encoded"].values.astype(int) # -1,0,1,2
chronic_te = 
test_feat["chronic_encoded"].values.astype(int)

def shift_objective(shifts):
    # shifts is [s_neg1, s_0, s_1, s_2] or just [s_0, s_1, s_2]
    # We'll map -1 to 0 (default)
    s0, s1, s2 = shifts
    mapping = {-1:s0, 0:s0, 1:s1, 2:s2}
    
    # Vectorized application
    # Create shift vector
    shift_vec = np.zeros_like(oof_raw)
    for c_val, s_val in mapping.items():
        shift_vec[chronic_tr == c_val] = s_val
        
    return mean_absolute_error(y, oof_raw + shift_vec)
print("\n[Post-Process] Optimizing Stratified Shifts...")
# Initial guess: median residuals per group
resid = y - oof_raw
init_shifts = []
for g in [0, 1, 2]: # approximate groups
    r = resid[(chronic_tr == g) | (chronic_tr == -1)] if g==0 
else resid[chronic_tr == g]
    init_shifts.append(np.median(r))
print(f"  Initial residual medians: {init_shifts}")
res = minimize(shift_objective, x0=init_shifts, 
method='Nelder-Mead', tol=1e-4)
opt_shifts = res.x
best_mae = res.fun
print(f"  Optimized Shifts: Group0/Unk={opt_shifts[0]:.4f}, 
Group1={opt_shifts[1]:.4f}, Group2={opt_shifts[2]:.4f}")
print(f"  Final Optimized OOF MAE: {best_mae:.3f}")
# Apply to Test
test_final = test_raw.copy()
shift_map = {-1:opt_shifts[0], 0:opt_shifts[0], 1:opt_shifts[1], 
2:opt_shifts[2]}
for i in range(len(test_final)):
    c = chronic_te[i]

    test_final[i] += shift_map.get(c, 0.0)
# Clip
y_max = np.max(y)
test_final = np.clip(test_final, 0, y_max * 1.5)
# -----------------------------
# Submission
# -----------------------------
sub = pd.DataFrame({
    "patient_id": test_df["patient_id"].values,
    "ed_cost_next3y_usd": np.round(test_final, 2)
})
out_path = Path(OUT_SUB_PATH)
out_path.parent.mkdir(parents=True, exist_ok=True)
sub.to_csv(out_path, index=False)
print("\n[Sanity Check]")
print("Submission Stats:")
print(sub[TARGET].describe())
print(f"\nDone. Saved to {OUT_SUB_PATH}")
print("Paste back: (1) Final Optimized OOF MAE, (2) 
Optimized Shifts values.")
log:
======================================
======================================
==============
ITERATION 9 | Back to Iter7 (Winner) + Automated 
Stratified Shift Optimization
Logic: Iter7 (MAE~451) was best. We refine it by 
optimizing shifts per chronic group.
======================================
======================================
==============
[load] Reading data...
Features: Full=40, Pruned=30
[Training] 5 Seeds x 7 Folds...

  Seed 1 MAE: A_RMSE_full_d5=429.64 | 
B_RMSE_pruned_d4=432.67 | C_MAE_pruned_d4=436.64
  Seed 2 MAE: A_RMSE_full_d5=432.63 | 
B_RMSE_pruned_d4=432.56 | C_MAE_pruned_d4=437.97
  Seed 3 MAE: A_RMSE_full_d5=429.25 | 
B_RMSE_pruned_d4=431.26 | C_MAE_pruned_d4=439.05
  Seed 4 MAE: A_RMSE_full_d5=430.80 | 
B_RMSE_pruned_d4=431.66 | C_MAE_pruned_d4=437.34
  Seed 5 MAE: A_RMSE_full_d5=430.62 | 
B_RMSE_pruned_d4=429.18 | C_MAE_pruned_d4=443.23
[Ensemble] Searching weights...
  Best Weights: A_RMSE_full_d5=0.60, 
B_RMSE_pruned_d4=0.40, C_MAE_pruned_d4=0.00
  Best Lambda: 0.00
  Ensemble MAE (Raw): 427.638
[Post-Process] Optimizing Stratified Shifts...
  Initial residual medians: 
[np.float64(-22.904879716014875), 
np.float64(-11.996815943285583), 
np.float64(16.364638281932002)]
  Optimized Shifts: Group0/Unk=-23.0480, 
Group1=-11.9968, Group2=16.3646
  Final Optimized OOF MAE: 427.428
[Sanity Check]
Submission Stats:
count     2000.000000
mean      3915.387060
std       1754.078236
min       1126.070000
25%       2586.582500
50%       3548.650000
75%       4874.462500
max      10599.760000
Name: ed_cost_next3y_usd, dtype: float64
Done. Saved to 
D:\AgentDs\agent_ds_healthcare\submission.csv

Paste back: (1) Final Optimized OOF MAE, (2) Optimized 
Shifts values.
得到real mae:
🚀 Submitting predictions...
✅ Prediction submitted successfully!
📊 Score: 450.5367 (MAE)
✅ Validation passed
✅ Submission successful!
   📊 Score: 450.5367
   📏 Metric: MAE
   ✔️  Validation: Passed
🎯 Next steps:
   1. Try incorporating relevant information outside this 
table!
   2. Move on to Healthcare Challenge 3!
Thought for 4m 15s
你这个 Iter9 很成功：LB = 450.5367（比你之前 459.x 明显进了一截），而且从现象上也把
三条路的优缺点“吃透了”。
但你现在最关键的信号其实是：
OOF 427.4 vs LB 450.5 差了 23 → 这不是“模型训练得更好”，而是 CV 评价比真实 test
更乐观（分布漂移 + 你做的后处理在 OOF 上仍会偏乐观）。
下面我直接给你结论 + 下一步怎么把 450.5 往下压（尽量快、尽量 LB-oriented）。
1) 你这个“优化 shift”的 Nelder-Mead 其实没有学到新东西
你发现优化结果几乎等于初始值：
Group0/Unk ≈ -23.048
Group1 ≈ -11.9968
Group2 ≈ 16.3646

这是必然的：
对 MAE（L1）来说，组内加一个常数的最优解就是“组内残差的中位数”。
也就是说，你的 minimize 本质上就是在数值上找 median —— 不会比 “直接取 median
residual” 更强，只会更慢、还更容易给你一种“我优化了很多”的错觉。
✅ 所以 Iter10 里建议把这段改成：
直接算组内 median residual
再做 层级 shrink（收缩） 防止对 test 过拟合
这一步通常能让 LB 更稳（有时还能小幅降）。
2) 你 OOF 太低的根源：CV 不像 test（你现在需要“对 test 分布负责”的
训练）
你 Iter9 的 base 模型（A/B）在 OOF 都 ~430，ensemble 427，但 LB 450。
这非常像：
train/test covariate shift（比如 chronic/insurance/zip/receipt pattern 在 test 分布不
同）
你现在的 CV 是在 “train 内部很好”，但 “对 test 不友好”
✅ 最直接、成本最低、最 Kaggle 的修法：
加 adversarial validation 权重（importance weighting），让模型训练更像 test。
你之前做过 shift weighting，但那时特征/模型不稳定。现在你 Iter7-features 已经稳定了，
再加权重通常更有用。
3) 你现在的 ensemble search 太“硬”，可以加一个“轻量 stacking 校准”
提高泛化
你现在是网格搜 (wA,wB,wC,lam)，然后再做 per-chronic shift。
网格搜对小数据经常会选到 “对 OOF 最优但对 LB 不稳” 的权重。
✅ 更稳的做法（非常轻量）：
用 OOF 预测（已经是 out-of-fold）当作特征
用一个 强正则的线性/Huber/Quantile 回归 做 stacking

然后再接一个“收缩版 group shift”
这通常比手工网格搜更稳，LB 也更容易下降。
Iter10 给你一套“最小改动但很可能再降 LB”的三件套
下面每个都是“插拔式”的，不需要你重写整份代码。
A) 把 shift 改成“层级收缩 median shift”（替换 minimize）
把你 Post-Process 那段改成下面这个函数（直接复制）：
python
def compute_shrunk_group_shifts(y, pred_oof, group, groups=(0,1,2), unk_value=-1,
    """
    y, pred_oof: OOF真实与预测
    group: chronic_encoded (可能含 -1)
    k: 收缩强度，越大越保守（推荐 80~200）
    cap: shift 最大绝对值限制，防止 test 翻车
    """
    resid = y - pred_oof
    global_med = float(np.median(resid))
    shifts = {}
    # 把 unknown 并到 0 组（你现在逻辑就是这样）
    group2 = group.copy()
    group2[group2 == unk_value] = 0
    for g in groups:
        mask = (group2 == g)
        r = resid[mask]
        if len(r) == 0:
            raw = global_med
            n = 0
        else:
            raw = float(np.median(r))
            n = int(mask.sum())

然后你原来：
res = minimize(...)
opt_shifts = res.x ...
换成：
这一步的目标不是让 OOF 更低，而是让 LB 更稳、更可能下降。


        # 层级收缩：n小的组更靠近 global_med
        alpha = n / (n + k)
        shrunk = alpha * raw + (1 - alpha) * global_med
        # cap 防翻车
        shrunk = float(np.clip(shrunk, -cap, cap))
        shifts[g] = shrunk
    # 0组 shift 同时用于 unknown
    shifts[unk_value] = shifts[0]
    return shifts, global_med


python
shifts, global_med = compute_shrunk_group_shifts(
    y=y,
    pred_oof=oof_raw,
    group=chronic_tr,
    k=120,       # 先用 120
    cap=80.0     # 先 cap 80
)
oof_final = oof_raw + np.vectorize(lambda c: shifts.get(c, shifts[0]))(chronic_tr
test_final = test_raw + np.vectorize(lambda c: shifts.get(c, shifts[0]))(chronic_t
print("Shrunk shifts:", shifts, "| global_med:", global_med)
print("OOF MAE after shrunk shifts:", mean_absolute_error(y, oof_final))

B) 加 adversarial importance weights（让训练更贴近 test）
你可以用你现有的 pruned 特征（或 full）做一个 train-vs-test 分类器，然后把权重喂给
CatBoostRegressor。
最省事、你已经装了 CatBoost：用 CatBoostClassifier。
Copy code
Copy code
python
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
def build_importance_weights(train_feat, test_feat, cols, seed=42, clip=(0.5, 2.0
    X_tr = train_feat[cols].copy()
    X_te = test_feat[cols].copy()
    X = pd.concat([X_tr, X_te], axis=0, ignore_index=True)
    y = np.r_[np.zeros(len(X_tr)), np.ones(len(X_te))].astype(int)
    oof_p = np.zeros(len(X), dtype=float)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr_idx, va_idx in skf.split(X, y):
        clf = CatBoostClassifier(
            loss_function="Logloss",
            depth=4,
            learning_rate=0.05,
            iterations=2000,
            l2_leaf_reg=8.0,
            random_strength=1.0,
            rsm=0.85,
            verbose=0,
            allow_writing_files=False,
            random_seed=seed + 13,
        )
        clf.fit(X.iloc[tr_idx], y[tr_idx])
        oof_p[va_idx] = clf.predict_proba(X.iloc[va_idx])[:, 1]
    auc = roc_auc_score(y, oof_p)
    print(f"[adv] train-vs-test AUC={auc:.4f}")

用法（在训练 loop 前）：
然后你 CatBoostRegressor 的 fit 改成：
这一步经常能把 LB 再往下推一点（尤其是你这种 OOF 很低但 LB 偏高的情况）。
C) 用“轻量 stacking”替代网格权重（更稳的泛化）
你现在的网格搜给了 C=0，等于你实际上只在用 A+B。
可以直接用 OOF 预测做一个正则回归来学组合（比网格稳）。
最简单、很鲁棒：HuberRegressor（对 outlier 更稳）。


    p_tr = np.clip(oof_p[:len(X_tr)], 1e-4, 1-1e-4)
    w = p_tr / (1 - p_tr)                    # odds
    w = np.clip(w, clip[0], clip[1]) ** power  # soften + clip
    return w


python
# 用 pruned（更稳）做权重
w_imp = build_importance_weights(train_feat, test_feat, feat_pruned, seed=SEED, c
Copy code
python
cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), sample_weight=w_imp[ti])
Copy code
python
from sklearn.linear_model import HuberRegressor
def fit_stacker_huber(y, oof_preds_dict, test_preds_dict, extra_cols=None):
    # extra_cols 可以加 baseline_next3y / chronic_encoded 等
    Z_tr = [oof_preds_dict[k] for k in sorted(oof_preds_dict.keys())]
    Z_te = [test_preds_dict[k] for k in sorted(test_preds_dict.keys())]

你可以把 A/B/C 的 seed-avg 预测放进去，再加 baseline：
然后对 oof_stack/test_stack  再接上面 A) 的 shrunk group shifts，通常比你当前“网格权
重 + optimize shift”更稳。
你现在这份 Iter9 的结论（非常重要）
你已经证明了：Iter7-features + A/B 模型 是你目前最稳的核心。
你所谓的“优化 chronic shift”其实只是 median residual，真正能提升 LB 的空间在：
1. shift 要收缩/限幅（减少对 test 的误校准）
2. 训练要贴近 test（adversarial weights）
3. 组合方式要更泛化（stacking 而不是网格硬搜）
Copy code
    if extra_cols is not None:
        for col_tr, col_te in extra_cols:
            Z_tr.append(col_tr)
            Z_te.append(col_te)
    Z_tr = np.column_stack(Z_tr)
    Z_te = np.column_stack(Z_te)
    reg = HuberRegressor(alpha=1e-3, epsilon=1.35, max_iter=5000)
    reg.fit(Z_tr, y)
    oof = reg.predict(Z_tr)
    te  = reg.predict(Z_te)
    return oof, te, reg.coef_, reg.intercept_
Copy code
python
oof_stack, test_stack, coef, bias = fit_stacker_huber(
    y=y,
    oof_preds_dict={m: oof_avg[m] for m in models},
    test_preds_dict={m: test_avg[m] for m in models},
    extra_cols=[(baseline_tr, baseline_te)]
)
print("Stacker coef:", coef, "bias:", bias)
print("OOF MAE stack:", mean_absolute_error(y, oof_stack))
Copy code

如果你按我上面 Iter10 的顺序做（先 A，再 B，再 C），你大概率会看到两件事：
OOF 可能不会再像 427 这么“好看”
但 LB 更有机会从 450.53 再往下掉（而且波动更小）
你跑完 Iter10 后，把下面 6 个值贴我，我就能精确告诉你下一步该往哪拧（不需要你再长
篇解释）：
1. adv AUC（train-vs-test）
2. 是否用了 importance weights（clip/power）
3. 是否用了 stacker（coef 大概长什么样）
4. shrunk shifts（k/cap）
5. OOF MAE（raw / after shift）
6. LB MAE
给我一个full code 优化我可以直接past in jupyter cell to 
run
Thought for 3m 3s
python
```

</details>


<details>
<summary>RAW — ITERATION 10 code + TypeError + FIXED version (Pool weights)</summary>

```text
# === ITERATION 10: Iter7-Core + (1) Adversarial Importance Weights + (2) Robust 
# Paste this whole cell into Jupyter and run.
# Output: D:\AgentDs\agent_ds_healthcare\submission.csv
import os, re, sys, gc, warnings, random
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
# -----------------------------

# Seeds / Repro
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
# -----------------------------
# Paths
# -----------------------------
DATA_DIR = r"D:\AgentDs\agent_ds_healthcare"
TRAIN_PATH = r"D:\AgentDs\agent_ds_healthcare\ed_cost_train.csv"
TEST_PATH  = r"D:\AgentDs\agent_ds_healthcare\ed_cost_test.csv"
PATIENTS_PATH = r"D:\AgentDs\agent_ds_healthcare\patients.csv"
ADM_TRAIN_PATH = r"D:\AgentDs\agent_ds_healthcare\admissions_train.csv"
ADM_TEST_PATH  = r"D:\AgentDs\agent_ds_healthcare\admissions_test.csv"
RECEIPTS_JOBLIB_PATH = r"D:\AgentDs\agent_ds_healthcare\receipts_parsed.joblib"
OUT_SUB_PATH = r"D:\AgentDs\agent_ds_healthcare\submission.csv"
ID_COL = "patient_id"
TARGET = "ed_cost_next3y_usd"
print("="*100)
print("ITERATION 10 | Iter7-Core + AdvWeights + RobustStack + Shrunk Chronic Shif
print("Goal: improve LB stability and push below your 450.5367 by training closer 
print("="*100)
# -----------------------------
# Minimal deps (auto-install if missing)
# -----------------------------
def _pip_install(pkg: str):
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
try:
    from joblib import load as joblib_load
except Exception:
    _pip_install("joblib")
    from joblib import load as joblib_load

try:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import mean_absolute_error, roc_auc_score
    from sklearn.linear_model import HuberRegressor
except Exception:
    _pip_install("scikit-learn")
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import mean_absolute_error, roc_auc_score
    from sklearn.linear_model import HuberRegressor
try:
    from catboost import CatBoostRegressor, CatBoostClassifier
except Exception:
    _pip_install("catboost")
    from catboost import CatBoostRegressor, CatBoostClassifier
# -----------------------------
# Config
# -----------------------------
class CFG:
    # Base CV bagging
    N_FOLDS = 7
    N_SEEDS = 5
    # CatBoost base
    ITERS = 3000
    ES_ROUNDS = 120
    LR = 0.03
    RSM = 0.80
    # Importance weighting (adv validation)
    ADV_FOLDS = 5
    ADV_ITERS = 2000
    ADV_CLIP = (0.60, 1.80)   # clip odds
    ADV_POWER = 0.60          # soften
    # Stacker
    META_FOLDS = 7

    HUBER_ALPHA = 1e-3
    HUBER_EPS = 1.35
    # Shrunk group shift
    SHIFT_K = 120            # larger => more conservative (80~200)
    SHIFT_CAP = 80.0         # max abs shift
    PRED_CLIP_MULT = 1.5     # final max clip = y_max * mult
# -----------------------------
# Helpers
# -----------------------------
def must_exist(path: str, name: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {name}: {path}")
def log_shape(name: str, df: pd.DataFrame):
    mem = df.memory_usage(deep=True).sum() / (1024**2)
    print(f"[{name}] shape={df.shape} | cols={df.shape[1]} | mem={mem:.2f} MB")
def qdict(x, qs=(0,0.01,0.05,0.1,0.5,0.9,0.95,0.99,1.0)):
    x = np.asarray(x, dtype=float)
    return {q: float(np.quantile(x, q)) for q in qs}
def standardize_zip3(z):
    if z is None or (isinstance(z, float) and np.isnan(z)):
        return None
    s = re.sub(r"\D", "", str(z).strip())
    if not s:
        return None
    return s.zfill(3)
def norm_code(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().upper()
    if s == "" or s.lower() == "nan":
        return None
    if re.fullmatch(r"\d+\.0+", s):
        s = s.split(".")[0]
    s = re.sub(r"\s+", "", s)

    return s
# -----------------------------
# Admissions aggregates (robust, low-noise)
# -----------------------------
def load_admissions_features(adm_train_path: str, adm_test_path: str) -> Optional
    dfs = []
    for path in [adm_train_path, adm_test_path]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "readmit_30d" in df.columns:
                df = df.drop(columns=["readmit_30d"])
            dfs.append(df)
    if not dfs:
        return None
    adm = pd.concat(dfs, ignore_index=True)
    need = ["patient_id", "charlson_band", "acuity_emergent"]
    if not all(c in adm.columns for c in need):
        return None
    adm["patient_id"] = pd.to_numeric(adm["patient_id"], errors="coerce")
    adm["charlson_band"] = pd.to_numeric(adm["charlson_band"], errors="coerce")
    adm["acuity_emergent"] = pd.to_numeric(adm["acuity_emergent"], errors="coerce
    out = adm.groupby("patient_id").agg(
        charlson_max=("charlson_band", "max"),
        charlson_mean=("charlson_band", "mean"),
        pct_emergent=("acuity_emergent", "mean"),
    ).reset_index()
    for c in ["charlson_max", "charlson_mean", "pct_emergent"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out
# -----------------------------
# Receipts (Iter7-style low-dim) — robust joblib loader + safe amount detection
# -----------------------------

def _extract_lineitems_df(obj) -> Optional[pd.DataFrame]:
    # Common formats: dict with "lineitems_df" etc, direct df, list of dfs
    if isinstance(obj, dict):
        for k in ["lineitems_df", "lineitems", "items_df", "items", "line_items_d
            if k in obj and isinstance(obj[k], pd.DataFrame):
                return obj[k].copy()
        # Some pipelines store long df directly under a key
        for k, v in obj.items():
            if isinstance(v, pd.DataFrame) and ("patient_id" in v.columns):
                return v.copy()
        return None
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if isinstance(obj, (list, tuple)):
        for v in obj:
            if isinstance(v, pd.DataFrame) and ("patient_id" in v.columns):
                return v.copy()
    return None
def _choose_amount_column(li: pd.DataFrame) -> Tuple[Optional[str], str]:
    """
    Returns (amount_col, mode)
    mode:
      - "direct": use amount_col as per-line amount
      - "qtyxunit": compute from qty*unit_price
      - "uniform_total": distribute patient-total uniformly across lines
      - "none": cannot compute
    """
    cols = set(li.columns)
    # direct per-line candidates (ordered)
    direct_candidates = [
        "line_total", "line_total_usd", "extended_price", "item_total",
        "line_cost", "amount"
    ]
    # sometimes "total" exists but is ambiguous; only use if nothing else
    fallback_candidates = ["total"]
    for c in direct_candidates:
        if c in cols:

            # Heuristic: if within-patient nunique is ~1 for almost all, it might 
            # We'll still accept for "line_total" / "extended_price" etc (more li
            return c, "direct"
    # qty * unit_price
    qty_candidates = ["qty", "quantity"]
    unit_candidates = ["unit_price", "price", "unit_cost"]
    qty_col = next((c for c in qty_candidates if c in cols), None)
    unit_col = next((c for c in unit_candidates if c in cols), None)
    if qty_col and unit_col:
        return f"{qty_col}__x__{unit_col}", "qtyxunit"
    # uniform distribute from patient total if present
    total_candidates = ["sum_line_total", "receipt_total", "sum_items", "total_am
    total_col = next((c for c in total_candidates if c in cols), None)
    if total_col:
        return total_col, "uniform_total"
    # last resort: "total"
    for c in fallback_candidates:
        if c in cols:
            return c, "direct"
    return None, "none"
def build_receipt_features_iter7(li: pd.DataFrame) -> Optional[pd.DataFrame]:
    li = li.copy()
    if ID_COL not in li.columns:
        return None
    # code col
    code_col = None
    for c in ["code", "cpt", "cpt_code", "hcpcs", "proc_code"]:
        if c in li.columns:
            code_col = c
            break
    if code_col is None:
        return None
    amt_col, amt_mode = _choose_amount_column(li)

    if amt_col is None or amt_mode == "none":
        return None
    # patient_id + code normalize
    li[ID_COL] = pd.to_numeric(li[ID_COL], errors="coerce").astype("Int64")
    li = li.dropna(subset=[ID_COL]).copy()
    li[ID_COL] = li[ID_COL].astype(int)
    li["code_norm"] = li[code_col].map(norm_code)
    li = li.dropna(subset=["code_norm"]).copy()
    # compute amt
    if amt_mode == "direct":
        li["amt"] = pd.to_numeric(li[amt_col], errors="coerce").fillna(0.0).astyp
        li.loc[li["amt"] < 0, "amt"] = 0.0
    elif amt_mode == "qtyxunit":
        qty_col, unit_col = amt_col.split("__x__")
        qty = pd.to_numeric(li[qty_col], errors="coerce").fillna(0.0).astype(float
        unit = pd.to_numeric(li[unit_col], errors="coerce").fillna(0.0).astype(fl
        li["amt"] = (qty * unit).astype(float)
        li.loc[li["amt"] < 0, "amt"] = 0.0
    elif amt_mode == "uniform_total":
        # distribute patient total uniformly (keeps totals consistent; shares bec
        total_per_patient = pd.to_numeric(li[amt_col], errors="coerce").fillna(0.
        total_per_patient = total_per_patient.groupby(li[ID_COL]).transform("first
        n_lines = li.groupby(ID_COL).cumcount()  # just to force groupby size lat
        sizes = li.groupby(ID_COL)["code_norm"].transform("size").astype(float).c
        li["amt"] = (total_per_patient / sizes).astype(float)
        li.loc[li["amt"] < 0, "amt"] = 0.0
    # totals
    receipt_total = li.groupby(ID_COL)["amt"].sum().rename("receipt_total")
    li = li.join(receipt_total, on=ID_COL)
    denom = li["receipt_total"].replace(0.0, np.nan)
    # buckets (Iter7)
    is_em = li["code_norm"].isin(["99281","99282","99283","99284","99285"])
    em_map = {"99281":1,"99282":2,"99283":3,"99284":4,"99285":5}

    em_level = li["code_norm"].map(em_map).fillna(0).astype(float)
    is_crit = li["code_norm"].isin(["99291","99292"])
    is_obs  = li["code_norm"].str.startswith("G037", na=False)
    is_high = li["code_norm"].isin(["31500","36556","32551","36620","92950"])
    code_num = pd.to_numeric(li["code_norm"].where(li["code_norm"].str.fullmatch(
    is_imaging = code_num.between(70000, 79999)
    is_proc_general = code_num.between(10000, 69999)
    is_proc_any = is_high | (is_proc_general & (~is_high) & (~is_em) & (~is_crit)
    # HHI
    share = (li["amt"] / denom).fillna(0.0)
    cost_hhi = (share * share).groupby(li[ID_COL]).sum().rename("cost_hhi")
    # counts
    n_unique_codes = li.groupby(ID_COL)["code_norm"].nunique().rename("n_unique_c
    n_em_codes = is_em.astype(int).groupby(li[ID_COL]).sum().rename("n_em_codes")
    max_em_level = em_level.groupby(li[ID_COL]).max().rename("max_em_level")
    # flags
    has_critical_care = is_crit.astype(int).groupby(li[ID_COL]).max().rename("has_
    has_high_acuity   = is_high.astype(int).groupby(li[ID_COL]).max().rename("has_
    has_imaging       = is_imaging.astype(int).groupby(li[ID_COL]).max().rename("
    has_observation   = is_obs.astype(int).groupby(li[ID_COL]).max().rename("has_
    has_99285         = (li["code_norm"].eq("99285").astype(int).groupby(li[ID_CO
    # bucket totals (weighted by amt)
    em_total   = li.loc[is_em].groupby(ID_COL)["amt"].sum().rename("em_total")
    crit_total = li.loc[is_crit].groupby(ID_COL)["amt"].sum().rename("crit_total"
    proc_total = li.loc[is_proc_any].groupby(ID_COL)["amt"].sum().rename("proc_tot
    high_total = li.loc[is_high].groupby(ID_COL)["amt"].sum().rename("high_total"
    out = pd.concat([
        n_unique_codes, cost_hhi, n_em_codes, max_em_level,
        has_critical_care, has_high_acuity, has_imaging, has_observation, has_992
        receipt_total
    ], axis=1).reset_index()
    for s in [em_total, crit_total, proc_total, high_total]:

        out = out.merge(s.reset_index(), on=ID_COL, how="left")
    for c in ["em_total","crit_total","proc_total","high_total","receipt_total"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    d2 = out["receipt_total"].replace(0.0, np.nan)
    out["pct_cost_critical"] = (out["crit_total"] / d2).replace([np.inf, -np.inf]
    out["pct_cost_procedure"] = (out["proc_total"] / d2).replace([np.inf, -np.inf
    out["pct_cost_high_acuity"] = (out["high_total"] / d2).replace([np.inf, -np.i
    out["cost_per_em"] = np.where(
        out["n_em_codes"] > 0,
        out["receipt_total"] / out["n_em_codes"].clip(lower=1),
        out["receipt_total"]
    )
    # simple composite (Iter7-ish; stable)
    out["n_high_acuity_total"] = (out["has_high_acuity"].fillna(0).astype(int) + 
    # drop helpers
    out.drop(columns=["em_total","crit_total","proc_total","high_total","receipt_t
    # numeric fill
    for c in out.columns:
        if c == ID_COL:
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out
def load_receipts_features(joblib_path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(joblib_path):
        return None
    try:
        obj = joblib_load(joblib_path)
        li = _extract_lineitems_df(obj)
        if li is None or not isinstance(li, pd.DataFrame):
            return None
        # If it already looks aggregated (has cost_hhi etc), just return unique p

        if (ID_COL in li.columns) and ("cost_hhi" in li.columns) and ("n_unique_c
            df = li.copy()
            df[ID_COL] = pd.to_numeric(df[ID_COL], errors="coerce").astype("Int64
            df = df.dropna(subset=[ID_COL]).copy()
            df[ID_COL] = df[ID_COL].astype(int)
            df = df.drop_duplicates(ID_COL, keep="last")
            # ensure numeric
            for c in df.columns:
                if c == ID_COL: continue
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
            return df[[ID_COL] + [c for c in df.columns if c != ID_COL]]
        # Else: build Iter7 low-dim from lineitems
        feat = build_receipt_features_iter7(li)
        if feat is None:
            return None
        feat[ID_COL] = pd.to_numeric(feat[ID_COL], errors="coerce").astype("Int64
        feat = feat.dropna(subset=[ID_COL]).copy()
        feat[ID_COL] = feat[ID_COL].astype(int)
        feat = feat.drop_duplicates(ID_COL, keep="last")
        return feat
    except Exception as e:
        print(f"[warn] receipts joblib load failed: {type(e).__name__}: {e}")
        return None
# -----------------------------
# Feature engineering (Iter7-core)
# -----------------------------
def build_features(ed_df: pd.DataFrame,
                   patients_df: pd.DataFrame,
                   adm_df: Optional[pd.DataFrame],
                   rcpt_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    feat = ed_df.copy()
    # chronic encoding
    chronic_map = {"PNEUMONIA": 0, "DIABETESCOMP": 1, "HF": 2}
    feat["primary_chronic"] = feat["primary_chronic"].astype(str)
    feat["chronic_encoded"] = feat["primary_chronic"].str.upper().map(chronic_map

    # priors
    feat["prior_ed_visits_5y"] = pd.to_numeric(feat["prior_ed_visits_5y"], errors
    feat["prior_ed_cost_5y_usd"] = pd.to_numeric(feat["prior_ed_cost_5y_usd"], er
    feat["prior_cost_cap20k"] = feat["prior_ed_cost_5y_usd"].clip(upper=20000.0)
    feat["sqrt_prior_cost"] = np.sqrt(feat["prior_ed_cost_5y_usd"].clip(lower=0.0
    feat["log_prior_cost"] = np.log1p(feat["prior_ed_cost_5y_usd"].clip(lower=0.0
    feat["log_prior_cost_cap20k"] = np.log1p(feat["prior_cost_cap20k"].clip(lower
    feat["log_visits"] = np.log1p(feat["prior_ed_visits_5y"].clip(lower=0.0))
    feat["cost_per_visit"] = feat["prior_ed_cost_5y_usd"] / feat["prior_ed_visits_
    # baseline anchor
    feat["baseline_next3y"] = feat["prior_ed_cost_5y_usd"] * (3.0 / 5.0)
    # patients
    p = patients_df.copy()
    p[ID_COL] = pd.to_numeric(p[ID_COL], errors="coerce").astype(int)
    p["age"] = pd.to_numeric(p["age"], errors="coerce").fillna(p["age"].median())
    p["sex_encoded"] = (p["sex"].astype(str).str.upper() == "M").astype(int)
    ins = p["insurance"].astype(str).str.lower()
    ins_map = {"private": 2, "public": 1, "self_pay": 0, "selfpay": 0}
    p["insurance_encoded"] = ins.map(ins_map).fillna(-1).astype(float)
    z3 = p["zip3"].apply(standardize_zip3).astype("string")
    zr = z3.fillna("000").str.replace(r"\D", "", regex=True).str.zfill(3).str[0]
    p["zip_region"] = pd.to_numeric(zr, errors="coerce").fillna(-1).astype(float)
    feat = feat.merge(p[[ID_COL, "age", "sex_encoded", "insurance_encoded", "zip_
    feat["ins_x_chronic"] = feat["insurance_encoded"].fillna(-1) * feat["chronic_
    # admissions
    if adm_df is not None:
        feat = feat.merge(adm_df, on=ID_COL, how="left")
        for c in ["charlson_max", "charlson_mean", "pct_emergent"]:
            if c in feat.columns:
                feat[c] = pd.to_numeric(feat[c], errors="coerce").fillna(0.0)
    # receipts
    if rcpt_df is not None:

        feat = feat.merge(rcpt_df, on=ID_COL, how="left")
        for c in rcpt_df.columns:
            if c == ID_COL:
                continue
            feat[c] = pd.to_numeric(feat[c], errors="coerce").fillna(0.0)
    # light interactions
    if "pct_cost_critical" in feat.columns:
        feat["logprior_x_pctcritical"] = feat["log_prior_cost"] * feat["pct_cost_
    else:
        feat["logprior_x_pctcritical"] = 0.0
    if "n_high_acuity_total" in feat.columns:
        feat["logprior_x_highacu"] = feat["log_prior_cost"] * feat["n_high_acuity_
    else:
        feat["logprior_x_highacu"] = 0.0
    if "n_unique_codes" in feat.columns:
        feat["cost_per_code"] = feat["prior_ed_cost_5y_usd"] / feat["n_unique_cod
    else:
        feat["cost_per_code"] = feat["prior_ed_cost_5y_usd"]
    # numeric safety
    feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    for c in feat.columns:
        if c in [ID_COL, "primary_chronic", TARGET]:
            continue
        feat[c] = pd.to_numeric(feat[c], errors="coerce").fillna(0.0)
    return feat
def get_numeric_feature_cols(df: pd.DataFrame) -> List[str]:
    exclude = {ID_COL, "primary_chronic", TARGET, "sex", "insurance", "zip3"}
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols

def drop_constants(cols: List[str], df: pd.DataFrame) -> List[str]:
    out = []
    for c in cols:
        if c not in df.columns:
            continue
        if df[c].nunique(dropna=False) <= 1:
            continue
        out.append(c)
    return out
# -----------------------------
# Adversarial importance weights (train-vs-test)
# -----------------------------
def build_importance_weights(train_feat: pd.DataFrame,
                             test_feat: pd.DataFrame,
                             cols: List[str],
                             seed: int = 42,
                             clip: Tuple[float, float] = (0.6, 1.8),
                             power: float = 0.6) -> np.ndarray:
    X_tr = train_feat[cols].copy()
    X_te = test_feat[cols].copy()
    X = pd.concat([X_tr, X_te], axis=0, ignore_index=True)
    y = np.r_[np.zeros(len(X_tr)), np.ones(len(X_te))].astype(int)
    oof_p = np.zeros(len(X), dtype=float)
    skf = StratifiedKFold(n_splits=CFG.ADV_FOLDS, shuffle=True, random_state=seed
    for fold, (ti, vi) in enumerate(skf.split(X, y), 1):
        clf = CatBoostClassifier(
            loss_function="Logloss",
            depth=4,
            learning_rate=0.05,
            iterations=CFG.ADV_ITERS,
            l2_leaf_reg=8.0,
            random_strength=1.0,
            rsm=0.85,
            verbose=0,
            allow_writing_files=False,

            random_seed=seed + 1000 + fold * 17,
        )
        clf.fit(X.iloc[ti], y[ti])
        oof_p[vi] = clf.predict_proba(X.iloc[vi])[:, 1]
        del clf
        gc.collect()
    auc = roc_auc_score(y, oof_p)
    print(f"[adv] train-vs-test AUC={auc:.4f}  (higher => more shift; weights mat
    p_tr = np.clip(oof_p[:len(X_tr)], 1e-4, 1 - 1e-4)
    odds = p_tr / (1 - p_tr)
    w = np.clip(odds, clip[0], clip[1]) ** power
    print(f"[adv] weights stats: min={w.min():.3f} p50={np.median(w):.3f} p90={np
    return w
# -----------------------------
# Base training (multi-seed bagging) — 3 models: A full RMSE, B pruned RMSE, C pr
# -----------------------------
def train_models(train_feat: pd.DataFrame,
                 test_feat: pd.DataFrame,
                 feat_full: List[str],
                 feat_pruned: List[str],
                 w_imp: Optional[np.ndarray]) -> Tuple[Dict[str, List[np.ndarray]
    y = train_feat[TARGET].values.astype(float)
    y_log = np.log1p(y)
    # stratify: chronic + target bins
    tmp = train_feat[["primary_chronic", TARGET]].copy()
    tmp["cost_bin"] = pd.qcut(tmp[TARGET], q=5, labels=False, duplicates="drop")
    tmp["strat"] = tmp["primary_chronic"].astype(str) + "_" + tmp["cost_bin"].asty
    strat = LabelEncoder().fit_transform(tmp["strat"].values)
    model_specs = {
        "A_RMSE_full_d5": dict(
            loss_function="RMSE", eval_metric="MAE",
            depth=5, l2_leaf_reg=5, min_data_in_leaf=28,
            learning_rate=CFG.LR, iterations=CFG.ITERS,
            rsm=CFG.RSM, random_strength=1.0,
        ),

        "B_RMSE_pruned_d4": dict(
            loss_function="RMSE", eval_metric="MAE",
            depth=4, l2_leaf_reg=4, min_data_in_leaf=32,
            learning_rate=CFG.LR, iterations=CFG.ITERS,
            rsm=CFG.RSM, random_strength=2.0,
        ),
        "C_LOGRMSE_pruned_d4": dict(
            loss_function="RMSE", eval_metric="RMSE",
            depth=4, l2_leaf_reg=5, min_data_in_leaf=36,
            learning_rate=CFG.LR, iterations=CFG.ITERS,
            rsm=CFG.RSM, random_strength=1.0,
        ),
    }
    model_cols = {
        "A_RMSE_full_d5": feat_full,
        "B_RMSE_pruned_d4": feat_pruned,
        "C_LOGRMSE_pruned_d4": feat_pruned,
    }
    oof_by_seed = {m: [] for m in model_specs}
    test_by_seed = {m: [] for m in model_specs}
    print("\n[base training] CatBoost CPU | shallow trees | rsm=0.8 | multi-seed 
    print(f"Seeds={CFG.N_SEEDS}, Folds={CFG.N_FOLDS}, Models={list(model_specs.key
    for seed_idx in range(CFG.N_SEEDS):
        rs = SEED + seed_idx * 7
        kf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=rs)
        oof_seed = {m: np.zeros(len(train_feat), dtype=float) for m in model_spec
        test_seed = {m: np.zeros(len(test_feat), dtype=float) for m in model_spec
        for fold, (ti, vi) in enumerate(kf.split(train_feat, strat), 1):
            for mname, params in model_specs.items():
                cols = model_cols[mname]
                X_tr = train_feat[cols].iloc[ti]
                X_va = train_feat[cols].iloc[vi]
                X_te = test_feat[cols]

                if mname.startswith("C_LOG"):
                    y_tr = y_log[ti]
                    y_va = y_log[vi]
                else:
                    y_tr = y[ti]
                    y_va = y[vi]
                cb = CatBoostRegressor(
                    **params,
                    task_type="CPU",
                    thread_count=-1,
                    verbose=0,
                    allow_writing_files=False,
                    random_seed=rs + fold * 31 + (hash(mname) % 997),
                    early_stopping_rounds=CFG.ES_ROUNDS,
                )
                sw_tr = None if w_imp is None else w_imp[ti]
                sw_va = None if w_imp is None else w_imp[vi]
                cb.fit(
                    X_tr, y_tr,
                    eval_set=(X_va, y_va),
                    sample_weight=sw_tr,
                    eval_sample_weight=sw_va,
                    verbose=0
                )
                pred_va = cb.predict(X_va)
                pred_te = cb.predict(X_te)
                if mname.startswith("C_LOG"):
                    pred_va = np.expm1(pred_va)
                    pred_te = np.expm1(pred_te)
                pred_va = np.clip(pred_va, 0.0, None)
                pred_te = np.clip(pred_te, 0.0, None)
                oof_seed[mname][vi] = pred_va
                test_seed[mname] += pred_te / CFG.N_FOLDS

                del cb
                gc.collect()
        # per-seed MAE
        maes = {m: float(mean_absolute_error(y, oof_seed[m])) for m in model_spec
        print(f"  Seed {seed_idx+1}/{CFG.N_SEEDS} OOF MAE: " + " | ".join([f"{m}=
        for m in model_specs:
            oof_by_seed[m].append(oof_seed[m])
            test_by_seed[m].append(test_seed[m])
    print("\n[seed-averaged OOF MAE per model]")
    for m in model_specs:
        oof_avg_m = np.mean(np.vstack(oof_by_seed[m]), axis=0)
        print(f"  {m:20s}: {mean_absolute_error(y, oof_avg_m):.2f}")
    return oof_by_seed, test_by_seed
# -----------------------------
# Robust stacking (Huber) with CV on meta-features
# -----------------------------
def robust_stack_huber(y: np.ndarray,
                       Z_tr: np.ndarray,
                       Z_te: np.ndarray,
                       strat_labels: np.ndarray,
                       w_imp: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarr
    """
    Z_tr: (n_train, d) meta features
    Z_te: (n_test, d)  meta features
    """
    oof = np.zeros(len(y), dtype=float)
    kf = StratifiedKFold(n_splits=CFG.META_FOLDS, shuffle=True, random_state=SEED 
    for fold, (ti, vi) in enumerate(kf.split(Z_tr, strat_labels), 1):
        reg = HuberRegressor(alpha=CFG.HUBER_ALPHA, epsilon=CFG.HUBER_EPS, max_it
        sw = None if w_imp is None else w_imp[ti]
        reg.fit(Z_tr[ti], y[ti], sample_weight=sw)
        oof[vi] = reg.predict(Z_tr[vi])

    oof = np.clip(oof, 0.0, None)
    mae = float(mean_absolute_error(y, oof))
    print(f"[stack] meta OOF MAE (Huber CV): {mae:.3f}")
    # fit full
    reg_full = HuberRegressor(alpha=CFG.HUBER_ALPHA, epsilon=CFG.HUBER_EPS, max_i
    reg_full.fit(Z_tr, y, sample_weight=w_imp if w_imp is not None else None)
    te = np.clip(reg_full.predict(Z_te), 0.0, None)
    meta = {
        "coef": reg_full.coef_.tolist(),
        "intercept": float(reg_full.intercept_),
        "meta_oof_mae": mae,
    }
    print(f"[stack] full coef={np.round(reg_full.coef_, 4)} | intercept={reg_full
    return oof, te, meta
# -----------------------------
# Shrunk chronic shifts (median-residual with hierarchical shrink)
# -----------------------------
def compute_shrunk_group_shifts(y: np.ndarray,
                                pred_oof: np.ndarray,
                                group: np.ndarray,
                                groups=(0, 1, 2),
                                unk_value=-1,
                                k=120,
                                cap=80.0) -> Tuple[Dict[int, float], float]:
    resid = y - pred_oof
    global_med = float(np.median(resid))
    group2 = group.copy()
    group2[group2 == unk_value] = 0
    shifts = {}
    for g in groups:
        mask = (group2 == g)
        r = resid[mask]
        if len(r) == 0:
            raw = global_med
            n = 0

        else:
            raw = float(np.median(r))
            n = int(mask.sum())
        alpha = n / (n + k)
        shrunk = alpha * raw + (1 - alpha) * global_med
        shrunk = float(np.clip(shrunk, -cap, cap))
        shifts[g] = shrunk
    shifts[unk_value] = shifts[0]
    return shifts, global_med
def apply_group_shifts(pred: np.ndarray, group: np.ndarray, shifts: Dict[int, flo
    out = pred.copy()
    for g, s in shifts.items():
        out[group == g] += s
    # any unexpected group -> use 0 group shift
    mask_unknown = ~np.isin(group, list(shifts.keys()))
    if mask_unknown.any():
        out[mask_unknown] += shifts.get(0, 0.0)
    return out
# =============================
# MAIN
# =============================
must_exist(TRAIN_PATH, "TRAIN")
must_exist(TEST_PATH, "TEST")
must_exist(PATIENTS_PATH, "patients")
must_exist(ADM_TRAIN_PATH, "admissions_train")
must_exist(ADM_TEST_PATH, "admissions_test")
if not os.path.exists(RECEIPTS_JOBLIB_PATH):
    print("[warn] receipts_parsed.joblib missing -> receipts features will be abs
print("\n[load] reading CSVs...")
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)
patients_df = pd.read_csv(PATIENTS_PATH)
log_shape("ed_cost_train", train_df)
log_shape("ed_cost_test", test_df)
log_shape("patients", patients_df)

print("\n[target stats]")
print(train_df[TARGET].describe().to_string())
# ids
for df in [train_df, test_df, patients_df]:
    df[ID_COL] = pd.to_numeric(df[ID_COL], errors="coerce").astype(int)
print("\n[admissions] building aggregates...")
adm_df = load_admissions_features(ADM_TRAIN_PATH, ADM_TEST_PATH)
print("  admissions features:", None if adm_df is None else adm_df.shape)
print("\n[receipts] loading receipts_parsed.joblib -> Iter7 low-dim...")
rcpt_df = load_receipts_features(RECEIPTS_JOBLIB_PATH) if os.path.exists(RECEIPTS_
if rcpt_df is None:
    print("  receipts features: None")
else:
    print(f"  receipts features: {rcpt_df.shape} | sample cols={ [c for c in rcpt_
print("\n[features] building train/test feature frames...")
train_feat = build_features(train_df, patients_df, adm_df, rcpt_df)
test_feat  = build_features(test_df,  patients_df, adm_df, rcpt_df)
# full numeric features
feat_full = drop_constants(get_numeric_feature_cols(train_feat), train_feat)
# Iter7-pruned set (stable)
pruned_candidates = [
    "prior_ed_visits_5y","prior_ed_cost_5y_usd","prior_cost_cap20k","sqrt_prior_c
    "cost_per_visit","log_visits",
    "baseline_next3y",
    "chronic_encoded","age","sex_encoded","insurance_encoded","zip_region","ins_x_
    "charlson_max","charlson_mean","pct_emergent",
    "cost_per_em","cost_hhi","pct_cost_procedure","pct_cost_critical","pct_cost_h
    "n_high_acuity_total","has_critical_care","has_99285","max_em_level","n_uniqu
    "logprior_x_pctcritical","logprior_x_highacu","cost_per_code",
]
feat_pruned = drop_constants([c for c in pruned_candidates if c in train_feat.col
print(f"  FULL feature count:   {len(feat_full)}")

print(f"  PRUNED feature count: {len(feat_pruned)}")
print(f"  PRUNED features: {feat_pruned}")
# Fill medians for selected cols
for c in set(feat_full + feat_pruned + [TARGET, "baseline_next3y", "chronic_encod
    if c in train_feat.columns:
        med = train_feat[c].median() if not train_feat[c].isna().all() else 0.0
        train_feat[c] = pd.to_numeric(train_feat[c], errors="coerce").fillna(med)
    if c in test_feat.columns:
        med = train_feat[c].median() if (c in train_feat.columns and not train_fe
        test_feat[c] = pd.to_numeric(test_feat[c], errors="coerce").fillna(med)
# Baseline / y
y = train_feat[TARGET].values.astype(float)
baseline_tr = train_feat["baseline_next3y"].values.astype(float)
baseline_te = test_feat["baseline_next3y"].values.astype(float)
# Strat labels (for meta CV too)
tmp = train_feat[["primary_chronic", TARGET]].copy()
tmp["cost_bin"] = pd.qcut(tmp[TARGET], q=5, labels=False, duplicates="drop")
tmp["strat"] = tmp["primary_chronic"].astype(str) + "_" + tmp["cost_bin"].astype(
strat_labels = LabelEncoder().fit_transform(tmp["strat"].values)
# -----------------------------
# 1) Importance weights
# -----------------------------
print("\n" + "="*70)
print("[STEP 1] Adversarial importance weights (train vs test)")
print("="*70)
w_imp = build_importance_weights(train_feat, test_feat, feat_pruned, seed=SEED, c
# -----------------------------
# 2) Train base models
# -----------------------------
print("\n" + "="*70)
print("[STEP 2] Base models (Iter7 core) with importance weighting")
print("="*70)
oof_by_seed, test_by_seed = train_models(train_feat, test_feat, feat_full, feat_p
models = list(oof_by_seed.keys())

oof_avg = {m: np.mean(np.vstack(oof_by_seed[m]), axis=0) for m in models}
test_avg = {m: np.mean(np.vstack(test_by_seed[m]), axis=0) for m in models}
print("\n[base avg] OOF MAE summary:")
for m in models:
    print(f"  {m:20s}: {mean_absolute_error(y, oof_avg[m]):.3f}")
# -----------------------------
# 3) Robust stacking (meta model)
# -----------------------------
print("\n" + "="*70)
print("[STEP 3] Robust stacking (Huber) on OOF preds + baseline")
print("="*70)
# meta features: base preds + baseline
Z_tr = np.column_stack([oof_avg[m] for m in models] + [baseline_tr])
Z_te = np.column_stack([test_avg[m] for m in models] + [baseline_te])
oof_stack, test_stack, meta_info = robust_stack_huber(
    y=y, Z_tr=Z_tr, Z_te=Z_te, strat_labels=strat_labels, w_imp=w_imp
)
print("[stack] OOF quantiles:", qdict(oof_stack))
print("[stack] TEST quantiles:", qdict(test_stack))
# -----------------------------
# 4) Shrunk chronic shifts (safer calibration)
# -----------------------------
print("\n" + "="*70)
print("[STEP 4] Shrunk chronic shifts (median residual per chronic, with shrink + 
print("="*70)
chronic_tr = train_feat["chronic_encoded"].values.astype(int)  # -1,0,1,2
chronic_te = test_feat["chronic_encoded"].values.astype(int)
shifts, global_med = compute_shrunk_group_shifts(
    y=y,
    pred_oof=oof_stack,
    group=chronic_tr,
    k=CFG.SHIFT_K,

    cap=CFG.SHIFT_CAP
)
print(f"[shift] global median residual = {global_med:+.3f}")
print(f"[shift] shrunk shifts (k={CFG.SHIFT_K}, cap={CFG.SHIFT_CAP}): {shifts}")
oof_final = apply_group_shifts(oof_stack, chronic_tr, shifts)
test_final = apply_group_shifts(test_stack, chronic_te, shifts)
oof_final = np.clip(oof_final, 0.0, None)
test_final = np.clip(test_final, 0.0, None)
mae_stack = float(mean_absolute_error(y, oof_stack))
mae_final = float(mean_absolute_error(y, oof_final))
print(f"[OOF] stack MAE (pre-shift) : {mae_stack:.3f}")
print(f"[OOF] final MAE (post-shift): {mae_final:.3f}")
print("[OOF] final quantiles:", qdict(oof_final))
# clip test to safe range
y_max = float(np.max(y))
test_final = np.clip(test_final, 0.0, y_max * CFG.PRED_CLIP_MULT)
# -----------------------------
# 5) Write submission
# -----------------------------
sub = pd.DataFrame({
    ID_COL: test_df[ID_COL].values.astype(int),
    TARGET: np.round(test_final.astype(float), 2),
})[[ID_COL, TARGET]]
out_path = Path(OUT_SUB_PATH)
out_path.parent.mkdir(parents=True, exist_ok=True)
sub.to_csv(out_path, index=False)
print("\n" + "="*70)
print("[SUBMISSION sanity checks]")
print("="*70)
print("submission shape:", sub.shape)
print("columns:", list(sub.columns))
print("any NaNs:", bool(np.isnan(sub[TARGET]).any()))

print("pred min/median/max:", float(sub[TARGET].min()), float(sub[TARGET].median(
print("pred quantiles:", qdict(sub[TARGET].values))
print("\nSaved submission to:", str(out_path))
print("\nPaste back after submit:")
print("  (1) leaderboard MAE")
print("  (2) [adv] AUC + weights stats")
print("  (3) base model avg OOF MAEs")
print("  (4) stack MAE pre/post shift + shifts dict")
-----------------------------------------------------------------
----------
TypeError                                 Traceback (most recent call 
last)
Cell In[15], line 853
    851 print("[STEP 2] Base models (Iter7 core) with 
importance weighting")
    852 print("="*70)
--> 853 oof_by_seed, test_by_seed = 
train_models(train_feat, test_feat, feat_full, feat_pruned, 
w_imp)
    855 models = list(oof_by_seed.keys())
    856 oof_avg = {m: np.mean(np.vstack(oof_by_seed[m]), 
axis=0) for m in models}
Cell In[15], line 636, in train_models(train_feat, test_feat, 
feat_full, feat_pruned, w_imp)
    633 sw_tr = None if w_imp is None else w_imp[ti]
    634 sw_va = None if w_imp is None else w_imp[vi]
--> 636 cb.fit(
    637     X_tr, y_tr,
    638     eval_set=(X_va, y_va),
    639     sample_weight=sw_tr,
    640     eval_sample_weight=sw_va,
    641     verbose=0

    642 )
    644 pred_va = cb.predict(X_va)
    645 pred_te = cb.predict(X_te)
TypeError: CatBoostRegressor.fit() got an unexpected 
keyword argument 'eval_sample_weight'
debug
Thought for 48s
你这个报错的原因很简单：你装的 CatBoost 版本不支持 eval_sample_weight  这个参数。
要给验证集加权（early stopping 用到的 eval），最稳的写法是 用 catboost.Pool(...,
weight=...) 。
下面给你一个可直接粘贴运行的“修复版全量 cell”（我只改了关键点：cb.fit()  改成用
Pool ，并兼容没有权重的情况）。你直接把之前那格替换掉跑即可。
python
# === ITERATION 10 (FIXED): Iter7-Core + AdvWeights + RobustStack + Shrunk Chroni
# Fix: CatBoostRegressor.fit() does NOT support eval_sample_weight in your versio
#      Use Pool(weight=...) for train/eval.
import os, re, sys, gc, warnings, random
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
DATA_DIR = r"D:\AgentDs\agent_ds_healthcare"

TRAIN_PATH = r"D:\AgentDs\agent_ds_healthcare\ed_cost_train.csv"
TEST_PATH  = r"D:\AgentDs\agent_ds_healthcare\ed_cost_test.csv"
PATIENTS_PATH = r"D:\AgentDs\agent_ds_healthcare\patients.csv"
ADM_TRAIN_PATH = r"D:\AgentDs\agent_ds_healthcare\admissions_train.csv"
ADM_TEST_PATH  = r"D:\AgentDs\agent_ds_healthcare\admissions_test.csv"
RECEIPTS_JOBLIB_PATH = r"D:\AgentDs\agent_ds_healthcare\receipts_parsed.joblib"
OUT_SUB_PATH = r"D:\AgentDs\agent_ds_healthcare\submission.csv"
ID_COL = "patient_id"
TARGET = "ed_cost_next3y_usd"
print("="*100)
print("ITERATION 10 | FIXED (Pool weights) | Iter7-Core + AdvWeights + RobustStac
print("="*100)
def _pip_install(pkg: str):
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
try:
    from joblib import load as joblib_load
except Exception:
    _pip_install("joblib")
    from joblib import load as joblib_load
try:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import mean_absolute_error, roc_auc_score
    from sklearn.linear_model import HuberRegressor
except Exception:
    _pip_install("scikit-learn")
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import mean_absolute_error, roc_auc_score
    from sklearn.linear_model import HuberRegressor
try:
    from catboost import CatBoostRegressor, CatBoostClassifier, Pool
except Exception:

    _pip_install("catboost")
    from catboost import CatBoostRegressor, CatBoostClassifier, Pool
class CFG:
    # Base CV bagging
    N_FOLDS = 7
    N_SEEDS = 5
    # CatBoost base
    ITERS = 3000
    ES_ROUNDS = 120
    LR = 0.03
    RSM = 0.80
    # Adversarial weighting
    ADV_FOLDS = 5
    ADV_ITERS = 2000
    ADV_CLIP = (0.60, 1.80)
    ADV_POWER = 0.60
    # Stacker
    META_FOLDS = 7
    HUBER_ALPHA = 1e-3
    HUBER_EPS = 1.35
    # Shrunk group shift
    SHIFT_K = 120
    SHIFT_CAP = 80.0
    PRED_CLIP_MULT = 1.5
def must_exist(path: str, name: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {name}: {path}")
def log_shape(name: str, df: pd.DataFrame):
    mem = df.memory_usage(deep=True).sum() / (1024**2)
    print(f"[{name}] shape={df.shape} | cols={df.shape[1]} | mem={mem:.2f} MB")
def qdict(x, qs=(0,0.01,0.05,0.1,0.5,0.9,0.95,0.99,1.0)):
    x = np.asarray(x, dtype=float)

    return {q: float(np.quantile(x, q)) for q in qs}
def standardize_zip3(z):
    if z is None or (isinstance(z, float) and np.isnan(z)):
        return None
    s = re.sub(r"\D", "", str(z).strip())
    if not s:
        return None
    return s.zfill(3)
def norm_code(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().upper()
    if s == "" or s.lower() == "nan":
        return None
    if re.fullmatch(r"\d+\.0+", s):
        s = s.split(".")[0]
    s = re.sub(r"\s+", "", s)
    return s
# -----------------------------
# Admissions aggregates
# -----------------------------
def load_admissions_features(adm_train_path: str, adm_test_path: str) -> Optional
    dfs = []
    for path in [adm_train_path, adm_test_path]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "readmit_30d" in df.columns:
                df = df.drop(columns=["readmit_30d"])
            dfs.append(df)
    if not dfs:
        return None
    adm = pd.concat(dfs, ignore_index=True)
    need = ["patient_id", "charlson_band", "acuity_emergent"]
    if not all(c in adm.columns for c in need):
        return None

    adm["patient_id"] = pd.to_numeric(adm["patient_id"], errors="coerce")
    adm["charlson_band"] = pd.to_numeric(adm["charlson_band"], errors="coerce")
    adm["acuity_emergent"] = pd.to_numeric(adm["acuity_emergent"], errors="coerce
    out = adm.groupby("patient_id").agg(
        charlson_max=("charlson_band", "max"),
        charlson_mean=("charlson_band", "mean"),
        pct_emergent=("acuity_emergent", "mean"),
    ).reset_index()
    for c in ["charlson_max", "charlson_mean", "pct_emergent"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out
# -----------------------------
# Receipts loader + Iter7 low-dim builder (same as before, keep minimal)
# -----------------------------
def _extract_lineitems_df(obj) -> Optional[pd.DataFrame]:
    if isinstance(obj, dict):
        for k in ["lineitems_df", "lineitems", "items_df", "items", "line_items_d
            if k in obj and isinstance(obj[k], pd.DataFrame):
                return obj[k].copy()
        for k, v in obj.items():
            if isinstance(v, pd.DataFrame) and ("patient_id" in v.columns):
                return v.copy()
        return None
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if isinstance(obj, (list, tuple)):
        for v in obj:
            if isinstance(v, pd.DataFrame) and ("patient_id" in v.columns):
                return v.copy()
    return None
def _choose_amount_column(li: pd.DataFrame) -> Tuple[Optional[str], str]:
    cols = set(li.columns)
    direct_candidates = ["line_total", "line_total_usd", "extended_price", "item_t
    for c in direct_candidates:
        if c in cols:
            return c, "direct"

    qty_col = next((c for c in ["qty", "quantity"] if c in cols), None)
    unit_col = next((c for c in ["unit_price", "price", "unit_cost"] if c in cols
    if qty_col and unit_col:
        return f"{qty_col}__x__{unit_col}", "qtyxunit"
    total_candidates = ["sum_line_total", "receipt_total", "sum_items", "total_am
    total_col = next((c for c in total_candidates if c in cols), None)
    if total_col:
        return total_col, "uniform_total"
    if "total" in cols:
        return "total", "direct"
    return None, "none"
def build_receipt_features_iter7(li: pd.DataFrame) -> Optional[pd.DataFrame]:
    li = li.copy()
    if ID_COL not in li.columns:
        return None
    code_col = None
    for c in ["code", "cpt", "cpt_code", "hcpcs", "proc_code"]:
        if c in li.columns:
            code_col = c
            break
    if code_col is None:
        return None
    amt_col, amt_mode = _choose_amount_column(li)
    if amt_col is None or amt_mode == "none":
        return None
    li[ID_COL] = pd.to_numeric(li[ID_COL], errors="coerce").astype("Int64")
    li = li.dropna(subset=[ID_COL]).copy()
    li[ID_COL] = li[ID_COL].astype(int)
    li["code_norm"] = li[code_col].map(norm_code)
    li = li.dropna(subset=["code_norm"]).copy()
    if amt_mode == "direct":

        li["amt"] = pd.to_numeric(li[amt_col], errors="coerce").fillna(0.0).astyp
    elif amt_mode == "qtyxunit":
        qty_col, unit_col = amt_col.split("__x__")
        qty = pd.to_numeric(li[qty_col], errors="coerce").fillna(0.0).astype(float
        unit = pd.to_numeric(li[unit_col], errors="coerce").fillna(0.0).astype(fl
        li["amt"] = (qty * unit).astype(float)
    else:  # uniform_total
        total_per_patient = pd.to_numeric(li[amt_col], errors="coerce").fillna(0.
        total_per_patient = total_per_patient.groupby(li[ID_COL]).transform("first
        sizes = li.groupby(ID_COL)["code_norm"].transform("size").astype(float).c
        li["amt"] = (total_per_patient / sizes).astype(float)
    li.loc[li["amt"] < 0, "amt"] = 0.0
    receipt_total = li.groupby(ID_COL)["amt"].sum().rename("receipt_total")
    li = li.join(receipt_total, on=ID_COL)
    denom = li["receipt_total"].replace(0.0, np.nan)
    is_em = li["code_norm"].isin(["99281","99282","99283","99284","99285"])
    em_map = {"99281":1,"99282":2,"99283":3,"99284":4,"99285":5}
    em_level = li["code_norm"].map(em_map).fillna(0).astype(float)
    is_crit = li["code_norm"].isin(["99291","99292"])
    is_obs  = li["code_norm"].str.startswith("G037", na=False)
    is_high = li["code_norm"].isin(["31500","36556","32551","36620","92950"])
    code_num = pd.to_numeric(li["code_norm"].where(li["code_norm"].str.fullmatch(
    is_imaging = code_num.between(70000, 79999)
    is_proc_general = code_num.between(10000, 69999)
    is_proc_any = is_high | (is_proc_general & (~is_high) & (~is_em) & (~is_crit)
    share = (li["amt"] / denom).fillna(0.0)
    cost_hhi = (share * share).groupby(li[ID_COL]).sum().rename("cost_hhi")
    n_unique_codes = li.groupby(ID_COL)["code_norm"].nunique().rename("n_unique_c
    n_em_codes = is_em.astype(int).groupby(li[ID_COL]).sum().rename("n_em_codes")
    max_em_level = em_level.groupby(li[ID_COL]).max().rename("max_em_level")
    has_critical_care = is_crit.astype(int).groupby(li[ID_COL]).max().rename("has_
    has_high_acuity   = is_high.astype(int).groupby(li[ID_COL]).max().rename("has_

    has_imaging       = is_imaging.astype(int).groupby(li[ID_COL]).max().rename("
    has_observation   = is_obs.astype(int).groupby(li[ID_COL]).max().rename("has_
    has_99285         = (li["code_norm"].eq("99285").astype(int).groupby(li[ID_CO
    em_total   = li.loc[is_em].groupby(ID_COL)["amt"].sum().rename("em_total")
    crit_total = li.loc[is_crit].groupby(ID_COL)["amt"].sum().rename("crit_total"
    proc_total = li.loc[is_proc_any].groupby(ID_COL)["amt"].sum().rename("proc_tot
    high_total = li.loc[is_high].groupby(ID_COL)["amt"].sum().rename("high_total"
    out = pd.concat([
        n_unique_codes, cost_hhi, n_em_codes, max_em_level,
        has_critical_care, has_high_acuity, has_imaging, has_observation, has_992
        receipt_total
    ], axis=1).reset_index()
    for s in [em_total, crit_total, proc_total, high_total]:
        out = out.merge(s.reset_index(), on=ID_COL, how="left")
    for c in ["em_total","crit_total","proc_total","high_total","receipt_total"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    d2 = out["receipt_total"].replace(0.0, np.nan)
    out["pct_cost_critical"] = (out["crit_total"] / d2).replace([np.inf, -np.inf]
    out["pct_cost_procedure"] = (out["proc_total"] / d2).replace([np.inf, -np.inf
    out["pct_cost_high_acuity"] = (out["high_total"] / d2).replace([np.inf, -np.i
    out["cost_per_em"] = np.where(
        out["n_em_codes"] > 0,
        out["receipt_total"] / out["n_em_codes"].clip(lower=1),
        out["receipt_total"]
    )
    out["n_high_acuity_total"] = (out["has_high_acuity"].fillna(0).astype(int) + 
    out.drop(columns=["em_total","crit_total","proc_total","high_total","receipt_t
    for c in out.columns:
        if c == ID_COL: continue
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out

def load_receipts_features(joblib_path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(joblib_path):
        return None
    try:
        obj = joblib_load(joblib_path)
        li = _extract_lineitems_df(obj)
        if li is None:
            return None
        feat = build_receipt_features_iter7(li)
        if feat is None:
            return None
        feat[ID_COL] = pd.to_numeric(feat[ID_COL], errors="coerce").astype("Int64
        feat = feat.dropna(subset=[ID_COL]).copy()
        feat[ID_COL] = feat[ID_COL].astype(int)
        feat = feat.drop_duplicates(ID_COL, keep="last")
        return feat
    except Exception as e:
        print(f"[warn] receipts joblib load failed: {type(e).__name__}: {e}")
        return None
# -----------------------------
# Build features (Iter7-core)
# -----------------------------
def build_features(ed_df: pd.DataFrame,
                   patients_df: pd.DataFrame,
                   adm_df: Optional[pd.DataFrame],
                   rcpt_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    feat = ed_df.copy()
    chronic_map = {"PNEUMONIA": 0, "DIABETESCOMP": 1, "HF": 2}
    feat["primary_chronic"] = feat["primary_chronic"].astype(str)
    feat["chronic_encoded"] = feat["primary_chronic"].str.upper().map(chronic_map
    feat["prior_ed_visits_5y"] = pd.to_numeric(feat["prior_ed_visits_5y"], errors
    feat["prior_ed_cost_5y_usd"] = pd.to_numeric(feat["prior_ed_cost_5y_usd"], er
    feat["prior_cost_cap20k"] = feat["prior_ed_cost_5y_usd"].clip(upper=20000.0)
    feat["sqrt_prior_cost"] = np.sqrt(feat["prior_ed_cost_5y_usd"].clip(lower=0.0
    feat["log_prior_cost"] = np.log1p(feat["prior_ed_cost_5y_usd"].clip(lower=0.0
    feat["log_prior_cost_cap20k"] = np.log1p(feat["prior_cost_cap20k"].clip(lower

    feat["log_visits"] = np.log1p(feat["prior_ed_visits_5y"].clip(lower=0.0))
    feat["cost_per_visit"] = feat["prior_ed_cost_5y_usd"] / feat["prior_ed_visits_
    feat["baseline_next3y"] = feat["prior_ed_cost_5y_usd"] * (3.0/5.0)
    p = patients_df.copy()
    p[ID_COL] = pd.to_numeric(p[ID_COL], errors="coerce").astype(int)
    p["age"] = pd.to_numeric(p["age"], errors="coerce").fillna(p["age"].median())
    p["sex_encoded"] = (p["sex"].astype(str).str.upper() == "M").astype(int)
    ins = p["insurance"].astype(str).str.lower()
    ins_map = {"private": 2, "public": 1, "self_pay": 0, "selfpay": 0}
    p["insurance_encoded"] = ins.map(ins_map).fillna(-1).astype(float)
    z3 = p["zip3"].apply(standardize_zip3).astype("string")
    zr = z3.fillna("000").str.replace(r"\D", "", regex=True).str.zfill(3).str[0]
    p["zip_region"] = pd.to_numeric(zr, errors="coerce").fillna(-1).astype(float)
    feat = feat.merge(p[[ID_COL, "age", "sex_encoded", "insurance_encoded", "zip_
    feat["ins_x_chronic"] = feat["insurance_encoded"].fillna(-1) * feat["chronic_
    if adm_df is not None:
        feat = feat.merge(adm_df, on=ID_COL, how="left")
        for c in ["charlson_max", "charlson_mean", "pct_emergent"]:
            if c in feat.columns:
                feat[c] = pd.to_numeric(feat[c], errors="coerce").fillna(0.0)
    if rcpt_df is not None:
        feat = feat.merge(rcpt_df, on=ID_COL, how="left")
        for c in rcpt_df.columns:
            if c == ID_COL: continue
            feat[c] = pd.to_numeric(feat[c], errors="coerce").fillna(0.0)
    feat["logprior_x_pctcritical"] = feat["log_prior_cost"] * pd.to_numeric(feat.
    feat["logprior_x_highacu"] = feat["log_prior_cost"] * pd.to_numeric(feat.get(
    if "n_unique_codes" in feat.columns:
        feat["cost_per_code"] = feat["prior_ed_cost_5y_usd"] / feat["n_unique_cod
    else:
        feat["cost_per_code"] = feat["prior_ed_cost_5y_usd"]

    feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    for c in feat.columns:
        if c in [ID_COL, "primary_chronic", TARGET]:
            continue
        feat[c] = pd.to_numeric(feat[c], errors="coerce").fillna(0.0)
    return feat
def get_numeric_feature_cols(df: pd.DataFrame) -> List[str]:
    exclude = {ID_COL, "primary_chronic", TARGET, "sex", "insurance", "zip3"}
    cols = []
    for c in df.columns:
        if c in exclude: 
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols
def drop_constants(cols: List[str], df: pd.DataFrame) -> List[str]:
    out = []
    for c in cols:
        if c in df.columns and df[c].nunique(dropna=False) > 1:
            out.append(c)
    return out
# -----------------------------
# Adversarial weights
# -----------------------------
def build_importance_weights(train_feat: pd.DataFrame,
                             test_feat: pd.DataFrame,
                             cols: List[str],
                             seed: int = 42,
                             clip: Tuple[float, float] = (0.6, 1.8),
                             power: float = 0.6) -> np.ndarray:
    X_tr = train_feat[cols].copy()
    X_te = test_feat[cols].copy()
    X = pd.concat([X_tr, X_te], axis=0, ignore_index=True)
    y = np.r_[np.zeros(len(X_tr)), np.ones(len(X_te))].astype(int)

    oof_p = np.zeros(len(X), dtype=float)
    skf = StratifiedKFold(n_splits=CFG.ADV_FOLDS, shuffle=True, random_state=seed
    for fold, (ti, vi) in enumerate(skf.split(X, y), 1):
        clf = CatBoostClassifier(
            loss_function="Logloss",
            depth=4,
            learning_rate=0.05,
            iterations=CFG.ADV_ITERS,
            l2_leaf_reg=8.0,
            random_strength=1.0,
            rsm=0.85,
            verbose=0,
            allow_writing_files=False,
            random_seed=seed + 1000 + fold * 17,
        )
        clf.fit(X.iloc[ti], y[ti])
        oof_p[vi] = clf.predict_proba(X.iloc[vi])[:, 1]
        del clf
        gc.collect()
    auc = roc_auc_score(y, oof_p)
    print(f"[adv] train-vs-test AUC={auc:.4f}")
    p_tr = np.clip(oof_p[:len(X_tr)], 1e-4, 1 - 1e-4)
    odds = p_tr / (1 - p_tr)
    w = np.clip(odds, clip[0], clip[1]) ** power
    print(f"[adv] weights stats: min={w.min():.3f} p50={np.median(w):.3f} p90={np
    return w
# -----------------------------
# TRAIN MODELS (FIXED: use Pool for weights)
# -----------------------------
def train_models(train_feat: pd.DataFrame,
                 test_feat: pd.DataFrame,
                 feat_full: List[str],
                 feat_pruned: List[str],
                 w_imp: Optional[np.ndarray]) -> Tuple[Dict[str, List[np.ndarray]
    y = train_feat[TARGET].values.astype(float)
    y_log = np.log1p(y)

    tmp = train_feat[["primary_chronic", TARGET]].copy()
    tmp["cost_bin"] = pd.qcut(tmp[TARGET], q=5, labels=False, duplicates="drop")
    tmp["strat"] = tmp["primary_chronic"].astype(str) + "_" + tmp["cost_bin"].asty
    strat = LabelEncoder().fit_transform(tmp["strat"].values)
    model_specs = {
        "A_RMSE_full_d5": dict(
            loss_function="RMSE", eval_metric="MAE",
            depth=5, l2_leaf_reg=5, min_data_in_leaf=28,
            learning_rate=CFG.LR, iterations=CFG.ITERS,
            rsm=CFG.RSM, random_strength=1.0,
        ),
        "B_RMSE_pruned_d4": dict(
            loss_function="RMSE", eval_metric="MAE",
            depth=4, l2_leaf_reg=4, min_data_in_leaf=32,
            learning_rate=CFG.LR, iterations=CFG.ITERS,
            rsm=CFG.RSM, random_strength=2.0,
        ),
        "C_LOGRMSE_pruned_d4": dict(
            loss_function="RMSE", eval_metric="RMSE",
            depth=4, l2_leaf_reg=5, min_data_in_leaf=36,
            learning_rate=CFG.LR, iterations=CFG.ITERS,
            rsm=CFG.RSM, random_strength=1.0,
        ),
    }
    model_cols = {
        "A_RMSE_full_d5": feat_full,
        "B_RMSE_pruned_d4": feat_pruned,
        "C_LOGRMSE_pruned_d4": feat_pruned,
    }
    oof_by_seed = {m: [] for m in model_specs}
    test_by_seed = {m: [] for m in model_specs}
    print("\n[base training] (FIXED) using Pool(weight=...) for train/eval")
    print(f"Seeds={CFG.N_SEEDS}, Folds={CFG.N_FOLDS}, Models={list(model_specs.key
    for seed_idx in range(CFG.N_SEEDS):
        rs = SEED + seed_idx * 7

        kf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=rs)
        oof_seed = {m: np.zeros(len(train_feat), dtype=float) for m in model_spec
        test_seed = {m: np.zeros(len(test_feat), dtype=float) for m in model_spec
        for fold, (ti, vi) in enumerate(kf.split(train_feat, strat), 1):
            for mname, params in model_specs.items():
                cols = model_cols[mname]
                X_tr = train_feat[cols].iloc[ti]
                X_va = train_feat[cols].iloc[vi]
                X_te = test_feat[cols]
                if mname.startswith("C_LOG"):
                    y_tr = y_log[ti]
                    y_va = y_log[vi]
                else:
                    y_tr = y[ti]
                    y_va = y[vi]
                cb = CatBoostRegressor(
                    **params,
                    task_type="CPU",
                    thread_count=-1,
                    verbose=0,
                    allow_writing_files=False,
                    random_seed=rs + fold * 31 + (hash(mname) % 997),
                    early_stopping_rounds=CFG.ES_ROUNDS,
                )
                if w_imp is None:
                    cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0)
                else:
                    sw_tr = w_imp[ti]
                    sw_va = w_imp[vi]
                    train_pool = Pool(X_tr, y_tr, weight=sw_tr)
                    val_pool = Pool(X_va, y_va, weight=sw_va)
                    cb.fit(train_pool, eval_set=val_pool, verbose=0)
                pred_va = cb.predict(X_va)

                pred_te = cb.predict(X_te)
                if mname.startswith("C_LOG"):
                    pred_va = np.expm1(pred_va)
                    pred_te = np.expm1(pred_te)
                pred_va = np.clip(pred_va, 0.0, None)
                pred_te = np.clip(pred_te, 0.0, None)
                oof_seed[mname][vi] = pred_va
                test_seed[mname] += pred_te / CFG.N_FOLDS
                del cb
                gc.collect()
        maes = {m: float(mean_absolute_error(y, oof_seed[m])) for m in model_spec
        print(f"  Seed {seed_idx+1}/{CFG.N_SEEDS} OOF MAE: " + " | ".join([f"{m}=
        for m in model_specs:
            oof_by_seed[m].append(oof_seed[m])
            test_by_seed[m].append(test_seed[m])
    print("\n[seed-averaged OOF MAE per model]")
    for m in model_specs:
        oof_avg_m = np.mean(np.vstack(oof_by_seed[m]), axis=0)
        print(f"  {m:20s}: {mean_absolute_error(y, oof_avg_m):.2f}")
    return oof_by_seed, test_by_seed
# -----------------------------
# Robust stacking (Huber) with meta CV
# -----------------------------
def robust_stack_huber(y: np.ndarray,
                       Z_tr: np.ndarray,
                       Z_te: np.ndarray,
                       strat_labels: np.ndarray,
                       w_imp: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarr
    oof = np.zeros(len(y), dtype=float)
    kf = StratifiedKFold(n_splits=CFG.META_FOLDS, shuffle=True, random_state=SEED 

    for fold, (ti, vi) in enumerate(kf.split(Z_tr, strat_labels), 1):
        reg = HuberRegressor(alpha=CFG.HUBER_ALPHA, epsilon=CFG.HUBER_EPS, max_it
        sw = None if w_imp is None else w_imp[ti]
        reg.fit(Z_tr[ti], y[ti], sample_weight=sw)
        oof[vi] = reg.predict(Z_tr[vi])
    oof = np.clip(oof, 0.0, None)
    mae = float(mean_absolute_error(y, oof))
    print(f"[stack] meta OOF MAE (Huber CV): {mae:.3f}")
    reg_full = HuberRegressor(alpha=CFG.HUBER_ALPHA, epsilon=CFG.HUBER_EPS, max_i
    reg_full.fit(Z_tr, y, sample_weight=w_imp if w_imp is not None else None)
    te = np.clip(reg_full.predict(Z_te), 0.0, None)
    meta = {"coef": reg_full.coef_.tolist(), "intercept": float(reg_full.intercept
    print(f"[stack] full coef={np.round(reg_full.coef_, 4)} | intercept={reg_full
    return oof, te, meta
# -----------------------------
# Shrunk chronic shifts
# -----------------------------
def compute_shrunk_group_shifts(y: np.ndarray,
                                pred_oof: np.ndarray,
                                group: np.ndarray,
                                groups=(0, 1, 2),
                                unk_value=-1,
                                k=120,
                                cap=80.0) -> Tuple[Dict[int, float], float]:
    resid = y - pred_oof
    global_med = float(np.median(resid))
    group2 = group.copy()
    group2[group2 == unk_value] = 0
    shifts = {}
    for g in groups:
        mask = (group2 == g)
        r = resid[mask]
        raw = float(np.median(r)) if len(r) else global_med
        n = int(mask.sum()) if len(r) else 0

        alpha = n / (n + k)
        shrunk = alpha * raw + (1 - alpha) * global_med
        shrunk = float(np.clip(shrunk, -cap, cap))
        shifts[g] = shrunk
    shifts[unk_value] = shifts[0]
    return shifts, global_med
def apply_group_shifts(pred: np.ndarray, group: np.ndarray, shifts: Dict[int, flo
    out = pred.copy()
    for g, s in shifts.items():
        out[group == g] += s
    mask_unknown = ~np.isin(group, list(shifts.keys()))
    if mask_unknown.any():
        out[mask_unknown] += shifts.get(0, 0.0)
    return out
# =============================
# MAIN RUN
# =============================
must_exist(TRAIN_PATH, "TRAIN")
must_exist(TEST_PATH, "TEST")
must_exist(PATIENTS_PATH, "patients")
must_exist(ADM_TRAIN_PATH, "admissions_train")
must_exist(ADM_TEST_PATH, "admissions_test")
if not os.path.exists(RECEIPTS_JOBLIB_PATH):
    print("[warn] receipts_parsed.joblib missing -> receipts features absent (lik
print("\n[load] reading CSVs...")
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)
patients_df = pd.read_csv(PATIENTS_PATH)
log_shape("ed_cost_train", train_df)
log_shape("ed_cost_test", test_df)
log_shape("patients", patients_df)
print("\n[target stats]")
print(train_df[TARGET].describe().to_string())

for df in [train_df, test_df, patients_df]:
    df[ID_COL] = pd.to_numeric(df[ID_COL], errors="coerce").astype(int)
print("\n[admissions] building aggregates...")
adm_df = load_admissions_features(ADM_TRAIN_PATH, ADM_TEST_PATH)
print("  admissions features:", None if adm_df is None else adm_df.shape)
print("\n[receipts] loading receipts_parsed.joblib -> Iter7 low-dim...")
rcpt_df = load_receipts_features(RECEIPTS_JOBLIB_PATH) if os.path.exists(RECEIPTS_
print("  receipts features:", None if rcpt_df is None else rcpt_df.shape)
print("\n[features] building train/test frames...")
train_feat = build_features(train_df, patients_df, adm_df, rcpt_df)
test_feat  = build_features(test_df,  patients_df, adm_df, rcpt_df)
feat_full = drop_constants(get_numeric_feature_cols(train_feat), train_feat)
pruned_candidates = [
    "prior_ed_visits_5y","prior_ed_cost_5y_usd","prior_cost_cap20k","sqrt_prior_c
    "cost_per_visit","log_visits",
    "baseline_next3y",
    "chronic_encoded","age","sex_encoded","insurance_encoded","zip_region","ins_x_
    "charlson_max","charlson_mean","pct_emergent",
    "cost_per_em","cost_hhi","pct_cost_procedure","pct_cost_critical","pct_cost_h
    "n_high_acuity_total","has_critical_care","has_99285","max_em_level","n_uniqu
    "logprior_x_pctcritical","logprior_x_highacu","cost_per_code",
]
feat_pruned = drop_constants([c for c in pruned_candidates if c in train_feat.col
print(f"  FULL feature count:   {len(feat_full)}")
print(f"  PRUNED feature count: {len(feat_pruned)}")
# fill medians
for c in set(feat_full + feat_pruned + [TARGET, "baseline_next3y", "chronic_encod
    if c in train_feat.columns:
        med = train_feat[c].median() if not train_feat[c].isna().all() else 0.0
        train_feat[c] = pd.to_numeric(train_feat[c], errors="coerce").fillna(med)
    if c in test_feat.columns:
        med = train_feat[c].median() if (c in train_feat.columns and not train_fe

        test_feat[c] = pd.to_numeric(test_feat[c], errors="coerce").fillna(med)
y = train_feat[TARGET].values.astype(float)
baseline_tr = train_feat["baseline_next3y"].values.astype(float)
baseline_te = test_feat["baseline_next3y"].values.astype(float)
# strat labels
tmp = train_feat[["primary_chronic", TARGET]].copy()
tmp["cost_bin"] = pd.qcut(tmp[TARGET], q=5, labels=False, duplicates="drop")
tmp["strat"] = tmp["primary_chronic"].astype(str) + "_" + tmp["cost_bin"].astype(
strat_labels = LabelEncoder().fit_transform(tmp["strat"].values)
print("\n" + "="*70)
print("[STEP 1] Adversarial importance weights")
print("="*70)
w_imp = build_importance_weights(train_feat, test_feat, feat_pruned, seed=SEED, c
print("\n" + "="*70)
print("[STEP 2] Base models with weights (Pool fix)")
print("="*70)
oof_by_seed, test_by_seed = train_models(train_feat, test_feat, feat_full, feat_p
models = list(oof_by_seed.keys())
oof_avg = {m: np.mean(np.vstack(oof_by_seed[m]), axis=0) for m in models}
test_avg = {m: np.mean(np.vstack(test_by_seed[m]), axis=0) for m in models}
print("\n[base avg] OOF MAE:")
for m in models:
    print(f"  {m:20s}: {mean_absolute_error(y, oof_avg[m]):.3f}")
print("\n" + "="*70)
print("[STEP 3] Robust stacking (Huber) on base preds + baseline")
print("="*70)
Z_tr = np.column_stack([oof_avg[m] for m in models] + [baseline_tr])
Z_te = np.column_stack([test_avg[m] for m in models] + [baseline_te])
oof_stack, test_stack, meta_info = robust_stack_huber(y, Z_tr, Z_te, strat_labels
print("[stack] OOF quantiles:", qdict(oof_stack))
print("\n" + "="*70)

print("[STEP 4] Shrunk chronic shifts (safer)")
print("="*70)
chronic_tr = train_feat["chronic_encoded"].values.astype(int)
chronic_te = test_feat["chronic_encoded"].values.astype(int)
shifts, global_med = compute_shrunk_group_shifts(
    y=y, pred_oof=oof_stack, group=chronic_tr,
    k=CFG.SHIFT_K, cap=CFG.SHIFT_CAP
)
print(f"[shift] global median residual={global_med:+.3f}")
print(f"[shift] shifts={shifts}")
oof_final = np.clip(apply_group_shifts(oof_stack, chronic_tr, shifts), 0.0, None)
test_final = np.clip(apply_group_shifts(test_stack, chronic_te, shifts), 0.0, Non
mae_pre = float(mean_absolute_error(y, oof_stack))
mae_post = float(mean_absolute_error(y, oof_final))
print(f"[OOF] stack MAE pre-shift : {mae_pre:.3f}")
print(f"[OOF] final MAE post-shift: {mae_post:.3f}")
# final clip
y_max = float(np.max(y))
test_final = np.clip(test_final, 0.0, y_max * CFG.PRED_CLIP_MULT)
print("\n" + "="*70)
print("[STEP 5] Write submission")
print("="*70)
sub = pd.DataFrame({
    ID_COL: test_df[ID_COL].values.astype(int),
    TARGET: np.round(test_final.astype(float), 2),
})[[ID_COL, TARGET]]
out_path = Path(OUT_SUB_PATH)
out_path.parent.mkdir(parents=True, exist_ok=True)
sub.to_csv(out_path, index=False)
print("Saved:", str(out_path))
print("submission stats:", sub[TARGET].describe().to_string())
print("pred quantiles:", qdict(sub[TARGET].values))

你现在重新跑就不会再出现 eval_sample_weight  的 TypeError 了。
如果你跑完发现 LB 还没降（甚至略升），最常见原因是 importance weights 太激进。到时
候你把 [adv] AUC  和 weight stats 发我，我会直接告诉你该把 ADV_CLIP / ADV_POWER /
SHIFT_K / SHIFT_CAP  调成哪组更稳。
print("\nAfter you submit, paste back:")
print("  (1) LB MAE")
print("  (2) [adv] AUC + weight stats")
print("  (3) base avg OOF MAEs")
print("  (4) stack MAE pre/post + shifts")
```

</details>


## 6) Reproducible workflow skeleton (derived from the artifacts)

This is a *checklist-style* skeleton you (or another AI) can use to produce a single consolidated, reproducible pipeline later.


1. **Data load & invariants**

   - Read: `ed_cost_train.csv`, `ed_cost_test.csv`, `patients.csv`, `admissions_train.csv`, `admissions_test.csv`, `receipts_parsed.joblib`.

   - Enforce `patient_id` typing and merges.

   - Receipts sanity: confirm per-line amount column; ensure per-patient totals are not accidentally treated as per-line amounts.


2. **Feature engineering (Iter7-core)**

   - Low-dim receipts: EM/critical/observation/high-acuity buckets; HHI; counts; pct shares; `cost_per_em`.

   - Patient encodings: age, sex, insurance encoding, zip3→zip_region.

   - Admissions aggregates: charlson max/mean + emergent rate.

   - Stable transforms: log/sqrt/cap for prior costs; baseline anchor `baseline_next3y = prior_cost * 0.6`.


3. **Model training**

   - Multi-seed × stratified CV folds.

   - Base models (examples): RMSE(full), RMSE(pruned), MAE(pruned) or LOGRMSE(pruned).

   - Strong regularization (depth 4–5, RSM~0.8, min_data_in_leaf tuned).


4. **Ensembling / stacking**

   - Prefer stability-aware objectives (mean + std penalty across seeds).

   - Option A: grid weights + small baseline blend (`lam`).

   - Option B: robust meta-model (Huber) on OOF predictions + baseline.


5. **Calibration / residual shifts (use cautiously)**

   - Global median shift or per-chronic median residual shift.

   - Prefer **hierarchical shrink + cap** to avoid overfitting LB.


6. **Distribution shift handling**

   - Adversarial validation (train-vs-test classifier) → importance weights.

   - Apply weights consistently in training & validation (via CatBoost `Pool(weight=...)`).


7. **Output & logging**

   - Write `submission.csv` with columns `(patient_id, ed_cost_next3y_usd)`.

   - Always log: per-seed OOF MAE, ensemble meta (weights/lam/shift), prediction quantiles, and any calibration parameters.

