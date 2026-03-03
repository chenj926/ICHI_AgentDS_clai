# AgentDS Healthcare Challenge 2 — 迭代优化与突破（Prompt + 迭代信息整理）
## 来源
- 文件：`IEEE ICHI - 迭代优化与突破.pdf`
- 目的：把本次对话里**用户 prompts（含贴的代码）**与**迭代信息/改动点/建议路线**整理成可直接喂给更强 AI 的 Markdown，方便后续总结出可复现流程。

## 背景与目标（从对话内容抽取）
- 你在复盘 v8 notebook：**Iteration 2–6 长期卡在 ~450 MAE**，只有一次突破到 **MAE≈449.4152（CODE 18 / Code17++）**。
- 你希望解释：
  - 为什么 Iter2–6 卡住？
  - CODE18 做对了什么才破 450？
  - 下一步如何把 449.x 压到 448.x？

## 1) 用户 Prompts（逐条原文抽取）

### Prompt 1（长 prompt，包含附带的 CODE18 全量/片段）
<details>
<summary>展开查看 Prompt 1 原文</summary>

```text
你看到attach的v8的code, 为什么除了iteration 1 很差之外, 
剩下的iteration 2-6都被卡在450~, 我唯一突破的code (mae 
449.4152), 做到了什么能让我们突破:
# === CODE 18 / "Code17++" (built directly on your ~451 
Iter7 / v3-Iter15/16 spirit) ===
# Less is more: keep LOW-DIM + shallow CatBoost + 
strong regularization + multi-seed bagging + STABLE 
ensemble.
# Changes vs your Iter7:
#   (1) Receipts: add ONLY a few robust "price level" + 
"concentration" features (median unit_price, top1/top3 
share, gini, max_line_total)
#   (2) Models: still 3 models, but explicitly add 
subsample(0.8)+Bernoulli to match code16 anti-overfit; 
slightly stronger L2/min_leaf
#   (3) Aggregation: use trimmed-mean across seeds (drop 
min/max) for robustness (often helps LB)
#   (4) Test preds: optional light "full-train per seed" blend 
(kept small, still fast)
# Output: D:\AgentDs\agent_ds_healthcare\submission.csv
import os, re, sys, gc, math, warnings, random, zlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
pd.set_option("display.max_columns", 250)
pd.set_option("display.width", 200)
SEED = 42
Challenge2_baseline_ichi_…
File
ChatGPT 5.2 Thinking

random.seed(SEED)
np.random.seed(SEED)
# -----------------------------
# Paths (must match prompt)
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
RECEIPTS_PDF_DIR = 
r"D:\AgentDs\agent_ds_healthcare\receipts_pdf"  # last 
resort only (we will NOT parse)
OUT_SUB_PATH = 
r"D:\AgentDs\agent_ds_healthcare\submission.csv"
ID_COL = "patient_id"
TARGET = "ed_cost_next3y_usd"
print("="*95)
print("CODE 18 | v3/code16 spirit: LOW-DIM receipts + 
shallow CatBoost + strong regularization + pruning + 
multi-seed + STABLE ensemble")
print("Goal: push LB down from ~451 by reducing 
generalization gap (NO over-engineering).")
print("="*95)
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
# Config (keep fast; code16-like regularization)
# -----------------------------
class CFG:
    N_FOLDS = 7
    N_SEEDS = 5
    ITERS = 3200
    ES_ROUNDS = 130
    LR = 0.03
    RSM = 0.80
    SUBSAMPLE = 0.80
    # seed-robust aggregation
    TRIM_K = 1  # with 5 seeds -> drop min/max and 
average middle 3
    # ensemble search

    W_STEP = 0.05
    LAM_GRID = [0.00, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]  # 
allow a bit more baseline blend
    SHIFT_GRID = [0.0, 0.5, 1.0]
    # stability objective (LB-oriented)
    STD_PEN = 0.20
    LAM_PEN = 4.0
    SHIFT_PEN = 0.001
    # optional small full-train-per-seed blend (often helps a 
bit; still cheap)
    USE_FULLFIT_BLEND = True
    FULLFIT_BLEND_W = 0.35  # final test_pred = (1-
w)*foldbag + w*fullfit
# -----------------------------
# Utilities
# -----------------------------
def must_exist(path: str, name: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {name}: {path}")
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
def safe_num_series(s: pd.Series, default=0.0) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce")
    v = v.replace([np.inf, -np.inf], np.nan).fillna(default)
    return v
def stable_hash(s: str) -> int:
    return int(zlib.crc32(s.encode("utf-8")) & 0xffffffff)
def gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return 0.0
    if np.all(x == 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    cumx = np.cumsum(x)
    return float((n + 1 - 2*np.sum(cumx)/cumx[-1]) / n)
def trimmed_mean(mat: np.ndarray, trim_k: int = 1) -> 
np.ndarray:
    """
    mat: (n_seeds, n_samples)
    if n_seeds >= 2*trim_k + 1, drop extremes along axis=0 
then mean.
    """
    mat = np.asarray(mat, dtype=float)
    if mat.ndim != 2:
        raise ValueError("trimmed_mean expects 2D array")

    n = mat.shape[0]
    if trim_k <= 0 or n < 2*trim_k + 1:
        return np.mean(mat, axis=0)
    s = np.sort(mat, axis=0)
    return np.mean(s[trim_k:n-trim_k, :], axis=0)
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
    for c in 

["charlson_max","charlson_mean","pct_emergent"]:
        out[c] = safe_num_series(out[c], default=0.0)
    out["patient_id"] = pd.to_numeric(out["patient_id"], 
errors="coerce").astype("Int64")
    out = out.dropna(subset=["patient_id"]).copy()
    out["patient_id"] = out["patient_id"].astype(int)
    return out
# -----------------------------
# Low-dim receipts features from parsed lineitems (v3-
style, + tiny robust add-ons)
# -----------------------------
def build_receipt_features_from_lineitems(li: pd.DataFrame) 
-> pd.DataFrame:
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
m_items","item_total","extended_price","sum_line_total"]:
        if c in li.columns:
            total_col = c
            break
    unit_col = None
    for c in ["unit_price","unitprice","unit_cost","unitcost"]:
        if c in li.columns:
            unit_col = c
            break
    qty_col = None
    for c in ["qty","quantity","units"]:

        if c in li.columns:
            qty_col = c
            break
    if "patient_id" not in li.columns or code_col is None or 
total_col is None:
        raise ValueError("receipts lineitems missing required 
columns (need patient_id, code, line_total-like).")
    li["patient_id"] = pd.to_numeric(li["patient_id"], 
errors="coerce").astype("Int64")
    li = li.dropna(subset=["patient_id"]).copy()
    li["patient_id"] = li["patient_id"].astype(int)
    li["code"] = li[code_col].map(norm_code)
    li = li.dropna(subset=["code"]).copy()
    li["amt"] = pd.to_numeric(li[total_col], 
errors="coerce").fillna(0.0).astype(float)
    li.loc[li["amt"] < 0, "amt"] = 0.0
    if unit_col is not None:
        li["unit_price"] = pd.to_numeric(li[unit_col], 
errors="coerce").replace([np.inf,-np.inf], np.nan)
    else:
        li["unit_price"] = np.nan
    if qty_col is not None:
        li["qty"] = pd.to_numeric(li[qty_col], 
errors="coerce").replace([np.inf,-np.inf], np.nan)
    else:
        li["qty"] = np.nan
    # totals
    receipt_total = li.groupby("patient_id")
["amt"].sum().rename("receipt_total")
    li = li.join(receipt_total, on="patient_id")
    denom = li["receipt_total"].replace(0.0, np.nan)
    share = (li["amt"] / denom).fillna(0.0)

    # concentration add-ons (tiny, robust)
    cost_hhi = (share * 
share).groupby(li["patient_id"]).sum().rename("cost_hhi")
    top1_share = 
share.groupby(li["patient_id"]).max().rename("top1_share")
    # top3 share
    def _topk_sum(s, k=3):
        a = np.sort(s.values.astype(float))[::-1]
        return float(a[:k].sum()) if a.size else 0.0
    top3_share = 
share.groupby(li["patient_id"]).apply(lambda s: 
_topk_sum(s, 3)).rename("top3_share")
    gini_amt = li.groupby("patient_id")["amt"].apply(lambda 
s: gini(s.values)).rename("gini_line_total")
    max_line_total = li.groupby("patient_id")
["amt"].max().rename("max_line_total")
    # code numeric where possible
    code_num = 
pd.to_numeric(li["code"].where(li["code"].str.fullmatch(r"\d
+"), None), errors="coerce")
    # buckets
    em_codes = ["99281","99282","99283","99284","99285"]
    is_em = li["code"].isin(em_codes)
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
    # basic counts
    n_unique_codes = li.groupby("patient_id")
["code"].nunique().rename("n_unique_codes")
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
    # flags (key codes from your EDA)
    def has_code(code: str, name: str):
        return 
(li["code"].eq(code).astype(int).groupby(li["patient_id"]).ma
x()).rename(name)
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

    # price level add-ons (tiny, robust): median unit_price 
overall + for EM/imaging/lab
    def _median_unit(mask):
        if unit_col is None:
            return None
        sub = li.loc[mask & li["unit_price"].notna(), 
["patient_id","unit_price"]]
        if sub.empty:
            return None
        return sub.groupby("patient_id")
["unit_price"].median()
    med_unit_all = 
li.loc[li["unit_price"].notna()].groupby("patient_id")
["unit_price"].median().rename("median_unit_price")
    med_unit_em = _median_unit(is_em)
    if med_unit_em is not None:
        med_unit_em = 
med_unit_em.rename("median_unit_price_em")
    med_unit_img = _median_unit(is_imaging)
    if med_unit_img is not None:
        med_unit_img = 
med_unit_img.rename("median_unit_price_imaging")
    med_unit_lab = _median_unit(is_lab)
    if med_unit_lab is not None:
        med_unit_lab = 
med_unit_lab.rename("median_unit_price_lab")
    # assemble
    out = pd.concat([
        n_unique_codes,
        receipt_total,
        cost_hhi, top1_share, top3_share, gini_amt, 
max_line_total,
        n_em_codes, max_em_level, avg_em_level, n_high_em,
        has_critical_care, has_high_acuity, has_observation, 
has_imaging,
        has_intub_31500, has_cvc_36556, has_cpr_92950, 
has_artline_36620,

        has_ct_head_70450, has_99285, has_ct_abdpel_74177, 
has_troponin_84484, has_obs_G0378,
        n_procedures, n_imaging, n_lab
    ], axis=1).reset_index()
    # merge totals
    for s in [em_total, crit_total, proc_total, img_total, 
lab_total, high_total]:
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
    out["cost_per_em"] = np.where(out["n_em_codes"] > 0, 
out["receipt_total"] / out["n_em_codes"].clip(lower=1), 
out["receipt_total"])
    # composite high acuity count (tiny)
    out["n_high_acuity_total"] = (
        out["has_intub_31500"].fillna(0).astype(int)
        + out["has_cvc_36556"].fillna(0).astype(int)
        + out["has_cpr_92950"].fillna(0).astype(int)
        + out["has_artline_36620"].fillna(0).astype(int)

        + out["has_critical_care"].fillna(0).astype(int)
    ).astype(int)
    # attach unit-price medians
    out = out.merge(med_unit_all.reset_index(), 
on="patient_id", how="left")
    if med_unit_em is not None:
        out = out.merge(med_unit_em.reset_index(), 
on="patient_id", how="left")
    else:
        out["median_unit_price_em"] = np.nan
    if med_unit_img is not None:
        out = out.merge(med_unit_img.reset_index(), 
on="patient_id", how="left")
    else:
        out["median_unit_price_imaging"] = np.nan
    if med_unit_lab is not None:
        out = out.merge(med_unit_lab.reset_index(), 
on="patient_id", how="left")
    else:
        out["median_unit_price_lab"] = np.nan
    # log transforms (very few)
    for c in 
["median_unit_price","median_unit_price_em","median_unit
_price_imaging","median_unit_price_lab","max_line_total"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out[c] = out[c].replace([np.inf,-np.inf], np.nan)
        out[c] = out[c].fillna(0.0)
        out["log1p_" + c] = np.log1p(out[c].clip(lower=0.0))
    # cleanup helper totals (drop raw totals to avoid 
duplicating prior cost)
    out.drop(columns=[c for c in 
["em_total","crit_total","proc_total","img_total","lab_total","
high_total","receipt_total"] if c in out.columns], 
inplace=True)
    # fill numeric
    for c in out.columns:

        if c == "patient_id":
            continue
        out[c] = pd.to_numeric(out[c], 
errors="coerce").replace([np.inf,-np.inf], np.nan).fillna(0.0)
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
                return 
build_receipt_features_from_lineitems(data[k])
        # else: try coerce
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
            return build_receipt_features_from_lineitems(df)
        return df
    # if list/tuple

    if isinstance(data, (list, tuple)):
        dfs = [x for x in data if isinstance(x, pd.DataFrame)]
        for df in dfs:
            if "patient_id" in df.columns and any(c in 
df.columns for c in 
["code","cpt","cpt_code","hcpcs","proc_code"]):
                return build_receipt_features_from_lineitems(df)
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
                   rcpt_df: Optional[pd.DataFrame]) -> 
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
safe_num_series(feat["prior_ed_visits_5y"], 
default=0.0).clip(lower=0.0)
    feat["prior_ed_cost_5y_usd"] = 
safe_num_series(feat["prior_ed_cost_5y_usd"], 
default=0.0).clip(lower=0.0)

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
errors="coerce").astype("Int64")
    p = p.dropna(subset=["patient_id"]).copy()
    p["patient_id"] = p["patient_id"].astype(int)
    p["age"] = pd.to_numeric(p["age"], errors="coerce")
    if p["age"].isna().any():
        p["age"] = p["age"].fillna(p["age"].median())
    p["age"] = p["age"].clip(lower=0, upper=110)
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
                feat[c] = safe_num_series(feat[c], default=0.0)
    # receipts features
    if rcpt_df is not None:
        feat = feat.merge(rcpt_df, on="patient_id", 
how="left")
        for c in rcpt_df.columns:
            if c == "patient_id": 
                continue
            feat[c] = safe_num_series(feat[c], default=np.nan)
            if feat[c].isna().any():
                feat[c] = feat[c].fillna(feat[c].median())
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
            feat[c] = feat[c].fillna(feat[c].median() if not 
feat[c].isna().all() else 0.0)
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
    # 3 models (keep "less is more")
    # explicitly add Bernoulli subsample (row sampling) + 
rsm (col sampling) -> code16-style anti-overfit
    model_specs = {
        "A_RMSE_full_d5": dict(
            loss_function="RMSE", eval_metric="MAE",
            depth=5, l2_leaf_reg=8, min_data_in_leaf=40,
            learning_rate=CFG.LR, iterations=CFG.ITERS,
            rsm=CFG.RSM, bootstrap_type="Bernoulli", 
subsample=CFG.SUBSAMPLE,
            random_strength=1.0,
        ),
        "B_RMSE_pruned_d4": dict(
            loss_function="RMSE", eval_metric="MAE",
            depth=4, l2_leaf_reg=6, min_data_in_leaf=50,
            learning_rate=CFG.LR, iterations=CFG.ITERS,
            rsm=CFG.RSM, bootstrap_type="Bernoulli", 
subsample=CFG.SUBSAMPLE,
            random_strength=2.0,
        ),
        "C_MAE_pruned_d4": dict(

            loss_function="MAE", eval_metric="MAE",
            depth=4, l2_leaf_reg=12, min_data_in_leaf=55,
            learning_rate=CFG.LR, iterations=CFG.ITERS,
            rsm=CFG.RSM, bootstrap_type="Bernoulli", 
subsample=CFG.SUBSAMPLE,
            random_strength=1.5,
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
rsm=0.8 | subsample=0.8 | multi-seed bagging")
    print("Models:", list(model_specs.keys()))
    print(f"Seeds={CFG.N_SEEDS}, Folds={CFG.N_FOLDS}\n")
    for seed_idx in range(CFG.N_SEEDS):
        rs = SEED + seed_idx * 13
        kf = StratifiedKFold(n_splits=CFG.N_FOLDS, 
shuffle=True, random_state=rs)
        oof_seed = {m: np.zeros(len(train_feat), dtype=float) 
for m in model_specs.keys()}
        test_seed_foldbag = {m: np.zeros(len(test_feat), 
dtype=float) for m in model_specs.keys()}
        best_iters_seed = {m: [] for m in model_specs.keys()}
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
                    random_seed=int(rs + fold * 31 + 
stable_hash(mname) % 997),
                    early_stopping_rounds=CFG.ES_ROUNDS,
                )
                cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0)
                try:
                    bi = int(cb.get_best_iteration())
                except Exception:
                    bi = None
                if bi is not None and bi > 0:
                    best_iters[mname].append(bi)
                    best_iters_seed[mname].append(bi)
                pred_va = cb.predict(X_va)
                pred_te = cb.predict(X_te)
                oof_seed[mname][vi] = pred_va
                test_seed_foldbag[mname] += pred_te / 
CFG.N_FOLDS
                del cb
                gc.collect()
        # optional: full-fit per seed (cheap) to use all data a 
bit (still strongly regularized)
        test_seed_final = {}
        if CFG.USE_FULLFIT_BLEND:
            for mname, params in model_specs.items():
                cols = model_featcols[mname]

                X_all = train_feat[cols]
                X_te = test_feat[cols]
                # use median best iteration for this seed/model 
(or a safe fallback)
                if best_iters_seed[mname]:
                    it_med = 
int(np.median(best_iters_seed[mname]))
                else:
                    it_med = int(CFG.ITERS * 0.45)
                it_use = int(max(300, min(CFG.ITERS, it_med + 
150)))
                params_full = dict(params)
                params_full["iterations"] = it_use  # no early 
stopping in full fit
                cb_full = CatBoostRegressor(
                    **params_full,
                    task_type="CPU",
                    thread_count=-1,
                    verbose=0,
                    allow_writing_files=False,
                    random_seed=int(rs + 999 + 
stable_hash("FULL_"+mname) % 997),
                )
                cb_full.fit(X_all, y, verbose=0)
                pred_te_full = cb_full.predict(X_te)
                del cb_full
                gc.collect()
                test_seed_final[mname] = (1.0 - 
CFG.FULLFIT_BLEND_W) * test_seed_foldbag[mname] + 
CFG.FULLFIT_BLEND_W * pred_te_full
        else:
            test_seed_final = test_seed_foldbag
        # per-seed MAE
        seed_maes = {m: float(mean_absolute_error(y, 
oof_seed[m])) for m in model_specs.keys()}
        print(f"  Seed {seed_idx+1}/{CFG.N_SEEDS} OOF MAE: 

" + " | ".join([f"{m}={seed_maes[m]:.2f}" for m in 
model_specs.keys()]))
        for m in model_specs.keys():
            oof_by_seed[m].append(oof_seed[m])
            test_by_seed[m].append(test_seed_final[m])
    print("\n[seed-aggregated OOF MAE per model] 
(trimmed mean across seeds)")
    for m in oof_by_seed.keys():
        mat = np.vstack(oof_by_seed[m])
        avg_oof = trimmed_mean(mat, trim_k=CFG.TRIM_K)
        print(f"  {m:18s}: {mean_absolute_error(y, 
avg_oof):.2f}")
    print("\n[median best_iteration per model] (reference)")
    for m in best_iters.keys():
        if best_iters[m]:
            print(f"  {m:18s}: {int(np.median(best_iters[m]))}")
        else:
            print(f"  {m:18s}: (n/a)")
    return oof_by_seed, test_by_seed, best_iters
# -----------------------------
# Ensemble selection (stability across seeds) - for 3 models
# -----------------------------
def stable_ensemble_search(train_feat: pd.DataFrame,
                           oof_by_seed: Dict[str, List[np.ndarray]],
                           baseline_vec: np.ndarray) -> 
Tuple[np.ndarray, Dict]:
    y = train_feat[TARGET].values.astype(float)
    model_names = list(oof_by_seed.keys())
    assert len(model_names) == 3, "This search expects 
exactly 3 models."
    # trimmed mean OOF per model
    oof_agg = {m: 
trimmed_mean(np.vstack(oof_by_seed[m]), 
trim_k=CFG.TRIM_K) for m in model_names}

    step = CFG.W_STEP
    grid = np.round(np.arange(0.0, 1.0 + 1e-9, step), 10)
    best = None
    top_list = []
    def eval_combo(wA, wB, wC, lam, shift_mult):
        maes = []
        # shift derived from aggregated OOF (NOT per seed) 
-> reduces overfit
        pred_avg = wA*oof_agg[model_names[0]] + 
wB*oof_agg[model_names[1]] + 
wC*oof_agg[model_names[2]]
        pred_avg = (1.0-lam)*pred_avg + lam*baseline_vec
        shift = float(np.median(y - pred_avg)) * shift_mult
        for s in range(CFG.N_SEEDS):
            pred = wA*oof_by_seed[model_names[0]][s] + 
wB*oof_by_seed[model_names[1]][s] + 
wC*oof_by_seed[model_names[2]][s]
            pred = (1.0-lam)*pred + lam*baseline_vec
            pred = pred + shift
            maes.append(float(mean_absolute_error(y, pred)))
        mean_m = float(np.mean(maes))
        std_m = float(np.std(maes, ddof=0))
        obj = mean_m + CFG.STD_PEN*std_m + 
CFG.LAM_PEN*lam + CFG.SHIFT_PEN*abs(shift_mult)
        return obj, mean_m, std_m, shift
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
                    obj, mean_m, std_m, shift = eval_combo(wA, 
wB, wC, lam, sm)
                    rec = (obj, mean_m, std_m, wA, wB, wC, lam, 
sm, shift)
                    top_list.append(rec)
                    if best is None or obj < best[0]:
                        best = rec
    top_list.sort(key=lambda x: x[0])
    print("\n[ensemble search] Top candidates (robust 
objective = mean + std_pen + simplicity_pen):")
    for i, rec in enumerate(top_list[:10], 1):
        obj, mean_m, std_m, wA, wB, wC, lam, sm, shift = rec
        print(f"  #{i:02d} obj={obj:.3f} meanMAE={mean_m:.3f} 
std={std_m:.3f} | w=({wA:.2f},{wB:.2f},{wC:.2f}) | lam=
{lam:.2f} | shift_mult={sm:.1f} | shift={shift:.2f}")
    obj, mean_m, std_m, wA, wB, wC, lam, sm, shift = best
    mA, mB, mC = model_names
    oof_final = wA*oof_agg[mA] + wB*oof_agg[mB] + 
wC*oof_agg[mC]
    oof_final = (1.0-lam)*oof_final + lam*baseline_vec
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
    return oof_final, meta
# -----------------------------
# Optional small group shift (very conservative)
# -----------------------------
def apply_chronic_group_shift(train_feat: pd.DataFrame, 

pred_oof: np.ndarray, shrink: float = 0.25) -> 
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
# Main
# -----------------------------
must_exist(TRAIN_PATH, "TRAIN")
must_exist(TEST_PATH, "TEST")
must_exist(PATIENTS_PATH, "patients")
must_exist(ADM_TRAIN_PATH, "admissions_train")
must_exist(ADM_TEST_PATH, "admissions_test")
if not os.path.exists(RECEIPTS_JOBLIB_PATH):
    print("[warn] receipts_parsed.joblib missing -> receipts 
features will be empty (likely worse).")
print("\n[load] reading CSVs...")
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
patients = pd.read_csv(PATIENTS_PATH)
adm_tr = pd.read_csv(ADM_TRAIN_PATH)
adm_te = pd.read_csv(ADM_TEST_PATH)
log_shape("ed_cost_train", train)
log_shape("ed_cost_test", test)
log_shape("patients", patients)
print("\n[target stats]")
print(train[TARGET].describe().to_string())

# ids
for df in [train, test, patients]:
    df[ID_COL] = pd.to_numeric(df[ID_COL], 
errors="coerce").astype(int)
# admissions
print("\n[admissions] building robust aggregates...")
adm_df = load_admissions_features(ADM_TRAIN_PATH, 
ADM_TEST_PATH)
if adm_df is None:
    print("  admissions features: None")
else:
    print(f"  admissions features: {adm_df.shape} | cols=
{list(adm_df.columns)}")
# receipts
print("\n[receipts] loading receipts_parsed.joblib and 
building low-dim receipt features...")
rcpt_df = None
if os.path.exists(RECEIPTS_JOBLIB_PATH):
    try:
        rcpt_df = load_receipts_joblib(RECEIPTS_JOBLIB_PATH)
        if rcpt_df is not None:
            rcpt_df["patient_id"] = 
pd.to_numeric(rcpt_df["patient_id"], 
errors="coerce").astype("Int64")
            rcpt_df = rcpt_df.dropna(subset=
["patient_id"]).copy()
            rcpt_df["patient_id"] = 
rcpt_df["patient_id"].astype(int)
            rcpt_df = rcpt_df.drop_duplicates("patient_id", 
keep="last")
            print(f"  receipt_feat shape: {rcpt_df.shape}")
            print(f"  receipt_feat cols ({len(rcpt_df.columns)-1}): 
{[c for c in rcpt_df.columns if c!='patient_id']}")
        else:
            print("  [warn] could not build receipt features from 
joblib structure.")
    except Exception as e:
        print(f"  [warn] receipts joblib load/build failed: {e}")

        rcpt_df = None
else:
    print("  [warn] receipts joblib missing; skipping receipts 
features.")
# build features
print("\n[features] building train/test feature frames...")
train_feat = build_features(train, patients, adm_df, rcpt_df)
test_feat  = build_features(test,  patients, adm_df, rcpt_df)
# choose features
feat_full = get_numeric_feature_cols(train_feat)
feat_full = [c for c in feat_full if c != TARGET]
feat_full = drop_constants(feat_full, train_feat)
# PRUNED set: stable low-dim list (extend your iter7 list 
with ONLY the new robust features)
pruned_candidates = [
    # priors + transforms
    
"prior_ed_visits_5y","prior_ed_cost_5y_usd","prior_cost_cap
20k","sqrt_prior_cost","log_prior_cost","log_prior_cost_cap2
0k","cost_per_visit","log_visits",
    "baseline_next3y",
    # demographics
    
"chronic_encoded","age","sex_encoded","insurance_encod
ed","zip_region","ins_x_chronic",
    # admissions
    "charlson_max","charlson_mean","pct_emergent",
    # receipt robust (old)
    
"cost_per_em","cost_hhi","pct_cost_em","pct_cost_procedur
e","pct_cost_critical","pct_cost_high_acuity","pct_cost_imagi
ng","pct_cost_lab",
    
"n_high_acuity_total","has_critical_care","has_99285","max_
em_level","avg_em_level","n_high_em","n_unique_codes",
    
"top1_share","top3_share","gini_line_total","max_line_total",

    
"median_unit_price","median_unit_price_em","median_unit
_price_imaging","median_unit_price_lab",
    
"log1p_median_unit_price","log1p_median_unit_price_em",
"log1p_median_unit_price_imaging","log1p_median_unit_p
rice_lab","log1p_max_line_total",
    # light interactions
    "logprior_x_pctcritical","logprior_x_highacu",
    # stable ratio
    "cost_per_code",
]
feat_pruned = [c for c in pruned_candidates if c in 
train_feat.columns]
feat_pruned = drop_constants(feat_pruned, train_feat)
print(f"  FULL feature count:   {len(feat_full)}")
print(f"  PRUNED feature count: {len(feat_pruned)}")
print(f"  PRUNED features: {feat_pruned}")
# safety fill
for c in set(feat_full + feat_pruned):
    med = train_feat[c].median() if c in train_feat.columns 
and not train_feat[c].isna().all() else 0.0
    train_feat[c] = pd.to_numeric(train_feat[c], 
errors="coerce").replace([np.inf,-np.inf], np.nan).fillna(med)
    test_feat[c]  = pd.to_numeric(test_feat[c], 
errors="coerce").replace([np.inf,-np.inf], np.nan).fillna(med)
# train
oof_by_seed, test_by_seed, best_iters = 
train_models(train_feat, test_feat, feat_full, feat_pruned)
# baseline vectors for blending
baseline_oof  = 
train_feat["baseline_next3y"].values.astype(float)
baseline_test = 
test_feat["baseline_next3y"].values.astype(float)
# stable ensemble search on OOF

oof_ens, meta = stable_ensemble_search(train_feat, 
oof_by_seed, baseline_oof)
# build final test ensemble using chosen weights + 
baseline blend + shift
mA, mB, mC = meta["models_order"]
wA, wB, wC = meta["weights"]
lam = meta["lam_baseline"]
shift = meta["shift_value"]
test_agg = {m: trimmed_mean(np.vstack(test_by_seed[m]), 
trim_k=CFG.TRIM_K) for m in meta["models_order"]}
test_ens = wA*test_agg[mA] + wB*test_agg[mB] + 
wC*test_agg[mC]
test_ens = (1.0-lam)*test_ens + lam*baseline_test
test_ens = test_ens + shift
# optional chronic shift (very conservative; require 
noticeable OOF gain)
y = train_feat[TARGET].values.astype(float)
base_mae = float(mean_absolute_error(y, oof_ens))
best_oof = oof_ens
best_shift = {"type": "none"}
for shrink in [0.0, 0.20, 0.30]:
    if shrink <= 0:
        continue
    oof2, shifts = apply_chronic_group_shift(train_feat, 
oof_ens, shrink=shrink)
    m2 = float(mean_absolute_error(y, oof2))
    if m2 < base_mae - 0.12:
        base_mae = m2
        best_oof = oof2
        best_shift = {"type": "chronic_group", "shrink": shrink, 
"shifts": shifts}
if best_shift["type"] == "chronic_group":
    test_chronic = 
test_feat["primary_chronic"].astype(str).values
    for g, s in best_shift["shifts"].items():

        test_ens[test_chronic == g] += s
# clip predictions to a reasonable range (LB-safe)
y_max = float(np.max(y))
test_ens = np.clip(test_ens, 0.0, y_max * 1.5)
# feature importance from a full-fit Model A (quick insight)
print("\n[full-train] fitting Model A on full train for feature 
importance...")
A_params = dict(
    loss_function="RMSE", eval_metric="MAE",
    depth=5, l2_leaf_reg=8, min_data_in_leaf=40,
    learning_rate=CFG.LR, iterations=CFG.ITERS,
    rsm=CFG.RSM, bootstrap_type="Bernoulli", 
subsample=CFG.SUBSAMPLE,
    random_strength=1.0,
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
print("\n" + "="*75)
print("[FINAL OOF]")
print(f"  ensemble OOF MAE (stable search + optional 
chronic shift): {final_oof_mae:.3f}")

print("  ensemble meta:", meta)
print("  extra shift:", best_shift["type"], 
("shrink="+str(best_shift.get("shrink")) if 
best_shift["type"]!="none" else ""))
print("  OOF pred quantiles:", qdict(best_oof, qs=
(0,0.01,0.05,0.1,0.5,0.9,0.95,0.99,1.0)))
print("="*75)
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
我们改怎么做继续提升, 突破449, 448呢?
```
</details>

### Prompt 2（追加请求）
```text
直接给我code
```

## 2) 迭代信息与过程解析（按对话中给出的“迭代/代码”组织）

### 2.1 为什么 v8 的 Iteration 2–6 会卡在 ~450（对话内解释要点）
- **共同点**：仍然围绕同一套核心信号（prior + admissions + receipts）+ CatBoost 做“后处理/微调”（shift / log / calibration / weights / tweedie），在小样本噪声 + test 分布下收益被抵消。
- **Iter2：Hierarchical Shift Optimization**
  - 仍是 Iter7 特征+模型，只是把 chronic shift 交给优化器自动找。
  - 风险：shift 容易在 train 内吃噪声，且 OOF 可能被 strat/shift 设计间接污染（分组样本少时尤甚），LB 不稳。
- **Iter3：Hybrid-Log ensemble + shift**
  - 加 log-target 模型追多样性。
  - 风险：MAE 最优点对应条件中位数；log/exp 回来会引入系统偏差，靠 shift 修补会变成“补来补去”。
- **Iter4：Tail + cross-fit group calibration**
  - cross-fit 比直接 shift 更不易泄漏，但分组太细会高方差（组样本太少），分组太粗又无增益。
- **Iter5：AdvWeights + stack + shrunk shifts**
  - adversarial weights 常见两种失败：
    1) train/test drift 不大 → 权重只是放大噪声；
    2) drift 存在但与目标关系不一致 → 反而伤 MAE。
- **Iter6：Tweedie**
  - target 并非 0-inflated（最小值 ~306），Tweedie 的优势发挥不出且 loss mismatch，变差是预期内现象。

### 2.2 CODE18（≈449.4152）为什么能破 450（对话内归因）
- 关键不是“神奇 feature”，而是**显著降低泛化误差的方差**：
  1) **低维 + 强正则 + 浅树**：receipts 只加“价格水平 + 集中度”的稳健统计（median unit_price、top1/top3 share、gini、max_line_total）。
  2) **显式子采样（subsample + rsm）**：近似结构化 bagging，降低小样本强相关特征下的过拟合。
  3) **多 seed + trimmed mean**：去掉极端 seed，压掉“偶然学歪的一把”。
  4) **ensemble 搜索目标惩罚跨 seed 的 std**：追求 mean + λ·std + 简单性惩罚，更贴近 leaderboard。
  5) **baseline（prior_cost×3/5）允许少量混入但被惩罚**：做 bias–variance 折中。
- 总结句：**砍掉“OOF 看起来强但 test 崩”的方差部分**，所以能稳定破 450。

### 2.3 继续从 449 → 448 的建议路线（对话内给出的 3 个优先改动）
- 目标：在“稳定派”的框架内，尽量不升维、少引入高自由度后处理。
- **A1：加入 per-code mix 低维强信息（强推）**
  - 由于 receipts 的 code universe 约 18 个左右，适合做稠密低维特征：
    - `pct_cost_{code}`：该 code 总金额 / receipt_total
    - `cnt_{code}`：该 code 出现次数
    - （可选）`log1p_cost_{code}`
  - 实现方式：在 `build_receipt_features_from_lineitems()` 里，清洗后 groupby(patient_id, code) 聚合并 pivot。
- **B1：按不确定性动态收缩到 baseline（预计可再掉 0.3–0.8）**
  - 你已有不确定性来源：跨 seed 的预测分歧 / 不同模型之间分歧。
  - 做法：
    1) 计算 `std_pred_i`（例如 3 models × N_SEEDS 的 OOF 堆叠后的 std）
    2) 映射到 `alpha_i`（越不确定，alpha 越大）
    3) `pred_i = (1-alpha_i)*pred_i + alpha_i*baseline_i`
  - 只需很小的网格：
    - `alpha_max ∈ {0.15,0.25,0.35,0.45}`
    - `q_low ∈ {0.5,0.6,0.7}`
    - `q_high ∈ {0.9,0.95,0.975}`
- **A2：把 group shift 换成经验贝叶斯收缩（EB shrink）**
  - 用组样本量自动决定 shift 强度：
    - `shift_g = median_resid_g * n_g / (n_g + k)`
  - 样本多 → shift 接近真实偏差；样本少 → shift 自动缩回 0。
- 建议实验顺序（对话内）：
  1) 先上 A1（其它不动）
  2) 再叠加 B1
  3) 最后需要时才叠加 A2（且只在粗分组上）

## 3) 生成的补丁代码（CODE19 / Code18+++）

### 3.1 CODE19 的核心意图（对话内描述）
- 在 CODE18 框架上加入：
  - per-code mix（share/cnt）
  - uncertainty shrink（动态 baseline 收缩）
  - EB group shift（组残差收缩）

### 3.2 CODE19 原文（对话中贴出的版本）
<details>
<summary>展开查看 CODE19 全量文本</summary>

```text
Thought for 4m 7s
python
# === CODE 19 / "Code18+++": per-code mix features + uncertainty shrink + EB grou
# Goal: push LB below 449 by adding low-dim orthogonal receipt signal + variance 
# Output: D:\AgentDs\agent_ds_healthcare\submission.csv
import os, re, sys, gc, math, warnings, random, zlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
pd.set_option("display.max_columns", 300)
pd.set_option("display.width", 220)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
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
Copy code

ID_COL = "patient_id"
TARGET = "ed_cost_next3y_usd"
# Known receipt code universe observed in your logs (keep low-dim & stable)
KNOWN_CODES = [
    "31500","36556","36620","70450","71045","74177","84484","85025","87070","9295
    "99281","99282","99283","99284","99285","99291","99292","G0378"
]
OTHER_CODE = "OTHER"
print("="*95)
print("CODE 19 | Code18+++ : +per-code mix (share/cnt) + uncertainty shrink + EB 
print("="*95)
# -----------------------------
# Minimal deps
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
# Config (keep stable / regularized)
# -----------------------------
class CFG:
    N_FOLDS = 7
    N_SEEDS = 5
    ITERS = 3200
    ES_ROUNDS = 130
    LR = 0.03
    RSM = 0.80
    SUBSAMPLE = 0.80
    TRIM_K = 1  # with 5 seeds -> drop min/max
    # ensemble search
    W_STEP = 0.05
    LAM_GRID = [0.00, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
    SHIFT_GRID = [0.0, 0.5, 1.0]
    STD_PEN = 0.20
    LAM_PEN = 4.0
    SHIFT_PEN = 0.001
    USE_FULLFIT_BLEND = True
    FULLFIT_BLEND_W = 0.35
    # uncertainty shrink search (dynamic baseline pull for high-uncertainty sampl
    UNC_ALPHA_MAX_GRID = [0.15, 0.25, 0.35, 0.45]
    UNC_QLOW_GRID = [0.50, 0.60, 0.70]
    UNC_QHIGH_GRID = [0.90, 0.95, 0.975]
    UNC_ALPHA_PEN = 0.03  # small simplicity penalty
    # EB group shift
    EB_K_GRID = [20, 40, 60, 80, 120]

# -----------------------------
# Utilities
# -----------------------------
def must_exist(path: str, name: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {name}: {path}")
def log_shape(name: str, df: pd.DataFrame):
    mem = df.memory_usage(deep=True).sum() / (1024**2)
    print(f"[{name}] shape={df.shape} | cols={df.shape[1]} | mem={mem:.2f} MB")
def qdict(x, qs=(0,0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99,1.0)):
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
def safe_num_series(s: pd.Series, default=0.0) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce")
    v = v.replace([np.inf, -np.inf], np.nan).fillna(default)
    return v

def stable_hash(s: str) -> int:
    return int(zlib.crc32(s.encode("utf-8")) & 0xffffffff)
def gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return 0.0
    if np.all(x == 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    cumx = np.cumsum(x)
    return float((n + 1 - 2*np.sum(cumx)/cumx[-1]) / n)
def trimmed_mean(mat: np.ndarray, trim_k: int = 1) -> np.ndarray:
    mat = np.asarray(mat, dtype=float)
    if mat.ndim != 2:
        raise ValueError("trimmed_mean expects 2D array")
    n = mat.shape[0]
    if trim_k <= 0 or n < 2*trim_k + 1:
        return np.mean(mat, axis=0)
    s = np.sort(mat, axis=0)
    return np.mean(s[trim_k:n-trim_k, :], axis=0)
# -----------------------------
# Admissions features
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
    need = ["patient_id","charlson_band","acuity_emergent"]
    if not all(c in adm.columns for c in need):
        return None
    adm["patient_id"] = pd.to_numeric(adm["patient_id"], errors="coerce")
    adm["charlson_band"] = pd.to_numeric(adm["charlson_band"], errors="coerce")
    adm["acuity_emergent"] = pd.to_numeric(adm["acuity_emergent"], errors="coerce
    out = adm.groupby("patient_id").agg(
        charlson_max=("charlson_band","max"),
        charlson_mean=("charlson_band","mean"),
        pct_emergent=("acuity_emergent","mean"),
    ).reset_index()
    for c in ["charlson_max","charlson_mean","pct_emergent"]:
        out[c] = safe_num_series(out[c], default=0.0)
    out["patient_id"] = pd.to_numeric(out["patient_id"], errors="coerce").astype(
    out = out.dropna(subset=["patient_id"]).copy()
    out["patient_id"] = out["patient_id"].astype(int)
    return out
# -----------------------------
# Receipts features: low-dim + per-code share/cnt (A1)
# -----------------------------
def build_receipt_features_from_lineitems(li: pd.DataFrame) -> pd.DataFrame:
    li = li.copy()
    # locate columns
    code_col = None
    for c in ["code","cpt","cpt_code","hcpcs","proc_code"]:
        if c in li.columns:
            code_col = c
            break
    total_col = None
    for c in ["line_total","line_total_usd","total","amount","line_cost","sum_ite

        if c in li.columns:
            total_col = c
            break
    unit_col = None
    for c in ["unit_price","unitprice","unit_cost","unitcost"]:
        if c in li.columns:
            unit_col = c
            break
    qty_col = None
    for c in ["qty","quantity","units"]:
        if c in li.columns:
            qty_col = c
            break
    if "patient_id" not in li.columns or code_col is None or total_col is None:
        raise ValueError("receipts lineitems missing required columns (need patie
    li["patient_id"] = pd.to_numeric(li["patient_id"], errors="coerce").astype("I
    li = li.dropna(subset=["patient_id"]).copy()
    li["patient_id"] = li["patient_id"].astype(int)
    li["code"] = li[code_col].map(norm_code)
    li = li.dropna(subset=["code"]).copy()
    li["amt"] = pd.to_numeric(li[total_col], errors="coerce").fillna(0.0).astype(
    li.loc[li["amt"] < 0, "amt"] = 0.0
    if unit_col is not None:
        li["unit_price"] = pd.to_numeric(li[unit_col], errors="coerce").replace([
    else:
        li["unit_price"] = np.nan
    if qty_col is not None:
        li["qty"] = pd.to_numeric(li[qty_col], errors="coerce").replace([np.inf,-
    else:
        li["qty"] = np.nan
    # totals

    receipt_total = li.groupby("patient_id")["amt"].sum().rename("receipt_total")
    li = li.join(receipt_total, on="patient_id")
    denom = li["receipt_total"].replace(0.0, np.nan)
    share = (li["amt"] / denom).fillna(0.0)
    # concentration add-ons
    cost_hhi = (share * share).groupby(li["patient_id"]).sum().rename("cost_hhi")
    top1_share = share.groupby(li["patient_id"]).max().rename("top1_share")
    def _topk_sum(s, k=3):
        a = np.sort(s.values.astype(float))[::-1]
        return float(a[:k].sum()) if a.size else 0.0
    top3_share = share.groupby(li["patient_id"]).apply(lambda s: _topk_sum(s, 3))
    gini_amt = li.groupby("patient_id")["amt"].apply(lambda s: gini(s.values)).re
    max_line_total = li.groupby("patient_id")["amt"].max().rename("max_line_total
    # code numeric where possible
    code_num = pd.to_numeric(li["code"].where(li["code"].str.fullmatch(r"\d+"), N
    # buckets
    em_codes = ["99281","99282","99283","99284","99285"]
    is_em = li["code"].isin(em_codes)
    em_map = {"99281":1,"99282":2,"99283":3,"99284":4,"99285":5}
    em_level = li["code"].map(em_map).fillna(0).astype(float)
    is_crit = li["code"].isin(["99291","99292"])
    is_obs = li["code"].str.startswith("G037", na=False)
    is_high = li["code"].isin(["31500","36556","32551","36620","92950"])
    is_lab = code_num.between(80000, 89999)
    is_imaging = code_num.between(70000, 79999)
    is_proc_general = code_num.between(10000, 69999)
    is_proc_any = is_high | (is_proc_general & (~is_high) & (~is_em) & (~is_crit)
    n_unique_codes = li.groupby("patient_id")["code"].nunique().rename("n_unique_
    # EM stats
    n_em_codes = is_em.astype(int).groupby(li["patient_id"]).sum().rename("n_em_c

    max_em_level = em_level.groupby(li["patient_id"]).max().rename("max_em_level"
    sum_em_level = (em_level * is_em.astype(int)).groupby(li["patient_id"]).sum()
    avg_em_level = (sum_em_level / n_em_codes.replace(0, np.nan)).fillna(0.0).ren
    n_high_em = ((em_level >= 4) & is_em).astype(int).groupby(li["patient_id"]).s
    # totals by bucket
    em_total = li.loc[is_em].groupby("patient_id")["amt"].sum().rename("em_total"
    crit_total = li.loc[is_crit].groupby("patient_id")["amt"].sum().rename("crit_t
    proc_total = li.loc[is_proc_any].groupby("patient_id")["amt"].sum().rename("p
    img_total = li.loc[is_imaging].groupby("patient_id")["amt"].sum().rename("img_
    lab_total = li.loc[is_lab].groupby("patient_id")["amt"].sum().rename("lab_tot
    high_total = li.loc[is_high].groupby("patient_id")["amt"].sum().rename("high_t
    # counts by bucket
    n_procedures = is_proc_any.astype(int).groupby(li["patient_id"]).sum().rename
    n_imaging = is_imaging.astype(int).groupby(li["patient_id"]).sum().rename("n_
    n_lab = is_lab.astype(int).groupby(li["patient_id"]).sum().rename("n_lab")
    # flags
    def has_code(code: str, name: str):
        return (li["code"].eq(code).astype(int).groupby(li["patient_id"]).max()).
    has_critical_care = is_crit.astype(int).groupby(li["patient_id"]).max().renam
    has_high_acuity = is_high.astype(int).groupby(li["patient_id"]).max().rename(
    has_observation = is_obs.astype(int).groupby(li["patient_id"]).max().rename("
    has_imaging = is_imaging.astype(int).groupby(li["patient_id"]).max().rename("
    has_intub_31500 = has_code("31500","has_intub_31500")
    has_cvc_36556 = has_code("36556","has_cvc_36556")
    has_cpr_92950 = has_code("92950","has_cpr_92950")
    has_artline_36620 = has_code("36620","has_artline_36620")
    has_ct_head_70450 = has_code("70450","has_ct_head_70450")
    has_99285 = has_code("99285","has_99285")
    has_ct_abdpel_74177 = has_code("74177","has_ct_abdpel_74177")
    has_troponin_84484 = has_code("84484","has_troponin_84484")
    has_obs_G0378 = has_code("G0378","has_obs_G0378")
    # price level add-ons: median unit_price overall + by buckets
    def _median_unit(mask):
        if unit_col is None:

            return None
        sub = li.loc[mask & li["unit_price"].notna(), ["patient_id","unit_price"]
        if sub.empty:
            return None
        return sub.groupby("patient_id")["unit_price"].median()
    med_unit_all = li.loc[li["unit_price"].notna()].groupby("patient_id")["unit_p
    med_unit_em = _median_unit(is_em)
    if med_unit_em is not None:
        med_unit_em = med_unit_em.rename("median_unit_price_em")
    med_unit_img = _median_unit(is_imaging)
    if med_unit_img is not None:
        med_unit_img = med_unit_img.rename("median_unit_price_imaging")
    med_unit_lab = _median_unit(is_lab)
    if med_unit_lab is not None:
        med_unit_lab = med_unit_lab.rename("median_unit_price_lab")
    # ---- NEW: per-code mix features (share + count) over known codes + OTHER --
    li["code_known"] = li["code"].where(li["code"].isin(KNOWN_CODES), OTHER_CODE)
    amt_by_code = (
        li.groupby(["patient_id","code_known"])["amt"].sum().unstack(fill_value=0
    )
    cnt_by_code = (
        li.groupby(["patient_id","code_known"])["amt"].size().unstack(fill_value=
    )
    # ensure all columns exist
    all_codes = KNOWN_CODES + [OTHER_CODE]
    for c in all_codes:
        if c not in amt_by_code.columns:
            amt_by_code[c] = 0.0
        if c not in cnt_by_code.columns:
            cnt_by_code[c] = 0
    amt_by_code = amt_by_code[all_codes].astype(float)
    cnt_by_code = cnt_by_code[all_codes].astype(float)  # keep numeric; cast late
    rec_total2 = amt_by_code.sum(axis=1).replace(0.0, np.nan)
    share_by_code = amt_by_code.div(rec_total2, axis=0).fillna(0.0).astype(float)

    share_by_code.columns = [f"share_code_{c}" for c in share_by_code.columns]
    cnt_by_code.columns = [f"cnt_code_{c}" for c in cnt_by_code.columns]
    # assemble (all index=patient_id)
    frames = [
        n_unique_codes,
        receipt_total,
        cost_hhi, top1_share, top3_share, gini_amt, max_line_total,
        n_em_codes, max_em_level, avg_em_level, n_high_em,
        has_critical_care, has_high_acuity, has_observation, has_imaging,
        has_intub_31500, has_cvc_36556, has_cpr_92950, has_artline_36620,
        has_ct_head_70450, has_99285, has_ct_abdpel_74177, has_troponin_84484, ha
        n_procedures, n_imaging, n_lab,
        share_by_code, cnt_by_code
    ]
    if med_unit_all is not None:
        frames.append(med_unit_all)
    if med_unit_em is not None:
        frames.append(med_unit_em)
    if med_unit_img is not None:
        frames.append(med_unit_img)
    if med_unit_lab is not None:
        frames.append(med_unit_lab)
    out = pd.concat(frames, axis=1).reset_index()
    # merge bucket totals to compute pct ratios, then drop totals again
    for s in [em_total, crit_total, proc_total, img_total, lab_total, high_total]
        out = out.merge(s.reset_index(), on="patient_id", how="left")
    for c in ["em_total","crit_total","proc_total","img_total","lab_total","high_
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    denom2 = out["receipt_total"].replace(0.0, np.nan)
    out["pct_cost_em"] = (out["em_total"] / denom2).replace([np.inf,-np.inf], np.
    out["pct_cost_procedure"] = (out["proc_total"] / denom2).replace([np.inf,-np.
    out["pct_cost_critical"] = (out["crit_total"] / denom2).replace([np.inf,-np.i
    out["pct_cost_imaging"] = (out["img_total"] / denom2).replace([np.inf,-np.inf
    out["pct_cost_lab"] = (out["lab_total"] / denom2).replace([np.inf,-np.inf], n
    out["pct_cost_high_acuity"] = (out["high_total"] / denom2).replace([np.inf,-n

    out["cost_per_em"] = np.where(out["n_em_codes"] > 0, out["receipt_total"] / o
    out["n_high_acuity_total"] = (
        out["has_intub_31500"].fillna(0).astype(int)
        + out["has_cvc_36556"].fillna(0).astype(int)
        + out["has_cpr_92950"].fillna(0).astype(int)
        + out["has_artline_36620"].fillna(0).astype(int)
        + out["has_critical_care"].fillna(0).astype(int)
    ).astype(int)
    # log transforms (few)
    for c in ["median_unit_price","median_unit_price_em","median_unit_price_imagi
        if c not in out.columns:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf,-np.inf], 
        out["log1p_" + c] = np.log1p(out[c].clip(lower=0.0))
    # drop raw totals (keep only ratios + mix + counts)
    out.drop(columns=[c for c in ["em_total","crit_total","proc_total","img_total
    # numeric fill
    for c in out.columns:
        if c == "patient_id":
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf,-np.inf], 
    return out
def load_receipts_joblib(joblib_path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(joblib_path):
        return None
    data = joblib_load(joblib_path)
    if isinstance(data, dict):
        for k in ["lineitems_df","lineitems","items_df","items","line_items_df","
            if k in data and isinstance(data[k], pd.DataFrame):
                return build_receipt_features_from_lineitems(data[k])
        try:
            df = pd.DataFrame.from_dict(data, orient="index")

            df.index.name = "patient_id"
            df = df.reset_index()
            return df
        except Exception:
            return None
    if isinstance(data, pd.DataFrame):
        df = data
        if "patient_id" in df.columns and any(c in df.columns for c in ["code","c
            return build_receipt_features_from_lineitems(df)
        return df
    if isinstance(data, (list, tuple)):
        dfs = [x for x in data if isinstance(x, pd.DataFrame)]
        for df in dfs:
            if "patient_id" in df.columns and any(c in df.columns for c in ["code
                return build_receipt_features_from_lineitems(df)
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
                   rcpt_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    feat = ed_df.copy()
    chronic_map = {"PNEUMONIA":0, "DIABETESCOMP":1, "HF":2}
    feat["primary_chronic"] = feat["primary_chronic"].astype(str)
    feat["chronic_encoded"] = feat["primary_chronic"].str.upper().map(chronic_map
    feat["prior_ed_visits_5y"] = safe_num_series(feat["prior_ed_visits_5y"], defa
    feat["prior_ed_cost_5y_usd"] = safe_num_series(feat["prior_ed_cost_5y_usd"], 

    feat["prior_cost_cap20k"] = feat["prior_ed_cost_5y_usd"].clip(upper=20000.0)
    feat["sqrt_prior_cost"] = np.sqrt(feat["prior_ed_cost_5y_usd"].clip(lower=0.0
    feat["log_prior_cost"] = np.log1p(feat["prior_ed_cost_5y_usd"].clip(lower=0.0
    feat["log_prior_cost_cap20k"] = np.log1p(feat["prior_cost_cap20k"].clip(lower
    feat["log_visits"] = np.log1p(feat["prior_ed_visits_5y"].clip(lower=0.0))
    feat["cost_per_visit"] = feat["prior_ed_cost_5y_usd"] / feat["prior_ed_visits_
    feat["baseline_next3y"] = feat["prior_ed_cost_5y_usd"] * (3.0/5.0)
    p = patients_df.copy()
    p["patient_id"] = pd.to_numeric(p["patient_id"], errors="coerce").astype("Int
    p = p.dropna(subset=["patient_id"]).copy()
    p["patient_id"] = p["patient_id"].astype(int)
    p["age"] = pd.to_numeric(p["age"], errors="coerce")
    if p["age"].isna().any():
        p["age"] = p["age"].fillna(p["age"].median())
    p["age"] = p["age"].clip(lower=0, upper=110)
    p["sex_encoded"] = (p["sex"].astype(str).str.upper() == "M").astype(int)
    ins = p["insurance"].astype(str).str.lower()
    ins_map = {"private":2, "public":1, "self_pay":0, "selfpay":0}
    p["insurance_encoded"] = ins.map(ins_map).fillna(-1).astype(float)
    z3 = p["zip3"].apply(standardize_zip3).astype("string")
    zr = z3.fillna("000").str.replace(r"\D","", regex=True).str.zfill(3).str[0]
    p["zip_region"] = pd.to_numeric(zr, errors="coerce").fillna(-1).astype(float)
    feat = feat.merge(p[["patient_id","age","sex_encoded","insurance_encoded","zi
    feat["ins_x_chronic"] = feat["insurance_encoded"].fillna(-1) * feat["chronic_
    if adm_df is not None:
        feat = feat.merge(adm_df, on="patient_id", how="left")
        for c in ["charlson_max","charlson_mean","pct_emergent"]:
            if c in feat.columns:
                feat[c] = safe_num_series(feat[c], default=0.0)
        # small bin for EB shift grouping (optional)
        if "charlson_max" in feat.columns:

            feat["charlson_bin3"] = np.clip(np.floor(feat["charlson_max"].fillna(
    else:
        feat["charlson_bin3"] = 0
    if rcpt_df is not None:
        feat = feat.merge(rcpt_df, on="patient_id", how="left")
        for c in rcpt_df.columns:
            if c == "patient_id":
                continue
            feat[c] = pd.to_numeric(feat[c], errors="coerce").replace([np.inf,-np
            if feat[c].isna().any():
                med = feat[c].median() if not feat[c].isna().all() else 0.0
                feat[c] = feat[c].fillna(med)
    if "pct_cost_critical" in feat.columns:
        feat["logprior_x_pctcritical"] = feat["log_prior_cost"] * feat["pct_cost_
    if "n_high_acuity_total" in feat.columns:
        feat["logprior_x_highacu"] = feat["log_prior_cost"] * feat["n_high_acuity_
    if "n_unique_codes" in feat.columns:
        feat["cost_per_code"] = feat["prior_ed_cost_5y_usd"] / feat["n_unique_cod
    feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    for c in feat.columns:
        if c in ["patient_id", "primary_chronic", TARGET]:
            continue
        feat[c] = pd.to_numeric(feat[c], errors="coerce")
        if feat[c].isna().any():
            feat[c] = feat[c].fillna(feat[c].median() if not feat[c].isna().all() 
    return feat
def get_numeric_feature_cols(df: pd.DataFrame) -> List[str]:
    exclude = {"patient_id","primary_chronic",TARGET,"sex","insurance","zip3"}
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
# Training
# -----------------------------
def train_models(train_feat: pd.DataFrame, test_feat: pd.DataFrame,
                 feat_full: List[str], feat_pruned: List[str]) -> Tuple[Dict[str, 
    y = train_feat[TARGET].values.astype(float)
    tmp = train_feat[["primary_chronic", TARGET]].copy()
    tmp["cost_bin"] = pd.qcut(tmp[TARGET], q=5, labels=False, duplicates="drop")
    tmp["strat"] = tmp["primary_chronic"].astype(str) + "_" + tmp["cost_bin"].asty
    strat = LabelEncoder().fit_transform(tmp["strat"].values)
    # fallback if too granular for N_FOLDS
    vc = pd.Series(strat).value_counts()
    if int(vc.min()) < CFG.N_FOLDS:
        print(f"[warn] strat too granular (min class count {int(vc.min())}). Fall
        strat = LabelEncoder().fit_transform(train_feat["primary_chronic"].astype
    model_specs = {
        "A_RMSE_full_d5": dict(
            loss_function="RMSE", eval_metric="MAE",
            depth=5, l2_leaf_reg=8, min_data_in_leaf=40,
            learning_rate=CFG.LR, iterations=CFG.ITERS,
            rsm=CFG.RSM, bootstrap_type="Bernoulli", subsample=CFG.SUBSAMPLE,
            random_strength=1.0,
        ),
        "B_RMSE_pruned_d4": dict(

            loss_function="RMSE", eval_metric="MAE",
            depth=4, l2_leaf_reg=6, min_data_in_leaf=50,
            learning_rate=CFG.LR, iterations=CFG.ITERS,
            rsm=CFG.RSM, bootstrap_type="Bernoulli", subsample=CFG.SUBSAMPLE,
            random_strength=2.0,
        ),
        "C_MAE_pruned_d4": dict(
            loss_function="MAE", eval_metric="MAE",
            depth=4, l2_leaf_reg=12, min_data_in_leaf=55,
            learning_rate=CFG.LR, iterations=CFG.ITERS,
            rsm=CFG.RSM, bootstrap_type="Bernoulli", subsample=CFG.SUBSAMPLE,
            random_strength=1.5,
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
    print("\n[training] CatBoost CPU | shallow trees | rsm=0.8 | subsample=0.8 | 
    print("Models:", list(model_specs.keys()))
    print(f"Seeds={CFG.N_SEEDS}, Folds={CFG.N_FOLDS}\n")
    for seed_idx in range(CFG.N_SEEDS):
        rs = SEED + seed_idx * 13
        kf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=rs)
        oof_seed = {m: np.zeros(len(train_feat), dtype=float) for m in model_spec
        test_seed_foldbag = {m: np.zeros(len(test_feat), dtype=float) for m in mo
        best_iters_seed = {m: [] for m in model_specs.keys()}
        for fold, (ti, vi) in enumerate(kf.split(train_feat, strat), 1):
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
                    random_seed=int(rs + fold * 31 + stable_hash(mname) % 997),
                    early_stopping_rounds=CFG.ES_ROUNDS,
                )
                cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0)
                try:
                    bi = int(cb.get_best_iteration())
                except Exception:
                    bi = None
                if bi is not None and bi > 0:
                    best_iters[mname].append(bi)
                    best_iters_seed[mname].append(bi)
                pred_va = cb.predict(X_va)
                pred_te = cb.predict(X_te)
                oof_seed[mname][vi] = pred_va
                test_seed_foldbag[mname] += pred_te / CFG.N_FOLDS
                del cb
                gc.collect()
        # optional: full-fit per seed blend
        test_seed_final = {}
        if CFG.USE_FULLFIT_BLEND:
            for mname, params in model_specs.items():
                cols = model_featcols[mname]
                X_all = train_feat[cols]
                X_te = test_feat[cols]

                if best_iters_seed[mname]:
                    it_med = int(np.median(best_iters_seed[mname]))
                else:
                    it_med = int(CFG.ITERS * 0.45)
                it_use = int(max(300, min(CFG.ITERS, it_med + 150)))
                params_full = dict(params)
                params_full["iterations"] = it_use
                cb_full = CatBoostRegressor(
                    **params_full,
                    task_type="CPU",
                    thread_count=-1,
                    verbose=0,
                    allow_writing_files=False,
                    random_seed=int(rs + 999 + stable_hash("FULL_"+mname) % 997),
                )
                cb_full.fit(X_all, y, verbose=0)
                pred_te_full = cb_full.predict(X_te)
                del cb_full
                gc.collect()
                test_seed_final[mname] = (1.0 - CFG.FULLFIT_BLEND_W) * test_seed_
        else:
            test_seed_final = test_seed_foldbag
        seed_maes = {m: float(mean_absolute_error(y, oof_seed[m])) for m in model_
        print(f"  Seed {seed_idx+1}/{CFG.N_SEEDS} OOF MAE: " + " | ".join([f"{m}=
        for m in model_specs.keys():
            oof_by_seed[m].append(oof_seed[m])
            test_by_seed[m].append(test_seed_final[m])
    print("\n[seed-aggregated OOF MAE per model] (trimmed mean across seeds)")
    for m in oof_by_seed.keys():
        mat = np.vstack(oof_by_seed[m])
        avg_oof = trimmed_mean(mat, trim_k=CFG.TRIM_K)
        print(f"  {m:18s}: {mean_absolute_error(y, avg_oof):.2f}")
    print("\n[median best_iteration per model] (reference)")

    for m in best_iters.keys():
        if best_iters[m]:
            print(f"  {m:18s}: {int(np.median(best_iters[m]))}")
        else:
            print(f"  {m:18s}: (n/a)")
    return oof_by_seed, test_by_seed, best_iters
# -----------------------------
# Ensemble selection (stability across seeds)
# -----------------------------
def stable_ensemble_search(train_feat: pd.DataFrame,
                           oof_by_seed: Dict[str, List[np.ndarray]],
                           baseline_vec: np.ndarray) -> Tuple[Dict, List[Tuple]]:
    y = train_feat[TARGET].values.astype(float)
    model_names = list(oof_by_seed.keys())
    assert len(model_names) == 3, "This search expects exactly 3 models."
    oof_agg = {m: trimmed_mean(np.vstack(oof_by_seed[m]), trim_k=CFG.TRIM_K) for 
    step = CFG.W_STEP
    grid = np.round(np.arange(0.0, 1.0 + 1e-9, step), 10)
    best = None
    top_list = []
    def eval_combo(wA, wB, wC, lam, shift_mult):
        maes = []
        pred_avg = wA*oof_agg[model_names[0]] + wB*oof_agg[model_names[1]] + wC*o
        pred_avg = (1.0-lam)*pred_avg + lam*baseline_vec
        shift = float(np.median(y - pred_avg)) * shift_mult
        for s in range(CFG.N_SEEDS):
            pred = wA*oof_by_seed[model_names[0]][s] + wB*oof_by_seed[model_names
            pred = (1.0-lam)*pred + lam*baseline_vec
            pred = pred + shift
            maes.append(float(mean_absolute_error(y, pred)))
        mean_m = float(np.mean(maes))

        std_m = float(np.std(maes, ddof=0))
        obj = mean_m + CFG.STD_PEN*std_m + CFG.LAM_PEN*lam + CFG.SHIFT_PEN*abs(sh
        return obj, mean_m, std_m, shift
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
                    obj, mean_m, std_m, shift = eval_combo(wA, wB, wC, lam, sm)
                    rec = (obj, mean_m, std_m, wA, wB, wC, lam, sm, shift)
                    top_list.append(rec)
                    if best is None or obj < best[0]:
                        best = rec
    top_list.sort(key=lambda x: x[0])
    print("\n[ensemble search] Top candidates (obj = mean + std_pen + simplicity_
    for i, rec in enumerate(top_list[:10], 1):
        obj, mean_m, std_m, wA, wB, wC, lam, sm, shift = rec
        print(f"  #{i:02d} obj={obj:.3f} meanMAE={mean_m:.3f} std={std_m:.3f} | w
    obj, mean_m, std_m, wA, wB, wC, lam, sm, shift = best
    meta = {
        "models_order": model_names,
        "weights": (float(wA), float(wB), float(wC)),
        "lam_baseline": float(lam),
        "shift_mult": float(sm),
        "shift_value": float(shift),
        "oof_mean_mae_across_seeds": float(mean_m),
        "oof_std_mae_across_seeds": float(std_m),
    }
    return meta, top_list
# -----------------------------

# Build seed-level ensemble matrices (needed for uncertainty shrink)
# -----------------------------
def build_seed_ensemble_mats(oof_by_seed, test_by_seed, meta, baseline_oof, basel
    mA, mB, mC = meta["models_order"]
    wA, wB, wC = meta["weights"]
    lam = meta["lam_baseline"]
    shift = meta["shift_value"]
    oof_mat = []
    test_mat = []
    for s in range(CFG.N_SEEDS):
        o_s = wA*oof_by_seed[mA][s] + wB*oof_by_seed[mB][s] + wC*oof_by_seed[mC][
        o_s = (1.0-lam)*o_s + lam*baseline_oof
        o_s = o_s + shift
        oof_mat.append(o_s)
        t_s = wA*test_by_seed[mA][s] + wB*test_by_seed[mB][s] + wC*test_by_seed[m
        t_s = (1.0-lam)*t_s + lam*baseline_test
        t_s = t_s + shift
        test_mat.append(t_s)
    return np.vstack(oof_mat), np.vstack(test_mat)
# -----------------------------
# Uncertainty shrink (B1)
# -----------------------------
def apply_uncertainty_shrink(pred_center, pred_std, baseline, alpha_max, ql_val, 
    denom = float(max(qh_val - ql_val, 1e-9))
    alpha = np.clip((pred_std - ql_val) / denom, 0.0, 1.0) * float(alpha_max)
    out = (1.0 - alpha) * pred_center + alpha * baseline
    return out, alpha
def search_uncertainty_shrink(y, pred_center, pred_std, baseline):
    best = None
    cands = []
    for alpha_max in CFG.UNC_ALPHA_MAX_GRID:
        for qlow in CFG.UNC_QLOW_GRID:
            for qhigh in CFG.UNC_QHIGH_GRID:
                if qhigh <= qlow + 1e-9:

                    continue
                ql_val = float(np.quantile(pred_std, qlow))
                qh_val = float(np.quantile(pred_std, qhigh))
                pred2, alpha = apply_uncertainty_shrink(pred_center, pred_std, ba
                mae = float(mean_absolute_error(y, pred2))
                obj = mae + CFG.UNC_ALPHA_PEN * float(alpha_max)
                rec = (obj, mae, float(alpha_max), float(qlow), float(qhigh), ql_
                cands.append(rec)
                if best is None or obj < best[0]:
                    best = rec
    cands.sort(key=lambda x: x[0])
    print("\n[uncertainty shrink] Top candidates (obj = MAE + alpha_pen):")
    for i, rec in enumerate(cands[:10], 1):
        obj, mae, alpha_max, qlow, qhigh, ql_val, qh_val, amean = rec
        print(f"  #{i:02d} obj={obj:.4f} mae={mae:.4f} | alpha_max={alpha_max:.2f
    obj, mae, alpha_max, qlow, qhigh, ql_val, qh_val, amean = best
    meta = dict(alpha_max=alpha_max, qlow=qlow, qhigh=qhigh, ql_val=ql_val, qh_va
    return meta
# -----------------------------
# EB group shift (A2)
# -----------------------------
def make_key(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    parts = []
    for c in cols:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            s = pd.to_numeric(s, errors="coerce").fillna(-999).round().astype(int
        else:
            s = s.astype(str)
        parts.append(s)
    key = parts[0]
    for p in parts[1:]:
        key = key.str.cat(p, sep="|")
    return key
def apply_eb_group_shift(train_feat, y, pred, group_cols, k):

    key_tr = make_key(train_feat, group_cols)
    resid = y - pred
    tmp = pd.DataFrame({"key": key_tr.values, "resid": resid})
    med = tmp.groupby("key")["resid"].median()
    n = tmp.groupby("key")["resid"].size()
    shrink = (n / (n + float(k))).astype(float)
    shift = (med * shrink).to_dict()
    pred2 = pred + key_tr.map(shift).fillna(0.0).values.astype(float)
    return pred2, shift
def apply_shift_map(test_feat, pred, shift_map, group_cols):
    key_te = make_key(test_feat, group_cols)
    return pred + key_te.map(shift_map).fillna(0.0).values.astype(float)
def search_eb_group_shift(train_feat, y, pred_oof):
    base_mae = float(mean_absolute_error(y, pred_oof))
    best = (base_mae, None, None, None, None)
    print("\n[EB group shift] searching... (only accept if improves)")
    group_sets = [
        ["primary_chronic"],
        ["primary_chronic","insurance_encoded"],
        ["primary_chronic","insurance_encoded","charlson_bin3"],
    ]
    for cols in group_sets:
        if not all(c in train_feat.columns for c in cols):
            continue
        for k in CFG.EB_K_GRID:
            pred2, shift_map = apply_eb_group_shift(train_feat, y, pred_oof, cols
            mae = float(mean_absolute_error(y, pred2))
            print(f"  cols={cols} k={k:3d} -> MAE={mae:.4f} (delta {mae-base_mae:
            if mae < best[0]:
                best = (mae, cols, k, shift_map, pred2)
    mae, cols, k, shift_map, pred2 = best
    if cols is None or mae > base_mae - 0.05:  # require meaningful improvement t
        print("  -> EB shift skipped (no robust improvement).")
        return dict(use=False, base_mae=base_mae)
    print(f"  -> EB shift SELECTED: cols={cols} k={k} | MAE {base_mae:.4f} -> {ma
    return dict(use=True, base_mae=base_mae, best_mae=mae, cols=cols, k=k, shift_

# -----------------------------
# Main
# -----------------------------
must_exist(TRAIN_PATH, "TRAIN")
must_exist(TEST_PATH, "TEST")
must_exist(PATIENTS_PATH, "patients")
must_exist(ADM_TRAIN_PATH, "admissions_train")
must_exist(ADM_TEST_PATH, "admissions_test")
if not os.path.exists(RECEIPTS_JOBLIB_PATH):
    print("[warn] receipts_parsed.joblib missing -> receipts features will be emp
print("\n[load] reading CSVs...")
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
patients = pd.read_csv(PATIENTS_PATH)
log_shape("ed_cost_train", train)
log_shape("ed_cost_test", test)
log_shape("patients", patients)
print("\n[target stats]")
print(train[TARGET].describe().to_string())
# ids
for df in [train, test, patients]:
    df[ID_COL] = pd.to_numeric(df[ID_COL], errors="coerce").astype(int)
# admissions
print("\n[admissions] building robust aggregates...")
adm_df = load_admissions_features(ADM_TRAIN_PATH, ADM_TEST_PATH)
if adm_df is None:
    print("  admissions features: None")
else:
    print(f"  admissions features: {adm_df.shape} | cols={list(adm_df.columns)}")
# receipts
print("\n[receipts] loading receipts_parsed.joblib and building low-dim receipt f
rcpt_df = None
if os.path.exists(RECEIPTS_JOBLIB_PATH):

    try:
        rcpt_df = load_receipts_joblib(RECEIPTS_JOBLIB_PATH)
        if rcpt_df is not None:
            rcpt_df["patient_id"] = pd.to_numeric(rcpt_df["patient_id"], errors="
            rcpt_df = rcpt_df.dropna(subset=["patient_id"]).copy()
            rcpt_df["patient_id"] = rcpt_df["patient_id"].astype(int)
            rcpt_df = rcpt_df.drop_duplicates("patient_id", keep="last")
            print(f"  receipt_feat shape: {rcpt_df.shape}")
            print(f"  receipt_feat cols ({len(rcpt_df.columns)-1}): {len([c for c 
        else:
            print("  [warn] could not build receipt features from joblib structur
    except Exception as e:
        print(f"  [warn] receipts joblib load/build failed: {e}")
        rcpt_df = None
else:
    print("  [warn] receipts joblib missing; skipping receipts features.")
# build features
print("\n[features] building train/test feature frames...")
train_feat = build_features(train, patients, adm_df, rcpt_df)
test_feat  = build_features(test,  patients, adm_df, rcpt_df)
# choose features
feat_full = get_numeric_feature_cols(train_feat)
feat_full = [c for c in feat_full if c != TARGET]
feat_full = drop_constants(feat_full, train_feat)
# PRUNED set: stable low-dim list + per-code mix
pruned_candidates = [
    # priors + transforms
    "prior_ed_visits_5y","prior_ed_cost_5y_usd","prior_cost_cap20k","sqrt_prior_c
    "baseline_next3y",
    # demographics
    "chronic_encoded","age","sex_encoded","insurance_encoded","zip_region","ins_x_
    # admissions
    "charlson_max","charlson_mean","pct_emergent","charlson_bin3",
    # receipt robust
    "cost_per_em","cost_hhi","pct_cost_em","pct_cost_procedure","pct_cost_critica
    "n_high_acuity_total","has_critical_care","has_99285","max_em_level","avg_em_
    "top1_share","top3_share","gini_line_total","max_line_total",

    "median_unit_price","median_unit_price_em","median_unit_price_imaging","media
    "log1p_median_unit_price","log1p_median_unit_price_em","log1p_median_unit_pri
    # light interactions
    "logprior_x_pctcritical","logprior_x_highacu",
    # stable ratio
    "cost_per_code",
]
# add per-code mix features
for c in KNOWN_CODES + [OTHER_CODE]:
    pruned_candidates.append(f"share_code_{c}")
    pruned_candidates.append(f"cnt_code_{c}")
feat_pruned = [c for c in pruned_candidates if c in train_feat.columns]
feat_pruned = drop_constants(feat_pruned, train_feat)
print(f"  FULL feature count:   {len(feat_full)}")
print(f"  PRUNED feature count: {len(feat_pruned)}")
# safety fill
for c in set(feat_full + feat_pruned):
    med = train_feat[c].median() if c in train_feat.columns and not train_feat[c]
    train_feat[c] = pd.to_numeric(train_feat[c], errors="coerce").replace([np.inf
    test_feat[c]  = pd.to_numeric(test_feat[c], errors="coerce").replace([np.inf,
# train
oof_by_seed, test_by_seed, best_iters = train_models(train_feat, test_feat, feat_
# baseline vectors for blending / shrink
baseline_oof  = train_feat["baseline_next3y"].values.astype(float)
baseline_test = test_feat["baseline_next3y"].values.astype(float)
# stable ensemble search on OOF
meta, _ = stable_ensemble_search(train_feat, oof_by_seed, baseline_oof)
# build seed-level ensemble mats (for uncertainty shrink)
oof_seed_mat, test_seed_mat = build_seed_ensemble_mats(oof_by_seed, test_by_seed, 
# central predictions (trimmed mean across seeds AFTER ensembling)
oof_center  = trimmed_mean(oof_seed_mat, trim_k=CFG.TRIM_K)

test_center = trimmed_mean(test_seed_mat, trim_k=CFG.TRIM_K)
# seed-based uncertainty
oof_std  = np.std(oof_seed_mat, axis=0, ddof=0)
test_std = np.std(test_seed_mat, axis=0, ddof=0)
y = train_feat[TARGET].values.astype(float)
base_mae = float(mean_absolute_error(y, oof_center))
print("\n" + "="*75)
print("[BASE ENSEMBLE]")
print(f"  OOF MAE (trimmed mean across seed-ensembles): {base_mae:.4f}")
print("  meta:", meta)
print("="*75)
# uncertainty shrink search + apply
unc_meta = search_uncertainty_shrink(y, oof_center, oof_std, baseline_oof)
oof_unc, alpha_oof = apply_uncertainty_shrink(
    oof_center, oof_std, baseline_oof,
    alpha_max=unc_meta["alpha_max"],
    ql_val=unc_meta["ql_val"],
    qh_val=unc_meta["qh_val"],
)
test_unc, alpha_test = apply_uncertainty_shrink(
    test_center, test_std, baseline_test,
    alpha_max=unc_meta["alpha_max"],
    ql_val=unc_meta["ql_val"],
    qh_val=unc_meta["qh_val"],
)
mae_unc = float(mean_absolute_error(y, oof_unc))
print("\n[UNCERTAINTY SHRINK APPLIED]")
print(f"  OOF MAE: {mae_unc:.4f} (delta {mae_unc-base_mae:+.4f})")
print(f"  unc_meta: {unc_meta}")
print(f"  alpha_oof mean/p95/max: {float(np.mean(alpha_oof)):.4f} / {float(np.qua
# EB group shift search + apply (on top of uncertainty shrink)
eb_meta = search_eb_group_shift(train_feat, y, oof_unc)
oof_final = oof_unc.copy()
test_final = test_unc.copy()
extra_shift = {"type": "none"}

if eb_meta.get("use", False):
    cols = eb_meta["cols"]
    shift_map = eb_meta["shift_map"]
    # apply to OOF
    oof_final = apply_shift_map(train_feat, oof_unc, shift_map, cols)
    # apply to test
    test_final = apply_shift_map(test_feat, test_unc, shift_map, cols)
    extra_shift = {"type": "eb_group", "cols": cols, "k": eb_meta["k"]}
final_oof_mae = float(mean_absolute_error(y, oof_final))
print("\n" + "="*75)
print("[FINAL OOF]")
print(f"  OOF MAE (base -> unc_shrink -> EB_shift): {final_oof_mae:.4f}")
print("  extra shift:", extra_shift)
print("  OOF pred quantiles:", qdict(oof_final, qs=(0,0.01,0.05,0.1,0.5,0.9,0.95,
print("="*75)
# clip predictions (LB-safe)
y_max = float(np.max(y))
test_final = np.clip(test_final, 0.0, y_max * 1.5)
# feature importance from a full-fit Model A (quick insight)
print("\n[full-train] fitting Model A on full train for feature importance...")
A_params = dict(
    loss_function="RMSE", eval_metric="MAE",
    depth=5, l2_leaf_reg=8, min_data_in_leaf=40,
    learning_rate=CFG.LR, iterations=CFG.ITERS,
    rsm=CFG.RSM, bootstrap_type="Bernoulli", subsample=CFG.SUBSAMPLE,
    random_strength=1.0,
    task_type="CPU", thread_count=-1,
    verbose=0, allow_writing_files=False,
    random_seed=SEED,
)
mA_full = CatBoostRegressor(**A_params)
mA_full.fit(train_feat[feat_full], y, verbose=0)
try:
    imp = mA_full.get_feature_importance()
    imp_df = pd.DataFrame({"feature": feat_full, "importance": imp}).sort_values(
    print("\n[Top 15 feature importances] (Model A full)")
    print(imp_df.to_string(index=False))

except Exception as e:
    print(f"[warn] feature importance failed: {e}")
del mA_full
gc.collect()
# write submission
sub = pd.DataFrame({
    "patient_id": test["patient_id"].values.astype(int),
    "ed_cost_next3y_usd": np.round(test_final.astype(float), 2)
})[["patient_id","ed_cost_next3y_usd"]]
out_path = Path(OUT_SUB_PATH)
out_path.parent.mkdir(parents=True, exist_ok=True)
sub.to_csv(out_path, index=False)
print("\n[SUBMISSION sanity checks]")
print("submission shape:", sub.shape)
print("columns:", list(sub.columns))
print("any NaNs:", bool(np.isnan(sub["ed_cost_next3y_usd"]).any()))
print("pred min/median/max:", float(sub["ed_cost_next3y_usd"].min()), float(sub["
print("pred quantiles:", qdict(sub["ed_cost_next3y_usd"].values))
print("\nSaved submission to:", str(out_path))
print("\nPaste back: (1) leaderboard MAE, (2) FINAL OOF MAE, (3) meta + unc_meta 

```
</details>

## 4) 可复现流程骨架（供后续 AI 总结/固化为“标准流程”）

### 4.1 数据与路径（固定约束）
- `DATA_DIR = D:\AgentDs\agent_ds_healthcare`
- `ed_cost_train.csv / ed_cost_test.csv / patients.csv / admissions_train.csv / admissions_test.csv`
- receipts 特征优先来自 `receipts_parsed.joblib`（避免解析 PDF）

### 4.2 特征工程（稳定优先）
1) **基础先验（low-risk）**：prior_cost / prior_visits + log/sqrt/cap + cost_per_visit + baseline_next3y
2) **患者信息**：age、sex、insurance、zip_region 等低维编码 + 少量交互
3) **admissions 聚合**：charlson_max/mean、pct_emergent
4) **receipts 低维稳健统计**：
   - bucket totals/ratios（em/proc/img/lab/crit/high acuity）
   - 集中度：HHI、top1/top3、gini、max_line_total
   - 价格水平：median_unit_price（overall + by bucket）
   - （扩展）per-code mix：share_code_{code} / cnt_code_{code}

### 4.3 训练（小样本稳健化）
- StratifiedKFold：`primary_chronic × target_bin`（若过细导致 min class count < n_folds 则降级为 chronic-only）
- 多 seed bagging（例如 5 seeds）
- 模型：浅树 CatBoost（depth 4–5）+ 强正则（rsm=0.8, subsample=0.8, l2/min_leaf）
- 每 seed：fold-bagging；可选 full-fit blend（轻量）
- 跨 seed 聚合：trimmed mean（drop min/max）

### 4.4 稳定性导向的 ensemble 搜索
- 仅 3 个模型权重搜索（步长 0.05）+ baseline 混合 `lam` + shift_mult
- 目标函数（示例）：`obj = mean_mae + STD_PEN*std_mae + LAM_PEN*lam + SHIFT_PEN*|shift_mult|`

### 4.5 后处理（只做低自由度、可解释、可控的稳健项）
1) uncertainty shrink：按 seed 分歧对高不确定样本增加 baseline 权重
2) EB group shift：按组样本量对残差中位数做收缩后加回

### 4.6 产出与日志
- 输出 `submission.csv`（patient_id, ed_cost_next3y_usd）
- 打印：OOF MAE、ensemble meta、预测分位数、NaN 检查等

