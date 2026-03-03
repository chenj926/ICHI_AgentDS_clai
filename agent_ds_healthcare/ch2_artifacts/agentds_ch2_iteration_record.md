# AgentDS Healthcare Challenge 2 — ED Cost Forecasting 迭代记录（基于用户PROMPT与提交MAE）

来源：来自 `extracted_user_prompts.txt` 的用户消息（不含assistant回答）。
覆盖范围：从 baseline 提交（LB 459.2684）到 Code27（LB 455.0833），重点包含每次提交的 MAE 与关键日志摘要。

## 快速总览（按时间顺序）

- PROMPT 04 — **Baseline** — LB MAE **459.2684**
- PROMPT 07 — **Code 17** — LB MAE **456.6848** | Δvs上一轮 -2.5836
- PROMPT 08 — **Code 18** — LB MAE **449.4152** | Δvs上一轮 -7.2696
- PROMPT 09 — **Code 19** — LB MAE **450.0905** | Δvs上一轮 +0.6753
- PROMPT 10 — **Code 20** — LB MAE **449.0221** | Δvs上一轮 -1.0684
- PROMPT 11 — **Code 21** — LB MAE **452.9674** | Δvs上一轮 +3.9453
- PROMPT 12 — **Code 22** — LB MAE **448.1754** | Δvs上一轮 -4.7920
- PROMPT 13 — **Code 23** — LB MAE **449.2911** | Δvs上一轮 +1.1157
- PROMPT 14 — **Code 24** — LB MAE **448.1754** | Δvs上一轮 -1.1157
- PROMPT 15 — **Code 25** — LB MAE **448.1210** | Δvs上一轮 -0.0544
- PROMPT 16 — **Code 26** — LB MAE **448.1413** | Δvs上一轮 +0.0203
- PROMPT 17 — **Code 27** — LB MAE **455.0833** | Δvs上一轮 +6.9420

## 详细记录

### R01 | PROMPT 04

- 标识: **Baseline**
- Leaderboard MAE: **459.2684**
- Plan: robust receipts+admissions+patients features -> 5-

**关键配置（从日志抽取）**
- CatBoost Task
- Seeds
- Folds
- RSM
- Subsample

**数据/特征摘要（从日志抽取）**
- ed_cost_train shape
- ed_cost_test shape
- patients shape
- receipt_feat shape
- FULL feature count
- PRUNED feature count

**提交回执片段（Submitting predictions / Score）**
- this is the real mae form the leaderboard:

**原始PROMPT开头摘录（前约120行，用于快速定位）**
```text
this is the real mae form the leaderboard:
459.2684
and here is our full log:
Plan: robust receipts+admissions+patients features -> 5-
fold CV -> CatBoost(MAE)+CatBoost(log1p) + baseline 
blend (tuned weights+clip+bias) -> train full -> 
submission.csv
Receipts loaded shape: (31864, 15) | sample cols: 
['patient_id', 'zip3_receipt_raw', 'insurance_receipt', 
'receipt_total', 'pdf_path', 'n_line_items', 'sum_line_total', 
'sum_unit_x_qty', 'parse_ok', 'zip3_receipt', 'code', 
'description']
Receipt rcpt_sum_items vs prior_ed_cost_5y_usd 
match_rate(train): 1.0000
WARNING: strat too granular (min class count 1). Falling 
back to stratify by primary_chronic.
Default metric period is 5 because MAE is/are not 
implemented for GPU
Model backend: CatBoost | Preferred task_type: GPU
Default metric period is 5 because MAE is/are not 
implemented for GPU
Fold 1 MAE (default blend): 907.248
Default metric period is 5 because MAE is/are not 
implemented for GPU
Fold 2 MAE (default blend): 933.306
Default metric period is 5 because MAE is/are not 
implemented for GPU
Fold 3 MAE (default blend): 846.651
Default metric period is 5 because MAE is/are not 
implemented for GPU
Fold 4 MAE (default blend): 852.588
Default metric period is 5 because MAE is/are not 
implemented for GPU
Fold 5 MAE (default blend): 884.036
OOF MAE raw: 1386.089 | log: 452.934 | baseline: 1973.234
Best OOF ensemble: {'mae': 449.40836165958086, 'w_raw': 
0.05, 'w_log': 0.9, 'w_base': 0.04999999999999993, 'clip_q': 
0.997, 'clip_hi': 9558.498209999989, 'bias_mode': 'group', 
'bias_obj': {'DiabetesComp': 98.15052216429012, 'HF': 
202.7015782361268, 'Pneumonia': 25.87477966712072}}
Default metric period is 5 because MAE is/are not 
implemented for GPU
Top 10 feature importances (raw model):
                               feature  importance
                       primary_chronic   53.490383
                  prior_ed_cost_5y_usd   28.913490
            rcpt_high_acuity_spend_sum   11.540394
                        rcpt_pdf_total    1.699253
                      prior_cost_log1p    1.159068
                       baseline_next3y    1.050749
                        rcpt_sum_items    0.877271
                         adm_dx_cnt_hf    0.336450
                 adm_dow_cnt_adm_dow_6    0.234927
rcpt_high_acuity_spend_per_prior_visit    0.161706
Submission sanity:
 shape: (2000, 2)
 columns: ['patient_id', 'ed_cost_next3y_usd']
 pred NaNs: 0
 pred min/median/max: 1015.6486711222069 
3566.6808514059426 9558.498209999989
Overall CV MAE (best ensemble): 449.40836165958086
Models used: CatBoost raw MAE (iter=4999) + CatBoost 
log1p RMSE (iter=800) + baseline blend (w_raw=0.05, 
w_log=0.90, w_base=0.05)
Saved submission to: 
D:\AgentDs\agent_ds_healthcare\submission.csv
Paste back CV MAE and these logs for next iteration.
ANd keep improve from here! 突破450!
```

---

### R02 | PROMPT 07

- 标识: **CODE 17**
- Leaderboard MAE: **456.6848**
- 相对上一轮变化: **-2.5836**（负数=提升）
- 标题/描述: Code17 (from transition (Code16->Code17))
- Goal: improve LB beyond ~452 by reducing
- Plan: (Code16->Code17) shallow+strong-reg CatBoost

**关键配置（从日志抽取）**
- CatBoost Task: CPU
- Seeds: 5
- Folds: 7
- RSM: 0.8
- Subsample: 0.8

**数据/特征摘要（从日志抽取）**
- ed_cost_train shape
- ed_cost_test shape
- patients shape
- receipt_feat shape
- FULL feature count
- PRUNED feature count

**Ensemble 权重搜索（Top 片段）**
- considered in ensemble search (helps LB sometimes)
- # Output: D:\AgentDs\agent_ds_healthcare\submission.csv
- # -----------------------------
- # Paths (must match prompt)
- # -----------------------------
- # -----------------------------
- # Minimal deps
- # -----------------------------
- # -----------------------------
- # Config (keep fast)
- # -----------------------------
- # iterations are upper bounds; early stopping typically
- …（更多见原prompt全文）

**后处理/校准/shift 相关（抽取含 shift 关键词的行）**
- def apply_chronic_group_shift(train_feat: pd.DataFrame,
- # optional chronic shift (very conservative; only apply if
- best_shift = {"type": "none"}
- oof2, shifts = apply_chronic_group_shift(train_feat,
- best_shift = {"type": "chronic_group", "shrink": shrink,
- # apply chosen chronic shift to test
- if best_shift["type"] == "chronic_group":
- for g, s in best_shift["shifts"].items():
- chronic shift): {final_oof_mae:.2f}")
- print("  extra shift:", best_shift["type"],
- ("shrink="+str(best_shift.get("shrink")) if
- best_shift["type"]!="none" else ""))

**提交回执片段（Submitting predictions / Score）**
- real MAE:
- Paste back CV MAE + these logs + new leaderboard MAE
- print("\nPaste back: (1) leaderboard MAE, (2) FINAL OOF

**原始PROMPT开头摘录（前约120行，用于快速定位）**
```text
real MAE:
456.6848
our log:
Plan: (Code16->Code17) shallow+strong-reg CatBoost 
(depth 4/5, rsm=0.8, subsample=0.8) + stability pruning + 
multi-seed; model space: log1p(y) and delta-
log(y/baseline), choose best by CV; train full; save 
submission.csv
Receipts loaded shape: (31864, 17) | cols: ['patient_id', 
'zip3_receipt_raw', 'insurance_receipt', 'receipt_total', 
'pdf_path', 'n_line_items', 'sum_line_total', 'sum_unit_x_qty', 
'parse_ok', 'zip3_receipt', 'code', 'description', 'qty', 
'unit_price', 'line_total', 'code_str', 'cat']
Receipt rcpt_sum_items vs prior_ed_cost_5y_usd 
match_rate(train): 1.0000
Pruning summary: total_feats=90 | mandatory=19 | 
selected=60
Selected head: ['primary_chronic', 'prior_ed_visits_5y', 
'prior_ed_cost_5y_usd', 'age', 'sex', 'insurance', 'zip3', 
'adm_los_days_mean', 'adm_los_days_max', 
'adm_los_days_sum', 'adm_acuity_emergent_mean', 
'adm_charlson_band_mean', 'adm_charlson_band_max', 
'adm_charlson_band_sum', 'adm_ed_visits_6m_mean', 
'adm_discharge_weekday_mode', 'adm_primary_dx_mode', 
'adm_primary_dx_nunique', 'adm_los_days_sum_log1p', 
'adm_charlson_band_sum_log1p', 'rcpt_n_lines', 
'rcpt_n_unique_codes', 'rcpt_sum_items', 
'rcpt_sum_unit_x_qty', 'rcpt_n_line_items_meta']
Fold 1 MAE | log=459.194 | deltalog=463.682 | 
ens(0.5/0.5)=453.791 | best_it log=861 dlt=960
Fold 2 MAE | log=451.107 | deltalog=458.545 | 
ens(0.5/0.5)=444.061 | best_it log=646 dlt=1215
Fold 3 MAE | log=442.797 | deltalog=429.819 | 
ens(0.5/0.5)=429.431 | best_it log=418 dlt=753
Fold 4 MAE | log=434.198 | deltalog=447.365 | 
ens(0.5/0.5)=429.703 | best_it log=2095 dlt=1474
Fold 5 MAE | log=455.378 | deltalog=451.449 | 
ens(0.5/0.5)=449.317 | best_it log=645 dlt=993
Overall CV MAE | log=448.535 | deltalog=450.172 | 
ens=441.260 | baseline=1973.234
Chosen approach for final training: ens  | CV MAE: 
441.2604
Final model(s): Ensemble: 0.5*log + 0.5*deltalog, each 
multi-seed avg (depth=4,rsm=0.8,subsample=0.8)
Selected feature count: 60
Submission sanity:
 shape: (2000, 2)
 columns: ['patient_id', 'ed_cost_next3y_usd']
 pred NaNs: 0
 pred min/median/max: 827.3384745500957 
3547.0192748521613 11201.384185819214
Saved submission to: 
D:\AgentDs\agent_ds_healthcare\submission.csv
Paste back CV MAE + these logs + new leaderboard MAE 
for next iteration.
ANd here I have a best curent model:
# === ITERATION 7: Anti-overfit++ on the v3(Iter15/16) 
path (keep it practical, fast, and LB-oriented) ===
# Core ideas:
#   - Still LOW-DIM receipts features (buckets + shares + 
HHI + EM stats + key procedure flags)
#   - Shallow trees (depth 4-5), strong RSM (0.8), multi-
seed bagging
#   - Stronger "generalization" ensemble selection: choose 
weights by stability across seeds (mean + std), not just 
best OOF
#   - Add ONE diverse model: MAE-loss on pruned features 
(metric-aligned but still regularized)
#   - Tiny baseline shrink option (toward baseline_next3y) 
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
```

---

### R03 | PROMPT 08

- 标识: **CODE 18**
- Leaderboard MAE: **449.4152**
- 相对上一轮变化: **-7.2696**（负数=提升）
- 标题/描述: v3/code16 spirit: LOW-DIM receipts + shallow
- Goal: push LB down from ~451 by reducing generalization

**关键配置（从日志抽取）**
- CatBoost Task: CPU
- Seeds: 5
- Folds: 7
- RSM: 0.8
- Subsample: 0.8

**数据/特征摘要（从日志抽取）**
- ed_cost_train shape: 2000, 5
- ed_cost_test shape: 2000, 4
- patients shape: 4000, 5
- receipt_feat shape: 4000, 44
- FULL feature count: 64
- PRUNED feature count: 49

**OOF（逐seed）**
- Seed 1/5: A_RMSE_full_d5=431.98 |
- Seed 2/5: A_RMSE_full_d5=432.97 |
- Seed 3/5: A_RMSE_full_d5=431.83 |
- Seed 4/5: A_RMSE_full_d5=432.06 |
- Seed 5/5: A_RMSE_full_d5=431.87 |

**OOF（seed聚合）**
- [seed-aggregated OOF MAE per model] (trimmed mean
- A_RMSE_full_d5    : 429.51
- B_RMSE_pruned_d4  : 428.30
- C_MAE_pruned_d4   : 435.40

**Ensemble 权重搜索（Top 片段）**
- [ensemble search] Top candidates (robust objective =
- #01 obj=430.791 meanMAE=430.637 std=0.768 | w=
- #02 obj=430.792 meanMAE=430.638 std=0.768 | w=
- #03 obj=430.794 meanMAE=430.639 std=0.769 | w=
- #04 obj=430.795 meanMAE=430.634 std=0.805 | w=
- #05 obj=430.796 meanMAE=430.634 std=0.805 | w=
- #06 obj=430.796 meanMAE=430.634 std=0.805 | w=
- #07 obj=430.806 meanMAE=430.660 std=0.729 | w=
- #08 obj=430.807 meanMAE=430.660 std=0.730 | w=
- #09 obj=430.807 meanMAE=430.660 std=0.730 | w=
- #10 obj=430.820 meanMAE=430.651 std=0.844 | w=

**后处理/校准/shift 相关（抽取含 shift 关键词的行）**
- extra shift: chronic_group shrink=0.3

**提交回执片段（Submitting predictions / Score）**
- Paste back: (1) leaderboard MAE, (2) FINAL OOF MAE, (3)
- 🚀 Submitting predictions...
- ✅ Prediction submitted successfully!
- 📊 Score: 449.4152 (MAE)
- 📊 Score: 449.4152

**原始PROMPT开头摘录（前约120行，用于快速定位）**
```text
our 
log:===================================
======================================
======================
CODE 18 | v3/code16 spirit: LOW-DIM receipts + shallow 
CatBoost + strong regularization + pruning + multi-seed + 
STABLE ensemble
Goal: push LB down from ~451 by reducing generalization 
gap (NO over-engineering).
======================================
======================================
===================
[load] reading CSVs...
[ed_cost_train] shape=(2000, 5) | cols=5 | mem=0.17 MB
[ed_cost_test] shape=(2000, 4) | cols=4 | mem=0.15 MB
[patients] shape=(4000, 5) | cols=5 | mem=0.49 MB
[target stats]
count     2000.00000
mean      3908.25191
std       1822.40192
min        306.88000
25%       2548.76750
50%       3569.09500
75%       4956.42250
max      11184.61000
[admissions] building robust aggregates...
  admissions features: (4000, 4) | cols=['patient_id', 
'charlson_max', 'charlson_mean', 'pct_emergent']
[receipts] loading receipts_parsed.joblib and building low-
dim receipt features...
  receipt_feat shape: (4000, 44)
  receipt_feat cols (43): ['n_unique_codes', 'cost_hhi', 
'top1_share', 'top3_share', 'gini_line_total', 'max_line_total', 
'n_em_codes', 'max_em_level', 'avg_em_level', 'n_high_em', 
'has_critical_care', 'has_high_acuity', 'has_observation', 
'has_imaging', 'has_intub_31500', 'has_cvc_36556', 
'has_cpr_92950', 'has_artline_36620', 'has_ct_head_70450', 
'has_99285', 'has_ct_abdpel_74177', 'has_troponin_84484', 
'has_obs_G0378', 'n_procedures', 'n_imaging', 'n_lab', 
'pct_cost_em', 'pct_cost_procedure', 'pct_cost_critical', 
'pct_cost_imaging', 'pct_cost_lab', 'pct_cost_high_acuity', 
'cost_per_em', 'n_high_acuity_total', 'median_unit_price', 
'median_unit_price_em', 'median_unit_price_imaging', 
'median_unit_price_lab', 'log1p_median_unit_price', 
'log1p_median_unit_price_em', 
'log1p_median_unit_price_imaging', 
'log1p_median_unit_price_lab', 'log1p_max_line_total']
[features] building train/test feature frames...
  FULL feature count:   64
  PRUNED feature count: 49
  PRUNED features: ['prior_ed_visits_5y', 
'prior_ed_cost_5y_usd', 'prior_cost_cap20k', 
'sqrt_prior_cost', 'log_prior_cost', 'log_prior_cost_cap20k', 
'cost_per_visit', 'log_visits', 'baseline_next3y', 
'chronic_encoded', 'age', 'sex_encoded', 
'insurance_encoded', 'zip_region', 'ins_x_chronic', 
'charlson_max', 'charlson_mean', 'pct_emergent', 
'cost_per_em', 'cost_hhi', 'pct_cost_em', 
'pct_cost_procedure', 'pct_cost_critical', 
'pct_cost_high_acuity', 'pct_cost_imaging', 'pct_cost_lab', 
'n_high_acuity_total', 'has_critical_care', 'has_99285', 
'max_em_level', 'avg_em_level', 'n_high_em', 
'n_unique_codes', 'top1_share', 'top3_share', 
'gini_line_total', 'max_line_total', 'median_unit_price', 
'median_unit_price_em', 'median_unit_price_imaging', 
'median_unit_price_lab', 'log1p_median_unit_price', 
'log1p_median_unit_price_em', 
'log1p_median_unit_price_imaging', 
'log1p_median_unit_price_lab', 'log1p_max_line_total', 
'logprior_x_pctcritical', 'logprior_x_highacu', 
'cost_per_code']
[training] CatBoost CPU | shallow trees | rsm=0.8 | 
subsample=0.8 | multi-seed bagging
Models: ['A_RMSE_full_d5', 'B_RMSE_pruned_d4', 
'C_MAE_pruned_d4']
Seeds=5, Folds=7
  Seed 1/5 OOF MAE: A_RMSE_full_d5=431.98 | 
B_RMSE_pruned_d4=430.37 | C_MAE_pruned_d4=436.85
  Seed 2/5 OOF MAE: A_RMSE_full_d5=432.97 | 
B_RMSE_pruned_d4=432.87 | C_MAE_pruned_d4=441.95
  Seed 3/5 OOF MAE: A_RMSE_full_d5=431.83 | 
B_RMSE_pruned_d4=430.74 | C_MAE_pruned_d4=436.77
  Seed 4/5 OOF MAE: A_RMSE_full_d5=432.06 | 
B_RMSE_pruned_d4=430.08 | C_MAE_pruned_d4=438.76
  Seed 5/5 OOF MAE: A_RMSE_full_d5=431.87 | 
B_RMSE_pruned_d4=432.39 | C_MAE_pruned_d4=439.58
[seed-aggregated OOF MAE per model] (trimmed mean 
across seeds)
  A_RMSE_full_d5    : 429.51
  B_RMSE_pruned_d4  : 428.30
  C_MAE_pruned_d4   : 435.40
[median best_iteration per model] (reference)
  A_RMSE_full_d5    : 596
  B_RMSE_pruned_d4  : 711
  C_MAE_pruned_d4   : 619
[ensemble search] Top candidates (robust objective = 
mean + std_pen + simplicity_pen):
  #01 obj=430.791 meanMAE=430.637 std=0.768 | w=
(0.40,0.60,0.00) | lam=0.00 | shift_mult=0.0 | shift=0.00
  #02 obj=430.792 meanMAE=430.638 std=0.768 | w=
(0.40,0.60,0.00) | lam=0.00 | shift_mult=0.5 | shift=0.55
  #03 obj=430.794 meanMAE=430.639 std=0.769 | w=
(0.40,0.60,0.00) | lam=0.00 | shift_mult=1.0 | shift=1.10
  #04 obj=430.795 meanMAE=430.634 std=0.805 | w=
(0.35,0.65,0.00) | lam=0.00 | shift_mult=0.0 | shift=-0.00
  #05 obj=430.796 meanMAE=430.634 std=0.805 | w=
(0.35,0.65,0.00) | lam=0.00 | shift_mult=0.5 | shift=-0.17
  #06 obj=430.796 meanMAE=430.634 std=0.805 | w=
(0.35,0.65,0.00) | lam=0.00 | shift_mult=1.0 | shift=-0.34
  #07 obj=430.806 meanMAE=430.660 std=0.729 | w=
(0.45,0.55,0.00) | lam=0.00 | shift_mult=0.0 | shift=0.00
  #08 obj=430.807 meanMAE=430.660 std=0.730 | w=
(0.45,0.55,0.00) | lam=0.00 | shift_mult=0.5 | shift=0.27
  #09 obj=430.807 meanMAE=430.660 std=0.730 | w=
(0.45,0.55,0.00) | lam=0.00 | shift_mult=1.0 | shift=0.54
  #10 obj=430.820 meanMAE=430.651 std=0.844 | w=
(0.30,0.70,0.00) | lam=0.00 | shift_mult=0.0 | shift=-0.00
[full-train] fitting Model A on full train for feature 
importance...
[Top 10 feature importances] (Model A full)
```

---

### R04 | PROMPT 09

- 标识: **CODE 19**
- Leaderboard MAE: **450.0905**
- 相对上一轮变化: **+0.6753**（负数=提升）
- 标题/描述: Less-is-More++: shallow+strong-reg + multi-

**关键配置（从日志抽取）**
- CatBoost Task: CPU
- Seeds: 5
- Folds: 7
- RSM: 0.8
- Subsample: 0.8

**数据/特征摘要（从日志抽取）**
- ed_cost_train shape: 2000, 5
- ed_cost_test shape: 2000, 4
- patients shape: 4000, 5
- receipt_feat shape: 4000, 44
- FULL feature count: 64
- PRUNED feature count: 49

**OOF（逐seed）**
- Seed 1/5: A_RMSE_full_d5=428.20 |
- Seed 2/5: A_RMSE_full_d5=432.34 |
- Seed 3/5: A_RMSE_full_d5=432.12 |
- Seed 4/5: A_RMSE_full_d5=436.99 |
- Seed 5/5: A_RMSE_full_d5=433.22 |

**OOF（seed聚合）**
- [seed-aggregated OOF MAE per model] (trimmed mean
- A_RMSE_full_d5    : 429.25
- B_RMSE_pruned_d4  : 427.76
- D_LOG_pruned_d4   : 444.48

**提交回执片段（Submitting predictions / Score）**
- 🚀 Submitting predictions...
- ✅ Prediction submitted successfully!
- 📊 Score: 450.0905 (MAE)
- 📊 Score: 450.0905
- Paste back: (1) leaderboard MAE, (2) raw ensemble OOF

**原始PROMPT开头摘录（前约120行，用于快速定位）**
```text
🚀 Submitting predictions...
✅ Prediction submitted successfully!
📊 Score: 450.0905 (MAE)
✅ Validation passed
✅ Submission successful!
   📊 Score: 450.0905
   📏 Metric: MAE
   ✔️  Validation: Passed
======================================
======================================
====================
CODE 19 | Less-is-More++: shallow+strong-reg + multi-
seed + trimmed mean + (safe) calibration; aim: improve LB 
< 449
======================================
======================================
====================
[load] reading CSVs...
[ed_cost_train] shape=(2000, 5) | cols=5 | mem=0.17 MB
[ed_cost_test] shape=(2000, 4) | cols=4 | mem=0.15 MB
[patients] shape=(4000, 5) | cols=5 | mem=0.49 MB
[target stats]
count     2000.00000
mean      3908.25191
std       1822.40192
min        306.88000
25%       2548.76750
50%       3569.09500
75%       4956.42250
max      11184.61000
[admissions] building robust aggregates...
  admissions features: (4000, 4)
[receipts] building low-dim receipt features...
  receipt_feat shape: (4000, 44) | n_features=43
[features] building train/test feature frames...
  FULL feature count:   64
  PRUNED feature count: 49
[training] CatBoost CPU | shallow | rsm=0.8 | 
subsample=0.8 | multi-seed | trimmed-mean aggregation
Models: ['A_RMSE_full_d5', 'B_RMSE_pruned_d4', 
'D_LOG_pruned_d4']
Seeds=5, Folds=7
  Seed 1/5 OOF MAE: A_RMSE_full_d5=428.20 | 
B_RMSE_pruned_d4=427.77 | D_LOG_pruned_d4=444.59
  Seed 2/5 OOF MAE: A_RMSE_full_d5=432.34 | 
B_RMSE_pruned_d4=431.95 | D_LOG_pruned_d4=448.39
  Seed 3/5 OOF MAE: A_RMSE_full_d5=432.12 | 
B_RMSE_pruned_d4=430.59 | D_LOG_pruned_d4=446.87
  Seed 4/5 OOF MAE: A_RMSE_full_d5=436.99 | 
B_RMSE_pruned_d4=433.59 | D_LOG_pruned_d4=449.55
  Seed 5/5 OOF MAE: A_RMSE_full_d5=433.22 | 
B_RMSE_pruned_d4=431.19 | D_LOG_pruned_d4=447.03
[seed-aggregated OOF MAE per model] (trimmed mean 
across seeds)
  A_RMSE_full_d5    : 429.25
  B_RMSE_pruned_d4  : 427.76
  D_LOG_pruned_d4   : 444.48
[median best_iteration per model] (reference)
  A_RMSE_full_d5    : 603
  B_RMSE_pruned_d4  : 734
  D_LOG_pruned_d4   : 553
[baseline]
  baseline_next3y OOF MAE: 1973.234
  OOF pred mins (A,B,D): {'A_RMSE_full_d5': 
1097.0887148001384, 'B_RMSE_pruned_d4': 
1076.076134186275, 'D_LOG_pruned_d4': 
979.0738396811215}
[candidate search] Top 10 (objective = CFcal_MAE + 
0.25*seed_std + 1.5*lam):
  #01 obj=428.206 | CFcal=427.708 | raw=427.918 | 
std=1.991 | wA=0.25 wB=0.75 wD=0.00 | lam=0.00 | 
min=1081.3 max=10306.2
  #02 obj=428.258 | CFcal=427.752 | raw=427.968 | 
std=2.027 | wA=0.30 wB=0.70 wD=0.00 | lam=0.00 | 
min=1082.4 max=10306.6
  #03 obj=428.277 | CFcal=427.750 | raw=428.079 | 
std=2.109 | wA=0.40 wB=0.60 wD=0.00 | lam=0.00 | 
min=1084.5 max=10307.5
  #04 obj=428.289 | CFcal=427.773 | raw=428.021 | 
std=2.065 | wA=0.35 wB=0.65 wD=0.00 | lam=0.00 | 
min=1083.4 max=10307.0
  #05 obj=428.346 | CFcal=427.806 | raw=428.143 | 
std=2.158 | wA=0.45 wB=0.55 wD=0.00 | lam=0.00 | 
min=1085.5 max=10307.9
  #06 obj=428.411 | CFcal=427.913 | raw=428.171 | 
std=1.996 | wA=0.25 wB=0.70 wD=0.05 | lam=0.00 | 
min=1077.3 max=10279.7
  #07 obj=428.454 | CFcal=427.945 | raw=428.222 | 
std=2.036 | wA=0.30 wB=0.65 wD=0.05 | lam=0.00 | 
min=1078.4 max=10280.2
  #08 obj=428.456 | CFcal=427.902 | raw=428.209 | 
std=2.214 | wA=0.50 wB=0.50 wD=0.00 | lam=0.00 | 
min=1086.6 max=10308.4
  #09 obj=428.504 | CFcal=427.986 | raw=428.283 | 
std=2.072 | wA=0.35 wB=0.60 wD=0.05 | lam=0.00 | 
min=1079.4 max=10280.6
  #10 obj=428.559 | CFcal=428.031 | raw=428.345 | 
std=2.114 | wA=0.40 wB=0.55 wD=0.05 | lam=0.00 | 
min=1080.5 max=10281.1
[best config]
{'wA': 0.25, 'wB': 0.75, 'wD': 0.0, 'lam_baseline': 0.0, 
'raw_mae': 427.918, 'cf_cal_mae': 427.708, 'seed_std': 1.991}
[OOF metrics]
  raw ensemble OOF MAE: 427.918
  CF-calibrated OOF MAE: 427.708
  OOF quantiles (raw ensemble): {0: 1081.3292793397409, 
0.01: 1389.1603740785959, 0.05: 1758.2910385229115, 0.1: 
2015.8008382827763, 0.5: 3512.091360727207, 0.9: 
6491.895494477464, 0.95: 7345.833229057238, 0.99: 
8691.034780377586, 1.0: 10306.153772664396}
  OOF quantiles (CF-cal): {0: 1091.8821886893836, 0.01: 
1391.587684287993, 0.05: 1758.6380834999661, 0.1: 
2008.4759305610091, 0.5: 3531.1498961973657, 0.9: 
6522.930608267557, 0.95: 7377.61929558277, 0.99: 
8698.303532207434, 1.0: 10338.533465423898}
[feature importance] fitting Model B (pruned, depth=4) on 
full train...
                feature  importance
        chronic_encoded   16.297410
        sqrt_prior_cost    9.361286
```

---

### R05 | PROMPT 10

- 标识: **CODE 20**
- Leaderboard MAE: **449.0221**
- 相对上一轮变化: **-1.0684**（负数=提升）
- 标题/描述: Code18-core + tiny robust residual shifts (cross-
- Goal: beat LB 449.4152

**关键配置（从日志抽取）**
- CatBoost Task: CPU
- Seeds: 5
- Folds: 7
- RSM: 0.8
- Subsample: 0.8

**数据/特征摘要（从日志抽取）**
- ed_cost_train shape: 2000, 5
- ed_cost_test shape: 2000, 4
- patients shape: 4000, 5
- receipt_feat shape: 4000, 44
- FULL feature count: 64
- PRUNED feature count: 49

**OOF（逐seed）**
- Seed 1/5: A_full_d5=429.66 |
- Seed 2/5: A_full_d5=431.72 |
- Seed 3/5: A_full_d5=430.53 |
- Seed 4/5: A_full_d5=433.35 |
- Seed 5/5: A_full_d5=432.09 |

**OOF（seed聚合）**
- [seed-aggregated OOF MAE per model] (trimmed mean)
- A_full_d5   : 429.32
- B_pruned_d4 : 428.21

**Ensemble 权重搜索（Top 片段）**
- [ensemble search] Top 8 (obj=mean+0.2*std+6*lam):
- #01 obj=430.600 mean=430.440 std=0.800 | wA=0.45
- #02 obj=430.603 mean=430.442 std=0.802 | wA=0.50
- #03 obj=430.622 mean=430.460 std=0.807 | wA=0.40
- #04 obj=430.626 mean=430.462 std=0.819 | wA=0.55
- #05 obj=430.668 mean=430.504 std=0.818 | wA=0.35
- #06 obj=430.670 mean=430.502 std=0.842 | wA=0.60
- #07 obj=430.728 mean=430.553 std=0.875 | wA=0.65
- #08 obj=430.733 mean=430.565 std=0.839 | wA=0.30
- #01 MAE=428.412 | chronic_factor=0.45 |
- #02 MAE=428.426 | chronic_factor=0.35 |
- #03 MAE=428.433 | chronic_factor=0.30 |
- …（更多见原prompt全文）

**后处理/校准/shift 相关（抽取含 shift 关键词的行）**
- chronic shifts: {np.str_('DiabetesComp'):

**提交回执片段（Submitting predictions / Score）**
- 🚀 Submitting predictions...
- ✅ Prediction submitted successfully!
- 📊 Score: 449.0221 (MAE)
- 📊 Score: 449.0221
- Paste back: (1) leaderboard MAE, (2) base OOF MAE, (3)

**原始PROMPT开头摘录（前约120行，用于快速定位）**
```text
🚀 Submitting predictions...
✅ Prediction submitted successfully!
📊 Score: 449.0221 (MAE)
✅ Validation passed
✅ Submission successful!
   📊 Score: 449.0221
   📏 Metric: MAE
   ✔️  Validation: Passed
======================================
======================================
========================
CODE 20 | Code18-core + tiny robust residual shifts (cross-
fitted) | goal: beat LB 449.4152
======================================
======================================
========================
[load] reading CSVs...
[ed_cost_train] shape=(2000, 5) | cols=5 | mem=0.17 MB
[ed_cost_test] shape=(2000, 4) | cols=4 | mem=0.15 MB
[patients] shape=(4000, 5) | cols=5 | mem=0.49 MB
[target stats]
count     2000.00000
mean      3908.25191
std       1822.40192
min        306.88000
25%       2548.76750
50%       3569.09500
75%       4956.42250
max      11184.61000
[admissions] building aggregates...
  admissions features: (4000, 4)
[receipts] building low-dim receipt features...
  receipt_feat shape: (4000, 44) | n_features=43
[features] building train/test feature frames...
  FULL feature count:   64
  PRUNED feature count: 49
[training] CatBoost CPU | depth(5/4) | rsm=0.8 | 
subsample=0.8 | multi-seed | 7-fold
Models: ['A_full_d5', 'B_pruned_d4']
Seeds=5, Folds=7
  Seed 1/5 OOF MAE: A_full_d5=429.66 | 
B_pruned_d4=431.63
  Seed 2/5 OOF MAE: A_full_d5=431.72 | 
B_pruned_d4=429.85
  Seed 3/5 OOF MAE: A_full_d5=430.53 | 
B_pruned_d4=433.18
  Seed 4/5 OOF MAE: A_full_d5=433.35 | 
B_pruned_d4=431.47
  Seed 5/5 OOF MAE: A_full_d5=432.09 | 
B_pruned_d4=430.56
[seed-aggregated OOF MAE per model] (trimmed mean)
  A_full_d5   : 429.32
  B_pruned_d4 : 428.21
[median best_iteration] (ref)
  A_full_d5   : 596
  B_pruned_d4 : 689
[ensemble search] Top 8 (obj=mean+0.2*std+6*lam):
  #01 obj=430.600 mean=430.440 std=0.800 | wA=0.45 
wB=0.55 | lam=0.00
  #02 obj=430.603 mean=430.442 std=0.802 | wA=0.50 
wB=0.50 | lam=0.00
  #03 obj=430.622 mean=430.460 std=0.807 | wA=0.40 
wB=0.60 | lam=0.00
  #04 obj=430.626 mean=430.462 std=0.819 | wA=0.55 
wB=0.45 | lam=0.00
  #05 obj=430.668 mean=430.504 std=0.818 | wA=0.35 
wB=0.65 | lam=0.00
  #06 obj=430.670 mean=430.502 std=0.842 | wA=0.60 
wB=0.40 | lam=0.00
  #07 obj=430.728 mean=430.553 std=0.875 | wA=0.65 
wB=0.35 | lam=0.00
  #08 obj=430.733 mean=430.565 std=0.839 | wA=0.30 
wB=0.70 | lam=0.00
[chosen ensemble] {'wA': 0.45, 'wB': 0.55, 'lam': 0.0, 
'mean_mae': 430.4403379395482, 'std_mae': 
0.8002349370370401}
[base ensemble]
  OOF MAE: 428.507
  pred quantiles: {0: 1023.7016969393061, 0.01: 
1390.2282495939444, 0.05: 1754.852357621023, 0.1: 
2020.3051014078267, 0.5: 3509.231686085173, 0.9: 
6492.395996205892, 0.95: 7334.802448033061, 0.99: 
8649.918725012143, 1.0: 10278.201389433469}
[shift tuning] Top 10 (cross-fitted MAE):
  #01 MAE=428.412 | chronic_factor=0.45 | 
flag_factor=0.00
  #02 MAE=428.426 | chronic_factor=0.35 | 
flag_factor=0.00
  #03 MAE=428.433 | chronic_factor=0.30 | 
flag_factor=0.00
  #04 MAE=428.441 | chronic_factor=0.25 | 
flag_factor=0.00
  #05 MAE=428.462 | chronic_factor=0.15 | 
flag_factor=0.00
  #06 MAE=428.503 | chronic_factor=0.45 | 
flag_factor=0.25
  #07 MAE=428.507 | chronic_factor=0.00 | 
flag_factor=0.00
  #08 MAE=428.528 | chronic_factor=0.35 | 
flag_factor=0.25
  #09 MAE=428.541 | chronic_factor=0.30 | 
flag_factor=0.25
  #10 MAE=428.556 | chronic_factor=0.25 | 
flag_factor=0.25
[chosen shifts] {'cf': 0.45, 'ff': 0.0, 'cv_mae': 
428.4121165882631}
[final after shifts]
  OOF MAE: 428.264  | delta vs base: -0.243
  chronic shifts: {np.str_('DiabetesComp'): 
-0.41355766401927324, np.str_('HF'): 9.851538477514577, 
np.str_('Pneumonia'): -9.070108940425035}
  flag shifts (order=['has_intub_31500', 'has_cvc_36556', 
'has_cpr_92950', 'has_artline_36620', 'has_critical_care', 
'has_ct_head_70450', 'has_ct_abdpel_74177', 
'has_troponin_84484', 'has_99285', 'has_obs_G0378']): [0.0, 
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0]
  pred quantiles: {0: 1014.6315879988811, 0.01: 
1389.8146919299252, 0.05: 1751.1724207204609, 0.1: 
2015.5744995861542, 0.5: 3512.441058061503, 0.9: 
6502.247534683406, 0.95: 7344.653986510576, 0.99:
```

---

### R06 | PROMPT 11

- 标识: **CODE 21**
- Leaderboard MAE: **452.9674**
- 相对上一轮变化: **+3.9453**（负数=提升）
- 标题/描述: Code20-core + blend-bagging (top blends

**关键配置（从日志抽取）**
- CatBoost Task: CPU
- Seeds: 5
- Folds: 7
- RSM: 0.8
- Subsample: 0.8

**数据/特征摘要（从日志抽取）**
- ed_cost_train shape: 2000, 5
- ed_cost_test shape: 2000, 4
- patients shape: 4000, 5
- receipt_feat shape: 4000, 46
- FULL feature count: 70
- PRUNED feature count: 56

**OOF（逐seed）**
- Seed 1/5: A_full_d5=432.00 |
- Seed 2/5: A_full_d5=432.90 |
- Seed 3/5: A_full_d5=434.94 |
- Seed 4/5: A_full_d5=431.70 |
- Seed 5/5: A_full_d5=432.05 |

**OOF（seed聚合）**
- [seed-aggregated OOF MAE per model] (trimmed mean)
- A_full_d5   : 430.01
- B_pruned_d4 : 430.32

**后处理/校准/shift 相关（抽取含 shift 关键词的行）**
- across lam) + chronic+highacu tiny residual shifts | aim: LB
- chronic shifts: {np.str_('DiabetesComp'): 11.55,

**提交回执片段（Submitting predictions / Score）**
- 🚀 Submitting predictions...
- ✅ Prediction submitted successfully!
- 📊 Score: 452.9674 (MAE)
- 📊 Score: 452.9674
- Paste back: (1) leaderboard MAE, (2) base OOF MAE, (3)

**原始PROMPT开头摘录（前约120行，用于快速定位）**
```text
🚀 Submitting predictions...
✅ Prediction submitted successfully!
📊 Score: 452.9674 (MAE)
✅ Validation passed
✅ Submission successful!
   📊 Score: 452.9674
   📏 Metric: MAE
   ✔️  Validation: Passed
======================================
======================================
==================================
CODE 21 | Code20-core + blend-bagging (top blends 
across lam) + chronic+highacu tiny residual shifts | aim: LB 
< 449.0
======================================
======================================
==================================
[load] reading CSVs...
[ed_cost_train] shape=(2000, 5) | cols=5 | mem=0.17 MB
[ed_cost_test] shape=(2000, 4) | cols=4 | mem=0.15 MB
[patients] shape=(4000, 5) | cols=5 | mem=0.49 MB
[target stats]
count     2000.00000
mean      3908.25191
std       1822.40192
min        306.88000
25%       2548.76750
50%       3569.09500
75%       4956.42250
max      11184.61000
[admissions] building aggregates...
  admissions features: ((4000, 9), ['patient_id', 'adm_n', 
'charlson_max', 'charlson_mean', 'pct_emergent', 
'adm_los_mean', 'adm_los_max', 'adm_edvis6m_mean', 
'adm_primary_dx_nuniq'])
[receipts] building low-dim receipt features...
  receipt_feat shape: (4000, 46) | n_features=45
[features] building train/test feature frames...
  Receipt rcpt_sum_items vs prior_ed_cost_5y_usd 
match_rate(train): 1.0000
  Receipt rcpt_sum_items vs prior_ed_cost_5y_usd 
match_rate(test):  1.0000
  FULL feature count:   70
  PRUNED feature count: 56
[baseline stats]
  baseline_next3y quantiles: {0: 30.0, 0.01: 30.0, 0.05: 
126.78989999999999, 0.1: 270.0018, 0.5: 1301.226, 0.9: 
6154.710600000001, 0.95: 8483.0742, 0.99: 13351.53954, 
1.0: 23848.074}
[training] CatBoost CPU | depth(5/4) | rsm=0.8 | 
subsample=0.8 | multi-seed | 7-fold
Models: ['A_full_d5', 'B_pruned_d4']
Seeds=5, Folds=7
  Seed 1/5 OOF MAE: A_full_d5=432.00 | 
B_pruned_d4=431.12
  Seed 2/5 OOF MAE: A_full_d5=432.90 | 
B_pruned_d4=432.51
  Seed 3/5 OOF MAE: A_full_d5=434.94 | 
B_pruned_d4=432.95
  Seed 4/5 OOF MAE: A_full_d5=431.70 | 
B_pruned_d4=435.62
  Seed 5/5 OOF MAE: A_full_d5=432.05 | 
B_pruned_d4=433.89
[seed-aggregated OOF MAE per model] (trimmed mean)
  A_full_d5   : 430.01
  B_pruned_d4 : 430.32
[median best_iteration] (ref)
  A_full_d5   : 604
  B_pruned_d4 : 688
[blend-bagging] selected blends (obj=mean+0.2*std), one 
per lam + top overall:
  #01 obj=432.080 mean=431.924 std=0.783 | wA=0.55 
wB=0.45 | lam=0.00
  #02 obj=432.544 mean=432.412 std=0.657 | wA=0.55 
wB=0.45 | lam=0.02
  #03 obj=438.046 mean=437.914 std=0.664 | wA=0.55 
wB=0.45 | lam=0.05
  #04 obj=449.837 mean=449.727 std=0.553 | wA=0.60 
wB=0.40 | lam=0.08
  #05 obj=432.087 mean=431.925 std=0.811 | wA=0.50 
wB=0.50 | lam=0.00
[base (blend-bagged) ensemble]
  OOF MAE: 431.715
  pred quantiles: {0: 1012.4547515462455, 0.01: 
1359.688524246625, 0.05: 1708.4569056945343, 0.1: 
1963.9049212353627, 0.5: 3447.9322290430446, 0.9: 
6467.928557726796, 0.95: 7362.382416044668, 0.99: 
8756.292284671596, 1.0: 10636.601123341969}
[high-acuity support]
  train highacu rate: 0.737 | count: 1474
  test  highacu rate: 0.725 | count: 1450
[shift tuning] Top 10 (cross-fitted MAE):
  #01 MAE=430.121 | chronic_factor=0.60 | 
highacu_factor=1.00
  #02 MAE=430.126 | chronic_factor=0.80 | 
highacu_factor=1.00
  #03 MAE=430.131 | chronic_factor=0.80 | 
highacu_factor=0.85
  #04 MAE=430.142 | chronic_factor=0.60 | 
highacu_factor=0.85
  #05 MAE=430.145 | chronic_factor=0.80 | 
highacu_factor=0.65
  #06 MAE=430.168 | chronic_factor=0.80 | 
highacu_factor=0.45
  #07 MAE=430.178 | chronic_factor=0.60 | 
highacu_factor=0.65
  #08 MAE=430.181 | chronic_factor=0.45 | 
highacu_factor=1.00
  #09 MAE=430.206 | chronic_factor=0.80 | 
highacu_factor=0.25
  #10 MAE=430.211 | chronic_factor=0.45 | 
highacu_factor=0.85
[chosen shifts] {'cf': 0.6, 'hf': 1.0, 'cv_mae': 
430.1208867382407}
[final after shifts]
  OOF MAE: 429.55 | delta vs base: -2.165
  chronic shifts: {np.str_('DiabetesComp'): 11.55, 
np.str_('HF'): 36.316, np.str_('Pneumonia'): -1.166}
  highacu shift: 19.896
  pred quantiles: {0: 1011.2885317378174, 0.01:
```

---

### R07 | PROMPT 12

- 标识: **CODE 22**
- Leaderboard MAE: **448.1754**
- 相对上一轮变化: **-4.7920**（负数=提升）
- 标题/描述: Back to Code20 route: (2-model shallow
- Goal: improve LB from 449.0221 toward <448 WITHOUT

**关键配置（从日志抽取）**
- CatBoost Task: CPU
- Seeds: 5
- Folds: 7
- RSM: 0.8
- Subsample: 0.8

**数据/特征摘要（从日志抽取）**
- ed_cost_train shape: 2000, 5
- ed_cost_test shape: 2000, 4
- patients shape: 4000, 5
- receipt_feat shape: 4000, 45
- FULL feature count: 64
- PRUNED feature count: 49

**OOF（逐seed）**
- Seed 1/5: A_full_d5=429.99 |
- Seed 2/5: A_full_d5=431.99 |
- Seed 3/5: A_full_d5=432.29 |
- Seed 4/5: A_full_d5=432.93 |
- Seed 5/5: A_full_d5=432.91 |

**OOF（seed聚合）**
- [seed-aggregated OOF MAE per model] (trimmed mean)
- A_full_d5   : 429.34
- B_pruned_d4 : 427.41

**Ensemble 权重搜索（Top 片段）**
- [ensemble search] Top 8 (obj=mean+0.2*std):
- #01 obj=430.448 mean=430.294 std=0.769 | wA=0.30
- #02 obj=430.450 mean=430.294 std=0.777 | wA=0.35
- #03 obj=430.467 mean=430.315 std=0.757 | wA=0.25
- #04 obj=430.470 mean=430.312 std=0.791 | wA=0.40
- #05 obj=430.506 mean=430.357 std=0.743 | wA=0.20
- #06 obj=430.513 mean=430.351 std=0.809 | wA=0.45
- #07 obj=430.565 mean=430.418 std=0.733 | wA=0.15
- #08 obj=430.572 mean=430.406 std=0.829 | wA=0.50
- #01 obj=427.572 | CV_MAE=427.552 |
- #02 obj=427.581 | CV_MAE=427.561 |
- #03 obj=427.593 | CV_MAE=427.573 |
- …（更多见原prompt全文）

**后处理/校准/shift 相关（抽取含 shift 关键词的行）**
- [best shift config] {'obj': 427.57223969664295, 'cv_mae':
- [apply shifts] YES (cross-fit improvement met threshold)
- chronic shifts: {np.str_('DiabetesComp'): -1.449,

**提交回执片段（Submitting predictions / Score）**
- 🚀 Submitting predictions...
- ✅ Prediction submitted successfully!
- 📊 Score: 448.1754 (MAE)
- 📊 Score: 448.1754
- Paste back: (1) leaderboard MAE, (2) base/final OOF MAE,

**原始PROMPT开头摘录（前约120行，用于快速定位）**
```text
🚀 Submitting predictions...
✅ Prediction submitted successfully!
📊 Score: 448.1754 (MAE)
✅ Validation passed
✅ Submission successful!
   📊 Score: 448.1754
   📏 Metric: MAE
   ✔️  Validation: Passed
======================================
======================================
==================================
CODE 22 | Back to Code20 route: (2-model shallow 
CatBoost + strong reg + multi-seed + trimmed mean) + 
tiny LOW-PRIOR residual correction
Goal: improve LB from 449.0221 toward <448 WITHOUT 
changing the core recipe.
======================================
======================================
==================================
[load] reading CSVs...
[ed_cost_train] shape=(2000, 5) | cols=5 | mem=0.17 MB
[ed_cost_test] shape=(2000, 4) | cols=4 | mem=0.15 MB
[patients] shape=(4000, 5) | cols=5 | mem=0.49 MB
[target stats]
count     2000.00000
mean      3908.25191
std       1822.40192
min        306.88000
25%       2548.76750
50%       3569.09500
75%       4956.42250
max      11184.61000
[target tail counts]
  y<500: 1
  y<800: 9
  y<1000: 21
  y<1500: 83
[admissions] building aggregates...
  admissions features: ((4000, 4), ['patient_id', 
'charlson_max', 'charlson_mean', 'pct_emergent'])
[receipts] building low-dim receipt features...
  receipt_feat shape: (4000, 45) | n_features=44
  Receipt rcpt_sum_items vs prior_ed_cost_5y_usd 
match_rate(train): 1.0000
  Receipt rcpt_sum_items vs prior_ed_cost_5y_usd 
match_rate(test):  1.0000
[features] building train/test feature frames...
  FULL feature count:   64
  PRUNED feature count: 49
  PRUNED head: ['prior_ed_visits_5y', 
'prior_ed_cost_5y_usd', 'prior_cost_cap20k', 
'sqrt_prior_cost', 'log_prior_cost', 'log_prior_cost_cap20k', 
'cost_per_visit', 'log_visits', 'baseline_next3y', 
'chronic_encoded', 'age', 'sex_encoded', 
'insurance_encoded', 'zip_region', 'ins_x_chronic', 
'charlson_max', 'charlson_mean', 'pct_emergent', 
'cost_per_em', 'cost_hhi', 'pct_cost_em', 
'pct_cost_procedure', 'pct_cost_critical', 
'pct_cost_high_acuity', 'pct_cost_imaging']
[training] CatBoost CPU | shallow | rsm=0.8 | 
subsample=0.8 | multi-seed | 7-fold
Models: ['A_full_d5', 'B_pruned_d4']
Seeds=5, Folds=7
  Seed 1/5 OOF MAE: A_full_d5=429.99 | 
B_pruned_d4=430.59
  Seed 2/5 OOF MAE: A_full_d5=431.99 | 
B_pruned_d4=429.53
  Seed 3/5 OOF MAE: A_full_d5=432.29 | 
B_pruned_d4=431.26
  Seed 4/5 OOF MAE: A_full_d5=432.93 | 
B_pruned_d4=431.48
  Seed 5/5 OOF MAE: A_full_d5=432.91 | 
B_pruned_d4=430.97
[seed-aggregated OOF MAE per model] (trimmed mean)
  A_full_d5   : 429.34
  B_pruned_d4 : 427.41
[median best_iteration] (ref)
  A_full_d5   : 550
  B_pruned_d4 : 670
[ensemble search] Top 8 (obj=mean+0.2*std):
  #01 obj=430.448 mean=430.294 std=0.769 | wA=0.30 
wB=0.70
  #02 obj=430.450 mean=430.294 std=0.777 | wA=0.35 
wB=0.65
  #03 obj=430.467 mean=430.315 std=0.757 | wA=0.25 
wB=0.75
  #04 obj=430.470 mean=430.312 std=0.791 | wA=0.40 
wB=0.60
  #05 obj=430.506 mean=430.357 std=0.743 | wA=0.20 
wB=0.80
  #06 obj=430.513 mean=430.351 std=0.809 | wA=0.45 
wB=0.55
  #07 obj=430.565 mean=430.418 std=0.733 | wA=0.15 
wB=0.85
  #08 obj=430.572 mean=430.406 std=0.829 | wA=0.50 
wB=0.50
[chosen ensemble] {'wA': 0.3, 'wB': 0.7, 'mean_mae': 
430.29380457938913, 'std_mae': 0.7691922876588294, 
'obj': 430.4476430369209}
[base ensemble]
  OOF MAE: 427.634
  OOF pred quantiles: {0: 1081.9608387261962, 0.01: 
1383.9058491927785, 0.05: 1754.446898692773, 0.1: 
2018.9569322894267, 0.5: 3510.0374493865106, 0.9: 
6513.432934013248, 0.95: 7352.056240102546, 0.99: 
8644.928656440043, 1.0: 10260.955320849702}
  TEST pred quantiles: {0: 1107.6031717188569, 0.01: 
1357.736476943465, 0.05: 1690.225051097339, 0.1: 
1979.4051340121348, 0.5: 3556.1582628737183, 0.9: 
6473.30726747712, 0.95: 7430.015214241114, 0.99: 
8791.600535579095, 1.0: 10393.167138033632}
[low-prior group]
  threshold prior_ed_cost_5y_usd <= 450.00
  train low-prior rate=0.100 | n=200
  test  low-prior rate=0.103 | n=206
[rare high-acuity group]
  definition: skipped (no threshold in [3,2,1] gives rate 
5%-20%)
  train high-acuity rate=0.000 | n=0
  test  high-acuity rate=0.000 | n=0
```

---

### R08 | PROMPT 13

- 标识: **CODE 23**
- Leaderboard MAE: **449.2911**
- 相对上一轮变化: **+1.1157**（负数=提升）
- 标题/描述: Code22++: same core, but ensemble weight

**关键配置（从日志抽取）**
- CatBoost Task: CPU
- Seeds: 5
- Folds: 7
- RSM: 0.8
- Subsample: 0.8

**数据/特征摘要（从日志抽取）**
- ed_cost_train shape: 2000, 5
- ed_cost_test shape: 2000, 4
- patients shape: 4000, 5
- receipt_feat shape: 4000, 45
- FULL feature count: 64
- PRUNED feature count: 49

**OOF（逐seed）**
- Seed 1/5: A_full=429.97 | B_pruned=428.65
- Seed 2/5: A_full=431.88 | B_pruned=433.20
- Seed 3/5: A_full=430.61 | B_pruned=430.96
- Seed 4/5: A_full=434.04 | B_pruned=432.45
- Seed 5/5: A_full=428.83 | B_pruned=431.97

**OOF（seed聚合）**
- [seed-aggregated OOF MAE per model] (trimmed mean)
- A_full  : 428.84
- B_pruned: 428.59

**Ensemble 权重搜索（Top 片段）**
- [ensemble search | ALIGNED] Top 10 (obj = trimmedMAE +
- #01 obj=428.302 | trimmedMAE=428.226 |
- seed_std=1.518 | wA=0.35 wB=0.65
- #02 obj=428.312 | trimmedMAE=428.237 |
- seed_std=1.519 | wA=0.40 wB=0.60
- #03 obj=428.327 | trimmedMAE=428.251 |
- seed_std=1.521 | wA=0.30 wB=0.70
- #04 obj=428.348 | trimmedMAE=428.272 |
- seed_std=1.522 | wA=0.45 wB=0.55
- #05 obj=428.366 | trimmedMAE=428.290 |
- seed_std=1.527 | wA=0.25 wB=0.75
- #06 obj=428.399 | trimmedMAE=428.323 |
- …（更多见原prompt全文）

**后处理/校准/shift 相关（抽取含 shift 关键词的行）**
- [best shift config] {'obj': 428.2263545459559, 'cv_mae':
- [apply shifts] NO (improvement too small)

**提交回执片段（Submitting predictions / Score）**
- 🚀 Submitting predictions...
- ✅ Prediction submitted successfully!
- 📊 Score: 449.2911 (MAE)
- 📊 Score: 449.2911
- Paste back: (1) leaderboard MAE, (2) base/final OOF MAE,

**原始PROMPT开头摘录（前约120行，用于快速定位）**
```text
here is the real mae:
🚀 Submitting predictions...
✅ Prediction submitted successfully!
📊 Score: 449.2911 (MAE)
✅ Validation passed
✅ Submission successful!
   📊 Score: 449.2911
   📏 Metric: MAE
   ✔️  Validation: Passed
and here is our log:
======================================
======================================
==================================
CODE 23 | Code22++: same core, but ensemble weight 
selection uses FINAL trimmed-mean OOF MAE (aligned to 
inference aggregator)
Aim: push LB below 448.1754 with minimal risk (less-is-
more).
======================================
======================================
==================================
[load] reading CSVs...
[ed_cost_train] shape=(2000, 5) | cols=5 | mem=0.17 MB
[ed_cost_test] shape=(2000, 4) | cols=4 | mem=0.15 MB
[patients] shape=(4000, 5) | cols=5 | mem=0.49 MB
[target stats]
count     2000.00000
mean      3908.25191
std       1822.40192
min        306.88000
25%       2548.76750
50%       3569.09500
75%       4956.42250
max      11184.61000
[admissions] building aggregates...
  admissions features: ((4000, 4), ['patient_id', 
'charlson_max', 'charlson_mean', 'pct_emergent'])
[receipts] building low-dim receipt features...
  receipt_feat shape: (4000, 45) | n_features=44
[features] building train/test feature frames...
  Receipt rcpt_sum_items vs prior_ed_cost_5y_usd 
match_rate(train): 1.0000
  Receipt rcpt_sum_items vs prior_ed_cost_5y_usd 
match_rate(test):  1.0000
  FULL feature count:   64
  PRUNED feature count: 49
[training] CatBoost CPU | shallow | rsm=0.8 | 
subsample=0.8 | multi-seed | 7-fold
Models: ['A_full', 'B_pruned']
Seeds=5, Folds=7
  Seed 1/5 OOF MAE: A_full=429.97 | B_pruned=428.65
  Seed 2/5 OOF MAE: A_full=431.88 | B_pruned=433.20
  Seed 3/5 OOF MAE: A_full=430.61 | B_pruned=430.96
  Seed 4/5 OOF MAE: A_full=434.04 | B_pruned=432.45
  Seed 5/5 OOF MAE: A_full=428.83 | B_pruned=431.97
[seed-aggregated OOF MAE per model] (trimmed mean)
  A_full  : 428.84
  B_pruned: 428.59
[median best_iteration] (ref)
  A_full  : 589
  B_pruned: 703
[ensemble search | ALIGNED] Top 10 (obj = trimmedMAE + 
0.05*seed_std):
  #01 obj=428.302 | trimmedMAE=428.226 | 
seed_std=1.518 | wA=0.35 wB=0.65
  #02 obj=428.312 | trimmedMAE=428.237 | 
seed_std=1.519 | wA=0.40 wB=0.60
  #03 obj=428.327 | trimmedMAE=428.251 | 
seed_std=1.521 | wA=0.30 wB=0.70
  #04 obj=428.348 | trimmedMAE=428.272 | 
seed_std=1.522 | wA=0.45 wB=0.55
  #05 obj=428.366 | trimmedMAE=428.290 | 
seed_std=1.527 | wA=0.25 wB=0.75
  #06 obj=428.399 | trimmedMAE=428.323 | 
seed_std=1.533 | wA=0.50 wB=0.50
  #07 obj=428.412 | trimmedMAE=428.335 | 
seed_std=1.534 | wA=0.20 wB=0.80
  #08 obj=428.462 | trimmedMAE=428.385 | 
seed_std=1.546 | wA=0.55 wB=0.45
  #09 obj=428.463 | trimmedMAE=428.386 | 
seed_std=1.542 | wA=0.15 wB=0.85
  #10 obj=428.503 | trimmedMAE=428.425 | 
seed_std=1.561 | wA=0.60 wB=0.40
[chosen ensemble | ALIGNED] {'wA': 0.35, 'wB': 0.65, 
'trimmed_mae': 428.2263545459559, 'seed_std': 
1.5177810792522253, 'obj': 428.30224359991854}
[base ensemble]
  OOF MAE: 428.226
  OOF quantiles: {0: 1078.1590095100817, 0.01: 
1384.8675562293943, 0.05: 1755.6734630759036, 0.1: 
2011.2735553393927, 0.5: 3501.0155837159095, 0.9: 
6500.621140593186, 0.95: 7343.29136750545, 0.99: 
8649.541819180851, 1.0: 10163.004452110501}
  TEST quantiles: {0: 1103.5472171402307, 0.01: 
1353.0976878233002, 0.05: 1692.6758934131626, 0.1: 
1981.1975653553382, 0.5: 3558.1812441214856, 0.9: 
6470.967386014974, 0.95: 7434.276151816874, 0.99: 
8796.52899814398, 1.0: 10375.960832121951}
[correction groups]
  low-prior thr prior_ed_cost_5y_usd <= 450.00 | 
train_rate=0.100 n=200 | test_rate=0.103 n=206
  rare-highacu: skipped (no threshold in [3,2,1] gives rate 
5%-20%) | train_rate=0.000 n=0 | test_rate=0.000 n=0
[shift tuning] Top 12 (objective = CV_MAE + simplicity 
penalties):
  #01 obj=428.226 | CV_MAE=428.226 | 
chronic_factor=0.00 lowpr_factor=0.00 
highacu_factor=0.00
  #02 obj=428.234 | CV_MAE=428.214 | 
chronic_factor=0.25 lowpr_factor=0.00 
highacu_factor=0.00
  #03 obj=428.236 | CV_MAE=428.216 | 
chronic_factor=0.35 lowpr_factor=0.00 
highacu_factor=0.00
  #04 obj=428.237 | CV_MAE=428.217 | 
chronic_factor=0.15 lowpr_factor=0.00 
highacu_factor=0.00
  #05 obj=428.242 | CV_MAE=428.222 | 
chronic_factor=0.45 lowpr_factor=0.00 
highacu_factor=0.00
```

---

### R09 | PROMPT 14

- 标识: **CODE 24**
- Leaderboard MAE: **448.1754**
- 相对上一轮变化: **-1.1157**（负数=提升）
- 标题/描述: Code22-core + (NEW) low-util baseline blend
- Plan: build low-dim features -> 7-fold x 5-seed CatBoost(A

**关键配置（从日志抽取）**
- CatBoost Task: CPU
- Seeds: 5
- Folds: 7
- RSM: 0.8
- Subsample: 0.8

**数据/特征摘要（从日志抽取）**
- ed_cost_train shape: 2000, 5
- ed_cost_test shape: 2000, 4
- patients shape: 4000, 5
- receipt_feat shape: 4000, 45
- FULL feature count: 64
- PRUNED feature count: 49

**OOF（逐seed）**
- Seed 1/5: A_full_d5=429.99 |
- Seed 2/5: A_full_d5=431.99 |
- Seed 3/5: A_full_d5=432.29 |
- Seed 4/5: A_full_d5=432.93 |
- Seed 5/5: A_full_d5=432.91 |

**OOF（seed聚合）**
- [seed-aggregated OOF MAE per model] (trimmed mean)
- A_full_d5   : 429.34
- B_pruned_d4 : 427.41

**Ensemble 权重搜索（Top 片段）**
- [ensemble search] Top 8 (obj=mean+0.2*std):
- #01 obj=430.448 mean=430.294 std=0.769 | wA=0.30
- #02 obj=430.450 mean=430.294 std=0.777 | wA=0.35
- #03 obj=430.467 mean=430.315 std=0.757 | wA=0.25
- #04 obj=430.470 mean=430.312 std=0.791 | wA=0.40
- #05 obj=430.506 mean=430.357 std=0.743 | wA=0.20
- #06 obj=430.513 mean=430.351 std=0.809 | wA=0.45
- #07 obj=430.565 mean=430.418 std=0.733 | wA=0.15
- #08 obj=430.572 mean=430.406 std=0.829 | wA=0.50
- #01 obj=427.589 | CV_MAE=427.569 |
- #02 obj=427.590 | CV_MAE=427.570 |
- #03 obj=427.591 | CV_MAE=427.571 |
- …（更多见原prompt全文）

**后处理/校准/shift 相关（抽取含 shift 关键词的行）**
- cross-fitted tiny corrections: chronic shift + optional
- chronic shifts: {np.str_('DiabetesComp'): -1.449,

**提交回执片段（Submitting predictions / Score）**
- 🚀 Submitting predictions...
- ✅ Prediction submitted successfully!
- 📊 Score: 448.1754 (MAE)
- 📊 Score: 448.1754
- Paste back: (1) leaderboard MAE, (2) Base OOF MAE, (3)

**原始PROMPT开头摘录（前约120行，用于快速定位）**
```text
🚀 Submitting predictions...
✅ Prediction submitted successfully!
📊 Score: 448.1754 (MAE)
✅ Validation passed
✅ Submission successful!
   📊 Score: 448.1754
   📏 Metric: MAE
   ✔️  Validation: Passed
======================================
======================================
==================================
CODE 24 | Code22-core + (NEW) low-util baseline blend 
correction (cross-fitted).
Plan: build low-dim features -> 7-fold x 5-seed CatBoost(A 
full d5 + B pruned d4) -> trimmed-mean -> w-search ->
      cross-fitted tiny corrections: chronic shift + optional 
low-util blend-to-baseline -> train full -> submission.csv
======================================
======================================
==================================
[load] reading CSVs...
[ed_cost_train] shape=(2000, 5) | cols=5 | mem=0.17 MB
[ed_cost_test] shape=(2000, 4) | cols=4 | mem=0.15 MB
[patients] shape=(4000, 5) | cols=5 | mem=0.49 MB
[target stats]
count     2000.00000
mean      3908.25191
std       1822.40192
min        306.88000
25%       2548.76750
50%       3569.09500
75%       4956.42250
max      11184.61000
[admissions] building aggregates...
  admissions features: ((4000, 4), ['patient_id', 
'charlson_max', 'charlson_mean', 'pct_emergent'])
[receipts] building low-dim receipt features...
  receipt_feat shape: (4000, 45) | n_features=44
[features] building train/test feature frames...
  Receipt rcpt_sum_items vs prior_ed_cost_5y_usd 
match_rate(train): 1.0000
  Receipt rcpt_sum_items vs prior_ed_cost_5y_usd 
match_rate(test):  1.0000
  FULL feature count:   64
  PRUNED feature count: 49
[training] CatBoost CPU | depth(5/4) | rsm=0.8 | 
subsample=0.8 | multi-seed | 7-fold
Models: ['A_full_d5', 'B_pruned_d4']
Seeds=5, Folds=7
  Seed 1/5 OOF MAE: A_full_d5=429.99 | 
B_pruned_d4=430.59
  Seed 2/5 OOF MAE: A_full_d5=431.99 | 
B_pruned_d4=429.53
  Seed 3/5 OOF MAE: A_full_d5=432.29 | 
B_pruned_d4=431.26
  Seed 4/5 OOF MAE: A_full_d5=432.93 | 
B_pruned_d4=431.48
  Seed 5/5 OOF MAE: A_full_d5=432.91 | 
B_pruned_d4=430.97
[seed-aggregated OOF MAE per model] (trimmed mean)
  A_full_d5   : 429.34
  B_pruned_d4 : 427.41
[median best_iteration] (ref)
  A_full_d5   : 550
  B_pruned_d4 : 670
[ensemble search] Top 8 (obj=mean+0.2*std):
  #01 obj=430.448 mean=430.294 std=0.769 | wA=0.30 
wB=0.70
  #02 obj=430.450 mean=430.294 std=0.777 | wA=0.35 
wB=0.65
  #03 obj=430.467 mean=430.315 std=0.757 | wA=0.25 
wB=0.75
  #04 obj=430.470 mean=430.312 std=0.791 | wA=0.40 
wB=0.60
  #05 obj=430.506 mean=430.357 std=0.743 | wA=0.20 
wB=0.80
  #06 obj=430.513 mean=430.351 std=0.809 | wA=0.45 
wB=0.55
  #07 obj=430.565 mean=430.418 std=0.733 | wA=0.15 
wB=0.85
  #08 obj=430.572 mean=430.406 std=0.829 | wA=0.50 
wB=0.50
[chosen ensemble] {'wA': 0.3, 'wB': 0.7, 'mean_mae': 
430.29380457938913, 'std_mae': 0.7691922876588294, 
'obj': 430.4476430369209}
[base ensemble]
  OOF MAE: 427.634
  OOF quantiles: {0: 1081.9608387261962, 0.01: 
1383.9058491927785, 0.05: 1754.446898692773, 0.1: 
2018.9569322894267, 0.5: 3510.0374493865106, 0.9: 
6513.432934013248, 0.95: 7352.056240102546, 0.99: 
8644.928656440043, 1.0: 10260.955320849702}
  TEST quantiles: {0: 1107.6031717188569, 0.01: 
1357.736476943465, 0.05: 1690.225051097339, 0.1: 
1979.4051340121348, 0.5: 3556.1582628737183, 0.9: 
6473.30726747712, 0.95: 7430.015214241114, 0.99: 
8791.600535579095, 1.0: 10393.167138033632}
[low-util group]
  definition: prior_cost<=q10(450.00) AND prior_visits<=1
  train low-util rate=0.088 | n=176
  test  low-util rate=0.089 | n=179
[correction search] Top 12 (obj = CV_MAE + penalties):
  #01 obj=427.589 | CV_MAE=427.569 | 
chronic_factor=0.55 | lowutil_lam=0.00
  #02 obj=427.590 | CV_MAE=427.570 | 
chronic_factor=0.65 | lowutil_lam=0.00
  #03 obj=427.591 | CV_MAE=427.571 | 
chronic_factor=0.75 | lowutil_lam=0.00
  #04 obj=427.594 | CV_MAE=427.574 | 
chronic_factor=0.45 | lowutil_lam=0.00
  #05 obj=427.600 | CV_MAE=427.580 | 
chronic_factor=0.35 | lowutil_lam=0.00
  #06 obj=427.611 | CV_MAE=427.591 | 
chronic_factor=0.25 | lowutil_lam=0.00
  #07 obj=427.625 | CV_MAE=427.605 | 
chronic_factor=0.15 | lowutil_lam=0.00
  #08 obj=427.634 | CV_MAE=427.634 | 
chronic_factor=0.00 | lowutil_lam=0.00
  #09 obj=428.933 | CV_MAE=428.851 | 
chronic_factor=0.45 | lowutil_lam=0.10
```

---

### R10 | PROMPT 15

- 标识: **CODE 25**
- Leaderboard MAE: **448.1210**
- 相对上一轮变化: **-0.0544**（负数=提升）
- 标题/描述: Code22-core + one-SE (prefer smaller A weight)
- Goal: beat LB 448.1754 by slightly reducing overfit to A_full

**关键配置（从日志抽取）**
- CatBoost Task: CPU
- Seeds: 5
- Folds: 7
- RSM: 0.8
- Subsample: 0.8

**数据/特征摘要（从日志抽取）**
- ed_cost_train shape: 2000, 5
- ed_cost_test shape: 2000, 4
- patients shape: 4000, 5
- receipt_feat shape: 4000, 45
- FULL feature count: 64
- PRUNED feature count: 49

**OOF（逐seed）**
- Seed 1/5: A_full_d5=429.99 |
- Seed 2/5: A_full_d5=431.99 |
- Seed 3/5: A_full_d5=432.29 |
- Seed 4/5: A_full_d5=432.93 |
- Seed 5/5: A_full_d5=432.91 |

**OOF（seed聚合）**
- [seed-aggregated OOF MAE per model] (trimmed mean)
- A_full_d5   : 429.34
- B_pruned_d4 : 427.41

**Ensemble 权重搜索（Top 片段）**
- [ensemble search] Top 10 (obj=mean+0.2*std):
- #01 obj=430.448 mean=430.294 std=0.769 | wA=0.30
- #02 obj=430.450 mean=430.294 std=0.777 | wA=0.35
- #03 obj=430.467 mean=430.315 std=0.757 | wA=0.25
- #04 obj=430.470 mean=430.312 std=0.791 | wA=0.40
- #05 obj=430.506 mean=430.357 std=0.743 | wA=0.20
- #06 obj=430.513 mean=430.351 std=0.809 | wA=0.45
- #07 obj=430.565 mean=430.418 std=0.733 | wA=0.15
- #08 obj=430.572 mean=430.406 std=0.829 | wA=0.50
- #09 obj=430.649 mean=430.505 std=0.719 | wA=0.10
- #10 obj=430.652 mean=430.482 std=0.850 | wA=0.55
- best:    obj=430.448 | wA=0.30 wB=0.70
- …（更多见原prompt全文）

**后处理/校准/shift 相关（抽取含 shift 关键词的行）**
- [chronic shift tuning] Top 10 (obj=CV_MAE+penalty):
- [best chronic shift] {'base_mae': 427.53798583390187, 'obj':
- [apply chronic shift] YES
- chronic shifts: {np.str_('DiabetesComp'): -3.781,
- (3) chosen one-SE weight + bag_list, (4) chronic shift meta.

**提交回执片段（Submitting predictions / Score）**
- the real MAE is :
- 🚀 Submitting predictions...
- ✅ Prediction submitted successfully!
- 📊 Score: 448.1210 (MAE)
- 📊 Score: 448.1210
- Paste back: (1) leaderboard MAE, (2) base/final OOF MAE,

**原始PROMPT开头摘录（前约120行，用于快速定位）**
```text
the real MAE is :
🚀 Submitting predictions...
✅ Prediction submitted successfully!
📊 Score: 448.1210 (MAE)
✅ Validation passed
✅ Submission successful!
   📊 Score: 448.1210
   📏 Metric: MAE
   ✔️  Validation: Passed
and our log:
======================================
======================================
==================================
CODE 25 | Code22-core + one-SE (prefer smaller A weight) 
+ tiny weight-bagging around chosen weight.
Goal: beat LB 448.1754 by slightly reducing overfit to A_full 
while keeping diversity.
======================================
======================================
==================================
[load] reading CSVs...
[ed_cost_train] shape=(2000, 5) | cols=5 | mem=0.17 MB
[ed_cost_test] shape=(2000, 4) | cols=4 | mem=0.15 MB
[patients] shape=(4000, 5) | cols=5 | mem=0.49 MB
[target stats]
count     2000.00000
mean      3908.25191
std       1822.40192
min        306.88000
25%       2548.76750
50%       3569.09500
75%       4956.42250
max      11184.61000
[admissions] building aggregates...
  admissions features: ((4000, 4), ['patient_id', 
'charlson_max', 'charlson_mean', 'pct_emergent'])
[receipts] building low-dim receipt features...
  receipt_feat shape: (4000, 45) | n_features=44
[features] building train/test feature frames...
  Receipt rcpt_sum_items vs prior_ed_cost_5y_usd 
match_rate(train): 1.0000
  Receipt rcpt_sum_items vs prior_ed_cost_5y_usd 
match_rate(test):  1.0000
  FULL feature count:   64
  PRUNED feature count: 49
[training] CatBoost CPU | depth(5/4) | rsm=0.8 | 
subsample=0.8 | multi-seed | 7-fold
Models: ['A_full_d5', 'B_pruned_d4']
Seeds=5, Folds=7
  Seed 1/5 OOF MAE: A_full_d5=429.99 | 
B_pruned_d4=430.59
  Seed 2/5 OOF MAE: A_full_d5=431.99 | 
B_pruned_d4=429.53
  Seed 3/5 OOF MAE: A_full_d5=432.29 | 
B_pruned_d4=431.26
  Seed 4/5 OOF MAE: A_full_d5=432.93 | 
B_pruned_d4=431.48
  Seed 5/5 OOF MAE: A_full_d5=432.91 | 
B_pruned_d4=430.97
[seed-aggregated OOF MAE per model] (trimmed mean)
  A_full_d5   : 429.34
  B_pruned_d4 : 427.41
[median best_iteration] (ref)
  A_full_d5   : 550
  B_pruned_d4 : 670
[ensemble search] Top 10 (obj=mean+0.2*std):
  #01 obj=430.448 mean=430.294 std=0.769 | wA=0.30 
wB=0.70
  #02 obj=430.450 mean=430.294 std=0.777 | wA=0.35 
wB=0.65
  #03 obj=430.467 mean=430.315 std=0.757 | wA=0.25 
wB=0.75
  #04 obj=430.470 mean=430.312 std=0.791 | wA=0.40 
wB=0.60
  #05 obj=430.506 mean=430.357 std=0.743 | wA=0.20 
wB=0.80
  #06 obj=430.513 mean=430.351 std=0.809 | wA=0.45 
wB=0.55
  #07 obj=430.565 mean=430.418 std=0.733 | wA=0.15 
wB=0.85
  #08 obj=430.572 mean=430.406 std=0.829 | wA=0.50 
wB=0.50
  #09 obj=430.649 mean=430.505 std=0.719 | wA=0.10 
wB=0.90
  #10 obj=430.652 mean=430.482 std=0.850 | wA=0.55 
wB=0.45
[one-SE selection] best vs simplest-within-tol
  best:    obj=430.448 | wA=0.30 wB=0.70
  chosen:  obj=430.506 | wA=0.20 wB=0.80 | tol=0.08
[weight-bagging] weights from 0.20 to 0.30 step=0.05 -> 
[0.2, 0.25, 0.3]
[base ensemble after weight-bagging]
  per-weight OOF MAE (trimmed-mean): {0.2: 427.458, 
0.25: 427.53, 0.3: 427.634}
  bagged OOF MAE: 427.538
  OOF quantiles: {0: 1080.8602566802203, 0.01: 
1386.075915620909, 0.05: 1752.008245362574, 0.1: 
2018.5749328698655, 0.5: 3510.3284861244288, 0.9: 
6515.432527367393, 0.95: 7351.625218384794, 0.99: 
8639.34917537169, 1.0: 10261.113835591837}
  TEST quantiles: {0: 1105.4116223603114, 0.01: 
1357.275788145009, 0.05: 1690.2921768217097, 0.1: 
1978.5752143232457, 0.5: 3556.2140722550976, 0.9: 
6474.297541661167, 0.95: 7429.40853370033, 0.99: 
8793.601040169602, 1.0: 10400.126362759329}
[chronic shift tuning] Top 10 (obj=CV_MAE+penalty):
  #01 obj=427.452 | CV_MAE=427.432 | 
chronic_factor=0.75
  #02 obj=427.459 | CV_MAE=427.439 | 
chronic_factor=0.65
  #03 obj=427.469 | CV_MAE=427.449 | 
chronic_factor=0.55
  #04 obj=427.480 | CV_MAE=427.460 | 
chronic_factor=0.45
  #05 obj=427.491 | CV_MAE=427.471 | 
chronic_factor=0.35
  #06 obj=427.505 | CV_MAE=427.485 | 
chronic_factor=0.25
  #07 obj=427.524 | CV_MAE=427.504 | 
chronic_factor=0.15
```

---

### R11 | PROMPT 16

- 标识: **CODE 26**
- Leaderboard MAE: **448.1413**
- 相对上一轮变化: **+0.0203**（负数=提升）
- 标题/描述: Code22/25 core + NEW stacking (median/MAE-

**关键配置（从日志抽取）**
- CatBoost Task: CPU
- Seeds: 5
- Folds: 7
- RSM: 0.8
- Subsample: 0.8

**数据/特征摘要（从日志抽取）**
- ed_cost_train shape: 2000, 5
- ed_cost_test shape: 2000, 4
- patients shape: 4000, 5
- receipt_feat shape: 4000, 45
- FULL feature count: 64
- PRUNED feature count: 49

**OOF（逐seed）**
- Seed 1/5: A_full_d5=429.99 |
- Seed 2/5: A_full_d5=431.99 |
- Seed 3/5: A_full_d5=432.29 |
- Seed 4/5: A_full_d5=432.93 |
- Seed 5/5: A_full_d5=432.91 |

**OOF（seed聚合）**
- [seed-aggregated OOF MAE per model] (trimmed mean)
- A_full_d5   : 429.34
- B_pruned_d4 : 427.41

**后处理/校准/shift 相关（抽取含 shift 关键词的行）**
- global_shift_factor=0.00 | chronic_factor=1.00
- global_shift_factor=0.00 | chronic_factor=0.85
- global_shift_factor=0.00 | chronic_factor=0.75
- global_shift_factor=0.00 | chronic_factor=0.65
- global_shift_factor=0.00 | chronic_factor=0.55
- global_shift_factor=0.50 | chronic_factor=1.00
- global_shift_factor=0.00 | chronic_factor=0.45
- global_shift_factor=0.50 | chronic_factor=0.85
- global_shift_factor=0.00 | chronic_factor=0.35
- global_shift_factor=0.50 | chronic_factor=0.75
- global_shift_factor=1.00 | chronic_factor=1.00
- global_shift_factor=0.00 | chronic_factor=0.25
- chronic_shifts: {np.str_('DiabetesComp'): -5.099,

**提交回执片段（Submitting predictions / Score）**
- 🚀 Submitting predictions...
- ✅ Prediction submitted successfully!
- 📊 Score: 448.1413 (MAE)
- 📊 Score: 448.1413

**原始PROMPT开头摘录（前约120行，用于快速定位）**
```text
======================================
======================================
======================================
======
CODE 26 | Code22/25 core + NEW stacking (median/MAE-
aligned QuantileRegressor) + conservative blend + 
global+chronic correction
Aim: explore a new 'diverse but simple' path to push LB 
beyond ~448.12 toward <447.
======================================
======================================
======================================
======
[load] reading CSVs...
[ed_cost_train] shape=(2000, 5) | cols=5 | mem=0.17 MB
[ed_cost_test] shape=(2000, 4) | cols=4 | mem=0.15 MB
[patients] shape=(4000, 5) | cols=5 | mem=0.49 MB
[target stats]
count     2000.00000
mean      3908.25191
std       1822.40192
min        306.88000
25%       2548.76750
50%       3569.09500
75%       4956.42250
max      11184.61000
[admissions] building aggregates...
  admissions features: ((4000, 4), ['patient_id', 
'charlson_max', 'charlson_mean', 'pct_emergent'])
[receipts] building low-dim receipt features...
  receipt_feat shape: (4000, 45) | n_features=44
[features] building train/test feature frames...
  Receipt rcpt_sum_items vs prior_ed_cost_5y_usd 
match_rate(train): 1.0000
  Receipt rcpt_sum_items vs prior_ed_cost_5y_usd 
match_rate(test):  1.0000
  FULL feature count:   64
  PRUNED feature count: 49
[training] CatBoost CPU | depth(5/4) | rsm=0.8 | 
subsample=0.8 | multi-seed | 7-fold
Models: ['A_full_d5', 'B_pruned_d4']
Seeds=5, Folds=7
  Seed 1/5 OOF MAE: A_full_d5=429.99 | 
B_pruned_d4=430.59
  Seed 2/5 OOF MAE: A_full_d5=431.99 | 
B_pruned_d4=429.53
  Seed 3/5 OOF MAE: A_full_d5=432.29 | 
B_pruned_d4=431.26
  Seed 4/5 OOF MAE: A_full_d5=432.93 | 
B_pruned_d4=431.48
  Seed 5/5 OOF MAE: A_full_d5=432.91 | 
B_pruned_d4=430.97
[seed-aggregated OOF MAE per model] (trimmed mean)
  A_full_d5   : 429.34
  B_pruned_d4 : 427.41
[median best_iteration] (ref)
  A_full_d5   : 550
  B_pruned_d4 : 670
[A/B weight search] Top 10 (obj=mean+0.2*std):
  #01 obj=430.448 mean=430.294 std=0.769 | wA=0.30 
wB=0.70
  #02 obj=430.450 mean=430.294 std=0.777 | wA=0.35 
wB=0.65
  #03 obj=430.467 mean=430.315 std=0.757 | wA=0.25 
wB=0.75
  #04 obj=430.470 mean=430.312 std=0.791 | wA=0.40 
wB=0.60
  #05 obj=430.506 mean=430.357 std=0.743 | wA=0.20 
wB=0.80
  #06 obj=430.513 mean=430.351 std=0.809 | wA=0.45 
wB=0.55
  #07 obj=430.565 mean=430.418 std=0.733 | wA=0.15 
wB=0.85
  #08 obj=430.572 mean=430.406 std=0.829 | wA=0.50 
wB=0.50
  #09 obj=430.649 mean=430.505 std=0.719 | wA=0.10 
wB=0.90
  #10 obj=430.652 mean=430.482 std=0.850 | wA=0.55 
wB=0.45
[one-SE selection] best vs chosen (simpler-within-tol)
  best:    obj=430.448 | wA=0.30 wB=0.70
  chosen:  obj=430.506 | wA=0.20 wB=0.80 | tol=0.10
[A/B bagging] wA from 0.20 to 0.35 step=0.05 -> [0.2, 
0.25, 0.3, 0.35]
[base A/B bagged]
  per-weight OOF MAE (trimmed): {0.2: 427.458, 0.25: 
427.53, 0.3: 427.634, 0.35: 427.739}
  bagged OOF MAE: 427.586
[stacking] Meta model: QuantileRegressor(q=0.5)
[stacking] alpha candidates (lower MAE better):
  alpha=1e-02 | CV_MAE=427.741
  alpha=3e-03 | CV_MAE=427.835
  alpha=1e-03 | CV_MAE=427.942
  alpha=3e-04 | CV_MAE=428.019
  alpha=1e-04 | CV_MAE=428.033
[stacking] best alpha=1e-02 (MAE=427.741) | chosen 
alpha=1e-02 within tol=0.08
[stack OOF MAE] 427.741 | stack_info={'alpha_best': 0.01, 
'alpha_chosen': 0.01, 'cv_mae_best': 427.7409481511412, 
'cv_mae_chosen': 427.7409481511412}
[blend bag+stack] Top 10 (obj = MAE + tiny_w_penalty):
  #01 obj=427.571 | MAE=427.567 | w_stack=0.20
  #02 obj=427.571 | MAE=427.566 | w_stack=0.25
  #03 obj=427.572 | MAE=427.566 | w_stack=0.30
  #04 obj=427.573 | MAE=427.570 | w_stack=0.15
  #05 obj=427.575 | MAE=427.568 | w_stack=0.35
  #06 obj=427.576 | MAE=427.574 | w_stack=0.10
  #07 obj=427.580 | MAE=427.579 | w_stack=0.05
  #08 obj=427.582 | MAE=427.574 | w_stack=0.40
  #09 obj=427.586 | MAE=427.586 | w_stack=0.00
  #10 obj=427.588 | MAE=427.579 | w_stack=0.45
[blend bag+stack] best vs chosen (smaller w within tol)
  best:   obj=427.571 | MAE=427.567 | w_stack=0.20
  chosen: obj=427.586 | MAE=427.586 | w_stack=0.00 | 
tol=0.05
[blend bag+stack] bag w_stack from 0.00 to 0.10 
step=0.05 -> [0.0, 0.05, 0.1]
[bag+stack blended]
  blended OOF MAE: 427.579
  OOF quantiles: {0: 1080.8020420387895, 0.01:
```

---

### R12 | PROMPT 17

- 标识: **CODE 27**
- Leaderboard MAE: **455.0833**
- 相对上一轮变化: **+6.9420**（负数=提升）
- 标题/描述: Bold attempt to break ~448: Code22/25 core +

**关键配置（从日志抽取）**
- CatBoost Task: CPU
- Seeds
- Folds: 7
- RSM
- Subsample

**数据/特征摘要（从日志抽取）**
- ed_cost_train shape: 2000, 5
- ed_cost_test shape: 2000, 4
- patients shape: 4000, 5
- receipt_feat shape: 4000, 53
- FULL feature count: 76
- PRUNED feature count: 53

**OOF（逐seed）**
- Seed 1/5: 432.926
- Seed 2/5: 434.223
- Seed 3/5: 433.380
- Seed 4/5: 435.612
- Seed 5/5: 434.011
- Seed 1/5: 434.188
- Seed 2/5: 434.190
- Seed 3/5: 431.726
- Seed 4/5: 432.816
- Seed 5/5: 432.295
- …（共 18 行，见原prompt全文）

**OOF（seed聚合）**
- [seed-aggregated OOF MAE] (lower is better)
- A_full_d5          : 430.772
- B_pruned_d4_plain  : 429.416
- B_pruned_d4_monoQ  : 1439.624
- C_cat_d4           : 440.401

**后处理/校准/shift 相关（抽取含 shift 关键词的行）**
- [correction] chronic median-residual shift (cross-fitted);
- best chronic shift: {'obj': 430.96656927541255, 'cv_mae':
- SKIP chronic shift (gain too small or cf=0).
- final OOF MAE (after optional scale + chronic shift):
- chronic shift applied: False | shift_info: {'obj':
- scale/chronic shift applied.

**提交回执片段（Submitting predictions / Score）**
- 🚀 Submitting predictions...
- ✅ Prediction submitted successfully!
- 📊 Score: 455.0833 (MAE)
- 📊 Score: 455.0833
- Paste back: (1) leaderboard MAE, (2) FINAL OOF MAE, (3) B

**原始PROMPT开头摘录（前约120行，用于快速定位）**
```text
🚀 Submitting predictions...
✅ Prediction submitted successfully!
📊 Score: 455.0833 (MAE)
✅ Validation passed
✅ Submission successful!
   📊 Score: 455.0833
   📏 Metric: MAE
   ✔️  Validation: Passed
======================================
======================================
====================================
CODE 27 | Bold attempt to break ~448: Code22/25 core + 
NEW categorical CatBoost (dx3 + top_code_grp) + 
optional monotone Quantile + optional RF(MAE).
Philosophy: Earn complexity -> add ONE new signal path, 
keep regularization strong, weight search uses fold-robust 
objective.
======================================
======================================
====================================
[load] reading CSVs...
[ed_cost_train] shape=(2000, 5) | cols=5 | mem=0.17 MB
[ed_cost_test] shape=(2000, 4) | cols=4 | mem=0.15 MB
[patients] shape=(4000, 5) | cols=5 | mem=0.49 MB
[target stats]
count     2000.00000
mean      3908.25191
std       1822.40192
min        306.88000
25%       2548.76750
50%       3569.09500
75%       4956.42250
max      11184.61000
[admissions] building aggregates (adds dx3_mode as 
categorical)...
  admissions features: (4000, 9) | cols=['patient_id', 'adm_n', 
'charlson_max', 'charlson_mean', 'pct_emergent', 
'adm_los_mean', 'adm_edvis6m_mean', 'dx3_mode', 
'dx3_nuniq']
[receipts] building low-dim receipt features 
(+top_code_grp categorical) from receipts_parsed.joblib ...
  receipt_feat shape: (4000, 53) | n_features=52
  Receipt rcpt_sum_items vs prior_ed_cost_5y_usd 
match_rate(train): 1.0000
  Receipt rcpt_sum_items vs prior_ed_cost_5y_usd 
match_rate(test):  1.0000
[features] building train/test feature frames...
  FULL feature count:   76
  PRUNED feature count: 53
[training] CatBoost CPU | shallow | strong reg | multi-seed 
CV
Folds=7 | Seeds_AB=5 | Seeds_CAT=3
Models: A_full_d5 (RMSE), B_pruned_d4 (RMSE), 
B_mono_d4 (Quantile+monotone), C_cat_d4 (MAE+cats)
[A_full_d5]
  Seed 1/5 OOF MAE: 432.926
  Seed 2/5 OOF MAE: 434.223
  Seed 3/5 OOF MAE: 433.380
  Seed 4/5 OOF MAE: 435.612
  Seed 5/5 OOF MAE: 434.011
[B_pruned_d4 | plain]
  Seed 1/5 OOF MAE: 434.188
  Seed 2/5 OOF MAE: 434.190
  Seed 3/5 OOF MAE: 431.726
  Seed 4/5 OOF MAE: 432.816
  Seed 5/5 OOF MAE: 432.295
[B_pruned_d4 | mono+quantile]
  Seed 1/5 OOF MAE: 1439.875
  Seed 2/5 OOF MAE: 1440.123
  Seed 3/5 OOF MAE: 1440.514
  Seed 4/5 OOF MAE: 1439.800
  Seed 5/5 OOF MAE: 1440.499
[C_cat_d4 | MAE + categorical dx3/top_code]
  Seed 1/3 OOF MAE: 441.495
  Seed 2/3 OOF MAE: 445.307
  Seed 3/3 OOF MAE: 442.556
[seed-aggregated OOF MAE] (lower is better)
  A_full_d5          : 430.772
  B_pruned_d4_plain  : 429.416
  B_pruned_d4_monoQ  : 1439.624
  C_cat_d4           : 440.401
[choose B] selected: B_plain  (plain=429.416, 
monoQ=1439.624)
[A/B weight search] (uses trimmed-mean across seeds of 
per-seed blend)
  Top 8 (obj2 = fold_obj + 0.02*wA):
   #01 obj2=431.519 | mean=429.418 std=17.508 | 
wA=0.00 wB=1.00
   #02 obj2=431.550 | mean=429.442 std=17.556 | 
wA=0.05 wB=0.95
   #03 obj2=431.582 | mean=429.468 std=17.597 | 
wA=0.10 wB=0.90
   #04 obj2=431.600 | mean=429.480 std=17.644 | 
wA=0.15 wB=0.85
   #05 obj2=431.626 | mean=429.499 std=17.691 | 
wA=0.20 wB=0.80
   #06 obj2=431.678 | mean=429.543 std=17.755 | 
wA=0.25 wB=0.75
   #07 obj2=431.735 | mean=429.591 std=17.815 | 
wA=0.30 wB=0.70
   #08 obj2=431.795 | mean=429.642 std=17.881 | 
wA=0.35 wB=0.65
[one-SE A/B selection]
  best   wA=0.00 (obj2=431.519)
  chosen wA=0.00 (obj2=431.519) | tol=0.10
[A/B bagging] bag_wA=[0.0, 0.05, 0.1]
  per-weight OOF MAE: {0.0: 429.416, 0.05: 429.44, 0.1: 
429.466}
  base A/B bagged OOF MAE: 429.436
[RF(MAE) diversity] training 7-fold 
RandomForestRegressor(absolute_error) on PRUNED 
numerics (single run)...
  RF OOF MAE: 467.025
[blend search] objective = fold_mean + std_pen*fold_std + 
penalties on wC/wRF (prefer small)
  Top 10 blends:
   #01 obj2=431.429 | mean=429.375 std=17.086 | 
base=0.90 C=0.10 RF=0.00
   #02 obj2=431.450 | mean=429.418 std=16.880 | 
base=0.85 C=0.15 RF=0.00
```

---
