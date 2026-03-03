---
source_pdf: "IEEE ICHI - LB优化与模型调整.pdf"
domain: "AgentDS Healthcare"
challenge: 2
task: "ED Cost Forecasting"
metric: "MAE (lower is better)"
prepared_date: "2026-02-19"
timezone: "America/Toronto"
---

# AgentDS Healthcare Ch2 — LB优化与模型调整（Iter7 → Iter8）迭代记录

## 0. 这份文档怎么用
- 目标：把这份 PDF 中的**用户 prompts + 两次关键迭代（Iter7/Iter8）的策略与实现**整理成可被更强 AI 继续“总结可复现流程/提炼通用 prompt/产出 paper 方法叙事”的结构化材料。
- 说明：**注意**：以下脚本来自 PDF 抽取，可能存在少量由于分页/断行导致的换行或字符丢失（尤其是某些长字符串/列表项）。用于“迭代证据/思路复盘/给更强 AI 进一步整理”完全足够；如果要直接运行，请优先以你本地 notebook/py 原始版本为准，或让后续 AI 先做一次“断行修复”。

---

## 1. 原始用户 Prompts（提取 + 清洗）

### P1（目标指令）
> 继续在这个451的基础上提升，争取超越450

### P2（跑完后要求回传的信息）
跑完你回我5个东西（我就能继续做下一次“<450决胜微调”）：
1) LB MAE（public/private 哪个都行，给具体数）
2) final OOF MAE
3) ensemble meta（weights / lam / shift）
4) calibration chosen（有没有选中 chronic×insurance 或加 charlson_bin）
5) 预测分位数（脚本已打印）

---

## 2. Iteration 7：Anti-overfit++（v3-spirit，LB oriented）

### 2.1 意图 / 核心思想（脚本头部注释要点）
- 仍然坚持 LOW-DIM receipts 特征（buckets + shares + HHI + EM stats + 关键 procedure flags）。
- CatBoost 浅树：depth 4–5，RSM=0.8，多 seed bagging。
- Ensemble 权重选择：用“跨 seeds 的稳定性（mean + std）”而不是只追 best OOF。
- 引入 ONE 个多样化模型：在 pruned 特征上用 MAE loss（metric-aligned 但仍强正则）。
- Ensemble search 中允许极小的 baseline_next3y shrink（小 lam 网格）。

### 2.2 输入 / 输出（Paths & IO）
- 数据目录：`D:\AgentDs\agent_ds_healthcare`
- 输入：
  - `ed_cost_train.csv`
  - `ed_cost_test.csv`
  - `patients.csv`
  - `admissions_train.csv`
  - `admissions_test.csv`
  - `receipts_parsed.joblib`（用于 receipts 低维特征；脚本注明 pdf 目录仅作 last resort，不解析）
- 输出：
  - `D:\AgentDs\agent_ds_healthcare\submission.csv`

### 2.3 CFG（训练与 ensemble search 超参）
- N_FOLDS: 7
- N_SEEDS: 5
- ITERS: 3000
- ES_ROUNDS: 120
- LR: 0.03
- RSM: 0.80
- W_STEP: 0.05
- LAM_GRID: [0.00, 0.03, 0.05, 0.08, 0.10, 0.15]
- SHIFT_GRID: [0.0, 0.5, 1.0]
- STD_PEN: 0.20
- LAM_PEN: 8.0
- SHIFT_PEN: 0.002

### 2.4 模型族（CatBoost，3 模型）
- A_RMSE_full_d5: loss_function="RMSE", eval_metric="MAE", depth=5, l2_leaf_reg=5, min_data_in_leaf=28, rsm=CFG.RSM, random_strength=1.0
- B_RMSE_pruned_d4: loss_function="RMSE", eval_metric="MAE", depth=4, l2_leaf_reg=4, min_data_in_leaf=32, rsm=CFG.RSM, random_strength=2.0
- C_MAE_pruned_d4: loss_function="MAE", eval_metric="MAE", depth=4, l2_leaf_reg=7, min_data_in_leaf=36, rsm=CFG.RSM, random_strength=1.5

### 2.5 特征工程（摘要）
- tabular priors：`prior_ed_visits_5y`、`prior_ed_cost_5y_usd` 及其变换（sqrt/log/cap）
- baseline：`baseline_next3y = prior_ed_cost_5y_usd * (3/5)`
- patients：`age`、`sex_encoded`、`insurance_encoded`、`zip_region`、`ins_x_chronic`
- admissions：`charlson_max/mean`、`pct_emergent`
- receipts（从 joblib lineitems 聚合的 low-dim features）：分桶 totals、counts、flags、share/HHI 等 + 少量交互项

### 2.6 PRUNED 特征集合（用于 pruned 模型与稳定性）
（从脚本 `pruned_candidates` 抽取并清洗）
- prior_ed_visits_5y
- prior_ed_cost_5y_usd
- sqrt_prior_cost
- log_prior_cost
- log_prior_cost_cap20k
- cost_per_visit
- log_visits
- baseline_next3y
- chronic_encoded
- age
- sex_encoded
- insurance_encoded
- zip_region
- ins_x_chronic
- charlson_max
- charlson_mean
- pct_emergent
- cost_per_em
- cost_hhi
- pct_cost_procedure
- pct_cost_critical
- pct_cost_high_acuity
- n_high_acuity_total
- has_critical_care
- has_99285
- max_em_level
- n_unique_codes
- logprior_x_pctcritical
- logprior_x_highacu
- cost_per_code

### 2.7 Ensemble search（稳定性导向）
- 权重网格：`W_STEP = 0.05`
- baseline blend：`LAM_GRID = [0, 0.03, 0.05, 0.08, 0.10, 0.15]`
- 全局 shift：`SHIFT_GRID = [0.0, 0.5, 1.0]`（基于 median residual 的缩放）
- 目标函数（LB-oriented）：
  - `obj = mean_MAE_across_seeds + STD_PEN*std_MAE_across_seeds + LAM_PEN*lam + SHIFT_PEN*|shift_mult|`
  - 其中 `STD_PEN=0.20`，`LAM_PEN=8.0`，`SHIFT_PEN=0.002`

### 2.8 可选后处理：chronic group shift（Iter7 版本）
- 思路：按 `primary_chronic` 的 median residual 做 shrink（例：0.3/0.5）
- 风险：使用全量 residual 直接算 shift 会有轻微“乐观偏差”（Iter8 明确要解决这一点）

---

## 3. Iteration 8：Tail+Calibration++（在 Iter7 上的升级）

### 3.1 升级点（脚本头部注释要点）
- 新增 ONE 个 tail-friendly 多样化模型：log1p 目标回归（预测后再 expm1 回到原尺度）。
- 将 Iter7 的 group shift 升级为 CROSS-FIT group-median calibration（避免“用自己的残差校准自己”的泄漏/乐观偏差）。
- 在不破坏低维策略的前提下，补充超稳定的 receipts / admissions 统计特征（n_line_items / top1_share / entropy / max_line_amt / n_adm 等）。

### 3.2 CFG（相比 Iter7 的关键变化）
- N_FOLDS: 5
- N_SEEDS: 6
- ITERS: 3000
- ES_ROUNDS: 140
- LR: 0.03
- RSM: 0.80
- BOOTSTRAP_TYPE: "Bernoulli"
- SUBSAMPLE: 0.85
- W_STEP: 0.05
- LAM_GRID: [0.00, 0.03, 0.05, 0.08, 0.10, 0.15]
- SHIFT_GRID: [0.0, 0.5, 1.0]
- STD_PEN: 0.20
- LAM_PEN: 8.0
- SHIFT_PEN: 0.002

### 3.3 模型族（CatBoost，4 模型）
- A_RMSE_full_d5: loss_function="RMSE", eval_metric="MAE", depth=5, l2_leaf_reg=5, min_data_in_leaf=28, bootstrap_type=CFG.BOOTSTRAP_TYPE, subsample=CFG.SUBSAMPLE, rsm=CFG.RSM, random_strength=1.0
- B_RMSE_pruned_d4: loss_function="RMSE", eval_metric="MAE", depth=4, l2_leaf_reg=4, min_data_in_leaf=32, bootstrap_type=CFG.BOOTSTRAP_TYPE, subsample=CFG.SUBSAMPLE, rsm=CFG.RSM, random_strength=2.0
- C_MAE_pruned_d4: loss_function="MAE", eval_metric="MAE", depth=4, l2_leaf_reg=7, min_data_in_leaf=36, bootstrap_type=CFG.BOOTSTRAP_TYPE, subsample=CFG.SUBSAMPLE, rsm=CFG.RSM, random_strength=1.5
- D_LOGRMSE_pruned_d5: log1p 目标（训练/验证在 log-space，推理后 expm1 回到原尺度），loss_function="RMSE", eval_metric="RMSE", depth=5, l2_leaf_reg=6, min_data_in_leaf=30, bootstrap_type=CFG.BOOTSTRAP_TYPE, subsample=CFG.SUBSAMPLE, rsm=CFG.RSM, random_strength=1.3

### 3.4 receipts/admissions 新增“超稳定”统计特征（从脚本函数与注释整理）
- receipts line-items：`n_line_items`、`mean_line_amt`、`std_line_amt`、`max_line_amt`、`top1_share`、`cost_entropy`、`cost_hhi`
- admissions：`n_adm`、`charlson_sum`、`n_emergent`（在聚合中新增）
- 注：Iter8 的 pruned_candidates 列表在 PDF 抽取时有多处断行截断；以上字段来自 Iter8 代码中明确计算/命名的聚合特征。

### 3.5 Ensemble search（泛化到 N models 的 simplex grid）
- 用整数组合 `units = int(1/W_STEP)` 枚举所有 weight compositions，使得 `sum(w)=1`
- 仍然保留 `lam`（baseline blend）与 `shift_mult`（median shift）网格
- 仍然使用跨 seeds 的 mean+std 稳健目标，降低“选权重过拟合 OOF 噪声”的风险

### 3.6 CROSS-FIT group-median calibration（关键：避免泄漏/乐观偏差）
- 目标：替代 Iter7 的 `apply_chronic_group_shift`
- 核心机制：
  - 用 StratifiedKFold 做 cross-fit：在每个 fold 的训练子集上统计 group median residual，再只应用到验证子集（OOF）
  - 对 group shift 做 count-based shrink：`cnt/(cnt+k_smooth)`（脚本中 `k_smooth=60.0`）
  - `min_count=25`（脚本参数）
- calibration 候选（从脚本整理）：
- 候选 1：group_cols = [primary_chronic]，shrink ≈ 0.30
- 候选 2：group_cols = [primary_chronic, insurance_cat]，shrink ≈ 0.25
- 候选 3：group_cols = [primary_chronic, insurance_cat, charlson_bin]（仅当 charlson_bin 存在；PDF 中该行的 shrink 数值被截断，需回原脚本确认）
- 启用规则（脚本显式）：
- 只有当 OOF MAE 改善超过 0.08（best_mae - 0.08）时才会启用某个 calibration（降低 LB overfit 风险）。

### 3.7 对应 P2：跑完后你需要回传的 5 项（用于下一轮“<450 决胜微调”）
- LB MAE（public/private 任意，给数值）
- final OOF MAE
- ensemble meta（weights / lam / shift）
- calibration chosen（是否选中 chronic×insurance 或加 charlson_bin）
- 预测分位数（脚本打印）

---

## 4. 若 Iter8 仍卡在 450.xx：下一步方向（PDF 尾部提示）
- 方向：`ratio residual` 路线  
  - 思路：建模 `y / baseline` 的偏离（ratio），再用小权重混合回原 ensemble  
  - 预期：常见于最后 0.5–1.5 MAE 的增益来源（脚本尾部说明）

---

## 5. 可复现流程骨架（给后续 AI 继续总结/抽象 prompt 用）

### 5.1 数据与目录（硬约束）
- 所有数据位于：`D:\AgentDs\agent_ds_healthcare\`
- 必备文件：`ed_cost_train.csv`、`ed_cost_test.csv`、`patients.csv`、`admissions_train.csv`、`admissions_test.csv`、`receipts_parsed.joblib`
- 输出文件：`submission.csv`（两版脚本都写到同一路径）

### 5.2 Pipeline 步骤（可复现最小链路）
- 准备环境：Python 3.x；依赖主要为 joblib、scikit-learn、catboost（脚本内含自动 pip 安装逻辑）。
- 确保数据路径完全一致（Windows）：D:\AgentDs\agent_ds_healthcare\ 目录下包含 ed_cost_train.csv / ed_cost_test.csv / patients.csv / admissions_train.csv / admissions_test.csv / receipts_parsed.joblib。
- 运行 Iter7 或 Iter8 单脚本：它会自行构建 features → 多折 CV → multi-seed bagging → 稳健 ensemble search → （Iter8 还会做 cross-fit calibration）→ 输出 submission.csv。
- 提交 submission.csv 到 leaderboard，记录 LB MAE。
- 把脚本末尾要求的日志（OOF MAE、ensemble meta、calibration chosen、pred quantiles）一并保存（用于下一轮微调与 paper 复现）。

### 5.3 建议保存的“复现实验资产”
- 每次运行的 console log（包含：seed-level OOF、final OOF、ensemble top candidates、最终 meta、calibration chosen、pred quantiles）
- 提交回执：leaderboard MAE（public/private）+ 提交时间
- 该轮脚本版本号 / Git commit（如果有）

---

## 附录 A：Iter7 脚本（PDF 原样摘录）
<details>
<summary>展开/折叠 Iter7 原始脚本文本</summary>

```python
# === ITERATION 7: Anti-overfit++ on the v3(Iter15/16)
path (keep it practical, fast, and LB-oriented) ===
# Core ideas:
# - Still LOW-DIM receipts features (buckets + shares +
HHI + EM stats + key procedure flags)
# - Shallow trees (depth 4-5), strong RSM (0.8), multi-
seed bagging
# - Stronger "generalization" ensemble selection: choose
weights by stability across seeds (mean + std), not just
best OOF
# - Add ONE diverse model: MAE-loss on pruned features
(metric-aligned but still regularized)
# - Tiny baseline shrink option (toward baseline_next3y)
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
TEST_PATH =
r"D:\AgentDs\agent_ds_healthcare\ed_cost_test.csv"
PATIENTS_PATH =
r"D:\AgentDs\agent_ds_healthcare\patients.csv"
ADM_TRAIN_PATH =
r"D:\AgentDs\agent_ds_healthcare\admissions_train.csv"
ADM_TEST_PATH =
r"D:\AgentDs\agent_ds_healthcare\admissions_test.csv"
RECEIPTS_JOBLIB_PATH =
r"D:\AgentDs\agent_ds_healthcare\receipts_parsed.joblib"
RECEIPTS_PDF_DIR =
r"D:\AgentDs\agent_ds_healthcare\receipts_pdf" # last
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
LAM_GRID = [0.00, 0.03, 0.05, 0.08, 0.10, 0.15] #
baseline blend (small)
SHIFT_GRID = [0.0, 0.5, 1.0] # global median shift
multiplier
# small complexity penalties (LB-oriented)
STD_PEN = 0.20 # penalize variance across seeds
(stability)
LAM_PEN = 8.0 # penalize baseline blend (discourage
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
ChatGPT 5.2 Thinking
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
print(f" Seed {seed_idx+1}/{CFG.N_SEEDS} OOF MAE:
" + " | ".join([f"{m}={seed_maes[m]:.2f}" for m in
model_specs.keys()]))
for m in model_specs.keys():
oof_by_seed[m].append(oof_seed[m])
test_by_seed[m].append(test_seed[m])
print("\n[seed-averaged OOF MAE per model]")
for m in oof_by_seed.keys():
avg_oof = np.mean(np.vstack(oof_by_seed[m]),
axis=0)
print(f" {m:18s}: {mean_absolute_error(y,
avg_oof):.2f}")
# print median best iters
print("\n[median best_iteration per model] (for
reference)")
for m in best_iters.keys():
if best_iters[m]:
print(f" {m:18s}: {int(np.median(best_iters[m]))}")
else:
print(f" {m:18s}: (n/a)")
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
print(f" #{i:02d} obj={obj:.3f} meanMAE={mean_m:.3f}
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
(baseline_vec[:len(test_final)]*0 + (baseline_vec.mean())) #
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
print(" admissions features: None")
else:
print(f" admissions features: {adm_df.shape} | cols=
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
print(f" receipt_feat shape: {pdf_df.shape}")
print(f" receipt_feat cols ({len(pdf_df.columns)-1}):
{[c for c in pdf_df.columns if c!='patient_id']}")
else:
print(" [warn] could not build receipt features from
joblib structure.")
except Exception as e:
print(f" [warn] receipts joblib load/build failed: {e}")
pdf_df = None
else:
print(" [warn] receipts joblib missing; skipping receipts
features.")
# sanity invariant: sum(line totals) == prior_ed_cost_5y_usd
(should match); check using available receipt feature proxy
if possible
# Our low-dim features do not include sum_items; use n/a.
# build features
print("\n[features] building train/test feature frames...")
train_feat = build_features(train, patients, adm_df, pdf_df)
test_feat = build_features(test, patients, adm_df, pdf_df)
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
test_feat[c] = pd.to_numeric(test_feat[c],
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
print(f" FULL feature count: {len(feat_full)}")
print(f" PRUNED feature count: {len(feat_pruned)}")
print(f" PRUNED features: {feat_pruned}")
miss_train = int(train_feat[feat_full].isna().sum().sum())
miss_test = int(test_feat[feat_full].isna().sum().sum())
print(f" Missing cells after fill (full): train={miss_train} test=
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
print(f" ensemble OOF MAE (stable search + optional
chronic shift): {final_oof_mae:.2f}")
print(" ensemble meta:", meta)
print(" extra shift:", best_shift["type"],
("shrink="+str(best_shift.get("shrink")) if
best_shift["type"]!="none" else ""))
print(" OOF pred quantiles:", qdict(best_oof, qs=
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

---

## 附录 B：Iter8 脚本（PDF 原样摘录）
<details>
<summary>展开/折叠 Iter8 原始脚本文本</summary>

```python
# === ITERATION 8: Tail+Calibration++ (LB-oriented, practical, fast) ===
# Upgrades vs Iter7:
# (1) Add ONE diverse model: log1p-target CatBoost (tail friendly)
# (2) Replace leakage-prone group shift with CROSS-FIT group-median calibration
# (3) Add a few ultra-stable low-dim receipt/admission stats (n_line_items, top1
# Output: D:\AgentDs\agent_ds_healthcare\submission.csv
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
TRAIN_PATH = r"D:\AgentDs\agent_ds_healthcare\ed_cost_train.csv"
TEST_PATH = r"D:\AgentDs\agent_ds_healthcare\ed_cost_test.csv"
PATIENTS_PATH = r"D:\AgentDs\agent_ds_healthcare\patients.csv"
ADM_TRAIN_PATH = r"D:\AgentDs\agent_ds_healthcare\admissions_train.csv"
ADM_TEST_PATH = r"D:\AgentDs\agent_ds_healthcare\admissions_test.csv"
RECEIPTS_JOBLIB_PATH = r"D:\AgentDs\agent_ds_healthcare\receipts_parsed.joblib"
RECEIPTS_PDF_DIR = r"D:\AgentDs\agent_ds_healthcare\receipts_pdf" # last resort
OUT_SUB_PATH = r"D:\AgentDs\agent_ds_healthcare\submission.csv"
TARGET = "ed_cost_next3y_usd"
print("="*92)
print("ITERATION 8 | Tail+Calibration++: +log1p model + cross-fit group calibrati
print("Goal: push LB below 450 by reducing tail bias + calibration leakage + impr
print("="*92)
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
# Config (keep fast, but add 1 model => keep runtime similar via folds/seeds balan
# -----------------------------
class CFG:
# Keep runtime similar after adding 4th model
N_FOLDS = 5
N_SEEDS = 6
ITERS = 3000
ES_ROUNDS = 140
LR = 0.03
RSM = 0.80
# extra regularization (helps generalization, usually LB-positive)
BOOTSTRAP_TYPE = "Bernoulli"
SUBSAMPLE = 0.85
# ensemble search
W_STEP = 0.05
LAM_GRID = [0.00, 0.03, 0.05, 0.08, 0.10, 0.15] # baseline blend (small)
SHIFT_GRID = [0.0, 0.5, 1.0] # global median shift multiplier
# stability penalties
STD_PEN = 0.20
LAM_PEN = 8.0
SHIFT_PEN = 0.002
# -----------------------------
# Utilities
# -----------------------------
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
def must_exist(path: str, name: str):
if not os.path.exists(path):
raise FileNotFoundError(f"Missing {name}: {path}")
def safe_clip_nonneg(a: np.ndarray) -> np.ndarray:
a = np.asarray(a, dtype=float)
a[a < 0] = 0.0
return a
# -----------------------------
# Admissions features (add a couple stable counts)
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
adm = adm.dropna(subset=["patient_id"]).copy()
adm["patient_id"] = adm["patient_id"].astype(int)
out = adm.groupby("patient_id").agg(
n_adm=("patient_id","size"),
charlson_max=("charlson_band","max"),
charlson_mean=("charlson_band","mean"),
charlson_sum=("charlson_band","sum"),
pct_emergent=("acuity_emergent","mean"),
n_emergent=("acuity_emergent","sum"),
).reset_index()
for c in ["n_adm","charlson_max","charlson_mean","charlson_sum","pct_emergent
out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
return out
# -----------------------------
# Low-dim receipts features from parsed lineitems
# Add: n_line_items, mean/std/max amt, top1_share, entropy
# -----------------------------
def build_pdf_features_from_lineitems(li: pd.DataFrame) -> pd.DataFrame:
li = li.copy()
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
if code_col is None or total_col is None or "patient_id" not in li.columns:
raise ValueError("Lineitems DF missing required columns.")
li["patient_id"] = pd.to_numeric(li["patient_id"], errors="coerce").astype("I
li = li.dropna(subset=["patient_id"]).copy()
li["patient_id"] = li["patient_id"].astype(int)
li["code"] = li[code_col].map(norm_code)
li = li.dropna(subset=["code"]).copy()
li["amt"] = pd.to_numeric(li[total_col], errors="coerce").fillna(0.0).astype(
li.loc[li["amt"] < 0, "amt"] = 0.0
# totals
total = li.groupby("patient_id")["amt"].sum().rename("receipt_total")
li = li.join(total, on="patient_id")
denom = li["receipt_total"].replace(0.0, np.nan)
# code numeric where possible
code_num = pd.to_numeric(li["code"].where(li["code"].str.fullmatch(r"\d+"), N
# buckets
is_em = li["code"].isin(["99281","99282","99283","99284","99285"])
em_map = {"99281":1,"99282":2,"99283":3,"99284":4,"99285":5}
em_level = li["code"].map(em_map).fillna(0).astype(float)
is_crit = li["code"].isin(["99291","99292"])
is_obs = li["code"].str.startswith("G037", na=False)
is_high = li["code"].isin(["31500","36556","32551","36620","92950"]) # airway
is_lab = code_num.between(80000, 89999)
is_imaging = code_num.between(70000, 79999)
is_proc_general = code_num.between(10000, 69999)
is_proc_any = is_high | (is_proc_general & (~is_high) & (~is_em) & (~is_crit)
# per-line shares
share = (li["amt"] / denom).fillna(0.0)
cost_hhi = (share * share).groupby(li["patient_id"]).sum().rename("cost_hhi")
entropy = (-(share * np.log(share + 1e-12))).groupby(li["patient_id"]).sum().
top1_share = li.groupby("patient_id").apply(lambda g: float(g["amt"].max() /
# stable amount stats
n_line_items = li.groupby("patient_id").size().rename("n_line_items")
mean_line_amt = li.groupby("patient_id")["amt"].mean().rename("mean_line_amt"
std_line_amt = li.groupby("patient_id")["amt"].std(ddof=0).fillna(0.0).renam
max_line_amt = li.groupby("patient_id")["amt"].max().rename("max_line_amt")
# basic counts
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
has_critical_care = is_crit.astype(int).groupby(li["patient_id"]).max().renam
has_high_acuity = is_high.astype(int).groupby(li["patient_id"]).max().rename(
has_observation = is_obs.astype(int).groupby(li["patient_id"]).max().rename("
has_imaging = is_imaging.astype(int).groupby(li["patient_id"]).max().rename("
def has_code(code: str, name: str):
return (li["code"].eq(code).astype(int).groupby(li["patient_id"]).max()).
has_intub_31500 = has_code("31500","has_intub_31500")
has_cvc_36556 = has_code("36556","has_cvc_36556")
has_cpr_92950 = has_code("92950","has_cpr_92950")
has_artline_36620 = has_code("36620","has_artline_36620")
has_ct_head_70450 = has_code("70450","has_ct_head_70450")
has_99285 = has_code("99285","has_99285")
has_ct_abdpel_74177 = has_code("74177","has_ct_abdpel_74177")
has_troponin_84484 = has_code("84484","has_troponin_84484")
has_obs_G0378 = has_code("G0378","has_obs_G0378")
out = pd.concat([
n_unique_codes,
n_line_items, mean_line_amt, std_line_amt, max_line_amt,
top1_share, entropy, cost_hhi,
n_em_codes, max_em_level, avg_em_level, n_high_em,
has_critical_care, has_high_acuity, has_observation, has_imaging,
has_intub_31500, has_cvc_36556, has_cpr_92950, has_artline_36620,
has_ct_head_70450, has_99285, has_ct_abdpel_74177, has_troponin_84484, ha
n_procedures, n_imaging, n_lab,
total
], axis=1).reset_index()
# merge totals (may be missing)
for s in [em_total, crit_total, proc_total, img_total, lab_total, high_total]
out = out.merge(s.reset_index(), on="patient_id", how="left")
for c in ["em_total","crit_total","proc_total","img_total","lab_total","high_
if c not in out.columns:
out[c] = 0.0
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
# drop helper totals
out.drop(columns=[c for c in ["em_total","crit_total","proc_total","img_total
# fill numeric
for c in out.columns:
if c == "patient_id":
continue
out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
return out
def load_receipts_joblib(joblib_path: str) -> Optional[pd.DataFrame]:
if not os.path.exists(joblib_path):
return None
data = joblib_load(joblib_path)
if isinstance(data, dict):
for k in ["lineitems_df","lineitems","items_df","items","line_items_df","
if k in data and isinstance(data[k], pd.DataFrame):
return build_pdf_features_from_lineitems(data[k])
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
return build_pdf_features_from_lineitems(df)
return df
if isinstance(data, (list, tuple)):
dfs = [x for x in data if isinstance(x, pd.DataFrame)]
for df in dfs:
if "patient_id" in df.columns and any(c in df.columns for c in ["code
return build_pdf_features_from_lineitems(df)
for df in dfs:
if "patient_id" in df.columns:
return df
return None
# -----------------------------
# Feature engineering (numeric-only, keep v3 spirit)
# -----------------------------
def build_features(ed_df: pd.DataFrame,
patients_df: pd.DataFrame,
adm_df: Optional[pd.DataFrame],
pdf_df: Optional[pd.DataFrame]) -> pd.DataFrame:
feat = ed_df.copy()
chronic_map = {"PNEUMONIA":0, "DIABETESCOMP":1, "HF":2}
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
# patients
p = patients_df.copy()
p["patient_id"] = pd.to_numeric(p["patient_id"], errors="coerce").astype(int)
p["age"] = pd.to_numeric(p["age"], errors="coerce")
p["age"] = p["age"].fillna(p["age"].median())
p["sex_encoded"] = (p["sex"].astype(str).str.upper() == "M").astype(int)
ins = p["insurance"].astype(str).str.lower()
ins_map = {"private":2, "public":1, "self_pay":0, "selfpay":0}
p["insurance_encoded"] = ins.map(ins_map).fillna(-1).astype(float)
p["insurance_cat"] = p["insurance_encoded"].fillna(-1).astype(int)
z3 = p["zip3"].apply(standardize_zip3).astype("string")
zr = z3.fillna("000").str.replace(r"\D","", regex=True).str.zfill(3).str[0]
p["zip_region"] = pd.to_numeric(zr, errors="coerce").fillna(-1).astype(float)
# a stable age bin (helps calibration & sometimes model)
p["age_bin"] = pd.cut(p["age"], bins=[0,30,50,70,200], labels=False, include_
feat = feat.merge(p[["patient_id","age","age_bin","sex_encoded","insurance_en
on="patient_id", how="left")
feat["ins_x_chronic"] = feat["insurance_encoded"].fillna(-1) * feat["chronic_
# admissions
if adm_df is not None:
feat = feat.merge(adm_df, on="patient_id", how="left")
for c in ["n_adm","charlson_max","charlson_mean","charlson_sum","pct_emer
if c in feat.columns:
feat[c] = pd.to_numeric(feat[c], errors="coerce").fillna(feat[c].
if "charlson_max" in feat.columns:
feat["charlson_bin"] = np.clip(np.round(feat["charlson_max"]).astype(
else:
feat["charlson_bin"] = 0
else:
feat["charlson_bin"] = 0
# receipts
if pdf_df is not None:
feat = feat.merge(pdf_df, on="patient_id", how="left")
for c in pdf_df.columns:
if c == "patient_id": continue
feat[c] = pd.to_numeric(feat[c], errors="coerce").fillna(feat[c].medi
# light interactions
if "pct_cost_critical" in feat.columns:
feat["logprior_x_pctcritical"] = feat["log_prior_cost"] * feat["pct_cost_
if "n_high_acuity_total" in feat.columns:
feat["logprior_x_highacu"] = feat["log_prior_cost"] * feat["n_high_acuity_
# stable ratio
if "n_unique_codes" in feat.columns:
feat["cost_per_code"] = feat["prior_ed_cost_5y_usd"] / feat["n_unique_cod
# numeric safety
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
def make_strat_labels(df_train_feat: pd.DataFrame, y: np.ndarray) -> np.ndarray:
tmp = df_train_feat[["primary_chronic"]].copy()
tmp["y"] = y
tmp["cost_bin"] = pd.qcut(tmp["y"], q=5, labels=False, duplicates="drop")
tmp["strat"] = tmp["primary_chronic"].astype(str) + "_" + tmp["cost_bin"].asty
strat = LabelEncoder().fit_transform(tmp["strat"].values)
return strat
# -----------------------------
# Training (multi-seed + fold bagging) with 4 models (add log1p)
# -----------------------------
def train_models(train_feat: pd.DataFrame, test_feat: pd.DataFrame,
feat_full: List[str], feat_pruned: List[str]) -> Tuple[Dict[str,
y = train_feat[TARGET].values.astype(float)
strat = make_strat_labels(train_feat, y)
# 4 models: add one log1p target model
model_specs = {
"A_RMSE_full_d5": dict(
loss_function="RMSE", eval_metric="MAE",
depth=5, l2_leaf_reg=5, min_data_in_leaf=28,
learning_rate=CFG.LR, iterations=CFG.ITERS,
rsm=CFG.RSM, random_strength=1.0,
bootstrap_type=CFG.BOOTSTRAP_TYPE, subsample=CFG.SUBSAMPLE,
),
"B_RMSE_pruned_d4": dict(
loss_function="RMSE", eval_metric="MAE",
depth=4, l2_leaf_reg=4, min_data_in_leaf=32,
learning_rate=CFG.LR, iterations=CFG.ITERS,
rsm=CFG.RSM, random_strength=2.0,
bootstrap_type=CFG.BOOTSTRAP_TYPE, subsample=CFG.SUBSAMPLE,
),
"C_MAE_pruned_d4": dict(
loss_function="MAE", eval_metric="MAE",
depth=4, l2_leaf_reg=7, min_data_in_leaf=36,
learning_rate=CFG.LR, iterations=CFG.ITERS,
rsm=CFG.RSM, random_strength=1.5,
bootstrap_type=CFG.BOOTSTRAP_TYPE, subsample=CFG.SUBSAMPLE,
),
"D_LOGRMSE_pruned_d5": dict(
loss_function="RMSE", eval_metric="RMSE", # in log-space
depth=5, l2_leaf_reg=6, min_data_in_leaf=30,
learning_rate=CFG.LR, iterations=CFG.ITERS,
rsm=CFG.RSM, random_strength=1.3,
bootstrap_type=CFG.BOOTSTRAP_TYPE, subsample=CFG.SUBSAMPLE,
),
}
model_featcols = {
"A_RMSE_full_d5": feat_full,
"B_RMSE_pruned_d4": feat_pruned,
"C_MAE_pruned_d4": feat_pruned,
"D_LOGRMSE_pruned_d5": feat_pruned,
}
model_target_mode = {
"A_RMSE_full_d5": "raw",
"B_RMSE_pruned_d4": "raw",
"C_MAE_pruned_d4": "raw",
"D_LOGRMSE_pruned_d5": "log1p",
}
oof_by_seed = {m: [] for m in model_specs.keys()}
test_by_seed = {m: [] for m in model_specs.keys()}
best_iters = {m: [] for m in model_specs.keys()}
print("\n[training] CatBoost CPU | shallow + rsm + subsample | multi-seed bag
print("Models:", list(model_specs.keys()))
print(f"Seeds={CFG.N_SEEDS}, Folds={CFG.N_FOLDS}\n")
for seed_idx in range(CFG.N_SEEDS):
rs = SEED + seed_idx * 7
kf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=rs)
oof_seed = {m: np.zeros(len(train_feat), dtype=float) for m in model_spec
test_seed = {m: np.zeros(len(test_feat), dtype=float) for m in model_spec
for fold, (ti, vi) in enumerate(kf.split(train_feat, strat), 1):
for mname, params in model_specs.items():
cols = model_featcols[mname]
mode = model_target_mode[mname]
X_tr = train_feat[cols].iloc[ti]
y_tr = y[ti]
X_va = train_feat[cols].iloc[vi]
y_va = y[vi]
X_te = test_feat[cols]
if mode == "log1p":
y_tr_fit = np.log1p(y_tr)
y_va_fit = np.log1p(y_va)
else:
y_tr_fit = y_tr
y_va_fit = y_va
cb = CatBoostRegressor(
**params,
task_type="CPU",
thread_count=-1,
verbose=0,
allow_writing_files=False,
random_seed=rs + fold * 31 + (hash(mname) % 997),
early_stopping_rounds=CFG.ES_ROUNDS,
)
cb.fit(X_tr, y_tr_fit, eval_set=(X_va, y_va_fit), verbose=0)
try:
best_iters[mname].append(int(cb.get_best_iteration()))
except Exception:
pass
pred_va = cb.predict(X_va)
pred_te = cb.predict(X_te)
if mode == "log1p":
pred_va = np.expm1(pred_va)
pred_te = np.expm1(pred_te)
pred_va = safe_clip_nonneg(pred_va)
pred_te = safe_clip_nonneg(pred_te)
oof_seed[mname][vi] = pred_va
test_seed[mname] += pred_te / CFG.N_FOLDS
del cb
gc.collect()
# per-seed MAE on RAW scale (important)
seed_maes = {m: float(mean_absolute_error(y, oof_seed[m])) for m in model_
print(f" Seed {seed_idx+1}/{CFG.N_SEEDS} OOF MAE: " + " | ".join([f"{m}=
for m in model_specs.keys():
oof_by_seed[m].append(oof_seed[m])
test_by_seed[m].append(test_seed[m])
print("\n[seed-averaged OOF MAE per model]")
for m in oof_by_seed.keys():
avg_oof = np.mean(np.vstack(oof_by_seed[m]), axis=0)
print(f" {m:20s}: {mean_absolute_error(y, avg_oof):.2f}")
print("\n[median best_iteration per model] (for reference)")
for m in best_iters.keys():
if best_iters[m]:
print(f" {m:20s}: {int(np.median(best_iters[m]))}")
else:
print(f" {m:20s}: (n/a)")
return oof_by_seed, test_by_seed, best_iters
# -----------------------------
# Ensemble selection (stability across seeds) — generic N-model simplex grid
# -----------------------------
def stable_ensemble_search(train_feat: pd.DataFrame,
oof_by_seed: Dict[str, List[np.ndarray]],
test_by_seed: Dict[str, List[np.ndarray]],
baseline_oof: np.ndarray,
baseline_test: np.ndarray) -> Tuple[np.ndarray, np.nda
y = train_feat[TARGET].values.astype(float)
model_names = list(oof_by_seed.keys())
n_models = len(model_names)
# seed-avg
oof_avg = {m: np.mean(np.vstack(oof_by_seed[m]), axis=0) for m in model_names
test_avg = {m: np.mean(np.vstack(test_by_seed[m]), axis=0) for m in model_nam
# stack for speed
OOF_AVG = np.vstack([oof_avg[m] for m in model_names]) # (n_mod
TEST_AVG = np.vstack([test_avg[m] for m in model_names]) # (n_mod
OOF_SEEDS = [np.vstack([oof_by_seed[m][s] for m in model_names]) for s in ran
# integer simplex grid
step = CFG.W_STEP
units = int(round(1.0 / step))
def compositions(total, k):
if k == 1:
yield (total,)
else:
for i in range(total + 1):
for rest in compositions(total - i, k - 1):
yield (i,) + rest
best = None
top_list = []
for w_int in compositions(units, n_models):
w = np.array(w_int, dtype=float) / units
if abs(w.sum() - 1.0) > 1e-9:
continue
# precompute base avg pred
pred_avg_base = w @ OOF_AVG
for lam in CFG.LAM_GRID:
# baseline blend on avg
pred_avg = (1.0 - lam) * pred_avg_base + lam * baseline_oof
for sm in CFG.SHIFT_GRID:
shift = float(np.median(y - pred_avg)) * sm
maes = []
for s in range(CFG.N_SEEDS):
pred = w @ OOF_SEEDS[s]
pred = (1.0 - lam) * pred + lam * baseline_oof
pred = pred + shift
maes.append(float(mean_absolute_error(y, pred)))
mean_m = float(np.mean(maes))
std_m = float(np.std(maes, ddof=0))
obj = mean_m + CFG.STD_PEN * std_m + CFG.LAM_PEN * lam + CFG.SHIF
rec = (obj, mean_m, std_m, w.copy(), float(lam), float(sm), float
top_list.append(rec)
if best is None or obj < best[0]:
best = rec
top_list.sort(key=lambda x: x[0])
print("\n[ensemble search] Top candidates by robust objective (mean + std + s
for i, rec in enumerate(top_list[:10], 1):
obj, mean_m, std_m, w, lam, sm, shift = rec
w_str = ",".join([f"{x:.2f}" for x in w])
print(f" #{i:02d} obj={obj:.3f} meanMAE={mean_m:.3f} std={std_m:.3f} | w
obj, mean_m, std_m, w, lam, sm, shift = best
oof_final = w @ OOF_AVG
test_final = w @ TEST_AVG
oof_final = (1.0 - lam) * oof_final + lam * baseline_oof
test_final = (1.0 - lam) * test_final + lam * baseline_test
oof_final = oof_final + shift
test_final = test_final + shift
meta = {
"models_order": model_names,
"weights": tuple(float(x) for x in w),
"lam_baseline": float(lam),
"shift_mult": float(sm),
"shift_value": float(shift),
"oof_mean_mae_across_seeds": float(mean_m),
"oof_std_mae_across_seeds": float(std_m),
}
return oof_final, test_final, meta
# -----------------------------
# CROSS-FIT group-median calibration (avoids leakage)
# -----------------------------
def _group_key(df: pd.DataFrame, cols: List[str]) -> pd.Series:
# stable string key
tmp = df[cols].copy()
for c in cols:
if pd.api.types.is_numeric_dtype(tmp[c]):
# make bins stable for numeric discrete columns
tmp[c] = tmp[c].fillna(-1).astype(int).astype(str)
else:
tmp[c] = tmp[c].astype(str)
return tmp.astype(str).agg("|".join, axis=1)
def crossfit_group_calibration(train_feat: pd.DataFrame,
base_oof: np.ndarray,
group_cols: List[str],
shrink: float,
min_count: int = 25,
k_smooth: float = 60.0,
n_splits: int = 5,
random_state: int = 42) -> Tuple[np.ndarray, Dict[
"""
Returns:
oof_cal (cross-fitted)
shifts_full (fit on full residuals, for applying to test)
"""
y = train_feat[TARGET].values.astype(float)
strat = make_strat_labels(train_feat, y)
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_st
oof_cal = base_oof.copy()
for tr, va in skf.split(train_feat, strat):
resid_tr = y[tr] - base_oof[tr]
key_tr = _group_key(train_feat.iloc[tr], group_cols)
# compute median residual per group on TR only
df_tr = pd.DataFrame({"key": key_tr.values, "resid": resid_tr})
grp = df_tr.groupby("key")["resid"]
med = grp.median()
cnt = grp.size()
# shrink by count
shrink_n = (cnt / (cnt + k_smooth)).clip(lower=0.0, upper=1.0)
shift_map = (med * shrink * shrink_n).to_dict()
key_va = _group_key(train_feat.iloc[va], group_cols)
oof_cal[va] = oof_cal[va] + key_va.map(shift_map).fillna(0.0).values
# fit shifts on full data for test application
resid_full = y - base_oof
key_full = _group_key(train_feat, group_cols)
df_full = pd.DataFrame({"key": key_full.values, "resid": resid_full})
grp_full = df_full.groupby("key")["resid"]
med_full = grp_full.median()
cnt_full = grp_full.size()
shrink_n_full = (cnt_full / (cnt_full + k_smooth)).clip(lower=0.0, upper=1.0)
shifts_full = (med_full * shrink * shrink_n_full).to_dict()
return oof_cal, shifts_full
def apply_shifts_to_test(test_feat: pd.DataFrame, base_test: np.ndarray, group_co
key_te = _group_key(test_feat, group_cols)
return base_test + key_te.map(shifts_full).fillna(0.0).values
# -----------------------------
# Main load
# -----------------------------
must_exist(TRAIN_PATH, "TRAIN")
must_exist(TEST_PATH, "TEST")
must_exist(PATIENTS_PATH, "patients")
must_exist(ADM_TRAIN_PATH, "admissions_train")
must_exist(ADM_TEST_PATH, "admissions_test")
if not os.path.exists(RECEIPTS_JOBLIB_PATH):
print("[warn] receipts_parsed.joblib missing -> receipts features will be empt
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
for df in [train, test, patients]:
df["patient_id"] = pd.to_numeric(df["patient_id"], errors="coerce").astype(int
print("\n[admissions] building robust aggregates (+counts)...")
adm_df = load_admissions_features(ADM_TRAIN_PATH, ADM_TEST_PATH)
if adm_df is None:
print(" admissions features: None")
else:
print(f" admissions features: {adm_df.shape} | cols={list(adm_df.columns)}")
print("\n[receipts] loading receipts_parsed.joblib and building low-dim receipt f
pdf_df = None
if os.path.exists(RECEIPTS_JOBLIB_PATH):
try:
pdf_df = load_receipts_joblib(RECEIPTS_JOBLIB_PATH)
if pdf_df is not None:
pdf_df["patient_id"] = pd.to_numeric(pdf_df["patient_id"], errors="co
pdf_df = pdf_df.dropna(subset=["patient_id"]).copy()
pdf_df["patient_id"] = pdf_df["patient_id"].astype(int)
pdf_df = pdf_df.drop_duplicates("patient_id", keep="last")
print(f" receipt_feat shape: {pdf_df.shape}")
print(f" receipt_feat cols ({len(pdf_df.columns)-1}): {[c for c in p
else:
print(" [warn] could not build receipt features from joblib structur
except Exception as e:
print(f" [warn] receipts joblib load/build failed: {e}")
pdf_df = None
else:
print(" [warn] receipts joblib missing; skipping receipts features.")
print("\n[features] building train/test feature frames...")
train_feat = build_features(train, patients, adm_df, pdf_df)
test_feat = build_features(test, patients, adm_df, pdf_df)
feat_full = get_numeric_feature_cols(train_feat)
feat_full = [c for c in feat_full if c != TARGET]
# PRUNED set: stable features + new stable stats
pruned_candidates = [
# priors + transforms
"prior_ed_visits_5y","prior_ed_cost_5y_usd","sqrt_prior_cost","log_prior_cost
"baseline_next3y",
# demographics
"chronic_encoded","age","age_bin","sex_encoded","insurance_encoded","insuranc
# admissions (expanded)
"n_adm","charlson_max","charlson_mean","charlson_sum","charlson_bin","pct_eme
# receipt robust + new stable stats
"cost_per_em","cost_hhi","cost_entropy","top1_share","n_line_items","mean_lin
"pct_cost_procedure","pct_cost_critical","pct_cost_high_acuity",
"n_high_acuity_total","has_critical_care","has_99285","max_em_level","n_uniqu
# light interactions
"logprior_x_pctcritical","logprior_x_highacu",
# stable ratio
"cost_per_code",
]
feat_pruned = [c for c in pruned_candidates if c in train_feat.columns]
# fill safety
for c in set(feat_full + feat_pruned):
if c in train_feat.columns:
med = train_feat[c].median() if not train_feat[c].isna().all() else 0.0
train_feat[c] = pd.to_numeric(train_feat[c], errors="coerce").fillna(med)
test_feat[c] = pd.to_numeric(test_feat[c], errors="coerce").fillna(med)
feat_full = drop_constants(feat_full, train_feat)
feat_pruned = drop_constants(feat_pruned, train_feat)
print(f" FULL feature count: {len(feat_full)}")
print(f" PRUNED feature count: {len(feat_pruned)}")
print(f" PRUNED features: {feat_pruned}")
miss_train = int(train_feat[feat_full].isna().sum().sum())
miss_test = int(test_feat[feat_full].isna().sum().sum())
print(f" Missing cells after fill (full): train={miss_train} test={miss_test}")
# train
oof_by_seed, test_by_seed, best_iters = train_models(train_feat, test_feat, feat_
# baseline vectors
baseline_oof = train_feat["baseline_next3y"].values.astype(float)
baseline_test = test_feat["baseline_next3y"].values.astype(float)
# stable ensemble search (now 4 models)
oof_ens, test_ens, meta = stable_ensemble_search(train_feat, oof_by_seed, test_by_
y = train_feat[TARGET].values.astype(float)
base_mae = float(mean_absolute_error(y, oof_ens))
print("\n" + "="*70)
print("[BASE ENSEMBLE OOF]")
print(f" OOF MAE (stable search): {base_mae:.3f}")
print(" ensemble meta:", meta)
print("="*70)
# -----------------------------
# Cross-fit group calibration candidates (LB-oriented)
# -----------------------------
print("\n[calibration] trying CROSS-FIT group-median calibration candidates (no l
calib_candidates = []
# always safe
calib_candidates.append((["primary_chronic"], 0.30))
# usually best ROI
calib_candidates.append((["primary_chronic","insurance_cat"], 0.25))
# optional: add charlson bin if present (can help HF vs others)
if "charlson_bin" in train_feat.columns:
calib_candidates.append((["primary_chronic","insurance_cat","charlson_bin"],
best_oof = oof_ens.copy()
best_test = test_ens.copy()
best_calib = {"type": "none"}
best_mae = base_mae
for cols, shrink in calib_candidates:
try:
oof_cal, shifts_full = crossfit_group_calibration(
train_feat, oof_ens, group_cols=cols,
shrink=shrink, min_count=25, k_smooth=60.0,
n_splits=5, random_state=SEED+11
)
mae_cal = float(mean_absolute_error(y, oof_cal))
print(f" candidate cols={cols} shrink={shrink:.2f} -> OOF MAE={mae_cal:.
# accept only if improvement is meaningful to reduce LB-overfit risk
if mae_cal < best_mae - 0.08:
best_mae = mae_cal
best_oof = oof_cal
best_test = apply_shifts_to_test(test_feat, test_ens, cols, shifts_fu
best_calib = {"type":"group_median_crossfit", "cols": cols, "shrink":
except Exception as e:
print(f" [warn] calibration candidate failed cols={cols}: {e}")
print("\n[calibration] chosen:", best_calib)
# final clip (LB-safe, but not too aggressive)
y_max = float(np.max(y))
best_test = np.clip(best_test, 0.0, y_max * 1.6)
final_oof_mae = float(mean_absolute_error(y, best_oof))
print("\n" + "="*70)
print("[FINAL OOF]")
print(f" final OOF MAE (ensemble + optional cross-fit calibration): {final_oof_m
print(" ensemble meta:", meta)
print(" calibration:", best_calib)
print(" OOF pred quantiles:", qdict(best_oof, qs=(0,0.01,0.05,0.1,0.5,0.9,0.95,0
print("="*70)
# submission
sub = pd.DataFrame({
"patient_id": test["patient_id"].values.astype(int),
"ed_cost_next3y_usd": np.round(best_test.astype(float), 2)
})
sub = sub[["patient_id", "ed_cost_next3y_usd"]]
out_path = Path(OUT_SUB_PATH)
out_path.parent.mkdir(parents=True, exist_ok=True)
sub.to_csv(out_path, index=False)
print("\n[SUBMISSION sanity checks]")
print("submission shape:", sub.shape)
print("submission columns exactly:", list(sub.columns))
print("any NaNs in preds:", bool(np.isnan(sub["ed_cost_next3y_usd"]).any()))
print("pred min/median/max:", float(sub["ed_cost_next3y_usd"].min()), float(sub["
print("pred quantiles:", qdict(sub["ed_cost_next3y_usd"].values))
print("\nSaved submission to:", str(out_path))
print("\nPaste back: (1) leaderboard MAE, (2) FINAL OOF MAE, (3) ensemble meta, (4
```

</details>

---

## 附录 C：PDF 尾部的“回传要求 + 下一步提示”（原样摘录）
```text
跑完你回我 个东西（我就能继续做下⼀次 决胜微调 ）
5 “<450 ”
（ 哪个都⾏，给具体数）
1. LB MAE public/private
2. final OOF MAE
（ ）
3. ensemble meta weights/lam/shift
（有没有选中 或加 ）
4. calibration chosen chronic×insurance charlson_bin
预测分位数（脚本已打印）
5.
如果这版 还是卡在 ，我下⼀步会⾛ 模型（预测 的
LB 450.xx “ratio residual y / baseline
偏离） ⼩权重混合 ，它通常是这种数据集最后 的来源。
+ ” 0.5~1.5 MAE
```
