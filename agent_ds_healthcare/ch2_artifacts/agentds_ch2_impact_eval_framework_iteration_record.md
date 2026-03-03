# AgentDS Healthcare Ch2 — 影响评估框架与跨 Challenge 迁移实验（从 PDF 抽取整理）

- 来源：`IEEE ICHI - Ch2影响评估框架.pdf`（111 页）
- 整理日期：2026-02-19
- 目的：把这份对话里的 **用户 prompts + 迭代动作 + 关键结果（MAE / OOF / Macro-F1）** 整理成可交给其他 AI 的结构化材料，用来进一步总结出 **可复现的流程**。

## 关键结论与数字（从对话中直接抽取）

- Ch1 你给出的最好成绩：Macro-F1 = 0.9014（你在对话里声明的）。
- Ch2 你给出的当前最强基线：Code18（你标注的 *current best code*）≈ MAE 449（Leaderboard）。
- Ch1→Ch2 迁移实验：Code19（Code18 + CH1-risk bridge）Leaderboard MAE = 450.9103（对话中提交回执）。
- 同一次 Code19 运行打印的 FINAL OOF：ensemble OOF MAE = 429.695（显示出明显 OOF≠LB）。
- Code19 内部训练的“辅助 Ch1 风险模型”：aux OOF macro-F1@0.5 = 0.7034（显著低于你宣称的 0.9014）。
- 诊断主线：**OOF 很好但 LB 卡/倒退** → 更像是 *train/test 分布漂移 + CV 失真*，而不是“再加新特征”。
- 下一步建议主线：用 **adversarial validation → importance weighting** 做 drift-aware 训练（对话里提出 Code20）。

## 迭代时间线（按对话顺序）

### Iter-0：你问“其他 challenge 的 description / 代码是否能证明影响 Ch2？”
- 你的 prompt：想用其他 challenge（如 description.md、其他题结果/代码）来证明或增强 Ch2。
- 输出要点（对话中的框架）：
  - 可能直接增益：领域机制假设（慢病负担/保险/就诊强度）、可复用工程（解析缓存、merge 粒度、防泄漏）、验证框架（adversarial validation、稳健集成）。
  - 可能负面：把别题指标硬套、泄漏式代理特征、过强医学映射但数据不支持。
  - 推荐验证方式：提炼→最小实现→双验证（OOF + sanity holdout）。

### Iter-1：你提出 Ch1 很强（Macro-F1 = 0.9014），想迁移到 Ch2
- 你的 prompt：强调 Ch1 Macro-F1 很高，问是否能变成 Ch2 的证明/增益。
- 对话建议：只迁移 **低维、OOF 一致** 的“风险信号”，不要把高维文本直接搬进 Ch2。
- 推荐做法：在 Ch2 代码里训练一个辅助 Ch1 模型，产出 admissions_train 的 OOF 概率 + admissions_test 的 full-fit 概率，再按 patient_id 聚合为 4–6 个 risk 特征（mean/max/std/count/entropy）并 merge 进 Ch2。

### Iter-2：你提供基线（Code18）与 Ch1 当前 best submission
- 你的输入内容：
  - 说明你 Ch1 当前 best submission（admission_id, readmit_30d）。
  - 贴出 Ch2 当前 best code：Code18（你标注 MAE≈449）。
- 对话诊断：只用 Ch1 的 submission 文件（仅 admissions_test 预测）会导致 train/test feature mismatch；更稳的是“在 Ch2 内部训练辅助 Ch1”，保证 train 侧也有 OOF 预测概率。

### Iter-3：实现迁移：你要求 one-cell 完整代码 → 得到 Code19
- 你的 prompt：`give me full code in 1 cell!!!`
- 输出：Code19 = Code18 + CH1-risk bridge（在同一 cell 内训练辅助 Ch1 模型、聚合 patient risk 特征、merge 回归特征）。

### Iter-4：你运行 Code19 并反馈结果：LB 没变好（甚至倒退）
- 你贴的回执/日志要点：
  - Leaderboard MAE = 450.9103
  - FINAL OOF ensemble MAE = 429.695
  - aux Ch1 模型 OOF macro-F1@0.5 = 0.7034
- 你的结论性提问：`好像没啥用?`

### Iter-5：对话给出的解释与下一步方向（Code20）
- 为什么 Code19 对 LB 无增益（对话里的三条主因）：
  1) 任务目标不对齐：Ch1 是 30 天再入院分类；Ch2 是未来 3 年总成本回归。
  2) 你在 Code19 里训练出来的 aux Ch1 模型并不强（macro-F1≈0.7034），喂给 Ch2 的 risk 特征噪声更大。
  3) 更核心的瓶颈是 **CV/LB mismatch（分布漂移）**：OOF 很低但 LB 高很多。
- 下一步被建议的路线：adversarial validation → importance weighting（对话提供 Code20 的一整套实现思路）。

## 可复现流程（把对话里的方法论提炼成 checklist）

### A. 迁移/证明一个“新信息源”是否真的有用（对话框架浓缩）
- 提炼：先把新来源（其他 challenge / description / 模型）的贡献写成“特征族/信号族”清单（不要直接搬整套模型）。
- 最小实现：每次只加 **一个** 信号族到当前最稳基线（如 Code18），控制自由度。
- 双验证：
  - 5-fold（或你当前 N_FOLDS）OOF MAE
  - 再加一个 sanity holdout（按成本分位分层 + 固定 seed）看是否一致改善
- 判据：**只降 OOF、不降 holdout/LB**，优先判定为“没有可证明的泛化价值”。

### B. Ch1 → Ch2 迁移的“最稳实现方式”（按对话）
- 不要直接用 Ch1 submission（只覆盖 admissions_test）；要在 Ch2 代码里训练一个 aux Ch1 模型，产出：
  - admissions_train：OOF 概率（避免 train 侧用真标签导致 mismatch）
  - admissions_test：full-fit 概率
- patient_id 聚合成低维 risk 特征（mean/max/std/count/entropy），再 merge 到 Ch2。
- 强约束：维度要低（4–6 个），否则小样本（train=2000）很容易过拟合。

### C. 当出现“OOF 很好但 LB 卡/倒退”时的优先级（对话结论）
- 第一优先：检查/缓解 train-test 分布漂移（adversarial validation AUC、importance weighting、稳健集成）。
- 第二优先：稳健化而不是加复杂度（trimmed mean、one-SE、weight bagging、shift 的 cross-fit/保守化）。
- 最后才是：加新高维信息源（文本、复杂 stacking、更多类别展开），因为自由度会快速放大 CV 噪声。

## 附录：原始 prompts / 日志摘录（便于喂给其他 AI）

> 说明：以下为从 PDF 文本抽取后做了最小清理（去掉多余空行、修剪行尾空格），可能仍存在 PDF 导出导致的断行/字体替换。

### Prompt-1（你问：其他 challenge / description.md 是否能“证明影响”Ch2）

```text
你看看我其他上传的 其他 的结果或者
description.md cha
会对我们 有证明影响吗？
code ch2
```

### Assistant-1（对应 Prompt-1 的框架回答：哪些会增益/哪些会误导/三步验证法）

```text
我现在没法直接 看到你其他上传的 其他 的结果或代码 因
“ description.md / challenge ”——
为在当前这个对话⾥并没有可检索的上传⽂件（⼯具侧显⽰ ）。所以我
NoSourcesAvailable
不能基于具体内容下结论。
但我可以先把你问的这个问题拆成两层，给你⼀个可执⾏的判断框架：哪些 其他
“ challenge
的东西 会对 （ ）有正⾯证明 影响，哪些反⽽会误导或拖累。
” Ch2 ED Cost Forecasting /
哪些 其他 的内容 很可能对 有直接增益
1) “ challenge ” Ch2
领域共识 业务假设（能当作 证明 的那种）
A. / “ ”
如果 ⾥明确写了类似：
description.md
与 慢病负担（ ） 强相关
ED cost ** Charlson / comorbidity **
与 保险类型 ⽀付⽅强相关
/
与 急诊就诊强度、住院历史、复诊 再⼊院⻛险相关
/
与 处置 检查（ ）与⾦额分布相关
/ procedure codes
这种是 可迁移的因果 机制假设 ，对 很有⽤：
“ / ” Ch2
你可以把它转成更稳的特征族（例如：强度、集中度、价格⽔平、尾部⻛险 ），并⽤
proxy
去验证增益。
CV/holdout
可复⽤的数据⼯程：解析、缓存、对⻬、去泄漏
B.
任何别的 ⾥如果有：
challenge code
更鲁棒的 （容错、覆盖率统计、 ）
PDF parsing fallback
缓存策略（ ）
joblib / parquet / hash key
统⼀（ 、 聚合粒度⼀致）
merge key patient_id admissions
防⽌ 处理不⼀致、类别缺失处理⼀致
train/test
这类会显著减少 暗坑 ，提升泛化稳定性，间接提⾼ 。
“ ” LB

验证框架（尤其是 稳定性导向）
C. drift /
如果其他 有成熟的：
challenge
（检验 分布差）
adversarial validation train-test
稳健集成：多
seed + trimmed mean / median
选择 权重时考虑 跨 稳定性（均值 ⽅差）
ensemble seed +
这些对你⽬前 卡 这种很关键：很多时候不是模型不够强，⽽是泛化策略不够稳。
“ 450”
哪些 其他 的内容 对 可能是负⾯影响（看起来聪明但
2) “ challenge ” Ch2
会害你）
把别的任务⽬标 指标硬套过来
A. /
⽐如别的 是分类、 、 ，或者是别的回归⽬标分布完全不同。
challenge AUC logloss
你如果照搬：
过深模型、过多⾼维⽂本特征
过度复杂的
stacking
以 最优为唯⼀标准的权重搜索
OOF
很容易造成你看到的现象： 很好但 不动 甚⾄变差。
OOF LB /
泄漏式特征 或接近标签的代理
B. “ ”
任何描述 代码如果⽤了：
/
明显来⾃未来的信息（例如后续结算、未来随访、明显后验字段）
或者⽤了 分布做了不当的 标准化
test-only target encoding/
会让 假好， 卡死甚⾄暴死。
CV LB
过强的 医疗知识映射 但数据不⽀持
C. “ ”
⽐如强⾏把 映射到复杂层级、或者引⼊外部知识库但没对⻬，会让特征噪声更
CPT/ICD
⼤。对你现在这种⼩样本（ ）⾮常危险。
train 2000
你要的 证明影响 ：最靠谱的做法
3) “ ”
你现在想知道 有没有证明价值 ，我建议⽤⼀个三步验证（不需要⼤改代码）：
“ ”

提炼：从 其他 ⾥只抽取 假设 特征族 清单（不要直接搬整
1. description.md / challenge “ / ”
套模型）。
最⼩实现：每次只加 个特征族（例如： 集中度 价格⽔平）
2. 1 receipt +
双验证：
3.
5fold OOF MAE
再加⼀个 （⽐如按 分位分层 固定 ）看是否⼀致改善
sanity holdout cost + seed
如果 改善但 不改善，基本就不是 可证明的增益 。
OOF holdout “ ”
我现在需要你怎么做（不问你⼀堆问题，给你最⼩动作）
4)
把这些东西直接贴到对话⾥任意⼀种即可（越少越好也⾏）：
⾥ 相关的那⼏段（尤其是任务、数据字段含义、任何
description.md Ch2 domain
）
insight
或者你认为 可能影响 的其他 代码⾥：
“ Ch2” challenge
特征⼯程部分 验证 集成部分（不⽤全贴）
+ /
我拿到⽂本后，我可以直接告诉你：
哪些点能 转成 的稳健特征 验证策略
“ Ch2 / ”
哪些点要避免（会让你继续卡 ）
450
最优的 迁移优先级 （按性价⽐排序）
“ backlog”
你直接粘贴内容就⾏，不⽤整理。
description.md
File
这个 我 的数据很⾼
! ch1 Macro-F1
```

### Prompt-2（你声明：Ch1 Macro-F1=0.9014）

```text
0.9014
```

### Assistant-2（对应 Prompt-2：把 Ch1 作为低维风险信号迁移到 Ch2 的建议）

```text
你 做到 ，对 很可能是 有⽤的侧信息 但它不会⾃动把
Ch1 Macro-F1=0.9014 Ch2 “ ”—— MAE
从 拉到 ；更现实的是：它能给 提供⼀个额外、低维、强泛化的 健康严重度
450 400 Ch2 “ /

复发⻛险 信号，有机会帮你从平台期⾥抠出⼀点 。原因在数据设计⾥写得很清楚：三题
” LB
共享同⼀套 （同⼀个 ），⽽ 的疾病类别也⾼度
patients.csv patient_id join key Ch1/Ch2
重合（ 的 与 的 都是
Ch1 primary_dx Ch2 primary_chronic HF / Pneumonia /
），所以 读⼊院 出院⽂本学到的⻛险 在机制上和 未来 使⽤ 成本 是相
DiabetesComp “ + ” “ ED / ”
关的（同⼀个⼈、同⼀类慢病、同样的保险 年龄等⼈⼝学信息）。
/ description
你该怎么把 的强结果 转化 为 的增益（最稳、最不容易过拟合的做法）
Ch1 “ ” Ch2
只做⼀件事：把 模型的预测当成 的低维特征（不要把 那套⾼维⽂本直接搬进
Ch1 Ch2 Ch1
）。
Ch2
具体落地建议（按收益 ⻛险⽐排序）：
/
级 特征（推荐）
1. Ch1→patient “readmit risk”
在 （ 级）上做 ，得到每条 的
Ch1 admission 5-fold OOF admission
（不要⽤ 概率）。
p(readmit_30d) in-sample
按 聚合成 个低维特征：
patient_id 3–6 risk_mean / risk_max / risk_std /
。
n_admissions
进 的 ，作为额外输⼊。
merge Ch2 ed_cost_train/test
把 的出院 只做 主题低维压缩 再聚合（可选）
2. Ch1 note “ ”
出院 维 按 做均值 最⼤值聚合；维度控
note TF-IDF → SVD(8~16 ) → patient_id /
制住。
别做的事（容易害你继续卡 ）
3. 450
直接把 的⾼维 向量整块塞进 （⼩样本 很容易过拟合）。
Ch1 TF-IDF Ch2 2000
⽤ 的真实标签 做 聚合特征（虽然不是 标签泄漏，
Ch1 readmit_30d patient Ch2
但会让 变得 虚⾼ 、 不动的概率更⼤）。
CV “ ” LB
最快验证 到底有没有证明影响
“ ”
你不⽤猜，做⼀个最⼩实验就⾏：
在你当前最稳的 低维 基础上，只加上⾯那 个 特
Code18/ CatBoost 3–6 ch1_risk_*
征；
看 你那个 是否⼀起下降（只降 不降 ，就说
OOF MAE + sanity holdout OOF holdout
明没泛化价值）。
如果你愿意我直接 落地到代码 ，你把你 现在那套最强 （哪怕只贴 训练 产
“ ” Ch1 pipeline +
出预测概率 那⼀段）贴出来，我会把它改成：输出 级 特征并⾃动 到
patient risk merge
（保持你 ⻛格，不引⼊⼤规模⽂本特征）。
Ch2 low-dim description

Sources
```

### Prompt-3（你贴：Ch1 best submission + Ch2 best Code18）

```text
这是我们
current best submission data for ch1
this is our current best code (MAE 449):
# === CODE 18 / "Code17++" (built directly on your ~451
Iter7 / v3-Iter15/16 spirit) ===
# Less is more: keep LOW-DIM + shallow CatBoost +
strong regularization + multi-seed bagging + STABLE
ensemble.
# Changes vs your Iter7:
# (1) Receipts: add ONLY a few robust "price level" +
"concentration" features (median unit_price, top1/top3
share, gini, max_line_total)
# (2) Models: still 3 models, but explicitly add
subsample(0.8)+Bernoulli to match code16 anti-overfit;
slightly stronger L2/min_leaf
# (3) Aggregation: use trimmed-mean across seeds (drop
min/max) for robustness (often helps LB)
# (4) Test preds: optional light "full-train per seed" blend
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
random.seed(SEED)
np.random.seed(SEED)
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
TRIM_K = 1 # with 5 seeds -> drop min/max and
average middle 3

# ensemble search
W_STEP = 0.05
LAM_GRID = [0.00, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20] #
allow a bit more baseline blend
SHIFT_GRID = [0.0, 0.5, 1.0]
# stability objective (LB-oriented)
STD_PEN = 0.20
LAM_PEN = 4.0
SHIFT_PEN = 0.001
# optional small full-train-per-seed blend (often helps a
bit; still cheap)
USE_FULLFIT_BLEND = True
FULLFIT_BLEND_W = 0.35 # final test_pred = (1-
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
params_full["iterations"] = it_use # no early
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

print(f" Seed {seed_idx+1}/{CFG.N_SEEDS} OOF MAE:
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
print(f" {m:18s}: {mean_absolute_error(y,
avg_oof):.2f}")
print("\n[median best_iteration per model] (reference)")
for m in best_iters.keys():
if best_iters[m]:
print(f" {m:18s}: {int(np.median(best_iters[m]))}")
else:
print(f" {m:18s}: (n/a)")
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
print(f" #{i:02d} obj={obj:.3f} meanMAE={mean_m:.3f}
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
test = pd.read_csv(TEST_PATH)
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
print(" admissions features: None")
else:
print(f" admissions features: {adm_df.shape} | cols=
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
print(f" receipt_feat shape: {rcpt_df.shape}")
print(f" receipt_feat cols ({len(rcpt_df.columns)-1}):
{[c for c in rcpt_df.columns if c!='patient_id']}")
else:
print(" [warn] could not build receipt features from
joblib structure.")
except Exception as e:

print(f" [warn] receipts joblib load/build failed: {e}")
rcpt_df = None
else:
print(" [warn] receipts joblib missing; skipping receipts
features.")
# build features
print("\n[features] building train/test feature frames...")
train_feat = build_features(train, patients, adm_df, rcpt_df)
test_feat = build_features(test, patients, adm_df, rcpt_df)
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
print(f" FULL feature count: {len(feat_full)}")
print(f" PRUNED feature count: {len(feat_pruned)}")
print(f" PRUNED features: {feat_pruned}")
# safety fill
for c in set(feat_full + feat_pruned):
med = train_feat[c].median() if c in train_feat.columns
and not train_feat[c].isna().all() else 0.0
train_feat[c] = pd.to_numeric(train_feat[c],
errors="coerce").replace([np.inf,-np.inf], np.nan).fillna(med)
test_feat[c] = pd.to_numeric(test_feat[c],
errors="coerce").replace([np.inf,-np.inf], np.nan).fillna(med)
# train
oof_by_seed, test_by_seed, best_iters =
train_models(train_feat, test_feat, feat_full, feat_pruned)
# baseline vectors for blending
baseline_oof =
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
print(f" ensemble OOF MAE (stable search + optional

chronic shift): {final_oof_mae:.3f}")
print(" ensemble meta:", meta)
print(" extra shift:", best_shift["type"],
("shrink="+str(best_shift.get("shrink")) if
best_shift["type"]!="none" else ""))
print(" OOF pred quantiles:", qdict(best_oof, qs=
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
```

### Assistant-3（对应 Prompt-3：为什么 submission 文件不够；给出 aux Ch1→patient risk 的 PATCH）

```text
你这个 提交⽂件 （ ）本⾝只
Ch1 submission_v25_global.csv admission_id, readmit_30d
能告诉我们 上每条 的 预测，没法 直接 对 产⽣稳定增
admissions_test admission 0/1 “ ” Ch2
益，原因很简单：
是 级回归； 是 级分类（需要先把 聚合到
Ch2 patient_id Ch1 admission_id admission
）。
patient description description
你这份提交 只覆盖 （ ⾏），训练集 没有对应 同
admissions_test 5000 admissions_train “
分布 的预测（只有真标签）。如果你⽤ ⽤真标签、 ⽤预测标签 ， 会出现
” “train test ” Ch2
明显 ，通常 看起来变好但 不动。
train/test feature mismatch OOF LB
✅ 最稳的玩法是：在 代码⾥训练⼀个 辅助 模型 ，产出：
Ch2 “ Ch1 ”
的 概率（不是⽤真标签）
admissions_train OOF
的 概率
admissions_test full-fit
然后把这两部分拼起来，按 聚合成 个低维 特征
patient_id 4~6 risk
（ ）， 进你当前 的特征表。
mean/max/std/count/entropy merge Code18
下⾯我给你⼀个 直接可粘贴的补丁（你只需要把它插进你 ⾥⼏个位置；不改你现
“ ” Code18
有主结构）。
✅ ：把 ⻛险信号做成 的低维特征（ ⼀致、可缓存）
PATCH Ch1 Ch2 OOF
你把下⾯代码插⼊到你的 ⾥：
Code18
在 ⾥新增
1. “ Paths DISCHARGE_NOTES_PATH ”
在 后新增（ ）
2. “ imports sklearn text + CatBoostClassifier ”
新增函数
3. “ build_ch1_patient_risk_features ”
增加⼀个参数 并
4. “build_features() ch1_risk_df merge”
加上新特征名
5. “pruned_candidates ”
⾥在 前先算
6. “main build_features ch1_risk_df ”
python
# =============================
# (1) PATHS: add this with your other paths
# =============================
DISCHARGE_NOTES_PATH = r"D:\AgentDs\agent_ds_healthcare\discharge_notes.json"
CH1_RISK_CACHE_PATH = r"D:\AgentDs\agent_ds_healthcare\cache_ch1_patient_risk.jo
# =============================
# (2) IMPORTS: add these near your sklearn/catboost imports

# =============================
import json
try:
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import f1_score
except Exception:
_pip_install("scikit-learn")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import f1_score
try:
from joblib import dump as joblib_dump
except Exception:
_pip_install("joblib")
from joblib import dump as joblib_dump
try:
from catboost import CatBoostClassifier
except Exception:
_pip_install("catboost")
from catboost import CatBoostClassifier
# =============================
# (3) NEW: build ch1 patient-level risk features (OOF-consistent)
# =============================
def _load_discharge_notes(notes_path: str) -> pd.DataFrame:
if not os.path.exists(notes_path):
return pd.DataFrame(columns=["admission_id", "note"])
try:
with open(notes_path, "r", encoding="utf-8") as f:
data = json.load(f)
df = pd.DataFrame(data)
if "admission_id" not in df.columns:
return pd.DataFrame(columns=["admission_id", "note"])
if "note" not in df.columns:
df["note"] = ""
df["admission_id"] = pd.to_numeric(df["admission_id"], errors="coerce")

df = df.dropna(subset=["admission_id"]).copy()
df["admission_id"] = df["admission_id"].astype(int)
df["note"] = df["note"].astype(str).fillna("")
return df[["admission_id", "note"]]
except Exception:
return pd.DataFrame(columns=["admission_id", "note"])
def build_ch1_patient_risk_features(adm_train_path: str,
adm_test_path: str,
patients_df: pd.DataFrame,
notes_path: str,
cache_path: str,
n_splits: int = 5,
svd_dim: int = 16,
seed: int = 42) -> pd.DataFrame:
"""
Returns patient-level features:
- ch1_readmit_p_mean / max / std
- ch1_n_admissions
- ch1_dx_entropy (how mixed the dx is across admissions)
Uses OOF probs for admissions_train and full-fit probs for admissions_test.
"""
if os.path.exists(cache_path):
try:
obj = joblib_load(cache_path)
if isinstance(obj, pd.DataFrame) and "patient_id" in obj.columns:
return obj
except Exception:
pass
if (not os.path.exists(adm_train_path)) or (not os.path.exists(adm_test_path)
print("[ch1-risk] admissions_train/test missing -> skip ch1 risk.")
return pd.DataFrame(columns=["patient_id","ch1_readmit_p_mean","ch1_readm
adm_tr = pd.read_csv(adm_train_path)
adm_te = pd.read_csv(adm_test_path)
# basic schema guard
need_cols = ["admission_id","patient_id","primary_dx","los_days","acuity_emer

for c in need_cols:
if c not in adm_tr.columns:
print(f"[ch1-risk] missing col in admissions_train: {c} -> skip ch1 r
return pd.DataFrame(columns=["patient_id","ch1_readmit_p_mean","ch1_r
if c not in adm_te.columns:
print(f"[ch1-risk] missing col in admissions_test: {c} -> skip ch1 ri
return pd.DataFrame(columns=["patient_id","ch1_readmit_p_mean","ch1_r
if "readmit_30d" not in adm_tr.columns:
print("[ch1-risk] admissions_train has no readmit_30d -> skip ch1 risk.")
return pd.DataFrame(columns=["patient_id","ch1_readmit_p_mean","ch1_readm
# ids
for df in [adm_tr, adm_te]:
df["admission_id"] = pd.to_numeric(df["admission_id"], errors="coerce")
df["patient_id"] = pd.to_numeric(df["patient_id"], errors="coerce")
df.dropna(subset=["admission_id","patient_id"], inplace=True)
df["admission_id"] = df["admission_id"].astype(int)
df["patient_id"] = df["patient_id"].astype(int)
adm_tr["is_train"] = 1
adm_te["is_train"] = 0
adm_all = pd.concat([adm_tr, adm_te], ignore_index=True)
# merge patients
p = patients_df.copy()
p["patient_id"] = pd.to_numeric(p["patient_id"], errors="coerce")
p = p.dropna(subset=["patient_id"]).copy()
p["patient_id"] = p["patient_id"].astype(int)
p["age"] = pd.to_numeric(p["age"], errors="coerce").fillna(p["age"].median()
p["sex_encoded"] = (p["sex"].astype(str).str.upper() == "M").astype(int)
ins = p["insurance"].astype(str).str.lower()
ins_map = {"private":2, "public":1, "self_pay":0, "selfpay":0}
p["insurance_encoded"] = ins.map(ins_map).fillna(-1).astype(float)
z3 = p["zip3"].apply(standardize_zip3).astype("string")
zr = z3.fillna("000").str.replace(r"\D","", regex=True).str.zfill(3).str[0]
p["zip_region"] = pd.to_numeric(zr, errors="coerce").fillna(-1).astype(float)
adm_all = adm_all.merge(p[["patient_id","age","sex_encoded","insurance_encode

# merge notes
notes_df = _load_discharge_notes(notes_path)
if not notes_df.empty:
adm_all = adm_all.merge(notes_df, on="admission_id", how="left")
else:
adm_all["note"] = ""
adm_all["note"] = adm_all["note"].astype(str).fillna("")
# dx encoding
dx_map = {"PNEUMONIA":0, "DIABETESCOMP":1, "HF":2}
adm_all["primary_dx"] = adm_all["primary_dx"].astype(str)
adm_all["dx_encoded"] = adm_all["primary_dx"].str.upper().map(dx_map).fillna(
# numeric features
num_cols = ["los_days","acuity_emergent","charlson_band","ed_visits_6m","disc
"age","sex_encoded","insurance_encoded","zip_region","dx_encoded"
for c in num_cols:
adm_all[c] = pd.to_numeric(adm_all[c], errors="coerce").replace([np.inf,-
adm_all[c] = adm_all[c].fillna(adm_all[c].median() if not adm_all[c].isna
# note length (tiny but useful)
adm_all["note_len"] = adm_all["note"].str.len().clip(0, 500)
num_cols2 = num_cols + ["note_len"]
# text -> tfidf -> svd (low-dim)
X_num = adm_all[num_cols2].values.astype(float)
use_text = True
try:
tfidf = TfidfVectorizer(
max_features=12000,
ngram_range=(1,2),
min_df=2,
)
X_t = tfidf.fit_transform(adm_all["note"].values)
# guard svd_dim
max_svd = min(svd_dim, X_t.shape[1]-1) if X_t.shape[1] > 1 else 0
if max_svd <= 0:
use_text = False

X_svd = np.zeros((len(adm_all), 0), dtype=float)
else:
svd = TruncatedSVD(n_components=max_svd, random_state=seed)
X_svd = svd.fit_transform(X_t).astype(float)
except Exception:
use_text = False
X_svd = np.zeros((len(adm_all), 0), dtype=float)
X_all = np.hstack([X_num, X_svd])
# split train/test admissions
tr_mask = (adm_all["is_train"].values.astype(int) == 1)
X_tr = X_all[tr_mask]
y_tr = pd.to_numeric(adm_all.loc[tr_mask, "readmit_30d"], errors="coerce").fi
X_te = X_all[~tr_mask]
# stratify by dx + label for stable OOF
strat = (adm_all.loc[tr_mask, "primary_dx"].astype(str).str.upper().fillna("U
strat = LabelEncoder().fit_transform(strat)
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
oof = np.zeros(len(y_tr), dtype=float)
# lightweight, regularized classifier (LB-safe: we only need a smooth risk sig
base_params = dict(
loss_function="Logloss",
eval_metric="Logloss",
depth=4,
learning_rate=0.05,
iterations=2200,
l2_leaf_reg=10,
min_data_in_leaf=60,
rsm=0.85,
bootstrap_type="Bernoulli",
subsample=0.85,
random_strength=1.2,
task_type="CPU",
thread_count=-1,
verbose=0,
allow_writing_files=False,

)
for fold, (ti, vi) in enumerate(skf.split(X_tr, strat), 1):
cb = CatBoostClassifier(
**base_params,
random_seed=int(seed + 1000 + fold*37),
early_stopping_rounds=120,
)
cb.fit(X_tr[ti], y_tr[ti], eval_set=(X_tr[vi], y_tr[vi]), verbose=0)
oof[vi] = cb.predict_proba(X_tr[vi])[:, 1]
del cb
gc.collect()
# quick sanity
try:
pred_lbl = (oof >= 0.5).astype(int)
mf1 = float(f1_score(y_tr, pred_lbl, average="macro"))
print(f"[ch1-risk] OOF macro-F1 (aux model, threshold=0.5): {mf1:.4f} | t
except Exception:
pass
# full-fit -> admissions_test probs
cb_full = CatBoostClassifier(**base_params, random_seed=int(seed + 9999))
cb_full.fit(X_tr, y_tr, verbose=0)
p_te = cb_full.predict_proba(X_te)[:, 1] if len(X_te) else np.array([], dtype
del cb_full
gc.collect()
# attach preds back to admissions
adm_pred_tr = adm_all.loc[tr_mask, ["admission_id","patient_id","primary_dx"]
adm_pred_tr["readmit_p"] = oof
adm_pred_te = adm_all.loc[~tr_mask, ["admission_id","patient_id","primary_dx"
adm_pred_te["readmit_p"] = p_te
adm_pred = pd.concat([adm_pred_tr, adm_pred_te], ignore_index=True)
# patient aggregates
g = adm_pred.groupby("patient_id")["readmit_p"]
out = pd.DataFrame({
"patient_id": g.mean().index.astype(int),

"ch1_readmit_p_mean": g.mean().values.astype(float),
"ch1_readmit_p_max": g.max().values.astype(float),
"ch1_readmit_p_std": g.std().fillna(0.0).values.astype(float),
"ch1_n_admissions": g.size().values.astype(int),
})
# dx entropy (1 feature)
dx_ct = adm_pred.groupby(["patient_id","primary_dx"]).size().unstack(fill_val
dx_prop = dx_ct.div(dx_ct.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
ent = -(dx_prop * np.log(dx_prop + 1e-12)).sum(axis=1)
out = out.merge(ent.rename("ch1_dx_entropy").reset_index(), on="patient_id",
out["ch1_dx_entropy"] = pd.to_numeric(out["ch1_dx_entropy"], errors="coerce")
# cache
try:
joblib_dump(out, cache_path)
print(f"[ch1-risk] cached -> {cache_path}")
except Exception:
pass
return out
# =============================
# (4) CHANGE: update build_features signature and merge ch1 risk df inside
# =============================
# Change your build_features(...) definition to add ch1_risk_df param:
# def build_features(ed_df, patients_df, adm_df, rcpt_df, ch1_risk_df):
# Then inside build_features, after merging patients/admissions/receipts, add:
# ---- add this block inside build_features() ----
# if ch1_risk_df is not None and (not ch1_risk_df.empty):
# feat = feat.merge(ch1_risk_df, on="patient_id", how="left")
# for c in ["ch1_readmit_p_mean","ch1_readmit_p_max","ch1_readmit_p_std","
# if c in feat.columns:
# feat[c] = pd.to_numeric(feat[c], errors="coerce").replace([np.inf
# feat[c] = feat[c].fillna(feat[c].median() if not feat[c].isna().
#
# # tiny interactions (still low-risk)

# if "ch1_readmit_p_mean" in feat.columns:
# feat["logprior_x_readmitp"] = feat["log_prior_cost"] * feat["ch1_rea
# feat["visits_x_readmitp"] = feat["log_visits"] * feat["ch1_rea
# =============================
# (5) CHANGE: add these to your pruned_candidates list
# =============================
# "ch1_readmit_p_mean","ch1_readmit_p_max","ch1_readmit_p_std","ch1_n_admission
# "logprior_x_readmitp","visits_x_readmitp",
# =============================
# (6) MAIN: compute ch1_risk_df before building train/test features
# =============================
# right before:
# train_feat = build_features(train, patients, adm_df, rcpt_df)
# add:
# print("\n[ch1-risk] building auxiliary patient readmission risk features (OOF-c
# ch1_risk_df = build_ch1_patient_risk_features(
# ADM_TRAIN_PATH, ADM_TEST_PATH, patients, DISCHARGE_NOTES_PATH,
# cache_path=CH1_RISK_CACHE_PATH, n_splits=5, svd_dim=16, seed=SEED
# )
# print(f" ch1 risk features: {ch1_risk_df.shape} | cols={list(ch1_risk_df.column
# then call:
# train_feat = build_features(train, patients, adm_df, rcpt_df, ch1_risk_df)
# test_feat = build_features(test, patients, adm_df, rcpt_df, ch1_risk_df)
你跑完后，最关键看这 个信号
2
（代码⾥会 ⼀个⼤概值）
1. Ch1-risk OOF macro-F1 print
的 是否下降，同时你看 是否动
2. Ch2 FINAL OOF MAE LB
如果你这步能让 ⼩降但 终于动了，基本就说明：你之前卡 的问题，确实是 缺
OOF LB 450
⼀个 跨数据源的稳定严重度信号（⽽不是继续堆 细节）。
“ ” receipt

如果你希望我把它直接整合成 完整的 ⼀整段可替换 （你直接复制跑，不
“ Code19 one-cell”
⽤⼿动改 处），我也可以直接给你整合版。
6
Sources
```

### Prompt-4（你要求：one-cell 完整代码）

```text
give me full code in 1 cell!!!
```

### Assistant-4（输出：Code19 one-cell 全量代码）

```text
python
# === CODE 19 / "Code18 + CH1-risk bridge" ===
# Goal: push LB below ~449 by adding ONE low-dim, OOF-consistent "readmission risk
# (admissions_train readmit_30d) + discharge notes -> patient-level risk agg
# Keep the Code18 spirit: LOW-DIM + shallow CatBoost + strong regularization + mu
import os, re, sys, gc, math, warnings, random, zlib, json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
pd.set_option("display.max_columns", 250)
pd.set_option("display.width", 200)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
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
DISCHARGE_NOTES_PATH = r"D:\AgentDs\agent_ds_healthcare\discharge_notes.json"
# caches
CH1_RISK_CACHE_PATH = r"D:\AgentDs\agent_ds_healthcare\cache_ch1_patient_risk.jo
OUT_SUB_PATH = r"D:\AgentDs\agent_ds_healthcare\submission.csv"
ID_COL = "patient_id"
TARGET = "ed_cost_next3y_usd"
print("="*110)
print("CODE 19 | Code18 + CH1-risk bridge: LOW-DIM receipts + shallow CatBoost +
print("Add-on: train auxiliary CH1 readmit model (OOF-consistent) -> patient risk
print("="*110)
# -----------------------------
# Minimal deps
# -----------------------------
def _pip_install(pkg: str):
import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
try:
from joblib import load as joblib_load, dump as joblib_dump
except Exception:
_pip_install("joblib")
from joblib import load as joblib_load, dump as joblib_dump
try:
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD
except Exception:
_pip_install("scikit-learn")
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
try:
from catboost import CatBoostRegressor, CatBoostClassifier
except Exception:
_pip_install("catboost")
from catboost import CatBoostRegressor, CatBoostClassifier
# -----------------------------
# Config (keep fast; code16-like regularization)
# -----------------------------
class CFG:
# Ch2 main
N_FOLDS = 7
N_SEEDS = 5
ITERS = 3200
ES_ROUNDS = 130
LR = 0.03
RSM = 0.80
SUBSAMPLE = 0.80
TRIM_K = 1 # with 5 seeds -> drop min/max and average middle 3
# ensemble search
W_STEP = 0.05
LAM_GRID = [0.00, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
SHIFT_GRID = [0.0, 0.5, 1.0]
# stability objective (LB-oriented)
STD_PEN = 0.20
LAM_PEN = 4.0
SHIFT_PEN = 0.001

# optional small full-train-per-seed blend (often helps a bit; still cheap)
USE_FULLFIT_BLEND = True
FULLFIT_BLEND_W = 0.35 # final test_pred = (1-w)*foldbag + w*fullfit
# CH1 aux model -> patient risk features
CH1_N_SPLITS = 5
CH1_SVD_DIM = 16
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
"""
mat: (n_seeds, n_samples)
if n_seeds >= 2*trim_k + 1, drop extremes along axis=0 then mean.
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
# Admissions features (keep simple, robust) for Ch2 main

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
# CH1 AUX: discharge notes loader + patient-level risk features (OOF-consistent)
# -----------------------------
def _load_discharge_notes(notes_path: str) -> pd.DataFrame:

if not os.path.exists(notes_path):
return pd.DataFrame(columns=["admission_id", "note"])
try:
with open(notes_path, "r", encoding="utf-8") as f:
data = json.load(f)
df = pd.DataFrame(data)
if "admission_id" not in df.columns:
return pd.DataFrame(columns=["admission_id", "note"])
if "note" not in df.columns:
df["note"] = ""
df["admission_id"] = pd.to_numeric(df["admission_id"], errors="coerce")
df = df.dropna(subset=["admission_id"]).copy()
df["admission_id"] = df["admission_id"].astype(int)
df["note"] = df["note"].astype(str).fillna("")
return df[["admission_id", "note"]]
except Exception:
return pd.DataFrame(columns=["admission_id", "note"])
def build_ch1_patient_risk_features(adm_train_path: str,
adm_test_path: str,
patients_df: pd.DataFrame,
notes_path: str,
cache_path: str,
n_splits: int = 5,
svd_dim: int = 16,
seed: int = 42) -> pd.DataFrame:
"""
Returns patient-level features:
- ch1_readmit_p_mean / max / std
- ch1_n_admissions
- ch1_dx_entropy
Uses OOF probs for admissions_train and full-fit probs for admissions_test.
"""
if os.path.exists(cache_path):
try:
obj = joblib_load(cache_path)
if isinstance(obj, pd.DataFrame) and "patient_id" in obj.columns:
print(f"[ch1-risk] loaded cache -> {cache_path}")
return obj
except Exception:

pass
if (not os.path.exists(adm_train_path)) or (not os.path.exists(adm_test_path)
print("[ch1-risk] admissions_train/test missing -> skip ch1 risk.")
return pd.DataFrame(columns=["patient_id","ch1_readmit_p_mean","ch1_readm
adm_tr = pd.read_csv(adm_train_path)
adm_te = pd.read_csv(adm_test_path)
if "readmit_30d" not in adm_tr.columns:
print("[ch1-risk] admissions_train has no readmit_30d -> skip ch1 risk.")
return pd.DataFrame(columns=["patient_id","ch1_readmit_p_mean","ch1_readm
# required cols
need_cols = ["admission_id","patient_id","primary_dx","los_days","acuity_emer
for c in need_cols:
if c not in adm_tr.columns or c not in adm_te.columns:
print(f"[ch1-risk] missing col: {c} -> skip ch1 risk.")
return pd.DataFrame(columns=["patient_id","ch1_readmit_p_mean","ch1_r
# ids
for df in [adm_tr, adm_te]:
df["admission_id"] = pd.to_numeric(df["admission_id"], errors="coerce")
df["patient_id"] = pd.to_numeric(df["patient_id"], errors="coerce")
df.dropna(subset=["admission_id","patient_id"], inplace=True)
df["admission_id"] = df["admission_id"].astype(int)
df["patient_id"] = df["patient_id"].astype(int)
adm_tr = adm_tr.copy()
adm_te = adm_te.copy()
adm_tr["is_train"] = 1
adm_te["is_train"] = 0
adm_all = pd.concat([adm_tr, adm_te], ignore_index=True)
# merge patients (low-dim demographics)
p = patients_df.copy()
p["patient_id"] = pd.to_numeric(p["patient_id"], errors="coerce")
p = p.dropna(subset=["patient_id"]).copy()
p["patient_id"] = p["patient_id"].astype(int)
p["age"] = pd.to_numeric(p["age"], errors="coerce")

p["age"] = p["age"].fillna(p["age"].median() if not p["age"].isna().all() els
p["sex_encoded"] = (p["sex"].astype(str).str.upper() == "M").astype(int)
ins = p["insurance"].astype(str).str.lower()
ins_map = {"private":2, "public":1, "self_pay":0, "selfpay":0}
p["insurance_encoded"] = ins.map(ins_map).fillna(-1).astype(float)
z3 = p["zip3"].apply(standardize_zip3).astype("string")
zr = z3.fillna("000").str.replace(r"\D","", regex=True).str.zfill(3).str[0]
p["zip_region"] = pd.to_numeric(zr, errors="coerce").fillna(-1).astype(float)
adm_all = adm_all.merge(p[["patient_id","age","sex_encoded","insurance_encode
# merge notes
notes_df = _load_discharge_notes(notes_path)
if not notes_df.empty:
adm_all = adm_all.merge(notes_df, on="admission_id", how="left")
else:
adm_all["note"] = ""
adm_all["note"] = adm_all["note"].astype(str).fillna("")
# dx encoding
dx_map = {"PNEUMONIA":0, "DIABETESCOMP":1, "HF":2}
adm_all["primary_dx"] = adm_all["primary_dx"].astype(str)
adm_all["dx_encoded"] = adm_all["primary_dx"].str.upper().map(dx_map).fillna(
# numeric features
num_cols = ["los_days","acuity_emergent","charlson_band","ed_visits_6m","disc
"age","sex_encoded","insurance_encoded","zip_region","dx_encoded"
for c in num_cols:
adm_all[c] = pd.to_numeric(adm_all[c], errors="coerce").replace([np.inf,-
adm_all[c] = adm_all[c].fillna(adm_all[c].median() if not adm_all[c].isna
adm_all["note_len"] = adm_all["note"].str.len().clip(0, 800)
X_num = adm_all[num_cols + ["note_len"]].values.astype(float)
# text -> tfidf -> svd (low-dim)
use_text = True
X_svd = np.zeros((len(adm_all), 0), dtype=float)
try:
tfidf = TfidfVectorizer(
max_features=12000,

ngram_range=(1,2),
min_df=2
)
X_t = tfidf.fit_transform(adm_all["note"].values)
max_svd = min(svd_dim, X_t.shape[1]-1) if X_t.shape[1] > 1 else 0
if max_svd <= 0:
use_text = False
else:
svd = TruncatedSVD(n_components=max_svd, random_state=seed)
X_svd = svd.fit_transform(X_t).astype(float)
except Exception:
use_text = False
X_svd = np.zeros((len(adm_all), 0), dtype=float)
X_all = np.hstack([X_num, X_svd])
tr_mask = (adm_all["is_train"].values.astype(int) == 1)
X_tr = X_all[tr_mask]
y_tr = pd.to_numeric(adm_all.loc[tr_mask, "readmit_30d"], errors="coerce").fi
X_te = X_all[~tr_mask]
# stratify by dx + label
strat = (adm_all.loc[tr_mask, "primary_dx"].astype(str).str.upper().fillna("U
strat = LabelEncoder().fit_transform(strat)
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
oof = np.zeros(len(y_tr), dtype=float)
base_params = dict(
loss_function="Logloss",
eval_metric="Logloss",
depth=4,
learning_rate=0.05,
iterations=2200,
l2_leaf_reg=10,
min_data_in_leaf=60,
rsm=0.85,
bootstrap_type="Bernoulli",
subsample=0.85,
random_strength=1.2,

task_type="CPU",
thread_count=-1,
verbose=0,
allow_writing_files=False,
)
for fold, (ti, vi) in enumerate(skf.split(X_tr, strat), 1):
cb = CatBoostClassifier(
**base_params,
random_seed=int(seed + 1000 + fold*37),
early_stopping_rounds=120,
)
cb.fit(X_tr[ti], y_tr[ti], eval_set=(X_tr[vi], y_tr[vi]), verbose=0)
oof[vi] = cb.predict_proba(X_tr[vi])[:, 1]
del cb
gc.collect()
try:
pred_lbl = (oof >= 0.5).astype(int)
mf1 = float(f1_score(y_tr, pred_lbl, average="macro"))
print(f"[ch1-risk] aux OOF macro-F1@0.5: {mf1:.4f} | text={use_text} | sv
except Exception:
pass
# full-fit for admissions_test
cb_full = CatBoostClassifier(**base_params, random_seed=int(seed + 9999))
cb_full.fit(X_tr, y_tr, verbose=0)
p_te = cb_full.predict_proba(X_te)[:, 1] if len(X_te) else np.array([], dtype
del cb_full
gc.collect()
# attach preds back to admissions
adm_pred_tr = adm_all.loc[tr_mask, ["admission_id","patient_id","primary_dx"]
adm_pred_tr["readmit_p"] = oof
adm_pred_te = adm_all.loc[~tr_mask, ["admission_id","patient_id","primary_dx"
adm_pred_te["readmit_p"] = p_te
adm_pred = pd.concat([adm_pred_tr, adm_pred_te], ignore_index=True)
# patient aggregates
g = adm_pred.groupby("patient_id")["readmit_p"]

out = pd.DataFrame({
"patient_id": g.mean().index.astype(int),
"ch1_readmit_p_mean": g.mean().values.astype(float),
"ch1_readmit_p_max": g.max().values.astype(float),
"ch1_readmit_p_std": g.std().fillna(0.0).values.astype(float),
"ch1_n_admissions": g.size().values.astype(int),
})
# dx entropy
dx_ct = adm_pred.groupby(["patient_id","primary_dx"]).size().unstack(fill_val
dx_prop = dx_ct.div(dx_ct.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
ent = -(dx_prop * np.log(dx_prop + 1e-12)).sum(axis=1)
out = out.merge(ent.rename("ch1_dx_entropy").reset_index(), on="patient_id",
out["ch1_dx_entropy"] = pd.to_numeric(out["ch1_dx_entropy"], errors="coerce")
# cache
try:
joblib_dump(out, cache_path)
print(f"[ch1-risk] cached -> {cache_path}")
except Exception:
pass
return out
# -----------------------------
# Low-dim receipts features from parsed lineitems (v3-style, + tiny robust add-on
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
receipt_total = li.groupby("patient_id")["amt"].sum().rename("receipt_total")

li = li.join(receipt_total, on="patient_id")
denom = li["receipt_total"].replace(0.0, np.nan)
share = (li["amt"] / denom).fillna(0.0)
# concentration
cost_hhi = (share * share).groupby(li["patient_id"]).sum().rename("cost_hhi")
top1_share = share.groupby(li["patient_id"]).max().rename("top1_share")
def _topk_sum(s, k=3):
a = np.sort(s.values.astype(float))[::-1]
return float(a[:k].sum()) if a.size else 0.0
top3_share = share.groupby(li["patient_id"]).apply(lambda s: _topk_sum(s, 3))
gini_amt = li.groupby("patient_id")["amt"].apply(lambda s: gini(s.values)).re
max_line_total = li.groupby("patient_id")["amt"].max().rename("max_line_total
code_num = pd.to_numeric(li["code"].where(li["code"].str.fullmatch(r"\d+"), N
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
n_em_codes = is_em.astype(int).groupby(li["patient_id"]).sum().rename("n_em_c
max_em_level = em_level.groupby(li["patient_id"]).max().rename("max_em_level"
sum_em_level = (em_level * is_em.astype(int)).groupby(li["patient_id"]).sum()
avg_em_level = (sum_em_level / n_em_codes.replace(0, np.nan)).fillna(0.0).ren
n_high_em = ((em_level >= 4) & is_em).astype(int).groupby(li["patient_id"]).s
em_total = li.loc[is_em].groupby("patient_id")["amt"].sum().rename("em_total"
crit_total = li.loc[is_crit].groupby("patient_id")["amt"].sum().rename("crit_t

proc_total = li.loc[is_proc_any].groupby("patient_id")["amt"].sum().rename("p
img_total = li.loc[is_imaging].groupby("patient_id")["amt"].sum().rename("img_
lab_total = li.loc[is_lab].groupby("patient_id")["amt"].sum().rename("lab_tot
high_total = li.loc[is_high].groupby("patient_id")["amt"].sum().rename("high_t
n_procedures = is_proc_any.astype(int).groupby(li["patient_id"]).sum().rename
n_imaging = is_imaging.astype(int).groupby(li["patient_id"]).sum().rename("n_
n_lab = is_lab.astype(int).groupby(li["patient_id"]).sum().rename("n_lab")
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
# price level add-ons: median unit_price overall + by bucket
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
out = pd.concat([
n_unique_codes,
receipt_total,
cost_hhi, top1_share, top3_share, gini_amt, max_line_total,
n_em_codes, max_em_level, avg_em_level, n_high_em,
has_critical_care, has_high_acuity, has_observation, has_imaging,
has_intub_31500, has_cvc_36556, has_cpr_92950, has_artline_36620,
has_ct_head_70450, has_99285, has_ct_abdpel_74177, has_troponin_84484, ha
n_procedures, n_imaging, n_lab
], axis=1).reset_index()
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

# attach unit-price medians
out = out.merge(med_unit_all.reset_index(), on="patient_id", how="left")
if med_unit_em is not None:
out = out.merge(med_unit_em.reset_index(), on="patient_id", how="left")
else:
out["median_unit_price_em"] = np.nan
if med_unit_img is not None:
out = out.merge(med_unit_img.reset_index(), on="patient_id", how="left")
else:
out["median_unit_price_imaging"] = np.nan
if med_unit_lab is not None:
out = out.merge(med_unit_lab.reset_index(), on="patient_id", how="left")
else:
out["median_unit_price_lab"] = np.nan
for c in ["median_unit_price","median_unit_price_em","median_unit_price_imagi
out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf,-np.inf],
out["log1p_" + c] = np.log1p(out[c].clip(lower=0.0))
# cleanup helper totals
out.drop(columns=[c for c in ["em_total","crit_total","proc_total","img_total
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
# Feature engineering (numeric-only, v3 style) + CH1-risk merge
# -----------------------------
def build_features(ed_df: pd.DataFrame,
patients_df: pd.DataFrame,
adm_df: Optional[pd.DataFrame],
rcpt_df: Optional[pd.DataFrame],
ch1_risk_df: Optional[pd.DataFrame]) -> pd.DataFrame:
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
# baseline for LB-safe blending
feat["baseline_next3y"] = feat["prior_ed_cost_5y_usd"] * (3.0/5.0)
# patients encodings
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
# admissions aggregates (Ch2 classic)
if adm_df is not None:
feat = feat.merge(adm_df, on="patient_id", how="left")
for c in ["charlson_max","charlson_mean","pct_emergent"]:
if c in feat.columns:

feat[c] = safe_num_series(feat[c], default=0.0)
# receipts features
if rcpt_df is not None:
feat = feat.merge(rcpt_df, on="patient_id", how="left")
for c in rcpt_df.columns:
if c == "patient_id":
continue
feat[c] = safe_num_series(feat[c], default=np.nan)
if feat[c].isna().any():
feat[c] = feat[c].fillna(feat[c].median())
# CH1 risk features (OOF-consistent)
if ch1_risk_df is not None and (not ch1_risk_df.empty):
feat = feat.merge(ch1_risk_df, on="patient_id", how="left")
for c in ["ch1_readmit_p_mean","ch1_readmit_p_max","ch1_readmit_p_std","c
if c in feat.columns:
feat[c] = pd.to_numeric(feat[c], errors="coerce").replace([np.inf
feat[c] = feat[c].fillna(feat[c].median() if not feat[c].isna().a
if "ch1_readmit_p_mean" in feat.columns:
feat["logprior_x_readmitp"] = feat["log_prior_cost"] * feat["ch1_read
feat["visits_x_readmitp"] = feat["log_visits"] * feat["ch1_read
# a couple of LIGHT interactions (still low-risk)
if "pct_cost_critical" in feat.columns:
feat["logprior_x_pctcritical"] = feat["log_prior_cost"] * feat["pct_cost_
if "n_high_acuity_total" in feat.columns:
feat["logprior_x_highacu"] = feat["log_prior_cost"] * feat["n_high_acuity_
# derived stable ratios
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
# -----------------------------
# Training (multi-seed + fold bagging)
# -----------------------------
def train_models(train_feat: pd.DataFrame, test_feat: pd.DataFrame,
feat_full: List[str], feat_pruned: List[str]) -> Tuple[Dict[str,
y = train_feat[TARGET].values.astype(float)
# stratify: chronic + target bin (v3)
tmp = train_feat[["primary_chronic", TARGET]].copy()
tmp["cost_bin"] = pd.qcut(tmp[TARGET], q=5, labels=False, duplicates="drop")
tmp["strat"] = tmp["primary_chronic"].astype(str) + "_" + tmp["cost_bin"].asty
strat = LabelEncoder().fit_transform(tmp["strat"].values)
# 3 models (keep "less is more")

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
# optional: full-fit per seed
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
print(f" Seed {seed_idx+1}/{CFG.N_SEEDS} OOF MAE: " + " | ".join([f"{m}=
for m in model_specs.keys():
oof_by_seed[m].append(oof_seed[m])

test_by_seed[m].append(test_seed_final[m])
print("\n[seed-aggregated OOF MAE per model] (trimmed mean across seeds)")
for m in oof_by_seed.keys():
mat = np.vstack(oof_by_seed[m])
avg_oof = trimmed_mean(mat, trim_k=CFG.TRIM_K)
print(f" {m:18s}: {mean_absolute_error(y, avg_oof):.2f}")
print("\n[median best_iteration per model] (reference)")
for m in best_iters.keys():
if best_iters[m]:
print(f" {m:18s}: {int(np.median(best_iters[m]))}")
else:
print(f" {m:18s}: (n/a)")
return oof_by_seed, test_by_seed, best_iters
# -----------------------------
# Ensemble selection (stability across seeds) - for 3 models
# -----------------------------
def stable_ensemble_search(train_feat: pd.DataFrame,
oof_by_seed: Dict[str, List[np.ndarray]],
baseline_vec: np.ndarray) -> Tuple[np.ndarray, Dict]:
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
print("\n[ensemble search] Top candidates (robust objective = mean + std_pen
for i, rec in enumerate(top_list[:10], 1):
obj, mean_m, std_m, wA, wB, wC, lam, sm, shift = rec
print(f" #{i:02d} obj={obj:.3f} meanMAE={mean_m:.3f} std={std_m:.3f} | w
obj, mean_m, std_m, wA, wB, wC, lam, sm, shift = best
mA, mB, mC = model_names
oof_final = wA*oof_agg[mA] + wB*oof_agg[mB] + wC*oof_agg[mC]
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
def apply_chronic_group_shift(train_feat: pd.DataFrame, pred_oof: np.ndarray, shr
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
print("[warn] receipts_parsed.joblib missing -> receipts features will be empt

print("\n[load] reading CSVs...")
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
patients = pd.read_csv(PATIENTS_PATH)
log_shape("ed_cost_train", train)
log_shape("ed_cost_test", test)
log_shape("patients", patients)
print("\n[target stats]")
print(train[TARGET].describe().to_string())
# ids
for df in [train, test, patients]:
df[ID_COL] = pd.to_numeric(df[ID_COL], errors="coerce").astype(int)
# Ch1-risk patient features (OOF-consistent)
print("\n[ch1-risk] building auxiliary patient readmission risk features (OOF-con
try:
ch1_risk_df = build_ch1_patient_risk_features(
ADM_TRAIN_PATH, ADM_TEST_PATH,
patients_df=patients,
notes_path=DISCHARGE_NOTES_PATH,
cache_path=CH1_RISK_CACHE_PATH,
n_splits=CFG.CH1_N_SPLITS,
svd_dim=CFG.CH1_SVD_DIM,
seed=SEED
)
print(f" ch1 risk features: {ch1_risk_df.shape} | cols={list(ch1_risk_df.col
except Exception as e:
print(f" [warn] ch1 risk build failed: {e}")
ch1_risk_df = None
# admissions aggregates for Ch2 main
print("\n[admissions] building robust aggregates (Ch2 classic)...")
adm_df = load_admissions_features(ADM_TRAIN_PATH, ADM_TEST_PATH)
if adm_df is None:
print(" admissions features: None")
else:

print(f" admissions features: {adm_df.shape} | cols={list(adm_df.columns)}")
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
print(f" receipt_feat shape: {rcpt_df.shape}")
print(f" receipt_feat cols ({len(rcpt_df.columns)-1}): {[c for c in
else:
print(" [warn] could not build receipt features from joblib structur
except Exception as e:
print(f" [warn] receipts joblib load/build failed: {e}")
rcpt_df = None
else:
print(" [warn] receipts joblib missing; skipping receipts features.")
# build features
print("\n[features] building train/test feature frames...")
train_feat = build_features(train, patients, adm_df, rcpt_df, ch1_risk_df)
test_feat = build_features(test, patients, adm_df, rcpt_df, ch1_risk_df)
# choose features
feat_full = get_numeric_feature_cols(train_feat)
feat_full = [c for c in feat_full if c != TARGET]
feat_full = drop_constants(feat_full, train_feat)
# PRUNED set: stable low-dim list (extend Code18 with ONLY ch1-risk low-dim)
pruned_candidates = [
# priors + transforms
"prior_ed_visits_5y","prior_ed_cost_5y_usd","prior_cost_cap20k","sqrt_prior_c
"baseline_next3y",
# demographics
"chronic_encoded","age","sex_encoded","insurance_encoded","zip_region","ins_x_

# admissions
"charlson_max","charlson_mean","pct_emergent",
# ch1 risk
"ch1_readmit_p_mean","ch1_readmit_p_max","ch1_readmit_p_std","ch1_n_admission
"logprior_x_readmitp","visits_x_readmitp",
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
feat_pruned = [c for c in pruned_candidates if c in train_feat.columns]
feat_pruned = drop_constants(feat_pruned, train_feat)
print(f" FULL feature count: {len(feat_full)}")
print(f" PRUNED feature count: {len(feat_pruned)}")
print(f" PRUNED features: {feat_pruned}")
# safety fill
for c in set(feat_full + feat_pruned):
med = train_feat[c].median() if c in train_feat.columns and not train_feat[c]
train_feat[c] = pd.to_numeric(train_feat[c], errors="coerce").replace([np.inf
test_feat[c] = pd.to_numeric(test_feat[c], errors="coerce").replace([np.inf,
# train
oof_by_seed, test_by_seed, best_iters = train_models(train_feat, test_feat, feat_
# baseline vectors for blending
baseline_oof = train_feat["baseline_next3y"].values.astype(float)
baseline_test = test_feat["baseline_next3y"].values.astype(float)
# stable ensemble search on OOF
oof_ens, meta = stable_ensemble_search(train_feat, oof_by_seed, baseline_oof)
# build final test ensemble using chosen weights + baseline blend + shift

mA, mB, mC = meta["models_order"]
wA, wB, wC = meta["weights"]
lam = meta["lam_baseline"]
shift = meta["shift_value"]
test_agg = {m: trimmed_mean(np.vstack(test_by_seed[m]), trim_k=CFG.TRIM_K) for m
test_ens = wA*test_agg[mA] + wB*test_agg[mB] + wC*test_agg[mC]
test_ens = (1.0-lam)*test_ens + lam*baseline_test
test_ens = test_ens + shift
# optional chronic shift (very conservative; require noticeable OOF gain)
y = train_feat[TARGET].values.astype(float)
base_mae = float(mean_absolute_error(y, oof_ens))
best_oof = oof_ens
best_shift = {"type": "none"}
for shrink in [0.0, 0.20, 0.30]:
if shrink <= 0:
continue
oof2, shifts = apply_chronic_group_shift(train_feat, oof_ens, shrink=shrink)
m2 = float(mean_absolute_error(y, oof2))
if m2 < base_mae - 0.12:
base_mae = m2
best_oof = oof2
best_shift = {"type": "chronic_group", "shrink": shrink, "shifts": shifts
if best_shift["type"] == "chronic_group":
test_chronic = test_feat["primary_chronic"].astype(str).values
for g, s in best_shift["shifts"].items():
test_ens[test_chronic == g] += s
# clip predictions to a reasonable range (LB-safe)
y_max = float(np.max(y))
test_ens = np.clip(test_ens, 0.0, y_max * 1.5)
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
print("\n[Top feature importances] (Model A full)")
print(imp_df.to_string(index=False))
except Exception as e:
print(f"[warn] feature importance failed: {e}")
del mA_full
gc.collect()
# final logs
final_oof_mae = float(mean_absolute_error(y, best_oof))
print("\n" + "="*90)
print("[FINAL OOF]")
print(f" ensemble OOF MAE (stable search + optional chronic shift): {final_oof_m
print(" ensemble meta:", meta)
print(" extra shift:", best_shift["type"], ("shrink="+str(best_shift.get("shrink
print(" OOF pred quantiles:", qdict(best_oof, qs=(0,0.01,0.05,0.1,0.5,0.9,0.95,0
print("="*90)
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
print("any NaNs in preds:", bool(np.isnan(sub["ed_cost_next3y_usd"]).any()))
print("pred min/median/max:", float(sub["ed_cost_next3y_usd"].min()), float(sub["
print("pred quantiles:", qdict(sub["ed_cost_next3y_usd"].values))
print("\nSaved submission to:", str(out_path))
print("\nPaste back: (1) leaderboard MAE, (2) FINAL OOF MAE, (3) ensemble meta +
🚀
```

### Prompt-5（你贴 Code19 运行回执/日志 + 提问：好像没啥用？）

```text
Submitting predictions...
✅
Prediction submitted successfully!
📊
Score: 450.9103 (MAE)
✅
Validation passed
✅
Submission successful!
📊
Score: 450.9103
📏
Metric: MAE
✔
Validation: Passed
======================================
======================================
==================================
CODE 19 | Code18 + CH1-risk bridge: LOW-DIM receipts +
shallow CatBoost + strong reg + multi-seed + STABLE
ensemble
Add-on: train auxiliary CH1 readmit model (OOF-
consistent) -> patient risk aggregates -> feed to Ch2
regression.
======================================
======================================
==================================
[load] reading CSVs...
[ed_cost_train] shape=(2000, 5) | cols=5 | mem=0.17 MB
[ed_cost_test] shape=(2000, 4) | cols=4 | mem=0.15 MB

[patients] shape=(4000, 5) | cols=5 | mem=0.49 MB
[target stats]
count 2000.00000
mean 3908.25191
std 1822.40192
min 306.88000
25% 2548.76750
50% 3569.09500
75% 4956.42250
max 11184.61000
[ch1-risk] building auxiliary patient readmission risk
features (OOF-consistent)...
[ch1-risk] aux OOF macro-F1@0.5: 0.7034 | text=True |
svd_dim=16
[ch1-risk] cached ->
D:\AgentDs\agent_ds_healthcare\cache_ch1_patient_risk.jo
blib
ch1 risk features: (4000, 6) | cols=['patient_id',
'ch1_readmit_p_mean', 'ch1_readmit_p_max',
'ch1_readmit_p_std', 'ch1_n_admissions', 'ch1_dx_entropy']
[admissions] building robust aggregates (Ch2 classic)...
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
FULL feature count: 71
PRUNED feature count: 56
PRUNED features: ['prior_ed_visits_5y',
'prior_ed_cost_5y_usd', 'prior_cost_cap20k',
'sqrt_prior_cost', 'log_prior_cost', 'log_prior_cost_cap20k',
'cost_per_visit', 'log_visits', 'baseline_next3y',
'chronic_encoded', 'age', 'sex_encoded',
'insurance_encoded', 'zip_region', 'ins_x_chronic',
'charlson_max', 'charlson_mean', 'pct_emergent',
'ch1_readmit_p_mean', 'ch1_readmit_p_max',
'ch1_readmit_p_std', 'ch1_n_admissions', 'ch1_dx_entropy',
'logprior_x_readmitp', 'visits_x_readmitp', 'cost_per_em',
'cost_hhi', 'pct_cost_em', 'pct_cost_procedure',
'pct_cost_critical', 'pct_cost_high_acuity',
'pct_cost_imaging', 'pct_cost_lab', 'n_high_acuity_total',
'has_critical_care', 'has_99285', 'max_em_level',
'avg_em_level', 'n_high_em', 'n_unique_codes', 'top1_share',
'top3_share', 'gini_line_total', 'max_line_total',
'median_unit_price', 'median_unit_price_em',
'median_unit_price_imaging', 'median_unit_price_lab',
'log1p_median_unit_price', 'log1p_median_unit_price_em',
'log1p_median_unit_price_imaging',
'log1p_median_unit_price_lab', 'log1p_max_line_total',
'logprior_x_pctcritical', 'logprior_x_highacu',
'cost_per_code']
[training] CatBoost CPU | shallow trees | rsm=0.8 |
subsample=0.8 | multi-seed bagging
Models: ['A_RMSE_full_d5', 'B_RMSE_pruned_d4',
'C_MAE_pruned_d4']
Seeds=5, Folds=7

Seed 1/5 OOF MAE: A_RMSE_full_d5=432.68 |
B_RMSE_pruned_d4=433.02 | C_MAE_pruned_d4=438.48
Seed 2/5 OOF MAE: A_RMSE_full_d5=433.54 |
B_RMSE_pruned_d4=433.13 | C_MAE_pruned_d4=444.12
Seed 3/5 OOF MAE: A_RMSE_full_d5=435.45 |
B_RMSE_pruned_d4=433.56 | C_MAE_pruned_d4=440.99
Seed 4/5 OOF MAE: A_RMSE_full_d5=435.55 |
B_RMSE_pruned_d4=435.68 | C_MAE_pruned_d4=444.18
Seed 5/5 OOF MAE: A_RMSE_full_d5=434.33 |
B_RMSE_pruned_d4=432.19 | C_MAE_pruned_d4=443.68
[seed-aggregated OOF MAE per model] (trimmed mean
across seeds)
A_RMSE_full_d5 : 431.40
B_RMSE_pruned_d4 : 429.58
C_MAE_pruned_d4 : 438.19
[median best_iteration per model] (reference)
A_RMSE_full_d5 : 578
B_RMSE_pruned_d4 : 657
C_MAE_pruned_d4 : 669
[ensemble search] Top candidates (robust objective =
mean + std_pen + simplicity_pen):
#01 obj=432.973 meanMAE=432.753 std=1.099 | w=
(0.40,0.60,0.00) | lam=0.00 | shift_mult=0.0 | shift=-0.00
#02 obj=432.973 meanMAE=432.753 std=1.098 | w=
(0.40,0.60,0.00) | lam=0.00 | shift_mult=0.5 | shift=-0.30
#03 obj=432.974 meanMAE=432.754 std=1.097 | w=
(0.40,0.60,0.00) | lam=0.00 | shift_mult=1.0 | shift=-0.60
#04 obj=432.983 meanMAE=432.764 std=1.095 | w=
(0.45,0.55,0.00) | lam=0.00 | shift_mult=0.0 | shift=-0.00
#05 obj=432.984 meanMAE=432.765 std=1.093 | w=
(0.45,0.55,0.00) | lam=0.00 | shift_mult=0.5 | shift=-0.43
#06 obj=432.986 meanMAE=432.767 std=1.091 | w=
(0.45,0.55,0.00) | lam=0.00 | shift_mult=1.0 | shift=-0.86
#07 obj=432.986 meanMAE=432.765 std=1.107 | w=
(0.35,0.65,0.00) | lam=0.00 | shift_mult=0.0 | shift=-0.00
#08 obj=432.987 meanMAE=432.765 std=1.107 | w=
(0.35,0.65,0.00) | lam=0.00 | shift_mult=0.5 | shift=-0.05

#09 obj=432.988 meanMAE=432.765 std=1.106 | w=
(0.35,0.65,0.00) | lam=0.00 | shift_mult=1.0 | shift=-0.11
#10 obj=433.016 meanMAE=432.798 std=1.086 | w=
(0.50,0.50,0.00) | lam=0.00 | shift_mult=0.0 | shift=-0.00
[full-train] fitting Model A on full train for feature
importance...
[Top feature importances] (Model A full)
feature importance
chronic_encoded 15.270650
prior_ed_cost_5y_usd 9.223246
log_prior_cost_cap20k 7.763313
sqrt_prior_cost 6.761881
log_prior_cost 6.617104
prior_cost_cap20k 6.518574
cost_per_em 4.328865
prior_ed_visits_5y 2.233407
baseline_next3y 2.178595
cost_per_visit 2.148896
log_visits 2.044706
cost_per_code 1.846830
======================================
======================================
==============
[FINAL OOF]
ensemble OOF MAE (stable search + optional chronic
shift): 429.695
ensemble meta: {'models_order': ['A_RMSE_full_d5',
'B_RMSE_pruned_d4', 'C_MAE_pruned_d4'], 'weights': (0.4,
0.6, 0.0), 'lam_baseline': 0.0, 'shift_mult': 0.0, 'shift_value':
-0.0, 'oof_mean_mae_across_seeds': 432.75264656175824,
'oof_std_mae_across_seeds': 1.0993495506785935}
extra shift: chronic_group shrink=0.2
OOF pred quantiles: {0: 1048.0989640409848, 0.01:
1411.3553444441761, 0.05: 1772.840408581087, 0.1:
2012.0897620350563, 0.5: 3496.9648981987666, 0.9:
6510.167976279022, 0.95: 7333.8861121212585, 0.99:
8643.613637385735, 1.0: 10277.346526675536}

======================================
======================================
==============
[SUBMISSION sanity checks]
submission shape: (2000, 2)
submission columns exactly: ['patient_id',
'ed_cost_next3y_usd']
any NaNs in preds: False
pred min/median/max: 1089.82 3565.25 10390.04
pred quantiles: {0: 1089.82, 0.01: 1354.0901, 0.05:
1683.9235, 0.1: 1971.588, 0.25: 2585.7375, 0.5: 3565.25,
0.75: 4898.655, 0.9: 6511.809, 0.95: 7413.348499999999,
0.99: 8792.561999999998, 1.0: 10390.04}
Saved submission to:
D:\AgentDs\agent_ds_healthcare\submission.csv
Paste back: (1) leaderboard MAE, (2) FINAL OOF MAE, (3)
ensemble meta + extra shift, (4) pred quantiles.
好像没啥⽤
?
```

### Assistant-5（对 Code19 失效的解释 + Code20（drift-aware weighting）方向）

```text
对，你这个 的 在 上基本没起作⽤：你从 还倒
Code19 “Ch1-risk bridge” LB 449.x → 450.91
退了⼀点，虽然 很漂亮（ ），但 这件事在你这题上已经很明确了。
OOF 429.7 OOF≠LB
原因我觉得主要就三条（都和任务定义 分布有关）：
/
⽬标不完全对⻬
1.
是 天再⼊院 ⼆分类（ ）， 是 未来 年 总成本 回归
Ch1 “30 ” Macro-F1 Ch2 “ 3 ED ”
（ ）。短期 ⻛险和 年成本相关性不⼀定强，尤其你已经有很强的先验变
MAE readmit 3
量 。这决定了 信号很可能只是 弱
prior_ed_cost_5y_usd / prior_ed_visits_5y Ch1 “
增益或噪声 。
” description description
你这个 模型本⾝不够强
2. aux Ch1
⾥ 只有 ，远低于你说的 。这意味着我们喂给
log aux OOF macro-F1 0.7034 0.9014 Ch2
的 本⾝就偏噪声， 不动很正常。
ch1_readmit_p_* LB

你真正的瓶颈是 分布漂移 失真
3. “train/test + CV ”
你现在最典型的现象： 但 ，差 ，这⼏乎就是 没模拟 分
OOF 429 LB 450 20+ “CV test
布 。所以你继续加特征（哪怕是 ⻛险）很容易只让 变好、 不动甚⾄变
” Ch1 OOF LB
差。
下⼀步怎么搞：不要再 加新信息 ，先把 泛化 对⻬
“ ” “ ” test
你现在最该试的是： （漂移权重）。
adversarial validation → importance weighting
核⼼思想：训练⼀个 区分 的分类器，得到每个 样本有多像 ，然后在
“ train vs test” train test
回归训练时 给更像 的样本更⼤权重。这类⽅法在你这种 好但 卡 的场景，往
test “OOF LB ”
往⽐堆特征更有效。
下⾯给你 （ 个 ，全量可跑）：在 ⻛格不变的前提下，新增：
CODE 20 1 cell Code18
训练 分类器，输出 权重分布
train-vs-test AUC +
⽤ 正确传 （不会再 ）
CatBoost Pool sample_weight TypeError
搜索⽤ （更贴近 ）
ensemble weighted MAE test
你直接复制跑，输出还是 。
D:\AgentDs\agent_ds_healthcare\submission.csv
这版我默认 不再⽤ （因为你刚验证 没益处），先把漂移问题解决。
ch1_risk LB
python Copy code
# === CODE 20 / "Code18 + Drift-Aware Importance Weighting" ===
# Goal: fix "OOF good but LB stuck" by aligning training with test distribution v
# Keep spirit: LOW-DIM + shallow CatBoost + strong reg + multi-seed bagging + stab
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
random.seed(SEED)
np.random.seed(SEED)
# -----------------------------
# Paths
# -----------------------------
DATA_DIR = r"D:\AgentDs\agent_ds_healthcare"
TRAIN_PATH = r"D:\AgentDs\agent_ds_healthcare\ed_cost_train.csv"
TEST_PATH = r"D:\AgentDs\agent_ds_healthcare\ed_cost_test.csv"
PATIENTS_PATH = r"D:\AgentDs\agent_ds_healthcare\patients.csv"
ADM_TRAIN_PATH = r"D:\AgentDs\agent_ds_healthcare\admissions_train.csv"
ADM_TEST_PATH = r"D:\AgentDs\agent_ds_healthcare\admissions_test.csv"
RECEIPTS_JOBLIB_PATH = r"D:\AgentDs\agent_ds_healthcare\receipts_parsed.joblib"
OUT_SUB_PATH = r"D:\AgentDs\agent_ds_healthcare\submission.csv"
ID_COL = "patient_id"
TARGET = "ed_cost_next3y_usd"
print("="*110)
print("CODE 20 | Drift-aware importance weighting: adversarial train-vs-test weig
print("Goal: reduce CV/LB mismatch (OOF good but LB stuck) by training more on 't
print("="*110)
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

from sklearn.metrics import mean_absolute_error, roc_auc_score
except Exception:
_pip_install("scikit-learn")
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, roc_auc_score
try:
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
except Exception:
_pip_install("catboost")
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
# -----------------------------
# Config
# -----------------------------
class CFG:
# main CV
N_FOLDS = 7
N_SEEDS = 5
ITERS = 3200
ES_ROUNDS = 130
LR = 0.03
RSM = 0.80
SUBSAMPLE = 0.80
TRIM_K = 1 # 5 seeds -> drop min/max, avg middle 3
# ensemble search
W_STEP = 0.05
LAM_GRID = [0.00, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30] # allow mo
SHIFT_GRID = [0.0, 0.5, 1.0]
STD_PEN = 0.20
LAM_PEN = 1.5 # reduce penalty vs Code18 (drift cases often need a b
SHIFT_PEN = 0.001
USE_FULLFIT_BLEND = True
FULLFIT_BLEND_W = 0.35
# adversarial weighting
ADV_FOLDS = 5

W_CLIP = (0.25, 4.0) # clip importance weights
W_POWER = 1.0 # 1.0 = plain; try 0.8 if too aggressive
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
def mae_w(y_true: np.ndarray, y_pred: np.ndarray, w: Optional[np.ndarray] = None)
y_true = np.asarray(y_true, dtype=float)
y_pred = np.asarray(y_pred, dtype=float)
if w is None:
return float(np.mean(np.abs(y_true - y_pred)))
w = np.asarray(w, dtype=float)
w = np.clip(w, 1e-12, np.inf)
return float(np.average(np.abs(y_true - y_pred), weights=w))
def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
v = np.asarray(values, dtype=float)
w = np.asarray(weights, dtype=float)
m = ~np.isnan(v)

v, w = v[m], w[m]
if v.size == 0:
return 0.0
order = np.argsort(v)
v, w = v[order], w[order]
cw = np.cumsum(w)
cutoff = 0.5 * cw[-1]
return float(v[np.searchsorted(cw, cutoff)])
# -----------------------------
# Admissions features (simple, robust)
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
# Receipts features from parsed lineitems (low-dim)
# -----------------------------
def build_receipt_features_from_lineitems(li: pd.DataFrame) -> pd.DataFrame:
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
receipt_total = li.groupby("patient_id")["amt"].sum().rename("receipt_total")
li = li.join(receipt_total, on="patient_id")
denom = li["receipt_total"].replace(0.0, np.nan)
share = (li["amt"] / denom).fillna(0.0)
cost_hhi = (share * share).groupby(li["patient_id"]).sum().rename("cost_hhi")
top1_share = share.groupby(li["patient_id"]).max().rename("top1_share")
def _topk_sum(s, k=3):
a = np.sort(s.values.astype(float))[::-1]
return float(a[:k].sum()) if a.size else 0.0
top3_share = share.groupby(li["patient_id"]).apply(lambda s: _topk_sum(s, 3))
gini_amt = li.groupby("patient_id")["amt"].apply(lambda s: gini(s.values)).re
max_line_total = li.groupby("patient_id")["amt"].max().rename("max_line_total
code_num = pd.to_numeric(li["code"].where(li["code"].str.fullmatch(r"\d+"), N
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
n_em_codes = is_em.astype(int).groupby(li["patient_id"]).sum().rename("n_em_c
max_em_level = em_level.groupby(li["patient_id"]).max().rename("max_em_level"
sum_em_level = (em_level * is_em.astype(int)).groupby(li["patient_id"]).sum()
avg_em_level = (sum_em_level / n_em_codes.replace(0, np.nan)).fillna(0.0).ren
n_high_em = ((em_level >= 4) & is_em).astype(int).groupby(li["patient_id"]).s
em_total = li.loc[is_em].groupby("patient_id")["amt"].sum().rename("em_total"
crit_total = li.loc[is_crit].groupby("patient_id")["amt"].sum().rename("crit_t
proc_total = li.loc[is_proc_any].groupby("patient_id")["amt"].sum().rename("p
img_total = li.loc[is_imaging].groupby("patient_id")["amt"].sum().rename("img_
lab_total = li.loc[is_lab].groupby("patient_id")["amt"].sum().rename("lab_tot
high_total = li.loc[is_high].groupby("patient_id")["amt"].sum().rename("high_t
n_procedures = is_proc_any.astype(int).groupby(li["patient_id"]).sum().rename
n_imaging = is_imaging.astype(int).groupby(li["patient_id"]).sum().rename("n_
n_lab = is_lab.astype(int).groupby(li["patient_id"]).sum().rename("n_lab")
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
# median unit price (robust)
med_unit_all = li.loc[li["unit_price"].notna()].groupby("patient_id")["unit_p
out = pd.concat([
n_unique_codes, cost_hhi, top1_share, top3_share, gini_amt, max_line_tota

n_em_codes, max_em_level, avg_em_level, n_high_em,
has_critical_care, has_high_acuity, has_observation, has_imaging,
has_intub_31500, has_cvc_36556, has_cpr_92950, has_artline_36620,
has_ct_head_70450, has_99285, has_ct_abdpel_74177, has_troponin_84484, ha
n_procedures, n_imaging, n_lab
], axis=1).reset_index()
for s in [em_total, crit_total, proc_total, img_total, lab_total, high_total]
out = out.merge(s.reset_index(), on="patient_id", how="left")
for c in ["em_total","crit_total","proc_total","img_total","lab_total","high_
out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
# need receipt_total only to compute shares
receipt_total2 = li.groupby("patient_id")["amt"].sum().rename("receipt_total"
out = out.merge(receipt_total2.reset_index(), on="patient_id", how="left")
out["receipt_total"] = pd.to_numeric(out["receipt_total"], errors="coerce").f
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
# unit price
out = out.merge(med_unit_all.reset_index(), on="patient_id", how="left")
out["median_unit_price"] = pd.to_numeric(out["median_unit_price"], errors="co
out["log1p_median_unit_price"] = np.log1p(out["median_unit_price"].clip(lower
out["max_line_total"] = pd.to_numeric(out["max_line_total"], errors="coerce")
out["log1p_max_line_total"] = np.log1p(out["max_line_total"].clip(lower=0.0))

# drop helper totals
out.drop(columns=[c for c in ["em_total","crit_total","proc_total","img_total
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
# Feature engineering (numeric-only)
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
p["age"] = p["age"].fillna(p["age"].median() if not p["age"].isna().all() els
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
if rcpt_df is not None:
feat = feat.merge(rcpt_df, on="patient_id", how="left")
for c in rcpt_df.columns:
if c == "patient_id":
continue
feat[c] = pd.to_numeric(feat[c], errors="coerce").replace([np.inf,-np
if feat[c].isna().any():
feat[c] = feat[c].fillna(feat[c].median() if not feat[c].isna().a
if "pct_cost_critical" in feat.columns:
feat["logprior_x_pctcritical"] = feat["log_prior_cost"] * feat["pct_cost_
if "n_high_acuity_total" in feat.columns:
feat["logprior_x_highacu"] = feat["log_prior_cost"] * feat["n_high_acuity_
if "n_unique_codes" in feat.columns:
feat["cost_per_code"] = feat["prior_ed_cost_5y_usd"] / feat["n_unique_cod
feat.replace([np.inf,-np.inf], np.nan, inplace=True)
for c in feat.columns:
if c in ["patient_id","primary_chronic",TARGET]:
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
# Adversarial validation -> importance weights
# -----------------------------
def build_importance_weights(train_feat: pd.DataFrame,
test_feat: pd.DataFrame,
feat_cols: List[str],
n_splits: int = 5,
seed: int = 42) -> np.ndarray:
X_tr = train_feat[feat_cols].copy()
X_te = test_feat[feat_cols].copy()
# fill with train medians for stability
for c in feat_cols:
med = pd.to_numeric(X_tr[c], errors="coerce").median()
X_tr[c] = pd.to_numeric(X_tr[c], errors="coerce").replace([np.inf,-np.inf
X_te[c] = pd.to_numeric(X_te[c], errors="coerce").replace([np.inf,-np.inf
X = pd.concat([X_tr, X_te], ignore_index=True)
y = np.array([0]*len(X_tr) + [1]*len(X_te), dtype=int)

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
oof = np.zeros(len(X), dtype=float)
params = dict(
loss_function="Logloss",
eval_metric="AUC",
depth=4,
learning_rate=0.05,
iterations=2500,
l2_leaf_reg=10,
min_data_in_leaf=80,
rsm=0.85,
bootstrap_type="Bernoulli",
subsample=0.85,
random_strength=1.0,
task_type="CPU",
thread_count=-1,
verbose=0,
allow_writing_files=False,
)
for fold, (ti, vi) in enumerate(skf.split(X, y), 1):
cb = CatBoostClassifier(**params, random_seed=int(seed + 100 + fold*19),
cb.fit(X.iloc[ti], y[ti], eval_set=(X.iloc[vi], y[vi]), verbose=0)
oof[vi] = cb.predict_proba(X.iloc[vi])[:, 1]
del cb
gc.collect()
auc = float(roc_auc_score(y, oof))
print(f"\n[drift] adversarial OOF AUC(train vs test) = {auc:.4f} (0.5=no dri
# importance weights for TRAIN rows: w = p(test|x)/p(train|x)
p = np.clip(oof[:len(X_tr)], 1e-3, 1-1e-3)
w = (p / (1.0 - p)) ** float(CFG.W_POWER)
w = np.clip(w, CFG.W_CLIP[0], CFG.W_CLIP[1])
w = w / np.mean(w)
print("[drift] importance weight stats (train):", {
"min": float(np.min(w)), "p10": float(np.quantile(w,0.10)), "p50": float(

"p90": float(np.quantile(w,0.90)), "max": float(np.max(w))
})
return w
# -----------------------------
# Training (multi-seed + fold bagging) with weights
# -----------------------------
def train_models(train_feat: pd.DataFrame, test_feat: pd.DataFrame,
feat_full: List[str], feat_pruned: List[str],
w_imp: Optional[np.ndarray]) -> Tuple[Dict[str, List[np.ndarray]
y = train_feat[TARGET].values.astype(float)
tmp = train_feat[["primary_chronic", TARGET]].copy()
tmp["cost_bin"] = pd.qcut(tmp[TARGET], q=5, labels=False, duplicates="drop")
tmp["strat"] = tmp["primary_chronic"].astype(str) + "_" + tmp["cost_bin"].asty
strat = LabelEncoder().fit_transform(tmp["strat"].values)
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
depth=4, l2_leaf_reg=6, min_data_in_leaf=55,
learning_rate=CFG.LR, iterations=CFG.ITERS,
rsm=CFG.RSM, bootstrap_type="Bernoulli", subsample=CFG.SUBSAMPLE,
random_strength=2.0,
),
"C_MAE_pruned_d4": dict(
loss_function="MAE", eval_metric="MAE",
depth=4, l2_leaf_reg=12, min_data_in_leaf=60,
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
sw_tr = None if w_imp is None else w_imp[ti]
sw_va = None if w_imp is None else w_imp[vi]
train_pool = Pool(X_tr, y_tr, weight=sw_tr) if sw_tr is not None
valid_pool = Pool(X_va, y_va, weight=sw_va) if sw_va is not None
cb = CatBoostRegressor(
**params,

task_type="CPU",
thread_count=-1,
verbose=0,
allow_writing_files=False,
random_seed=int(rs + fold * 31 + stable_hash(mname) % 997),
early_stopping_rounds=CFG.ES_ROUNDS,
)
cb.fit(train_pool, eval_set=valid_pool, verbose=0)
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
# optional: full-fit per seed (still weighted)
test_seed_final = {}
if CFG.USE_FULLFIT_BLEND:
for mname, params in model_specs.items():
cols = model_featcols[mname]
X_all = train_feat[cols]
X_te = test_feat[cols]
sw_all = None if w_imp is None else w_imp
if best_iters_seed[mname]:
it_med = int(np.median(best_iters_seed[mname]))
else:
it_med = int(CFG.ITERS * 0.45)
it_use = int(max(300, min(CFG.ITERS, it_med + 150)))

params_full = dict(params)
params_full["iterations"] = it_use
train_pool = Pool(X_all, y, weight=sw_all) if sw_all is not None
cb_full = CatBoostRegressor(
**params_full,
task_type="CPU",
thread_count=-1,
verbose=0,
allow_writing_files=False,
random_seed=int(rs + 999 + stable_hash("FULL_"+mname) % 997),
)
cb_full.fit(train_pool, verbose=0)
pred_te_full = cb_full.predict(X_te)
del cb_full
gc.collect()
test_seed_final[mname] = (1.0 - CFG.FULLFIT_BLEND_W) * test_seed_
else:
test_seed_final = test_seed_foldbag
# report (unweighted + weighted)
seed_maes = {m: float(mean_absolute_error(y, oof_seed[m])) for m in model_
seed_wmaes = {m: mae_w(y, oof_seed[m], w_imp) if w_imp is not None else s
print(f" Seed {seed_idx+1}/{CFG.N_SEEDS} OOF MAE: " + " | ".join([f"{m}=
print(f" OOF wMAE: " + " | ".join([f"{m}={seed_wmaes[m]:.2
for m in model_specs.keys():
oof_by_seed[m].append(oof_seed[m])
test_by_seed[m].append(test_seed_final[m])
return oof_by_seed, test_by_seed, best_iters
# -----------------------------
# Ensemble selection (weighted stability)
# -----------------------------
def stable_ensemble_search(train_feat: pd.DataFrame,
oof_by_seed: Dict[str, List[np.ndarray]],
baseline_vec: np.ndarray,

w_imp: Optional[np.ndarray]) -> Tuple[np.ndarray, Dict
y = train_feat[TARGET].values.astype(float)
model_names = list(oof_by_seed.keys())
assert len(model_names) == 3
oof_agg = {m: trimmed_mean(np.vstack(oof_by_seed[m]), trim_k=CFG.TRIM_K) for
step = CFG.W_STEP
grid = np.round(np.arange(0.0, 1.0 + 1e-9, step), 10)
best = None
top_list = []
def eval_combo(wA, wB, wC, lam, shift_mult):
maes = []
pred_avg = wA*oof_agg[model_names[0]] + wB*oof_agg[model_names[1]] + wC*o
pred_avg = (1.0-lam)*pred_avg + lam*baseline_vec
# shift based on (weighted) median residual
resid = y - pred_avg
if w_imp is None:
shift = float(np.median(resid)) * shift_mult
else:
shift = weighted_median(resid, w_imp) * shift_mult
for s in range(CFG.N_SEEDS):
pred = wA*oof_by_seed[model_names[0]][s] + wB*oof_by_seed[model_names
pred = (1.0-lam)*pred + lam*baseline_vec
pred = pred + shift
maes.append(mae_w(y, pred, w_imp))
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
print("\n[ensemble search] Top candidates (objective uses wMAE across seeds +
for i, rec in enumerate(top_list[:10], 1):
obj, mean_m, std_m, wA, wB, wC, lam, sm, shift = rec
print(f" #{i:02d} obj={obj:.3f} mean_wMAE={mean_m:.3f} std={std_m:.3f} |
obj, mean_m, std_m, wA, wB, wC, lam, sm, shift = best
mA, mB, mC = model_names
oof_final = wA*oof_agg[mA] + wB*oof_agg[mB] + wC*oof_agg[mC]
oof_final = (1.0-lam)*oof_final + lam*baseline_vec
oof_final = oof_final + shift
meta = {
"models_order": model_names,
"weights": (float(wA), float(wB), float(wC)),
"lam_baseline": float(lam),
"shift_mult": float(sm),
"shift_value": float(shift),
"oof_mean_wmae_across_seeds": float(mean_m),
"oof_std_wmae_across_seeds": float(std_m),
}
return oof_final, meta
# -----------------------------
# Optional chronic-group shift (weighted)
# -----------------------------
def apply_chronic_group_shift(train_feat: pd.DataFrame, pred_oof: np.ndarray,
w_imp: Optional[np.ndarray], shrink: float = 0.25)

y = train_feat[TARGET].values.astype(float)
chronic = train_feat["primary_chronic"].astype(str).values
resid = y - pred_oof
shifts = {}
for g in np.unique(chronic):
idx = (chronic == g)
if w_imp is None:
med = float(np.median(resid[idx]))
else:
med = weighted_median(resid[idx], w_imp[idx])
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
print("\n[load] reading CSVs...")
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
patients = pd.read_csv(PATIENTS_PATH)
log_shape("ed_cost_train", train)
log_shape("ed_cost_test", test)
log_shape("patients", patients)
print("\n[target stats]")
print(train[TARGET].describe().to_string())
for df in [train, test, patients]:
df[ID_COL] = pd.to_numeric(df[ID_COL], errors="coerce").astype(int)

print("\n[admissions] building robust aggregates...")
adm_df = load_admissions_features(ADM_TRAIN_PATH, ADM_TEST_PATH)
if adm_df is None:
print(" admissions features: None")
else:
print(f" admissions features: {adm_df.shape} | cols={list(adm_df.columns)}")
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
print(f" receipt_feat shape: {rcpt_df.shape}")
else:
print(" [warn] could not build receipt features from joblib structur
except Exception as e:
print(f" [warn] receipts joblib load/build failed: {e}")
rcpt_df = None
else:
print(" [warn] receipts joblib missing; skipping receipts features.")
print("\n[features] building train/test feature frames...")
train_feat = build_features(train, patients, adm_df, rcpt_df)
test_feat = build_features(test, patients, adm_df, rcpt_df)
feat_full = get_numeric_feature_cols(train_feat)
feat_full = [c for c in feat_full if c != TARGET]
feat_full = drop_constants(feat_full, train_feat)
pruned_candidates = [
"prior_ed_visits_5y","prior_ed_cost_5y_usd","prior_cost_cap20k","sqrt_prior_c
"cost_per_visit","log_visits","baseline_next3y",
"chronic_encoded","age","sex_encoded","insurance_encoded","zip_region","ins_x_
"charlson_max","charlson_mean","pct_emergent",
# receipts

"cost_per_em","cost_hhi","pct_cost_em","pct_cost_procedure","pct_cost_critica
"n_high_acuity_total","has_critical_care","has_99285","max_em_level","avg_em_
"top1_share","top3_share","gini_line_total","max_line_total","median_unit_pri
"logprior_x_pctcritical","logprior_x_highacu","cost_per_code",
]
feat_pruned = [c for c in pruned_candidates if c in train_feat.columns]
feat_pruned = drop_constants(feat_pruned, train_feat)
print(f" FULL feature count: {len(feat_full)}")
print(f" PRUNED feature count: {len(feat_pruned)}")
print(f" PRUNED features: {feat_pruned}")
# fill safety
for c in set(feat_full + feat_pruned):
med = train_feat[c].median() if c in train_feat.columns and not train_feat[c]
train_feat[c] = pd.to_numeric(train_feat[c], errors="coerce").replace([np.inf
test_feat[c] = pd.to_numeric(test_feat[c], errors="coerce").replace([np.inf,
# build drift weights using PRUNED (more stable)
w_imp = build_importance_weights(train_feat, test_feat, feat_pruned, n_splits=CFG
# train weighted models
oof_by_seed, test_by_seed, best_iters = train_models(train_feat, test_feat, feat_
baseline_oof = train_feat["baseline_next3y"].values.astype(float)
baseline_test = test_feat["baseline_next3y"].values.astype(float)
# weighted ensemble search
oof_ens, meta = stable_ensemble_search(train_feat, oof_by_seed, baseline_oof, w_i
# test ensemble
mA, mB, mC = meta["models_order"]
wA, wB, wC = meta["weights"]
lam = meta["lam_baseline"]
shift = meta["shift_value"]
test_agg = {m: trimmed_mean(np.vstack(test_by_seed[m]), trim_k=CFG.TRIM_K) for m
test_ens = wA*test_agg[mA] + wB*test_agg[mB] + wC*test_agg[mC]
test_ens = (1.0-lam)*test_ens + lam*baseline_test
test_ens = test_ens + shift

# optional chronic shift (weighted) — only if it helps unweighted MAE by a tiny m
y = train_feat[TARGET].values.astype(float)
base_mae = float(mean_absolute_error(y, oof_ens))
best_oof = oof_ens
best_shift = {"type": "none"}
for shrink in [0.0, 0.15, 0.25]:
if shrink <= 0:
continue
oof2, shifts = apply_chronic_group_shift(train_feat, oof_ens, w_imp, shrink=s
m2 = float(mean_absolute_error(y, oof2))
if m2 < base_mae - 0.05:
base_mae = m2
best_oof = oof2
best_shift = {"type": "chronic_group", "shrink": shrink, "shifts": shifts
if best_shift["type"] == "chronic_group":
test_chronic = test_feat["primary_chronic"].astype(str).values
for g, s in best_shift["shifts"].items():
test_ens[test_chronic == g] += s
# clip (LB-safe)
y_max = float(np.max(y))
test_ens = np.clip(test_ens, 0.0, y_max * 1.5)
final_oof_mae = float(mean_absolute_error(y, best_oof))
final_oof_wmae = mae_w(y, best_oof, w_imp)
print("\n" + "="*90)
print("[FINAL OOF]")
print(f" unweighted MAE: {final_oof_mae:.3f}")
print(f" weighted wMAE: {final_oof_wmae:.3f}")
print(" ensemble meta:", meta)
print(" extra shift:", best_shift["type"], ("shrink="+str(best_shift.get("shrink
print(" OOF pred quantiles:", qdict(best_oof, qs=(0,0.01,0.05,0.1,0.5,0.9,0.95,0
print("="*90)
# write submission
sub = pd.DataFrame({

"patient_id": test["patient_id"].values.astype(int),
"ed_cost_next3y_usd": np.round(test_ens.astype(float), 2)
})[["patient_id","ed_cost_next3y_usd"]]
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
print("\nPaste back: (1) leaderboard MAE, (2) drift AUC + weight stats, (3) FINAL
你跑完 后，我只看 个输出就能判断 有没有希望破
Code20 3 “ 449”
（漂移强不强）
1. adversarial OOF AUC
（权重是否极端）
2. importance weight stats
新的 （最关键）
3. LB MAE
如果 很⾼（⽐如 ），这条路通常就对；如果 接近 ，那说明漂移不⼤，你
AUC >0.70 AUC 0.5
卡 的原因就得换⽅向（⽐如更强的 尾部 分位数 建模或更合理的 ）。
450 “ / ” target transform
Sources
```
