# AgentDS Healthcare Challenge 2 — ED Cost Forecasting (MAE)
## Final iteration log (Iteration 01–06) + prompts + results (open‑source ready)

> Generated: 2026-02-24
>
> **What this file is**: A single, paper‑citeable record that combines (1) the *exact prompt pack* used to steer LLM coding, and (2) the *iteration‑by‑iteration* experimental deltas and measured outcomes (OOF + Leaderboard).
>
> **Primary artifacts referenced**:
> - `IEEE ICHI - Code31 迭代与补丁.pdf` (prompt/rules + narrative + code excerpts)
> - `Challenge2_baseline_ichi_v9.ipynb` (final runnable code cells + printed run summaries)

---
## A) Problem statement (public)
We forecast each patient’s **3‑year future Emergency Department cost** (`ed_cost_next3y_usd`) and are evaluated by **Mean Absolute Error (MAE)** (lower is better). The dataset is small (**train=2000, test=2000**) and multi‑source:

- `ed_cost_train.csv / ed_cost_test.csv` — core table (prior ED visits/cost in last 5y, chronic category, target).
- `patients.csv` — demographics + insurance + zip3.
- `admissions_train.csv / admissions_test.csv` — admissions aggregates (Charlson etc).
- `receipts_parsed.joblib` — parsed receipt line‑items aggregated into robust low‑dim features.

Key practical constraint (drives all design choices): **runtime must stay minute‑level (<5 min per full run)**; no heavy deep models / no high‑dim text features.

---
## B) Prompt pack (the *actual* steering instructions we used)
This section is what you can include in an open‑source repo to show *exactly* how the LLM was instructed.

### B.1 Handoff prompt (from PDF; lightly reformatted for readability)
```text
任务与约束（给接⼿ 的开场⼀句话）
## 0) AI
任务：预测 （未来 年 成本），
* ed_cost_next3y_usd 3 ED
指标 ，训练集 ，测试集 。
**MAE** **2000** **2000**
数据源（你们⼿⾥）：
*
（核⼼表）
* ed_cost_train.csv / ed_cost_test.csv
（⼈⼝学、保险、 ）
* patients.csv zip3
（住院
* admissions_train.csv / admissions_test.csv /
⼊院汇总信息）
（收费 项⽬聚合，可复现
* receipts_parsed.joblib /
）
prior cost
以及更 重 的： 、
* “ ” discharge_notes.json
等
vitals_timeseries.json
现实约束： ⾏ 的 回归， 。
* **2000 ** tabular “less is more”
训练不能拖到 ⼩时（你明确说不⾏）；最好 分钟
1~2 3~5
以内（你们最强 就是这个级别）。
code
---
⽬前最佳： 来⾃ （成功要点必
## 1) MAE 448.0793 Code31
须继承）
的核⼼结构（必须保留的⻣架）
### 1.1 Code31
的极简版 个真正有效的⼩技
**Code31 = Code25-core + 1
巧（ ） ，整体流程是：
chronic shift **
轻量特征 （全是低维、强先验、稳健）
1. ** **
历史：
* ED
* prior_ed_cost_5y_usd , prior_ed_visits_5y
变换： 等
* log1p , sqrt , cost_per_visit
强基线：
* baseline_next3y = prior_ed_cost_5y_usd *
(3/5)
患者⼈⼝学： （低维编码）
* age , sex , insurance +
（只取 的⾸位做 ，避免 ）
zip_region zip3 region one-hot
聚合（⾮常关键但仍然低维）：
* Admissions
* charlson_max , charlson_mean , pct_emergent
聚合（重要点： 它和 ⾼度⼀致 ）：
* Receipts ** prior cost **
有 等，但你们 明确做了去重 避免重
* rcpt_sum_items ** /
复信息 （如移除 ）
** rcpt_sum_items
两个 模型（ ）
2. CatBoost A/B
： ，⽤ 特征
* A depth=5 FULL
： ，⽤ 特征
* B depth=4 PRUNED
， 、 ，早停
* CPU loss_function=RMSE eval_metric=MAE
（ ），学习率约
es=120 0.03
折 （ ），总体 分钟级别
* 7 CV + 3 seeds FAST_MODE ~ 1~2
权重搜索 （很稳的 ）
3. A/B + weight-bagging ensemble
在 ⽹格（步⻓ ）上做搜索，⽤
* wA 0.05 mean + 0.2*std
的⽬标选权重
⽤ 思路选 更简单的权重
* one-SE “ ”
在附近⼏个权重上 （例如 ）
* bag [0.25, 0.30, 0.35]
关键成功点： （按慢病组做残差校正）
4. ** Chronic shift **
按 （ ）
* primary_chronic HF / Pneumonia / DiabetesComp
分组
计算每组 的 （ 最
* residual = y - pred **median** MAE
优统计量）
做 （样本少的组收缩）、做 （限制幅度）
* shrink cap
调参 ，并且只有当 ⾜够才
* CV chronic_factor gain apply
⾥它 贡献了真正可⻅的 提升 ，并且对应
* Code31 ** OOF **
也最优。
LB
的关键结果（要让接⼿ 牢记）
### 1.2 Code31 AI
： （当前最优）
* **Leaderboard MAE 448.0793 **
训练⽇志⾥（ ）：
* Code31
基础 ：
* AB bag OOF **427.928**
后 ： （ ）
* chronic shift OOF **427.480** delta -0.448
最终没启⽤（ ），说明 再往
* baseline shrink w=1.0 “
拉回 在 上没收益
baseline ” CV
训练总耗时
* ~ **97s**
接⼿ 的第⼀原则： 先别动⻣架 。你们最强的提升
> AI ** **
来⾃ 低维强先验 稳健化 ⼩校
“ + ensemble + chronic shift
正 。
```

### B.2 Non‑negotiable rules (the manager rubric)
These were repeatedly enforced to avoid "false improvements" and keep experiments comparable:

- **Freeze the Code31 skeleton**: same joins, same receipts feature logic, same folds/seeds defaults, same evaluation harness.
- **Single‑point change only**: every iteration must change *one* thing; everything else stays fixed.
- **Credible OOF gain threshold**: improvements smaller than ~0.2–0.3 MAE are treated as noise at this stage.
- **Hard runtime gate**: full pipeline (A/B/C + ensemble + shift) must stay **< 5 minutes**.
- **Always print audit outputs**: OOF base/final, chosen weights, shift meta, prediction quantiles, and a `pred_hash`/submission md5 to prove submissions changed.

### B.3 Iteration prompt template (the reusable patch request)
```text
You must implement a strict SINGLE-POINT change from the previous best code.
Do not change features unless explicitly stated.
Keep runtime < 5 minutes.
Output MUST include:
  - base OOF MAE, after-shift OOF MAE, final OOF MAE
  - chosen weights (+ bag list)
  - shift_meta (+ whether applied)
  - baseline_shrink_meta / bin_shift_meta (if relevant)
  - OOF and TEST prediction quantiles
  - pred_hash (crc32 of rounded test preds) and/or submission.csv md5
Return FULL runnable code (single-cell) so we can paste-run immediately.
```

---
## C) Fixed pipeline skeleton (what stays constant across iterations)
We keep the modeling pipeline intentionally conservative and audit-friendly:

- **Low‑dim strong priors**: prior ED cost/visits + mild transforms; simple demographic encodings; admissions aggregates; low‑dim receipts aggregates.
- **Two feature views**: FULL vs PRUNED (PRUNED reduces variance).
- **CatBoost base learners** on CPU with early stopping, 7‑fold CV, multi-seed bagging.
- **Weight search** uses mean(MAE)+STD_PEN*std(MAE), then **local weight-bagging** around the chosen point.
- **Chronic residual shift**: small, shrink/capped per‑chronic residual correction (only applied if CV gain passes a threshold).
- **Baseline shrink / bin calibration** are *optional* and always gated (allowed to become a no‑op).

---
## D) Iteration record (Iteration 01–06) — prompts + deltas + results
All six iterations below are the **final GPT track** referenced by the notebook’s `# Iteration 01–06` headings, with the corresponding leaderboard MAE written in those markdown cells.

### Iteration 01 — CODE 31 baseline (standalone)

**Prompt excerpt (what we actually told the LLM right before generating this code):**
```text
seed residual
（ ⼤幅提升但 明显变差，典型 幻觉）；
OOF LB CV MAE
、 、 等都使
loss Bayesian/MVS bootstrap residual target LB
变差。
>
经验教训：样本太⼩， 改善不⼀定带来 改善，必
> OOF LB
须保证真正的单点改动和可⽐性（避免 特征维度变
receipts
化导致多变量实验），并优先做 稳定、强先验、低维 的改
“ ”
动。下⼀步建议优先尝试：预测 护栏、
clipping monotonic
、 更稳健化（但不要⾛到分箱校准
constraints chronic shift
过拟合）。
>
约束：训练必须保持分钟级（ ），不能 。请在
> <5min 1hr+
⻣架上做严格单点迭代，并输出 、
Code31 feature diff OOF
稳定性、预测分位数。
best code:
#
======================================
======
```

**Single-point change (delta):**
- Initialize the frozen baseline: CatBoost A/B (FULL/PRUNED) + one-SE weight-bagging + chronic shift + optional baseline shrink (gated).

**Code pointer (for open-source reproducibility):**
- Notebook: `Challenge2_baseline_ichi_v9.ipynb`  |  code cell index: `6`  |  markdown label cell: `5`

**Measured results (from the code cell’s printed final summary):**
- OOF base MAE: 427.928
- OOF after chronic shift: 427.480
- OOF final MAE: 427.480
- TEST pred_hash: (n/a)
- Leaderboard MAE (as recorded in notebook): **448.0793**

**Run summary block (verbatim, for audit):**
```text
[FINAL SUMMARY]
  base OOF MAE (AB bag):     427.928
  after chronic shift MAE:   427.480  (delta=-0.448)
  final OOF MAE:             427.480  (delta=-0.448)
  weight meta: {'best': {'wA': 0.4, 'wB': 0.6, 'obj': 429.95209383956944, 'mean': 429.8237377239825, 'std': 0.6417805779347993}, 'chosen': {'wA': 0.25, 'wB': 0.75, 'obj': 430.0175242980049, 'mean': 429.88362267530505, 'std': 0.6695081134992206}, 'bag_list_wA': [0.25, 0.3, 0.35]}
  chronic shift meta: {'base_mae': 427.92823920810883, 'obj': 427.7605539008683, 'cv_mae': 427.7405539008683, 'cf': 0.75, 'cv_gain_vs_base': 0.18768530724054244} | applied: True
  baseline shrink meta: {'applied': False, 'gain': 0.0}
  OOF quantiles: {0: 1051.4363103641224, 0.01: 1384.4678380057126, 0.05: 1733.884346554134, 0.1: 2007.8912172017415, 0.5: 3507.61531389114, 0.9: 6540.899781908962, 0.95: 7354.264882531222, 0.99: 8688.295267226058, 1.0: 10200.874143037847}
  TEST quantiles: {0: 1087.3213552195787, 0.01: 1335.6438665205355, 0.05: 1677.4788605558024, 0.1: 1968.1392845230012, 0.5: 3556.039936787505, 0.9: 6486.3635112501315, 0.95: 7442.877960013898, 0.99: 8813.76833940386, 1.0: 10400.723100303465}
====================================================================================================

[SUBMISSION sanity checks]
  path: D:\AgentDs\agent_ds_healthcare\submission.csv
  shape: (2000, 2)
  cols: ['patient_id', 'ed_cost_next3y_usd']
  any NaNs: False
  min/median/max: 1087.32 3556.04 10400.72
  quantiles: {0: 1087.32, 0.01: 1335.6407000000002, 0.05: 1677.4755, 0.1: 1968.142, 0.25: 2588.9975, 0.5: 3556.04, 0.75: 4897.505, 0.9: 6486.365000000002, 0.95: 7442.8735, 0.99: 8813.771599999998, 1.0: 10400.72}

[done] total wall time: 79.4s

Paste back: leaderboard MAE + (base_mae, final_mae) + chosen weight bag_list + shift_meta + baseline_shrink_meta.
```

**Manager decision / verdict (what we did next):**
- Keep as baseline reference (first stable 448.0793 submission).

### Iteration 02 — CODE 31+ audit-hardening (same predictions)

**Prompt excerpt (what we actually told the LLM right before generating this code):**
```text
baseline shrink
且 等价于 不做 。所以你以为 加了⼀个新技
best_w_model=1.0 gain=0.0 → “ shrink” “
巧 ，但实际上这次 的最终预测 ，跟你之前最
” run = AB weight-bag + chronic shift
优⽅案⼀样，就会出现 ⼏乎没变 。
“ ”
你这次⽇志⾥所有关键签名都与 完全⼀致：
Code31
、 参数、 选择（ 选了 ，
folds/seeds A/B weight one-SE wA=0.25 bag=
）、 、 数值都⼀致。
[0.25,0.30,0.35] chronic_factor=0.75 shifts
这会导致预测⽂件很可能 都⼀样 分数当然也⼀模⼀样。
byte-level → LB
你要确认 到底 有没有变 ，最稳的⼀步是：对 打⼀个
“ submission ” submission.csv md5
（下⾯我在 ⾥直接加了⾃动 和记忆对⽐）。
full code md5
你要的： 可运⾏ （含
3) 1-cell Full Standalone Code feature hash +
）
submission md5 guardrail
直接整段复制进⼀个 运⾏即可。
cell
默认路径仍是 ，你要改⽬录只改 ⼀⾏。
D:\AgentDs\agent_ds_healthcare DATA_DIR
# ==P=la=i=n= t=e=x=t===================================
```

**Single-point change (delta):**
- Add guardrails only (feature fingerprints + receipts invariants + submission.csv md5), without altering model/preds.

**Code pointer (for open-source reproducibility):**
- Notebook: `Challenge2_baseline_ichi_v9.ipynb`  |  code cell index: `12`  |  markdown label cell: `11`

**Measured results (from the code cell’s printed final summary):**
- OOF base MAE: 427.928
- OOF after chronic shift: 427.480
- OOF final MAE: 427.480
- TEST pred_hash: (n/a)
- Leaderboard MAE (as recorded in notebook): **448.0793**

**Run summary block (verbatim, for audit):**
```text
[FINAL SUMMARY]
  base OOF MAE (AB bag):     427.928
  after chronic shift MAE:   427.480  (delta=-0.448)
  final OOF MAE:             427.480  (delta=-0.448)
  chosen weight bag_list_wA: [0.25, 0.3, 0.35]
  chronic shift meta: {'base_mae': 427.92823920810883, 'obj': 427.7605539008683, 'cv_mae': 427.7405539008683, 'cf': 0.75, 'cv_gain_vs_base': 0.18768530724054244} | applied: True | shifts: {'DiabetesComp': -1.254, 'HF': 13.62, 'Pneumonia': -20.282}
  baseline shrink meta: {'applied': False, 'gain': 0.0, 'chosen_w_model': 1.0}
  OOF quantiles: {0: 1051.4363103641224, 0.01: 1384.4678380057126, 0.05: 1733.884346554134, 0.1: 2007.8912172017415, 0.5: 3507.61531389114, 0.9: 6540.899781908962, 0.95: 7354.264882531222, 0.99: 8688.295267226058, 1.0: 10200.874143037847}
  TEST quantiles: {0: 1087.3213552195787, 0.01: 1335.6438665205355, 0.05: 1677.4788605558024, 0.1: 1968.1392845230012, 0.5: 3556.039936787505, 0.9: 6486.3635112501315, 0.95: 7442.877960013898, 0.99: 8813.76833940386, 1.0: 10400.723100303465}
====================================================================================================

[SUBMISSION sanity checks]
  path: D:\AgentDs\agent_ds_healthcare\submission.csv
  shape: (2000, 2)
  cols: ['patient_id', 'ed_cost_next3y_usd']
  any NaNs: False
  min/median/max: 1087.32 3556.04 10400.72
  quantiles: {0: 1087.32, 0.01: 1335.6407000000002, 0.05: 1677.4755, 0.1: 1968.142, 0.25: 2588.9975, 0.5: 3556.04, 0.75: 4897.505, 0.9: 6486.365000000002, 0.95: 7442.8735, 0.99: 8813.771599999998, 1.0: 10400.72}

[SUBMISSION md5]
  md5: c3f21e1ed923959a1696bb40c7b5fba1
  (no previous md5 found)

[done] total wall time: 104.2s

Paste back: leaderboard MAE + (base_mae, final_mae) + chosen weight bag_list_wA + shift_meta + baseline_shrink_meta.
```

**Manager decision / verdict (what we did next):**
- Keep as the *reproducibility* baseline (same LB) — makes later diffs trustworthy.

### Iteration 03 — CODE 44 Ridge diversity (failed on LB)

**Prompt excerpt (what we actually told the LLM right before generating this code):**
```text
Code31 final_OOF=427.480 | Code31 LB=448.0793")
我希望你在这个突破了的⽅向上继续
improve!
Thought for 18m 1s
你这个现象（ 改了但 ⼀模⼀样 也⼀样 ）基本只有两种原因：
“ LB / log ”
新模型最终被 规则选成了 权重（⽐如 ） 最终 跟
1. one‑SE 0 wC=0 / wD=0 → submission
上⼀版完全相同；
你虽然加了模型，但 搜索 权重 的实现让最终落回同⼀个权重点
2. ensemble / bagging
（也会导致 不变）。
submission
你现在的突破⽅向（更强正则 更强多样性）是对的。下⼀步最 ⼲净 的就是：加⼀个完全
+ “ ”
不同的归纳偏置： （线性），然后做 权重搜索，并保持你
Ridge 4‑way simplex Code31/43
的所有安全机制（ 、 、 、 ）。
one‑SE bag chronic shift baseline shrink
下⾯给你⼀份完整可单 直接跑的 （
cell CODE 44 Code43 + SINGLE CHANGE: Ridge as
）。
Model D
我还加了⼀个⾮常实⽤的 （ 值的 ），你⼀眼能看出 这次
pred_hash submission CRC32 “
到底有没有变 。
submission ”
# ============================================
```

**Single-point change (delta):**
- Add Model D = Ridge regression; extend ensemble to 4-way simplex weight search (A/B/C/D) while keeping all other logic identical.

**Code pointer (for open-source reproducibility):**
- Notebook: `Challenge2_baseline_ichi_v9.ipynb`  |  code cell index: `14`  |  markdown label cell: `13`

**Measured results (from the code cell’s printed final summary):**
- OOF base MAE: 426.773
- OOF after chronic shift: 426.416
- OOF final MAE: 426.416
- TEST pred_hash (crc32): 225292366
- Leaderboard MAE (as recorded in notebook): **448.7924**

**Run summary block (verbatim, for audit):**
```text
[FINAL SUMMARY - CODE 44]
  SINGLE CHANGE: Added Model D = Ridge for ensemble diversity
  base OOF MAE (ABCD bag):   426.773
  after chronic shift MAE:   426.416  (delta=-0.356)
  final OOF MAE:             426.416  (delta=-0.356)
  chosen weights: {'wA': 0.1, 'wB': 0.5, 'wC': 0.3, 'wD': 0.1, 'obj': 428.5130786512598, 'mean': 428.3702592598699, 'std': 0.714096956949409}
  best weights: {'wA': 0.15, 'wB': 0.35, 'wC': 0.4, 'wD': 0.1, 'obj': 428.4373669709596, 'mean': 428.30148993591246, 'std': 0.6793851752356944}
  AB-only ref: {'wA': 0.4, 'wB': 0.6, 'wC': 0.0, 'wD': 0.0, 'obj': 429.95209383956944, 'mean': 429.8237377239825, 'std': 0.6417805779347993}
  ABC-only ref: {'wA': 0.2, 'wB': 0.4, 'wC': 0.4, 'wD': 0.0, 'obj': 429.2815209463075, 'mean': 429.1482109826456, 'std': 0.6665498183095325}
  chronic shift meta: {'base_mae': 426.7727370689358, 'obj': 426.634343067666, 'cv_mae': 426.614343067666, 'cf': 0.75, 'cv_gain_vs_base': 0.15839400126975534} | applied: True
  baseline shrink meta: {'applied': False, 'gain': 0.0}
  OOF quantiles: {0: 967.8304194772783, 0.01: 1319.984103901501, 0.05: 1721.0104433450076, 0.1: 2003.908051532762, 0.5: 3521.236048582974, 0.9: 6502.599001934906, 0.95: 7326.67424028398, 0.99: 8685.85720554266, 1.0: 10460.081612196247}
  TEST quantiles: {0: 980.8454853993477, 0.01: 1288.9202460285856, 0.05: 1654.7165603401543, 0.1: 1956.4379217742728, 0.5: 3558.40358734658, 0.9: 6460.177216280456, 0.95: 7381.5264749204325, 0.99: 8795.13292132188, 1.0: 10579.973084880436}
  pred_hash(crc32 of rounded test preds): 225292366

  >>> If chosen wD > 0 and LB improves -> Ridge diversity is helping.
  >>> If chosen wD = 0 and pred_hash matches Code43 -> submission will be identical.
====================================================================================================

[SUBMISSION sanity checks]
  path: D:\AgentDs\agent_ds_healthcare\submission.csv
  shape: (2000, 2)
  cols: ['patient_id', 'ed_cost_next3y_usd']
  any NaNs: False
  min/median/max: 980.85 3558.3999999999996 10579.97
  quantiles: {0: 980.85, 0.01: 1288.9247, 0.05: 1654.719, 0.1: 1956.435, 0.25: 2603.365, 0.5: 3558.3999999999996, 0.75: 4887.3825, 0.9: 6460.180000000002, 0.95: 7381.5239999999985, 0.99: 8795.128999999999, 1.0: 10579.97}

[done] total wall time: 140.7s

Paste back:
  leaderboard MAE + (base_mae, final_mae) + chosen weights (wA,wB,wC,wD) + shift_meta + baseline_shrink_meta + pred_hash
```

**Manager decision / verdict (what we did next):**
- Reject for leaderboard (OOF improved but LB worsened → classic CV–LB mismatch).

### Iteration 04 — CODE 43 regularized diversity breakthrough (BEST)

**Prompt excerpt (what we actually told the LLM right before generating this code):**
```text
heavy-tail
轻量的 校准
2. OOF
⽐如⽤ 做⼀层线性校准：
OOF y ≈ a * pred + b
或做⾮常轻微的分位数映射（不要太激进，否则也会像 ⼀样容易反噬）
chronic shift
结论给你⼀句话版本
主线回滚到 （ 或强制 ）是对的。
447.x Code43 / wD=0
这次退步⼤概率是 弱但不相关 的 作为 参与平均导致的（以及
“ ” Ridge base model
有⼀定 ⻛险）。
chronic shift overfit
下⼀步想稳扎稳打提升 ，优先做 对⻬（分桶 对抗验证），以及 ⽤ 做⼆层
LB CV / ridge
、⽽不是做第四个 模型。
blender base
我需要 给我
code, code
Thought for 16m 27s
下⾯给你 路线的 （ 三模型 权重搜索 ）完整单
447.x Code43 A/B/C + 3-way + chronic shift
代码，直接整段粘贴跑即可。
cell
你只需要确认 路径和 ⽂件存在。
DATA_DIR receipts_parsed.joblib
# ============================================
```

**Single-point change (delta):**
- Add Model C = shallower, stronger-regularized CatBoost (depth=3) on PRUNED features; move from AB to ABC weight search (one-SE + bagging).

**Code pointer (for open-source reproducibility):**
- Notebook: `Challenge2_baseline_ichi_v9.ipynb`  |  code cell index: `18`  |  markdown label cell: `17`

**Measured results (from the code cell’s printed final summary):**
- OOF base MAE: 427.504
- OOF after chronic shift: 427.118
- OOF final MAE: 427.118
- TEST pred_hash (crc32): 3230597766
- Leaderboard MAE (as recorded in notebook): **447.9542**

**Run summary block (verbatim, for audit):**
```text
[FINAL SUMMARY - CODE 43]
  base OOF MAE (ABC bag):    427.504
  after chronic shift MAE:   427.118  (delta=-0.386)
  final OOF MAE:             427.118  (delta=-0.386)
  chosen weights: {'wA': 0.2, 'wB': 0.5, 'wC': 0.3, 'obj': 429.3363412117731, 'mean': 429.2013211075919, 'std': 0.6751005209059112}
  best weights:   {'wA': 0.2, 'wB': 0.4, 'wC': 0.4, 'obj': 429.2815209463075, 'mean': 429.1482109826456, 'std': 0.6665498183095325}
  AB-only ref:    {'wA': 0.4, 'wB': 0.6, 'obj': 429.95209383956944, 'mean': 429.8237377239825}
  chronic shift meta: {'base_mae': 427.50442903461686, 'obj': 427.3881728582975, 'cv_mae': 427.3681728582975, 'cf': 0.75, 'cv_gain_vs_base': 0.13625617631936393} | applied: True
  baseline shrink meta: {'applied': False, 'gain': 0.0}
  OOF quantiles: {0: 1060.6426711421016, 0.01: 1379.3048438335516, 0.05: 1736.3384890209277, 0.1: 2007.99492735281, 0.5: 3509.7822311189643, 0.9: 6535.783459919906, 0.95: 7348.483139735989, 0.99: 8694.26910282528, 1.0: 10224.885424138714}
  TEST quantiles: {0: 1072.5112386101946, 0.01: 1339.6777455322483, 0.05: 1677.5343436846451, 0.1: 1966.8540511267147, 0.5: 3553.1797196353054, 0.9: 6475.3076681301245, 0.95: 7434.414752370273, 0.99: 8807.270204483106, 1.0: 10444.287889428582}
  pred_hash(crc32 of rounded test preds): 3230597766
====================================================================================================

[SUBMISSION sanity checks]
  path: D:\AgentDs\agent_ds_healthcare\submission.csv
  shape: (2000, 2)
  cols: ['patient_id', 'ed_cost_next3y_usd']
  any NaNs: False
  min/median/max: 1072.51 3553.1800000000003 10444.29
  quantiles: {0: 1072.51, 0.01: 1339.679, 0.05: 1677.5339999999999, 0.1: 1966.854, 0.25: 2591.6375, 0.5: 3553.1800000000003, 0.75: 4893.3825, 0.9: 6475.305000000004, 0.95: 7434.415499999999, 0.99: 8807.271099999998, 1.0: 10444.29}

[done] total wall time: 118.2s
```

**Manager decision / verdict (what we did next):**
- Adopt as best-known (first time entering 447.x).

### Iteration 05 — CODE 45 weight-pick = best (failed on LB)

**Prompt excerpt (what we actually told the LLM right before generating this code):**
```text
的唯⼀改动（相对 ）
Code45 Code43
把 权重选择从 偏保守（会刻意压低 ）改为 直接选 最优点（通常会
3-way “one-SE wC ” “ obj
给 更⼤权重），然后照旧做邻域 。
Model C ” weight-bagging
你现在的 ⾥：
Code43
最优点（ 最⼩）是
3-way obj (wA=0.20, wB=0.40, wC=0.40)
但 规则最后选的是 （它在 近似最优 ⾥优先选更⼩的 ）
one-SE (0.20, 0.50, 0.30) “ ” wC
⽽ 的突破核⼼就是：更强正则的 的多样性有⽤。
Claude Model C
所以这个改动的假设是： 过度保守，压掉了 该给 的权重 ， 可能因此损失⼀点
one-SE “ C ” LB
点。
这不是加特征，不是复杂后处理，不会引⼊ 那种 分箱校准 幻觉 。
Code33 “ CV ”
训练时间基本不变（仍是 模型 ）。
3 * 7 folds * 3 seeds
✅ （ 直接跑，输出 ）
Code45 1 cell submission.csv
默认 （就是本次唯⼀改动）。
CFG.WEIGHT_PICK_MODE="best"
如果你想完全复现 ，把它改回 即可。
Code43 "one_se"
# ============================================
```

**Single-point change (delta):**
- Change weight selection rule ONLY: choose the objective-minimizing BEST point instead of one-SE conservative pick.

**Code pointer (for open-source reproducibility):**
- Notebook: `Challenge2_baseline_ichi_v9.ipynb`  |  code cell index: `20`  |  markdown label cell: `19`

**Measured results (from the code cell’s printed final summary):**
- OOF base MAE: 427.444
- OOF after chronic shift: 427.094
- OOF final MAE: 427.094
- TEST pred_hash (crc32): 4255401435
- Leaderboard MAE (as recorded in notebook): **448.0529**

**Run summary block (verbatim, for audit):**
```text
[FINAL SUMMARY - CODE 45]
  WEIGHT_PICK_MODE: best  (Code45 default=best; Code43 was one_se)
  base OOF MAE (ABC bag):    427.444
  after chronic shift MAE:   427.094  (delta=-0.351)
  final OOF MAE:             427.094  (delta=-0.351)
  chosen weights: {'wA': 0.2, 'wB': 0.4, 'wC': 0.4, 'obj': 429.2815209463075, 'mean': 429.1482109826456, 'std': 0.6665498183095325}
  best weights:   {'wA': 0.2, 'wB': 0.4, 'wC': 0.4, 'obj': 429.2815209463075, 'mean': 429.1482109826456, 'std': 0.6665498183095325}
  AB-only ref:    {'wA': 0.4, 'wB': 0.6, 'obj': 429.95209383956944, 'mean': 429.8237377239825}
  chronic shift meta: {'base_mae': 427.44411699730705, 'obj': 427.3562206719334, 'cv_mae': 427.3362206719334, 'cf': 0.75, 'cv_gain_vs_base': 0.10789632537364469} | applied: True
  baseline shrink meta: {'applied': False, 'gain': 0.0}
  OOF quantiles: {0: 1068.4860992446354, 0.01: 1375.8372690073825, 0.05: 1734.1676293342546, 0.1: 2008.892898797754, 0.5: 3506.678993069584, 0.9: 6534.937734043596, 0.95: 7340.509005650822, 0.99: 8696.841762669428, 1.0: 10229.576050521693}
  TEST quantiles: {0: 1070.877284080643, 0.01: 1343.8601496980286, 0.05: 1676.7892325783102, 0.1: 1967.370220300722, 0.5: 3552.5138090487426, 0.9: 6474.58457530236, 0.95: 7436.496935273796, 0.99: 8804.704583547289, 1.0: 10454.565511193288}
  pred_hash(crc32 of rounded test preds): 4255401435
====================================================================================================

[SUBMISSION sanity checks]
  path: D:\AgentDs\agent_ds_healthcare\submission.csv
  shape: (2000, 2)
  cols: ['patient_id', 'ed_cost_next3y_usd']
  any NaNs: False
  min/median/max: 1070.88 3552.5150000000003 10454.57
  quantiles: {0: 1070.88, 0.01: 1343.8610999999999, 0.05: 1676.787, 0.1: 1967.373, 0.25: 2594.3725, 0.5: 3552.5150000000003, 0.75: 4896.23, 0.9: 6474.588000000004, 0.95: 7436.5005, 0.99: 8804.702899999998, 1.0: 10454.57}

[done] total wall time: 113.3s

Paste back: leaderboard MAE + (base_mae, final_mae) + chosen weights + shift_meta + pred_hash.
```

**Manager decision / verdict (what we did next):**
- Reject; revert to one-SE (best-point choice overfits CV noise, harms LB).

### Iteration 06 — CODE 46 gated baseline-bin residual calibration (no net gain)

**Prompt excerpt (what we actually told the LLM right before generating this code):**
```text
Code45 “Paste back”
Leaderboard MAE: 448.0529
(base_mae, final_mae): (427.444117, 427.093509)
base_mae = 427.44411699730705
final_mae = 427.09350894255795
chosen weights (wA,wB,wC): wA=0.20, wB=0.40, wC=0.40
shift_meta: {'base_mae': 427.44411699730705, 'obj': 427.3562206719334,
'cv_mae': 427.3362206719334, 'cf': 0.75, 'cv_gain_vs_base':
0.10789632537364469} | applied= True
applied shifts ≈ {DiabetesComp: +2.388, HF: +13.612, Pneumonia: -15.598}
baseline_shrink_meta: {'applied': False, 'gain': 0.0}
pred_hash: 4255401435
你说得对： 是负优化，我们回到 （ 选权重） 的路线继续试⼀个 微
Code45 Code43 one-SE “
⼩但不同 的改动，⽽且这次我直接给你 （单 可跑）。
” full code cell
下⾯是 ：在 完全不动 的前提
CODE 46 Code43 A/B/C + one-SE + bagging + chronic shift
下，只加⼀个很轻量的后处理：
✅ 按 分位数分箱的 校准（ ）
“ baseline_next3y residual bin residual shift ”
⽬的：修正 不同成本区间 系统性偏差，减少 的那种漂移
“ ” CV-LB gap
特点：极便宜（只在 上做⼏次⼩ ），可控、可关（ 就等于完全不变）
OOF CV factor=0
不涉及 （你吐槽 的那种东西，这⾥完全不碰）
AUC AUC≈0.5
# ============================================
```

**Single-point change (delta):**
- Add *optional* baseline_next3y quantile-bin residual calibration after chronic shift; calibration is CV-tuned and can auto-disable (bin_factor=0).

**Code pointer (for open-source reproducibility):**
- Notebook: `Challenge2_baseline_ichi_v9.ipynb`  |  code cell index: `22`  |  markdown label cell: `21`

**Measured results (from the code cell’s printed final summary):**
- OOF base MAE: 427.504
- OOF after chronic shift: 427.118
- OOF after bin calibration: 427.118
- OOF final MAE: 427.118
- TEST pred_hash (crc32): 562791831
- Leaderboard MAE (as recorded in notebook): **447.9542**

**Run summary block (verbatim, for audit):**
```text
[FINAL SUMMARY - CODE 46]
  SINGLE CHANGE: Added bin residual calibration after chronic shift (baseline_next3y quantile bins)
  base OOF MAE (ABC bag):      427.504
  after chronic shift MAE:     427.118  (delta=-0.386)
  after bin calibration MAE:   427.118  (delta=+0.000)
  final OOF MAE:               427.118  (delta=-0.386)
  chosen weights: {'wA': 0.2, 'wB': 0.5, 'wC': 0.3, 'obj': 429.3363412117731, 'mean': 429.2013211075919, 'std': 0.6751005209059112}
  best weights:   {'wA': 0.2, 'wB': 0.4, 'wC': 0.4, 'obj': 429.2815209463075, 'mean': 429.1482109826456, 'std': 0.6665498183095325}
  AB-only ref:    {'wA': 0.4, 'wB': 0.6, 'obj': 429.95209383956944, 'mean': 429.8237377239825}
  chronic shift meta: {'base_mae': 427.50442903461686, 'obj': 427.3881728582975, 'cv_mae': 427.3681728582975, 'cf': 0.75, 'cv_gain_vs_base': 0.13625617631936393} | applied: True
  bin shift meta: {'base_mae': 427.1180249041759, 'obj': 427.1180249041759, 'cv_mae': 427.1180249041759, 'bin_factor': 0.0, 'cv_gain_vs_base': 0.0, 'n_bins': 8} | applied: False
  baseline shrink meta: {'applied': False, 'gain': 0.0}
  OOF quantiles: {0: 1060.6426711421016, 0.01: 1379.3048438335516, 0.05: 1736.3384890209277, 0.1: 2007.99492735281, 0.5: 3509.7822311189643, 0.9: 6535.783459919906, 0.95: 7348.483139735989, 0.99: 8694.26910282528, 1.0: 10224.885424138714}
  TEST quantiles: {0: 1072.5112386101946, 0.01: 1339.6777455322483, 0.05: 1677.5343436846451, 0.1: 1966.8540511267147, 0.5: 3553.1797196353054, 0.9: 6475.3076681301245, 0.95: 7434.414752370273, 0.99: 8807.270204483106, 1.0: 10444.287889428582}
  pred_hash(crc32 of rounded test preds): 562791831
====================================================================================================

[SUBMISSION sanity checks]
  path: D:\AgentDs\agent_ds_healthcare\submission.csv
  shape: (2000, 2)
  cols: ['patient_id', 'ed_cost_next3y_usd']
  any NaNs: False
  min/median/max: 1072.51 3553.1800000000003 10444.29
  quantiles: {0: 1072.51, 0.01: 1339.679, 0.05: 1677.5339999999999, 0.1: 1966.854, 0.25: 2591.6375, 0.5: 3553.1800000000003, 0.75: 4893.3825, 0.9: 6475.305000000004, 0.95: 7434.415499999999, 0.99: 8807.271099999998, 1.0: 10444.29}

[done] total wall time: 117.3s

Paste back: leaderboard MAE + (base_mae, final_mae) + chosen weights + shift_meta + bin_shift_meta + baseline_shrink_meta + pred_hash.
```

**Manager decision / verdict (what we did next):**
- Keep as safe optional layer; in this run it gated to no-op and matched the best LB.

---
## E) Cross-iteration takeaways (paper-ready)
### E.1 What actually worked
- **Regularized tree diversity** (Model C depth=3) improved leaderboard even without improving OOF dramatically.
- **Conservative selection (one-SE)** acts like a generalization prior; it helped LB compared to choosing the best CV objective point.

### E.2 What repeatedly failed (and why)
- **OOF↓ does not guarantee LB↓** at this stage: Ridge (Iteration 03) improved OOF but degraded LB, suggesting mismatch / sensitivity to distribution tails.
- **Aggressive tuning without gating** (e.g., picking best weights) tends to overfit fold noise and harms LB (Iteration 05).

### E.3 Engineering principles that prevented wasted submissions
- Add `pred_hash` / submission md5 checks so we never submit an unchanged file by accident (Iteration 02).
- Any post-processing must be gated and allowed to become a no-op (Iteration 06).
- Strictly enforce single-point deltas — otherwise ablations become non-attributable.

---
## F) Appendix (optional): Iteration 07 (not part of the requested 01–06 set)
- Notebook label: `Iteration 07`  |  LB recorded: **448.1441**
- Summary: Model C uses **Huber** loss (robust diversity) instead of RMSE; everything else kept identical to Code43 skeleton.

Prompt excerpt:
```text
群 这条路基本没信息（⾄少⽬前特征下），所以接下来我们别再⾛ 先分类再分模型 分校
” “ /
正 的⽅向（那种很容易⽩忙、还会引⼊ ）。
” shift
你要的是回到 （ ）这个⽅向，做⼀个很⼩但更有希望的改变。我这次给你
Code43 447.9542
⼀个单点改动，⾮常符合 ⼩⼼假设，⼤胆尝试 ：
“ ”
（建议尝试）
CODE 47
单⼀改动：让 ⽤ （带 ）代替
Model C Huber loss delta RMSE
你已经验证：更强正则 更浅的 带来了 提升（ ）。
/ Model C LB 447.95x
但你也看到： 在 上更好， 却崩 说明我们更需要 对 更鲁棒
Ridge CV LB → outlier/shift
的多样性，⽽不是 更低 的多样性。
“CV ”
是 与 的折中：⼩误差⽤ （稳定、好拟合），⼤误差转 （抗离
Huber RMSE MAE L2 L1
群、抗 ），常⽤于 ⼤ 的场景。
shift CV-LB gap
下⾯是完整可直接粘贴 跑的 （不依赖外部代码）。
1 cell full code
（除了 的 变成 ，其它尽量保持 的⻣架与逻辑⼀致。）
Model C loss_function Huber Code43
Python
# ===============================================================================
```

Verbatim summary:
```text
[FINAL SUMMARY - CODE 47]
  SINGLE CHANGE: Model C loss_function=Huber (robust) instead of RMSE
  base OOF MAE (ABC bag):    427.996
  after chronic shift MAE:   427.579  (delta=-0.417)
  final OOF MAE:             427.579  (delta=-0.417)
  chosen weights: {'wA': 0.25, 'wB': 0.75, 'wC': 0.0, 'obj': 430.0175242980049, 'mean': 429.88362267530505, 'std': 0.6695081134992206}
  best weights:   {'wA': 0.4, 'wB': 0.6, 'wC': 0.0, 'obj': 429.95209383956944, 'mean': 429.8237377239825, 'std': 0.6417805779347993}
  chronic shift meta: {'base_mae': 427.99634152109167, 'obj': 427.83067695246496, 'cv_mae': 427.810676952465, 'cf': 0.75, 'cv_gain_vs_base': 0.18566456862669156} | applied: True
  baseline shrink meta: {'applied': False, 'gain': 0.0}
  OOF quantiles: {0: 1044.1798483315329, 0.01: 1388.161128679174, 0.05: 1731.4399011596208, 0.1: 2005.7768131027137, 0.5: 3504.3159774707565, 0.9: 6541.108389804956, 0.95: 7354.086638968208, 0.99: 8687.08799697188, 1.0: 10219.535677594396}
  TEST quantiles: {0: 1081.7247206723575, 0.01: 1334.2929724099101, 0.05: 1677.3024028983116, 0.1: 1966.4985089131815, 0.5: 3556.3728028979845, 0.9: 6482.899218456129, 0.95: 7441.5802269492715, 0.99: 8818.34921912536, 1.0: 10413.225944142385}
  pred_hash(crc32 of rounded test preds): 1997099653
====================================================================================================

[SUBMISSION sanity checks]
  path: D:\AgentDs\agent_ds_healthcare\submission.csv
  shape: (2000, 2)
  cols: ['patient_id', 'ed_cost_next3y_usd']
  any NaNs: False
  min/median/max: 1081.72 3556.37 10413.23
  quantiles: {0: 1081.72, 0.01: 1334.2940999999998, 0.05: 1677.303, 0.1: 1966.5, 0.25: 2589.8725, 0.5: 3556.37, 0.75: 4893.615, 0.9: 6482.903000000001, 0.95: 7441.581499999999, 0.99: 8818.352399999998, 1.0: 10413.23}

[done] total wall time: 158.5s

Paste back:
  leaderboard MAE + (base_mae, final_mae) + chosen weights + shift_meta + baseline_shrink_meta + pred_hash
```
