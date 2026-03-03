---
title: "AgentDS Healthcare Challenge 2 (ED Cost Forecasting) — Human Manager Prompts + Code + MAE 迭代整理"
created: 2026-02-24
purpose: |
  1) 把 Human Manager 侧的 prompt（含关键介入时刻）结构化归档；
  2) 把关键代码版本（CODE25/CODE32 等）与 MAE 迭代信息整理成可复现流程；
  3) 便于后续交给更强的 AI 进一步做 prompt 总结、方法论抽象与论文写作。
source_files:
  - "IEEE ICHI - Healthcare AI Challenge.pdf" (prompt pack + single mega prompt)
  - "IEEE ICHI - ED Cost Forecasting Strategy-3.pdf" (迭代式 Research & Validation 工作流 prompt)
  - "ED成本预测策略.pdf" (v3/v4/v5 MAE 记录与方法演进摘要)
  - "IEEE ICHI - Healthcare Cost Forecasting Strategy.pdf" (早期基线 493.9 & target 448.7)
  - "IEEE ICHI - Code 25 Challenge 2.pdf" (CODE25 代码与 CODE32 paste-back/MAE 元信息)
notes:
  - PDF 文本抽取存在行尾截断（例如 expert -> exper），必要处我做了“语义补全/润色版”，同时保留“原始抽取版”以便追溯。
  - 本文档不试图粘贴 658 页完整代码；只抽取“可复现流程必需”的关键结构、关键函数思想与迭代元信息。
---

# 0. 术语与角色

## 0.1 Human Manager（人类管理者）职责
Human Manager 的核心职责不是“逐行指挥 AI 写代码”，而是：

- **定义目标与约束**：明确 metric=MAE、提交格式、运行环境（Windows+RTX4060）、数据文件路径、计算预算等。
- **提供上下文与证据包（Data Pack）**：包括已完成的 EDA 结论、已知的信号、历史最好 MAE、失败模式。
- **制定迭代协议**：每次运行后必须 paste-back 哪些日志/指标；每次只做少量高 ROI 变更；控制迭代节奏。
- **关键介入**（本文件重点）：在 AI 输出发散、格式不稳定、计算过度、或改动不产生收益时，用 prompt “收敛行为”。

## 0.2 AI Worker（被管理的 AI）职责
- 端到端决策：特征工程、CV、建模、集成、训练全量、生成 submission。
- 输出必须可复制运行（notebook cell / 单 cell）。
- 不做无意义 EDA 叙述；以可执行代码为主。

---

# 1. Human Manager Prompt 资产库（含关键介入时刻）

> 说明：以下 prompt 原文来自 `IEEE ICHI - Healthcare AI Challenge.pdf` 的「Autopilot Prompt Pack」与「Single Mega Prompt」部分。

## 1.1 关键介入时刻 #1：Autopilot Prompt Pack（3-message 结构）

**介入动机：** Human Manager 不能/不想“微观管理”AI，但需要让 AI 自动产出可运行 notebook cells，并且遵守提交格式与 MAE 优化目标。

### Message 1 — System / Role（角色与工作方式）
**原始抽取版（可能有截断）：**
```text
You are a world-class Senior Machine Learning Scientist + Healthcare domain exper
You are pragmatic, results-driven, and you write production-quality, reproducible
Mission:
Win the AgentDS Healthcare Challenge 2 (ED Cost Forecasting) by materially improv
You must make your own modeling decisions end-to-end (feature engineering, valida
Working style:
- Be concrete and executable. Prefer code over long prose.
- Make strong assumptions when needed; do not ask clarifying questions unless abs
- Think in ablations: propose a small number of high-ROI experiments, run them, c
- Be careful about leakage, split consistency, and submission format.
- If you cite any external facts (papers, library behaviors), cite sources. If un
```

**语义补全/润色版（便于复用，建议作为 System message）：**
```text
You are a world-class Senior Machine Learning Scientist + Healthcare domain expert.
You are pragmatic, results-driven, and you write production-quality, reproducible code.

Mission:
Win the AgentDS Healthcare Challenge 2 (ED Cost Forecasting) by materially improving MAE on the leaderboard.

You must make your own modeling decisions end-to-end (feature engineering, validation, model selection, ensembling, and submission generation).

Working style:
- Be concrete and executable. Prefer code over long prose.
- Make strong assumptions when needed; do not ask clarifying questions unless absolutely necessary.
- Think in ablations: propose a small number of high-ROI experiments, run them, compare results, iterate.
- Be careful about leakage, split consistency, and submission format.
- If you cite any external facts (papers, library behaviors), cite sources. If unsure, state uncertainty.
```

### Message 2 — Context / Data Pack（挑战定义 + 环境/文件 + 数据事实 + 已知信号）
> **这是 Human Manager 最“值钱”的输入**：把你已经做过的 EDA 结论、当前最好分数、失败模式、以及你不想再做的事情，全部塞进来。

**原始抽取版（可能有截断）：**
```text
### Challenge
Domain: Healthcare
Task: ED Cost Forecasting (regression)
Metric: MAE (lower is better)
Goal: predict total ED costs over the next 3 years for each patient in test set.

### Local constraints
- Environment: local Windows
- GPU: RTX 4060 (CUDA 12)
- Prefer solutions that can run on this setup without massive hyperparameter swee
### Files available in working directory
- ed_cost_train.csv, ed_cost_test.csv
- patients.csv
- admissions_train.csv, admissions_test.csv
- receipts_pdf/ (many PDFs; optional to parse)
- receipts_parsed.joblib (already-parsed receipt features; preferred over re-pars
- sample_submission.csv (format reference)
Load hints (already working):
train_costs = pd.read_csv("ed_cost_train.csv")
test_costs  = pd.read_csv("ed_cost_test.csv")
same for patients.csv, admissions_train/test.csv
### Dataset facts you MUST respect
- Each patient_id has one synthetic ED billing summary PDF.
- The PDF line items sum to a final TOTAL.
- The TOTAL matches prior_ed_cost_5y_usd in the ed_cost table (this is guaranteed
- Our parsing observations:
  TRAIN match_rate: pdf_total vs prior = 0.895 | sum_items vs prior = 1.0
  TEST  match_rate: pdf_total vs prior = 0.89  | sum_items vs prior = 1.0
  => Use sum_items (sum of line totals) as the reliable reconstruction of prior c
### What we already tried / current baseline
- Multiple iterations (v3/v4/v5 notebooks exist). We did enough EDA already.
- Current results: MAE ~460 typical; best run so far ~455 MAE.
- We need a practical, code-first improvement path to break 455 and push toward MA
### Observed signal from receipt-derived features
These high-acuity receipt codes correlate with higher residuals (underprediction)
has_intub_31500, has_cvc_36556, has_cpr_92950, has_critical_care,
has_artline_36620, has_ct_head_70450, has_99285, has_ct_abdpel_74177, has_troponi
### Non-negotiable requirements
- I do NOT want more EDA narrative.
- I want Jupyter notebook CELL code I can copy-paste directly (not a .py file).
```

**可复用模板（建议 Human Manager 每次迭代时更新该块，尤其是“当前最好 MAE”“已知信号”“失败模式”）：**
```text
### Challenge
- Domain: Healthcare
- Task: ED Cost Forecasting (regression)
- Metric: MAE (lower is better)
- Goal: predict total ED costs over the next 3 years for each patient in the test set.

### Local constraints
- Environment: local Windows
- GPU: RTX 4060 (CUDA 12)
- Prefer solutions that run in this setup without massive hyperparameter sweeps.

### Files available
- ed_cost_train.csv, ed_cost_test.csv
- patients.csv
- admissions_train.csv, admissions_test.csv
- receipts_pdf/ (many PDFs; optional to parse)
- receipts_parsed.joblib (preferred over re-parsing PDFs)
- sample_submission.csv

### Dataset facts you MUST respect
- Each patient_id has one synthetic ED billing summary PDF.
- PDF line items sum to a final TOTAL.
- TOTAL matches prior_ed_cost_5y_usd (data consistency signal).

### What we already know (EDA / prior runs)
- Current results: MAE ~460 typical; best run so far ~455 MAE.
- Observed signal from receipt-derived features: high-acuity codes correlate with higher residuals (underprediction).
  Examples: has_intub_31500, has_cvc_36556, has_cpr_92950, has_critical_care, ...
- We do NOT want more EDA narrative; code-first only.

### Non-negotiable
- Output notebook CELL code I can copy-paste directly (not a .py file).
```

### Message 3 — Output Contract（强制输出格式 + 迭代协议）
**原始抽取版（可能有截断）：**
```text
### Your output must follow this structure (no exceptions)
1) "Plan (high-level, minimal)"
- 5–8 bullets max, each bullet = one concrete modeling idea / experiment.
- Prioritize by expected MAE gain vs effort. Avoid huge sweeps.
2) "Notebook Cells (copy-paste ready)"
Provide code blocks as notebook cells in the exact order I should run them:
- Cell 0: imports + settings + seed + optional pip installs
- Cell 1: load data + merge tables + basic sanity checks
- Cell 2: build receipt features from receipts_parsed.joblib (do not parse PDFs u
- Cell 3: build admissions aggregated features per patient_id (no label leakage)
- Cell 4: modeling baseline optimized for MAE (not RMSE)
- Cell 5: stronger model(s) + ensembling/stacking if justified
- Cell 6: train final + predict test + write submission.csv
Each cell should be runnable as-is.
3) "Guardrails / Sanity checks"
- List 6–10 checks (submission format, row counts, NaNs, leakage checks, target r
4) "Next iteration hooks"
- Tell me exactly what metrics/logs to paste back after I run it, so you can prop
### Additional constraints
- Optimize MAE directly when possible (MAE/L1 objectives).
- Be mindful of heavy tails and outliers (cost data).

额外给你两条「跟进迭代」用的短 prompt（你跑完一轮发回去就行）
Follow-up Prompt A（你给它你最新CV结果后，让它自我迭代）
Follow-up Prompt B（当你怀疑是特征/泄漏/提交格式问题时）

- If you use external libraries (xgboost/catboost/lightgbm), include install note
- Do NOT browse excessively: if you have web access, cite at most 3–6 high-qualit
```

**关键点（Human Manager 的“控制阀”）：**
- 强制输出结构：Plan（少量 bullet）+ Notebook Cells（严格按 cell 顺序）  
- 强制完成：CV 报告 MAE → 全量训练 → 生成 submission.csv → sanity checks  
- **关键迭代协议（paste-back 模板）**：这就是后续“最关键的介入点”，确保每次都可复现、可比较。

#### Message 3 中包含的“粘贴回传（paste-back）模板”
```text
Here are the results from your last notebook:
- CV MAE (folds): [...]
- Overall CV MAE: ...
- Public/validation notes: ...
- Top error segments (if any): ...
Now propose the next 3 highest-ROI improvements.
Rules:
- Must be incremental and code-first (new/modified notebook cells).
- No broad EDA.
- Keep compute reasonable on RTX 4060.
- Explain why each change should reduce MAE and what failure mode it targets.
```

#### Message 3 中包含的“故障诊断/审计模板（可选）”
```text
I suspect the bottleneck is either feature leakage, receipt feature construction,
Please:
1) audit the pipeline for leakage risks (patient_id joins, train/test overlap, ta
2) audit receipt feature aggregation (sum_items, missing pdf_total),
3) ensure MAE-optimized training (objective/eval_metric),
4) provide corrected cells only (diff-style), not a full re-write.

你这套 prompt 里最关键的“硬钉子”我已经帮你写进去了
PDF 的 TOTAL 必须匹配 prior_ed_cost_5y_usd（数据说明里写死的）
你给的 sample receipt 也清楚展示了“行项目 + TOTAL”的结构
（Code/Description/Qty/Unit/Line Total/TOTAL）
你们的 parsing 结论（sum_items 完美匹配 prior）被写成“必须遵守的事实”
性
输出合同强制它：不做长篇 EDA、只给可运行 cell、必须生成 submission.csv
小建议（不会变成微操，但能显著减少跑偏）
1)
 RMSE**：
很多库默认回归是 RMSE/平方误差，你们评估是 MAE，目标函数不对会卡在某个水平。你
在 prompt 里已经强制了“MAE/L1 objective”。（例如 XGBoost 就明确支持
reg:absoluteerror  作为 L1 回归目标）
2. 如果它选 CatBoost：
CatBoost 文档里也说明了 MAE/Quantile 情况下的叶子估计策略与 CPU/GPU 支持等细
节；你的硬件上它通常是很“省心”的强 baseline 候选。
如果你愿意，我也可以把上面 Message 2 里的 “Observed signal / residual lift 表”再整理成
一个更短的“Evidence pack”（只留最有用的 10 条），进一步减少 prompt 长度、但不损失关
键信息。

description
receipt_8
XGBoost Docum…
CatBoost

我要一段大prompt! 而且output是一个cell不是cells, 然后
ed_cost_train再详细一点就是:
"D:\AgentDs\agent_ds_healthcare\ed_cost_test.csv"
我所有data都在"D:\AgentDs\agent_ds_healthcare\.
再给我一版整合的, 当然信息千万不要省略和丢

Thought for 11m 47s
下面是你要的 「单段超大 Prompt（整合版）」——你直接复制从
```

---

## 1.2 关键介入时刻 #2：Single Mega Prompt（强制“只输出一个 Jupyter cell”）

**介入动机：** Human Manager 希望把 Message1/2/3 合并，直接丢给 AI，且强制 **只输出一个 python 代码块**（一个 cell），在 cell 内完成“读数据→训练CV→训练全量→预测→写 submission.csv”。

**原始抽取版（BEGIN PROMPT ~ END PROMPT）：**
```text
BEGIN PROMPT
You are a world-class Senior Machine Learning Scientist + Healthcare domain expert +
Kaggle/benchmark competition winner.
You will operate as an autonomous top-tier MLE/DS: you will decide the modeling strategy
end-to-end and implement it.
0) Mission (must obey)
We are competing in AgentDS Healthcare Domain — Challenge 2: ED Cost Forecasting
(regression).
We must predict ed_cost_next3y_usd for each patient_id in test set.
Evaluation metric: MAE (Mean Absolute Error) — lower is better.
Our current best MAE is around 455. We need a practical improvement path to approach
and ideally beat MAE < 400.
You MUST:
Make your own decisions (features, validation, model, ensembling, objectives) without
asking me to micro-manage.
Focus on practical, code-first improvements, not long EDA narrative.
Respect limited compute: local Windows + RTX 4060 (CUDA 12). Avoid massive
sweeps.
You are allowed to use GPU if it helps, but your solution must also be able to run
CPU-only as fallback.
1) Non-negotiable output format (CRITICAL)

Your entire response MUST be exactly ONE Jupyter notebook cell, i.e. one single Python
code block:
No extra text outside the code block. No multiple cells. No markdown explanations.
Inside that one cell, you may print a concise plan and print CV results, but keep it brief.
That single cell must be runnable end-to-end (no dependencies on previous cells) and
must:
1. Load data from the exact Windows paths below
2. Build patient-level features by merging tables + receipts_parsed.joblib + admissions
aggregates
3. Train with cross-validation and report MAE
4. Train final model(s) on full train
5. Predict test
6. Write a submission.csv  that strictly matches the sample submission format: columns
exactly:
patient_id
ed_cost_next3y_usd
7. Save submission to: D:\AgentDs\agent_ds_healthcare\submission.csv  (overwrite ok)
8. Print final sanity checks (shape, missing, column names) and the location of the saved
file.
2) File system & exact paths (Windows)
All data is located under:
DATA_DIR = r"D:\AgentDs\agent_ds_healthcare"
Use these exact paths:
ED cost:
TRAIN: r"D:\AgentDs\agent_ds_healthcare\ed_cost_train.csv"
TEST: r"D:\AgentDs\agent_ds_healthcare\ed_cost_test.csv"
python
# ... all code here ...

Patients:
r"D:\AgentDs\agent_ds_healthcare\patients.csv"
Admissions:
TRAIN: r"D:\AgentDs\agent_ds_healthcare\admissions_train.csv"
TEST: r"D:\AgentDs\agent_ds_healthcare\admissions_test.csv"
Receipts:
PDFs folder (optional / last resort only):
r"D:\AgentDs\agent_ds_healthcare\receipts_pdf"
Parsed receipts features (preferred):
r"D:\AgentDs\agent_ds_healthcare\receipts_parsed.joblib"
Sample submission reference:
r"D:\AgentDs\agent_ds_healthcare\submission.csv" (This is the sample format
reference ONLY; still overwrite with your generated predictions.)
Important: Use raw strings r"..." for Windows paths.
3) Dataset facts you MUST respect (do not debate)
We have one synthetic ED billing receipt PDF per patient_id.
Each PDF has line items (CPT/HCPCS-like codes, quantities, line totals) and a final TOTAL.
Key invariant from our EDA:
The receipt is constructed so that the sum of line totals (sum_items) matches the
row’s prior_ed_cost_5y_usd perfectly.
The PDF “TOTAL” field is sometimes missing/NaN in our parser, but sum_items is
reliable.
Our observed parsing quality:
TRAIN match_rate:
pdf_total vs prior = 0.895
sum_items vs prior = 1.0
TEST match_rate:
pdf_total vs prior = 0.89
sum_items vs prior = 1.0
Therefore:
Treat sum_items  as the reliable reconstruction of prior cost from receipts.

Do NOT rely on pdf_total being always present.

Worst 15 pdf_total mismatches (train) (note: abs_diff_pdf_total shown; many are 0 because
mismatch often means missing rather than numeric diff):
patient_id prior_ed_cost_5y_usd pdf_total sum_items abs_diff_pdf_total abs_diff_sum_items
1080 1031.30 1031.30 1031.30 0.0 0.000000e+00
1807 5664.15 5664.15 5664.15 0.0 9.094947e-13
3038 1025.28 1025.28 1025.28 0.0 0.000000e+00
1314 1852.77 1852.77 1852.77 0.0 2.273737e-13
3321 8489.80 8489.80 8489.80 0.0 0.000000e+00
1296 2912.06 2912.06 2912.06 0.0 0.000000e+00
33 735.11 735.11 735.11 0.0 0.000000e+00
3605 1737.56 1737.56 1737.56 0.0 0.000000e+00
1903 2325.33 2325.33 2325.33 0.0 4.547474e-13
1488 4377.89 4377.89 4377.89 0.0 0.000000e+00
1106 7359.35 7359.35 7359.35 0.0 0.000000e+00
2148 2885.07 2885.07 2885.07 0.0 4.547474e-13
821 597.53 597.53 597.53 0.0 0.000000e+00
686 4502.53 4502.53 4502.53 0.0 0.000000e+00
1303 306.45 306.45 306.45 0.0 5.684342e-14

Worst 15 sum_items mismatches (train) (these show pdf_total NaN but sum_items matches
prior):
patient_id prior_ed_cost_5y_usd pdf_total sum_items abs_diff_pdf_total abs_diff_sum_items
2841 16734.76 NaN 16734.76 NaN 3.637979e-12
3617 25827.83 NaN 25827.83 NaN 3.637979e-12
761 17680.51 NaN 17680.51 NaN 3.637979e-12
3810 25937.30 NaN 25937.30 NaN 3.637979e-12
2293 19299.59 NaN 19299.59 NaN 3.637979e-12
1548 19284.94 NaN 19284.94 NaN 3.637979e-12
3957 20391.86 NaN 20391.86 NaN 3.637979e-12
2612 21882.69 NaN 21882.69 NaN 3.637979e-12
835 12170.04 NaN 12170.04 NaN 3.637979e-12
3985 23004.74 NaN 23004.74 NaN 3.637979e-12
2292 16544.19 NaN 16544.19 NaN 3.637979e-12
2415 21941.82 NaN 21941.82 NaN 3.637979e-12
3459 18000.13 NaN 18000.13 NaN 3.637979e-12
594 20131.91 NaN 20131.91 NaN 3.637979e-12
3879 30797.47 NaN 30797.47 NaN 3.637979e-12
Also observed residual lift vs baseline (mean resid), by feature=1 (these codes tend to
indicate higher costs / underprediction risk):
feature support_n mean_resid_lift mean_target
has_intub_31500 678 140.609167 4950.480796
has_cvc_36556 682 123.421400 5000.174751
has_cpr_92950 664 109.065437 4893.741777
has_critical_care 1001 99.563237 4761.644076
has_artline_36620 682 96.552485 4934.775572
has_ct_head_70450 420 48.260357 3772.546429
has_99285 454 39.363778 3768.006498
has_ct_abdpel_74177 474 33.109884 3832.512405
has_troponin_84484 448 27.195614 3740.529107
has_obs_G0378 435 -9.453460 3845.414506

Interpretation:
These receipt-derived “high acuity” procedure indicators matter.
Prior_cost/prior_visits alone likely underpredict high-acuity patients unless features
are used well.
4) What you can assume about columns (but still verify in
code)
ed_cost_train/test likely have:
patient_id
primary_chronic
prior_ed_visits_5y
prior_ed_cost_5y_usd
(train only) ed_cost_next3y_usd target
patients.csv likely has:
patient_id, age, sex, insurance, zip3
admissions_train/test likely has:
admission_id, patient_id, primary_dx, los_days, acuity_emergent, charlson_band,
ed_visits_6m, discharge_weekday
(admissions_train has readmit_30d, but that column must NOT be used as feature
because it won’t exist in admissions_test and it’s a different target.)
5) Practical expectations & constraints
We already did plenty of EDA; do NOT spend output tokens on generic EDA or
correlation talk.
Prior attempts around MAE 460; best around 455. We need a stronger, practical
pipeline.
Many complex models did not help much so far; you must focus on what typically
moves MAE in this kind of synthetic healthcare cost regression: correct objective
(L1/MAE), robust handling of heavy tails, better receipts + admissions aggregations,
and careful ensembling.

6) Online research requirement (lightweight — DO NOT
over-search)
You may (optionally) do a small amount of online search (3–6 high-quality sources max)
about:
healthcare cost prediction / utilization forecasting / ED cost forecasting
modeling heavy-tailed costs with MAE (L1) objectives
competitions or papers with similar “prior utilization + claims codes” style features
tips for MAE optimization / robust regression
But do not over-search: keep it minimal and focus on actionable modeling decisions.
If you cite sources, cite them briefly in comments inside the code cell (short, not long
prose).
If you cannot browse, proceed anyway.
7) LLM usage (allowed, but be pragmatic)
You are an LLM. You can use your domain knowledge to:
group procedure codes into clinically meaningful buckets (airway, vascular access,
imaging, labs, critical care, observation, etc.)
create robust derived features from receipts codes (counts, spend shares, top-k codes,
interactions)
But do not create nonsense or overly complex ontology. Focus on features that help
MAE.
8) Required behaviors inside the code
Inside the single code cell you output, you must:
Set global SEED for reproducibility.
Implement a reliable CV scheme and report fold MAEs + overall MAE.
Avoid leakage: never use ed_cost_next3y_usd in feature engineering; never use
admissions_train.readmit_30d.
Prefer receipts_parsed.joblib; only parse PDFs if joblib is missing (and if parsing, do it
efficiently).
Handle categorical features properly (e.g., CatBoost native, or one-hot/target
encoding as appropriate).

Use MAE-aligned objectives when possible. Use robust models suitable for tabular
data.
Keep runtime reasonable on a 4060; no huge hyperparameter sweeps.
Produce final submission.csv  with correct columns/order and row count equal to
ed_cost_test rows.
9) What to print at the end (inside the code)
Print:
overall CV MAE
which model(s) / ensemble used
top 10 feature importances (if available)
sanity checks:
submission shape
submission columns exactly
any NaNs in predictions
min/median/max prediction
instructions: “Paste back CV MAE and these logs for next iteration”.
Now: produce the ONE-CELL python solution that best improves MAE beyond our current
455 baseline.
END PROMPT
```

**这一版 Mega Prompt 相比 3-message pack 的关键变化（可写入论文的人机协作方法学）：**
1) **输出约束更强**：只允许一个 cell → 更利于自动执行、减少 copy/paste 错误。  
2) **路径与 I/O 约束更强**：常见比赛失败点是“读错文件/写错 submission”。  
3) **末尾打印协议**：强制输出 CV MAE、特征重要性、submission sanity checks，并提示下一轮 paste-back。  

---

## 1.3 关键介入时刻 #3（可选）：强制“研究-验证”迭代工作流（先 EDA 再建模）

来源：`IEEE ICHI - ED Cost Forecasting Strategy-3.pdf`

**介入动机：** 当 Human Manager 想避免 AI 直接输出一坨最终 pipeline，而是要求先形成证据链（hypothesis→EDA→反馈→再建模）。

```text
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
```

---

# 2. Code 资产与结构化摘要（CODE25 / CODE32）

> 说明：代码原文在 `IEEE ICHI - Code 25 Challenge 2.pdf`（658 页）。这里抽取“可复现流程”的关键结构与关键迭代信息。

## 2.1 CODE25（当前最好 code 的核心思想）

**CODE25 头部原文摘录（含 MAE 信息与核心改动点）：**
```text
我们还在是challenge 2, 这是我们目前最好的code (MAE 
448.1210):
# === CODE 25 | Code22 route (best LB 448.1754) + "one-
standard-error" weight choice + small weight-bagging 
toward simpler (more B) ===
# Key idea (less-is-more): when several A/B blends are 
near-tied, prefer the simpler one (smaller A_full weight),
# and bag a tiny local neighborhood of weights to reduce 
sensitivity.
# Everything else stays Code22-core: same features, same 
CatBoost params, same multi-seed + trimmed-mean, same 
chronic residual shift.
#
# Output: D:\AgentDs\agent_ds_healthcare\submission.csv
import os, re, sys, gc, math, warnings, random, zlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
pd.set_option("display.max_columns", 260)
pd.set_option("display.width", 220)
SEED = 42
random.seed(SEED)
```

### CODE25：从方法上做了什么（结构化总结）
从头部注释可以抽象出 CODE25 的“可复现设计”：

1) **A/B 两套预测（A_full vs B_simpler）**  
   - 目的：在不改变核心特征/模型的大前提下，用 A/B 多样性来对抗过拟合与不稳定。
2) **One-standard-error（one-SE）权重选择 + “偏向更简单”**  
   - 当多个 A/B blending 权重几乎并列（near-tied），选更简单的（更小的 A 权重）。
3) **小范围 weight-bagging**  
   - 在 chosen weight 附近取小范围权重集合，平均，以减少对单点权重的敏感性。
4) **多 seed + trimmed mean**  
   - 用多随机种子训练；对 seed 预测做 trimmed mean，提升稳定性。
5) **chronic residual shift（慢性病组残差校正）**  
   - 对某些 chronic group 进行 cross-fitted residual shift，修正系统性欠/过预测。

### CODE25：可复现 pipeline 的“最小抽象伪代码”
```python
# 0) load tables & receipt features (prefer receipts_parsed.joblib)
X_train, y_train, X_test = build_features(...)

# 1) train 2 model variants (A/B) with multi-seed CV -> get OOF & test preds
oof_A, test_A = train_variant_A_multi_seed(...)
oof_B, test_B = train_variant_B_multi_seed(...)

# 2) AB weight search with stability objective (mean MAE + std penalty)
wA_chosen, bag_list = choose_weight_oneSE_prefer_simpler(oof_A, oof_B, y_train)

# 3) weight-bag predictions (average multiple weights near chosen)
oof_blend, test_blend = bag_weights(oof_A, oof_B, test_A, test_B, bag_list)

# 4) chronic residual shift (cross-fitted) applied on blended preds
oof_shifted, test_shifted, shift_meta = apply_chronic_shift(oof_blend, test_blend, y_train, group=primary_chronic)

# 5) optional global bias correction (guardrail: only apply if MAE improves)
test_final, bias_meta = optional_bias_correction(oof_shifted, test_shifted, y_train)

# 6) train full, predict, write submission.csv, print sanity checks + paste-back protocol
save_submission(test_final)
```

---

## 2.2 CODE32：Human Manager 要求的 “paste-back” 元信息（用于可复现迭代日志）

### 2.2.1 Human Manager 的 paste-back 协议（关键介入点）
```text
Paste back: leaderboard MAE + (base_mae, 
after_shift_mae, after_bias_mae, final_mae) + bias_meta + 
chosen weight bag_list + shift_meta.
这次调整太过于微小了, 连我们自己的validation MAE和real 
MAE都一模一样, 再继续做微创迭代争取突破MAE 448 到
447.x
Thought for 27m 48s
下面先把你要求的 CODE32 paste-back 信息整理出来，然后解释为什么这次“几乎没动”，
最后给你一版 CODE33（
```

> 解释：Human Manager 明确要求每次 run 都 paste-back 一组“最小必要指标”，这样 AI 才能在下一轮做有效、可比较的微创迭代。

### 2.2.2 CODE32 结果（原文 paste-back）
```text
(1) Leaderboard MAE
448.0793
(2) (base_mae, after_shift_mae, after_bias_mae, final_mae)
base_mae (AB bag OOF): 427.92823920810883
after_shift_mae: 427.47982859847690
after_bias_mae: 427.47982859847690
final_mae: 427.47982859847690
(3) chosen weight + bag_list
chosen wA/wB: wA=0.25, wB=0.75
bag_list_wA: [0.25, 0.30, 0.35]
(4) bias_meta
(5) chronic shift meta
{
  'applied': False,
  'mae_before': 427.4798,
  'mae_after': 427.4798,
  'gain': 0.0,
  'raw_bias': -0.4709,
  'bias_used': -0.4709,
  'min_gain': 0.05,
  'cap_abs': 250.0
}
{
  'base_mae': 427.92823920810883,
  'obj': 427.7605539008683,
  'cv_mae': 427.7405539008683,
  'cf': 0.75,
  'cv_gain_vs_base': 0.18768530724054244
}
# applied=True
# shifts: DiabetesComp -1.254, HF +13.62, Pneumonia -20.282
Python
```

**从该 paste-back 可得的结构化记录：**
- **Leaderboard MAE**：448.0793  
- **OOF 变化轨迹**（base → after shift → after bias → final）：  
  - base_mae: 427.9282  
  - after_shift_mae: 427.4798  
  - after_bias_mae: 427.4798  
  - final_mae: 427.4798  
- **AB 权重**：wA=0.25, wB=0.75；bag_list=[0.25, 0.30, 0.35]  
- **bias_meta**：applied=False（说明“全局偏置校正”这刀没砍下去，因为 guardrail 判断无收益）  
- **chronic shift meta**：包含 cv_gain_vs_base 等（可用于论文写“校正模块带来的收益”）

---

# 3. MAE 迭代信息汇总（按时间粗粒度）

> 说明：这里把不同阶段文档里出现的 MAE/Score 摘出来，做一张“可复现迭代时间线”。  
> 由于这些来自不同阶段/不同 code 线（v3/v4/v5 vs code25/32），请在论文中注明“阶段划分”。

## 3.1 早期基线（Iteration 1）
来源：`IEEE ICHI - Healthcare Cost Forecasting Strategy.pdf`

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
```

## 3.2 中期 v3/v4/v5（OOF 与 LB）
来源：`ED成本预测策略.pdf`

```text
group_ratio ：OOF MAE 3571.495 | out_oof_mae 1630.570（明显结构性失败）
powerlaw_log ：OOF MAE 540.008 | out_oof_mae 637.761
isotonic_group ：OOF MAE 505.529 | out_oof_mae 478.924
lgb_light_sublinear ：OOF MAE 464.763 | out_oof_mae 403.818
OOF MAE — LGB : 468.93
OOF MAE — XGB : 467.28
OOF MAE — CAT : 456.04
OOF MAE — ENSEMBLE: 456.99 (w=0.2/0.2/0.6)
提交（Leaderboard）：460.1811 MAE
输出里出现过多组 OOF 结果（例如 raw 450.x / calibrated 443.x 等），但最终提交：
473.7564 MAE
v5（含 Iter25 EDA + crossV1 提交）
```

```text
例如：[FINAL] Ensemble_LGB_CB_w0.56 | OOF MAE=452.589 | OOF outlier
MAE=398.328
也有多 seed LGB / CB 的 OOF 汇总输出（范围大约 456–458/462–471 等）
v5 的一次提交（crossV1 文件）：473.8055 MAE
```

## 3.3 后期（CODE25 / CODE32，冲击 448 → 447.x）
来源：`IEEE ICHI - Code 25 Challenge 2.pdf`

- CODE25: “目前最好的 code (MAE 448.1210)”（并提到 best LB 448.1754 的基线）  
- CODE32: LB MAE 448.0793；并给出 base/shift/bias/final 的 OOF 分解。

---

# 4. 可复现流程（论文可写的人机协作方法）

下面给一套 **Prompt → Code → Run → Paste-back → Next Iteration** 的闭环流程（你可以直接写进论文的「Methods」或「Appendix」）。

## 4.1 实验资产管理（强烈建议）
建议在仓库/实验目录中维护：

- `prompts/`
  - `prompt_pack_v1_messages.md`（3-message）
  - `prompt_mega_v2_one_cell.md`（BEGIN~END）
  - `prompt_iter_template.md`（paste-back 模板）
- `runs/`
  - `run_0001/`
    - `prompt_used.md`（当次实际发送给 AI 的 prompt）
    - `notebook.ipynb` 或 `cells.py`（当次代码）
    - `logs.txt`（CV MAE、weights、shift_meta、bias_meta、runtime）
    - `submission.csv`
    - `leaderboard_score.txt`（手工填）
- `results/mae_timeline.csv`（结构化记录）

## 4.2 闭环流程（每轮迭代都严格执行）

### Step 0 — 固定运行环境与数据路径
- 固定随机种子（多 seed 列表也要固定）。
- 固定输入文件路径与输出 submission 路径（避免“读错文件/写错格式”导致的噪声）。

### Step 1 — Human Manager 选择 prompt 形态（介入点）
- **初期/探索期**：用 3-message Prompt Pack（利于分层注入信息与迭代更新）。
- **收敛/自动化期**：切换到 Single Mega Prompt（单 cell，减少人为操作误差）。
- **需要证据链时**：用 Strategy-3 的 Research & Validation gating（先 EDA 再建模）。

### Step 2 — AI Worker 产出代码并本地运行
- 只接受“可直接运行”的 notebook cells / 单 cell。
- 运行后收集并保存：
  - overall CV MAE
  - folds MAE（如果有）
  - chosen weight / bag_list
  - shift_meta / bias_meta
  - submission sanity checks（shape/cols/NaNs/min/median/max）
  - wall time

### Step 3 — Human Manager paste-back（关键介入点）
严格按“paste-back 协议”把指标贴回给 AI（例如 CODE32 要求的那一组）。

> 原则：**paste-back 的信息越结构化、越稳定，下一轮 AI 的改进越可控。**

### Step 4 — 下一轮迭代的约束（控制变量）
- 每轮只做 1–3 个“高 ROI”改动（A/B test 思维）。
- 任何校正模块（如 bias correction）必须带 guardrail：**只有当 OOF 改善超过阈值才启用**（CODE32 bias_meta 的 applied=False 就是例子）。
- 避免“过搜/过调参”；优先可解释、可复现、可在 4060 上跑完的改动。

### Step 5 — 终止条件
- LB MAE 改善 < ε 连续 N 轮（例如 ε=0.05，N=3）；
- 或计算预算耗尽；
- 或达到论文需要的 “method + ablation evidence” 覆盖度。

---

# 5. 给后续更强 AI 的“二次分析任务清单”（你可以直接复制给它）

1) 从本文档抽象出一套“Human Manager Prompting Framework”：  
   - 角色定义、约束层、证据层、输出协议层、迭代协议层。  
2) 对 CODE25 的 pipeline 写一个 1-2 页的论文方法描述（含公式/伪代码）：  
   - AB blend + one-SE selection + bagging + multi-seed trimmed mean + chronic residual shift + bias guardrail。  
3) 生成 `mae_timeline.csv` 的标准字段定义，并给出如何自动解析 logs 的方法。  
4) 设计下一步冲击 447.x 的“微创但有效”改动清单（每项要说明：改动点、预期收益、风险、如何验证）。

---

# Appendix A — 完整 Mega Prompt（便于直接复用）
> 见 1.2 节的 `BEGIN PROMPT ... END PROMPT` 块。

# Appendix B — CODE32 paste-back（便于复用）
> 见 2.2 节。
