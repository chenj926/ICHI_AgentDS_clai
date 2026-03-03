# AgentDS Healthcare Challenge 2 — ED Cost Forecasting Insights

**Source PDF:** `IEEE ICHI - ED Cost Forecasting Insights.pdf`

**Pages:** 253  
**Extracted text chars:** 358256  
**Extraction timestamp:** 2026-02-20 04:59 UTC

> This Markdown is designed for downstream AI analysis and reproducibility.
> - It includes **verbatim page-by-page text** (Appendix).
> - It also includes **auto-extracted event chunks** (prompts, submissions, errors) to reduce navigation cost.
> - No content is intentionally truncated.

## Auto-extracted event chunks (for fast navigation)

### A. Explicit prompts: “Your Task” blocks

- Pages containing marker(s): 3, 139, 161

- Extracted page ranges (with context padding): 2–4, 138–140, 160–162



#### Chunk 1: Pages 2–4

```text

[Page 2]
Tabular Data (ed_cost_*.csv):
patient_id: Join key.
primary_chronic: {HF (Heart F ailure), DiabetesComp,  
Pneumonia}.
prior_ed_visits_5y: Integer count of visits in the prior 5  
years.
prior_ed_cost_5y_usd: T otal cost (USD) in the prior 5 years.
ed_cost_next3y_usd (T arget): The regression target for the  
next 3 years.
Note: There are other columns (like ZIP3) that need  
analysis.
Unstructured Data (receipt_<patient_id>.pdf):
These are synthetic "billing summaries."
Content: Line items containing CPT/HCPCS-like codes,  
quantities, and line totals.
Validation: The final "T OTAL" in the PDF matches the  
prior_ed_cost_5y_usd in the CSV.
Key Constraints & Environment:
Hardware: Local Windows environment with a single  
NVIDIA R TX 4060 (CUD A 12). The solution must be  
efficient enough to run here (e.g., careful selection of Local  
LLMs, efficient OCR/parsing, or feature extraction).
Search Constraint: Y ou should search for relevant papers  
and challenges (e.g., similar Kaggle competitions, papers  
on ED cost regression) to inform your strategy, but do not  
over-search. W e need practical, applicable strategies, not  
just a literature review.

[Page 3]
Your T ask:
Using your domain knowledge and the provided files (I will  
attach 3 sample PDFs, the description.md, and the CSV  
samples), please develop a comprehensive S trategy and  
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
NLP/R egex?
Correction of P ast Errors: Since we fixed the missing PDF  
issue, how do we best align the tabular features with the  
extracted PDF features to lower MAE?
Immediate Next S tep:
Please SEAR CH for similar problems/papers first, then

[Page 4]
THINK deeply about the medical/economic implications of  
the data columns and PDF contents. Finally, output a  
detailed S tep-by-S tep Master Plan.
(I have attached the dataset samples and description  
below)
What the lit eratur e says will actually mov e the needle (and why it
matt ers her e)
1. “Cost -on-cost” is usually the str ongest signal.
Across healthcare cost prediction work, prior spending/utilization is often the best
single predictor of future spending, and gradient boosting  tends to be a top-
performing workhorse on this kind of data. 
2. High-cost tails ar e syst ematically har d (classic underpr ediction pr oblem).
Traditional risk models (e.g., CMS-HC C–style risk adjustment) tend to overpredict low-
cost and underpr edict high-cost  patients—so if you want to win, you need explicit
“tail hygiene” (loss choice, calibration, segmentation, and features that identify
impending spikes). 
3. Area-lev el socioeconomic conditions corr elate with higher spending , and in
diabetes crises, social determinants strongly predict recurrence.
So your ZIP3 + insurance  combo is not “just demographic garnish”—it can be a real
driver proxy. 
These three points map perfectly to your setup: you already have strong cost-on-cost
features (prior 5y cost/visits), you fixed PDF completeness (so richer utilization phenotype
is now available), and you have geography/insurance signals in the receipts +
patients.csv . 
The “niche” clinical economics: why HF ≠ Diabet esComp ≠ Pneumonia
for futur e ED cost
Think in cost spik e mechanisms  (what creates a big future ED cost episode) rather than
“regr Heart F ailure (HF)
HF is a progressiv e, relapse-pr one condition. Costs jump when patients hit a worsening
HF ev ent (fluid overload, pulmonary edema, arrhythmia, cardiorenal syndrome), andPMC
MedP AC+1
JAMA Network +2
description
description

```

#### Chunk 2: Pages 138–140

```text

[Page 138]
What I need fr om y our next run (so w e can it erate scientifically)
Paste:
The per-fold log lines  (baseline_mae, model_mae, best_iter).
OOF MAE (raw)  and OOF MAE calibrated .
If any fold falls back to CPU (it will say so).
Once we have stable OOF ~450-ish without pathological spikes, the next gain (based on
your ED A) is a two-stage r esidual model  where stage-2 predicts residuals using only# Save outputs
oof_df = pd.DataFrame({ "patient_id" : train2[ "patient_id" ].astype( int), "y_true" : y
oof_df.to_csv(RUN_DIR / "oof_predictions.csv" , index= False)
sub = pd.DataFrame({ "patient_id" : test2[ "patient_id" ].astype( int), "ed_cost_next3y
sub.to_csv(RUN_DIR / "submission.csv" , index= False)
sub.to_csv(SUBMISSION_PATH, index= False)
run_log = {
    "run_id" : RUN_ID,
    "pipeline_version" : PIPELINE_VERSION,
    "xgboost_version" : getattr(xgb, "__version__" , "unknown" ),
    "use_cuda_requested" : WANT_CUDA,
    "n_splits" : N_SPLITS,
    "feature_matrix_shape" : [int(X_train.shape[ 0]), int(X_train.shape[ 1])],
    "feature_matrix_nnz" : int(X_train.nnz),
    "oof_mae_raw" : float(oof_mae),
    "oof_mae_calibrated" : float(oof_mae_cal) if USE_GROUP_MEDIAN_CALIBRATION else 
    "folds": fold_logs,
}
with open(RUN_DIR / "run_log.json" , "w", encoding= "utf-8") as f:
    json.dump(run_log, f, indent= 2)
log(f"Saved run artifacts to: {RUN_DIR} ")
log(f"Saved submission -> {SUBMISSION_PATH} ")

[Page 139]
receipt phenotype  (severity + complexity) — that’s usually the cleanest way to exploit those
high-lift codes without the “prior cost dominates everything” effect.
Sour ces
Sour ces
System Update: Phase 2 (Modeling V1) Execution  
Complete.
I have executed your v1 pipeline. Below are the raw logs  
from the local run and the actual feedback from the  
competition platform.
1. Internal Logs (Cross-V alidation):
Plaintext
[06:21:12] X_train shape=(2000, 65623), nnz=90215
[06:21:12] T arget stats: mean=3908.252, p50=3569.095 ...
[Fold 1] T rain-MAE: 191.44 | V al-MAE: 460.98
[Fold 2] T rain-MAE: 168.24 | V al-MAE: 440.29
[Fold 4] T rain-MAE: 163.64 | V al-MAE: 458.88
[06:24:10] OOF MAE (raw): 450.806
[06:24:10] OOF MAE (calibrated): 443.163
2. External R esult (Leaderboard Submission):
Plaintext
🚀 Submitting predictions...
✅ Submission successful!
📊 Score: 463.7263 (MAE)
Your T ask:
As the Lead Scientist, review these results.
Diagnosis: Compare the Internal CV (443), External LB  
(463), and the T rain/V al curves in the logs. What does the  
X_train shape combined with the T rain vs V al gap tell you

[Page 140]
about our current model's behavior?
Strategize V2: Based on your diagnosis, what is the single  
most critical change we need to make to the pipeline to  
close the Generalization Gap?
Output:
Analysis: The "Why".
The Plan: The precise engineering steps for V2.
The Code: The updated v2 code
Analysis — the “Why”
Your numbers are the classic signature of covariate shif t + high-cap acity sp arse featur es:
Internal CV (calibrat ed) 443 vs LB 463 (~+20 MAE)  means your CV is optimistic
relative to the hidden test distribution. That usually happens when train folds look
like each other , but test looks meaningfully differ ent in feature distribution. This is
precisely the covariate shift setting:  while we hope  is
stable. 
X_train shape (2000, 65623)  with only ~90k non-zeros means you have huge featur e
space dominat ed by hashed code t okens. The model can “memorize” idiosyncratic
code patterns and interactions in the training distribution. That’s why Train-MAE
~160–190  while Val-MAE ~440–460 : high variance/high capacity. When test has a
shifted mix of codes/complexity patterns, that memorization doesn’t transfer.
Your receipt ED A already hinted the structure: “severity phenotype” codes exist, but
the long-tail code mix is complex. Hash features help CV, but they also amplif y shif t
sensitivity .
The single most critical change t o close the gap
Make training shif t-awar e using adv ersarial v alidation → impor tance w eighting.
Mechanism:p  (x)= train p  (x) test p(y∣x)
Journal of Machi …+1

```

#### Chunk 3: Pages 160–162

```text

[Page 160]
What t o paste back (so w e can it erate like scientists)
1. Adversarial AUC
2. OOF MAE (unweighted)  and OOF MAE (importance-weighted proxy)
3. Whether calibration A CCEPTED  or REJECTED
4. The platform score for this submission
If the weight ed pr oxy tracks the LB tightly (even if unweighted CV doesn’t), then we’ve
successfully aligned the training objective to the actual hidden test distribution — that’s
the fastest route to closing the 20-MAE gap.    adv_auc= float(adv_auc),
    weight_stats=sanity_stats(w_train),
    feature_shape=[ int(X_train.shape[ 0]), int(X_train.shape[ 1])],
    feature_nnz= int(X_train.nnz),
    oof_mae= float(oof_mae),
    oof_wmae= float(oof_wmae),
    calibration_used= bool(cal_used),
    calibration_info=cal_info,
    folds=fold_logs,
)
with open(RUN_DIR/ "run_log.json" , "w", encoding= "utf-8") as f:
    json.dump(run_log, f, indent= 2)
log(f"Saved run artifacts to: {RUN_DIR} ")
log(f"Saved submission -> {SUBMISSION_PATH} ")
Sour ces
Sour ces
System Update: Phase 2 R esults & Diagnosis
I have executed your V2 pipeline (Adversarial W eighting).  
The results are conclusive, but they disprove our previous  
theory.

[Page 161]
The Hard Evidence:
Adversarial V alidation A UC = 0.5274: This is effectively  
random.
Critical Deduction: There is NO significant covariate shift  
between T rain and T est. The "Adversarial W eighting"  
strategy was mathematically unnecessary and did not help  
(LB stayed flat at ~462).
Persistent Overfitting:
Train MAE: ~230
Validation MAE: ~445
Gap: >200 points. The model is memorizing specific  
patient noise rather than learning generalizable rules.
The Benchmark Gap: The known state-of-the-art for this  
specific challenge is MAE ~400. W e are stuck at 460. W e 
are missing a massive structural component of the  
problem.
Your T ask: The S trategic Pivot
We are hitting a wall. S tandard X GBoost regression on  
these features is capping out. As the Domain Expert, you  
need to re-formulate the problem.
Think about these angles (but do not be limited by them):
The "Zero" Problem: Are we forcing a regressor to predict  
continuous values for patients who might have $0 cost?
Target Definitions: W e are predicting T otal Cost. Is this the  
most stable target? Or should we be modeling Cost =  
(Probability of Visit) * (Cost per Visit)?
Interaction F ailures: W e have primary_chronic and we have

[Page 162]
receipt_codes. Are we failing to capture how they amplify  
each other? (e.g., A generic "Chest X-Ray" might be cheap  
for Pneumonia but a signal of impending doom for Heart  
Failure).
Action R equired:
Diagnose the S tagnation: Why did V2 fail to close the  
generalization gap despite the lack of distribution shift?
Propose the "Phase 3" Architecture: Don't just tweak  
parameters. Propose a fundamental change in how we  
model the target or engineer the features to hit that 400  
MAE benchmark.
Generate v3: The new implementation.
Here is also the real log for refernece:
[06:41:14] xgboost version: 3.0.0
[06:41:14] PIPELINE_VERSION:  
EDCOST_V2_adv_weighting_2026-02-14
[06:41:15] Building features (cache miss/version change)...
[06:41:15] Saved feature cache:  
D:\AgentDs\agent_ds_healthcare\cache_iter10\feat_matrix_
cache_v2.joblib
[06:41:15] X_train shape=(2000, 65623), nnz=90215
[06:41:15] T arget: mean=3908.252, p50=3569.095,  
p95=7541.847, max=11184.610
[06:41:15] T raining adversarial validator (train vs test)...
[06:42:01] Saved adversarial cache:  
D:\AgentDs\agent_ds_healthcare\cache_iter10\adv_scores_
cache_v2.joblib
[06:42:01] Adversarial A UC (train vs test) = 0.5274  (0.5=no  
shift; higher=more shift)
[06:42:01] Importance weights stats: {'min':  
0.2361346185207367, 'max': 3.778153896331787, 'mean':  
0.999999995559454, 'p01': 0.2739429384469986, 'p50':  
0.9470024108886719, 'p99': 2.8402921867370603}
[06:42:01] S tarting shift-aware CV training...
[0] validation_0-mae:1413.43999 validation_1-

```



### B. Submissions & leaderboard outputs

- Pages containing marker(s): 139, 183, 184

- Extracted page ranges (with context padding): 138–140, 182–185



#### Chunk 1: Pages 138–140

```text

[Page 138]
What I need fr om y our next run (so w e can it erate scientifically)
Paste:
The per-fold log lines  (baseline_mae, model_mae, best_iter).
OOF MAE (raw)  and OOF MAE calibrated .
If any fold falls back to CPU (it will say so).
Once we have stable OOF ~450-ish without pathological spikes, the next gain (based on
your ED A) is a two-stage r esidual model  where stage-2 predicts residuals using only# Save outputs
oof_df = pd.DataFrame({ "patient_id" : train2[ "patient_id" ].astype( int), "y_true" : y
oof_df.to_csv(RUN_DIR / "oof_predictions.csv" , index= False)
sub = pd.DataFrame({ "patient_id" : test2[ "patient_id" ].astype( int), "ed_cost_next3y
sub.to_csv(RUN_DIR / "submission.csv" , index= False)
sub.to_csv(SUBMISSION_PATH, index= False)
run_log = {
    "run_id" : RUN_ID,
    "pipeline_version" : PIPELINE_VERSION,
    "xgboost_version" : getattr(xgb, "__version__" , "unknown" ),
    "use_cuda_requested" : WANT_CUDA,
    "n_splits" : N_SPLITS,
    "feature_matrix_shape" : [int(X_train.shape[ 0]), int(X_train.shape[ 1])],
    "feature_matrix_nnz" : int(X_train.nnz),
    "oof_mae_raw" : float(oof_mae),
    "oof_mae_calibrated" : float(oof_mae_cal) if USE_GROUP_MEDIAN_CALIBRATION else 
    "folds": fold_logs,
}
with open(RUN_DIR / "run_log.json" , "w", encoding= "utf-8") as f:
    json.dump(run_log, f, indent= 2)
log(f"Saved run artifacts to: {RUN_DIR} ")
log(f"Saved submission -> {SUBMISSION_PATH} ")

[Page 139]
receipt phenotype  (severity + complexity) — that’s usually the cleanest way to exploit those
high-lift codes without the “prior cost dominates everything” effect.
Sour ces
Sour ces
System Update: Phase 2 (Modeling V1) Execution  
Complete.
I have executed your v1 pipeline. Below are the raw logs  
from the local run and the actual feedback from the  
competition platform.
1. Internal Logs (Cross-V alidation):
Plaintext
[06:21:12] X_train shape=(2000, 65623), nnz=90215
[06:21:12] T arget stats: mean=3908.252, p50=3569.095 ...
[Fold 1] T rain-MAE: 191.44 | V al-MAE: 460.98
[Fold 2] T rain-MAE: 168.24 | V al-MAE: 440.29
[Fold 4] T rain-MAE: 163.64 | V al-MAE: 458.88
[06:24:10] OOF MAE (raw): 450.806
[06:24:10] OOF MAE (calibrated): 443.163
2. External R esult (Leaderboard Submission):
Plaintext
🚀 Submitting predictions...
✅ Submission successful!
📊 Score: 463.7263 (MAE)
Your T ask:
As the Lead Scientist, review these results.
Diagnosis: Compare the Internal CV (443), External LB  
(463), and the T rain/V al curves in the logs. What does the  
X_train shape combined with the T rain vs V al gap tell you

[Page 140]
about our current model's behavior?
Strategize V2: Based on your diagnosis, what is the single  
most critical change we need to make to the pipeline to  
close the Generalization Gap?
Output:
Analysis: The "Why".
The Plan: The precise engineering steps for V2.
The Code: The updated v2 code
Analysis — the “Why”
Your numbers are the classic signature of covariate shif t + high-cap acity sp arse featur es:
Internal CV (calibrat ed) 443 vs LB 463 (~+20 MAE)  means your CV is optimistic
relative to the hidden test distribution. That usually happens when train folds look
like each other , but test looks meaningfully differ ent in feature distribution. This is
precisely the covariate shift setting:  while we hope  is
stable. 
X_train shape (2000, 65623)  with only ~90k non-zeros means you have huge featur e
space dominat ed by hashed code t okens. The model can “memorize” idiosyncratic
code patterns and interactions in the training distribution. That’s why Train-MAE
~160–190  while Val-MAE ~440–460 : high variance/high capacity. When test has a
shifted mix of codes/complexity patterns, that memorization doesn’t transfer.
Your receipt ED A already hinted the structure: “severity phenotype” codes exist, but
the long-tail code mix is complex. Hash features help CV, but they also amplif y shif t
sensitivity .
The single most critical change t o close the gap
Make training shif t-awar e using adv ersarial v alidation → impor tance w eighting.
Mechanism:p  (x)= train p  (x) test p(y∣x)
Journal of Machi …+1

```

#### Chunk 2: Pages 182–185

```text

[Page 182]
1. OOF MAE baseline-only  and OOF MAE V3  (and the delta)
2. Fold logs (baseline_MAE vs v3_MAE + best_iter)
3. The new LB score
If V3 is doing what it’s designed to do, you should see:
baseline-only MAE is “okay”
V3 improves baseline consistently across folds
variance across folds narrows (less 409 ↔ 483 chaos)
LB should move meaningfully toward ~440–420 territory; then we can iterate the last
~20–40 points.
If V3 still plateaus, the next escalation is true fr equency–sev erity modeling  (a two-part /
Tweedie-style claim framework) — the standard approach for nonnegative cost outcomes
in healthcare/insurance. Scikit-learn +1
Sour ces
Sour ces
[14:54:21] xgboost version: 3.0.0
[14:54:21] S tarting V3 CV: baseline + residual | n=2000
[0] validation_0-mae:483.80710 validation_1-
mae:542.05221
[200] validation_0-mae:407.12944 validation_1-
mae:533.40803
[347] validation_0-mae:383.98902 validation_1-
mae:536.64686
[14:54:24] [Fold 1/5] mode=cuda | baseline_MAE=542.87 |  
v3_MAE=529.15 | improve=13.72 | best_iter=98 |  
code_effects=18 | time=2.5s
[0] validation_0-mae:485.09723 validation_1-
mae:524.23722
[200] validation_0-mae:406.61182 validation_1-
mae:515.21914
[277] validation_0-mae:393.88378 validation_1-
mae:516.37441
[14:54:25] [Fold 2/5] mode=cuda | baseline_MAE=525.42 |  
v3_MAE=511.84 | improve=13.58 | best_iter=28 |  
code_effects=18 | time=0.9s
[0] validation_0-mae:488.53184 validation_1-

[Page 183]
mae:508.19811
[200] validation_0-mae:406.09336 validation_1-
mae:503.97365
[294] validation_0-mae:390.67594 validation_1-
mae:505.94988
[14:54:26] [Fold 3/5] mode=cuda | baseline_MAE=509.11 |  
v3_MAE=499.59 | improve=9.52 | best_iter=45 |  
code_effects=18 | time=0.9s
[0] validation_0-mae:485.81878 validation_1-
mae:550.67443
[200] validation_0-mae:404.42422 validation_1-
mae:539.12232
[341] validation_0-mae:383.61780 validation_1-
mae:542.71394
[14:54:27] [Fold 4/5] mode=cuda | baseline_MAE=552.22 |  
v3_MAE=534.94 | improve=17.28 | best_iter=91 |  
code_effects=18 | time=1.0s
[0] validation_0-mae:478.27440 validation_1-
mae:549.30858
[200] validation_0-mae:398.56244 validation_1-
mae:525.69052
[377] validation_0-mae:373.63341 validation_1-
mae:526.59098
[14:54:28] [Fold 5/5] mode=cuda | baseline_MAE=551.50 |  
v3_MAE=524.53 | improve=26.98 | best_iter=128 |  
code_effects=18 | time=1.1s
[14:54:28] OOF MAE baseline-only: 536.227467
[14:54:28] OOF MAE V3 (baseline+residual): 520.010242   
(delta +16.217226)
[14:54:28] Fitting final baseline on full train...
[14:54:28] Final fit n_estimators = 291 (median_best_iter +  
200)
[14:54:28] Saved run artifacts:  
D:\AgentDs\agent_ds_healthcare\cache_iter10\run_v3_202
60214_145421
[14:54:28] Saved submission ->  
D:\AgentDs\agent_ds_healthcare\submission_ICHI_V1.csv
🚀 Submitting predictions...
✅ Prediction submitted successfully!

[Page 184]
📊 Score: 546.2783 (MAE)
✅ Validation passed
✅ Submission successful!
   📊 Score: 546.2783
   📏 Metric: MAE
   ✔  Validation: P assed
🎯 Next steps:
   1. Try incorporating relevant information outside this  
table!
   2. Move on to Healthcare Challenge 3!
怎么train 和 real 都是 500 多了 ??
你这个  V3 变成  500+ （ OOF 520 / LB 546 ）不是 “ 数据突然变难了 ”，⽽是  我们把模型的主
信号砍掉了 ，导致系统性退化。
为什么会  500+ （根因）
1. V3 的  baseline 设计太弱
你⽤的是  primary_chronic × insurance × prior_cost_decile  的分组中位数当
baseline。它丢掉了最强的连续信号（ prior_ed_cost_5y_usd  的连续关系、
prior_ed_visits_5y 、以及  receipt composition 的细粒度） ， baseline 本⾝就  536
MAE，已经⽐你  V1/V2 的  443–452 差⼀⼤截。
2. V3 的 “code effect embedding” ⼏乎没学到东西
每折只⽣成  18 个  code_effects ，等于残差模型只能靠很少的离散效果做修正，修不回
baseline 的巨⼤误差  → 结果⾃然整体  500+ 。
3. V2 失败不是 “shif t” 问题
你的 adversarial A UC=0.527 （ ≈ 随机），说明  train/test 没有明显  covariate shif t，所以
加权不该有收益。我们真正的问题是： 模型缺少关键结构性信息  + ⾼维表⽰导致⾼⽅
差（Train MAE 很低、 Val MAE 卡在  440–480 之间，且折间波动⼤） 。
Phase 3 真正的战略转向（能解释为什么  SOTA ~400 ）结论： V3 的失败是 “ 架构砍主信号  + 残差修复能⼒不⾜ ” ，不是平台或数据异常。

[Page 185]
你之前只⽤了  ed_cost_*.csv + patients.csv + receipt 。但 description.md 明确说明 还
有能⽤来提取患者严重程度 / 共病 / ⽣理不稳定性的⽂件：
admissions_* + discharge_notes.json + stays_* + vitals_timeseries.json （都可⽤
patient_id  join）。这类 “ 跨表病程 / 住院严重度 / ⽣命体征稳定性 ” 特征，往往是医疗成本预
测⾥最能拉开差距的结构信号；⽽ “ 两段式 / 频率 - 严重度 ” 等成本建模思路也正是健康经济学
⾥常⻅路线。
单⼀最关键改动：把  Challenge 1/3 的信息做成  patient -level 聚合特征，加⼊  ED cost 回
归。
这就是你现在离  ~400 还差  60+ 的 “ 结构性组件 ” 。
V3 ⼯程计划（精确步骤）
1. 构建 patient -level 辅助特征 （不泄漏  ED ⽬标）
admissions （ train+test ）：次数、 LOS 统计、 Charlson band 统计、 acuity_emergent ⽐
例、ed_visits_6m 均值、 primary_dx 分布
discharge_notes ：按  patient 拼接⽂本  → TF-IDF → SVD(32 维 )
stays（ train+test ）：次数、 unit_type 分布、 admission_reason 分布
vitals_timeseries ：按  stay 汇总  HR/SBP/DBP/T emp/RR 的  mean/std/min/max/slope →
再按 patient 聚合（ mean/max ）
vitals notes ：按  patient 拼接  → TF-IDF → SVD(16 维 )
2. receipt 特征继续⽤你  EDA 证明有效的：⾼  lift code + 复杂度 / 分散度（不再⽤  65k
hashing；改成 “Top-K code count/share” 低维表⽰）
3. 回到 MAE 对⻬⽬标： XGBoost reg:absoluteerror （这是最贴合  metric 的⽬标之⼀）
并保留你在  V1 证实有效的  分组残差校准 （primary_chronic × prior_cost_decile ）。
V3 新实现（ 1-cell ，可直接跑）PMC +1
XGBoost Docum …
说明：这版会⾃动检查  admissions/stays/vitals/notes ⽂件是否存在；存在就⽤，不存在
就跳过（鲁棒） 。
缓存：会把 “patient-level 辅助特征 ” 缓存成  joblib ，⼆次运⾏很快。
python

```



### C. Errors / tracebacks

- Pages containing marker(s): 34, 97, 101, 102, 103, 112, 114, 115, 116, 126, 133, 147, 153, 175, 177, 189, 198, 221, 236, 237, 238

- Extracted page ranges (with context padding): 34, 97, 101–103, 112, 114–116, 126, 133, 147, 153, 175, 177, 189, 198, 221, 236–238



#### Chunk 1: Pages 34–34

```text

[Page 34]
What I want y ou to send b ack af ter you run it (Phase 2)
Just paste:
The printed summary sections (target quantiles, chronic summary, PDF match rates).
The top 15 r ows of spearman_receipt_features.csv
The top 20 codes  from code_lift_table.csv  (residual-lift sorted)
Any surprising plots (or tell me which plot filenames look odd)
Then I’ll refine the feature hypotheses (what to keep, what to drop, what to stratify) before
we touch any model.if __name__ == "__main__" :
    main()
Sour ces
Sour ces
got bug:
-----------------------------------------------------------------
----------
AttributeError                            T raceback (most recent call  
last)
Cell In[5], line 715
    711     print(" - tail comparison:  
tail_feature_comparison.csv")
    714 if __name__ == "__main__":
--> 715     main()
Cell In[5], line 529, in main()
    527 rec_rows = []
    528 for pid, obj in parsed.items():
--> 529     feats = build_receipt_features(obj)
    530     feats["patient_id"] = int(pid)
    531     rec_rows.append(feats)
Cell In[5], line 246, in build_receipt_features(parsed)
    245 def build_receipt_features(parsed: dict):
--> 246     items = parsed.get("items", [])

```

#### Chunk 2: Pages 97–97

```text

[Page 97]
    num_cols = base_num + receipt_num + curated_feats
    # Fill numeric NaNs
    for df in [train_df, test_df]:
        for c in num_cols:
            if c not in df.columns:
                df[c] = np.nan
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors= "coerce" ).fillna(
    # One-hot for categoricals (fit on combined for consistent columns)
    all_cat = pd.concat([train_df[cat_cols], test_df[cat_cols]], axis= 0, ignore_i
    try:
        ohe = OneHotEncoder(handle_unknown= "ignore" , sparse_output= True)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown= "ignore" , sparse= True)
    X_cat_all = ohe.fit_transform(all_cat)
    # Dense numeric -> sparse
    X_num_all = sp.csr_matrix(pd.concat([train_df[num_cols], test_df[num_cols]], 
    # Hash codes
    if USE_HASH_CODES:
        hasher = FeatureHasher(n_features=HASH_DIM, input_type= "dict")
        X_hash_all = hasher.transform(hash_dicts)
        X_all = sp.hstack([X_num_all, X_cat_all, X_hash_all], format="csr")
    else:
        X_all = sp.hstack([X_num_all, X_cat_all], format="csr")
    X_train = X_all[: len(train_df)]
    X_test  = X_all[ len(train_df):]
    return train_df, test_df, X_train, X_test, ohe
# -----------------------------
# CV + training
# -----------------------------
def make_strat_labels (train_df: pd.DataFrame, n_bins: int = 10) -> np.ndarray:

```

#### Chunk 3: Pages 101–103

```text

[Page 101]
What t o do af ter you run it
1. Paste the printed OOF MAE raw vs calibrat ed.
2. Tell me whether pseudohuber blend  improved or worsened OOF MAE (you can flip
USE_PSEUDOHUBER_BLEND ).
If calibration helps but not enough, the next step is a 2-stage r esidual model  where
Stage-2 predicts the residual  using receipt-heavy features (especially the severe procedure
+ critical care patterns). That’s often a clean MAE gain in this exact structure.    mae_raw = mean_absolute_error(y, oof)
    mae_cal = mean_absolute_error(y, oof_cal)
    print(f"OOF MAE raw: {mae_raw: .6f} | calibrated: {mae_cal: .6f}")
    test_pred_final = test_cal
else:
    test_pred_final = test_pred
# Final post-processing
test_pred_final = np.clip(test_pred_final, 0.0, None)
# Write submission
sub = pd.DataFrame({
    "patient_id" : test2[ "patient_id" ].astype( int).values,
    "ed_cost_next3y_usd" : test_pred_final
})
sub.to_csv(SUBMISSION_PATH, index= False)
print(f"\nSaved submission -> {SUBMISSION_PATH} ")
print(f"Done in {time.time()-t0: .1f}s")
Sour ces
Sour ces
-----------------------------------------------------------------
----------
TypeError                                 T raceback (most recent call  
last)
Cell In[13], line 589
    586 y = train2["ed_cost_next3y_usd"].astype(float).values

[Page 102]
    588 print("\nT raining CV...")
--> 589 oof, test_pred = run_cv_training(train2, X_train, y,  
test2, X_test)
    591 # Optional calibration (group median residual shift)
    592 if USE_GR OUP_MEDIAN_CALIBRA TION:
Cell In[13], line 528, in run_cv_training(train_df, X_train, y,  
test_df, X_test)
    525 y_tr, y_va = y[tr_idx], y[va_idx]
    527 # Model A: L1/MAE-aligned objective
--> 528 m_abs = fit_one_model(X_tr, y_tr, X_va, y_va,  
objective="reg:absoluteerror", seed=SEED + fold)
    530 pred_va_abs = m_abs.predict(X_va)
    531 pred_te_abs = m_abs.predict(X_test)
Cell In[13], line 488, in fit_one_model(X_train, y_train,  
X_valid, y_valid, objective, seed)
    486 params["random_state"] = seed
    487 model = xgb.X GBRegressor(**params)
--> 488 model.fit(
    489     X_train, y_train,
    490     eval_set=[(X_valid, y_valid)],
    491     eval_metric="mae",
    492     verbose=VERBOSE_EV AL,
    493     early_stopping_rounds=EARL Y_STOP,
    494 )
    495 return model
File 
c:\Users\shend\AppData\Local\Programs\Python\Python3
12\Lib\site-packages\xgboost\core.py:729, in  
require_keyword_args.<locals>.throw_if.
<locals>.inner_f(*args, **kwargs)
    727 for k, arg in zip(sig.parameters, args):
    728     kwargs[k] = arg
--> 729 return func(**kwargs)
TypeError: X GBModel.fit() got an unexpected keyword  
argument 'eval_metric'

[Page 103]
Help me fix the bug, and make sure the new code is robust
That error happens because your installed X GBoost sklearn API does not accept
eval_metric  as a fit()  argument  anymore (it’s been moved to the construct or /
set_params ). XGBoost’s own docs show eval_metric  passed when creating
XGBRegressor , and fit()  called with eval_set  (no eval_metric  kwarg). 
Also, for GPU in X GBoost ≥2.x/3.x the recommended way is tree_method="hist",
device="cuda"  (with fallback for older installs). 
Below is a robust, 1-cell  replacement that:
fixes the eval_metric  bug (sets it in the constructor),
uses the modern GPU parameters (with fallback),
uses a safe_fit()  wrapper that only passes args your fit()  actually supports,
uses predict_best()  so early stopping works across versions.
Copy/paste into one Jupy ter cell  and run.XGBoost Docum …+1
XGBoost Docum …
python
# ============================================================
# AgentDS Challenge 2 — TRAINING PIPELINE (robust XGBoost sklearn API)
# Fixes: eval_metric in fit() TypeError, GPU param compatibility, early-stop pred
# ============================================================
import os, re, time, math, warnings, inspect
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import load, dump
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import StratifiedKFold

```

#### Chunk 4: Pages 112–112

```text

[Page 112]
    receipt_num = [
        "n_items" ,"n_unique_codes" ,"top1_share" ,"top3_share" ,"distributed_cost" ,
        "hhi","gini","bucket_entropy" ,
        "n_em_9928x" ,"max_em_level_9928x" ,
        "has_critical_care" ,"n_99291" ,"n_99292" ,"cc_time_proxy" ,
        "n_severe_proc" ,"severe_proc_share" ,
        "n_hi_diag" ,"hi_diag_share" ,
        "severity_score" ,
    ]
    curated_feats = []
    for c in CURATED_CODES:
        curated_feats += [ f"has_{c}", f"cnt_{c}", f"share_ {c}"]
    num_cols = base_num + receipt_num + curated_feats
    for df in [train_df, test_df]:
        for c in num_cols:
            if c not in df.columns:
                df[c] = 0.0
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors= "coerce" ).fillna(
    all_cat = pd.concat([train_df[cat_cols], test_df[cat_cols]], axis= 0, ignore_i
    try:
        ohe = OneHotEncoder(handle_unknown= "ignore" , sparse_output= True)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown= "ignore" , sparse= True)
    X_cat_all = ohe.fit_transform(all_cat)
    X_num_all = sp.csr_matrix(pd.concat([train_df[num_cols], test_df[num_cols]], 
    if USE_HASH_CODES:
        hasher = FeatureHasher(n_features=HASH_DIM, input_type= "dict")
        X_hash_all = hasher.transform(hash_dicts)
        X_all = sp.hstack([X_num_all, X_cat_all, X_hash_all], format="csr")
    else:
        X_all = sp.hstack([X_num_all, X_cat_all], format="csr")

```

#### Chunk 5: Pages 114–116

```text

[Page 114]
    # remove predictor to avoid API churn (some versions removed it)
    base.pop( "predictor" , None)
    if want_cuda:
        # Preferred modern API
        p1 = dict(base, tree_method= "hist", device= "cuda")
        try:
            _ = xgb.XGBRegressor(**p1)
            return p1, "GPU via device='cuda', tree_method='hist'"
        except TypeError:
            pass
        # Legacy fallback
        p2 = dict(base, tree_method= "gpu_hist" )
        try:
            _ = xgb.XGBRegressor(**p2)
            return p2, "GPU via tree_method='gpu_hist' (legacy)"
        except TypeError:
            pass
    # CPU fallback
    p3 = dict(base, tree_method= "hist")
    p3.pop( "device" , None)
    return p3, "CPU via tree_method='hist'"
def safe_fit (model, X_tr, y_tr, X_va, y_va, early_stopping_rounds= 200, verbose= 20
    """
    Pass only supported kwargs to model.fit() to avoid version-specific TypeError
    """
    sig = inspect.signature(model.fit)
    kwargs = {}
    if "eval_set"  in sig.parameters:
        kwargs[ "eval_set" ] = [(X_va, y_va)]
    if "verbose"  in sig.parameters:
        kwargs[ "verbose" ] = verbose
    elif "verbose_eval"  in sig.parameters:
        kwargs[ "verbose_eval" ] = verbose

[Page 115]
    if "early_stopping_rounds"  in sig.parameters and early_stopping_rounds is not 
        kwargs[ "early_stopping_rounds" ] = int(early_stopping_rounds)
    elif "callbacks"  in sig.parameters and early_stopping_rounds is not None:
        # callback-based early stopping as fallback
        try:
            from xgboost.callback import EarlyStopping
            kwargs[ "callbacks" ] = [EarlyStopping(rounds= int(early_stopping_rounds
        except Exception:
            pass
    try:
        model.fit(X_tr, y_tr, **kwargs)
        return model
    except TypeError as e:
        # last-resort: drop args one-by-one
        msg = str(e)
        for k in list(kwargs.keys()):
            try:
                kw2 = dict(kwargs)
                kw2.pop(k, None)
                model.fit(X_tr, y_tr, **kw2)
                print(f"[safe_fit] fit() succeeded after dropping ' {k}' (original 
                return model
            except TypeError:
                continue
        raise
def predict_best (model, X ):
    """
    Robust prediction for early-stopped models across xgboost versions.
    """
    # Newer versions: iteration_range
    if hasattr(model, "best_iteration" ) and model.best_iteration is not None:
        bi = int(model.best_iteration)
        try:
            return model.predict(X, iteration_range=( 0, bi + 1))
        except TypeError:
            pass
    # Older versions: best_ntree_limit
    if hasattr(model, "best_ntree_limit" ):

[Page 116]
        try:
            return model.predict(X, ntree_limit=model.best_ntree_limit)
        except TypeError:
            pass
    return model.predict(X)
def fit_one_model (X_tr, y_tr, X_va, y_va, objective: str, seed: int, want_cuda: b
    # Base params (we set eval_metric in constructor, NOT in fit)  :contentReferen
    base = dict(
        n_estimators= 8000,
        learning_rate= 0.03,
        max_depth= 6,
        min_child_weight= 10,
        subsample= 0.8,
        colsample_bytree= 0.8,
        reg_lambda= 1.0,
        reg_alpha= 0.0,
        random_state=seed,
        n_jobs=- 1,
        verbosity= 1,
        objective=objective,
        eval_metric= "mae",   # <-- FIX: constructor param, not fit()
    )
    params, mode = resolve_xgb_params(base, want_cuda=want_cuda)
    # print(mode)  # uncomment if you want per-model verbosity
    model = xgb.XGBRegressor(**params)
    # try GPU/selected mode; if XGBoostError occurs, fall back to CPU once
    try:
        model = safe_fit(model, X_tr, y_tr, X_va, y_va, early_stopping_rounds=EAR
        return model, mode
    except xgb.core.XGBoostError as e:
        print(f"[WARN] Training failed in mode ' {mode}'. Falling back to CPU. Err
        params_cpu, mode_cpu = resolve_xgb_params(base, want_cuda= False)
        model2 = xgb.XGBRegressor(**params_cpu)
        model2 = safe_fit(model2, X_tr, y_tr, X_va, y_va, early_stopping_rounds=EA
        return model2, mode_cpu

```

#### Chunk 6: Pages 126–126

```text

[Page 126]
def entropy_from_counts (counts: Counter ) -> float:
    if not counts:
        return 0.0
    vals = np.array( list(counts.values()), dtype= float)
    p = vals / (vals. sum() + 1e-12)
    return float(-(p * np.log(p + 1e-12)).sum())
def gini_from_amounts (x: np.ndarray ) -> float:
    x = x.astype( float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return 0.0
    if np.all(x == 0):
        return 0.0
    x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(x)
    g = (n + 1 - 2 * np.sum(cumx) / (cumx[- 1] + 1e-12)) / n
    return float(g)
def build_prior_bins (prior: pd.Series, n_bins: int = 10):
    q = np.quantile(prior.values.astype( float), np.linspace( 0, 1, n_bins + 1))
    bins = np.unique(q)
    if len(bins) < 3:
        bins = np.unique(np.quantile(prior.values.astype( float), [0, 0.5, 1.0]))
    return bins
def make_strat_labels (df: pd.DataFrame, n_bins: int = 10) -> np.ndarray:
    qbin = pd.qcut(df[ "prior_ed_cost_5y_usd" ].astype( float), q=n_bins, duplicates
    return (df["primary_chronic" ].astype( str) + "|" + qbin).values
def predict_best (model, X ):
    bi = getattr(model, "best_iteration" , None)
    if bi is not None:
        try:
            return model.predict(X, iteration_range=( 0, int(bi) + 1))
        except TypeError:
            pass
    return model.predict(X)

```

#### Chunk 7: Pages 133–133

```text

[Page 133]
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors= "coerce" ).fillna(
    all_cat = pd.concat([train_df[cat_cols], test_df[cat_cols]], axis= 0, ignore_i
    try:
        ohe = OneHotEncoder(handle_unknown= "ignore" , sparse_output= True)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown= "ignore" , sparse= True)
    X_cat_all = ohe.fit_transform(all_cat)
    X_num_all = sp.csr_matrix(pd.concat([train_df[num_cols], test_df[num_cols]], 
    if USE_HASH_CODES:
        hasher = FeatureHasher(n_features=HASH_DIM, input_type= "dict")
        X_hash_all = hasher.transform(hash_dicts)
        X_all = sp.hstack([X_num_all, X_cat_all, X_hash_all], format="csr")
    else:
        X_all = sp.hstack([X_num_all, X_cat_all], format="csr")
    X_train = X_all[: len(train_df)]
    X_test  = X_all[ len(train_df):]
    return train_df, test_df, X_train, X_test
# -----------------------------
# XGB constructor params (XGBoost 3 requires early_stopping_rounds in ctor)
# -----------------------------
def make_xgb_regressor (seed: int, use_cuda: bool = True):
    params = dict(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        min_child_weight=MIN_CHILD_WEIGHT,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE,
        reg_lambda=REG_LAMBDA,
        reg_alpha=REG_ALPHA,
        random_state=seed,
        n_jobs=- 1,
        verbosity= 1,
        objective= "reg:absoluteerror" ,

```

#### Chunk 8: Pages 147–147

```text

[Page 147]
    cdf = np.cumsum(w)
    cutoff = 0.5 * np.sum(w)
    j = int(np.searchsorted(cdf, cutoff, side= "left"))
    j = min(max(j, 0), len(v)-1)
    return float(v[j])
def predict_best (model, X ):
    bi = getattr(model, "best_iteration" , None)
    if bi is not None:
        try:
            return model.predict(X, iteration_range=( 0, int(bi) + 1))
        except TypeError:
            pass
    return model.predict(X)
def predict_best_proba (model, X ):
    bi = getattr(model, "best_iteration" , None)
    if bi is not None:
        try:
            return model.predict_proba(X, iteration_range=( 0, int(bi) + 1))[:, 1]
        except TypeError:
            pass
    return model.predict_proba(X)[:, 1]
def safe_fit (model, X_tr, y_tr, eval_set= None, sample_weight= None, sample_weight_
    sig = inspect.signature(model.fit)
    kwargs = {}
    if "eval_set"  in sig.parameters and eval_set is not None:
        kwargs[ "eval_set" ] = eval_set
    if "sample_weight"  in sig.parameters and sample_weight is not None:
        kwargs[ "sample_weight" ] = sample_weight
    if "sample_weight_eval_set"  in sig.parameters and sample_weight_eval_set is n
        kwargs[ "sample_weight_eval_set" ] = sample_weight_eval_set
    if "verbose"  in sig.parameters:
        kwargs[ "verbose" ] = verbose
    model.fit(X_tr, y_tr, **kwargs)
    return model
def sanity_stats (arr):
    arr = np.asarray(arr, dtype= float)

```

#### Chunk 9: Pages 153–153

```text

[Page 153]
        "prior_ed_cost_5y_usd" ,
        "prior_cost_per_year" ,
        "prior_visits_per_year" ,
        "prior_cost_per_visit" ,
    ]
    receipt_num = [
        "n_items" ,"n_unique_codes" ,"top1_share" ,"top3_share" ,"distributed_cost" ,
        "hhi","gini","bucket_entropy" ,
        "n_em_9928x" ,"max_em_level_9928x" ,
        "has_critical_care" ,"n_99291" ,"n_99292" ,"cc_time_proxy" ,
        "n_severe_proc" ,"severe_proc_share" ,
        "n_hi_diag" ,"hi_diag_share" ,
        "severity_score" ,
    ]
    curated = []
    for c in CURATED_CODES:
        curated += [ f"has_{c}", f"cnt_{c}", f"share_ {c}"]
    num_cols = base_num + receipt_num + curated
    for df in [train_df, test_df]:
        for c in num_cols:
            if c not in df.columns:
                df[c] = 0.0
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors= "coerce" ).fillna(
    all_cat = pd.concat([train_df[cat_cols], test_df[cat_cols]], axis= 0, ignore_i
    try:
        ohe = OneHotEncoder(handle_unknown= "ignore" , sparse_output= True)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown= "ignore" , sparse= True)
    X_cat_all = ohe.fit_transform(all_cat)
    X_num_all = sp.csr_matrix(pd.concat([train_df[num_cols], test_df[num_cols]], 
    if USE_HASH_CODES:
        hasher = FeatureHasher(n_features=HASH_DIM, input_type= "dict")
        X_hash_all = hasher.transform(hash_dicts)
        X_all = sp.hstack([X_num_all, X_cat_all, X_hash_all], format="csr")
    else:

```

#### Chunk 10: Pages 175–175

```text

[Page 175]
            "ce_sum" : float(vals.sum()),
            "ce_mean" : float(vals.mean()),
            "ce_max" : float(vals.max()),
            "ce_pos_sum" : float(pos.sum()) if len(pos) else 0.0,
            "ce_neg_sum" : float(neg.sum()) if len(neg) else 0.0,
            "ce_n_codes" : float(len(codes)),
        })
    return pd.DataFrame(rows)
def make_model (seed, use_cuda= True):
    p = dict(XGB_RESID_PARAMS)
    p["random_state" ] = seed
    if not use_cuda:
        p[ "device" ] = "cpu"
    return xgb.XGBRegressor(**p)
def predict_best (model, X ):
    bi = getattr(model, "best_iteration" , None)
    if bi is not None:
        try:
            return model.predict(X, iteration_range=( 0, int(bi) + 1))
        except TypeError:
            pass
    return model.predict(X)
# -----------------------------
# Load data
# -----------------------------
log(f"xgboost version: {getattr(xgb,'__version__' ,'unknown' )}")
train = normalize_cols(pd.read_csv(TRAIN_CSV))
test  = normalize_cols(pd.read_csv(TEST_CSV))
patients = normalize_cols(pd.read_csv(PATIENTS_CSV))
assert RECEIPT_CACHE.exists(), f"Missing {RECEIPT_CACHE} "
receipts = load(RECEIPT_CACHE)
if not isinstance (receipts, dict) or len(receipts) < 3000:
    raise RuntimeError( "receipts_parsed.joblib structure/size looks wrong. Rebuil
# Merge patients
train = train.merge(patients, on= "patient_id" , how="left")

```

#### Chunk 11: Pages 177–177

```text

[Page 177]
    all_df[c] = pd.to_numeric(all_df[c], errors= "coerce" ).fillna( 0.0).astype(np.f
# Fit one-hot encoder on all (no leakage; no y used)
try:
    ohe = OneHotEncoder(handle_unknown= "ignore" , sparse_output= False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown= "ignore" , sparse= False)
X_cat_all = ohe.fit_transform(all_df[cat_cols])
# Split back
train2 = all_df[all_df[ "_is_train" ] == 1].copy().reset_index(drop= True)
test2  = all_df[all_df[ "_is_train" ] == 0].copy().reset_index(drop= True)
X_cat_train = X_cat_all[: len(train2)]
X_cat_test  = X_cat_all[ len(train2):]
y = train2[ "ed_cost_next3y_usd" ].astype( float).values
# Global cost bins (no leakage)
bins = build_prior_bins(train2[ "prior_ed_cost_5y_usd" ].values, n_bins=COST_BINS)
# Stratify for CV (stable)
strat = (train2[ "primary_chronic" ].astype( str) + "|" +
         pd.qcut(train2[ "prior_ed_cost_5y_usd" ].astype( float), q=10, duplicates= "
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle= True, random_state=SEED)
# OOF containers
oof_pred = np.zeros( len(train2), dtype= float)
oof_base = np.zeros( len(train2), dtype= float)
test_pred_accum = np.zeros( len(test2), dtype= float)
test_base_full = None
fold_logs = []
log(f"Starting V3 CV: baseline + residual | n= {len(train2)}")
for fold, (tr_idx, va_idx) in enumerate (skf.split(np.zeros( len(train2)), strat), 
    t0 = time.time()
    # fold baseline (no leakage): compute medians using fold-train only

```

#### Chunk 12: Pages 189–189

```text

[Page 189]
        bins = np.unique(np.quantile(train_prior.values.astype( float), [0, 0.5, 1
    return bins
def predict_best (model, X ):
    bi = getattr(model, "best_iteration" , None)
    if bi is not None:
        try:
            return model.predict(X, iteration_range=( 0, int(bi) + 1))
        except TypeError:
            pass
    return model.predict(X)
def group_median_calibrate (train_df, oof_pred, y_true, test_df, test_pred, n_bins
    bins = build_prior_bins(train_df[ "prior_ed_cost_5y_usd" ].astype( float), n_bin
    tr_bin = pd.cut(train_df[ "prior_ed_cost_5y_usd" ].astype( float), bins=bins, in
    te_bin = pd.cut(test_df[ "prior_ed_cost_5y_usd" ].astype( float), bins=bins, inc
    tr_key = (train_df[ "primary_chronic" ].astype( str) + "|" + tr_bin).values
    te_key = (test_df[ "primary_chronic" ].astype( str) + "|" + te_bin).values
    resid = y_true - oof_pred
    shift = pd.Series(resid).groupby(tr_key).median().to_dict()
    oof_adj = oof_pred + np.array([shift.get(k, 0.0) for k in tr_key], dtype= float
    te_adj  = test_pred + np.array([shift.get(k, 0.0) for k in te_key], dtype= flo
    return oof_adj, te_adj, shift
def tfidf_svd_patient (text_by_pid: pd.Series, n_components: int, prefix: str):
    """
    Fit TF-IDF + SVD on all patients' text (train+test). Return dataframe with SV
    """
    texts = text_by_pid.fillna( "").astype( str).values
    vec = TfidfVectorizer(
        min_df= 2,
        max_features= 20000,
        ngram_range=( 1,2),
        token_pattern= r"(?u)\b\w+\b"
    )
    X = vec.fit_transform(texts)
    k = min(n_components, max(2, X.shape[ 1]-1)) if X.shape[ 1] > 2 else 2
    svd = TruncatedSVD(n_components=k, random_state=SEED)
    Z = svd.fit_transform(X)

```

#### Chunk 13: Pages 198–198

```text

[Page 198]
# receipts features
log("Building receipt features..." )
patient_ids = all_df[ "patient_id" ].astype( int).tolist()
rcpt_feat, code_list = receipt_patient_features(receipts, patient_ids)
all_df = all_df.merge(rcpt_feat, on= "patient_id" , how="left")
log(f"Receipt top-code features: {len(code_list)} codes")
# fill missing numeric from AUX joins
all_df = all_df.fillna( 0.0)
# split
train_df = all_df[all_df[ "_is_train" ] == 1].copy().reset_index(drop= True)
test_df  = all_df[all_df[ "_is_train" ] == 0].copy().reset_index(drop= True)
y = train_df[ "ed_cost_next3y_usd" ].astype( float).values
# feature columns
cat_cols = [ "primary_chronic" , "sex", "insurance" , "zip3"]
for c in cat_cols:
    train_df[c] = train_df[c].astype( str)
    test_df[c]  = test_df[c].astype( str)
drop_cols = { "ed_cost_next3y_usd" ,"_is_train" }
feat_cols = [c for c in train_df.columns if c not in drop_cols]
# separate numeric/cat
num_cols = [c for c in feat_cols if c not in cat_cols and c != "patient_id" ]
# convert numeric
for c in num_cols:
    train_df[c] = pd.to_numeric(train_df[c], errors= "coerce" ).fillna( 0.0).astype(
    test_df[c]  = pd.to_numeric(test_df[c], errors= "coerce" ).fillna( 0.0).astype(n
# one-hot
try:
    ohe = OneHotEncoder(handle_unknown= "ignore" , sparse_output= True)
except TypeError:
    ohe = OneHotEncoder(handle_unknown= "ignore" , sparse= True)
X_cat = ohe.fit_transform(pd.concat([train_df[cat_cols], test_df[cat_cols]], axis
X_cat_tr = X_cat[: len(train_df)]
X_cat_te = X_cat[ len(train_df):]

```

#### Chunk 14: Pages 221–221

```text

[Page 221]
-----------------------------------------------------------------
----------
AssertionError                            T raceback (most recent call  
last)
Cell In[36], line 109
    107 assert RECEIPT_CA CHE.exists(), f"Missing  
{RECEIPT_CA CHE}"
    108 receipts = load(RECEIPT_CA CHE)
--> 109 assert isinstance(receipts, dict) and len(receipts)  
>= 3000, "Bad receipts cache."
    111 all_df = pd.concat([train.assign(_split="train"),  
test.assign(_split="test")], axis=0, ignore_index=T rue)
    112 all_ids = all_df["patient_id"].astype(int).tolist()
AssertionError: Bad receipts cache.
你这个报错  100% 是  receipts_p arsed.joblib ⼜变成 “ ⾮  dict / 只有很少条⽬ ” 的坏缓存 （就
像你之前第⼀次  EDA 时缓存⾥只有  2 条⼀样）。这通常来⾃两类原因：
被其它实验覆盖 （例如某次写⼊了⼀个不同结构的对象到同⼀路径） ；
写⽂件中途被中断 导致⽂件不完整 / 损坏。joblib.dump/load  本质是  pickle 持久化，
写⼊不是原⼦操作；更稳妥的做法是 “ 写到临时⽂件 → 再  replace/rename 覆盖 ” ，避免
出现半写⼊⽂件。  
下⾯给你⼀个 ⼀键修复  + 继续跑  Iter25 ED A 的 Jupy ter 单  cell。它会：
1. 尝试加载  receipts_p arsed.joblib  → 校验结构 / 数量
2. 若坏：⾃动在  cache_iter10/  ⾥找  最⼤的可⽤  receipts_p arsed.joblib /
receipts_p arsed.bad_.joblib** 来恢复
3. 还不⾏：就从  receipts_pdf/  快速重建 （你机器之前  15~20s 就能跑完）
4. ⽤原⼦写⼊ 保存新的  joblib ，避免再写坏
5. 然后直接跑你要的  Iter25 baselines （ ratio / log-power / isotonic / very-light X GB ）joblib.readthedo …+3
python

```

#### Chunk 15: Pages 236–238

```text

[Page 236]
你接下来怎么做（最省时间）
1. 直接跑上⾯这⼀个  cell （它会⾃动修好  receipts cache ）
2. 把最后打印的  baseline_comp arison 表格贴出来（ 5 ⾏）
3. 我们就能⽴刻判断专家的  Iter25 假设是否成⽴：
log_power  / isotonic  是否显著优于  ratio_baseline
log_power + util/price  是否进⼀步下降
very_light_xgb  是否在不爆炸的情况下更好
如果这⼀步验证通过，我们  Phase 3 的建模主⼲就会变成：
sublinear b aseline （ log/isot onic ）  + boosting 学  residual （并且⽤  util/price 拆解作为核
⼼结构特征） ，⽽不是继续在  65k hash 上硬卷。log("\n[iter25] CV baseline comparison (lower is better):" )
print(summary.to_string(index= False))
summary.to_csv(OUT_DIR/ "baseline_comparison.csv" , index= False)
log(f"\nSaved outputs to: {OUT_DIR} ")
log("Key plots: scatter_log_prior_vs_log_future.png, ratio_by_prior_decile.png, p
log("Key table: baseline_comparison.csv" )
Sour ces
Sour ces
-----------------------------------------------------------------
----------
KeyError                                  T raceback (most recent call  
last)
File 
c:\Users\shend\AppData\Local\Programs\Python\Python3
12\Lib\site-packages\pandas\core\indexes\base.py:3812,  
in Index.get_loc(self, key)
   3811 try:
-> 3812     return self._engine.get_loc(casted_key)
   3813 except K eyError as err:

[Page 237]
File pandas/_libs/index.pyx:167, in  
pandas._libs.index.IndexEngine.get_loc()
File pandas/_libs/index.pyx:196, in  
pandas._libs.index.IndexEngine.get_loc()
File pandas/_libs/hashtable_class_helper.pxi:7088, in  
pandas._libs.hashtable.PyObjectHashT able.get_item()
File pandas/_libs/hashtable_class_helper.pxi:7096, in  
pandas._libs.hashtable.PyObjectHashT able.get_item()
KeyError: 'top3_share'
The above exception was the direct cause of the following  
exception:
KeyError                                  T raceback (most recent call  
last)
Cell In[38], line 321
    318 tops = pd.concat([tops[["patient_id"]],  
tops["amount"].apply(pd.Series)], axis=1)
    320 feat_rcpt = agg.merge(tops, on="patient_id",  
how="left")
--> 321 feat_rcpt["distributed_cost"] = 1.0 -  
feat_rcpt["top3_share"].fillna(0.0)
    322 feat_rcpt["std_total"] =  
feat_rcpt["std_total"].replace(0, np.nan).fillna(1e-9)
    323 feat_rcpt["price_index"] = feat_rcpt["sum_items"] /  
feat_rcpt["std_total"]
File 
c:\Users\shend\AppData\Local\Programs\Python\Python3
12\Lib\site-packages\pandas\core\frame.py:4113, in  
DataFrame.__getitem__(self, key)
   4111 if self.columns.nlevels > 1:
   4112     return self._getitem_multilevel(key)
-> 4113 indexer = self.columns.get_loc(key)
   4114 if is_integer(indexer):
   4115     indexer = [indexer]

[Page 238]
File 
c:\Users\shend\AppData\Local\Programs\Python\Python3
12\Lib\site-packages\pandas\core\indexes\base.py:3819,  
in Index.get_loc(self, key)
   3814     if isinstance(casted_key, slice) or (
   3815         isinstance(casted_key, abc.Iterable)
   3816         and any(isinstance(x, slice) for x in casted_key)
   3817     ):
   3818         raise InvalidIndexError(key)
-> 3819     raise K eyError(key) from err
   3820 except T ypeError:
   3821     # If we have a listlike key, _check_indexing_error  
will raise
   3822     #  InvalidIndexError. Otherwise we fall through  
and re-raise
   3823     #  the T ypeError.
   3824     self._check_indexing_error(key)
KeyError: 'top3_share'
这个 KeyError: 'top3_share'  本质原因是：你那段  groupby().apply()  之后的 “展开 / 透
视”逻辑在你当前  pandas 版本下没有稳定⽣成名为  top1_share/top3_share  的列，所以
merge 后  feat_rcpt  ⾥根本没有  top3_share  这列。
最稳的修法： 不要⽤  pivot/lev el_1 那套展开 ，改成  pandas 官⽅推荐的  named
aggr egation（groupby().agg(top1_share=(...), top3_share=(...)) ），保证列名稳定。
pandas ⽂档明确⽀持这种写法。  
（你之前那种 “ 返回  Series 再展开 ” 的⽅式也能做，但不同  pandas 输出形状 / 列名很容易
变，正好踩坑了。 ）
下⾯给你⼀份  单 cell 的“Iter25 ED A （带  receipts cache ⾃动修复） ” 完整代码，唯⼀关键改
动就是  top1/t op3 shar e 的计算现在⽤  named aggregation ，绝不会再缺列。Pandas
Stack Overflow +1
python

```



## Quick index of key markers (page-level)

### System Role

- Page 1: System R ole & P ersona:

- Page 12: System R ole & Objective:



### Your Task

- Page 3: Your T ask:

- Page 139: Your T ask:

- Page 161: Your T ask: The S trategic Pivot



### Submission

- Page 139: 🚀 Submitting predictions...

- Page 183: 🚀 Submitting predictions...

- Page 184: ✅ Submission successful!



### Score/MAE

- Page 1: Metric: Mean Absolute Error (MAE).

- Page 3: extracted PDF features to lower MAE?

- Page 9: This is where teams quietly lose MAE.

- Page 10: Phase 2 — “R eceipt phenotype” upgrade (o ften the big MAE dr op)

- Page 11: CatBoost(MAE) + X GBoost(MAE) → average

- Page 12: Only keep additions that improve OOF MAE and reduce underprediction in top

- Page 15: MAE. R eceipt-derived “severity/complexity” features should separate tail patients even

- Page 32: # Tail comparisons (who drives MAE)

- Page 70: problem — MAE gains will come from improving most o f the distribution , not only the far

- Page 72: This is exactly the kind of “orthogonal” signal that can improve MAE beyond prior spend.

- Page 79: This is exactly the kind of practical “data plumbing” win that moves MAE.

- Page 82: F) Int eractions (these usually add MAE wins in healthcar e cost)

- Page 85: Since the leaderboard metric is MAE , the cleanest alignment is to train with an

- Page 86: prior_cost_decile)  (often a free MAE win).

- Page 87: # Calibration (often helps MAE)

- Page 99: # Model A: L1/MAE-aligned objective

- Page 101: 1. Paste the printed OOF MAE raw vs calibrat ed.

- Page 102: 527 # Model A: L1/MAE-aligned objective

- Page 117: print(f"[Fold {fold}/{N_SPLITS} ] MAE={mae:.6f}")

- Page 118: print(f"OOF MAE raw: {mean_absolute_error(y, oof): .6f} | calibrated: {mean_ab

- Page 121: [Fold 1/5] MAE=479998797.222300

- Page 122: analyze the log and compare with actual real MAE later on.

- Page 137: log(f"OOF MAE (raw): {oof_mae: .6f}")

- Page 138: OOF MAE (raw)  and OOF MAE calibrated .

- Page 139: [Fold 1] T rain-MAE: 191.44 | V al-MAE: 460.98

- Page 140: Internal CV (calibrat ed) 443 vs LB 463 (~+20 MAE)  means your CV is optimistic

- Page 141: report unweight ed MAE  (traditional CV)

- Page 158: log( f"[Fold {fold}/{N_SPLITS_REG} ] mode={mode} | TrainMAE= {train_mae: .2f} | V

- Page 159: log( f"Calibrated OOF MAE: {oof_mae_cal: .6f} (delta {oof_mae - oof_mae_cal:+ .6

- Page 160: 2. OOF MAE (unweighted)  and OOF MAE (importance-weighted proxy)

- Page 161: Train MAE: ~230

- Page 162: MAE benchmark.

- Page 163: [06:42:43] [Fold 1/5] mode=cuda | T rainMAE=258.77 |

- Page 164: [06:45:21] [Fold 4/5] mode=cuda | T rainMAE=232.47 |

- Page 165: [06:46:30] Calibrated OOF MAE: 445.306969 (delta

- Page 167: 4. Evaluate MAE on final pr ediction : baseline + residual_pred .

- Page 179: log( f"[Fold {fold}/{N_SPLITS} ] mode={mode} | baseline_MAE= {mae_base: .2f} | v3_

- Page 182: 1. OOF MAE baseline-only  and OOF MAE V3  (and the delta)

- Page 183: [14:54:26] [Fold 3/5] mode=cuda | baseline_MAE=509.11 |

- Page 184: 📊 Score: 546.2783 (MAE)

- Page 185: 3. 回到 MAE 对⻬⽬标： XGBoost reg:absoluteerror （这是最贴合  metric 的⽬标之⼀）

- Page 186: # Metric-aligned: XGBoost reg:absoluteerror (MAE) + group-median residual calibra

- Page 200: log( f"[Fold {fold}/{N_SPLITS} ] mode={mode} | TrainMAE= {tr_mae: .2f} | ValMAE= {v

- Page 201: 2. 每折 TrainMAE / ValMAE / best_iter

- Page 202: objective="regression_l1" （ MAE / median ）

- Page 204: 你们要追  400 MAE ，我认为最可能的突破⼝不是 “ 再堆  100

- Page 206: 并分别报告  overall MAE 和  outlier MAE

- Page 207: 1. ratio b aseline vs log b aseline vs isot onic b aseline 的  OOF MAE

- Page 208: “输出 overall MAE + t op5% MAE （尾部） ”

- Page 220: very_light_xgb  应该在不爆炸的前提下，进⼀步压  MAE （如果  admissions 有⽤，会



### Traceback

- Page 34: AttributeError                            T raceback (most recent call

- Page 101: TypeError                                 T raceback (most recent call

- Page 221: AssertionError                            T raceback (most recent call

- Page 236: KeyError                                  T raceback (most recent call

- Page 237: KeyError                                  T raceback (most recent call



### TypeError

- Page 97: except TypeError:

- Page 101: TypeError                                 T raceback (most recent call

- Page 102: TypeError: X GBModel.fit() got an unexpected keyword

- Page 103: # Fixes: eval_metric in fit() TypeError, GPU param compatibility, early-stop pred

- Page 112: except TypeError:

- Page 114: except TypeError:

- Page 115: except TypeError as e:

- Page 116: except TypeError:

- Page 126: except TypeError:

- Page 133: except TypeError:

- Page 147: except TypeError:

- Page 153: except TypeError:

- Page 175: except TypeError:

- Page 177: except TypeError:

- Page 189: except TypeError:

- Page 198: except TypeError:

- Page 238: 3820 except T ypeError:



### KeyError

- Page 236: KeyError                                  T raceback (most recent call

- Page 237: KeyError: 'top3_share'

- Page 238: -> 3819     raise K eyError(key) from err



### AssertionError

- Page 221: AssertionError                            T raceback (most recent call



## Appendix: Full extracted text (page-by-page, verbatim)

> Each page is embedded as a code block. This ensures no prompt/log content is cut off.


### Page 1

```text

System R ole & P ersona:
You are a domain expert in Healthcare Economics and a  
Senior Scientist in Machine Learning/AI. Y ou specialize in  
"ED Cost Forecasting" and predictive modeling for hospital  
readmissions. Y ou possess deep knowledge of medical  
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
Current S tatus: W e previously struggled with high MAE  
because our PDF data was incomplete. W e have now fixed  
the pipeline so that every patient in the training set has a  
corresponding, complete R eceipt PDF.
The Data Landscape:
We are working with a multimodal dataset consisting of  
CSVs and synthetic billing PDFs.receipt_1.pdf
PDFreceipt_6.pdf
PDF
receipt_10.pdf
PDFdescription.md
File

```


### Page 2

```text

Tabular Data (ed_cost_*.csv):
patient_id: Join key.
primary_chronic: {HF (Heart F ailure), DiabetesComp,  
Pneumonia}.
prior_ed_visits_5y: Integer count of visits in the prior 5  
years.
prior_ed_cost_5y_usd: T otal cost (USD) in the prior 5 years.
ed_cost_next3y_usd (T arget): The regression target for the  
next 3 years.
Note: There are other columns (like ZIP3) that need  
analysis.
Unstructured Data (receipt_<patient_id>.pdf):
These are synthetic "billing summaries."
Content: Line items containing CPT/HCPCS-like codes,  
quantities, and line totals.
Validation: The final "T OTAL" in the PDF matches the  
prior_ed_cost_5y_usd in the CSV.
Key Constraints & Environment:
Hardware: Local Windows environment with a single  
NVIDIA R TX 4060 (CUD A 12). The solution must be  
efficient enough to run here (e.g., careful selection of Local  
LLMs, efficient OCR/parsing, or feature extraction).
Search Constraint: Y ou should search for relevant papers  
and challenges (e.g., similar Kaggle competitions, papers  
on ED cost regression) to inform your strategy, but do not  
over-search. W e need practical, applicable strategies, not  
just a literature review.

```


### Page 3

```text

Your T ask:
Using your domain knowledge and the provided files (I will  
attach 3 sample PDFs, the description.md, and the CSV  
samples), please develop a comprehensive S trategy and  
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
NLP/R egex?
Correction of P ast Errors: Since we fixed the missing PDF  
issue, how do we best align the tabular features with the  
extracted PDF features to lower MAE?
Immediate Next S tep:
Please SEAR CH for similar problems/papers first, then  

```


### Page 4

```text

THINK deeply about the medical/economic implications of  
the data columns and PDF contents. Finally, output a  
detailed S tep-by-S tep Master Plan.
(I have attached the dataset samples and description  
below)
What the lit eratur e says will actually mov e the needle (and why it
matt ers her e)
1. “Cost -on-cost” is usually the str ongest signal.
Across healthcare cost prediction work, prior spending/utilization is often the best
single predictor of future spending, and gradient boosting  tends to be a top-
performing workhorse on this kind of data. 
2. High-cost tails ar e syst ematically har d (classic underpr ediction pr oblem).
Traditional risk models (e.g., CMS-HC C–style risk adjustment) tend to overpredict low-
cost and underpr edict high-cost  patients—so if you want to win, you need explicit
“tail hygiene” (loss choice, calibration, segmentation, and features that identify
impending spikes). 
3. Area-lev el socioeconomic conditions corr elate with higher spending , and in
diabetes crises, social determinants strongly predict recurrence.
So your ZIP3 + insurance  combo is not “just demographic garnish”—it can be a real
driver proxy. 
These three points map perfectly to your setup: you already have strong cost-on-cost
features (prior 5y cost/visits), you fixed PDF completeness (so richer utilization phenotype
is now available), and you have geography/insurance signals in the receipts +
patients.csv . 
The “niche” clinical economics: why HF ≠ Diabet esComp ≠ Pneumonia
for futur e ED cost
Think in cost spik e mechanisms  (what creates a big future ED cost episode) rather than
“regr Heart F ailure (HF)
HF is a progressiv e, relapse-pr one condition. Costs jump when patients hit a worsening
HF ev ent (fluid overload, pulmonary edema, arrhythmia, cardiorenal syndrome), andPMC
MedP AC+1
JAMA Network +2
description
description

```


### Page 5

```text

utilization/cost remains elevated around those events. This “event-driven persistence” is
why HF tends to have higher future cost distribution. 
What that implies for y our featur es:
Signals of high-acuity ED car e (e.g., airway support, high-level E/M codes, cardiac
biomarkers) in the receipt are not just “past severity”—they’re a proxy for a phenotype
that tends to recur.
Diabet esComp
“DiabetesComp” cost spikes are often episodic crises  (DKA/HHS, severe hypoglycemia),
plus infections/foot complications. Recurr ence is str ongly tied t o access and social
determinants  (med affordability, adherence barriers, unstable housing, low follow-up).
Recurrent DKA is notably costly, and social factors predict it. 
What that implies for y our featur es:
A history of many “cheap-ish” ED workups can still mean futur e big spend  if it reflects
unstable control + barriers to outpatient care.
ZIP3 + insurance interactions matter more for diabetes-type trajectories than you’d
expect in a generic regression.
Pneumonia
Pneumonia can be “one-and-done,” but repeat pneumonia/r eturn visits/r eadmissions
can be a marker of frailty, chronic lung disease, immune compromise—i.e., hidden
comorbidity bur den. Readmissions contribute materially to pneumonia-related cost.
What that implies for y our featur es:
Repeated chest imaging / cultures / higher-acuity E/M codes may tag “recurrent
respiratory decompensation” vs a single mild episode.
ZIP3: what it r epresents her e, and how t o encode it t o win
What ZIP3 isPMC +1
PMC +1
PMC +1

```


### Page 6

```text

ZIP3 is the first thr ee digits o f the ZIP code , which corresponds to USPS “3-digit ZIP
prefix” areas used for sectional center facility (SCF) sortation—i.e., a coarse geography .
In your receipts, it appears explicitly (e.g., “ZIP3: 21 Insurance: private”). 
What it can pr oxy in ED cost for ecasting
Even in synthetic data, ZIP3 often stands in for:
Price lev el / char ge envir onment  (regional pricing, facility mix)
Access t o car e (ED as safety net)
Area-lev el depriv ation  (which is frequently associated with higher spending and
barriers to care) 
How t o encode (practical, leaderbo ard-safe)
Do all of the following (they’re Treat ZIP3 as a cat egorical str ing* (not an
integer).
CatBoost: native categorical.
XGBoost: one-hot if small cardinality; otherwise target encoding.
2. Frequency encoding : zip3_count , zip3_train_fraction .
Often helps when many rare ZIP3s exist.
3. Cross featur es (very high R OI):
(zip3, insurance)
(zip3, primary_chronic)
(insurance, primary_chronic)
4. Target encoding (CV-safe only)
If you use target encoding, do it within folds  so you don’t leak.
PDFs: how line it ems can pr edict futur e cost (ev en though T OTAL only
matches prior cost)
Your receipt PDFs are structur ed utilization phenotypes , not just totals. Example patterns
from your samples:Postal Explorer +1
receipt_1
JAMA Network +1
receipt_1
External enrichment (ADI/SDI) could help in real life, but unless rules explicitly allow
external data, I’d first win with internal encodings + int eractions .

```


### Page 7

```text

Patient 6 includes emer gency intub ation (31500)  plus higher-severity ED visit codes
and biomarkers. 
Patient 1 shows repeated moderate ED visits (99283), CBC, chest X-ray. 
Patient 10 has fewer items, mostly ED visit + chest imaging. 
The k ey: build featur es that sep arate “one big episode” vs “many small
episodes”
Two patients can have the same prior total, but:
One had one sev ere event (procedure-heavy, high-acuity)
The other had many low/moderat e visits  (poor control / poor access)
Those two patterns have differ ent futur e risk .
#  can still indicate high future cost
Yes—because the predictive signal is no sive,” it’s “this code indicates a
recurring clinical story.” Exam /M low-to-moderate codes (99281–99283) can
reflect frequent r eliance on ED , which often persists.
Repeated chest imaging/labs can indicate recurr ent car diopulmonar y sympt oms
(HF/pneumonia phenotypes).
What t o extract fr om PDFs (the winning featur e set)
You want three lay ers of receipt features:
Layer A — Det erministic structur ed featur es (must -have)
From parsed line items:
n_line_items , n_unique_codes , total_cost_pdf , total_cost_delta_vs_csv
ED visit sev erity distribution  using 99281–99285:
counts, max level, weighted mean level
fraction of high-severity (e.g., 99284–99285)
Procedur e intensity :
count of “big procedure” CPT s (often non-99xxx)
max line-item cost, top1/top3 cost share
Categor y buck ets by CPT pr efix/range  (robust even if codes vary):
E/M (99xxx), labs (often 8xxxx), imaging (7xxxx), procedures (1–6xxxx)
Price index / sit e-of-care proxy:
for common codes (e.g., 99283), compute unit price; compare to global medianreceipt_6
receipt_1
receipt_10
receipt_6
receipt_1
receipt_10

```


### Page 8

```text

this is a sneaky but powerful “regional pricing / facility mix” feature
Layer B — “Episode r econstruction” featur es (high R OI, still simple)
Receipts often list multiple ED visits. Y ou can segment a pseudo-episode whenever an ED
visit E/M code appears:
episode count (should correlate with prior_ed_visits_5y , but not identical)
per-episode cost stats: mean, max, std
worst-episode severity: max E/M level within any episode
“episode complexity”: number of ancillary codes attached to an episode
This captures utilization patterns the raw totals cannot.
Layer C — T ext/Bag-o f-codes embeddings (optional, o ften help ful)
Bag-o f-codes : treat CPT codes as tokens; use FeatureHasher  or TF-IDF on codes
TF-IDF on full r eceipt t ext (codes + descriptions), then Truncat edSVD  to ~50–200
dims
This usually beats “LLM extraction” because your PDFs are already semi-structured.
Local LLM vs R egex/NLP on R TX 4060: what I’ d do t o win
Recommendation: Regex + structur ed parsing fir st, then lightw eight t ext
featur es
Because:
Your PDFs are synthetic billing summaries and appear text-extractable  (no OCR
needed).
LLMs add latency + variance and often underperform on structured extraction unless
you heavily constrain them.
Use LLMs only if  you want an extra semantic embedding feature after you’ve built a
strong structured baseline:
If you go that route, prefer a small embedding model  (sentence-transformers / e5 /
bge-small) over a generative LLM. It’s cheaper and more stable.
Cache embeddings once.

```


### Page 9

```text

Fixing the p ast err or (missing PDFs): how t o align tabular + PDF
featur es pr operly now
This is where teams quietly lose MAE.
1. Build a r eceipt index k eyed by p atient_id
Assert: len(receipts)==len(ed_cost_train)+len(ed_cost_test)
Assert: every patient_id  in train/test has exactly one receipt.
2. Hard validation checks
abs(pdf_TOTAL - prior_ed_cost_5y_usd) < 0.01  for ~100% rows (should hold by
design). 
If any mismatch: log and decide a rule (usually trust CSV, but keep delta as a feature).
3. One featur e table, one join
Create receipt_features.parquet  with patient_id  as the only key.
Join once into train/test; never re-parse during modeling runs.
4. Missingness handling
Now that PDFs are complete, missingness should be ~0; still code defensively:
if parse fails, fill zeros + add receipt_parse_failed=1
Step-by-st ep Mast er Plan (optimized for leaderbo ard iteration)
# foundation (must do)
Implement fast PDF t ext extraction  (PyMuPDF) + robust line parsing.
Produce receipt_features  (Layer A) + caching.
Join patients.csv  to add age, sex, insurance, zip3  (per dataset spec).
Establish a CV scheme :
Stratify by primary_chronic
Also stratify by binned prior_ed_cost_5y_usd  (to stabilize tail behavior)
Phase 1 — S trong b aseline (should alr eady be competitiv e)
Train 2 models and average:
1. CatBoostR egressor  (MAE loss) with:
base tabular + patient demographics + receipt Layer A/B features
2. XGBoost  with objective MAE ( reg:absoluteerror ) and:description
description
description

```


### Page 10

```text

one-hot categoricals (or target-encoding CV-safe)
same numeric features
Add small engineere zed prior cost = prior_ed_cost_5y_usd / 5
annualized visits = prior_ed_visits_5y / 5
cost per visit
interaction: primary_chronic × annualized_cost
Phase 2 — “R eceipt phenotype” upgrade (o ften the big MAE dr op)
Add:
Episode reconstruction (Layer B)
Price-index features (unit price normalization for common codes)
Bag-of-codes hashed features (Layer C light):
FeatureHasher(n_features=2**18)  on tokens like:
code=99283
code=99283|chronic=HF
code_bucket=99xxx
Then retrain + blend.
Phase 3 — T ail handling (wher e winning mar gins liv e)
Because models underpredict high-cost patients in general , do:
1. Outlier -awar e calibration
On OOF predictions, compute median residual by:
primary_chronic
cost quantile bin of prior_ed_cost_5y_usd
Apply those median shifts to test predictions.
2. Two-model corr ection
Model A predicts baseline.
Model B predicts residual  using receipt-heavy  features (especially episode max
severity / big-procedure flags).
Final = A + B.
Phase 4 — Optional stacking fr om other tasks (only if allow ed + av ailable)
If the same patient_id  appears in admissions/stays tables:description
MedP AC

```


### Page 11

```text

Train an auxiliary model for readmission risk  (Challenge 1) and use predicted risk as a
feature for ED cost.
Train an auxiliary model for dischar ge readiness  (Challenge 3) and use as a frailty
proxy.
This is often a “free” signal amplifier if rules allow using all provided files.
Practical implementation not es (so it runs cleanly on R TX 4060)
Do not OCR  unless you discover PDFs are image-only (your samples look text-
parsable).
Parallelize PDF parsing with multiprocessing; cache to parquet.
Keep text feature dims controlled:
TF-IDF (max_features ~ 20k) + SVD (50–200)
or FeatureHasher (2^18) to avoid huge vocab memory
“Do this next” checklist (immediat e execution or der)
1. Receipt p arsing & v alidation
Extract: header (zip3, insurance), line items, T OTAL
Validate T OTAL vs CSV prior cost
Save receipt_features.parquet
2. Featur e set v1
Base: chronic + prior visits + prior cost (+ engineered rates)
Add: age/sex/insurance/zip3 from patients.csv 
Add: receipt Layer A
3. Train v1 ensemble
CatBoost(MAE) + X GBoost(MAE) → average
Track MAE by chronic group + by prior-cost quantiles
4. Featur e set v2
Add episode reconstruction (Layer B)
Add bag-of-codes hashing (Layer C light)
5. Tail calibration
Residual median shifts by (chronic × prior-cost-bin)
6. Submit + it eratedescription

```


### Page 12

```text

Only keep additions that improve OOF MAE and reduce underprediction in top
quantiles.
If you want, paste your current feature extraction script (or your latest “EDC OST V4” cell)
and I’ll mark exactly where to add: (1) episode features, (2) price-index fea afe
target encoder for ZIP3/insurance—those are usually the highest-R OI edits once PDFs are
complete.description
Sour ces
Sour ces
Sour ces
System R ole & Objective:
You are the Lead Healthcare AI Scientist and Domain  
Expert. W e are competing in the AgentDS Healthcare  
Challenge 2: ED Cost Forecasting.
The Goal:
Forecast ed_cost_next3y_usd (T otal ED costs for the next 3  
years) based on patient history, chronic conditions, and  
synthetic billing PDF receipts.
Operational Constraint - THE ITERA TIVE WORKFL OW:
DO NO T generate the final training pipeline or model  
architecture yet.
Instead, we will adopt a strict "R esearch & V alidation"  
workflow.
Phase 1 (Current): Y ou will formulate hypotheses based on  
Domain Knowledge + Online Search. Then, you will  
independently design and write an Exploratory Data  
Analysis (ED A) script for me to run locally.
Phase 2 (User Feedback): I will run your code and provide  
the logs/outputs/charts back to you.
Phase 3 (R efinement): Based on the data evidence, you will  
refine the feature engineering strategy and then we will  

```


### Page 13

```text

move to modeling.
Your Environment & R esources:
Hardware: Local Windows PC, NVIDIA R TX 4060 (CUD A 12).  
Efficiency is key.
Directory S tructure (Python pathlib format):
Python
from pathlib import P ath
BASE_DIR = P ath(r"D:\AgentDs\agent_ds_healthcare")
TRAIN_CSV = BASE_DIR / "ed_cost_train.csv"
TEST_CSV  = BASE_DIR / "ed_cost_test.csv"
PATIENT S_CSV = BASE_DIR / "patients.csv"
PDF_DIR = BASE_DIR / "receipts_pdf"
SUBMISSION_P ATH = BASE_DIR /  
"submission_ICHI_V1.csv"
# Cache for parsed PDF data to save time on re-runs
CACHE_DIR = BASE_DIR / "cache_iter10"
CACHE_DIR.mkdir(parents=T rue, exist_ok=T rue)
RECEIPT_CA CHE = CA CHE_DIR / "receipts_parsed.joblib"
Context & Domain Specifics (The "Niche"):
Problem: R egression of Future ED Costs.
Data:
ed_cost_*.csv: T abular patient history and demographics.
PDF R eceipts: S ynthetic billing summaries where the total  
matches the prior 5-year cost.
The Core Challenge: W e need to extract predictive signals  
from these PDFs and CSV s that standard models might  
miss. The relationships between past billing codes  
(CPT/HCPCS) and future risk are complex and non-linear.

```


### Page 14

```text

Your Immediate T ask:
Search & Hypothesize: Briefly search for "ED cost  
prediction features" or similar healthcare regression  
challenges to ground your thinking.
Design "Phase 1: Discovery & V alidation":
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
Domain Analysis: Y our high-level thoughts and  
hypotheses.
The Code: A clean, executable Python block that I can run  
immediately to generate the insights you need.
(I have attached the sample PDFs and CSV snippets for  
your reference in previous prompt)
Domain analysis (hypotheses gr ounded by quick sear ch)
From healthcare cost prediction literature, three patterns show up consistently and are
directly testable in your dataset:
1. Prior spend/utilization is the dominant b aseline signal , but it mostly captures “how
much happened,” not “what happened.” R eviews and empirical studies repeatedly
show strong temporal correlation of expenditures and that prior cost alone is a strong
predictor of future cost. PMC +2

```


### Page 15

```text

Hypothesis H1:  prior_ed_cost_5y_usd  will correlate strongly with
ed_cost_next3y_usd , but the relationship will be non-linear and differ by chronic
cohort.
2. The tail (high-cost p atients) is wher e models fail and wher e leaderbo ard gaps liv e.
High-cost/high-need work emphasizes that the top end is more temporally “sticky,”
but also harder—so identifying who is pois ed to spik e again  matters. 
Hypothesis H2:  Your target is heavy-tailed; a small fraction of patients drives most
MAE. R eceipt-derived “severity/complexity” features should separate tail patients even
within similar prior-cost bins.
3. ED-specific signals o ften hinge on “utilization phenotypes” (fr equent use, acuity ,
procedur es), not just diagnosis labels.  ED ML work frequently models frequent ED
use and resource intensity from administrative data. 
Hypothesis H3:  Receipt line items (CPT/HCPCS-like codes) contain phenotype signals
(acuity mix, procedure intensity, visit count proxy, spend concentration) that add
predictive structure beyond totals.
4. Geography + p ayer/cov erage is a meaningful pr oxy for access/price envir onment.
Even in synthetic setups, ZIP3/insurance often acts like a bundled signal for
socioeconomic + facility pricing context.
Hypothesis H4:  ZIP3 × insurance × primary_chronic  interactions show measurable
shifts in target distribution.
Data int egrity hypothesis (must p ass befor e modeling):
H0: Every patient has a receipt PDF; parsed TOTAL  matches prior_ed_cost_5y_usd  (within
pennies). Any mismatch indicates parsing bugs or file alignment problems.
Phase 1: Discov ery & V alidation — ED A script (run locally)
What it does
Verifies join integrity (train/test/patients), duplicates, missingness.
Profiles the target distribution and tail concentration.
Validates receipt coverage and TOTAL  consistency vs CSV prior cost.
Extracts receipt “utilization phenotype” features (counts, code diversity, E/M severity
proxy, spend concentration).
Computes correlations + code li ft (codes associated with higher future cost), including
a stratified r esidual lif t that partially controls for chronic + prior-cost-bin (so we don’tPMC +2
Nature +1

```


### Page 16

```text

fool ourselves with obvious confounding).
Saves plots + CSV summaries into CACHE_DIR/eda_outputs_iter10/ .
Impor tant:  This is ED A only — no training pipeline, no model fitting.
python
# ============================================================
# AgentDS Challenge 2 — Phase 1 EDA (CSV + PDF receipts)
# ============================================================
# Run this as a standalone script (recommended):  python eda_iter10.py
# Or run in a notebook cell (multiprocessing avoided; uses threads).
import os
import re
import math
import json
import time
import warnings
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump, load
# -----------------------------
# USER CONFIG (as provided)
# -----------------------------
BASE_DIR = Path( r"D:\AgentDs\agent_ds_healthcare" )
TRAIN_CSV = BASE_DIR / "ed_cost_train.csv"
TEST_CSV  = BASE_DIR / "ed_cost_test.csv"
PATIENTS_CSV = BASE_DIR / "patients.csv"
PDF_DIR = BASE_DIR / "receipts_pdf"
SUBMISSION_PATH = BASE_DIR / "submission_ICHI_V1.csv"
CACHE_DIR = BASE_DIR / "cache_iter10"
CACHE_DIR.mkdir(parents= True, exist_ok= True)
RECEIPT_CACHE = CACHE_DIR / "receipts_parsed.joblib"

```


### Page 17

```text

OUT_DIR = CACHE_DIR / "eda_outputs_iter10"
OUT_DIR.mkdir(parents= True, exist_ok= True)
# Parsing controls
MAX_PDFS = None          # set to an int for a quick dry run (e.g., 200). None = p
N_TOP_CODES = 40         # for code lift tables
CODE_MIN_SUPPORT = 20    # only analyze codes appearing in >= this many train pat
TOTAL_TOL = 0.01         # tolerance for matching PDF TOTAL vs prior cost
# Threaded parsing (Windows-friendly)
N_WORKERS = max(4, (os.cpu_count() or 8) - 2)
# -----------------------------
# Optional deps
# -----------------------------
try:
    import fitz  # PyMuPDF
except Exception as e:
    raise RuntimeError(
        "PyMuPDF not installed or failed to import. Install with: pip install pym
    ) from e
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs ):  # fallback
        return x
warnings.filterwarnings( "ignore" )
# ============================================================
# Helpers
# ============================================================
def _normalize_cols (df: pd.DataFrame ) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df

```


### Page 18

```text

def _ensure_col_alias (df: pd.DataFrame, aliases: dict) -> pd.DataFrame:
    """
    aliases: {canonical: [alt1, alt2, ...]}
    If canonical not present but an alt exists, rename alt -> canonical.
    """
    df = df.copy()
    cols = set(df.columns)
    for canon, alts in aliases.items():
        if canon in cols:
            continue
        for a in alts:
            if a in cols:
                df = df.rename(columns={a: canon})
                cols = set(df.columns)
                break
    return df
def savefig(path: Path ):
    plt.tight_layout()
    plt.savefig(path, dpi= 140)
    plt.close()
def safe_float (x):
    if x is None:
        return np.nan
    if isinstance (x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    s = s.replace( "$", "").replace( ",", "")
    try:
        return float(s)
    except Exception:
        return np.nan
def spearman_corr (a, b):
    # robust Spearman without scipy
    a = pd.Series(a).astype( float)
    b = pd.Series(b).astype( float)
    m = a.notna() & b.notna()
    if m.sum() < 3:

```


### Page 19

```text

        return np.nan
    return a[m].rank().corr(b[m].rank())
def print_header (title: str):
    print("\n" + "="*90)
    print(title)
    print("="*90)
# ============================================================
# Receipt parsing
# ============================================================
ZIP3_RE = re. compile(r"\bzip3\b\s*[:\-]?\s*(\d{3})" , re.IGNORECASE)
INS_RE  = re. compile(r"\binsurance\b\s*[:\-]?\s*([A-Za-z_]+)" , re.IGNORECASE)
TOTAL_RE = re. compile(r"\btotal\b\s*[:\-]?\s*\$?\s*([\d,]+(?:\.\d{2})?)" , re.IGNO
# CPT/HCPCS-like code pattern:
# - 5-digit numeric CPT (e.g., 99283)
# - HCPCS alphanumeric (e.g., A0425) — allow 1 letter + 4 digits, or generic 4-7 
CODE_RE = re. compile(r"^[A-Z]?\d{4,5}$|^[A-Z0-9]{4,7}$" )
def extract_pdf_text (pdf_path: Path ) -> str:
    doc = fitz. open(pdf_path)
    try:
        parts = []
        for i in range(doc.page_count):
            page = doc.load_page(i)
            parts.append(page.get_text( "text"))
        return "\n".join(parts)
    finally:
        doc.close()
def parse_line_item (line: str):
    """
    Heuristic: parse lines ending with an amount, with qty preceding it.
    Format often like:
        99283   Emergency Department Visit (Level 3)   1   245.00
    We'll infer:
        code = first token
        qty  = last integer before amount

```


### Page 20

```text

        amt  = trailing decimal number
        desc = middle
    """
    line = line.strip()
    if not line:
        return None
    # Must end with an amount
    m_amt = re.search( r"(\d[\d,]*\.\d{2})\s*$" , line)
    if not m_amt:
        return None
    amt = safe_float(m_amt.group( 1))
    left = line[:m_amt.start()].strip()
    # qty should end the left side
    m_qty = re.search( r"(\d+)\s*$" , left)
    if not m_qty:
        return None
    qty = int(m_qty.group( 1))
    left2 = left[:m_qty.start()].strip()
    if not left2:
        return None
    code = left2.split()[ 0].strip()
    if not CODE_RE. match(code):
        return None
    desc = left2[ len(code):].strip()
    return {"code": code, "desc": desc, "qty": qty, "amount" : float(amt)}
def parse_receipt (pdf_path: Path ):
    text = extract_pdf_text(pdf_path)
    zip3 = None
    m = ZIP3_RE.search(text)
    if m:
        zip3 = m.group( 1)
    insurance = None
    m = INS_RE.search(text)

```


### Page 21

```text

    if m:
        insurance = m.group( 1).lower()
    total = None
    # take the LAST match as the grand total (safer if subtotals exist)
    totals = TOTAL_RE.findall(text)
    if totals:
        total = safe_float(totals[- 1])
    # line items
    items = []
    for raw in text.splitlines():
        it = parse_line_item(raw)
        if it is not None:
            items.append(it)
    return {
        "pdf_path" : str(pdf_path),
        "zip3_receipt" : zip3,
        "insurance_receipt" : insurance,
        "pdf_total" : total,
        "n_items_raw" : len(items),
        "items": items,
        "text_len" : len(text),
    }
# ============================================================
# Receipt features (phenotypes)
# ============================================================
def em_level_from_code (code: str):
    # ED E/M often in 99281–99285
    if not code.isdigit():
        return None
    if code.startswith( "9928") and len(code) == 5:
        last = int(code[-1])
        if last in [1,2,3,4,5]:
            return last
    return None

```


### Page 22

```text

def code_bucket (code: str):
    # Coarse bucket by prefix
    if code.isdigit():
        if code.startswith( "99"):
            return "em_99xxx"
        first = code[ 0]
        return f"num_{first}xxxx"
    # HCPCS/alphanumeric
    return f"alpha_ {code[0].upper()}"
def build_receipt_features (parsed: dict):
    items = parsed.get( "items", [])
    if not items:
        return {
            "receipt_parse_failed" : 1,
            "receipt_n_items" : 0,
            "receipt_n_unique_codes" : 0,
            "receipt_sum_items" : 0.0,
            "receipt_top1_share" : np.nan,
            "receipt_top3_share" : np.nan,
            "receipt_em_count" : 0,
            "receipt_em_max" : np.nan,
            "receipt_em_mean" : np.nan,
            "receipt_non_em_count" : 0,
            "receipt_bucket_entropy" : np.nan,
            "receipt_zip3_from_pdf" : parsed.get( "zip3_receipt" ),
            "receipt_ins_from_pdf" : parsed.get( "insurance_receipt" ),
            "receipt_pdf_total" : parsed.get( "pdf_total" ),
        }
    amounts = np.array([it[ "amount" ] for it in items], dtype= float)
    codes = [it[ "code"] for it in items]
    uniq = len(set(codes))
    s = float(np.nansum(amounts))
    top_sorted = np.sort(amounts)[::- 1]
    top1 = float(top_sorted[ 0]) if len(top_sorted) else np.nan
    top3 = float(top_sorted[: 3].sum()) if len(top_sorted) >= 3 else float(top_sor
    top1_share = (top1 / s) if s > 0 else np.nan
    top3_share = (top3 / s) if s > 0 else np.nan

```


### Page 23

```text

    # E/M levels
    em_levels = [em_level_from_code(c) for c in codes]
    em_levels = [x for x in em_levels if x is not None]
    em_count = len(em_levels)
    em_max = float(np.max(em_levels)) if em_levels else np.nan
    em_mean = float(np.mean(em_levels)) if em_levels else np.nan
    non_em_count = sum(1 for c in codes if not (c.isdigit() and c.startswith( "99"
    # bucket entropy (how diverse is the "type" mix)
    buckets = [code_bucket(c) for c in codes]
    bc = Counter(buckets)
    probs = np.array( list(bc.values()), dtype= float)
    probs = probs / probs. sum()
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    return {
        "receipt_parse_failed" : 0,
        "receipt_n_items" : len(items),
        "receipt_n_unique_codes" : uniq,
        "receipt_sum_items" : s,
        "receipt_top1_share" : top1_share,
        "receipt_top3_share" : top3_share,
        "receipt_em_count" : em_count,
        "receipt_em_max" : em_max,
        "receipt_em_mean" : em_mean,
        "receipt_non_em_count" : non_em_count,
        "receipt_bucket_entropy" : entropy,
        "receipt_zip3_from_pdf" : parsed.get( "zip3_receipt" ),
        "receipt_ins_from_pdf" : parsed.get( "insurance_receipt" ),
        "receipt_pdf_total" : parsed.get( "pdf_total" ),
    }
# ============================================================
# Main EDA
# ============================================================
def main():
    print_header( "LOAD DATA" )
    train = _normalize_cols(pd.read_csv(TRAIN_CSV))

```


### Page 24

```text

    test  = _normalize_cols(pd.read_csv(TEST_CSV))
    # normalize common aliases
    aliases = {
        "patient_id" : ["patientid" , "id"],
        "primary_chronic" : ["chronic" , "primarycondition" ],
        "prior_ed_visits_5y" : ["prior_visits_5y" , "ed_visits_5y" ],
        "prior_ed_cost_5y_usd" : ["prior_cost_5y_usd" , "ed_cost_5y" , "prior_cost_5y
        "ed_cost_next3y_usd" : ["target" , "y", "cost_next3y" ],
        "zip3": ["zip", "zip_3", "zip_prefix" , "zip3_code" , "zip3"],
        "insurance" : ["payer", "insurance_type" ],
    }
    train = _ensure_col_alias(train, aliases)
    test  = _ensure_col_alias(test, aliases)
    print("Train shape:" , train.shape)
    print("Test  shape:" , test.shape)
    print("Train cols:" , list(train.columns))
    # patients
    patients = None
    if PATIENTS_CSV.exists():
        patients = _normalize_cols(pd.read_csv(PATIENTS_CSV))
        patients = _ensure_col_alias(patients, aliases)
        print("Patients shape:" , patients.shape)
        print("Patients cols:" , list(patients.columns))
    else:
        print("WARNING: patients.csv not found at" , PATIENTS_CSV)
    # integrity: patient_id
    assert "patient_id"  in train.columns, "train missing patient_id"
    assert "patient_id"  in test.columns, "test missing patient_id"
    dup_tr = train[ "patient_id" ].duplicated(). sum()
    dup_te = test[ "patient_id" ].duplicated(). sum()
    print(f"Duplicate patient_id — train: {dup_tr}  | test: {dup_te} ")
    overlap = set(train["patient_id" ]).intersection( set(test["patient_id" ]))
    print("Train/Test patient_id overlap:" , len(overlap))

```


### Page 25

```text

    # merge patients
    if patients is not None and "patient_id"  in patients.columns:
        train = train.merge(patients, on= "patient_id" , how="left", suffixes=( "", 
        test  = test.merge(patients, on= "patient_id" , how="left", suffixes=( "", "_
    # missingness
    print_header( "MISSINGNESS (TOP 30)" )
    miss = train.isna().mean().sort_values(ascending= False).head(30)
    print(miss.to_string())
    # target profile
    if "ed_cost_next3y_usd"  in train.columns:
        y = train[ "ed_cost_next3y_usd" ].astype( float)
        print_header( "TARGET DISTRIBUTION" )
        qs = y.quantile([ 0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1.0])
        print(qs.to_string())
        print("Mean:", y.mean(), "Std:", y.std(), "Skew(approx):" , y.skew())
        # histogram raw
        plt.figure()
        plt.hist(y.clip(upper=y.quantile( 0.995)), bins= 60)
        plt.title( "Target histogram (clipped at 99.5%)" )
        plt.xlabel( "ed_cost_next3y_usd" )
        plt.ylabel( "count")
        savefig(OUT_DIR / "target_hist_clipped.png" )
        # histogram log
        plt.figure()
        plt.hist(np.log1p(y), bins= 60)
        plt.title( "log1p(Target) histogram" )
        plt.xlabel( "log1p(ed_cost_next3y_usd)" )
        plt.ylabel( "count")
        savefig(OUT_DIR / "target_log_hist.png" )
        # tail concentration
        top = y.quantile( 0.95)
        tail_share = y[y >= top]. sum() / y.sum()
        print(f"Top 5% of patients contribute ~ {tail_share: .3f} of total target s
    # chronic cohort comparisons

```


### Page 26

```text

    if "primary_chronic"  in train.columns and "ed_cost_next3y_usd"  in train.colum
        print_header( "TARGET BY PRIMARY_CHRONIC" )
        g = train.groupby( "primary_chronic" )["ed_cost_next3y_usd" ]
        summary = g.agg([ "count", "mean", "median" ])
        summary[ "p90"] = g.quantile( 0.90)
        summary[ "p95"] = g.quantile( 0.95)
        summary[ "p99"] = g.quantile( 0.99)
        print(summary.sort_values( "mean", ascending= False).to_string())
        # boxplot (log)
        plt.figure(figsize=( 8,4))
        cats = list(train["primary_chronic" ].dropna().unique())
        data = [np.log1p(train.loc[train[ "primary_chronic" ]==c, "ed_cost_next3y_u
        plt.boxplot(data, labels=cats, showfliers= False)
        plt.title( "log1p(Target) by primary_chronic (fliers hidden)" )
        plt.ylabel( "log1p(ed_cost_next3y_usd)" )
        savefig(OUT_DIR / "target_by_chronic_box_log.png" )
    # key numeric relationships
    if "ed_cost_next3y_usd"  in train.columns:
        print_header( "BASIC RELATIONSHIPS (SCATTERS)" )
        if "prior_ed_cost_5y_usd"  in train.columns:
            x = train[ "prior_ed_cost_5y_usd" ].astype( float)
            y = train[ "ed_cost_next3y_usd" ].astype( float)
            plt.figure()
            plt.scatter(np.log1p(x), np.log1p(y), s= 10, alpha= 0.35)
            plt.title( "log1p(prior_ed_cost_5y_usd) vs log1p(target)" )
            plt.xlabel( "log1p(prior_ed_cost_5y_usd)" )
            plt.ylabel( "log1p(ed_cost_next3y_usd)" )
            savefig(OUT_DIR / "scatter_log_priorcost_vs_log_target.png" )
        if "prior_ed_visits_5y"  in train.columns:
            x = train[ "prior_ed_visits_5y" ].astype( float)
            y = train[ "ed_cost_next3y_usd" ].astype( float)
            plt.figure()
            plt.scatter(x, np.log1p(y), s= 10, alpha= 0.35)
            plt.title( "prior_ed_visits_5y vs log1p(target)" )
            plt.xlabel( "prior_ed_visits_5y" )
            plt.ylabel( "log1p(ed_cost_next3y_usd)" )
            savefig(OUT_DIR / "scatter_visits_vs_log_target.png" )

```


### Page 27

```text

        if "prior_ed_cost_5y_usd"  in train.columns and "prior_ed_visits_5y"  in tr
            cpv = train[ "prior_ed_cost_5y_usd" ].astype( float) / (train[ "prior_ed_v
            plt.figure()
            plt.hist(np.log1p(cpv), bins= 60)
            plt.title( "log1p(prior cost per visit) distribution" )
            plt.xlabel( "log1p(prior_ed_cost_5y_usd / prior_ed_visits_5y)" )
            plt.ylabel( "count")
            savefig(OUT_DIR / "prior_cost_per_visit_log_hist.png" )
    # correlation scan (numeric)
    if "ed_cost_next3y_usd"  in train.columns:
        print_header( "SPEARMAN CORRELATIONS (NUMERIC FEATURES)" )
        y = train[ "ed_cost_next3y_usd" ].astype( float)
        num_cols = [c for c in train.columns if c != "ed_cost_next3y_usd"  and pd.
        rows = []
        for c in num_cols:
            rows.append((c, spearman_corr(train[c], y)))
        corr_df = pd.DataFrame(rows, columns=[ "feature" , "spearman_r" ]).sort_valu
        print(corr_df.head( 20).to_string(index= False))
        corr_df.to_csv(OUT_DIR / "spearman_numeric_features.csv" , index= False)
    # ============================================================
    # RECEIPTS: coverage + parse + validation
    # ============================================================
    print_header( "RECEIPT PDF COVERAGE CHECK" )
    assert PDF_DIR.exists(), f"PDF_DIR not found: {PDF_DIR} "
    pdf_files = sorted(PDF_DIR.glob( "receipt_*.pdf" ))
    print("PDF files found:" , len(pdf_files), "in", PDF_DIR)
    # filename -> patient_id extraction (digits after receipt_)
    pid_to_pdf = {}
    extra_bad = []
    for p in pdf_files:
        m = re.search( r"receipt_(\d+)\.pdf$" , p.name)
        if not m:
            extra_bad.append(p.name)
            continue

```


### Page 28

```text

        pid_to_pdf[ int(m.group( 1))] = p
    all_ids = pd.concat([train[ "patient_id" ], test[ "patient_id" ]], axis= 0).astype
    missing_pdfs = [pid for pid in all_ids if pid not in pid_to_pdf]
    extra_pdfs = [pid for pid in pid_to_pdf.keys() if pid not in set(map(int, all_
    print("Missing PDFs for patient_ids:" , len(missing_pdfs))
    print("Extra PDFs not in train/test:" , len(extra_pdfs))
    (OUT_DIR / "missing_pdfs.txt" ).write_text( "\n".join(map(str, missing_pdfs[: 50
    (OUT_DIR / "extra_pdfs.txt" ).write_text( "\n".join(map(str, extra_pdfs[: 5000])
    # Parse receipts with cache
    parsed = None
    if RECEIPT_CACHE.exists():
        print("Loading receipt cache:" , RECEIPT_CACHE)
        parsed = load(RECEIPT_CACHE)
        print("Cached parsed receipts:" , len(parsed))
    else:
        print("Parsing receipts (threaded)... workers:" , N_WORKERS)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        items = [(pid, pid_to_pdf[pid]) for pid in pid_to_pdf.keys()]
        items = sorted(items, key= lambda x: x[0])
        if MAX_PDFS is not None:
            items = items[:MAX_PDFS]
        parsed = {}
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            futs = {ex.submit(parse_receipt, path): pid for pid, path in items}
            for fut in tqdm(as_completed(futs), total= len(futs), desc= "Parsing PD
                pid = futs[fut]
                try:
                    parsed[pid] = fut.result()
                except Exception as e:
                    parsed[pid] = {
                        "pdf_path" : str(pid_to_pdf.get(pid, "")),
                        "zip3_receipt" : None,
                        "insurance_receipt" : None,
                        "pdf_total" : None,

```


### Page 29

```text

                        "n_items_raw" : 0,
                        "items": [],
                        "text_len" : 0,
                        "parse_error" : repr(e),
                    }
        print(f"Parsed {len(parsed)} receipts in {time.time()-t0: .1f}s")
        dump(parsed, RECEIPT_CACHE)
        print("Saved receipt cache:" , RECEIPT_CACHE)
    # Build receipt feature table
    print_header( "RECEIPT FEATURE EXTRACTION" )
    rec_rows = []
    for pid, obj in parsed.items():
        feats = build_receipt_features(obj)
        feats[ "patient_id" ] = int(pid)
        rec_rows.append(feats)
    rec_feat = pd.DataFrame(rec_rows)
    print("Receipt feature df:" , rec_feat.shape)
    rec_feat.to_csv(OUT_DIR / "receipt_features_basic.csv" , index= False)
    # Merge receipt feats into train/test for EDA
    train2 = train.merge(rec_feat, on= "patient_id" , how="left")
    test2  = test.merge(rec_feat, on= "patient_id" , how="left")
    # Validate TOTAL alignment
    if "prior_ed_cost_5y_usd"  in train2.columns:
        print_header( "PDF TOTAL vs CSV prior_ed_cost_5y_usd VALIDATION" )
        for name, df in [("train", train2), ( "test", test2)]:
            prior = df[ "prior_ed_cost_5y_usd" ].astype( float)
            total = df[ "receipt_pdf_total" ].astype( float)
            diff = (total - prior). abs()
            ok = diff <= TOTAL_TOL
            match_rate = ok.mean()
            n_bad = int((~ok).sum())
            print(f"[{name}] match_rate= {match_rate: .4f} | mismatches= {n_bad}/{le
            # save worst mismatches
            bad = df.loc[~ok, [ "patient_id" , "prior_ed_cost_5y_usd" , "receipt_pdf_

```


### Page 30

```text

            if len(bad):
                bad[ "abs_diff" ] = (bad[ "receipt_pdf_total" ].astype( float) - bad[ "
                bad = bad.sort_values( "abs_diff" , ascending= False)
            bad.to_csv(OUT_DIR / f"pdf_total_mismatches_ {name}.csv", index= False)
            # plot diff
            plt.figure()
            plt.hist(diff.fillna(- 1), bins= 80)
            plt.title( f"abs(PDF_TOTAL - prior_ed_cost_5y_usd) — {name}")
            plt.xlabel( "abs diff (USD), NaN shown as -1" )
            plt.ylabel( "count")
            savefig(OUT_DIR / f"pdf_total_absdiff_hist_ {name}.png")
    # Receipt feature distributions (sanity)
    print_header( "RECEIPT FEATURE SANITY (DISTRIBUTIONS)" )
    for c in ["receipt_n_items" , "receipt_n_unique_codes" , "receipt_em_count" , "r
              "receipt_top1_share" , "receipt_bucket_entropy" ]:
        if c in train2.columns:
            s = train2[c]
            print(f"{c}: mean={pd.to_numeric(s, errors= 'coerce' ).mean(): .3f} | "
                  f"p50={pd.to_numeric(s, errors= 'coerce' ).quantile( 0.5):.3f} | "
                  f"p90={pd.to_numeric(s, errors= 'coerce' ).quantile( 0.9):.3f} | "
                  f"missing= {s.isna().mean(): .3f}")
            plt.figure()
            vals = pd.to_numeric(s, errors= "coerce" ).dropna()
            if len(vals):
                plt.hist(vals.clip(upper=vals.quantile( 0.995)), bins= 60)
                plt.title( f"{c} histogram (clipped at 99.5%)")
                plt.xlabel(c)
                plt.ylabel( "count")
                savefig(OUT_DIR / f"{c}_hist.png")
    # Correlate receipt features with target (train only)
    if "ed_cost_next3y_usd"  in train2.columns:
        print_header( "RECEIPT FEATURE ↔ TARGET (SPEARMAN)" )
        y = train2[ "ed_cost_next3y_usd" ].astype( float)
        receipt_cols = [c for c in train2.columns if c.startswith( "receipt_" ) and 
        rows = []
        for c in receipt_cols:

```


### Page 31

```text

            rows.append((c, spearman_corr(train2[c], y)))
        rc = pd.DataFrame(rows, columns=[ "receipt_feature" , "spearman_r" ]).sort_v
        print(rc.head( 20).to_string(index= False))
        rc.to_csv(OUT_DIR / "spearman_receipt_features.csv" , index= False)
    # ============================================================
    # Code lift analysis (train only)
    # ============================================================
    if "ed_cost_next3y_usd"  in train.columns:
        print_header( "CODE LIFT ANALYSIS (TRAIN ONLY)" )
        # patient_id -> set(codes)
        pid_to_codes = {}
        for pid, obj in parsed.items():
            codes = [it[ "code"] for it in obj.get( "items", [])]
            pid_to_codes[ int(pid)] = set(codes)
        train_pids = set(train["patient_id" ].astype( int).tolist())
        overall_mean = train[ "ed_cost_next3y_usd" ].astype( float).mean()
        # baseline residual control: (primary_chronic × prior_cost_bin) median
        tr = train2.copy()
        if "prior_ed_cost_5y_usd"  in tr.columns:
            tr[ "prior_cost_bin" ] = pd.qcut(tr[ "prior_ed_cost_5y_usd" ].astype( float
        else:
            tr[ "prior_cost_bin" ] = "all"
        group_keys = []
        if "primary_chronic"  in tr.columns:
            group_keys.append( "primary_chronic" )
        group_keys.append( "prior_cost_bin" )
        baseline = tr.groupby(group_keys)[ "ed_cost_next3y_usd" ].median().rename( "
        tr = tr.join(baseline, on=group_keys)
        tr[ "residual_vs_baseline" ] = tr["ed_cost_next3y_usd" ].astype( float) - tr[
        pid_to_y = dict(zip(tr["patient_id" ].astype( int), tr["ed_cost_next3y_usd"
        pid_to_resid = dict(zip(tr["patient_id" ].astype( int), tr["residual_vs_bas
        code_patients = defaultdict( list)
        for pid in train_pids:

```


### Page 32

```text

            for code in pid_to_codes.get(pid, set()):
                code_patients[code].append(pid)
        rows = []
        for code, pids in code_patients.items():
            n = len(pids)
            if n < CODE_MIN_SUPPORT:
                continue
            ys = np.array([pid_to_y[pid] for pid in pids], dtype= float)
            rs = np.array([pid_to_resid[pid] for pid in pids], dtype= float)
            rows.append({
                "code": code,
                "support_n" : n,
                "mean_target" : float(np.mean(ys)),
                "median_target" : float(np.median(ys)),
                "lift_mean_vs_overall" : float(np.mean(ys) - overall_mean),
                "mean_residual_vs_chronic_costbin_median" : float(np.mean(rs)),
            })
        code_lift = pd.DataFrame(rows)
        if len(code_lift):
            code_lift = code_lift.sort_values([ "mean_residual_vs_chronic_costbin_
            code_lift.to_csv(OUT_DIR / "code_lift_table.csv" , index= False)
            print("Top codes by residual lift (controls chronic + prior-cost-bin)
            print(code_lift.head(N_TOP_CODES).to_string(index= False))
            print("\nTop codes by raw mean target lift:" )
            code_lift2 = code_lift.sort_values([ "lift_mean_vs_overall" , "support_
            print(code_lift2.head(N_TOP_CODES).to_string(index= False))
        else:
            print("No codes met support threshold. Lower CODE_MIN_SUPPORT and re-
    # ============================================================
    # Tail comparisons (who drives MAE)
    # ============================================================
    if "ed_cost_next3y_usd"  in train2.columns:
        print_header( "TAIL COMPARISON (TOP 5% TARGET vs REST)" )
        y = train2[ "ed_cost_next3y_usd" ].astype( float)
        thr = y.quantile( 0.95)

```


### Page 33

```text

        top = train2[y >= thr].copy()
        rest = train2[y < thr].copy()
        print("Top5% threshold:" , thr, "| top_n:" , len(top), "| rest_n:" , len(res
        key_feats = [ "prior_ed_cost_5y_usd" , "prior_ed_visits_5y" ,
                     "receipt_n_items" , "receipt_n_unique_codes" , "receipt_em_cou
                     "receipt_em_max" , "receipt_top1_share" , "receipt_bucket_entr
        rows = []
        for f in key_feats:
            if f in train2.columns:
                a = pd.to_numeric(top[f], errors= "coerce" )
                b = pd.to_numeric(rest[f], errors= "coerce" )
                rows.append({
                    "feature" : f,
                    "top_mean" : float(a.mean()),
                    "rest_mean" : float(b.mean()),
                    "delta_mean" : float(a.mean() - b.mean()),
                    "top_p50" : float(a.quantile( 0.5)),
                    "rest_p50" : float(b.quantile( 0.5)),
                })
        tail_tbl = pd.DataFrame(rows).sort_values( "delta_mean" , ascending= False)
        print(tail_tbl.to_string(index= False))
        tail_tbl.to_csv(OUT_DIR / "tail_feature_comparison.csv" , index= False)
    # Save merged EDA tables for inspection
    train2.to_csv(OUT_DIR / "train_with_receipt_features.csv" , index= False)
    test2.to_csv(OUT_DIR / "test_with_receipt_features.csv" , index= False)
    print_header( "DONE")
    print("Outputs saved to:" , OUT_DIR)
    print("Key files:" )
    print(" - target plots: target_hist_clipped.png, target_log_hist.png" )
    print(" - pdf validation: pdf_total_absdiff_hist_train/test.png + pdf_total_m
    print(" - receipt features: receipt_features_basic.csv" )
    print(" - correlations: spearman_numeric_features.csv, spearman_receipt_featu
    print(" - code lift: code_lift_table.csv" )
    print(" - tail comparison: tail_feature_comparison.csv" )

```


### Page 34

```text

What I want y ou to send b ack af ter you run it (Phase 2)
Just paste:
The printed summary sections (target quantiles, chronic summary, PDF match rates).
The top 15 r ows of spearman_receipt_features.csv
The top 20 codes  from code_lift_table.csv  (residual-lift sorted)
Any surprising plots (or tell me which plot filenames look odd)
Then I’ll refine the feature hypotheses (what to keep, what to drop, what to stratify) before
we touch any model.if __name__ == "__main__" :
    main()
Sour ces
Sour ces
got bug:
-----------------------------------------------------------------
----------
AttributeError                            T raceback (most recent call  
last)
Cell In[5], line 715
    711     print(" - tail comparison:  
tail_feature_comparison.csv")
    714 if __name__ == "__main__":
--> 715     main()
Cell In[5], line 529, in main()
    527 rec_rows = []
    528 for pid, obj in parsed.items():
--> 529     feats = build_receipt_features(obj)
    530     feats["patient_id"] = int(pid)
    531     rec_rows.append(feats)
Cell In[5], line 246, in build_receipt_features(parsed)
    245 def build_receipt_features(parsed: dict):
--> 246     items = parsed.get("items", [])

```


### Page 35

```text

    247     if not items:
    248         return {
    249             "receipt_parse_failed": 1,
    250             "receipt_n_items": 0,
   (...)    262             "receipt_pdf_total":  
parsed.get("pdf_total"),
    263         }
AttributeError: 'int' object has no attribute 'get'
and unfinished log:
======================================
======================================
==============
LOAD D ATA
======================================
======================================
==============
Train shape: (2000, 5)
Test  shape: (2000, 4)
Train cols: ['patient_id', 'primary_chronic',  
'prior_ed_visits_5y', 'prior_ed_cost_5y_usd',  
'ed_cost_next3y_usd']
Patients shape: (4000, 5)
Patients cols: ['patient_id', 'age', 'sex', 'insurance', 'zip3']
Duplicate patient_id — train: 0 | test: 0
Train/T est patient_id overlap: 0
======================================
======================================
==============
MISSINGNESS (T OP 30)
======================================
======================================
==============
patient_id              0.0
primary_chronic         0.0
prior_ed_visits_5y      0.0
prior_ed_cost_5y_usd    0.0
ed_cost_next3y_usd      0.0

```


### Page 36

```text

age                     0.0
sex                     0.0
insurance               0.0
zip3                    0.0
======================================
======================================
==============
TARGET DISTRIBUTION
======================================
======================================
==============
0.00      306.8800
0.01      991.4990
0.05     1553.6960
0.10     1867.5730
0.25     2548.7675
0.50     3569.0950
0.75     4956.4225
0.90     6636.4750
0.95     7541.8475
0.99     8801.0213
1.00    11184.6100
Mean: 3908.25191 S td: 1822.401920306473 Skew(approx):  
0.816103887108234
Top 5% of patients contribute ~0.107 of total target spend  
(tail concentration proxy)
======================================
======================================
==============
TARGET B Y PRIMAR Y_CHR ONIC
======================================
======================================
==============
                 count         mean   median       p90       p95         
p99
primary_chronic                                                            
HF                 873  4720.807239  4377.79  7522.504   
8191.498  9468.0928

```


### Page 37

```text

DiabetesComp       725  3516.845283  3113.74  5866.244   
6705.846  8029.7624
Pneumonia          402  2849.567836  2590.54  4936.438   
5454.448  6177.9207
======================================
======================================
==============
BASIC REL ATIONSHIPS (SCA TTERS)
======================================
======================================
==============
======================================
======================================
==============
SPEARMAN C ORREL ATIONS (NUMERIC FEA TURES)
======================================
======================================
==============
             feature  spearman_r
prior_ed_cost_5y_usd    0.882309
  prior_ed_visits_5y    0.697989
          patient_id    0.056849
                 age    0.039905
                zip3    0.020266
======================================
======================================
==============
RECEIPT PDF C OVERA GE CHECK
======================================
======================================
==============
PDF files found: 4000 in  
D:\AgentDs\agent_ds_healthcare\receipts_pdf
Missing PDFs for patient_ids: 0
Extra PDFs not in train/test: 0
Loading receipt cache:  
D:\AgentDs\agent_ds_healthcare\cache_iter10\receipts_par

```


### Page 38

```text

sed.joblib
Cached parsed receipts: 2
======================================
======================================
==============
RECEIPT FEA TURE EX TRACTION
======================================
======================================
==============
You’re loading a stale / wr ong-format r eceipt cache .
Two smoking guns from your log:
Cached parsed receipts: 2  → you should have ~4000 (train+test). So the cache is
not the current parse output.
The crash says obj is an int → meaning parsed[pid]  is not a dict  (expected), so
build_receipt_features(obj)  fails on .get .
joblib.load()  will simply reconstruct what ever object was pr eviously dumped  at that
path. 
So your receipts_parsed.joblib  likely came from an earlier experiment (or a partial run)
and isn’t the dict-of-dicts this script expects.
Fast fix (do this once)
Delete or rename this file and rerun:
D:\AgentDs\agent_ds_healthcare\cache_iter10\receipts_parsed.joblib
That forces a full re-parse and should clear the bug.
Robust fix (r ecommended): aut o-det ect b ad cache + r ebuild
1) Add a cache v alidat or + “for ce reparse” flagjoblib.readthedo …+1

```


### Page 39

```text

Paste this near your config section:
2) Replace y our “lo ad cache” block with this
Find the part:
Replace it with:python
FORCE_REPARSE = False  # set True for one run if you want to force rebuild
def cache_is_valid (parsed_obj, expected_n: int) -> bool:
    if not isinstance (parsed_obj, dict):
        return False
    if len(parsed_obj) < int(0.98 * expected_n):  # allow a few failures
        return False
    # spot-check value structure
    for v in parsed_obj.values():
        if not isinstance (v, dict):
            return False
        if ("items" not in v) or ("pdf_total"  not in v):
            return False
        break
    return True
Copy codepython
if RECEIPT_CACHE.exists():
    print("Loading receipt cache:" , RECEIPT_CACHE)
    parsed = load(RECEIPT_CACHE)
    print("Cached parsed receipts:" , len(parsed))
else:
    ...
Copy codepython
expected_n = len(pid_to_pdf)
parsed = None
if RECEIPT_CACHE.exists() and (not FORCE_REPARSE):
    print("Loading receipt cache:" , RECEIPT_CACHE)

```


### Page 40

```text

    parsed = load(RECEIPT_CACHE)
    print("Cached parsed receipts:" , len(parsed) if hasattr(parsed, "__len__" ) el
    if not cache_is_valid(parsed, expected_n):
        print("⚠ Receipt cache INVALID (wrong type/size/structure). Rebuilding c
        # backup the old cache so you can inspect later
        backup = RECEIPT_CACHE.with_suffix( f".bad_{int(time.time())}.joblib")
        try:
            RECEIPT_CACHE.rename(backup)
            print("Backed up bad cache to:" , backup)
        except Exception as e:
            print("Could not rename bad cache (will overwrite):" , e)
        parsed = None
if parsed is None:
    print("Parsing receipts (threaded)... workers:" , N_WORKERS)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    items = [(pid, pid_to_pdf[pid]) for pid in pid_to_pdf.keys()]
    items = sorted(items, key= lambda x: x[0])
    if MAX_PDFS is not None:
        items = items[:MAX_PDFS]
    parsed = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(parse_receipt, path): pid for pid, path in items}
        for fut in tqdm(as_completed(futs), total= len(futs), desc= "Parsing PDFs" )
            pid = futs[fut]
            try:
                parsed[pid] = fut.result()
            except Exception as e:
                parsed[pid] = {
                    "pdf_path" : str(pid_to_pdf.get(pid, "")),
                    "zip3_receipt" : None,
                    "insurance_receipt" : None,
                    "pdf_total" : None,
                    "n_items_raw" : 0,
                    "items": [],
                    "text_len" : 0,

```


### Page 41

```text

3) Har den build_receipt_features (so it nev er crashes)
Change the start of build_receipt_features  to:
Optional impr ovement (p arsing stability)
If you later see “few line items parsed” for many PDFs, switch to:                    "parse_error" : repr(e),
                }
    print(f"Parsed {len(parsed)} receipts in {time.time()-t0: .1f}s")
    dump(parsed, RECEIPT_CACHE)
    print("Saved receipt cache:" , RECEIPT_CACHE)
python
def build_receipt_features (parsed: dict):
    if not isinstance (parsed, dict):
        return {
            "receipt_parse_failed" : 1,
            "receipt_n_items" : 0,
            "receipt_n_unique_codes" : 0,
            "receipt_sum_items" : 0.0,
            "receipt_top1_share" : np.nan,
            "receipt_top3_share" : np.nan,
            "receipt_em_count" : 0,
            "receipt_em_max" : np.nan,
            "receipt_em_mean" : np.nan,
            "receipt_non_em_count" : 0,
            "receipt_bucket_entropy" : np.nan,
            "receipt_zip3_from_pdf" : None,
            "receipt_ins_from_pdf" : None,
            "receipt_pdf_total" : np.nan,
        }
    items = parsed.get( "items", [])
    ...

```


### Page 42

```text

PyMuPDF notes this can improve reading order consistency. 
Quick int erpretation o f your p artial ED A log (useful signal alr eady)
Target is moderat ely sk ewed (skew ~0.82), not insanely heavy-tailed.
HF has the highest futur e ED cost  vs DiabetesComp vs Pneumonia (clear cohort
separation).
Baseline predictors are strong:
prior_ed_cost_5y_usd  Spearman ~0.88, prior_ed_visits_5y  ~0.70 → receipts need
to add orthogonal  signal (acuity/procedure mix, concentration, code diversity), not
“more of the same”.
What t o do now
1. Apply the robust cache fix above (or delete the cache file once).
2. Re-run ED A.
3. Send me:
PDF TOTAL vs CSV prior... match_rate  for train/test
Top 15 of spearman_receipt_features.csv
Top 20 rows of code_lift_table.csv  (residual-lift sorted)
Then we’ll refine which receipt features are genuinely predictive (and which are just proxies
for prior cost).
Copy codeCopy codepython
page.get_text( "text", sort=True)
Copy codepymupdf.readth …
Sour ces
Sour ces
I am in a jupyter notebook and I want you to give me the  
new full code I can copy into 1 cell

```


### Page 43

```text

Here’s a single-cell  notebook version that auto-det ects & quarantines a b ad joblib
cache  (since joblib.dump/load  can persist any Python object) and rebuilds r eceipts when
needed . 
It also uses page.get_text("text", sort=True)  for more stable reading order during text
extraction. Joblib
PyMuPDF
python
# ============================================================
# AgentDS Challenge 2 — Phase 1 EDA (CSV + PDF receipts) — ONE CELL
# Jupyter-friendly: thread parsing + robust cache validation
# ============================================================
import os
import re
import time
import math
import json
import warnings
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump, load
# -----------------------------
# Optional deps
# -----------------------------
try:
    import fitz  # PyMuPDF
except Exception as e:
    raise RuntimeError( "PyMuPDF missing. Install: pip install pymupdf" ) from e
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs ):  # fallback

```


### Page 44

```text

        return x
warnings.filterwarnings( "ignore" )
# -----------------------------
# USER CONFIG (as provided)
# -----------------------------
BASE_DIR = Path( r"D:\AgentDs\agent_ds_healthcare" )
TRAIN_CSV = BASE_DIR / "ed_cost_train.csv"
TEST_CSV  = BASE_DIR / "ed_cost_test.csv"
PATIENTS_CSV = BASE_DIR / "patients.csv"
PDF_DIR = BASE_DIR / "receipts_pdf"
SUBMISSION_PATH = BASE_DIR / "submission_ICHI_V1.csv"
CACHE_DIR = BASE_DIR / "cache_iter10"
CACHE_DIR.mkdir(parents= True, exist_ok= True)
# NOTE: if you set MAX_PDFS for a quick run, we automatically write to a separate 
MAX_PDFS = None          # set to int for quick dry run (e.g., 200). None = parse 
FORCE_REPARSE = False    # set True to force rebuilding cache once
RECEIPT_CACHE = CACHE_DIR / ( "receipts_parsed.joblib"  if MAX_PDFS is None else f"
OUT_DIR = CACHE_DIR / "eda_outputs_iter10"
OUT_DIR.mkdir(parents= True, exist_ok= True)
# Analysis knobs
N_TOP_CODES = 40
CODE_MIN_SUPPORT = 20
TOTAL_TOL = 0.01
# Threaded parsing (Windows/Jupyter friendly)
N_WORKERS = max(4, (os.cpu_count() or 8) - 2)
# ============================================================
# Helpers
# ============================================================
def _normalize_cols (df: pd.DataFrame ) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

```


### Page 45

```text

    return df
def _ensure_col_alias (df: pd.DataFrame, aliases: dict) -> pd.DataFrame:
    df = df.copy()
    cols = set(df.columns)
    for canon, alts in aliases.items():
        if canon in cols:
            continue
        for a in alts:
            if a in cols:
                df = df.rename(columns={a: canon})
                cols = set(df.columns)
                break
    return df
def print_header (title: str):
    print("\n" + "="*90)
    print(title)
    print("="*90)
def savefig(path: Path ):
    plt.tight_layout()
    plt.savefig(path, dpi= 140)
    plt.close()
def safe_float (x):
    if x is None:
        return np.nan
    if isinstance (x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    s = s.replace( "$", "").replace( ",", "")
    try:
        return float(s)
    except Exception:
        return np.nan
def safe_int (x):
    try:
        s = str(x).strip()

```


### Page 46

```text

        if s == "" or s.lower() == "nan":
            return np.nan
        return int(float(s))
    except Exception:
        return np.nan
def spearman_corr (a, b):
    a = pd.Series(a).astype( float)
    b = pd.Series(b).astype( float)
    m = a.notna() & b.notna()
    if m.sum() < 3:
        return np.nan
    return a[m].rank().corr(b[m].rank())
def cache_is_valid (parsed_obj, expected_n: int) -> bool:
    """Validate that cache is dict[int -> dict with required keys], and roughly c
    if not isinstance (parsed_obj, dict):
        return False
    if expected_n is not None:
        if len(parsed_obj) < int(0.98 * expected_n):
            return False
    # spot-check a few values
    checked = 0
    for v in parsed_obj.values():
        checked += 1
        if not isinstance (v, dict):
            return False
        if ("items" not in v) or ("pdf_total"  not in v) or ("text_len"  not in v):
            return False
        if checked >= 3:
            break
    return True
# ============================================================
# Receipt parsing
# ============================================================
# IMPORTANT: your synthetic ZIP3 can be 1–3 digits (e.g., "21"), so allow 1–3 dig
ZIP3_RE  = re. compile(r"\bzip3\b\s*[:\-]?\s*(\d{1,3})" , re.IGNORECASE)
INS_RE   = re. compile(r"\binsurance\b\s*[:\-]?\s*([A-Za-z_]+)" , re.IGNORECASE)
TOTAL_RE = re. compile(r"\btotal\b\s*[:\-]?\s*\$?\s*([\d,]+(?:\.\d{2})?)" , re.IGNO

```


### Page 47

```text

# CPT/HCPCS-ish codes
CODE_RE = re. compile(r"^[A-Z]?\d{4,5}$|^[A-Z0-9]{4,7}$" )
def extract_pdf_text (pdf_path: Path ) -> str:
    doc = fitz. open(pdf_path)
    try:
        parts = []
        for i in range(doc.page_count):
            page = doc.load_page(i)
            # sort=True improves reading order consistency for many PDFs
            parts.append(page.get_text( "text", sort=True))
        return "\n".join(parts)
    finally:
        doc.close()
def parse_line_item (line: str):
    """
    Heuristic parse:
        <CODE> <DESCRIPTION...> <QTY> <AMOUNT>
    Example:
        99283 Emergency Department Visit (Level 3) 1 245.00
    """
    line = line.strip()
    if not line:
        return None
    # Must end with an amount like 123.45
    m_amt = re.search( r"(\d[\d,]*\.\d{2})\s*$" , line)
    if not m_amt:
        return None
    amt = safe_float(m_amt.group( 1))
    left = line[:m_amt.start()].strip()
    # qty should end the left side
    m_qty = re.search( r"(\d+)\s*$" , left)
    if not m_qty:
        return None
    qty = int(m_qty.group( 1))
    left2 = left[:m_qty.start()].strip()

```


### Page 48

```text

    if not left2:
        return None
    code = left2.split()[ 0].strip()
    if not CODE_RE. match(code):
        return None
    desc = left2[ len(code):].strip()
    return {"code": code, "desc": desc, "qty": qty, "amount" : float(amt)}
def parse_receipt (pdf_path: Path ):
    text = extract_pdf_text(pdf_path)
    zip3 = None
    m = ZIP3_RE.search(text)
    if m:
        zip3 = m.group( 1)
    insurance = None
    m = INS_RE.search(text)
    if m:
        insurance = m.group( 1).lower()
    total = None
    totals = TOTAL_RE.findall(text)
    if totals:
        total = safe_float(totals[- 1])  # use last TOTAL occurrence
    items = []
    for raw in text.splitlines():
        it = parse_line_item(raw)
        if it is not None:
            items.append(it)
    return {
        "pdf_path" : str(pdf_path),
        "zip3_receipt" : zip3,
        "insurance_receipt" : insurance,
        "pdf_total" : total,
        "n_items_raw" : len(items),

```


### Page 49

```text

        "items": items,
        "text_len" : len(text),
    }
# ============================================================
# Receipt features (phenotypes)
# ============================================================
def em_level_from_code (code: str):
    # ED E/M codes often 99281–99285
    if not isinstance (code, str):
        return None
    if code.isdigit() and code.startswith( "9928") and len(code) == 5:
        last = int(code[-1])
        if last in [1,2,3,4,5]:
            return last
    return None
def code_bucket (code: str):
    if not isinstance (code, str) or not code:
        return "unknown"
    if code.isdigit():
        if code.startswith( "99"):
            return "em_99xxx"
        return f"num_{code[0]}xxxx"
    return f"alpha_ {code[0].upper()}"
def build_receipt_features (parsed_obj ):
    # HARDEN: never crash if cache contains unexpected values
    if not isinstance (parsed_obj, dict):
        return {
            "receipt_parse_failed" : 1,
            "receipt_n_items" : 0,
            "receipt_n_unique_codes" : 0,
            "receipt_sum_items" : 0.0,
            "receipt_top1_share" : np.nan,
            "receipt_top3_share" : np.nan,
            "receipt_em_count" : 0,
            "receipt_em_max" : np.nan,
            "receipt_em_mean" : np.nan,
            "receipt_non_em_count" : 0,

```


### Page 50

```text

            "receipt_bucket_entropy" : np.nan,
            "receipt_zip3_from_pdf" : None,
            "receipt_ins_from_pdf" : None,
            "receipt_pdf_total" : np.nan,
        }
    items = parsed_obj.get( "items", [])
    if not items:
        return {
            "receipt_parse_failed" : 1,
            "receipt_n_items" : 0,
            "receipt_n_unique_codes" : 0,
            "receipt_sum_items" : 0.0,
            "receipt_top1_share" : np.nan,
            "receipt_top3_share" : np.nan,
            "receipt_em_count" : 0,
            "receipt_em_max" : np.nan,
            "receipt_em_mean" : np.nan,
            "receipt_non_em_count" : 0,
            "receipt_bucket_entropy" : np.nan,
            "receipt_zip3_from_pdf" : parsed_obj.get( "zip3_receipt" ),
            "receipt_ins_from_pdf" : parsed_obj.get( "insurance_receipt" ),
            "receipt_pdf_total" : parsed_obj.get( "pdf_total" ),
        }
    amounts = np.array([it.get( "amount" , np.nan) for it in items], dtype= float)
    codes = [it.get( "code", "") for it in items]
    uniq = len(set(codes))
    s = float(np.nansum(amounts))
    top_sorted = np.sort(amounts[~np.isnan(amounts)])[::- 1]
    top1 = float(top_sorted[ 0]) if len(top_sorted) else np.nan
    top3 = float(top_sorted[: 3].sum()) if len(top_sorted) >= 3 else float(top_sor
    top1_share = (top1 / s) if (s and s > 0 and not np.isnan(top1)) else np.nan
    top3_share = (top3 / s) if (s and s > 0 and not np.isnan(top3)) else np.nan
    em_levels = [em_level_from_code(c) for c in codes]
    em_levels = [x for x in em_levels if x is not None]
    em_count = len(em_levels)
    em_max = float(np.max(em_levels)) if em_levels else np.nan

```


### Page 51

```text

    em_mean = float(np.mean(em_levels)) if em_levels else np.nan
    non_em_count = sum(1 for c in codes if not (isinstance (c, str) and c.isdigit(
    buckets = [code_bucket(c) for c in codes]
    bc = Counter(buckets)
    probs = np.array( list(bc.values()), dtype= float)
    probs = probs / probs. sum()
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    return {
        "receipt_parse_failed" : 0,
        "receipt_n_items" : len(items),
        "receipt_n_unique_codes" : uniq,
        "receipt_sum_items" : s,
        "receipt_top1_share" : top1_share,
        "receipt_top3_share" : top3_share,
        "receipt_em_count" : em_count,
        "receipt_em_max" : em_max,
        "receipt_em_mean" : em_mean,
        "receipt_non_em_count" : non_em_count,
        "receipt_bucket_entropy" : entropy,
        "receipt_zip3_from_pdf" : parsed_obj.get( "zip3_receipt" ),
        "receipt_ins_from_pdf" : parsed_obj.get( "insurance_receipt" ),
        "receipt_pdf_total" : parsed_obj.get( "pdf_total" ),
    }
# ============================================================
# Main EDA
# ============================================================
def main():
    print_header( "LOAD DATA" )
    train = _normalize_cols(pd.read_csv(TRAIN_CSV))
    test  = _normalize_cols(pd.read_csv(TEST_CSV))
    aliases = {
        "patient_id" : ["patientid" , "id"],
        "primary_chronic" : ["chronic" , "primarycondition" ],
        "prior_ed_visits_5y" : ["prior_visits_5y" , "ed_visits_5y" ],
        "prior_ed_cost_5y_usd" : ["prior_cost_5y_usd" , "ed_cost_5y" , "prior_cost_5y

```


### Page 52

```text

        "ed_cost_next3y_usd" : ["target" , "y", "cost_next3y" ],
        "zip3": ["zip", "zip_3", "zip_prefix" , "zip3_code" , "zip3"],
        "insurance" : ["payer", "insurance_type" ],
    }
    train = _ensure_col_alias(train, aliases)
    test  = _ensure_col_alias(test, aliases)
    print("Train shape:" , train.shape)
    print("Test  shape:" , test.shape)
    print("Train cols:" , list(train.columns))
    # patients
    patients = None
    if PATIENTS_CSV.exists():
        patients = _normalize_cols(pd.read_csv(PATIENTS_CSV))
        patients = _ensure_col_alias(patients, aliases)
        print("Patients shape:" , patients.shape)
        print("Patients cols:" , list(patients.columns))
    else:
        print("WARNING: patients.csv not found at" , PATIENTS_CSV)
    assert "patient_id"  in train.columns and "patient_id"  in test.columns, "Missi
    dup_tr = train[ "patient_id" ].duplicated(). sum()
    dup_te = test[ "patient_id" ].duplicated(). sum()
    print(f"Duplicate patient_id — train: {dup_tr}  | test: {dup_te} ")
    overlap = set(train["patient_id" ]).intersection( set(test["patient_id" ]))
    print("Train/Test patient_id overlap:" , len(overlap))
    # merge patients
    if patients is not None and "patient_id"  in patients.columns:
        train = train.merge(patients, on= "patient_id" , how="left", suffixes=( "", 
        test  = test.merge(patients, on= "patient_id" , how="left", suffixes=( "", "_
    # missingness
    print_header( "MISSINGNESS (TOP 30)" )
    miss = train.isna().mean().sort_values(ascending= False).head(30)
    print(miss.to_string())

```


### Page 53

```text

    # target profile
    if "ed_cost_next3y_usd"  in train.columns:
        y = train[ "ed_cost_next3y_usd" ].astype( float)
        print_header( "TARGET DISTRIBUTION" )
        qs = y.quantile([ 0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1.0])
        print(qs.to_string())
        print("Mean:", y.mean(), "Std:", y.std(), "Skew:", y.skew())
        # histogram raw
        plt.figure()
        plt.hist(y.clip(upper=y.quantile( 0.995)), bins= 60)
        plt.title( "Target histogram (clipped at 99.5%)" )
        plt.xlabel( "ed_cost_next3y_usd" )
        plt.ylabel( "count")
        savefig(OUT_DIR / "target_hist_clipped.png" )
        # histogram log
        plt.figure()
        plt.hist(np.log1p(y), bins= 60)
        plt.title( "log1p(Target) histogram" )
        plt.xlabel( "log1p(ed_cost_next3y_usd)" )
        plt.ylabel( "count")
        savefig(OUT_DIR / "target_log_hist.png" )
        top = y.quantile( 0.95)
        tail_share = y[y >= top]. sum() / y.sum()
        print(f"Top 5% contribute ~ {tail_share: .3f} of total target spend")
    # chronic summary
    if "primary_chronic"  in train.columns and "ed_cost_next3y_usd"  in train.colum
        print_header( "TARGET BY PRIMARY_CHRONIC" )
        g = train.groupby( "primary_chronic" )["ed_cost_next3y_usd" ]
        summary = g.agg([ "count", "mean", "median" ])
        summary[ "p90"] = g.quantile( 0.90)
        summary[ "p95"] = g.quantile( 0.95)
        summary[ "p99"] = g.quantile( 0.99)
        print(summary.sort_values( "mean", ascending= False).to_string())
        plt.figure(figsize=( 8,4))
        cats = list(train["primary_chronic" ].dropna().unique())

```


### Page 54

```text

        data = [np.log1p(train.loc[train[ "primary_chronic" ]==c, "ed_cost_next3y_u
        plt.boxplot(data, labels=cats, showfliers= False)
        plt.title( "log1p(Target) by primary_chronic (fliers hidden)" )
        plt.ylabel( "log1p(ed_cost_next3y_usd)" )
        savefig(OUT_DIR / "target_by_chronic_box_log.png" )
    # basic relationships
    if "ed_cost_next3y_usd"  in train.columns:
        print_header( "BASIC RELATIONSHIPS (SCATTERS)" )
        if "prior_ed_cost_5y_usd"  in train.columns:
            x = train[ "prior_ed_cost_5y_usd" ].astype( float)
            y = train[ "ed_cost_next3y_usd" ].astype( float)
            plt.figure()
            plt.scatter(np.log1p(x), np.log1p(y), s= 10, alpha= 0.35)
            plt.title( "log1p(prior_ed_cost_5y_usd) vs log1p(target)" )
            plt.xlabel( "log1p(prior_ed_cost_5y_usd)" )
            plt.ylabel( "log1p(ed_cost_next3y_usd)" )
            savefig(OUT_DIR / "scatter_log_priorcost_vs_log_target.png" )
        if "prior_ed_visits_5y"  in train.columns:
            x = train[ "prior_ed_visits_5y" ].astype( float)
            y = train[ "ed_cost_next3y_usd" ].astype( float)
            plt.figure()
            plt.scatter(x, np.log1p(y), s= 10, alpha= 0.35)
            plt.title( "prior_ed_visits_5y vs log1p(target)" )
            plt.xlabel( "prior_ed_visits_5y" )
            plt.ylabel( "log1p(ed_cost_next3y_usd)" )
            savefig(OUT_DIR / "scatter_visits_vs_log_target.png" )
        if "prior_ed_cost_5y_usd"  in train.columns and "prior_ed_visits_5y"  in tr
            cpv = train[ "prior_ed_cost_5y_usd" ].astype( float) / (train[ "prior_ed_v
            plt.figure()
            plt.hist(np.log1p(cpv), bins= 60)
            plt.title( "log1p(prior cost per visit) distribution" )
            plt.xlabel( "log1p(prior_ed_cost_5y_usd / prior_ed_visits_5y)" )
            plt.ylabel( "count")
            savefig(OUT_DIR / "prior_cost_per_visit_log_hist.png" )
    # numeric correlations
    if "ed_cost_next3y_usd"  in train.columns:

```


### Page 55

```text

        print_header( "SPEARMAN CORRELATIONS (NUMERIC FEATURES)" )
        y = train[ "ed_cost_next3y_usd" ].astype( float)
        num_cols = [c for c in train.columns if c != "ed_cost_next3y_usd"  and pd.
        rows = [(c, spearman_corr(train[c], y)) for c in num_cols]
        corr_df = pd.DataFrame(rows, columns=[ "feature" , "spearman_r" ]).sort_valu
        print(corr_df.head( 20).to_string(index= False))
        corr_df.to_csv(OUT_DIR / "spearman_numeric_features.csv" , index= False)
    # ============================================================
    # RECEIPTS: coverage + parse + validation
    # ============================================================
    print_header( "RECEIPT PDF COVERAGE CHECK" )
    assert PDF_DIR.exists(), f"PDF_DIR not found: {PDF_DIR} "
    pdf_files = sorted(PDF_DIR.glob( "receipt_*.pdf" ))
    print("PDF files found:" , len(pdf_files), "in", PDF_DIR)
    pid_to_pdf = {}
    bad_names = []
    for p in pdf_files:
        m = re.search( r"receipt_(\d+)\.pdf$" , p.name)
        if not m:
            bad_names.append(p.name)
            continue
        pid_to_pdf[ int(m.group( 1))] = p
    all_ids = pd.concat([train[ "patient_id" ], test[ "patient_id" ]], axis= 0).astype
    missing_pdfs = [pid for pid in all_ids if pid not in pid_to_pdf]
    extra_pdfs = [pid for pid in pid_to_pdf.keys() if pid not in set(map(int, all_
    print("Missing PDFs for patient_ids:" , len(missing_pdfs))
    print("Extra PDFs not in train/test:" , len(extra_pdfs))
    (OUT_DIR / "missing_pdfs.txt" ).write_text( "\n".join(map(str, missing_pdfs[: 50
    (OUT_DIR / "extra_pdfs.txt" ).write_text( "\n".join(map(str, extra_pdfs[: 5000])
    # Parse receipts with cache validation
    expected_n = len(all_ids) if MAX_PDFS is None else min(MAX_PDFS, len(all_ids)
    parsed = None
    if RECEIPT_CACHE.exists() and (not FORCE_REPARSE):

```


### Page 56

```text

        print("Loading receipt cache:" , RECEIPT_CACHE)
        parsed = load(RECEIPT_CACHE)
        print("Cached parsed receipts:" , len(parsed) if hasattr(parsed, "__len__"
        if not cache_is_valid(parsed, expected_n if MAX_PDFS is None else None):
            # If MAX_PDFS is set, we don't enforce size; we enforce structure only
            if MAX_PDFS is not None:
                ok_struct = cache_is_valid(parsed, None)
            else:
                ok_struct = False
            if not ok_struct:
                print("⚠ Receipt cache INVALID (wrong type/structure/size). Rebu
                backup = RECEIPT_CACHE.with_suffix( f".bad_{int(time.time())}.jobl
                try:
                    RECEIPT_CACHE.rename(backup)
                    print("Backed up bad cache to:" , backup)
                except Exception as e:
                    print("Could not rename bad cache (will overwrite):" , e)
                parsed = None
    if parsed is None:
        print("Parsing receipts (threaded)... workers:" , N_WORKERS)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        items = [(pid, pid_to_pdf[pid]) for pid in pid_to_pdf.keys()]
        items = sorted(items, key= lambda x: x[0])
        if MAX_PDFS is not None:
            items = items[:MAX_PDFS]
        parsed = {}
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            futs = {ex.submit(parse_receipt, path): pid for pid, path in items}
            for fut in tqdm(as_completed(futs), total= len(futs), desc= "Parsing PD
                pid = futs[fut]
                try:
                    parsed[pid] = fut.result()
                except Exception as e:
                    parsed[pid] = {
                        "pdf_path" : str(pid_to_pdf.get(pid, "")),

```


### Page 57

```text

                        "zip3_receipt" : None,
                        "insurance_receipt" : None,
                        "pdf_total" : None,
                        "n_items_raw" : 0,
                        "items": [],
                        "text_len" : 0,
                        "parse_error" : repr(e),
                    }
        print(f"Parsed {len(parsed)} receipts in {time.time()-t0: .1f}s")
        dump(parsed, RECEIPT_CACHE)
        print("Saved receipt cache:" , RECEIPT_CACHE)
    # Build receipt feature table
    print_header( "RECEIPT FEATURE EXTRACTION" )
    rec_rows = []
    for pid, obj in parsed.items():
        feats = build_receipt_features(obj)
        feats[ "patient_id" ] = int(pid)
        rec_rows.append(feats)
    rec_feat = pd.DataFrame(rec_rows)
    print("Receipt feature df:" , rec_feat.shape)
    rec_feat.to_csv(OUT_DIR / "receipt_features_basic.csv" , index= False)
    # Merge into train/test
    train2 = train.merge(rec_feat, on= "patient_id" , how="left")
    test2  = test.merge(rec_feat, on= "patient_id" , how="left")
    # Validate TOTAL alignment (train/test)
    if "prior_ed_cost_5y_usd"  in train2.columns:
        print_header( "PDF TOTAL vs CSV prior_ed_cost_5y_usd VALIDATION" )
        for name, df in [("train", train2), ( "test", test2)]:
            prior = df[ "prior_ed_cost_5y_usd" ].astype( float)
            total = pd.to_numeric(df[ "receipt_pdf_total" ], errors= "coerce" )
            diff = (total - prior). abs()
            ok = diff <= TOTAL_TOL
            match_rate = float(ok.mean())
            n_bad = int((~ok).sum())
            print(f"[{name}] match_rate= {match_rate: .4f} | mismatches= {n_bad}/{le

```


### Page 58

```text

            bad = df.loc[~ok, [ "patient_id" , "prior_ed_cost_5y_usd" , "receipt_pdf_
            if len(bad):
                bad[ "abs_diff" ] = (pd.to_numeric(bad[ "receipt_pdf_total" ], errors
                bad = bad.sort_values( "abs_diff" , ascending= False)
            bad.to_csv(OUT_DIR / f"pdf_total_mismatches_ {name}.csv", index= False)
            plt.figure()
            plt.hist(diff.fillna(- 1), bins= 80)
            plt.title( f"abs(PDF_TOTAL - prior_ed_cost_5y_usd) — {name}")
            plt.xlabel( "abs diff (USD), NaN shown as -1" )
            plt.ylabel( "count")
            savefig(OUT_DIR / f"pdf_total_absdiff_hist_ {name}.png")
    # Integrity check: receipt ZIP3 / insurance vs patients.csv (if present)
    if "zip3" in train2.columns and "receipt_zip3_from_pdf"  in train2.columns:
        print_header( "ZIP3 CONSISTENCY: patients.csv vs receipt PDFs (TRAIN)" )
        a = train2[ "zip3"].apply(safe_int)
        b = train2[ "receipt_zip3_from_pdf" ].apply(safe_int)
        m = (~pd.isna(a)) & (~pd.isna(b))
        if m.sum() > 0:
            match = float((a[m] == b[m]).mean())
            print(f"ZIP3 match rate where both present: {match:.4f} (n={int(m.sum
        else:
            print("No comparable ZIP3 pairs found (receipt parsing may not captur
    if "insurance"  in train2.columns and "receipt_ins_from_pdf"  in train2.columns
        print_header( "INSURANCE CONSISTENCY: patients.csv vs receipt PDFs (TRAIN)
        a = train2[ "insurance" ].astype( str).str.lower(). str.strip()
        b = train2[ "receipt_ins_from_pdf" ].astype( str).str.lower(). str.strip()
        m = (a != "nan") & (b != "nan")
        if m.sum() > 0:
            match = float((a[m] == b[m]).mean())
            print(f"Insurance match rate where both present: {match:.4f} (n={int(
        else:
            print("No comparable insurance pairs found (receipt parsing may not c
    # Receipt feature distributions
    print_header( "RECEIPT FEATURE SANITY (DISTRIBUTIONS)" )
    for c in ["receipt_n_items" , "receipt_n_unique_codes" , "receipt_em_count" , "r

```


### Page 59

```text

              "receipt_top1_share" , "receipt_bucket_entropy" , "receipt_parse_fail
        if c in train2.columns:
            s = pd.to_numeric(train2[c], errors= "coerce" )
            print(f"{c}: mean={s.mean(): .3f} | p50= {s.quantile( 0.5):.3f} | p90= {s
            plt.figure()
            vals = s.dropna()
            if len(vals):
                plt.hist(vals.clip(upper=vals.quantile( 0.995)), bins= 60)
                plt.title( f"{c} histogram (clipped at 99.5%)")
                plt.xlabel(c)
                plt.ylabel( "count")
                savefig(OUT_DIR / f"{c}_hist.png")
    # Receipt feature ↔ target correlations (train only)
    if "ed_cost_next3y_usd"  in train2.columns:
        print_header( "RECEIPT FEATURE ↔ TARGET (SPEARMAN)" )
        y = train2[ "ed_cost_next3y_usd" ].astype( float)
        receipt_cols = [c for c in train2.columns if c.startswith( "receipt_" ) and 
        rows = [(c, spearman_corr(train2[c], y)) for c in receipt_cols]
        rc = pd.DataFrame(rows, columns=[ "receipt_feature" , "spearman_r" ]).sort_v
        print(rc.head( 20).to_string(index= False))
        rc.to_csv(OUT_DIR / "spearman_receipt_features.csv" , index= False)
    # Code lift analysis (train only)
    if "ed_cost_next3y_usd"  in train2.columns:
        print_header( "CODE LIFT ANALYSIS (TRAIN ONLY)" )
        pid_to_codes = {}
        for pid, obj in parsed.items():
            if isinstance (obj, dict):
                codes = [it.get( "code", "") for it in obj.get( "items", []) if isi
            else:
                codes = []
            pid_to_codes[ int(pid)] = set(codes)
        tr = train2.copy()
        if "prior_ed_cost_5y_usd"  in tr.columns:
            tr[ "prior_cost_bin" ] = pd.qcut(tr[ "prior_ed_cost_5y_usd" ].astype( float
        else:
            tr[ "prior_cost_bin" ] = "all"

```


### Page 60

```text

        group_keys = []
        if "primary_chronic"  in tr.columns:
            group_keys.append( "primary_chronic" )
        group_keys.append( "prior_cost_bin" )
        baseline = tr.groupby(group_keys)[ "ed_cost_next3y_usd" ].median().rename( "
        tr = tr.join(baseline, on=group_keys)
        tr[ "residual_vs_baseline" ] = tr["ed_cost_next3y_usd" ].astype( float) - tr[
        overall_mean = tr[ "ed_cost_next3y_usd" ].astype( float).mean()
        pid_to_y = dict(zip(tr["patient_id" ].astype( int), tr["ed_cost_next3y_usd"
        pid_to_resid = dict(zip(tr["patient_id" ].astype( int), tr["residual_vs_bas
        train_pids = set(tr["patient_id" ].astype( int).tolist())
        code_patients = defaultdict( list)
        for pid in train_pids:
            for code in pid_to_codes.get(pid, set()):
                if code:
                    code_patients[code].append(pid)
        rows = []
        for code, pids in code_patients.items():
            n = len(pids)
            if n < CODE_MIN_SUPPORT:
                continue
            ys = np.array([pid_to_y[pid] for pid in pids], dtype= float)
            rs = np.array([pid_to_resid[pid] for pid in pids], dtype= float)
            rows.append({
                "code": code,
                "support_n" : n,
                "mean_target" : float(np.mean(ys)),
                "median_target" : float(np.median(ys)),
                "lift_mean_vs_overall" : float(np.mean(ys) - overall_mean),
                "mean_residual_vs_chronic_costbin_median" : float(np.mean(rs)),
            })
        code_lift = pd.DataFrame(rows)
        if len(code_lift):
            code_lift = code_lift.sort_values(

```


### Page 61

```text

                [ "mean_residual_vs_chronic_costbin_median" , "support_n" ],
                ascending=[ False, False]
            )
            code_lift.to_csv(OUT_DIR / "code_lift_table.csv" , index= False)
            print("Top codes by residual lift (controls chronic + prior-cost-bin)
            print(code_lift.head(N_TOP_CODES).to_string(index= False))
        else:
            print("No codes met support threshold. Consider lowering CODE_MIN_SUP
    # Tail comparison
    if "ed_cost_next3y_usd"  in train2.columns:
        print_header( "TAIL COMPARISON (TOP 5% TARGET vs REST)" )
        y = train2[ "ed_cost_next3y_usd" ].astype( float)
        thr = y.quantile( 0.95)
        top = train2[y >= thr].copy()
        rest = train2[y < thr].copy()
        print("Top5% threshold:" , thr, "| top_n:" , len(top), "| rest_n:" , len(res
        key_feats = [ "prior_ed_cost_5y_usd" , "prior_ed_visits_5y" ,
                     "receipt_n_items" , "receipt_n_unique_codes" , "receipt_em_cou
                     "receipt_em_max" , "receipt_top1_share" , "receipt_bucket_entr
        rows = []
        for f in key_feats:
            if f in train2.columns:
                a = pd.to_numeric(top[f], errors= "coerce" )
                b = pd.to_numeric(rest[f], errors= "coerce" )
                rows.append({
                    "feature" : f,
                    "top_mean" : float(a.mean()),
                    "rest_mean" : float(b.mean()),
                    "delta_mean" : float(a.mean() - b.mean()),
                    "top_p50" : float(a.quantile( 0.5)),
                    "rest_p50" : float(b.quantile( 0.5)),
                })
        tail_tbl = pd.DataFrame(rows).sort_values( "delta_mean" , ascending= False)
        print(tail_tbl.to_string(index= False))
        tail_tbl.to_csv(OUT_DIR / "tail_feature_comparison.csv" , index= False)
    # Save merged tables
    train2.to_csv(OUT_DIR / "train_with_receipt_features.csv" , index= False)

```


### Page 62

```text

If this still prints something like Cached parsed receipts: 2 , it will now auto-backup  that
cache as *.bad_<timestamp>.joblib  and rebuild cleanly.    test2.to_csv(OUT_DIR / "test_with_receipt_features.csv" , index= False)
    print_header( "DONE")
    print("Outputs saved to:" , OUT_DIR)
    print("Cache used:" , RECEIPT_CACHE)
    print("Key outputs:" )
    print(" - pdf_total_mismatches_train/test.csv" )
    print(" - spearman_receipt_features.csv" )
    print(" - code_lift_table.csv" )
    print(" - tail_feature_comparison.csv" )
# Run
main()
Sour ces
Sour ces
Here it the log for ED A:
======================================
======================================
==============
LOAD D ATA
======================================
======================================
==============
Train shape: (2000, 5)
Test  shape: (2000, 4)
Train cols: ['patient_id', 'primary_chronic',  
'prior_ed_visits_5y', 'prior_ed_cost_5y_usd',  
'ed_cost_next3y_usd']
Patients shape: (4000, 5)
Patients cols: ['patient_id', 'age', 'sex', 'insurance', 'zip3']
Duplicate patient_id — train: 0 | test: 0
Train/T est patient_id overlap: 0

```


### Page 63

```text

======================================
======================================
==============
MISSINGNESS (T OP 30)
======================================
======================================
==============
patient_id              0.0
primary_chronic         0.0
prior_ed_visits_5y      0.0
prior_ed_cost_5y_usd    0.0
ed_cost_next3y_usd      0.0
age                     0.0
sex                     0.0
insurance               0.0
zip3                    0.0
======================================
======================================
==============
TARGET DISTRIBUTION
======================================
======================================
==============
0.00      306.8800
0.01      991.4990
0.05     1553.6960
0.10     1867.5730
0.25     2548.7675
0.50     3569.0950
0.75     4956.4225
0.90     6636.4750
0.95     7541.8475
0.99     8801.0213
1.00    11184.6100
Mean: 3908.25191 S td: 1822.401920306473 Skew:  
0.816103887108234
Top 5% contribute ~0.107 of total target spend
======================================

```


### Page 64

```text

======================================
==============
TARGET B Y PRIMAR Y_CHR ONIC
======================================
======================================
==============
                 count         mean   median       p90       p95         
p99
primary_chronic                                                            
HF                 873  4720.807239  4377.79  7522.504   
8191.498  9468.0928
DiabetesComp       725  3516.845283  3113.74  5866.244   
6705.846  8029.7624
Pneumonia          402  2849.567836  2590.54  4936.438   
5454.448  6177.9207
======================================
======================================
==============
BASIC REL ATIONSHIPS (SCA TTERS)
======================================
======================================
==============
======================================
======================================
==============
SPEARMAN C ORREL ATIONS (NUMERIC FEA TURES)
======================================
======================================
==============
             feature  spearman_r
prior_ed_cost_5y_usd    0.882309
  prior_ed_visits_5y    0.697989
          patient_id    0.056849
                 age    0.039905
                zip3    0.020266
======================================
======================================

```


### Page 65

```text

==============
RECEIPT PDF C OVERA GE CHECK
======================================
======================================
==============
PDF files found: 4000 in  
D:\AgentDs\agent_ds_healthcare\receipts_pdf
Missing PDFs for patient_ids: 0
Extra PDFs not in train/test: 0
Loading receipt cache:  
D:\AgentDs\agent_ds_healthcare\cache_iter10\receipts_par
sed.joblib
Cached parsed receipts: 2
⚠ Receipt cache INV ALID (wrong type/structure/size).  
Rebuilding cache...
Backed up bad cache to:  
D:\AgentDs\agent_ds_healthcare\cache_iter10\receipts_par
sed.bad_1771064731.joblib
Parsing receipts (threaded)... workers: 20
MuPDF error: format error: cannot recognize xref format
MuPDF error: format error: cannot recognize xref format
MuPDF error: format error: cannot recognize xref format
Parsing PDFs: 100%| ██████████ | 4000/4000  
[00:15<00:00, 265.50it/s]
Parsed 4000 receipts in 15.8s
Saved receipt cache:  
D:\AgentDs\agent_ds_healthcare\cache_iter10\receipts_par
sed.joblib
======================================
======================================
==============
RECEIPT FEA TURE EX TRACTION
======================================
======================================
==============
Receipt feature df: (4000, 15)

```


### Page 66

```text

======================================
======================================
==============
PDF T OTAL vs CSV prior_ed_cost_5y_usd V ALID ATION
======================================
======================================
==============
[train] match_rate=0.8950 | mismatches=210/2000
[test] match_rate=0.8900 | mismatches=220/2000
======================================
======================================
==============
ZIP3 C ONSISTENCY: patients.csv vs receipt PDFs (TRAIN)
======================================
======================================
==============
ZIP3 match rate where both present: 1.0000 (n=2000)
======================================
======================================
==============
INSURANCE C ONSISTENCY: patients.csv vs receipt PDFs  
(TRAIN)
======================================
======================================
==============
Insurance match rate where both present: 1.0000 (n=2000)
======================================
======================================
==============
RECEIPT FEA TURE SANIT Y (DISTRIBUTIONS)
======================================
======================================
==============
receipt_n_items: mean=6.995 | p50=7.000 | p90=9.000 |  
missing=0.000
receipt_n_unique_codes: mean=5.122 | p50=5.000 |  

```


### Page 67

```text

p90=7.000 | missing=0.000
receipt_em_count: mean=1.673 | p50=2.000 | p90=4.000 |  
missing=0.000
receipt_em_max: mean=3.531 | p50=4.000 | p90=5.000 |  
missing=0.280
receipt_top1_share: mean=0.311 | p50=0.290 | p90=0.471 |  
missing=0.000
receipt_bucket_entropy: mean=1.024 | p50=1.011 |  
p90=1.427 | missing=0.000
receipt_parse_failed: mean=0.000 | p50=0.000 | p90=0.000  
| missing=0.000
======================================
======================================
==============
RECEIPT FEA TURE ↔ T ARGET (SPEARMAN)
======================================
======================================
==============
       receipt_feature  spearman_r
     receipt_sum_items    0.882361
     receipt_pdf_total    0.841586
       receipt_n_items    0.425981
  receipt_non_em_count    0.378321
       receipt_em_mean    0.279492
        receipt_em_max    0.236304
receipt_n_unique_codes    0.179226
receipt_bucket_entropy    0.064890
      receipt_em_count   -0.447675
    receipt_top1_share   -0.486048
    receipt_top3_share   -0.486794
  receipt_parse_failed         NaN
======================================
======================================
==============
CODE LIFT ANAL YSIS (TRAIN ONL Y)
======================================
======================================
==============

```


### Page 68

```text

Top codes by residual lift (controls chronic + prior-cost-
bin):
 code  support_n  mean_target  median_target   
lift_mean_vs_overall   
mean_residual_vs_chronic_costbin_median
31500        678  4950.480796       4757.870            
1042.228886                               140.609167
99291        684  4891.798626       4741.095             
983.546716                               124.138465
36556        682  5000.174751       4837.630            
1091.922841                               123.421400
99292        687  4978.879287       4817.180            
1070.627377                               119.033013
92950        664  4893.741777       4763.255             
985.489867                               109.065437
36620        682  4934.775572       4768.445            
1026.523662                                96.552485
70450        420  3772.546429       3528.805            
-135.705481                                48.260357
99285        454  3768.006498       3488.295            
-140.245412                                39.363778
74177        474  3832.512405       3652.390             
-75.739505                                33.109884
84484        448  3740.529107       3551.855            
-167.722803                                27.195614
99284        443  3822.749887       3637.030             
-85.502023                                25.133950
G0378        435  3845.414506       3591.930             
-62.837404                                -9.453460
85025        576  2999.141424       2874.330            
-909.110486                               -54.315208
99281        582  3061.065876       2953.655            
-847.186034                               -68.953866
87070        574  3004.134599       2861.310            
-904.117311                               -73.855436
99283        571  2926.558932       2777.510            
-981.692978                               -93.222697
71045        617  2985.395008       2867.620            
-922.856902                               -96.890835
99282        573  2931.181379       2794.860            

```


### Page 69

```text

-977.070531                               -99.710724
======================================
======================================
==============
TAIL C OMP ARISON (T OP 5% T ARGET vs REST)
======================================
======================================
==============
Top5% threshold: 7541.8475 | top_n: 100 | rest_n: 1900
               feature     top_mean   rest_mean   delta_mean       
top_p50    rest_p50
  prior_ed_cost_5y_usd 17116.294100 3313.013253  
13803.280847 15936.705000 2002.540000
    prior_ed_visits_5y     7.390000    3.348947     4.041053      
7.000000    3.000000
       receipt_n_items     8.500000    6.916316     1.583684      
9.000000    7.000000
        receipt_em_max     4.857143    3.511268     1.345875      
5.000000    4.000000
receipt_n_unique_codes     5.090000    5.123684     
-0.033684     5.000000    5.000000
receipt_bucket_entropy     0.914188    1.029736     
-0.115548     0.940118    1.011404
    receipt_top1_share     0.195139    0.317247    -0.122108      
0.191660    0.297751
      receipt_em_count     0.310000    1.745263    -1.435263      
0.000000    2.000000
======================================
======================================
==============
DONE
======================================
======================================
==============
Outputs saved to:  
D:\AgentDs\agent_ds_healthcare\cache_iter10\eda_output
s_iter10
Cache used:  

```


### Page 70

```text

D:\AgentDs\agent_ds_healthcare\cache_iter10\receipts_par
sed.joblib
Key outputs:
 - pdf_total_mismatches_train/test.csv
 - spearman_receipt_features.csv
 - code_lift_table.csv
 - tail_feature_comparison.csv
Got it — this ED A log is very informative. Here’s what it says scientifically, what to change
in your receipt feature design, and what we should validate next befor e any modeling .
1) What y our ED A proves (and what it br eaks)
✅ Data int egrity: tabular side is clean
No missingness, no ID overlap between train/test, and patients.csv joins cleanly.
✅ Target is not extr emely  heavy-tailed
Top 5% is only ~10.7% of total spend. So this isn’t the classic “1% drives 50%” cost
problem — MAE gains will come from improving most o f the distribution , not only the far
tail.
✅ Chronic cohor ts are meaningfully sep arated
HF >> DiabetesComp >> Pneumonia (mean/median). That’s consistent with real-world
chronic trajectories (HF tends to recur with decompensations and high-intensity ED care).
This supports cohort-aware feature interactions later.
✅ Prior cost dominat es (as expect ed)
Spearman ~0.88 with target. This aligns with healthcare cost prediction literature: past cost
is usually the strongest single predictor in claims-like settings. 
2) The biggest r ed flag: PDF “ TOTAL” mismat ch ~11%
You expected PDF grand total == prior_ed_cost_5y_usd , but:
train match_rate = 0.895PMC

```


### Page 71

```text

test match_rate = 0.890
That strongly suggests your TOTAL extraction heuristic is wr ong for ~10–11%  of PDFs
(likely multiple “T OTAL” occurrences: per-visit totals + final grand total, and your regex is
grabbing the wrong one).
Good news: you already  have a better proxy:
receipt_sum_items  Spearman = 0.882 , basically identical to prior_ed_cost_5y_usd .
Hypothesis t o validat e next
H_total: In “mismatch” cases, sum(line_item_amounts)  will still match
prior_ed_cost_5y_usd  closely → meaning only the T OTAL regex is wrong, not your line
parsing.
If this hypothesis holds, we can ignor e pdf_t otal entir ely for modeling and use sum_items
for sanity checks only.
3) The most impor tant modeling insight: y our “E/M” featur es ar e mis-
specified
You saw:
receipt_em_count  Spearman = -0.448
Top 5% patients have receipt_em_count  mean 0.31 (vs 1.75 for rest)
This is not saying “higher acuity lowers future cost.”
It’s saying your em_count  only counts 99281–99285  (standard ED visit level codes), but
your code lift shows the tail is dominated by:
99291 / 99292 (critical car e time codes)  
and severe procedures like:
31500 (emer gency endotracheal intub ation)  
92950 (CPR)  
36556 (central v enous cathet er) 
36620 (ar terial cathet er) 
So high-risk patients are often coded as critical car e + r esuscitation , not “9928x visit
levels.” That’s why your 9928x count behaves “inverted.”ACEP +2
Value Set Autho …+1
ACEP +1
Value Set Autho …
American Societ …

```


### Page 72

```text

Actionable r efinement
Split “ED intensity” into three separate phenotypes:
1. ED visit lev els (99281–99285)  — moderate pathway
2. Critical car e (99291, 99292)  — severe pathway (very predictive) 
3. Resuscitation/air way/lines  (31500, 92950, 36556, 36620) — extreme severity markers
4) Int erpreting the “ cost concentration” signals (t op1/t op3 shar e are
negativ e)
top1_share  and top3_share  are ~ -0.49 with target
Top 5% patients have lower top1_share
Clinical interpretation: chronic/sev ere ED ev ents ar e multi-r esour ce (many
labs/imaging/procedures), so costs are spread across many line items. “One code
dominates” looks more like a simpler/contained episode.
So keep these concentration features — they’re telling you “complexity vs single-driver.”
5) Code lif t table confirms clinically plausible  “severity → futur e cost”
signals
The strongest residual-lift codes after controlling for (chronic × prior_cost_bin) are exactly
“unstable physiology” markers:
airway intervention (31500) 
critical care time (99291/99292) 
CPR (92950) 
central/arterial lines (36556/36620) 
This is exactly the kind of “orthogonal” signal that can improve MAE beyond prior spend.
Also note: imaging + troponin appear in your top list (70450, 74177, 84484). These
correspond to high-intensity diagnostic pathways (CT head, CT A/P w contrast, troponin).ACEP +1
Value Set Autho …+3
Value Set Autho …+1
ACEP +1
ACEP +1
Value Set Autho …+1
AAPC +3

```


### Page 73

```text

Phase 3 deliv erable: What t o validat e next (tar geted, no
modeling)
A) Root-cause the T OTAL mismat ch
Compare abs(sum_items - prior_cost)  vs abs(pdf_total - prior_cost)
If sum_items matches ~100%, fix T OTAL parsing later but proceed.
B) Rebuild acuity featur es corr ectly
Create new features:
n_ed_em_9928x , max_ed_level_9928x , has_99285
n_critical_care_99291 , n_critical_care_99292 , has_critical_care
has_intubation_31500 , has_cpr_92950 , has_cvc_36556 , has_artline_36620
optional: has_observation_G0378  
Then evaluate association with:
target
and residual af ter contr olling for (chr onic × prior_cost_bin)  (like your code-lift
method)
One-cell “ED A add-on” t o run now (uses y our existing cache, fast)
Copy into one Jupyter cell:Novitas Solutions
python
import re
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load
BASE_DIR = Path( r"D:\AgentDs\agent_ds_healthcare" )
TRAIN_CSV = BASE_DIR / "ed_cost_train.csv"
TEST_CSV  = BASE_DIR / "ed_cost_test.csv"
PATIENTS_CSV = BASE_DIR / "patients.csv"
CACHE_DIR = BASE_DIR / "cache_iter10"
RECEIPT_CACHE = CACHE_DIR / "receipts_parsed.joblib"

```


### Page 74

```text

train = pd.read_csv(TRAIN_CSV)
test  = pd.read_csv(TEST_CSV)
patients = pd.read_csv(PATIENTS_CSV)
parsed = load(RECEIPT_CACHE)  # dict[patient_id] -> dict with "items", "pdf_total
def sum_items (obj):
    if not isinstance (obj, dict):
        return np.nan
    items = obj.get( "items", [])
    if not items:
        return 0.0
    return float(np.nansum([it.get( "amount" , np.nan) for it in items if isinstanc
def code_counts (obj):
    if not isinstance (obj, dict):
        return {}
    items = obj.get( "items", [])
    codes = [it.get( "code","") for it in items if isinstance (it, dict)]
    cc = {}
    # ED E/M (99281-99285)
    em9928 = [c for c in codes if isinstance (c,str) and c.isdigit() and c.startsw
    cc["n_ed_em_9928x" ] = len(em9928)
    cc["has_99285" ] = int("99285" in em9928)
    # critical care
    cc["n_99291" ] = sum(1 for c in codes if c=="99291")
    cc["n_99292" ] = sum(1 for c in codes if c=="99292")
    cc["has_critical_care" ] = int((cc["n_99291" ] + cc["n_99292" ]) > 0)
    # high acuity procedures
    cc["has_intub_31500" ] = int("31500" in codes)
    cc["has_cpr_92950" ] = int("92950" in codes)
    cc["has_cvc_36556" ] = int("36556" in codes)
    cc["has_artline_36620" ] = int("36620" in codes)
    # selected diagnostics from your lift list
    cc["has_ct_head_70450" ] = int("70450" in codes)
    cc["has_ct_abdpel_74177" ] = int("74177" in codes)
    cc["has_troponin_84484" ] = int("84484" in codes)
    # observation
    cc["has_obs_G0378" ] = int("G0378" in codes)

```


### Page 75

```text

    return cc
# Build receipt table
rows = []
for pid, obj in parsed.items():
    pid = int(pid)
    r = { "patient_id" : pid,
         "pdf_total" : obj.get( "pdf_total" ) if isinstance (obj, dict) else np.nan,
         "sum_items" : sum_items(obj)}
    r.update(code_counts(obj))
    rows.append(r)
rec = pd.DataFrame(rows)
# Merge
tr = train.merge(patients, on= "patient_id" , how="left").merge(rec, on= "patient_id
te = test.merge(patients, on= "patient_id" , how="left").merge(rec, on= "patient_id"
# --- A) TOTAL mismatch root cause ---
def match_rate (a, b, tol= 0.01):
    d = (pd.to_numeric(a, errors= "coerce" ) - pd.to_numeric(b, errors= "coerce" )).a
    return float((d <= tol).mean()), d
mr_total_tr, d_total_tr = match_rate(tr[ "pdf_total" ], tr["prior_ed_cost_5y_usd" ])
mr_sum_tr,   d_sum_tr   = match_rate(tr[ "sum_items" ], tr["prior_ed_cost_5y_usd" ])
print("TRAIN match_rate: pdf_total vs prior =" , mr_total_tr, "| sum_items vs prio
mr_total_te, d_total_te = match_rate(te[ "pdf_total" ], te["prior_ed_cost_5y_usd" ])
mr_sum_te,   d_sum_te   = match_rate(te[ "sum_items" ], te["prior_ed_cost_5y_usd" ])
print("TEST  match_rate: pdf_total vs prior =" , mr_total_te, "| sum_items vs prio
# Show worst 15 mismatches (train) for pdf_total and for sum_items
tmp = tr[[ "patient_id" ,"prior_ed_cost_5y_usd" ,"pdf_total" ,"sum_items" ]].copy()
tmp["abs_diff_pdf_total" ] = d_total_tr
tmp["abs_diff_sum_items" ] = d_sum_tr
print("\nWorst 15 pdf_total mismatches (train):" )
print(tmp.sort_values( "abs_diff_pdf_total" , ascending= False).head(15).to_string(i
print("\nWorst 15 sum_items mismatches (train):" )
print(tmp.sort_values( "abs_diff_sum_items" , ascending= False).head(15).to_string(i

```


### Page 76

```text

What t o paste back t o me (so I can lock in the featur e strat egy)
1. The two match rates:
sum_items vs prior  (train/test)
pdf_total vs prior  (train/test)
2. The “R esidual lift vs baseline” table printed at the end.
Once we confirm whether line-item sums are reliable and which acuity markers have the
strongest residual lift, we’ll finalize the receipt phenotype featur e set  and only then move
to modeling.# --- B) Residual-lift check for new acuity features ---
# baseline residual: median by (primary_chronic x prior_cost_decile)
tr2 = tr.copy()
tr2["prior_cost_bin" ] = pd.qcut(tr2[ "prior_ed_cost_5y_usd" ], q=10, duplicates= "dr
base = tr2.groupby([ "primary_chronic" ,"prior_cost_bin" ])["ed_cost_next3y_usd" ].me
tr2 = tr2.join(base, on=[ "primary_chronic" ,"prior_cost_bin" ])
tr2["resid"] = tr2[ "ed_cost_next3y_usd" ] - tr2[ "baseline" ]
feat_list = [
    "has_critical_care" ,"has_intub_31500" ,"has_cpr_92950" ,"has_cvc_36556" ,"has_art
    "has_ct_head_70450" ,"has_ct_abdpel_74177" ,"has_troponin_84484" ,"has_obs_G0378
]
print("\nResidual lift vs baseline (mean resid), by feature=1:" )
out = []
for f in feat_list:
    if f not in tr2.columns: 
        continue
    m = tr2[f]== 1
    if m.sum() < 20:
        continue
    out.append((f, int(m.sum()), float(tr2.loc[m, "resid"].mean()), float(tr2.loc[
out = pd.DataFrame(out, columns=[ "feature" ,"support_n" ,"mean_resid_lift" ,"mean_ta
print(out.to_string(index= False))
Sour ces
Sour ces

```


### Page 77

```text

here is the new log from ED A add-on:
TRAIN match_rate: pdf_total vs prior = 0.895 | sum_items  
vs prior = 1.0
TEST  match_rate: pdf_total vs prior = 0.89 | sum_items vs  
prior = 1.0
Worst 15 pdf_total mismatches (train):
 patient_id  prior_ed_cost_5y_usd  pdf_total  sum_items   
abs_diff_pdf_total  abs_diff_sum_items
       1080               1031.30    1031.30    1031.30                  
0.0        0.000000e+00
       1807               5664.15    5664.15    5664.15                  
0.0        9.094947e-13
       3038               1025.28    1025.28    1025.28                  
0.0        0.000000e+00
       1314               1852.77    1852.77    1852.77                  
0.0        2.273737e-13
       3321               8489.80    8489.80    8489.80                  
0.0        0.000000e+00
       1296               2912.06    2912.06    2912.06                  
0.0        0.000000e+00
         33                735.11     735.11     735.11                 0.0         
0.000000e+00
       3605               1737.56    1737.56    1737.56                  
0.0        0.000000e+00
       1903               2325.33    2325.33    2325.33                  
0.0        4.547474e-13
       1488               4377.89    4377.89    4377.89                  
0.0        0.000000e+00
       1106               7359.35    7359.35    7359.35                  
0.0        0.000000e+00
       2148               2885.07    2885.07    2885.07                  
0.0        4.547474e-13
        821                597.53     597.53     597.53                 0.0         
0.000000e+00
        686               4502.53    4502.53    4502.53                  
0.0        0.000000e+00
       1303                306.45     306.45     306.45                 0.0         

```


### Page 78

```text

5.684342e-14
Worst 15 sum_items mismatches (train):
 patient_id  prior_ed_cost_5y_usd  pdf_total  sum_items   
abs_diff_pdf_total  abs_diff_sum_items
       2841              16734.76        NaN   16734.76                  
NaN        3.637979e-12
       3617              25827.83        NaN   25827.83                  
NaN        3.637979e-12
        761              17680.51        NaN   17680.51                  
NaN        3.637979e-12
       3810              25937.30        NaN   25937.30                  
NaN        3.637979e-12
       2293              19299.59        NaN   19299.59                  
NaN        3.637979e-12
       1548              19284.94        NaN   19284.94                  
NaN        3.637979e-12
       3957              20391.86        NaN   20391.86                  
NaN        3.637979e-12
       2612              21882.69        NaN   21882.69                  
NaN        3.637979e-12
        835              12170.04        NaN   12170.04                  
NaN        3.637979e-12
       3985              23004.74        NaN   23004.74                  
NaN        3.637979e-12
       2292              16544.19        NaN   16544.19                  
NaN        3.637979e-12
       2415              21941.82        NaN   21941.82                  
NaN        3.637979e-12
       3459              18000.13        NaN   18000.13                  
NaN        3.637979e-12
        594              20131.91        NaN   20131.91                  
NaN        3.637979e-12
       3879              30797.47        NaN   30797.47                  
NaN        3.637979e-12
Residual lift vs baseline (mean resid), by feature=1:
            feature  support_n  mean_resid_lift  mean_target
    has_intub_31500        678       140.609167  4950.480796
      has_cvc_36556        682       123.421400  5000.174751

```


### Page 79

```text

      has_cpr_92950        664       109.065437  4893.741777
  has_critical_care       1001        99.563237  4761.644076
  has_artline_36620        682        96.552485  4934.775572
  has_ct_head_70450        420        48.260357  3772.546429
          has_99285        454        39.363778  3768.006498
has_ct_abdpel_74177        474        33.109884  3832.512405
 has_troponin_84484        448        27.195614  3740.529107
      has_obs_G0378        435        -9.453460  3845.414506
This is a really str ong set of diagnostics — you’ve essentially proven (1) the PDFs are
usable at scale on your machine, and (2) the right signal is in the line it ems, not in the
“TOTAL” string.
1) The “ TOTAL mismat ch” is not a data pr oblem — it’ s a p arsing
artifact
Your add-on results are definitive:
sum_items vs prior  match rate = 1.0 (train + test)
pdf_total vs prior  match rate ≈ 0.89
So the receipts are internally consist ent; your line-item parser is perfect in aggregate, but
the “ TOTAL” text is o ften missing / not extractable , leading to pdf_total = NaN  and
therefore “mismatch.” (This also explains why your earlier “worst mismatches” list looked
weird — NaNs don’t sort the way you expect.)
Decision (for the r est o f the pr oject):
Treat sum_items  as the canonical r eceipt t otal.
Treat pdf_total  as a low-quality field  used only as a “text-extraction health flag”
(e.g., total_found = 0/1 ).
In feature engineering, either drop pdf_total  entir ely or imput e pdf_total =
sum_items  when missing .
This is exactly the kind of practical “data plumbing” win that moves MAE.
2) The r eceipt signals that matt er are “severity phenotype” signals

```


### Page 80

```text

Your residual lift table (controlled for primary_chronic × prior_cost_decile ) is the key
evidence:
Intubation 31500 : +140.6 residual lift
Central venous catheter 36556 : +123.4
CPR 92950 : +109.1
Critical care ( 99291/99292 ) as a bucket: +99.6
Arterial line 36620 : +96.6
CT head 70450 : +48.3
CT A/P w contrast 74177 : +33.1
Troponin 84484 : +27.2
Observation G0378 : slightly negative
Clinically, that’s exactly what we’d expect: these codes reflect organ failur e risk /
resuscitation / inv asive monit oring , which are strong proxies for “this patient had (and
may again have) unstable physiology.”
Code definitions back this interpretation:
99291/99292  are critical care E/M time codes for critically ill/injured patients (first 30–
74 min; each additional 30 min). 
31500  is emergency endotracheal intubation. 
92950  is cardiopulmonary resuscitation. 
36556  is insertion of a non-tunneled central venous catheter (age ≥5). 
70450  is CT head/brain without contrast. 
74177  is CT abdomen/pelvis with contrast. 
G0378  is hospital outpatient observation services billed per hour. 
Translation for the competition:
These are “high-value” receipt features because they add signal even af ter you control for
prior spending.
3) Why y our earlier E/M featur es look ed “wr ong” (and how t o fix
them)
In the first ED A, receipt_em_count  was strongly negative and top-5% patients had lower
receipt_em_count . That’s because your em_count  only counted 99281–99285 , but theValue Set Autho …+2
Value Set Autho …
Value Set Autho …+1
Value Set Autho …
Value Set Autho …
Value Set Autho …
Novitas Solutions +1

```


### Page 81

```text

high-risk cases are often coded as critical car e (99291/99292)  plus procedures
(intubation/CPR/lines), not standard ED visit levels.
Fix: r eplace “ED E/M” with a 3-lane acuity model
Lane A: ED visit lev el codes  99281–99285
Lane B: critical car e 99291 + 99292 (and counts)
Lane C: r esuscitation/pr ocedur e sev erity (31500, 92950, 36556, 36620, etc.)
This will eliminate the misleading sign issue and give you monotone/separable severity
features.
4) Complexity featur es ar e also meaningful (and consist ent with ED
economics)
Your negative Spearman for top1_share/top3_share  and the tail comparison (top patients
have lower top1_share ) makes sense: severe ED episodes tend to involve many ser vices
(labs + imaging + procedures), so spend is distributed across line items rather than
concentrated in a single code.
Keep these featur es, but treat them as “episode complexity / multi-resource intensity”
proxies.
5) Updat ed Phase-3 featur e blueprint (still no model)
A) “R eceipt canonical t otal”
receipt_total = sum_items
Optional: total_found = int(notna(pdf_total))  (quality flag only)
B) High-acuity flags + counts (y our pr oven winner s)
has_intub_31500 , has_cpr_92950 , has_cvc_36556 , has_artline_36620
has_critical_care , n_99291 , n_99292  (and maybe cc_minutes_proxy =
74*has_99291 + 30*n_99292 )
C) Diagnostic int ensity
has_ct_head_70450 , has_ct_abdpel_74177 , has_troponin_84484

```


### Page 82

```text

Later: extend to other recurring labs/imaging from your global code-frequency table
D) ED visit “v olume” and “lev el” (but corr ectly defined)
n_ed_level_9928x , has_99285 , max_9928x_level , has_any_ed_eval = any(9928x or
critical care)
E) Complexity shape
n_items , n_unique_codes , entropy , top1_share , top3_share
F) Int eractions (these usually add MAE wins in healthcar e cost)
primary_chronic × severity_flags
insurance × severity_flags
zip3 × insurance  (price/access proxy)
6) The single most impor tant next v alidation (befor e modeling)
Right now you’ve proven global residual lift. Next we should verify heterogeneity :
1. Residual lif t by chr onic gr oup
Example hypothesis:
Severity flags may have stronger incremental effect in HF than Pneumonia.
2. Residual lif t by insurance
Synthetic datasets often encode access/coverage differences — this can be big.
If you run one more small check and it confirms these interactions, we’ll lock the final
engineered feature set and proceed to modeling with high confidence.
If you want, I’ll give you a one-cell follow-up  that prints:
% pdf_total missing  vs prior_cost_decile
per-chronic residual lift for each severity flag
per-insurance residual lift for each severity flag
…but the key strategic decisions are already clear from your logs:
Use sum_it ems as truth
Promot e sev erity/critical-car e/procedur e flags t o first-class featur es
Redesign ED E/M featur es int o 3 acuity lanes

```


### Page 83

```text

Sour ces
Sour ces
System Update: Phase 1 (ED A) Completed.
Here are the raw findings from the data analysis script you  
designed. R ead these carefully, as they contradict some  
standard assumptions and reveal the specific structure of  
this dataset.
The Evidence (Data Feedback):
Data Integrity:
pdf_total (parsed from text) has ~11% mismatch rate with  
prior_ed_cost_5y_usd.
sum_items (sum of parsed line items) has a 100% match  
rate with prior_ed_cost_5y_usd.
Conclusion: The text extraction of the "T otal" field is  
unreliable. The line-item sum is the ground truth.
Clinical Signal (R esidual Analysis):
We analyzed the residuals (deviations) of cost after  
controlling for primary_chronic.
High P ositive Lift Codes: The following CPT codes showed  
massive positive residuals (meaning patients with these  
codes cost significantly more than their history suggests):
31500 (Intubation), 92950 (CPR), 36556 (CV C), 36620 (Art  
Line).
99291 & 99292 (Critical Care).
70450 (CT Head), 74177 (CT A/P), 84484 (T roponin).

```


### Page 84

```text

Negative/Neutral Signal: S tandard E/M codes (99281-
99285) do not show this same explosive cost risk.
Complexity Signal:
top1_share (percentage of cost from the most expensive  
item) is negatively correlated with total complexity. High-
cost patients tend to have distributed costs across many  
complex items.
Your Objective - Phase 2: Engineering & Modeling
As the Domain Expert, you now have the empirical  
evidence needed to design the predictive pipeline.
Task:
Design and write the full training pipeline (train.py).
You must decide:
Feature R epresentation: How to best encode the "High  
Lift" codes and the "Complexity" signals discovered above?  
(e.g., Counts? Booleans? W eighted vectors?)
Target T ransformation: Given the "Niche" nature of ED  
costs (likely zero-inflated or heavy-tailed), what is the  
mathematically optimal target transformation or loss  
function?
Validation S trategy: How do we ensure our CV score aligns  
with the leaderboard, considering the distribution of  
primary_chronic?
Model Selection: Choose the most appropriate algorithm  
for this tabular + sparse feature set on a local R TX 4060.
Constraint:
Use the parsed data from receipts_parsed.joblib to save  
time.

```


### Page 85

```text

The solution must be robust and grounded in the findings  
above.
Action:
Think through the engineering implications of the findings,  
then generate the complete Python script.
And remember to give me 1 cell copy pastable code in  
jupyter.
Your ED A evidence basically locks in the “right” design:
Never trust pdf_total ; use sum_items  as the receipt total (it matches 100%).
The lift is driven by critical car e + r esuscitation/pr ocedur e codes
(31500/92950/36556/36620 + 99291/99292) and a few high-int ensity diagnostics
(70450/74177/84484).
Since the leaderboard metric is MAE , the cleanest alignment is to train with an
L1/MAE-style objectiv e (XGBoost reg:absoluteerror ). 
You can train efficiently on your R TX 4060 using XGBoost GPU  (gpu_hist ).
Below is a single Jupy ter cell  that is effectively your train.py  (you can save it as a file if
you want). It:
Loads CSV s + receipts_parsed.joblib  (no re-parsing).
Builds:
High-lif t code featur es (bool + count + amount + share)
Complexity featur es (top shares, entropy, HHI, Gini, etc.)
Hashed b ag-o f-codes  sparse features (captures “other codes” beyond your
curated list)
One-hot categorical features (sex/insurance/zip3/primary_chronic)
Uses Stratified CV  on (primary_chronic × prior_cost_decile)  to keep folds
leaderboard-like.
Trains an XGBoost GPU model  optimized for MAE, with an optional pseudohuber
second model for robustness (blended). XGBoost Docum …
XGBoost Docum …+1
XGBoost Docum …

```


### Page 86

```text

Applies group median r esidual calibration  by (primary_chronic ×
prior_cost_decile)  (often a free MAE win).
Writes submission to submission_ICHI_V1.csv .
python
# ============================================================
# AgentDS Challenge 2 — TRAINING PIPELINE (single-cell "train.py")
# Uses receipts_parsed.joblib (no PDF parsing)
# ============================================================
import os, re, time, math, warnings
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import load, dump
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
warnings.filterwarnings( "ignore" )
# -----------------------------
# Paths (your setup)
# -----------------------------
BASE_DIR = Path( r"D:\AgentDs\agent_ds_healthcare" )
TRAIN_CSV = BASE_DIR / "ed_cost_train.csv"
TEST_CSV  = BASE_DIR / "ed_cost_test.csv"
PATIENTS_CSV = BASE_DIR / "patients.csv"
CACHE_DIR = BASE_DIR / "cache_iter10"
CACHE_DIR.mkdir(parents= True, exist_ok= True)
RECEIPT_CACHE = CACHE_DIR / "receipts_parsed.joblib"
SUBMISSION_PATH = BASE_DIR / "submission_ICHI_V1.csv"

```


### Page 87

```text

# Optional caches to speed reruns (feature matrices)
FEAT_CACHE = CACHE_DIR / "feat_matrix_cache.joblib"
# -----------------------------
# Training config
# -----------------------------
SEED = 42
N_SPLITS = 5
# Hashing for bag-of-codes (sparse)
HASH_DIM = 2**16  # 65536; small dataset so this is fine
USE_HASH_CODES = True
# Ensemble options
USE_PSEUDOHUBER_BLEND = True   # blend reg:absoluteerror + reg:pseudohubererror
BLEND_W_ABS = 0.7
BLEND_W_HUBER = 0.3
# Calibration (often helps MAE)
USE_GROUP_MEDIAN_CALIBRATION = True
CALIB_BINS = 10  # prior-cost bins used in calibration
# XGBoost params (GPU)
XGB_BASE_PARAMS = dict(
    # Core
    n_estimators= 8000,
    learning_rate= 0.03,
    max_depth= 6,
    min_child_weight= 10,
    subsample= 0.8,
    colsample_bytree= 0.8,
    reg_lambda= 1.0,
    reg_alpha= 0.0,
    # GPU
    tree_method= "gpu_hist" ,
    predictor= "gpu_predictor" ,
    # Misc
    random_state=SEED,
    n_jobs= 0,

```


### Page 88

```text

    verbosity= 1,
)
EARLY_STOP = 200
VERBOSE_EVAL = 200
# -----------------------------
# Helper utilities
# -----------------------------
def set_seed (seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
set_seed(SEED)
def normalize_cols (df: pd.DataFrame ) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df
def safe_float (x):
    if x is None:
        return np.nan
    if isinstance (x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().replace( "$","").replace( ",","")
    try:
        return float(s)
    except Exception:
        return np.nan
def code_bucket (code: str) -> str:
    if not isinstance (code, str) or not code:
        return "unk"
    if code.isdigit():
        if code.startswith( "99"):
            return "em_99"
        return f"num_{code[0]}"
    return f"alpha_ {code[0].upper()}"

```


### Page 89

```text

def entropy_from_counts (counts: Counter ) -> float:
    if not counts:
        return 0.0
    vals = np.array( list(counts.values()), dtype= float)
    p = vals / (vals. sum() + 1e-12)
    return float(-(p * np.log(p + 1e-12)).sum())
def gini_from_amounts (x: np.ndarray ) -> float:
    # Gini coefficient for non-negative vector x
    x = x.astype( float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan
    if np.all(x == 0):
        return 0.0
    x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(x)
    g = (n + 1 - 2 * np.sum(cumx) / (cumx[- 1] + 1e-12)) / n
    return float(g)
def build_prior_bins (train_prior: pd.Series, n_bins: int = 10):
    # Quantile edges, then pd.cut for both train/test
    q = np.quantile(train_prior.values.astype( float), np.linspace( 0, 1, n_bins + 
    bins = np.unique(q)
    # ensure strictly increasing bins
    if len(bins) < 3:
        # fallback: 2 bins
        bins = np.unique(np.quantile(train_prior.values.astype( float), [0, 0.5, 1
    return bins
# -----------------------------
# Evidence-driven code lists
# -----------------------------
# High-lift severe procedures
SEVERE_PROC = [ "31500", "92950", "36556", "36620"]
# Critical care
CRIT_CARE = [ "99291", "99292"]
# High-intensity diagnostics from your lift table

```


### Page 90

```text

HI_DIAG = [ "70450", "74177", "84484"]
# Observation code (showed slightly negative lift in your add-on)
OBS = ["G0378"]
# Standard ED E/M visit levels (use as separate lane)
ED_LEVELS = [ "99281", "99282", "99283", "99284", "99285"]
CURATED_CODES = SEVERE_PROC + CRIT_CARE + HI_DIAG + OBS + ED_LEVELS
# -----------------------------
# Receipt feature engineering (uses items list from cache)
# -----------------------------
def receipt_features_from_items (obj: dict) -> dict:
    # obj: dict with key "items" list of {"code","qty","amount",...}
    items = obj.get( "items", []) if isinstance (obj, dict) else []
    if not items:
        # should not happen given your parse success, but be safe
        out = {
            "receipt_total" : 0.0,
            "n_items" : 0,
            "n_unique_codes" : 0,
            "top1_share" : np.nan,
            "top3_share" : np.nan,
            "hhi": np.nan,
            "gini": np.nan,
            "bucket_entropy" : 0.0,
            "n_em_9928x" : 0,
            "max_em_level_9928x" : np.nan,
            "has_critical_care" : 0,
            "n_99291" : 0,
            "n_99292" : 0,
            "cc_time_proxy" : 0.0,
            "n_severe_proc" : 0,
            "severe_proc_share" : np.nan,
            "n_hi_diag" : 0,
            "hi_diag_share" : np.nan,
        }
        # curated code slots
        for c in CURATED_CODES:
            out[ f"has_{c}"] = 0

```


### Page 91

```text

            out[ f"cnt_{c}"] = 0
            out[ f"amt_{c}"] = 0.0
            out[ f"share_ {c}"] = np.nan
        return out
    codes = []
    amts = []
    for it in items:
        if not isinstance (it, dict):
            continue
        c = str(it.get( "code", "")).strip()
        a = safe_float(it.get( "amount" , np.nan))
        if c:
            codes.append(c)
        amts.append(a)
    amts = np.array(amts, dtype= float)
    total = float(np.nansum(amts))
    if total <= 0:
        total = 1e-9
    # Basic counts
    code_counts = Counter(codes)
    n_items = int(len(items))
    n_unique = int(len(code_counts))
    # Cost concentration
    top_sorted = np.sort(amts[~np.isnan(amts)])[::- 1]
    top1 = float(top_sorted[ 0]) if len(top_sorted) else np.nan
    top3 = float(top_sorted[: 3].sum()) if len(top_sorted) >= 3 else float(top_sor
    top1_share = (top1 / total) if (not np.isnan(top1)) else np.nan
    top3_share = (top3 / total) if (not np.isnan(top3)) else np.nan
    # HHI on line-item amounts (complexity proxy)
    p = top_sorted / (top_sorted. sum() + 1e-12) if len(top_sorted) else np.array(
    hhi = float(np.sum(p * p)) if len(p) else np.nan
    gini = gini_from_amounts(amts)
    # Bucket entropy on code types
    bucket_counts = Counter(code_bucket(c) for c in codes)

```


### Page 92

```text

    bucket_entropy = entropy_from_counts(bucket_counts)
    # ED lane A: 99281–99285
    em9928 = [c for c in codes if c in ED_LEVELS]
    n_em_9928x = len(em9928)
    max_em_level = np.nan
    if em9928:
        # map 99281->1, ... 99285->5
        levels = [ int(c[-1]) for c in em9928 if c[-1].isdigit()]
        if levels:
            max_em_level = float(np.max(levels))
    # Critical care lane B
    n_99291 = int(code_counts.get( "99291", 0))
    n_99292 = int(code_counts.get( "99292", 0))
    has_cc = int((n_99291 + n_99292) > 0)
    # crude time proxy: first block 30–74 mins; each 99292 adds up to 30 mins
    cc_time_proxy = float((74 if n_99291 > 0 else 0) + 30 * n_99292)
    # Severe procedures lane C
    n_severe = int(sum(code_counts.get(c, 0) for c in SEVERE_PROC))
    amt_severe = 0.0
    # sum amounts for severe codes
    for it in items:
        if isinstance (it, dict) and str(it.get( "code","")).strip() in SEVERE_PROC
            amt_severe += safe_float(it.get( "amount" , 0.0)) or 0.0
    severe_share = float(amt_severe / total)
    # Diagnostics
    n_diag = int(sum(code_counts.get(c, 0) for c in HI_DIAG))
    amt_diag = 0.0
    for it in items:
        if isinstance (it, dict) and str(it.get( "code","")).strip() in HI_DIAG:
            amt_diag += safe_float(it.get( "amount" , 0.0)) or 0.0
    diag_share = float(amt_diag / total)
    # Severity score (simple evidence-driven weighting)
    has_intub = int(code_counts.get( "31500", 0) > 0)
    has_cpr   = int(code_counts.get( "92950", 0) > 0)
    has_cvc   = int(code_counts.get( "36556", 0) > 0)

```


### Page 93

```text

    has_art   = int(code_counts.get( "36620", 0) > 0)
    has_cth   = int(code_counts.get( "70450", 0) > 0)
    has_ctap  = int(code_counts.get( "74177", 0) > 0)
    has_trop  = int(code_counts.get( "84484", 0) > 0)
    severity_score = (
        5*has_intub + 5*has_cpr + 4*has_cvc + 4*has_art +
        3*has_cc + 1*has_cth + 1*has_ctap + 1*has_trop
    )
    out = {
        "receipt_total" : float(total),
        "n_items" : n_items,
        "n_unique_codes" : n_unique,
        "top1_share" : float(top1_share),
        "top3_share" : float(top3_share),
        "hhi": float(hhi),
        "gini": float(gini),
        "bucket_entropy" : float(bucket_entropy),
        "n_em_9928x" : int(n_em_9928x),
        "max_em_level_9928x" : float(max_em_level),
        "has_critical_care" : int(has_cc),
        "n_99291" : int(n_99291),
        "n_99292" : int(n_99292),
        "cc_time_proxy" : float(cc_time_proxy),
        "n_severe_proc" : int(n_severe),
        "severe_proc_share" : float(severe_share),
        "n_hi_diag" : int(n_diag),
        "hi_diag_share" : float(diag_share),
        "severity_score" : float(severity_score),
        # derived complexity: “distributed cost” grows as top3_share falls
        "distributed_cost" : float(1.0 - top3_share) if not np.isnan(top3_share) e
    }
    # Curated code features (bool + count + amount + share)
    # We compute amount per code by summing item amounts for that code
    amt_by_code = Counter()
    for it in items:
        if not isinstance (it, dict):
            continue

```


### Page 94

```text

        c = str(it.get( "code","")).strip()
        if not c:
            continue
        a = safe_float(it.get( "amount" , 0.0))
        if np.isnan(a):
            continue
        amt_by_code[c] += float(a)
    for c in CURATED_CODES:
        cnt = int(code_counts.get(c, 0))
        amt = float(amt_by_code.get(c, 0.0))
        out[ f"has_{c}"] = int(cnt > 0)
        out[ f"cnt_{c}"] = cnt
        out[ f"amt_{c}"] = amt
        out[ f"share_ {c}"] = float(amt / total)
    return out
def build_code_hash_dict (obj: dict) -> dict:
    # sparse token dict: code counts + bucket counts
    items = obj.get( "items", []) if isinstance (obj, dict) else []
    codes = []
    for it in items:
        if not isinstance (it, dict):
            continue
        c = str(it.get( "code","")).strip()
        if c:
            codes.append(c)
    cc = Counter(codes)
    bc = Counter(code_bucket(c) for c in codes)
    d = {}
    # code tokens
    for c, v in cc.items():
        d[ f"c={c}"] = float(v)
    # bucket tokens
    for b, v in bc.items():
        d[ f"b={b}"] = float(v)
    return d

```


### Page 95

```text

# -----------------------------
# Build feature matrices
# -----------------------------
def build_features (train_df, test_df, patients_df, receipts_dict ):
    # Merge demographics
    train_df = train_df.merge(patients_df, on= "patient_id" , how="left")
    test_df  = test_df.merge(patients_df, on= "patient_id" , how="left")
    # Receipt features for all ids (train+test)
    all_ids = pd.concat([train_df[ "patient_id" ], test_df[ "patient_id" ]], axis= 0).
    rec_rows = []
    hash_dicts = []
    missing = 0
    for pid in all_ids:
        obj = receipts_dict.get( int(pid), None)
        if obj is None:
            missing += 1
            obj = { "items": []}
        rec_rows.append({ "patient_id" : int(pid), **receipt_features_from_items(ob
        if USE_HASH_CODES:
            hash_dicts.append(build_code_hash_dict(obj))
    if missing:
        print(f"WARNING: {missing}  patient_ids missing from receipts cache. Fille
    rec_feat = pd.DataFrame(rec_rows)
    # Split back
    rec_train = rec_feat.iloc[: len(train_df)].copy()
    rec_test  = rec_feat.iloc[ len(train_df):].copy()
    train_df = train_df.merge(rec_train, on= "patient_id" , how="left")
    test_df  = test_df.merge(rec_test, on= "patient_id" , how="left")
    # Base numeric engineered from tabular
    train_df[ "prior_cost_per_year" ] = train_df[ "prior_ed_cost_5y_usd" ] / 5.0
    test_df[ "prior_cost_per_year" ]  = test_df[ "prior_ed_cost_5y_usd" ] / 5.0
    train_df[ "prior_visits_per_year" ] = train_df[ "prior_ed_visits_5y" ] / 5.0

```


### Page 96

```text

    test_df[ "prior_visits_per_year" ]  = test_df[ "prior_ed_visits_5y" ] / 5.0
    train_df[ "prior_cost_per_visit" ] = train_df[ "prior_ed_cost_5y_usd" ] / (train_
    test_df[ "prior_cost_per_visit" ]  = test_df[ "prior_ed_cost_5y_usd" ] / (test_df
    # Because receipt_total == prior_ed_cost_5y_usd, DON'T include receipt_total 
    # Instead include shares/complexity, which are informative beyond totals.
    # Define categorical and numeric columns
    cat_cols = [ "primary_chronic" , "sex", "insurance" , "zip3"]
    for c in cat_cols:
        train_df[c] = train_df[c].astype( str)
        test_df[c]  = test_df[c].astype( str)
    # Numeric feature set (dense)
    base_num = [
        "age",
        "prior_ed_visits_5y" ,
        "prior_ed_cost_5y_usd" ,
        "prior_cost_per_year" ,
        "prior_visits_per_year" ,
        "prior_cost_per_visit" ,
    ]
    receipt_num = [
        # complexity/shape
        "n_items" , "n_unique_codes" , "top1_share" , "top3_share" , "distributed_cost
        "hhi", "gini", "bucket_entropy" ,
        # acuity lanes
        "n_em_9928x" , "max_em_level_9928x" ,
        "has_critical_care" , "n_99291" , "n_99292" , "cc_time_proxy" ,
        "n_severe_proc" , "severe_proc_share" ,
        "n_hi_diag" , "hi_diag_share" ,
        "severity_score" ,
    ]
    # curated per-code numeric features (counts + shares; amounts can be redundan
    curated_feats = []
    for c in CURATED_CODES:
        curated_feats += [ f"has_{c}", f"cnt_{c}", f"share_ {c}"]

```


### Page 97

```text

    num_cols = base_num + receipt_num + curated_feats
    # Fill numeric NaNs
    for df in [train_df, test_df]:
        for c in num_cols:
            if c not in df.columns:
                df[c] = np.nan
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors= "coerce" ).fillna(
    # One-hot for categoricals (fit on combined for consistent columns)
    all_cat = pd.concat([train_df[cat_cols], test_df[cat_cols]], axis= 0, ignore_i
    try:
        ohe = OneHotEncoder(handle_unknown= "ignore" , sparse_output= True)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown= "ignore" , sparse= True)
    X_cat_all = ohe.fit_transform(all_cat)
    # Dense numeric -> sparse
    X_num_all = sp.csr_matrix(pd.concat([train_df[num_cols], test_df[num_cols]], 
    # Hash codes
    if USE_HASH_CODES:
        hasher = FeatureHasher(n_features=HASH_DIM, input_type= "dict")
        X_hash_all = hasher.transform(hash_dicts)
        X_all = sp.hstack([X_num_all, X_cat_all, X_hash_all], format="csr")
    else:
        X_all = sp.hstack([X_num_all, X_cat_all], format="csr")
    X_train = X_all[: len(train_df)]
    X_test  = X_all[ len(train_df):]
    return train_df, test_df, X_train, X_test, ohe
# -----------------------------
# CV + training
# -----------------------------
def make_strat_labels (train_df: pd.DataFrame, n_bins: int = 10) -> np.ndarray:

```


### Page 98

```text

    # stratify by primary_chronic x prior-cost quantile
    # (keeps fold distributions closer to leaderboard)
    prior = train_df[ "prior_ed_cost_5y_usd" ].astype( float)
    # qcut labels on train only
    qbin = pd.qcut(prior, q=n_bins, duplicates= "drop").astype( str)
    strat = train_df[ "primary_chronic" ].astype( str) + "|" + qbin
    return strat.values
def fit_one_model (X_train, y_train, X_valid, y_valid, objective: str, seed: int):
    params = dict(XGB_BASE_PARAMS)
    params[ "objective" ] = objective
    params[ "random_state" ] = seed
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric= "mae",
        verbose=VERBOSE_EVAL,
        early_stopping_rounds=EARLY_STOP,
    )
    return model
def group_median_calibrate (train_df, oof_pred, y_true, test_df, test_pred, n_bins
    # group by primary_chronic x prior-cost bin edges from TRAIN, then median res
    bins = build_prior_bins(train_df[ "prior_ed_cost_5y_usd" ].astype( float), n_bin
    tr_bin = pd.cut(train_df[ "prior_ed_cost_5y_usd" ].astype( float), bins=bins, in
    te_bin = pd.cut(test_df[ "prior_ed_cost_5y_usd" ].astype( float), bins=bins, inc
    tr_key = (train_df[ "primary_chronic" ].astype( str) + "|" + tr_bin).values
    te_key = (test_df[ "primary_chronic" ].astype( str) + "|" + te_bin).values
    resid = y_true - oof_pred
    shift = pd.Series(resid).groupby(tr_key).median()
    shift_dict = shift.to_dict()
    oof_adj = oof_pred + np.array([shift_dict.get(k, 0.0) for k in tr_key], dtype
    te_adj  = test_pred + np.array([shift_dict.get(k, 0.0) for k in te_key], dtyp
    return oof_adj, te_adj, shift
def run_cv_training (train_df, X_train, y, test_df, X_test ):

```


### Page 99

```text

    strat = make_strat_labels(train_df, n_bins= 10)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle= True, random_state=SEED)
    oof = np.zeros( len(train_df), dtype= float)
    test_pred_accum = np.zeros( len(test_df), dtype= float)
    fold_mae = []
    for fold, (tr_idx, va_idx) in enumerate (skf.split(np.zeros( len(train_df)), st
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        # Model A: L1/MAE-aligned objective
        m_abs = fit_one_model(X_tr, y_tr, X_va, y_va, objective= "reg:absoluteerro
        pred_va_abs = m_abs.predict(X_va)
        pred_te_abs = m_abs.predict(X_test)
        if USE_PSEUDOHUBER_BLEND:
            # Model B: smoother alternative
            m_hub = fit_one_model(X_tr, y_tr, X_va, y_va, objective= "reg:pseudohu
            pred_va_hub = m_hub.predict(X_va)
            pred_te_hub = m_hub.predict(X_test)
            pred_va = BLEND_W_ABS * pred_va_abs + BLEND_W_HUBER * pred_va_hub
            pred_te = BLEND_W_ABS * pred_te_abs + BLEND_W_HUBER * pred_te_hub
        else:
            pred_va = pred_va_abs
            pred_te = pred_te_abs
        oof[va_idx] = pred_va
        test_pred_accum += pred_te / N_SPLITS
        mae = mean_absolute_error(y_va, pred_va)
        fold_mae.append(mae)
        print(f"[Fold {fold}/{N_SPLITS} ] MAE: {mae:.6f}")
    print(f"\nOOF MAE (mean over folds): {np.mean(fold_mae): .6f} | OOF MAE (globa
    return oof, test_pred_accum

```


### Page 100

```text

# -----------------------------
# MAIN
# -----------------------------
t0 = time.time()
print("Loading CSVs..." )
train = normalize_cols(pd.read_csv(TRAIN_CSV))
test  = normalize_cols(pd.read_csv(TEST_CSV))
patients = normalize_cols(pd.read_csv(PATIENTS_CSV))
assert RECEIPT_CACHE.exists(), f"Missing receipt cache: {RECEIPT_CACHE} . Run your 
print("Loading receipts cache..." )
receipts = load(RECEIPT_CACHE)
if not isinstance (receipts, dict) or len(receipts) < 3000:
    raise RuntimeError( "receipts_parsed.joblib has unexpected structure/size. Reb
# Build matrices (use caching to speed reruns)
if FEAT_CACHE.exists():
    print("Loading cached feature matrices:" , FEAT_CACHE)
    cache = load(FEAT_CACHE)
    train2 = cache[ "train2" ]
    test2 = cache[ "test2"]
    X_train = cache[ "X_train" ]
    X_test = cache[ "X_test" ]
else:
    print("Building features..." )
    train2, test2, X_train, X_test, _ohe = build_features(train, test, patients, 
    dump({ "train2" : train2, "test2": test2, "X_train" : X_train, "X_test" : X_test}
    print("Saved feature cache:" , FEAT_CACHE)
# Target
y = train2[ "ed_cost_next3y_usd" ].astype( float).values
print("\nTraining CV..." )
oof, test_pred = run_cv_training(train2, X_train, y, test2, X_test)
# Optional calibration (group median residual shift)
if USE_GROUP_MEDIAN_CALIBRATION:
    print("\nApplying group-median calibration (primary_chronic x prior_cost_bin)
    oof_cal, test_cal, shift = group_median_calibrate(train2, oof, y, test2, test_

```


### Page 101

```text

What t o do af ter you run it
1. Paste the printed OOF MAE raw vs calibrat ed.
2. Tell me whether pseudohuber blend  improved or worsened OOF MAE (you can flip
USE_PSEUDOHUBER_BLEND ).
If calibration helps but not enough, the next step is a 2-stage r esidual model  where
Stage-2 predicts the residual  using receipt-heavy features (especially the severe procedure
+ critical care patterns). That’s often a clean MAE gain in this exact structure.    mae_raw = mean_absolute_error(y, oof)
    mae_cal = mean_absolute_error(y, oof_cal)
    print(f"OOF MAE raw: {mae_raw: .6f} | calibrated: {mae_cal: .6f}")
    test_pred_final = test_cal
else:
    test_pred_final = test_pred
# Final post-processing
test_pred_final = np.clip(test_pred_final, 0.0, None)
# Write submission
sub = pd.DataFrame({
    "patient_id" : test2[ "patient_id" ].astype( int).values,
    "ed_cost_next3y_usd" : test_pred_final
})
sub.to_csv(SUBMISSION_PATH, index= False)
print(f"\nSaved submission -> {SUBMISSION_PATH} ")
print(f"Done in {time.time()-t0: .1f}s")
Sour ces
Sour ces
-----------------------------------------------------------------
----------
TypeError                                 T raceback (most recent call  
last)
Cell In[13], line 589
    586 y = train2["ed_cost_next3y_usd"].astype(float).values

```


### Page 102

```text

    588 print("\nT raining CV...")
--> 589 oof, test_pred = run_cv_training(train2, X_train, y,  
test2, X_test)
    591 # Optional calibration (group median residual shift)
    592 if USE_GR OUP_MEDIAN_CALIBRA TION:
Cell In[13], line 528, in run_cv_training(train_df, X_train, y,  
test_df, X_test)
    525 y_tr, y_va = y[tr_idx], y[va_idx]
    527 # Model A: L1/MAE-aligned objective
--> 528 m_abs = fit_one_model(X_tr, y_tr, X_va, y_va,  
objective="reg:absoluteerror", seed=SEED + fold)
    530 pred_va_abs = m_abs.predict(X_va)
    531 pred_te_abs = m_abs.predict(X_test)
Cell In[13], line 488, in fit_one_model(X_train, y_train,  
X_valid, y_valid, objective, seed)
    486 params["random_state"] = seed
    487 model = xgb.X GBRegressor(**params)
--> 488 model.fit(
    489     X_train, y_train,
    490     eval_set=[(X_valid, y_valid)],
    491     eval_metric="mae",
    492     verbose=VERBOSE_EV AL,
    493     early_stopping_rounds=EARL Y_STOP,
    494 )
    495 return model
File 
c:\Users\shend\AppData\Local\Programs\Python\Python3
12\Lib\site-packages\xgboost\core.py:729, in  
require_keyword_args.<locals>.throw_if.
<locals>.inner_f(*args, **kwargs)
    727 for k, arg in zip(sig.parameters, args):
    728     kwargs[k] = arg
--> 729 return func(**kwargs)
TypeError: X GBModel.fit() got an unexpected keyword  
argument 'eval_metric'

```


### Page 103

```text

Help me fix the bug, and make sure the new code is robust
That error happens because your installed X GBoost sklearn API does not accept
eval_metric  as a fit()  argument  anymore (it’s been moved to the construct or /
set_params ). XGBoost’s own docs show eval_metric  passed when creating
XGBRegressor , and fit()  called with eval_set  (no eval_metric  kwarg). 
Also, for GPU in X GBoost ≥2.x/3.x the recommended way is tree_method="hist",
device="cuda"  (with fallback for older installs). 
Below is a robust, 1-cell  replacement that:
fixes the eval_metric  bug (sets it in the constructor),
uses the modern GPU parameters (with fallback),
uses a safe_fit()  wrapper that only passes args your fit()  actually supports,
uses predict_best()  so early stopping works across versions.
Copy/paste into one Jupy ter cell  and run.XGBoost Docum …+1
XGBoost Docum …
python
# ============================================================
# AgentDS Challenge 2 — TRAINING PIPELINE (robust XGBoost sklearn API)
# Fixes: eval_metric in fit() TypeError, GPU param compatibility, early-stop pred
# ============================================================
import os, re, time, math, warnings, inspect
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import load, dump
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import StratifiedKFold

```


### Page 104

```text

from sklearn.metrics import mean_absolute_error
import xgboost as xgb
warnings.filterwarnings( "ignore" )
np.set_printoptions(suppress= True)
# -----------------------------
# Paths (your setup)
# -----------------------------
BASE_DIR = Path( r"D:\AgentDs\agent_ds_healthcare" )
TRAIN_CSV = BASE_DIR / "ed_cost_train.csv"
TEST_CSV  = BASE_DIR / "ed_cost_test.csv"
PATIENTS_CSV = BASE_DIR / "patients.csv"
CACHE_DIR = BASE_DIR / "cache_iter10"
CACHE_DIR.mkdir(parents= True, exist_ok= True)
RECEIPT_CACHE = CACHE_DIR / "receipts_parsed.joblib"
SUBMISSION_PATH = BASE_DIR / "submission_ICHI_V1.csv"
# feature cache (keeps reruns fast)
FEAT_CACHE = CACHE_DIR / "feat_matrix_cache.joblib"
# -----------------------------
# Training config
# -----------------------------
SEED = 42
N_SPLITS = 5
HASH_DIM = 2**16
USE_HASH_CODES = True
USE_PSEUDOHUBER_BLEND = True
BLEND_W_ABS = 0.7
BLEND_W_HUBER = 0.3
USE_GROUP_MEDIAN_CALIBRATION = True
CALIB_BINS = 10
# Early stopping controls
EARLY_STOP = 200

```


### Page 105

```text

VERBOSE_EVAL = 200  # can set to False for silent
# -----------------------------
# Helpers
# -----------------------------
def set_seed (seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
set_seed(SEED)
def normalize_cols (df: pd.DataFrame ) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df
def safe_float (x):
    if x is None:
        return np.nan
    if isinstance (x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().replace( "$","").replace( ",","")
    try:
        return float(s)
    except Exception:
        return np.nan
def code_bucket (code: str) -> str:
    if not isinstance (code, str) or not code:
        return "unk"
    if code.isdigit():
        if code.startswith( "99"):
            return "em_99"
        return f"num_{code[0]}"
    return f"alpha_ {code[0].upper()}"
def entropy_from_counts (counts: Counter ) -> float:
    if not counts:
        return 0.0

```


### Page 106

```text

    vals = np.array( list(counts.values()), dtype= float)
    p = vals / (vals. sum() + 1e-12)
    return float(-(p * np.log(p + 1e-12)).sum())
def gini_from_amounts (x: np.ndarray ) -> float:
    x = x.astype( float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan
    if np.all(x == 0):
        return 0.0
    x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(x)
    g = (n + 1 - 2 * np.sum(cumx) / (cumx[- 1] + 1e-12)) / n
    return float(g)
def build_prior_bins (train_prior: pd.Series, n_bins: int = 10):
    q = np.quantile(train_prior.values.astype( float), np.linspace( 0, 1, n_bins + 
    bins = np.unique(q)
    if len(bins) < 3:
        bins = np.unique(np.quantile(train_prior.values.astype( float), [0, 0.5, 1
    return bins
# -----------------------------
# Evidence-driven code lists
# -----------------------------
SEVERE_PROC = [ "31500", "92950", "36556", "36620"]
CRIT_CARE   = [ "99291", "99292"]
HI_DIAG     = [ "70450", "74177", "84484"]
OBS         = [ "G0378"]
ED_LEVELS   = [ "99281", "99282", "99283", "99284", "99285"]
CURATED_CODES = SEVERE_PROC + CRIT_CARE + HI_DIAG + OBS + ED_LEVELS
# -----------------------------
# Receipt features (from receipts_parsed.joblib items)
# -----------------------------
def receipt_features_from_items (obj: dict) -> dict:
    items = obj.get( "items", []) if isinstance (obj, dict) else []

```


### Page 107

```text

    if not items:
        out = {
            "n_items" : 0,
            "n_unique_codes" : 0,
            "top1_share" : 0.0,
            "top3_share" : 0.0,
            "hhi": 1.0,
            "gini": 0.0,
            "bucket_entropy" : 0.0,
            "n_em_9928x" : 0,
            "max_em_level_9928x" : 0.0,
            "has_critical_care" : 0,
            "n_99291" : 0,
            "n_99292" : 0,
            "cc_time_proxy" : 0.0,
            "n_severe_proc" : 0,
            "severe_proc_share" : 0.0,
            "n_hi_diag" : 0,
            "hi_diag_share" : 0.0,
            "severity_score" : 0.0,
            "distributed_cost" : 0.0,
        }
        for c in CURATED_CODES:
            out[ f"has_{c}"] = 0
            out[ f"cnt_{c}"] = 0
            out[ f"share_ {c}"] = 0.0
        return out
    codes = []
    amts = []
    for it in items:
        if not isinstance (it, dict):
            continue
        c = str(it.get( "code", "")).strip()
        a = safe_float(it.get( "amount" , np.nan))
        if c:
            codes.append(c)
        amts.append(a)
    amts = np.array(amts, dtype= float)
    total = float(np.nansum(amts))

```


### Page 108

```text

    if not np.isfinite(total) or total <= 0:
        total = 1e-9
    code_counts = Counter(codes)
    n_items = int(len(items))
    n_unique = int(len(code_counts))
    top_sorted = np.sort(amts[~np.isnan(amts)])[::- 1]
    top1 = float(top_sorted[ 0]) if len(top_sorted) else 0.0
    top3 = float(top_sorted[: 3].sum()) if len(top_sorted) >= 3 else float(top_sor
    top1_share = top1 / total
    top3_share = top3 / total
    p = top_sorted / (top_sorted. sum() + 1e-12) if len(top_sorted) else np.array(
    hhi = float(np.sum(p * p)) if len(p) else 1.0
    gini = gini_from_amounts(amts)
    bucket_counts = Counter(code_bucket(c) for c in codes)
    bucket_entropy = entropy_from_counts(bucket_counts)
    em9928 = [c for c in codes if c in ED_LEVELS]
    n_em_9928x = len(em9928)
    max_em_level = float(max([int(c[-1]) for c in em9928], default= 0))
    n_99291 = int(code_counts.get( "99291", 0))
    n_99292 = int(code_counts.get( "99292", 0))
    has_cc = int((n_99291 + n_99292) > 0)
    cc_time_proxy = float((74 if n_99291 > 0 else 0) + 30 * n_99292)
    n_severe = int(sum(code_counts.get(c, 0) for c in SEVERE_PROC))
    amt_by_code = Counter()
    for it in items:
        if not isinstance (it, dict):
            continue
        c = str(it.get( "code","")).strip()
        a = safe_float(it.get( "amount" , 0.0))
        if c and np.isfinite(a):
            amt_by_code[c] += float(a)
    amt_severe = float(sum(amt_by_code.get(c, 0.0) for c in SEVERE_PROC))

```


### Page 109

```text

    severe_share = amt_severe / total
    n_diag = int(sum(code_counts.get(c, 0) for c in HI_DIAG))
    amt_diag = float(sum(amt_by_code.get(c, 0.0) for c in HI_DIAG))
    diag_share = amt_diag / total
    has_intub = int(code_counts.get( "31500", 0) > 0)
    has_cpr   = int(code_counts.get( "92950", 0) > 0)
    has_cvc   = int(code_counts.get( "36556", 0) > 0)
    has_art   = int(code_counts.get( "36620", 0) > 0)
    has_cth   = int(code_counts.get( "70450", 0) > 0)
    has_ctap  = int(code_counts.get( "74177", 0) > 0)
    has_trop  = int(code_counts.get( "84484", 0) > 0)
    severity_score = (
        5*has_intub + 5*has_cpr + 4*has_cvc + 4*has_art +
        3*has_cc + 1*has_cth + 1*has_ctap + 1*has_trop
    )
    out = {
        "n_items" : n_items,
        "n_unique_codes" : n_unique,
        "top1_share" : float(top1_share),
        "top3_share" : float(top3_share),
        "hhi": float(hhi),
        "gini": float(gini if np.isfinite(gini) else 0.0),
        "bucket_entropy" : float(bucket_entropy),
        "n_em_9928x" : int(n_em_9928x),
        "max_em_level_9928x" : float(max_em_level),
        "has_critical_care" : int(has_cc),
        "n_99291" : int(n_99291),
        "n_99292" : int(n_99292),
        "cc_time_proxy" : float(cc_time_proxy),
        "n_severe_proc" : int(n_severe),
        "severe_proc_share" : float(severe_share),
        "n_hi_diag" : int(n_diag),
        "hi_diag_share" : float(diag_share),
        "severity_score" : float(severity_score),
        "distributed_cost" : float(1.0 - top3_share),
    }

```


### Page 110

```text

    for c in CURATED_CODES:
        cnt = int(code_counts.get(c, 0))
        amt = float(amt_by_code.get(c, 0.0))
        out[ f"has_{c}"] = int(cnt > 0)
        out[ f"cnt_{c}"] = cnt
        out[ f"share_ {c}"] = float(amt / total)
    return out
def build_code_hash_dict (obj: dict) -> dict:
    items = obj.get( "items", []) if isinstance (obj, dict) else []
    codes = []
    for it in items:
        if isinstance (it, dict):
            c = str(it.get( "code","")).strip()
            if c:
                codes.append(c)
    cc = Counter(codes)
    bc = Counter(code_bucket(c) for c in codes)
    d = {}
    for c, v in cc.items():
        d[ f"c={c}"] = float(v)
    for b, v in bc.items():
        d[ f"b={b}"] = float(v)
    return d
# -----------------------------
# Feature builder
# -----------------------------
def build_features (train_df, test_df, patients_df, receipts_dict ):
    train_df = train_df.merge(patients_df, on= "patient_id" , how="left")
    test_df  = test_df.merge(patients_df, on= "patient_id" , how="left")
    all_ids = pd.concat([train_df[ "patient_id" ], test_df[ "patient_id" ]], axis= 0).
    rec_rows = []
    hash_dicts = []
    missing = 0

```


### Page 111

```text

    for pid in all_ids:
        obj = receipts_dict.get( int(pid), None)
        if obj is None:
            missing += 1
            obj = { "items": []}
        rec_rows.append({ "patient_id" : int(pid), **receipt_features_from_items(ob
        if USE_HASH_CODES:
            hash_dicts.append(build_code_hash_dict(obj))
    if missing:
        print(f"[WARN] Missing receipts in cache for {missing}  ids; filled empty 
    rec_feat = pd.DataFrame(rec_rows)
    rec_train = rec_feat.iloc[: len(train_df)].copy()
    rec_test  = rec_feat.iloc[ len(train_df):].copy()
    train_df = train_df.merge(rec_train, on= "patient_id" , how="left")
    test_df  = test_df.merge(rec_test, on= "patient_id" , how="left")
    # tabular engineered
    train_df[ "prior_cost_per_year" ]   = train_df[ "prior_ed_cost_5y_usd" ] / 5.0
    test_df[ "prior_cost_per_year" ]    = test_df[ "prior_ed_cost_5y_usd" ] / 5.0
    train_df[ "prior_visits_per_year" ] = train_df[ "prior_ed_visits_5y" ] / 5.0
    test_df[ "prior_visits_per_year" ]  = test_df[ "prior_ed_visits_5y" ] / 5.0
    train_df[ "prior_cost_per_visit" ]  = train_df[ "prior_ed_cost_5y_usd" ] / (train_
    test_df[ "prior_cost_per_visit" ]   = test_df[ "prior_ed_cost_5y_usd" ] / (test_d
    cat_cols = [ "primary_chronic" , "sex", "insurance" , "zip3"]
    for c in cat_cols:
        train_df[c] = train_df[c].astype( str)
        test_df[c]  = test_df[c].astype( str)
    base_num = [
        "age",
        "prior_ed_visits_5y" ,
        "prior_ed_cost_5y_usd" ,
        "prior_cost_per_year" ,
        "prior_visits_per_year" ,
        "prior_cost_per_visit" ,
    ]

```


### Page 112

```text

    receipt_num = [
        "n_items" ,"n_unique_codes" ,"top1_share" ,"top3_share" ,"distributed_cost" ,
        "hhi","gini","bucket_entropy" ,
        "n_em_9928x" ,"max_em_level_9928x" ,
        "has_critical_care" ,"n_99291" ,"n_99292" ,"cc_time_proxy" ,
        "n_severe_proc" ,"severe_proc_share" ,
        "n_hi_diag" ,"hi_diag_share" ,
        "severity_score" ,
    ]
    curated_feats = []
    for c in CURATED_CODES:
        curated_feats += [ f"has_{c}", f"cnt_{c}", f"share_ {c}"]
    num_cols = base_num + receipt_num + curated_feats
    for df in [train_df, test_df]:
        for c in num_cols:
            if c not in df.columns:
                df[c] = 0.0
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors= "coerce" ).fillna(
    all_cat = pd.concat([train_df[cat_cols], test_df[cat_cols]], axis= 0, ignore_i
    try:
        ohe = OneHotEncoder(handle_unknown= "ignore" , sparse_output= True)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown= "ignore" , sparse= True)
    X_cat_all = ohe.fit_transform(all_cat)
    X_num_all = sp.csr_matrix(pd.concat([train_df[num_cols], test_df[num_cols]], 
    if USE_HASH_CODES:
        hasher = FeatureHasher(n_features=HASH_DIM, input_type= "dict")
        X_hash_all = hasher.transform(hash_dicts)
        X_all = sp.hstack([X_num_all, X_cat_all, X_hash_all], format="csr")
    else:
        X_all = sp.hstack([X_num_all, X_cat_all], format="csr")

```


### Page 113

```text

    X_train = X_all[: len(train_df)]
    X_test  = X_all[ len(train_df):]
    return train_df, test_df, X_train, X_test
# -----------------------------
# Stratification + calibration
# -----------------------------
def make_strat_labels (train_df: pd.DataFrame, n_bins: int = 10) -> np.ndarray:
    prior = train_df[ "prior_ed_cost_5y_usd" ].astype( float)
    qbin = pd.qcut(prior, q=n_bins, duplicates= "drop").astype( str)
    strat = train_df[ "primary_chronic" ].astype( str) + "|" + qbin
    return strat.values
def group_median_calibrate (train_df, oof_pred, y_true, test_df, test_pred, n_bins
    bins = build_prior_bins(train_df[ "prior_ed_cost_5y_usd" ].astype( float), n_bin
    tr_bin = pd.cut(train_df[ "prior_ed_cost_5y_usd" ].astype( float), bins=bins, in
    te_bin = pd.cut(test_df[ "prior_ed_cost_5y_usd" ].astype( float), bins=bins, inc
    tr_key = (train_df[ "primary_chronic" ].astype( str) + "|" + tr_bin).values
    te_key = (test_df[ "primary_chronic" ].astype( str) + "|" + te_bin).values
    resid = y_true - oof_pred
    shift = pd.Series(resid).groupby(tr_key).median()
    shift_dict = shift.to_dict()
    oof_adj = oof_pred + np.array([shift_dict.get(k, 0.0) for k in tr_key], dtype
    te_adj  = test_pred + np.array([shift_dict.get(k, 0.0) for k in te_key], dtyp
    return oof_adj, te_adj
# -----------------------------
# XGBoost API-robust training wrappers
# -----------------------------
print("xgboost version:" , getattr(xgb, "__version__" , "unknown" ))
def resolve_xgb_params (base_params: dict, want_cuda: bool = True):
    """
    Prefer modern GPU params: tree_method='hist', device='cuda' (official docs).
    Fall back to older gpu_hist if needed; otherwise CPU hist.
    """
    base = dict(base_params)

```


### Page 114

```text

    # remove predictor to avoid API churn (some versions removed it)
    base.pop( "predictor" , None)
    if want_cuda:
        # Preferred modern API
        p1 = dict(base, tree_method= "hist", device= "cuda")
        try:
            _ = xgb.XGBRegressor(**p1)
            return p1, "GPU via device='cuda', tree_method='hist'"
        except TypeError:
            pass
        # Legacy fallback
        p2 = dict(base, tree_method= "gpu_hist" )
        try:
            _ = xgb.XGBRegressor(**p2)
            return p2, "GPU via tree_method='gpu_hist' (legacy)"
        except TypeError:
            pass
    # CPU fallback
    p3 = dict(base, tree_method= "hist")
    p3.pop( "device" , None)
    return p3, "CPU via tree_method='hist'"
def safe_fit (model, X_tr, y_tr, X_va, y_va, early_stopping_rounds= 200, verbose= 20
    """
    Pass only supported kwargs to model.fit() to avoid version-specific TypeError
    """
    sig = inspect.signature(model.fit)
    kwargs = {}
    if "eval_set"  in sig.parameters:
        kwargs[ "eval_set" ] = [(X_va, y_va)]
    if "verbose"  in sig.parameters:
        kwargs[ "verbose" ] = verbose
    elif "verbose_eval"  in sig.parameters:
        kwargs[ "verbose_eval" ] = verbose

```


### Page 115

```text

    if "early_stopping_rounds"  in sig.parameters and early_stopping_rounds is not 
        kwargs[ "early_stopping_rounds" ] = int(early_stopping_rounds)
    elif "callbacks"  in sig.parameters and early_stopping_rounds is not None:
        # callback-based early stopping as fallback
        try:
            from xgboost.callback import EarlyStopping
            kwargs[ "callbacks" ] = [EarlyStopping(rounds= int(early_stopping_rounds
        except Exception:
            pass
    try:
        model.fit(X_tr, y_tr, **kwargs)
        return model
    except TypeError as e:
        # last-resort: drop args one-by-one
        msg = str(e)
        for k in list(kwargs.keys()):
            try:
                kw2 = dict(kwargs)
                kw2.pop(k, None)
                model.fit(X_tr, y_tr, **kw2)
                print(f"[safe_fit] fit() succeeded after dropping ' {k}' (original 
                return model
            except TypeError:
                continue
        raise
def predict_best (model, X ):
    """
    Robust prediction for early-stopped models across xgboost versions.
    """
    # Newer versions: iteration_range
    if hasattr(model, "best_iteration" ) and model.best_iteration is not None:
        bi = int(model.best_iteration)
        try:
            return model.predict(X, iteration_range=( 0, bi + 1))
        except TypeError:
            pass
    # Older versions: best_ntree_limit
    if hasattr(model, "best_ntree_limit" ):

```


### Page 116

```text

        try:
            return model.predict(X, ntree_limit=model.best_ntree_limit)
        except TypeError:
            pass
    return model.predict(X)
def fit_one_model (X_tr, y_tr, X_va, y_va, objective: str, seed: int, want_cuda: b
    # Base params (we set eval_metric in constructor, NOT in fit)  :contentReferen
    base = dict(
        n_estimators= 8000,
        learning_rate= 0.03,
        max_depth= 6,
        min_child_weight= 10,
        subsample= 0.8,
        colsample_bytree= 0.8,
        reg_lambda= 1.0,
        reg_alpha= 0.0,
        random_state=seed,
        n_jobs=- 1,
        verbosity= 1,
        objective=objective,
        eval_metric= "mae",   # <-- FIX: constructor param, not fit()
    )
    params, mode = resolve_xgb_params(base, want_cuda=want_cuda)
    # print(mode)  # uncomment if you want per-model verbosity
    model = xgb.XGBRegressor(**params)
    # try GPU/selected mode; if XGBoostError occurs, fall back to CPU once
    try:
        model = safe_fit(model, X_tr, y_tr, X_va, y_va, early_stopping_rounds=EAR
        return model, mode
    except xgb.core.XGBoostError as e:
        print(f"[WARN] Training failed in mode ' {mode}'. Falling back to CPU. Err
        params_cpu, mode_cpu = resolve_xgb_params(base, want_cuda= False)
        model2 = xgb.XGBRegressor(**params_cpu)
        model2 = safe_fit(model2, X_tr, y_tr, X_va, y_va, early_stopping_rounds=EA
        return model2, mode_cpu

```


### Page 117

```text

# -----------------------------
# CV training
# -----------------------------
def run_cv_training (train_df, X_train, y, test_df, X_test ):
    strat = make_strat_labels(train_df, n_bins= 10)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle= True, random_state=SEED)
    oof = np.zeros( len(train_df), dtype= float)
    test_pred_accum = np.zeros( len(test_df), dtype= float)
    fold_mae = []
    for fold, (tr_idx, va_idx) in enumerate (skf.split(np.zeros( len(train_df)), st
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        m_abs, mode_abs = fit_one_model(X_tr, y_tr, X_va, y_va, objective= "reg:ab
        pred_va_abs = predict_best(m_abs, X_va)
        pred_te_abs = predict_best(m_abs, X_test)
        if USE_PSEUDOHUBER_BLEND:
            m_hub, mode_hub = fit_one_model(X_tr, y_tr, X_va, y_va, objective= "re
            pred_va_hub = predict_best(m_hub, X_va)
            pred_te_hub = predict_best(m_hub, X_test)
            pred_va = BLEND_W_ABS * pred_va_abs + BLEND_W_HUBER * pred_va_hub
            pred_te = BLEND_W_ABS * pred_te_abs + BLEND_W_HUBER * pred_te_hub
        else:
            pred_va = pred_va_abs
            pred_te = pred_te_abs
        oof[va_idx] = pred_va
        test_pred_accum += pred_te / N_SPLITS
        mae = mean_absolute_error(y_va, pred_va)
        fold_mae.append(mae)
        print(f"[Fold {fold}/{N_SPLITS} ] MAE={mae:.6f}")
    print(f"\nOOF MAE mean(folds): {np.mean(fold_mae): .6f} | OOF MAE global: {mea
    return oof, test_pred_accum

```


### Page 118

```text

# ============================================================
# MAIN
# ============================================================
t0 = time.time()
print("Loading CSVs..." )
train = normalize_cols(pd.read_csv(TRAIN_CSV))
test  = normalize_cols(pd.read_csv(TEST_CSV))
patients = normalize_cols(pd.read_csv(PATIENTS_CSV))
assert RECEIPT_CACHE.exists(), f"Missing receipt cache: {RECEIPT_CACHE} "
print("Loading receipts cache..." )
receipts = load(RECEIPT_CACHE)
if not isinstance (receipts, dict) or len(receipts) < 3000:
    raise RuntimeError( "receipts_parsed.joblib has unexpected structure/size. Reb
# Load/build feature matrices
if FEAT_CACHE.exists():
    print("Loading cached feature matrices:" , FEAT_CACHE)
    cache = load(FEAT_CACHE)
    train2 = cache[ "train2" ]
    test2 = cache[ "test2"]
    X_train = cache[ "X_train" ]
    X_test = cache[ "X_test" ]
else:
    print("Building features (first time)..." )
    train2, test2, X_train, X_test = build_features(train, test, patients, receipt
    dump({ "train2" : train2, "test2": test2, "X_train" : X_train, "X_test" : X_test}
    print("Saved feature cache:" , FEAT_CACHE)
y = train2[ "ed_cost_next3y_usd" ].astype( float).values
print("\nTraining CV..." )
oof, test_pred = run_cv_training(train2, X_train, y, test2, X_test)
if USE_GROUP_MEDIAN_CALIBRATION:
    print("\nApplying group-median calibration (primary_chronic x prior_cost_bin)
    oof_cal, test_cal = group_median_calibrate(train2, oof, y, test2, test_pred, 
    print(f"OOF MAE raw: {mean_absolute_error(y, oof): .6f} | calibrated: {mean_ab
    test_pred_final = test_cal
else:

```


### Page 119

```text

Why this fix es your crash
eval_metric  is now set in the model construct or (eval_metric="mae" ) and never
passed t o fit() , matching current X GBoost sklearn guidance. 
GPU config is modern : tree_method="hist", device="cuda"  with fallback to older
gpu_hist  or CPU. 
If you run this and it still throws a new error, paste the first ~20 lines of the traceback and
the printed xgboost version:  line — I’ll adapt the wrapper to that exact signature.    test_pred_final = test_pred
test_pred_final = np.clip(test_pred_final, 0.0, None)
sub = pd.DataFrame({
    "patient_id" : test2[ "patient_id" ].astype( int).values,
    "ed_cost_next3y_usd" : test_pred_final
})
sub.to_csv(SUBMISSION_PATH, index= False)
print(f"\nSaved submission -> {SUBMISSION_PATH} ")
print(f"Done in {time.time()-t0: .1f}s")
XGBoost Docum …+1
XGBoost Docum …
Sour ces
Sour ces
xgboost version: 3.0.0
Loading CSV s...
Loading receipts cache...
Loading cached feature matrices:  
D:\AgentDs\agent_ds_healthcare\cache_iter10\feat_matrix_
cache.joblib
Training CV...
[0] validation_0-mae:1386.40011
[200] validation_0-mae:452.14857
[400] validation_0-mae:449.13321
[600] validation_0-mae:449.27528
[800] validation_0-mae:449.70555

```


### Page 120

```text

[1000] validation_0-mae:449.68491
[1200] validation_0-mae:450.68834
[1400] validation_0-mae:450.88746
[1600] validation_0-mae:451.38257
[1800] validation_0-mae:451.55949
[2000] validation_0-mae:451.86153
[2200] validation_0-mae:452.02730
[2400] validation_0-mae:451.98221
[2600] validation_0-mae:451.97315
[2800] validation_0-mae:451.88378
[3000] validation_0-mae:452.47768
[3200] validation_0-mae:452.47586
[3400] validation_0-mae:452.32695
[3600] validation_0-mae:452.44605
[3800] validation_0-mae:452.36298
[4000] validation_0-mae:452.56180
[4200] validation_0-mae:452.62484
[4400] validation_0-mae:452.63789
[4600] validation_0-mae:452.55240
[4800] validation_0-mae:452.58254
[5000] validation_0-mae:452.75438
[5200] validation_0-mae:452.67358
[5400] validation_0-mae:452.81111
[5600] validation_0-mae:452.87176
[5800] validation_0-mae:452.98698
[6000] validation_0-mae:453.07104
[6200] validation_0-mae:453.03530
[6400] validation_0-mae:453.19724
[6600] validation_0-mae:453.22210
[6800] validation_0-mae:453.28672
[7000] validation_0-mae:453.32281
[7200] validation_0-mae:453.46055
[7400] validation_0-mae:453.46147
[7600] validation_0-mae:453.46577
[7800] validation_0-mae:453.47923
[7999] validation_0-mae:453.49593
[0] validation_0-mae:1599995972.48000
[200] validation_0-mae:1599995972.48000
[400] validation_0-mae:1599995972.48000
[600] validation_0-mae:1599995972.48000

```


### Page 121

```text

[800] validation_0-mae:1599995972.48000
[1000] validation_0-mae:1599995972.48000
[1200] validation_0-mae:1599995972.48000
[1400] validation_0-mae:1599995972.48000
[1600] validation_0-mae:1599995972.48000
[1800] validation_0-mae:1599995972.48000
[2000] validation_0-mae:1599995972.48000
[2200] validation_0-mae:1599995972.48000
[2400] validation_0-mae:1599995972.48000
[2600] validation_0-mae:1599995972.48000
[2800] validation_0-mae:1599995972.48000
[3000] validation_0-mae:1599995972.48000
[3200] validation_0-mae:1599995972.48000
[3400] validation_0-mae:1599995972.48000
[3600] validation_0-mae:1599995972.48000
[3800] validation_0-mae:1599995972.48000
[4000] validation_0-mae:1599995972.48000
[4200] validation_0-mae:1599995972.48000
[4400] validation_0-mae:1599995972.48000
[4600] validation_0-mae:1599995972.48000
[4800] validation_0-mae:1599995972.48000
[5000] validation_0-mae:1599995972.48000
[5200] validation_0-mae:1599995972.48000
[5400] validation_0-mae:1599995972.48000
[5600] validation_0-mae:1599995972.48000
[5800] validation_0-mae:1599995972.48000
[6000] validation_0-mae:1599995972.48000
[6200] validation_0-mae:1599995972.48000
[6400] validation_0-mae:1599995972.48000
[6600] validation_0-mae:1599995972.48000
[6800] validation_0-mae:1599995972.48000
[7000] validation_0-mae:1599995972.48000
[7200] validation_0-mae:1599995972.48000
[7400] validation_0-mae:1599995972.48000
[7600] validation_0-mae:1599995972.48000
[7800] validation_0-mae:1599995972.48000
[7999] validation_0-mae:1599995972.48000
[Fold 1/5] MAE=479998797.222300
[0] validation_0-mae:1421.75675
[200] validation_0-mae:447.07365

```


### Page 122

```text

[400] validation_0-mae:438.18042
I stopped training early since I found that the validation  
mae is very very poor, you must iteratively update the  
code to ensure robustness, and also taking consider to our  
previous eda results. Also , add better log info so we can  
analyze the log and compare with actual real MAE later on.
You hit two sep arate issues :
1. reg:pseudohubererror  is behaving lik e a br oken/null objectiv e in your run (it
flatlines with a constant huge MAE). There are known reports of pseudo-Huber
behaving like a null model / constant prediction in X GBoost. 
→ For robustness, disable pseudo-Huber  for now and stick to MAE-aligned
reg:absoluteerror .
2. Early st opping wasn’t being applied  (your MAE kept printing to 7999 rounds). In
newer X GBoost sklearn interface, eval_metric , early_stopping_rounds , and
callbacks  were remov ed fr om fit()  and must be passed in the construct or.
→ We’ll set early_stopping_rounds  in the model constructor (so it actually stops).
We’ll also keep GPU training using X GBoost’s device=cuda  parameter. 
Below is an updat ed, mor e robust 1-cell  pipeline with better logging + baselines. It will:
Train only reg:absoluteerror  (robust + directly aligned with MAE). 
Use construct or-based early st opping  (works in X GBoost 3.0.0). 
Print per-fold:
baseline MAE (group median by primary_chronic × prior_cost_bin )
model MAE
best_iteration / best_score
prediction sanity stats
Save:
run_dir/oof_predictions.csv
run_dir/submission.csv
run_dir/run_log.json
Version-stamps the feature cache so you don’t silently reuse stale matrices.GitHub
XGBoost Docum …
XGBoost Docum …
XGBoost Docum …
XGBoost Docum …

```


### Page 123

```text

python
# ============================================================
# AgentDS Challenge 2 — Robust XGBoost 3.0.0 Training (ONE CELL)
# Fixes:
#  - No pseudohuber (known to behave like null/constant in some setups)
#  - Early stopping moved to constructor (XGBoost sklearn breaking change)
#  - Stronger logs + baselines + sanity checks
# ============================================================
import os, re, time, json, math, warnings
from pathlib import Path
from datetime import datetime
from collections import Counter
import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import load, dump
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
warnings.filterwarnings( "ignore" )
np.set_printoptions(suppress= True)
# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path( r"D:\AgentDs\agent_ds_healthcare" )
TRAIN_CSV = BASE_DIR / "ed_cost_train.csv"
TEST_CSV  = BASE_DIR / "ed_cost_test.csv"
PATIENTS_CSV = BASE_DIR / "patients.csv"
CACHE_DIR = BASE_DIR / "cache_iter10"
CACHE_DIR.mkdir(parents= True, exist_ok= True)

```


### Page 124

```text

RECEIPT_CACHE = CACHE_DIR / "receipts_parsed.joblib"
SUBMISSION_PATH = BASE_DIR / "submission_ICHI_V1.csv"
# -----------------------------
# Run + cache versions (prevents stale feature reuse)
# -----------------------------
PIPELINE_VERSION = "edcost_xgb_abs_es_ctor_v3_2026-02-14"
FEAT_CACHE = CACHE_DIR / "feat_matrix_cache_v3.joblib"   # new filename to avoid s
RUN_ID = datetime.now().strftime( "%Y%m%d_%H%M%S" )
RUN_DIR = CACHE_DIR / f"run_{RUN_ID} "
RUN_DIR.mkdir(parents= True, exist_ok= True)
def log(msg):
    ts = datetime.now().strftime( "%H:%M:%S" )
    print(f"[{ts}] {msg}")
# -----------------------------
# Config
# -----------------------------
SEED = 42
N_SPLITS = 5
# sparse hashing of codes (captures long tail beyond curated codes)
USE_HASH_CODES = True
HASH_DIM = 2**16
# XGB training
N_ESTIMATORS = 5000
LEARNING_RATE = 0.05
MAX_DEPTH = 6
MIN_CHILD_WEIGHT = 10
SUBSAMPLE = 0.8
COLSAMPLE = 0.8
REG_LAMBDA = 1.0
REG_ALPHA = 0.0
EARLY_STOP = 250         # constructor-based
VERBOSE_EVAL = 200       # print every N rounds
WANT_CUDA = True         # will fall back to CPU automatically if GPU fails

```


### Page 125

```text

# Calibration
USE_GROUP_MEDIAN_CALIBRATION = True
CALIB_BINS = 10
# -----------------------------
# Helpers
# -----------------------------
def set_seed (seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
set_seed(SEED)
def normalize_cols (df):
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df
def safe_float (x):
    if x is None:
        return np.nan
    if isinstance (x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().replace( "$","").replace( ",","")
    try:
        return float(s)
    except Exception:
        return np.nan
def code_bucket (code: str) -> str:
    if not isinstance (code, str) or not code:
        return "unk"
    if code.isdigit():
        if code.startswith( "99"):
            return "em_99"
        return f"num_{code[0]}"
    return f"alpha_ {code[0].upper()}"

```


### Page 126

```text

def entropy_from_counts (counts: Counter ) -> float:
    if not counts:
        return 0.0
    vals = np.array( list(counts.values()), dtype= float)
    p = vals / (vals. sum() + 1e-12)
    return float(-(p * np.log(p + 1e-12)).sum())
def gini_from_amounts (x: np.ndarray ) -> float:
    x = x.astype( float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return 0.0
    if np.all(x == 0):
        return 0.0
    x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(x)
    g = (n + 1 - 2 * np.sum(cumx) / (cumx[- 1] + 1e-12)) / n
    return float(g)
def build_prior_bins (prior: pd.Series, n_bins: int = 10):
    q = np.quantile(prior.values.astype( float), np.linspace( 0, 1, n_bins + 1))
    bins = np.unique(q)
    if len(bins) < 3:
        bins = np.unique(np.quantile(prior.values.astype( float), [0, 0.5, 1.0]))
    return bins
def make_strat_labels (df: pd.DataFrame, n_bins: int = 10) -> np.ndarray:
    qbin = pd.qcut(df[ "prior_ed_cost_5y_usd" ].astype( float), q=n_bins, duplicates
    return (df["primary_chronic" ].astype( str) + "|" + qbin).values
def predict_best (model, X ):
    bi = getattr(model, "best_iteration" , None)
    if bi is not None:
        try:
            return model.predict(X, iteration_range=( 0, int(bi) + 1))
        except TypeError:
            pass
    return model.predict(X)

```


### Page 127

```text

def sanity_stats (name, arr ):
    arr = np.asarray(arr, dtype= float)
    return {
        f"{name}_min": float(np.nanmin(arr)),
        f"{name}_max": float(np.nanmax(arr)),
        f"{name}_mean": float(np.nanmean(arr)),
        f"{name}_p01": float(np.nanpercentile(arr, 1)),
        f"{name}_p50": float(np.nanpercentile(arr, 50)),
        f"{name}_p99": float(np.nanpercentile(arr, 99)),
    }
def ensure_finite (pred, context= "pred"):
    if not np.all(np.isfinite(pred)):
        raise RuntimeError( f"[SANITY FAIL] Non-finite values in {context} .")
    if np.nanmax(np. abs(pred)) > 1e7:
        raise RuntimeError( f"[SANITY FAIL] {context}  has extreme magnitude (>1e7)
# -----------------------------
# Evidence-driven code lists
# -----------------------------
SEVERE_PROC = [ "31500", "92950", "36556", "36620"]
CRIT_CARE   = [ "99291", "99292"]
HI_DIAG     = [ "70450", "74177", "84484"]
OBS         = [ "G0378"]
ED_LEVELS   = [ "99281", "99282", "99283", "99284", "99285"]
CURATED_CODES = SEVERE_PROC + CRIT_CARE + HI_DIAG + OBS + ED_LEVELS
# -----------------------------
# Receipt feature engineering (from receipts_parsed.joblib)
# -----------------------------
def receipt_features_from_items (obj: dict) -> dict:
    items = obj.get( "items", []) if isinstance (obj, dict) else []
    if not items:
        out = {
            "n_items" : 0, "n_unique_codes" : 0,
            "top1_share" : 0.0, "top3_share" : 0.0, "distributed_cost" : 0.0,
            "hhi": 1.0, "gini": 0.0, "bucket_entropy" : 0.0,
            "n_em_9928x" : 0, "max_em_level_9928x" : 0.0,
            "has_critical_care" : 0, "n_99291" : 0, "n_99292" : 0, "cc_time_proxy" : 
            "n_severe_proc" : 0, "severe_proc_share" : 0.0,

```


### Page 128

```text

            "n_hi_diag" : 0, "hi_diag_share" : 0.0,
            "severity_score" : 0.0
        }
        for c in CURATED_CODES:
            out[ f"has_{c}"] = 0
            out[ f"cnt_{c}"] = 0
            out[ f"share_ {c}"] = 0.0
        return out
    codes = []
    amts = []
    for it in items:
        if not isinstance (it, dict):
            continue
        c = str(it.get( "code","")).strip()
        a = safe_float(it.get( "amount" , np.nan))
        if c:
            codes.append(c)
        amts.append(a)
    amts = np.array(amts, dtype= float)
    total = float(np.nansum(amts))
    if not np.isfinite(total) or total <= 0:
        total = 1e-9
    cc = Counter(codes)
    n_items = int(len(items))
    n_unique = int(len(cc))
    top_sorted = np.sort(amts[~np.isnan(amts)])[::- 1]
    top1 = float(top_sorted[ 0]) if len(top_sorted) else 0.0
    top3 = float(top_sorted[: 3].sum()) if len(top_sorted) >= 3 else float(top_sor
    top1_share = top1 / total
    top3_share = top3 / total
    p = top_sorted / (top_sorted. sum() + 1e-12) if len(top_sorted) else np.array(
    hhi = float(np.sum(p * p)) if len(p) else 1.0
    gini = gini_from_amounts(amts)
    bucket_counts = Counter(code_bucket(c) for c in codes)
    bucket_entropy = entropy_from_counts(bucket_counts)

```


### Page 129

```text

    em9928 = [c for c in codes if c in ED_LEVELS]
    n_em_9928x = len(em9928)
    max_em_level = float(max([int(c[-1]) for c in em9928], default= 0))
    n_99291 = int(cc.get( "99291", 0))
    n_99292 = int(cc.get( "99292", 0))
    has_cc = int((n_99291 + n_99292) > 0)
    cc_time_proxy = float((74 if n_99291 > 0 else 0) + 30 * n_99292)
    n_severe = int(sum(cc.get(c, 0) for c in SEVERE_PROC))
    amt_by_code = Counter()
    for it in items:
        if not isinstance (it, dict):
            continue
        c = str(it.get( "code","")).strip()
        a = safe_float(it.get( "amount" , 0.0))
        if c and np.isfinite(a):
            amt_by_code[c] += float(a)
    amt_severe = float(sum(amt_by_code.get(c, 0.0) for c in SEVERE_PROC))
    severe_share = amt_severe / total
    n_diag = int(sum(cc.get(c, 0) for c in HI_DIAG))
    amt_diag = float(sum(amt_by_code.get(c, 0.0) for c in HI_DIAG))
    diag_share = amt_diag / total
    has_intub = int(cc.get( "31500", 0) > 0)
    has_cpr   = int(cc.get( "92950", 0) > 0)
    has_cvc   = int(cc.get( "36556", 0) > 0)
    has_art   = int(cc.get( "36620", 0) > 0)
    has_cth   = int(cc.get( "70450", 0) > 0)
    has_ctap  = int(cc.get( "74177", 0) > 0)
    has_trop  = int(cc.get( "84484", 0) > 0)
    severity_score = (
        5*has_intub + 5*has_cpr + 4*has_cvc + 4*has_art +
        3*has_cc + 1*has_cth + 1*has_ctap + 1*has_trop
    )

```


### Page 130

```text

    out = {
        "n_items" : n_items,
        "n_unique_codes" : n_unique,
        "top1_share" : float(top1_share),
        "top3_share" : float(top3_share),
        "distributed_cost" : float(1.0 - top3_share),
        "hhi": float(hhi),
        "gini": float(gini),
        "bucket_entropy" : float(bucket_entropy),
        "n_em_9928x" : int(n_em_9928x),
        "max_em_level_9928x" : float(max_em_level),
        "has_critical_care" : int(has_cc),
        "n_99291" : int(n_99291),
        "n_99292" : int(n_99292),
        "cc_time_proxy" : float(cc_time_proxy),
        "n_severe_proc" : int(n_severe),
        "severe_proc_share" : float(severe_share),
        "n_hi_diag" : int(n_diag),
        "hi_diag_share" : float(diag_share),
        "severity_score" : float(severity_score),
    }
    for c in CURATED_CODES:
        cnt = int(cc.get(c, 0))
        amt = float(amt_by_code.get(c, 0.0))
        out[ f"has_{c}"] = int(cnt > 0)
        out[ f"cnt_{c}"] = cnt
        out[ f"share_ {c}"] = float(amt / total)
    return out
def build_code_hash_dict (obj: dict) -> dict:
    items = obj.get( "items", []) if isinstance (obj, dict) else []
    codes = []
    for it in items:
        if isinstance (it, dict):
            c = str(it.get( "code","")).strip()
            if c:
                codes.append(c)
    cc = Counter(codes)

```


### Page 131

```text

    bc = Counter(code_bucket(c) for c in codes)
    d = {}
    for c, v in cc.items():
        d[ f"c={c}"] = float(v)
    for b, v in bc.items():
        d[ f"b={b}"] = float(v)
    return d
# -----------------------------
# Build feature matrices
# -----------------------------
def build_features (train_df, test_df, patients_df, receipts_dict ):
    train_df = train_df.merge(patients_df, on= "patient_id" , how="left")
    test_df  = test_df.merge(patients_df, on= "patient_id" , how="left")
    all_ids = pd.concat([train_df[ "patient_id" ], test_df[ "patient_id" ]], axis= 0).
    rec_rows = []
    hash_dicts = []
    missing = 0
    for pid in all_ids:
        obj = receipts_dict.get( int(pid))
        if obj is None:
            missing += 1
            obj = { "items": []}
        rec_rows.append({ "patient_id" : int(pid), **receipt_features_from_items(ob
        if USE_HASH_CODES:
            hash_dicts.append(build_code_hash_dict(obj))
    if missing:
        log( f"[WARN] Missing receipts for {missing}  ids; filled with empty receipt
    rec_feat = pd.DataFrame(rec_rows)
    rec_train = rec_feat.iloc[: len(train_df)].copy()
    rec_test  = rec_feat.iloc[ len(train_df):].copy()
    train_df = train_df.merge(rec_train, on= "patient_id" , how="left")
    test_df  = test_df.merge(rec_test, on= "patient_id" , how="left")

```


### Page 132

```text

    # Tabular engineered features
    train_df[ "prior_cost_per_year" ]   = train_df[ "prior_ed_cost_5y_usd" ] / 5.0
    test_df[ "prior_cost_per_year" ]    = test_df[ "prior_ed_cost_5y_usd" ] / 5.0
    train_df[ "prior_visits_per_year" ] = train_df[ "prior_ed_visits_5y" ] / 5.0
    test_df[ "prior_visits_per_year" ]  = test_df[ "prior_ed_visits_5y" ] / 5.0
    train_df[ "prior_cost_per_visit" ]  = train_df[ "prior_ed_cost_5y_usd" ] / (train_
    test_df[ "prior_cost_per_visit" ]   = test_df[ "prior_ed_cost_5y_usd" ] / (test_d
    cat_cols = [ "primary_chronic" , "sex", "insurance" , "zip3"]
    for c in cat_cols:
        train_df[c] = train_df[c].astype( str)
        test_df[c]  = test_df[c].astype( str)
    base_num = [
        "age",
        "prior_ed_visits_5y" ,
        "prior_ed_cost_5y_usd" ,
        "prior_cost_per_year" ,
        "prior_visits_per_year" ,
        "prior_cost_per_visit" ,
    ]
    receipt_num = [
        "n_items" ,"n_unique_codes" ,"top1_share" ,"top3_share" ,"distributed_cost" ,
        "hhi","gini","bucket_entropy" ,
        "n_em_9928x" ,"max_em_level_9928x" ,
        "has_critical_care" ,"n_99291" ,"n_99292" ,"cc_time_proxy" ,
        "n_severe_proc" ,"severe_proc_share" ,
        "n_hi_diag" ,"hi_diag_share" ,
        "severity_score" ,
    ]
    curated = []
    for c in CURATED_CODES:
        curated += [ f"has_{c}", f"cnt_{c}", f"share_ {c}"]
    num_cols = base_num + receipt_num + curated
    for df in [train_df, test_df]:
        for c in num_cols:
            if c not in df.columns:
                df[c] = 0.0

```


### Page 133

```text

        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors= "coerce" ).fillna(
    all_cat = pd.concat([train_df[cat_cols], test_df[cat_cols]], axis= 0, ignore_i
    try:
        ohe = OneHotEncoder(handle_unknown= "ignore" , sparse_output= True)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown= "ignore" , sparse= True)
    X_cat_all = ohe.fit_transform(all_cat)
    X_num_all = sp.csr_matrix(pd.concat([train_df[num_cols], test_df[num_cols]], 
    if USE_HASH_CODES:
        hasher = FeatureHasher(n_features=HASH_DIM, input_type= "dict")
        X_hash_all = hasher.transform(hash_dicts)
        X_all = sp.hstack([X_num_all, X_cat_all, X_hash_all], format="csr")
    else:
        X_all = sp.hstack([X_num_all, X_cat_all], format="csr")
    X_train = X_all[: len(train_df)]
    X_test  = X_all[ len(train_df):]
    return train_df, test_df, X_train, X_test
# -----------------------------
# XGB constructor params (XGBoost 3 requires early_stopping_rounds in ctor)
# -----------------------------
def make_xgb_regressor (seed: int, use_cuda: bool = True):
    params = dict(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        min_child_weight=MIN_CHILD_WEIGHT,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE,
        reg_lambda=REG_LAMBDA,
        reg_alpha=REG_ALPHA,
        random_state=seed,
        n_jobs=- 1,
        verbosity= 1,
        objective= "reg:absoluteerror" ,

```


### Page 134

```text

        eval_metric= "mae",
        early_stopping_rounds=EARLY_STOP,   # <-- KEY FIX for XGBoost >=2.1.4 / 3
        tree_method= "hist",
    )
    if use_cuda:
        params[ "device" ] = "cuda"  # GPU device param
    else:
        params[ "device" ] = "cpu"
    return xgb.XGBRegressor(**params)
def group_median_baseline (train_df, y_tr, idx_tr, idx_va, bins ):
    tr = train_df.iloc[idx_tr].copy()
    va = train_df.iloc[idx_va].copy()
    tr_bin = pd.cut(tr[ "prior_ed_cost_5y_usd" ].astype( float), bins=bins, include_
    va_bin = pd.cut(va[ "prior_ed_cost_5y_usd" ].astype( float), bins=bins, include_
    tr_key = (tr[ "primary_chronic" ].astype( str) + "|" + tr_bin).values
    va_key = (va[ "primary_chronic" ].astype( str) + "|" + va_bin).values
    med = pd.Series(y_tr).groupby(tr_key).median()
    med_dict = med.to_dict()
    global_med = float(np.median(y_tr))
    pred_va = np.array([med_dict.get(k, global_med) for k in va_key], dtype= float
    return pred_va
def group_median_calibrate (train_df, oof_pred, y_true, test_df, test_pred, n_bins
    bins = build_prior_bins(train_df[ "prior_ed_cost_5y_usd" ].astype( float), n_bin
    tr_bin = pd.cut(train_df[ "prior_ed_cost_5y_usd" ].astype( float), bins=bins, in
    te_bin = pd.cut(test_df[ "prior_ed_cost_5y_usd" ].astype( float), bins=bins, inc
    tr_key = (train_df[ "primary_chronic" ].astype( str) + "|" + tr_bin).values
    te_key = (test_df[ "primary_chronic" ].astype( str) + "|" + te_bin).values
    resid = y_true - oof_pred
    shift = pd.Series(resid).groupby(tr_key).median().to_dict()
    oof_adj = oof_pred + np.array([shift.get(k, 0.0) for k in tr_key], dtype= float
    te_adj  = test_pred + np.array([shift.get(k, 0.0) for k in te_key], dtype= flo
    return oof_adj, te_adj, shift
# ============================================================
# MAINChatGPT  5.2 Thinking

```


### Page 135

```text

# ============================================================
log(f"xgboost version: {getattr(xgb, '__version__' , 'unknown' )}")
log(f"PIPELINE_VERSION: {PIPELINE_VERSION} ")
log("Loading CSVs..." )
train = normalize_cols(pd.read_csv(TRAIN_CSV))
test  = normalize_cols(pd.read_csv(TEST_CSV))
patients = normalize_cols(pd.read_csv(PATIENTS_CSV))
assert RECEIPT_CACHE.exists(), f"Missing {RECEIPT_CACHE} "
log("Loading receipts cache..." )
receipts = load(RECEIPT_CACHE)
if not isinstance (receipts, dict) or len(receipts) < 3000:
    raise RuntimeError( "receipts_parsed.joblib has unexpected structure/size. Reb
# Load/build feature matrices
need_rebuild = True
if FEAT_CACHE.exists():
    try:
        cache = load(FEAT_CACHE)
        if cache.get( "version" ) == PIPELINE_VERSION:
            train2 = cache[ "train2" ]; test2 = cache[ "test2"]
            X_train = cache[ "X_train" ]; X_test = cache[ "X_test" ]
            need_rebuild = False
            log( f"Loaded feature cache: {FEAT_CACHE} ")
    except Exception:
        need_rebuild = True
if need_rebuild:
    log( "Building features (cache miss/version change)..." )
    train2, test2, X_train, X_test = build_features(train, test, patients, receipt
    dump({ "version" : PIPELINE_VERSION, "train2" : train2, "test2": test2, "X_train
    log( f"Saved feature cache: {FEAT_CACHE} ")
y = train2[ "ed_cost_next3y_usd" ].astype( float).values
log(f"X_train shape= {X_train.shape} , nnz={X_train.nnz} , density= {X_train.nnz/(X_t
log(f"Target stats: mean= {y.mean(): .3f}, p50= {np.percentile(y, 50):.3f}, p95= {np.p
# CV
strat = make_strat_labels(train2, n_bins= 10)
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle= True, random_state=SEED)

```


### Page 136

```text

oof = np.zeros( len(train2), dtype= float)
test_pred = np.zeros( len(test2), dtype= float)
global_bins = build_prior_bins(train2[ "prior_ed_cost_5y_usd" ].astype( float), n_bi
fold_logs = []
log("Starting CV training..." )
for fold, (tr_idx, va_idx) in enumerate (skf.split(np.zeros( len(train2)), strat), 
    t_fold = time.time()
    X_tr, X_va = X_train[tr_idx], X_train[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    # Baseline: group-median by (primary_chronic x prior_cost_bin)
    base_va = group_median_baseline(train2, y_tr, tr_idx, va_idx, bins=global_bin
    base_mae = mean_absolute_error(y_va, base_va)
    # Train model
    model = make_xgb_regressor(seed=SEED + fold, use_cuda=WANT_CUDA)
    try:
        # eval_set includes train + valid for better logs (valid is validation_1)
        model.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_va, y_va)], verbose=VERBO
        train_mode = "cuda" if WANT_CUDA else "cpu"
    except xgb.core.XGBoostError as e:
        log( f"[Fold {fold}] GPU training failed; falling back to CPU. Error: {rep
        model = make_xgb_regressor(seed=SEED + fold, use_cuda= False)
        model.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_va, y_va)], verbose=VERBO
        train_mode = "cpu"
    pred_va = predict_best(model, X_va)
    pred_te = predict_best(model, X_test)
    # Sanity checks
    ensure_finite(pred_va, context= f"fold{fold}_pred_va")
    ensure_finite(pred_te, context= f"fold{fold}_pred_te")
    # Clip to non-negative costs
    pred_va = np.clip(pred_va, 0.0, None)
    pred_te = np.clip(pred_te, 0.0, None)

```


### Page 137

```text

    fold_mae = mean_absolute_error(y_va, pred_va)
    oof[va_idx] = pred_va
    test_pred += pred_te / N_SPLITS
    best_iter = getattr(model, "best_iteration" , None)
    best_score = getattr(model, "best_score" , None)
    info = {
        "fold": fold,
        "train_mode" : train_mode,
        "n_tr": int(len(tr_idx)),
        "n_va": int(len(va_idx)),
        "baseline_group_median_mae" : float(base_mae),
        "model_mae" : float(fold_mae),
        "mae_improvement_vs_baseline" : float(base_mae - fold_mae),
        "best_iteration" : None if best_iter is None else int(best_iter),
        "best_score" : None if best_score is None else float(best_score),
        "time_sec" : float(time.time() - t_fold),
        **sanity_stats( "pred_va" , pred_va),
        **sanity_stats( "y_va", y_va),
    }
    fold_logs.append(info)
    log( f"[Fold {fold}/{N_SPLITS} ] mode={train_mode}  | baseline_mae= {base_mae: .3f
        f"improve= {base_mae-fold_mae: .3f} | best_iter= {info['best_iteration' ]} | t
oof_mae = mean_absolute_error(y, oof)
log(f"OOF MAE (raw): {oof_mae: .6f}")
# Optional calibration
if USE_GROUP_MEDIAN_CALIBRATION:
    log( "Applying group-median residual calibration (primary_chronic x prior_cost_
    oof_cal, test_cal, shift = group_median_calibrate(train2, oof, y, test2, test_
    oof_mae_cal = mean_absolute_error(y, oof_cal)
    log( f"OOF MAE calibrated: {oof_mae_cal: .6f} (delta= {oof_mae - oof_mae_cal:+ .6
    test_pred_final = np.clip(test_cal, 0.0, None)
else:
    test_pred_final = np.clip(test_pred, 0.0, None)

```


### Page 138

```text

What I need fr om y our next run (so w e can it erate scientifically)
Paste:
The per-fold log lines  (baseline_mae, model_mae, best_iter).
OOF MAE (raw)  and OOF MAE calibrated .
If any fold falls back to CPU (it will say so).
Once we have stable OOF ~450-ish without pathological spikes, the next gain (based on
your ED A) is a two-stage r esidual model  where stage-2 predicts residuals using only# Save outputs
oof_df = pd.DataFrame({ "patient_id" : train2[ "patient_id" ].astype( int), "y_true" : y
oof_df.to_csv(RUN_DIR / "oof_predictions.csv" , index= False)
sub = pd.DataFrame({ "patient_id" : test2[ "patient_id" ].astype( int), "ed_cost_next3y
sub.to_csv(RUN_DIR / "submission.csv" , index= False)
sub.to_csv(SUBMISSION_PATH, index= False)
run_log = {
    "run_id" : RUN_ID,
    "pipeline_version" : PIPELINE_VERSION,
    "xgboost_version" : getattr(xgb, "__version__" , "unknown" ),
    "use_cuda_requested" : WANT_CUDA,
    "n_splits" : N_SPLITS,
    "feature_matrix_shape" : [int(X_train.shape[ 0]), int(X_train.shape[ 1])],
    "feature_matrix_nnz" : int(X_train.nnz),
    "oof_mae_raw" : float(oof_mae),
    "oof_mae_calibrated" : float(oof_mae_cal) if USE_GROUP_MEDIAN_CALIBRATION else 
    "folds": fold_logs,
}
with open(RUN_DIR / "run_log.json" , "w", encoding= "utf-8") as f:
    json.dump(run_log, f, indent= 2)
log(f"Saved run artifacts to: {RUN_DIR} ")
log(f"Saved submission -> {SUBMISSION_PATH} ")

```


### Page 139

```text

receipt phenotype  (severity + complexity) — that’s usually the cleanest way to exploit those
high-lift codes without the “prior cost dominates everything” effect.
Sour ces
Sour ces
System Update: Phase 2 (Modeling V1) Execution  
Complete.
I have executed your v1 pipeline. Below are the raw logs  
from the local run and the actual feedback from the  
competition platform.
1. Internal Logs (Cross-V alidation):
Plaintext
[06:21:12] X_train shape=(2000, 65623), nnz=90215
[06:21:12] T arget stats: mean=3908.252, p50=3569.095 ...
[Fold 1] T rain-MAE: 191.44 | V al-MAE: 460.98
[Fold 2] T rain-MAE: 168.24 | V al-MAE: 440.29
[Fold 4] T rain-MAE: 163.64 | V al-MAE: 458.88
[06:24:10] OOF MAE (raw): 450.806
[06:24:10] OOF MAE (calibrated): 443.163
2. External R esult (Leaderboard Submission):
Plaintext
🚀 Submitting predictions...
✅ Submission successful!
📊 Score: 463.7263 (MAE)
Your T ask:
As the Lead Scientist, review these results.
Diagnosis: Compare the Internal CV (443), External LB  
(463), and the T rain/V al curves in the logs. What does the  
X_train shape combined with the T rain vs V al gap tell you  

```


### Page 140

```text

about our current model's behavior?
Strategize V2: Based on your diagnosis, what is the single  
most critical change we need to make to the pipeline to  
close the Generalization Gap?
Output:
Analysis: The "Why".
The Plan: The precise engineering steps for V2.
The Code: The updated v2 code
Analysis — the “Why”
Your numbers are the classic signature of covariate shif t + high-cap acity sp arse featur es:
Internal CV (calibrat ed) 443 vs LB 463 (~+20 MAE)  means your CV is optimistic
relative to the hidden test distribution. That usually happens when train folds look
like each other , but test looks meaningfully differ ent in feature distribution. This is
precisely the covariate shift setting:  while we hope  is
stable. 
X_train shape (2000, 65623)  with only ~90k non-zeros means you have huge featur e
space dominat ed by hashed code t okens. The model can “memorize” idiosyncratic
code patterns and interactions in the training distribution. That’s why Train-MAE
~160–190  while Val-MAE ~440–460 : high variance/high capacity. When test has a
shifted mix of codes/complexity patterns, that memorization doesn’t transfer.
Your receipt ED A already hinted the structure: “severity phenotype” codes exist, but
the long-tail code mix is complex. Hash features help CV, but they also amplif y shif t
sensitivity .
The single most critical change t o close the gap
Make training shif t-awar e using adv ersarial v alidation → impor tance w eighting.
Mechanism:p  (x)= train p  (x) test p(y∣x)
Journal of Machi …+1

```


### Page 141

```text

1. Train a classifier to distinguish train vs t est using your same features (adversarial
validation). 
2. Convert classifier probabilities to impor tance w eights  
(with balanced train/test, ). This is the standard covariate-shift
correction. 
3. Train your regressor with sample_weight=w  so it learns a model optimized for the
test-like region.
XGBoost supports per-instance weights (via instance weights / DMatrix weight), which is
exactly what we need. 
The Plan — V2 engineering st eps (pr ecise)
1. Keep y our featur e set  (curated “high-lift” + complexity + hashing), but compute
adversarial scor es:
Build X_all = [X_train; X_test]
Label is_test = 0/1
5-fold OOF P(test|x)  + AUC
2. Conv ert to impor tance w eights  for training rows:
w = clip(p/(1-p), w_min, w_max)
normalize mean(w)=1
log weight distribution + adversarial A UC (A UC >> 0.5 confirms shift)
3. Regression training becomes shif t-awar e:
XGBRegressor with objective="reg:absoluteerror"
sample_weight = w_train_fold
pass sample_weight_eval_set  if supported so early stopping monitors weighted
eval too (more test-aligned)
4. CV logging for LB alignment :
report unweight ed MAE  (traditional CV)
report weight ed MAE  using w_val  (proxy for LB under covariate shift)
5. Calibration becomes shif t-awar e (optional but robust):
compute group median residual shifts using weight ed median  residuals (weights
= importance weights)
apply only if weight ed OOF MAE impr ovesKaggle
w(x)≈ p  (x)/p  (x) test train
w∝ p/(1− p)
Journal of Machi …+1
XGBoost Docum …

```


### Page 142

```text

The Code — V2 (one-cell, r obust, lots o f logs)
Copy/paste this into one Jupy ter cell  and run.
python
# ============================================================
# AgentDS Challenge 2 — V2: Shift-Aware Training (Adversarial Importance Weighting
# One-cell pipeline, XGBoost 3.0.0 compatible, robust logging.
#
# Core change vs V1:
#   - Adversarial validation (train vs test) -> importance weights
#   - Train regressor with sample_weight to match test distribution
#
# Theory refs:
#   - Covariate shift / importance weighting: Sugiyama (JMLR 2007)
#   - Adversarial validation practice: Kaggle notebooks
# ============================================================
import os, re, time, json, math, warnings, inspect
from pathlib import Path
from datetime import datetime
from collections import Counter
import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import load, dump
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, roc_auc_score
import xgboost as xgb
warnings.filterwarnings( "ignore" )
np.set_printoptions(suppress= True)
# -----------------------------
# Paths

```


### Page 143

```text

# -----------------------------
BASE_DIR = Path( r"D:\AgentDs\agent_ds_healthcare" )
TRAIN_CSV = BASE_DIR / "ed_cost_train.csv"
TEST_CSV  = BASE_DIR / "ed_cost_test.csv"
PATIENTS_CSV = BASE_DIR / "patients.csv"
CACHE_DIR = BASE_DIR / "cache_iter10"
CACHE_DIR.mkdir(parents= True, exist_ok= True)
RECEIPT_CACHE = CACHE_DIR / "receipts_parsed.joblib"
SUBMISSION_PATH = BASE_DIR / "submission_ICHI_V1.csv"
# New caches
PIPELINE_VERSION = "EDCOST_V2_adv_weighting_2026-02-14"
FEAT_CACHE = CACHE_DIR / "feat_matrix_cache_v2.joblib"
ADV_CACHE  = CACHE_DIR / "adv_scores_cache_v2.joblib"
RUN_ID = datetime.now().strftime( "%Y%m%d_%H%M%S" )
RUN_DIR = CACHE_DIR / f"run_v2_ {RUN_ID} "
RUN_DIR.mkdir(parents= True, exist_ok= True)
def log(msg):
    ts = datetime.now().strftime( "%H:%M:%S" )
    print(f"[{ts}] {msg}")
# -----------------------------
# Config
# -----------------------------
SEED = 42
N_SPLITS_REG = 5
N_SPLITS_ADV = 5
# Features
USE_HASH_CODES = True
HASH_DIM = 2**16
# Importance weights (clip to avoid instability)
W_EPS = 1e-3
W_CLIP_MIN = 0.25
W_CLIP_MAX = 4.0
# XGB (regression)

```


### Page 144

```text

REG_PARAMS = dict(
    n_estimators= 6000,
    learning_rate= 0.04,
    max_depth= 6,
    min_child_weight= 20,      # a bit more conservative than V1
    subsample= 0.8,
    colsample_bytree= 0.75,
    reg_lambda= 5.0,
    reg_alpha= 0.1,
    objective= "reg:absoluteerror" ,
    eval_metric= "mae",
    early_stopping_rounds= 300,
    tree_method= "hist",
    device= "cuda",            # will fall back to cpu if needed
    n_jobs=- 1,
    verbosity= 1,
)
# XGB (adversarial classifier)
ADV_PARAMS = dict(
    n_estimators= 2500,
    learning_rate= 0.05,
    max_depth= 4,
    min_child_weight= 20,
    subsample= 0.8,
    colsample_bytree= 0.8,
    reg_lambda= 5.0,
    reg_alpha= 0.0,
    objective= "binary:logistic" ,
    eval_metric= "auc",
    early_stopping_rounds= 150,
    tree_method= "hist",
    device= "cuda",
    n_jobs=- 1,
    verbosity= 0,
)
VERBOSE_EVAL = 200
# Calibration

```


### Page 145

```text

USE_WEIGHTED_GROUP_CALIBRATION = True
CALIB_BINS = 10
# -----------------------------
# Helpers
# -----------------------------
def set_seed (seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
set_seed(SEED)
def normalize_cols (df):
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df
def safe_float (x):
    if x is None:
        return np.nan
    if isinstance (x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().replace( "$","").replace( ",","")
    try:
        return float(s)
    except Exception:
        return np.nan
def code_bucket (code: str) -> str:
    if not isinstance (code, str) or not code:
        return "unk"
    if code.isdigit():
        if code.startswith( "99"):
            return "em_99"
        return f"num_{code[0]}"
    return f"alpha_ {code[0].upper()}"
def entropy_from_counts (counts: Counter ) -> float:
    if not counts:
        return 0.0

```


### Page 146

```text

    vals = np.array( list(counts.values()), dtype= float)
    p = vals / (vals. sum() + 1e-12)
    return float(-(p * np.log(p + 1e-12)).sum())
def gini_from_amounts (x: np.ndarray ) -> float:
    x = x.astype( float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return 0.0
    if np.all(x == 0):
        return 0.0
    x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(x)
    g = (n + 1 - 2 * np.sum(cumx) / (cumx[- 1] + 1e-12)) / n
    return float(g)
def build_prior_bins (prior: pd.Series, n_bins: int = 10):
    q = np.quantile(prior.values.astype( float), np.linspace( 0, 1, n_bins + 1))
    bins = np.unique(q)
    if len(bins) < 3:
        bins = np.unique(np.quantile(prior.values.astype( float), [0, 0.5, 1.0]))
    return bins
def weighted_mae (y_true, y_pred, w ):
    y_true = np.asarray(y_true, dtype= float)
    y_pred = np.asarray(y_pred, dtype= float)
    w = np.asarray(w, dtype= float)
    w = np.clip(w, 1e-12, None)
    return float(np.sum(w * np. abs(y_true - y_pred)) / np. sum(w))
def weighted_median (values, weights ):
    v = np.asarray(values, dtype= float)
    w = np.asarray(weights, dtype= float)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[m], w[m]
    if len(v) == 0:
        return 0.0
    idx = np.argsort(v)
    v, w = v[idx], w[idx]

```


### Page 147

```text

    cdf = np.cumsum(w)
    cutoff = 0.5 * np.sum(w)
    j = int(np.searchsorted(cdf, cutoff, side= "left"))
    j = min(max(j, 0), len(v)-1)
    return float(v[j])
def predict_best (model, X ):
    bi = getattr(model, "best_iteration" , None)
    if bi is not None:
        try:
            return model.predict(X, iteration_range=( 0, int(bi) + 1))
        except TypeError:
            pass
    return model.predict(X)
def predict_best_proba (model, X ):
    bi = getattr(model, "best_iteration" , None)
    if bi is not None:
        try:
            return model.predict_proba(X, iteration_range=( 0, int(bi) + 1))[:, 1]
        except TypeError:
            pass
    return model.predict_proba(X)[:, 1]
def safe_fit (model, X_tr, y_tr, eval_set= None, sample_weight= None, sample_weight_
    sig = inspect.signature(model.fit)
    kwargs = {}
    if "eval_set"  in sig.parameters and eval_set is not None:
        kwargs[ "eval_set" ] = eval_set
    if "sample_weight"  in sig.parameters and sample_weight is not None:
        kwargs[ "sample_weight" ] = sample_weight
    if "sample_weight_eval_set"  in sig.parameters and sample_weight_eval_set is n
        kwargs[ "sample_weight_eval_set" ] = sample_weight_eval_set
    if "verbose"  in sig.parameters:
        kwargs[ "verbose" ] = verbose
    model.fit(X_tr, y_tr, **kwargs)
    return model
def sanity_stats (arr):
    arr = np.asarray(arr, dtype= float)

```


### Page 148

```text

    return dict(
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        mean= float(np.mean(arr)),
        p01= float(np.percentile(arr, 1)),
        p50= float(np.percentile(arr, 50)),
        p99= float(np.percentile(arr, 99)),
    )
# -----------------------------
# Evidence-driven code lists
# -----------------------------
SEVERE_PROC = [ "31500", "92950", "36556", "36620"]
CRIT_CARE   = [ "99291", "99292"]
HI_DIAG     = [ "70450", "74177", "84484"]
OBS         = [ "G0378"]
ED_LEVELS   = [ "99281", "99282", "99283", "99284", "99285"]
CURATED_CODES = SEVERE_PROC + CRIT_CARE + HI_DIAG + OBS + ED_LEVELS
# -----------------------------
# Receipt features from receipts_parsed.joblib
# -----------------------------
def receipt_features_from_items (obj: dict) -> dict:
    items = obj.get( "items", []) if isinstance (obj, dict) else []
    if not items:
        out = {
            "n_items" : 0, "n_unique_codes" : 0,
            "top1_share" : 0.0, "top3_share" : 0.0, "distributed_cost" : 0.0,
            "hhi": 1.0, "gini": 0.0, "bucket_entropy" : 0.0,
            "n_em_9928x" : 0, "max_em_level_9928x" : 0.0,
            "has_critical_care" : 0, "n_99291" : 0, "n_99292" : 0, "cc_time_proxy" : 
            "n_severe_proc" : 0, "severe_proc_share" : 0.0,
            "n_hi_diag" : 0, "hi_diag_share" : 0.0,
            "severity_score" : 0.0
        }
        for c in CURATED_CODES:
            out[ f"has_{c}"] = 0
            out[ f"cnt_{c}"] = 0
            out[ f"share_ {c}"] = 0.0
        return out

```


### Page 149

```text

    codes, amts = [], []
    for it in items:
        if not isinstance (it, dict):
            continue
        c = str(it.get( "code","")).strip()
        a = safe_float(it.get( "amount" , np.nan))
        if c:
            codes.append(c)
        amts.append(a)
    amts = np.array(amts, dtype= float)
    total = float(np.nansum(amts))
    if not np.isfinite(total) or total <= 0:
        total = 1e-9
    cc = Counter(codes)
    n_items = int(len(items))
    n_unique = int(len(cc))
    top_sorted = np.sort(amts[~np.isnan(amts)])[::- 1]
    top1 = float(top_sorted[ 0]) if len(top_sorted) else 0.0
    top3 = float(top_sorted[: 3].sum()) if len(top_sorted) >= 3 else float(top_sor
    top1_share = top1 / total
    top3_share = top3 / total
    p = top_sorted / (top_sorted. sum() + 1e-12) if len(top_sorted) else np.array(
    hhi = float(np.sum(p * p)) if len(p) else 1.0
    gini = gini_from_amounts(amts)
    bucket_counts = Counter(code_bucket(c) for c in codes)
    bucket_entropy = entropy_from_counts(bucket_counts)
    em9928 = [c for c in codes if c in ED_LEVELS]
    n_em_9928x = len(em9928)
    max_em_level = float(max([int(c[-1]) for c in em9928], default= 0))
    n_99291 = int(cc.get( "99291", 0))
    n_99292 = int(cc.get( "99292", 0))
    has_cc  = int((n_99291 + n_99292) > 0)
    cc_time_proxy = float((74 if n_99291 > 0 else 0) + 30 * n_99292)

```


### Page 150

```text

    n_severe = int(sum(cc.get(c, 0) for c in SEVERE_PROC))
    amt_by_code = Counter()
    for it in items:
        if not isinstance (it, dict):
            continue
        c = str(it.get( "code","")).strip()
        a = safe_float(it.get( "amount" , 0.0))
        if c and np.isfinite(a):
            amt_by_code[c] += float(a)
    amt_severe = float(sum(amt_by_code.get(c, 0.0) for c in SEVERE_PROC))
    severe_share = amt_severe / total
    n_diag = int(sum(cc.get(c, 0) for c in HI_DIAG))
    amt_diag = float(sum(amt_by_code.get(c, 0.0) for c in HI_DIAG))
    diag_share = amt_diag / total
    has_intub = int(cc.get( "31500", 0) > 0)
    has_cpr   = int(cc.get( "92950", 0) > 0)
    has_cvc   = int(cc.get( "36556", 0) > 0)
    has_art   = int(cc.get( "36620", 0) > 0)
    has_cth   = int(cc.get( "70450", 0) > 0)
    has_ctap  = int(cc.get( "74177", 0) > 0)
    has_trop  = int(cc.get( "84484", 0) > 0)
    severity_score = (
        5*has_intub + 5*has_cpr + 4*has_cvc + 4*has_art +
        3*has_cc + 1*has_cth + 1*has_ctap + 1*has_trop
    )
    out = {
        "n_items" : n_items,
        "n_unique_codes" : n_unique,
        "top1_share" : float(top1_share),
        "top3_share" : float(top3_share),
        "distributed_cost" : float(1.0 - top3_share),
        "hhi": float(hhi),
        "gini": float(gini),

```


### Page 151

```text

        "bucket_entropy" : float(bucket_entropy),
        "n_em_9928x" : int(n_em_9928x),
        "max_em_level_9928x" : float(max_em_level),
        "has_critical_care" : int(has_cc),
        "n_99291" : int(n_99291),
        "n_99292" : int(n_99292),
        "cc_time_proxy" : float(cc_time_proxy),
        "n_severe_proc" : int(n_severe),
        "severe_proc_share" : float(severe_share),
        "n_hi_diag" : int(n_diag),
        "hi_diag_share" : float(diag_share),
        "severity_score" : float(severity_score),
    }
    for c in CURATED_CODES:
        cnt = int(cc.get(c, 0))
        amt = float(amt_by_code.get(c, 0.0))
        out[ f"has_{c}"] = int(cnt > 0)
        out[ f"cnt_{c}"] = cnt
        out[ f"share_ {c}"] = float(amt / total)
    return out
def build_code_hash_dict (obj: dict) -> dict:
    items = obj.get( "items", []) if isinstance (obj, dict) else []
    codes = []
    for it in items:
        if isinstance (it, dict):
            c = str(it.get( "code","")).strip()
            if c:
                codes.append(c)
    cc = Counter(codes)
    bc = Counter(code_bucket(c) for c in codes)
    d = {}
    for c, v in cc.items():
        d[ f"c={c}"] = float(v)
    for b, v in bc.items():
        d[ f"b={b}"] = float(v)
    return d

```


### Page 152

```text

# -----------------------------
# Feature build
# -----------------------------
def build_features (train_df, test_df, patients_df, receipts_dict ):
    train_df = train_df.merge(patients_df, on= "patient_id" , how="left")
    test_df  = test_df.merge(patients_df, on= "patient_id" , how="left")
    all_ids = pd.concat([train_df[ "patient_id" ], test_df[ "patient_id" ]], axis= 0).
    rec_rows, hash_dicts = [], []
    for pid in all_ids:
        obj = receipts_dict.get( int(pid), { "items": []})
        rec_rows.append({ "patient_id" : int(pid), **receipt_features_from_items(ob
        if USE_HASH_CODES:
            hash_dicts.append(build_code_hash_dict(obj))
    rec_feat = pd.DataFrame(rec_rows)
    rec_train = rec_feat.iloc[: len(train_df)].copy()
    rec_test  = rec_feat.iloc[ len(train_df):].copy()
    train_df = train_df.merge(rec_train, on= "patient_id" , how="left")
    test_df  = test_df.merge(rec_test, on= "patient_id" , how="left")
    # tabular engineered
    train_df[ "prior_cost_per_year" ]   = train_df[ "prior_ed_cost_5y_usd" ] / 5.0
    test_df[ "prior_cost_per_year" ]    = test_df[ "prior_ed_cost_5y_usd" ] / 5.0
    train_df[ "prior_visits_per_year" ] = train_df[ "prior_ed_visits_5y" ] / 5.0
    test_df[ "prior_visits_per_year" ]  = test_df[ "prior_ed_visits_5y" ] / 5.0
    train_df[ "prior_cost_per_visit" ]  = train_df[ "prior_ed_cost_5y_usd" ] / (train_
    test_df[ "prior_cost_per_visit" ]   = test_df[ "prior_ed_cost_5y_usd" ] / (test_d
    # categoricals
    cat_cols = [ "primary_chronic" , "sex", "insurance" , "zip3"]
    for c in cat_cols:
        train_df[c] = train_df[c].astype( str)
        test_df[c]  = test_df[c].astype( str)
    base_num = [
        "age",
        "prior_ed_visits_5y" ,

```


### Page 153

```text

        "prior_ed_cost_5y_usd" ,
        "prior_cost_per_year" ,
        "prior_visits_per_year" ,
        "prior_cost_per_visit" ,
    ]
    receipt_num = [
        "n_items" ,"n_unique_codes" ,"top1_share" ,"top3_share" ,"distributed_cost" ,
        "hhi","gini","bucket_entropy" ,
        "n_em_9928x" ,"max_em_level_9928x" ,
        "has_critical_care" ,"n_99291" ,"n_99292" ,"cc_time_proxy" ,
        "n_severe_proc" ,"severe_proc_share" ,
        "n_hi_diag" ,"hi_diag_share" ,
        "severity_score" ,
    ]
    curated = []
    for c in CURATED_CODES:
        curated += [ f"has_{c}", f"cnt_{c}", f"share_ {c}"]
    num_cols = base_num + receipt_num + curated
    for df in [train_df, test_df]:
        for c in num_cols:
            if c not in df.columns:
                df[c] = 0.0
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors= "coerce" ).fillna(
    all_cat = pd.concat([train_df[cat_cols], test_df[cat_cols]], axis= 0, ignore_i
    try:
        ohe = OneHotEncoder(handle_unknown= "ignore" , sparse_output= True)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown= "ignore" , sparse= True)
    X_cat_all = ohe.fit_transform(all_cat)
    X_num_all = sp.csr_matrix(pd.concat([train_df[num_cols], test_df[num_cols]], 
    if USE_HASH_CODES:
        hasher = FeatureHasher(n_features=HASH_DIM, input_type= "dict")
        X_hash_all = hasher.transform(hash_dicts)
        X_all = sp.hstack([X_num_all, X_cat_all, X_hash_all], format="csr")
    else:

```


### Page 154

```text

        X_all = sp.hstack([X_num_all, X_cat_all], format="csr")
    X_train = X_all[: len(train_df)]
    X_test  = X_all[ len(train_df):]
    return train_df, test_df, X_train, X_test
# -----------------------------
# Adversarial validation -> weights
# -----------------------------
def adversarial_oof_scores (X_train, X_test ):
    X_all = sp.vstack([X_train, X_test], format="csr")
    y_adv = np.concatenate([np.zeros(X_train.shape[ 0], dtype= int), np.ones(X_test
    skf = StratifiedKFold(n_splits=N_SPLITS_ADV, shuffle= True, random_state=SEED)
    oof = np.zeros( len(y_adv), dtype= float)
    for fold, (tr_idx, va_idx) in enumerate (skf.split(np.zeros( len(y_adv)), y_adv
        clf = xgb.XGBClassifier(**{**ADV_PARAMS, "random_state" : SEED + fold})
        try:
            clf = safe_fit(clf, X_all[tr_idx], y_adv[tr_idx],
                           eval_set=[(X_all[va_idx], y_adv[va_idx])],
                           verbose= False)
        except xgb.core.XGBoostError:
            # CPU fallback
            p = dict(ADV_PARAMS); p[ "device" ] = "cpu"
            clf = xgb.XGBClassifier(**{**p, "random_state" : SEED + fold})
            clf = safe_fit(clf, X_all[tr_idx], y_adv[tr_idx],
                           eval_set=[(X_all[va_idx], y_adv[va_idx])],
                           verbose= False)
        oof[va_idx] = predict_best_proba(clf, X_all[va_idx])
    auc = roc_auc_score(y_adv, oof)
    return oof[:X_train.shape[ 0]], oof[X_train.shape[ 0]:], float(auc)
def probs_to_weights (p_test, clip_min= 0.25, clip_max= 4.0):
    p = np.clip(p_test, W_EPS, 1.0 - W_EPS)
    w = p / ( 1.0 - p)          # with balanced train/test, proportional to p_test/
    w = np.clip(w, clip_min, clip_max)
    w = w / np.mean(w)         # normalize mean weight to 1

```


### Page 155

```text

    return w.astype(np.float32)
# -----------------------------
# Weighted group calibration (optional)
# -----------------------------
def weighted_group_calibration (train_df, oof_pred, y_true, w_train, test_df, test_
    bins = build_prior_bins(train_df[ "prior_ed_cost_5y_usd" ].astype( float), n_bin
    tr_bin = pd.cut(train_df[ "prior_ed_cost_5y_usd" ].astype( float), bins=bins, in
    te_bin = pd.cut(test_df[ "prior_ed_cost_5y_usd" ].astype( float), bins=bins, inc
    tr_key = (train_df[ "primary_chronic" ].astype( str) + "|" + tr_bin).values
    te_key = (test_df[ "primary_chronic" ].astype( str) + "|" + te_bin).values
    resid = y_true - oof_pred
    # weighted median residual per group
    shift = {}
    for k in np.unique(tr_key):
        m = (tr_key == k)
        shift[k] = weighted_median(resid[m], w_train[m])
    oof_adj = oof_pred + np.array([shift.get(k, 0.0) for k in tr_key], dtype= float
    te_adj  = test_pred + np.array([shift.get(k, 0.0) for k in te_key], dtype= flo
    return oof_adj, te_adj, shift
# -----------------------------
# MAIN
# -----------------------------
log(f"xgboost version: {getattr(xgb,'__version__' ,'unknown' )}")
log(f"PIPELINE_VERSION: {PIPELINE_VERSION} ")
train = normalize_cols(pd.read_csv(TRAIN_CSV))
test  = normalize_cols(pd.read_csv(TEST_CSV))
patients = normalize_cols(pd.read_csv(PATIENTS_CSV))
assert RECEIPT_CACHE.exists(), f"Missing: {RECEIPT_CACHE} "
receipts = load(RECEIPT_CACHE)
if not isinstance (receipts, dict) or len(receipts) < 3000:
    raise RuntimeError( "receipts_parsed.joblib structure/size looks wrong. Rebuil
# Features (cached)

```


### Page 156

```text

need_rebuild = True
if FEAT_CACHE.exists():
    try:
        cache = load(FEAT_CACHE)
        if cache.get( "version" ) == PIPELINE_VERSION:
            train2, test2 = cache[ "train2" ], cache[ "test2"]
            X_train, X_test = cache[ "X_train" ], cache[ "X_test" ]
            need_rebuild = False
            log( f"Loaded feature cache: {FEAT_CACHE} ")
    except Exception:
        need_rebuild = True
if need_rebuild:
    log( "Building features (cache miss/version change)..." )
    train2, test2, X_train, X_test = build_features(train, test, patients, receipt
    dump({ "version" : PIPELINE_VERSION, "train2" : train2, "test2": test2, "X_train
    log( f"Saved feature cache: {FEAT_CACHE} ")
y = train2[ "ed_cost_next3y_usd" ].astype( float).values
log(f"X_train shape= {X_train.shape} , nnz={X_train.nnz} ")
log(f"Target: mean= {y.mean(): .3f}, p50= {np.percentile(y, 50):.3f}, p95= {np.percent
# -----------------------------
# Adversarial scores + weights (cached)
# -----------------------------
need_adv = True
if ADV_CACHE.exists():
    try:
        adv = load(ADV_CACHE)
        if adv.get( "version" ) == PIPELINE_VERSION:
            p_train, p_test, adv_auc = adv[ "p_train" ], adv["p_test" ], adv["auc"]
            need_adv = False
            log( f"Loaded adversarial cache: {ADV_CACHE} ")
    except Exception:
        need_adv = True
if need_adv:
    log( "Training adversarial validator (train vs test)..." )
    p_train, p_test, adv_auc = adversarial_oof_scores(X_train, X_test)
    dump({ "version" : PIPELINE_VERSION, "p_train" : p_train, "p_test" : p_test, "auc

```


### Page 157

```text

    log( f"Saved adversarial cache: {ADV_CACHE} ")
log(f"Adversarial AUC (train vs test) = {adv_auc: .4f}  (0.5=no shift; higher=more 
w_train = probs_to_weights(p_train, clip_min=W_CLIP_MIN, clip_max=W_CLIP_MAX)
log(f"Importance weights stats: {sanity_stats(w_train)} ")
# Save adv diagnostics
pd.DataFrame({ "patient_id" : train2[ "patient_id" ].astype( int), "p_test" : p_train, 
pd.DataFrame({ "patient_id" : test2[ "patient_id" ].astype( int), "p_test" : p_test}).t
# -----------------------------
# Regression CV (shift-aware weights)
# Stratify by primary_chronic x prior_cost_decile x adv_score_bin
# -----------------------------
prior_bin = pd.qcut(train2[ "prior_ed_cost_5y_usd" ].astype( float), q=10, duplicate
adv_bin   = pd.qcut(pd.Series(p_train), q= 5, duplicates= "drop").astype( str)
strat = (train2[ "primary_chronic" ].astype( str) + "|" + prior_bin + "|" + adv_bin)
skf = StratifiedKFold(n_splits=N_SPLITS_REG, shuffle= True, random_state=SEED)
oof = np.zeros( len(train2), dtype= float)
test_pred = np.zeros( len(test2), dtype= float)
fold_logs = []
log("Starting shift-aware CV training..." )
for fold, (tr_idx, va_idx) in enumerate (skf.split(np.zeros( len(train2)), strat), 
    t_fold = time.time()
    X_tr, X_va = X_train[tr_idx], X_train[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    w_tr, w_va = w_train[tr_idx], w_train[va_idx]
    reg = xgb.XGBRegressor(**{**REG_PARAMS, "random_state" : SEED + fold})
    # Fit with train weights; pass eval weights if supported (more test-aligned e
    eval_set = [(X_tr, y_tr), (X_va, y_va)]
    eval_wts = [w_tr, w_va]
    try:
        reg = safe_fit(reg, X_tr, y_tr, eval_set=eval_set, sample_weight=w_tr, sa
        mode = "cuda"
    except xgb.core.XGBoostError as e:

```


### Page 158

```text

        # CPU fallback
        p = dict(REG_PARAMS); p[ "device" ] = "cpu"
        reg = xgb.XGBRegressor(**{**p, "random_state" : SEED + fold})
        reg = safe_fit(reg, X_tr, y_tr, eval_set=eval_set, sample_weight=w_tr, sa
        mode = "cpu"
    pred_tr = np.clip(predict_best(reg, X_tr), 0.0, None)
    pred_va = np.clip(predict_best(reg, X_va), 0.0, None)
    pred_te = np.clip(predict_best(reg, X_test), 0.0, None)
    # Metrics
    train_mae = mean_absolute_error(y_tr, pred_tr)
    val_mae   = mean_absolute_error(y_va, pred_va)
    val_wmae  = weighted_mae(y_va, pred_va, w_va)  # proxy for LB under covariate 
    oof[va_idx] = pred_va
    test_pred += pred_te / N_SPLITS_REG
    bi = getattr(reg, "best_iteration" , None)
    bs = getattr(reg, "best_score" , None)
    info = dict(
        fold=fold,
        mode=mode,
        train_mae= float(train_mae),
        val_mae= float(val_mae),
        val_weighted_mae= float(val_wmae),
        best_iteration= None if bi is None else int(bi),
        best_score= None if bs is None else float(bs),
        time_sec= float(time.time() - t_fold),
    )
    fold_logs.append(info)
    log( f"[Fold {fold}/{N_SPLITS_REG} ] mode={mode} | TrainMAE= {train_mae: .2f} | V
        f"Val_wMAE= {val_wmae: .2f} | best_iter= {info['best_iteration' ]} | time= {in
oof_mae = mean_absolute_error(y, oof)
oof_wmae = weighted_mae(y, oof, w_train)
log(f"OOF MAE (unweighted): {oof_mae: .6f}")
log(f"OOF MAE (importance-weighted proxy): {oof_wmae: .6f}")

```


### Page 159

```text

# -----------------------------
# Optional weighted calibration (apply only if it improves weighted OOF)
# -----------------------------
test_pred_final = test_pred.copy()
cal_used = False
cal_info = {}
if USE_WEIGHTED_GROUP_CALIBRATION:
    log( "Trying weighted group-median residual calibration..." )
    oof_cal, test_cal, shift = weighted_group_calibration(train2, oof, y, w_train
    oof_mae_cal  = mean_absolute_error(y, oof_cal)
    oof_wmae_cal = weighted_mae(y, oof_cal, w_train)
    log( f"Calibrated OOF MAE: {oof_mae_cal: .6f} (delta {oof_mae - oof_mae_cal:+ .6
    log( f"Calibrated OOF wMAE: {oof_wmae_cal: .6f} (delta {oof_wmae - oof_wmae_cal
    # decision: only use calibration if weighted proxy improves
    if oof_wmae_cal < oof_wmae - 1e-6:
        test_pred_final = np.clip(test_cal, 0.0, None)
        cal_used = True
        cal_info = { "n_groups" : len(shift), "oof_mae_cal" : float(oof_mae_cal), "o
        log( "Calibration ACCEPTED (improved weighted proxy)." )
    else:
        log( "Calibration REJECTED (did not improve weighted proxy)." )
# -----------------------------
# Save artifacts
# -----------------------------
oof_df = pd.DataFrame({ "patient_id" : train2[ "patient_id" ].astype( int), "y_true" : y
oof_df.to_csv(RUN_DIR/ "oof_predictions.csv" , index= False)
sub = pd.DataFrame({ "patient_id" : test2[ "patient_id" ].astype( int), "ed_cost_next3y
sub.to_csv(RUN_DIR/ "submission.csv" , index= False)
sub.to_csv(SUBMISSION_PATH, index= False)
run_log = dict(
    run_id=RUN_ID,
    pipeline_version=PIPELINE_VERSION,
    xgboost_version= getattr(xgb, "__version__" , "unknown" ),

```


### Page 160

```text

What t o paste back (so w e can it erate like scientists)
1. Adversarial AUC
2. OOF MAE (unweighted)  and OOF MAE (importance-weighted proxy)
3. Whether calibration A CCEPTED  or REJECTED
4. The platform score for this submission
If the weight ed pr oxy tracks the LB tightly (even if unweighted CV doesn’t), then we’ve
successfully aligned the training objective to the actual hidden test distribution — that’s
the fastest route to closing the 20-MAE gap.    adv_auc= float(adv_auc),
    weight_stats=sanity_stats(w_train),
    feature_shape=[ int(X_train.shape[ 0]), int(X_train.shape[ 1])],
    feature_nnz= int(X_train.nnz),
    oof_mae= float(oof_mae),
    oof_wmae= float(oof_wmae),
    calibration_used= bool(cal_used),
    calibration_info=cal_info,
    folds=fold_logs,
)
with open(RUN_DIR/ "run_log.json" , "w", encoding= "utf-8") as f:
    json.dump(run_log, f, indent= 2)
log(f"Saved run artifacts to: {RUN_DIR} ")
log(f"Saved submission -> {SUBMISSION_PATH} ")
Sour ces
Sour ces
System Update: Phase 2 R esults & Diagnosis
I have executed your V2 pipeline (Adversarial W eighting).  
The results are conclusive, but they disprove our previous  
theory.

```


### Page 161

```text

The Hard Evidence:
Adversarial V alidation A UC = 0.5274: This is effectively  
random.
Critical Deduction: There is NO significant covariate shift  
between T rain and T est. The "Adversarial W eighting"  
strategy was mathematically unnecessary and did not help  
(LB stayed flat at ~462).
Persistent Overfitting:
Train MAE: ~230
Validation MAE: ~445
Gap: >200 points. The model is memorizing specific  
patient noise rather than learning generalizable rules.
The Benchmark Gap: The known state-of-the-art for this  
specific challenge is MAE ~400. W e are stuck at 460. W e 
are missing a massive structural component of the  
problem.
Your T ask: The S trategic Pivot
We are hitting a wall. S tandard X GBoost regression on  
these features is capping out. As the Domain Expert, you  
need to re-formulate the problem.
Think about these angles (but do not be limited by them):
The "Zero" Problem: Are we forcing a regressor to predict  
continuous values for patients who might have $0 cost?
Target Definitions: W e are predicting T otal Cost. Is this the  
most stable target? Or should we be modeling Cost =  
(Probability of Visit) * (Cost per Visit)?
Interaction F ailures: W e have primary_chronic and we have  

```


### Page 162

```text

receipt_codes. Are we failing to capture how they amplify  
each other? (e.g., A generic "Chest X-Ray" might be cheap  
for Pneumonia but a signal of impending doom for Heart  
Failure).
Action R equired:
Diagnose the S tagnation: Why did V2 fail to close the  
generalization gap despite the lack of distribution shift?
Propose the "Phase 3" Architecture: Don't just tweak  
parameters. Propose a fundamental change in how we  
model the target or engineer the features to hit that 400  
MAE benchmark.
Generate v3: The new implementation.
Here is also the real log for refernece:
[06:41:14] xgboost version: 3.0.0
[06:41:14] PIPELINE_VERSION:  
EDCOST_V2_adv_weighting_2026-02-14
[06:41:15] Building features (cache miss/version change)...
[06:41:15] Saved feature cache:  
D:\AgentDs\agent_ds_healthcare\cache_iter10\feat_matrix_
cache_v2.joblib
[06:41:15] X_train shape=(2000, 65623), nnz=90215
[06:41:15] T arget: mean=3908.252, p50=3569.095,  
p95=7541.847, max=11184.610
[06:41:15] T raining adversarial validator (train vs test)...
[06:42:01] Saved adversarial cache:  
D:\AgentDs\agent_ds_healthcare\cache_iter10\adv_scores_
cache_v2.joblib
[06:42:01] Adversarial A UC (train vs test) = 0.5274  (0.5=no  
shift; higher=more shift)
[06:42:01] Importance weights stats: {'min':  
0.2361346185207367, 'max': 3.778153896331787, 'mean':  
0.999999995559454, 'p01': 0.2739429384469986, 'p50':  
0.9470024108886719, 'p99': 2.8402921867370603}
[06:42:01] S tarting shift-aware CV training...
[0] validation_0-mae:1413.43999 validation_1-

```


### Page 163

```text

mae:1372.90322
[200] validation_0-mae:284.30608 validation_1-
mae:455.21691
[400] validation_0-mae:242.14643 validation_1-
mae:454.25546
[600] validation_0-mae:222.79154 validation_1-
mae:454.48350
[799] validation_0-mae:211.07954 validation_1-
mae:454.33877
[06:42:43] [Fold 1/5] mode=cuda | T rainMAE=258.77 |  
ValMAE=466.82 | V al_wMAE=453.67 | best_iter=500 |  
time=42.8s
[0] validation_0-mae:1406.38666 validation_1-
mae:1392.23714
[200] validation_0-mae:274.67028 validation_1-
mae:460.06332
[400] validation_0-mae:236.25204 validation_1-
mae:462.78238
[498] validation_0-mae:225.84875 validation_1-
mae:463.56335
[06:43:11] [Fold 2/5] mode=cuda | T rainMAE=300.18 |  
ValMAE=462.31 | V al_wMAE=459.90 | best_iter=198 |  
time=27.8s
[0] validation_0-mae:1418.94886 validation_1-
mae:1350.88774
[200] validation_0-mae:284.90712 validation_1-
mae:447.12073
[400] validation_0-mae:244.96932 validation_1-
mae:445.82572
[600] validation_0-mae:224.51337 validation_1-
mae:445.44840
[800] validation_0-mae:212.42714 validation_1-
mae:445.65982
[984] validation_0-mae:204.26144 validation_1-
mae:445.73722
[06:44:07] [Fold 3/5] mode=cuda | T rainMAE=247.52 |  
ValMAE=440.61 | V al_wMAE=444.91 | best_iter=685 |  
time=55.6s
[0] validation_0-mae:1392.38974 validation_1-
mae:1465.81313

```


### Page 164

```text

[200] validation_0-mae:285.16177 validation_1-
mae:417.74131
[400] validation_0-mae:247.77842 validation_1-
mae:413.89179
[600] validation_0-mae:229.63944 validation_1-
mae:414.00909
[800] validation_0-mae:216.93428 validation_1-
mae:412.84629
[1000] validation_0-mae:208.88580 validation_1-
mae:412.71262
[1200] validation_0-mae:203.50395 validation_1-
mae:412.41718
[1400] validation_0-mae:197.47869 validation_1-
mae:413.08933
[1461] validation_0-mae:195.62666 validation_1-
mae:413.12329
[06:45:21] [Fold 4/5] mode=cuda | T rainMAE=232.47 |  
ValMAE=409.93 | V al_wMAE=412.38 | best_iter=1161 |  
time=74.5s
[0] validation_0-mae:1393.97288 validation_1-
mae:1454.26333
[200] validation_0-mae:274.38666 validation_1-
mae:486.73732
[400] validation_0-mae:233.73922 validation_1-
mae:485.65427
[600] validation_0-mae:216.87505 validation_1-
mae:484.21417
[800] validation_0-mae:204.48856 validation_1-
mae:483.73260
[1000] validation_0-mae:197.61076 validation_1-
mae:483.19832
[1200] validation_0-mae:191.87559 validation_1-
mae:483.09148
[1244] validation_0-mae:190.69605 validation_1-
mae:483.03298
[06:46:30] [Fold 5/5] mode=cuda | T rainMAE=227.28 |  
ValMAE=482.53 | V al_wMAE=482.78 | best_iter=944 |  
time=68.6s
[06:46:30] OOF MAE (unweighted): 452.440004
[06:46:30] OOF MAE (importance-weighted proxy):  

```


### Page 165

```text

450.634481
[06:46:30] T rying weighted group-median residual  
calibration...
[06:46:30] Calibrated OOF MAE: 445.306969 (delta  
+7.133035)
[06:46:30] Calibrated OOF wMAE: 442.796133 (delta  
+7.838348)
[06:46:30] Calibration A CCEPTED (improved weighted  
proxy).
[06:46:30] Saved run artifacts to:  
D:\AgentDs\agent_ds_healthcare\cache_iter10\run_v2_202
60214_064114
[06:46:30] Saved submission ->  
D:\AgentDs\agent_ds_healthcare\submission_ICHI_V1.csv
Analysis — why V2 failed (and why w e’re stuck at ~460)
Adversarial A UC ≈ 0.527 means train/t est featur e distributions ar e basically the same , so
the “covariate shift” story isn’t the bottleneck. Adversarial validation is exactly meant to
detect train–test distribution differences; A UC near 0.5 implies a classifier can’t tell them
apart. 
What your logs do show clearly:
Massiv e train–v al gap  (Train MAE ~230 vs V al MAE ~445) + huge fold v ariance  (Val
MAE ranges ~410 → ~483).
Featur e space is huge  (65k columns) relative to only 2k r ows, and the sparse code
representation encourages “memorizing” idiosyncratic code combos.
So the stagnation is not shif t — it’s high-v ariance r epresentation . The hashed code space
is letting the model learn brittle, sample-specific interactions that reduce train MAE but
don’t generalize.
The piv ot
Instead of “more features / more boosting,” we need to change the pr oblem form  to
reduce variance:
Phase 3 ar chitectur e (structural change)Medium +1

```


### Page 166

```text

Baseline + R esidual (“fr equency/sev erity style” decomposition):
Health cost modeling often improves stability by separating a stable b aseline component
from a conditional / incr emental component  (classic in health economics with two-part or
multi-part cost models, and in insurance with frequency–severity decomposition). 
We’ll implement a competition-friendly analog :
1. Baseline  = group-median expected future cost using only stable drivers
primary_chronic × insurance × prior_cost_decile  → median target
This captures the big predictable component and is very hard to overfit.
2. Residual model  predicts:
using low-dimensional, generalizable r eceipt “phenotype” featur es, not high-
dimensional hashing.
3. New featur e representation for “all codes” without hashing
We replace the 65k hashed space with super vised code-effect embeddings :
In each fold, compute per-code shrunk mean r esidual  on the fold’s training split
For each patient, aggregate those code effects: sum/max/mean/pos_sum/neg_sum
This directly encodes your ED A insight: “certain codes imply systematic positive
residual risk,” but in a few stable featur es instead of 65k brittle ones.
This is the single most cr itical change : reduce v ariance by compr essing the code sp ace
into a super vised low-dimensional embedding and model r esiduals , not raw totals.
The Plan — pr ecise V3 engineering st eps
1. Drop hashed code featur es entir ely.
2. Build a fold-safe b aseline :
bins: prior_ed_cost_5y_usd  deciles (bins defined globally; no leakage)
group median target using only train-fold rows.
3. Train a residual model on receipt phenotype + curated high-lift codes + code-effect
aggr egates:
Code effects computed only fr om fold tr ain split  and applied to fold val split (no
leakage).PMC +2
r= y−baseline(x)

```


### Page 167

```text

Strong regularization, shallow trees.
4. Evaluate MAE on final pr ediction : baseline + residual_pred .
5. Fit final baseline on all train; fit final residual model on all train residuals; predict test;
submit.
V3 code (1-cell, copy/p aste)
python
# ============================================================
# AgentDS Challenge 2 — V3: Baseline + Residual + Supervised Code-Effect Embedding
# Key pivot:
#   - Remove high-dimensional hashed code features (overfit source)
#   - Build stable baseline from (primary_chronic x insurance x prior_cost_decile
#   - Train residual model on low-dim receipt phenotypes + curated high-lift code
#     + fold-safe supervised code-effect aggregate features.
#
# XGBoost 3.0.0 compatible: early_stopping_rounds + eval_metric in constructor.
# ============================================================
import os, time, json, math, warnings
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from joblib import load, dump
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
warnings.filterwarnings( "ignore" )
np.set_printoptions(suppress= True)
# -----------------------------

```


### Page 168

```text

# Paths
# -----------------------------
BASE_DIR = Path( r"D:\AgentDs\agent_ds_healthcare" )
TRAIN_CSV = BASE_DIR / "ed_cost_train.csv"
TEST_CSV  = BASE_DIR / "ed_cost_test.csv"
PATIENTS_CSV = BASE_DIR / "patients.csv"
CACHE_DIR = BASE_DIR / "cache_iter10"
CACHE_DIR.mkdir(parents= True, exist_ok= True)
RECEIPT_CACHE = CACHE_DIR / "receipts_parsed.joblib"
SUBMISSION_PATH = BASE_DIR / "submission_ICHI_V1.csv"
RUN_ID = datetime.now().strftime( "%Y%m%d_%H%M%S" )
RUN_DIR = CACHE_DIR / f"run_v3_ {RUN_ID} "
RUN_DIR.mkdir(parents= True, exist_ok= True)
def log(msg):
    ts = datetime.now().strftime( "%H:%M:%S" )
    print(f"[{ts}] {msg}")
# -----------------------------
# Config
# -----------------------------
SEED = 42
N_SPLITS = 5
# Baseline bins
COST_BINS = 10
# Supervised code effect embedding controls
CODE_MIN_SUPPORT = 15      # minimum patients in fold-train having code to comput
SHRINK_ALPHA = 50.0        # shrinkage toward 0: effect = sum_resid / (count + alp
# Residual model params (conservative / low-variance)
XGB_RESID_PARAMS = dict(
    n_estimators= 5000,
    learning_rate= 0.05,
    max_depth= 4,
    min_child_weight= 50,
    subsample= 0.75,
    colsample_bytree= 0.75,

```


### Page 169

```text

    reg_lambda= 15.0,
    reg_alpha= 0.5,
    objective= "reg:absoluteerror" ,
    eval_metric= "mae",
    early_stopping_rounds= 250,
    tree_method= "hist",
    device= "cuda",     # will fallback to cpu if needed
    n_jobs=- 1,
    verbosity= 1,
)
VERBOSE_EVAL = 200
# -----------------------------
# Evidence-driven code sets (from your EDA)
# -----------------------------
SEVERE_PROC = [ "31500", "92950", "36556", "36620"]
CRIT_CARE   = [ "99291", "99292"]
HI_DIAG     = [ "70450", "74177", "84484"]
OBS         = [ "G0378"]
ED_LEVELS   = [ "99281", "99282", "99283", "99284", "99285"]
CURATED_CODES = SEVERE_PROC + CRIT_CARE + HI_DIAG + OBS + ED_LEVELS
# -----------------------------
# Utilities
# -----------------------------
def normalize_cols (df):
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df
def safe_float (x):
    if x is None:
        return np.nan
    if isinstance (x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().replace( "$","").replace( ",","")
    try:
        return float(s)

```


### Page 170

```text

    except Exception:
        return np.nan
def gini_from_amounts (x):
    x = np.asarray(x, dtype= float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0
    if np.all(x == 0):
        return 0.0
    x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cumx) / (cumx[- 1] + 1e-12)) / n)
def entropy_from_counts (counts: Counter ) -> float:
    if not counts:
        return 0.0
    v = np.array( list(counts.values()), dtype= float)
    p = v / (v. sum() + 1e-12)
    return float(-(p * np.log(p + 1e-12)).sum())
def code_bucket (code: str) -> str:
    if not isinstance (code, str) or not code:
        return "unk"
    if code.isdigit():
        if code.startswith( "99"):
            return "em_99"
        return f"num_{code[0]}"
    return f"alpha_ {code[0].upper()}"
def build_prior_bins (prior, n_bins= 10):
    q = np.quantile(prior.astype( float), np.linspace( 0, 1, n_bins + 1))
    bins = np.unique(q)
    if len(bins) < 3:
        bins = np.unique(np.quantile(prior.astype( float), [0, 0.5, 1.0]))
    return bins
def baseline_group_median (train_df, y, idx_fit, idx_apply, bins ):
    fit = train_df.iloc[idx_fit]

```


### Page 171

```text

    app = train_df.iloc[idx_apply]
    fit_bin = pd.cut(fit[ "prior_ed_cost_5y_usd" ].astype( float), bins=bins, includ
    app_bin = pd.cut(app[ "prior_ed_cost_5y_usd" ].astype( float), bins=bins, includ
    fit_key = (fit[ "primary_chronic" ].astype( str) + "|" + fit["insurance" ].astype
    app_key = (app[ "primary_chronic" ].astype( str) + "|" + app["insurance" ].astype
    med = pd.Series(y[idx_fit]).groupby(fit_key).median()
    med_dict = med.to_dict()
    # fallbacks
    chronic_med = pd.Series(y[idx_fit]).groupby(fit[ "primary_chronic" ].astype( str
    global_med = float(np.median(y[idx_fit]))
    out = np.empty( len(idx_apply), dtype= float)
    for i, k in enumerate (app_key):
        if k in med_dict:
            out[i] = med_dict[k]
        else:
            ch = app.iloc[i][ "primary_chronic" ]
            out[i] = chronic_med.get( str(ch), global_med)
    return out
def extract_codes_and_amounts (obj):
    items = obj.get( "items", []) if isinstance (obj, dict) else []
    codes = []
    amounts = []
    for it in items:
        if not isinstance (it, dict):
            continue
        c = str(it.get( "code","")).strip()
        a = safe_float(it.get( "amount" , np.nan))
        if c:
            codes.append(c)
        amounts.append(a)
    return codes, np.array(amounts, dtype= float)
def receipt_phenotype (obj):
    codes, amts = extract_codes_and_amounts(obj)

```


### Page 172

```text

    total = float(np.nansum(amts))
    if not np.isfinite(total) or total <= 0:
        total = 1e-9
    cc = Counter(codes)
    n_items = int(len(amts))
    n_unique = int(len(cc))
    top_sorted = np.sort(amts[np.isfinite(amts)])[::- 1]
    top1 = float(top_sorted[ 0]) if len(top_sorted) else 0.0
    top3 = float(top_sorted[: 3].sum()) if len(top_sorted) >= 3 else float(top_sor
    top1_share = top1 / total
    top3_share = top3 / total
    distributed_cost = 1.0 - top3_share
    p = top_sorted / (top_sorted. sum() + 1e-12) if len(top_sorted) else np.array(
    hhi = float(np.sum(p * p)) if len(p) else 1.0
    gini = gini_from_amounts(amts)
    bucket_entropy = entropy_from_counts(Counter(code_bucket(c) for c in codes))
    # ED lane
    em9928 = [c for c in codes if c in ED_LEVELS]
    n_em_9928x = len(em9928)
    max_em_level = float(max([int(c[-1]) for c in em9928], default= 0))
    # critical care
    n_99291 = int(cc.get( "99291", 0))
    n_99292 = int(cc.get( "99292", 0))
    has_cc  = int((n_99291 + n_99292) > 0)
    cc_time_proxy = float((74 if n_99291 > 0 else 0) + 30 * n_99292)
    # severe procs share
    amt_by_code = Counter()
    for c, a in zip(codes, amts):
        if np.isfinite(a):
            amt_by_code[c] += float(a)
    amt_severe = float(sum(amt_by_code.get(c, 0.0) for c in SEVERE_PROC))
    severe_share = amt_severe / total
    n_severe = int(sum(cc.get(c, 0) for c in SEVERE_PROC))

```


### Page 173

```text

    # diagnostics
    amt_diag = float(sum(amt_by_code.get(c, 0.0) for c in HI_DIAG))
    diag_share = amt_diag / total
    n_diag = int(sum(cc.get(c, 0) for c in HI_DIAG))
    # severity score
    has_intub = int(cc.get( "31500", 0) > 0)
    has_cpr   = int(cc.get( "92950", 0) > 0)
    has_cvc   = int(cc.get( "36556", 0) > 0)
    has_art   = int(cc.get( "36620", 0) > 0)
    has_cth   = int(cc.get( "70450", 0) > 0)
    has_ctap  = int(cc.get( "74177", 0) > 0)
    has_trop  = int(cc.get( "84484", 0) > 0)
    severity_score = 5*has_intub + 5*has_cpr + 4*has_cvc + 4*has_art + 3*has_cc + 
    # curated per-code has/cnt/share
    out = {
        "n_items" : n_items,
        "n_unique_codes" : n_unique,
        "top1_share" : float(top1_share),
        "top3_share" : float(top3_share),
        "distributed_cost" : float(distributed_cost),
        "hhi": float(hhi),
        "gini": float(gini),
        "bucket_entropy" : float(bucket_entropy),
        "n_em_9928x" : int(n_em_9928x),
        "max_em_level_9928x" : float(max_em_level),
        "has_critical_care" : int(has_cc),
        "n_99291" : int(n_99291),
        "n_99292" : int(n_99292),
        "cc_time_proxy" : float(cc_time_proxy),
        "n_severe_proc" : int(n_severe),
        "severe_proc_share" : float(severe_share),
        "n_hi_diag" : int(n_diag),
        "hi_diag_share" : float(diag_share),
        "severity_score" : float(severity_score),
    }

```


### Page 174

```text

    for c in CURATED_CODES:
        cnt = int(cc.get(c, 0))
        amt = float(amt_by_code.get(c, 0.0))
        out[ f"has_{c}"] = int(cnt > 0)
        out[ f"cnt_{c}"] = cnt
        out[ f"share_ {c}"] = float(amt / total)
    return out, cc  # also return counts for code-effect embedding
def build_code_effects (train_pids, resid_by_pid, codes_by_pid, min_support= 15, al
    # Compute shrunk mean residual per code from fold-train only:
    # effect(code) = sum_resid / (count + alpha)
    sum_resid = defaultdict( float)
    cnt = defaultdict( int)
    for pid in train_pids:
        r = float(resid_by_pid[pid])
        codes = codes_by_pid[pid]
        # use unique codes (presence effect), not counts
        for c in codes:
            cnt[c] += 1
            sum_resid[c] += r
    eff = {}
    for c, n in cnt.items():
        if n >= min_support:
            eff[c] = sum_resid[c] / (n + alpha)
    return eff
def code_effect_aggregates (pid_list, codes_by_pid, eff_dict ):
    # Aggregate code effects into a few stable features
    rows = []
    for pid in pid_list:
        codes = codes_by_pid[pid]
        vals = [eff_dict.get(c, 0.0) for c in codes]
        vals = np.array(vals, dtype= float) if len(vals) else np.array([ 0.0], dtyp
        pos = vals[vals > 0]
        neg = vals[vals < 0]
        rows.append({

```


### Page 175

```text

            "ce_sum" : float(vals.sum()),
            "ce_mean" : float(vals.mean()),
            "ce_max" : float(vals.max()),
            "ce_pos_sum" : float(pos.sum()) if len(pos) else 0.0,
            "ce_neg_sum" : float(neg.sum()) if len(neg) else 0.0,
            "ce_n_codes" : float(len(codes)),
        })
    return pd.DataFrame(rows)
def make_model (seed, use_cuda= True):
    p = dict(XGB_RESID_PARAMS)
    p["random_state" ] = seed
    if not use_cuda:
        p[ "device" ] = "cpu"
    return xgb.XGBRegressor(**p)
def predict_best (model, X ):
    bi = getattr(model, "best_iteration" , None)
    if bi is not None:
        try:
            return model.predict(X, iteration_range=( 0, int(bi) + 1))
        except TypeError:
            pass
    return model.predict(X)
# -----------------------------
# Load data
# -----------------------------
log(f"xgboost version: {getattr(xgb,'__version__' ,'unknown' )}")
train = normalize_cols(pd.read_csv(TRAIN_CSV))
test  = normalize_cols(pd.read_csv(TEST_CSV))
patients = normalize_cols(pd.read_csv(PATIENTS_CSV))
assert RECEIPT_CACHE.exists(), f"Missing {RECEIPT_CACHE} "
receipts = load(RECEIPT_CACHE)
if not isinstance (receipts, dict) or len(receipts) < 3000:
    raise RuntimeError( "receipts_parsed.joblib structure/size looks wrong. Rebuil
# Merge patients
train = train.merge(patients, on= "patient_id" , how="left")

```


### Page 176

```text

test  = test.merge(patients, on= "patient_id" , how="left")
# Precompute receipt features + codes_by_pid (unique code set per patient)
all_df = pd.concat([train.assign(_is_train= 1), test.assign(_is_train= 0)], axis= 0, 
all_ids = all_df[ "patient_id" ].astype( int).tolist()
feat_rows = []
codes_by_pid = {}
for pid in all_ids:
    obj = receipts.get( int(pid), { "items": []})
    feats, cc = receipt_phenotype(obj)
    feat_rows.append(feats)
    codes_by_pid[ int(pid)] = set(cc.keys())
feat_df = pd.DataFrame(feat_rows)
all_df = pd.concat([all_df.reset_index(drop= True), feat_df.reset_index(drop= True)
# Categoricals for residual model
cat_cols = [ "primary_chronic" , "insurance" , "sex", "zip3"]
for c in cat_cols:
    all_df[c] = all_df[c].astype( str)
# Numeric feature list for residual model (exclude raw prior_cost as baseline alr
resid_num_cols = [
    "age",
    "prior_ed_visits_5y" ,
    "n_items" ,"n_unique_codes" ,"top1_share" ,"top3_share" ,"distributed_cost" ,
    "hhi","gini","bucket_entropy" ,
    "n_em_9928x" ,"max_em_level_9928x" ,
    "has_critical_care" ,"n_99291" ,"n_99292" ,"cc_time_proxy" ,
    "n_severe_proc" ,"severe_proc_share" ,
    "n_hi_diag" ,"hi_diag_share" ,
    "severity_score" ,
]
# curated code features
for c in CURATED_CODES:
    resid_num_cols += [ f"has_{c}", f"cnt_{c}", f"share_ {c}"]
# Fill numeric
for c in resid_num_cols:

```


### Page 177

```text

    all_df[c] = pd.to_numeric(all_df[c], errors= "coerce" ).fillna( 0.0).astype(np.f
# Fit one-hot encoder on all (no leakage; no y used)
try:
    ohe = OneHotEncoder(handle_unknown= "ignore" , sparse_output= False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown= "ignore" , sparse= False)
X_cat_all = ohe.fit_transform(all_df[cat_cols])
# Split back
train2 = all_df[all_df[ "_is_train" ] == 1].copy().reset_index(drop= True)
test2  = all_df[all_df[ "_is_train" ] == 0].copy().reset_index(drop= True)
X_cat_train = X_cat_all[: len(train2)]
X_cat_test  = X_cat_all[ len(train2):]
y = train2[ "ed_cost_next3y_usd" ].astype( float).values
# Global cost bins (no leakage)
bins = build_prior_bins(train2[ "prior_ed_cost_5y_usd" ].values, n_bins=COST_BINS)
# Stratify for CV (stable)
strat = (train2[ "primary_chronic" ].astype( str) + "|" +
         pd.qcut(train2[ "prior_ed_cost_5y_usd" ].astype( float), q=10, duplicates= "
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle= True, random_state=SEED)
# OOF containers
oof_pred = np.zeros( len(train2), dtype= float)
oof_base = np.zeros( len(train2), dtype= float)
test_pred_accum = np.zeros( len(test2), dtype= float)
test_base_full = None
fold_logs = []
log(f"Starting V3 CV: baseline + residual | n= {len(train2)}")
for fold, (tr_idx, va_idx) in enumerate (skf.split(np.zeros( len(train2)), strat), 
    t0 = time.time()
    # fold baseline (no leakage): compute medians using fold-train only

```


### Page 178

```text

    base_tr = baseline_group_median(train2, y, tr_idx, tr_idx, bins=bins)
    base_va = baseline_group_median(train2, y, tr_idx, va_idx, bins=bins)
    resid_tr = y[tr_idx] - base_tr
    # Build code-effect dictionary from fold-train residuals
    train_pids = train2.loc[tr_idx, "patient_id" ].astype( int).tolist()
    resid_by_pid = { int(pid): float(r) for pid, r in zip(train_pids, resid_tr)}
    eff = build_code_effects(train_pids, resid_by_pid, codes_by_pid,
                             min_support=CODE_MIN_SUPPORT, alpha=SHRINK_ALPHA)
    # Code-effect aggregate features (fold-safe)
    tr_pids = train2.loc[tr_idx, "patient_id" ].astype( int).tolist()
    va_pids = train2.loc[va_idx, "patient_id" ].astype( int).tolist()
    ce_tr = code_effect_aggregates(tr_pids, codes_by_pid, eff).values.astype(np.f
    ce_va = code_effect_aggregates(va_pids, codes_by_pid, eff).values.astype(np.f
    # Residual model features
    X_num_tr = train2.loc[tr_idx, resid_num_cols].values.astype(np.float32)
    X_num_va = train2.loc[va_idx, resid_num_cols].values.astype(np.float32)
    X_tr = np.hstack([X_num_tr, X_cat_train[tr_idx], ce_tr]).astype(np.float32)
    X_va = np.hstack([X_num_va, X_cat_train[va_idx], ce_va]).astype(np.float32)
    # Train residual model
    model = make_model(SEED + fold, use_cuda= True)
    try:
        model.fit(X_tr, resid_tr,
                  eval_set=[(X_tr, resid_tr), (X_va, y[va_idx] - base_va)],
                  verbose=VERBOSE_EVAL)
        mode = "cuda"
    except xgb.core.XGBoostError as e:
        log( f"[Fold {fold}] GPU failed -> CPU fallback. Error: {repr(e)[:180]}")
        model = make_model(SEED + fold, use_cuda= False)
        model.fit(X_tr, resid_tr,
                  eval_set=[(X_tr, resid_tr), (X_va, y[va_idx] - base_va)],
                  verbose=VERBOSE_EVAL)
        mode = "cpu"

```


### Page 179

```text

    pred_resid_va = predict_best(model, X_va)
    pred_y_va = np.clip(base_va + pred_resid_va, 0.0, None)
    # Save OOF
    oof_pred[va_idx] = pred_y_va
    oof_base[va_idx] = base_va
    # fold metrics
    mae_base = mean_absolute_error(y[va_idx], base_va)
    mae_full = mean_absolute_error(y[va_idx], pred_y_va)
    info = {
        "fold": fold,
        "mode": mode,
        "mae_baseline" : float(mae_base),
        "mae_v3" : float(mae_full),
        "improve" : float(mae_base - mae_full),
        "best_iteration" : int(getattr(model, "best_iteration" , -1)) if getattr(mo
        "time_sec" : float(time.time() - t0),
        "n_codes_effect" : int(len(eff)),
    }
    fold_logs.append(info)
    log( f"[Fold {fold}/{N_SPLITS} ] mode={mode} | baseline_MAE= {mae_base: .2f} | v3_
        f"improve= {mae_base-mae_full: .2f} | best_iter= {info['best_iteration' ]} | 
        f"code_effects= {info['n_codes_effect' ]} | time= {info['time_sec' ]:.1f}s")
# OOF results
oof_mae_base = mean_absolute_error(y, oof_base)
oof_mae_v3 = mean_absolute_error(y, oof_pred)
log(f"OOF MAE baseline-only: {oof_mae_base: .6f}")
log(f"OOF MAE V3 (baseline+residual): {oof_mae_v3: .6f}  (delta {oof_mae_base - oo
# -----------------------------
# Fit final model on full training and predict test
# -----------------------------
log("Fitting final baseline on full train..." )
idx_all = np.arange( len(train2))
base_train_full = baseline_group_median(train2, y, idx_all, idx_all, bins=bins)
base_test_full = baseline_group_median(pd.concat([train2, test2], axis= 0, ignore_
                                       np.concatenate([y, np.zeros( len(test2))]),
                                       np.arange( len(train2)),  # fit idx

```


### Page 180

```text

                                       np.arange( len(train2), len(train2)+ len(test
                                       bins=bins)
resid_full = y - base_train_full
# code effects on full train residual
train_pids_full = train2[ "patient_id" ].astype( int).tolist()
resid_by_pid_full = { int(pid): float(r) for pid, r in zip(train_pids_full, resid_
eff_full = build_code_effects(train_pids_full, resid_by_pid_full, codes_by_pid,
                              min_support=CODE_MIN_SUPPORT, alpha=SHRINK_ALPHA)
ce_train_full = code_effect_aggregates(train_pids_full, codes_by_pid, eff_full).v
test_pids_full = test2[ "patient_id" ].astype( int).tolist()
ce_test_full = code_effect_aggregates(test_pids_full, codes_by_pid, eff_full).val
X_num_train_full = train2[resid_num_cols].values.astype(np.float32)
X_num_test_full  = test2[resid_num_cols].values.astype(np.float32)
X_train_full = np.hstack([X_num_train_full, X_cat_train, ce_train_full]).astype(n
X_test_full  = np.hstack([X_num_test_full,  X_cat_test,  ce_test_full]).astype(np
# Choose n_estimators for final fit: use median best_iter from CV if available
best_iters = [f[ "best_iteration" ] for f in fold_logs if f["best_iteration" ] is not
final_n_estimators = int(np.median(best_iters)) if best_iters else int(XGB_RESID_
final_n_estimators = max(200, min(final_n_estimators + 200, XGB_RESID_PARAMS[ "n_e
log(f"Final fit n_estimators = {final_n_estimators}  (median_best_iter + 200)")
final_params = dict(XGB_RESID_PARAMS)
final_params[ "n_estimators" ] = final_n_estimators
final_params[ "early_stopping_rounds" ] = None  # final fit no early stop
final_model = xgb.XGBRegressor(**{**final_params, "random_state" : SEED + 999})
try:
    final_model.fit(X_train_full, resid_full, verbose= False)
    mode_final = "cuda"
except xgb.core.XGBoostError as e:
    log( f"Final GPU failed -> CPU fallback. Error: {repr(e)[:180]}")
    final_params[ "device" ] = "cpu"
    final_model = xgb.XGBRegressor(**{**final_params, "random_state" : SEED + 999}
    final_model.fit(X_train_full, resid_full, verbose= False)

```


### Page 181

```text

What t o send b ack af ter running V3    mode_final = "cpu"
pred_resid_test = final_model.predict(X_test_full)
pred_test = np.clip(base_test_full + pred_resid_test, 0.0, None)
sub = pd.DataFrame({ "patient_id" : test2[ "patient_id" ].astype( int).values,
                    "ed_cost_next3y_usd" : pred_test})
sub.to_csv(SUBMISSION_PATH, index= False)
sub.to_csv(RUN_DIR / "submission.csv" , index= False)
# Save OOF + logs
oof_out = pd.DataFrame({
    "patient_id" : train2[ "patient_id" ].astype( int),
    "y_true" : y,
    "oof_baseline" : oof_base,
    "oof_pred" : oof_pred
})
oof_out.to_csv(RUN_DIR / "oof_predictions.csv" , index= False)
run_log = {
    "run_id" : RUN_ID,
    "xgboost_version" : getattr(xgb, "__version__" , "unknown" ),
    "mode_final" : mode_final,
    "oof_mae_baseline" : float(oof_mae_base),
    "oof_mae_v3" : float(oof_mae_v3),
    "folds": fold_logs,
    "code_effect_min_support" : CODE_MIN_SUPPORT,
    "code_effect_shrink_alpha" : SHRINK_ALPHA,
    "final_n_estimators" : final_n_estimators,
}
with open(RUN_DIR / "run_log.json" , "w", encoding= "utf-8") as f:
    json.dump(run_log, f, indent= 2)
log(f"Saved run artifacts: {RUN_DIR} ")
log(f"Saved submission -> {SUBMISSION_PATH} ")

```


### Page 182

```text

1. OOF MAE baseline-only  and OOF MAE V3  (and the delta)
2. Fold logs (baseline_MAE vs v3_MAE + best_iter)
3. The new LB score
If V3 is doing what it’s designed to do, you should see:
baseline-only MAE is “okay”
V3 improves baseline consistently across folds
variance across folds narrows (less 409 ↔ 483 chaos)
LB should move meaningfully toward ~440–420 territory; then we can iterate the last
~20–40 points.
If V3 still plateaus, the next escalation is true fr equency–sev erity modeling  (a two-part /
Tweedie-style claim framework) — the standard approach for nonnegative cost outcomes
in healthcare/insurance. Scikit-learn +1
Sour ces
Sour ces
[14:54:21] xgboost version: 3.0.0
[14:54:21] S tarting V3 CV: baseline + residual | n=2000
[0] validation_0-mae:483.80710 validation_1-
mae:542.05221
[200] validation_0-mae:407.12944 validation_1-
mae:533.40803
[347] validation_0-mae:383.98902 validation_1-
mae:536.64686
[14:54:24] [Fold 1/5] mode=cuda | baseline_MAE=542.87 |  
v3_MAE=529.15 | improve=13.72 | best_iter=98 |  
code_effects=18 | time=2.5s
[0] validation_0-mae:485.09723 validation_1-
mae:524.23722
[200] validation_0-mae:406.61182 validation_1-
mae:515.21914
[277] validation_0-mae:393.88378 validation_1-
mae:516.37441
[14:54:25] [Fold 2/5] mode=cuda | baseline_MAE=525.42 |  
v3_MAE=511.84 | improve=13.58 | best_iter=28 |  
code_effects=18 | time=0.9s
[0] validation_0-mae:488.53184 validation_1-

```


### Page 183

```text

mae:508.19811
[200] validation_0-mae:406.09336 validation_1-
mae:503.97365
[294] validation_0-mae:390.67594 validation_1-
mae:505.94988
[14:54:26] [Fold 3/5] mode=cuda | baseline_MAE=509.11 |  
v3_MAE=499.59 | improve=9.52 | best_iter=45 |  
code_effects=18 | time=0.9s
[0] validation_0-mae:485.81878 validation_1-
mae:550.67443
[200] validation_0-mae:404.42422 validation_1-
mae:539.12232
[341] validation_0-mae:383.61780 validation_1-
mae:542.71394
[14:54:27] [Fold 4/5] mode=cuda | baseline_MAE=552.22 |  
v3_MAE=534.94 | improve=17.28 | best_iter=91 |  
code_effects=18 | time=1.0s
[0] validation_0-mae:478.27440 validation_1-
mae:549.30858
[200] validation_0-mae:398.56244 validation_1-
mae:525.69052
[377] validation_0-mae:373.63341 validation_1-
mae:526.59098
[14:54:28] [Fold 5/5] mode=cuda | baseline_MAE=551.50 |  
v3_MAE=524.53 | improve=26.98 | best_iter=128 |  
code_effects=18 | time=1.1s
[14:54:28] OOF MAE baseline-only: 536.227467
[14:54:28] OOF MAE V3 (baseline+residual): 520.010242   
(delta +16.217226)
[14:54:28] Fitting final baseline on full train...
[14:54:28] Final fit n_estimators = 291 (median_best_iter +  
200)
[14:54:28] Saved run artifacts:  
D:\AgentDs\agent_ds_healthcare\cache_iter10\run_v3_202
60214_145421
[14:54:28] Saved submission ->  
D:\AgentDs\agent_ds_healthcare\submission_ICHI_V1.csv
🚀 Submitting predictions...
✅ Prediction submitted successfully!

```


### Page 184

```text

📊 Score: 546.2783 (MAE)
✅ Validation passed
✅ Submission successful!
   📊 Score: 546.2783
   📏 Metric: MAE
   ✔  Validation: P assed
🎯 Next steps:
   1. Try incorporating relevant information outside this  
table!
   2. Move on to Healthcare Challenge 3!
怎么train 和 real 都是 500 多了 ??
你这个  V3 变成  500+ （ OOF 520 / LB 546 ）不是 “ 数据突然变难了 ”，⽽是  我们把模型的主
信号砍掉了 ，导致系统性退化。
为什么会  500+ （根因）
1. V3 的  baseline 设计太弱
你⽤的是  primary_chronic × insurance × prior_cost_decile  的分组中位数当
baseline。它丢掉了最强的连续信号（ prior_ed_cost_5y_usd  的连续关系、
prior_ed_visits_5y 、以及  receipt composition 的细粒度） ， baseline 本⾝就  536
MAE，已经⽐你  V1/V2 的  443–452 差⼀⼤截。
2. V3 的 “code effect embedding” ⼏乎没学到东西
每折只⽣成  18 个  code_effects ，等于残差模型只能靠很少的离散效果做修正，修不回
baseline 的巨⼤误差  → 结果⾃然整体  500+ 。
3. V2 失败不是 “shif t” 问题
你的 adversarial A UC=0.527 （ ≈ 随机），说明  train/test 没有明显  covariate shif t，所以
加权不该有收益。我们真正的问题是： 模型缺少关键结构性信息  + ⾼维表⽰导致⾼⽅
差（Train MAE 很低、 Val MAE 卡在  440–480 之间，且折间波动⼤） 。
Phase 3 真正的战略转向（能解释为什么  SOTA ~400 ）结论： V3 的失败是 “ 架构砍主信号  + 残差修复能⼒不⾜ ” ，不是平台或数据异常。

```


### Page 185

```text

你之前只⽤了  ed_cost_*.csv + patients.csv + receipt 。但 description.md 明确说明 还
有能⽤来提取患者严重程度 / 共病 / ⽣理不稳定性的⽂件：
admissions_* + discharge_notes.json + stays_* + vitals_timeseries.json （都可⽤
patient_id  join）。这类 “ 跨表病程 / 住院严重度 / ⽣命体征稳定性 ” 特征，往往是医疗成本预
测⾥最能拉开差距的结构信号；⽽ “ 两段式 / 频率 - 严重度 ” 等成本建模思路也正是健康经济学
⾥常⻅路线。
单⼀最关键改动：把  Challenge 1/3 的信息做成  patient -level 聚合特征，加⼊  ED cost 回
归。
这就是你现在离  ~400 还差  60+ 的 “ 结构性组件 ” 。
V3 ⼯程计划（精确步骤）
1. 构建 patient -level 辅助特征 （不泄漏  ED ⽬标）
admissions （ train+test ）：次数、 LOS 统计、 Charlson band 统计、 acuity_emergent ⽐
例、ed_visits_6m 均值、 primary_dx 分布
discharge_notes ：按  patient 拼接⽂本  → TF-IDF → SVD(32 维 )
stays（ train+test ）：次数、 unit_type 分布、 admission_reason 分布
vitals_timeseries ：按  stay 汇总  HR/SBP/DBP/T emp/RR 的  mean/std/min/max/slope →
再按 patient 聚合（ mean/max ）
vitals notes ：按  patient 拼接  → TF-IDF → SVD(16 维 )
2. receipt 特征继续⽤你  EDA 证明有效的：⾼  lift code + 复杂度 / 分散度（不再⽤  65k
hashing；改成 “Top-K code count/share” 低维表⽰）
3. 回到 MAE 对⻬⽬标： XGBoost reg:absoluteerror （这是最贴合  metric 的⽬标之⼀）
并保留你在  V1 证实有效的  分组残差校准 （primary_chronic × prior_cost_decile ）。
V3 新实现（ 1-cell ，可直接跑）PMC +1
XGBoost Docum …
说明：这版会⾃动检查  admissions/stays/vitals/notes ⽂件是否存在；存在就⽤，不存在
就跳过（鲁棒） 。
缓存：会把 “patient-level 辅助特征 ” 缓存成  joblib ，⼆次运⾏很快。
python

```


### Page 186

```text

# ============================================================
# EDCOST V3 — Multisource Patient-Level Features (Admissions + Notes + Stays + Vi
# Metric-aligned: XGBoost reg:absoluteerror (MAE) + group-median residual calibra
# XGBoost 3.0.0 sklearn API compatible: eval_metric + early_stopping_rounds in con
# ============================================================
import os, json, time, math, warnings
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import load, dump
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import xgboost as xgb
warnings.filterwarnings( "ignore" )
np.set_printoptions(suppress= True)
# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path( r"D:\AgentDs\agent_ds_healthcare" )
CACHE_DIR = BASE_DIR / "cache_iter10"
CACHE_DIR.mkdir(parents= True, exist_ok= True)
TRAIN_CSV = BASE_DIR / "ed_cost_train.csv"
TEST_CSV  = BASE_DIR / "ed_cost_test.csv"
PATIENTS_CSV = BASE_DIR / "patients.csv"
PDF_DIR = BASE_DIR / "receipts_pdf"                  # not used (we use receipts_p
RECEIPT_CACHE = CACHE_DIR / "receipts_parsed.joblib"
SUBMISSION_PATH = BASE_DIR / "submission_ICHI_V1.csv"

```


### Page 187

```text

# extra files (from description.md)
ADM_TR = BASE_DIR / "admissions_train.csv"
ADM_TE = BASE_DIR / "admissions_test.csv"
DISCH_NOTES = BASE_DIR / "discharge_notes.json"
STAY_TR = BASE_DIR / "stays_train.csv"
STAY_TE = BASE_DIR / "stays_test.csv"
VITALS_JSON = BASE_DIR / "vitals_timeseries.json"
RUN_ID = datetime.now().strftime( "%Y%m%d_%H%M%S" )
RUN_DIR = CACHE_DIR / f"run_v3_multi_ {RUN_ID} "
RUN_DIR.mkdir(parents= True, exist_ok= True)
PIPELINE_VERSION = "EDCOST_V3_MULTISOURCE_2026-02-14"
AUX_CACHE = CACHE_DIR / "patient_aux_features_v3.joblib"
FEAT_CACHE = CACHE_DIR / "feature_matrix_v3.joblib"
def log(msg):
    ts = datetime.now().strftime( "%H:%M:%S" )
    print(f"[{ts}] {msg}")
# -----------------------------
# Config
# -----------------------------
SEED = 42
N_SPLITS = 5
# receipt code top-K
TOPK_CODES = 40
CODE_MIN_PATIENT_SUPPORT = 30  # include codes appearing in >= this many patients 
# Text dims
DISCH_SVD_DIM = 32
VITALSNOTE_SVD_DIM = 16
# XGB params (conservative but strong)
XGB_PARAMS = dict(
    n_estimators= 8000,
    learning_rate= 0.03,
    max_depth= 6,

```


### Page 188

```text

    min_child_weight= 20,
    subsample= 0.8,
    colsample_bytree= 0.8,
    reg_lambda= 10.0,
    reg_alpha= 0.2,
    objective= "reg:absoluteerror" ,
    eval_metric= "mae",
    early_stopping_rounds= 300,      # ctor-based for xgb 3.x
    tree_method= "hist",
    device= "cuda",
    n_jobs=- 1,
    verbosity= 1,
)
VERBOSE_EVAL = 200
CALIB_BINS = 10
# -----------------------------
# Helpers
# -----------------------------
def normalize_cols (df):
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df
def safe_float (x):
    if x is None:
        return np.nan
    if isinstance (x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().replace( "$","").replace( ",","")
    try:
        return float(s)
    except Exception:
        return np.nan
def build_prior_bins (train_prior: pd.Series, n_bins: int = 10):
    q = np.quantile(train_prior.values.astype( float), np.linspace( 0, 1, n_bins + 
    bins = np.unique(q)
    if len(bins) < 3:

```


### Page 189

```text

        bins = np.unique(np.quantile(train_prior.values.astype( float), [0, 0.5, 1
    return bins
def predict_best (model, X ):
    bi = getattr(model, "best_iteration" , None)
    if bi is not None:
        try:
            return model.predict(X, iteration_range=( 0, int(bi) + 1))
        except TypeError:
            pass
    return model.predict(X)
def group_median_calibrate (train_df, oof_pred, y_true, test_df, test_pred, n_bins
    bins = build_prior_bins(train_df[ "prior_ed_cost_5y_usd" ].astype( float), n_bin
    tr_bin = pd.cut(train_df[ "prior_ed_cost_5y_usd" ].astype( float), bins=bins, in
    te_bin = pd.cut(test_df[ "prior_ed_cost_5y_usd" ].astype( float), bins=bins, inc
    tr_key = (train_df[ "primary_chronic" ].astype( str) + "|" + tr_bin).values
    te_key = (test_df[ "primary_chronic" ].astype( str) + "|" + te_bin).values
    resid = y_true - oof_pred
    shift = pd.Series(resid).groupby(tr_key).median().to_dict()
    oof_adj = oof_pred + np.array([shift.get(k, 0.0) for k in tr_key], dtype= float
    te_adj  = test_pred + np.array([shift.get(k, 0.0) for k in te_key], dtype= flo
    return oof_adj, te_adj, shift
def tfidf_svd_patient (text_by_pid: pd.Series, n_components: int, prefix: str):
    """
    Fit TF-IDF + SVD on all patients' text (train+test). Return dataframe with SV
    """
    texts = text_by_pid.fillna( "").astype( str).values
    vec = TfidfVectorizer(
        min_df= 2,
        max_features= 20000,
        ngram_range=( 1,2),
        token_pattern= r"(?u)\b\w+\b"
    )
    X = vec.fit_transform(texts)
    k = min(n_components, max(2, X.shape[ 1]-1)) if X.shape[ 1] > 2 else 2
    svd = TruncatedSVD(n_components=k, random_state=SEED)
    Z = svd.fit_transform(X)

```


### Page 190

```text

    cols = [ f"{prefix} _svd_{i:02d}" for i in range(Z.shape[ 1])]
    return pd.DataFrame(Z, columns=cols), { "tfidf_vocab" : len(vec.vocabulary_), "
# -----------------------------
# Receipt features (from receipts_parsed.joblib)
# -----------------------------
SEVERE_PROC = [ "31500", "92950", "36556", "36620"]
CRIT_CARE   = [ "99291", "99292"]
HI_DIAG     = [ "70450", "74177", "84484"]
OBS         = [ "G0378"]
ED_LEVELS   = [ "99281", "99282", "99283", "99284", "99285"]
CURATED_CODES = SEVERE_PROC + CRIT_CARE + HI_DIAG + OBS + ED_LEVELS
def receipt_patient_features (receipts_dict, patient_ids ):
    # pass 1: gather per-patient code support
    support = Counter()
    per_patient = {}
    for pid in patient_ids:
        obj = receipts_dict.get( int(pid), { "items": []})
        items = obj.get( "items", [])
        codes = []
        amt_by_code = Counter()
        amts = []
        for it in items:
            if not isinstance (it, dict):
                continue
            c = str(it.get( "code","")).strip()
            a = safe_float(it.get( "amount" , np.nan))
            if c:
                codes.append(c)
            if np.isfinite(a):
                amt_by_code[c] += float(a)
                amts.append( float(a))
        u = set(codes)
        for c in u:
            support[c] += 1
        per_patient[ int(pid)] = (Counter(codes), amt_by_code, np.array(amts, dtyp
    # select code vocab: top-K by support, but also ensure min support
    code_list = [c for c, n in support.most_common() if n >= CODE_MIN_PATIENT_SUP

```


### Page 191

```text

    code_list = code_list[:TOPK_CODES]
    # always include curated high-lift codes even if rare
    for c in CURATED_CODES:
        if c not in code_list:
            code_list.append(c)
    rows = []
    for pid in patient_ids:
        cc, amt_by_code, amts = per_patient[ int(pid)]
        total = float(np.nansum(amts))
        if not np.isfinite(total) or total <= 0:
            total = 1e-9
        # concentration / complexity
        top_sorted = np.sort(amts[np.isfinite(amts)])[::- 1]
        top1 = float(top_sorted[ 0]) if len(top_sorted) else 0.0
        top3 = float(top_sorted[: 3].sum()) if len(top_sorted) >= 3 else float(top_
        top1_share = top1 / total
        top3_share = top3 / total
        distributed_cost = 1.0 - top3_share
        # E/M + CC + severe
        n_em = sum(cc.get(c, 0) for c in ED_LEVELS)
        n_99291 = int(cc.get( "99291", 0))
        n_99292 = int(cc.get( "99292", 0))
        has_cc  = int((n_99291 + n_99292) > 0)
        cc_time_proxy = float((74 if n_99291> 0 else 0) + 30*n_99292)
        n_severe = int(sum(cc.get(c, 0) for c in SEVERE_PROC))
        severe_amt = float(sum(amt_by_code.get(c, 0.0) for c in SEVERE_PROC))
        severe_share = severe_amt / total
        n_diag = int(sum(cc.get(c, 0) for c in HI_DIAG))
        diag_amt = float(sum(amt_by_code.get(c, 0.0) for c in HI_DIAG))
        diag_share = diag_amt / total
        severity_score = (
            5*int(cc.get( "31500",0)>0) + 5*int(cc.get( "92950",0)>0) +
            4*int(cc.get( "36556",0)>0) + 4*int(cc.get( "36620",0)>0) +
            3*has_cc + int(cc.get( "70450",0)>0) + int(cc.get( "74177",0)>0) + int(

```


### Page 192

```text

        )
        r = dict(
            patient_id= int(pid),
            rcpt_n_items= int(len(amts)),
            rcpt_n_unique_codes= int(len(cc)),
            rcpt_top1_share= float(top1_share),
            rcpt_top3_share= float(top3_share),
            rcpt_distributed_cost= float(distributed_cost),
            rcpt_n_em_9928x= int(n_em),
            rcpt_has_cc= int(has_cc),
            rcpt_cc_time_proxy= float(cc_time_proxy),
            rcpt_n_severe_proc= int(n_severe),
            rcpt_severe_share= float(severe_share),
            rcpt_n_hi_diag= int(n_diag),
            rcpt_hi_diag_share= float(diag_share),
            rcpt_severity_score= float(severity_score),
        )
        # top code counts + shares
        for c in code_list:
            r[ f"code_cnt_ {c}"] = float(cc.get(c, 0))
            r[ f"code_share_ {c}"] = float(amt_by_code.get(c, 0.0) / total)
        rows.append(r)
    return pd.DataFrame(rows), code_list
# -----------------------------
# Admissions + discharge notes features
# -----------------------------
def build_admissions_features ():
    if not (ADM_TR.exists() and ADM_TE.exists()):
        log( "[AUX] admissions files missing, skipping admissions features." )
        return None
    adm_tr = normalize_cols(pd.read_csv(ADM_TR))
    adm_te = normalize_cols(pd.read_csv(ADM_TE))
    adm = pd.concat([adm_tr.assign(_split= "train"), adm_te.assign(_split= "test")]

```


### Page 193

```text

    # numeric aggregates
    agg = adm.groupby( "patient_id" ).agg(
        adm_n=( "admission_id" ,"count"),
        adm_los_mean=( "los_days" ,"mean"),
        adm_los_max=( "los_days" ,"max"),
        adm_los_std=( "los_days" ,"std"),
        adm_aci_mean=( "acuity_emergent" ,"mean"),
        adm_charlson_mean=( "charlson_band" ,"mean"),
        adm_charlson_max=( "charlson_band" ,"max"),
        adm_edvis6m_mean=( "ed_visits_6m" ,"mean"),
    ).reset_index()
    agg[ "adm_los_std" ] = agg[ "adm_los_std" ].fillna( 0.0)
    # categorical counts
    for col, prefix in [("primary_dx" ,"adm_dx" ), ("discharge_weekday" ,"adm_dow" )]
        ctab = pd.crosstab(adm[ "patient_id" ], adm[col]).reset_index()
        ctab.columns = [ "patient_id" ] + [f"{prefix} _{c}" for c in ctab.columns[ 1:
        agg = agg.merge(ctab, on= "patient_id" , how="left")
    # discharge notes -> patient text
    note_feat = None
    note_meta = None
    if DISCH_NOTES.exists():
        with open(DISCH_NOTES, "r", encoding= "utf-8") as f:
            notes = json.load(f)
        df_notes = pd.DataFrame(notes)
        df_notes = normalize_cols(df_notes)
        # join to admissions to get patient_id
        adm_note = adm.merge(df_notes, on= "admission_id" , how="left")
        pat_text = adm_note.groupby( "patient_id" )["note"].apply( lambda s: " ".joi
        pat_text = pat_text.set_index( "patient_id" )["note"]
        Z, meta = tfidf_svd_patient(pat_text.reindex(adm[ "patient_id" ].unique()).
        # align rows to patient ids in same order
        pid_order = pd.Index(adm[ "patient_id" ].unique())
        note_feat = pd.DataFrame({ "patient_id" : pid_order.values})
        note_feat = pd.concat([note_feat.reset_index(drop= True), Z.reset_index(dr
        note_meta = meta
    else:
        log( "[AUX] discharge_notes.json missing, skipping discharge note text feat

```


### Page 194

```text

    return agg, note_feat, note_meta
# -----------------------------
# Stays + vitals features
# -----------------------------
def build_stays_vitals_features ():
    if not (STAY_TR.exists() and STAY_TE.exists()):
        log( "[AUX] stays files missing, skipping stays/vitals features." )
        return None
    st_tr = normalize_cols(pd.read_csv(STAY_TR))
    st_te = normalize_cols(pd.read_csv(STAY_TE))
    st = pd.concat([st_tr.assign(_split= "train"), st_te.assign(_split= "test")], ax
    # stays aggregates
    agg = st.groupby( "patient_id" ).agg(
        stay_n=( "stay_id" ,"count"),
    ).reset_index()
    for col, prefix in [("unit_type" ,"stay_unit" ), ("admission_reason" ,"stay_reas
        ctab = pd.crosstab(st[ "patient_id" ], st[col]).reset_index()
        ctab.columns = [ "patient_id" ] + [f"{prefix} _{c}" for c in ctab.columns[ 1:
        agg = agg.merge(ctab, on= "patient_id" , how="left")
    # vitals per stay -> then patient agg
    vit_agg = None
    vit_text_feat = None
    vit_meta = None
    if VITALS_JSON.exists():
        with open(VITALS_JSON, "r", encoding= "utf-8") as f:
            vitals = json.load(f)
        rows = []
        note_rows = []
        for obj in vitals:
            sid = int(obj["stay_id" ])
            days = obj.get( "days", [])
            df = pd.DataFrame(days)
            # numeric summaries per stay
            feat = { "stay_id" : sid}
            for col in ["hr","sbp","dbp","temp_c" ,"rr"]:

```


### Page 195

```text

                if col not in df.columns:
                    feat[ f"{col}_mean"] = np.nan
                    feat[ f"{col}_std"] = np.nan
                    feat[ f"{col}_min"] = np.nan
                    feat[ f"{col}_max"] = np.nan
                    feat[ f"{col}_slope"] = np.nan
                    continue
                x = pd.to_numeric(df[col], errors= "coerce" )
                feat[ f"{col}_mean"] = float(x.mean())
                feat[ f"{col}_std"]  = float(x.std(ddof= 0) if np.isfinite(x.std(dd
                feat[ f"{col}_min"]  = float(x.min())
                feat[ f"{col}_max"]  = float(x.max())
                # slope day10 - day1 (if day field exists)
                if "day" in df.columns:
                    dmin = df.loc[df[ "day"]==1, col]
                    dmax = df.loc[df[ "day"]==10, col]
                    feat[ f"{col}_slope"] = float(pd.to_numeric(dmax, errors= "coer
                else:
                    feat[ f"{col}_slope"] = 0.0
            rows.append(feat)
            # text per stay
            if "note" in df.columns:
                txt = " ".join(df[ "note"].fillna( "").astype( str).tolist())
            else:
                txt = ""
            note_rows.append({ "stay_id" : sid, "vitals_note" : txt})
        vit_stay = pd.DataFrame(rows)
        vit_note = pd.DataFrame(note_rows)
        # join stay -> patient via st
        vit_stay = vit_stay.merge(st[[ "stay_id" ,"patient_id" ]], on="stay_id" , how
        vit_note = vit_note.merge(st[[ "stay_id" ,"patient_id" ]], on="stay_id" , how
        # patient aggregation of vitals numerics (mean + max)
        agg_mean = vit_stay.groupby( "patient_id" ).mean(numeric_only= True).add_pre
        agg_max  = vit_stay.groupby( "patient_id" ).max(numeric_only= True).add_pref
        vit_agg = agg_mean.merge(agg_max, on= "patient_id" , how="left")

```


### Page 196

```text

        # patient text aggregation of vitals notes -> TFIDF+SVD
        pat_text = vit_note.groupby( "patient_id" )["vitals_note" ].apply( lambda s: 
        pat_text = pat_text.set_index( "patient_id" )["vitals_note" ]
        # fit on all patients that appear in stays
        pid_order = pd.Index(st[ "patient_id" ].unique())
        Z, meta = tfidf_svd_patient(pat_text.reindex(pid_order).fillna( ""), VITAL
        vit_text_feat = pd.DataFrame({ "patient_id" : pid_order.values})
        vit_text_feat = pd.concat([vit_text_feat.reset_index(drop= True), Z.reset_
        vit_meta = meta
    else:
        log( "[AUX] vitals_timeseries.json missing, skipping vitals features." )
    return agg, vit_agg, vit_text_feat, vit_meta
# ============================================================
# MAIN
# ============================================================
log(f"xgboost version: {getattr(xgb,'__version__' ,'unknown' )}")
log(f"PIPELINE_VERSION: {PIPELINE_VERSION} ")
train = normalize_cols(pd.read_csv(TRAIN_CSV))
test  = normalize_cols(pd.read_csv(TEST_CSV))
patients = normalize_cols(pd.read_csv(PATIENTS_CSV))
assert RECEIPT_CACHE.exists(), f"Missing receipts cache: {RECEIPT_CACHE} "
receipts = load(RECEIPT_CACHE)
if not isinstance (receipts, dict) or len(receipts) < 3000:
    raise RuntimeError( "receipts_parsed.joblib structure/size looks wrong. Rebuil
# concat train/test
train["_is_train" ] = 1
test["_is_train" ] = 0
all_df = pd.concat([train, test], axis= 0, ignore_index= True)
all_df = all_df.merge(patients, on= "patient_id" , how="left")
# ---- build / load AUX patient features cache
if AUX_CACHE.exists():
    aux = load(AUX_CACHE)
    if aux.get( "version" ) == PIPELINE_VERSION:

```


### Page 197

```text

        log( f"Loaded AUX cache: {AUX_CACHE} ")
        adm_agg = aux.get( "adm_agg" )
        disch_svd = aux.get( "disch_svd" )
        stay_agg = aux.get( "stay_agg" )
        vit_num = aux.get( "vit_num" )
        vit_svd = aux.get( "vit_svd" )
    else:
        aux = None
else:
    aux = None
if aux is None:
    log( "Building AUX patient-level features (admissions/stays/vitals/notes)..." )
    adm_agg, disch_svd, disch_meta = build_admissions_features() if ADM_TR.exists
    stay_agg, vit_num, vit_svd, vit_meta = build_stays_vitals_features() if STAY_
    dump({
        "version" : PIPELINE_VERSION,
        "adm_agg" : adm_agg,
        "disch_svd" : disch_svd,
        "stay_agg" : stay_agg,
        "vit_num" : vit_num,
        "vit_svd" : vit_svd,
        "disch_meta" : disch_meta,
        "vit_meta" : vit_meta,
    }, AUX_CACHE)
    log( f"Saved AUX cache: {AUX_CACHE} ")
# join AUX to all_df
def safe_merge (base, feat ):
    if feat is None:
        return base
    return base.merge(feat, on= "patient_id" , how="left")
all_df = safe_merge(all_df, adm_agg)
all_df = safe_merge(all_df, disch_svd)
all_df = safe_merge(all_df, stay_agg)
all_df = safe_merge(all_df, vit_num)
all_df = safe_merge(all_df, vit_svd)

```


### Page 198

```text

# receipts features
log("Building receipt features..." )
patient_ids = all_df[ "patient_id" ].astype( int).tolist()
rcpt_feat, code_list = receipt_patient_features(receipts, patient_ids)
all_df = all_df.merge(rcpt_feat, on= "patient_id" , how="left")
log(f"Receipt top-code features: {len(code_list)} codes")
# fill missing numeric from AUX joins
all_df = all_df.fillna( 0.0)
# split
train_df = all_df[all_df[ "_is_train" ] == 1].copy().reset_index(drop= True)
test_df  = all_df[all_df[ "_is_train" ] == 0].copy().reset_index(drop= True)
y = train_df[ "ed_cost_next3y_usd" ].astype( float).values
# feature columns
cat_cols = [ "primary_chronic" , "sex", "insurance" , "zip3"]
for c in cat_cols:
    train_df[c] = train_df[c].astype( str)
    test_df[c]  = test_df[c].astype( str)
drop_cols = { "ed_cost_next3y_usd" ,"_is_train" }
feat_cols = [c for c in train_df.columns if c not in drop_cols]
# separate numeric/cat
num_cols = [c for c in feat_cols if c not in cat_cols and c != "patient_id" ]
# convert numeric
for c in num_cols:
    train_df[c] = pd.to_numeric(train_df[c], errors= "coerce" ).fillna( 0.0).astype(
    test_df[c]  = pd.to_numeric(test_df[c], errors= "coerce" ).fillna( 0.0).astype(n
# one-hot
try:
    ohe = OneHotEncoder(handle_unknown= "ignore" , sparse_output= True)
except TypeError:
    ohe = OneHotEncoder(handle_unknown= "ignore" , sparse= True)
X_cat = ohe.fit_transform(pd.concat([train_df[cat_cols], test_df[cat_cols]], axis
X_cat_tr = X_cat[: len(train_df)]
X_cat_te = X_cat[ len(train_df):]

```


### Page 199

```text

X_num_tr = sp.csr_matrix(train_df[num_cols].values)
X_num_te = sp.csr_matrix(test_df[num_cols].values)
X_tr_all = sp.hstack([X_num_tr, X_cat_tr], format="csr")
X_te_all = sp.hstack([X_num_te, X_cat_te], format="csr")
log(f"X shape train= {X_tr_all.shape} , nnz={X_tr_all.nnz} ")
log(f"Target stats: mean= {y.mean(): .3f}, p50= {np.percentile(y, 50):.3f}, p95= {np.p
# CV split
strat = (train_df[ "primary_chronic" ].astype( str) + "|" +
         pd.qcut(train_df[ "prior_ed_cost_5y_usd" ].astype( float), q=10, duplicates
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle= True, random_state=SEED)
oof = np.zeros( len(train_df), dtype= float)
test_pred = np.zeros( len(test_df), dtype= float)
fold_logs = []
log("Starting CV training (V3 multisource)..." )
for fold, (tr_idx, va_idx) in enumerate (skf.split(np.zeros( len(train_df)), strat)
    t0 = time.time()
    Xtr, Xva = X_tr_all[tr_idx], X_tr_all[va_idx]
    ytr, yva = y[tr_idx], y[va_idx]
    model = xgb.XGBRegressor(**{**XGB_PARAMS, "random_state" : SEED + fold})
    try:
        model.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], verbose=VERBOSE_EVA
        mode = "cuda"
    except xgb.core.XGBoostError as e:
        log( f"[Fold {fold}] GPU failed -> CPU fallback. Error: {repr(e)[:180]}")
        p = dict(XGB_PARAMS); p[ "device" ] = "cpu"
        model = xgb.XGBRegressor(**{**p, "random_state" : SEED + fold})
        model.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], verbose=VERBOSE_EVA
        mode = "cpu"
    pred_tr = np.clip(predict_best(model, Xtr), 0.0, None)
    pred_va = np.clip(predict_best(model, Xva), 0.0, None)
    pred_te = np.clip(predict_best(model, X_te_all), 0.0, None)

```


### Page 200

```text

    tr_mae = mean_absolute_error(ytr, pred_tr)
    va_mae = mean_absolute_error(yva, pred_va)
    oof[va_idx] = pred_va
    test_pred += pred_te / N_SPLITS
    info = dict(
        fold=fold, mode=mode,
        train_mae= float(tr_mae), val_mae= float(va_mae),
        best_iteration= int(getattr(model, "best_iteration" , -1)) if getattr(model
        best_score= float(getattr(model, "best_score" , np.nan)) if getattr(model,"
        time_sec= float(time.time()-t0)
    )
    fold_logs.append(info)
    log( f"[Fold {fold}/{N_SPLITS} ] mode={mode} | TrainMAE= {tr_mae: .2f} | ValMAE= {v
oof_mae = mean_absolute_error(y, oof)
log(f"OOF MAE (raw): {oof_mae: .6f}")
# calibration
oof_cal, test_cal, shift = group_median_calibrate(train_df, oof, y, test_df, test_
oof_mae_cal = mean_absolute_error(y, oof_cal)
log(f"OOF MAE (calibrated): {oof_mae_cal: .6f} (delta {oof_mae - oof_mae_cal:+ .6f}
test_pred_final = np.clip(test_cal, 0.0, None)
# Save artifacts
oof_df = pd.DataFrame({ "patient_id" : train_df[ "patient_id" ].astype( int), "y_true"
oof_df.to_csv(RUN_DIR / "oof_predictions.csv" , index= False)
sub = pd.DataFrame({ "patient_id" : test_df[ "patient_id" ].astype( int).values, "ed_c
sub.to_csv(SUBMISSION_PATH, index= False)
sub.to_csv(RUN_DIR / "submission.csv" , index= False)
run_log = dict(
    run_id=RUN_ID,
    pipeline_version=PIPELINE_VERSION,
    xgboost_version= getattr(xgb,"__version__" ,"unknown" ),
    oof_mae_raw= float(oof_mae),

```


### Page 201

```text

你现在应该期待看到什么
OOF 会明显回到  440–460 级别（⾄少不可能再  520 ）。
如果 admissions/stays/vitals/notes 的特征真的有强信号， OOF calibrat ed 有机会从
443 再往下压 ，LB 也会跟着动。
你下⼀步发我什么（我才能继续把你推到  ~400 ）
把下⾯⼏⾏⽇志贴回来：
1. Receipt top-code features: ... （确认  code 特征维度）
2. 每折 TrainMAE / ValMAE / best_iter
3. OOF MAE raw / calibrated
4. 平台 LB 分数
如果这版能把  LB 从  462 拉到  ~440 附近，我们再做最后⼀跳：
加⼊ vitals 数值趋势特征 更丰富（每个  vital 的  day-by-day slope/ 波动），
或者做⼀个 ** ⼩型  seed ensemble （ 3 个  random_state 平均） ** 降低⽅差。    oof_mae_cal= float(oof_mae_cal),
    n_features= int(X_tr_all.shape[ 1]),
    nnz= int(X_tr_all.nnz),
    receipt_code_features= len(code_list),
    folds=fold_logs,
)
with open(RUN_DIR / "run_log.json" , "w", encoding= "utf-8") as f:
    json.dump(run_log, f, indent= 2)
log(f"Saved run artifacts: {RUN_DIR} ")
log(f"Saved submission -> {SUBMISSION_PATH} ")
Sour ces
Sour ces

```


### Page 202

```text

这是⼀个专家和我说的新的 eda 的想法 :
你这次的 “ 补充  EDA” ⾮常关键：我们之前确实在  “ 把数值当
成字符串 / 把浮点误差当成数据错误 ” 这个坑⾥绕了好⼏
圈，导致后续的  feature 选择和模型判断产⽣了偏差。下⾯
我把核⼼问题、为什么会影响你们的模型选择、以及我建议
的 Iter25 新⼀轮  EDA （专⻔验证 “sublinear/ 对数型关系  + 
更强 baseline” ）  给你⼀个清晰的路线图  + ⼀段你可以直接
在 Jupyter 单  cell 运⾏的代码。
1) 你发现的偏差是对的：我们之前 “ 数据质量判断⽅式 ” 会误
导模型策略
A. abs_diff ⾥  e-10 级别的 “ 差异 ” 本质上不是  mismatch
你贴出来的结果⾥：
sum_items vs prior = 1.0 match （实际上代表  line items 的
解析完全正确）
pdf_total vs prior match_rate ≈ 0.895 （主要因为  pdf_total  
抽取规则在部分  PDF 上失败  -> NaN ，⽽不是  TOTAL 不等
于 prior）
这说明： receipt 真实内容没问题，问题是  parser 的 “TOTAL 
⾏抽取 ”不鲁棒。
在建模上，如果我们把这些  NaN 当成 “ 坏数据 ” ，或者在  
feature ⾥让关键字段缺失率到  12%+ （ Iter24 就是 
0.126），会直接把模型逼向 “ 学缺失模式 ” ⽽不是学医学 / 经
济信号。
结论： Receipt 特征应该以  sum(line_totals) 作为  total ，
TOTAL 字段只做校验，不要作为主来源。
B. 你提到的  “L1 vs quantile(α=0.5) ⼀样 ” 不是  bug ，是数学
上等价
LightGBM 的：
objective="regression_l1" （ MAE / median ）

```


### Page 203

```text

objective="quantile", alpha=0.5
在理论上都是在拟合  条件中位数，因此很多情况下会得到
⼏乎完全⼀致的  OOF （甚⾄  best_iter 也可能⼀样） ，这是
正常的。
所以 Phase 3.3 那个现象属于 “ 预期⾏为 ” ，不是训练出错。
2) 你那张散点图给了我们⼀个更本质的结构： future vs  
prior 是明显的  sublinear （近似  log / 幂律）
你贴的图⾥， prior_ed_cost_5y_usd 增⻓时，
ed_cost_next3y_usd 仍然上升但  边际增幅变⼩（凹函数） 。
这在健康经济学⾥⾮常常⻅：医疗费⽤分布通常  右偏  + 
heavy-tail + 异⽅差，⽤原始尺度直接回归经常会让模型在
不同 cost 区间的  bias 不⼀致。经典讨论就是 “ 对数变换  vs 
GLM Gamma” 等。
因此你们之前的  “group-median ratio （ y≈ratio·x ） ” baseline  
会天然错，因为它假设线性⽐例（常数  ratio ），⽽你的图显
⽰ ratio 随  x 增⼤⽽下降（典型  sublinear ）。
这也是你们  residual 两阶段（ baseline + residual ）为什么
⼀直表现差： baseline 本⾝结构错了， residual 学到的是 “ 纠
错噪声 ”。
3) “新增  admissions/stays/notes 为何没提升？ ”—— 不是它
们⼀定没⽤，⽽是我们需要更聪明的⽤法
你们 EDA ⾥  admissions 特征对  residual 的  Spearman ⼤多
只有 0.02~0.05 ，这意味着  线性 / 单变量相关不强，但不代
表对 ⾮线性交互  没价值（例如： HF + ⾼  CCI + emergent  
admission 的组合才会推⾼未来  ED 成本）。
同时， zip3 很可能在这个  synthetic 任务⾥同时承担两类信
号：
价格⽔平 / 收费⽔平（区域 / 机构差异）
社会经济 / 就医可及性（ SDOH ）：低  SES 区域  ED 使⽤更

```


### Page 204

```text

⾼、慢病管理更差是有⼤量⽂献⽀持的。
不过：如果你只把这些  cross features “ 全量堆进去 ” ，在 
2000 样本下⾮常容易  引⼊噪声  + 过拟合  + CV 偏乐观，从
⽽出现你们  crossV1 上榜分数更差（ 473 ）的情况。
4) 我建议：现在开启  Iter25 （专⻔验证  sublinear baseline  
+ receipt“ 利⽤量 / 价格拆解 ” ）
你们要追  400 MAE ，我认为最可能的突破⼝不是 “ 再堆  100 
个弱相关特征 ” ，⽽是：
Track 1：把  baseline 从 “ 线性⽐例 ” 升级为 “ 幂律 / 对数 / 单调
拟合”
per-group （ chronic×insurance ）拟合：
log1p(y) = a_g + b_g * log1p(prior_cost) + c_g *  
log1p(prior_visits)
这是最符合你散点图形状的  baseline 。
或者 per-group isotonic regression （单调但不强制函数形
状）
很适合凹函数  + ⼩样本的情况。
这类 baseline ⼀旦能把  MAE 从  600 拉到接近  430~450 ，
后⾯再让  boosting 学  residual 才有意义。
Track 2：把  receipts 拆成  “utilization volume” vs “price  
level”
你之前  EDA ⾥  price_index 和  prior/target 有中等相关，但
对 residual 有负相关，这提⽰：
价格⾼不等于未来⽤得多
你需要把  prior_cost 分解成：
标准化  utilization ：⽤全体的  code median price 作为 
reference price ，算  standardized_total

```


### Page 205

```text

price factor ： prior_total / standardized_total
这在理赔 / 费⽤建模⾥很常⻅（类似  claims severity  
的“frequency × severity” 拆解思路） ，⽽  LightGBM 也明确
⽀持 Gamma/T weedie 这类适合  skewed cost 的⽬标分布。
Track 3： Admissions 只先⽤ “ 少⽽硬 ” 的特征（别全塞）
建议只保留你  EDA ⾥确实有信号、且语义明确的：
adm_charlson_mean/max
adm_emergent_rate
adm_edvis6m_mean/max/sum
dx counts （ HF/DiabetesComp/Pneumonia ）
其余先不要加，避免噪声。
5) Iter25 单  cell ED A 代码（你复制到⼀个  Jupyter cell 直接
跑）
这个 cell 会做  4 件事：
⽤你 iter10 的  receipts_parsed.joblib （你确认  OK 的那个）
重建 receipt 特征，并⽤  tolerance 做校验
⽣成 utilization 标准化总额  + price_index
把 admissions （ train+test ）聚合成  patient-level 少量硬特
征
做 5-fold S tratified CV 对⽐  四种  baseline ：
ratio baseline （你们旧的）
power-law/log baseline （建议的新  baseline ）
isotonic baseline （单调⾮参）

```


### Page 206

```text

⼀个 very-light LightGBM （只⽤少量核⼼特征）
并分别报告  overall MAE 和  outlier MAE
你跑完把输出表格  + ⽣成的  png 发我，我们再决定下⼀步
是不是把  power-law/isotonic baseline 作为主⼲，然后再做  
residual boosting / ensemble 。帮我做新的验证  以及评估
这个可⾏性
你这位专家给的  Iter25 思路⾮常值得做 ，⽽且它正好解释了我们之前 “ 卡在  460” 的两个关键
痛点：
(i) baseline 形状错（把明显凹函数当线性⽐例 / 线性加和） ，导致  residual 学成 “ 纠错噪声 ” ；
(ii) high-dim hash code 让模型⾼⽅差 ，CV 乐观、 LB 不动。
下⾯我先把 “ 可⾏性评估  + 我们要验证什么 ” 讲清楚，然后给你⼀段  Jupy ter 单  cell 可运⾏的
Iter25 ED A code，专⻔验证：
sublinear （ log/ 幂律） baseline 是否能显著压  MAE
receipts 的  utilization vs price  拆解是否是 “ 硬信号 ”
admissions 只⽤ “ 少⽽硬 ” 的特征是否能稳定增益
以及：你提到的  L1 vs quantile(0.5)  本来就等价、不是  bug （避免我们再误判训练异
常）
0) 可⾏性评估：这条路为什么 “ 对症 ”
A) L1 与  quantile(α=0.5) 等价是数学事实
很多库⾥  L1/MAE  本质上在拟合条件中位数；⽽  分位数回归  τ=0.5  就是中位数回归（也叫
LAD）。
scikit-learn 的  QuantileRegressor  说明  quantile=0.5 预测的是  50% 分位数
（median ）  
论⽂也明确指出  τ=0.5 对应  median quantile regression / L AD 
所以你看到  “L1 vs quantile(0.5) OOF 很像 ” 是预期⾏为 ，不是训练坏了。
B) “futur e vs prior 是  sublinear ” ⾮常符合健康经济学规律scikit-learn
PMC

```


### Page 207

```text

健康⽀出建模⾥，常⻅做法是  log 模型  / GLM Gamma-log / T weedie  等，⽤来处理右偏、
异⽅差与凹形边际效应。
Manning (2001) 专⻔讨论了对数模型与替代⽅法在  skewed expenditure 上的统计表现
Buntin 等（ 2004 ）⽐较多种⽅法（含  one-/two-part ）预测  Medicare cost 
如果存在⼤量  0 成本， Tweedie 能同时处理  0 点质量  + 正连续成本  
（你们⽬前  train 的  min > 0 ，不⼀定需要 two-part ，但 “ 凹函数  baseline” 这个点⾮常
关键。）
C) receipts 拆  “utilization × price” 是⾮常合理的 “ 结构特征 ”
保险/理赔⾥经典就是  frequency × severity 或者 “ 数量  × 单价 ” 拆解。
频率与严重度分解建模总赔付是经典路径  
scikit-learn T weedie ⽰例⾥也展⽰了  frequency / severity 的思路  
你们的  receipts 恰好能做 “ 标准化单价 ” 的  utilization 估计，这是 很可能补上缺失结构的⼀
步。
D) Isot onic b aseline 是 “ 单调不设形状 ” ，适合⼩样本凹函数
IsotonicR egression 是  1D 单调拟合（不假设函数形状，只要求单调） ，⾮常适合你们这种
“明显随  prior_cost 单调上升但凹 ” 的关系验证。  
1) Iter25 我们要验证的结论（你跑完我就能决定  Phase 3 主⼲）
你跑完下⾯代码后，我们最关⼼  4 个输出：
1. ratio b aseline vs log b aseline vs isot onic b aseline 的  OOF MAE
如果 log/isotonic 明显优于  ratio （通常会） ，说明  baseline 形状确实是关键缺⼝。
2. receipts 的  standar dized_t otal （ utilization ） 和 price_index （ price fact or ） 是否对
residual 有贡献
如果“utilization_std_total” 解释⼒强⽽  price_index 弱，说明未来主要由 “ ⽤量 ” 驱动
如果 price_index 在某些组（保险 /zip3 ）很重要，说明它是 “ 价格环境 / 机构 ” 信号
3. admissions “ 少⽽硬 ” 的特征加⼊后， very-light booster 是否稳定改善（⽽不是  CV 虚
增）
4. 是否存在  0 成本  / 极低成本（判断  two-part / T weedie 是否必要）PubMed
ScienceDirect
PMC
ScienceDirect
scikit-learn
scikit-learn +1

```


### Page 208

```text

2) Iter25 单  cell ED A 代码（直接贴进  Jupy ter 运⾏）
✅ 特点：
“只⽤ receipts_parsed.joblib （不再碰  pdf_t otal ）”
“⽣成 utilization_st d_total + price_index ”
“admissions 存在则聚合少量硬特征，不存在⾃动跳过 ”
“5-fold CV 对⽐  4 种  baseline ： ratio / log-pow er / isot onic / v ery-light X GB”
“输出 overall MAE + t op5% MAE （尾部） ”
“画图并保存到  cache_iter10/eda_iter25/ ”
python
# ============================================================
# Iter25 EDA — Validate: sublinear baseline + utilization×price decomposition + s
# One Jupyter cell. Safe, no leakage, uses receipts_parsed.joblib.
# ============================================================
import os, re, json, time, math, warnings
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load, dump
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
warnings.filterwarnings( "ignore" )
np.set_printoptions(suppress= True)
# -----------------------------
# PATHS (EDIT IF NEEDED)
# -----------------------------
BASE_DIR = Path( r"D:\AgentDs\agent_ds_healthcare" )

```


### Page 209

```text

CACHE_DIR = BASE_DIR / "cache_iter10"
CACHE_DIR.mkdir(parents= True, exist_ok= True)
TRAIN_CSV = BASE_DIR / "ed_cost_train.csv"
TEST_CSV  = BASE_DIR / "ed_cost_test.csv"
PATIENTS_CSV = BASE_DIR / "patients.csv"
RECEIPT_CACHE = CACHE_DIR / "receipts_parsed.joblib"
# optional admissions
ADM_TR = BASE_DIR / "admissions_train.csv"
ADM_TE = BASE_DIR / "admissions_test.csv"
OUT_DIR = CACHE_DIR / "eda_iter25"
OUT_DIR.mkdir(parents= True, exist_ok= True)
SEED = 42
N_SPLITS = 5
TOL = 1e-2
# -----------------------------
# Utils
# -----------------------------
def log(msg):
    print(msg)
def norm_cols (df):
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df
def safe_float (x):
    if x is None:
        return np.nan
    if isinstance (x, (int,float,np.number)):
        return float(x)
    s = str(x).strip().replace( "$","").replace( ",","")
    try:
        return float(s)
    except Exception:
        return np.nan

```


### Page 210

```text

def safe_int (x, default= 1):
    try:
        if x is None:
            return default
        if isinstance (x, (int, np.integer)):
            return int(x)
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return default
        return int(float(s))
    except Exception:
        return default
def build_prior_bins (train_prior, n_bins= 10):
    q = np.quantile(train_prior.astype( float), np.linspace( 0,1,n_bins+ 1))
    bins = np.unique(q)
    if len(bins) < 3:
        bins = np.unique(np.quantile(train_prior.astype( float), [0,0.5,1.0]))
    return bins
def topk_mae (y_true, y_pred, q= 0.95):
    thr = np.quantile(y_true, q)
    m = y_true >= thr
    if m.sum() == 0:
        return np.nan
    return mean_absolute_error(y_true[m], y_pred[m])
# -----------------------------
# Load data
# -----------------------------
train = norm_cols(pd.read_csv(TRAIN_CSV))
test  = norm_cols(pd.read_csv(TEST_CSV))
patients = norm_cols(pd.read_csv(PATIENTS_CSV))
train = train.merge(patients, on= "patient_id" , how="left")
test  = test.merge(patients, on= "patient_id" , how="left")
assert RECEIPT_CACHE.exists(), f"Missing {RECEIPT_CACHE} "
receipts = load(RECEIPT_CACHE)
assert isinstance (receipts, dict) and len(receipts) >= 3000, "Bad receipts cache.

```


### Page 211

```text

all_df = pd.concat([train.assign(_split= "train"), test.assign(_split= "test")], ax
all_ids = all_df[ "patient_id" ].astype( int).tolist()
# -----------------------------
# Build line-item dataframe (for reference prices)
# -----------------------------
rows = []
for pid in all_ids:
    obj = receipts.get( int(pid), { "items":[]})
    items = obj.get( "items", []) if isinstance (obj, dict) else []
    for it in items:
        if not isinstance (it, dict):
            continue
        code = str(it.get( "code","")).strip()
        amt  = safe_float(it.get( "amount" , np.nan))
        qty  = safe_int(it.get( "qty", 1), default= 1)
        if code == "" or (not np.isfinite(amt)):
            continue
        qty = max(qty, 1)
        unit = float(amt) / qty
        rows.append(( int(pid), code, qty, float(amt), unit))
li = pd.DataFrame(rows, columns=[ "patient_id" ,"code","qty","amount" ,"unit_price" ]
log(f"[iter25] line_items rows = {len(li)} | unique codes = {li['code'].nunique()
global_med_unit = float(li["unit_price" ].median())
ref_price = li.groupby( "code")["unit_price" ].median().to_dict()
# standardized utilization total per patient
li["ref_unit" ] = li["code"].map(ref_price).fillna(global_med_unit)
li["std_amount" ] = li["qty"] * li["ref_unit" ]
agg = li.groupby( "patient_id" ).agg(
    sum_items=( "amount" ,"sum"),
    std_total=( "std_amount" ,"sum"),
    n_items=( "amount" ,"count"),
    n_unique_codes=( "code","nunique" ),
).reset_index()

```


### Page 212

```text

# top1/top3 shares for complexity
def top_shares (x):
    x = np.asarray(x, dtype= float)
    x = x[np.isfinite(x)]
    s = x. sum()
    if s <= 0 or len(x)==0:
        return pd.Series({ "top1_share" :0.0, "top3_share" :0.0})
    xs = np.sort(x)[::- 1]
    top1 = xs[ 0]
    top3 = xs[: 3].sum()
    return pd.Series({ "top1_share" : float(top1/s), "top3_share" : float(top3/s)})
tops = li.groupby( "patient_id" )["amount" ].apply( lambda s: top_shares(s.values)).r
# tops has duplicated columns structure due to series; clean it:
tops = tops.pivot(index= "patient_id" , columns= "level_1" , values= "amount" ).reset_i
# (above pivot trick is sometimes messy in pandas versions; robust alternative be
if "top1_share"  not in tops.columns or "top3_share"  not in tops.columns:
    tmp = li.groupby( "patient_id" )["amount" ].apply( lambda s: top_shares(s.values)
    tops = tmp.reset_index()
    # expand series
    tops = pd.concat([tops[[ "patient_id" ]], tops[ "amount" ].apply(pd.Series)], axi
feat_rcpt = agg.merge(tops, on= "patient_id" , how="left")
feat_rcpt[ "distributed_cost" ] = 1.0 - feat_rcpt[ "top3_share" ].fillna( 0.0)
# price index
feat_rcpt[ "std_total" ] = feat_rcpt[ "std_total" ].replace( 0, np.nan).fillna( 1e-9)
feat_rcpt[ "price_index" ] = feat_rcpt[ "sum_items" ] / feat_rcpt[ "std_total" ]
# merge receipt feats
all_df = all_df.merge(feat_rcpt, on= "patient_id" , how="left")
# -----------------------------
# Validate: sum_items == prior cost (tolerance) ; pdf_total is irrelevant here
# -----------------------------
if "prior_ed_cost_5y_usd"  in all_df.columns:
    diff = (all_df.loc[all_df[ "_split" ]=="train","sum_items" ] - all_df.loc[all_df
    match = float((diff <= TOL).mean())
    log( f"[iter25] TRAIN sum_items vs prior match_rate@ {TOL} = {match:.4f} | max_

```


### Page 213

```text

# -----------------------------
# Admissions: keep " 少⽽硬 " features only (if files exist)
# -----------------------------
adm_feat = None
if ADM_TR.exists() and ADM_TE.exists():
    adm_tr = norm_cols(pd.read_csv(ADM_TR))
    adm_te = norm_cols(pd.read_csv(ADM_TE))
    adm = pd.concat([adm_tr, adm_te], axis= 0, ignore_index= True)
    # expected columns (if missing, skip gracefully)
    colmap = {
        "los_days" :"los_days" ,
        "charlson_band" :"charlson_band" ,
        "acuity_emergent" :"acuity_emergent" ,
        "ed_visits_6m" :"ed_visits_6m" ,
        "primary_dx" :"primary_dx" ,
        "admission_id" :"admission_id"
    }
    for k in list(colmap.keys()):
        if colmap[k] not in adm.columns:
            colmap[k] = None
    g = adm.groupby( "patient_id" )
    out = pd.DataFrame({ "patient_id" : g.size().index.values})
    out[ "adm_n"] = g.size().values
    def agg_col(col, fn, name ):
        if col is None: 
            out[name] = 0.0
        else:
            out[name] = g[col].agg(fn).values
    agg_col(colmap[ "los_days" ], "mean", "adm_los_mean" )
    agg_col(colmap[ "los_days" ], "max",  "adm_los_max" )
    agg_col(colmap[ "charlson_band" ], "mean", "adm_charlson_mean" )
    agg_col(colmap[ "charlson_band" ], "max",  "adm_charlson_max" )
    # emergent rate is mean if it's 0/1
    agg_col(colmap[ "acuity_emergent" ], "mean", "adm_emergent_rate" )
    agg_col(colmap[ "ed_visits_6m" ], "mean", "adm_edvis6m_mean" )
    agg_col(colmap[ "ed_visits_6m" ], "max",  "adm_edvis6m_max" )

```


### Page 214

```text

    agg_col(colmap[ "ed_visits_6m" ], "sum",  "adm_edvis6m_sum" )
    # dx counts for the 3 chronic labels if present
    if colmap[ "primary_dx" ] is not None:
        for dx in ["HF","DiabetesComp" ,"Pneumonia" ]:
            out[ f"adm_dx_ {dx}"] = g[colmap[ "primary_dx" ]].apply( lambda s: int((s.
    else:
        for dx in ["HF","DiabetesComp" ,"Pneumonia" ]:
            out[ f"adm_dx_ {dx}"] = 0.0
    adm_feat = out
    all_df = all_df.merge(adm_feat, on= "patient_id" , how="left")
    log( "[iter25] admissions features added." )
else:
    log( "[iter25] admissions_train/test not found -> skip admissions features." )
all_df = all_df.fillna( 0.0)
# -----------------------------
# Core EDA plots: sublinear check
# -----------------------------
tr = all_df[all_df[ "_split" ]=="train"].copy()
y = tr["ed_cost_next3y_usd" ].astype( float).values
x = tr["prior_ed_cost_5y_usd" ].astype( float).values
plt.figure()
plt.scatter(np.log1p(x), np.log1p(y), s= 8, alpha= 0.35)
plt.xlabel( "log1p(prior_ed_cost_5y_usd)" )
plt.ylabel( "log1p(ed_cost_next3y_usd)" )
plt.title( "Sublinear check: log1p(prior_cost) vs log1p(future_cost)" )
plt.tight_layout()
plt.savefig(OUT_DIR/ "scatter_log_prior_vs_log_future.png" , dpi=150)
plt.show()
# ratio vs prior bins
tr["ratio_y_over_prior" ] = tr["ed_cost_next3y_usd" ] / (tr[ "prior_ed_cost_5y_usd" ] 
tr["prior_bin" ] = pd.qcut(tr[ "prior_ed_cost_5y_usd" ], q=10, duplicates= "drop")
ratio_by_bin = tr.groupby( "prior_bin" )["ratio_y_over_prior" ].mean()
plt.figure(figsize=( 8,4))
plt.plot( range(len(ratio_by_bin)), ratio_by_bin.values, marker= "o")

```


### Page 215

```text

plt.xticks( range(len(ratio_by_bin)), [ str(i) for i in range(len(ratio_by_bin))])
plt.xlabel( "prior_cost decile bin (0..9)" )
plt.ylabel( "mean(y / prior_cost)" )
plt.title( "Evidence of sublinear: ratio decreases with prior_cost" )
plt.tight_layout()
plt.savefig(OUT_DIR/ "ratio_by_prior_decile.png" , dpi=150)
plt.show()
# price_index distribution
plt.figure()
plt.hist(tr[ "price_index" ].clip(0, tr["price_index" ].quantile( 0.995)), bins= 60)
plt.title( "price_index = prior_total / standardized_total (clipped at 99.5%)" )
plt.xlabel( "price_index" )
plt.ylabel( "count")
plt.tight_layout()
plt.savefig(OUT_DIR/ "price_index_hist.png" , dpi=150)
plt.show()
# -----------------------------
# CV Baselines
# -----------------------------
# group key for baselines
tr["grp"] = tr["primary_chronic" ].astype( str) + "|" + tr["insurance" ].astype( str)
# stratify: primary_chronic x prior_cost decile
strat = (tr[ "primary_chronic" ].astype( str) + "|" +
         pd.qcut(tr[ "prior_ed_cost_5y_usd" ].astype( float), q=10, duplicates= "drop
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle= True, random_state=SEED)
def oof_ratio_baseline (df, y, splits ):
    oof = np.zeros( len(df), dtype= float)
    for tr_idx, va_idx in splits:
        dtr = df.iloc[tr_idx]
        dva = df.iloc[va_idx]
        ratio = y[tr_idx] / (dtr[ "prior_ed_cost_5y_usd" ].values + 1e-9)
        med = pd.Series(ratio).groupby(dtr[ "grp"].values).median().to_dict()
        gmed = float(np.median(ratio))
        pred = (dva[ "prior_ed_cost_5y_usd" ].values) * np.array([med.get(g, gmed) 
        oof[va_idx] = np.clip(pred, 0.0, None)

```


### Page 216

```text

    return oof
def oof_log_power_baseline (df, y, splits, use_extra= False):
    """
    log1p(y) = a_g + b_g*log1p(prior_cost) + c_g*log1p(prior_visits) (+ optional 
    """
    oof = np.zeros( len(df), dtype= float)
    for tr_idx, va_idx in splits:
        dtr = df.iloc[tr_idx]
        dva = df.iloc[va_idx]
        ylog = np.log1p(y[tr_idx])
        # global fit fallback
        def build_X(d):
            cols = [
                np.ones( len(d)),
                np.log1p(d[ "prior_ed_cost_5y_usd" ].values),
                np.log1p(d[ "prior_ed_visits_5y" ].values),
            ]
            if use_extra:
                cols += [
                    np.log1p(d[ "std_total" ].values),
                    np.log1p(d[ "price_index" ].values),
                ]
            return np.vstack(cols).T
        Xg = build_X(dtr)
        rg = Ridge(alpha= 1.0, fit_intercept= False)
        rg.fit(Xg, ylog)
        # per-group fits
        preds = np.zeros( len(dva), dtype= float)
        for g, idx_va_g in dva.groupby( "grp").groups.items():
            idx_va = np.array( list(idx_va_g), dtype= int)
            # training rows in same group
            mask_tr = (dtr[ "grp"].values == g)
            if mask_tr. sum() >= 30:
                X = build_X(dtr.loc[mask_tr])
                r = Ridge(alpha= 1.0, fit_intercept= False)
                r.fit(X, np.log1p(y[tr_idx][mask_tr]))

```


### Page 217

```text

                Xva = build_X(dva.loc[idx_va])
                plog = r.predict(Xva)
            else:
                Xva = build_X(dva.loc[idx_va])
                plog = rg.predict(Xva)
            preds[idx_va] = np.expm1(plog)
        oof[va_idx] = np.clip(preds, 0.0, None)
    return oof
def oof_isotonic_baseline (df, y, splits ):
    """
    per-group isotonic on log1p(prior_cost) -> log1p(y)
    """
    oof = np.zeros( len(df), dtype= float)
    for tr_idx, va_idx in splits:
        dtr = df.iloc[tr_idx]
        dva = df.iloc[va_idx]
        xtr = np.log1p(dtr[ "prior_ed_cost_5y_usd" ].values)
        ytr = np.log1p(y[tr_idx])
        # global isotonic fallback
        iso_g = IsotonicRegression(increasing= True, out_of_bounds= "clip")
        iso_g.fit(xtr, ytr)
        preds = np.zeros( len(dva), dtype= float)
        for g, idx_va_g in dva.groupby( "grp").groups.items():
            idx_va = np.array( list(idx_va_g), dtype= int)
            mask_tr = (dtr[ "grp"].values == g)
            xva = np.log1p(dva.loc[idx_va, "prior_ed_cost_5y_usd" ].values)
            if mask_tr. sum() >= 40:
                iso = IsotonicRegression(increasing= True, out_of_bounds= "clip")
                iso.fit(np.log1p(dtr.loc[mask_tr, "prior_ed_cost_5y_usd" ].values)
                        np.log1p(y[tr_idx][mask_tr]))
                plog = iso.predict(xva)
            else:
                plog = iso_g.predict(xva)
            preds[idx_va] = np.expm1(plog)
        oof[va_idx] = np.clip(preds, 0.0, None)
    return oof

```


### Page 218

```text

def oof_light_xgb (df, y, splits ):
    # minimal hard features (stable)
    use_cols = [
        "age","prior_ed_visits_5y" ,"prior_ed_cost_5y_usd" ,
        "std_total" ,"price_index" ,
        "n_items" ,"n_unique_codes" ,"top1_share" ,"top3_share" ,"distributed_cost"
    ]
    # add admissions hard features if present
    for c in ["adm_n","adm_charlson_mean" ,"adm_charlson_max" ,"adm_emergent_rate" ,
        if c in df.columns:
            use_cols.append(c)
    X_num = df[use_cols].astype( float).values
    # categoricals
    cat_cols = [ "primary_chronic" ,"insurance" ,"sex","zip3"]
    ohe = OneHotEncoder(handle_unknown= "ignore" , sparse_output= False)
    X_cat = ohe.fit_transform(df[cat_cols].astype( str))
    X = np.hstack([X_num, X_cat]).astype(np.float32)
    oof = np.zeros( len(df), dtype= float)
    params = dict(
        n_estimators= 3000,
        learning_rate= 0.05,
        max_depth= 4,
        min_child_weight= 50,
        subsample= 0.8,
        colsample_bytree= 0.8,
        reg_lambda= 15.0,
        reg_alpha= 0.5,
        objective= "reg:absoluteerror" ,
        eval_metric= "mae",
        early_stopping_rounds= 200,
        tree_method= "hist",
        device= "cuda",
        n_jobs=- 1,
        verbosity= 0,
    )
    for fold, (tr_idx, va_idx) in enumerate (splits, 1):
        Xtr, Xva = X[tr_idx], X[va_idx]

```


### Page 219

```text

        ytr, yva = y[tr_idx], y[va_idx]
        m = xgb.XGBRegressor(**{**params, "random_state" : SEED+fold})
        try:
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose= False)
        except xgb.core.XGBoostError:
            params_cpu = dict(params); params_cpu[ "device" ] = "cpu"
            m = xgb.XGBRegressor(**{**params_cpu, "random_state" : SEED+fold})
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose= False)
        pred = m.predict(Xva)
        oof[va_idx] = np.clip(pred, 0.0, None)
    return oof
splits = list(skf.split(np.zeros( len(tr)), strat))
log("\n[iter25] Running baselines (5-fold CV)..." )
pred_ratio = oof_ratio_baseline(tr, y, splits)
pred_log   = oof_log_power_baseline(tr, y, splits, use_extra= False)
pred_log2  = oof_log_power_baseline(tr, y, splits, use_extra= True)
pred_iso   = oof_isotonic_baseline(tr, y, splits)
pred_xgb   = oof_light_xgb(tr, y, splits)
def report(name, pred ):
    mae = mean_absolute_error(y, pred)
    mae95 = topk_mae(y, pred, q= 0.95)
    mae99 = topk_mae(y, pred, q= 0.99)
    return {"method" : name, "mae": mae, "mae_top5%" : mae95, "mae_top1%" : mae99}
summary = pd.DataFrame([
    report( "ratio_baseline (group median y/prior)" , pred_ratio),
    report( "log_power (log1p y ~ log1p prior + log1p visits)" , pred_log),
    report( "log_power + util/price (adds log std_total + log price_index)" , pred_
    report( "isotonic (per-group log1p prior -> log1p y)" , pred_iso),
    report( "very_light_xgb (core + util/price + few admissions)" , pred_xgb),
]).sort_values( "mae")
log("\n[iter25] CV baseline comparison (lower is better):" )
print(summary.to_string(index= False))
summary.to_csv(OUT_DIR/ "baseline_comparison.csv" , index= False)

```


### Page 220

```text

3) 你跑完后怎么判断 “ 这条路是否可⾏ ”
你只需要把下⾯这些贴回来：
1. baseline_comparison.csv  打印出来的那⼀⾏  summary （越低越好）
2. 三张图的结论（不⽤发图也⾏，告诉我趋势） ：
ratio_by_prior_decile.png  是否明显下降（⽀持  sublinear ）
price_index_hist.png  是否有分散（说明  price factor 有信息）
3. admissions 是否存在、被加进去了（代码会打印）
预期（如果专家思路正确）
log_power  / isotonic  会显著优于  ratio_baseline
log_power + util/price  若优于  log_power ，说明  utilization/price 拆解是 “ 硬信号 ”
very_light_xgb  应该在不爆炸的前提下，进⼀步压  MAE （如果  admissions 有⽤，会
更明显）
4) 下⼀步（等你贴回结果后我会做什么）
如果 isotonic/log  baseline 明显赢： Phase 3 主⼲就改成  sublinear b aseline +
boosting 学  residual（这次  baseline 不是错形状的） 。
如果 util/price  有贡献：把它变成核⼼结构特征，后续再考虑更强的
frequency×severity 或  Tweedie ⻛格建模（理论上能处理  skew/ 半连续，但你们⽬前训
练集似乎⽆  0 ，所以不急） 。  
把 Iter25 的  summary 贴回来，我们就能⾮常明确地决定：
是⾛ “log b aseline + r esidual boosting” 还是  “isot onic b aseline + r esidual”，以及
util/price 拆解值不值得成为主特征。log(f"\nSaved outputs to: {OUT_DIR} ")
log("Key plots: scatter_log_prior_vs_log_future.png, ratio_by_prior_decile.png, p
log("Key table: baseline_comparison.csv" )
PMC +2
Sour ces
Sour ces

```


### Page 221

```text

-----------------------------------------------------------------
----------
AssertionError                            T raceback (most recent call  
last)
Cell In[36], line 109
    107 assert RECEIPT_CA CHE.exists(), f"Missing  
{RECEIPT_CA CHE}"
    108 receipts = load(RECEIPT_CA CHE)
--> 109 assert isinstance(receipts, dict) and len(receipts)  
>= 3000, "Bad receipts cache."
    111 all_df = pd.concat([train.assign(_split="train"),  
test.assign(_split="test")], axis=0, ignore_index=T rue)
    112 all_ids = all_df["patient_id"].astype(int).tolist()
AssertionError: Bad receipts cache.
你这个报错  100% 是  receipts_p arsed.joblib ⼜变成 “ ⾮  dict / 只有很少条⽬ ” 的坏缓存 （就
像你之前第⼀次  EDA 时缓存⾥只有  2 条⼀样）。这通常来⾃两类原因：
被其它实验覆盖 （例如某次写⼊了⼀个不同结构的对象到同⼀路径） ；
写⽂件中途被中断 导致⽂件不完整 / 损坏。joblib.dump/load  本质是  pickle 持久化，
写⼊不是原⼦操作；更稳妥的做法是 “ 写到临时⽂件 → 再  replace/rename 覆盖 ” ，避免
出现半写⼊⽂件。  
下⾯给你⼀个 ⼀键修复  + 继续跑  Iter25 ED A 的 Jupy ter 单  cell。它会：
1. 尝试加载  receipts_p arsed.joblib  → 校验结构 / 数量
2. 若坏：⾃动在  cache_iter10/  ⾥找  最⼤的可⽤  receipts_p arsed.joblib /
receipts_p arsed.bad_.joblib** 来恢复
3. 还不⾏：就从  receipts_pdf/  快速重建 （你机器之前  15~20s 就能跑完）
4. ⽤原⼦写⼊ 保存新的  joblib ，避免再写坏
5. 然后直接跑你要的  Iter25 baselines （ ratio / log-power / isotonic / very-light X GB ）joblib.readthedo …+3
python

```


### Page 222

```text

# ============================================================
# Iter25 EDA (ROBUST) — auto-fix receipts_parsed.joblib then run baselines
# One Jupyter cell
# ============================================================
import os, re, json, time, math, warnings
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load, dump
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
warnings.filterwarnings( "ignore" )
np.set_printoptions(suppress= True)
# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = Path( r"D:\AgentDs\agent_ds_healthcare" )
CACHE_DIR = BASE_DIR / "cache_iter10"
CACHE_DIR.mkdir(parents= True, exist_ok= True)
TRAIN_CSV = BASE_DIR / "ed_cost_train.csv"
TEST_CSV  = BASE_DIR / "ed_cost_test.csv"
PATIENTS_CSV = BASE_DIR / "patients.csv"
PDF_DIR = BASE_DIR / "receipts_pdf"
RECEIPT_CACHE = CACHE_DIR / "receipts_parsed.joblib"

```


### Page 223

```text

# optional admissions
ADM_TR = BASE_DIR / "admissions_train.csv"
ADM_TE = BASE_DIR / "admissions_test.csv"
OUT_DIR = CACHE_DIR / "eda_iter25"
OUT_DIR.mkdir(parents= True, exist_ok= True)
SEED = 42
N_SPLITS = 5
TOL = 1e-2
# -----------------------------
# Utils
# -----------------------------
def log(msg):
    ts = datetime.now().strftime( "%H:%M:%S" )
    print(f"[{ts}] {msg}")
def norm_cols (df):
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df
def safe_float (x):
    if x is None:
        return np.nan
    if isinstance (x, (int,float,np.number)):
        return float(x)
    s = str(x).strip().replace( "$","").replace( ",","")
    try:
        return float(s)
    except Exception:
        return np.nan
def safe_int (x, default= 1):
    try:
        if x is None:
            return default
        if isinstance (x, (int, np.integer)):
            return int(x)

```


### Page 224

```text

        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return default
        return int(float(s))
    except Exception:
        return default
def topk_mae (y_true, y_pred, q= 0.95):
    thr = np.quantile(y_true, q)
    m = y_true >= thr
    return mean_absolute_error(y_true[m], y_pred[m]) if m.sum() else np.nan
def build_prior_bins (train_prior, n_bins= 10):
    q = np.quantile(train_prior.astype( float), np.linspace( 0,1,n_bins+ 1))
    bins = np.unique(q)
    if len(bins) < 3:
        bins = np.unique(np.quantile(train_prior.astype( float), [0,0.5,1.0]))
    return bins
# -----------------------------
# Receipt cache: validate / recover / rebuild
# -----------------------------
CODE_RE = re. compile(r"^[A-Z]?\d{4,5}$|^[A-Z0-9]{4,7}$" )
def validate_receipts (obj, expected_n, min_frac= 0.95):
    if not isinstance (obj, dict):
        return False, f"type={type(obj)} not dict"
    if len(obj) < int(min_frac * expected_n):
        return False, f"len={len(obj)} < {int(min_frac*expected_n)}"
    # spot check a few entries
    checked = 0
    for k,v in obj.items():
        checked += 1
        if not isinstance (v, dict):
            return False, f"value for key {k} is {type(v)} not dict"
        if "items" not in v:
            return False, f"value for key {k} missing 'items'"
        if not isinstance (v["items"], list):
            return False, f"value for key {k} items is {type(v['items'])} not list
        break

```


### Page 225

```text

    return True, "ok"
def atomic_dump_joblib (obj, path: Path ):
    tmp = path.with_suffix( f".tmp_{int(time.time())}.joblib")
    dump(obj, tmp)
    os.replace(tmp, path)  # atomic replace on same filesystem
def try_load_joblib (path: Path ):
    try:
        return load(path)
    except Exception as e:
        return None
def recover_best_receipts (cache_dir: Path, expected_n: int):
    cands = list(cache_dir.glob( "receipts_parsed*.joblib" )) + list(cache_dir.glob
    # sort by size desc then mtime desc
    cands = sorted(set(cands), key= lambda p: (p.stat().st_size, p.stat().st_mtime
    for p in cands:
        obj = try_load_joblib(p)
        if obj is None:
            continue
        ok, reason = validate_receipts(obj, expected_n)
        if ok:
            return obj, p
    return None, None
def parse_line_item (line: str):
    line = line.strip()
    if not line:
        return None
    m_amt = re.search( r"(\d[\d,]*\.\d{2})\s*$" , line)
    if not m_amt:
        return None
    amt = safe_float(m_amt.group( 1))
    left = line[:m_amt.start()].strip()
    m_qty = re.search( r"(\d+)\s*$" , left)
    if not m_qty:
        return None
    qty = int(m_qty.group( 1))

```


### Page 226

```text

    left2 = left[:m_qty.start()].strip()
    if not left2:
        return None
    code = left2.split()[ 0].strip()
    if not CODE_RE. match(code):
        return None
    desc = left2[ len(code):].strip()
    return {"code": code, "desc": desc, "qty": qty, "amount" : float(amt)}
def rebuild_receipts_from_pdfs (pdf_dir: Path ):
    import fitz
    from concurrent.futures import ThreadPoolExecutor, as_completed
    try:
        from tqdm.auto import tqdm
    except Exception:
        def tqdm(x, **kwargs ): return x
    pdfs = sorted(pdf_dir.glob( "receipt_*.pdf" ))
    pid_paths = []
    for p in pdfs:
        m = re.search( r"receipt_(\d+)\.pdf$" , p.name)
        if m:
            pid_paths.append(( int(m.group( 1)), p))
    def parse_one (pid, path: Path ):
        out = { "pdf_path" : str(path), "items": [], "text_len" : 0}
        try:
            with fitz.open(path) as doc:
                text = []
                for i in range(doc.page_count):
                    text.append(doc.load_page(i).get_text( "text", sort=True))
                text = "\n".join(text)
            out[ "text_len" ] = len(text)
            items = []
            for ln in text.splitlines():
                it = parse_line_item(ln)
                if it is not None:
                    items.append(it)

```


### Page 227

```text

            out[ "items"] = items
            return pid, out
        except Exception as e:
            out[ "parse_error" ] = repr(e)
            return pid, out
    receipts = {}
    n_workers = max(4, (os.cpu_count() or 8) - 2)
    log( f"[receipts] rebuilding from PDFs: {len(pid_paths)} files | workers= {n_wo
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(parse_one, pid, path): pid for pid, path in pid_paths}
        for fut in tqdm(as_completed(futs), total= len(futs), desc= "Parsing PDFs" )
            pid, obj = fut.result()
            receipts[ int(pid)] = obj
    log( f"[receipts] rebuilt {len(receipts)} receipts in {time.time()-t0: .1f}s")
    return receipts
# -----------------------------
# Load train/test to know expected_n
# -----------------------------
train = norm_cols(pd.read_csv(TRAIN_CSV))
test  = norm_cols(pd.read_csv(TEST_CSV))
patients = norm_cols(pd.read_csv(PATIENTS_CSV))
train = train.merge(patients, on= "patient_id" , how="left")
test  = test.merge(patients, on= "patient_id" , how="left")
expected_n = len(train) + len(test)
# -----------------------------
# Ensure receipts cache is valid
# -----------------------------
receipts = None
src = None
if RECEIPT_CACHE.exists():
    obj = try_load_joblib(RECEIPT_CACHE)
    if obj is not None:
        ok, reason = validate_receipts(obj, expected_n)
        if ok:

```


### Page 228

```text

            receipts = obj
            src = RECEIPT_CACHE
        else:
            log( f"[receipts] current cache INVALID: {reason} ")
if receipts is None:
    # try recover from other caches
    rec, p = recover_best_receipts(CACHE_DIR, expected_n)
    if rec is not None:
        receipts = rec
        src = p
        log( f"[receipts] recovered valid cache from: {p.name}  (len={len(receipts)
        # copy back to main path with atomic write
        atomic_dump_joblib(receipts, RECEIPT_CACHE)
        log( f"[receipts] wrote recovered cache -> {RECEIPT_CACHE} ")
    else:
        # rebuild
        receipts = rebuild_receipts_from_pdfs(PDF_DIR)
        ok, reason = validate_receipts(receipts, expected_n)
        log( f"[receipts] rebuilt validation: {ok} ({reason} )")
        atomic_dump_joblib(receipts, RECEIPT_CACHE)
        log( f"[receipts] saved rebuilt cache -> {RECEIPT_CACHE} ")
        src = RECEIPT_CACHE
log(f"[receipts] using cache source: {src} | type= {type(receipts)} | len= {len(rec
# ============================================================
# Iter25 EDA starts here
# ============================================================
all_df = pd.concat([train.assign(_split= "train"), test.assign(_split= "test")], ax
all_ids = all_df[ "patient_id" ].astype( int).tolist()
# Build line-item table for reference prices
rows = []
for pid in all_ids:
    obj = receipts.get( int(pid), { "items": []})
    items = obj.get( "items", []) if isinstance (obj, dict) else []
    for it in items:
        if not isinstance (it, dict):

```


### Page 229

```text

            continue
        code = str(it.get( "code","")).strip()
        amt  = safe_float(it.get( "amount" , np.nan))
        qty  = safe_int(it.get( "qty", 1), default= 1)
        if not code or (not np.isfinite(amt)):
            continue
        qty = max(qty, 1)
        unit = float(amt) / qty
        rows.append(( int(pid), code, qty, float(amt), unit))
li = pd.DataFrame(rows, columns=[ "patient_id" ,"code","qty","amount" ,"unit_price" ]
log(f"[iter25] line_items rows= {len(li)} | unique_codes= {li['code'].nunique()}")
global_med_unit = float(li["unit_price" ].median())
ref_price = li.groupby( "code")["unit_price" ].median().to_dict()
li["ref_unit" ] = li["code"].map(ref_price).fillna(global_med_unit)
li["std_amount" ] = li["qty"] * li["ref_unit" ]
# aggregate: sum_items (truth), std_total (utilization proxy), price_index
agg = li.groupby( "patient_id" ).agg(
    sum_items=( "amount" ,"sum"),
    std_total=( "std_amount" ,"sum"),
    n_items=( "amount" ,"count"),
    n_unique_codes=( "code","nunique" ),
).reset_index()
def compute_topshares (arr):
    x = np.asarray(arr, dtype= float)
    x = x[np.isfinite(x)]
    s = x. sum()
    if s <= 0 or len(x)==0:
        return pd.Series({ "top1_share" :0.0, "top3_share" :0.0})
    xs = np.sort(x)[::- 1]
    return pd.Series({ "top1_share" : float(xs[0]/s), "top3_share" : float(xs[:3].su
tops = li.groupby( "patient_id" )["amount" ].apply( lambda s: compute_topshares(s.val
tops = pd.concat([tops[[ "patient_id" ]], tops[ "amount" ].apply(pd.Series)], axis= 1)
feat_rcpt = agg.merge(tops, on= "patient_id" , how="left")

```


### Page 230

```text

feat_rcpt[ "distributed_cost" ] = 1.0 - feat_rcpt[ "top3_share" ].fillna( 0.0)
feat_rcpt[ "std_total" ] = feat_rcpt[ "std_total" ].replace( 0, np.nan).fillna( 1e-9)
feat_rcpt[ "price_index" ] = feat_rcpt[ "sum_items" ] / feat_rcpt[ "std_total" ]
all_df = all_df.merge(feat_rcpt, on= "patient_id" , how="left")
# Validate sum_items == prior in train
tr = all_df[all_df[ "_split" ]=="train"].copy()
diff = (tr[ "sum_items" ] - tr["prior_ed_cost_5y_usd" ]).abs()
match = float((diff <= TOL).mean())
log(f"[iter25] TRAIN sum_items vs prior match_rate@ {TOL} = {match:.4f} | max_abs_
# Admissions ( 少⽽硬 )
if ADM_TR.exists() and ADM_TE.exists():
    adm_tr = norm_cols(pd.read_csv(ADM_TR))
    adm_te = norm_cols(pd.read_csv(ADM_TE))
    adm = pd.concat([adm_tr, adm_te], axis= 0, ignore_index= True)
    g = adm.groupby( "patient_id" )
    out = pd.DataFrame({ "patient_id" : g.size().index.values})
    out[ "adm_n"] = g.size().values
    def add_agg(col, fn, name, default= 0.0):
        if col in adm.columns:
            out[name] = g[col].agg(fn).values
        else:
            out[name] = default
    add_agg( "los_days" ,"mean","adm_los_mean" )
    add_agg( "los_days" ,"max","adm_los_max" )
    add_agg( "charlson_band" ,"mean","adm_charlson_mean" )
    add_agg( "charlson_band" ,"max","adm_charlson_max" )
    add_agg( "acuity_emergent" ,"mean","adm_emergent_rate" )
    add_agg( "ed_visits_6m" ,"mean","adm_edvis6m_mean" )
    add_agg( "ed_visits_6m" ,"max","adm_edvis6m_max" )
    add_agg( "ed_visits_6m" ,"sum","adm_edvis6m_sum" )
    if "primary_dx"  in adm.columns:
        for dx in ["HF","DiabetesComp" ,"Pneumonia" ]:
            out[ f"adm_dx_ {dx}"] = g["primary_dx" ].apply( lambda s: int((s.astype( st
    else:

```


### Page 231

```text

        for dx in ["HF","DiabetesComp" ,"Pneumonia" ]:
            out[ f"adm_dx_ {dx}"] = 0.0
    all_df = all_df.merge(out, on= "patient_id" , how="left")
    log( "[iter25] admissions features added." )
else:
    log( "[iter25] admissions files not found -> skip." )
all_df = all_df.fillna( 0.0)
tr = all_df[all_df[ "_split" ]=="train"].copy()
y = tr["ed_cost_next3y_usd" ].astype( float).values
x = tr["prior_ed_cost_5y_usd" ].astype( float).values
# -----------------------------
# Plots
# -----------------------------
plt.figure()
plt.scatter(np.log1p(x), np.log1p(y), s= 8, alpha= 0.35)
plt.xlabel( "log1p(prior_ed_cost_5y_usd)" )
plt.ylabel( "log1p(ed_cost_next3y_usd)" )
plt.title( "Sublinear check: log1p(prior) vs log1p(future)" )
plt.tight_layout()
plt.savefig(OUT_DIR/ "scatter_log_prior_vs_log_future.png" , dpi=150)
plt.show()
tr["ratio_y_over_prior" ] = tr["ed_cost_next3y_usd" ] / (tr[ "prior_ed_cost_5y_usd" ] 
tr["prior_bin" ] = pd.qcut(tr[ "prior_ed_cost_5y_usd" ], q=10, duplicates= "drop")
ratio_by_bin = tr.groupby( "prior_bin" )["ratio_y_over_prior" ].mean()
plt.figure(figsize=( 8,4))
plt.plot( range(len(ratio_by_bin)), ratio_by_bin.values, marker= "o")
plt.xticks( range(len(ratio_by_bin)), [ str(i) for i in range(len(ratio_by_bin))])
plt.xlabel( "prior_cost decile bin (0..9)" )
plt.ylabel( "mean(y / prior_cost)" )
plt.title( "Sublinear evidence: ratio decreases with prior_cost" )
plt.tight_layout()
plt.savefig(OUT_DIR/ "ratio_by_prior_decile.png" , dpi=150)
plt.show()
plt.figure()

```


### Page 232

```text

plt.hist(tr[ "price_index" ].clip(0, tr["price_index" ].quantile( 0.995)), bins= 60)
plt.title( "price_index = prior_total / standardized_total (clipped 99.5%)" )
plt.xlabel( "price_index" )
plt.ylabel( "count")
plt.tight_layout()
plt.savefig(OUT_DIR/ "price_index_hist.png" , dpi=150)
plt.show()
# -----------------------------
# CV baselines
# -----------------------------
tr["grp"] = tr["primary_chronic" ].astype( str) + "|" + tr["insurance" ].astype( str)
strat = (tr[ "primary_chronic" ].astype( str) + "|" +
         pd.qcut(tr[ "prior_ed_cost_5y_usd" ].astype( float), q=10, duplicates= "drop
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle= True, random_state=SEED)
splits = list(skf.split(np.zeros( len(tr)), strat))
def oof_ratio_baseline (df, y, splits ):
    oof = np.zeros( len(df), dtype= float)
    for tr_idx, va_idx in splits:
        dtr = df.iloc[tr_idx]; dva = df.iloc[va_idx]
        ratio = y[tr_idx] / (dtr[ "prior_ed_cost_5y_usd" ].values + 1e-9)
        med = pd.Series(ratio).groupby(dtr[ "grp"].values).median().to_dict()
        gmed = float(np.median(ratio))
        pred = (dva[ "prior_ed_cost_5y_usd" ].values) * np.array([med.get(g, gmed) 
        oof[va_idx] = np.clip(pred, 0.0, None)
    return oof
def oof_log_power_baseline (df, y, splits, use_extra= False):
    oof = np.zeros( len(df), dtype= float)
    def build_X(d):
        cols = [
            np.ones( len(d)),
            np.log1p(d[ "prior_ed_cost_5y_usd" ].values),
            np.log1p(d[ "prior_ed_visits_5y" ].values),
        ]
        if use_extra:
            cols += [np.log1p(d[ "std_total" ].values), np.log1p(d[ "price_index" ].v
        return np.vstack(cols).T

```


### Page 233

```text

    for tr_idx, va_idx in splits:
        dtr = df.iloc[tr_idx]; dva = df.iloc[va_idx]
        ylog = np.log1p(y[tr_idx])
        Xg = build_X(dtr)
        rg = Ridge(alpha= 1.0, fit_intercept= False).fit(Xg, ylog)
        preds = np.zeros( len(dva), dtype= float)
        for g, idx_va_g in dva.groupby( "grp").groups.items():
            idx_va = np.array( list(idx_va_g), dtype= int)
            mask_tr = (dtr[ "grp"].values == g)
            Xva = build_X(dva.loc[idx_va])
            if mask_tr. sum() >= 30:
                r = Ridge(alpha= 1.0, fit_intercept= False).fit(build_X(dtr.loc[mas
                plog = r.predict(Xva)
            else:
                plog = rg.predict(Xva)
            preds[idx_va] = np.expm1(plog)
        oof[va_idx] = np.clip(preds, 0.0, None)
    return oof
def oof_isotonic_baseline (df, y, splits ):
    oof = np.zeros( len(df), dtype= float)
    for tr_idx, va_idx in splits:
        dtr = df.iloc[tr_idx]; dva = df.iloc[va_idx]
        xtr = np.log1p(dtr[ "prior_ed_cost_5y_usd" ].values)
        ytr = np.log1p(y[tr_idx])
        iso_g = IsotonicRegression(increasing= True, out_of_bounds= "clip").fit(xtr
        preds = np.zeros( len(dva), dtype= float)
        for g, idx_va_g in dva.groupby( "grp").groups.items():
            idx_va = np.array( list(idx_va_g), dtype= int)
            mask_tr = (dtr[ "grp"].values == g)
            xva = np.log1p(dva.loc[idx_va, "prior_ed_cost_5y_usd" ].values)
            if mask_tr. sum() >= 40:
                iso = IsotonicRegression(increasing= True, out_of_bounds= "clip").f
                    np.log1p(dtr.loc[mask_tr, "prior_ed_cost_5y_usd" ].values),
                    np.log1p(y[tr_idx][mask_tr])

```


### Page 234

```text

                )
                plog = iso.predict(xva)
            else:
                plog = iso_g.predict(xva)
            preds[idx_va] = np.expm1(plog)
        oof[va_idx] = np.clip(preds, 0.0, None)
    return oof
def oof_very_light_xgb (df, y, splits ):
    use_cols = [
        "age","prior_ed_visits_5y" ,"prior_ed_cost_5y_usd" ,
        "std_total" ,"price_index" ,
        "n_items" ,"n_unique_codes" ,"top1_share" ,"top3_share" ,"distributed_cost"
    ]
    for c in ["adm_n","adm_charlson_mean" ,"adm_charlson_max" ,"adm_emergent_rate" ,
              "adm_dx_HF" ,"adm_dx_DiabetesComp" ,"adm_dx_Pneumonia" ]:
        if c in df.columns:
            use_cols.append(c)
    X_num = df[use_cols].astype( float).values
    cat_cols = [ "primary_chronic" ,"insurance" ,"sex","zip3"]
    ohe = OneHotEncoder(handle_unknown= "ignore" , sparse_output= False)
    X_cat = ohe.fit_transform(df[cat_cols].astype( str))
    X = np.hstack([X_num, X_cat]).astype(np.float32)
    params = dict(
        n_estimators= 2500,
        learning_rate= 0.05,
        max_depth= 4,
        min_child_weight= 80,
        subsample= 0.8,
        colsample_bytree= 0.8,
        reg_lambda= 20.0,
        reg_alpha= 1.0,
        objective= "reg:absoluteerror" ,
        eval_metric= "mae",
        early_stopping_rounds= 150,
        tree_method= "hist",
        device= "cuda",
        n_jobs=- 1,

```


### Page 235

```text

        verbosity= 0,
    )
    oof = np.zeros( len(df), dtype= float)
    for fold, (tr_idx, va_idx) in enumerate (splits, 1):
        Xtr, Xva = X[tr_idx], X[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]
        m = xgb.XGBRegressor(**{**params, "random_state" : SEED+fold})
        try:
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose= False)
        except xgb.core.XGBoostError:
            p = dict(params); p[ "device" ] = "cpu"
            m = xgb.XGBRegressor(**{**p, "random_state" : SEED+fold})
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose= False)
        oof[va_idx] = np.clip(m.predict(Xva), 0.0, None)
    return oof
log("[iter25] Running baselines (5-fold CV)..." )
pred_ratio = oof_ratio_baseline(tr, y, splits)
pred_log   = oof_log_power_baseline(tr, y, splits, use_extra= False)
pred_log2  = oof_log_power_baseline(tr, y, splits, use_extra= True)
pred_iso   = oof_isotonic_baseline(tr, y, splits)
pred_xgb   = oof_very_light_xgb(tr, y, splits)
def report(name, pred ):
    return {
        "method" : name,
        "mae": mean_absolute_error(y, pred),
        "mae_top5%" : topk_mae(y, pred, q= 0.95),
        "mae_top1%" : topk_mae(y, pred, q= 0.99),
    }
summary = pd.DataFrame([
    report( "ratio_baseline (group median y/prior)" , pred_ratio),
    report( "log_power (log y ~ log prior + log visits)" , pred_log),
    report( "log_power + util/price (adds log std_total + log price_index)" , pred_
    report( "isotonic (per-group log prior -> log y)" , pred_iso),
    report( "very_light_xgb (core + util/price + few admissions)" , pred_xgb),
]).sort_values( "mae")

```


### Page 236

```text

你接下来怎么做（最省时间）
1. 直接跑上⾯这⼀个  cell （它会⾃动修好  receipts cache ）
2. 把最后打印的  baseline_comp arison 表格贴出来（ 5 ⾏）
3. 我们就能⽴刻判断专家的  Iter25 假设是否成⽴：
log_power  / isotonic  是否显著优于  ratio_baseline
log_power + util/price  是否进⼀步下降
very_light_xgb  是否在不爆炸的情况下更好
如果这⼀步验证通过，我们  Phase 3 的建模主⼲就会变成：
sublinear b aseline （ log/isot onic ）  + boosting 学  residual （并且⽤  util/price 拆解作为核
⼼结构特征） ，⽽不是继续在  65k hash 上硬卷。log("\n[iter25] CV baseline comparison (lower is better):" )
print(summary.to_string(index= False))
summary.to_csv(OUT_DIR/ "baseline_comparison.csv" , index= False)
log(f"\nSaved outputs to: {OUT_DIR} ")
log("Key plots: scatter_log_prior_vs_log_future.png, ratio_by_prior_decile.png, p
log("Key table: baseline_comparison.csv" )
Sour ces
Sour ces
-----------------------------------------------------------------
----------
KeyError                                  T raceback (most recent call  
last)
File 
c:\Users\shend\AppData\Local\Programs\Python\Python3
12\Lib\site-packages\pandas\core\indexes\base.py:3812,  
in Index.get_loc(self, key)
   3811 try:
-> 3812     return self._engine.get_loc(casted_key)
   3813 except K eyError as err:

```


### Page 237

```text

File pandas/_libs/index.pyx:167, in  
pandas._libs.index.IndexEngine.get_loc()
File pandas/_libs/index.pyx:196, in  
pandas._libs.index.IndexEngine.get_loc()
File pandas/_libs/hashtable_class_helper.pxi:7088, in  
pandas._libs.hashtable.PyObjectHashT able.get_item()
File pandas/_libs/hashtable_class_helper.pxi:7096, in  
pandas._libs.hashtable.PyObjectHashT able.get_item()
KeyError: 'top3_share'
The above exception was the direct cause of the following  
exception:
KeyError                                  T raceback (most recent call  
last)
Cell In[38], line 321
    318 tops = pd.concat([tops[["patient_id"]],  
tops["amount"].apply(pd.Series)], axis=1)
    320 feat_rcpt = agg.merge(tops, on="patient_id",  
how="left")
--> 321 feat_rcpt["distributed_cost"] = 1.0 -  
feat_rcpt["top3_share"].fillna(0.0)
    322 feat_rcpt["std_total"] =  
feat_rcpt["std_total"].replace(0, np.nan).fillna(1e-9)
    323 feat_rcpt["price_index"] = feat_rcpt["sum_items"] /  
feat_rcpt["std_total"]
File 
c:\Users\shend\AppData\Local\Programs\Python\Python3
12\Lib\site-packages\pandas\core\frame.py:4113, in  
DataFrame.__getitem__(self, key)
   4111 if self.columns.nlevels > 1:
   4112     return self._getitem_multilevel(key)
-> 4113 indexer = self.columns.get_loc(key)
   4114 if is_integer(indexer):
   4115     indexer = [indexer]

```


### Page 238

```text

File 
c:\Users\shend\AppData\Local\Programs\Python\Python3
12\Lib\site-packages\pandas\core\indexes\base.py:3819,  
in Index.get_loc(self, key)
   3814     if isinstance(casted_key, slice) or (
   3815         isinstance(casted_key, abc.Iterable)
   3816         and any(isinstance(x, slice) for x in casted_key)
   3817     ):
   3818         raise InvalidIndexError(key)
-> 3819     raise K eyError(key) from err
   3820 except T ypeError:
   3821     # If we have a listlike key, _check_indexing_error  
will raise
   3822     #  InvalidIndexError. Otherwise we fall through  
and re-raise
   3823     #  the T ypeError.
   3824     self._check_indexing_error(key)
KeyError: 'top3_share'
这个 KeyError: 'top3_share'  本质原因是：你那段  groupby().apply()  之后的 “展开 / 透
视”逻辑在你当前  pandas 版本下没有稳定⽣成名为  top1_share/top3_share  的列，所以
merge 后  feat_rcpt  ⾥根本没有  top3_share  这列。
最稳的修法： 不要⽤  pivot/lev el_1 那套展开 ，改成  pandas 官⽅推荐的  named
aggr egation（groupby().agg(top1_share=(...), top3_share=(...)) ），保证列名稳定。
pandas ⽂档明确⽀持这种写法。  
（你之前那种 “ 返回  Series 再展开 ” 的⽅式也能做，但不同  pandas 输出形状 / 列名很容易
变，正好踩坑了。 ）
下⾯给你⼀份  单 cell 的“Iter25 ED A （带  receipts cache ⾃动修复） ” 完整代码，唯⼀关键改
动就是  top1/t op3 shar e 的计算现在⽤  named aggregation ，绝不会再缺列。Pandas
Stack Overflow +1
python

```


### Page 239

```text

# ============================================================
# Iter25 EDA (ROBUST) — auto-fix receipts_parsed.joblib then run baselines
# FIX: compute top1_share/top3_share via named aggregation (stable column names)
# ============================================================
import os, re, json, time, math, warnings
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load, dump
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
warnings.filterwarnings( "ignore" )
np.set_printoptions(suppress= True)
# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = Path( r"D:\AgentDs\agent_ds_healthcare" )
CACHE_DIR = BASE_DIR / "cache_iter10"
CACHE_DIR.mkdir(parents= True, exist_ok= True)
TRAIN_CSV = BASE_DIR / "ed_cost_train.csv"
TEST_CSV  = BASE_DIR / "ed_cost_test.csv"
PATIENTS_CSV = BASE_DIR / "patients.csv"
PDF_DIR = BASE_DIR / "receipts_pdf"
RECEIPT_CACHE = CACHE_DIR / "receipts_parsed.joblib"

```


### Page 240

```text

# optional admissions
ADM_TR = BASE_DIR / "admissions_train.csv"
ADM_TE = BASE_DIR / "admissions_test.csv"
OUT_DIR = CACHE_DIR / "eda_iter25"
OUT_DIR.mkdir(parents= True, exist_ok= True)
SEED = 42
N_SPLITS = 5
TOL = 1e-2
# -----------------------------
# Utils
# -----------------------------
def log(msg):
    ts = datetime.now().strftime( "%H:%M:%S" )
    print(f"[{ts}] {msg}")
def norm_cols (df):
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df
def safe_float (x):
    if x is None:
        return np.nan
    if isinstance (x, (int,float,np.number)):
        return float(x)
    s = str(x).strip().replace( "$","").replace( ",","")
    try:
        return float(s)
    except Exception:
        return np.nan
def safe_int (x, default= 1):
    try:
        if x is None:
            return default
        if isinstance (x, (int, np.integer)):
            return int(x)

```


### Page 241

```text

        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return default
        return int(float(s))
    except Exception:
        return default
def topk_mae (y_true, y_pred, q= 0.95):
    thr = np.quantile(y_true, q)
    m = y_true >= thr
    return mean_absolute_error(y_true[m], y_pred[m]) if m.sum() else np.nan
# -----------------------------
# Receipt cache: validate / recover / rebuild
# -----------------------------
CODE_RE = re. compile(r"^[A-Z]?\d{4,5}$|^[A-Z0-9]{4,7}$" )
def validate_receipts (obj, expected_n, min_frac= 0.95):
    if not isinstance (obj, dict):
        return False, f"type={type(obj)} not dict"
    if len(obj) < int(min_frac * expected_n):
        return False, f"len={len(obj)} < {int(min_frac*expected_n)}"
    # spot check
    for k,v in obj.items():
        if not isinstance (v, dict):
            return False, f"value for key {k} is {type(v)} not dict"
        if "items" not in v or not isinstance (v["items"], list):
            return False, f"value for key {k} missing/invalid items"
        break
    return True, "ok"
def atomic_dump_joblib (obj, path: Path ):
    tmp = path.with_suffix( f".tmp_{int(time.time())}.joblib")
    dump(obj, tmp)
    os.replace(tmp, path)
def try_load_joblib (path: Path ):
    try:
        return load(path)
    except Exception:

```


### Page 242

```text

        return None
def recover_best_receipts (cache_dir: Path, expected_n: int):
    cands = list(cache_dir.glob( "receipts_parsed*.joblib" )) + list(cache_dir.glob
    cands = sorted(set(cands), key= lambda p: (p.stat().st_size, p.stat().st_mtime
    for p in cands:
        obj = try_load_joblib(p)
        if obj is None:
            continue
        ok, _ = validate_receipts(obj, expected_n)
        if ok:
            return obj, p
    return None, None
def parse_line_item (line: str):
    line = line.strip()
    if not line:
        return None
    m_amt = re.search( r"(\d[\d,]*\.\d{2})\s*$" , line)
    if not m_amt:
        return None
    amt = safe_float(m_amt.group( 1))
    left = line[:m_amt.start()].strip()
    m_qty = re.search( r"(\d+)\s*$" , left)
    if not m_qty:
        return None
    qty = int(m_qty.group( 1))
    left2 = left[:m_qty.start()].strip()
    if not left2:
        return None
    code = left2.split()[ 0].strip()
    if not CODE_RE. match(code):
        return None
    desc = left2[ len(code):].strip()
    return {"code": code, "desc": desc, "qty": qty, "amount" : float(amt)}
def rebuild_receipts_from_pdfs (pdf_dir: Path ):

```


### Page 243

```text

    import fitz
    from concurrent.futures import ThreadPoolExecutor, as_completed
    try:
        from tqdm.auto import tqdm
    except Exception:
        def tqdm(x, **kwargs ): return x
    pdfs = sorted(pdf_dir.glob( "receipt_*.pdf" ))
    pid_paths = []
    for p in pdfs:
        m = re.search( r"receipt_(\d+)\.pdf$" , p.name)
        if m:
            pid_paths.append(( int(m.group( 1)), p))
    def parse_one (pid, path: Path ):
        out = { "pdf_path" : str(path), "items": [], "text_len" : 0}
        try:
            with fitz.open(path) as doc:
                text = []
                for i in range(doc.page_count):
                    text.append(doc.load_page(i).get_text( "text", sort=True))
                text = "\n".join(text)
            out[ "text_len" ] = len(text)
            items = []
            for ln in text.splitlines():
                it = parse_line_item(ln)
                if it is not None:
                    items.append(it)
            out[ "items"] = items
            return pid, out
        except Exception as e:
            out[ "parse_error" ] = repr(e)
            return pid, out
    receipts = {}
    n_workers = max(4, (os.cpu_count() or 8) - 2)
    log( f"[receipts] rebuilding from PDFs: {len(pid_paths)} files | workers= {n_wo
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_workers) as ex:

```


### Page 244

```text

        futs = {ex.submit(parse_one, pid, path): pid for pid, path in pid_paths}
        for fut in tqdm(as_completed(futs), total= len(futs), desc= "Parsing PDFs" )
            pid, obj = fut.result()
            receipts[ int(pid)] = obj
    log( f"[receipts] rebuilt {len(receipts)} receipts in {time.time()-t0: .1f}s")
    return receipts
# -----------------------------
# Load train/test to know expected_n
# -----------------------------
train = norm_cols(pd.read_csv(TRAIN_CSV))
test  = norm_cols(pd.read_csv(TEST_CSV))
patients = norm_cols(pd.read_csv(PATIENTS_CSV))
train = train.merge(patients, on= "patient_id" , how="left")
test  = test.merge(patients, on= "patient_id" , how="left")
expected_n = len(train) + len(test)
# -----------------------------
# Ensure receipts cache is valid
# -----------------------------
receipts = None
src = None
if RECEIPT_CACHE.exists():
    obj = try_load_joblib(RECEIPT_CACHE)
    if obj is not None:
        ok, reason = validate_receipts(obj, expected_n)
        if ok:
            receipts = obj
            src = RECEIPT_CACHE
        else:
            log( f"[receipts] current cache INVALID: {reason} ")
if receipts is None:
    rec, p = recover_best_receipts(CACHE_DIR, expected_n)
    if rec is not None:
        receipts = rec
        src = p
        log( f"[receipts] recovered valid cache from: {p.name}  (len={len(receipts)
        atomic_dump_joblib(receipts, RECEIPT_CACHE)

```


### Page 245

```text

        log( f"[receipts] wrote recovered cache -> {RECEIPT_CACHE} ")
    else:
        receipts = rebuild_receipts_from_pdfs(PDF_DIR)
        ok, reason = validate_receipts(receipts, expected_n)
        log( f"[receipts] rebuilt validation: {ok} ({reason} )")
        atomic_dump_joblib(receipts, RECEIPT_CACHE)
        log( f"[receipts] saved rebuilt cache -> {RECEIPT_CACHE} ")
        src = RECEIPT_CACHE
log(f"[receipts] using cache source: {src} | type= {type(receipts)} | len= {len(rec
# ============================================================
# Iter25 EDA
# ============================================================
all_df = pd.concat([train.assign(_split= "train"), test.assign(_split= "test")], ax
all_ids = all_df[ "patient_id" ].astype( int).tolist()
# Build line-item table
rows = []
for pid in all_ids:
    obj = receipts.get( int(pid), { "items": []})
    items = obj.get( "items", []) if isinstance (obj, dict) else []
    for it in items:
        if not isinstance (it, dict):
            continue
        code = str(it.get( "code","")).strip()
        amt  = safe_float(it.get( "amount" , np.nan))
        qty  = safe_int(it.get( "qty", 1), default= 1)
        if not code or (not np.isfinite(amt)):
            continue
        qty = max(qty, 1)
        unit = float(amt) / qty
        rows.append(( int(pid), code, qty, float(amt), unit))
li = pd.DataFrame(rows, columns=[ "patient_id" ,"code","qty","amount" ,"unit_price" ]
log(f"[iter25] line_items rows= {len(li)} | unique_codes= {li['code'].nunique()}")
global_med_unit = float(li["unit_price" ].median())
ref_price = li.groupby( "code")["unit_price" ].median().to_dict()

```


### Page 246

```text

li["ref_unit" ] = li["code"].map(ref_price).fillna(global_med_unit)
li["std_amount" ] = li["qty"] * li["ref_unit" ]
# Aggregate totals
agg = li.groupby( "patient_id" ).agg(
    sum_items=( "amount" ,"sum"),
    std_total=( "std_amount" ,"sum"),
    n_items=( "amount" ,"count"),
    n_unique_codes=( "code","nunique" ),
).reset_index()
# -----------------------------
# FIXED: compute top shares with named aggregation (stable columns)
# -----------------------------
def _topk_share (s, k):
    v = pd.to_numeric(s, errors= "coerce" ).dropna().values.astype( float)
    tot = v. sum()
    if tot <= 0 or len(v) == 0:
        return 0.0
    v.sort()
    return float(v[-min(k, len(v)):].sum() / tot)
tops = li.groupby( "patient_id" ).agg(
    top1_share=( "amount" , lambda s: _topk_share(s, 1)),
    top3_share=( "amount" , lambda s: _topk_share(s, 3)),
).reset_index()
feat_rcpt = agg.merge(tops, on= "patient_id" , how="left")
feat_rcpt[ "distributed_cost" ] = 1.0 - feat_rcpt[ "top3_share" ].fillna( 0.0)
feat_rcpt[ "std_total" ] = feat_rcpt[ "std_total" ].replace( 0, np.nan).fillna( 1e-9)
feat_rcpt[ "price_index" ] = feat_rcpt[ "sum_items" ] / feat_rcpt[ "std_total" ]
all_df = all_df.merge(feat_rcpt, on= "patient_id" , how="left")
# Validate sum_items == prior in train
tr = all_df[all_df[ "_split" ]=="train"].copy()
diff = (tr[ "sum_items" ] - tr["prior_ed_cost_5y_usd" ]).abs()
match = float((diff <= TOL).mean())
log(f"[iter25] TRAIN sum_items vs prior match_rate@ {TOL} = {match:.4f} | max_abs_

```


### Page 247

```text

# Admissions ( 少⽽硬 )
if ADM_TR.exists() and ADM_TE.exists():
    adm_tr = norm_cols(pd.read_csv(ADM_TR))
    adm_te = norm_cols(pd.read_csv(ADM_TE))
    adm = pd.concat([adm_tr, adm_te], axis= 0, ignore_index= True)
    g = adm.groupby( "patient_id" )
    out = pd.DataFrame({ "patient_id" : g.size().index.values})
    out[ "adm_n"] = g.size().values
    def add_agg(col, fn, name, default= 0.0):
        out[name] = g[col].agg(fn).values if col in adm.columns else default
    add_agg( "los_days" ,"mean","adm_los_mean" )
    add_agg( "los_days" ,"max","adm_los_max" )
    add_agg( "charlson_band" ,"mean","adm_charlson_mean" )
    add_agg( "charlson_band" ,"max","adm_charlson_max" )
    add_agg( "acuity_emergent" ,"mean","adm_emergent_rate" )
    add_agg( "ed_visits_6m" ,"mean","adm_edvis6m_mean" )
    add_agg( "ed_visits_6m" ,"max","adm_edvis6m_max" )
    add_agg( "ed_visits_6m" ,"sum","adm_edvis6m_sum" )
    if "primary_dx"  in adm.columns:
        for dx in ["HF","DiabetesComp" ,"Pneumonia" ]:
            out[ f"adm_dx_ {dx}"] = g["primary_dx" ].apply( lambda s: int((s.astype( st
    else:
        for dx in ["HF","DiabetesComp" ,"Pneumonia" ]:
            out[ f"adm_dx_ {dx}"] = 0.0
    all_df = all_df.merge(out, on= "patient_id" , how="left")
    log( "[iter25] admissions features added." )
else:
    log( "[iter25] admissions files not found -> skip." )
all_df = all_df.fillna( 0.0)
tr = all_df[all_df[ "_split" ]=="train"].copy()
y = tr["ed_cost_next3y_usd" ].astype( float).values
x = tr["prior_ed_cost_5y_usd" ].astype( float).values
# -----------------------------

```


### Page 248

```text

# Plots
# -----------------------------
plt.figure()
plt.scatter(np.log1p(x), np.log1p(y), s= 8, alpha= 0.35)
plt.xlabel( "log1p(prior_ed_cost_5y_usd)" )
plt.ylabel( "log1p(ed_cost_next3y_usd)" )
plt.title( "Sublinear check: log1p(prior) vs log1p(future)" )
plt.tight_layout()
plt.savefig(OUT_DIR/ "scatter_log_prior_vs_log_future.png" , dpi=150)
plt.show()
tr["ratio_y_over_prior" ] = tr["ed_cost_next3y_usd" ] / (tr[ "prior_ed_cost_5y_usd" ] 
tr["prior_bin" ] = pd.qcut(tr[ "prior_ed_cost_5y_usd" ], q=10, duplicates= "drop")
ratio_by_bin = tr.groupby( "prior_bin" )["ratio_y_over_prior" ].mean()
plt.figure(figsize=( 8,4))
plt.plot( range(len(ratio_by_bin)), ratio_by_bin.values, marker= "o")
plt.xticks( range(len(ratio_by_bin)), [ str(i) for i in range(len(ratio_by_bin))])
plt.xlabel( "prior_cost decile bin (0..9)" )
plt.ylabel( "mean(y / prior_cost)" )
plt.title( "Sublinear evidence: ratio decreases with prior_cost" )
plt.tight_layout()
plt.savefig(OUT_DIR/ "ratio_by_prior_decile.png" , dpi=150)
plt.show()
plt.figure()
plt.hist(tr[ "price_index" ].clip(0, tr["price_index" ].quantile( 0.995)), bins= 60)
plt.title( "price_index = prior_total / standardized_total (clipped 99.5%)" )
plt.xlabel( "price_index" )
plt.ylabel( "count")
plt.tight_layout()
plt.savefig(OUT_DIR/ "price_index_hist.png" , dpi=150)
plt.show()
# -----------------------------
# CV baselines
# -----------------------------
tr["grp"] = tr["primary_chronic" ].astype( str) + "|" + tr["insurance" ].astype( str)
strat = (tr[ "primary_chronic" ].astype( str) + "|" +
         pd.qcut(tr[ "prior_ed_cost_5y_usd" ].astype( float), q=10, duplicates= "drop
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle= True, random_state=SEED)

```


### Page 249

```text

splits = list(skf.split(np.zeros( len(tr)), strat))
def oof_ratio_baseline (df, y, splits ):
    oof = np.zeros( len(df), dtype= float)
    for tr_idx, va_idx in splits:
        dtr = df.iloc[tr_idx]; dva = df.iloc[va_idx]
        ratio = y[tr_idx] / (dtr[ "prior_ed_cost_5y_usd" ].values + 1e-9)
        med = pd.Series(ratio).groupby(dtr[ "grp"].values).median().to_dict()
        gmed = float(np.median(ratio))
        pred = dva[ "prior_ed_cost_5y_usd" ].values * np.array([med.get(g, gmed) fo
        oof[va_idx] = np.clip(pred, 0.0, None)
    return oof
def oof_log_power_baseline (df, y, splits, use_extra= False):
    oof = np.zeros( len(df), dtype= float)
    def build_X(d):
        cols = [
            np.ones( len(d)),
            np.log1p(d[ "prior_ed_cost_5y_usd" ].values),
            np.log1p(d[ "prior_ed_visits_5y" ].values),
        ]
        if use_extra:
            cols += [np.log1p(d[ "std_total" ].values), np.log1p(d[ "price_index" ].v
        return np.vstack(cols).T
    for tr_idx, va_idx in splits:
        dtr = df.iloc[tr_idx]; dva = df.iloc[va_idx]
        ylog = np.log1p(y[tr_idx])
        rg = Ridge(alpha= 1.0, fit_intercept= False).fit(build_X(dtr), ylog)
        preds = np.zeros( len(dva), dtype= float)
        for g, idx_va_g in dva.groupby( "grp").groups.items():
            idx_va = np.array( list(idx_va_g), dtype= int)
            mask_tr = (dtr[ "grp"].values == g)
            Xva = build_X(dva.loc[idx_va])
            if mask_tr. sum() >= 30:
                r = Ridge(alpha= 1.0, fit_intercept= False).fit(build_X(dtr.loc[mas
                plog = r.predict(Xva)

```


### Page 250

```text

            else:
                plog = rg.predict(Xva)
            preds[idx_va] = np.expm1(plog)
        oof[va_idx] = np.clip(preds, 0.0, None)
    return oof
def oof_isotonic_baseline (df, y, splits ):
    oof = np.zeros( len(df), dtype= float)
    for tr_idx, va_idx in splits:
        dtr = df.iloc[tr_idx]; dva = df.iloc[va_idx]
        xtr = np.log1p(dtr[ "prior_ed_cost_5y_usd" ].values)
        ytr = np.log1p(y[tr_idx])
        iso_g = IsotonicRegression(increasing= True, out_of_bounds= "clip").fit(xtr
        preds = np.zeros( len(dva), dtype= float)
        for g, idx_va_g in dva.groupby( "grp").groups.items():
            idx_va = np.array( list(idx_va_g), dtype= int)
            mask_tr = (dtr[ "grp"].values == g)
            xva = np.log1p(dva.loc[idx_va, "prior_ed_cost_5y_usd" ].values)
            if mask_tr. sum() >= 40:
                iso = IsotonicRegression(increasing= True, out_of_bounds= "clip").f
                    np.log1p(dtr.loc[mask_tr, "prior_ed_cost_5y_usd" ].values),
                    np.log1p(y[tr_idx][mask_tr])
                )
                plog = iso.predict(xva)
            else:
                plog = iso_g.predict(xva)
            preds[idx_va] = np.expm1(plog)
        oof[va_idx] = np.clip(preds, 0.0, None)
    return oof
def oof_very_light_xgb (df, y, splits ):
    use_cols = [
        "age","prior_ed_visits_5y" ,"prior_ed_cost_5y_usd" ,
        "std_total" ,"price_index" ,
        "n_items" ,"n_unique_codes" ,"top1_share" ,"top3_share" ,"distributed_cost"
    ]
    for c in ["adm_n","adm_charlson_mean" ,"adm_charlson_max" ,"adm_emergent_rate" ,
              "adm_dx_HF" ,"adm_dx_DiabetesComp" ,"adm_dx_Pneumonia" ]:

```


### Page 251

```text

        if c in df.columns:
            use_cols.append(c)
    X_num = df[use_cols].astype( float).values
    cat_cols = [ "primary_chronic" ,"insurance" ,"sex","zip3"]
    ohe = OneHotEncoder(handle_unknown= "ignore" , sparse_output= False)
    X_cat = ohe.fit_transform(df[cat_cols].astype( str))
    X = np.hstack([X_num, X_cat]).astype(np.float32)
    params = dict(
        n_estimators= 2500,
        learning_rate= 0.05,
        max_depth= 4,
        min_child_weight= 80,
        subsample= 0.8,
        colsample_bytree= 0.8,
        reg_lambda= 20.0,
        reg_alpha= 1.0,
        objective= "reg:absoluteerror" ,
        eval_metric= "mae",
        early_stopping_rounds= 150,
        tree_method= "hist",
        device= "cuda",
        n_jobs=- 1,
        verbosity= 0,
    )
    oof = np.zeros( len(df), dtype= float)
    for fold, (tr_idx, va_idx) in enumerate (splits, 1):
        Xtr, Xva = X[tr_idx], X[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]
        m = xgb.XGBRegressor(**{**params, "random_state" : SEED+fold})
        try:
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose= False)
        except xgb.core.XGBoostError:
            p = dict(params); p[ "device" ] = "cpu"
            m = xgb.XGBRegressor(**{**p, "random_state" : SEED+fold})
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose= False)
        oof[va_idx] = np.clip(m.predict(Xva), 0.0, None)
    return oof

```


### Page 252

```text

跑完你把这  2 样东西贴给我就⾏
1. baseline_comparison  表格（ 5 ⾏）
2. 你观察到的三张图结论（尤其  ratio 是否随  prior decile 下降、 price_index 是否分散）
然后我就能判断 “sublinear baseline + util/price 拆解 ” 到底能不能把我们从  460 拉向  400 。log("[iter25] Running baselines (5-fold CV)..." )
pred_ratio = oof_ratio_baseline(tr, y, splits)
pred_log   = oof_log_power_baseline(tr, y, splits, use_extra= False)
pred_log2  = oof_log_power_baseline(tr, y, splits, use_extra= True)
pred_iso   = oof_isotonic_baseline(tr, y, splits)
pred_xgb   = oof_very_light_xgb(tr, y, splits)
def report(name, pred ):
    return {
        "method" : name,
        "mae": mean_absolute_error(y, pred),
        "mae_top5%" : topk_mae(y, pred, q= 0.95),
        "mae_top1%" : topk_mae(y, pred, q= 0.99),
    }
summary = pd.DataFrame([
    report( "ratio_baseline (group median y/prior)" , pred_ratio),
    report( "log_power (log y ~ log prior + log visits)" , pred_log),
    report( "log_power + util/price (adds log std_total + log price_index)" , pred_
    report( "isotonic (per-group log prior -> log y)" , pred_iso),
    report( "very_light_xgb (core + util/price + few admissions)" , pred_xgb),
]).sort_values( "mae")
log("\n[iter25] CV baseline comparison (lower is better):" )
print(summary.to_string(index= False))
summary.to_csv(OUT_DIR/ "baseline_comparison.csv" , index= False)
log(f"\nSaved outputs to: {OUT_DIR} ")
log("Key plots: scatter_log_prior_vs_log_future.png, ratio_by_prior_decile.png, p
log("Key table: baseline_comparison.csv" )

```


### Page 253

```text

Sour ces
Sour ces

```
