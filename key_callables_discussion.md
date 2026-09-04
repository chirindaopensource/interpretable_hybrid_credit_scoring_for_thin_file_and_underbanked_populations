## **Discussion of the Inputs, Processes and Outputs (IPOs) of the Final Orchestrator Callables**

Below is a discussion of the Inputs, Processes and Outputs (IPOs) of the **twenty-three final task-specific
and top-level orchestrator callables** in the "*Interpretable hybrid credit scoring for thin-file and
underbanked populations: A fairness-aware residual learning framework with evidence from East African
financial inclusion data*" end-to-end research pipeline (Kanziga, Gaba & Kanamugire, 2026;
arXiv:2608.26837).

**Scope note, stated for fidelity.** These twenty-three are the *task* orchestrators: the nineteen per-task
orchestrators of Task 1 → Task 19, the Task 20 orchestrator `run_full_study`, the Task 21 robustness
orchestrator, the Task 22 Taiwan continuity-benchmark orchestrator, and the top-level Task 23 wrapper
`run_research_pipeline`. The module-level import cell (cell 17) and the corpus preamble (cell 20, which
loads `study_config.json` and constructs the schema-conformant `zindi_raw`/`taiwan_raw` ingested frames) are
documented inside the entries that consume them; they are not counted as separate entries, but they are not
omitted either.

All signatures, equation numbers and LaTeX excerpts below were verified against the source module
`interpretable_hybrid_credit_scoring_for_thin_file_and_underbanked_populations_remediated.py`, the assessed
discussion in `key_callables_discussion.txt`, and `latex_context.txt` rather than recalled. Every entry
cites its workbook anchor (code cell and orchestrator definition line).

---

### Task 1 — `orchestrate_validate_ingested_parameters(config, zindi_raw, taiwan_raw, output_dir) → Dict[str, Any]`

*Workbook anchor: cell 21, L6404.*

**Inputs.** `config: Dict[str, Any]` — the frozen study configuration, the single source of truth for the
study; `zindi_raw: pd.DataFrame` — the raw Zindi FSD frame, 23,524 × 13, zero missing values, declared raw
column order, exhaustive categorical level sets (including the verbatim source typo `Divorced/Seperated`); `taiwan_raw: pd.DataFrame`
— the raw UCI Taiwan Credit Default frame, 30,000 × 25, columns `ID`, `LIMIT_BAL`, `SEX`, `EDUCATION`,
`MARRIAGE`, `AGE`, `PAY_0`, `PAY_2`…`PAY_6`, `BILL_AMT1`…`BILL_AMT6`, `PAY_AMT1`…`PAY_AMT6`,
`default.payment.next.month`; `output_dir: Path` — persistence root for the validation manifest.

**Process.** *Step 1 — configuration validation.* `validate_study_config` (helpers `_check`, `_get_nested`,
`_enforce_pin_type_contracts`) iterates a frozen tuple of required top-level keys, asserts each present,
recursively validates every nested block with declared value constraints — `math.isclose` for the
partition-fraction identity $\tau_{\text{train}} + \tau_{\text{val}} + \tau_{\text{test}} = 1.0$,
`isinstance` type contracts, set membership for the hyperparameter grids — and verifies that the
equation-parameter map contains every entry the downstream equations consume. *Step 2 — Zindi frame
validation.* `validate_zindi_raw` asserts `shape == (23524, 13)`, compares the ordered columns against
`raw_column_order` exactly (order mismatch with matching column set ⇒ reorder, never halt, and record the
reordering in the manifest), asserts `bank_account` is the integer label, checks the positive-class rate
$0.1408$ within $10^{-4}$ via `math.isclose`, checks zero missing values per column via `isnull().sum()`,
and verifies the country composition exactly (Kenya 6,068; Rwanda 8,735; Tanzania 6,620; Uganda 2,101).
*Step 3 — Taiwan frame validation.* `validate_taiwan_raw` asserts `shape == (30000, 25)`, ordered columns,
label `default.payment.next.month`, positive rate $\approx 0.22$ within $10^{-2}$, the UCI quirk that
`PAY_0` exists while `PAY_1` does not, and the exact categorical code sets of `SEX`, `EDUCATION`,
`MARRIAGE`, halting on any out-of-set code while preserving declared unknown codes (`EDUCATION ∈ {0, 5, 6}`, `MARRIAGE = 0`).
*Step 4 — composition.* The three sub-reports are fused into one structured validation report, persisted as
a JSON manifest in `output_dir`; the first failed check raises (fail-fast).

**Transformation.** Three untrusted ingested parameters → one validated contract report plus a persisted
JSON manifest; the Zindi frame, if its column order drifted, is returned reordered with the reordering
recorded. Every later stage may trust shape, dtype, level-set and rate invariants without re-verifying them.

**Role.** The research gate for the Section 4.1 invariants — *"The labelled training set contains 23,524
respondents spanning survey years 2016–2018 (Rwanda 8,735; Tanzania 6,620; Kenya 6,068; Uganda 2,101), with
a marginal positive-class rate of 14.08% (account ownership)"* — and for the benchmark facts of Section 4.3
(30,000 card holders, 23 features, ≈22% positives). It enforces the study's reporting discipline ("Logging
every random seed, every hyperparameter selection, every environment version, and every deviation from the
stated protocol") at the boundary of the pipeline, where a wrong assumption is cheapest to catch, and it
certifies the config rule that "where the manuscript does not specify a value, the config supplies a
rational placeholder and marks it explicitly as UNSPECIFIED IN MANUSCRIPT".

<br>

### Task 2 — `orchestrate_cleanse_data(config, zindi_raw, taiwan_raw, output_dir) → Dict[str, Any]`

*Workbook anchor: cell 22, L6919.*

**Inputs.** The certified `zindi_raw` and `taiwan_raw` frames from Task 1 (no re-validation inside the
cleanser); the same `config`; the same `output_dir`.

**Process.** *Step 1 — non-predictive column removal.* `drop_non_predictive_columns` uses named-column
access exclusively (`DataFrame.drop(axis=1, columns=[...])`): Zindi loses `uniqueid` and `year`, Taiwan
loses `ID`; the remaining Zindi column set is asserted equal to the ten features plus `bank_account`, and
Taiwan to 24 columns; original row order is preserved. *Step 2 — categorical standardisation.*
`standardize_categorical_values` applies `Series.str.strip()` to every object-dtype column — internal case,
internal spacing and the verbatim typo `Divorced/Seperated` are untouched — then re-validates every unique
value against `study_config["dataset_parameters"]["zindi_fsd"]["categorical_levels"]`; an unseen level is
logged to the `unseen_levels` registry, with the rule: halt if it appears in training (it contradicts the
fixed-level claim), preserve it if only in validation/test (the neutral-WoE rule of Section 5.1 governs it
there). *Step 3 — dtype contracts and audit.* `enforce_dtype_contracts` casts `household_size`,
`age_of_respondent`, `bank_account` (Zindi) and the Taiwan features plus label to `int64` and asserts the
dtypes; `_frame_digest` computes SHA-256 digests of the original and cleansed frames;
`_build_level_validation` assembles the level-validation result; the audit record — dropped columns,
level-validation results, both hash pairs, timestamp — is persisted *before any fitting occurs*.

**Transformation.** Validated raw frames → cleansed frames (Zindi 11 columns; Taiwan 24 columns) + a
tamper-evident audit record + the `unseen_levels` registry. The hash pair is the integrity witness: any
later mutation of a cleansed frame breaks the recorded chain.

**Role.** Implements the Section 4.1 drop decisions verbatim — *"The year column (2016–2018) and the
uniqueid identifier are dropped prior to fitting: year because it is perfectly collinear with country in
this sample (each survey year covers one country), and uniqueid because it is a unique-per-row identifier
with no predictive content"* — and the Section 4.3 inheritance rule for Taiwan. The audit record is the
reporting-discipline instrument: the cleansing stage is the first place the pipeline could silently diverge
from the manuscript's data contract, so it is the first place that is hash-sealed.

<br>

### Task 3 — `orchestrate_create_partitions(config, zindi_cleansed, taiwan_cleansed, unseen_levels, output_dir) → Dict[str, Any]`

*Workbook anchor: cell 23, L7356.*

**Inputs.** The two cleansed frames; the `unseen_levels` registry carried forward; `config`; `output_dir`.

**Process.** *Step 1 — the stratified splitter.* `implement_stratified_splitter` chains two
`sklearn.model_selection.StratifiedShuffleSplit` applications: first `n_splits=1, test_size=0.4, random_state=42`
→ 60% training block + 40% holdout; then `test_size=0.5` on the holdout (same seed) → 20% validation + 20%
test. The chained design is declared a rational placeholder (the manuscript fixes the 60/20/20 proportions,
not the algorithm) and is recorded in the manifest. *Step 2 — Zindi integrity.* `split_zindi_and_verify`
asserts $\lvert\text{train}\rvert = 14{,}114$ and $\lvert\text{val}\rvert, \lvert\text{test}\rvert = 4{,}705
\pm 1$, and the test positive rate $= 0.1407$ within $10^{-3}$, logging actual values whenever the $\pm 1$
rounding tolerance engages. *Step 3 — Taiwan integrity.* `split_taiwan_and_verify` applies the identical
splitter and seed, records all three sizes and positive rates, and documents the expected drift: Taiwan's
published numbers come from an inherited pipeline with an unspecified seed.

**Transformation.** Cleansed rows → three disjoint stratified index arrays per dataset, frozen in two
persisted partition manifests (sizes, rates, seed, algorithm). From this point the test partition is
read-only by contract, and the manifests — not row order — are the authoritative partition definition for
every downstream stage.

**Role.** Implements the protocol guarantee of Section 5.3: *"To prevent information leakage, all
preprocessing, hyperparameter tuning, residual-learning, and Platt-calibration stages are performed
exclusively on the training and validation partitions. The test partition is reserved strictly for final
evaluation."* Freezing the boundaries here is what makes that sentence enforceable downstream; and the seed
(42) plus the split algorithm are the first two entries of the paper's own fidelity-diagnosis checklist when
a reproduced number drifts.

<br>

### Task 4 — `orchestrate_fit_preprocessing_stats(config, zindi_frame, zindi_manifest, taiwan_frame, taiwan_manifest, output_dir) → Dict[str, Any]`

*Workbook anchor: cell 24, L7738.*

**Inputs.** The cleansed frames with their frozen partition manifests (the manifests supply the training-row
index sets; only those rows may contribute statistics); `config`; `output_dir`.

**Process.** *Step 1 — feature-set derivation.* `derive_taiwan_feature_sets` establishes the Taiwan
continuous/categorical block split the placeholder protocol requires (the thesis pipeline is not restated in
the paper; the split is declared and flagged UNSPECIFIED IN MANUSCRIPT). *Step 2 — imputation statistics.*
`compute_imputation_stats` computes the median of the numerical features and the mode of every categorical
feature via `Series.mode().iloc[0]` (first sorted mode, for determinism), on the training partition only;
the statistics are stored even though Zindi has zero missing values, for interface uniformity and for
Taiwan. *Step 3 — winsorization statistics.* `compute_winsorization_stats` computes the 1st and 99th
percentiles of the numerical features via `numpy.percentile` on the training partition only, records the
exact percentile values in the audit log, and applies the clipping bounds unrounded — a percentile of a
discrete integer column may be an unobserved boundary value, and the raw value, not a rounded one, is
frozen. *Step 4 — frozen application.* `apply_frozen_stats` (via `_fit_dataset_stats`) fills missing values
with the frozen medians/modes and clips the numerical features at the frozen bounds, identically on
training, validation and test; every Taiwan placeholder decision is logged and flagged; the frozen
`preprocessing_stats` dictionary is persisted to JSON.

**Transformation.** Clean partitions → partitions whose missing cells and tails are governed exclusively by
training-learned constants, plus the immutable `preprocessing_stats` block (medians, modes, `winsor_bounds`)
that every later transformation consumes. Validation and test rows are transformed by statistics they never
contributed to.

**Role.** Implements Section 5.1 verbatim: *"Missing values are imputed by median (numerical) and mode
(categorical) on the training partition only, and the imputation values are then applied to validation and
test"*; *"Numerical features are winsorized at the 1st and 99th percentiles of the training partition"*. The
training-only freeze is the leakage-prevention contract in its purest form — *"the discipline of computing
and freezing every transformation statistic on the training partition only, to guarantee that no information
from validation or test partitions enters model estimation"* — and the exact-value audit log is what makes
later drift diagnosable rather than mysterious.

<br>

### Task 5 — `orchestrate_fit_woe_encoding(config, zindi_train_df, zindi_y, zindi_frames, zindi_logistic_features, taiwan_train_df, taiwan_y, taiwan_frames, taiwan_logistic_features, taiwan_continuous, taiwan_categorical, output_dir) → Dict[str, Any]`

*Workbook anchor: cell 25, L8654.*

**Inputs.** Per dataset: the training partition frame, its label vector, the full partition-frame
dictionary, the ordered logistic-branch feature tuple; for Taiwan additionally the explicit continuous and
categorical block tuples; `config`; `output_dir`.

**Process.** *Step 1 — monotonic binning of the continuous features.* `fit_monotonic_bins` starts from 20
quantile bins on the training partition (`_qcut_edges`, `_bins_from_edges`), computes each bin's WoE via
`woe_from_counts` under Equation (2) with the pinned class convention ($g$ counts $y=0$, $b$ counts $y=1$,
$G, B$ the training totals), and enforces monotonicity with `_enforce_monotonicity`: whenever the WoE
sequence changes direction, the adjacent pair with the closest WoE values is merged (`_merge_bins`, test
`_is_monotone`), repeated until monotone; `_enforce_min_frequency` then enforces a minimum bin frequency of
5% of training observations; `_interval_label` and `assert_interval_key_uniqueness` keep bin keys stable and
unique. *Step 2 — WoE scores.* `compute_woe_scores` recomputes $g_k, b_k, G, B$ under the frozen boundaries
on training and forms the final WoE for every bin and every declared categorical level via Equation (2); a
zero count in either class takes the documented neutral WoE = 0 treatment; `_assert_woe_key_coverage` checks
key coverage before freezing. *Step 3 — frozen application.* `apply_woe_mapping` bins each validation/test
observation by the frozen cut points (continuous) or raw level (categorical) and replaces it with the
training-learned WoE; any level or bin unseen in training maps to neutral 0.0 and is logged; identical
column ordering across the three matrices is asserted before persistence.

**Transformation.** Raw numeric and categorical columns → monotone-risk WoE scores. Each bin or level $k$ is
replaced by the training-frozen value

$$\mathrm{WoE}_k = \ln\!\left(\frac{g_k/G}{b_k/B}\right), \tag{2}$$

so the logistic branch receives a matrix whose every column is a monotone log-odds contrast against the
training population.

**Role.** Implements the scorecard encoding of Section 3.1 — *"WoE-encoded features and L2 regularization on
β together preserve the auditable per-feature decomposition that regulators require of a scorecard"* — under
Section 5.1's leakage rules: *"Rare WoE values encountered in held-out folds that do not appear in training
are assigned a neutral WoE = 0, preserving the log-odds-zero contribution and preventing information
leakage."* The monotonicity enforcement is the industry convention (Siddiqi 2012; Thomas 2017) that makes
the coefficient table a risk ladder rather than an arbitrary contrast set.

<br>

### Task 6 — `orchestrate_iv_filter_and_ohe(config, train_df, y, woe_map, cut_points, woe_matrices, partition_frames, categorical_features, continuous_features, output_dir, verify_manuscript_targets) → Dict[str, Any]`

*Workbook anchor: cell 26, L9173.*

**Inputs.** The training partition frame and labels; the frozen WoE artefacts from Task 5; the WoE matrices
of all three partitions; the raw partition frames; the categorical and continuous feature tuples;
`output_dir`; the manuscript-target verification flag. (Invoked once per dataset — Zindi and Taiwan.)

**Process.** *Step 1 — Information Value and the retention filter.* `compute_information_values` computes,
per feature on the training partition,

$$\mathrm{IV} = \sum_k \left(\frac{g_k}{G} - \frac{b_k}{B}\right)\mathrm{WoE}_k,$$

from the frozen WoE values exactly; `apply_iv_filter` retains features with $\mathrm{IV} \ge 0.02$ and
filters the WoE matrices to the retained columns; `assert_zindi_iv_targets` (when
`verify_manuscript_targets`) asserts that all ten Zindi features pass and that the three strongest IVs match
the manuscript — `education_level` ≈ 0.97, `cellphone_access` ≈ 0.71, `job_type` ≈ 0.63 — within absolute
tolerance 0.02. *Step 2 — one-hot fitting.* `fit_one_hot_encoding` expands every categorical feature into
indicator columns via `pandas.get_dummies(drop_first=False)` on training, so every training-observed level
receives a column; the sorted unique levels per feature are frozen; continuous features are appended
unchanged (rare levels are accepted, not collapsed — the manuscript specifies no rare-level handling). *Step
3 — frozen application.* `apply_one_hot_encoding` applies `get_dummies` to validation and test, reindexes
against the frozen training column list with zero fill (a value absent from the frozen vocabulary yields an
all-zero block), and asserts `X_ohe_val.columns.equals(X_ohe_train.columns)` and likewise for test. The IV
table, retained feature set and frozen column list are persisted.

**Transformation.** Raw categorical levels → a dense indicator design aligned to a frozen training
vocabulary: $X_{\mathrm{ohe}}^{\mathrm{train/val/test}}$ with identical column ordering, while the logistic
branch continues on the WoE matrix filtered to the retained set — the two-representation split the
manuscript prescribes.

**Role.** Implements Section 5.1's feature economy verbatim: *"For Zindi specifically, all ten input
features (country, location_type, cellphone_access, gender_of_respondent, relationship_with_head,
marital_status, education_level, job_type, household_size, age_of_respondent) survive the IV ≥ 0.02 filter;
the strongest predictor by IV is education_level (IV = 0.97), followed by cellphone_access (0.71) and
job_type (0.63)"* — the three-value assertion is this orchestrator's precision check. It also implements the
independent representation rule: *"for the residual learner, categorical features are one-hot encoded, which
permits the boosting model to capture nonlinear interactions among categorical levels."* The IV filter binds
the logistic branch's feature economy; the boosting branch keeps the full set.

<br>


### Task 7 — `orchestrate_fit_logistic(config, X_woe_train, y_train, X_woe_val, y_val, X_woe_test, y_test, output_dir, feature_names) → Dict[str, Any]`

*Workbook anchor: cell 27, L9600.*

**Inputs.** The WoE design matrices of all three partitions with aligned label vectors; the ordered
`feature_names` of the WoE matrix (the coefficient vector must later attach to feature identities for
adverse-action explanations); `config`; `output_dir`.

**Process.** *Step 1 — C selection.* `select_C_cv` runs `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`;
for each $C \in \{0.01, 0.1, 1, 10\}$ it fits `LogisticRegression(solver='lbfgs', penalty='l2', class_weight='balanced', C=C, max_iter=1000)`
on the fold's training rows, scores AUC on the fold's validation rows, and selects the argmax mean; `lbfgs`
non-convergence at extreme $C$ is absorbed by `max_iter` and logged. *Step 2 — the final fit.*
`fit_final_logistic` refits the same estimator on the full training partition at the selected $C$ and
extracts $\hat{\beta}$ and $\hat{\beta}_0$, persisted for adverse-action explanations. *Step 3 — probability
generation.* `predict_lr_partitions` applies `predict_proba` to all three matrices, forming

$$p_{\mathrm{LR}}(x) = \sigma\!\left(\hat{\beta}^{\top} x + \hat{\beta}_0\right) = \frac{1}{1 + \exp\!\left(-(\hat{\beta}^{\top} x + \hat{\beta}_0)\right)}, \tag{1}$$

records the validation AUC for comparison against the manuscript's test value ($\approx 0.854$), and asserts
every probability lies in the open interval $(0, 1)$.

**Transformation.** WoE matrix → probability vector through the sigmoid link on the linear predictor. The
same pass yields the explanation object $(\hat{\beta}, \hat{\beta}_0, \text{feature names})$, so the
interpretable branch produces its score and its coefficient table simultaneously.

**Role.** Implements the baseline of Section 3.1 — *"The logistic regression baseline estimates
$p_{\mathrm{LR}}(x_i) = \sigma(\beta^\top x_i + \beta_0)$ … with $\beta \in \mathbb{R}^p$ fitted by maximum
likelihood under balanced class weighting"* — under the Table 1 configuration (`lbfgs`, L2, balanced, $C$
grid $\{0.01, 0.1, 1, 10\}$). The balanced weighting is kept exactly as specified; its probability-scale
consequence (the raw LR's $\mathrm{ECE} = 0.234$ on the imbalanced Zindi target) is deliberately left
uncorrected here — calibration belongs to Task 13, and the paper's ablation design requires the uncalibrated
baseline to be measured as it is.

<br>

### Task 8 — `orchestrate_compute_residuals(labels, p_lr) → Dict[str, Any]`

*Workbook anchor: cell 28, L9907.*

**Inputs.** Two dictionaries keyed by partition (train/val/test): the binary label vectors and the Task 7
probability vectors. No configuration enters; the step is arithmetic plus validation.

**Process.** *Step 1 — training residuals.* `compute_training_residuals` forms, per training observation,

$$r_i = y_i - p_{\mathrm{LR}}(x_i), \tag{3}$$

the regression target of the residual learner. *Step 2 — validation and test residuals.*
`compute_partition_residuals` applies the identical formula to validation and test; these are read-only
artefacts, retained for evaluation and for the alternating-minimization procedure, never used to fit the
booster. *Step 3 — distribution validation.* `validate_residual_distribution` asserts $r_i \in [-1, 1]$ (a
probability minus a binary label is bounded), computes mean, standard deviation, minimum, maximum, and halts
if the training standard deviation falls below $10^{-3}$ — a near-constant LR would make residual learning
meaningless — logging the summary into the run manifest.

**Transformation.** Label and probability vectors → signed, bounded residual vectors
$r^{\mathrm{train/val/test}}$, positionally aligned to the partitions; the training vector becomes the
boosting branch's regression target, the other two are evaluation-only artefacts.

**Role.** Implements Equation (3) and Section 3.2's residual-definition argument verbatim: *"Pearson or
deviance residuals are more standard for GLM diagnostics, but the raw response residual is adopted here
because it provides a direct regression target for the secondary learner and aligns with the
residual-learning paradigm popularized for deep networks by He et al."* The orchestrator enforces the
algebraic implication of that choice — the residuals are heteroscedastic by construction, hence the
squared-error objective in Task 9 — and the variance floor protects the residual-learning premise from a
degenerate baseline.

<br>

### Task 9 — `orchestrate_fit_residual_xgboost(config, X_ohe_matrices, residuals, p_lr, labels, lr_only_val_auc, output_dir) → Dict[str, Any]`

*Workbook anchor: cell 29, L10313.*

**Inputs.** The one-hot matrices of all three partitions; the three residual vectors; the LR probability and
label vectors (the provisional-hybrid scoring used by early stopping); the LR-only validation AUC (the
contribution-check baseline); `config`; `output_dir`.

**Process.** *Step 1 — configure and fit.* `fit_residual_booster` builds `XGBRegressor(objective='reg:squarederror', n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, reg_lambda=2)`
and trains it on $X_{\mathrm{ohe}}^{\mathrm{train}}$ against $r^{\mathrm{train}}$ under the custom
`_HybridAucEarlyStopping` monitor: if the config declares early stopping, training halts after `patience=50`
rounds without improvement in the validation AUC of the provisional hybrid $p_{\mathrm{LR}}(x) +
\hat{r}(x)$; otherwise the fixed 400 estimators are used. The paper's ranges (300–500 trees, depth 3–4, rate
0.03–0.05) resolve to the config placeholders, flagged UNSPECIFIED IN MANUSCRIPT. *Step 2 — predictions at
the frozen iteration.* `predict_residuals` predicts on all three matrices at the frozen best iteration,
records the actual tree count, and asserts no NaN in any vector. *Step 3 — contribution validation.*
`validate_residual_contribution` computes the Pearson correlation between $r$ and $\hat{r}$ on training
(`scipy.stats.pearsonr`), asserts it positive, and records the provisional hybrid validation AUC against
`lr_only_val_auc`.

**Transformation.** One-hot features → residual-score vector

$$\hat{r}(x) = f_{\mathrm{GBM}}(x) = \sum_{m=1}^{M} \nu\, h_m(x), \tag{4}$$

the nonlinear correction term of the hybrid — with $h_m$ the regression tree fitted at iteration $m$ and
$\nu$ the learning rate.

**Role.** Implements Section 3.2 under the Section 5.2 configuration, whose design rationale is
load-bearing: *"The shallow tree depths (3–4) for the residual learner are deliberate: deeper trees allow
the boosting component to dominate the prediction and erode the interpretability ratio. Low learning rates
(0.03–0.05) encourage gradual residual correction rather than aggressive nonlinear replacement of the
logistic baseline."* The hybrid-AUC early-stopping target operationalises that sentence: the model is
selected on the quantity the paper cares about — the combined prediction — not on the residual MSE alone.

<br>

### Task 10 — `orchestrate_fit_standalone_xgboost(config, X_ohe_matrices, labels, output_dir, verify_manuscript_targets) → Dict[str, Any]`

*Workbook anchor: cell 30, L10663.*

**Inputs.** The one-hot matrices of all three partitions with aligned binary labels; the config's standalone
hyperparameters; the verification flag; `output_dir`.

**Process.** *Step 1 — configuration.* `configure_standalone` builds `XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, reg_lambda=2, objective='binary:logistic', eval_metric='auc')`
and computes $\text{scale\_pos\_weight} = n_{\text{neg}}/n_{\text{pos}}$ on the training partition only,
recording the exact value. *Step 2 — fit and predict.* `fit_and_predict` fits on
$X_{\mathrm{ohe}}^{\mathrm{train}}$ with the aligned labels and produces probability scores for all three
partitions; no post-hoc calibration is applied to the standalone model. *Step 3 — benchmark recording.*
`record_benchmark` computes test AUC (`roc_auc_score`), Brier (`brier_score_loss`) and F1 at 0.5
(`precision_recall_fscore_support`), and — when the flag is set — verifies the manuscript targets (AUC
$\approx 0.873$, Brier $\approx 0.143$, F1 $\approx 0.513$) within $0.01$, invoking the documented
fidelity-diagnosis order on deviation.

**Transformation.** One-hot features + binary labels → a pure-boosting probability score. This model is the
discrimination ceiling: it never sees the linear branch, so its test metrics define what the hybrid must
match while adding calibration and interpretability.

**Role.** Implements the Table 2 comparator row (Zindi XGBoost AUC 0.873, Brier 0.143, F1 0.513) and Section
5.2's imbalance strategy — `scale_pos_weight` "is recomputed from the Zindi class balance". Its research
role is to separate the two claims the paper makes about the hybrid: matching the forest on rank-ordering
($\Delta\mathrm{AUC} = -0.004$, 95% CI $[-0.009, +0.001]$, $p = 0.92$) while adding the Platt-calibration
gain and the per-prediction interpretability decomposition that a pure forest cannot offer.

<br>

### Task 11 — `orchestrate_select_adaptive_weights(config, X_woe_train, y_train, X_ohe_train, selected_C, p_lr, output_dir) → Dict[str, Any]`

*Workbook anchor: cell 31, L11118.*

**Inputs.** The WoE and one-hot training matrices with training labels; the Task 7 selected $C$; the
final-model LR probability vectors of all partitions; `config`; `output_dir`.

**Process.** *Step 1 — the uncertainty proxy and the weight schedule.* `compute_uncertainty` and
`compute_adaptive_weight` implement

$$u_{\mathrm{LR}}(x) = 4 \cdot p_{\mathrm{LR}}(x)\,(1 - p_{\mathrm{LR}}(x)) \in [0, 1], \tag{8}$$

$$\alpha^{*}(x) = \alpha_{\min} + (\alpha_{\max} - \alpha_{\min})\, u_{\mathrm{LR}}(x)^{\gamma}, \tag{9}$$

with $\gamma > 0$ controlling concavity and the output verified bounded. *Step 2 — grid search.*
`grid_search_adaptive_weights` enumerates the 27 triples ($\alpha_{\min} \in \{0.0, 0.1, 0.2\}$,
$\alpha_{\max} \in \{0.3, 0.5, 0.8\}$, $\gamma \in \{0.5, 1.0, 2.0\}$) and runs 5-fold stratified CV per
triple: per fold it fits the LR, computes fold-training residuals, fits the residual booster on them,
computes fold-validation $u_{\mathrm{LR}}$ and $\alpha^{*}$, forms the Equation (5) hybrid and scores
validation AUC; the argmax triple is selected. *Step 3 — freezing.* `freeze_adaptive_weights` recomputes
$u_{\mathrm{LR}}$ and $\alpha^{*}$ for every observation of all three partitions from the final
full-training LR probabilities — never from fold models — asserts non-negativity and boundedness, and
persists the triple and the per-observation arrays with the full CV trace.

**Transformation.** Final LR probabilities → per-observation local-uncertainty weights that scale the
residual correction before it enters the hybrid sum; a three-dimensional grid choice collapses to one
selected triple with a complete selection record.

**Role.** Implements Section 3.4 — *"The weight $\alpha^{*}(x)$ in Eq. (5) is allowed to vary with the local
uncertainty of the logistic baseline"* — under the Section 5.2 grids. This is the mechanism that lets the
residual branch contribute most where the baseline is most uncertain, and the frozen $\alpha^{*}$ is what
makes the interpretability ratio of Equation (6) a per-prediction quantity rather than a global constant.

> **One registered deviation, disclosed rather than buried.** Section 5.2 states that "static and adaptive
> weighting hyperparameters are selected via 5-fold and 3-fold stratified cross-validation respectively", in
> tension with Section 3.4's 5-fold statement. The implementation standardises on 5-fold for the adaptive grid
> and logs the resolution; the choice is the more conservative of the two readings (more folds, smaller
> validation slices, identical seed discipline).

<br>

### Task 12 — `orchestrate_build_hybrid_and_rho(config, p_lr, r_hat, alpha_star, labels, output_dir, verify_manuscript_targets) → Dict[str, Any]`

*Workbook anchor: cell 32, L11499.*

**Inputs.** The three partition dictionaries of LR probabilities, residual predictions and frozen adaptive
weights; the label vectors; `config`; `output_dir`; the verification flag.

**Process.** *Step 1 — hybrid probabilities.* `compute_hybrid_probabilities` forms

$$p_{\mathrm{Hybrid}}(x) = \mathrm{clip}\!\left(p_{\mathrm{LR}}(x) + \alpha^{*}(x)\,\hat{r}(x),\, 0,\, 1\right), \tag{5}$$

via `numpy.clip` with NaN and range assertions, storing the *unclipped* additive components for Step 2.
*Step 2 — ratio and regions.* `compute_rho_and_regions` computes, from the unclipped components,

$$\rho(x) = \frac{|p_{\mathrm{LR}}(x)|}{|p_{\mathrm{LR}}(x)| + |\alpha^{*}(x)\,\hat{r}(x)|} \in [0, 1], \tag{6}$$

with the zero-denominator guard $\rho := 1$ when the denominator falls below $10^{-12}$ (theoretically
impossible, numerically reachable), and assigns the three regions,

$$\mathrm{Region}(x) = \begin{cases} \text{Fully interpretable} & \rho(x) \ge 0.9,\\ \text{Partially interpretable} & 0.7 \le \rho(x) < 0.9,\\ \text{ML-driven} & \rho(x) < 0.7; \end{cases} \tag{7}$$

regions are computed on the uncalibrated scale — Platt calibration (Task 13) changes only the probability
scale, never $\rho$ or membership. *Step 3 — test aggregation.* `aggregate_region_stats` computes per-region
shares, within-region positive-class rates, mean $\rho$ per region and the full $\rho$ distribution,
verifying Table 3's Zindi values (shares 6.4% / 34.1% / 59.4%; positive rates 53.8% / 19.7% / 6.5%; each
within $\pm 1$ percentage point) and asserting the bimodality of the $\rho$ distribution (primary mode below
0.7, secondary mode near 1.0) when the flag is set.

**Transformation.** Three per-observation vectors → one clipped hybrid score, one auditability scalar
$\rho(x) \in [0,1]$, and one three-level region label, per observation. The unclipped components are
retained because $\rho$ is a function of the pre-clip additive parts; clipping would distort the ratio at
the boundary.

**Role.** Implements the paper's second methodological pillar — Equations (5), (6), (7) — together with
Section 3.3's central interpretive caution, stated nearly verbatim there: *"the ratio is defined on the
probability scale rather than the logit scale, so borrowers for whom $p_{\mathrm{LR}}(x)$ is large (close to
one) tend to fall in the fully-interpretable region when the residual correction $\alpha^{*}(x)\hat{r}(x)$
is bounded, which mechanically couples the fully-interpretable region to the high-$p_{\mathrm{LR}}$ tail of
the score distribution"* — so the Table 3 concentration of high-base-rate borrowers in the fully
interpretable region (53.8% Zindi, 69.5% Taiwan) "is therefore a joint property of Eq. (6) and the
well-calibrated behaviour of the logistic-regression baseline on that tail", not an independent empirical
discovery.

<br>


### Task 13 — `orchestrate_fit_platt(p_hybrid, p_lr, labels, output_dir, verify_manuscript_targets) → Dict[str, Any]`

*Workbook anchor: cell 33, L12134.*

**Inputs.** The hybrid and LR probability vectors of all three partitions (validation for fitting, test for
application); the aligned label vectors; `output_dir`; the verification flag.

**Process.** *Step 1 — validation fit.* `fit_platt` (through `platt_equation_10`) fits a univariate logistic
regression of the binary labels on the validation hybrid probabilities under

$$p_{\mathrm{cal}}(x) = \frac{1}{1 + \exp\!\left(a \cdot p_{\mathrm{Hybrid}}(x) + b\right)}, \tag{10}$$

in an eight-substep internal procedure: input validation; unpenalised maximum-likelihood fitting (a
heavily-regularised fallback is documented for small validation sets); recovery of the library-convention
coefficients; derivation of the manuscript-convention $(a, b)$ by sign negation where the library convention
differs; a numerical round-trip assertion of the two conventions; strict-monotonicity verification of the
fitted map; logging; return. *Step 2 — frozen test application.* `apply_platt` evaluates the frozen logistic
function on the test hybrid probabilities; region assignments remain based on the *uncalibrated*
probabilities — calibration changes only the probability scale, never $\rho(x)$ or region membership — and
both the uncalibrated and calibrated test vectors are stored. *Step 3 — the LR+Platt ablation.*
`fit_lr_platt_ablation` fits a second Platt model on the validation *LR* probabilities (identical
procedure), applies it to the test LR probabilities, and records Brier and ECE (`compute_ece`, Equation (11)
with $K = 10$ bins); the expected Zindi values ($\approx 0.088$, $\approx 0.016$) are verified when the flag
is set.

**Transformation.** Uncalibrated hybrid probabilities → sigmoid-rescaled probabilities whose scale is
corrected toward the empirical positive rate while rank order is preserved; the ablation additionally yields
a second calibrated vector from the LR branch alone, isolating the calibration attribution.

**Role.** Implements Section 3.5 verbatim — *"We apply Platt scaling post-hoc, $p_{\mathrm{cal}}(x) =
\frac{1}{1 + \exp(a \cdot p_{\mathrm{Hybrid}}(x) + b)}$, with $a, b$ estimated on a held-out validation
fold"* — under Section 5.3's rule that all "Platt-calibration stages are performed exclusively on the
training and validation partitions". The ablation operationalises the study's principle of isolating "the
sources of predictive gains from calibration gains": the paper's Zindi numbers (LR+Platt Brier 0.088 versus
calibrated hybrid 0.085) attribute most of the calibration improvement to Platt scaling of the
class-weighted baseline, while the calibrated hybrid's ECE falls to 0.024 against 0.234 for the raw LR — the
tenfold reduction Table 2 and Figure 1 report.

<br>

### Task 14 — `orchestrate_compute_metrics(probabilities, labels_test, regions_test, p_lr_test, p_hybrid_test, region_stats, output_dir, verify_manuscript_targets) → Dict[str, Any]`

*Workbook anchor: cell 34, L12549.*

**Inputs.** A dictionary of test probability vectors for the five candidate models — LR, standalone XGBoost,
hybrid uncalibrated, hybrid Platt-calibrated, LR+Platt; the test labels; the test region assignments; the
test LR and hybrid probabilities (per-region AUC inputs); the Task 12 region statistics; `output_dir`; the
verification flag.

**Process.** *Step 1 — discrimination metrics.* `compute_discrimination_metrics` computes, per model on the
test partition, AUC (`roc_auc_score`), the Gini coefficient $\mathrm{Gini} = 2\,\mathrm{AUC} - 1$, and the
minority-class F1 at threshold 0.5 (`precision_recall_fscore_support`), aligned by test row order. *Step 2 —
calibration metrics.* `compute_calibration_metrics` computes the Brier score $\mathrm{Brier} =
\frac{1}{n}\sum_i (p_i - y_i)^2$ (`brier_score_loss`) and the Expected Calibration Error with $K = 10$
equal-width bins on $[0, 1]$,

$$\mathrm{ECE} = \sum_{k=1}^{K} \frac{|B_k|}{n}\,\bigl|\mathrm{acc}(B_k) - \mathrm{conf}(B_k)\bigr|, \tag{11}$$

with empty bins contributing zero, plus the reliability-diagram data (per-bin confidence and accuracy).
*Step 3 — per-region AUC and tables.* `compute_per_region_auc` computes LR and hybrid AUC within each of the
three interpretability regions; `assemble_results_tables` assembles Table 2 (predictive performance) and
Table 3 (region behaviour); `assert_headline_values` — when enabled — verifies the Zindi headline values
(calibrated hybrid AUC $0.869 \pm 0.005$, Brier $0.085 \pm 0.005$, ECE $0.024 \pm 0.005$) and, on deviation,
walks the fidelity-diagnosis checklist in its documented order: split seed → binning algorithm → rare-level
handling → calibration fitting → library versions → statistical-test implementation. Tables persist to JSON.

**Transformation.** Five probability vectors + region labels → the manuscript's reporting tables: rank
metrics (AUC, Gini), threshold metrics (F1 at 0.5), calibration metrics (Brier, ECE, reliability data) and
region-stratified AUCs.

**Role.** Implements Section 5.4's evaluation stack verbatim: *"Discrimination is evaluated via the Area
Under the ROC Curve (AUC), the Gini coefficient ($2\,\mathrm{AUC}-1$), and the minority-class F1-score.
Calibration is evaluated via the Brier score and the Expected Calibration Error of Eq. (11) with $K = 10$
equal-width probability bins, complemented by reliability diagrams."* The F1 convention follows Section
6.1's explicit caveat: *"F1 at 0.5 is reported for comparison with the methodological literature, not as the
operational decision rule"* — Platt scaling shifts probabilities toward the base rate and lowers the
0.5-threshold F1 even when rank-ordering is unchanged.

<br>

### Task 15 — `orchestrate_run_fairness_audit(config, test_frame, rho_test, regions_test, p_cal_test, y_test, output_dir, verify_manuscript_targets) → Dict[str, Any]`

*Workbook anchor: cell 35, L13137.*

**Inputs.** The test partition frame carrying the raw protected attributes; the test $\rho$ vector; the
region assignments; the calibrated hybrid test probabilities; the test labels; `config` (protected-attribute
mappings, reference groups); `output_dir`; the verification flag.

**Process.** *Step 1 — recode the protected attributes.* `recode_protected_attributes` builds the audit
frame with `row_id`, `rho`, `region`, `p_cal` and the protected columns: gender (as recorded), location
(urban/rural), education — collapsed per Section 4.5 to primary-or-less versus secondary-and-above, with
`Other` and `Don't know/Refused` mapped to secondary-and-above under a documented placeholder — and country
(Kenya, Rwanda, Tanzania, Uganda); `assert_cell_counts` checks the per-cell counts against Table 4 within
$\pm 5$ observations (the manuscript's counts depend on the exact split). *Step 2 — routing rates.*
`compute_routing_rates` (with `mask_indices`) estimates, per subgroup $g$, the routing rate
$\hat{\Pr}[\rho(x) < 0.7 \mid A = g]$ with 95% percentile bootstrap confidence intervals, $B = 500$
resamples drawn with replacement within (class × attribute) strata; an empty stratum in a resample falls
back to class-only stratification for that resample; the expected point estimates are verified and the
non-overlap of the rural/urban, primary/secondary and Uganda/Kenya intervals is checked when the flag is
set. *Step 3 — pooled and per-region DI and EOD.* `compute_di_and_eod` (with `_eod_for`) forms predicted
positives $\hat{y} = \mathbf{1}[p_{\mathrm{cal}} \ge 0.5]$ and computes, pooled and per region,

$$\mathrm{DI}(g) = \frac{\Pr[\hat{y} = 1 \mid A = g]}{\Pr[\hat{y} = 1 \mid A = g_{\mathrm{ref}}]}, \qquad \mathrm{EOD} = \max_{g}\left\{\bigl|\mathrm{TPR}(g) - \mathrm{TPR}(g_{\mathrm{ref}})\bigr|,\; \bigl|\mathrm{FPR}(g) - \mathrm{FPR}(g_{\mathrm{ref}})\bigr|\right\},$$

applying the degenerate-cell rule: a reference group with zero predicted positives in a region reports the
explicit undefined string "---", never zero. All reports persist with the post-treatment caveat reiterated
in the output.

**Transformation.** Test predictions and raw protected attributes → a per-region fairness profile: routing
rates with bootstrap uncertainty, disparate-impact ratios and equalized-odds differences, pooled and
stratified by the three interpretability regions (Tables 4–6 analogues).

**Role.** Implements Section 3.8 — "which is the empirical focus of the paper" — asking *"whether protected
subgroups (gender, urban/rural residence, education, country) are distributed evenly across those regions,
or whether some groups are systematically routed to the opaque ML-driven region where adverse-action
explanations become harder to produce"*. It reproduces the Section 6.3 findings (routing gaps of 18, 32 and
22 percentage points along location, education and country; gender essentially zero) and encodes two
statistical disciplines verbatim: per-region DI and EOD *"are consequently descriptive stratifications of
the audit surface rather than fairness metrics defined on well-behaved subpopulations"*, because
conditioning on $\rho(x)$ conditions on a variable *"causally downstream of $X$ and therefore a
post-treatment variable in the sense of Rosenbaum"*, while *"the routing-rate audit (list item 1) is not
affected, because $\rho(x)$ is the outcome rather than a conditioning variable"*; and the near-empty
predicted-positive cells of the ML-driven region are *"undefined, not zero, and must be reported explicitly
as such"*. The four-fifths rule ($\mathrm{DI} \in [0.8, 1.25]$) is reported as a reference threshold, not a
decision rule; reference groups are the majority subgroups.

<br>

### Task 16 — `orchestrate_run_bootstrap_delong(config, predictions, labels_test, output_dir, verify_manuscript_targets) → Dict[str, Any]`

*Workbook anchor: cell 36, L13519.*

**Inputs.** The frozen test prediction vectors of the hybrid, LR and standalone XGBoost; the test labels;
`config` (resample counts: $B = 2000$ for AUC differences); `output_dir`; the verification flag.

**Process.** *Step 1 — stratified paired bootstrap.* `stratified_paired_bootstrap_auc` runs $B = 2000$
resamples; each resample draws with replacement within the positive-class and negative-class test strata
separately (preserving the marginal class distribution), recomputes both AUCs on the resampled observations
and records $\Delta\mathrm{AUC}^{(b)}$; the 95% percentile interval is formed from the 2.5th and 97.5th
percentiles, for both comparisons (Hybrid vs LR, Hybrid vs XGBoost). *Step 2 — the DeLong test.*
`delong_test` implements the Sun–Xu fast midrank algorithm to estimate the covariance matrix of the two
correlated AUC estimates and computes

$$z = \frac{\widehat{\mathrm{AUC}}_1 - \widehat{\mathrm{AUC}}_2}{\sqrt{\widehat{\mathrm{Var}}(\widehat{\mathrm{AUC}}_1 - \widehat{\mathrm{AUC}}_2)}},$$

reporting the one-sided p-value for $H_1\!:\;\mathrm{AUC}_{\mathrm{Hybrid}} >
\mathrm{AUC}_{\mathrm{comparator}}$ and the two-sided p-value via the normal CDF; the implementation is
validated on a synthetic dataset with known AUCs before use. *Step 3 — consolidation.*
`consolidate_inference_table` assembles Table 9 (comparison, $\Delta\mathrm{AUC}$, bootstrap CI, one-sided
and two-sided DeLong p) for both datasets where applicable and records in the audit log that no
multiple-comparison correction is applied, consistent with the descriptive-audit intent; the table is
serialised to JSON.

**Transformation.** Paired test scores → per-comparison inference records: point $\Delta\mathrm{AUC}$,
percentile bootstrap CI, DeLong $z$ and one-/two-sided p-values.

**Role.** Implements Section 5.6's inference protocol — "Confidence intervals on all AUC differences,
disparate-impact ratios, and equalized-odds differences are constructed by stratified paired bootstrap on
the held-out test partition … For AUC-difference intervals we use $B = 2000$ resamples", with significance
"assessed by the paired DeLong test on correlated ROC curves" computed "by the fast midrank algorithm of Sun
and Xu" — and Section 6.5's headline evidence on Zindi: Hybrid vs LR $\Delta\mathrm{AUC} = +0.015$ (CI
$[+0.011, +0.020]$, $p < 10^{-9}$); Hybrid vs XGBoost $-0.004$ (CI $[-0.009, +0.001]$, $p = 0.92$); Taiwan:
$+0.057$ ($[+0.048, +0.066]$, $p < 0.001$) and $+0.001$ ($[-0.004, +0.006]$, $p = 0.37$). The bootstrap CI
is the primary interval statement; the DeLong test the primary hypothesis test; the absence of
multiple-comparison correction is a recorded design choice of the descriptive audit.

<br>

### Task 17 — `orchestrate_run_shap_analysis(config, booster, best_iteration, X_ohe_test, feature_names, regions_test, output_dir, verify_manuscript_targets) → Dict[str, Any]`

*Workbook anchor: cell 37, L13973.*

**Inputs.** The fitted residual XGBoost booster at its frozen best iteration; the one-hot test matrix; the
frozen one-hot feature names; the test region assignments (used to restrict to the ML-driven region, $\rho <
0.7$); `config`; `output_dir`; the verification flag.

**Process.** *Step 1 — TreeSHAP attributions.* `compute_tree_shap` restricts
$X_{\mathrm{ohe}}^{\mathrm{test}}$ to the ML-driven region ($n = 2{,}796$ on Zindi) and instantiates
`shap.TreeExplainer` on the *residual regressor only* — never on the full hybrid, preserving the additive
decomposition of Equation (5) — computing SHAP values for every observation in the region, aligned by
`row_id`. *Step 2 — ranking.* `rank_features_by_mean_abs_shap` computes the mean absolute SHAP per feature,

$$\overline{|\phi_j|} = \frac{1}{n_{\mathrm{ML}}} \sum_{x \in \text{ML region}} |\phi_j(x)|,$$

ranks features in descending order and, when the flag is set, asserts the top two ranks match the manuscript
(cellphone access = no; age of respondent). *Step 3 — stability screen and conditional diagnostics.*
`bootstrap_stability_screen` runs $B = 500$ observation-level resamples from the region, recomputes the
ranking per resample, computes each feature's rank standard deviation, verifies the top-2 ranks at $\sigma =
0.00$ and the top-5 set preserved in 74.0% of resamples, and computes the conditional mean SHAP for the
country dummies (Kenya $\approx -0.055$ among `country_Kenya = 1`; Rwanda $\approx +0.019$ among `country_Rwanda = 1`).
The aggregated report persists.

**Transformation.** ML-driven-region features → per-observation additive attributions of the residual
branch, reduced to a mean-|SHAP| ranking with an uncertainty screen over the ranking itself and conditional
diagnostics on the country dummies.

**Role.** Implements Section 3.6 verbatim: *"We compute TreeSHAP attributions on $f_{\mathrm{GBM}}$ for
these observations, restricted to the residual component to preserve the additive decomposition of Eq. (5).
Stability of the SHAP feature rankings is assessed by bootstrap resampling of the test set ($B = 500$), and
observed instability is treated as a flag on individual explanations rather than a property of the framework
as a whole."* It delivers Section 6.7's mechanism evidence: the residual learner's dominant features are the
same infrastructure/demographic features the LR already sees, and the country-dummy signs show it
"correcting country-specific miscalibration of the linear baseline within the uncertain subpopulation"
rather than importing alternative-data signal.

<br>

### Task 18 — `orchestrate_run_thin_file(config, train_df, test_df, model_probs_test, y_test, rho_test, full_test_metrics, output_dir, verify_manuscript_targets) → Dict[str, Any]`

*Workbook anchor: cell 38, L14491.*

**Inputs.** The Zindi training and test partition frames (standardisation statistics from training only);
the test probability vectors of LR, standalone XGBoost and the calibrated hybrid; the test labels; the test
$\rho$ vector; the full-test metrics of the calibrated hybrid (the H1 comparison baseline); `config`;
`output_dir`; the verification flag.

**Process.** *Step 1 — the composite index.* `construct_thin_file_index` maps `education_level` to the
config's ordinal scale, forms a cellphone binary ($\mathbf{1}[\text{cellphone\_access} = \text{Yes}]$) and a
formal-job indicator from the config's job-type mapping; each component is z-score standardised on the
training partition only and the index is their sum; the 25th-percentile threshold $\tau$ of the *training*
index is frozen (every mapping is documented as an UNSPECIFIED placeholder where the manuscript is silent).
*Step 2 — subgroup performance.* `apply_threshold_and_compute` defines the thin-file subgroup as test
observations with index below $\tau$, asserts the expected profile ($n = 1{,}559$; share 33.1%; positive
rate 2.4%) within tolerances, and computes AUC, Brier and ECE for the three models on the subgroup with $B =
2000$ stratified paired-bootstrap 95% CIs on the pairwise $\Delta\mathrm{AUC}$. *Step 3 — the pre-registered
hypotheses.* `test_hypotheses` compares the hybrid-over-LR AUC gain on the subgroup (expected $+0.058$)
against the full population ($+0.015$) — H1 — and the subgroup mean $\rho$ (expected 0.718) against the
full-test mean (0.703) — H2 — interpreting the absence of a downward shift through the mechanism the paper
states. The report persists.

**Transformation.** Three proxy features → a training-standardised composite index, its frozen
bottom-quartile threshold, and a subgroup-conditioned performance evaluation with bootstrap intervals,
scored against the full-test benchmarks.

**Role.** Implements Section 4.5's operational definition verbatim — *"The thin-file subgroup of Section 3.9
is defined as the bottom quartile of a composite index obtained by summing the standardized values of
education level, cellphone access (binary), and a formal-job indicator (binary)"* — and Section 6.4's two
findings: *"The hybrid AUC gain over LR is larger on thin-file than on the full population"* ($+0.058$ vs
$+0.015$; H1 supported), while *"the $\rho(x)$ distribution on the thin-file subgroup is not shifted toward
smaller values as hypothesised"* (0.718 vs 0.703; H2 not supported), for the reason the paper gives: *"the
thin-file population is so confidently predicted negative by the LR baseline … that the LR predictive
variance $4\,p_{\mathrm{LR}}(x)(1 - p_{\mathrm{LR}}(x))$ stays small, the adaptive weight $\alpha^{*}(x)$
stays low, and the residual correction has limited effect on the final prediction."* The H1/H2 verdicts
operationalise the study's pre-registration logic, including the explicit acknowledgement that H2 fails.

<br>


### Task 19 — `orchestrate_run_alternating_min(config, X_woe_train, X_ohe_train, y_train, X_woe_val, X_ohe_val, p_lr_val_initial, y_val, X_woe_test, X_ohe_test, y_test, beta_initial, beta_0_initial, booster_initial, best_iteration_initial, two_stage_test_auc, two_stage_fit_seconds, output_dir, verify_manuscript_targets) → Dict[str, Any]`

*Workbook anchor: cell 39, L15048.*

**Inputs.** The full WoE and one-hot design matrices of training, validation and test with labels; the
two-stage initialisation $(\beta^{(0)}, \beta_0^{(0)}, f_{\mathrm{GBM}}^{(0)}, \text{best iteration})$; the
two-stage test AUC and wall-clock fit time (the comparison baselines); `config`; `output_dir`; the
verification flag.

**Process.** *Step 1 — the fixed-point iteration.* `implement_fixed_point_iteration` (helpers `_sigmoid`,
`_joint_loss`) alternates, from the two-stage initialisation: (i) hold $f_{\mathrm{GBM}}$ fixed and update
$\beta$ by L-BFGS on the joint loss

$$\mathcal{L}(\beta, f_{\mathrm{GBM}}) = \sum_{i=1}^{N} \ell\!\left(y_i,\, \mathrm{clip}\!\left(p_{\mathrm{LR}}(x_i;\beta) + \alpha^{*}(x_i)\,f_{\mathrm{GBM}}(x_i),\, 0,\, 1\right)\right) + \lambda_{\mathrm{LR}}\|\beta\|_2^2 + \Omega(f_{\mathrm{GBM}}), \tag{12}$$

with $\ell$ the binary cross-entropy $\ell(y, p) = -y\ln p - (1-y)\ln(1-p)$ and $\Omega$ the XGBoost
complexity penalty — critically, $p_{\mathrm{LR}}$, $u_{\mathrm{LR}}$ and $\alpha^{*}$ are recomputed from
the *updated* $\beta$ before any hybrid prediction is formed; (ii) hold $\beta$ fixed, recompute the
residuals $r_i = y_i - p_{\mathrm{LR}}(x_i;\beta)$ and refit the residual booster on them; (iii) iterate
until the change in validation AUC falls below the $10^{-3}$ tolerance. *Step 2 — trajectory recording.*
`run_and_record_trajectory` records the iteration count, the validation-AUC sequence, the final test
$\Delta\mathrm{AUC}$ against the two-stage model, and per-iteration elapsed time via `time.perf_counter`
(the booster refit dominates runtime; expected total $\approx 4\times$ the two-stage fit). *Step 3 —
conclusion.* `document_conclusion` records convergence in three iterations on Zindi with a final
$\Delta\mathrm{AUC}$ of $+0.0006$ — smaller than the tolerance — so two-stage training remains the
operational deployment default; it states explicitly that step (ii) minimizes squared error on the residuals
rather than the joint binary cross-entropy, so the scheme is *not* coordinate descent on $\mathcal{L}$, and
that a true functional-gradient descent is left as future work. The scheme is not run on Taiwan. The report
persists.

**Transformation.** Two-stage parameter initialisation → a fixed-point sequence over $(\beta,
f_{\mathrm{GBM}})$ whose endpoint is scored on the test partition and compared against the two-stage
endpoint, with the full convergence trajectory and compute accounting.

**Role.** Implements Section 3.10's joint formulation — Equation (12) — and its deliberately honest Section
6.6 verdict: *"The tolerance is larger than the reported improvement, so the procedure halts before it can
produce a distinguishable change; we therefore do not interpret this as evidence that two-stage is globally
optimal, only that this particular fixed-point scheme does not improve on it in a detectable way, and that
the additional compute of the joint iteration (approximately four times the two-stage cost in our runs) is
not justified."* The methodological caveat is carried in the output, not buried: the residual-refit sub-step
is squared error on residuals, which is why the scheme "is not coordinate descent on $\mathcal{L}$" and why
"a true functional-gradient descent on the joint loss (which would flow gradients through both branches) is
left" as future work.

> **One registered caveat, disclosed rather than buried.** The convergence tolerance ($10^{-3}$ on
> validation AUC) exceeds the measured improvement ($+0.0006$ test $\Delta\mathrm{AUC}$); the orchestrator
> therefore refuses to claim joint superiority — exactly as the manuscript does — and records the $\approx
> 4\times$ wall-clock ratio as the compute cost of the methodological clarity the joint formulation buys.

<br>

### Task 20 — `run_full_study(config, zindi_raw, taiwan_raw, output_dir, run_taiwan, verify_manuscript_targets, verify_structural, verify_manuscript_values, verify_frozen_statistics) → Dict[str, Any]`

*Workbook anchor: cell 40, L17925.*

**Inputs.** The three ingested parameters (`config`, `zindi_raw`, `taiwan_raw`) and `output_dir`; the
`run_taiwan` flag gating Task 22; three verification flags scoping the Step 3 verifier — structural checks,
manuscript-value checks, and the frozen-statistics-from-training checks.

**Process.** *Step 1 — state initialisation and the callable registry.* `run_full_study` initialises the
shared `state` dictionary (raw frames, config, output dir) and builds the registry of the nineteen task
orchestrators through the per-task adapters `_adapter_validate_inputs` … `_adapter_run_alternating_min`
(each a state-dict wrapper of one orchestrator) plus `build_registry`, which verifies that all nineteen
names are present; the preamble helpers `_frame_digest` and `_partition_frames` serve the whole stage. *Step
2 — the sequential pipeline with frozen-state passing.* `run_pipeline` executes the registry in strict
dependency order; before and after each callable it snapshots the state by serialising the frozen statistics
to JSON (`_json_safe`) and recording the SHA-256 hash of every key DataFrame (`_snapshot_hashes`,
`_digest_array`), writes per-stage checkpoints (`_write_stage_checkpoint`), measures peak resident-set size
(`_measure_rss_bytes` through the guarded `resource`/`psutil` probes), and enforces the test-partition
information barrier (`_enforce_partition_access`, `_verify_test_indices_unchanged`, `_verify_test_barrier`,
`_verify_frozen_statistics_from_training`): the test partition lives under a protected key whose hash is
verified at every checkpoint, so no callable can mutate the test indices or labels without detection. *Step
3 — verification, checkpointing and reporting.* After every stage the lightweight verifier
(`run_lightweight_verifier` over `VerificationCheck` specifications — the nine intermediate quantities, e.g. `zindi_train_n == 14114`,
the selected $\alpha$ triple, region shares, routing-gap point estimates, the $\Delta\mathrm{AUC}$
confidence intervals, the SHAP top-five order, the thin-file sample size, the alternating-minimisation
iteration count — with `_resolve_check_actual`, `_compare_check`, `_verify_equals`, `_verify_in`,
`_verify_subset`, `_verify_gt`) runs, writing a fidelity-diagnostic report (`_write_fidelity_diagnostic`) on
failure and exercising its own negative self-test (`verifier_negative_self_test`); at the end,
`build_reproduction_report` compares every final metric against
`study_config["study_metadata"]["headline_results"]`, the Task 22 continuity chain runs conditionally
(`_run_taiwan_continuity_chain`, `build_taiwan_reproduction_report`) when `run_taiwan` is True, and
`_print_summary_table` prints the summary.

**Transformation.** Raw frames + config → a fully verified study state: every frozen statistic, fitted
model, table and report, with per-stage checkpoints, hash barriers and a reproduction report attesting the
manuscript targets. The state is the study's complete audit trail, not merely its results.

**Role.** The operational core of the paper's release statement — *"The implementation, including the
fairness audit module (with stratified paired bootstrap and DeLong test) and the alternating fixed-point
joint-optimization routine (run_zindi_paper.py, ~800 lines, end-to-end runtime under 5 minutes on a recent
laptop CPU), will be released at a public repository prior to journal submission under a permissive
open-source license"* — and the mechanical enforcement of Section 5.3's discipline: *"all preprocessing,
hyperparameter tuning, residual-learning, and Platt-calibration stages are performed exclusively on the
training and validation partitions. The test partition is reserved strictly for final evaluation."* The
information barrier is hash-enforced rather than conventional, and the per-stage verifier turns the
reporting discipline ("log every random seed, every hyperparameter selection, every environment version, and
every deviation from the stated protocol") into executed code.

<br>

### Task 21 — `orchestrate_run_robustness(config, analysis, X_ohe_train, X_ohe_val, X_ohe_test, r_train, p_lr, alpha_star, y_val, y_test, output_dir) → Dict[str, Any]`

*Workbook anchor: cell 41, L18787.*

**Inputs.** The Task 15 audit analysis frame ($\rho$, regions, protected attributes, calibrated
probabilities); the one-hot matrices of all partitions; the training residuals; the final LR probabilities
and frozen adaptive weights of all partitions; the validation and test labels; `config`; `output_dir`.

**Process.** *Step 1 — threshold sensitivity.* `threshold_sensitivity` recomputes the region assignments
under the alternative interpretability-threshold pairs $(\rho_{\mathrm{fully}}, \rho_{\mathrm{ml}}) = (0.85,
0.65)$ and $(0.95, 0.75)$, re-runs the routing-rate audit under each, and recomputes F1, DI and EOD at
decision thresholds 0.4 and 0.6 in addition to the primary 0.5, documenting the small-sample caveats that
alternative thresholds create (especially in the ML-driven region). *Step 2 — LightGBM sensitivity.*
`lightgbm_sensitivity` refits the residual learner as `LGBMRegressor(objective='regression', learning_rate=0.03, num_leaves=7, subsample=0.8, feature_fraction=0.8, lambda_l2=2)`,
rebuilds the hybrid with the *same* frozen adaptive weights from Task 11, refits the Platt calibration on
the validation partition, and recomputes the test AUC, Brier, ECE, region shares and routing-rate gaps,
treating the differences as sensitivity evidence. *Step 3 — reference-group sensitivity.*
`reference_group_sensitivity` re-runs the fairness audit with the alternative reference groups (female,
rural, primary-or-less, Rwanda), recomputes routing gaps, DI ratios and EOD, and documents that routing-rate
gaps and pooled DI/EOD are the least reference-sensitive while per-region values can shift materially under
small denominators — reusing the degenerate-cell rule throughout. All three reports persist.

**Transformation.** Frozen chain artefacts → a sensitivity envelope over (i) the interpretability and
decision thresholds, (ii) the boosting library, and (iii) the fairness reference groups, each compared
against the primary configuration.

**Role.** Converts the manuscript's declared robustness intentions into executed, persisted analyses:
Section 3.3 — the thresholds "are indicative rather than theoretical; their sensitivity is evaluated
empirically"; Section 3.2 — "LightGBM retained as a sensitivity check"; Section 5.5 — the reference group
"is set to the majority subgroup on each attribute (male for gender, urban for location, secondary-and-above
for education, Kenya for country), with sensitivity to that choice reported as a robustness check".

<br>

### Task 22 — `orchestrate_taiwan_reproduction(config, taiwan_cleansed, taiwan_partitions, output_dir, verify_manuscript_targets) → Dict[str, Any]`

*Workbook anchor: cell 42, L19302.*

**Inputs.** The cleansed Taiwan frame (24 columns) and its frozen partition manifest from Task 3; `config`
carrying the declared Taiwan placeholder protocol; `output_dir`; the verification flag.

**Process.** *Step 1 — placeholder preprocessing.* `preprocess_taiwan_placeholder` (with
`_partition_frames`) applies the identical preprocessing callables used for Zindi: drop `ID`; compute
medians/modes and the 1st/99th percentiles on the Taiwan training partition only, freeze and apply them;
monotonic binning and WoE encoding for the logistic branch; one-hot encoding for the boosting branch — every
decision logged and flagged UNSPECIFIED IN MANUSCRIPT under the config's placeholder protocol, because the
thesis pipeline is not restated in the paper. *Step 2 — fit and evaluate.* `fit_and_evaluate_taiwan`
executes the same model chain as Zindi through the shared callables — C selection, LR, residuals, residual
XGBoost, standalone XGBoost, adaptive-weight grid search, hybrid and $\rho$, Platt calibration, metrics —
and verifies the continuity numbers within stated tolerances when the flag is set: calibrated hybrid AUC
$0.776 \pm 0.005$, Brier $0.136 \pm 0.005$, ECE $0.011 \pm 0.003$; fully-interpretable share $6.3\% \pm 1$
pp at a $69.5\% \pm 2$ pp default rate; $\Delta\mathrm{AUC}$ vs LR $+0.057$ (CI $[+0.048, +0.066]$, $p <
0.001$) and vs XGB $+0.001$ (CI $[-0.004, +0.006]$, $p = 0.37$). *Step 3 — deviation investigation.*
`investigate_deviations` walks the fidelity-diagnosis checklist in its documented order (split seed and
partition indices → binning algorithm variant → rare-level WoE handling → calibration fitting details →
library-version behaviour → DeLong implementation variant) when any number deviates beyond tolerance,
*without* modifying the Zindi pipeline to force a match, and documents the deviations transparently as
arising from the placeholder protocol. The reproduction report persists with every placeholder decision
flagged.

**Transformation.** The raw Taiwan frame → a full independent run of the Zindi pipeline under the
inherited-preprocessing placeholder, producing the continuity-benchmark report with all model artefacts and
deviation records.

**Role.** Implements Section 4.3's continuity contract — the dataset *"is retained from the thesis without
alteration of preprocessing, to permit direct numerical continuity with the headline results of Kanziga
(2026)"* — and Section 6.1's benchmark row of Table 2 (calibrated hybrid AUC 0.776, Brier 0.136, ECE 0.011,
against LR 0.719 and XGBoost 0.775). The honesty rule of the reproduction is the fidelity checklist: where
the placeholder protocol drifts from the thesis numbers, the orchestrator documents rather than tunes, per
the study's standard of "Reporting discrepancies transparently even when the qualitative patterns reproduce,
and documenting all environment versions and seeds to permit drift diagnosis".

<br>

### Task 23 (top level) — `run_research_pipeline(config, zindi_raw, taiwan_raw, output_dir, run_taiwan, verify_manuscript_targets) → Dict[str, Any]`

*Workbook anchor: cell 43, L19745; invoked by the execution cell 44.*

**Inputs.** The three ingested parameters — the study configuration and the two raw frames as manufactured
by the corpus preamble (cell 20: `study_config.json` plus the schema-conformant synthetic corpus built under
the documented deterministic seed, flagged as a placeholder for the real Zindi/Taiwan CSV downloads) —
`output_dir`, the `run_taiwan` flag and the manuscript-target verification flag.

**Process.** *Step 1 — the verified chain.* `run_tasks_1_to_19_chain` invokes the Task 20 `run_full_study`
orchestrator exactly once, with the test-partition information barrier and the manuscript-target flag
propagated, and validates that the returned state carries every key the downstream stages consume. *Step 2 —
robustness and the Taiwan chain.* `run_robustness_and_taiwan` executes the verified Task 21 robustness
orchestrator on the chain artefacts and, when `run_taiwan` is True, the Task 22 Taiwan reproduction
orchestrator — reusing the verified orchestrators without duplication. *Step 3 — consolidation.*
`consolidate_artifacts` assembles the artefacts of every executed stage into a single dictionary keyed by
stage, persists the JSON run manifest carrying the JSON-safe subset, the execution log and the reproduction
report, verifies that the manifest file exists, and returns the consolidated artefacts dictionary (with the
reproduction pass/total counts).

**Transformation.** The three ingested parameters → one consolidated, stage-keyed artefacts dictionary plus
a persisted run manifest — the study's complete executable record: every stage's outputs, the execution log,
and the reproduction report against the manuscript's headline results.

**Role.** The top-level orchestrator is the executable embodiment of the paper's governance claim — the
framework as *"an operational governance framework for auditable, fair, and interpretable machine-learning
credit decisioning that simultaneously preserves predictive accuracy, satisfies regulator-visible
explanation requirements, and proactively prevents protected subgroups from being concealed within opaque
model regions, thereby rendering digital credit safer, more explainable, and more equitable for thin-file
and underbanked populations"* — executed as: Tasks 1–19 run exactly once on a hash-protected test partition;
robustness and the Taiwan continuity benchmark reuse the verified chain without duplication; the persisted
manifest makes every seed, selection and deviation auditable after the fact, in the form the Code and Data
Availability section promises (released implementation, ~800 lines, end-to-end runtime under 5 minutes on a
recent laptop CPU, permissive open-source license).

<br><br>
