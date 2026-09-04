# Interpretable Hybrid Credit Scoring for Thin-File and Underbanked Populations

[![License: MIT](https://img.shields.io/badge/License-MIT-blue?style=flat)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2608.26837-B31B1B?style=flat&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2608.26837)
[![Method](https://img.shields.io/badge/Method-Residual--Learning%20Hybrid-orange?style=flat)](https://arxiv.org/abs/2608.26837)
[![Focus](https://img.shields.io/badge/Focus-Fair%20ML%20%7C%20Financial%20Inclusion-00529B?style=flat)](https://github.com/chirindaopensource/interpretable_hybrid_credit_scoring_for_thin_file_and_underbanked_populations)

**Repository:** `https://github.com/chirindaopensource/interpretable_hybrid_credit_scoring_for_thin_file_and_underbanked_populations`
**Owner:** 2026 Craig Chirinda (Open Source Projects)

This repo independently implements the 2026 paper by Kanziga, Gaba, and Kanamugire ([arXiv:2608.26837](https://arxiv.org/abs/2608.26837)). The full citation appears below.

The paper provides no code. This repo turns its math into a repeatable research workflow. It links a logistic scorecard to a boosted residual learner. For each borrower, it shows the scorecard's share of the risk score and checks fairness in three regions.

> **Read before running.** This workspace has no raw CSV files, so the notebook ships with synthetic data that match the required schema. They have the published row counts, column order, and class balance. The full pipeline runs, but its results do **not** reproduce the paper. Use real Zindi and Taiwan data to repeat the study.

## Contents

- [What the pipeline delivers](#what-the-pipeline-delivers)
- [Reproduction status and data](#reproduction-status-and-data)
- [Method](#method)
- [Requirements and installation](#requirements-and-installation)
- [Inputs](#inputs)
- [Run the study](#run-the-study)
- [Pipeline tasks](#pipeline-tasks)
- [Outputs](#outputs)
- [Validation and operating rules](#validation-and-operating-rules)
- [Registered divergences](#registered-divergences)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [Citation](#citation)

## What the pipeline delivers

The notebook `interpretable_hybrid_credit_scoring_for_thin_file_and_underbanked_populations_draft.ipynb` has **170 callables**: 165 functions, 2 classes, and 3 class methods. It groups them into 23 tasks and two pipeline layers.

The pipeline provides:

- A WoE logistic scorecard fitted under a strict training-only information barrier.
- A shallow XGBoost model that learns raw-response residuals from the scorecard.
- Adaptive residual weights selected across all 27 configured parameter triples.
- Per-borrower interpretability ratios and three interpretability regions.
- Platt calibration, five-model evaluation, TreeSHAP analysis, and thin-file analysis.
- Pooled and region-level fairness audits for four protected attributes.
- Stratified paired-bootstrap intervals and the Sun–Xu fast-midrank DeLong test.
- Deterministic checkpoints, per-stage telemetry, verification records, and a run manifest.
- Optional Taiwan continuity benchmarking and a documented robustness suite.

## Reproduction status and data

### The data requirement

The code needs these real data sets:

| Dataset | Required source and role |
|---|---|
| Zindi Financial Inclusion in Africa | The main study data: 23,524 respondents in Kenya, Rwanda, Tanzania, and Uganda, drawn from Financial Sector Deepening Network survey instruments collected in 2016–2018. |
| Taiwan Credit Default (UCI) | The 30,000-account continuity benchmark. |

No program can generate these source data. The shipped notebook builds synthetic frames from `np.random.default_rng(71)` only to test each rule and code path.

The synthetic Zindi label is the top 14.08% of a fixed logistic score from six features: `age_of_respondent`, `household_size`, `education_level`, `cellphone_access`, `job_type`, and `country`. Its IV filter keeps those six features and drops four as noise. Its discrimination metrics far exceed the published values. Do not draw real-world conclusions from this corpus.

To repeat the study, change the *Study Inputs* cell to read files. First, add a gate that rejects generator-derived frames.

### Label semantics

For Zindi, `bank_account = 1` means the person owns a formal bank account. It does **not** mean default. The paper uses account ownership as a proxy for an eligibility gate. Do not read the label as a loan-default event.

## Method

The model links a scorecard to a residual learner. The scorecard gives the clear part of the score. The residual learner adds a bounded correction.

### Scorecard encoding and feature selection

The scorecard uses Weight-of-Evidence (WoE) inputs. For bin or level $`k`$, $`g_k`$ counts $`y=0`$ training rows and $`b_k`$ counts $`y=1`$ rows. $`G`$ and $`B`$ are their totals.

```math
\text{WoE}_k = \ln\!\left(\frac{g_k/G}{b_k/B}\right), \qquad
\text{IV} = \sum_k\left(\frac{g_k}{G} - \frac{b_k}{B}\right)\text{WoE}_k .
```

A zero count gets the neutral value $`\text{WoE}_k=0`$. A category missing from training gets the same value. Keep features with $`\text{IV} \ge 0.02`$.

For continuous features, start with 20 quantile bins. Merge to make WoE monotone. Merge again to meet the five-percent training-frequency floor. Then check monotonicity once more.

### Residual-learning hybrid

The scorecard uses L2 logistic regression with balanced class weights. Five-fold stratified CV picks $`C`$ from $`\{0.01, 0.1, 1, 10\}`$.

The residual learner fits:

```math
r_i = y_i - p_{\text{LR}}(x_i), \qquad
p_{\text{Hybrid}}(x) = \mathrm{clip}\!\left(p_{\text{LR}}(x) + \alpha^*(x)\,\hat r(x),\;0,\;1\right).
```

The booster uses squared-error loss. Its raw-response target keeps its output on the probability scale. The two branch terms can therefore be compared directly.

The XGBoost learner uses depth 3, rate 0.03, 400 estimators, row and column subsampling of 0.8, and $`\lambda=2`$. A custom `xgboost.callback.TrainingCallback` stops on validation AUC of the provisional hybrid. Native early stopping cannot use this metric because it cannot see $`p_{\text{LR}}`$.

### Adaptive weighting and interpretability

The residual gets more weight when the scorecard is unsure:

```math
u_{\text{LR}}(x)=4p_{\text{LR}}(x)\left(1-p_{\text{LR}}(x)\right), \qquad
\alpha^*(x)=\alpha_{\min}+(\alpha_{\max}-\alpha_{\min})u_{\text{LR}}(x)^\gamma.
```

The search tests all 27 candidates in
$`\{0.0,0.1,0.2\}\times\{0.3,0.5,0.8\}\times\{0.5,1.0,2.0\}`$
with five-fold stratified CV. Each fold refits all three parts.

The linear share is:

```math
\rho(x)=\frac{|p_{\text{LR}}(x)|}{|p_{\text{LR}}(x)|+|\alpha^*(x)\hat r(x)|}.
```

Each prediction falls in one region:

| Region | Rule |
|---|---|
| Fully interpretable | $`\rho(x)\ge0.9`$ |
| Partially interpretable | $`0.7\le\rho(x)<0.9`$ |
| ML-driven | $`\rho(x)<0.7`$ |

When the residual is zero, $`\rho(x)=1`$. The cutoffs are useful guides, not theory. The robustness suite also tests $`(0.85,0.65)`$ and $`(0.95,0.75)`$.

### Calibration and evaluation

The hybrid stays between zero and one, but it may need calibration. The pipeline fits Platt scaling by unpenalised maximum likelihood on validation data:

```math
p_{\text{cal}}(x)=\frac{1}{1+\exp\!\left(a\,p_{\text{Hybrid}}(x)+b\right)}.
```

It reports AUC, Gini, minority-class F1 at 0.5, Brier score, ECE with 10 equal-width bins, and reliability data for five models. It also builds Tables 2 and 3.

Region assignment always uses **uncalibrated** hybrid scores. Calibration can change F1 and Brier score. It cannot change $`\rho(x)`$ or the region.

### Fairness, inference, explanations, and thin-file analysis

The decision rule is $`\hat y=\mathbf{1}[p_{\text{cal}}^{\text{test}}\ge0.5]`$. The audit reports disparate impact (DI) and equalised-odds difference (EOD), both overall and by region. It reports ML-driven routing as $`\hat{\Pr}[\rho(x)<0.7\mid A=g]`$ with 95% percentile intervals from 500 class-by-attribute resamples.

If a region has no reference-group predicted positives, use `UNDEFINED = "---"`, never `0.0`. Region-level DI and EOD describe a split; they are not causal fairness measures. $`\rho(x)`$ is downstream of the model features. The report uses the U.S. EEOC four-fifths band: $`0.8`$–$`1.25`$.

For model comparison, use:

- A stratified paired bootstrap with $`B=2000`$. It resamples positive and negative test rows separately. It takes score vectors, so it cannot refit models in the loop.
- The Sun–Xu fast-midrank DeLong test for paired ROC curves. It reports one-sided and two-sided results. The audit records that it does not correct for multiple comparisons.

For the ML-driven region, TreeSHAP explains the sliced residual booster, not the full hybrid. The stability screen uses 500 row-level bootstrap resamples. It reports rank standard deviations and top-five-set preservation.

The thin-file index adds training-standardised education, cellphone, and formal-job parts. The pipeline fixes its cut at the 25th training percentile and tests both registered hypotheses. It retains the paper's result that H2 is not supported.

### Joint optimisation

Task 19 takes turns: L-BFGS scorecard update, residual update, booster refit, and validation-AUC check. It stops at $`10^{-3}`$. It rebuilds $`\alpha^*(x;\beta)`$ in the objective at each step. This prevents stale weights. Two-stage training is the default. The joint run does not use Taiwan.

![Project overview](https://raw.githubusercontent.com/chirindaopensource/interpretable_hybrid_credit_scoring_for_thin_file_and_underbanked_populations/main/interpretable_hybrid_credit_scoring_for_thin_file_and_underbanked_populations_ipo_main.png)

## Requirements and installation

### Requirements

- Python 3.11+.
- Verified package versions: `numpy` 2.3.5, `pandas` 2.3.3, `scipy` 1.16.3, `scikit-learn` 1.7.2, `xgboost` 3.4.1, `lightgbm` 4.7.0, and `shap` 0.52.0.
- `psutil` for per-stage memory logs on Windows. Its import is guarded; if missing, memory data becomes `None` and the study continues.
- The Zindi and Taiwan source data to repeat the study.

A CPU-only laptop is sufficient. At Zindi scale, a full run takes about four minutes. The 27-triple, five-fold adaptive search takes about 200 seconds. The 2,000-resample bootstrap takes about 19 seconds.

### Installation

```sh
git clone https://github.com/chirindaopensource/interpretable_hybrid_credit_scoring_for_thin_file_and_underbanked_populations.git
cd interpretable_hybrid_credit_scoring_for_thin_file_and_underbanked_populations
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Alternatively:

```sh
pip install numpy pandas scipy scikit-learn xgboost lightgbm shap matplotlib jupyter psutil
```

Keep `study_config.json` in the working directory. Place real source files under `data/`. Start the notebook with:

```sh
jupyter notebook interpretable_hybrid_credit_scoring_for_thin_file_and_underbanked_populations_draft.ipynb
```

## Inputs

Task 1 validates each input before fitting begins.

| Input | Contract |
|---|---|
| `config: Dict[str, Any]` | Parsed from `study_config.json`. The file has 32 top-level sections, an equation-parameter map covering Equations (1)–(12) plus five unnumbered equations, and pinned hyperparameters. |
| `zindi_raw: pd.DataFrame` | Shape **(23,524, 13)** in the declared column order. `bank_account` is `int64`, with **1 = formal-account ownership**, and a positive rate of **0.1408**. Country counts must be Rwanda 8,735; Tanzania 6,620; Kenya 6,068; Uganda 2,101. The frame has zero missing values. Preserve `"Divorced/Seperated"` exactly. |
| `taiwan_raw: pd.DataFrame` | Shape **(30,000, 25)**. `default.payment.next.month` has a positive rate of about **0.22**. `PAY_0` is present and `PAY_1` is absent. `SEX` is a subset of {1,2}; `EDUCATION` of {0,…,6}; and `MARRIAGE` of {0,1,2,3}. |

Task 2 drops Zindi's `uniqueid` and `year`; `year` is perfectly collinear with `country`. It drops Taiwan's `ID`, leaving 24 Taiwan columns.

## Run the study

All Task 1–23 callables live in the same notebook namespace. There is no separate `.py` module to import. Load the configuration and real frames in the *Study Inputs* cell, then call the top-level orchestrator:

```python
from pathlib import Path
from typing import Any, Dict

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger("interpretable_hybrid_credit_scoring")

study_artifacts: Dict[str, Any] = run_research_pipeline(
    config=config,
    zindi_raw=zindi_raw,
    taiwan_raw=taiwan_raw,
    output_dir=Path.cwd() / "notebook_outputs",
    run_taiwan=False,
    verify_manuscript_targets=True,
)
```

Use `verify_manuscript_targets=False` only for a synthetic smoke run. A synthetic corpus cannot match the paper's Information Values or headline metrics. Set it to `True` when real data are in place.

The callable signature is:

```python
run_research_pipeline(
    config,
    zindi_raw,
    taiwan_raw,
    output_dir,
    run_taiwan=False,
    verify_manuscript_targets=True,
) -> Dict[str, Any]
```

The return value includes `stages_executed`, `pipeline_order`, `chain`, `robustness`, `taiwan`, `execution_log`, and `reproduction_report`.

For advanced control, call `run_full_study(config, zindi_raw, taiwan_raw, output_dir, run_taiwan, verify_manuscript_targets, verify_structural, verify_manuscript_values, verify_frozen_statistics)`. It exposes structural verification, manuscript-value verification, and the runtime frozen-statistics check independently.

## Pipeline tasks

| Task | Orchestrator | What it does |
|---:|---|---|
| 1 | `orchestrate_validate_ingested_parameters` | Validates the 32 configuration sections, equation map, types, float identities, schemas, label rates, country composition, level sets, and the source typo. It records a column reorder instead of halting. |
| 2 | `orchestrate_cleanse_data` | Drops non-predictive identifiers, strips object-column whitespace, validates exhaustive levels and integer types, and writes a cleansing audit with SHA-256 digests and UTC time. |
| 3 | `orchestrate_create_partitions` | Makes the chained 60/40 then 50/50 stratified split. Zindi yields 14,114 / 4,705 / 4,705; Taiwan yields 18,000 / 6,000 / 6,000. |
| 4 | `orchestrate_fit_preprocessing_stats` | Fits training medians, sorted-first categorical modes, and 1st/99th-percentile winsorization bounds. |
| 5 | `orchestrate_fit_woe_encoding` | Fits 20-quantile monotonic WoE bins, applies fallback and merge rules, enforces the five-percent frequency floor, and guards against rounded-key collisions. |
| 6 | `orchestrate_iv_filter_and_ohe` | Retains IV-qualified features and builds frozen-level one-hot matrices. Unseen levels map to all-zero indicators. |
| 7 | `orchestrate_fit_logistic` | Selects $`C`$ by five-fold CV and fits `lbfgs` logistic regression with L2 penalty, balanced weights, and `max_iter=1000`. |
| 8 | `orchestrate_compute_residuals` | Computes $`y-p_{\text{LR}}`$ on every partition, checks the $`[-1,1]`$ range, and halts on an almost constant training residual. |
| 9 | `orchestrate_fit_residual_xgboost` | Fits the residual XGBoost model with provisional-hybrid-AUC early stopping and requires positive residual-prediction correlation. |
| 10 | `orchestrate_fit_standalone_xgboost` | Fits the standalone `binary:logistic` comparator with training-derived `scale_pos_weight=n_0/n_1`. |
| 11 | `orchestrate_select_adaptive_weights` | Evaluates the 27 adaptive triples by five-fold CV and freezes each partition's $`\alpha^*`$ values. |
| 12 | `orchestrate_build_hybrid_and_rho` | Builds clipped predictions, calculates $`\rho`$, assigns regions, and summarizes region shares and outcomes. |
| 13 | `orchestrate_fit_platt` | Fits unpenalised Platt calibration on validation, checks Equation (10), checks strict monotonicity, applies the frozen model to test, and fits the LR+Platt ablation. |
| 14 | `orchestrate_compute_metrics` | Computes metrics and reliability data for all five models; creates Tables 2 and 3. |
| 15 | `orchestrate_run_fairness_audit` | Recodes protected attributes, measures routing with bootstrap intervals, and calculates pooled and per-region DI and EOD. |
| 16 | `orchestrate_run_bootstrap_delong` | Runs the stratified paired bootstrap and Sun–Xu DeLong test; produces Table 9. |
| 17 | `orchestrate_run_shap_analysis` | Explains the sliced residual booster for $`\rho<0.7`$ and runs the 500-resample stability screen. |
| 18 | `orchestrate_run_thin_file` | Builds the frozen thin-file index, evaluates subgroup performance and calibration, and tests the two registered hypotheses. |
| 19 | `orchestrate_run_alternating_min` | Runs the fixed-point joint-optimisation routine and records its path and per-iteration cost. It skips Taiwan. |
| 20 | `run_full_study` | Runs Tasks 1–19 in dependency order, enforces the information barrier, checkpoints state, captures telemetry and warnings, and verifies every stage. |
| 21 | `orchestrate_run_robustness` | Tests alternative region thresholds, decision thresholds, LightGBM, and reference groups. |
| 22 | `orchestrate_taiwan_reproduction` | Applies the Taiwan continuity protocol and its six-item fidelity checklist. |
| 23 | `run_research_pipeline` | Runs the study chain, robustness suite, optional Taiwan benchmark, and final manifest. |

## Outputs

`run_research_pipeline` returns a consolidated dictionary and writes a versioned archive.

| Key | Contents |
|---|---|
| `stages_executed` | Stages that ran, such as `['chain', 'robustness']`, plus `taiwan` when enabled. |
| `pipeline_order` | The canonical 19-stage dependency order. |
| `chain` | Validation, cleansing, partitions, frozen statistics, IV values, retained features, models, adaptive results, region statistics, and Platt parameters in both conventions. |
| `robustness` | Region-threshold, decision-threshold, LightGBM, and reference-group sensitivity results. |
| `taiwan` | Taiwan benchmark results and deviation report, or `None`. |
| `execution_log` | Stage order, time, memory and delta, warnings, checkpoint path, added state keys, and verification counts. |
| `reproduction_report` | Each headline result, its reproduced value, target, tolerance, discrepancy, and pass/fail status. |

The generated directory has this layout:

```text
notebook_outputs/
├── run_manifest.json
├── reproduction_report.json
├── checkpoints/                    # 00_validate_inputs.json … 18_run_alternating_min.json
├── t1_validate/validation_report.json
├── t2_cleanse/                     # Audit and cleansed CSVs
├── t3_split/                       # Frozen partition manifests
├── t4_preprocessing/               # Medians, modes, winsorization bounds
├── t5_woe/woe_artifacts.json
├── t6_ohe/iv_ohe_artifacts.json
├── t7_logistic/logistic_artifacts.json
├── t9_residual/residual_booster_artifacts.json
├── t10_standalone/standalone_xgb_artifacts.json
├── t11_adaptive/adaptive_weights_artifacts.json
├── t12_hybrid/hybrid_rho_artifacts.json
├── t13_platt/platt_artifacts.json
├── t14_metrics/
├── t15_fairness/fairness_audit_artifacts.json
├── t16_inference/inference_artifacts.json
├── t17_shap/shap_artifacts.json
├── t18_thin_file/thin_file_artifacts.json
└── t19_joint/joint_optimization_artifacts.json
```

## Validation and operating rules

### Information barrier

Fit every transform on training data only, then freeze it. This covers imputation, winsorization, WoE bins and tables, one-hot levels, $`C`$, adaptive weights, and Platt parameters.

The runner checks each stage's data access before and after it runs. With `verify_frozen_statistics=True`, it rebuilds medians and winsorization bounds from training data only. This directly checks for use of held-out rows.

### Two verification classes

Structural checks stop invalid shapes, label domains, and array alignment. Value checks record a mismatch with a paper target but let the run go on. A mismatch is a study result, not a code fault.

The checker uses exact match, absolute tolerance, percentage-point tolerance, and ordered-sequence match. It skips weak membership and threshold tests. A self-test corrupts each value and proves that its check fails.

### Fairness and inference rules

- Never replace an undefined fairness cell with `0.0`.
- Never use test data to choose $`C`$, early stopping, adaptive weights, or calibration.
- Never refit an estimator inside the bootstrap loop.
- Treat per-region fairness estimates as descriptive, not causal.
- Record every methodological departure where it occurs.

## Registered divergences

| Item | Implementation decision |
|---|---|
| Adaptive-weight CV | Section 3.4 specifies five folds, while Table 1 says three. The implementation uses five folds for every triple and logs the resolution. |
| Probability-scale $`\rho`$ | The ratio uses probability, not logit, scale. High-$`p_{\text{LR}}`$ borrowers can therefore cluster in the fully interpretable region when residual corrections are bounded. A logit-scale variant is future work. |
| Platt parameterisation | Equation (10) uses a positive exponent, while `sklearn` uses a negative one. The artifacts store both pairs, with $`a=-w`$ and $`b=-c`$, and verify the mapping to $`10^{-10}`$. |
| Thin-file standard deviation | The paper does not state the estimator. The implementation uses population standard deviation, `ddof=0`, and records it as a placeholder. |
| Joint optimisation | The booster minimizes residual squared error, not joint cross-entropy. The method is therefore not coordinate descent on the stated joint loss. |
| Taiwan preprocessing | The paper inherits Taiwan preprocessing from earlier work without restating it. Each decision is logged as `UNSPECIFIED IN MANUSCRIPT`; the Zindi pipeline is not altered to resolve it. |

## Configuration

`study_config.json` is the single source of truth; `config.yaml` mirrors it. It controls:

- Partition fractions and random seed.
- Imputation, winsorization, WoE, IV, and monotonic-binning rules.
- Logistic, residual-XGBoost, standalone-XGBoost, and LightGBM settings.
- Adaptive-weight grids, $`\rho`$ thresholds, calibration, and metrics.
- Protected attributes, reference groups, fairness rules, bootstrap counts, and DeLong settings.
- TreeSHAP stability settings and thin-file index rules.
- `verify_structural`, `verify_manuscript_values`, and `verify_frozen_statistics`.

The settings and run manifest give a reason for every `UNSPECIFIED IN MANUSCRIPT` parameter.

## Project structure

```text
interpretable_hybrid_credit_scoring_for_thin_file_and_underbanked_populations/
├── interpretable_hybrid_credit_scoring_for_thin_file_and_underbanked_populations_draft.ipynb
├── interpretable_hybrid_credit_scoring_for_thin_file_and_underbanked_populations_ipo_main.png
├── study_config.json
├── config.yaml
├── requirements.txt
├── LICENSE
├── README.md
├── data/
│   ├── zindi_financial_inclusion.csv
│   └── taiwan_credit_default.csv
└── notebook_outputs/
```

## Contributing

Contributions are welcome. Fork the repo, make a feature branch, and send a pull request that explains the change.

Contributions must:

- Follow PEP 8. Add full type hints, technical docstrings, checks, and error handling.
- Preserve the training-only information barrier.
- Fail on missing provider data instead of silently generating substitutes.
- Add a failing self-test for every new checker rule.
- Report degenerate fairness cells as undefined.
- State each method change where it starts.

## Recommended extensions

- Add a real-data provenance gate that rejects generator-derived frames.
- Implement and compare a logit-scale interpretability ratio.
- Replace the current fixed-point scheme with true functional-gradient joint optimisation.
- Add intersectional and multi-group fairness analysis.
- Optimize decision thresholds against asymmetric lending costs.
- Generate adverse-action reason codes from WoE scorecard contributions.
- Add region-aware routing constraints for protected groups.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Citation

If you use this code or methodology, cite the original arXiv preprint:

```bibtex
@misc{kanziga2026interpretable,
  title={Interpretable hybrid credit scoring for thin-file and underbanked
         populations: A fairness-aware residual learning framework with
         evidence from East African financial inclusion data},
  author={Kanziga, Belise and Gaba, Ya\'e U. and Kanamugire, Olivier},
  year={2026},
  month={aug},
  eprint={2608.26837},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2608.26837}
}
```

For the implementation:

```text
Chirinda, C. (2026). Interpretable Hybrid Credit Scoring for Thin-File and
Underbanked Populations: An Open Source Implementation. GitHub repository:
https://github.com/chirindaopensource/interpretable_hybrid_credit_scoring_for_thin_file_and_underbanked_populations
```

## Acknowledgments

- Belise Kanziga, Yaé U. Gaba, and Olivier Kanamugire for the research behind the residual-learning hybrid and interpretability-region fairness audit.
- Zindi and the Financial Sector Deepening Network for the Financial Inclusion in Africa data, and the UCI Machine Learning Repository for the Taiwan continuity data.
- Lundberg and Lee for TreeSHAP; DeLong, DeLong, and Clarke-Pearson; Sun and Xu; and Platt.
- The developers of scikit-learn, XGBoost, LightGBM, SHAP, SciPy, NumPy, Pandas, Matplotlib, and Jupyter.
