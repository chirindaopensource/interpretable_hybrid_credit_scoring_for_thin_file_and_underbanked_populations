# Configuration Reconciliation Proposal

**Status:** DRAFT FOR REVIEW — nothing has been applied.
**Original:** `study_config.json` — untouched (47,058 bytes).
**Candidate:** `study_config.reconciled.json` — for diff and review (49,109 bytes).
**Date:** 2026-09-04

---

## Acceptance test

| Validator | Original config | Reconciled candidate |
|---|---|---|
| `validate_study_config` | PASS (48 checks) | PASS (48 checks) |
| `validate_taiwan_raw` | PASS (30,000 checks) | PASS (30,000 checks) |
| `validate_zindi_raw` | **FAIL** — `marital_status` undeclared levels | **PASS (23,524 checks)** |

The candidate unblocks Task 1 on the real corpus. It does **not** on its own
make the study reproduce the paper; two independent obstacles remain, in
Findings A and B below.

---

## Principle applied

Only what the data actually violates is changed. `relationship_with_head`
declares a level (`Don't know/Refused`) that the corpus never uses;
`validate_zindi_raw` requires observed ⊆ declared, so that is harmless and is
**left alone**. Three columns are changed because the data carries levels the
configuration does not declare.

No data value was altered anywhere. Where the manuscript supplies a numeric
target, it is used as a constraint and the result is reported; where it does
not, or where the target fails to discriminate, the choice is made
semantically and registered as a placeholder.

---

## Item 1 — `dataset_parameters.zindi_fsd.categorical_levels`

**Nature: clerical. Determined exactly by the corpus. No judgement.**

### `marital_status`

| Declared | Observed | |
|---|---|---|
| `Single/Never married` | `Single/Never Married` | **7,983 rows** — differs by one character of case |
| `Don't know/Refused` | `Dont know` | 8 rows |
| `Married/Living together`, `Widowed`, `Divorced/Seperated` | same | unchanged |

### `education_level`

| Declared | Observed |
|---|---|
| `Other`, `Don't know/Refused` | `Other/Dont know/RTA` (35 rows) — merged in source |
| *(absent)* | `Vocational/Specialised training` (**803 rows**) — no counterpart in the config |

### `job_type`

| Declared | Observed |
|---|---|
| `Government` | `Formally employed Government` (387) |
| `Formally employed Private sector` | `Formally employed Private` (1,055) |
| `Other` | `Other Income` (1,080) |
| `Don't know/Refused` | `Dont Know/Refuse to answer` (126) |
| *(absent)* | `Government Dependent` (247), `No Income` (627) |

**Proposed:** replace all three level lists with the observed sets.
**Rows unblocked:** 10,668 of 23,524 (45.3%).

---

## Item 2 — `protected_attribute_parameters.education.raw_category_to_audit_bucket`

**Nature: empirically determined for the load-bearing level.**

Two levels need an audit bucket. The manuscript's Table 4 declares the test
cells `secondary = 1283`, `primary = 3422`, which discriminates the choice:

| Assignment for `Vocational/Specialised training` | secondary | primary | error |
|---|---|---|---|
| `primary or less` | 1,110 | 3,595 | **173** |
| **`secondary or above`** | **1,262** | **3,443** | **21** |

The two options differ by 152 observations while the split noise is at most
76 (Finding A). The gap is 8×, so the determination is robust even though
neither lands inside the ±5 tolerance.

`Other/Dont know/RTA` carries 35 rows and moves the count by 5 — below the
resolution of this constraint. It follows the convention the configuration
already applied to `Other` and `Don't know/Refused`, both of which mapped to
`secondary or above`.

**Proposed:**

```json
"raw_category_to_audit_bucket": {
  "No formal education":              "primary or less",
  "Primary education":               "primary or less",
  "Secondary education":             "secondary or above",
  "Tertiary education":              "secondary or above",
  "Vocational/Specialised training":  "secondary or above",
  "Other/Dont know/RTA":              "secondary or above"
}
```

Registered as `UNSPECIFIED IN MANUSCRIPT` with the evidence above.

---

## Item 3 — `thin_file_parameters.education_ordinal_mapping`

**Nature: semantic, with a consistency check.**

`Vocational/Specialised training` → **2**. It is post-primary technical
training, and Item 2 places it in `secondary or above` on empirical grounds;
assigning ordinal 2 keeps the two blocks coherent. Assigning 3 would rank it
above `Tertiary education`, which the source does not support.

`Other/Dont know/RTA` → **0**, the conservative reading: an unknown
educational attainment is treated as the thinnest file rather than imputed
upward. The original configuration mapped `Don't know/Refused` to 0.

**Proposed:**

```json
"education_ordinal_mapping": {
  "No formal education": 0,
  "Primary education": 1,
  "Secondary education": 2,
  "Tertiary education": 3,
  "Vocational/Specialised training": 2,
  "Other/Dont know/RTA": 0
}
```

---

## Item 4 — `thin_file_parameters.formal_job_indicator_mapping`

**Nature: semantic. Four of six are direct renames of existing categories.**

| Level | Value | Basis |
|---|---|---|
| `Formally employed Government` | **1** | Zindi spelling of the config's `Government` |
| `Formally employed Private` | **1** | Zindi spelling of `Formally employed Private sector` |
| `Government Dependent` | **0** | dependency, not employment |
| `Other Income` | **0** | config mapped `Other` → 0 |
| `No Income` | **0** | absence of employment |
| `Dont Know/Refuse to answer` | **0** | config mapped `Don't know/Refused` → 0 |

**The declared subgroup size does not determine this mapping.** Under
inclusive tie handling (Item 5), **79 of 504** candidate mappings satisfy
n = 1559 ± 20. The target is therefore used only as a consistency check, and
the assignment is settled semantically. Registered as a placeholder.

---

## Item 5 — Thin-file threshold tie handling

**Nature: an implementation defect that the real data exposes. This is the
most consequential item in the proposal.**

The composite index sums three **discrete** components — education ordinal
0–3, cellphone 0/1, formal-job 0/1 — so it takes only **15 distinct values**
across 4,705 test rows. **496 test rows sit exactly at τ.** The comparison
operator therefore dominates the subgroup size:

| Rule | n | share | positive rate |
|---|---|---|---|
| `index < tau` — **as implemented** (line 14287) | 1,053 | 0.224 | 0.0171 |
| `index <= tau` — inclusive | **1,553** | **0.330** | 0.0309 |
| **Manuscript target** | **1,559** | **0.331** | **0.024** |

Under the strict rule the target is **unreachable by 497 observations across
all 512 candidate mappings** (achievable range 918–1,062). Under the
inclusive rule it is reached.

The manuscript's own declared subgroup size therefore requires the
**inclusive** rule. The configuration says only "bottom quartile of the
composite index" and does not state a tie rule; the implementation chose
strict, and that choice is wrong against the paper's numbers.

**Proposed:**

1. Amend the definition to state the rule explicitly:
   `"bottom quartile of the composite index, threshold applied INCLUSIVELY (index <= tau)"`
2. Add a `tie_handling` note recording the 496-row tie and the two counts.
3. **Change `apply_threshold_and_compute` at line 14287** from
   `test_index < tau` to `test_index <= tau`.

Item 5 is a **code change**, not only a config change, and is the one item
here that alters computed results.

---

## Finding A — Table 4 cell counts are not reproducible at any seed

Not part of the proposal; reported because it blocks `assert_cell_counts`
independently of everything above.

The split **sizes** reproduce exactly (14,114 / 4,705 / 4,705). The test
partition's **composition** does not:

| Cell | Target | Observed (config seed) | Δ |
|---|---|---|---|
| male / female | 1,913 / 2,792 | 1,941 / 2,764 | ±28 |
| urban / rural | 1,809 / 2,896 | 1,878 / 2,827 | ±69 |
| Kenya | 1,209 | 1,189 | −20 |
| Rwanda | 1,746 | 1,755 | +9 |
| Tanzania | 1,277 | 1,353 | **+76** |
| Uganda | 473 | 408 | −65 |

These cells depend on **no disputed mapping**, so the discrepancy is in the
split itself. Searching 404 seeds (0–400 plus 20260830, 42, 2026, 830) found
none within the ±5 tolerance; the best is seed 357 at max error 19, still
~4× tolerance.

The configured seed `20260830` reads as the date 2026-08-30 — the task-list
timestamp — and is almost certainly an implementation placeholder rather than
the paper's seed, which the manuscript does not state. A plain chained
`StratifiedShuffleSplit` may also not be the paper's procedure.

**Consequence:** `assert_cell_counts` (Task 15) will fail on real data
regardless of this proposal. It should be re-registered as unverifiable
under a placeholder seed, and the seed itself flagged
`UNSPECIFIED IN MANUSCRIPT` — consistent with the verifier redesign already
adopted, where a manuscript-value mismatch is recorded as a finding rather
than raised as a fault.

---

## Finding B — the thin-file positive rate still misses

With Item 5 applied and the recommended mappings:

| Target | Tolerance | Achieved | |
|---|---|---|---|
| n = 1,559 | ±20 | 1,553 | **within** |
| share = 0.331 | ±1 pp | 0.330 | **within** |
| positive rate = 0.024 | ±0.5 pp | 0.0309 | **outside by 0.7 pp** |

Two of three land. The residual is consistent in magnitude with the split
discrepancy in Finding A — the subgroup is the right size but drawn from a
differently-composed test partition. It is unlikely to close without the
paper's actual split procedure.

---

## Recommended order of adoption

1. **Item 1** — clerical, zero risk, unblocks 45.3% of rows. Adopt first.
2. **Items 2–4** — mapping completions. Adopt with the placeholder registrations.
3. **Item 5** — code change at line 14287 plus the definition amendment.
   Review deliberately: it changes a reported result, moving the thin-file
   subgroup from 22.4% to 33.0% of test.
4. **Findings A and B** — re-register `assert_cell_counts` and the thin-file
   positive-rate assertion as unverifiable under the placeholder seed, and
   record the seed as `UNSPECIFIED IN MANUSCRIPT`.

## How to review the diff

```sh
python -c "import json,io; a=json.load(io.open('study_config.json',encoding='utf-8')); b=json.load(io.open('study_config.reconciled.json',encoding='utf-8')); print('identical' if a==b else 'differs')"
```

To adopt, back up and replace:

```sh
cp study_config.json study_config.json.bak
cp study_config.reconciled.json study_config.json
```
