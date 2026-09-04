# Data Provenance

Source records for the two datasets used by
*"Interpretable hybrid credit scoring for thin-file and underbanked
populations"* (Kanziga, Gaba & Kanamugire, 2026, arXiv:2608.26837).

Retrieved **2026-09-04**. Every transformation applied to a distributed file
is recorded below; no value in either corpus was altered.

---

## 1. Taiwan Credit Default — continuity benchmark

| | |
|---|---|
| **Source** | UCI Machine Learning Repository, dataset 350, *Default of Credit Card Clients* (Yeh & Lien, 2009) |
| **URL** | `https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip` |
| **Retrieved** | 2026-09-04, HTTP 200, 5,539,494 bytes |
| **Access** | Public. No authentication. |
| **Licence** | CC BY 4.0 (UCI ML Repository terms) |

### Files

| File | Bytes | SHA-256 |
|---|---|---|
| `default of credit card clients.xls` (source of record, as distributed) | 5,539,328 | `30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933` |
| `taiwan_credit_default.csv` (study-conformant) | 2,897,209 | `959736ebc535d84d4881eae09f54cd19cc9407a65ad4690d1c98d42f7a2cd6c2` |

The downloaded ZIP had SHA-256
`56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602`
and was removed after extraction; the extracted `.xls` above is retained.

### Transformations applied

The workbook carries a **two-row header**: row 0 holds the placeholder codes
`X1`…`X23`/`Y`, row 1 holds the real column names. It is therefore read with
`header=1`.

1. **Label column renamed.** UCI names it `default payment next month`
   (spaces); `study_config.json → raw_data_schemas.taiwan_credit_schema`
   declares the R-style `default.payment.next.month` (dots). The **name**
   changed; no value did.
2. **Columns reordered** to the declared schema order.
3. **Cast to `int64`** — every column in this dataset is integral by
   definition; the cast is lossless.

### Verification

- Shape **30,000 × 25** — matches the declared contract exactly.
- Zero missing values.
- `PAY_0` present, `PAY_1` absent, as the source specifies.
- **`validate_taiwan_raw` PASSES** — 30,000 checks recorded.

---

## 2. Zindi Financial Inclusion in Africa — primary corpus

| | |
|---|---|
| **Source** | Zindi competition *Financial Inclusion in Africa*, `Train.csv` |
| **Competition** | `https://zindi.africa/competitions/financial-inclusion-in-africa` |
| **Underlying survey** | FinScope surveys, 2016–2018, via the Financial Sector Deepening Network |
| **Coverage** | Kenya, Rwanda, Tanzania, Uganda |

### Access note — why this file did not come from Zindi directly

Zindi serves competition data only to an authenticated account that has
accepted the competition rules. The data endpoint returns **HTTP 401**
without credentials:

```
HTTP 401  <-  https://api.zindi.africa/v1/competitions/financial-inclusion-in-africa/files/Train.csv
```

The file was therefore obtained from public mirrors and **authenticated by
content** against the study's own published composition. If you hold a Zindi
account, downloading `Train.csv` directly and diffing it against
`zindi_financial_inclusion_raw.csv` is the recommended confirmation step.

### Cross-source corroboration

The file was downloaded from **three independent public repositories** and
compared:

| Mirror | Bytes | Content SHA-256 (newline-normalised) |
|---|---|---|
| `adivyas99/Financial-Inclusion-Zindi-Project` → `train.csv` | 2,864,059 | `8c81af3d6e49a3c8419728b945528c2faf23d36ce373c68f17fdfa87b824a888` |
| `datascientist-kenn/Zindi-Financial-Inclusion-in-Africa` → `Train_v2.csv` | 2,864,060 | `8c81af3d6e49a3c8419728b945528c2faf23d36ce373c68f17fdfa87b824a888` |
| `redhaam/Financial-inclusion-in-Africa` → `Train_v2.csv` | 2,864,060 | `8c81af3d6e49a3c8419728b945528c2faf23d36ce373c68f17fdfa87b824a888` |

Two are byte-identical; the third differs only by a trailing newline. All
three carry **identical content**.

### Content authentication against the published composition

Every element of the study's declared contract is reproduced exactly:

| Property | Contract | Observed | |
|---|---|---|---|
| Rows | 23,524 | 23,524 | MATCH |
| Columns | 13 | 13 | MATCH |
| Rwanda | 8,735 | 8,735 | MATCH |
| Tanzania | 6,620 | 6,620 | MATCH |
| Kenya | 6,068 | 6,068 | MATCH |
| Uganda | 2,101 | 2,101 | MATCH |
| Positive rate | 0.1408 | 0.140792 | MATCH |
| Missing values | 0 | 0 | MATCH |
| Verbatim typo `Divorced/Seperated` | required | present | MATCH |

Four exact country counts, the exact row count and the exact positive rate
cannot be matched by coincidence. This is the authentic, unmodified file.

### Files

| File | Bytes | SHA-256 |
|---|---|---|
| `zindi_financial_inclusion_raw.csv` (source of record, byte-faithful) | 2,864,060 | `05c0e0801e85eaccd1457012a2af155da77d1ebb3c48371949faea231741ed34` |
| `zindi_financial_inclusion.csv` (study-conformant) | 2,860,749 | `e3322fb761c6326888b62ff3e06413413bb7ad21f10027c9d83b1319744fe484` |

### Transformations applied

Exactly two, both lossless, applied only to the study-conformant file:

1. **Columns reordered** from the distributed order to
   `dataset_parameters.zindi_fsd.raw_column_order`. The study's own
   `validate_zindi_raw` reorders and records this rather than halting, so
   the reorder is sanctioned; doing it here merely avoids a spurious audit
   entry.
2. **`bank_account` encoded** `Yes → 1`, `No → 0`. Required: the declared
   contract is an integer label with *1 = owns formal bank account*, and
   `validate_zindi_raw` asserts an integer dtype. Counts are preserved
   exactly — `{No: 20212, Yes: 3312}` → `{0: 20212, 1: 3312}`.

**No categorical value was modified.** See the finding below.

---

## 3. FINDING — `study_config.json` declares categorical levels that the real Zindi data does not use

`validate_zindi_raw` **FAILS** on the authentic file:

```
ValueError: zindi_raw['marital_status'] has undeclared levels
            {'Single/Never Married', 'Dont know'}.
```

The validator halts on the first offending column, but **three** columns
diverge. The data is ground truth; the configuration is what is wrong. The
declared levels were evidently authored from the manuscript's prose rather
than from the distributed file, and they are self-consistent with the
notebook's synthetic corpus — which is why this never surfaced before.

### `marital_status` — 7,991 of 23,524 rows (34.0%)

| Level in data | Rows | Declared? |
|---|---|---|
| Married/Living together | 10,749 | yes |
| **Single/Never Married** | 7,983 | **no** — config declares `Single/Never married` (lower-case *m*) |
| Widowed | 2,708 | yes |
| Divorced/Seperated | 2,076 | yes |
| **Dont know** | 8 | **no** — config declares `Don't know/Refused` |

The 7,983-row mismatch is a **single character of letter case**.

### `education_level` — 838 rows (3.6%)

| Level in data | Rows | Declared? |
|---|---|---|
| Primary education | 12,791 | yes |
| No formal education | 4,515 | yes |
| Secondary education | 4,223 | yes |
| Tertiary education | 1,157 | yes |
| **Vocational/Specialised training** | 803 | **no** — no vocational category exists in the config |
| **Other/Dont know/RTA** | 35 | **no** — config declares `Other` and `Don't know/Refused` separately |

### `job_type` — 3,522 rows (15.0%)

| Level in data | Rows | Declared? |
|---|---|---|
| Self employed | 6,437 | yes |
| Informally employed | 5,597 | yes |
| Farming and Fishing | 5,441 | yes |
| Remittance Dependent | 2,527 | yes |
| **Other Income** | 1,080 | **no** |
| **Formally employed Private** | 1,055 | **no** — config declares `Formally employed Private sector` |
| **No Income** | 627 | **no** |
| **Formally employed Government** | 387 | **no** — config declares `Government` |
| **Government Dependent** | 247 | **no** |
| **Dont Know/Refuse to answer** | 126 | **no** |

**10,668 of 23,524 rows (45.3%) carry at least one undeclared level.**

### Blast radius beyond Task 1

The same three columns are mapped by two further configuration blocks, and
both are equally affected:

| Block | Consumer | Unmapped levels | Rows |
|---|---|---|---|
| `protected_attribute_parameters.education.raw_category_to_audit_bucket` | Task 15 fairness recoding | `Vocational/Specialised training`, `Other/Dont know/RTA` | 838 |
| `thin_file_parameters.education_ordinal_mapping` | Task 18 composite index | `Vocational/Specialised training`, `Other/Dont know/RTA` | 838 |
| `thin_file_parameters.formal_job_indicator_mapping` | Task 18 formal-job indicator | six job levels | 3,522 |

An unmapped level in the Task 18 mappings would silently produce `NaN` in
the thin-file index rather than raising, so this must be reconciled
deliberately — not left to fail closed at Task 1 alone.

### Required reconciliation

`study_config.json` must be updated to the real level sets before the
pipeline can run on real data. Four blocks need edits:

1. `dataset_parameters.zindi_fsd.categorical_levels` — `marital_status`,
   `education_level`, `job_type`
2. `protected_attribute_parameters.education.raw_category_to_audit_bucket`
   — add the two unmapped education levels, with a stated rationale for the
   bucket each is assigned to
3. `thin_file_parameters.education_ordinal_mapping` — assign an ordinal to
   `Vocational/Specialised training` (a genuine methodological decision:
   it is not obviously ordered against `Secondary education`) and to
   `Other/Dont know/RTA`
4. `thin_file_parameters.formal_job_indicator_mapping` — the six job
   levels, of which `Formally employed Government` and
   `Formally employed Private` are clearly formal employment and should
   map to 1

Items 2–4 are **substantive methodological choices**, not clerical fixes,
and each should be registered as an `UNSPECIFIED IN MANUSCRIPT` placeholder
with its rationale, consistent with the project's existing practice.

---

## Reproducing this download

```sh
# Taiwan (public, no authentication)
curl -sSL -o uci_taiwan.zip \
  "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
unzip uci_taiwan.zip

# Zindi (preferred: authenticated download from the competition page)
#   https://zindi.africa/competitions/financial-inclusion-in-africa/data
# then diff against zindi_financial_inclusion_raw.csv to confirm.
```

Converting the `.xls` requires `xlrd` (`pip install xlrd`); `openpyxl`
handles only `.xlsx`.
