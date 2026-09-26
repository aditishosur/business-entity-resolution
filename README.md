# Business Entity Resolution

AWS ML Challenge 2026 solution for resolving noisy business records across three independent sources.

## Project Status

The project contains:

- A baseline in-memory solver.
- A memory-safe SQLite-backed streaming solver.
- Country-aware candidate generation.
- Name and address normalization.
- Similarity-based matching.
- Singleton handling.
- Evaluation and threshold-selection workflow.
- Validation metric tests.
- Output-format validation instructions.
- Reproducible PowerShell commands.

> The streaming solver should be used for the complete dataset because the raw challenge data contains millions of records.

### Current Evaluation Status

The evaluation workflow has been implemented on a deterministic **80/20 train/validation split** of Source 1 using seed `42`.

The validation workflow currently:

- Trains the matcher only on the training portion.
- Keeps the validation ground truth withheld during model fitting.
- Uses the complete Source 2 and Source 3 candidate universe.
- Generates candidates using country-aware exact normalized name, address, and postal-code blocking.
- Computes precision, recall, macro `F_0.5`, singleton accuracy, and candidate recall.
- Sweeps multiple decision thresholds.
- Produces validation results, threshold comparisons, and a reproducibility report.

The evaluated thresholds are: `0.70, 0.75, 0.80, 0.85, 0.86, 0.90, 0.95`

---

## Problem

For every Source 1 business, identify all matching records from Source 2 and Source 3.

The data contains:

- No shared business identifier.
- Typos and punctuation differences.
- Abbreviations and legal suffixes.
- Address variations.
- Missing address components.
- Transliteration differences.
- Multiple possible matches.
- Singleton Source 1 records with no matches.
- An unseen country in the test data, including France.

The evaluation metric is **macro F₀.₅**, which penalizes false positives more heavily than false negatives.

---

## Evaluation & Validation

### Train / Validation Split

Source 1 is split deterministically into:

- 80% training
- 20% validation
- Random seed: `42`

The split is performed at the Source 1 entity level, while the complete ground-truth match sets are preserved.

Source 2 and Source 3 records remain available as the candidate universe. Validation ground truth is not used for model fitting.

### Metrics

The evaluation workflow reports:

- Precision
- Recall
- Macro F₀.₅
- Singleton accuracy
- Number of predictions
- Candidate recall

For each Source 1 entity with ground-truth matches, F₀.₅ is calculated separately and then macro-averaged.

The metric is:

```
F_0.5 = (1.25 × precision × recall) / (0.25 × precision + recall)
```

### Threshold Sweep

The current validation evaluates the following thresholds:

| Threshold | Precision | Recall   | Macro F₀.₅ | Singleton Accuracy | Predictions | Candidate Recall |
|-----------|-----------|----------|------------|---------------------|-------------|-------------------|
| 0.70      | 0.680628  | 0.361964 | 0.542397   | 0.950556            | 290,434     | 0.353037          |
| 0.75      | 0.677369  | 0.355453 | 0.537752   | 0.956601            | 287,141     | 0.353037          |
| 0.80      | 0.670030  | 0.346341 | 0.529404   | 0.960066            | 282,907     | 0.353037          |
| 0.85      | 0.659580  | 0.334681 | 0.517930   | 0.962847            | 277,531     | 0.353037          |
| 0.86      | 0.656730  | 0.331762 | 0.514896   | 0.963209            | 276,185     | 0.353037          |
| 0.90      | 0.640205  | 0.316078 | 0.497699   | 0.965949            | 268,157     | 0.353037          |
| 0.95      | 0.614141  | 0.297219 | 0.473356   | 0.968206            | 256,202     | 0.353037          |

Under the current model and blocking configuration, **0.70** produced the highest measured Macro F₀.₅ on the validation split, with a score of **0.542397**.

> This is a validation result rather than a universal threshold recommendation. Any subsequent model or blocking changes should be re-evaluated.

### Validation Artifacts

The evaluation workflow produces:

```
experiments/
└── validation_full/
    ├── validation_results.csv
    ├── threshold_comparison.csv
    └── validation_report.md
```

**`validation_results.csv`**
Contains per-Source-1 validation results, including prediction and candidate-recall information.

> The full validation file is generated locally but is excluded from the Git repository because it is approximately 125 MB, exceeding GitHub's standard 100 MB file limit.

**`threshold_comparison.csv`**
Contains the aggregate metric results for every evaluated threshold.

**`validation_report.md`**
Contains:

- Validation methodology.
- Dataset and split configuration.
- Threshold comparison.
- Final measured validation result.
- Reproducibility information.
- Candidate-generation details.
- Limitations and validation notes.

### Validation Tests

Metric behavior is covered by automated tests for:

- Singleton correct prediction.
- Singleton incorrect prediction.
- Correct single-match prediction.
- Partial multi-match prediction.
- Extra prediction / false-positive handling.
- Macro F₀.₅.
- Singleton accuracy.
- Candidate recall.

Current evaluation test status: **8 passed**

Run the tests with:

```bash
pytest tests/test_evaluation.py
```

---

## Repository Structure

```
business-entity-resolution/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── src/
│   ├── solve.py
│   ├── solve_streaming.py
│   └── evaluate.py
├── tests/
│   └── test_evaluation.py
├── docs/
├── experiments/
│   ├── analyze_validation.py
│   └── validation_full/
│       ├── threshold_comparison.csv
│       └── validation_report.md
└── output/
```

---

## Running the Streaming Solver

For the complete challenge dataset, use the streaming solver rather than the baseline in-memory implementation.

```powershell
python -m src.solve_streaming `
  --data-dir "<path-to-dataset>" `
  --output-dir "output"
```

The streaming implementation uses a temporary SQLite-backed index to avoid loading the complete Source 2 and Source 3 datasets into memory simultaneously.

## Running Validation

The complete validation workflow can be run with:

```powershell
python -m src.evaluate `
  --data-dir "<path-to-dataset>" `
  --output-dir "experiments\validation_full" `
  --source1-sample 0 `
  --chunk-size 50000
```

For a smaller smoke test during development:

```powershell
python -m src.evaluate `
  --data-dir "<path-to-dataset>" `
  --output-dir "experiments\validation_smoke" `
  --source1-sample 5000 `
  --chunk-size 50000
```

The temporary SQLite validation index is removed after the evaluation completes.

---

## Reproducibility

The recorded full-validation environment includes:

| Item                    | Value                                       |
|-------------------------|----------------------------------------------|
| Python                  | 3.12.10                                       |
| OS                      | Windows 11                                    |
| CPU                     | Intel64 Family 6 Model 186 Stepping 3         |
| Branch                  | `feature/evaluation`                          |
| Validation commit       | `59db0bd4afc32ae7fab9a5c1ac414e9891db1b07`    |
| Split seed              | 42                                             |
| Train/validation split  | 80/20                                          |

> The exact validation runtime was not captured during the full run.

### Current Validation Result

The current full validation run produced:

| Metric              | Value       |
|----------------------|-------------|
| Threshold             | 0.70        |
| Precision              | 0.680628    |
| Recall                | 0.361964    |
| Macro F₀.₅             | 0.542397    |
| Singleton accuracy    | 0.950556    |
| Predictions           | 290,434     |
| Candidate recall      | 0.353037    |

These results describe the current model and candidate-generation configuration and should be rechecked after any changes to matching, features, or blocking.

---

## Notes

The validation workflow is intended to support model and threshold decisions before the final challenge pipeline is executed.

The final submission must still be validated against the complete test pipeline and the challenge's required output format before submission.