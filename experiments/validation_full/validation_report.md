# Validation Report

## Dataset and split

- Source 1 records used: 2,206,821
- Training Source 1 records: 1,765,457
- Validation Source 1 records: 441,364
- Validation fraction: 20.00%
- Random seed: 42
- Source 1 sample limit: all available records

## Evaluation implementation

- Evaluation uses the repository's SQLite-backed streaming architecture.
- Source 2 and Source 3 remain the complete candidate universe.
- Candidate blocking uses country + exact name, country + exact address, and country + postal.
- Each blocking query is capped at 300 candidates.
- Model training uses only the training Source 1 split.
- Validation ground truth is never used during model fitting.

## Metrics

- Precision, recall and F0.5 are calculated per Source 1 entity and macro-averaged.
- Singleton accuracy measures whether entities with no true matches receive an empty prediction.
- Candidate recall measures whether true matches survive candidate generation.

 Candidate recall interpretation: The reported candidate recall of 0.353037 is calculated as a macro average over non-singleton Source 1 entities (entities with at least one ground-truth match). Final recall, in contrast, is averaged across all Source 1 entities, including singleton entities. Therefore, candidate recall and final recall are not directly equivalent, and final recall can be slightly higher than candidate recall.

## Threshold comparison

| Threshold | Precision | Recall | Macro F0.5 | Singleton Accuracy | Predictions | Candidate Recall |
|---:|---:|---:|---:|---:|---:|---:|
| 0.70 | 0.680628 | 0.361964 | 0.542397 | 0.950556 | 290,434 | 0.353037 |
| 0.75 | 0.677369 | 0.355453 | 0.537752 | 0.956601 | 287,141 | 0.353037 |
| 0.80 | 0.670030 | 0.346341 | 0.529404 | 0.960066 | 282,907 | 0.353037 |
| 0.85 | 0.659580 | 0.334681 | 0.517930 | 0.962847 | 277,531 | 0.353037 |
| 0.86 | 0.656730 | 0.331762 | 0.514896 | 0.963209 | 276,185 | 0.353037 |
| 0.90 | 0.640205 | 0.316078 | 0.497699 | 0.965949 | 268,157 | 0.353037 |
| 0.95 | 0.614141 | 0.297219 | 0.473356 | 0.968206 | 256,202 | 0.353037 |

## Measured validation result

- Threshold with highest measured Macro F0.5: **0.70**
- Corresponding Macro F0.5: **0.542397**

## Limitations

- These are validation results from the supplied training ground truth.
- They are not test-set results because the challenge test set has no public ground truth.
- Candidate recall is a blocking-stage ceiling: a true match excluded by blocking cannot be recovered by thresholding.
- If a Source 1 sample was used, the results describe that deterministic validation sample rather than the entire Source 1 training population.
