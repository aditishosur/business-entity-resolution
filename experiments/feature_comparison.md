# Matching feature investigation

## Validation status

The deterministic validation data referenced by the repository is not present in this checkout: no `train_source1.tsv` or `train_ground_truth.tsv` is available under the repository, its `dataset/`/`student_resource/` locations, or the accessible local Downloads and temporary directories. Consequently, no new metric is asserted here and the comparison CSV intentionally has blank metric fields. The pre-existing baseline artifact is retained unchanged; it was not represented as a newly reproduced result.

## Implemented candidates for the next local validation run

| Feature | Reason | Status | Precision safeguard |
|---|---|---|---|
| Trailing legal suffix core equality | Handles `Acme Ltd` / `Acme Limited` | Implemented, unvalidated | Only a classifier feature; never a direct match rule. Only trailing tokens are removed. |
| Word-order-insensitive name token-set equality | Handles reordered words | Implemented, unvalidated | Separate from legal-core equality; no direct acceptance. |
| Character trigram Dice similarity, name and address | Better explains small local edits than only edit ratio | Implemented, unvalidated | Lightweight set Dice feature; no candidate expansion. |
| House-number equality | Helps distinguish same-street candidates | Implemented, unvalidated | Postal token is excluded and this is classifier evidence only. |
| Existing exact name/address, postal, Jaccard, containment, edit similarity, country | Existing baseline signals | Retained | Existing exact-match override is unchanged. |
| Abbreviation/initial expansion | Could create broad collisions | Rejected pending data evidence | Not implemented. |
| Candidate/blocking expansion | Candidate recall must be assessed by blocking owner | Not changed | Avoids unmeasured runtime/memory and precision impact. |

Both `src.solve.py` and the production `src.solve_streaming.py`, as well as `src.evaluate.py`, use the same shared 15-value feature definition. Run the full evaluator with the supplied local dataset before selecting a threshold or deploying the added signals.
