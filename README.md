# Business Entity Resolution

This is a local-only, reproducible pipeline for the supplied ML Challenge 2026 data. It uses normalized names and addresses, country-aware blocking, token and edit-similarity features, and a logistic regression matcher trained from the supplied training ground truth. It performs no external data lookup.

## Run

From the `student_resource` directory:

```bash
python -m pip install -r code/business_entity_resolution/requirements.txt
python code/business_entity_resolution/src/solve.py \
  --data-dir dataset \
  --output-dir output
```

The default training sample is 50,000 Source 1 records. Increase `--train-sample` when more memory and time are available. The two required files are written to `output/`.

Validate them with:

```bash
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

The pipeline treats country as an arbitrary string label, so unseen test countries remain eligible for matching. Its blocking keys are full normalized name, full normalized address, postal code, and rare name/address tokens within country. Candidate lists are exactly the records passed to the final scorer.
