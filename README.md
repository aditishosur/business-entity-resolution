# Business Entity Resolution

ML Challenge 2026 solution for resolving noisy business records across three independent sources.

## Project Status

The project contains:

- A baseline in-memory solver.
- A memory-safe SQLite-backed streaming solver.
- Country-aware candidate generation.
- Name and address normalization.
- Similarity-based matching.
- Singleton handling.
- Output-format validation instructions.
- Reproducible PowerShell commands.

The streaming solver should be used for the complete dataset because the raw challenge data contains millions of records.

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

The evaluation metric is macro `F_0.5`, which penalizes false positives more heavily than false negatives.

## Repository Structure

```text
business-entity-resolution/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── src/
│   ├── solve.py
│   └── solve_streaming.py
├── tests/
├── docs/
├── experiments/
└── output/
