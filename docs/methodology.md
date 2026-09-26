# ML Challenge 2026: Business Entity Resolution

**Team Name:** Codex local baseline  
**Team Members:** [add team members]  
**Submission Date:** 2026-09-26

## 1. Executive Summary

The solution combines country-aware blocking with an explainable pairwise logistic regression matcher. Names and addresses are Unicode-normalized, tokenized, and compared with exact, token-overlap, containment, and edit-similarity features. All processing uses only the supplied training and test files.

## 2. Methodology

### 2.1 Problem Analysis

The data contains independent records with noisy business names, partial or reordered addresses, punctuation variation, legal suffix variation, and missing components. Country is retained as an open string feature and is never restricted to the training countries.

### 2.2 Solution Strategy

**Approach Type:** Blocking + pairwise classifier  
**Core Innovation:** Multiple complementary blocking keys preserve recall while bounded block sizes prevent common names or addresses from creating an unmanageable Cartesian product.

## 3. Candidate Generation (Blocking)

Keys are built from country plus full normalized name, full normalized address, postal code, up to three longer name tokens, and up to four longer address tokens. Blocks larger than 300 records are discarded as non-discriminative. The final candidate set is the union of these keys and is written to `candidate_pairs.tsv`; final matches are always a subset of it.

## 4. Matching Model

Features include exact normalized name/address agreement, postal agreement, name and address token Jaccard overlap, token containment, character edit similarity, and country agreement. The model is a class-balanced logistic regression trained on candidate pairs labeled from `train_ground_truth.tsv`, with at most eight sampled negatives per Source 1 row. A precision-oriented probability threshold is applied at inference, with a small allowance for strong exact-field evidence.

## 5. Results & Error Analysis

The generated submission files should be validated with the supplied validator. The exact leaderboard score is unavailable locally because test labels are withheld. Expected false positives are common-name or shared-address collisions; expected false negatives are severe transliterations, highly incomplete addresses, and records whose useful tokens occur only in oversized blocks.

## 6. Conclusion

The pipeline is deterministic, auditable, and self-contained. It produces both the scored match file and the blocking audit file while respecting the no-external-lookup rule.

## Appendix

### A. Code Artefacts

The entry point is `code/business_entity_resolution/src/solve.py`. Dependencies are pinned in `requirements.txt`; the README contains reproduction and validator commands.

### B. Additional Results

The solver prints the number of output rows and fitted model coefficients. Local evaluation can be added by holding out a labeled Source 1 sample and applying the same `predict` function.
