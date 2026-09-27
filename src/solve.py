"""Local-only business entity resolution pipeline.

The implementation deliberately uses explainable string features and blocking
keys derived from the supplied data.  It does not perform external lookups.
"""
from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.matching_features import feature_row


COLS = ["entity_id", "business_name", "business_address", "country"]
TOKEN_RE = re.compile(r"[a-z0-9]+")
POSTAL_RE = re.compile(r"(?<!\d)(\d{5,6})(?!\d)")
COMMON = {"the", "and", "of", "for", "a", "an", "co", "company", "inc", "ltd", "llc", "limited", "corp", "corporation", "pvt", "private"}


def norm(value: object) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode().lower()
    text = text.replace("&", " and ")
    return " ".join(TOKEN_RE.findall(text))


def tokens(text: object) -> set[str]:
    if text is None or (not isinstance(text, str) and pd.isna(text)):
        return set()
    text = str(text)
    return set(text.split()) if text else set()


def postal(text: str) -> str:
    match = POSTAL_RE.search(text or "")
    return match.group(1) if match else ""


def read_records(path: Path, limit: int | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, nrows=limit)
    for col in COLS:
        if col not in frame:
            raise ValueError(f"Missing column {col} in {path}")
    frame = frame[COLS].copy()
    frame["name_n"] = frame.business_name.map(norm)
    frame["address_n"] = frame.business_address.map(norm)
    frame["country_n"] = frame.country.map(norm)
    frame["postal_n"] = frame.address_n.map(postal)
    frame["name_t"] = frame.name_n.map(tokens)
    frame["address_t"] = frame.address_n.map(tokens)
    return frame


def keys(row: pd.Series):
    country = row.country_n
    if row.name_n:
        yield "N|" + country + "|" + row.name_n
    if row.address_n:
        yield "A|" + country + "|" + row.address_n
    if row.postal_n:
        yield "P|" + country + "|" + row.postal_n
    for token in sorted((row.name_t - COMMON), key=lambda x: (-len(x), x))[:3]:
        if len(token) >= 4:
            yield "NT|" + country + "|" + token
    for token in sorted((row.address_t - COMMON), key=lambda x: (-len(x), x))[:4]:
        if len(token) >= 5:
            yield "AT|" + country + "|" + token


class CandidateIndex:
    def __init__(self, records: pd.DataFrame, max_bucket: int = 300):
        self.records = records.reset_index(drop=True)
        self.max_bucket = max_bucket
        buckets: dict[str, list[int]] = defaultdict(list)
        overflow: set[str] = set()
        for i, row in self.records.iterrows():
            for key in keys(row):
                if key in overflow:
                    continue
                bucket = buckets[key]
                bucket.append(i)
                if len(bucket) > max_bucket:
                    del buckets[key]
                    overflow.add(key)
        self.buckets = dict(buckets)

    def candidates(self, row: pd.Series) -> list[int]:
        result: set[int] = set()
        for key in keys(row):
            result.update(self.buckets.get(key, ()))
        return sorted(result)


def pair_features(left: pd.DataFrame, right: pd.DataFrame, pairs: list[tuple[int, int]]) -> np.ndarray:
    values = [feature_row(left.iloc[i], right.iloc[j]) for i, j in pairs]
    return np.asarray(values, dtype=np.float32) if values else np.empty((0, 15), dtype=np.float32)


class LogisticMatcher:
    """Small MIT-licensed-style implementation of pairwise logistic regression."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))

    def fit(self, X: np.ndarray, y: np.ndarray, steps: int = 500, learning_rate: float = 0.08):
        rng = np.random.default_rng(self.seed)
        self.coef_ = rng.normal(0.0, 0.01, X.shape[1]).astype(np.float64)
        self.intercept_ = 0.0
        positive = max(1, int(y.sum()))
        negative = max(1, len(y) - positive)
        weights = np.where(y == 1, len(y) / (2.0 * positive), len(y) / (2.0 * negative))
        for _ in range(steps):
            probability = self._sigmoid(X @ self.coef_ + self.intercept_)
            error = (probability - y) * weights
            self.coef_ -= learning_rate * ((X.T @ error) / len(y) + 0.001 * self.coef_)
            self.intercept_ -= learning_rate * float(error.mean())
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted")
        positive = self._sigmoid(X @ self.coef_ + self.intercept_)
        return np.column_stack((1.0 - positive, positive))


def truth_map(path: Path) -> dict[str, set[str]]:
    gt = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return {row.source1_entity_id: set(filter(None, row.matched_entity_ids.split(","))) for row in gt.itertuples()}


def fit_model(train_dir: Path, sample_size: int, seed: int):
    source1 = read_records(train_dir / "train_source1.tsv", sample_size)
    source2 = read_records(train_dir / "train_source2.tsv")
    source3 = read_records(train_dir / "train_source3.tsv")
    other = pd.concat([source2, source3], ignore_index=True)
    index = CandidateIndex(other)
    truth = truth_map(train_dir / "train_ground_truth.tsv")
    rng = np.random.default_rng(seed)
    pairs: list[tuple[int, int]] = []
    labels: list[int] = []
    for i, row in source1.iterrows():
        candidates = index.candidates(row)
        wanted = truth.get(row.entity_id, set())
        positive = {j for j in candidates if other.iloc[j].entity_id in wanted}
        for j in positive:
            pairs.append((i, j)); labels.append(1)
        negatives = [j for j in candidates if j not in positive]
        if len(negatives) > 8:
            negatives = list(rng.choice(negatives, size=8, replace=False))
        for j in negatives:
            pairs.append((i, int(j))); labels.append(0)
    X = pair_features(source1, other, pairs)
    y = np.asarray(labels, dtype=np.int8)
    if len(np.unique(y)) < 2:
        raise RuntimeError("Training sample did not contain both positive and negative candidate pairs")
    model = LogisticMatcher(seed=seed).fit(X, y)
    return model


def predict(model, source1: pd.DataFrame, other: pd.DataFrame, index: CandidateIndex, threshold: float):
    matching: list[tuple[str, str]] = []
    candidates_out: list[tuple[str, str]] = []
    for i, row in source1.iterrows():
        candidates = index.candidates(row)
        candidates_out.append((row.entity_id, ",".join(other.iloc[j].entity_id for j in candidates)))
        if not candidates:
            # The submission requires exactly one matching row for every Source 1 row.
            matching.append((row.entity_id, ""))
            continue
        pairs = [(i, j) for j in candidates]
        probabilities = model.predict_proba(pair_features(source1, other, pairs))[:, 1]
        chosen = []
        for j, p in zip(candidates, probabilities):
            b = other.iloc[j]
            f = feature_row(row, b)
            # Exact agreement on a non-empty name or address is a strong, auditable signal.
            exact = f[0] or f[1]
            if p >= threshold or (exact and p >= max(0.50, threshold - 0.20)):
                chosen.append((float(p), b.entity_id))
        chosen.sort(reverse=True)
        matching.append((row.entity_id, ",".join(entity_id for _, entity_id in chosen)))
    return matching, candidates_out


def write_pairs(path: Path, header: tuple[str, str], rows: list[tuple[str, str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True, help="student_resource/dataset")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-sample", type=int, default=50000)
    parser.add_argument("--threshold", type=float, default=0.86)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    model = fit_model(args.data_dir / "train", args.train_sample, args.seed)
    source1 = read_records(args.data_dir / "test/test_source1.tsv")
    source2 = read_records(args.data_dir / "test/test_source2.tsv")
    source3 = read_records(args.data_dir / "test/test_source3.tsv")
    other = pd.concat([source2, source3], ignore_index=True)
    index = CandidateIndex(other)
    matching, candidates = predict(model, source1, other, index, args.threshold)
    write_pairs(args.output_dir / "matching_results.tsv", ("source1_entity_id", "matched_entity_ids"), matching)
    write_pairs(args.output_dir / "candidate_pairs.tsv", ("source1_entity_id", "candidate_entity_ids"), candidates)
    print(f"wrote {len(matching):,} matching rows and {len(candidates):,} candidate rows")
    print("model coefficients:", np.round(model.coef_, 4).tolist())


if __name__ == "__main__":
    main()
