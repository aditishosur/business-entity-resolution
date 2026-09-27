"""Memory-bounded entity-resolution pipeline backed by SQLite.

Only a bounded pandas chunk and the current Source 1 batch are held in memory.
All business data used by the matcher comes from the supplied TSV files.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import os
import re
import sqlite3
import time
import unicodedata
from ctypes import wintypes
from pathlib import Path

import numpy as np
import pandas as pd

from src.solve import LogisticMatcher
from src.matching_features import feature_row


TOKEN_RE = re.compile(r"[a-z0-9]+")
POSTAL_RE = re.compile(r"(?<!\d)(\d{5,6})(?!\d)")
COLS = ["entity_id", "business_name", "business_address", "country"]


def memory_gb() -> float:
    """Return this process's resident memory using only the standard library."""
    if os.name == "nt":
        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("page_fault_count", ctypes.c_ulong),
                        ("peak_working_set", ctypes.c_size_t), ("working_set", ctypes.c_size_t),
                        ("quota_peak_paged_pool", ctypes.c_size_t), ("paged_pool", ctypes.c_size_t),
                        ("quota_peak_non_paged_pool", ctypes.c_size_t), ("non_paged_pool", ctypes.c_size_t),
                        ("pagefile", ctypes.c_size_t), ("peak_pagefile", ctypes.c_size_t)]
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        get_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        get_info.restype = wintypes.BOOL
        if get_info(handle, ctypes.byref(counters), counters.cb):
            return counters.working_set / (1024 ** 3)
        return 0.0
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    except ImportError:
        return 0.0


def log(message: str, started: float):
    print(f"[{time.strftime('%H:%M:%S')}] {message} | elapsed={time.monotonic() - started:.1f}s | memory={memory_gb():.2f}GB", flush=True)


def norm(value: object) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().lower()
    return " ".join(TOKEN_RE.findall(text.replace("&", " and ")))


def postal(text: str) -> str:
    match = POSTAL_RE.search(text or "")
    return match.group(1) if match else ""


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[COLS].copy()
    frame["name_n"] = frame["business_name"].map(norm)
    frame["address_n"] = frame["business_address"].map(norm)
    frame["country_n"] = frame["country"].map(norm)
    frame["postal_n"] = frame["address_n"].map(postal)
    return frame


def init_db(path: Path) -> sqlite3.Connection:
    if path.exists():
        path.unlink()
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("""CREATE TABLE records (
        row_id INTEGER PRIMARY KEY,
        entity_id TEXT NOT NULL UNIQUE,
        name_n TEXT NOT NULL,
        address_n TEXT NOT NULL,
        country_n TEXT NOT NULL,
        postal_n TEXT NOT NULL
    )""")
    connection.execute("CREATE INDEX idx_name ON records(country_n, name_n)")
    connection.execute("CREATE INDEX idx_address ON records(country_n, address_n)")
    connection.execute("CREATE INDEX idx_postal ON records(country_n, postal_n)")
    return connection


def build_db(paths: list[Path], db_path: Path, chunk_size: int, started: float, label: str):
    connection = init_db(db_path)
    total = 0
    try:
        for path in paths:
            for frame in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                                     usecols=COLS, chunksize=chunk_size):
                frame = normalize_frame(frame)
                rows = frame[["entity_id", "name_n", "address_n", "country_n", "postal_n"]].itertuples(index=False, name=None)
                connection.executemany(
                    "INSERT INTO records(entity_id,name_n,address_n,country_n,postal_n) VALUES (?,?,?,?,?)",
                    rows,
                )
                connection.commit()
                total += len(frame)
                if total % (chunk_size * 5) == 0:
                    log(f"{label}: indexed {total:,} records", started)
        log(f"{label}: indexed {total:,} records", started)
        return connection
    except Exception:
        connection.close()
        raise


def source1_rows(path: Path, chunk_size: int, limit: int | None = None):
    read = 0
    for frame in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                             usecols=COLS, chunksize=chunk_size):
        if limit is not None and read + len(frame) > limit:
            frame = frame.iloc[: limit - read]
        if len(frame) == 0:
            break
        yield normalize_frame(frame)
        read += len(frame)
        if limit is not None and read >= limit:
            break


def record_from_row(row: pd.Series) -> dict:
    return {
        "entity_id": row.entity_id,
        "name_n": row.name_n,
        "address_n": row.address_n,
        "country_n": row.country_n,
        "postal_n": row.postal_n,
    }


def candidates(connection: sqlite3.Connection, row: dict) -> list[dict]:
    ids: set[int] = set()
    queries = [
        ("SELECT row_id FROM records WHERE country_n=? AND name_n=? AND name_n<>'' LIMIT 300", (row["country_n"], row["name_n"])),
        ("SELECT row_id FROM records WHERE country_n=? AND address_n=? AND address_n<>'' LIMIT 300", (row["country_n"], row["address_n"])),
        ("SELECT row_id FROM records WHERE country_n=? AND postal_n=? AND postal_n<>'' LIMIT 300", (row["country_n"], row["postal_n"])),
    ]
    for query, params in queries:
        ids.update(item[0] for item in connection.execute(query, params))
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    rows = connection.execute(
        f"SELECT row_id,entity_id,name_n,address_n,country_n,postal_n FROM records WHERE row_id IN ({placeholders})",
        tuple(ids),
    ).fetchall()
    return [{"row_id": item[0], "entity_id": item[1], "name_n": item[2], "address_n": item[3],
             "country_n": item[4], "postal_n": item[5]} for item in rows]


def ground_truth(path: Path, ids: set[str]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for frame in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=100_000):
        selected = frame[frame.source1_entity_id.isin(ids)]
        for row in selected.itertuples(index=False):
            result[row.source1_entity_id] = set(filter(None, row.matched_entity_ids.split(",")))
    return result


def train_model(train_dir: Path, train_sample: int, chunk_size: int, db_dir: Path, seed: int, started: float):
    source1 = pd.concat(list(source1_rows(train_dir / "train_source1.tsv", chunk_size, train_sample)), ignore_index=True)
    truth = ground_truth(train_dir / "train_ground_truth.tsv", set(source1.entity_id))
    db_path = db_dir / "train.sqlite"
    connection = build_db([train_dir / "train_source2.tsv", train_dir / "train_source3.tsv"], db_path, chunk_size, started, "train")
    rng = np.random.default_rng(seed)
    pairs: list[list[float]] = []
    labels: list[int] = []
    for number, (_, series) in enumerate(source1.iterrows(), 1):
        left = record_from_row(series)
        found = candidates(connection, left)
        wanted = truth.get(left["entity_id"], set())
        positives = [item for item in found if item["entity_id"] in wanted]
        negatives = [item for item in found if item["entity_id"] not in wanted]
        if len(negatives) > 8:
            negatives = [negatives[i] for i in rng.choice(len(negatives), 8, replace=False)]
        for item in positives:
            pairs.append(feature_row(left, item)); labels.append(1)
        for item in negatives:
            pairs.append(feature_row(left, item)); labels.append(0)
        if number % 5000 == 0:
            log(f"train: built pairs for {number:,} Source 1 rows", started)
    connection.close()
    db_path.unlink(missing_ok=True)
    X = np.asarray(pairs, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int8)
    if len(y) == 0 or len(np.unique(y)) < 2:
        raise RuntimeError("Training sample did not produce both positive and negative candidate pairs")
    model = LogisticMatcher(seed=seed).fit(X, y)
    log(f"train: fitted matcher on {len(y):,} pairs", started)
    return model


def write_outputs(model, source1_path: Path, source2_path: Path, source3_path: Path, output_dir: Path,
                  db_dir: Path, chunk_size: int, limit: int | None, threshold: float, started: float):
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / "test.sqlite"
    connection = build_db([source2_path, source3_path], db_path, chunk_size, started, "test")
    matching_path = output_dir / "matching_results.tsv"
    candidate_path = output_dir / "candidate_pairs.tsv"
    processed = total_candidates = total_matches = 0
    with matching_path.open("w", newline="", encoding="utf-8") as matching_handle, candidate_path.open("w", newline="", encoding="utf-8") as candidate_handle:
        matching_writer = csv.writer(matching_handle, delimiter="\t", lineterminator="\n")
        candidate_writer = csv.writer(candidate_handle, delimiter="\t", lineterminator="\n")
        matching_writer.writerow(["source1_entity_id", "matched_entity_ids"])
        candidate_writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for frame in source1_rows(source1_path, chunk_size, limit):
            for _, series in frame.iterrows():
                left = record_from_row(series)
                found = candidates(connection, left)
                candidate_ids = [item["entity_id"] for item in found]
                selected: list[tuple[float, str]] = []
                if found:
                    feature_values = np.asarray([feature_row(left, item) for item in found], dtype=np.float32)
                    probabilities = model.predict_proba(feature_values)[:, 1]
                    for item, probability in zip(found, probabilities):
                        exact = bool(item["name_n"] and item["name_n"] == left["name_n"]) or bool(item["address_n"] and item["address_n"] == left["address_n"])
                        if probability >= threshold or (exact and probability >= max(0.50, threshold - 0.20)):
                            selected.append((float(probability), item["entity_id"]))
                selected.sort(reverse=True)
                matching_writer.writerow([left["entity_id"], ",".join(item[1] for item in selected)])
                candidate_writer.writerow([left["entity_id"], ",".join(candidate_ids)])
                processed += 1
                total_candidates += len(candidate_ids)
                total_matches += len(selected)
            if processed % (chunk_size * 5) == 0 or limit is not None:
                log(f"test: processed {processed:,} Source 1 rows; candidates={total_candidates:,}; matches={total_matches:,}", started)
    connection.close()
    db_path.unlink(missing_ok=True)
    log(f"complete: rows={processed:,}; candidates={total_candidates:,}; matches={total_matches:,}", started)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-sample", type=int, default=50_000)
    parser.add_argument("--source1-limit", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--threshold", type=float, default=0.86)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--work-dir", type=Path, default=None)
    args = parser.parse_args()
    started = time.monotonic()
    work_dir = args.work_dir or (args.output_dir / ".index")
    work_dir.mkdir(parents=True, exist_ok=True)
    log("starting streaming pipeline", started)
    model = train_model(args.data_dir / "train", args.train_sample, args.chunk_size, work_dir, args.seed, started)
    write_outputs(model, args.data_dir / "test/test_source1.tsv", args.data_dir / "test/test_source2.tsv",
                  args.data_dir / "test/test_source3.tsv", args.output_dir, work_dir, args.chunk_size,
                  args.source1_limit, args.threshold, started)
    try:
        (work_dir / "train.sqlite").unlink(missing_ok=True)
        (work_dir / "test.sqlite").unlink(missing_ok=True)
        work_dir.rmdir()
    except OSError:
        pass


if __name__ == "__main__":
    main()
