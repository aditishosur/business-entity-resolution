"""
Validation and threshold evaluation for the streaming entity-resolution solver.

The repository documents src/solve_streaming.py as the solver to use for the
complete challenge dataset. This evaluator therefore uses the same:
    - normalization
    - country-aware exact name/address/postal blocking
    - 300-candidate limit
    - feature definitions
    - LogisticMatcher
    - exact-match probability override

The Source 1 training/validation split is deterministic.

The Source 2 and Source 3 TSV files are never copied or modified. A temporary
SQLite index is created under the evaluation output directory and deleted when
the evaluation finishes.
"""

from __future__ import annotations

import argparse
import csv
import re
import sqlite3
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from src.solve import LogisticMatcher
from src.matching_features import feature_row


COLS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]

TOKEN_RE = re.compile(r"[a-z0-9]+")
POSTAL_RE = re.compile(r"(?<!\d)(\d{5,6})(?!\d)")

MAX_CANDIDATES_PER_BLOCK = 300

THRESHOLDS = [
    0.70,
    0.75,
    0.80,
    0.85,
    0.86,
    0.90,
    0.95,
]


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def norm(value: object) -> str:
    if value is None or (
        not isinstance(value, str)
        and pd.isna(value)
    ):
        return ""

    text = (
        unicodedata
        .normalize("NFKD", str(value))
        .encode("ascii", "ignore")
        .decode()
        .lower()
    )

    return " ".join(
        TOKEN_RE.findall(
            text.replace("&", " and ")
        )
    )


def postal(text: str) -> str:
    match = POSTAL_RE.search(text or "")
    return match.group(1) if match else ""


def token_set(text: str) -> set[str]:
    return set(text.split()) if text else set()


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[COLS].copy()

    frame["name_n"] = frame["business_name"].map(norm)
    frame["address_n"] = frame["business_address"].map(norm)
    frame["country_n"] = frame["country"].map(norm)
    frame["postal_n"] = frame["address_n"].map(postal)

    return frame


# ---------------------------------------------------------------------------
# SQLite streaming index
# ---------------------------------------------------------------------------

def init_db(db_path: Path) -> sqlite3.Connection:
    if db_path.exists():
        db_path.unlink()

    connection = sqlite3.connect(db_path)

    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")

    connection.execute(
        """
        CREATE TABLE records (
            row_id INTEGER PRIMARY KEY,
            entity_id TEXT NOT NULL UNIQUE,
            name_n TEXT NOT NULL,
            address_n TEXT NOT NULL,
            country_n TEXT NOT NULL,
            postal_n TEXT NOT NULL
        )
        """
    )

    connection.execute(
        """
        CREATE INDEX idx_name
        ON records(country_n, name_n)
        """
    )

    connection.execute(
        """
        CREATE INDEX idx_address
        ON records(country_n, address_n)
        """
    )

    connection.execute(
        """
        CREATE INDEX idx_postal
        ON records(country_n, postal_n)
        """
    )

    return connection


def build_db(
    paths: list[Path],
    db_path: Path,
    chunk_size: int,
) -> sqlite3.Connection:

    print("Building SQLite candidate index...")

    connection = init_db(db_path)

    total = 0

    try:
        for path in paths:

            print(f"Indexing {path.name}...")

            for frame in pd.read_csv(
                path,
                sep="\t",
                dtype=str,
                keep_default_na=False,
                usecols=COLS,
                chunksize=chunk_size,
            ):

                frame = normalize_frame(frame)

                rows = frame[
                    [
                        "entity_id",
                        "name_n",
                        "address_n",
                        "country_n",
                        "postal_n",
                    ]
                ].itertuples(
                    index=False,
                    name=None,
                )

                connection.executemany(
                    """
                    INSERT INTO records(
                        entity_id,
                        name_n,
                        address_n,
                        country_n,
                        postal_n
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    rows,
                )

                connection.commit()

                total += len(frame)

                if total % (
                    chunk_size * 10
                ) == 0:
                    print(
                        f"  indexed {total:,} records"
                    )

        print(
            f"SQLite index complete: {total:,} records"
        )

        return connection

    except Exception:
        connection.close()
        raise


# ---------------------------------------------------------------------------
# Source 1 loading / split
# ---------------------------------------------------------------------------

def load_source1(
    path: Path,
    limit: int | None,
    chunk_size: int,
) -> pd.DataFrame:

    frames = []
    loaded = 0

    for frame in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=COLS,
        chunksize=chunk_size,
    ):

        if limit is not None:

            remaining = limit - loaded

            if remaining <= 0:
                break

            frame = frame.iloc[
                :remaining
            ]

        frames.append(
            normalize_frame(frame)
        )

        loaded += len(frame)

        if limit is not None and loaded >= limit:
            break

    if not frames:
        raise RuntimeError(
            "No Source 1 records were loaded."
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


def split_source1(
    source1: pd.DataFrame,
    validation_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    rng = np.random.default_rng(seed)

    indices = np.arange(len(source1))
    rng.shuffle(indices)

    validation_size = int(
        round(
            len(source1)
            * validation_fraction
        )
    )

    validation_indices = indices[
        :validation_size
    ]

    train_indices = indices[
        validation_size:
    ]

    train = source1.iloc[
        train_indices
    ].reset_index(drop=True)

    validation = source1.iloc[
        validation_indices
    ].reset_index(drop=True)

    return train, validation


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def load_ground_truth(
    path: Path,
    source1_ids: set[str],
) -> dict[str, set[str]]:

    result: dict[str, set[str]] = {}

    for frame in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=100_000,
    ):

        selected = frame[
            frame.source1_entity_id.isin(
                source1_ids
            )
        ]

        for row in selected.itertuples(
            index=False
        ):

            result[
                row.source1_entity_id
            ] = set(
                filter(
                    None,
                    row.matched_entity_ids.split(","),
                )
            )

    return result


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def candidates(
    connection: sqlite3.Connection,
    row: pd.Series,
) -> list[dict]:

    ids: set[int] = set()

    queries = [
        (
            """
            SELECT row_id
            FROM records
            WHERE country_n = ?
              AND name_n = ?
              AND name_n <> ''
            LIMIT 300
            """,
            (
                row.country_n,
                row.name_n,
            ),
        ),
        (
            """
            SELECT row_id
            FROM records
            WHERE country_n = ?
              AND address_n = ?
              AND address_n <> ''
            LIMIT 300
            """,
            (
                row.country_n,
                row.address_n,
            ),
        ),
        (
            """
            SELECT row_id
            FROM records
            WHERE country_n = ?
              AND postal_n = ?
              AND postal_n <> ''
            LIMIT 300
            """,
            (
                row.country_n,
                row.postal_n,
            ),
        ),
    ]

    for query, params in queries:

        ids.update(
            item[0]
            for item in connection.execute(
                query,
                params,
            )
        )

    if not ids:
        return []

    placeholders = ",".join(
        "?" for _ in ids
    )

    rows = connection.execute(
        f"""
        SELECT
            row_id,
            entity_id,
            name_n,
            address_n,
            country_n,
            postal_n
        FROM records
        WHERE row_id IN ({placeholders})
        """,
        tuple(ids),
    ).fetchall()

    return [
        {
            "row_id": item[0],
            "entity_id": item[1],
            "name_n": item[2],
            "address_n": item[3],
            "country_n": item[4],
            "postal_n": item[5],
        }
        for item in rows
    ]


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_model(
    connection: sqlite3.Connection,
    train_source1: pd.DataFrame,
    truth: dict[str, set[str]],
    seed: int,
    max_negatives: int = 8,
) -> LogisticMatcher:

    rng = np.random.default_rng(seed)

    features = []
    labels = []

    total_pairs = 0

    print(
        f"Building training pairs for "
        f"{len(train_source1):,} Source 1 records..."
    )

    for number, (_, left) in enumerate(
        train_source1.iterrows(),
        start=1,
    ):

        found = candidates(
            connection,
            left,
        )

        wanted = truth.get(
            left.entity_id,
            set(),
        )

        positives = [
            item
            for item in found
            if item["entity_id"] in wanted
        ]

        negatives = [
            item
            for item in found
            if item["entity_id"] not in wanted
        ]

        if len(negatives) > max_negatives:

            selected = rng.choice(
                len(negatives),
                size=max_negatives,
                replace=False,
            )

            negatives = [
                negatives[int(i)]
                for i in selected
            ]

        for item in positives:

            features.append(
                feature_row(
                    left,
                    item,
                )
            )

            labels.append(1)

        for item in negatives:

            features.append(
                feature_row(
                    left,
                    item,
                )
            )

            labels.append(0)

        total_pairs += (
            len(positives)
            + len(negatives)
        )

        if number % 1_000 == 0:
            print(
                f"  processed "
                f"{number:,}/{len(train_source1):,} "
                f"Source 1 rows; "
                f"pairs={total_pairs:,}"
            )

    X = np.asarray(
        features,
        dtype=np.float32,
    )

    y = np.asarray(
        labels,
        dtype=np.int8,
    )

    if len(y) == 0:
        raise RuntimeError(
            "No training pairs were generated."
        )

    if len(np.unique(y)) < 2:
        raise RuntimeError(
            "Training pairs contain only one class."
        )

    print(
        f"Fitting LogisticMatcher on "
        f"{len(y):,} pairs..."
    )

    model = LogisticMatcher(
        seed=seed
    ).fit(
        X,
        y,
    )

    print("Model training complete.")

    return model


# ---------------------------------------------------------------------------
# Validation scoring
# ---------------------------------------------------------------------------

def generate_validation_scores(
    connection: sqlite3.Connection,
    model: LogisticMatcher,
    validation: pd.DataFrame,
) -> tuple[
    list[set[str]],
    list[dict[str, float]],
]:

    candidate_sets = []
    probability_maps = []

    print(
        f"Generating validation candidates for "
        f"{len(validation):,} Source 1 records..."
    )

    for number, (_, left) in enumerate(
        validation.iterrows(),
        start=1,
    ):

        found = candidates(
            connection,
            left,
        )

        candidate_sets.append(
            {
                item["entity_id"]
                for item in found
            }
        )

        if not found:

            probability_maps.append({})
            continue

        X = np.asarray(
            [
                feature_row(
                    left,
                    item,
                )
                for item in found
            ],
            dtype=np.float32,
        )

        probabilities = (
            model
            .predict_proba(X)[:, 1]
        )

        probability_maps.append(
            {
                item["entity_id"]: float(probability)
                for item, probability in zip(
                    found,
                    probabilities,
                )
            }
        )

        if number % 1_000 == 0:
            print(
                f"  scored "
                f"{number:,}/{len(validation):,}"
            )

    return (
        candidate_sets,
        probability_maps,
    )


def apply_threshold(
    connection: sqlite3.Connection,
    validation: pd.DataFrame,
    probability_maps: list[dict[str, float]],
    threshold: float,
) -> list[set[str]]:

    predictions = []

    for (_, left), probability_map in zip(
        validation.iterrows(),
        probability_maps,
    ):

        chosen = set()

        for entity_id, probability in (
            probability_map.items()
        ):

            # Reproduce solve_streaming.py's
            # exact-match acceptance override.
            rows = connection.execute(
                """
                SELECT
                    name_n,
                    address_n
                FROM records
                WHERE entity_id = ?
                """,
                (entity_id,),
            ).fetchall()

            exact = False

            for name_n, address_n in rows:

                exact_name = bool(
                    left.name_n
                    and name_n
                    and left.name_n == name_n
                )

                exact_address = bool(
                    left.address_n
                    and address_n
                    and left.address_n
                    == address_n
                )

                if exact_name or exact_address:
                    exact = True
                    break

            if (
                probability >= threshold
                or (
                    exact
                    and probability
                    >= max(
                        0.50,
                        threshold - 0.20,
                    )
                )
            ):
                chosen.add(entity_id)

        predictions.append(chosen)

    return predictions


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def entity_metrics(
    truth: set[str],
    prediction: set[str],
) -> tuple[float, float, float]:

    if not truth:

        score = (
            1.0
            if not prediction
            else 0.0
        )

        return score, score, score

    tp = len(
        truth & prediction
    )

    fp = len(
        prediction - truth
    )

    fn = len(
        truth - prediction
    )

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    denominator = (
        0.25 * precision
        + recall
    )

    f05 = (
        1.25 * precision * recall
        / denominator
        if denominator
        else 0.0
    )

    return precision, recall, f05


def macro_metrics(
    truths: list[set[str]],
    predictions: list[set[str]],
) -> tuple[float, float, float]:

    if len(truths) != len(predictions):
        raise ValueError(
            "truths and predictions "
            "must have equal length"
        )

    if not truths:
        return 0.0, 0.0, 0.0

    values = [
        entity_metrics(
            truth,
            prediction,
        )
        for truth, prediction in zip(
            truths,
            predictions,
        )
    ]

    return (
        float(
            np.mean(
                [value[0] for value in values]
            )
        ),
        float(
            np.mean(
                [value[1] for value in values]
            )
        ),
        float(
            np.mean(
                [value[2] for value in values]
            )
        ),
    )

def macro_f05(
    truths: list[set[str]],
    predictions: list[set[str]],
) -> float:
    """Return the macro-averaged F0.5 score."""
    return macro_metrics(truths, predictions)[2]


def singleton_accuracy(
    truths: list[set[str]],
    predictions: list[set[str]],
) -> float:

    values = [
        not prediction
        for truth, prediction in zip(
            truths,
            predictions,
        )
        if not truth
    ]

    return (
        float(np.mean(values))
        if values
        else 1.0
    )


def candidate_recall(
    truths: list[set[str]],
    candidates_list: list[set[str]],
) -> float:

    values = []

    for truth, candidate_set in zip(
        truths,
        candidates_list,
    ):

        if truth:

            values.append(
                len(
                    truth & candidate_set
                )
                / len(truth)
            )

    return (
        float(np.mean(values))
        if values
        else 1.0
    )


# ---------------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------------

def write_validation_results(
    path: Path,
    validation: pd.DataFrame,
    truths: list[set[str]],
    predictions: list[set[str]],
    candidates_list: list[set[str]],
):

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.writer(
            handle,
            lineterminator="\n",
        )

        writer.writerow(
            [
                "source1_entity_id",
                "true_match_count",
                "true_matched_entity_ids",
                "predicted_match_count",
                "predicted_entity_ids",
                "candidate_count",
                "candidate_entity_ids",
                "candidate_true_match_count",
                "candidate_recall",
                "precision",
                "recall",
                "f05",
                "singleton",
            ]
        )

        for row, truth, prediction, candidate_set in zip(
            validation.itertuples(index=False),
            truths,
            predictions,
            candidates_list,
        ):

            precision, recall, f05 = (
                entity_metrics(
                    truth,
                    prediction,
                )
            )

            candidate_true_match_count = len(
                truth & candidate_set
            )

            candidate_recall = (
                candidate_true_match_count / len(truth)
                if truth
                else 1.0
            )

            writer.writerow(
                [
                    row.entity_id,
                    len(truth),
                    ",".join(sorted(truth)),
                    len(prediction),
                    ",".join(sorted(prediction)),
                    len(candidate_set),
                    ",".join(sorted(candidate_set)),
                    candidate_true_match_count,
                    candidate_recall,
                    precision,
                    recall,
                    f05,
                    not truth,
                ]
            )


def write_report(
    path: Path,
    *,
    source1_records: int,
    train_records: int,
    validation_records: int,
    validation_fraction: float,
    seed: int,
    train_sample: int | None,
    results: pd.DataFrame,
):

    best = results.loc[
        results["macro_f05"].idxmax()
    ]

    lines = [
        "# Validation Report",
        "",
        "## Dataset and split",
        "",
        f"- Source 1 records used: {source1_records:,}",
        f"- Training Source 1 records: {train_records:,}",
        f"- Validation Source 1 records: {validation_records:,}",
        f"- Validation fraction: {validation_fraction:.2%}",
        f"- Random seed: {seed}",
        (
            f"- Source 1 sample limit: {train_sample:,}"
            if train_sample is not None
            else "- Source 1 sample limit: all available records"
        ),
        "",
        "## Evaluation implementation",
        "",
        "- Evaluation uses the repository's SQLite-backed streaming architecture.",
        "- Source 2 and Source 3 remain the complete candidate universe.",
        "- Candidate blocking uses country + exact name, country + exact address, and country + postal.",
        f"- Each blocking query is capped at {MAX_CANDIDATES_PER_BLOCK} candidates.",
        "- Model training uses only the training Source 1 split.",
        "- Validation ground truth is never used during model fitting.",
        "",
        "## Metrics",
        "",
        "- Precision, recall and F0.5 are calculated per Source 1 entity and macro-averaged.",
        "- Singleton accuracy measures whether entities with no true matches receive an empty prediction.",
        "- Candidate recall measures whether true matches survive candidate generation.",
        "",
        "## Threshold comparison",
        "",
        "| Threshold | Precision | Recall | Macro F0.5 | Singleton Accuracy | Predictions | Candidate Recall |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in results.itertuples(index=False):

        lines.append(
            f"| {row.threshold:.2f} | "
            f"{row.precision:.6f} | "
            f"{row.recall:.6f} | "
            f"{row.macro_f05:.6f} | "
            f"{row.singleton_accuracy:.6f} | "
            f"{row.num_predictions:,} | "
            f"{row.candidate_recall:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Measured validation result",
            "",
            f"- Threshold with highest measured Macro F0.5: **{best.threshold:.2f}**",
            f"- Corresponding Macro F0.5: **{best.macro_f05:.6f}**",
            "",
            "## Limitations",
            "",
            "- These are validation results from the supplied training ground truth.",
            "- They are not test-set results because the challenge test set has no public ground truth.",
            "- Candidate recall is a blocking-stage ceiling: a true match excluded by blocking cannot be recovered by thresholding.",
            "- If a Source 1 sample was used, the results describe that deterministic validation sample rather than the entire Source 1 training population.",
            "",
        ]
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/validation"),
    )

    parser.add_argument(
        "--source1-sample",
        type=int,
        default=50_000,
        help=(
            "Number of Source 1 records to use. "
            "Default: 50,000. Use 0 for all."
        ),
    )

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50_000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    train_dir = (
        args.data_dir / "train"
    )

    source1_path = (
        train_dir
        / "train_source1.tsv"
    )

    source2_path = (
        train_dir
        / "train_source2.tsv"
    )

    source3_path = (
        train_dir
        / "train_source3.tsv"
    )

    ground_truth_path = (
        train_dir
        / "train_ground_truth.tsv"
    )

    output_dir = args.output_dir
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    db_dir = (
        output_dir
        / ".evaluation_index"
    )

    db_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    db_path = (
        db_dir
        / "validation.sqlite"
    )

    try:

        print("=" * 60)
        print("ENTITY RESOLUTION VALIDATION")
        print("=" * 60)

        # ---------------------------------------------------------------
        # Source 1
        # ---------------------------------------------------------------

        limit = (
            None
            if args.source1_sample == 0
            else args.source1_sample
        )

        print(
            "\nLoading Source 1..."
        )

        source1 = load_source1(
            source1_path,
            limit,
            args.chunk_size,
        )

        print(
            f"Loaded {len(source1):,} Source 1 records."
        )

        train_source1, validation_source1 = (
            split_source1(
                source1,
                args.validation_fraction,
                args.seed,
            )
        )

        print(
            f"Training split: "
            f"{len(train_source1):,}"
        )

        print(
            f"Validation split: "
            f"{len(validation_source1):,}"
        )

        # ---------------------------------------------------------------
        # Ground truth
        # ---------------------------------------------------------------

        print(
            "\nLoading ground truth..."
        )

        ground_truth = load_ground_truth(
            ground_truth_path,
            set(source1.entity_id),
        )

        print(
            f"Ground-truth records loaded: "
            f"{len(ground_truth):,}"
        )

        # ---------------------------------------------------------------
        # SQLite candidate universe
        # ---------------------------------------------------------------

        connection = build_db(
            [
                source2_path,
                source3_path,
            ],
            db_path,
            args.chunk_size,
        )

        # ---------------------------------------------------------------
        # Model
        # ---------------------------------------------------------------

        train_truth = {
            entity_id: ground_truth.get(
                entity_id,
                set(),
            )
            for entity_id
            in train_source1.entity_id
        }

        model = train_model(
            connection,
            train_source1,
            train_truth,
            args.seed,
        )

        # ---------------------------------------------------------------
        # Validation candidate probabilities
        # ---------------------------------------------------------------

        (
            candidate_sets,
            probability_maps,
        ) = generate_validation_scores(
            connection,
            model,
            validation_source1,
        )

        validation_truths = [
            ground_truth.get(
                entity_id,
                set(),
            )
            for entity_id
            in validation_source1.entity_id
        ]

        # ---------------------------------------------------------------
        # Threshold sweep
        # ---------------------------------------------------------------

        print(
            "\nRunning threshold sweep..."
        )

        rows = []

        all_predictions = {}

        for threshold in THRESHOLDS:

            predictions = apply_threshold(
                connection,
                validation_source1,
                probability_maps,
                threshold,
            )

            all_predictions[
                threshold
            ] = predictions

            precision, recall, f05 = (
                macro_metrics(
                    validation_truths,
                    predictions,
                )
            )

            rows.append(
                {
                    "threshold": threshold,
                    "precision": precision,
                    "recall": recall,
                    "macro_f05": f05,
                    "singleton_accuracy": singleton_accuracy(
                        validation_truths,
                        predictions,
                    ),
                    "num_predictions": sum(
                        bool(prediction)
                        for prediction
                        in predictions
                    ),
                    "candidate_recall": candidate_recall(
                        validation_truths,
                        candidate_sets,
                    ),
                }
            )

        results = pd.DataFrame(rows)

        results.to_csv(
            output_dir
            / "threshold_comparison.csv",
            index=False,
        )

        # ---------------------------------------------------------------
        # Best threshold
        # ---------------------------------------------------------------

        best_index = results[
            "macro_f05"
        ].idxmax()

        best_threshold = float(
            results.loc[
                best_index,
                "threshold",
            ]
        )

        best_predictions = all_predictions[
            best_threshold
        ]

        write_validation_results(
            output_dir
            / "validation_results.csv",
            validation_source1,
            validation_truths,
            best_predictions,
            candidate_sets,
        )

        write_report(
            output_dir
            / "validation_report.md",
            source1_records=len(source1),
            train_records=len(train_source1),
            validation_records=len(
                validation_source1
            ),
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            train_sample=limit,
            results=results,
        )

        # ---------------------------------------------------------------
        # Summary
        # ---------------------------------------------------------------

        print("\n")
        print("=" * 60)
        print("VALIDATION COMPLETE")
        print("=" * 60)

        print(
            results.to_string(
                index=False
            )
        )

        print(
            f"\nMeasured threshold: "
            f"{best_threshold:.2f}"
        )

        print(
            f"Measured Macro F0.5: "
            f"{results.loc[best_index, 'macro_f05']:.6f}"
        )

        print("\nFiles:")

        print(
            output_dir
            / "validation_results.csv"
        )

        print(
            output_dir
            / "threshold_comparison.csv"
        )

        print(
            output_dir
            / "validation_report.md"
        )

    finally:

        # Close the database if it was created.
        try:
            connection.close()
        except (
            NameError,
            UnboundLocalError,
        ):
            pass

        # Remove the temporary SQLite index.
        if db_path.exists():

            try:
                db_path.unlink()
                print(
                    "\nTemporary validation index deleted."
                )
            except OSError as exc:
                print(
                    f"\nWARNING: Could not delete "
                    f"temporary index: {exc}"
                )

        try:
            db_dir.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    main()
