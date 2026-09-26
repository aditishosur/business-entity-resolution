import ast
import pandas as pd
from pathlib import Path


CSV_PATH = Path("experiments/validation_smoke/validation_results.csv")


def parse_ids(value):
    """Parse the CSV representation of a list of entity IDs."""
    if pd.isna(value) or value == "":
        return set()

    value = str(value).strip()

    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return set(parsed)
    except (ValueError, SyntaxError):
        pass

    # Fallback for comma-separated representations
    return {
        item.strip()
        for item in value.split(",")
        if item.strip()
    }


def main():
    df = pd.read_csv(CSV_PATH)

    true_sets = df["true_matched_entity_ids"].apply(parse_ids)
    predicted_sets = df["predicted_entity_ids"].apply(parse_ids)
    candidate_sets = df["candidate_entity_ids"].apply(parse_ids)

    true_counts = true_sets.apply(len)
    candidate_counts = candidate_sets.apply(len)

    singleton_mask = true_counts == 0
    matched_mask = true_counts > 0

    # True matches that survived candidate generation.
    candidate_true_counts = [
        len(true_ids & candidate_ids)
        for true_ids, candidate_ids
        in zip(true_sets, candidate_sets)
    ]

    # True matches that were actually predicted.
    correct_prediction_counts = [
        len(true_ids & predicted_ids)
        for true_ids, predicted_ids
        in zip(true_sets, predicted_sets)
    ]

    candidate_true_counts = pd.Series(candidate_true_counts)
    correct_prediction_counts = pd.Series(correct_prediction_counts)

    # Among non-singletons:
    # all true matches absent from candidates
    blocking_failure = (
        matched_mask
        & (candidate_true_counts == 0)
    )

    # At least one true match was a candidate, but none was predicted.
    matching_failure = (
        matched_mask
        & (candidate_true_counts > 0)
        & (correct_prediction_counts == 0)
    )

    # At least one correct match was predicted.
    successful_prediction = (
        matched_mask
        & (correct_prediction_counts > 0)
    )

    # Candidate counts that reach the 300-per-block cap.
    # Note: candidate_count is the UNION of the blocking results,
    # so >300 does not necessarily mean one block hit the cap.
    candidate_300_plus = candidate_counts >= 300

    print("=" * 60)
    print("VALIDATION DIAGNOSTICS")
    print("=" * 60)

    print(f"\nValidation entities: {len(df):,}")

    print("\nGround truth:")
    print(f"  Singleton entities: {singleton_mask.sum():,}")
    print(f"  Entities with >=1 true match: {matched_mask.sum():,}")

    print("\nCandidate generation:")
    print(
        f"  Entities with all true matches absent from candidates: "
        f"{blocking_failure.sum():,}"
    )
    print(
        f"  Entities with >=1 true match surviving candidates: "
        f"{(matched_mask & (candidate_true_counts > 0)).sum():,}"
    )

    print("\nMatching:")
    print(
        f"  Entities with candidates but no correct prediction: "
        f"{matching_failure.sum():,}"
    )
    print(
        f"  Entities with >=1 correct prediction: "
        f"{successful_prediction.sum():,}"
    )

    print("\nCandidate counts:")
    print(f"  Average candidates/entity: {candidate_counts.mean():.2f}")
    print(f"  Median candidates/entity: {candidate_counts.median():.0f}")
    print(f"  Minimum candidates/entity: {candidate_counts.min():,}")
    print(f"  Maximum candidates/entity: {candidate_counts.max():,}")
    print(
        f"  Entities with >=300 candidates: "
        f"{candidate_300_plus.sum():,}"
    )

    print("\nCandidate-count distribution:")
    print(candidate_counts.describe().to_string())

    print("\nCandidate recall diagnostic:")
    total_true_matches = true_counts[matched_mask].sum()
    surviving_true_matches = candidate_true_counts[matched_mask].sum()

    if total_true_matches:
        recall = surviving_true_matches / total_true_matches
        print(f"  True matches: {total_true_matches:,}")
        print(f"  True matches surviving blocking: {surviving_true_matches:,}")
        print(f"  Candidate recall: {recall:.6f}")
    else:
        print("  No non-singleton entities found.")


if __name__ == "__main__":
    main()