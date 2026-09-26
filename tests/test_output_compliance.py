import csv
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from typing import Iterable


MATCHING_HEADER = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_HEADER = ["source1_entity_id", "candidate_entity_ids"]


def _read_tsv(path: Path, expected_header: list[str], label: str, errors: list[str]):
    rows = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t", strict=True)
            try:
                header = next(reader)
            except StopIteration:
                errors.append(f"{label}: missing header")
                return rows
            if header != expected_header:
                errors.append(f"{label}: header must be {expected_header!r}, got {header!r}")
            try:
                for line_number, row in enumerate(reader, start=2):
                    if len(row) != 2:
                        errors.append(
                            f"{label}: row {line_number} must have exactly 2 tab-separated fields"
                        )
                    padded = (row + [""])[:2]
                    rows.append((padded[0], padded[1]))
            except csv.Error as error:
                errors.append(f"{label}: malformed TSV near line {reader.line_num}: {error}")
    except (OSError, UnicodeError, csv.Error) as error:
        errors.append(f"{label}: unable to read TSV: {error}")
    return rows


def _parse_ids(value: str, label: str, row_number: int, errors: list[str]) -> list[str]:
    if value == "":
        return []
    values = value.split(",")
    if any(item == "" for item in values):
        errors.append(f"{label}: row {row_number} contains an empty ID in a nonempty list")
    return [item for item in values if item != ""]


def validate_outputs(
    matching_path: Path,
    candidate_path: Path,
    source1_ids: Iterable[str],
    valid_source23_ids: Iterable[str],
) -> list[str]:
    """Return output-contract violations; blank match/candidate lists are valid."""
    errors: list[str] = []
    expected_ids = list(source1_ids)
    expected_counts = Counter(item for item in expected_ids if isinstance(item, str) and item != "")
    if len(expected_counts) != len(expected_ids):
        errors.append("Source 1 test IDs must be nonempty strings")
    if any(count != 1 for count in expected_counts.values()):
        errors.append("Source 1 test IDs must be unique")
    expected_source1 = set(expected_counts)
    valid_other = {
        item for item in valid_source23_ids if isinstance(item, str) and item != ""
    }

    matching_rows = _read_tsv(matching_path, MATCHING_HEADER, "matching_results.tsv", errors)
    candidate_rows = _read_tsv(candidate_path, CANDIDATE_HEADER, "candidate_pairs.tsv", errors)
    if len(matching_rows) != len(expected_ids):
        errors.append(
            f"matching_results.tsv: expected {len(expected_ids)} data rows, got {len(matching_rows)}"
        )
    if len(candidate_rows) != len(expected_ids):
        errors.append(
            f"candidate_pairs.tsv: expected {len(expected_ids)} data rows, got {len(candidate_rows)}"
        )

    matching_counts = Counter(source1_id for source1_id, _ in matching_rows if source1_id)
    candidate_counts = Counter(source1_id for source1_id, _ in candidate_rows if source1_id)
    matching_lists: dict[str, list[list[str]]] = {}
    candidate_lists: dict[str, list[list[str]]] = {}

    for label, rows, counts, lists in (
        ("matching_results.tsv", matching_rows, matching_counts, matching_lists),
        ("candidate_pairs.tsv", candidate_rows, candidate_counts, candidate_lists),
    ):
        for row_number, (source1_id, list_value) in enumerate(rows, start=2):
            if not source1_id:
                errors.append(f"{label}: row {row_number} has an empty Source 1 ID")
                continue
            if source1_id not in expected_source1:
                errors.append(f"{label}: row {row_number} has unexpected Source 1 ID {source1_id!r}")
            lists.setdefault(source1_id, []).append(
                _parse_ids(list_value, label, row_number, errors)
            )
        if any(count > 1 for count in counts.values()):
            errors.append(f"{label}: duplicate Source 1 IDs")
        missing = expected_source1 - set(counts)
        if missing:
            errors.append(f"{label}: missing Source 1 IDs {sorted(missing)!r}")
        repeated_or_other = {item for item, count in counts.items() if count != 1}
        if repeated_or_other:
            errors.append(f"{label}: each expected Source 1 ID must appear exactly once")

    for row_number, (source1_id, list_value) in enumerate(matching_rows, start=2):
        if not source1_id:
            continue
        matched_ids = _parse_ids(list_value, "matching_results.tsv", row_number, errors)
        if len(matched_ids) != len(set(matched_ids)):
            errors.append(f"matching_results.tsv: row {row_number} has duplicate matched IDs")
        for matched_id in matched_ids:
            if matched_id not in valid_other:
                errors.append(
                    f"matching_results.tsv: row {row_number} references invalid matched ID {matched_id!r}"
                )
            if matched_id in expected_source1:
                errors.append(
                    f"matching_results.tsv: row {row_number} contains a Source 1 ID in its match list"
                )

    for row_number, (source1_id, list_value) in enumerate(candidate_rows, start=2):
        if not source1_id:
            continue
        candidate_ids = _parse_ids(list_value, "candidate_pairs.tsv", row_number, errors)
        for candidate_id in candidate_ids:
            if candidate_id not in valid_other:
                errors.append(
                    f"candidate_pairs.tsv: row {row_number} references invalid candidate ID {candidate_id!r}"
                )

    for source1_id, match_rows in matching_lists.items():
        candidate_rows_for_id = candidate_lists.get(source1_id, [])
        if len(match_rows) == 1 and len(candidate_rows_for_id) == 1:
            missing_candidates = set(match_rows[0]) - set(candidate_rows_for_id[0])
            if missing_candidates:
                errors.append(
                    f"Source 1 ID {source1_id!r} has matches absent from candidates: "
                    f"{sorted(missing_candidates)!r}"
                )

    return errors


class OutputComplianceTests(unittest.TestCase):
    source1_ids = ["s1", "s2"]
    source23_ids = ["a", "b"]

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)
        self.matching_path = self.directory / "matching_results.tsv"
        self.candidate_path = self.directory / "candidate_pairs.tsv"
        self.write_valid_outputs()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def write_tsv(self, path, header, rows):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)

    def write_valid_outputs(self):
        self.write_tsv(self.matching_path, MATCHING_HEADER, [("s1", "a"), ("s2", "")])
        self.write_tsv(self.candidate_path, CANDIDATE_HEADER, [("s1", "a,b"), ("s2", "")])

    def validate(self):
        return validate_outputs(
            self.matching_path,
            self.candidate_path,
            self.source1_ids,
            self.source23_ids,
        )

    def test_accepts_compliant_files_and_preserved_empty_lists(self):
        self.assertEqual(self.validate(), [])

    def test_requires_exact_headers(self):
        self.write_tsv(self.matching_path, ["source1_entity_id", "matches"], [("s1", "a"), ("s2", "")])
        errors = self.validate()
        self.assertTrue(any("matching_results.tsv: header must be" in error for error in errors))

    def test_detects_non_tab_delimited_file(self):
        self.matching_path.write_text(
            "source1_entity_id,matched_entity_ids\ns1,a\ns2,\n", encoding="utf-8"
        )
        errors = self.validate()
        self.assertTrue(any("matching_results.tsv: header must be" in error for error in errors))
        self.assertTrue(any("exactly 2 tab-separated fields" in error for error in errors))

    def test_detects_missing_source1_rows_and_wrong_row_count(self):
        for path, header, label in (
            (self.matching_path, MATCHING_HEADER, "matching_results.tsv"),
            (self.candidate_path, CANDIDATE_HEADER, "candidate_pairs.tsv"),
        ):
            with self.subTest(output=label):
                self.write_tsv(path, header, [("s1", "a")])
                errors = self.validate()
                self.assertTrue(any(f"{label}: expected 2 data rows, got 1" in error for error in errors))
                self.assertTrue(any(f"{label}: missing Source 1 IDs ['s2']" in error for error in errors))
                self.write_valid_outputs()

    def test_detects_duplicate_source1_ids(self):
        for path, header, label in (
            (self.matching_path, MATCHING_HEADER, "matching_results.tsv"),
            (self.candidate_path, CANDIDATE_HEADER, "candidate_pairs.tsv"),
        ):
            with self.subTest(output=label):
                self.write_tsv(path, header, [("s1", "a"), ("s1", "b")])
                errors = self.validate()
                self.assertTrue(any(f"{label}: duplicate Source 1 IDs" in error for error in errors))
                self.write_valid_outputs()

    def test_detects_unexpected_source1_ids(self):
        self.write_tsv(self.matching_path, MATCHING_HEADER, [("s1", "a"), ("unknown", "")])
        errors = self.validate()
        self.assertTrue(any("unexpected Source 1 ID 'unknown'" in error for error in errors))

    def test_detects_duplicate_matched_ids(self):
        self.write_tsv(self.matching_path, MATCHING_HEADER, [("s1", "a,a"), ("s2", "")])
        errors = self.validate()
        self.assertTrue(any("duplicate matched IDs" in error for error in errors))

    def test_detects_invalid_source23_references(self):
        self.write_tsv(self.matching_path, MATCHING_HEADER, [("s1", "not-valid"), ("s2", "")])
        errors = self.validate()
        self.assertTrue(any("invalid matched ID 'not-valid'" in error for error in errors))

        self.write_tsv(self.candidate_path, CANDIDATE_HEADER, [("s1", "not-valid"), ("s2", "")])
        errors = self.validate()
        self.assertTrue(any("invalid candidate ID 'not-valid'" in error for error in errors))

    def test_detects_source1_id_inside_match_list(self):
        self.write_tsv(self.matching_path, MATCHING_HEADER, [("s1", "s2"), ("s2", "")])
        errors = self.validate()
        self.assertTrue(any("contains a Source 1 ID in its match list" in error for error in errors))

    def test_detects_final_match_missing_from_candidate_list(self):
        self.write_tsv(self.candidate_path, CANDIDATE_HEADER, [("s1", "b"), ("s2", "")])
        errors = self.validate()
        self.assertTrue(any("matches absent from candidates" in error for error in errors))

    def test_malformed_empty_values_report_errors_without_crashing(self):
        self.matching_path.write_text(
            "source1_entity_id\tmatched_entity_ids\n\t\ns2\n", encoding="utf-8"
        )
        errors = self.validate()
        self.assertTrue(any("empty Source 1 ID" in error for error in errors))
        self.assertTrue(any("exactly 2 tab-separated fields" in error for error in errors))

    def test_empty_id_inside_nonempty_list_is_reported_without_crashing(self):
        self.write_tsv(self.candidate_path, CANDIDATE_HEADER, [("s1", "a,,b"), ("s2", "")])
        errors = self.validate()
        self.assertTrue(any("empty ID in a nonempty list" in error for error in errors))

    def test_missing_file_is_reported_without_crashing(self):
        self.candidate_path.unlink()
        errors = self.validate()
        self.assertTrue(any("candidate_pairs.tsv: unable to read TSV" in error for error in errors))


if __name__ == "__main__":
    unittest.main()