import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import solve
import solve_streaming


def baseline_frame(records):
    frame = pd.DataFrame(records)
    frame["name_n"] = frame["business_name"].map(solve.norm)
    frame["address_n"] = frame["business_address"].map(solve.norm)
    frame["country_n"] = frame["country"].map(solve.norm)
    frame["postal_n"] = frame["address_n"].map(solve.postal)
    frame["name_t"] = frame["name_n"].map(solve.tokens)
    frame["address_t"] = frame["address_n"].map(solve.tokens)
    return frame


def streaming_index(records, directory):
    path = Path(directory) / "records.sqlite"
    connection = solve_streaming.init_db(path)
    frame = solve_streaming.normalize_frame(pd.DataFrame(records))
    values = frame[["entity_id", "name_n", "address_n", "country_n", "postal_n"]]
    connection.executemany(
        "INSERT INTO records(entity_id,name_n,address_n,country_n,postal_n) VALUES (?,?,?,?,?)",
        values.itertuples(index=False, name=None),
    )
    connection.commit()
    return connection


def stream_query_row(name="", address="", country="United States"):
    frame = solve_streaming.normalize_frame(pd.DataFrame([{
        "entity_id": "query",
        "business_name": name,
        "business_address": address,
        "country": country,
    }]))
    return solve_streaming.record_from_row(frame.iloc[0])


class BlockingCandidateTests(unittest.TestCase):
    def test_normalization_used_for_blocking(self):
        value = "Caf\u00e9 & Co., \u6771\u4eac"
        self.assertEqual(solve.norm(value), "cafe and co")
        self.assertEqual(solve_streaming.norm(value), "cafe and co")
        self.assertEqual(solve.postal("12 Rue 75001"), "75001")
        self.assertEqual(solve_streaming.postal("12 Rue 75001"), "75001")
        self.assertEqual(solve.postal("12 Rue 1234"), "")

    def test_baseline_emits_each_blocking_key_family(self):
        row = baseline_frame([{
            "entity_id": "query",
            "business_name": "Acme Holdings",
            "business_address": "12 Beacon Avenue 12345",
            "country": "United States",
        }]).iloc[0]

        self.assertEqual(set(solve.keys(row)), {
            "N|united states|acme holdings",
            "A|united states|12 beacon avenue 12345",
            "P|united states|12345",
            "NT|united states|holdings",
            "NT|united states|acme",
            "AT|united states|avenue",
            "AT|united states|beacon",
            "AT|united states|12345",
        })

    def test_baseline_token_minimum_lengths_and_common_exclusions(self):
        row = baseline_frame([{
            "entity_id": "query",
            "business_name": "The And Acme Inc Ltd abc",
            "business_address": "The Road Street Company",
            "country": "Canada",
        }]).iloc[0]
        generated = set(solve.keys(row))

        self.assertIn("NT|canada|acme", generated)
        self.assertFalse(any(key.startswith("NT|") and key.rsplit("|", 1)[1] in {
            "the", "and", "inc", "ltd", "abc"
        } for key in generated))
        self.assertIn("AT|canada|street", generated)
        self.assertNotIn("AT|canada|road", generated)
        self.assertFalse(any(key.startswith("AT|") and key.endswith("|company") for key in generated))

    def test_baseline_drops_bucket_at_301_not_300(self):
        for size, expected_count in ((300, 300), (301, 0)):
            with self.subTest(bucket_size=size):
                records = [{
                    "entity_id": f"candidate-{index}",
                    "business_name": "Shared Enterprise",
                    "business_address": f"Unit {index}",
                    "country": "United States",
                } for index in range(size)]
                index = solve.CandidateIndex(baseline_frame(records))
                query = baseline_frame([{
                    "entity_id": "query",
                    "business_name": "Shared Enterprise",
                    "business_address": "",
                    "country": "United States",
                }]).iloc[0]

                self.assertEqual(len(index.candidates(query)), expected_count)

    def test_baseline_candidates_are_deduplicated(self):
        records = baseline_frame([{
            "entity_id": "candidate",
            "business_name": "Acme Holdings",
            "business_address": "12 Beacon Avenue 12345",
            "country": "United States",
        }])
        query = baseline_frame([{
            "entity_id": "query",
            "business_name": "Acme Holdings",
            "business_address": "12 Beacon Avenue 12345",
            "country": "United States",
        }]).iloc[0]

        self.assertEqual(solve.CandidateIndex(records).candidates(query), [0])

    def test_baseline_country_is_part_of_each_block(self):
        records = baseline_frame([{
            "entity_id": "canadian",
            "business_name": "Acme Holdings",
            "business_address": "12 Beacon Avenue 12345",
            "country": "Canada",
        }])
        query = baseline_frame([{
            "entity_id": "query",
            "business_name": "Acme Holdings",
            "business_address": "12 Beacon Avenue 12345",
            "country": "United States",
        }]).iloc[0]

        self.assertEqual(solve.CandidateIndex(records).candidates(query), [])

    def test_streaming_retrieves_exact_name_address_and_postal_candidates(self):
        records = [
            {
                "entity_id": "by-name",
                "business_name": "Acme Holdings",
                "business_address": "90 Pine Lane 54321",
                "country": "United States",
            },
            {
                "entity_id": "by-address",
                "business_name": "Different Company",
                "business_address": "12 Beacon Avenue 12345",
                "country": "United States",
            },
            {
                "entity_id": "by-postal",
                "business_name": "Another Company",
                "business_address": "50 Cedar Avenue 12345",
                "country": "United States",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            connection = streaming_index(records, directory)
            try:
                found = solve_streaming.candidates(
                    connection,
                    stream_query_row("Acme Holdings", "12 Beacon Avenue 12345"),
                )
            finally:
                connection.close()

        self.assertEqual({item["entity_id"] for item in found}, {
            "by-name", "by-address", "by-postal"
        })

    def test_streaming_deduplicates_and_isolates_country(self):
        records = [
            {
                "entity_id": "same-country",
                "business_name": "Acme Holdings",
                "business_address": "12 Beacon Avenue 12345",
                "country": "United States",
            },
            {
                "entity_id": "other-country",
                "business_name": "Acme Holdings",
                "business_address": "12 Beacon Avenue 12345",
                "country": "Canada",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            connection = streaming_index(records, directory)
            try:
                found = solve_streaming.candidates(
                    connection,
                    stream_query_row("Acme Holdings", "12 Beacon Avenue 12345"),
                )
            finally:
                connection.close()

        self.assertEqual([item["entity_id"] for item in found], ["same-country"])

    def test_streaming_limit_returns_300_without_assuming_which_rows(self):
        records = [{
            "entity_id": f"candidate-{index}",
            "business_name": "Shared Enterprise",
            "business_address": f"Unit {index}",
            "country": "United States",
        } for index in range(305)]
        expected_ids = {row["entity_id"] for row in records}
        with tempfile.TemporaryDirectory() as directory:
            connection = streaming_index(records, directory)
            try:
                found = solve_streaming.candidates(
                    connection,
                    stream_query_row("Shared Enterprise"),
                )
            finally:
                connection.close()

        found_ids = {item["entity_id"] for item in found}
        self.assertEqual(len(found_ids), 300)
        self.assertTrue(found_ids <= expected_ids)

    def test_token_only_blocking_is_current_baseline_streaming_divergence(self):
        records = [{
            "entity_id": "shared-token-only",
            "business_name": "Acme Global Trading",
            "business_address": "90 Pine Lane",
            "country": "United States",
        }]
        baseline_query = baseline_frame([{
            "entity_id": "query",
            "business_name": "Acme Global Services",
            "business_address": "12 Beacon Avenue",
            "country": "United States",
        }]).iloc[0]
        baseline_candidates = solve.CandidateIndex(baseline_frame(records)).candidates(baseline_query)

        with tempfile.TemporaryDirectory() as directory:
            connection = streaming_index(records, directory)
            try:
                streaming_candidates = solve_streaming.candidates(
                    connection,
                    stream_query_row("Acme Global Services", "12 Beacon Avenue"),
                )
            finally:
                connection.close()

        self.assertEqual([records[position]["entity_id"] for position in baseline_candidates], [
            "shared-token-only"
        ])
        self.assertEqual(streaming_candidates, [])

    def test_repeatability_baseline_order_and_streaming_candidate_set(self):
        records = [
            {
                "entity_id": f"candidate-{index}",
                "business_name": "Acme Holdings",
                "business_address": f"12 Beacon Avenue {12340 + index}",
                "country": "United States",
            } for index in range(3)
        ]
        baseline_records = baseline_frame(records)
        baseline_query = baseline_frame([{
            "entity_id": "query",
            "business_name": "Acme Holdings",
            "business_address": "",
            "country": "United States",
        }]).iloc[0]
        baseline_index = solve.CandidateIndex(baseline_records)
        baseline_first = baseline_index.candidates(baseline_query)
        baseline_second = baseline_index.candidates(baseline_query)
        self.assertEqual(baseline_first, baseline_second)
        self.assertEqual(baseline_first, sorted(baseline_first))

        with tempfile.TemporaryDirectory() as directory:
            connection = streaming_index(records, directory)
            try:
                query = stream_query_row("Acme Holdings")
                streaming_first = solve_streaming.candidates(connection, query)
                streaming_second = solve_streaming.candidates(connection, query)
            finally:
                connection.close()

        self.assertEqual(
            {item["entity_id"] for item in streaming_first},
            {item["entity_id"] for item in streaming_second},
        )


if __name__ == "__main__":
    unittest.main()