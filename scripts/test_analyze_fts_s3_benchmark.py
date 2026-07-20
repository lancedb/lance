#!/usr/bin/env python3

import struct
import unittest

from scripts.analyze_fts_s3_benchmark import canonical_result, parity_result


def score_bits(score: float) -> int:
    return struct.unpack("!I", struct.pack("!f", score))[0]


def record(row_ids: list[int], scores: list[float]) -> dict[str, object]:
    return {
        "row_ids": row_ids,
        "score_bits": [score_bits(score) for score in scores],
    }


class ParityResultTest(unittest.TestCase):
    def test_allows_different_rows_at_cutoff_tie(self) -> None:
        baseline = record([1, 2, 3], [3.0, 2.0, 1.0])
        candidate = record([1, 2, 4], [3.0, 2.0, 1.0])

        self.assertNotEqual(canonical_result(baseline), canonical_result(candidate))
        self.assertEqual(parity_result(baseline), parity_result(candidate))

    def test_requires_exact_rows_above_cutoff(self) -> None:
        baseline = record([1, 2, 3], [3.0, 2.0, 1.0])
        candidate = record([1, 4, 3], [3.0, 2.0, 1.0])

        self.assertNotEqual(parity_result(baseline), parity_result(candidate))

    def test_requires_exact_cutoff_score_and_count(self) -> None:
        baseline = record([1, 2, 3], [3.0, 2.0, 1.0])
        different_score = record([1, 2, 3], [3.0, 2.0, 0.5])
        different_count = record([1, 2, 3, 4], [3.0, 2.0, 1.0, 1.0])

        self.assertNotEqual(parity_result(baseline), parity_result(different_score))
        self.assertNotEqual(parity_result(baseline), parity_result(different_count))

    def test_rejects_mismatched_rows_and_scores(self) -> None:
        with self.assertRaisesRegex(ValueError, "different row and score counts"):
            parity_result({"row_ids": [1], "score_bits": []})

    def test_rejects_non_finite_scores(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-finite score"):
            parity_result(record([1], [float("nan")]))


if __name__ == "__main__":
    unittest.main()
