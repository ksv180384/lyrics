import unittest
from types import SimpleNamespace

from app import (
    _ordered_transcription_matches,
    _repair_repeated_text_blocks,
    _repair_untrusted_ranges,
)


class RepeatedBlockRepairTests(unittest.TestCase):
    def test_chooses_consistent_late_cluster_for_broken_repeat(self):
        chorus = ["line a", "line b", "line c", "line d", "line e", "line f", "line g", "line h"]
        source_starts = [54.38, 56.20, 57.88, 59.76, 61.92, 63.74, 65.56, 67.24]
        source_ends = [55.68, 57.38, 59.36, 61.24, 63.12, 65.06, 67.10, 68.96]
        broken_starts = [100.62, 102.48, 104.46, 104.76, 105.10, 124.80, 127.90, 129.26]
        broken_ends = [102.32, 104.46, 104.76, 105.10, 114.52, 127.04, 129.10, 130.92]
        texts = ["intro", *chorus, "verse", *chorus, "outro"]
        starts = [36.60, *source_starts, 69.54, *broken_starts, 142.20]
        ends = [37.98, *source_ends, 99.92, *broken_ends, 144.60]
        segments = [
            SimpleNamespace(text=text, start=start, end=end, words=[])
            for text, start, end in zip(texts, starts, ends)
        ]

        repaired_starts, _ = _repair_repeated_text_blocks(
            segments, starts.copy(), ends.copy()
        )

        self.assertAlmostEqual(repaired_starts[10], 116.40, places=2)
        self.assertAlmostEqual(repaired_starts[17], 129.26, places=2)

    def test_ignores_far_match_for_reliable_repeated_line(self):
        segments = [
            SimpleNamespace(text="repeat phrase", start=10.0, end=13.0, words=[]),
            SimpleNamespace(text="repeat phrase", start=32.0, end=35.0, words=[]),
        ]
        transcript_words = [
            SimpleNamespace(word="repeat", start=32.0, end=32.8),
            SimpleNamespace(word=" phrase", start=32.9, end=33.8),
        ]

        matches = _ordered_transcription_matches(segments, transcript_words)

        self.assertNotIn(0, matches)
        self.assertAlmostEqual(matches[1][1], 32.0)

    def test_preserves_refined_repeat_after_stretched_first_line(self):
        chorus = ["line a", "line b", "line c", "line d", "line e", "line f"]
        source_starts = [10.10, 13.70, 17.36, 22.24, 22.66, 24.18]
        source_ends = [13.38, 16.68, 21.82, 22.38, 23.74, 24.34]
        refined_starts = [104.58, 108.00, 111.76, 116.54, 117.14, 118.64]
        refined_ends = [107.88, 111.20, 116.30, 116.94, 118.18, 118.84]
        forced_starts = [80.70, 108.00, 111.66, 116.56, 117.12, 118.66]
        texts = ["intro", *chorus, "bridge", *chorus, "outro"]
        starts = [7.0, *source_starts, 40.0, *refined_starts, 119.5]
        ends = [9.0, *source_ends, 75.96, *refined_ends, 135.12]
        segment_starts = [7.0, *source_starts, 40.0, *forced_starts, 119.5]
        segments = [
            SimpleNamespace(text=text, start=start, end=end, words=[])
            for text, start, end in zip(texts, segment_starts, ends)
        ]

        repaired_starts, _ = _repair_repeated_text_blocks(
            segments, starts.copy(), ends.copy()
        )

        self.assertEqual(repaired_starts[8:14], refined_starts)

    def test_preserves_coherent_tail_after_instrumental_break(self):
        starts = [116.40, 118.22, 119.90, 121.78, 123.94, 125.76, 127.58, 129.26, 142.20, 144.70]
        ends = [117.70, 119.40, 121.38, 123.26, 125.14, 127.08, 129.12, 130.98, 144.60, 146.32]
        transcript_words = [SimpleNamespace(end=148.0)]

        repaired_starts, _ = _repair_untrusted_ranges(
            starts.copy(), ends.copy(), {7}, transcript_words, 148.04
        )

        self.assertEqual(repaired_starts[-2:], [142.20, 144.70])


if __name__ == "__main__":
    unittest.main()
