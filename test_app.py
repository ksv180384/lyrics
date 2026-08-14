import unittest
from types import SimpleNamespace

from app import (
    _build_line_start_times,
    _is_section_label,
    _lyrics_for_alignment,
    _collapsed_prefix_retry_offset,
    _has_severely_collapsed_prefix,
    _ordered_transcription_matches,
    _repair_collapsed_segments_before_late_groups,
    _repair_repeated_segment_starts,
    _repair_repeated_text_blocks,
    _repair_untrusted_ranges,
    _segment_end,
    _segment_start,
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

    def test_recovers_late_phrase_after_collapsed_false_words(self):
        def word(text, start, end):
            return SimpleNamespace(word=text, start=start, end=end)

        segments = [
            SimpleNamespace(
                text="Les annees passent",
                start=114.94,
                end=116.86,
                words=[
                    word(" Les", 114.94, 115.16),
                    word(" annees", 115.16, 115.16),
                    word(" passent", 115.16, 116.86),
                ],
            ),
            SimpleNamespace(
                text="Les amis restent",
                start=117.0,
                end=117.08,
                words=[
                    word(" Les", 117.0, 117.08),
                    word(" amis", 117.08, 117.08),
                    word(" restent", 117.08, 117.08),
                ],
            ),
            SimpleNamespace(
                text="La vie defile",
                start=117.08,
                end=128.2,
                words=[
                    word(" La", 117.08, 117.12),
                    word(" vie", 117.12, 117.24),
                    word(" defile", 125.56, 128.2),
                ],
            ),
        ]
        starts = [_segment_start(segment) for segment in segments]
        ends = [_segment_end(segment) for segment in segments]

        repaired_starts, repaired_ends = (
            _repair_collapsed_segments_before_late_groups(
                segments, starts, ends, set()
            )
        )

        self.assertEqual(
            [round(value, 2) for value in repaired_starts],
            [120.79, 122.96, 124.56],
        )
        self.assertEqual(
            [round(value, 2) for value in repaired_ends],
            [122.71, 124.31, 126.06],
        )

    def test_ignores_match_shifted_almost_eight_seconds(self):
        segments = [
            SimpleNamespace(
                text="On a tellement de souvenirs",
                start=138.66,
                end=140.70,
                words=[],
            )
        ]
        transcript_words = [
            SimpleNamespace(word="On", start=131.0, end=131.2),
            SimpleNamespace(word=" a", start=131.2, end=131.4),
            SimpleNamespace(word=" tellement", start=131.4, end=132.0),
            SimpleNamespace(word=" de", start=132.0, end=132.2),
            SimpleNamespace(word=" souvenirs", start=132.2, end=133.0),
        ]

        matches = _ordered_transcription_matches(segments, transcript_words)

        self.assertEqual(matches, {})

    def test_detects_severely_collapsed_opening_lines(self):
        segments = [
            SimpleNamespace(text=f"line {index}", start=index, end=index, words=[])
            for index in range(10)
        ]

        self.assertTrue(_has_severely_collapsed_prefix(segments))

    def test_caps_intro_retry_at_twenty_seconds(self):
        segments = [
            SimpleNamespace(
                text="unique opening phrase",
                start=0.0,
                end=0.0,
                words=[],
            )
        ]
        transcription = SimpleNamespace(
            segments=[
                SimpleNamespace(
                    words=[
                        SimpleNamespace(
                            word="unique", start=41.0, end=41.8
                        ),
                        SimpleNamespace(
                            word=" opening", start=41.9, end=42.8
                        ),
                        SimpleNamespace(
                            word=" phrase", start=42.9, end=43.8
                        ),
                    ]
                )
            ]
        )

        self.assertEqual(
            _collapsed_prefix_retry_offset(segments, transcription),
            20.0,
        )

    def test_ignores_isolated_early_match_when_choosing_retry_offset(self):
        segments = [
            SimpleNamespace(text=text, start=0.0, end=0.0, words=[])
            for text in [
                "alpha bravo",
                "charlie delta",
                "echo foxtrot",
                "golf hotel",
            ]
        ]
        transcript_words = [
            SimpleNamespace(word="alpha", start=11.0, end=11.7),
            SimpleNamespace(word=" bravo", start=11.8, end=12.6),
            SimpleNamespace(word="charlie", start=41.0, end=41.8),
            SimpleNamespace(word=" delta", start=41.9, end=42.7),
            SimpleNamespace(word="echo", start=46.0, end=46.7),
            SimpleNamespace(word=" foxtrot", start=46.8, end=47.7),
            SimpleNamespace(word="golf", start=51.0, end=51.7),
            SimpleNamespace(word=" hotel", start=51.8, end=52.6),
        ]
        transcription = SimpleNamespace(
            segments=[SimpleNamespace(words=transcript_words)]
        )

        self.assertEqual(
            _collapsed_prefix_retry_offset(segments, transcription),
            20.0,
        )

    def test_ends_before_stray_word_after_instrumental_gap(self):
        segment = SimpleNamespace(
            start=150.0,
            end=176.0,
            words=[
                SimpleNamespace(word="Joe", start=150.0, end=150.8),
                SimpleNamespace(word=" Joe", start=150.9, end=151.7),
                SimpleNamespace(word=" Joe", start=175.0, end=176.0),
            ],
        )

        self.assertEqual(_segment_end(segment), 151.7)

    def test_preserves_coherent_tail_after_instrumental_break(self):
        starts = [116.40, 118.22, 119.90, 121.78, 123.94, 125.76, 127.58, 129.26, 142.20, 144.70]
        ends = [117.70, 119.40, 121.38, 123.26, 125.14, 127.08, 129.12, 130.98, 144.60, 146.32]
        transcript_words = [SimpleNamespace(end=148.0)]

        repaired_starts, _ = _repair_untrusted_ranges(
            starts.copy(), ends.copy(), {7}, transcript_words, 148.04
        )

        self.assertEqual(repaired_starts[-2:], [142.20, 144.70])


class SectionLabelTests(unittest.TestCase):
    def test_excludes_section_label_from_alignment_text(self):
        lines_fr = ["first line", "", "[Post-refrain]", "second line"]

        self.assertTrue(_is_section_label("[Post-refrain]"))
        self.assertFalse(_is_section_label("[00:12.34]actual lyric"))
        self.assertEqual(_lyrics_for_alignment(lines_fr), "first line\n\n\nsecond line")

    def test_section_label_uses_next_vocal_start(self):
        lines_fr = ["first line", "", "[Post-refrain]", "second line"]
        segments = [
            SimpleNamespace(start=10.0, end=12.0, words=[]),
            SimpleNamespace(start=20.0, end=22.0, words=[]),
        ]

        starts = _build_line_start_times(
            lines_fr, segments, [10.0, 20.0], [12.0, 22.0]
        )

        self.assertEqual(starts, [10.0, 12.0, 20.0, 20.0])


class AdjacentRepeatRecoveryTests(unittest.TestCase):
    def test_recovers_all_occurrences_after_stretched_repeat(self):
        segments = [
            SimpleNamespace(text="Balance ton quoi (Ah-ah)", start=151.0, end=168.0, words=[]),
            SimpleNamespace(text="Balance ton quoi (Ah-ah)", start=169.0, end=170.0, words=[]),
            SimpleNamespace(text="Balance ton quoi", start=170.0, end=170.0, words=[]),
        ]
        transcript_words = []
        for start in (151.0, 156.5, 168.4):
            transcript_words.extend([
                SimpleNamespace(word="Balance", start=start, end=start + 0.5),
                SimpleNamespace(word=" ton", start=start + 0.5, end=start + 0.8),
                SimpleNamespace(word=" quoi", start=start + 0.8, end=start + 1.2),
            ])

        matches = _ordered_transcription_matches(segments, transcript_words)

        self.assertEqual(
            [round(matches[index][1], 1) for index in range(3)],
            [151.0, 156.5, 168.4],
        )


    def test_preserves_trusted_irregular_repeat_cadence(self):
        segments = [
            SimpleNamespace(text="Balance ton quoi", start=start, end=end, words=[])
            for start, end in [(151.0, 153.0), (160.0, 162.0), (168.4, 170.0)]
        ]
        starts = [151.0, 160.0, 168.4]
        ends = [153.0, 162.0, 170.0]

        repaired_starts, _ = _repair_repeated_segment_starts(
            segments, starts.copy(), ends.copy(), 180.0, {0, 1, 2}
        )

        self.assertEqual(repaired_starts, starts)


if __name__ == "__main__":
    unittest.main()
