import unittest
from types import SimpleNamespace

from app import (
    _collapsed_adjacent_duplicate_range,
    _collapsed_transcription_retry_start,
    _remove_spoken_segment_range,
    _repair_repeated_blocks_before_late_anchor,
    _broken_suffix_start,
    _finalize_segment_boundaries,
    _build_line_start_times,
    _is_section_label,
    _lyrics_for_alignment,
    _collapsed_prefix_retry_offset,
    _has_severely_collapsed_prefix,
    _ordered_transcription_matches,
    _repair_collapsed_segments_before_late_groups,
    _repair_repeated_segment_starts,
    _repair_collapsed_repeated_tail,
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

    def test_uses_refined_starts_to_infer_repeat_shift(self):
        texts = ["lead", "a", "b", "c", "middle", "a", "b", "c", "tail"]
        refined_starts = [0.0, 10.0, 12.0, 14.0, 20.0, 40.0, 42.0, 60.0, 70.0]
        forced_starts = [0.0, 10.0, 12.0, 14.0, 20.0, 30.0, 30.0, 30.0, 70.0]
        ends = [start + 1.0 for start in refined_starts]
        segments = [
            SimpleNamespace(text=text, start=start, end=end, words=[])
            for text, start, end in zip(texts, forced_starts, ends)
        ]

        repaired_starts, _ = _repair_repeated_text_blocks(
            segments,
            refined_starts.copy(),
            ends.copy(),
        )

        self.assertEqual(repaired_starts[5:8], [40.0, 42.0, 44.0])

    def test_prefers_trusted_shift_over_larger_untrusted_cluster(self):
        block = ["a", "b", "c", "d", "e"]
        texts = ["lead", *block, "middle", *block, "tail"]
        source = [10.0, 14.0, 17.0, 21.0, 23.0]
        destination = [72.0, 76.0, 72.0, 83.0, 85.0]
        starts = [0.0, *source, 30.0, *destination, 100.0]
        ends = [start + 1.0 for start in starts]
        segments = [
            SimpleNamespace(text=text, start=start, end=end, words=[])
            for text, start, end in zip(texts, starts, ends)
        ]

        repaired_starts, _ = _repair_repeated_text_blocks(
            segments,
            starts.copy(),
            ends.copy(),
            trusted={9},
        )

        self.assertEqual(repaired_starts[7:12], [65.0, 69.0, 72.0, 76.0, 78.0])

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

    def test_detects_three_collapsed_opening_lines_before_large_gap(self):
        segments = [
            SimpleNamespace(text=f"line {index}", start=2.2, end=2.24, words=[])
            for index in range(3)
        ]
        segments.extend(
            SimpleNamespace(
                text=f"line {index}",
                start=24.48 + index * 2,
                end=25.08 + index * 2,
                words=[],
            )
            for index in range(5)
        )

        self.assertTrue(_has_severely_collapsed_prefix(segments))

    def test_ignores_only_two_collapsed_opening_lines(self):
        segments = [
            SimpleNamespace(text="line 0", start=2.2, end=2.24, words=[]),
            SimpleNamespace(text="line 1", start=2.24, end=2.24, words=[]),
        ]
        segments.extend(
            SimpleNamespace(
                text=f"line {index}",
                start=24.48 + index * 2,
                end=25.08 + index * 2,
                words=[],
            )
            for index in range(6)
        )

        self.assertFalse(_has_severely_collapsed_prefix(segments))

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


class AdjacentDuplicateStanzaTests(unittest.TestCase):
    @staticmethod
    def _segments(collapsed=True):
        texts = ["intro", "a", "b", "c", "a", "b", "c", "outro"]
        starts = [0.0, 10.0, 12.0, 14.0, 20.0, 20.0, 20.0, 40.0]
        ends = [1.0, 11.0, 13.0, 15.0, 20.0, 20.0, 20.0, 41.0]
        if not collapsed:
            starts[4:7] = [20.0, 22.0, 24.0]
            ends[4:7] = [21.0, 23.0, 25.0]
        return [
            SimpleNamespace(text=text, start=start, end=end, words=[])
            for text, start, end in zip(texts, starts, ends)
        ]

    def test_finds_only_the_collapsed_second_adjacent_copy(self):
        self.assertEqual(
            _collapsed_adjacent_duplicate_range(self._segments()),
            (4, 7),
        )
        self.assertIsNone(
            _collapsed_adjacent_duplicate_range(self._segments(collapsed=False))
        )

    def test_removes_matching_lines_in_all_languages(self):
        fr = ["intro", "", "a", "b", "c", "", "a", "b", "c", "", "outro"]
        ru = ["ru " + line if line else "" for line in fr]
        tr = ["tr " + line if line else "" for line in fr]

        filtered_fr, filtered_ru, filtered_tr = _remove_spoken_segment_range(
            fr, ru, tr, (4, 7)
        )

        self.assertEqual(filtered_fr, ["intro", "", "a", "b", "c", "", "outro"])
        self.assertEqual(len(filtered_fr), len(filtered_ru))
        self.assertEqual(len(filtered_fr), len(filtered_tr))

    def test_transcription_retry_starts_before_collapsed_run(self):
        segments = [
            SimpleNamespace(text=str(index), start=float(index), end=float(index + 1), words=[])
            for index in range(6)
        ]
        for index in range(3, 6):
            segments[index].end = segments[index].start

        self.assertEqual(_collapsed_transcription_retry_start(segments), 1)


class LateAnchorRepeatedBlockTests(unittest.TestCase):
    def test_moves_collapsed_repeat_next_to_following_anchor(self):
        texts = ["lead", "a", "b", "c", "anchor", "middle", "a", "b", "c", "late"]
        starts = [0.0, 2.0, 4.0, 6.0, 8.0, 20.0, 22.0, 24.0, 26.0, 40.0]
        ends = [start + 1.0 for start in starts]
        segments = [
            SimpleNamespace(text=text, start=start, end=end, words=[])
            for text, start, end in zip(texts, starts, ends)
        ]
        segments[6].end = segments[6].start

        repaired_starts, _ = _repair_repeated_blocks_before_late_anchor(
            segments,
            starts.copy(),
            ends.copy(),
        )

        self.assertEqual(repaired_starts[6:9], [34.0, 36.0, 38.0])

    def test_ignores_block_with_internal_duplicate_lines(self):
        texts = ["lead", "a", "a", "b", "anchor", "middle", "a", "a", "b", "late"]
        starts = [0.0, 2.0, 4.0, 6.0, 8.0, 20.0, 22.0, 24.0, 26.0, 40.0]
        ends = [start + 1.0 for start in starts]
        segments = [
            SimpleNamespace(text=text, start=start, end=end, words=[])
            for text, start, end in zip(texts, starts, ends)
        ]
        segments[6].end = segments[6].start

        repaired_starts, _ = _repair_repeated_blocks_before_late_anchor(
            segments,
            starts.copy(),
            ends.copy(),
        )

        self.assertEqual(repaired_starts, starts)


class CollapsedRepeatedTailTests(unittest.TestCase):
    def test_preserves_gap_and_backfills_only_final_series(self):
        texts = ["other", "repeat", "repeat", "other", "repeat", "repeat", "repeat", "repeat"]
        starts = [0.0, 10.0, 14.0, 20.0, 30.0, 60.0, 92.0, 93.0]
        ends = [start + 1.0 for start in starts]
        segments = [
            SimpleNamespace(text=text, start=start, end=end, words=[])
            for text, start, end in zip(texts, starts, ends)
        ]

        repaired_starts, _ = _repair_collapsed_repeated_tail(
            segments,
            starts.copy(),
            ends.copy(),
            100.0,
        )

        self.assertEqual(repaired_starts[4:], [30.0, 60.0, 64.0, 68.0])


class BrokenSuffixRecoveryTests(unittest.TestCase):
    @staticmethod
    def _segments(collapsed_tail=True):
        segments = [
            SimpleNamespace(text=f"line {index}", start=index * 2.0, end=index * 2.0 + 1.0, words=[])
            for index in range(15)
        ]
        if collapsed_tail:
            for index in range(11, 15):
                segments[index].start = 40.0
                segments[index].end = 40.0
        return segments

    def test_finds_boundary_before_large_jump_and_collapsed_tail(self):
        segments = self._segments()
        starts = [float(index * 2) for index in range(10)] + [32.0, 40.0, 40.0, 40.0, 40.0]

        self.assertEqual(_broken_suffix_start(segments, starts, 45.0), 9)

    def test_ignores_instrumental_gap_without_collapsed_tail(self):
        segments = self._segments(collapsed_tail=False)
        starts = [float(index * 2) for index in range(10)] + [32.0, 34.0, 36.0, 38.0, 40.0]

        self.assertIsNone(_broken_suffix_start(segments, starts, 45.0))

    def test_backfills_last_failed_line_inside_real_duration(self):
        starts, ends = _finalize_segment_boundaries(
            [209.78, 211.70, 212.74, 216.18, 218.54],
            [211.16, 212.74, 213.90, 218.53, 218.54],
            218.54,
        )

        self.assertGreater(starts[-1], starts[-2])
        self.assertLess(starts[-1], 218.54)
        self.assertEqual(ends[-1], 218.54)
        self.assertEqual(starts, sorted(starts))


if __name__ == "__main__":
    unittest.main()
