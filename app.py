import json
import os
import re
import statistics
import subprocess
import tempfile
import unicodedata
import uuid
import traceback
from itertools import combinations
from difflib import SequenceMatcher
from datetime import datetime, timezone

from flask import Flask, render_template, request, jsonify, send_file

import stable_whisper

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

WHISPER_MODEL = os.getenv('WHISPER_MODEL', 'large-v3-turbo')
WHISPER_DEVICE = os.getenv('WHISPER_DEVICE')
WHISPER_DOWNLOAD_ROOT = os.getenv('WHISPER_DOWNLOAD_ROOT')

model = None


def get_model():
    """Лениво загружает модель Whisper при первом обращении и повторно использует её."""
    global model
    if model is None:
        load_options = {}
        if WHISPER_DEVICE:
            load_options['device'] = WHISPER_DEVICE
        if WHISPER_DOWNLOAD_ROOT:
            load_options['download_root'] = WHISPER_DOWNLOAD_ROOT
        print(f"Loading Whisper model '{WHISPER_MODEL}' on device '{WHISPER_DEVICE or 'auto'}'...")
        model = stable_whisper.load_model(WHISPER_MODEL, **load_options)
    return model


def format_lrc_timestamp(seconds: float) -> str:
    """Преобразует секунды в временную метку LRC формата [мм:сс.сс]."""
    if seconds < 0:
        seconds = 0.0
    total_centiseconds = int(round(seconds * 100))
    m, centiseconds = divmod(total_centiseconds, 60 * 100)
    s, cs = divmod(centiseconds, 100)
    return f"[{m:02d}:{s:02d}.{cs:02d}]"


def format_plain_duration(seconds: float) -> str:
    """Возвращает длительность без квадратных скобок для интерфейса и метаданных."""
    tag = format_lrc_timestamp(seconds)
    return tag[1:-1]


def format_filename_duration(seconds: float) -> str:
    """Форматирует длительность для безопасного включения в имя файла."""
    total_seconds = max(0, int(round(seconds)))
    m, s = divmod(total_seconds, 60)
    return f"{m:02d}m{s:02d}s"


def get_audio_duration(audio_path: str) -> float:
    """Получает полную длительность аудиофайла через ffprobe."""
    completed = subprocess.run(
        [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            audio_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(completed.stdout.strip())


def _safe_filename_part(value: str) -> str:
    """Очищает часть имени файла от запрещённых символов и лишних пробелов."""
    value = re.sub(r'[\\/:*?"<>|\x00-\x1f]', ' ', value.strip())
    value = re.sub(r'\s+', ' ', value)
    return value.strip(' .') or 'song'


def _song_base_filename(artist: str, title: str, duration: float) -> str:
    """Формирует базовое имя файлов песни из исполнителя, названия и длительности."""
    artist_part = _safe_filename_part(artist)
    title_part = _safe_filename_part(title)
    duration_part = format_filename_duration(duration)
    return f"{artist_part} - {title_part} - {duration_part}"


def _unique_song_base(base: str) -> str:
    """Подбирает уникальное базовое имя, не перезаписывая ранее созданные LRC."""
    candidate = base
    index = 2
    while any(os.path.exists(os.path.join(UPLOAD_DIR, f"{candidate} - {lang.upper()}.lrc")) for lang in ('fr', 'ru', 'tr')):
        candidate = f"{base} ({index})"
        index += 1
    return candidate


def _song_meta_path(file_id: str) -> str:
    """Возвращает путь к JSON-файлу метаданных песни."""
    return os.path.join(UPLOAD_DIR, f"{file_id}.json")


def _read_song_meta(file_id: str) -> dict:
    """Читает сохранённые метаданные песни из JSON."""
    with open(_song_meta_path(file_id), 'r', encoding='utf-8') as f:
        return json.load(f)


def _song_paths(meta: dict) -> dict:
    """Преобразует имена языковых файлов из метаданных в полные пути."""
    return {
        lang: os.path.join(UPLOAD_DIR, filename)
        for lang, filename in meta.get('files', {}).items()
    }


def _read_song_payload(file_id: str) -> dict:
    """Собирает метаданные и содержимое трёх LRC-файлов для ответа API."""
    meta = _read_song_meta(file_id)
    paths = _song_paths(meta)
    lrc = {}
    for lang in ('fr', 'ru', 'tr'):
        with open(paths[lang], 'r', encoding='utf-8') as f:
            lrc[lang] = f.read()
    return {
        'download_id': file_id,
        'song': meta,
        'lrc_fr': lrc['fr'],
        'lrc_ru': lrc['ru'],
        'lrc_tr': lrc['tr'],
    }


def _split_lines_preserve(text: str) -> list:
    """Разбивает текст на строки, сохраняя пустые строки как элементы списка."""
    return text.splitlines()


def _nonempty_line_count(lines: list) -> int:
    """Подсчитывает количество непустых строк текста."""
    return sum(1 for ln in lines if ln.strip())


def _is_section_label(line: str) -> bool:
    """Определяет служебную пометку секции, которая не произносится в аудио."""
    return bool(re.fullmatch(r'\s*\[[^\[\]\n]+\]\s*', line or ''))


def _is_spoken_lyric_line(line: str) -> bool:
    """Возвращает True только для строки, которую нужно выравнивать с вокалом."""
    return bool((line or '').strip()) and not _is_section_label(line)


def _lyrics_for_alignment(lines: list) -> str:
    """Сохраняет разбиение текста, исключая из alignment служебные пометки."""
    return '\n'.join(line if _is_spoken_lyric_line(line) else '' for line in lines)


def _spoken_words(seg) -> list:
    """Возвращает только непустые слова сегмента Whisper."""
    return [word for word in (getattr(seg, 'words', None) or []) if getattr(word, 'word', '').strip()]


def _typical_word_duration(words: list) -> float:
    """Оценивает типичную длительность слова, исключая аномально длинные значения."""
    durations = [float(word.end) - float(word.start) for word in words if 0.06 <= float(word.end) - float(word.start) <= 2.0]
    return statistics.median(durations) if durations else 0.5


def _is_boundary_outlier(word, words: list) -> bool:
    """Определяет, растянула ли модель граничное слово на паузу или проигрыш."""
    duration = float(word.end) - float(word.start)
    if len(words) == 1:
        return duration > 1.2
    # stable-ts иногда растягивает единственное слово последней строки до конца
    # файла. Такое слово тоже нужно считать выбросом: соседних слов для сравнения
    # внутри сегмента в этом случае нет.
    return duration > max(3.0, _typical_word_duration(words) * 6)


def _late_word_group_index(words: list):
    """Find a late word group after collapsed false words at a segment boundary."""
    for index in range(1, len(words)):
        gap = float(words[index].start) - float(words[index - 1].end)
        prefix_duration = sum(
            max(0.0, float(word.end) - float(word.start))
            for word in words[:index]
        )
        if gap > 3.0 and prefix_duration < 0.5:
            return index
    return None


def _segment_start(seg) -> float:
    """Возвращает надёжное начало сегмента с коррекцией аномального первого слова."""
    words = _spoken_words(seg)
    if not words:
        return float(seg.start)
    late_group_index = _late_word_group_index(words)
    if late_group_index is not None:
        return max(
            float(words[0].start),
            float(words[late_group_index].start) - 0.5 * late_group_index,
        )
    first = words[0]
    if _is_boundary_outlier(first, words):
        typical_duration = 0.5 if len(words) == 1 else _typical_word_duration(words)
        return max(float(first.start), float(first.end) - typical_duration)
    return float(first.start)


def _segment_end(seg) -> float:
    """Возвращает надёжный конец сегмента с коррекцией аномального последнего слова."""
    words = _spoken_words(seg)
    if not words:
        return float(seg.end)
    for index in range(1, len(words)):
        gap = float(words[index].start) - float(words[index - 1].end)
        prefix_duration = sum(
            max(0.0, float(word.end) - float(word.start))
            for word in words[:index]
        )
        if gap > 5.0 and prefix_duration >= 0.5:
            return float(words[index - 1].end)
    if _late_word_group_index(words) is not None:
        last = words[-1]
        return min(float(last.end), float(last.start) + 0.5)
    last = words[-1]
    if _is_boundary_outlier(last, words):
        if len(words) == 1:
            return float(last.end)
        return min(float(last.end), float(last.start) + _typical_word_duration(words))
    return float(last.end)


def _repair_collapsed_segments_before_late_groups(
    segments: list,
    starts: list,
    ends: list,
    trusted: set,
) -> tuple:
    """Backfill collapsed lines immediately before a reliable late word group."""
    for anchor, segment in enumerate(segments):
        words = _spoken_words(segment)
        if _late_word_group_index(words) is None:
            continue

        cursor = starts[anchor]
        for index in range(anchor - 1, max(-1, anchor - 4), -1):
            previous_words = _spoken_words(segments[index])
            if index in trusted or not previous_words:
                break
            has_collapsed_word = any(
                float(word.end) - float(word.start) < 0.06
                for word in previous_words
            )
            if not has_collapsed_word:
                break

            raw_span = float(segments[index].end) - float(segments[index].start)
            estimated_duration = max(1.2, len(previous_words) * 0.45)
            if 0.5 <= raw_span <= 3.0:
                estimated_duration = max(estimated_duration, raw_span)
            cursor -= estimated_duration + 0.25
            starts[index] = cursor
            ends[index] = min(starts[index + 1], cursor + estimated_duration)

    return starts, ends


def _normalize_for_match(text: str) -> str:
    """Нормализует текст для нечувствительного к регистру, акцентам и пунктуации сравнения."""
    # Бэки и адлибы в скобках часто отсутствуют в транскрипции Whisper. Они
    # остаются в LRC, но не должны мешать найти основную строку в аудио.
    text_without_adlibs = re.sub(r'\([^)]*\)', '', text)
    decomposed = unicodedata.normalize('NFKD', text_without_adlibs.casefold())
    return ''.join(char for char in decomposed if char.isalnum())


def _word_window_boundaries(words: list) -> tuple:
    """Вычисляет очищенные временные границы последовательности распознанных слов."""
    first, last = words[0], words[-1]
    if len(words) == 1:
        end = float(last.end)
        start = end - 0.5 if _is_boundary_outlier(first, words) else float(first.start)
        return max(float(first.start), start), end
    typical_duration = _typical_word_duration(words)
    start = float(first.end) - typical_duration if _is_boundary_outlier(first, words) else float(first.start)
    end = float(last.start) + typical_duration if _is_boundary_outlier(last, words) else float(last.end)
    return start, end


def _transcription_candidates(segment, transcript_words: list) -> list:
    """Возвращает уверенные варианты строки во всей проверочной транскрипции."""
    target = _normalize_for_match(segment.text)
    if len(target) < 5:
        return []
    candidates = []
    max_chars, min_chars = int(len(target) * 1.5) + 8, max(1, int(len(target) * 0.6))
    for i, word in enumerate(transcript_words):
        candidate_text = ''
        for j in range(i, len(transcript_words)):
            candidate_text += _normalize_for_match(transcript_words[j].word)
            if len(candidate_text) > max_chars:
                break
            if len(candidate_text) < min_chars:
                continue
            candidate_start, candidate_end = _word_window_boundaries(transcript_words[i:j + 1])
            similarity = SequenceMatcher(None, target, candidate_text).ratio()
            threshold = 0.82 if len(target) < 10 else 0.80
            if similarity >= threshold:
                candidates.append((i, j + 1, similarity, candidate_start, candidate_end))

    # Оставляем лучшие варианты, но не отбрасываем повторы припева: далее их
    # разрулит глобальный поиск с обязательным сохранением порядка строк.
    return sorted(candidates, key=lambda item: item[2], reverse=True)[:24]


def _collapsed_transcription_retry_start(segments: list):
    """Return a safe retry point before a severe run of collapsed segments."""
    run_start = 0
    while run_start < len(segments):
        run_end = run_start
        while (
            run_end < len(segments)
            and _segment_end(segments[run_end]) - _segment_start(segments[run_end]) <= 0.35
        ):
            run_end += 1
        if run_end - run_start >= 3:
            return max(0, run_start - 2)
        run_start = max(run_start + 1, run_end)
    return None


def _ordered_transcription_matches(
    segments: list,
    transcript_words: list,
    allow_far_from=None,
) -> dict:
    """Находит максимальную цепочку непересекающихся строк в порядке песни."""
    normalized_lines = [_normalize_for_match(segment.text) for segment in segments]
    far_candidate_floor = 0.0
    if allow_far_from is not None and allow_far_from > 0:
        far_candidate_floor = _segment_end(segments[allow_far_from - 1]) - 1.0
    relaxed_repeat_windows = {}
    run_start = 0
    while run_start < len(segments):
        run_end = run_start + 1
        while (
            run_end < len(segments)
            and normalized_lines[run_end] == normalized_lines[run_start]
        ):
            run_end += 1
        if run_end - run_start >= 2 and any(
            float(segments[index].end) - float(segments[index].start) > 10
            for index in range(run_start, run_end)
        ):
            window = (
                min(_segment_start(segments[index]) for index in range(run_start, run_end)) - 3,
                max(float(segments[index].end) for index in range(run_start, run_end)) + 3,
            )
            for index in range(run_start, run_end):
                relaxed_repeat_windows[index] = window
        run_start = run_end

    nodes = []
    for line_index, segment in enumerate(segments):
        forced_start = _segment_start(segment)
        segment_is_stretched = float(segment.end) - float(segment.start) > 10
        repeat_window = relaxed_repeat_windows.get(line_index)
        for candidate in _transcription_candidates(segment, transcript_words):
            allow_far = (
                allow_far_from is not None and line_index >= allow_far_from
            )
            if allow_far and candidate[3] < far_candidate_floor:
                continue
            # Repeated chorus lines can match several real occurrences. A far
            # match is useful only when forced alignment itself stretched this
            # segment across an instrumental passage or a collapsed suffix is
            # being recovered from the last reliable boundary.
            if not allow_far and repeat_window is not None and not (
                repeat_window[0] <= candidate[3] <= repeat_window[1]
            ):
                continue
            if (
                not allow_far
                and repeat_window is None
                and not segment_is_stretched
                and abs(candidate[3] - forced_start) > 5
            ):
                continue
            if not allow_far and repeat_window is None and segment_is_stretched and not (
                forced_start - 3 <= candidate[3] <= float(segment.end) + 3
            ):
                continue
            nodes.append((line_index, *candidate))

    # Уникальные строки служат более сильными опорами, чем короткие повторы
    # припева: иначе длинная цепочка из одних «почему» может перескочить через
    # точно распознанный мост песни.
    occurrence_counts = {
        text: normalized_lines.count(text)
        for text in set(normalized_lines)
    }
    line_weights = [
        1.0
        + (1.5 if occurrence_counts[text] == 1 else 0.0)
        + min(0.75, len(text) / 40)
        for text in normalized_lines
    ]
    scores = [node[3] * line_weights[node[0]] for node in nodes]
    previous = [-1] * len(nodes)
    for current, node in enumerate(nodes):
        line_index, word_start = node[0], node[1]
        for prior in range(current):
            prior_node = nodes[prior]
            if prior_node[0] >= line_index or prior_node[2] > word_start:
                continue
            score = scores[prior] + node[3] * line_weights[line_index]
            if score > scores[current]:
                scores[current] = score
                previous[current] = prior

    if not nodes:
        return {}
    cursor = max(range(len(nodes)), key=scores.__getitem__)
    matches = {}
    while cursor >= 0:
        line_index, _, _, similarity, start, end = nodes[cursor]
        matches[line_index] = (similarity, start, end)
        cursor = previous[cursor]

    # Глобальная цепочка иногда пропускает соседний повтор ради более
    # сильной дальней опоры. Для растянутой серии выбираем локально первые
    # непересекающиеся появления — так сохраняются реальные паузы припева.
    run_start = 0
    while run_start < len(segments):
        run_end = run_start + 1
        while (
            run_end < len(segments)
            and normalized_lines[run_end] == normalized_lines[run_start]
        ):
            run_end += 1
        repeat_window = relaxed_repeat_windows.get(run_start)
        run_length = run_end - run_start
        if repeat_window is not None:
            candidates = sorted(
                (
                    candidate
                    for candidate in _transcription_candidates(
                        segments[run_start], transcript_words
                    )
                    if repeat_window[0] <= candidate[3] <= repeat_window[1]
                ),
                key=lambda candidate: (candidate[0], candidate[1], -candidate[2]),
            )
            viable_chains = [
                chain
                for chain in combinations(candidates, run_length)
                if (
                    all(left[1] <= right[0] for left, right in zip(chain, chain[1:]))
                    and abs(
                        chain[-1][3] - _segment_start(segments[run_end - 1])
                    ) <= 3
                )
            ]
            if viable_chains:
                chain = min(
                    viable_chains,
                    key=lambda candidate_chain: (
                        candidate_chain[-1][1],
                        candidate_chain[0][0],
                        -sum(candidate[2] for candidate in candidate_chain),
                    ),
                )
                for offset, candidate in enumerate(chain):
                    _, _, similarity, start, end = candidate
                    matches[run_start + offset] = (similarity, start, end)
        run_start = run_end
    return matches


def _recover_matches_before_large_gaps(
    segments: list,
    matches: dict,
    transcript_words: list,
) -> dict:
    """Recover lines that forced alignment left before a false large gap."""
    forced_starts = [_segment_start(segment) for segment in segments]
    for right in range(1, len(segments)):
        if forced_starts[right] - forced_starts[right - 1] <= 10:
            continue

        cursor = forced_starts[right]
        for i in range(right - 1, max(-1, right - 9), -1):
            viable = [
                candidate
                for candidate in _transcription_candidates(segments[i], transcript_words)
                if candidate[2] >= 0.86
                and candidate[3] >= forced_starts[i] - 3
                and candidate[4] <= cursor + 0.8
            ]
            if not viable:
                break

            _, _, similarity, start, end = max(
                viable,
                key=lambda candidate: (candidate[2], candidate[3]),
            )
            if abs(start - forced_starts[i]) <= 3:
                break
            matches[i] = (similarity, start, end)
            cursor = start

    return matches

def _collapsed_adjacent_duplicate_range(segments: list):
    """Find a second adjacent stanza copy that collapsed during alignment."""
    normalized = [_normalize_for_match(segment.text) for segment in segments]
    for length in range(min(8, len(segments) // 2), 2, -1):
        for second_start in range(length, len(segments) - length + 1):
            first_start = second_start - length
            if (
                normalized[first_start:second_start]
                != normalized[second_start:second_start + length]
            ):
                continue
            second = segments[second_start:second_start + length]
            collapsed = sum(
                1
                for segment in second
                if _segment_end(segment) - _segment_start(segment) <= 0.35
            )
            span = _segment_end(second[-1]) - _segment_start(second[0])
            if collapsed >= length - 1 or span < max(1.0, length * 0.5):
                return second_start, second_start + length
    return None


def _collapsed_unmatched_repeated_range(
    segments: list,
    matched_indices: set,
):
    """Find an earlier stanza copy that is absent from this audio version."""
    normalized = [_normalize_for_match(segment.text) for segment in segments]
    for destination in range(3, len(segments) - 3):
        best = None
        for source in range(destination):
            length = 0
            while (
                destination + length < len(segments)
                and source + length < destination
                and normalized[source + length]
                == normalized[destination + length]
            ):
                length += 1
            if length < 3:
                continue
            source_deltas = [
                _segment_start(segments[index + 1])
                - _segment_start(segments[index])
                for index in range(source, source + length - 1)
            ]
            if any(delta < 0.5 or delta > 12.0 for delta in source_deltas):
                continue
            if len(set(normalized[destination:destination + length])) < length:
                continue
            candidate = (length, source)
            if best is None or candidate > best:
                best = candidate

        if best is None:
            continue
        length, _ = best
        raw_destination = segments[destination:destination + length]
        collapsed = sum(
            _segment_end(segment) - _segment_start(segment) <= 0.35
            for segment in raw_destination
        )
        if collapsed < length - 1:
            continue
        if any(
            index in matched_indices
            for index in range(destination, destination + length)
        ):
            continue
        if destination + length not in matched_indices:
            continue
        return destination, destination + length
    return None


def _remove_spoken_segment_range(
    lines_fr: list,
    lines_ru: list,
    lines_tr: list,
    segment_range: tuple,
) -> tuple:
    """Remove the corresponding lines in all languages and collapse blank rows."""
    spoken_line_indices = [
        index
        for index, line in enumerate(lines_fr)
        if _is_spoken_lyric_line(line)
    ]
    start, end = segment_range
    removed = set(spoken_line_indices[start:end])
    filtered = ([], [], [])
    for index, values in enumerate(zip(lines_fr, lines_ru, lines_tr)):
        if index in removed:
            continue
        if (
            not values[0].strip()
            and filtered[0]
            and not filtered[0][-1].strip()
        ):
            continue
        for target, value in zip(filtered, values):
            target.append(value)
    return filtered


def _same_repeated_phrase(left: str, right: str) -> bool:
    """Считает соседние полную и сокращённую формы припева одним повтором."""
    left_norm = _normalize_for_match(left)
    right_norm = _normalize_for_match(right)
    shorter, longer = sorted((left_norm, right_norm), key=len)
    return shorter == longer or (len(shorter) >= 6 and longer.startswith(shorter))


def _repair_repeated_segment_starts(
    segments: list,
    starts: list,
    ends: list,
    duration: float,
    trusted: set = None,
) -> tuple:
    """Восстанавливает равномерный ритм серий, где alignment пропустил повторы."""
    trusted = trusted or set()
    run_start = 0
    while run_start < len(segments):
        run_end = run_start + 1
        while (
            run_end < len(segments)
            and _same_repeated_phrase(segments[run_end - 1].text, segments[run_end].text)
        ):
            run_end += 1

        split_at = next(
            (
                index
                for index in range(run_start + 1, run_end)
                if starts[index] - starts[index - 1] > 12.0
            ),
            None,
        )
        if split_at is not None:
            run_end = split_at

        if run_end - run_start >= 3 and not all(
            index in trusted for index in range(run_start, run_end)
        ):
            deltas = [
                starts[i + 1] - starts[i]
                for i in range(run_start, run_end - 1)
                if 2.0 <= starts[i + 1] - starts[i] <= 12.0
            ]
            if deltas:
                if len(deltas) == 2 and max(deltas) > min(deltas) * 1.15:
                    cadence = min(deltas)
                else:
                    cadence = statistics.median(deltas)
                tolerance = max(0.8, cadence * 0.25)
                clustered = [delta for delta in deltas if abs(delta - cadence) <= tolerance]
                if clustered:
                    cadence = statistics.median(clustered)

                anchor = starts[run_start]
                for offset, i in enumerate(range(run_start, run_end)):
                    starts[i] = anchor + cadence * offset
                next_start = starts[run_end] if run_end < len(starts) else duration
                last = run_end - 1
                last_text = _normalize_for_match(segments[last].text)
                end_offset = (
                    min(1.2, cadence * 0.35)
                    if len(last_text) <= 8
                    else cadence * 0.75
                )
                ends[last] = min(next_start, duration, starts[last] + end_offset)

        run_start = run_end

    for i in range(len(ends) - 1):
        ends[i] = min(max(ends[i], starts[i]), starts[i + 1])
    if ends:
        ends[-1] = min(duration, max(ends[-1], starts[-1]))
    return starts, ends


def _repair_patterned_refrain_blocks(
    segments: list,
    starts: list,
    ends: list,
    trusted: set = None,
) -> tuple:
    """Restore A-A-A-B / A-A-A-B refrains from their reliable opening beat."""
    trusted = trusted or set()
    normalized = [_normalize_for_match(segment.text) for segment in segments]
    blocks = []
    index = 0
    while index + 7 < len(segments):
        short_line = normalized[index]
        long_line = normalized[index + 3]
        if (
            len(short_line) >= 6
            and len(long_line) >= 6
            and normalized[index:index + 3] == [short_line] * 3
            and normalized[index + 4:index + 7] == [short_line] * 3
            and normalized[index + 7] == long_line
            and long_line != short_line
        ):
            length = 8
            if index + 8 < len(segments):
                final_line = normalized[index + 8]
                if (
                    len(final_line) >= 5
                    and (
                        short_line.startswith(final_line)
                        or final_line.startswith(short_line)
                    )
                ):
                    length = 9
            blocks.append((index, length))
            index += length
        else:
            index += 1

    if not blocks:
        return starts, ends

    cadence_candidates = []
    for block_start, _ in blocks:
        for left in (block_start, block_start + 1):
            delta = starts[left + 1] - starts[left]
            if left in trusted and left + 1 in trusted and 1.0 <= delta <= 3.0:
                cadence_candidates.append(delta)
    if not cadence_candidates:
        return starts, ends
    cadence = statistics.median(cadence_candidates)

    for block_start, length in blocks:
        trusted_count = sum(
            index in trusted
            for index in range(block_start, block_start + min(8, length))
        )
        anchor = (
            starts[block_start]
            if block_start in trusted
            else _segment_start(segments[block_start])
        )
        offsets = [
            0.0,
            cadence,
            cadence * 2,
            cadence * 3,
            cadence * 4 + 0.5,
            cadence * 5 + 0.5,
            cadence * 6 + 0.5,
            cadence * 7 + 0.5,
            cadence * 8 + 1.0,
        ][:length]
        expected = [anchor + offset for offset in offsets]
        if trusted_count >= min(8, length) and max(
            abs(starts[block_start + offset] - expected[offset])
            for offset in range(length)
        ) <= 1.0:
            continue
        for offset, expected_start in enumerate(expected):
            starts[block_start + offset] = expected_start

    for index in range(len(ends) - 1):
        ends[index] = min(max(ends[index], starts[index]), starts[index + 1])
    if ends:
        ends[-1] = max(ends[-1], starts[-1])
    return starts, ends


def _repair_collapsed_repeated_tail(
    segments: list,
    starts: list,
    ends: list,
    duration: float,
) -> tuple:
    """Backfill a collapsed final repeat series after a real instrumental gap."""
    if len(segments) < 4:
        return starts, ends

    normalized = [_normalize_for_match(segment.text) for segment in segments]
    run_start = len(segments) - 1
    while run_start > 0 and normalized[run_start - 1] == normalized[-1]:
        run_start -= 1
    if len(segments) - run_start < 3:
        return starts, ends

    split_at = next(
        (
            index
            for index in range(run_start + 1, len(segments))
            if starts[index] - starts[index - 1] > 12.0
        ),
        None,
    )
    if split_at is None or len(segments) - split_at < 3:
        return starts, ends
    if not any(
        starts[index] - starts[index - 1] > 12.0
        or starts[index] - starts[index - 1] < 0.5
        for index in range(split_at + 1, len(segments))
    ):
        return starts, ends

    cadences = [
        starts[index] - starts[index - 1]
        for index in range(1, split_at)
        if (
            normalized[index] == normalized[index - 1]
            and 2.0 <= starts[index] - starts[index - 1] <= 6.0
        )
    ]
    if not cadences:
        return starts, ends
    cadence = min(4.5, statistics.median(cadences))
    anchor = starts[split_at]
    for offset, index in enumerate(range(split_at, len(segments))):
        starts[index] = min(duration, anchor + cadence * offset)
        ends[index] = min(duration, starts[index] + min(1.5, cadence * 0.5))
    return starts, ends


def _repair_untrusted_ranges(
    starts: list,
    ends: list,
    trusted: set,
    transcript_words: list,
    duration: float,
) -> tuple:
    """Не позволяет ошибочной старой метке сдвинуть последующие надёжные совпадения."""
    anchors = sorted(trusted)
    for left, right in zip(anchors, anchors[1:]):
        if right - left <= 1:
            continue
        block_is_invalid = any(
            starts[i] <= starts[i - 1] or starts[i] >= starts[right]
            for i in range(left + 1, right)
        )
        if block_is_invalid:
            step = (starts[right] - starts[left]) / (right - left)
            for i in range(left + 1, right):
                starts[i] = starts[left] + step * (i - left)
                ends[i] = starts[i + 1] if i + 1 < right else starts[right]

    if anchors and anchors[-1] < len(starts) - 1:
        last_anchor = anchors[-1]
        tail_is_invalid = (
            starts[last_anchor + 1] <= starts[last_anchor]
            or starts[last_anchor + 1] - ends[last_anchor] > max(20.0, duration * 0.12)
            or starts[-1] >= duration - 0.25
        )
        if tail_is_invalid:
            transcript_end = max(
                (float(word.end) for word in transcript_words),
                default=ends[last_anchor],
            )
            tail_count = len(starts) - last_anchor - 1
            tail_start = min(max(ends[last_anchor], starts[last_anchor]), transcript_end)
            step = max(0.0, transcript_end - tail_start) / (tail_count + 1)
            for offset, i in enumerate(range(last_anchor + 1, len(starts)), start=1):
                starts[i] = tail_start + step * offset
                ends[i] = (
                    starts[i + 1]
                    if i + 1 < len(starts)
                    else max(starts[i], transcript_end)
                )

    return starts, ends


def _has_usable_collapsed_repeat_anchor(segments: list) -> bool:
    """Return whether a collapsed repeat still has a credible first onset."""
    if len(segments) < 3:
        return False
    durations = [
        _segment_end(segment) - _segment_start(segment)
        for segment in segments
    ]
    collapsed = sum(duration <= 0.35 for duration in durations)
    return (
        0.05 <= durations[0] <= 0.35
        and collapsed >= max(2, (len(segments) + 1) // 2)
    )


def _repair_repeated_text_blocks(
    segments: list,
    starts: list,
    ends: list,
    trusted: set = None,
) -> tuple:
    """Копирует проверенный ритм на поздний дословно повторяющийся блок."""
    trusted = trusted or set()
    normalized = [_normalize_for_match(segment.text) for segment in segments]
    forced_starts = [_segment_start(segment) for segment in segments]
    destination = 1

    while destination < len(segments):
        best = None
        for source in range(destination):
            length = 0
            while (
                destination + length < len(segments)
                and source + length < destination
                and normalized[source + length] == normalized[destination + length]
            ):
                length += 1
            if length < 3:
                continue

            source_deltas = [
                forced_starts[i + 1] - forced_starts[i]
                for i in range(source, source + length - 1)
            ]
            if any(delta < 0.5 or delta > 12 for delta in source_deltas):
                continue
            candidate = (length, source)
            if best is None or candidate > best:
                best = candidate

        if best is None:
            destination += 1
            continue

        length, source = best
        source_deltas = [
            starts[source + i + 1] - starts[source + i]
            for i in range(length - 1)
        ]
        destination_deltas = [
            starts[destination + i + 1] - starts[destination + i]
            for i in range(length - 1)
        ]
        raw_destination_deltas = [
            forced_starts[destination + i + 1]
            - forced_starts[destination + i]
            for i in range(length - 1)
        ]
        refined_compression = any(
            destination_delta < source_delta * 0.45
            and raw_delta >= source_delta * 0.65
            for source_delta, destination_delta, raw_delta in zip(
                source_deltas,
                destination_deltas,
                raw_destination_deltas,
            )
        )
        raw_repeat_is_coherent = all(
            0.5 <= delta <= 12.0
            for delta in raw_destination_deltas
        )
        anchored_collapsed_repeat = _has_usable_collapsed_repeat_anchor(
            segments[destination:destination + length]
        )
        timing_is_broken = (
            anchored_collapsed_repeat
            or refined_compression
            or any(
                destination_delta > 12
                or (destination_delta < 0.5 and source_delta >= 0.75)
                for source_delta, destination_delta in zip(
                    source_deltas,
                    destination_deltas,
                )
            )
        )
        if not timing_is_broken:
            destination += 1
            continue

        source_start = starts[source]
        source_offsets = [starts[source + i] - source_start for i in range(length)]
        if raw_repeat_is_coherent and refined_compression:
            # A single transcription match can jump to an earlier repeated
            # phrase and squeeze two otherwise healthy forced onsets together.
            # In that case the internally coherent raw block is safer as-is.
            for offset in range(length):
                index = destination + offset
                starts[index] = forced_starts[index]
                ends[index] = _segment_end(segments[index])
            destination += length
            continue
        # A broken repeat can split into two plausible timing clusters: the
        # beginning may align to an early instrumental while the final lines
        # align to the real vocals. Infer a shared shift from the entire block.
        if anchored_collapsed_repeat:
            # Forced alignment often finds the first sung line of a repeated
            # stanza and then collapses its next lines onto that same instant.
            # Preserve that useful onset and copy only the proven source rhythm.
            destination_start = forced_starts[destination]
        else:
            shift_candidates = [
                starts[destination + i] - starts[source + i]
                for i in range(length)
            ]
            shift_tolerance = 1.0
            ranked_shifts = []
            source_gap_before = (
                source_start - ends[source - 1]
                if source > 0
                else None
            )
            for candidate_shift in shift_candidates:
                inliers = [
                    shift
                    for shift in shift_candidates
                    if abs(shift - candidate_shift) <= shift_tolerance
                ]
                shift = statistics.median(inliers)
                destination_start = source_start + shift
                context_error = 0.0
                if destination > 0 and source_gap_before is not None:
                    destination_gap_before = destination_start - ends[destination - 1]
                    context_error = abs(destination_gap_before - source_gap_before)
                trusted_inliers = sum(
                    1
                    for offset, candidate in enumerate(shift_candidates)
                    if (
                        destination + offset in trusted
                        and abs(candidate - candidate_shift) <= shift_tolerance
                    )
                )
                ranked_shifts.append(
                    (trusted_inliers, len(inliers), -context_error, shift)
                )

            _, _, _, block_shift = max(ranked_shifts)
            destination_start = source_start + block_shift
        for offset in range(length):
            starts[destination + offset] = (
                destination_start + source_offsets[offset]
            )
            ends[destination + offset] = (
                destination_start + ends[source + offset] - source_start
            )
        destination += length

    for i in range(len(ends) - 1):
        ends[i] = min(max(ends[i], starts[i]), starts[i + 1])
    return starts, ends


def _repair_earlier_repeated_text_blocks(
    segments: list,
    starts: list,
    ends: list,
    trusted: set = None,
) -> tuple:
    """Use a healthy later repeat to repair an earlier collapsed occurrence."""
    trusted = trusted or set()
    normalized = [_normalize_for_match(segment.text) for segment in segments]
    for destination in range(len(segments) - 3):
        best = None
        for source in range(destination + 1, len(segments) - 2):
            length = 0
            while (
                destination + length < source
                and source + length < len(segments)
                and normalized[destination + length]
                == normalized[source + length]
            ):
                length += 1
            if length < 3:
                continue
            source_deltas = [
                starts[index + 1] - starts[index]
                for index in range(source, source + length - 1)
            ]
            if any(delta < 0.5 or delta > 12.0 for delta in source_deltas):
                continue
            candidate = (length, -source, source, source_deltas)
            if best is None or candidate[:2] > best[:2]:
                best = candidate

        if best is None:
            continue
        length, _, source, source_deltas = best
        last = destination + length - 1
        if destination not in trusted or last not in trusted:
            continue
        destination_deltas = [
            starts[index + 1] - starts[index]
            for index in range(destination, last)
        ]
        if not any(
            destination_delta < source_delta * 0.55
            or destination_delta > source_delta * 1.8
            for destination_delta, source_delta in zip(
                destination_deltas,
                source_deltas,
            )
        ):
            continue

        source_span = starts[source + length - 1] - starts[source]
        destination_span = starts[last] - starts[destination]
        if source_span <= 0 or destination_span <= 0:
            continue
        scale = destination_span / source_span
        destination_start = starts[destination]
        for offset in range(1, length - 1):
            index = destination + offset
            starts[index] = destination_start + (
                starts[source + offset] - starts[source]
            ) * scale
        for index in range(destination, last):
            ends[index] = min(max(ends[index], starts[index]), starts[index + 1])
    return starts, ends


def _repair_repeated_blocks_before_late_anchor(
    segments: list,
    starts: list,
    ends: list,
) -> tuple:
    """Move collapsed repeated blocks next to reliable following anchors."""
    normalized = [_normalize_for_match(segment.text) for segment in segments]
    while True:
        candidates = []
        for destination in range(1, len(segments) - 2):
            for source in range(destination):
                max_length = 0
                while (
                    destination + max_length < len(segments)
                    and source + max_length < destination
                    and normalized[source + max_length]
                    == normalized[destination + max_length]
                ):
                    max_length += 1
                for length in range(3, max_length + 1):
                    if len(set(normalized[destination:destination + length])) < length:
                        continue
                    if destination + length >= len(segments):
                        continue
                    raw_destination = segments[destination:destination + length]
                    if _has_usable_collapsed_repeat_anchor(raw_destination):
                        # Its first forced onset is more direct evidence than
                        # the distance from the following transcription anchor.
                        continue
                    if not any(
                        _segment_end(segment) - _segment_start(segment) <= 0.35
                        for segment in raw_destination
                    ):
                        continue
                    source_gap_after = (
                        starts[source + length] - starts[source + length - 1]
                    )
                    destination_gap_after = (
                        starts[destination + length]
                        - starts[destination + length - 1]
                    )
                    if (
                        not 0.5 <= source_gap_after <= 10.0
                        or destination_gap_after
                        <= max(10.0, source_gap_after * 2.0)
                    ):
                        continue
                    is_full_block_start = (
                        source == 0
                        or normalized[source - 1] != normalized[destination - 1]
                    )
                    candidates.append(
                        (
                            int(is_full_block_start),
                            destination_gap_after / source_gap_after,
                            length,
                            destination,
                            source,
                            source_gap_after,
                        )
                    )
        if not candidates:
            break

        _, _, length, destination, source, source_gap_after = max(candidates)
        source_offsets = [
            starts[source + offset] - starts[source]
            for offset in range(length)
        ]
        destination_last = starts[destination + length] - source_gap_after
        destination_start = destination_last - source_offsets[-1]
        if destination > 0 and destination_start < ends[destination - 1]:
            break
        for offset in range(length):
            index = destination + offset
            starts[index] = destination_start + source_offsets[offset]
            source_duration = max(
                0.0,
                ends[source + offset] - starts[source + offset],
            )
            next_start = (
                destination_start + source_offsets[offset + 1]
                if offset + 1 < length
                else starts[destination + length]
            )
            ends[index] = min(next_start, starts[index] + source_duration)

    return starts, ends


def _refine_segment_boundaries(segments: list, transcription, duration: float) -> tuple:
    """Уточняет границы сегментов по уверенным совпадениям проверочной транскрипции."""
    transcript_words = [word for segment in transcription.segments for word in (getattr(segment, 'words', None) or []) if getattr(word, 'word', '').strip()]
    retry_start = _collapsed_transcription_retry_start(segments)
    matches = _ordered_transcription_matches(
        segments,
        transcript_words,
        allow_far_from=retry_start,
    )
    matches = _recover_matches_before_large_gaps(
        segments,
        matches,
        transcript_words,
    )
    starts, ends, trusted = [], [], set()
    transcript_start = min((float(word.start) for word in transcript_words), default=0.0)
    for i, segment in enumerate(segments):
        forced_start, forced_end = _segment_start(segment), _segment_end(segment)
        _, matched_start, matched_end = matches.get(i, (0.0, forced_start, forced_end))
        # Транскрипция песен нередко пропускает вступление. Не переносим на её
        # первое найденное слово строки, которые forced alignment уже поставил раньше.
        if forced_start < transcript_start <= matched_start:
            matched_start, matched_end = forced_start, forced_end
        elif i in matches:
            trusted.add(i)
        # Уверенное глобальное совпадение точнее forced alignment уже при
        # расхождении от полутора секунд.
        if max(abs(matched_start - forced_start), abs(matched_end - forced_end)) > 1.5:
            starts.append(matched_start); ends.append(matched_end)
        else:
            starts.append(forced_start); ends.append(forced_end)
    starts, ends = _repair_collapsed_segments_before_late_groups(
        segments,
        starts,
        ends,
        trusted,
    )
    starts, ends = _repair_untrusted_ranges(
        starts,
        ends,
        trusted,
        transcript_words,
        duration,
    )
    # После уточнения гарантируем неубывающие начала и корректные интервалы сегментов.
    for i in range(1, len(starts)):
        starts[i] = max(starts[i], starts[i - 1])
        ends[i] = max(ends[i], starts[i])
    for i in range(len(ends) - 1):
        ends[i] = min(ends[i], starts[i + 1])
    starts, ends = _repair_repeated_segment_starts(
        segments,
        starts,
        ends,
        duration,
        trusted,
    )
    starts, ends = _repair_patterned_refrain_blocks(
        segments,
        starts,
        ends,
        trusted,
    )
    starts, ends = _repair_repeated_text_blocks(
        segments,
        starts,
        ends,
        trusted,
    )
    starts, ends = _repair_earlier_repeated_text_blocks(
        segments,
        starts,
        ends,
        trusted,
    )
    starts, ends = _repair_collapsed_repeated_tail(
        segments,
        starts,
        ends,
        duration,
    )
    return _repair_repeated_blocks_before_late_anchor(
        segments,
        starts,
        ends,
    )


def _broken_suffix_start(
    segments: list,
    starts: list,
    duration: float,
):
    """Find a late block that jumped forward before a collapsed tail."""
    if len(segments) < 8 or len(starts) != len(segments):
        return None
    tail_start = max(0, len(segments) - 24)
    collapsed_tail = [
        index
        for index in range(tail_start, len(segments))
        if (
            _segment_end(segments[index]) - _segment_start(segments[index]) <= 0.25
            or starts[index] >= duration - 0.25
        )
    ]
    if len(collapsed_tail) < 4:
        return None

    first_collapsed = min(collapsed_tail)
    if starts[first_collapsed] - starts[first_collapsed - 1] > 10:
        return max(1, first_collapsed - 1)
    forced_starts = [_segment_start(segment) for segment in segments]
    gaps = []
    for index in range(1, first_collapsed):
        gap = max(
            starts[index] - starts[index - 1],
            forced_starts[index] - forced_starts[index - 1],
        )
        if gap > 10:
            gaps.append((gap, index))
    if not gaps:
        return None
    _, right = max(gaps)
    return max(1, right - 1)


def _late_tail_block_start(segments: list):
    """Find a short final lyric block after a long omitted-vocal/instrumental gap."""
    candidates = [
        index
        for index in range(1, len(segments))
        if (
            3 <= len(segments) - index <= 8
            and _segment_start(segments[index])
            - _segment_end(segments[index - 1]) > 8.0
        )
    ]
    return candidates[-1] if candidates else None


def _realign_late_tail_block(
    model,
    audio_path: str,
    language: str,
    segments: list,
) -> list:
    """Realign only a short final block so preceding omitted vocals cannot trap it."""
    tail_index = _late_tail_block_start(segments)
    if tail_index is None:
        return segments
    previous_end = _segment_end(segments[tail_index - 1])
    offset = max(0.0, previous_end + 2.0)
    tail_lyrics = '\n'.join(
        str(segment.text).strip()
        for segment in segments[tail_index:]
    )
    handle = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    clipped_path = handle.name
    handle.close()
    try:
        subprocess.run(
            [
                'ffmpeg', '-y', '-ss', f'{offset:.3f}', '-i', audio_path,
                '-map', '0:a:0', '-vn', '-c:a', 'pcm_s16le', clipped_path,
            ],
            capture_output=True,
            check=True,
        )
        recovered = model.align(
            clipped_path,
            tail_lyrics,
            language=language,
            original_split=True,
        )
        if recovered is None:
            return segments
        recovered.offset_time(offset)
        tail = list(recovered.segments)
        if len(tail) != len(segments) - tail_index:
            return segments
        starts = [_segment_start(segment) for segment in tail]
        if (
            starts[0] < previous_end
            or starts[0] > _segment_start(segments[tail_index]) + 3.0
            or any(current < previous for previous, current in zip(starts, starts[1:]))
        ):
            return segments
        return [*segments[:tail_index], *tail]
    finally:
        try:
            os.remove(clipped_path)
        except OSError:
            pass


def _realign_broken_suffix(
    model,
    audio_path: str,
    language: str,
    segments: list,
    starts: list,
    ends: list,
    duration: float,
):
    """Retry only a displaced suffix after the last reliable boundary."""
    suffix_index = _broken_suffix_start(segments, starts, duration)
    if suffix_index is None:
        return None
    offset = max(0.0, min(duration - 1.0, ends[suffix_index - 1] - 1.0))
    suffix_lyrics = '\n'.join(
        str(segment.text).strip() for segment in segments[suffix_index:]
    )
    suffix = '.wav'
    handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    clipped_path = handle.name
    handle.close()
    try:
        subprocess.run(
            [
                'ffmpeg', '-y', '-ss', f'{offset:.3f}', '-i', audio_path,
                '-map', '0:a:0', '-vn', '-c:a', 'pcm_s16le', clipped_path,
            ],
            capture_output=True,
            check=True,
        )
        recovered = model.align(
            clipped_path,
            suffix_lyrics,
            language=language,
            original_split=True,
        )
        if recovered is None:
            return None
        recovered.offset_time(offset)
        recovered_segments = list(recovered.segments)
        if len(recovered_segments) != len(segments) - suffix_index:
            return None
        recovered_starts = [_segment_start(segment) for segment in recovered_segments]
        collapsed = sum(
            1
            for segment in recovered_segments
            if _segment_end(segment) - _segment_start(segment) <= 0.25
        )
        if (
            collapsed > 2
            or recovered_starts[0] < starts[suffix_index - 1] - 0.5
            or recovered_starts[0] > starts[suffix_index - 1] + 10
            or any(
                current < previous
                for previous, current in zip(recovered_starts, recovered_starts[1:])
            )
        ):
            return None
        recovered_segments = _realign_late_tail_block(
            model,
            audio_path,
            language,
            recovered_segments,
        )
        return suffix_index, recovered_segments
    finally:
        try:
            os.remove(clipped_path)
        except OSError:
            pass


def _finalize_segment_boundaries(
    starts: list,
    ends: list,
    duration: float,
) -> tuple:
    """Keep repaired timestamps monotonic and inside the real audio."""
    starts = [min(duration, max(0.0, float(start))) for start in starts]
    ends = [min(duration, max(0.0, float(end))) for end in ends]
    trailing = len(starts)
    while trailing > 0 and (
        starts[trailing - 1] >= duration - 0.05
        and ends[trailing - 1] - starts[trailing - 1] <= 0.10
    ):
        trailing -= 1
    trailing_count = len(starts) - trailing
    if 0 < trailing_count <= 3 and trailing > 0:
        deltas = [
            starts[index] - starts[index - 1]
            for index in range(max(1, trailing - 5), trailing)
            if 0.5 <= starts[index] - starts[index - 1] <= 5.0
        ]
        cadence = statistics.median(deltas) if deltas else 1.0
        available = max(0.0, duration - starts[trailing - 1])
        cadence = min(cadence, available / (trailing_count + 0.25))
        for offset, index in enumerate(range(trailing, len(starts)), start=1):
            starts[index] = min(duration - 0.01, starts[trailing - 1] + cadence * offset)
            ends[index] = duration

    for index in range(1, len(starts)):
        starts[index] = max(starts[index], starts[index - 1])
    for index in range(len(ends)):
        ends[index] = max(starts[index], min(duration, ends[index]))
        if index + 1 < len(starts):
            ends[index] = min(ends[index], starts[index + 1])
    return starts, ends


def _needs_transcription_refinement(segments: list, duration: float) -> bool:
    """Ищет разрывы, сжатие текста и растянутые до конца файла слова."""
    if any(_segment_start(current) - _segment_end(previous) > 10 for previous, current in zip(segments, segments[1:])):
        return True
    # Forced alignment может растянуть конец предыдущей строки до начала
    # следующей. Тогда разрыв между сегментами не виден, но их начала уже
    # расходятся на десятки секунд.
    starts = [_segment_start(segment) for segment in segments]
    if any(current - previous > 10 for previous, current in zip(starts, starts[1:])):
        return True
    if any(
        float(word.end) - float(word.start) > 10
        for segment in segments
        for word in _spoken_words(segment)
    ):
        return True
    if any(current - previous < 0.25 for previous, current in zip(starts, starts[1:])):
        return True
    # Неуспешный последний сегмент stable-ts бывает пустым и получает end,
    # равный длительности файла. В этом случае смотрим на начало последней строки.
    if starts and duration - starts[-1] > max(30.0, duration * 0.20):
        return True
    reliable_end = max((_segment_end(segment) for segment in segments), default=0.0)
    return duration - reliable_end > max(20.0, duration * 0.15)


def _has_displaced_repeated_prefix(segments: list, matches: dict) -> bool:
    """Detect an opening stanza aligned before its first audible refrain word."""
    normalized = [_normalize_for_match(segment.text) for segment in segments]
    repeated_indices = [
        line_index
        for line_index, text in enumerate(normalized)
        if line_index >= 3 and text and text in normalized[:line_index]
    ]
    for line_index in repeated_indices:
        repeated_start = _segment_start(segments[line_index])
        first_copy = normalized[:line_index].index(normalized[line_index])
        next_start = (
            _segment_start(segments[line_index + 1])
            if line_index + 1 < len(segments)
            else repeated_start
        )
        next_match = matches.get(line_index + 1)
        if next_match is not None and next_match[2] - next_match[1] <= 10:
            next_start = max(next_start, next_match[1])
        # In the broken Garou alignment the first stanza fills the instrumental
        # intro, its repeated cue lands on the first real "Gitan", and the next
        # line jumps thirty seconds to the second stanza. This shape is useful
        # even when Whisper turns the intro into one unusably long word.
        if (
            line_index + 1 < len(segments)
            and _segment_start(segments[first_copy]) < 5
            and repeated_start >= 15
            and _segment_end(segments[line_index - 1]) <= repeated_start + 1.5
            and next_start - repeated_start > 15
        ):
            return True

    if not matches:
        return False
    for line_index in sorted(matches):
        if line_index < 3:
            continue
        text = normalized[line_index]
        if not text or text not in normalized[:line_index]:
            continue
        # A trustworthy transcription match for a repeated cue (for example
        # the second textual "Gitan") can actually be the first sung cue when
        # forced alignment placed the whole preceding stanza in an instrumental
        # intro. Any earlier transcription anchor would disprove that pattern.
        if any(index < line_index for index in matches):
            continue
        _, matched_start, _ = matches[line_index]
        if matched_start < 15:
            continue
        if _segment_end(segments[line_index - 1]) > matched_start + 1.5:
            continue
        if matched_start - _segment_start(segments[0]) < 10:
            continue
        return True
    return False


def _has_severely_collapsed_prefix(segments: list, transcription=None) -> bool:
    """Detect when many opening lyric lines were squeezed into the intro."""
    prefix = segments[:min(12, len(segments))]
    if len(prefix) < 8:
        return False
    leading_collapsed = 0
    for segment in prefix:
        if _segment_end(segment) - _segment_start(segment) > 0.25:
            break
        leading_collapsed += 1
    if (
        leading_collapsed >= 3
        and leading_collapsed < len(prefix)
        and _segment_start(prefix[leading_collapsed])
        - _segment_end(prefix[leading_collapsed - 1]) > 10
    ):
        return True
    starts = [_segment_start(segment) for segment in prefix]
    collapsed = sum(
        1
        for segment in prefix
        if _segment_end(segment) - _segment_start(segment) <= 0.25
    )
    if collapsed >= 4 and starts[-1] - starts[0] <= 20:
        return True
    if transcription is None:
        return False
    transcript_words = [
        word
        for segment in transcription.segments
        for word in (getattr(segment, 'words', None) or [])
        if getattr(word, 'word', '').strip()
    ]
    matches = _ordered_transcription_matches(
        segments,
        transcript_words,
        allow_far_from=_collapsed_transcription_retry_start(segments),
    )
    return _has_displaced_repeated_prefix(segments, matches)


def _collapsed_prefix_retry_offset(segments: list, transcription) -> float:
    """Choose a safe point before the first reliable lyric match."""
    transcript_words = [
        word
        for segment in transcription.segments
        for word in (getattr(segment, 'words', None) or [])
        if getattr(word, 'word', '').strip()
    ]
    candidate_points = []
    for line_index, segment in enumerate(segments[:min(12, len(segments))]):
        for candidate in _transcription_candidates(segment, transcript_words):
            _, _, similarity, start, end = candidate
            if similarity >= 0.88 and end - start <= 8:
                candidate_points.append((line_index, start))
    if not candidate_points:
        return 0.0

    clustered_starts = [
        start
        for _, start in candidate_points
        if len({
            line_index
            for line_index, other_start in candidate_points
            if start <= other_start <= start + 20
        }) >= 3
    ]
    first_reliable_start = min(clustered_starts or [
        start for _, start in candidate_points
    ])
    return min(20.0, max(0.0, first_reliable_start - 10.0))


def _realign_after_collapsed_prefix(
    model,
    audio_path: str,
    lyrics: str,
    language: str,
    segments: list,
    transcription,
):
    """Retry alignment after removing an instrumental intro that trapped lyrics."""
    offset = _collapsed_prefix_retry_offset(segments, transcription)
    if offset < 3:
        return None

    suffix = '.wav'
    handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    clipped_path = handle.name
    handle.close()
    try:
        subprocess.run(
            [
                'ffmpeg', '-y', '-ss', f'{offset:.3f}', '-i', audio_path,
                '-map', '0:a:0', '-vn', '-c:a', 'pcm_s16le', clipped_path,
            ],
            capture_output=True,
            check=True,
        )
        recovered = model.align(
            clipped_path,
            lyrics,
            language=language,
            original_split=True,
        )
        if recovered is None:
            return None
        recovered.offset_time(offset)
        recovered_segments = list(recovered.segments)
        if _has_severely_collapsed_prefix(recovered_segments):
            return None
        return recovered
    finally:
        try:
            os.remove(clipped_path)
        except OSError:
            pass


def _build_line_start_times(
    lines_fr: list,
    segments: list,
    segment_starts: list = None,
    segment_ends: list = None,
) -> list:
    """
    Одна временная метка на строку FR.
    Пустые строки в Whisper не дают отдельных сегментов: сегментов столько же,
    сколько непустых строк. Для пустой строки берём конец предыдущей фразы (пауза).
    """
    n = len(lines_fr)
    ns = len(segments)
    if ns == 0:
        raise ValueError('Модель не вернула сегментов — проверьте аудио и текст FR')

    if segment_starts is None:
        segment_starts = [_segment_start(seg) for seg in segments]
    if segment_ends is None:
        segment_ends = [_segment_end(seg) for seg in segments]

    if ns == n:
        return segment_starts

    n_spoken = sum(1 for line in lines_fr if _is_spoken_lyric_line(line))
    if ns != n_spoken:
        raise ValueError(
            f'После выравнивания сегментов {ns}, произносимых строк во FR {n_spoken}, всего строк {n}. '
            'Число произносимых строк должно совпадать с числом сегментов. '
            'Уберите лишние переносы или проверьте, что текст FR совпадает с песней.'
        )

    times = [0.0] * n
    j = 0
    for i in range(n):
        if _is_section_label(lines_fr[i]):
            times[i] = segment_starts[j] if j < ns else segment_ends[-1]
        elif not lines_fr[i].strip():
            if j > 0:
                times[i] = segment_ends[j - 1]
            else:
                times[i] = max(0.0, segment_starts[0])
        else:
            times[i] = segment_starts[j]
            j += 1

    return times


def _lrc_line(tag: str, line: str) -> str:
    """Объединяет временную метку и текст, не добавляя текст к пустой строке."""
    line = line if line is not None else ''
    if line.strip() == '':
        return tag
    return f"{tag}{line}"


def _song_label(artist: str, title: str) -> str:
    """Формирует отображаемую подпись песни из исполнителя и названия."""
    artist = artist.strip()
    title = title.strip()
    if artist and title:
        return f"{artist} - {title}"
    return artist or title


def _build_lrc(
    lines: list,
    starts: list,
    song_label: str,
    first_lyric_start: float,
    lyric_end: float,
    duration: float,
) -> str:
    """Собирает полный LRC-текст с заголовком, строками песни и конечными метками."""
    title_pause = first_lyric_start / 2
    out = [
        _lrc_line(format_lrc_timestamp(0.0), song_label),
        _lrc_line(format_lrc_timestamp(title_pause), ''),
    ]
    for i, line in enumerate(lines):
        out.append(_lrc_line(format_lrc_timestamp(starts[i]), line))
    lyric_end_tag = format_lrc_timestamp(lyric_end)
    duration_tag = format_lrc_timestamp(duration)
    out.append(_lrc_line(lyric_end_tag, ""))
    if duration_tag != lyric_end_tag:
        out.append(_lrc_line(duration_tag, ""))
    return "\n".join(out) + "\n"


def align_multilang_lrc(
    audio_path: str,
    artist: str,
    title: str,
    lyrics_fr: str,
    lines_ru: list,
    lines_tr: list,
    language: str = 'fr',
) -> tuple:
    """
    Выравнивание по французскому тексту; одинаковые таймкоды на каждой строке для FR, RU, TR.
    Возвращает три строки LRC: (fr, ru, tr).
    """
    lines_fr = _split_lines_preserve(lyrics_fr)
    n = len(lines_fr)
    if n != len(lines_ru) or n != len(lines_tr):
        raise ValueError(
            f'Число строк должно совпадать: FR={len(lines_fr)}, RU={len(lines_ru)}, TR={len(lines_tr)}'
        )
    if n == 0:
        raise ValueError('Текст FR пуст')

    m = get_model()
    alignment_lyrics = _lyrics_for_alignment(lines_fr)
    result = m.align(audio_path, alignment_lyrics, language=language, original_split=True)
    segments = list(result.segments)
    duplicate_range = _collapsed_adjacent_duplicate_range(segments)
    if duplicate_range is not None:
        print('Collapsed adjacent duplicate stanza detected; removing the extra copy...')
        lines_fr, lines_ru, lines_tr = _remove_spoken_segment_range(
            lines_fr,
            lines_ru,
            lines_tr,
            duplicate_range,
        )
        n = len(lines_fr)
        alignment_lyrics = _lyrics_for_alignment(lines_fr)
        result = m.align(
            audio_path,
            alignment_lyrics,
            language=language,
            original_split=True,
        )
        segments = list(result.segments)
    segment_starts = [_segment_start(seg) for seg in segments]
    segment_ends = [_segment_end(seg) for seg in segments]
    duration = get_audio_duration(audio_path)
    # Второй проход запускается только при подозрительном разрыве, чтобы не замедлять обычные песни.
    if _needs_transcription_refinement(segments, duration):
        print('Suspicious alignment gap detected; verifying timestamps with transcription...')
        transcription = m.transcribe(audio_path, language=language, regroup=False)
        transcript_words = [
            word
            for segment in transcription.segments
            for word in (getattr(segment, 'words', None) or [])
            if getattr(word, 'word', '').strip()
        ]
        verification_matches = _ordered_transcription_matches(
            segments,
            transcript_words,
            allow_far_from=_collapsed_transcription_retry_start(segments),
        )
        unmatched_repeat = _collapsed_unmatched_repeated_range(
            segments,
            set(verification_matches),
        )
        if unmatched_repeat is not None:
            print('Unsung repeated stanza detected; removing it from this audio version...')
            lines_fr, lines_ru, lines_tr = _remove_spoken_segment_range(
                lines_fr,
                lines_ru,
                lines_tr,
                unmatched_repeat,
            )
            n = len(lines_fr)
            alignment_lyrics = _lyrics_for_alignment(lines_fr)
            result = m.align(
                audio_path,
                alignment_lyrics,
                language=language,
                original_split=True,
            )
            segments = list(result.segments)
        recovered_prefix = False
        if _has_severely_collapsed_prefix(segments, transcription):
            print('Collapsed lyric prefix detected; retrying after the instrumental intro...')
            recovered = _realign_after_collapsed_prefix(
                m,
                audio_path,
                alignment_lyrics,
                language,
                segments,
                transcription,
            )
            if recovered is not None:
                segments = list(recovered.segments)
                segment_starts = [_segment_start(seg) for seg in segments]
                segment_ends = [_segment_end(seg) for seg in segments]
                recovered_prefix = True
        if _needs_transcription_refinement(segments, duration):
            segment_starts, segment_ends = _refine_segment_boundaries(
                segments,
                transcription,
                duration,
            )
            recovered_suffix = _realign_broken_suffix(
                m,
                audio_path,
                language,
                segments,
                segment_starts,
                segment_ends,
                duration,
            )
            if recovered_suffix is not None:
                suffix_index, suffix_segments = recovered_suffix
                print('Displaced lyric suffix detected; retrying from the last reliable boundary...')
                segment_starts[suffix_index:] = [
                    _segment_start(segment) for segment in suffix_segments
                ]
                segment_ends[suffix_index:] = [
                    _segment_end(segment) for segment in suffix_segments
                ]
                segment_starts, segment_ends = _repair_collapsed_repeated_tail(
                    segments,
                    segment_starts,
                    segment_ends,
                    duration,
                )
    segment_starts, segment_ends = _finalize_segment_boundaries(
        segment_starts, segment_ends, duration
    )
    starts = _build_line_start_times(lines_fr, segments, segment_starts, segment_ends)
    first_lyric_start = next(
        starts[i] for i, line in enumerate(lines_fr)
        if _is_spoken_lyric_line(line)
    )
    lyric_end = max(segment_ends)
    lyric_end = min(duration, lyric_end)
    label = _song_label(artist, title)

    return (
        _build_lrc(lines_fr, starts, label, first_lyric_start, lyric_end, duration),
        _build_lrc(lines_ru, starts, label, first_lyric_start, lyric_end, duration),
        _build_lrc(lines_tr, starts, label, first_lyric_start, lyric_end, duration),
        duration,
        lyric_end,
    )


@app.route('/')
def index():
    """Отображает главную страницу веб-интерфейса."""
    return render_template('index.html')


@app.route('/align', methods=['POST'])
def align():
    """Принимает аудио и три текста, выполняет выравнивание и сохраняет LRC-файлы."""
    if 'audio' not in request.files:
        return jsonify({'error': 'MP3 файл не загружен'}), 400

    audio_file = request.files['audio']
    artist = request.form.get('artist', '')
    title = request.form.get('title', '')
    lyrics_fr = request.form.get('lyrics_fr', '')
    lyrics_ru = request.form.get('lyrics_ru', '')
    lyrics_tr = request.form.get('lyrics_tr', '')

    if not lyrics_fr.strip():
        return jsonify({'error': 'Заполните поле FR (французский текст)'}), 400

    if not artist.strip() or not title.strip():
        return jsonify({'error': 'Укажите исполнителя и название песни'}), 400

    if not audio_file.filename:
        return jsonify({'error': 'MP3 файл не выбран'}), 400

    lines_fr = _split_lines_preserve(lyrics_fr)
    lines_ru = _split_lines_preserve(lyrics_ru)
    lines_tr = _split_lines_preserve(lyrics_tr)
    if len(lines_fr) != len(lines_ru) or len(lines_fr) != len(lines_tr):
        return jsonify({
            'error': (
                f'Число строк должно совпадать: FR — {len(lines_fr)}, '
                f'RU — {len(lines_ru)}, TR — {len(lines_tr)}'
            ),
        }), 400

    file_id = str(uuid.uuid4())
    ext = os.path.splitext(audio_file.filename)[1] or '.mp3'
    audio_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")

    try:
        audio_file.save(audio_path)
        language = request.form.get('language', 'fr')
        lrc_fr, lrc_ru, lrc_tr, duration, lyric_end = align_multilang_lrc(
            audio_path,
            artist,
            title,
            lyrics_fr,
            lines_ru,
            lines_tr,
            language=language,
        )
        base = _unique_song_base(_song_base_filename(artist, title, duration))
        files = {}
        for suffix, content in (('fr', lrc_fr), ('ru', lrc_ru), ('tr', lrc_tr)):
            filename = f"{base} - {suffix.upper()}.lrc"
            files[suffix] = filename
            path = os.path.join(UPLOAD_DIR, filename)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)

        meta = {
            'id': file_id,
            'artist': artist.strip(),
            'title': title.strip(),
            'label': _song_label(artist, title),
            'duration': duration,
            'duration_text': format_plain_duration(duration),
            'lyric_end': lyric_end,
            'lyric_end_text': format_plain_duration(lyric_end),
            'created_at': datetime.now(timezone.utc).isoformat(),
            'files': files,
        }
        with open(_song_meta_path(file_id), 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        return jsonify({
            'lrc_fr': lrc_fr,
            'lrc_ru': lrc_ru,
            'lrc_tr': lrc_tr,
            'download_id': file_id,
            'song': meta,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Ошибка обработки: {str(e)}'}), 500
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


@app.route('/download/<file_id>/<lang>')
def download(file_id, lang):
    """Возвращает выбранный языковой LRC-файл для скачивания."""
    if lang not in ('fr', 'ru', 'tr'):
        return jsonify({'error': 'Неверный язык'}), 400

    download_name = {'fr': 'lyrics-fr.lrc', 'ru': 'lyrics-ru.lrc', 'tr': 'lyrics-tr.lrc'}[lang]
    meta_path = _song_meta_path(file_id)
    if os.path.exists(meta_path):
        meta = _read_song_meta(file_id)
        lrc_path = _song_paths(meta).get(lang)
        download_name = meta.get('files', {}).get(lang, download_name)
    else:
        lrc_path = os.path.join(UPLOAD_DIR, f"{file_id}_{lang}.lrc")

    if not os.path.exists(lrc_path):
        return jsonify({'error': 'Файл не найден'}), 404

    return send_file(
        lrc_path,
        as_attachment=True,
        download_name=download_name,
        mimetype='text/plain; charset=utf-8',
    )


@app.route('/songs')
def songs():
    """Возвращает список ранее обработанных песен и их метаданные."""
    items = []
    for name in os.listdir(UPLOAD_DIR):
        if not name.endswith('.json'):
            continue
        file_id = name[:-5]
        try:
            meta = _read_song_meta(file_id)
        except Exception:
            continue
        items.append({
            'id': meta.get('id', file_id),
            'artist': meta.get('artist', ''),
            'title': meta.get('title', ''),
            'label': meta.get('label', ''),
            'duration_text': meta.get('duration_text', ''),
            'created_at': meta.get('created_at', ''),
            'files': meta.get('files', {}),
        })

    items.sort(key=lambda item: item.get('created_at') or '', reverse=True)
    return jsonify({'songs': items})


@app.route('/songs/<file_id>')
def song(file_id):
    """Возвращает метаданные и тексты одной сохранённой песни по её идентификатору."""
    if not re.fullmatch(r'[0-9a-fA-F-]{36}', file_id):
        return jsonify({'error': 'Неверный ID песни'}), 400
    if not os.path.exists(_song_meta_path(file_id)):
        return jsonify({'error': 'Песня не найдена'}), 404
    return jsonify(_read_song_payload(file_id))
if __name__ == '__main__':
    print(f"Server starts on http://localhost:5555. Whisper model '{WHISPER_MODEL}' will load on first alignment.")
    app.run(debug=False, host='0.0.0.0', port=5555)
