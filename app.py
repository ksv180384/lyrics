import json
import os
import re
import statistics
import subprocess
import unicodedata
import uuid
import traceback
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
    return len(words) > 1 and duration > max(3.0, _typical_word_duration(words) * 6)


def _segment_start(seg) -> float:
    """Возвращает надёжное начало сегмента с коррекцией аномального первого слова."""
    words = _spoken_words(seg)
    if not words:
        return float(seg.start)
    first = words[0]
    if _is_boundary_outlier(first, words):
        return max(float(first.start), float(first.end) - _typical_word_duration(words))
    return float(first.start)


def _segment_end(seg) -> float:
    """Возвращает надёжный конец сегмента с коррекцией аномального последнего слова."""
    words = _spoken_words(seg)
    if not words:
        return float(seg.end)
    last = words[-1]
    if _is_boundary_outlier(last, words):
        return min(float(last.end), float(last.start) + _typical_word_duration(words))
    return float(last.end)


def _normalize_for_match(text: str) -> str:
    """Нормализует текст для нечувствительного к регистру, акцентам и пунктуации сравнения."""
    decomposed = unicodedata.normalize('NFKD', text.casefold())
    return ''.join(char for char in decomposed if char.isalnum())


def _word_window_boundaries(words: list) -> tuple:
    """Вычисляет очищенные временные границы последовательности распознанных слов."""
    first, last = words[0], words[-1]
    start = float(first.end) - _typical_word_duration(words) if _is_boundary_outlier(first, words) else float(first.start)
    end = float(last.start) + _typical_word_duration(words) if _is_boundary_outlier(last, words) else float(last.end)
    return start, end


def _match_transcribed_boundary(segment, transcript_words: list) -> tuple:
    """Ищет строку forced alignment среди слов проверочной транскрипции."""
    forced_start, forced_end = _segment_start(segment), _segment_end(segment)
    target = _normalize_for_match(segment.text)
    # Короткие повторяющиеся фразы неоднозначны: их оставляем на forced alignment.
    if len(target) < 10:
        return forced_start, forced_end
    best = None
    max_chars, min_chars = int(len(target) * 1.5) + 8, max(1, int(len(target) * 0.6))
    # Ищем текст рядом с исходной меткой; окно 35 секунд покрывает длинные проигрыши.
    for i, word in enumerate(transcript_words):
        if float(word.end) < forced_start - 35:
            continue
        if float(word.start) > forced_start + 35:
            break
        candidate_text = ''
        for j in range(i, len(transcript_words)):
            candidate_text += _normalize_for_match(transcript_words[j].word)
            if len(candidate_text) > max_chars:
                break
            if len(candidate_text) < min_chars:
                continue
            candidate_start, candidate_end = _word_window_boundaries(transcript_words[i:j + 1])
            distance = abs(candidate_start - forced_start)
            if distance > 35:
                continue
            similarity = SequenceMatcher(None, target, candidate_text).ratio()
            rank = similarity - distance * 0.001
            if best is None or rank > best[0]:
                best = (rank, similarity, candidate_start, candidate_end)
    # Порог 0.80 не позволяет исправлять время по слабому текстовому совпадению.
    if best is None or best[1] < 0.80:
        return forced_start, forced_end
    return best[2], best[3]


def _refine_segment_boundaries(segments: list, transcription) -> tuple:
    """Уточняет границы сегментов по уверенным совпадениям проверочной транскрипции."""
    transcript_words = [word for segment in transcription.segments for word in (getattr(segment, 'words', None) or []) if getattr(word, 'word', '').strip()]
    starts, ends = [], []
    for segment in segments:
        forced_start, forced_end = _segment_start(segment), _segment_end(segment)
        matched_start, matched_end = _match_transcribed_boundary(segment, transcript_words)
        # Заменяем обе границы только при существенном расхождении более трёх секунд.
        if max(abs(matched_start - forced_start), abs(matched_end - forced_end)) > 3:
            starts.append(matched_start); ends.append(matched_end)
        else:
            starts.append(forced_start); ends.append(forced_end)
    # После уточнения гарантируем неубывающие начала и корректные интервалы сегментов.
    for i in range(1, len(starts)):
        starts[i] = max(starts[i], starts[i - 1])
        ends[i] = max(ends[i], starts[i])
    for i in range(len(ends) - 1):
        ends[i] = min(ends[i], starts[i + 1])
    return starts, ends


def _needs_transcription_refinement(segments: list) -> bool:
    """Проверяет, есть ли разрыв более 10 секунд, требующий второго прохода."""
    return any(_segment_start(current) - _segment_end(previous) > 10 for previous, current in zip(segments, segments[1:]))


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

    n_ne = _nonempty_line_count(lines_fr)
    if ns != n_ne:
        raise ValueError(
            f'После выравнивания сегментов {ns}, непустых строк во FR {n_ne}, всего строк {n}. '
            'Число непустых строк должно совпадать с числом сегментов. '
            'Уберите лишние переносы или проверьте, что текст FR совпадает с песней.'
        )

    times = [0.0] * n
    j = 0
    for i in range(n):
        if not lines_fr[i].strip():
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
    out.append(_lrc_line(format_lrc_timestamp(lyric_end), ''))
    out.append(_lrc_line(format_lrc_timestamp(duration), ''))
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
    result = m.align(audio_path, lyrics_fr, language=language, original_split=True)
    segments = list(result.segments)
    segment_starts = [_segment_start(seg) for seg in segments]
    segment_ends = [_segment_end(seg) for seg in segments]
    # Второй проход запускается только при подозрительном разрыве, чтобы не замедлять обычные песни.
    if _needs_transcription_refinement(segments):
        print('Suspicious alignment gap detected; verifying timestamps with transcription...')
        transcription = m.transcribe(audio_path, language=language, regroup=False)
        segment_starts, segment_ends = _refine_segment_boundaries(segments, transcription)
    starts = _build_line_start_times(lines_fr, segments, segment_starts, segment_ends)
    first_lyric_start = next(starts[i] for i, line in enumerate(lines_fr) if line.strip())
    lyric_end = max(segment_ends)
    duration = max(get_audio_duration(audio_path), lyric_end)
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
