import os
import uuid
import traceback

from flask import Flask, render_template, request, jsonify, send_file

import stable_whisper

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

WHISPER_MODEL = os.getenv('WHISPER_MODEL', 'medium')
WHISPER_DEVICE = os.getenv('WHISPER_DEVICE')
WHISPER_DOWNLOAD_ROOT = os.getenv('WHISPER_DOWNLOAD_ROOT')

model = None


def get_model():
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
    """[mm:ss.m] или [mm:ss.mmm] — как в примере: целые секунды с .0, иначе миллисекунды."""
    if seconds < 0:
        seconds = 0.0
    m = int(seconds // 60)
    s = round(seconds - m * 60, 3)
    if s >= 60:
        s = 59.999
    whole = int(s)
    frac = s - whole
    if abs(frac) < 1e-6:
        return f"[{m:02d}:{whole:02d}.0]"
    ms = int(round(frac * 1000))
    return f"[{m:02d}:{whole:02d}.{ms:03d}]"


def _split_lines_preserve(text: str) -> list:
    return text.splitlines()


def _nonempty_line_count(lines: list) -> int:
    return sum(1 for ln in lines if ln.strip())


def _build_line_start_times(lines_fr: list, segments: list) -> list:
    """
    Одна временная метка на строку FR.
    Пустые строки в Whisper не дают отдельных сегментов: сегментов столько же,
    сколько непустых строк. Для пустой строки берём конец предыдущей фразы (пауза).
    """
    n = len(lines_fr)
    ns = len(segments)
    if ns == 0:
        raise ValueError('Модель не вернула сегментов — проверьте аудио и текст FR')

    if ns == n:
        return [float(seg.start) for seg in segments]

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
                times[i] = float(segments[j - 1].end)
            else:
                times[i] = max(0.0, float(segments[0].start))
        else:
            times[i] = float(segments[j].start)
            j += 1

    return times


def _lrc_line(tag: str, line: str) -> str:
    line = line if line is not None else ''
    if line.strip() == '':
        return tag
    return f"{tag}{line}"


def align_multilang_lrc(
    audio_path: str,
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
    starts = _build_line_start_times(lines_fr, segments)

    out_fr, out_ru, out_tr = [], [], []
    for i in range(n):
        tag = format_lrc_timestamp(starts[i])
        out_fr.append(_lrc_line(tag, lines_fr[i]))
        out_ru.append(_lrc_line(tag, lines_ru[i]))
        out_tr.append(_lrc_line(tag, lines_tr[i]))

    return (
        "\n".join(out_fr) + "\n",
        "\n".join(out_ru) + "\n",
        "\n".join(out_tr) + "\n",
    )


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/align', methods=['POST'])
def align():
    if 'audio' not in request.files:
        return jsonify({'error': 'MP3 файл не загружен'}), 400

    audio_file = request.files['audio']
    lyrics_fr = request.form.get('lyrics_fr', '')
    lyrics_ru = request.form.get('lyrics_ru', '')
    lyrics_tr = request.form.get('lyrics_tr', '')

    if not lyrics_fr.strip():
        return jsonify({'error': 'Заполните поле FR (французский текст)'}), 400

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
        lrc_fr, lrc_ru, lrc_tr = align_multilang_lrc(
            audio_path,
            lyrics_fr,
            lines_ru,
            lines_tr,
            language=language,
        )
        for suffix, content in (('fr', lrc_fr), ('ru', lrc_ru), ('tr', lrc_tr)):
            path = os.path.join(UPLOAD_DIR, f"{file_id}_{suffix}.lrc")
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)

        return jsonify({
            'lrc_fr': lrc_fr,
            'lrc_ru': lrc_ru,
            'lrc_tr': lrc_tr,
            'download_id': file_id,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Ошибка обработки: {str(e)}'}), 500
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


@app.route('/download/<file_id>/<lang>')
def download(file_id, lang):
    if lang not in ('fr', 'ru', 'tr'):
        return jsonify({'error': 'Неверный язык'}), 400

    lrc_path = os.path.join(UPLOAD_DIR, f"{file_id}_{lang}.lrc")
    if not os.path.exists(lrc_path):
        return jsonify({'error': 'Файл не найден'}), 404

    names = {'fr': 'lyrics-fr.lrc', 'ru': 'lyrics-ru.lrc', 'tr': 'lyrics-tr.lrc'}
    return send_file(
        lrc_path,
        as_attachment=True,
        download_name=names[lang],
        mimetype='text/plain; charset=utf-8',
    )


if __name__ == '__main__':
    print("Загрузка модели Whisper (medium)...")
    get_model()
    print("Модель загружена. Сервер запускается на http://localhost:5555")
    app.run(debug=False, host='0.0.0.0', port=5555)
