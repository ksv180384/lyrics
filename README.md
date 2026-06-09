# Liryc

Flask app for aligning multilingual lyrics with audio through `stable-ts` / Whisper.

## Run with Docker

CPU mode:

```bash
docker compose up --build liryc
```

Open http://localhost:5555.

## Run with NVIDIA GPU

GPU mode can speed up Whisper alignment when the host has an NVIDIA GPU, recent NVIDIA drivers, and NVIDIA Container Toolkit installed.

```bash
docker compose --profile gpu up --build liryc-gpu
```

Open http://localhost:5555.

## Configuration

You can choose another Whisper model:

```bash
WHISPER_MODEL=small docker compose up --build liryc
WHISPER_MODEL=large-v3 docker compose --profile gpu up --build liryc-gpu
```

By default, model files are cached in the Docker volume `whisper-models`, and generated `.lrc` files are kept in `./uploads`.
