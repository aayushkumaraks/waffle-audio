FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/voiceModels/pocket-tts-cache \
    VOICECHAT_TTS_BACKEND=pocket \
    VOICECHAT_API_HOST=0.0.0.0 \
    VOICECHAT_API_PORT=3000

WORKDIR /app

# PortAudio is needed for the optional microphone CLI; web mode uses browser audio.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libportaudio2 portaudio19-dev pulseaudio-utils curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY . .
RUN mkdir -p /app/voiceModels/pocket-tts-cache && chmod -R a+rwx /app/voiceModels
EXPOSE 3000
CMD ["uv", "run", "--frozen", "--no-sync", "python", "-m", "services.api.app"]
