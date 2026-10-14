# Stage 1: Build the React webapp
FROM node:20-slim AS webapp-build
WORKDIR /app/tools/webapp
COPY tools/webapp/package*.json ./
RUN npm ci
COPY tools/webapp/ ./
RUN npm run build

# Stage 2: Application runtime
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/voiceModels/pocket-tts-cache \
    VOICECHAT_TTS_BACKEND=pocket \
    VOICECHAT_API_HOST=0.0.0.0 \
    VOICECHAT_API_PORT=3000 \
    VOICECHAT_WEBAPP_PORT=5173 \
    VOICECHAT_WEBRTC_UDP_PORT_MIN=50000 \
    VOICECHAT_WEBRTC_UDP_PORT_MAX=50100 \
    VOICECHAT_WEBRTC_ADVERTISED_IP= \
    OLLAMA_BASE_URL=http://host.docker.internal:11434 \
    OLLAMA_HOST=http://host.docker.internal:11434 \
    OLLAMA_MODEL=qwen3.5:4b

WORKDIR /app

# PortAudio is needed for the optional microphone CLI; web mode uses browser audio.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libportaudio2 portaudio19-dev pulseaudio-utils curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js runtime from node image for running the webapp server
COPY --from=node:20-slim /usr/local/bin /usr/local/bin
COPY --from=node:20-slim /usr/local/lib/node_modules /usr/local/lib/node_modules

RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .

# Copy production webapp build and node dependencies from builder stage
COPY --from=webapp-build /app/tools/webapp/dist /app/tools/webapp/dist
COPY --from=webapp-build /app/tools/webapp/node_modules /app/tools/webapp/node_modules

RUN mkdir -p /app/voiceModels/pocket-tts-cache && chmod -R a+rwx /app/voiceModels
RUN chmod +x /app/scripts/start-all.sh

EXPOSE 3000 5173 50000-50100/udp
CMD ["/app/scripts/start-all.sh"]
