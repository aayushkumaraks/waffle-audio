# Waffle Webapp

The Waffle Webapp is the production React single-page application for the Waffle Audio assistant.

## Requirements

- Node.js 20+
- npm 10+
- A running Waffle Audio API
- A browser with microphone and WebRTC support for voice modes

## Development

From the repository root:

```bash
bash ./tools/webapp/start.sh
```

Or:

```bash
cd tools/webapp
npm ci
npm run dev -- --host 0.0.0.0
```

Vite normally starts at `http://localhost:5173`. The development app automatically targets `http://localhost:8000` when it detects Vite's development ports.

If the API is elsewhere, change the **API base URL** in the header. The value is persisted in the browser.

You can also set `VITE_API_URL` before starting Vite:

```bash
VITE_API_URL=http://localhost:8000 npm run dev -- --host 0.0.0.0
```

## Production

Build the SPA:

```bash
cd tools/webapp
npm ci
npm run build
```

The generated bundle is written to `tools/webapp/dist/`.

The API allows the local Vite/preview origins by default. For a different frontend origin, set `VOICECHAT_CORS_ALLOW_ORIGINS` to a comma-separated allowlist, for example:

```bash
export VOICECHAT_CORS_ALLOW_ORIGINS="https://app.example.com"
```

The Vite configuration uses relative asset paths so the generated SPA can be mounted by the FastAPI service at:

```
/ui
```

The API already mounts `tools/webapp` at `/ui`. Start the API using the repository's normal API entrypoint, then open `http://localhost:8000/ui` (or the configured API host/port).

For a local preview of the production bundle:

```bash
npm run preview -- --host 0.0.0.0
```

## Functional modes

### Text chat

1. Confirm the API indicator is green.
2. Type a message.
3. Press Enter or the send button.
4. The browser calls `/llm/generate`, displays the response, then requests WAV audio from `/tts/synthesize`.

### Push-to-talk

1. Hold the microphone button.
2. Release to stop recording.
3. The browser converts the recording to mono 16 kHz PCM.
4. PCM is sent to `/stt/push`.
5. The browser waits for a completed transcript from `/stt/transcripts`.
6. The transcript is processed through the same LLM + TTS turn flow.

Microphone access requires a secure context in browsers: HTTPS in production, or localhost during development.

### Live voice

1. Click **Live voice**.
2. Grant microphone permission.
3. The browser creates a WebRTC offer and sends it to `/webrtc/offer`.
4. The API streams synthesized assistant audio back over WebRTC.
5. The UI polls transcript and conversation-history endpoints for visible conversation state.
6. Click **End live** to close the peer connection and microphone tracks.

The live path intentionally relies on the server's existing VAD/STT/LLM/TTS pipeline rather than duplicating that pipeline in the browser.

## Architecture

```text
tools/webapp/
├── src/
│   ├── components/
│   │   ├── AppHeader.jsx
│   │   ├── Composer.jsx
│   │   └── MessageList.jsx
│   ├── hooks/
│   │   └── useVoiceChat.js
│   ├── services/
│   │   └── api.js
│   ├── App.jsx
│   ├── App.css
│   └── main.jsx
├── index.html
├── package.json
├── package-lock.json
├── vite.config.js
└── start.sh
```

The UI layer is separated from transport and voice orchestration. The API client centralizes HTTP error handling, while `useVoiceChat` owns browser media, WebRTC lifecycle, conversation state, and mode transitions.

## Operational notes

- The browser never needs direct access to Ollama, STT, or TTS providers.
- API errors are surfaced as conversation/system messages instead of being silently swallowed in user-initiated turns.
- Live-session polling failures are tolerated because the media connection is independent of the UI polling loop.
- Microphone tracks and WebRTC peers are explicitly stopped on disconnect and component unmount.
- The API URL is stored locally in the browser and can be changed without rebuilding the app.
