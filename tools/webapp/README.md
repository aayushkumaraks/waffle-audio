# Waffle Webapp

The Waffle Webapp is the React single-page application for the Waffle Audio voice assistant.

## Requirements

- Node.js 20+
- npm 10+
- The Waffle Audio API running

## Start the webapp

From the repository root:

```bash
./tools/webapp/start.sh
```

The script enters the webapp directory, installs dependencies if `node_modules` is missing, and starts Vite on all interfaces.

Alternatively:

```bash
cd tools/webapp
npm install
npm run dev -- --host 0.0.0.0
```

Vite will print the local URL, normally:

```
http://localhost:5173
```

Open that URL in a browser.

## API endpoint

By default, the SPA uses the browser's current origin as the API base URL. This is suitable when the API serves the webapp itself.

The API endpoint can also be changed from the endpoint field in the webapp header.

For a separately running API, enter its base URL, for example:

```
http://localhost:8000
```

## Production build

```bash
cd tools/webapp
npm install
npm run build
```

The production bundle is generated in:

```
tools/webapp/dist/
```

Preview the production build with:

```bash
npm run preview
```

## Architecture

The SPA is intentionally split by responsibility:

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
├── vite.config.js
└── start.sh
```

The browser-facing interaction layer lives in the React app. API calls remain behind the small API client, while voice/conversation orchestration is isolated in the voice-chat hook.

## Modes

- **Text:** send a typed message and receive the assistant response with TTS playback.
- **Push-to-talk:** hold the microphone button, release to transcribe and process the turn.
- **Live voice:** establish a WebRTC session for continuous voice interaction.
