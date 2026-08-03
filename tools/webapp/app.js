/* VoiceChat browser client
 *
 * Flow: mic recording → /stt/push (PCM chunks) → poll /stt/transcripts
 *       → /llm/generate (with history) → /tts/synthesize (WAV) → <Audio>
 */

// ─── Config ──────────────────────────────────────────────────────────────────

const STT_SAMPLE_RATE    = 16000;      // Moonshine expects 16 kHz
const STT_CHUNK_SAMPLES  = 16000;      // push in 1-second chunks
const STT_SILENCE_SECS   = 1.2;        // trailing silence to flush the stream
const STT_POLL_MS        = 400;
const STT_TIMEOUT_MS     = 15_000;
const MAX_RECORD_SECS    = 60;

// ─── State ───────────────────────────────────────────────────────────────────

let history          = [];   // [{role, content}]
let busy             = false;
let mediaRecorder    = null;
let audioChunks      = [];
let recordTimeout    = null;
let currentAudio     = null; // HTMLAudioElement being played

// ─── DOM refs ────────────────────────────────────────────────────────────────

// Auto-detect API origin when served from the same host, fall back to localhost
const _detectedOrigin = window.location.protocol !== "file:"
  ? window.location.origin
  : "http://localhost:3000";

const urlInput    = document.getElementById("urlInput");
const statusDot   = document.getElementById("statusDot");
const statusLabel = document.getElementById("statusLabel");
const messagesEl  = document.getElementById("messages");
const talkBtn     = document.getElementById("talkBtn");
const textInput   = document.getElementById("textInput");
const sendBtn     = document.getElementById("sendBtn");
const clearBtn    = document.getElementById("clearBtn");

// ─── Helpers ─────────────────────────────────────────────────────────────────

const baseUrl = () => urlInput.value.replace(/\/$/, "");
const sleep   = ms => new Promise(r => setTimeout(r, ms));

function setStatus(label, state /* idle|recording|busy */) {
  statusLabel.textContent = label;
  const isBusy = state === "busy" || state === "recording";
  talkBtn.disabled = isBusy && state !== "recording";
  sendBtn.disabled = isBusy;
  textInput.disabled = isBusy;
  if (state === "recording") {
    talkBtn.classList.add("recording");
  } else {
    talkBtn.classList.remove("recording");
  }
}

function addMessage(role, text, opts = {}) {
  const el = document.createElement("div");
  el.className = `msg ${role}` + (opts.typing ? " typing" : "");
  el.innerHTML = `<div class="role">${role === "user" ? "You" : "Assistant"}</div>
                  <div class="content">${escHtml(text)}</div>`;
  if (opts.id) el.id = opts.id;
  messagesEl.appendChild(el);
  el.scrollIntoView({ behavior: "smooth", block: "end" });
  return el;
}

function updateMessage(id, text, { done = false } = {}) {
  const el = document.getElementById(id);
  if (!el) return;
  el.querySelector(".content").textContent = text;
  if (done) el.classList.remove("typing");
  el.scrollIntoView({ behavior: "smooth", block: "end" });
}

function escHtml(str) {
  return str.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

function showError(msg) {
  const el = document.createElement("div");
  el.style.cssText = "color:var(--red);font-size:0.8rem;text-align:center;padding:4px";
  el.textContent = "⚠ " + msg;
  messagesEl.appendChild(el);
  el.scrollIntoView({ behavior: "smooth", block: "end" });
}

// ─── API calls ───────────────────────────────────────────────────────────────

async function apiPost(path, body) {
  const res = await fetch(baseUrl() + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.json().then(d => d.detail).catch(() => res.statusText);
    throw new Error(detail);
  }
  return res;
}

async function checkHealth() {
  try {
    const res = await fetch(baseUrl() + "/health");
    if (res.ok) {
      statusDot.className = "ok";
    } else {
      statusDot.className = "err";
    }
  } catch {
    statusDot.className = "err";
  }
}

// ─── Audio recording ─────────────────────────────────────────────────────────

async function startRecording() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  audioChunks = [];
  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };
  mediaRecorder.start(100); // emit chunks every 100 ms

  // safety: auto-stop after MAX_RECORD_SECS
  recordTimeout = setTimeout(() => stopAndProcess(), MAX_RECORD_SECS * 1000);
}

async function stopRecording() {
  clearTimeout(recordTimeout);
  if (!mediaRecorder || mediaRecorder.state === "inactive") return;

  await new Promise(resolve => {
    mediaRecorder.onstop = resolve;
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach(t => t.stop());
  });
}

async function decodeToMono16k(blob) {
  const arrayBuffer = await blob.arrayBuffer();
  const ctx = new AudioContext();
  const decoded = await ctx.decodeAudioData(arrayBuffer);
  await ctx.close();

  // Mix down to mono if needed
  const monoBuffer = decoded.numberOfChannels > 1
    ? mixToMono(decoded)
    : decoded;

  // Resample to 16 kHz with OfflineAudioContext
  const outFrames = Math.ceil(monoBuffer.duration * STT_SAMPLE_RATE);
  const offline = new OfflineAudioContext(1, outFrames, STT_SAMPLE_RATE);
  const src = offline.createBufferSource();
  src.buffer = monoBuffer;
  src.connect(offline.destination);
  src.start();

  const rendered = await offline.startRendering();
  return rendered.getChannelData(0); // Float32Array
}

function mixToMono(buffer) {
  const ctx = new OfflineAudioContext(1, buffer.length, buffer.sampleRate);
  const src = ctx.createBufferSource();
  src.buffer = buffer;
  src.connect(ctx.destination);
  src.start();
  // Return synchronously — we use the original buffer channels averaged
  const out = new Float32Array(buffer.length);
  for (let c = 0; c < buffer.numberOfChannels; c++) {
    const ch = buffer.getChannelData(c);
    for (let i = 0; i < ch.length; i++) out[i] += ch[i];
  }
  for (let i = 0; i < out.length; i++) out[i] /= buffer.numberOfChannels;

  const mono = new AudioBuffer({ numberOfChannels: 1, length: buffer.length, sampleRate: buffer.sampleRate });
  mono.copyToChannel(out, 0);
  return mono;
}

// ─── STT pipeline ────────────────────────────────────────────────────────────

async function pushAudioToSTT(samples) {
  // Clear any previous transcript events
  await fetch(baseUrl() + "/stt/transcripts", { method: "DELETE" });

  // Push speech samples in chunks
  for (let i = 0; i < samples.length; i += STT_CHUNK_SAMPLES) {
    const chunk = Array.from(samples.subarray(i, i + STT_CHUNK_SAMPLES));
    await apiPost("/stt/push", { sample_rate: STT_SAMPLE_RATE, samples: chunk });
  }

  // Push trailing silence so Moonshine flushes the utterance
  const silence = new Array(Math.floor(STT_SAMPLE_RATE * STT_SILENCE_SECS)).fill(0);
  await apiPost("/stt/push", { sample_rate: STT_SAMPLE_RATE, samples: silence });
}

async function pollForTranscript() {
  const deadline = Date.now() + STT_TIMEOUT_MS;

  while (Date.now() < deadline) {
    await sleep(STT_POLL_MS);

    const res = await fetch(baseUrl() + "/stt/transcripts");
    if (!res.ok) continue;

    const events = await res.json();
    // Find the last "completed" event
    const completed = [...events].reverse().find(e => e.kind === "completed");
    if (completed && completed.text.trim()) {
      return completed.text.trim();
    }
  }

  throw new Error("Transcription timed out. Please try again.");
}

// ─── LLM call ────────────────────────────────────────────────────────────────

async function generateResponse(userText) {
  history.push({ role: "user", content: userText });

  const res = await apiPost("/llm/generate", {
    messages: history,
    timeout_seconds: 90,
  });

  const { text } = await res.json();
  history.push({ role: "assistant", content: text });
  return text;
}

// ─── TTS playback ────────────────────────────────────────────────────────────

async function speakText(text) {
  const res = await apiPost("/tts/synthesize", { text });
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);

  return new Promise((resolve, reject) => {
    const audio = new Audio(url);
    currentAudio = audio;
    audio.onended = () => {
      URL.revokeObjectURL(url);
      currentAudio = null;
      resolve();
    };
    audio.onerror = () => {
      URL.revokeObjectURL(url);
      currentAudio = null;
      reject(new Error("Audio playback failed."));
    };
    audio.play().catch(reject);
  });
}

// ─── Main conversation turn ──────────────────────────────────────────────────

async function runTurn(userText) {
  busy = true;

  addMessage("user", userText);

  const typingId = "typing-" + Date.now();
  addMessage("assistant", "…", { typing: true, id: typingId });

  try {
    setStatus("Thinking…", "busy");
    const responseText = await generateResponse(userText);

    updateMessage(typingId, responseText, { done: true });

    setStatus("Speaking…", "busy");
    await speakText(responseText);

    setStatus("Idle", "idle");
  } catch (err) {
    updateMessage(typingId, "[Error]", { done: true });
    showError(err.message);
    setStatus("Idle", "idle");
  } finally {
    busy = false;
  }
}

async function stopAndProcess() {
  if (!mediaRecorder || mediaRecorder.state === "inactive") return;

  setStatus("Transcribing…", "busy");
  talkBtn.disabled = true;

  await stopRecording();

  try {
    const blob = new Blob(audioChunks, { type: "audio/webm" });
    if (blob.size < 100) {
      setStatus("Idle", "idle");
      busy = false;
      return;
    }

    const samples = await decodeToMono16k(blob);
    await pushAudioToSTT(samples);

    const transcript = await pollForTranscript();
    await runTurn(transcript);
  } catch (err) {
    showError(err.message);
    setStatus("Idle", "idle");
    busy = false;
  }
}

// ─── Button handlers ─────────────────────────────────────────────────────────

talkBtn.addEventListener("pointerdown", async e => {
  e.preventDefault();
  if (busy) return;

  // Stop any playing audio so the user can speak again
  if (currentAudio) {
    currentAudio.pause();
    currentAudio = null;
  }

  busy = true;
  setStatus("Recording…", "recording");

  try {
    await startRecording();
  } catch (err) {
    showError("Microphone access denied: " + err.message);
    setStatus("Idle", "idle");
    busy = false;
  }
});

talkBtn.addEventListener("pointerup", () => stopAndProcess());
talkBtn.addEventListener("pointerleave", () => {
  if (mediaRecorder && mediaRecorder.state !== "inactive") stopAndProcess();
});

sendBtn.addEventListener("click", async () => {
  const text = textInput.value.trim();
  if (!text || busy) return;
  textInput.value = "";
  textInput.style.height = "";
  await runTurn(text);
});

textInput.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendBtn.click();
  }
});

// Auto-grow textarea
textInput.addEventListener("input", () => {
  textInput.style.height = "auto";
  textInput.style.height = Math.min(textInput.scrollHeight, 120) + "px";
});

clearBtn.addEventListener("click", () => {
  history = [];
  messagesEl.innerHTML = "";
});

urlInput.addEventListener("change", () => checkHealth());

// ─── Init ─────────────────────────────────────────────────────────────────────

urlInput.value = _detectedOrigin;
checkHealth();
setInterval(checkHealth, 10_000);
setStatus("Idle", "idle");
