import { useState, useRef, useEffect, useCallback } from 'react'

// ─── Config ──────────────────────────────────────────────────────────────────

const STT_SAMPLE_RATE   = 16000
const STT_CHUNK_SAMPLES = 16000
const STT_SILENCE_SECS  = 1.2
const STT_POLL_MS       = 400
const STT_TIMEOUT_MS    = 15_000
const MAX_RECORD_SECS   = 60

const DETECTED_ORIGIN = window.location.protocol !== 'file:'
  ? window.location.origin
  : 'http://localhost:3000'

const sleep = ms => new Promise(r => setTimeout(r, ms))

// ─── App ─────────────────────────────────────────────────────────────────────

export default function App() {
  const [apiUrl, setApiUrl]           = useState(DETECTED_ORIGIN)
  const [healthState, setHealthState] = useState('')      // '' | 'ok' | 'err'
  const [messages, setMessages]       = useState([])      // {id, role, content, typing, liveTranscript}
  const [statusLabel, setStatusLabel] = useState('Idle')
  const [statusState, setStatusState] = useState('idle')  // idle | recording | busy | live
  const [textValue, setTextValue]     = useState('')
  const [isLive, setIsLive]           = useState(false)

  // Mutable refs — don't need to trigger re-renders
  const busyRef            = useRef(false)
  const historyRef         = useRef([])
  const mediaRecorderRef   = useRef(null)
  const audioChunksRef     = useRef([])
  const recordTimeoutRef   = useRef(null)
  const currentAudioRef    = useRef(null)
  const rtcPeerRef         = useRef(null)
  const pollTimerRef       = useRef(null)
  const knownHistoryLenRef = useRef(0)
  const remoteAudioRef     = useRef(null)
  const messagesEndRef     = useRef(null)

  const baseUrl = useCallback(() => apiUrl.replace(/\/$/, ''), [apiUrl])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages])

  useEffect(() => {
    checkHealth()
    const id = setInterval(checkHealth, 10_000)
    return () => clearInterval(id)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiUrl])

  // ── Helpers ────────────────────────────────────────────────────────────────

  function checkHealth() {
    fetch(baseUrl() + '/health')
      .then(res => setHealthState(res.ok ? 'ok' : 'err'))
      .catch(() => setHealthState('err'))
  }

  function addMsg(role, content, opts = {}) {
    const id = opts.id ?? ('msg-' + Date.now() + '-' + Math.random().toString(36).slice(2))
    setMessages(prev => [
      ...prev,
      { id, role, content, typing: !!opts.typing, liveTranscript: !!opts.liveTranscript },
    ])
    return id
  }

  function updateMsg(id, content, { done = false } = {}) {
    setMessages(prev =>
      prev.map(m => m.id === id ? { ...m, content, typing: done ? false : m.typing } : m)
    )
  }

  function removeLiveTranscript() {
    setMessages(prev => prev.filter(m => !m.liveTranscript))
  }

  function showError(msg) {
    setMessages(prev => [...prev, { id: 'err-' + Date.now(), role: 'error', content: msg }])
  }

  // ── API ────────────────────────────────────────────────────────────────────

  async function apiPost(path, body) {
    const res = await fetch(baseUrl() + path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    if (!res.ok) {
      const detail = await res.json().then(d => d.detail).catch(() => res.statusText)
      throw new Error(detail)
    }
    return res
  }

  // ── Audio recording ────────────────────────────────────────────────────────

  async function startRecording() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false })
    audioChunksRef.current = []
    const mr = new MediaRecorder(stream)
    mediaRecorderRef.current = mr
    mr.ondataavailable = e => { if (e.data.size > 0) audioChunksRef.current.push(e.data) }
    mr.start(100)
    recordTimeoutRef.current = setTimeout(() => stopAndProcess(), MAX_RECORD_SECS * 1000)
  }

  async function stopRecording() {
    clearTimeout(recordTimeoutRef.current)
    const mr = mediaRecorderRef.current
    if (!mr || mr.state === 'inactive') return
    await new Promise(resolve => {
      mr.onstop = resolve
      mr.stop()
      mr.stream.getTracks().forEach(t => t.stop())
    })
  }

  async function decodeToMono16k(blob) {
    const arrayBuffer = await blob.arrayBuffer()
    const ctx = new AudioContext()
    const decoded = await ctx.decodeAudioData(arrayBuffer)
    await ctx.close()

    const monoBuffer = decoded.numberOfChannels > 1 ? mixToMono(decoded) : decoded

    const outFrames = Math.ceil(monoBuffer.duration * STT_SAMPLE_RATE)
    const offline = new OfflineAudioContext(1, outFrames, STT_SAMPLE_RATE)
    const src = offline.createBufferSource()
    src.buffer = monoBuffer
    src.connect(offline.destination)
    src.start()
    const rendered = await offline.startRendering()
    return rendered.getChannelData(0)
  }

  function mixToMono(buffer) {
    const out = new Float32Array(buffer.length)
    for (let c = 0; c < buffer.numberOfChannels; c++) {
      const ch = buffer.getChannelData(c)
      for (let i = 0; i < ch.length; i++) out[i] += ch[i]
    }
    for (let i = 0; i < out.length; i++) out[i] /= buffer.numberOfChannels
    const mono = new AudioBuffer({ numberOfChannels: 1, length: buffer.length, sampleRate: buffer.sampleRate })
    mono.copyToChannel(out, 0)
    return mono
  }

  // ── STT pipeline ───────────────────────────────────────────────────────────

  async function pushAudioToSTT(samples) {
    await fetch(baseUrl() + '/stt/transcripts', { method: 'DELETE' })
    for (let i = 0; i < samples.length; i += STT_CHUNK_SAMPLES) {
      const chunk = Array.from(samples.subarray(i, i + STT_CHUNK_SAMPLES))
      await apiPost('/stt/push', { sample_rate: STT_SAMPLE_RATE, samples: chunk })
    }
    const silence = new Array(Math.floor(STT_SAMPLE_RATE * STT_SILENCE_SECS)).fill(0)
    await apiPost('/stt/push', { sample_rate: STT_SAMPLE_RATE, samples: silence })
  }

  async function pollForTranscript() {
    const deadline = Date.now() + STT_TIMEOUT_MS
    while (Date.now() < deadline) {
      await sleep(STT_POLL_MS)
      const res = await fetch(baseUrl() + '/stt/transcripts')
      if (!res.ok) continue
      const events = await res.json()
      const completed = [...events].reverse().find(e => e.kind === 'completed')
      if (completed && completed.text.trim()) return completed.text.trim()
    }
    throw new Error('Transcription timed out. Please try again.')
  }

  // ── LLM ────────────────────────────────────────────────────────────────────

  async function generateResponse(userText) {
    historyRef.current.push({ role: 'user', content: userText })
    const res = await apiPost('/llm/generate', {
      messages: historyRef.current,
      timeout_seconds: 90,
    })
    const { text } = await res.json()
    historyRef.current.push({ role: 'assistant', content: text })
    return text
  }

  // ── TTS ────────────────────────────────────────────────────────────────────

  async function speakText(text) {
    const res = await apiPost('/tts/synthesize', { text })
    const blob = await res.blob()
    const url = URL.createObjectURL(blob)
    return new Promise((resolve, reject) => {
      const audio = new Audio(url)
      currentAudioRef.current = audio
      audio.onended = () => { URL.revokeObjectURL(url); currentAudioRef.current = null; resolve() }
      audio.onerror = () => { URL.revokeObjectURL(url); currentAudioRef.current = null; reject(new Error('Audio playback failed.')) }
      audio.play().catch(reject)
    })
  }

  // ── Conversation turn ──────────────────────────────────────────────────────

  async function runTurn(userText) {
    busyRef.current = true
    addMsg('user', userText)
    const typingId = 'typing-' + Date.now()
    addMsg('assistant', '…', { typing: true, id: typingId })
    try {
      setStatusLabel('Thinking…'); setStatusState('busy')
      const responseText = await generateResponse(userText)
      updateMsg(typingId, responseText, { done: true })
      setStatusLabel('Speaking…'); setStatusState('busy')
      await speakText(responseText)
      setStatusLabel('Idle'); setStatusState('idle')
    } catch (err) {
      updateMsg(typingId, '[Error]', { done: true })
      showError(err.message)
      setStatusLabel('Idle'); setStatusState('idle')
    } finally {
      busyRef.current = false
    }
  }

  async function stopAndProcess() {
    const mr = mediaRecorderRef.current
    if (!mr || mr.state === 'inactive') return
    setStatusLabel('Transcribing…'); setStatusState('busy')
    try {
      await stopRecording()
      const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' })
      if (blob.size < 100) {
        setStatusLabel('Idle'); setStatusState('idle')
        busyRef.current = false
        return
      }
      const samples = await decodeToMono16k(blob)
      await pushAudioToSTT(samples)
      const transcript = await pollForTranscript()
      await runTurn(transcript)
    } catch (err) {
      showError(err.message)
      setStatusLabel('Idle'); setStatusState('idle')
      busyRef.current = false
    }
  }

  // ── WebRTC ─────────────────────────────────────────────────────────────────

  async function connectRTC() {
    let stream
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false })
    } catch (err) {
      showError('Microphone access denied: ' + err.message)
      return
    }

    const pc = new RTCPeerConnection({ iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] })
    rtcPeerRef.current = pc

    for (const track of stream.getAudioTracks()) pc.addTrack(track, stream)
    pc.ontrack = e => {
      if (e.streams[0] && remoteAudioRef.current) remoteAudioRef.current.srcObject = e.streams[0]
    }

    const offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    // Wait for ICE gathering (max 3 s)
    await new Promise(resolve => {
      const t = setTimeout(resolve, 3000)
      if (pc.iceGatheringState === 'complete') { clearTimeout(t); resolve(); return }
      pc.onicegatheringstatechange = () => {
        if (pc.iceGatheringState === 'complete') { clearTimeout(t); resolve() }
      }
    })

    let answer
    try {
      const res = await fetch(baseUrl() + '/webrtc/offer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type }),
      })
      if (!res.ok) throw new Error(await res.json().then(d => d.detail).catch(() => res.statusText))
      answer = await res.json()
    } catch (err) {
      showError('WebRTC handshake failed: ' + err.message)
      await disconnectRTC()
      return
    }

    await pc.setRemoteDescription(answer)
    pc.onconnectionstatechange = () => {
      if (pc.connectionState === 'connected') {
        setIsLive(true)
        setStatusLabel('Live – just speak'); setStatusState('live')
        knownHistoryLenRef.current = 0
        pollTimerRef.current = setInterval(pollLiveState, 600)
      } else if (['failed', 'disconnected', 'closed'].includes(pc.connectionState)) {
        endLiveMode()
      }
    }
  }

  async function disconnectRTC() {
    clearInterval(pollTimerRef.current)
    pollTimerRef.current = null
    removeLiveTranscript()
    const pc = rtcPeerRef.current
    if (pc) {
      pc.onconnectionstatechange = null
      pc.close()
      rtcPeerRef.current = null
    }
    if (remoteAudioRef.current) remoteAudioRef.current.srcObject = null
    endLiveMode()
  }

  function endLiveMode() {
    clearInterval(pollTimerRef.current)
    pollTimerRef.current = null
    removeLiveTranscript()
    rtcPeerRef.current = null
    setIsLive(false)
    setStatusLabel('Idle'); setStatusState('idle')
  }

  async function pollLiveState() {
    try {
      const tRes = await fetch(baseUrl() + '/stt/transcripts')
      if (tRes.ok) {
        const events = await tRes.json()
        const live = [...events].reverse().find(e => e.kind !== 'completed')
        if (live && live.text.trim()) {
          setMessages(prev => {
            const existing = prev.find(m => m.liveTranscript)
            if (existing) return prev.map(m => m.liveTranscript ? { ...m, content: live.text } : m)
            return [...prev, { id: 'live-transcript', role: 'user', content: live.text, liveTranscript: true }]
          })
        } else {
          removeLiveTranscript()
        }
      }

      const hRes = await fetch(baseUrl() + '/conversation/history')
      if (!hRes.ok) return
      const turns = await hRes.json()
      if (turns.length > knownHistoryLenRef.current) {
        setMessages(prev => {
          const filtered = prev.filter(m => !m.liveTranscript)
          const newTurns = turns.slice(knownHistoryLenRef.current).map((t, i) => ({
            id: 'hist-' + Date.now() + '-' + i,
            role: t.role,
            content: t.content,
          }))
          return [...filtered, ...newTurns]
        })
        knownHistoryLenRef.current = turns.length
      }
    } catch { /* Network hiccup */ }
  }

  // ── Event handlers ─────────────────────────────────────────────────────────

  async function handleTalkDown(e) {
    e.preventDefault()
    if (busyRef.current || isLive) return
    if (currentAudioRef.current) { currentAudioRef.current.pause(); currentAudioRef.current = null }
    busyRef.current = true
    setStatusLabel('Recording…'); setStatusState('recording')
    try {
      await startRecording()
    } catch (err) {
      showError('Microphone access denied: ' + err.message)
      setStatusLabel('Idle'); setStatusState('idle')
      busyRef.current = false
    }
  }

  function handleTalkUp() {
    const mr = mediaRecorderRef.current
    if (mr && mr.state !== 'inactive') stopAndProcess()
  }

  async function handleSend() {
    const text = textValue.trim()
    if (!text || busyRef.current) return
    setTextValue('')
    await runTurn(text)
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  function handleClear() {
    historyRef.current = []
    setMessages([])
    if (rtcPeerRef.current) {
      fetch(baseUrl() + '/conversation/history', { method: 'DELETE' }).catch(() => {})
      fetch(baseUrl() + '/stt/transcripts', { method: 'DELETE' }).catch(() => {})
      knownHistoryLenRef.current = 0
    }
  }

  // ── Derived state ──────────────────────────────────────────────────────────

  const isBusy       = statusState === 'busy' || statusState === 'recording'
  const talkDisabled = isLive || (isBusy && statusState !== 'recording')

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <>
      <header>
        <h1>VoiceChat</h1>
        <div id="statusDot" className={healthState} title="API connection" />
        <input
          id="urlInput"
          type="text"
          placeholder="API base URL"
          spellCheck="false"
          value={apiUrl}
          onChange={e => setApiUrl(e.target.value)}
          onBlur={checkHealth}
        />
        <button
          id="connectBtn"
          className={isLive ? 'live' : ''}
          onClick={() => isLive ? disconnectRTC() : connectRTC()}
        >
          {isLive ? '⏹ Disconnect' : '📞 Live'}
        </button>
        <button id="clearBtn" onClick={handleClear}>Clear</button>
      </header>

      <div id="messages" role="log" aria-live="polite">
        {messages.length === 0 && (
          <div className="empty-hint">
            Click 📞 Live for real-time voice, or hold 🎤 for push-to-talk.
          </div>
        )}
        {messages.map(m => {
          if (m.role === 'error') return (
            <div key={m.id} style={{ color: 'var(--red)', fontSize: '0.8rem', textAlign: 'center', padding: '4px' }}>
              ⚠ {m.content}
            </div>
          )
          return (
            <div
              key={m.id}
              className={[
                'msg',
                m.role,
                m.typing ? 'typing' : '',
                m.liveTranscript ? 'live-transcript' : '',
              ].filter(Boolean).join(' ')}
            >
              <div className="role">{m.role === 'user' ? 'You' : 'Assistant'}</div>
              <div className="content">{m.content}</div>
            </div>
          )
        })}
        <div ref={messagesEndRef} />
      </div>

      <audio ref={remoteAudioRef} autoPlay playsInline style={{ display: 'none' }} />

      <footer>
        <div id="textRow">
          <textarea
            id="textInput"
            rows={1}
            placeholder="Or type a message…"
            value={textValue}
            disabled={isBusy}
            onChange={e => {
              setTextValue(e.target.value)
              e.target.style.height = 'auto'
              e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px'
            }}
            onKeyDown={handleKeyDown}
          />
          <button id="sendBtn" disabled={isBusy} onClick={handleSend}>Send</button>
        </div>
        <button
          id="talkBtn"
          className={statusState === 'recording' ? 'recording' : ''}
          disabled={talkDisabled}
          title="Hold to talk"
          onPointerDown={handleTalkDown}
          onPointerUp={handleTalkUp}
          onPointerLeave={handleTalkUp}
        >
          🎤
        </button>
        <div id="statusLabel">{statusLabel}</div>
      </footer>
    </>
  )
}
