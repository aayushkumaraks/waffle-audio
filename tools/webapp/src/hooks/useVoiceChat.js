import { useCallback, useEffect, useRef, useState } from 'react'
import { createApiClient } from '../services/api'

const SAMPLE_RATE = 16000
const CHUNK_SAMPLES = 16000
const SILENCE_SECONDS = 1.2
const STT_TIMEOUT = 15000
const origin = window.location.protocol !== 'file:' ? window.location.origin : 'http://localhost:3000'

export function useVoiceChat() {
  const [apiUrl, setApiUrl] = useState(origin)
  const [healthState, setHealthState] = useState('')
  const [messages, setMessages] = useState([])
  const [statusLabel, setStatusLabel] = useState('Ready')
  const [statusState, setStatusState] = useState('idle')
  const [text, setText] = useState('')
  const [isLive, setIsLive] = useState(false)
  const busyRef = useRef(false)
  const historyRef = useRef([])
  const recorderRef = useRef(null)
  const chunksRef = useRef([])
  const audioRef = useRef(null)
  const peerRef = useRef(null)
  const remoteAudioRef = useRef(null)
  const pollRef = useRef(null)
  const knownHistoryRef = useRef(0)
  const endRef = useRef(null)
  const api = createApiClient(() => apiUrl)

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }) }, [messages])
  const checkHealth = useCallback(async () => {
    try { const r = await api.get('/health'); setHealthState(r.ok ? 'ok' : 'err') } catch { setHealthState('err') }
  }, [apiUrl])
  useEffect(() => { checkHealth(); const t = setInterval(checkHealth, 10000); return () => clearInterval(t) }, [checkHealth])
  const addMessage = (role, content, extra = {}) => setMessages(current => [...current, { id: extra.id || crypto.randomUUID(), role, content, ...extra }])

  const generateResponse = async userText => {
    historyRef.current.push({ role: 'user', content: userText })
    const r = await api.postJson('/llm/generate', { messages: historyRef.current, timeout_seconds: 90 })
    const data = await r.json()
    historyRef.current.push({ role: 'assistant', content: data.text })
    return data.text
  }

  const speak = async responseText => {
    const r = await api.postJson('/tts/synthesize', { text: responseText })
    const url = URL.createObjectURL(await r.blob())
    await new Promise((resolve, reject) => {
      const audio = new Audio(url); audioRef.current = audio
      audio.onended = () => { URL.revokeObjectURL(url); audioRef.current = null; resolve() }
      audio.onerror = () => { URL.revokeObjectURL(url); audioRef.current = null; reject(new Error('Audio playback failed')) }
      audio.play().catch(reject)
    })
  }

  const runTurn = async userText => {
    if (!userText.trim() || busyRef.current) return
    busyRef.current = true
    addMessage('user', userText)
    const id = crypto.randomUUID()
    addMessage('assistant', 'Thinking…', { id, typing: true })
    try {
      setStatusLabel('Thinking…'); setStatusState('busy')
      const response = await generateResponse(userText)
      setMessages(current => current.map(m => m.id === id ? { ...m, content: response, typing: false } : m))
      setStatusLabel('Speaking…'); await speak(response)
      setStatusLabel('Ready'); setStatusState('idle')
    } catch (error) {
      setMessages(current => current.map(m => m.id === id ? { ...m, content: error.message, role: 'error', typing: false } : m))
      setStatusLabel('Ready'); setStatusState('idle')
    } finally { busyRef.current = false }
  }

  const decodeToMono16k = async blob => {
    const context = new AudioContext()
    const decoded = await context.decodeAudioData(await blob.arrayBuffer())
    await context.close()
    const offline = new OfflineAudioContext(1, Math.ceil(decoded.duration * SAMPLE_RATE), SAMPLE_RATE)
    const source = offline.createBufferSource(); source.buffer = decoded; source.connect(offline.destination); source.start()
    return (await offline.startRendering()).getChannelData(0)
  }

  const finishRecording = async () => {
    const recorder = recorderRef.current
    if (!recorder || recorder.state === 'inactive') return
    setStatusLabel('Transcribing…'); setStatusState('busy')
    await new Promise(resolve => { recorder.onstop = resolve; recorder.stop(); recorder.stream.getTracks().forEach(t => t.stop()) })
    const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
    try {
      if (blob.size < 100) throw new Error('No audio captured')
      await api.delete('/stt/transcripts')
      const samples = await decodeToMono16k(blob)
      for (let i = 0; i < samples.length; i += CHUNK_SAMPLES) await api.postJson('/stt/push', { sample_rate: SAMPLE_RATE, samples: Array.from(samples.subarray(i, i + CHUNK_SAMPLES)) })
      await api.postJson('/stt/push', { sample_rate: SAMPLE_RATE, samples: new Array(Math.floor(SAMPLE_RATE * SILENCE_SECONDS)).fill(0) })
      const deadline = Date.now() + STT_TIMEOUT
      let transcript
      while (Date.now() < deadline) {
        await new Promise(r => setTimeout(r, 400))
        const events = await (await api.get('/stt/transcripts')).json()
        const completed = [...events].reverse().find(e => e.kind === 'completed')
        if (completed?.text?.trim()) { transcript = completed.text.trim(); break }
      }
      if (!transcript) throw new Error('Transcription timed out')
      await runTurn(transcript)
    } catch (error) {
      addMessage('error', error.message); busyRef.current = false; setStatusLabel('Ready'); setStatusState('idle')
    }
  }

  const startTalk = async () => {
    if (busyRef.current || isLive) return
    busyRef.current = true
    if (audioRef.current) { audioRef.current.pause(); audioRef.current = null }
    setStatusLabel('Listening…'); setStatusState('recording')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false })
      const recorder = new MediaRecorder(stream); recorderRef.current = recorder; chunksRef.current = []
      recorder.ondataavailable = e => { if (e.data.size) chunksRef.current.push(e.data) }
      recorder.start(100)
      setTimeout(() => finishRecording(), 60000)
    } catch (error) {
      addMessage('error', 'Microphone access denied: ' + error.message); busyRef.current = false; setStatusLabel('Ready'); setStatusState('idle')
    }
  }

  const stopTalk = () => { if (recorderRef.current?.state !== 'inactive') finishRecording() }

  const disconnectLive = () => {
    clearInterval(pollRef.current); pollRef.current = null
    peerRef.current?.close(); peerRef.current = null
    if (remoteAudioRef.current) remoteAudioRef.current.srcObject = null
    setMessages(current => current.filter(m => !m.liveTranscript))
    setIsLive(false); setStatusLabel('Ready'); setStatusState('idle')
  }

  const pollLive = async () => {
    try {
      const events = await (await api.get('/stt/transcripts')).json()
      const live = [...events].reverse().find(e => e.kind !== 'completed')
      setMessages(current => {
        const base = current.filter(m => !m.liveTranscript)
        return live?.text?.trim() ? [...base, { id: 'live-transcript', role: 'user', content: live.text, liveTranscript: true }] : base
      })
      const response = await api.get('/conversation/history')
      const turns = await response.json()
      if (turns.length > knownHistoryRef.current) {
        setMessages(current => [...current.filter(m => !m.liveTranscript), ...turns.slice(knownHistoryRef.current).map(t => ({ id: crypto.randomUUID(), role: t.role, content: t.content }))])
        knownHistoryRef.current = turns.length
      }
    } catch {}
  }

  const connectLive = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false })
      const peer = new RTCPeerConnection({ iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] })
      peerRef.current = peer
      stream.getAudioTracks().forEach(track => peer.addTrack(track, stream))
      peer.ontrack = e => { if (remoteAudioRef.current && e.streams[0]) remoteAudioRef.current.srcObject = e.streams[0] }
      const offer = await peer.createOffer(); await peer.setLocalDescription(offer)
      await new Promise(resolve => {
        const timer = setTimeout(resolve, 3000)
        if (peer.iceGatheringState === 'complete') { clearTimeout(timer); resolve() }
        else peer.onicegatheringstatechange = () => { if (peer.iceGatheringState === 'complete') { clearTimeout(timer); resolve() } }
      })
      const response = await fetch(api.root() + '/webrtc/offer', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ sdp: peer.localDescription.sdp, type: peer.localDescription.type }) })
      if (!response.ok) throw new Error(await response.json().then(d => d.detail).catch(() => response.statusText))
      await peer.setRemoteDescription(await response.json())
      peer.onconnectionstatechange = () => {
        if (peer.connectionState === 'connected') { setIsLive(true); setStatusLabel('Live conversation'); setStatusState('live'); knownHistoryRef.current = 0; pollRef.current = setInterval(pollLive, 600) }
        if (['failed', 'disconnected', 'closed'].includes(peer.connectionState)) disconnectLive()
      }
    } catch (error) { addMessage('error', 'Live voice failed: ' + error.message); peerRef.current?.close(); peerRef.current = null }
  }

  const clearConversation = () => {
    historyRef.current = []; setMessages([]); knownHistoryRef.current = 0
    api.delete('/conversation/history').catch(() => {}); api.delete('/stt/transcripts').catch(() => {})
  }

  useEffect(() => () => { clearInterval(pollRef.current); peerRef.current?.close(); audioRef.current?.pause() }, [])

  return { apiUrl, setApiUrl, healthState, messages, text, setText, statusLabel, statusState, isLive, isBusy: statusState === 'busy' || statusState === 'recording', endRef, remoteAudioRef, runTurn, startTalk, stopTalk, connectLive, disconnectLive, clearConversation }
}
