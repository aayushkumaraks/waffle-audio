import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { createApiClient, normalizeBaseUrl } from '../services/api'

const SAMPLE_RATE = 16000
const CHUNK_SAMPLES = 16000
const SILENCE_SECONDS = 1.2
const STT_TIMEOUT_MS = 15000
const MAX_RECORDING_MS = 60000
const LIVE_POLL_MS = 700
const API_STORAGE_KEY = 'waffle.apiUrl'

function defaultApiUrl() {
  const configured = import.meta.env.VITE_API_URL
  if (configured) return normalizeBaseUrl(configured)

  const { protocol, hostname, port } = window.location
  if (protocol === 'file:') return 'http://localhost:8000'
  if (port === '5173' || port === '4173') return `http://${hostname}:8000`
  return window.location.origin
}

function makeId() {
  return globalThis.crypto?.randomUUID?.() || `message-${Date.now()}-${Math.random().toString(36).slice(2)}`
}

function getRecorderMimeType() {
  if (typeof MediaRecorder === 'undefined') return ''
  return [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/mp4',
  ].find(type => MediaRecorder.isTypeSupported(type)) || ''
}

export function useVoiceChat() {
  const [apiUrl, setApiUrlState] = useState(() => {
    try {
      return localStorage.getItem(API_STORAGE_KEY) || defaultApiUrl()
    } catch {
      return defaultApiUrl()
    }
  })
  const [healthState, setHealthState] = useState('checking')
  const [messages, setMessages] = useState([])
  const [statusLabel, setStatusLabel] = useState('Ready')
  const [statusState, setStatusState] = useState('idle')
  const [text, setText] = useState('')
  const [isLive, setIsLive] = useState(false)

  const busyRef = useRef(false)
  const historyRef = useRef([])
  const recorderRef = useRef(null)
  const recordingTimerRef = useRef(null)
  const chunksRef = useRef([])
  const audioRef = useRef(null)
  const peerRef = useRef(null)
  const localStreamRef = useRef(null)
  const remoteAudioRef = useRef(null)
  const pollRef = useRef(null)
  const knownHistoryRef = useRef(0)
  const endRef = useRef(null)
  const mountedRef = useRef(true)
  const api = useMemo(() => createApiClient(() => apiUrl), [apiUrl])

  const setApiUrl = useCallback(value => {
    const next = normalizeBaseUrl(value)
    setApiUrlState(next)
    try {
      localStorage.setItem(API_STORAGE_KEY, next)
    } catch {
      // Local storage is optional.
    }
  }, [])

  const addMessage = useCallback((role, content, extra = {}) => {
    setMessages(current => [...current, { id: extra.id || makeId(), role, content, ...extra }])
  }, [])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages])

  const checkHealth = useCallback(async () => {
    try {
      const response = await api.get('/health')
      if (mountedRef.current) setHealthState(response.ok ? 'ok' : 'err')
    } catch {
      if (mountedRef.current) setHealthState('err')
    }
  }, [api])

  useEffect(() => {
    checkHealth()
    const timer = window.setInterval(checkHealth, 10000)
    return () => window.clearInterval(timer)
  }, [checkHealth])

  const stopPlayback = useCallback(() => {
    const audio = audioRef.current
    if (!audio) return
    audio.pause()
    audio.currentTime = 0
    audioRef.current = null
  }, [])

  const generateResponse = useCallback(async userText => {
    const nextHistory = [...historyRef.current, { role: 'user', content: userText }]
    const response = await api.postJson('/llm/generate', {
      messages: nextHistory,
      timeout_seconds: 90,
    })
    const data = await response.json()
    if (!data.text?.trim()) throw new Error('The assistant returned an empty response.')
    historyRef.current = [...nextHistory, { role: 'assistant', content: data.text }]
    return data.text
  }, [api])

  const speak = useCallback(async responseText => {
    const response = await api.postJson('/tts/synthesize', { text: responseText })
    const blob = await response.blob()
    if (!blob.size) throw new Error('The TTS service returned empty audio.')

    const url = URL.createObjectURL(blob)
    await new Promise((resolve, reject) => {
      const audio = new Audio(url)
      audioRef.current = audio

      const cleanup = () => {
        URL.revokeObjectURL(url)
        if (audioRef.current === audio) audioRef.current = null
      }
      audio.onended = () => { cleanup(); resolve() }
      audio.onerror = () => { cleanup(); reject(new Error('Audio playback failed.')) }

      audio.play().catch(error => {
        cleanup()
        reject(new Error(`Audio playback was blocked by the browser: ${error.message}`))
      })
    })
  }, [api])

  const runTurn = useCallback(async userText => {
    const cleanText = userText.trim()
    if (!cleanText || busyRef.current || isLive) return

    busyRef.current = true
    stopPlayback()
    addMessage('user', cleanText)

    const responseId = makeId()
    addMessage('assistant', 'Thinking…', { id: responseId, typing: true })
    setStatusLabel('Thinking…')
    setStatusState('busy')

    try {
      const response = await generateResponse(cleanText)
      setMessages(current => current.map(message =>
        message.id === responseId
          ? { ...message, content: response, typing: false }
          : message,
      ))

      setStatusLabel('Speaking…')
      await speak(response)
      if (mountedRef.current) {
        setStatusLabel('Ready')
        setStatusState('idle')
      }
    } catch (error) {
      setMessages(current => current.map(message =>
        message.id === responseId
          ? { ...message, role: 'error', content: error.message || 'The request failed.', typing: false }
          : message,
      ))
      setStatusLabel('Ready')
      setStatusState('idle')
    } finally {
      busyRef.current = false
    }
  }, [addMessage, generateResponse, isLive, speak, stopPlayback])

  const decodeToMono16k = useCallback(async blob => {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext
    if (!AudioContextClass || !window.OfflineAudioContext) {
      throw new Error('This browser does not support audio decoding required for push-to-talk.')
    }

    const context = new AudioContextClass()
    try {
      const decoded = await context.decodeAudioData(await blob.arrayBuffer())
      const length = Math.max(1, Math.ceil(decoded.duration * SAMPLE_RATE))
      const offline = new OfflineAudioContext(1, length, SAMPLE_RATE)
      const source = offline.createBufferSource()
      source.buffer = decoded
      source.connect(offline.destination)
      source.start()
      return (await offline.startRendering()).getChannelData(0)
    } finally {
      await context.close().catch(() => {})
    }
  }, [])

  const finishRecording = useCallback(async () => {
    const recorder = recorderRef.current
    if (!recorder || recorder.state === 'inactive') return

    window.clearTimeout(recordingTimerRef.current)
    recordingTimerRef.current = null
    setStatusLabel('Transcribing…')
    setStatusState('busy')

    const stopped = new Promise(resolve => {
      recorder.addEventListener('stop', resolve, { once: true })
    })
    recorder.stop()
    recorder.stream.getTracks().forEach(track => track.stop())
    await stopped
    recorderRef.current = null

    try {
      const blob = new Blob(chunksRef.current, { type: recorder.mimeType || 'audio/webm' })
      chunksRef.current = []
      if (blob.size < 100) throw new Error('No audio was captured.')

      await api.delete('/stt/transcripts')
      const samples = await decodeToMono16k(blob)

      for (let offset = 0; offset < samples.length; offset += CHUNK_SAMPLES) {
        const chunk = Array.from(samples.subarray(offset, offset + CHUNK_SAMPLES))
        if (chunk.length) {
          await api.postJson('/stt/push', { sample_rate: SAMPLE_RATE, samples: chunk })
        }
      }

      await api.postJson('/stt/push', {
        sample_rate: SAMPLE_RATE,
        samples: new Array(Math.floor(SAMPLE_RATE * SILENCE_SECONDS)).fill(0),
      })

      const deadline = Date.now() + STT_TIMEOUT_MS
      let transcript = ''
      while (Date.now() < deadline) {
        await new Promise(resolve => window.setTimeout(resolve, 350))
        const events = await (await api.get('/stt/transcripts')).json()
        const completed = [...events].reverse().find(event => event.kind === 'completed')
        if (completed?.text?.trim()) {
          transcript = completed.text.trim()
          break
        }
      }

      if (!transcript) throw new Error('Transcription timed out.')
      busyRef.current = false
      await runTurn(transcript)
    } catch (error) {
      addMessage('error', error.message || 'Push-to-talk failed.')
      busyRef.current = false
      setStatusLabel('Ready')
      setStatusState('idle')
    }
  }, [addMessage, api, decodeToMono16k, runTurn])

  const startTalk = useCallback(async () => {
    if (busyRef.current || isLive || recorderRef.current) return

    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
      addMessage('error', 'This browser does not support microphone recording.')
      return
    }

    busyRef.current = true
    stopPlayback()
    setStatusLabel('Listening…')
    setStatusState('recording')

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
        video: false,
      })
      const mimeType = getRecorderMimeType()
      const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream)
      recorderRef.current = recorder
      chunksRef.current = []
      recorder.ondataavailable = event => {
        if (event.data.size) chunksRef.current.push(event.data)
      }
      recorder.start(100)
      recordingTimerRef.current = window.setTimeout(() => finishRecording(), MAX_RECORDING_MS)
    } catch (error) {
      streamCleanup()
      addMessage('error', `Microphone access failed: ${error.message}`)
      busyRef.current = false
      setStatusLabel('Ready')
      setStatusState('idle')
    }
  }, [addMessage, finishRecording, isLive, stopPlayback])

  const stopTalk = useCallback(() => {
    if (recorderRef.current?.state !== 'inactive') finishRecording()
  }, [finishRecording])

  const disconnectLive = useCallback(() => {
    window.clearInterval(pollRef.current)
    pollRef.current = null
    window.clearTimeout(recordingTimerRef.current)
    recordingTimerRef.current = null

    peerRef.current?.close()
    peerRef.current = null

    localStreamRef.current?.getTracks().forEach(track => track.stop())
    localStreamRef.current = null

    if (remoteAudioRef.current) remoteAudioRef.current.srcObject = null

    setMessages(current => current.filter(message => !message.liveTranscript))
    setIsLive(false)
    busyRef.current = false
    setStatusLabel('Ready')
    setStatusState('idle')
  }, [])

  const pollLive = useCallback(async () => {
    if (!mountedRef.current || !peerRef.current) return

    try {
      const events = await (await api.get('/stt/transcripts')).json()
      const live = [...events].reverse().find(event => event.kind === 'updated' || event.kind === 'started')

      setMessages(current => {
        const base = current.filter(message => !message.liveTranscript)
        return live?.text?.trim()
          ? [...base, { id: 'live-transcript', role: 'user', content: live.text, liveTranscript: true }]
          : base
      })

      const turns = await (await api.get('/conversation/history')).json()
      if (turns.length > knownHistoryRef.current) {
        const newTurns = turns.slice(knownHistoryRef.current)
        setMessages(current => [
          ...current.filter(message => !message.liveTranscript),
          ...newTurns.map(turn => ({ id: makeId(), role: turn.role, content: turn.content })),
        ])
        knownHistoryRef.current = turns.length
      }
    } catch {
      // A transient polling failure should not terminate the media session.
    }
  }, [api])

  const connectLive = useCallback(async () => {
    if (busyRef.current || isLive) return

    if (!navigator.mediaDevices?.getUserMedia || !window.RTCPeerConnection) {
      addMessage('error', 'This browser does not support live voice.')
      return
    }

    busyRef.current = true
    stopPlayback()
    setStatusLabel('Connecting…')
    setStatusState('busy')

    let peer
    try {
      await api.delete('/stt/transcripts')
      const existingHistory = await (await api.get('/conversation/history')).json()
      knownHistoryRef.current = existingHistory.length

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
        video: false,
      })
      localStreamRef.current = stream

      peer = new RTCPeerConnection()
      peerRef.current = peer

      stream.getAudioTracks().forEach(track => peer.addTrack(track, stream))
      peer.ontrack = event => {
        if (remoteAudioRef.current && event.streams[0]) {
          remoteAudioRef.current.srcObject = event.streams[0]
          remoteAudioRef.current.play().catch(() => {})
        }
      }

      peer.onconnectionstatechange = () => {
        if (!mountedRef.current) return
        if (peer.connectionState === 'connected') {
          setIsLive(true)
          busyRef.current = false
          setStatusLabel('Live conversation')
          setStatusState('live')
          knownHistoryRef.current = 0
          window.clearInterval(pollRef.current)
          pollRef.current = window.setInterval(pollLive, LIVE_POLL_MS)
        } else if (['failed', 'disconnected', 'closed'].includes(peer.connectionState)) {
          disconnectLive()
        }
      }

      const offer = await peer.createOffer()
      await peer.setLocalDescription(offer)
      await new Promise((resolve, reject) => {
        if (peer.iceGatheringState === 'complete') {
          resolve()
          return
        }
        const timeout = window.setTimeout(() => {
          cleanup()
          reject(new Error('Timed out gathering ICE candidates.'))
        }, 10000)
        const onStateChange = () => {
          if (peer.iceGatheringState === 'complete') {
            cleanup()
            resolve()
          }
        }
        const cleanup = () => {
          window.clearTimeout(timeout)
          peer.removeEventListener('icegatheringstatechange', onStateChange)
        }
        peer.addEventListener('icegatheringstatechange', onStateChange)
      })

      const response = await fetch(api.root() + '/webrtc/offer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify({
          sdp: peer.localDescription?.sdp,
          type: peer.localDescription?.type,
        }),
      })
      if (!response.ok) {
        const data = await response.json().catch(() => ({}))
        throw new Error(data.detail || response.statusText || 'WebRTC negotiation failed.')
      }

      await peer.setRemoteDescription(await response.json())
      if (peer.connectionState === 'connected') {
        setIsLive(true)
        busyRef.current = false
        setStatusLabel('Live conversation')
        setStatusState('live')
        knownHistoryRef.current = 0
        window.clearInterval(pollRef.current)
        pollRef.current = window.setInterval(pollLive, LIVE_POLL_MS)
      }
    } catch (error) {
      peer?.close()
      localStreamRef.current?.getTracks().forEach(track => track.stop())
      localStreamRef.current = null
      peerRef.current = null
      busyRef.current = false
      setStatusLabel('Ready')
      setStatusState('idle')
      addMessage('error', `Live voice failed: ${error.message}`)
    }
  }, [addMessage, api, disconnectLive, isLive, pollLive, stopPlayback])

  const clearConversation = useCallback(() => {
    historyRef.current = []
    knownHistoryRef.current = 0
    setMessages([])
    api.delete('/conversation/history').catch(() => {})
    api.delete('/stt/transcripts').catch(() => {})
  }, [api])

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      window.clearInterval(pollRef.current)
      window.clearTimeout(recordingTimerRef.current)
      recorderRef.current?.stream.getTracks().forEach(track => track.stop())
      peerRef.current?.close()
      localStreamRef.current?.getTracks().forEach(track => track.stop())
      stopPlayback()
    }
  }, [stopPlayback])

  return {
    apiUrl,
    setApiUrl,
    healthState,
    messages,
    text,
    setText,
    statusLabel,
    statusState,
    isLive,
    isBusy: statusState === 'busy' || statusState === 'recording',
    endRef,
    remoteAudioRef,
    runTurn,
    startTalk,
    stopTalk,
    connectLive,
    disconnectLive,
    clearConversation,
  }
}

function streamCleanup() {
  // The stream is not available when getUserMedia itself rejects.
}
