import AppHeader from './components/AppHeader'
import MessageList from './components/MessageList'
import Composer from './components/Composer'
import { useVoiceChat } from './hooks/useVoiceChat'
import './App.css'

export default function App() {
  const chat = useVoiceChat()
  const send = () => { const value = chat.text.trim(); if (!value) return; chat.setText(''); chat.runTurn(value) }

  return (
    <div className="app-shell">
      <AppHeader apiUrl={chat.apiUrl} onApiUrlChange={chat.setApiUrl} healthState={chat.healthState}
        isLive={chat.isLive} onToggleLive={chat.isLive ? chat.disconnectLive : chat.connectLive} onClear={chat.clearConversation} />
      <MessageList messages={chat.messages} endRef={chat.endRef} />
      <audio ref={chat.remoteAudioRef} autoPlay playsInline className="remote-audio" />
      <Composer text={chat.text} onTextChange={chat.setText} onSend={send} onTalkStart={chat.startTalk} onTalkEnd={chat.stopTalk}
        statusLabel={chat.statusLabel} statusState={chat.statusState} disabled={chat.isBusy || chat.isLive} />
    </div>
  )
}
