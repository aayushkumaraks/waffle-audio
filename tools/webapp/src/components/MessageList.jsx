export default function MessageList({ messages, endRef }) {
  if (!messages.length) return (
    <main className="conversation conversation-empty" role="log" aria-live="polite">
      <div className="welcome-card">
        <div className="welcome-waffle" aria-hidden="true">{Array.from({ length: 9 }, (_, i) => <span key={i} />)}</div>
        <p className="eyebrow">Ready when you are</p>
        <h1>Have a conversation.</h1>
        <p>Talk naturally, type a message, or switch to live voice. Your conversation stays in this session.</p>
      </div>
    </main>
  )
  return (
    <main className="conversation" role="log" aria-live="polite">
      <div className="conversation-inner">
        {messages.map(message => (
          <article key={message.id} className={"message message-" + message.role + (message.liveTranscript ? " message-live" : "")}>
            <div className="message-meta"><span>{message.role === "user" ? "You" : message.role === "error" ? "System" : "Waffle"}</span></div>
            <div className="message-body">{message.content}</div>
          </article>
        ))}
        <div ref={endRef} />
      </div>
    </main>
  )
}
