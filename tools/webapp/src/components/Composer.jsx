export default function Composer({ text, onTextChange, onSend, onTalkStart, onTalkEnd, statusLabel, statusState, disabled }) {
  return (
    <footer className="composer-shell">
      <div className="composer">
        <form className="text-composer" onSubmit={e => { e.preventDefault(); onSend() }}>
          <textarea value={text} rows={1} placeholder="Write a message…" disabled={disabled}
            onChange={e => onTextChange(e.target.value)}
            onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); onSend() } }} />
          <button className="send-button" type="submit" disabled={disabled || !text.trim()} aria-label="Send message">↑</button>
        </form>
        <button className={"talk-button " + (statusState === "recording" ? "recording" : "")} disabled={disabled}
          onPointerDown={e => { e.preventDefault(); onTalkStart() }} onPointerUp={onTalkEnd} onPointerLeave={onTalkEnd}
          aria-label="Hold to talk"><span className="mic-icon">✦</span></button>
        <div className="composer-status"><span className={"status-pip " + statusState} />{statusLabel}<span className="status-hint">Hold the button to speak</span></div>
      </div>
    </footer>
  )
}
