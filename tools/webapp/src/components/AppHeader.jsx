export default function AppHeader({ apiUrl, onApiUrlChange, healthState, isLive, onToggleLive, onClear }) {
  return (
    <header className="app-header">
      <div className="brand">
        <div className="brand-mark" aria-hidden="true"><span /><span /><span /><span /></div>
        <div><div className="brand-name">Waffle</div><div className="brand-subtitle">Voice assistant</div></div>
      </div>
      <div className="header-actions">
        <label className="endpoint">
          <span className="sr-only">API base URL</span>
          <span className={"health-dot " + healthState} />
          <input value={apiUrl} onChange={e => onApiUrlChange(e.target.value)} aria-label="API base URL" spellCheck="false" />
        </label>
        <button className={"button button-live " + (isLive ? "active" : "")} onClick={onToggleLive}>
          <span className="button-icon">{isLive ? "■" : "●"}</span>{isLive ? "End live" : "Live voice"}
        </button>
        <button className="button button-quiet" onClick={onClear}>Clear</button>
      </div>
    </header>
  )
}
