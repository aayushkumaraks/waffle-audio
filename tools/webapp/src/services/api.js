export function createApiClient(baseUrl) {
  const root = () => baseUrl().replace(/\/$/, '')
  async function request(path, options = {}) {
    const response = await fetch(root() + path, options)
    if (!response.ok) {
      const detail = await response.json().then(data => data.detail).catch(() => response.statusText)
      throw new Error(detail || 'Request failed')
    }
    return response
  }
  return {
    get: path => request(path),
    postJson: (path, body) => request(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }),
    delete: path => request(path, { method: 'DELETE' }),
    root,
  }
}
