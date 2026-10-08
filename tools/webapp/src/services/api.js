export function normalizeBaseUrl(value) {
  return value.trim().replace(/\/+$/, '')
}

export function createApiClient(baseUrl) {
  const root = () => normalizeBaseUrl(baseUrl())

  async function request(path, options = {}) {
    const response = await fetch(root() + path, {
      ...options,
      headers: {
        Accept: 'application/json',
        ...(options.body ? { 'Content-Type': 'application/json' } : {}),
        ...(options.headers || {}),
      },
    })

    if (!response.ok) {
      let detail = response.statusText || 'Request failed'
      try {
        const data = await response.json()
        detail = data.detail || data.message || detail
      } catch {
        // Preserve the HTTP status when the response is not JSON.
      }
      throw new Error(detail)
    }

    return response
  }

  return {
    root,
    get: path => request(path),
    postJson: (path, body, options = {}) =>
      request(path, {
        ...options,
        method: 'POST',
        body: JSON.stringify(body),
      }),
    delete: path => request(path, { method: 'DELETE' }),
  }
}
