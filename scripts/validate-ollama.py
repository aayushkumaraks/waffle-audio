#!/usr/bin/env python3
"""Validate Ollama connectivity and perform a minimal chat request."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request


base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434").rstrip("/")
model = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")


def request(path: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="GET" if data is None else "POST",
    )
    with urllib.request.urlopen(req, timeout=15 if payload is None else 120) as response:
        return json.load(response)


try:
    version = request("/api/version")
except Exception as exc:
    print(f"ERROR: Cannot reach Ollama at {base_url}: {exc}", file=sys.stderr)
    print(
        "Start Ollama on the host and ensure it accepts connections from Docker. "
        "Use host.docker.internal for Docker Desktop; on Linux, configure Ollama "
        "to listen on an address reachable from the container.",
        file=sys.stderr,
    )
    sys.exit(1)

print(f"Connected to Ollama {version.get('version', '(version unknown)')} at {base_url}")

try:
    tags = request("/api/tags")
    available = {item.get("name", "") for item in tags.get("models", [])}
    if not any(name == model or name.startswith(f"{model}:") for name in available):
        print(f"ERROR: Ollama model '{model}' is not installed.", file=sys.stderr)
        print(f"Run on the host: ollama pull {model}", file=sys.stderr)
        sys.exit(2)

    result = request(
        "/api/chat",
        {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with the single word OK."}],
            "stream": False,
            "think": False,
        },
    )
    answer = result.get("message", {}).get("content", "").strip()
    if not answer:
        raise RuntimeError("Ollama returned an empty chat response.")
    print(f"Chat API check passed using '{model}'. Response: {answer[:120]}")
except urllib.error.HTTPError as exc:
    detail = exc.read().decode("utf-8", errors="replace")[:500]
    print(f"ERROR: Ollama API returned HTTP {exc.code}: {detail}", file=sys.stderr)
    sys.exit(1)
except Exception as exc:
    print(f"ERROR: Ollama chat validation failed: {exc}", file=sys.stderr)
    sys.exit(1)
