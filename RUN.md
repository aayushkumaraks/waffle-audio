# Running Waffle Audio

Quick reference for running Waffle Audio. For full details, see the [Running & CLI Guide](docs/running.md).

---

## 1. One-Click Start (CLI App)

The fastest way to install dependencies, verify models, and start the voice assistant:

```bash
./run.sh
```

On Windows:
```cmd
run.bat
```

> [!TIP]
> On your first run, `./run.sh` automatically launches the interactive **Setup Wizard** (`setup.py`) to personalize your assistant and detect your microphone and Ollama setup. You can re-run the wizard anytime with `./run.sh --setup` or `uv run python setup.py --force`.

Or start the CLI app directly with `uv`:

```bash
uv run python -m services.conversation.cli
```

---

## 2. Common CLI Flags

| Command / Flag | Purpose |
|---|---|
| `./run.sh -v` / `--verbose` | Show real-time turn latency timeline and compare with benchmarks |
| `./run.sh --benchmark` | Enable latency metrics, tool execution logging, and session averages |
| `./run.sh --trigger-mode` | Only respond when wake phrase `"Hey <ASSISTANT_NAME>"` is spoken |
| `./run.sh --voice <name>` | Start with a specific voice or cloned voice from `cloned_voices/` |
| `./run.sh --setup` | Re-run the interactive setup wizard before starting |

---

## 3. Voice Cloning

Clone any custom voice in 10–15 seconds:

```bash
uv run python clone_voice.py
```

See [docs/voice-cloning.md](docs/voice-cloning.md) for full instructions.

---

## 4. MCP Tools & Search

Manage installed Model Context Protocol (MCP) tools and web search:

```bash
uv run python install_tool.py
```

See [docs/tools-and-mcp.md](docs/tools-and-mcp.md) for presets and options.

---

## 5. HTTP API Server

Start the headless FastAPI server process:

```bash
uv run python -m services.api.app
```

Interactive Swagger documentation is available at: `http://localhost:3000/docs`

---

## 6. Running Unit Tests

Run the complete test suite across all services and scripts:

```bash
# Run via pytest (recommended)
uv run pytest

# Or run via Python unittest runner
uv run python -m unittest discover tests
```

---

## 7. Full Documentation

- **[Installation & Setup](docs/setup.md)**
- **[CLI & Running Guide](docs/running.md)**
- **[Configuration Reference](docs/configuration.md)**
- **[System Architecture](docs/architecture.md)**
- **[Audio Processing](docs/audio-processing.md)**
- **[Tools & MCP](docs/tools-and-mcp.md)**
- **[Latency Benchmarks](docs/benchmarks.md)**
- **[Voice Cloning](docs/voice-cloning.md)**
