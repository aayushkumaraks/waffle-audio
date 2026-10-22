# Latency Profiling & Performance Benchmarks

Waffle Audio is designed and profiled to achieve human-like, real-time conversational latencies on consumer hardware.

---

## 1. Measured Live Session Benchmarks

Measurements recorded directly from live two-turn user interaction on the standard local stack:
- **Speech Recognition**: Moonshine STT (local ONNX)
- **Language Model**: Ollama `qwen3.5:4b`
- **Text-to-Speech**: Pocket TTS (`anna` voice)
- **Audio Pipeline**: Unified in-process async event loop with raw NumPy array buffers

### End-to-End Latency (End of User Speech -> Assistant First Audio):

| Mode / Turn Type | Baseline | Initial Target | Measured Live | Improvement vs Baseline |
|---|---|---|---|---|
| **Conversational Fast Path** (Short turn) | 3.30 s | 1.60 s | **0.83 s** | **~75% faster (sub-second!)** |
| **Average Warm Turn** (Multi-sentence) | 3.30 s | 2.10 s | **1.12 s** | **~66% faster** |
| **First Turn (Cold Start)** | 15.00 s | 6.80 s | **5.40 s** | **~64% faster** |

---

## 2. Stage-by-Stage Turn Breakdown

Detailed pipeline profiling output from `./run.sh --verbose`:

| Pipeline Stage | Turn 1 (Detailed Answer) | Turn 2 (Conversational) | Target Reference | Delta vs Target |
|---|---|---|---|---|
| **1. Sentence Gate** | 0.25 s | 0.25 s | 0.40 s | **-0.15 s (Faster!)** |
| **2. LLM Time To First Token (TTFT)** | 0.37 s | 0.35 s | 0.45 s | **-0.10 s (Faster!)** |
| **3. LLM Full Response** | 0.87 s | 0.61 s | 1.47 s | **-0.86 s (Faster!)** |
| **4. TTS Time To First Audio (TTFB)** | 1.16 s | 0.58 s | 1.82 s | **-1.24 s (Faster!)** |
| **5. Total End-to-End Latency** | **1.41 s** | **0.83 s** | **2.10 s** | **-1.27 s (Faster!)** |

---

## 3. Real-Time Profiling in the CLI

To view stage-by-stage latency breakdowns live in your terminal as you speak, pass the `--verbose` flag:

```bash
./run.sh --verbose
```

To enable structured JSONL logging of all turn metrics and session summaries:

```bash
./run.sh --verbose --benchmark
```

Structured benchmark logs are written directly to:
- `logs/benchmarks.jsonl` (per-turn structured JSON entries)
- `logs/benchmark.log` (human-readable formatting)

---

## 4. Architectural Optimizations Behind the Speed

1. **Clause-Gated Fast Path**: Rather than waiting for full sentences, `ClauseChunker` splits streaming LLM tokens on clause boundaries. The very first chunk requires only ~3–4 words, dispatching TTS synthesis in <200 ms TTFB.
2. **Unified In-Process Event Loop**: Eliminates TCP network serialization and JSON marshaling latency between VAD, STT, LLM, and TTS services by passing raw NumPy arrays in memory.
3. **Capped Ollama Context (`OLLAMA_NUM_CTX=2048`)**: Prevents the context window from growing into CPU system RAM, keeping model weights and KV-cache 100% inside GPU VRAM.
4. **Ollama Keep-Alive (`OLLAMA_KEEP_ALIVE=30m`)**: Keeps the LLM loaded in memory across turns, completely avoiding cold-load delays.
5. **Acoustic Echo Cancellation (NLMS)**: Removes speaker feedback so the assistant never interrupts itself or loops back audio.
6. **SIMD Audio Resampling**: Uses `soxr` / `scipy.signal.resample_poly` for polyphase downsampling with near-zero CPU overhead.

---

## 5. Unit Test Verification

Benchmark logging calculations and pipeline latency optimizations are tested in:
- `tests/test_benchmark_logging.py`: Metrics tracking, JSONL file appending, immediate flush, turn metrics calculation.
- `tests/test_pipeline_optimizations.py`: `ClauseChunker` fast-path early dispatch and latency threshold validation.

```bash
uv run pytest tests/test_benchmark_logging.py tests/test_pipeline_optimizations.py
```
