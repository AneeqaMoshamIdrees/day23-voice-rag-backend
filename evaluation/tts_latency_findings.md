
# Day 23 — TTS Streaming Latency Findings

## Test Setup

- Model: XTTS v2 (coqui-tts 0.27.5)

- Hardware: CPU-only (no GPU available)

- Reference voice: 10.5s WAV sample, 22050Hz mono

- Test text: 4-sentence RAG-style answer (~380 characters)

- Model load time (one-time, excluded from per-request measurements): 22.61s

## Results

| Approach | Time To First Audio (TTFA) | Total Synthesis Time |

|---|---|---|

| Non-streaming (single full-text request) | 66.47s | 66.47s |

| Streaming (sentence-by-sentence chunking) | 25.32s | 78.35s |

**TTFA improvement: 41.15s faster (61.9% reduction)**

## Findings

1. **Streaming significantly reduces perceived latency.** By synthesizing

   and returning the first sentence immediately rather than waiting for

   the complete answer, users hear audio 61.9% sooner.

2. **Streaming has a small total-time cost.** Total synthesis time

   increased slightly (66.47s → 78.35s) because each chunk incurs its

   own per-call overhead in XTTS's inference pipeline. This is an

   acceptable tradeoff: for a conversational voice assistant, minimizing

   time-to-first-audio matters far more to perceived responsiveness than

   minimizing total generation time, since the user can begin listening

   to earlier chunks while later chunks are still synthesizing.

3. **Primary bottleneck: CPU-only inference.** XTTS v2 is a

   transformer-based autoregressive model designed with GPU acceleration

   in mind. On CPU, per-sentence synthesis ranges from ~13s to ~25s

   depending on sentence length, making even single-chunk latency high

   in absolute terms. A GPU-equipped deployment would be expected to

   reduce these times by an order of magnitude, based on documented

   XTTS v2 benchmarks.

4. **Accent drift.** The cloned voice preserves timbre/vocal texture

   accurately from the reference sample, but exhibits a shift toward

   generic English prosody rather than fully preserving the speaker's

   native accent — a documented limitation of XTTS v2's zero-shot

   cloning rather than an implementation issue.

