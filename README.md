# Hologlass

Real-time speech translation for a heads-up display, running **entirely on your
own machine**. Speak Spanish or French at it; English captions appear.

No cloud. No API keys. No account. Your voice never leaves the device, and
nothing is written to disk.

This is stage one of a privacy-first alternative to cloud-backed smart glasses:
the software layer, built as a desktop simulation of the display. It does one
thing, and it does it without sending your voice anywhere.

Everything it runs on is open source — Whisper, OPUS-MT and Silero VAD, all under
MIT or Apache-2.0. See [Built on](#built-on).

> **Status:** working prototype, measured on a Ryzen 5 5600X, CPU only.
> Response latency **~1.2 s**, real-time factor **0.10**. Accuracy is preferred
> over speed where the two conflict; see the benchmark section for what that
> choice cost and why it was made.

## Privacy

The whole point of the project, so it goes first.

- Audio is captured, transcribed, and translated **locally**. It is never uploaded.
- **No API keys. No inference-time network calls.** The only network access in the
  entire project is `scripts/fetch_models.py`, run once during setup.
- **Nothing is written to disk.** Audio exists in memory for the length of one
  utterance and is discarded.
- The server binds to `127.0.0.1` by default, so it is not reachable from your
  network unless you deliberately change that.

You do not have to take this on trust. There is exactly one file that touches the
network, and CI fails the build if PyTorch ever enters the runtime dependency set.

## Architecture

```
microphone ──► VAD ──► segmenter ──► ASR ──► MT ──► WebSocket ──► HUD
   16 kHz    Silero v5   utterance  Whisper Marian   FastAPI     browser
   512-frame   ONNX       gating      int8    int8
```

Every inference stage runs on **CTranslate2 or ONNX Runtime**. PyTorch is not a
runtime dependency; it is needed only at setup time to convert the translation
models. That takes the install from roughly 2.5 GB to a few hundred MB.

### The design decision that matters

The unit of work is the **utterance**, not a fixed time slice.

An earlier version of this project transcribed every five seconds regardless of
whether anyone was speaking, then re-translated an overlapping window of previous
chunks on each tick. Cost per second of speech grew with the context window, most
of that work was discarded, and chunk boundaries landed mid-word.

Here, voice activity detection gates the pipeline. Silence costs about a
millisecond of VAD instead of a full transcription pass, each utterance is
transcribed once and translated once, and boundaries land where a human would put
them.

The segmenter also applies five rules the naive version lacked, each fixing an
observed failure:

| Rule | Failure it prevents |
|---|---|
| Hysteresis (separate enter/exit thresholds) | State flapping on marginal frames, splitting one sentence into three |
| Pre-roll buffer | Clipped onsets — Silero needs a few frames to become confident, and the first consonant was being lost |
| Trailing-silence trim | Paying ASR cost to transcribe the silence that proved the utterance ended |
| Minimum speech duration | Coughs and keyboard clicks producing hallucinated transcripts |
| Maximum duration cap | An unbroken monologue never producing output at all |

### Why translation stays on the CPU

Marian int8 translates a sentence in **48-71 ms** (measured, Ryzen 5 5600X). That
is far below the ASR cost and below the threshold of perception. Accelerating it
would add a device dependency to save time nobody can notice. ASR is the only
stage where compute would buy anything, and a quantized `base` model already keeps
up with speech on one desktop core.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements-dev.txt
python -m scripts.fetch_models  # one-time, downloads and converts models
```

## Run

```bash
python -m server
```

Open <http://127.0.0.1:8000>, choose a language, press Start.

## Benchmark

`bench/run.py` replays recorded audio through the real pipeline, so runs are
deterministic and comparable.

```bash
python -m bench.run --audio sample.wav --model-size tiny base small
python -m bench.run --audio sample.wav --reference sample.es.txt --target sample.en.txt
```

Reports per-stage latency, real-time factor, and — given reference files — WER
and BLEU. It reports **response latency**, not just processing time: the endpoint
hold is not compute, but the speaker waits for it all the same, and a benchmark
that omits it understates the number that matters by hundreds of milliseconds.

### Where the time goes

Measured on a Ryzen 5 5600X, CPU only, Whisper `base` int8, French → English.

| Stage | Cost |
|---|---|
| Endpoint hold | 600 ms |
| VAD | ~1 ms/frame, 121 ms total over 24 s |
| ASR | 565 ms |
| MT | 55 ms |
| **Response latency** | **~1220 ms** |
| Real-time factor | 0.10 |

### The latency/accuracy trade, and why it went this way

The largest single term is not compute. It is `min_silence_ms`, the silence the
segmenter waits through before deciding the speaker has finished. Cutting it from
600 ms to 350 ms is worth **281 ms, a 23% reduction** — far more than any decode
setting achieves.

It was tried, measured, and reverted.

A shorter hold means results are computed on shorter fragments, and Whisper leans
heavily on hearing a complete phrase. The speed was real; so was the accuracy
cost. For a translation tool, a caption that arrives 280 ms sooner and says
something subtly wrong is worse than one that waits and is right.

Several decode-level settings were also tried as "free" wins — disabling the
temperature fallback ladder, suppressing timestamp tokens, capping how much audio
provisional results re-read. A sweep isolating each one produced a **byte-identical
transcript** in every case, at latency differences smaller than run-to-run noise.
They were reverted too: they bought nothing and gave up Whisper's own error
recovery.

The honest summary is that there was no free lunch here. The only lever that
moved latency meaningfully was the one that also cost accuracy.

One piece of that work was kept, because it is free in the other direction.
`continuation_grace_ms` holds an utterance open briefly after emitting, so a pause
of up to 850 ms rejoins the same sentence rather than splitting it. At the
original hold this costs no latency at all and yields **longer, more complete
utterances than the pipeline started with** — which is exactly what Whisper wants.

### Why `base`

Model size is a real trade, and the fast option loses on the merits rather than
on latency.

| Model | ASR p50 | Transcript of the same French clip |
|---|---|---|
| tiny | 255 ms | *"je suis désiré pour le haut de la tête"* — wrong sentence |
| **base** | **532 ms** | *"je suis testé le audio de voir si ça fonctionne"* — correct meaning |
| small | 1613 ms | *"je vais tester l'audio pour voir si ça fonctionne"* — best French |

`tiny` is twice as fast and invents a different sentence, so its speed buys
nothing. `small` is three times slower for a grammatical improvement that does
not change the meaning. `base` is the point where the output is trustworthy and
the pipeline still keeps up with speech.

## Tests

```bash
pytest -q      # 15 tests, no microphone or model downloads required
ruff check .
```

Tests substitute fake ASR and MT backends and drive the segmenter with scripted
VAD probabilities, so the logic most likely to be subtly wrong is verified in
milliseconds and identically on every run.

## Layout

| Path | Purpose |
|---|---|
| `core/` | Audio capture, VAD segmentation, ASR and MT backends, pipeline. No UI imports. |
| `server/` | FastAPI app and WebSocket transport. |
| `web/` | Static heads-up-display page. No build step, no dependencies. |
| `bench/` | Latency and accuracy measurement harness. |
| `scripts/` | One-time model fetch and conversion. The only network access in the project. |
| `tests/` | Deterministic tests against fake backends and recorded fixtures. |

## Built on

This project does no machine learning of its own. It orchestrates open models and
runtimes built by others, and the credit for the hard parts belongs to them.

| Component | Role | By | Licence |
|---|---|---|---|
| [Whisper](https://github.com/openai/whisper) | speech recognition | OpenAI | MIT |
| [OPUS-MT](https://huggingface.co/Helsinki-NLP) | translation | University of Helsinki | Apache-2.0 |
| [Silero VAD](https://github.com/snakers4/silero-vad) | voice activity detection | Silero Team | MIT |
| [CTranslate2](https://github.com/OpenNMT/CTranslate2) | quantized inference runtime | OpenNMT | MIT |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | Whisper on CTranslate2 | SYSTRAN | MIT |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | VAD inference | Microsoft | MIT |
| [FastAPI](https://github.com/fastapi/fastapi) | server and WebSocket transport | Sebastián Ramírez | MIT |

Every licence above was verified against the project's own repository or model
card. Full attribution is in [NOTICE](NOTICE).

Because all of them are permissive, the whole stack can run offline with no
account, no key and no terms of service. That is what makes the privacy claim
structural rather than a promise.

## Licence

Apache License 2.0. See [LICENSE](LICENSE).

You may use, modify and distribute this, including commercially, provided you
preserve the copyright and licence notices and state any changes you made.
