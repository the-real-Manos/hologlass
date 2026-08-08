"""Central configuration for the translation pipeline.

Every tunable lives here rather than being scattered through the code, so the
benchmark harness can sweep parameters without touching the pipeline itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
VAD_MODEL_PATH = MODELS_DIR / "silero_vad.onnx"
MT_MODELS_DIR = MODELS_DIR / "mt"

# Silero VAD v5 is trained for 16 kHz and requires exactly 512-sample frames.
# Neither value is free to change.
SAMPLE_RATE = 16_000
FRAME_SAMPLES = 512
FRAME_MS = FRAME_SAMPLES / SAMPLE_RATE * 1000  # 32 ms


@dataclass(frozen=True)
class LanguagePair:
    """A supported source language and the model that translates it to English."""

    name: str
    code: str
    hf_model: str

    @property
    def ct2_dir(self) -> Path:
        """Local directory holding the CTranslate2-converted weights."""
        return MT_MODELS_DIR / self.hf_model.split("/")[-1]


LANGUAGES: dict[str, LanguagePair] = {
    "Spanish": LanguagePair("Spanish", "es", "Helsinki-NLP/opus-mt-es-en"),
    "French": LanguagePair("French", "fr", "Helsinki-NLP/opus-mt-fr-en"),
}

DEFAULT_LANGUAGE = "Spanish"


@dataclass
class SegmenterConfig:
    """Voice-activity segmentation behaviour.

    The defaults trade a little latency for a lot of stability: hysteresis stops
    the segmenter flickering on marginal frames, and the pre-roll keeps the
    consonant at the start of an utterance instead of clipping it.
    """

    speech_threshold: float = 0.50
    """Probability above which a frame starts (or sustains) speech."""

    silence_threshold: float = 0.35
    """Probability below which a frame counts as silence. Lower than
    `speech_threshold` on purpose: the gap is the hysteresis band."""

    min_silence_ms: int = 600
    """Trailing silence before a result is emitted.

    Pure additive latency on every utterance: nothing can be shown until this
    elapses, so it is tempting to shorten. Doing so was measured and reverted.

    Cutting it to 350 ms did save 281 ms, but a shorter hold means results are
    computed on shorter fragments, and Whisper depends heavily on hearing a
    complete phrase. The speed was real and so was the accuracy cost; on this
    project accuracy wins. This value is deliberately back at its original."""

    continuation_grace_ms: int = 250
    """How long an emitted utterance stays open to being continued.

    Retained from the latency work, where it existed to make a short hold safe.
    It is worth keeping at the original hold for its own sake: a pause of up to
    `min_silence_ms + continuation_grace_ms` now rejoins the same utterance
    instead of splitting it, so a slow speaker gets one complete sentence rather
    than two fragments. Since Whisper is markedly better on complete sentences,
    this is an accuracy gain that costs no latency at all."""

    pre_roll_ms: int = 300
    """Audio retained from before speech was detected. Silero needs a few frames
    to become confident, and without this the onset is lost."""

    min_speech_ms: int = 250
    """Utterances shorter than this are discarded as noise blips."""

    max_utterance_ms: int = 12_000
    """Hard cap. Forces a flush so a monologue still produces output."""

    interim_every_ms: int = 900
    """Cadence for provisional transcripts while speech is ongoing."""


@dataclass
class ASRConfig:
    model_size: str = "base"
    device: str = "cpu"
    compute_type: str = "int8"
    beam_size: int = 1
    """Greedy by default. Beam search costs latency for marginal gain on the
    short utterances this pipeline produces."""

    cpu_threads: int = 0
    """0 lets CTranslate2 choose. The benchmark sweeps this."""

    temperature: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    """Fallback ladder for failed decodes: Whisper's own default.

    When a decode trips its compression-ratio or log-prob check, it retries at
    the next temperature. Those retries are what rescue a repetition loop or a
    low-confidence garble.

    Truncating this ladder, and removing it outright, were both tried as latency
    optimisations. Neither changed the transcript on the available fixture and
    neither moved the median beyond run-to-run noise, so both gave up error
    recovery for nothing. Left at the full ladder until there is evidence that
    shortening it buys something real."""

    without_timestamps: bool = False
    """Whether to suppress timestamp tokens.

    Measured on the French fixture, suppressing them changed the transcript not
    at all and the latency by less than run-to-run noise. Since the gain was
    unmeasurable, the default stays at Whisper's own trained behaviour rather
    than an optimisation that only looked free."""


@dataclass
class MTConfig:
    beam_size: int = 2
    max_decoding_length: int = 128
    compute_type: str = "int8"
    inter_threads: int = 1
    intra_threads: int = 0
    cache_size: int = 256
    """Interim transcripts repeat heavily; caching avoids re-translating them."""


@dataclass
class AppConfig:
    language: str = DEFAULT_LANGUAGE
    translate_interim: bool = True
    """Translating interims feels more responsive but roughly doubles MT load."""

    interim_max_seconds: float | None = None
    """Cap on how much audio a provisional transcript re-reads, or None for all.

    Whisper has no streaming mode, so every interim re-transcribes the utterance
    from its start and the cost grows with the length of the sentence. Capping it
    at 4 s was tried and reverted: a truncated tail starts mid-word and is
    transcribed noticeably worse, and provisional text is exactly what the
    speaker watches while talking, so it read as a general loss of accuracy.

    Uncapped by default. `max_utterance_ms` already bounds how long an utterance
    can run, so the cost has a ceiling regardless."""

    segmenter: SegmenterConfig = field(default_factory=SegmenterConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    mt: MTConfig = field(default_factory=MTConfig)

    @property
    def language_pair(self) -> LanguagePair:
        return LANGUAGES[self.language]
