"""Speech recognition backends.

The project targets CPU inference. A quantized `base` model keeps up with speech
on an ordinary desktop core, and committing to CPU means the thing runs anywhere
rather than only where a particular accelerator and its driver stack are present
-- which is the right constraint for something simulating a wearable.

Everything still sits behind `ASRBackend`. That is not a hedge about hardware: it
is what lets the tests substitute a fake engine and run in milliseconds with no
model, and what lets the benchmark compare model sizes and quantization levels
through one code path.
"""

from __future__ import annotations

import logging
import time
from typing import Protocol, runtime_checkable

import numpy as np

from core.config import SAMPLE_RATE, ASRConfig

logger = logging.getLogger(__name__)


class ASRResult:
    """Transcribed text plus the time it cost to produce."""

    __slots__ = ("text", "elapsed_ms")

    def __init__(self, text: str, elapsed_ms: float) -> None:
        self.text = text
        self.elapsed_ms = elapsed_ms


@runtime_checkable
class ASRBackend(Protocol):
    """Contract every speech recognition engine must satisfy."""

    name: str

    def transcribe(self, audio: np.ndarray, language: str) -> ASRResult:
        """Transcribe mono float32 audio at SAMPLE_RATE into `language` text."""
        ...

    def warmup(self, language: str) -> None:
        """Run one throwaway inference so the first real utterance is not slow."""
        ...


class FasterWhisperASR:
    """Whisper via CTranslate2, quantized to int8.

    CTranslate2 is used rather than the reference PyTorch implementation for
    three reasons: it is several times faster on CPU, int8 weights cut memory to
    roughly a quarter, and it removes torch from the runtime entirely.

    Greedy decoding is the default. On utterances of a few seconds, beam search
    costs meaningful latency and rarely changes the output.
    """

    def __init__(self, config: ASRConfig | None = None) -> None:
        from faster_whisper import WhisperModel  # imported lazily: heavy

        self._cfg = config or ASRConfig()
        self.name = f"faster-whisper/{self._cfg.model_size}/{self._cfg.compute_type}/{self._cfg.device}"

        logger.info("loading %s", self.name)
        started = time.perf_counter()
        self._model = WhisperModel(
            self._cfg.model_size,
            device=self._cfg.device,
            compute_type=self._cfg.compute_type,
            cpu_threads=self._cfg.cpu_threads,
        )
        logger.info("loaded in %.1f s", time.perf_counter() - started)

    def transcribe(self, audio: np.ndarray, language: str) -> ASRResult:
        started = time.perf_counter()
        segments, _info = self._model.transcribe(
            audio.astype(np.float32, copy=False),
            language=language,
            beam_size=self._cfg.beam_size,
            # The pipeline has already done voice-activity gating, so the audio
            # handed over is known to contain speech. Running Whisper's own VAD
            # again would duplicate that cost.
            vad_filter=False,
            # Suppresses the "thank you for watching" style hallucinations that
            # Whisper emits when handed near-silence.
            condition_on_previous_text=False,
            temperature=list(self._cfg.temperature),
            without_timestamps=self._cfg.without_timestamps,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return ASRResult(text, (time.perf_counter() - started) * 1000.0)

    def warmup(self, language: str) -> None:
        """Transcribe half a second of silence to force lazy init and allocation."""
        self.transcribe(np.zeros(SAMPLE_RATE // 2, dtype=np.float32), language)
        logger.info("%s warm", self.name)


def build_asr(config: ASRConfig | None = None) -> ASRBackend:
    """Construct the configured ASR backend."""
    return FasterWhisperASR(config)
