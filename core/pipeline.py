"""Pipeline orchestration: audio in, translation events out.

This is the layer the previous prototype got wrong. It transcribed on a fixed
five-second timer regardless of whether anyone was speaking, then re-translated
an overlapping window of past chunks on every tick, so the cost per second of
speech grew with the size of the context window and most of that work was thrown
away.

Here the unit of work is the utterance. Each one is transcribed once and
translated once. Provisional results exist for responsiveness but never cause the
final result to be recomputed.

Exposed as a generator so it stays synchronous and testable. The server runs it
on a worker thread; the benchmark drains it directly.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

from core.asr import ASRBackend, build_asr
from core.audio import AudioSource
from core.config import SAMPLE_RATE, VAD_MODEL_PATH, AppConfig
from core.mt import MTBackend, build_mt
from core.types import EventKind, StageTimings, TranslationEvent
from core.vad import Segment, SileroVad, UtteranceSegmenter

logger = logging.getLogger(__name__)


class TranslationPipeline:
    """Wires VAD, ASR, and MT into a single stream of events."""

    def __init__(
        self,
        config: AppConfig | None = None,
        asr: ASRBackend | None = None,
        mt: MTBackend | None = None,
    ) -> None:
        self.config = config or AppConfig()
        language = self.config.language_pair

        self.asr = asr or build_asr(self.config.asr)
        self.mt = mt or build_mt(language, self.config.mt)
        self.segmenter = UtteranceSegmenter(
            SileroVad(VAD_MODEL_PATH), self.config.segmenter
        )
        self._last_interim_text = ""

    def warmup(self) -> None:
        """Pay first-inference cost up front rather than on the user's first word."""
        self.asr.warmup(self.config.language_pair.code)
        self.mt.warmup()

    def reset(self) -> None:
        self.segmenter.reset()
        self._last_interim_text = ""

    def run(self, source: AudioSource) -> Iterator[TranslationEvent]:
        """Consume frames from `source`, yielding events until it is exhausted."""
        try:
            for frame in source.frames():
                for segment in self.segmenter.process_frame(frame):
                    event = self._handle(segment)
                    if event is not None:
                        yield event

            tail = self.segmenter.flush()
            if tail is not None:
                event = self._handle(tail)
                if event is not None:
                    yield event
        finally:
            source.close()

    def _handle(self, segment: Segment) -> TranslationEvent | None:
        language = self.config.language_pair
        audio = segment.audio

        if not segment.is_final and self.config.interim_max_seconds is not None:
            # Opt-in only. Truncating provisional audio makes it start mid-word,
            # which Whisper transcribes noticeably worse; see interim_max_seconds.
            limit = int(self.config.interim_max_seconds * SAMPLE_RATE)
            if audio.size > limit:
                audio = audio[-limit:]

        asr_result = self.asr.transcribe(audio, language.code)
        text = asr_result.text.strip()

        if not text:
            return None

        if not segment.is_final:
            # Whisper often returns an unchanged provisional transcript across
            # consecutive interims. Re-sending it would cost an MT call and cause
            # the display to flicker for no reason.
            if text == self._last_interim_text:
                return None
            self._last_interim_text = text
        else:
            self._last_interim_text = ""

        translate = segment.is_final or self.config.translate_interim
        mt_ms = 0.0
        target = ""
        if translate:
            mt_result = self.mt.translate(text)
            target = mt_result.text
            mt_ms = mt_result.elapsed_ms

        return TranslationEvent(
            utterance_id=segment.utterance_id,
            kind=EventKind.FINAL if segment.is_final else EventKind.INTERIM,
            source_text=text,
            target_text=target,
            audio_duration_ms=segment.duration_ms,
            timings=StageTimings(
                vad_ms=segment.vad_ms,
                asr_ms=asr_result.elapsed_ms,
                mt_ms=mt_ms,
            ),
        )
