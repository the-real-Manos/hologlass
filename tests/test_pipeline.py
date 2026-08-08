"""Pipeline orchestration, tested with fake backends.

Substituting the models keeps these tests fast and deterministic while still
exercising the logic that actually caused problems in the previous prototype:
how often translation is invoked, and whether provisional results are
deduplicated before they reach the client.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.asr import ASRResult
from core.config import FRAME_SAMPLES, AppConfig
from core.mt import MTResult
from core.pipeline import TranslationPipeline
from core.types import EventKind
from core.vad import Segment


class FakeASR:
    name = "fake-asr"

    def __init__(self, transcripts: list[str]) -> None:
        self.transcripts = list(transcripts)
        self.calls = 0

    def transcribe(self, audio: np.ndarray, language: str) -> ASRResult:
        text = self.transcripts[min(self.calls, len(self.transcripts) - 1)]
        self.calls += 1
        return ASRResult(text, 1.0)

    def warmup(self, language: str) -> None:
        return None


class FakeMT:
    name = "fake-mt"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def translate(self, text: str) -> MTResult:
        self.calls.append(text)
        return MTResult(f"EN[{text}]", 1.0)

    def warmup(self) -> None:
        return None


def make_pipeline(asr, mt, **config_kwargs) -> TranslationPipeline:
    pipeline = TranslationPipeline.__new__(TranslationPipeline)
    pipeline.config = AppConfig(**config_kwargs)
    pipeline.asr = asr
    pipeline.mt = mt
    pipeline.segmenter = None  # not exercised; segments are injected directly
    pipeline._last_interim_text = ""
    return pipeline


def segment(utterance_id: int, is_final: bool) -> Segment:
    audio = np.zeros(FRAME_SAMPLES * 30, dtype=np.float32)
    return Segment(
        utterance_id=utterance_id,
        audio=audio,
        is_final=is_final,
        duration_ms=960.0,
        vad_ms=2.0,
    )


def test_final_utterance_is_translated_exactly_once():
    asr, mt = FakeASR(["hola mundo"]), FakeMT()
    pipeline = make_pipeline(asr, mt)

    event = pipeline._handle(segment(1, is_final=True))

    assert event is not None
    assert event.kind is EventKind.FINAL
    assert event.target_text == "EN[hola mundo]"
    assert mt.calls == ["hola mundo"], "one translation per utterance, no re-work"


def test_unchanged_interim_is_suppressed():
    """Whisper repeats provisional text; resending it flickers the display and
    costs a needless MT call."""
    asr, mt = FakeASR(["hola", "hola", "hola mundo"]), FakeMT()
    pipeline = make_pipeline(asr, mt)

    first = pipeline._handle(segment(1, is_final=False))
    duplicate = pipeline._handle(segment(1, is_final=False))
    changed = pipeline._handle(segment(1, is_final=False))

    assert first is not None
    assert duplicate is None, "identical interim should be dropped"
    assert changed is not None
    assert mt.calls == ["hola", "hola mundo"]


def test_interim_translation_can_be_disabled():
    asr, mt = FakeASR(["hola"]), FakeMT()
    pipeline = make_pipeline(asr, mt, translate_interim=False)

    event = pipeline._handle(segment(1, is_final=False))

    assert event is not None
    assert event.target_text == ""
    assert mt.calls == [], "interim translation disabled means no MT work"


def test_empty_transcript_produces_no_event():
    asr, mt = FakeASR([""]), FakeMT()
    pipeline = make_pipeline(asr, mt)

    assert pipeline._handle(segment(1, is_final=True)) is None
    assert mt.calls == []


def test_timings_are_attributed_per_stage():
    asr, mt = FakeASR(["hola"]), FakeMT()
    pipeline = make_pipeline(asr, mt)

    event = pipeline._handle(segment(1, is_final=True))

    assert event.timings.vad_ms == pytest.approx(2.0)
    assert event.timings.asr_ms == pytest.approx(1.0)
    assert event.timings.mt_ms == pytest.approx(1.0)
    assert event.timings.total_ms == pytest.approx(4.0)


def test_event_serialises_for_the_websocket():
    asr, mt = FakeASR(["hola"]), FakeMT()
    pipeline = make_pipeline(asr, mt)

    payload = pipeline._handle(segment(1, is_final=True)).to_json()

    assert payload["kind"] == "final"
    assert payload["source_text"] == "hola"
    assert payload["real_time_factor"] > 0
    assert "total_ms" in payload["timings"]
