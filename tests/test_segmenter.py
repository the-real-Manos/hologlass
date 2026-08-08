"""Segmenter behaviour, tested against a stub VAD.

The segmenter is the component most likely to be quietly wrong, and its bugs
show up as clipped words or missing utterances rather than exceptions. Driving it
with a scripted probability sequence makes each rule verifiable on its own, with
no model, no audio hardware, and no run-to-run variation.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.config import FRAME_MS, FRAME_SAMPLES, SegmenterConfig
from core.vad import Segment, UtteranceSegmenter


class ScriptedVad:
    """Returns a predetermined probability per frame.

    Stands in for SileroVad, whose interface is a single call returning a float.
    """

    def __init__(self, probabilities: list[float]) -> None:
        self._probabilities = probabilities
        self._index = 0

    def __call__(self, frame: np.ndarray) -> float:
        value = self._probabilities[min(self._index, len(self._probabilities) - 1)]
        self._index += 1
        return value

    def reset(self) -> None:
        self._index = 0


def frames_for(ms: float) -> int:
    return int(round(ms / FRAME_MS))


def silence(ms: float) -> list[float]:
    return [0.02] * frames_for(ms)


def speech(ms: float) -> list[float]:
    return [0.95] * frames_for(ms)


def run(probabilities: list[float], config: SegmenterConfig) -> list[Segment]:
    segmenter = UtteranceSegmenter(ScriptedVad(probabilities), config)
    frame = np.zeros(FRAME_SAMPLES, dtype=np.float32)
    produced: list[Segment] = []
    for _ in probabilities:
        produced.extend(segmenter.process_frame(frame))
    tail = segmenter.flush()
    if tail is not None:
        produced.append(tail)
    return produced


@pytest.fixture
def config() -> SegmenterConfig:
    # Interim generation off by default so tests assert on finals only.
    return SegmenterConfig(interim_every_ms=100_000)


def test_emits_one_final_per_utterance(config):
    segments = run(silence(300) + speech(1500) + silence(1000), config)

    assert len(segments) == 1
    assert segments[0].is_final
    assert segments[0].utterance_id == 1


def test_two_utterances_get_distinct_ids(config):
    probabilities = (
        silence(300) + speech(1000) + silence(1000) + speech(1000) + silence(1000)
    )
    segments = run(probabilities, config)

    assert [s.utterance_id for s in segments] == [1, 2]
    assert all(s.is_final for s in segments)


def test_short_blip_is_discarded(config):
    """A cough is shorter than min_speech_ms and must not reach the ASR."""
    segments = run(silence(300) + speech(120) + silence(1000), config)

    assert segments == []


def test_pre_roll_is_prepended(config):
    """Audio from before detection must survive, or onsets get clipped."""
    config = SegmenterConfig(interim_every_ms=100_000, pre_roll_ms=300)
    segments = run(silence(600) + speech(1000) + silence(1000), config)

    assert len(segments) == 1
    # Roughly speech + pre-roll + the retained silence tail, not speech alone.
    assert segments[0].duration_ms > 1200


def test_hysteresis_does_not_split_on_a_dip(config):
    """A single marginal frame mid-word must not end the utterance."""
    probabilities = (
        silence(300) + speech(600) + [0.40] * 3 + speech(600) + silence(1000)
    )
    segments = run(probabilities, config)

    assert len(segments) == 1, "dip between thresholds should not close the utterance"


def test_max_duration_forces_a_flush():
    config = SegmenterConfig(interim_every_ms=100_000, max_utterance_ms=2000)
    segments = run(silence(300) + speech(5000), config)

    assert segments, "continuous speech must still produce output"
    assert segments[0].duration_ms <= 2200


def test_interims_precede_the_final():
    config = SegmenterConfig(interim_every_ms=400)
    segments = run(silence(300) + speech(1600) + silence(1000), config)

    assert [s.is_final for s in segments][-1] is True
    assert any(not s.is_final for s in segments), "expected provisional results"
    assert len({s.utterance_id for s in segments}) == 1


def test_trailing_silence_is_trimmed(config):
    """The silence proving the utterance ended should not be sent to the ASR."""
    short_tail = run(silence(300) + speech(1000) + silence(700), config)
    long_tail = run(silence(300) + speech(1000) + silence(3000), config)

    assert short_tail and long_tail
    assert long_tail[0].duration_ms == pytest.approx(short_tail[0].duration_ms, abs=FRAME_MS * 2)


def test_breath_pause_amends_rather_than_splits(config):
    """A pause longer than min_silence_ms but inside the grace window.

    The point of the two-stage endpoint: the first result is emitted quickly,
    and speech resuming corrects it in place instead of producing a second
    caption for the same sentence.
    """
    config = SegmenterConfig(interim_every_ms=100_000, min_silence_ms=350, continuation_grace_ms=250)
    probabilities = silence(300) + speech(800) + silence(450) + speech(800) + silence(1200)

    segments = run(probabilities, config)

    assert len({s.utterance_id for s in segments}) == 1, "a breath must not start a new utterance"
    assert len(segments) == 2, "expected a fast first result then an amended one"
    assert segments[1].duration_ms > segments[0].duration_ms, "the amendment should cover more audio"


def test_pause_beyond_grace_starts_a_new_utterance(config):
    config = SegmenterConfig(interim_every_ms=100_000, min_silence_ms=350, continuation_grace_ms=250)
    probabilities = silence(300) + speech(800) + silence(1500) + speech(800) + silence(1200)

    segments = run(probabilities, config)

    assert [s.utterance_id for s in segments] == [1, 2]


def test_result_is_emitted_at_min_silence_not_after_the_grace_window(config):
    """The grace window must not delay the first result, only allow amending it."""
    config = SegmenterConfig(interim_every_ms=100_000, min_silence_ms=350, continuation_grace_ms=250)
    segmenter = UtteranceSegmenter(
        ScriptedVad(silence(300) + speech(800) + silence(2000)), config
    )
    frame = np.zeros(FRAME_SAMPLES, dtype=np.float32)

    emitted_at = None
    for index in range(frames_for(300 + 800 + 2000)):
        if segmenter.process_frame(frame):
            emitted_at = index
            break

    assert emitted_at is not None
    elapsed_silence = (emitted_at * FRAME_MS) - (300 + 800)
    assert elapsed_silence < 350 + FRAME_MS * 2, "result waited for the grace window"


def test_reset_clears_state(config):
    segmenter = UtteranceSegmenter(ScriptedVad(speech(2000)), config)
    frame = np.zeros(FRAME_SAMPLES, dtype=np.float32)
    for _ in range(20):
        segmenter.process_frame(frame)

    assert segmenter.is_speaking
    segmenter.reset()
    assert not segmenter.is_speaking
    assert segmenter.flush() is None
