"""Tests against the real Silero graph.

These exist because of a bug that produced no error at all. The ONNX graph
declares its audio input as [None, None], so feeding it a bare 512-sample frame
instead of the required 512 + 64 samples of context is accepted silently and
returns a near-zero probability for every input. The pipeline ran for thirty
seconds on clear speech and reported nothing wrong.

Every other test in the suite substitutes the VAD, which is what let this
survive. A stub cannot catch a broken contract with a real model, so these tests
deliberately use the genuine weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.config import FRAME_SAMPLES, SAMPLE_RATE, VAD_MODEL_PATH
from core.vad import VAD_CONTEXT_SAMPLES, SileroVad

pytestmark = pytest.mark.skipif(
    not VAD_MODEL_PATH.exists(),
    reason="VAD weights not fetched; run python -m scripts.fetch_models",
)


def voice_like(seconds: float = 2.0) -> np.ndarray:
    """A harmonic stack with a syllable-rate envelope.

    Not real speech, but structured enough that a working VAD scores it far above
    silence, and crucially distinguishable from the near-zero the broken code
    returned for everything.
    """
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    signal = sum(np.sin(2 * np.pi * f * t) / (i + 1) for i, f in enumerate([200, 400, 600, 800, 1000]))
    signal *= 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)
    return (signal / np.abs(signal).max() * 0.4).astype(np.float32)


def probabilities(audio: np.ndarray) -> np.ndarray:
    vad = SileroVad(VAD_MODEL_PATH)
    values = [
        vad(audio[start : start + FRAME_SAMPLES])
        for start in range(0, len(audio) - FRAME_SAMPLES + 1, FRAME_SAMPLES)
    ]
    return np.array(values)


def test_context_samples_are_prepended():
    """The graph must receive 576 samples per 512-sample frame."""
    captured = {}
    vad = SileroVad(VAD_MODEL_PATH)
    original = vad._session.run

    def spy(outputs, inputs):
        captured["width"] = inputs["input"].shape[-1]
        return original(outputs, inputs)

    vad._session.run = spy
    vad(np.zeros(FRAME_SAMPLES, dtype=np.float32))

    assert captured["width"] == FRAME_SAMPLES + VAD_CONTEXT_SAMPLES


def test_structured_audio_scores_far_above_silence():
    """The regression itself: everything scored ~0.0005 before the fix."""
    speech = probabilities(voice_like())
    silence = probabilities(np.zeros(SAMPLE_RATE * 2, dtype=np.float32))

    assert silence.max() < 0.05, "silence should not register as speech"
    assert speech.max() > 0.20, (
        f"structured audio peaked at {speech.max():.4f}; the model is receiving "
        "malformed input and is not actually evaluating anything"
    )
    assert speech.max() > silence.max() * 10


def test_context_carries_between_frames():
    """Second frame's context must be the first frame's tail, not zeros."""
    vad = SileroVad(VAD_MODEL_PATH)
    frame = np.linspace(-0.5, 0.5, FRAME_SAMPLES, dtype=np.float32)

    vad(frame)

    assert np.allclose(vad._context, frame[-VAD_CONTEXT_SAMPLES:])


def test_reset_clears_context_and_state():
    vad = SileroVad(VAD_MODEL_PATH)
    vad(np.full(FRAME_SAMPLES, 0.3, dtype=np.float32))
    assert vad._context.any()

    vad.reset()

    assert not vad._context.any()
    assert not vad._state.any()


def test_wrong_frame_size_is_rejected():
    """The graph would accept it silently; we must not."""
    vad = SileroVad(VAD_MODEL_PATH)

    with pytest.raises(ValueError, match="512-sample frame"):
        vad(np.zeros(256, dtype=np.float32))
