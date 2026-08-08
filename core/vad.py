"""Streaming voice-activity detection and utterance segmentation.

Runs the Silero VAD v5 ONNX graph directly on ONNX Runtime. The `silero-vad` pip
package is deliberately not used: it hard-depends on torch and torchaudio, which
would put roughly 2 GB of PyTorch back into a runtime that otherwise does not
need it.

Segmentation is where end-to-end latency is won or lost. Transcribing on a fixed
timer means paying for ASR on silence and cutting words at arbitrary boundaries;
gating on speech means ASR runs once per utterance, on audio that starts and ends
where a human would put the boundary.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from core.config import FRAME_MS, FRAME_SAMPLES, SAMPLE_RATE, SegmenterConfig

logger = logging.getLogger(__name__)


# Silero v5 expects each frame to arrive with the tail of the previous frame
# prepended, giving the first convolution layer real history instead of zeros.
# At 16 kHz that is 64 samples, so the tensor handed to the graph is 576 wide,
# not 512.
#
# This is not optional and it is not validated by the runtime: the graph declares
# its input as [None, None], so passing a bare 512-sample frame is accepted
# silently and returns a near-zero probability for every input, including
# unmistakable speech. A VAD that never fires and never errors is the result.
VAD_CONTEXT_SAMPLES = 64


class SileroVad:
    """Frame-by-frame speech probability, with carried recurrent state.

    The model is stateful in two separate ways: an LSTM state that the graph
    returns on each call, and the audio context described above. Both must be
    carried between calls, so frames have to be fed in order and `reset()` called
    between independent streams.
    """

    def __init__(self, model_path: Path, sample_rate: int = SAMPLE_RATE) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Silero VAD weights not found at {model_path}. "
                "Run: python -m scripts.fetch_models"
            )

        opts = ort.SessionOptions()
        # One thread is plenty for a 512-sample frame, and it keeps the VAD from
        # competing with ASR for cores.
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 3

        self._session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._sample_rate = sample_rate
        self._input_names = {i.name for i in self._session.get_inputs()}
        # v5 exposes a single packed "state" tensor; v4 used separate h/c inputs.
        self._is_v5 = "state" in self._input_names
        self.reset()

    def reset(self) -> None:
        """Clear recurrent state and audio context. Call between streams."""
        self._context = np.zeros(VAD_CONTEXT_SAMPLES, dtype=np.float32)
        if self._is_v5:
            self._state = np.zeros((2, 1, 128), dtype=np.float32)
        else:
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        """Return the probability that `frame` contains speech.

        `frame` must be exactly FRAME_SAMPLES of float32 mono audio. The context
        samples are prepended here, so callers pass plain 512-sample frames.
        """
        if frame.shape[-1] != FRAME_SAMPLES:
            raise ValueError(f"expected {FRAME_SAMPLES}-sample frame, got {frame.shape[-1]}")

        frame = frame.astype(np.float32, copy=False)
        sr = np.array(self._sample_rate, dtype=np.int64)

        if self._is_v5:
            batch = np.concatenate((self._context, frame)).reshape(1, -1)
            self._context = frame[-VAD_CONTEXT_SAMPLES:].copy()
            out, self._state = self._session.run(
                None, {"input": batch, "state": self._state, "sr": sr}
            )
        else:
            # v4 takes bare frames; it has no separate context input.
            out, self._h, self._c = self._session.run(
                None, {"input": frame.reshape(1, -1), "h": self._h, "c": self._c, "sr": sr}
            )
        return float(out.reshape(-1)[0])


@dataclass
class Segment:
    """A slice of audio the segmenter considers worth transcribing."""

    utterance_id: int
    audio: np.ndarray
    is_final: bool
    duration_ms: float
    vad_ms: float
    """Time spent inside the VAD producing this segment, for latency attribution."""


class UtteranceSegmenter:
    """Turns a frame stream into utterances, with provisional results en route.

    Improvements over a naive threshold gate, each of which fixes a specific
    failure seen in the previous prototype:

    - **Hysteresis.** Separate enter/exit thresholds stop the state flapping on
      frames that sit near the boundary.
    - **Pre-roll.** Silero needs a few frames to become confident, so the frames
      immediately before detection are retained. Without this, plosives at the
      start of an utterance are clipped and the ASR guesses the first word.
    - **Trailing-silence trim.** The silence that proves the utterance ended is
      not sent to the ASR, which would otherwise pay to transcribe it.
    - **Minimum speech duration.** Coughs and door clicks are dropped instead of
      producing hallucinated transcripts.
    - **Maximum duration.** An unbroken monologue still yields output.
    """

    def __init__(
        self,
        vad: SileroVad,
        config: SegmenterConfig | None = None,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self._vad = vad
        self._cfg = config or SegmenterConfig()
        self._sample_rate = sample_rate

        self._pre_roll: deque[np.ndarray] = deque(maxlen=self._frames(self._cfg.pre_roll_ms))
        self._buffer: list[np.ndarray] = []
        self._speaking = False
        self._closing = False
        self._silence_frames = 0
        self._voiced_frames = 0
        self._frames_since_interim = 0
        self._utterance_id = 0
        self._vad_ms = 0.0

        self.last_probability = 0.0
        """Most recent speech probability, surfaced for live diagnostics.

        Signal level and speech probability are independent: a microphone can be
        plainly working while the VAD never fires, and without this the two are
        indistinguishable from outside.
        """

    @staticmethod
    def _frames(ms: float) -> int:
        return max(1, math.ceil(ms / FRAME_MS))

    def reset(self) -> None:
        self._vad.reset()
        self._pre_roll.clear()
        self._close()
        self._vad_ms = 0.0

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    def process_frame(self, frame: np.ndarray) -> list[Segment]:
        """Feed one frame; return any segments it completed or triggered."""
        started = time.perf_counter()
        probability = self._vad(frame)
        self._vad_ms += (time.perf_counter() - started) * 1000.0
        self.last_probability = probability

        if self._speaking:
            return self._handle_speaking(frame, probability)
        if self._closing:
            return self._handle_closing(frame, probability)
        return self._handle_idle(frame, probability)

    def _handle_idle(self, frame: np.ndarray, probability: float) -> list[Segment]:
        if probability < self._cfg.speech_threshold:
            self._pre_roll.append(frame)
            return []

        # Speech onset. Seed the buffer with the retained pre-roll so the start
        # of the word survives.
        self._utterance_id += 1
        self._buffer = [*self._pre_roll, frame]
        self._pre_roll.clear()
        self._speaking = True
        self._silence_frames = 0
        self._voiced_frames = 1
        self._frames_since_interim = 0
        logger.debug("speech onset, utterance %d", self._utterance_id)
        return []

    def _handle_speaking(self, frame: np.ndarray, probability: float) -> list[Segment]:
        self._buffer.append(frame)
        self._frames_since_interim += 1

        if probability < self._cfg.silence_threshold:
            self._silence_frames += 1
        else:
            self._silence_frames = 0
            self._voiced_frames += 1

        if self._silence_frames >= self._frames(self._cfg.min_silence_ms):
            final = self._finalise()
            return [final] if final else []

        if self._buffered_ms() >= self._cfg.max_utterance_ms:
            logger.debug("utterance %d hit max duration, forcing flush", self._utterance_id)
            # No continuation grace here: the cap exists to bound buffer growth,
            # and staying open would let it grow anyway.
            final = self._finalise(trim_silence=False, allow_continuation=False)
            return [final] if final else []

        if self._frames_since_interim >= self._frames(self._cfg.interim_every_ms):
            self._frames_since_interim = 0
            return [self._emit(self._buffer, is_final=False)]

        return []

    def _handle_closing(self, frame: np.ndarray, probability: float) -> list[Segment]:
        """A result has been emitted, but the utterance may still continue.

        Holding the utterance open briefly is what lets `min_silence_ms` be short
        without splitting sentences at every breath.
        """
        self._buffer.append(frame)

        if probability >= self._cfg.speech_threshold:
            logger.debug("utterance %d continued within grace window", self._utterance_id)
            self._speaking = True
            self._closing = False
            self._silence_frames = 0
            self._voiced_frames += 1
            return []

        self._silence_frames += 1
        settled = self._frames(self._cfg.min_silence_ms + self._cfg.continuation_grace_ms)
        if self._silence_frames >= settled:
            self._close()
        return []

    def flush(self) -> Segment | None:
        """Close any in-progress utterance. Call when the stream ends."""
        if self._closing:
            # A result was already emitted for this utterance; nothing to add.
            self._close()
            return None
        if not self._speaking:
            return None
        return self._finalise(trim_silence=False, allow_continuation=False)

    def _finalise(self, trim_silence: bool = True, allow_continuation: bool = True) -> Segment | None:
        frames = self._buffer
        if trim_silence and self._silence_frames:
            # Keep a short tail so the final consonant is not cut, drop the rest.
            keep = self._frames(150)
            drop = max(0, self._silence_frames - keep)
            if drop:
                frames = frames[:-drop]

        # Measured from frames the VAD actually scored as voiced. Deriving it
        # from buffer length instead would count the pre-roll and the retained
        # silence tail as speech, which lets short noise blips through.
        voiced_ms = self._voiced_frames * FRAME_MS

        if voiced_ms < self._cfg.min_speech_ms:
            logger.debug("dropping utterance %d: only %.0f ms of speech", self._utterance_id, voiced_ms)
            self._close()
            self._vad_ms = 0.0
            return None

        self._speaking = False
        self._frames_since_interim = 0

        if allow_continuation:
            # Keep the buffer and the utterance id: if speech resumes, the next
            # result supersedes this one rather than starting a new caption.
            self._closing = True
        else:
            self._close()

        return self._emit(frames, is_final=True)

    def _close(self) -> None:
        """Retire the current utterance for good."""
        self._speaking = False
        self._closing = False
        self._buffer = []
        self._silence_frames = 0
        self._voiced_frames = 0
        self._frames_since_interim = 0

    def _emit(self, frames: list[np.ndarray], is_final: bool) -> Segment:
        audio = np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32)
        segment = Segment(
            utterance_id=self._utterance_id,
            audio=audio,
            is_final=is_final,
            duration_ms=len(audio) / self._sample_rate * 1000.0,
            vad_ms=self._vad_ms,
        )
        self._vad_ms = 0.0
        return segment

    def _buffered_ms(self) -> float:
        return len(self._buffer) * FRAME_MS
