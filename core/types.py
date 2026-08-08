"""Event and timing types shared across the pipeline, server, and benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class EventKind(str, Enum):
    """Whether a result may still change.

    INTERIM results are provisional and will be superseded; a client should
    replace the current line rather than append. FINAL results are settled.
    """

    INTERIM = "interim"
    FINAL = "final"


@dataclass
class StageTimings:
    """Per-stage wall-clock cost, in milliseconds.

    Recorded on every event so latency can be attributed to a stage rather than
    guessed at. The benchmark harness aggregates these directly.
    """

    vad_ms: float = 0.0
    asr_ms: float = 0.0
    mt_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.vad_ms + self.asr_ms + self.mt_ms


@dataclass
class TranslationEvent:
    """One transcription/translation result emitted by the pipeline."""

    utterance_id: int
    kind: EventKind
    source_text: str
    target_text: str
    audio_duration_ms: float
    timings: StageTimings = field(default_factory=StageTimings)

    @property
    def real_time_factor(self) -> float:
        """Processing time divided by audio duration.

        Below 1.0 means the pipeline is faster than real time and can keep up
        with continuous speech. This is the headline number for the benchmark.
        """
        if self.audio_duration_ms <= 0:
            return 0.0
        return self.timings.total_ms / self.audio_duration_ms

    def to_json(self) -> dict[str, Any]:
        """Serialise for the WebSocket transport."""
        payload = asdict(self)
        payload["kind"] = self.kind.value
        payload["timings"]["total_ms"] = round(self.timings.total_ms, 1)
        payload["real_time_factor"] = round(self.real_time_factor, 3)
        return payload
