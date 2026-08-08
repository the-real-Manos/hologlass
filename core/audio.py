"""Audio sources.

Two implementations behind one interface. The microphone drives the live product;
the file source makes the whole pipeline testable and benchmarkable with no
hardware, no permissions, and identical output every run. Deterministic replay is
what allows the benchmark numbers to mean anything.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from core.config import FRAME_SAMPLES, SAMPLE_RATE

logger = logging.getLogger(__name__)

# Bounds the backlog if the pipeline briefly falls behind. At 16 kHz this is a
# few seconds of audio; beyond that, dropping the oldest frames is better than
# growing without limit and translating something the speaker has forgotten.
MAX_QUEUED_BLOCKS = 256


class AudioSource(Protocol):
    """Yields fixed-size mono float32 frames at SAMPLE_RATE."""

    def frames(self) -> Iterator[np.ndarray]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class InputDevice:
    """A selectable capture device."""

    index: int
    name: str
    host_api: str
    channels: int
    default_samplerate: int
    is_default: bool

    @property
    def label(self) -> str:
        return f"{self.name} ({self.host_api})"


def list_input_devices() -> list[InputDevice]:
    """Enumerate every device that can capture audio.

    Windows exposes the same physical microphone several times, once per host
    API, and they do not behave identically: MME is the most likely to refuse an
    unusual sample rate, WASAPI the most likely to accept it. Showing the host
    API is therefore not clutter, it is the information needed to pick one that
    works.
    """
    import sounddevice as sd

    try:
        default_index = sd.default.device[0]
    except (TypeError, IndexError):
        default_index = None

    devices: list[InputDevice] = []
    for index, info in enumerate(sd.query_devices()):
        if info["max_input_channels"] < 1:
            continue
        try:
            host_api = sd.query_hostapis(info["hostapi"])["name"]
        except Exception:  # noqa: BLE001 - host API lookup is best-effort
            host_api = "unknown"
        devices.append(
            InputDevice(
                index=index,
                # Driver names are frequently padded or truncated.
                name=info["name"].strip(),
                host_api=host_api,
                channels=int(info["max_input_channels"]),
                default_samplerate=int(info["default_samplerate"]),
                is_default=index == default_index,
            )
        )
    return devices


class _StreamingResampler:
    """Linear resampler that keeps its phase across blocks.

    Resampling each block independently would introduce a discontinuity at every
    boundary, which the VAD hears as a click. Carrying the fractional read
    position and the unconsumed tail between calls avoids that.

    Linear interpolation is adequate here: the target is a 16 kHz speech model,
    not audio playback.
    """

    def __init__(self, source_rate: int, target_rate: int) -> None:
        self.ratio = source_rate / target_rate
        self._carry = np.zeros(0, dtype=np.float32)
        self._phase = 0.0

    def __call__(self, block: np.ndarray) -> np.ndarray:
        data = np.concatenate((self._carry, block)) if self._carry.size else block
        if data.size < 2:
            self._carry = data
            return np.zeros(0, dtype=np.float32)

        count = int(np.floor((data.size - 1 - self._phase) / self.ratio)) + 1
        if count <= 0:
            self._carry = data
            return np.zeros(0, dtype=np.float32)

        positions = self._phase + np.arange(count) * self.ratio
        output = np.interp(positions, np.arange(data.size), data).astype(np.float32)

        # Retain everything the next interpolation could still need.
        next_position = positions[-1] + self.ratio
        consumed = min(int(np.floor(next_position)), data.size - 1)
        self._carry = data[consumed:].copy()
        self._phase = next_position - consumed
        return output


class MicrophoneSource:
    """Live capture via sounddevice.

    The callback runs on a realtime audio thread, so it does the minimum: convert
    to mono, resample if the device forced a different rate, record the signal
    level, and hand the block to a queue. Anything heavier risks dropouts.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        device: int | None = None,
        capture: bool = True,
    ) -> None:
        self._sample_rate = sample_rate
        self._device = device
        # Monitor mode reports levels without feeding the pipeline, so the queue
        # is not filled by audio nobody will consume.
        self._capture = capture

        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=MAX_QUEUED_BLOCKS)
        self._stream = None
        self._running = False
        self._resampler: _StreamingResampler | None = None
        self._dropped = 0

        self.device_rate: int | None = None
        self.resampling = False
        self.level_rms = 0.0
        self.level_peak = 0.0

    # -- stream lifecycle -------------------------------------------------

    def _open(self, rate: int) -> None:
        import sounddevice as sd

        self._stream = sd.InputStream(
            samplerate=rate,
            channels=1,
            dtype="float32",
            # Let PortAudio pick. Forcing a block size is another constraint the
            # driver can reject, and frames() re-chunks to 512 anyway.
            blocksize=0,
            device=self._device,
            callback=self._callback,
        )
        self._stream.start()

    def start(self) -> None:
        """Open the stream, falling back to the device's native rate if needed.

        Requesting 16 kHz outright fails on some Windows drivers, MME especially.
        Rather than surfacing that as "no audio", reopen at whatever the device
        does support and resample on the way in.
        """
        if self._running:
            return

        try:
            self._open(self._sample_rate)
            self.device_rate = self._sample_rate
            self.resampling = False
        except Exception as first_error:  # noqa: BLE001 - retried below
            native = self._native_rate()
            logger.warning(
                "device rejected %d Hz (%s); retrying at its native %d Hz",
                self._sample_rate, first_error, native,
            )
            try:
                self._open(native)
            except Exception as second_error:
                raise RuntimeError(
                    f"could not open audio device {self._device}: {second_error}"
                ) from second_error

            self.device_rate = native
            self.resampling = native != self._sample_rate
            if self.resampling:
                self._resampler = _StreamingResampler(native, self._sample_rate)

        self._running = True
        logger.info(
            "microphone started: device=%s rate=%d Hz%s",
            self._device if self._device is not None else "default",
            self.device_rate,
            " (resampling to 16 kHz)" if self.resampling else "",
        )

    def _native_rate(self) -> int:
        import sounddevice as sd

        try:
            info = sd.query_devices(self._device, "input")
            return int(info["default_samplerate"])
        except Exception:  # noqa: BLE001 - fall back to a near-universal rate
            return 44_100

    def _callback(self, indata, _frames, _time, status) -> None:
        if status:
            logger.debug("audio status: %s", status)

        block = indata[:, 0]
        # Level is measured pre-resample so it reflects what the device actually
        # delivered, which is the point of the meter.
        if block.size:
            self.level_peak = float(np.abs(block).max())
            self.level_rms = float(np.sqrt(np.mean(block.astype(np.float64) ** 2)))

        if not self._capture:
            return

        if self._resampler is not None:
            block = self._resampler(block.copy())
            if not block.size:
                return
        else:
            block = block.copy()

        try:
            self._queue.put_nowait(block)
        except queue.Full:
            self._dropped += 1
            if self._dropped % 50 == 1:
                logger.warning("audio queue full, dropping blocks (%d so far)", self._dropped)

    # -- consumption ------------------------------------------------------

    def frames(self) -> Iterator[np.ndarray]:
        """Yield exactly FRAME_SAMPLES-long frames, which the VAD requires."""
        if not self._running:
            self.start()

        pending = np.zeros(0, dtype=np.float32)
        while self._running:
            block = self._queue.get()
            if block is None:
                break

            pending = np.concatenate((pending, block))
            while pending.size >= FRAME_SAMPLES:
                yield pending[:FRAME_SAMPLES]
                pending = pending[FRAME_SAMPLES:]

    def close(self) -> None:
        if not self._running:
            return
        self._running = False
        # Unblocks frames() if it is parked on an empty queue. A queue already at
        # capacity means frames() is behind and will see `_running` anyway.
        with contextlib.suppress(queue.Full):
            self._queue.put_nowait(None)
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        logger.info("microphone stopped")


def _read_wav(path: Path) -> tuple[np.ndarray, int, int]:
    """Read a WAV into float32 samples, returning (samples, channels, rate).

    The stdlib `wave` module handles integer PCM only and raises outright on
    IEEE-float files, which is what scipy writes whenever it is handed a float
    array. Since that is how most recording scripts save audio, refusing those
    files would make the benchmark harness needlessly fussy about its input.

    Handles PCM 8/16/24/32-bit and IEEE float 32/64-bit, including the
    WAVE_FORMAT_EXTENSIBLE wrapper.
    """
    with path.open("rb") as handle:
        header = handle.read(12)
        if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            raise ValueError(f"{path.name}: not a RIFF/WAVE file")

        format_tag = channels = bits = 0
        rate = 0
        raw = b""

        while chunk_header := handle.read(8):
            if len(chunk_header) < 8:
                break
            chunk_id, size = struct.unpack("<4sI", chunk_header)
            body = handle.read(size)
            if size % 2:  # chunks are word-aligned
                handle.read(1)

            if chunk_id == b"fmt " and len(body) >= 16:
                format_tag, channels, rate, _, _, bits = struct.unpack("<HHIIHH", body[:16])
                # EXTENSIBLE stores the real format tag in the subformat GUID.
                if format_tag == 0xFFFE and len(body) >= 26:
                    format_tag = struct.unpack("<H", body[24:26])[0]
            elif chunk_id == b"data":
                raw = body

    if not raw:
        raise ValueError(f"{path.name}: no data chunk")

    if format_tag == 3 and bits == 32:
        samples = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    elif format_tag == 3 and bits == 64:
        samples = np.frombuffer(raw, dtype="<f8").astype(np.float32)
    elif format_tag == 1 and bits == 16:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif format_tag == 1 and bits == 32:
        samples = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif format_tag == 1 and bits == 8:
        # 8-bit PCM is unsigned and centred on 128.
        samples = (np.frombuffer(raw, dtype="<u1").astype(np.float32) - 128.0) / 128.0
    elif format_tag == 1 and bits == 24:
        packed = np.frombuffer(raw, dtype="<u1").reshape(-1, 3)
        widened = np.zeros((len(packed), 4), dtype="<u1")
        widened[:, 1:] = packed  # keep the sign in the top byte
        samples = widened.view("<i4").reshape(-1).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"{path.name}: unsupported WAV format tag {format_tag} at {bits}-bit")

    return samples, max(1, channels), rate


class WavFileSource:
    """Replays a WAV file as frames, for tests and benchmarks.

    Runs as fast as the pipeline can consume by default, so a benchmark is not
    bounded by wall-clock audio duration.
    """

    def __init__(self, path: Path, sample_rate: int = SAMPLE_RATE) -> None:
        self._path = Path(path)
        self._target_rate = sample_rate
        self._audio = self._load()

    @property
    def duration_ms(self) -> float:
        return len(self._audio) / self._target_rate * 1000.0

    def _load(self) -> np.ndarray:
        samples, channels, rate = _read_wav(self._path)
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        if rate != self._target_rate:
            logger.warning("resampling %d Hz -> %d Hz; prefer pre-converted fixtures", rate, self._target_rate)
            samples = _StreamingResampler(rate, self._target_rate)(samples)
        return samples

    def frames(self) -> Iterator[np.ndarray]:
        total = len(self._audio)
        for start in range(0, total - FRAME_SAMPLES + 1, FRAME_SAMPLES):
            yield self._audio[start : start + FRAME_SAMPLES]
        remainder = total % FRAME_SAMPLES
        if remainder:
            yield np.pad(self._audio[-remainder:], (0, FRAME_SAMPLES - remainder))

    def close(self) -> None:
        return None
