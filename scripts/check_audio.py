"""Audio and VAD diagnostic.

Answers the question the level meter cannot: the microphone is clearly producing
signal, so why is nothing being transcribed? Signal level and speech probability
are different things, and only the second one gates the pipeline.

Records from a device, prints level and Silero probability as you speak, then
reports what the segmenter would have done with it and recommends thresholds.

    python -m scripts.check_audio                        # 10s from the default device
    python -m scripts.check_audio --device 42 --seconds 15
    python -m scripts.check_audio --save tests/fixtures/es_sample.wav

Saving also produces the fixture the benchmark needs, so one recording serves
both purposes.
"""

from __future__ import annotations

import argparse
import sys
import time
import wave
from pathlib import Path

import numpy as np

from core.audio import MicrophoneSource, list_input_devices
from core.config import FRAME_SAMPLES, SAMPLE_RATE, VAD_MODEL_PATH, SegmenterConfig
from core.vad import SileroVad, UtteranceSegmenter

BAR_WIDTH = 28


def bar(value: float, width: int = BAR_WIDTH) -> str:
    filled = int(round(max(0.0, min(1.0, value)) * width))
    return "#" * filled + "." * (width - filled)


def show_devices() -> None:
    print("Input devices:\n")
    for device in list_input_devices():
        mark = "  <-- default" if device.is_default else ""
        print(f"  [{device.index:3}] {device.label}{mark}")


def record(device: int | None, seconds: float) -> tuple[np.ndarray, list[float]]:
    """Capture audio while printing a live level and VAD readout."""
    vad = SileroVad(VAD_MODEL_PATH)
    source = MicrophoneSource(device=device)
    source.start()

    if source.resampling:
        print(f"note: device opened at {source.device_rate} Hz, resampling to {SAMPLE_RATE} Hz")

    print(f"\nRecording {seconds:.0f}s. Speak normally.\n")
    print(f"{'level':^{BAR_WIDTH}} | {'speech probability':^{BAR_WIDTH}}")
    print("-" * (BAR_WIDTH * 2 + 3))

    collected: list[np.ndarray] = []
    probabilities: list[float] = []
    deadline = time.time() + seconds
    last_print = 0.0

    for frame in source.frames():
        collected.append(frame)
        probability = vad(frame)
        probabilities.append(probability)

        now = time.time()
        if now - last_print > 0.12:
            last_print = now
            peak = float(np.abs(frame).max())
            flag = "  SPEECH" if probability > 0.5 else ""
            print(f"\r{bar(peak * 4)} | {bar(probability)} {probability:4.2f}{flag}   ", end="")
            sys.stdout.flush()

        if now > deadline:
            break

    source.close()
    print("\n")
    return (np.concatenate(collected) if collected else np.zeros(0, dtype=np.float32)), probabilities


def analyse(audio: np.ndarray, probabilities: list[float]) -> None:
    if audio.size == 0:
        print("No audio captured at all. The device did not deliver any frames.")
        return

    peak = float(np.abs(audio).max())
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    peak_db = 20 * np.log10(peak) if peak > 0 else -100.0

    probs = np.array(probabilities)
    above_50 = float((probs > 0.50).mean())
    above_35 = float((probs > 0.35).mean())
    above_20 = float((probs > 0.20).mean())

    print("=" * 60)
    print(f"  duration        {audio.size / SAMPLE_RATE:.1f} s")
    print(f"  peak level      {peak:.4f}  ({peak_db:.0f} dB)")
    print(f"  rms level       {rms:.4f}")
    print()
    print(f"  VAD max         {probs.max():.3f}")
    print(f"  VAD mean        {probs.mean():.3f}")
    print(f"  frames > 0.50   {above_50 * 100:5.1f}%   (current speech threshold)")
    print(f"  frames > 0.35   {above_35 * 100:5.1f}%   (current silence threshold)")
    print(f"  frames > 0.20   {above_20 * 100:5.1f}%")
    print("=" * 60)

    # Replay through the real segmenter so the verdict reflects actual behaviour
    # rather than the raw threshold counts.
    for label, config in (
        ("current defaults", SegmenterConfig()),
        ("lowered thresholds", SegmenterConfig(speech_threshold=0.30, silence_threshold=0.20)),
        ("very permissive", SegmenterConfig(speech_threshold=0.15, silence_threshold=0.10, min_speech_ms=150)),
    ):
        segmenter = UtteranceSegmenter(_Replay(probabilities), config)
        found = []
        for index in range(len(probabilities)):
            frame = audio[index * FRAME_SAMPLES : (index + 1) * FRAME_SAMPLES]
            if frame.size < FRAME_SAMPLES:
                break
            found.extend(segmenter.process_frame(frame))
        tail = segmenter.flush()
        if tail:
            found.append(tail)
        finals = [s for s in found if s.is_final]
        print(f"  {label:<20} -> {len(finals)} utterance(s) would reach the ASR")

    print()
    if probs.max() < 0.20:
        print("VERDICT: Silero sees no speech at all in this recording.")
        print("  Either the signal is far too quiet for the model, or what was")
        print("  captured is not voice. Raise the input gain in Windows Sound")
        print("  settings and re-run before changing any thresholds.")
    elif above_50 < 0.02:
        print("VERDICT: speech is present but rarely crosses the 0.50 threshold.")
        print("  Lowering speech_threshold in core/config.py should fix this.")
    else:
        print("VERDICT: the VAD is firing normally. If nothing was transcribed,")
        print("  the problem is downstream of segmentation.")


class _Replay:
    """Feeds recorded probabilities back through the segmenter."""

    def __init__(self, probabilities: list[float]) -> None:
        self._probabilities = probabilities
        self._index = 0

    def __call__(self, _frame: np.ndarray) -> float:
        value = self._probabilities[min(self._index, len(self._probabilities) - 1)]
        self._index += 1
        return value

    def reset(self) -> None:
        self._index = 0


def save_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.clip(audio, -1.0, 1.0)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes((samples * 32767).astype(np.int16).tobytes())
    print(f"\nsaved -> {path}  ({audio.size / SAMPLE_RATE:.1f}s, 16 kHz mono)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", type=int, help="input device index (see --list)")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--save", type=Path, help="write the recording to a 16 kHz mono WAV")
    parser.add_argument("--list", action="store_true", help="list input devices and exit")
    args = parser.parse_args(argv)

    if args.list:
        show_devices()
        return 0

    audio, probabilities = record(args.device, args.seconds)
    analyse(audio, probabilities)
    if args.save:
        save_wav(args.save, audio)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
