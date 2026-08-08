"""Latency and accuracy benchmark.

Replays recorded audio through the real pipeline and reports where the time goes.
Because the file source is deterministic, two runs of the same configuration are
directly comparable, which is what makes a claim like "3x faster" mean anything.

    python -m bench.run --audio tests/fixtures/es_sample.wav
    python -m bench.run --audio sample.wav --model-size tiny base small
    python -m bench.run --audio sample.wav --reference sample.es.txt --target sample.en.txt

Place a reference transcript alongside the audio to get WER, and a reference
translation to get BLEU. Without them the run reports latency only.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from core.audio import WavFileSource
from core.config import LANGUAGES, REPO_ROOT, AppConfig, ASRConfig
from core.pipeline import TranslationPipeline
from core.types import EventKind

logger = logging.getLogger("bench")
RESULTS_DIR = REPO_ROOT / "bench" / "results"


@dataclass
class RunResult:
    """One configuration measured against one audio file."""

    model_size: str
    compute_type: str
    device: str
    cpu_threads: int
    audio_file: str
    audio_duration_s: float
    endpoint_hold_ms: int = 0
    """Silence the segmenter waits for before emitting.

    Not compute, but the user waits for it just the same, so a benchmark that
    reports only processing time understates response latency by this much."""

    utterances: int = 0
    wall_clock_s: float = 0.0
    load_s: float = 0.0

    asr_ms: list[float] = field(default_factory=list)
    mt_ms: list[float] = field(default_factory=list)
    vad_ms: list[float] = field(default_factory=list)

    transcript: str = ""
    translation: str = ""
    wer: float | None = None
    bleu: float | None = None

    @property
    def real_time_factor(self) -> float:
        """Wall-clock processing time over audio duration. Under 1.0 keeps up."""
        if self.audio_duration_s <= 0:
            return 0.0
        return self.wall_clock_s / self.audio_duration_s

    def percentile(self, values: list[float], fraction: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
        return ordered[index]

    @property
    def response_ms(self) -> float:
        """What the speaker actually waits: endpoint hold, then ASR, then MT.

        This is the headline latency figure. Processing time alone omits the
        silence threshold, which is often the largest single term.
        """
        asr = statistics.median(self.asr_ms) if self.asr_ms else 0.0
        mt = statistics.median(self.mt_ms) if self.mt_ms else 0.0
        return self.endpoint_hold_ms + asr + mt

    def summary(self) -> dict[str, float | str | int]:
        return {
            "model": self.model_size,
            "compute": self.compute_type,
            "device": self.device,
            "threads": self.cpu_threads or "auto",
            "utterances": self.utterances,
            "response_p50": round(self.response_ms, 1),
            "hold": self.endpoint_hold_ms,
            "rtf": round(self.real_time_factor, 3),
            "asr_p50": round(statistics.median(self.asr_ms), 1) if self.asr_ms else 0.0,
            "asr_p95": round(self.percentile(self.asr_ms, 0.95), 1),
            "mt_p50": round(statistics.median(self.mt_ms), 1) if self.mt_ms else 0.0,
            "vad_total": round(sum(self.vad_ms), 1),
            "load_s": round(self.load_s, 1),
            "wer": round(self.wer, 3) if self.wer is not None else "—",
            "bleu": round(self.bleu, 1) if self.bleu is not None else "—",
        }


def measure(
    audio_path: Path,
    language: str,
    asr_config: ASRConfig,
    translate_interim: bool,
    endpoint_hold_ms: int | None = None,
) -> RunResult:
    source = WavFileSource(audio_path)
    config = AppConfig(language=language, translate_interim=translate_interim)
    config.asr = asr_config
    if endpoint_hold_ms is not None:
        config.segmenter.min_silence_ms = endpoint_hold_ms

    load_started = time.perf_counter()
    pipeline = TranslationPipeline(config=config)
    pipeline.warmup()
    load_s = time.perf_counter() - load_started

    result = RunResult(
        model_size=asr_config.model_size,
        compute_type=asr_config.compute_type,
        device=asr_config.device,
        cpu_threads=asr_config.cpu_threads,
        audio_file=audio_path.name,
        audio_duration_s=source.duration_ms / 1000.0,
        endpoint_hold_ms=config.segmenter.min_silence_ms,
        load_s=load_s,
    )

    source_parts: list[str] = []
    target_parts: list[str] = []

    started = time.perf_counter()
    for event in pipeline.run(source):
        result.vad_ms.append(event.timings.vad_ms)
        if event.kind is not EventKind.FINAL:
            continue
        result.utterances += 1
        result.asr_ms.append(event.timings.asr_ms)
        if event.timings.mt_ms:
            result.mt_ms.append(event.timings.mt_ms)
        source_parts.append(event.source_text)
        target_parts.append(event.target_text)
    result.wall_clock_s = time.perf_counter() - started

    result.transcript = " ".join(source_parts)
    result.translation = " ".join(target_parts)
    return result


def score(result: RunResult, reference: Path | None, target_reference: Path | None) -> None:
    if reference and reference.exists():
        import jiwer

        result.wer = jiwer.wer(reference.read_text(encoding="utf-8").strip(), result.transcript)

    if target_reference and target_reference.exists():
        import sacrebleu

        result.bleu = sacrebleu.sentence_bleu(
            result.translation, [target_reference.read_text(encoding="utf-8").strip()]
        ).score


def render_table(results: list[RunResult]) -> str:
    rows = [r.summary() for r in results]
    if not rows:
        return "(no results)"

    headers = list(rows[0])
    widths = {h: max(len(h), *(len(str(row[h])) for row in rows)) for h in headers}

    lines = [
        " | ".join(h.ljust(widths[h]) for h in headers),
        "-|-".join("-" * widths[h] for h in headers),
    ]
    lines += [" | ".join(str(row[h]).ljust(widths[h]) for h in headers) for row in rows]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio", required=True, type=Path, nargs="+", help="WAV files to replay")
    parser.add_argument("--language", default="Spanish", choices=sorted(LANGUAGES))
    parser.add_argument("--model-size", nargs="+", default=["base"], help="Whisper sizes to compare")
    parser.add_argument("--compute-type", nargs="+", default=["int8"], help="e.g. int8 float32")
    parser.add_argument("--threads", nargs="+", type=int, default=[0], help="CPU thread counts (0 = auto)")
    parser.add_argument("--hold", nargs="+", type=int, help="endpoint hold values in ms to compare")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--translate-interim", action="store_true")
    parser.add_argument("--reference", type=Path, help="reference transcript for WER")
    parser.add_argument("--target", type=Path, help="reference translation for BLEU")
    parser.add_argument("--save", action="store_true", help="write JSON to bench/results/")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    results: list[RunResult] = []
    for audio in args.audio:
        if not audio.exists():
            parser.error(f"audio file not found: {audio}")
        for model_size in args.model_size:
            for compute_type in args.compute_type:
                for threads in args.threads:
                    for hold in args.hold or [None]:
                        logger.warning(
                            "running %s/%s/%d threads/hold=%s on %s",
                            model_size, compute_type, threads, hold or "default", audio.name,
                        )
                        result = measure(
                            audio,
                            args.language,
                            ASRConfig(
                                model_size=model_size,
                                compute_type=compute_type,
                                device=args.device,
                                cpu_threads=threads,
                            ),
                            args.translate_interim,
                            endpoint_hold_ms=hold,
                        )
                        score(result, args.reference, args.target)
                        results.append(result)

    print()
    print(render_table(results))
    print()
    for result in results:
        print(f"[{result.model_size}/{result.compute_type}] {result.transcript}")
        print(f"[{result.model_size}/{result.compute_type}] -> {result.translation}")

    if args.save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        path = RESULTS_DIR / f"bench-{stamp}.json"
        path.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")
        print(f"\nsaved -> {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
