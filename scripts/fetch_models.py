"""One-time model setup.

Downloads the Silero VAD graph and converts the Marian translation models to
CTranslate2 int8. This is the only step in the project that touches the network.

    python -m scripts.fetch_models                    # everything
    python -m scripts.fetch_models --language Spanish # one language
    python -m scripts.fetch_models --skip-asr         # VAD + MT only

Requires the dev dependencies: conversion loads the source weights with torch,
which the runtime itself does not need.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import urllib.request

from core.config import LANGUAGES, MODELS_DIR, MT_MODELS_DIR, VAD_MODEL_PATH, ASRConfig

SILERO_VAD_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx"
)

logger = logging.getLogger("fetch_models")


def fetch_vad(force: bool = False) -> None:
    if VAD_MODEL_PATH.exists() and not force:
        logger.info("VAD model already present at %s", VAD_MODEL_PATH)
        return

    VAD_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info("downloading Silero VAD -> %s", VAD_MODEL_PATH)

    temporary = VAD_MODEL_PATH.with_suffix(".onnx.part")
    with urllib.request.urlopen(SILERO_VAD_URL, timeout=60) as response, temporary.open("wb") as out:
        shutil.copyfileobj(response, out)
    temporary.replace(VAD_MODEL_PATH)

    size_kb = VAD_MODEL_PATH.stat().st_size / 1024
    logger.info("VAD model ready (%.0f KB)", size_kb)


def convert_translation_model(language_name: str, force: bool = False) -> None:
    """Convert a Helsinki-NLP Marian checkpoint to CTranslate2 int8."""
    language = LANGUAGES[language_name]
    target = language.ct2_dir

    if target.exists() and not force:
        logger.info("%s already converted at %s", language_name, target)
        return

    from ctranslate2.converters import TransformersConverter
    from transformers import MarianTokenizer

    MT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("converting %s -> %s (int8)", language.hf_model, target)

    converter = TransformersConverter(language.hf_model)
    converter.convert(str(target), quantization="int8", force=force)

    # The tokenizer lives alongside the converted weights so the runtime never
    # needs to reach Hugging Face again.
    MarianTokenizer.from_pretrained(language.hf_model).save_pretrained(str(target))
    logger.info("%s ready", language_name)


def fetch_asr(config: ASRConfig) -> None:
    """Trigger faster-whisper's own download of the pre-converted CT2 weights."""
    from faster_whisper import WhisperModel

    logger.info("fetching Whisper '%s' (%s)", config.model_size, config.compute_type)
    WhisperModel(config.model_size, device=config.device, compute_type=config.compute_type)
    logger.info("Whisper '%s' ready", config.model_size)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--language", choices=sorted(LANGUAGES), action="append",
                        help="limit conversion to this language (repeatable)")
    parser.add_argument("--model-size", default=ASRConfig.model_size, help="Whisper size to prefetch")
    parser.add_argument("--skip-asr", action="store_true", help="do not prefetch Whisper weights")
    parser.add_argument("--force", action="store_true", help="re-download and re-convert")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    fetch_vad(force=args.force)

    for name in args.language or sorted(LANGUAGES):
        convert_translation_model(name, force=args.force)

    if not args.skip_asr:
        fetch_asr(ASRConfig(model_size=args.model_size))

    logger.info("all models ready under %s", MODELS_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
