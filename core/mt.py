"""Machine translation backends.

A short sentence costs tens of milliseconds through a quantized Marian model,
measured at 71ms for Spanish and 48ms for French on a Ryzen 5 5600X. That is
comfortably below the ASR cost and below the threshold of perception, so
translation is never the bottleneck worth optimising.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from typing import Protocol, runtime_checkable

from core.config import LanguagePair, MTConfig

logger = logging.getLogger(__name__)


class MTResult:
    __slots__ = ("text", "elapsed_ms", "cached")

    def __init__(self, text: str, elapsed_ms: float, cached: bool = False) -> None:
        self.text = text
        self.elapsed_ms = elapsed_ms
        self.cached = cached


@runtime_checkable
class MTBackend(Protocol):
    name: str

    def translate(self, text: str) -> MTResult: ...

    def warmup(self) -> None: ...


class CT2MarianMT:
    """Helsinki-NLP Marian via CTranslate2, quantized to int8.

    Same weights as the `transformers` pipeline the prototype used, executed on a
    runtime built for inference. The tokenizer still comes from `transformers`,
    but MarianTokenizer is pure Python over sentencepiece and pulls in no torch.
    """

    def __init__(self, language: LanguagePair, config: MTConfig | None = None) -> None:
        import ctranslate2
        from transformers import MarianTokenizer

        self._cfg = config or MTConfig()
        self._language = language
        model_dir = language.ct2_dir

        if not model_dir.exists():
            raise FileNotFoundError(
                f"Converted translation model not found at {model_dir}. "
                f"Run: python -m scripts.fetch_models --language {language.name}"
            )

        self.name = f"ct2-marian/{language.code}-en/{self._cfg.compute_type}"
        logger.info("loading %s", self.name)
        started = time.perf_counter()

        self._translator = ctranslate2.Translator(
            str(model_dir),
            device="cpu",
            compute_type=self._cfg.compute_type,
            inter_threads=self._cfg.inter_threads,
            intra_threads=self._cfg.intra_threads,
        )
        self._tokenizer = MarianTokenizer.from_pretrained(str(model_dir))
        self._cache: OrderedDict[str, str] = OrderedDict()
        logger.info("loaded in %.1f s", time.perf_counter() - started)

    def translate(self, text: str) -> MTResult:
        text = text.strip()
        if not text:
            return MTResult("", 0.0)

        cached = self._cache.get(text)
        if cached is not None:
            self._cache.move_to_end(text)
            return MTResult(cached, 0.0, cached=True)

        started = time.perf_counter()
        try:
            tokens = self._tokenizer.convert_ids_to_tokens(self._tokenizer.encode(text))
            results = self._translator.translate_batch(
                [tokens],
                beam_size=self._cfg.beam_size,
                max_decoding_length=self._cfg.max_decoding_length,
            )
            hypothesis = results[0].hypotheses[0]
            output = self._tokenizer.decode(
                self._tokenizer.convert_tokens_to_ids(hypothesis),
                skip_special_tokens=True,
            )
        except Exception:
            # A translation failure must not take down the audio thread; the
            # transcript is still useful on its own.
            logger.exception("translation failed for %r", text[:80])
            return MTResult("", (time.perf_counter() - started) * 1000.0)

        self._remember(text, output)
        return MTResult(output, (time.perf_counter() - started) * 1000.0)

    def _remember(self, source: str, target: str) -> None:
        self._cache[source] = target
        while len(self._cache) > self._cfg.cache_size:
            self._cache.popitem(last=False)

    def warmup(self) -> None:
        self.translate("hola")
        self._cache.clear()
        logger.info("%s warm", self.name)


def build_mt(language: LanguagePair, config: MTConfig | None = None) -> MTBackend:
    return CT2MarianMT(language, config)
