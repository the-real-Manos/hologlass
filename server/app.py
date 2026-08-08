"""FastAPI application: serves the HUD and streams results over a WebSocket.

Replaces the Streamlit prototype, which ran its capture loop inside a script
rerun. That blocked the server thread, could not serve a second client, and
repainted the entire widget tree on a timer.

Here the pipeline runs on a worker thread and pushes events into the event loop
as they occur, so the browser is updated when there is something to say rather
than ten times a second regardless.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from core.asr import ASRBackend, build_asr
from core.audio import MicrophoneSource, list_input_devices
from core.config import LANGUAGES, AppConfig
from core.mt import MTBackend, build_mt
from core.pipeline import TranslationPipeline

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class ModelRegistry:
    """Caches loaded models across sessions.

    Whisper is language-agnostic so one instance serves everything. Translation
    models are per-language and cached individually, which makes switching
    language cost one model load the first time and nothing after that.
    """

    def __init__(self) -> None:
        self._asr: ASRBackend | None = None
        self._mt: dict[str, MTBackend] = {}
        self._lock = threading.Lock()

    def get(self, config: AppConfig) -> tuple[ASRBackend, MTBackend]:
        with self._lock:
            if self._asr is None:
                self._asr = build_asr(config.asr)
                self._asr.warmup(config.language_pair.code)

            language = config.language_pair
            if language.name not in self._mt:
                backend = build_mt(language, config.mt)
                backend.warmup()
                self._mt[language.name] = backend

            return self._asr, self._mt[language.name]


LEVEL_INTERVAL_S = 0.1


class Session:
    """One browser connection's capture-and-translate run."""

    def __init__(self, websocket: WebSocket, registry: ModelRegistry) -> None:
        self._websocket = websocket
        self._registry = registry
        self._thread: threading.Thread | None = None
        self._source: MicrophoneSource | None = None
        self._pipeline: TranslationPipeline | None = None
        self._level_task: asyncio.Task | None = None
        self._stopping = threading.Event()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    async def monitor(self, device: int | None) -> None:
        """Open the device and report levels only.

        Diagnosing "is my microphone working" should not require loading two
        models and waiting for a full transcription. This opens the stream, skips
        the pipeline entirely, and streams the signal level so the answer is
        visible within a second.
        """
        await self.stop()

        self._source = MicrophoneSource(device=device, capture=False)
        try:
            await asyncio.get_running_loop().run_in_executor(None, self._source.start)
        except Exception as exc:  # noqa: BLE001 - reported to the client
            self._source = None
            await self._send({"type": "error", "message": str(exc)})
            return

        self._start_level_stream()
        await self._send({
            "type": "status",
            "state": "monitoring",
            **self._device_status(),
        })

    async def start(self, language: str, device: int | None) -> None:
        await self.stop()

        config = AppConfig(language=language)
        await self._send({"type": "status", "state": "loading", "language": language})

        loop = asyncio.get_running_loop()
        # Model loading is blocking and can take seconds on a cold start, so it
        # must not run on the event loop.
        asr, mt = await loop.run_in_executor(None, self._registry.get, config)
        pipeline = self._pipeline = TranslationPipeline(config=config, asr=asr, mt=mt)

        self._source = MicrophoneSource(device=device)
        try:
            await loop.run_in_executor(None, self._source.start)
        except Exception as exc:  # noqa: BLE001 - reported to the client
            self._source = None
            await self._send({"type": "error", "message": str(exc)})
            return

        self._stopping.clear()
        self._thread = threading.Thread(
            target=self._worker, args=(pipeline, self._source, loop), daemon=True
        )
        self._thread.start()
        self._start_level_stream()

        await self._send({
            "type": "status",
            "state": "listening",
            "language": language,
            "asr": asr.name,
            "mt": mt.name,
            **self._device_status(),
        })

    def _device_status(self) -> dict[str, Any]:
        source = self._source
        if source is None:
            return {}
        return {
            "device_rate": source.device_rate,
            "resampling": source.resampling,
        }

    def _start_level_stream(self) -> None:
        self._level_task = asyncio.create_task(self._stream_levels())

    async def _stream_levels(self) -> None:
        """Push the input level to the client so silence is visibly diagnosable."""
        try:
            while self._source is not None:
                payload = {
                    "type": "level",
                    "rms": round(self._source.level_rms, 5),
                    "peak": round(self._source.level_peak, 5),
                }
                # Speech probability is what actually gates the pipeline, and it
                # is independent of level. Reporting both makes "the mic works
                # but nothing transcribes" a visible state rather than a mystery.
                if self._pipeline is not None and self._pipeline.segmenter is not None:
                    payload["vad"] = round(self._pipeline.segmenter.last_probability, 3)
                    payload["speaking"] = self._pipeline.segmenter.is_speaking
                await self._send(payload)
                await asyncio.sleep(LEVEL_INTERVAL_S)
        except asyncio.CancelledError:
            raise

    def _worker(self, pipeline: TranslationPipeline, source: MicrophoneSource, loop) -> None:
        """Drain the pipeline on a worker thread, forwarding events to the loop."""
        try:
            for event in pipeline.run(source):
                if self._stopping.is_set():
                    break
                self._dispatch(loop, {"type": "translation", **event.to_json()})
        except Exception as exc:  # noqa: BLE001 - surfaced to the client instead
            logger.exception("pipeline failed")
            self._dispatch(loop, {"type": "error", "message": str(exc)})
        finally:
            self._dispatch(loop, {"type": "status", "state": "stopped"})

    def _dispatch(self, loop, payload: dict[str, Any]) -> None:
        asyncio.run_coroutine_threadsafe(self._send(payload), loop)

    async def _send(self, payload: dict[str, Any]) -> None:
        # A client that has already gone away is not an error worth propagating
        # back into the worker thread.
        with contextlib.suppress(WebSocketDisconnect, RuntimeError):
            await self._websocket.send_json(payload)

    async def stop(self) -> None:
        self._stopping.set()

        if self._level_task is not None:
            self._level_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._level_task
            self._level_task = None

        if self._source is not None:
            # Closing the source ends the frame generator, which unwinds the
            # pipeline and lets the worker thread exit cleanly.
            self._source.close()
            self._source = None
        self._pipeline = None
        if self._thread is not None:
            await asyncio.get_running_loop().run_in_executor(None, self._thread.join, 3.0)
            self._thread = None


def create_app() -> FastAPI:
    app = FastAPI(title="Hologlass", docs_url=None, redoc_url=None)
    registry = ModelRegistry()

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/api/languages")
    async def languages() -> dict[str, list[str]]:
        return {"languages": sorted(LANGUAGES)}

    @app.get("/api/devices")
    async def devices() -> dict[str, list[dict[str, Any]]]:
        return {
            "devices": [
                {
                    "index": device.index,
                    "label": device.label,
                    "host_api": device.host_api,
                    "default_samplerate": device.default_samplerate,
                    "is_default": device.is_default,
                }
                for device in list_input_devices()
            ]
        }

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        session = Session(websocket, registry)
        await websocket.send_json({
            "type": "ready",
            "languages": sorted(LANGUAGES),
            "devices": [
                {"index": d.index, "label": d.label, "is_default": d.is_default}
                for d in list_input_devices()
            ],
        })

        def requested_device(message: dict[str, Any]) -> int | None:
            """None means "system default", which is a valid choice."""
            value = message.get("device")
            return None if value in (None, "", "default") else int(value)

        try:
            while True:
                message = await websocket.receive_json()
                action = message.get("action")

                if action == "start":
                    language = message.get("language", AppConfig().language)
                    if language not in LANGUAGES:
                        await websocket.send_json(
                            {"type": "error", "message": f"unknown language: {language}"}
                        )
                        continue
                    await session.start(language, requested_device(message))
                elif action == "monitor":
                    await session.monitor(requested_device(message))
                elif action == "stop":
                    await session.stop()
                else:
                    await websocket.send_json(
                        {"type": "error", "message": f"unknown action: {action!r}"}
                    )
        except WebSocketDisconnect:
            logger.info("client disconnected")
        finally:
            await session.stop()

    return app


app = create_app()
