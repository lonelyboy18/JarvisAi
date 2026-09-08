"""FastAPI + WebSocket server.

Ports jarvis/web.py's provider/settings REST endpoints and replaces its
`_broadcast_to_ws` + `run_coroutine_threadsafe` bridge with a direct
EventBus subscription per client — everything already runs on one asyncio
event loop, so no thread-safety bridge is needed.

The WebSocket also accepts `activate` / `interrupt` / `chat` messages, which
drive VoicePipeline directly. This is the manual trigger that stands in for
voice wake-up while wake-word detection is disabled (see ROADMAP.md) — it's
also just a generally useful way to exercise the pipeline from the LunaOrb
frontend or a REST client without needing a microphone.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from luna_core.config import LunaConfig
from luna_core.conversation import is_stop_command
from luna_core.events import EventBus
from luna_core.logging_setup import get_logger
from luna_core.providers import ProviderRouter
from luna_core.voice_pipeline import VoicePipeline

logger = get_logger("server")

_FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"


def create_app(
    config: LunaConfig,
    event_bus: EventBus,
    pipeline: VoicePipeline,
    provider_router: ProviderRouter,
) -> FastAPI:
    app = FastAPI(title="Luna")

    @app.get("/api/state")
    async def api_state() -> JSONResponse:
        return JSONResponse({"state": pipeline.state.current.value})

    @app.get("/api/providers")
    async def api_providers() -> JSONResponse:
        providers = {
            key: {"label": p.label, "model": p.model, "type": p.type, "base_url": p.base_url}
            for key, p in provider_router.providers().items()
        }
        return JSONResponse({"providers": providers, "active": provider_router.active})

    @app.post("/api/providers/{provider_key}")
    async def api_set_provider(provider_key: str) -> JSONResponse:
        if provider_router.switch_active(provider_key):
            return JSONResponse({"status": "ok", "active": provider_key})
        return JSONResponse({"status": "error", "message": "Unknown provider"}, status_code=400)

    @app.get("/api/settings")
    async def api_get_settings() -> JSONResponse:
        return JSONResponse(
            {
                "tts": config.tts.model_dump(),
                "stt": config.stt.model_dump(),
                "wake_word": config.wake_word.model_dump(),
                "llm": {"active_provider": provider_router.active, "temperature": config.llm.temperature},
            }
        )

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        queue = event_bus.subscribe()
        forward_task = asyncio.create_task(_forward_events(ws, queue))
        await ws.send_text(json.dumps({"type": "connected", "data": {"message": "Luna online."}, "timestamp": 0}))
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                await _handle_inbound(msg, pipeline)
        except WebSocketDisconnect:
            pass
        finally:
            forward_task.cancel()
            event_bus.unsubscribe(queue)

    if _FRONTEND_DIST.exists():
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")
    else:
        logger.warning("frontend build not found at %s — run `npm run build` in frontend/", _FRONTEND_DIST)

    return app


async def _forward_events(ws: WebSocket, queue: asyncio.Queue[Any]) -> None:
    try:
        while True:
            event = await queue.get()
            await ws.send_text(json.dumps(event.to_dict()))
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.exception("event forwarding failed")


async def _handle_inbound(msg: dict[str, Any], pipeline: VoicePipeline) -> None:
    msg_type = msg.get("type")
    if msg_type == "activate":
        asyncio.create_task(pipeline.activate())
    elif msg_type == "interrupt":
        asyncio.create_task(pipeline.interrupt())
    elif msg_type == "chat":
        text = str(msg.get("text", ""))
        if not text.strip():
            return
        if is_stop_command(text):
            asyncio.create_task(pipeline.interrupt())
            return
        asyncio.create_task(pipeline.activate_with_text(text))
    else:
        logger.debug("ignoring unknown websocket message type=%r", msg_type)
