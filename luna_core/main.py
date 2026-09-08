"""Luna entry point.

Builds the dependency graph explicitly (no globals/singletons beyond
config's cache) and runs the FastAPI server + voice pipeline together on one
asyncio event loop. The wake-word listener only starts if a trained model is
both enabled and present on disk — see ROADMAP.md decision on why it ships
inactive in this phase.

Run with: python -m luna_core.main
"""
from __future__ import annotations

import asyncio
import webbrowser
from pathlib import Path

import uvicorn

from luna_core.audio.stt import STTEngine
from luna_core.audio.tts import PiperTTS
from luna_core.config import load_config
from luna_core.conversation import ConversationLoop
from luna_core.events import EventBus
from luna_core.logging_setup import configure_logging, get_logger
from luna_core.permissions.engine import PermissionEngine
from luna_core.permissions.models import PermissionLevel
from luna_core.providers import ProviderRouter
from luna_core.server import create_app
from luna_core.state_machine import StateMachine
from luna_core.voice_pipeline import VoicePipeline

logger = get_logger("main")


async def run() -> None:
    config = load_config()
    configure_logging(config.logging.level, config.logging.dir)

    event_bus = EventBus()
    state_machine = StateMachine(event_bus)
    provider_router = ProviderRouter(config.llm)
    permission_engine = PermissionEngine(
        default_level=PermissionLevel(config.permissions.default_level),
        tool_levels={k: PermissionLevel(v) for k, v in config.permissions.tools.items()},
        event_bus=event_bus,
    )
    stt_engine = STTEngine(config.stt)
    tts_engine = PiperTTS(config.tts)
    conversation = ConversationLoop(provider_router, permission_engine, config.context.max_messages)

    pipeline = VoicePipeline(config, event_bus, state_machine, stt_engine, tts_engine, conversation)

    app = create_app(config, event_bus, pipeline, provider_router)
    server_config = uvicorn.Config(app, host=config.server.host, port=config.server.port, log_level="warning")
    server = uvicorn.Server(server_config)

    tasks = [asyncio.create_task(server.serve())]

    wake_model_path = config.wake_word.model_path
    if config.wake_word.enabled and wake_model_path and Path(wake_model_path).exists():
        listener = pipeline.build_wake_listener()
        tasks.append(asyncio.create_task(listener.run()))
        logger.info("wake word ACTIVE — say %r", config.wake_word.phrase)
    else:
        reason = "disabled in config" if not config.wake_word.enabled else f"model not found at {wake_model_path!r}"
        logger.warning(
            "wake word INACTIVE (%s) — trigger turns via the WebSocket 'activate'/'chat' "
            "message until a trained model is configured. See ROADMAP.md.",
            reason,
        )

    logger.info("Luna server: http://%s:%s", config.server.host, config.server.port)
    try:
        webbrowser.open(f"http://localhost:{config.server.port}")
    except Exception:
        logger.debug("could not open browser automatically", exc_info=True)

    await asyncio.gather(*tasks)


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
