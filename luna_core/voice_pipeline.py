"""VoicePipeline — wires wake word, mic gate, STT, conversation, and TTS
together, driving StateMachine transitions and publishing progress on the
EventBus.

This is the async equivalent of jarvis/main.py's handle_wake /
_handle_wake_inner, minus tool dispatch (Phase 2) and with cancellation /
barge-in as first-class instead of bolted on with a global threading.Event.
"""
from __future__ import annotations

import asyncio

from luna_core.audio.capture import MicGate
from luna_core.audio.stt import STTEngine, record_until_silence
from luna_core.audio.tts import PiperTTS
from luna_core.audio.wake_word import WakeWordListener
from luna_core.cancellation import CancellationToken, LunaCancelled
from luna_core.config import LunaConfig
from luna_core.conversation import ConversationLoop, is_stop_command
from luna_core.events import EventBus, EventType
from luna_core.logging_setup import get_logger
from luna_core.state_machine import LunaState, StateMachine

logger = get_logger("pipeline")


class VoicePipeline:
    def __init__(
        self,
        config: LunaConfig,
        event_bus: EventBus,
        state_machine: StateMachine,
        stt: STTEngine,
        tts: PiperTTS,
        conversation: ConversationLoop,
    ) -> None:
        self._config = config
        self._bus = event_bus
        self._state = state_machine
        self._stt = stt
        self._tts = tts
        self._conversation = conversation
        self._mic_gate = MicGate()
        self._cancellation = CancellationToken()
        self._turn_lock = asyncio.Lock()

    @property
    def mic_gate(self) -> MicGate:
        return self._mic_gate

    @property
    def state(self) -> StateMachine:
        return self._state

    def build_wake_listener(self) -> WakeWordListener:
        """Constructs (but does not start) the wake-word listener, wired to
        this pipeline's activate/interrupt and mic gate. main.py only calls
        .run() on it when a trained model is actually configured."""
        return WakeWordListener(
            config=self._config.wake_word,
            mic_gate=self._mic_gate,
            on_wake=self.activate,
            on_interrupt=self.interrupt,
            is_busy=lambda: self._state.is_busy,
        )

    async def activate(self) -> None:
        """Start a full listen-think-speak turn (records real audio). Called
        by the wake-word listener, or manually via the WebSocket `activate`
        message while voice wake-up is disabled."""
        if self._turn_lock.locked():
            logger.info("activate() called while a turn is already running — ignoring")
            return
        async with self._turn_lock:
            self._cancellation.reset()
            try:
                await self._run_voice_turn()
            except LunaCancelled:
                logger.info("turn cancelled")
            except Exception:
                logger.exception("unhandled error in voice pipeline turn")
                await self._safe_transition(LunaState.ERROR, "pipeline_error")
                await self._safe_transition(LunaState.IDLE, "recovered_from_error")
            finally:
                self._cancellation.reset()

    async def activate_with_text(self, text: str) -> None:
        """Same shape as activate() but skips wake/record/STT — the
        WebSocket typed-chat testing path that stands in for voice
        activation until a trained wake-word model is configured."""
        if self._turn_lock.locked():
            logger.info("activate_with_text() called while a turn is already running — ignoring")
            return
        async with self._turn_lock:
            self._cancellation.reset()
            try:
                await self._state.transition_to(LunaState.WAKE, "typed_input")
                self._bus.publish_type(EventType.WAKE_DETECTED, source="typed")
                await self._run_thinking_and_speaking(text)
            except LunaCancelled:
                logger.info("typed turn cancelled")
            except Exception:
                logger.exception("unhandled error in typed turn")
                await self._safe_transition(LunaState.ERROR, "pipeline_error")
                await self._safe_transition(LunaState.IDLE, "recovered_from_error")
            finally:
                self._cancellation.reset()

    async def interrupt(self) -> None:
        """Barge-in: stop whatever is running instead of starting a new
        turn. Also driven by the WebSocket `interrupt` message."""
        logger.info("interrupt requested (state=%s)", self._state.current.value)
        self._cancellation.cancel()
        self._tts.stop()
        if self._state.current not in (LunaState.IDLE, LunaState.INTERRUPTED):
            await self._safe_transition(LunaState.INTERRUPTED, "interrupt")
            await self._safe_transition(LunaState.IDLE, "interrupt_resolved")

    async def _run_voice_turn(self) -> None:
        await self._state.transition_to(LunaState.WAKE, "wake_detected")
        self._bus.publish_type(EventType.WAKE_DETECTED, source="voice")

        await self._state.transition_to(LunaState.LISTENING, "recording")
        self._mic_gate.pause()
        try:
            await asyncio.sleep(0.15)  # let the wake-word mic stream release the device
            audio = await record_until_silence(self._config.stt, self._cancellation)
        finally:
            self._mic_gate.resume()

        self._cancellation.check()
        text = await self._stt.transcribe(audio)
        self._bus.publish_type(EventType.TRANSCRIPT, text=text)

        if not text.strip():
            await self._state.transition_to(LunaState.IDLE, "empty_transcript")
            return

        await self._run_thinking_and_speaking(text)

    async def _run_thinking_and_speaking(self, text: str) -> None:
        if is_stop_command(text):
            logger.info("stop command recognized post-wake-word")
            self._cancellation.cancel()
            await self._state.transition_to(LunaState.INTERRUPTED, "stop_command")
            await self._state.transition_to(LunaState.IDLE, "stop_resolved")
            return

        await self._state.transition_to(LunaState.THINKING, "calling_llm")
        reply = await self._conversation.handle_utterance(text, self._cancellation)
        self._bus.publish_type(EventType.LLM_RESPONSE, text=reply)

        self._cancellation.check()
        await self._state.transition_to(LunaState.SPEAKING, "speaking")
        try:
            await self._tts.speak(reply, self._cancellation)
        finally:
            if self._state.current == LunaState.SPEAKING:
                await self._state.transition_to(LunaState.IDLE, "done_speaking")

    async def _safe_transition(self, target: LunaState, reason: str) -> None:
        try:
            await self._state.transition_to(target, reason)
        except Exception:
            logger.exception("failed to transition to %s", target)
