# Luna Architecture

Luna runs everything — the voice pipeline, the WebSocket server, and every future tool call —
on a **single asyncio event loop**. That one decision drives most of what's below: there is no
`threading.Event`/`run_coroutine_threadsafe` bridging anywhere in `luna_core`, because there's
no second thread to bridge to. Where a call is genuinely blocking (mic I/O, model inference,
subprocess I/O), it's wrapped in `await asyncio.to_thread(...)` for the duration of that one
call, not run on a long-lived background thread.

## Package layout

```
luna_core/
  config.py            typed (pydantic) config, loaded once from config.yaml
  logging_setup.py      stdlib logging, "luna.*" namespace
  events.py              Event / EventType / EventBus
  state_machine.py        LunaState / StateMachine
  cancellation.py          CancellationToken / LunaCancelled
  permissions/
    models.py               PermissionLevel / PermissionRequest / PermissionDecision
    engine.py                PermissionEngine.authorize()
  providers.py               LLMProvider / OllamaProvider / OpenAICompatibleProvider / ProviderRouter
  audio/
    capture.py                MicGate
    wake_word.py                WakeWordListener (openWakeWord)
    stt.py                       STTEngine (faster-whisper) + record_until_silence()
    tts.py                        PiperTTS (subprocess-isolated)
  conversation.py                 ConversationLoop
  voice_pipeline.py                 VoicePipeline (orchestrator)
  server.py                          FastAPI app + /ws
  main.py                            entry point (python -m luna_core.main)

frontend/            Vite + React + TypeScript — LunaOrb UI, connects to /ws
```

## EventBus

```python
class EventBus:
    def subscribe(self) -> asyncio.Queue[Event]: ...
    def publish(self, event: Event) -> None: ...
```

Every subscriber (each WebSocket client, plus any internal observer) gets its own
`asyncio.Queue`. `publish()` is a non-blocking `put_nowait()` fan-out to every queue — a slow
consumer drops events (bounded queue, `QueueFull` is swallowed) rather than ever blocking a
publisher. This is the direct replacement for jarvis's `_event_listeners: list[Callable]` +
`asyncio.run_coroutine_threadsafe(...)` bridge, which existed only because jarvis's voice loop
ran on a plain OS thread outside the FastAPI event loop. Luna has no such thread, so the bridge
has no reason to exist.

## StateMachine

States: `IDLE, WAKE, LISTENING, THINKING, EXECUTING, SPEAKING, INTERRUPTED, ERROR`.

`transition_to()` is `async` and serialized with an internal `asyncio.Lock` — the normal
pipeline flow and a barge-in interrupt can both attempt a transition from different coroutines
concurrently (the interrupt path runs independently of whatever turn is in progress), so the
lock prevents a torn state update. Every transition is validated against an explicit table
(`ERROR` and `INTERRUPTED` are reachable from any state — they're the two "something went
wrong or was cancelled" escapes) and, on success, published to the `EventBus` as a
`state_changed` event with `{from, to, reason}`. This is what makes barge-in *observable*: the
frontend sees `SPEAKING → INTERRUPTED → IDLE` as real events, not an internal side effect that
happens to also silence the speaker.

`IDLE → WAKE → LISTENING → THINKING → SPEAKING → IDLE` is the full voice turn. `WAKE → THINKING`
(skipping `LISTENING`) exists for the manual/typed-input path (see below) — text that's already
available doesn't need a recording step.

## Async mic sharing (MicGate)

`luna_core/audio/capture.py`'s `MicGate` is an `asyncio.Event` wrapper: `pause()` clears it,
`resume()` sets it, `wait_until_resumed()` awaits it. That's only a correct coordination
primitive because both consumers — `WakeWordListener.run()` and `record_until_silence()` — are
themselves `async def` loops that `await asyncio.to_thread(mic.read, ...)` / `await
asyncio.to_thread(model.predict, ...)` per chunk, rather than blocking the event loop for an
entire read loop on a background thread. Each iteration hands control back to the event loop,
so checking `gate.is_paused` between iterations needs no thread-safety machinery — everything
happens on the same loop. jarvis's version used a `threading.Event` because its wake-word loop
ran on a real OS thread; Luna's doesn't, so the simpler primitive is also the correct one.

On resume, `WakeWordListener` calls `model.reset()` before reading again, clearing openWakeWord's
internal prediction buffer so a stray sound right after STT finishes recording doesn't
false-trigger on stale state.

## Barge-in and cooldown

`WakeWordListener` takes an `is_busy: Callable[[], bool]` (backed by `StateMachine.is_busy`,
true during `LISTENING/THINKING/EXECUTING/SPEAKING`). A detection while busy calls
`on_interrupt()` instead of `on_wake()` — `VoicePipeline.interrupt()` cancels the shared
`CancellationToken`, stops TTS playback, and transitions `→ INTERRUPTED → IDLE`. It does **not**
start a new turn; the user has to say the wake word again. A cooldown window
(`wake_word.cooldown_seconds` normally, `interrupt_cooldown_seconds` — longer — after an
interrupt) stops Luna's own "Yes?" or response audio from re-triggering the listener. During
`LISTENING`, the wake-word mic is paused by `MicGate`, so the listener structurally cannot fire
during active recording — no race to reason about there.

`is_busy` conservatively includes `THINKING` (LLM call in flight), not just `EXECUTING`/
`SPEAKING` as literally specified — a stray wake word during "thinking" gets treated as
barge-in rather than silently queuing a second turn once the first resolves.

## Cancellation

`CancellationToken` wraps one `asyncio.Event`; `.check()` raises `LunaCancelled` if set. One
token is created per turn (`VoicePipeline` owns it) and threaded through every step:
between audio chunks in `record_until_silence`, before/after the LLM call in
`ConversationLoop.handle_utterance`, and between sentences (plus continuously during playback)
in `PiperTTS.speak`. "Luna stop" / "cancel" / "never mind" are matched against the *raw
transcript* in `voice_pipeline._run_thinking_and_speaking` — before it's ever sent to an LLM —
so a stop command doesn't have to round-trip through a model call to take effect.

## PermissionEngine and why Phase 1 goes through it anyway

`jarvis/tools/router.py` dispatched every tool call the LLM requested straight from
`response.message.tool_calls`, with zero gating — whatever the model asked for, it got.
`PermissionEngine.authorize(PermissionRequest) -> PermissionDecision` is the fix: every tool
name is looked up against a per-tool policy (`ALLOW` / `DENY` / `ASK`, config-driven), unknown
tools **default to DENY**, and every decision is logged and published to the `EventBus`.

Phase 1 registers zero tools (`ProviderRouter.chat(..., tools=[])`), so there's nothing to
dispatch — but `ConversationLoop._reject_tool_calls()` still exists: if a model hallucinates a
`tool_calls` entry anyway (it happens), each one is run through `PermissionEngine.authorize()`
before the reply is finalized. Since no tool is registered, the default-deny policy rejects it
and the user gets "I can't do that yet." This is a real gated code path today, not a comment
promising one for Phase 2 — when Phase 2 adds real tools, the dispatch point is already wired
to the gate; it's the *policy* (which tools are `ALLOW` vs `ASK`) that changes, not the
enforcement mechanism.

`PermissionEngine.set_approver()` exists for `ASK`-level tools but nothing registers one in
Phase 1 — an `ASK` decision with no approver degrades to a safe deny rather than hanging.

## ProviderRouter

`LLMProvider` is an ABC with one method: `async def chat(messages, *, tools, temperature) ->
ChatResult`. `OllamaProvider` wraps `ollama.AsyncClient`; `OpenAICompatibleProvider` wraps
`openai.AsyncOpenAI` (LM Studio, NVIDIA NIM, OpenAI itself, anything OpenAI-compatible).
`ProviderRouter.chat()` delegates to the configured active provider and falls back to the
`ollama` provider on any exception — the same resilience jarvis's `_process_request` had,
rebuilt as an injectable, unit-testable class instead of module-level functions closing over a
global config re-read on every call.

## Manual activation path

Wake-word detection ships **disabled** in this repo (see ROADMAP.md for why and how to finish
it). Until a trained `hey_luna.onnx` is configured, `VoicePipeline.activate()` can be triggered
manually over the WebSocket:

- `{"type": "activate"}` — runs the *full* voice turn (`WAKE → LISTENING → …`), including real
  microphone recording via `record_until_silence`. This is what LunaOrb's "Activate" button
  sends.
- `{"type": "chat", "text": "..."}` — skips straight to `WAKE → THINKING` with the given text
  (`VoicePipeline.activate_with_text`), no microphone involved. What LunaOrb's text box sends,
  and what the automated WebSocket test in `tests/luna_core/test_server.py` uses (it must not
  touch real audio hardware).
- `{"type": "interrupt"}` — same barge-in path a wake word during `EXECUTING`/`SPEAKING` would
  trigger.

`WakeWordListener` itself is unaffected by any of this — it's fully implemented and unit-tested
against a mocked openWakeWord model (`tests/luna_core/test_wake_word.py`); `main.py` simply
doesn't start it unless `wake_word.enabled` is true *and* `wake_word.model_path` points at a
file that exists.

## TTS: Piper, always as a subprocess

Piper's phonemizer dependency is **espeak-ng, GPL-3.0-or-later**. Importing a Piper Python
binding in-process would pull GPL-licensed code into the same process as the rest of Luna
(MIT-licensed); `luna_core/audio/tts.py`'s `PiperTTS` always shells out via
`asyncio.create_subprocess_exec([sys.executable, "-m", "piper", ...])`, keeping that boundary at
the OS process level regardless of how Piper itself is packaged. Audio is streamed
sentence-by-sentence (`--output-raw` on stdout) so playback can start before the whole reply has
synthesized, mirroring jarvis's `speak_streamed` but subprocess-based and cancellation-aware —
`speak()` checks the shared `CancellationToken` between sentences and continuously during
playback (polling `sounddevice`'s active stream), and `stop()` kills both the in-flight
subprocess and playback for barge-in.

### Voice choice

Default voice: **`en_US-lessac-high`** (see the Phase 1 validation report for the actual
time-to-first-audio / total-synth-time benchmark numbers measured on this machine, and for
`en_US-amy-medium`'s numbers as the documented lighter/faster fallback — swapping is a one-line
change to `tts.voice_model_path` in `config.yaml`, no code change). Piper voices are licensed
**per voice**, not under one blanket license — the specific license for each voice used here is
recorded in the Phase 1 report, pulled from that voice's model card before shipping either as
default.

## Config

`luna_core/config.py` defines `LunaConfig` as a tree of pydantic `BaseModel`s
(`ServerConfig`, `WakeWordConfig`, `SttConfig`, `TtsConfig`, `LlmConfig`, `ContextConfig`,
`PermissionsConfig`, `LoggingConfig`), loaded once via `load_config()` and cached; every module
takes its config section through its constructor rather than re-reading `config.yaml` itself —
jarvis's every module had its own `_load_config()` reading the file fresh on every call. pydantic
is already a transitive dependency (FastAPI requires it), so this adds no new dependency.

## Known limitation: memory

Phase 1 keeps a bounded in-process message list (`ConversationLoop._history`, capped at
`context.max_messages`) — enough for "LLM conversation," not a persistent cross-session memory
system like jarvis's SQLite+ChromaDB fact store. Deferred to a later phase; flagged here so it
isn't mistaken for an oversight.
