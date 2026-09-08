# Luna Roadmap

Luna is a local, async-native, permission-gated AI voice assistant. It replaces the earlier
Jarvis codebase in this repo (thread-based, ungated tool dispatch) with an architecture built
around an async `EventBus`/`StateMachine`, a `PermissionEngine` that gates every tool call by
default, and a `ProviderRouter` abstraction over LLM backends. See [ARCHITECTURE.md](ARCHITECTURE.md)
for how those pieces fit together, and [TESTING.md](TESTING.md) for how to verify changes.

## Phase 0 — Foundation ✅

The core, tool-agnostic architecture everything else builds on. Documented retroactively
alongside Phase 1 (see the Phase 1 validation report) since it didn't exist as a separate delivery.

- [x] `luna_core/config.py` — typed (pydantic) config loaded once from `config.yaml`
- [x] `luna_core/logging_setup.py` — structured logging, console + rotating file handler
- [x] `luna_core/events.py` — async `EventBus` (per-subscriber `asyncio.Queue` fan-out)
- [x] `luna_core/state_machine.py` — `LunaState` + `StateMachine` with a validated transition
      table, every change published to the `EventBus`
- [x] `luna_core/cancellation.py` — shared `CancellationToken` / `LunaCancelled`
- [x] `luna_core/permissions/engine.py` — `PermissionEngine.authorize()`, default-deny,
      audit-logged, no bypass path
- [x] `luna_core/providers.py` — `ProviderRouter` over Ollama / OpenAI-compatible backends,
      with fallback-to-Ollama on error

## Phase 1 — Voice pipeline ✅ (wake word scaffolded, inactive)

Microphone capture, local wake-word detection, speech-to-text, LLM conversation through the
provider layer, text-to-speech with a female voice, and the LunaOrb frontend wired to real
state changes over a local WebSocket. No computer-control tools yet — that's Phase 2.

- [x] Mic capture + async `MicGate` pause/resume between wake-word listening and STT recording
      (`luna_core/audio/capture.py`)
- [ ] **Wake word ("Hey Luna") — listener built, ships DISABLED.** openWakeWord has no
      pretrained "Hey Luna" model; training one needs a custom pipeline (GPU-based Docker, or a
      ~75–90 min Colab run) that this session could not complete — no local GPU
      (`nvidia-smi` not found) and no way to drive an interactive Colab session. `luna_core/audio/wake_word.py`
      is fully implemented and unit-tested (cooldown, barge-in, mic-gate integration, model reset
      on resume) against a mocked model. **To finish it:**
      1. Run one of:
         - [CoreWorxLab/openwakeword-training](https://github.com/CoreWorxLab/openwakeword-training)
           (Docker, local, needs a GPU) — synthesizes positive samples via Kokoro TTS + your own
           recorded voice (weighted higher), negatives from unrelated phrases.
         - [alfiedennen/openwakeword-colab-2026](https://github.com/alfiedennen/openwakeword-colab-2026)
           (Colab, MIT-licensed, run-all notebook, ~75–90 min on Colab Pro / ~2x on free T4).
         - Target phrase: `["hey luna", "hey luna"]`.
      2. Drop the resulting `hey_luna.onnx` at the path you set in `config.yaml`'s
         `wake_word.model_path`.
      3. Set `wake_word.enabled: true`.
      4. **License note:** if your training run used the ACAV100M negative-audio corpus (common
         in both pipelines above), the resulting model inherits ACAV100M's CC-BY-NC-SA-4.0
         restriction — fine for personal use, but flag it if this project is ever distributed.
      Until then, trigger turns via the WebSocket `activate` (records real mic audio) or `chat`
      (typed text) messages — see [ARCHITECTURE.md](ARCHITECTURE.md#manual-activation-path).
- [x] Barge-in: wake word heard again during `EXECUTING`/`SPEAKING` (and, conservatively,
      `THINKING`) interrupts instead of starting a second listen cycle, routed through
      `StateMachine.transition_to(LunaState.INTERRUPTED)`; cooldown window prevents Luna's own
      "Yes?"/response audio from self-triggering (`luna_core/audio/wake_word.py`)
- [x] Shared `CancellationToken` checked at each pipeline step; "Luna stop" / "cancel" /
      "never mind" recognized on the transcript before it reaches the LLM
      (`luna_core/conversation.py`, `luna_core/voice_pipeline.py`)
- [x] STT via faster-whisper, async (`luna_core/audio/stt.py`)
- [x] LLM conversation via `ProviderRouter` (`luna_core/conversation.py`)
- [x] TTS via Piper, subprocess-isolated, female voice
      (`luna_core/audio/tts.py`) — voice choice and benchmark numbers in ARCHITECTURE.md
- [x] LunaOrb React component wired to real `StateMachine` state over `/ws`
      (`frontend/src/components/LunaOrb.tsx`, `frontend/src/hooks/useLunaSocket.ts`)
- [x] Any `tool_calls` an LLM hallucinates (Phase 1 registers none) are routed through
      `PermissionEngine.authorize()` before anything happens — see
      [ARCHITECTURE.md](ARCHITECTURE.md#permission-gating-from-day-one)

## Phase 2 — Computer-control tools (next)

- Tool registry + schemas (screen vision, desktop automation, files, web, system) — every
  dispatch goes through `PermissionEngine.authorize()`, no exceptions, unlike jarvis's
  `tools/router.py`.
- Interactive `ASK`-level approval flow: `PermissionEngine.set_approver()` wired to a real UI
  prompt (LunaOrb shows a pending-approval state) instead of Phase 1's safe-deny default.
- Persistent memory (deferred from Phase 1 — a simple bounded in-process history was enough
  for "LLM conversation," but multi-session recall needs real storage).
- Finish wake-word training (see Phase 1 above) if not already done.
