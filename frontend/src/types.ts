// Mirrors luna_core/state_machine.py's LunaState and luna_core/events.py's
// EventType — keep these in sync by hand; the WebSocket wire format is the
// only contract between the two, there's no shared schema generation yet.

export type LunaState =
  | "idle"
  | "wake"
  | "listening"
  | "thinking"
  | "executing"
  | "speaking"
  | "interrupted"
  | "error";

export type EventType =
  | "state_changed"
  | "wake_detected"
  | "transcript"
  | "llm_response"
  | "tool_call"
  | "tool_denied"
  | "error"
  | "mute_changed"
  | "provider_changed"
  | "connected";

export interface LunaEvent {
  type: EventType;
  data: Record<string, unknown>;
  timestamp: number;
}

export interface StateChangedData {
  from: LunaState;
  to: LunaState;
  reason: string | null;
}
