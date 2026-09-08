import { useCallback, useEffect, useRef, useState } from "react";
import type { LunaEvent, LunaState, StateChangedData } from "../types";

export interface TranscriptEntry {
  role: "user" | "assistant";
  text: string;
  timestamp: number;
}

interface UseLunaSocketResult {
  state: LunaState;
  connected: boolean;
  transcript: TranscriptEntry[];
  lastEvent: LunaEvent | null;
  activate: () => void;
  interrupt: () => void;
  sendChat: (text: string) => void;
}

const RECONNECT_DELAY_MS = 1500;

function wsUrl(): string {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  return `${protocol}://${window.location.host}/ws`;
}

/** Connects to luna_core/server.py's /ws endpoint, tracks LunaState off
 * `state_changed` events, and exposes activate/interrupt/chat — the manual
 * triggers that stand in for voice wake-up while wake-word detection is
 * disabled (see ROADMAP.md). Reconnects automatically on drop. */
export function useLunaSocket(): UseLunaSocketResult {
  const [state, setState] = useState<LunaState>("idle");
  const [connected, setConnected] = useState(false);
  const [transcript, setTranscript] = useState<TranscriptEntry[]>([]);
  const [lastEvent, setLastEvent] = useState<LunaEvent | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    let cancelled = false;
    let reconnectTimer: ReturnType<typeof setTimeout> | undefined;

    const connect = () => {
      const socket = new WebSocket(wsUrl());
      wsRef.current = socket;

      socket.onopen = () => setConnected(true);
      socket.onclose = () => {
        setConnected(false);
        if (!cancelled) {
          reconnectTimer = setTimeout(connect, RECONNECT_DELAY_MS);
        }
      };
      socket.onerror = () => socket.close();
      socket.onmessage = (event: MessageEvent<string>) => {
        const parsed = JSON.parse(event.data) as LunaEvent;
        setLastEvent(parsed);

        if (parsed.type === "state_changed") {
          const data = parsed.data as unknown as StateChangedData;
          setState(data.to);
        } else if (parsed.type === "transcript") {
          const text = String(parsed.data.text ?? "");
          if (text.trim()) {
            setTranscript((prev) => [...prev, { role: "user", text, timestamp: parsed.timestamp }]);
          }
        } else if (parsed.type === "llm_response") {
          const text = String(parsed.data.text ?? "");
          if (text.trim()) {
            setTranscript((prev) => [...prev, { role: "assistant", text, timestamp: parsed.timestamp }]);
          }
        }
      };
    };

    connect();
    return () => {
      cancelled = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      wsRef.current?.close();
    };
  }, []);

  const send = useCallback((payload: Record<string, unknown>) => {
    const socket = wsRef.current;
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(payload));
    }
  }, []);

  const activate = useCallback(() => send({ type: "activate" }), [send]);
  const interrupt = useCallback(() => send({ type: "interrupt" }), [send]);
  const sendChat = useCallback((text: string) => send({ type: "chat", text }), [send]);

  return { state, connected, transcript, lastEvent, activate, interrupt, sendChat };
}
