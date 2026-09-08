import type { LunaState } from "../types";
import "./LunaOrb.css";

interface LunaOrbProps {
  state: LunaState;
}

const STATE_LABEL: Record<LunaState, string> = {
  idle: "Idle",
  wake: "Waking",
  listening: "Listening",
  thinking: "Thinking",
  executing: "Executing",
  speaking: "Speaking",
  interrupted: "Interrupted",
  error: "Error",
};

/** Visual state indicator, driven entirely by the LunaState the backend
 * reports over the WebSocket — see hooks/useLunaSocket.ts. No local state
 * of its own, so it can never drift from what the pipeline is actually
 * doing. */
export function LunaOrb({ state }: LunaOrbProps) {
  return (
    <div className="luna-orb-wrap" data-state={state}>
      <div className="luna-orb">
        <div className="luna-orb-ring luna-orb-ring-2" />
        <div className="luna-orb-ring luna-orb-ring-1" />
        <div className="luna-orb-core" />
      </div>
      <div className="luna-orb-label">{STATE_LABEL[state]}</div>
    </div>
  );
}
