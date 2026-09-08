import { useState, type FormEvent } from "react";
import "./App.css";
import { LunaOrb } from "./components/LunaOrb";
import { useLunaSocket } from "./hooks/useLunaSocket";

export default function App() {
  const { state, connected, transcript, activate, interrupt, sendChat } = useLunaSocket();
  const [draft, setDraft] = useState("");

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault();
    if (!draft.trim()) return;
    sendChat(draft.trim());
    setDraft("");
  };

  return (
    <div className="luna-app">
      <header className="luna-header">
        <h1>Luna</h1>
        <span className={`luna-status-dot ${connected ? "connected" : "disconnected"}`} title={connected ? "Connected" : "Disconnected"} />
      </header>

      <LunaOrb state={state} />

      <div className="luna-controls">
        <button onClick={activate} disabled={!connected}>
          Activate
        </button>
        <button onClick={interrupt} disabled={!connected}>
          Interrupt
        </button>
      </div>

      <p className="luna-hint">
        Voice wake-word ("Hey Luna") is not trained yet — use Activate (records real mic audio) or type below.
      </p>

      <form className="luna-chat-form" onSubmit={handleSubmit}>
        <input
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          placeholder='Type instead of saying "Hey Luna"…'
          disabled={!connected}
        />
        <button type="submit" disabled={!connected}>
          Send
        </button>
      </form>

      <div className="luna-transcript">
        {transcript.map((entry, i) => (
          <div key={i} className={`luna-transcript-entry luna-transcript-${entry.role}`}>
            <strong>{entry.role === "user" ? "You" : "Luna"}:</strong> {entry.text}
          </div>
        ))}
      </div>
    </div>
  );
}
