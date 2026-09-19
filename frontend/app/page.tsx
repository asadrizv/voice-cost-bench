"use client";

import { LiveKitRoom, RoomAudioRenderer, useDataChannel } from "@livekit/components-react";
import { useCallback, useEffect, useState } from "react";
import { CallButton } from "@/components/CallButton";
import { LatencyBreakdown } from "@/components/LatencyBreakdown";
import { LiveCostPanel } from "@/components/LiveCostPanel";
import { PipelineToggle } from "@/components/PipelineToggle";
import { Transcript } from "@/components/Transcript";
import { api, DEFAULT_ENDPOINTER, ENDPOINTERS, type Endpointer, type Pipeline, type TokenResponse } from "@/lib/api";
import { useLiveMetrics } from "@/lib/useLiveMetrics";

type CallStatus = { state: string; reason?: string; ended_by?: string };

function StatusListener({ onStatus }: { onStatus: (s: CallStatus) => void }) {
  useDataChannel("call-status", (msg) => {
    try {
      onStatus(JSON.parse(new TextDecoder().decode(msg.payload)));
    } catch {
      /* ignore malformed status */
    }
  });
  return null;
}

export default function CallPage() {
  const [pipeline, setPipeline] = useState<Pipeline>("api");
  const [persona, setPersona] = useState("law_firm");
  const [endpointer, setEndpointer] = useState<Endpointer>(DEFAULT_ENDPOINTER);
  const [session, setSession] = useState<TokenResponse | null>(null);
  const [callId, setCallId] = useState<string | null>(null);
  const [connecting, setConnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const live = useLiveMetrics(callId);
  const [env, setEnv] = useState<{ local: boolean; simulated: boolean }>({ local: false, simulated: false });
  // Calling before the server's default pipeline arrives would silently use the toggle's
  // initial value; a keyless API pipeline then rejects the call.
  const [configLoaded, setConfigLoaded] = useState(false);
  useEffect(() => {
    api
      .config()
      .then((c) => {
        setEnv({ local: Boolean(c.selfhosted_on_local_machine), simulated: Boolean(c.simulated) });
        if (c.default_pipeline === "api" || c.default_pipeline === "selfhosted") setPipeline(c.default_pipeline);
      })
      .catch(() => undefined)
      .finally(() => setConfigLoaded(true));
  }, []);

  const start = useCallback(async () => {
    setError(null);
    setNotice(null);
    setConnecting(true);
    try {
      const token = await api.token({ pipeline, persona, endpointer });
      setSession(token);
      setCallId(token.call_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setConnecting(false);
    }
  }, [pipeline, persona, endpointer]);

  const end = useCallback(() => setSession(null), []);
  const reset = useCallback(() => {
    setSession(null);
    setCallId(null);
    setError(null);
    setNotice(null);
  }, []);

  return (
    <main>
      <div className="row" style={{ margin: "16px 0 20px" }}>
        <div>
          <h1>Hartley &amp; Weber reception</h1>
          <p className="secondary small" style={{ margin: "2px 0 0" }}>
            Talk to the intake agent. The same conversation runs on either pipeline; cost and latency update live.
          </p>
        </div>
        <span className="spacer" />
        <PipelineToggle value={pipeline} onChange={setPipeline} disabled={!!session} />
        <select aria-label="Language" value={persona} disabled={!!session} onChange={(e) => setPersona(e.target.value)}>
          <option value="law_firm">English</option>
          <option value="law_firm_de">Deutsch</option>
        </select>
        <select aria-label="Endpointing" value={endpointer} disabled={!!session} onChange={(e) => setEndpointer(e.target.value as Endpointer)}>
          {Object.entries(ENDPOINTERS).map(([value, label]) => (
            <option key={value} value={value}>{label}</option>
          ))}
        </select>
        <CallButton active={!!session} connecting={connecting || !configLoaded} onStart={start} onEnd={end} />
        <button type="button" className="btn ghost" onClick={reset} disabled={!!session}>Reset</button>
      </div>

      {error && <div className="banner error" role="alert">{error}</div>}
      {notice && !session && <div className="banner" role="status" data-testid="call-notice">{notice}</div>}
      {env.simulated && (
        <div className="banner warn" role="note">
          <strong>Simulated providers.</strong> Replies are a tone and a fixed script; costs are fiction.
        </div>
      )}
      {!env.simulated && env.local && pipeline === "selfhosted" && (
        <div className="banner warn" role="note">
          <strong>Self-hosted models are running on this machine.</strong> Latency reflects a laptop, and cost is priced as
          a share of the benchmark L40S. Neither is a benchmark figure.
        </div>
      )}

      {session && (
        <LiveKitRoom serverUrl={session.url} token={session.token} connect audio video={false}
          onDisconnected={() => setSession(null)}
          onError={(e) => setError(`LiveKit: ${e.message}`)}>
          <RoomAudioRenderer />
          <StatusListener
            onStatus={(s) => {
              if (s.state === "rejected") {
                setError(`Call rejected: ${s.reason ?? "unknown reason"}`);
                setSession(null);
              }
              if (s.state === "failed") setError("The call failed on the server; see agent logs.");
              if (s.state === "ended" && s.ended_by === "agent") {
                setNotice("Clara ended the call.");
                setSession(null);
              }
            }}
          />
        </LiveKitRoom>
      )}

      <div className="grid cols-2">
        <LiveCostPanel live={live} />
        <LatencyBreakdown turns={live.turns} />
      </div>
      <div className="section">
        <Transcript turns={live.turns} />
      </div>
    </main>
  );
}
