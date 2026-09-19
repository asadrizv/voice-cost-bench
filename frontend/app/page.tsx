"use client";

import { LiveKitRoom, RoomAudioRenderer, useDataChannel } from "@livekit/components-react";
import { useCallback, useEffect, useState } from "react";
import { CallButton } from "@/components/CallButton";
import { LatencyBreakdown } from "@/components/LatencyBreakdown";
import { LiveCostPanel } from "@/components/LiveCostPanel";
import { PipelineToggle } from "@/components/PipelineToggle";
import { Transcript } from "@/components/Transcript";
import { api, type Pipeline, type TokenResponse } from "@/lib/api";
import { useLiveMetrics } from "@/lib/useLiveMetrics";

function StatusListener({ onStatus }: { onStatus: (s: { state: string; reason?: string }) => void }) {
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
  const [endpointer, setEndpointer] = useState("semantic");
  const [session, setSession] = useState<TokenResponse | null>(null);
  const [callId, setCallId] = useState<string | null>(null);
  const [connecting, setConnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const live = useLiveMetrics(callId);
  const [env, setEnv] = useState<{ local: boolean; simulated: boolean }>({ local: false, simulated: false });
  useEffect(() => {
    api
      .config()
      .then((c) => {
        setEnv({ local: Boolean(c.selfhosted_on_local_machine), simulated: Boolean(c.simulated) });
        if (c.default_pipeline === "api" || c.default_pipeline === "selfhosted") setPipeline(c.default_pipeline);
      })
      .catch(() => undefined);
  }, []);

  const start = useCallback(async () => {
    setError(null);
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
        <select aria-label="Endpointing" value={endpointer} disabled={!!session} onChange={(e) => setEndpointer(e.target.value)}>
          <option value="semantic">Semantic endpointing</option>
          <option value="silence">Silence threshold</option>
        </select>
        <CallButton active={!!session} connecting={connecting} onStart={start} onEnd={end} />
        <button type="button" className="btn ghost" onClick={reset} disabled={!!session}>Reset</button>
      </div>

      {error && <div className="banner error" role="alert">{error}</div>}
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
