"use client";

import { ms, usd } from "@/lib/format";
import type { LiveTurn } from "@/lib/useLiveMetrics";

export function Transcript({ turns }: { turns: LiveTurn[] }) {
  return (
    <section className="card" aria-label="Transcript">
      <div className="card-head"><h2>Conversation</h2></div>
      <div className="transcript" data-testid="transcript">
        {turns.length === 0 && <p className="muted small">The receptionist greets you when the call connects.</p>}
        {turns.map((t) => (
          <div key={t.index} style={{ display: "contents" }}>
            {t.userText && <div className="bubble caller">{t.userText}</div>}
            {t.agentText && (
              <div className="bubble">
                {t.agentText}
                <div className="meta">
                  {t.latency ? `${ms(t.latency.perceived_delay)} to reply · ` : ""}
                  {usd(t.cost, 5)}
                  {t.interrupted ? " · interrupted" : ""}
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </section>
  );
}
