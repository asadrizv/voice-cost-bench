"use client";

import { useEffect, useState } from "react";
import { api, type LiveEvent } from "./api";

export interface LiveTurn {
  index: number;
  userText: string;
  agentText: string;
  interrupted: boolean;
  latency: LiveEvent["latency"];
  cost: number;
}

export interface LiveState {
  connected: boolean;
  ended: boolean;
  last: LiveEvent | null;
  turns: LiveTurn[];
}

const EMPTY: LiveState = { connected: false, ended: false, last: null, turns: [] };

/** Subscribes to the API's SSE stream for one call. Replays from the call's start, so a
 *  late subscription still sees the greeting. */
export function useLiveMetrics(callId: string | null): LiveState {
  const [state, setState] = useState<LiveState>(EMPTY);

  useEffect(() => {
    if (!callId) return;
    setState({ ...EMPTY });
    const source = new EventSource(api.liveUrl(callId));
    const onEvent = (raw: MessageEvent<string>) => {
      const event = JSON.parse(raw.data) as LiveEvent;
      setState((prev) => {
        const turns =
          event.kind === "turn" && event.turn_index != null
            ? [
                ...prev.turns.filter((t) => t.index !== event.turn_index),
                {
                  index: event.turn_index,
                  userText: event.user_text,
                  agentText: event.agent_text,
                  interrupted: event.interrupted,
                  latency: event.latency,
                  cost: Number(event.turn_cost?.total ?? 0),
                },
              ].sort((a, b) => a.index - b.index)
            : prev.turns;
        return { connected: true, ended: prev.ended || event.kind === "ended", last: event, turns };
      });
      if (event.kind === "ended") source.close();
    };
    for (const kind of ["started", "turn", "tick", "ended"]) {
      source.addEventListener(kind, onEvent as EventListener);
    }
    source.onopen = () => setState((s) => ({ ...s, connected: true }));
    return () => source.close();
  }, [callId]);

  return state;
}
