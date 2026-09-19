# On-premise deployment

The stack does not ship an on-premise or Kubernetes deployment yet, and won't until a
client's architecture says what to build.

## Why this is out of scope for now

The implementation plan defers it explicitly ("Phase 9 — not before a client"): an
umbrella Helm chart around `vllm/vllm-stack`, Whisper and Kokoro as plain Deployments, no
operator, no time-slicing or MIG. Building it speculatively means maintaining a chart for
a topology nobody has asked for, and every GPU service is already plain HTTP/WebSocket so
the deployment target stays open.

It is still a real differentiator (no DACH competitor offers on-prem), so it comes back
the moment a client needs it. Reopen with the client's cluster constraints.

## Prior requests

- #16: "On-premise deployment for larger firms"
