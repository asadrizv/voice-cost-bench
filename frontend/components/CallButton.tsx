"use client";

export function CallButton({
  active,
  connecting,
  onStart,
  onEnd,
}: {
  active: boolean;
  connecting: boolean;
  onStart: () => void;
  onEnd: () => void;
}) {
  if (active) {
    return (
      <button type="button" className="btn danger" onClick={onEnd} data-testid="end-call">
        End call
      </button>
    );
  }
  return (
    <button type="button" className="btn" onClick={onStart} disabled={connecting} data-testid="start-call">
      {connecting ? "Connecting…" : "Call"}
    </button>
  );
}
