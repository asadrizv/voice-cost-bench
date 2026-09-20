"use client";

import type { Pipeline } from "@/lib/api";
import { PIPELINE_LABEL } from "@/lib/format";

export function PipelineToggle({
  value,
  options,
  onChange,
  disabled,
}: {
  value: Pipeline;
  options: readonly Pipeline[];
  onChange: (p: Pipeline) => void;
  disabled?: boolean;
}) {
  return (
    <div className="segmented" role="group" aria-label="Pipeline for the next call">
      {options.map((p) => (
        <button key={p} type="button" aria-pressed={value === p} disabled={disabled}
          onClick={() => onChange(p)} data-testid={`pipeline-${p}`}>
          {PIPELINE_LABEL[p]}
        </button>
      ))}
    </div>
  );
}
