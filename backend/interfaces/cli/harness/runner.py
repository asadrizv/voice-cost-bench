from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field

from backend.application.ports.endpoint_detector import ReportsDecisions
from backend.application.ports.pipeline_provider import Pipeline
from backend.application.services.concurrency_supervisor import CapacityExceeded
from backend.application.services.endpointing import EndpointerKind
from backend.application.use_cases.start_call import BudgetExceeded
from backend.domain.entities.call import Call
from backend.domain.value_objects.pipeline_kind import PipelineKind
from backend.infrastructure.simulated.gpu_model import (
    SimulatedGpu,
    SimulatedLlm,
    SimulatedStt,
    SimulatedTts,
)
from backend.infrastructure.transport.paced_output import PacedAudioOutput
from backend.interfaces.cli.harness.caller import Conversation, SyntheticCaller
from backend.interfaces.container import Container

log = logging.getLogger("loadtest")


@dataclass
class LevelRun:
    concurrency: int
    duration_s: float
    calls: list[Call] = field(default_factory=list)
    failed: int = 0
    rejected: int = 0
    reply_timeouts: int = 0
    lateness_ms: list[float] = field(default_factory=list)
    caller_observed_ms: list[float] = field(default_factory=list)
    endpoint_inference_ms: list[float] = field(default_factory=list)
    endpoint_failures: int = 0
    unanswered_turns: int = 0
    wall_s: float = 0.0


class LevelRunner:
    """Holds `concurrency` synthetic calls open for `duration_s`: each caller starts a new
    call as soon as its previous one hangs up, so the GPU sees steady load. Calls still in
    progress at the deadline are allowed to finish and are counted."""

    def __init__(
        self,
        container: Container,
        pipeline: PipelineKind,
        conversation: Conversation,
        endpointer: EndpointerKind,
        simulated: SimulatedGpu | None = None,
    ) -> None:
        self._c = container
        self._kind = pipeline
        self._conv = conversation
        self._endpointer = endpointer
        self._simulated = simulated

    async def run(self, concurrency: int, duration_s: float) -> LevelRun:
        result = LevelRun(concurrency, duration_s)
        deadline = time.monotonic() + duration_s
        started = time.monotonic()
        await asyncio.gather(*(self._caller_loop(i, deadline, result) for i in range(concurrency)))
        result.wall_s = time.monotonic() - started
        return result

    async def _caller_loop(self, index: int, deadline: float, result: LevelRun) -> None:
        # Stagger starts so callers don't all speak in lockstep.
        await asyncio.sleep(random.uniform(0, min(2.0, 0.1 * index)))
        while time.monotonic() < deadline:
            try:
                ctx = await self._c.start_call.execute(
                    self._kind, self._conv.persona, source="loadtest"
                )
            except (CapacityExceeded, BudgetExceeded) as exc:
                result.rejected += 1
                log.warning("caller %d rejected: %s", index, exc)
                if isinstance(exc, BudgetExceeded):
                    return
                await asyncio.sleep(1)
                continue
            if self._simulated is not None:
                gpu = self._simulated
                ctx.pipeline = Pipeline(
                    ctx.pipeline.kind,
                    SimulatedStt(gpu, self._conv.turns),
                    SimulatedLlm(gpu),
                    SimulatedTts(gpu),
                    uses_gpu=True,
                )
            output = PacedAudioOutput(self._c.clock)
            detector = self._c.endpointer(self._endpointer)
            session = self._c.session(ctx, output, detector)
            caller = SyntheticCaller(self._conv, session, output)
            try:
                call = await session.run(caller.frames())
                result.calls.append(call)
                result.caller_observed_ms.extend(output.caller_observed_ms)
                result.unanswered_turns += output.unanswered_turns
            except Exception:
                log.exception("call %s failed", ctx.call.id)
                result.failed += 1
            if isinstance(detector, ReportsDecisions):
                result.endpoint_inference_ms.extend(detector.inference_ms)
                result.endpoint_failures += detector.failures
            result.reply_timeouts += caller.timeouts
            result.lateness_ms.extend(caller.pacer.lateness_ms)
