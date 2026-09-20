from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace

from backend.application.ports.component_catalogue import (
    Component,
    ComponentCatalogue,
    ComponentKind,
)
from backend.application.ports.running_engines import EngineReportUnavailable, RunningEngines
from backend.domain.value_objects.pipeline_kind import PipelineKind

REPORTED_BY_SERVICE = (ComponentKind.STT, ComponentKind.TTS)


@dataclass(frozen=True)
class DescribedComponent:
    component: Component
    unconfirmed_reason: str
    """Empty when what runs is known; otherwise why the catalogue's default is listed."""

    @property
    def confirmed(self) -> bool:
        return not self.unconfirmed_reason


class DescribeComponents:
    """Every component that touches a call on a pipeline. Self-hosted STT and TTS are named
    from what their services report running, because each service picks its engine from its
    own environment; everything else is selected by this process's configuration."""

    def __init__(self, catalogue: ComponentCatalogue, engines: RunningEngines | None) -> None:
        self._catalogue = catalogue
        self._engines = engines

    async def execute(self, pipeline: PipelineKind) -> list[DescribedComponent]:
        asked = [
            self._confirm(component)
            if pipeline is PipelineKind.SELFHOSTED and component.kind in REPORTED_BY_SERVICE
            else self._as_configured(component)
            for component in self._catalogue.components(pipeline)
        ]
        # Probed together: each service has its own timeout, and asking them one after the
        # other made a cold /transparency cost the sum of those timeouts.
        return [entry for group in await asyncio.gather(*asked) for entry in group]

    async def _as_configured(self, component: Component) -> list[DescribedComponent]:
        return [DescribedComponent(component, "")]

    async def _confirm(self, default: Component) -> list[DescribedComponent]:
        """One entry per engine the service runs: a German deployment runs two TTS engines,
        and naming only one of them would misstate what processes a call."""
        kind = default.kind.value
        if self._engines is None:
            return [
                DescribedComponent(
                    default, f"calls are simulated; the {kind} service was not asked"
                )
            ]
        try:
            reported = await self._engines.report(default.kind)
        except EngineReportUnavailable as exc:
            return _default(default, f"the {kind} service did not report what it runs ({exc})")
        if not reported:
            return _default(default, f"the {kind} service reports no engine")
        entries = []
        for engine in reported:
            entry = self._catalogue.engine(default.kind, engine.id)
            if entry is None:
                return _default(
                    default, f"the {kind} service runs {engine.id!r}, which has no catalogue entry"
                )
            entries.append(DescribedComponent(replace(entry, model=engine.model), ""))
        return entries


def _default(component: Component, why: str) -> list[DescribedComponent]:
    """The catalogue's entry, standing in for a running engine that could not be named."""
    return [DescribedComponent(component, f"{why}; listed from the catalogue's default")]
