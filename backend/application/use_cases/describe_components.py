from __future__ import annotations

from dataclasses import dataclass, replace

from backend.application.ports.component_catalogue import (
    Component,
    ComponentKind,
    EngineCatalogue,
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

    def __init__(self, catalogue: EngineCatalogue, engines: RunningEngines | None) -> None:
        self._catalogue = catalogue
        self._engines = engines

    async def execute(self, pipeline: PipelineKind) -> list[DescribedComponent]:
        return [
            await self._confirm(c)
            if pipeline is PipelineKind.SELFHOSTED and c.kind in REPORTED_BY_SERVICE
            else DescribedComponent(c, "")
            for c in self._catalogue.components(pipeline)
        ]

    async def _confirm(self, default: Component) -> DescribedComponent:
        kind = default.kind.value
        if self._engines is None:
            return DescribedComponent(
                default, f"calls are simulated; the {kind} service was not asked"
            )
        try:
            reported = await self._engines.report(default.kind)
        except EngineReportUnavailable as exc:
            return DescribedComponent(
                default,
                f"the {kind} service did not report what it runs ({exc}); "
                "listed from the catalogue's default",
            )
        match reported:
            case [engine]:
                entry = self._catalogue.engine(default.kind, engine.id)
                if entry is None:
                    return DescribedComponent(
                        default,
                        f"the {kind} service runs {engine.id!r}, which has no catalogue entry; "
                        "listed from the catalogue's default",
                    )
                return DescribedComponent(replace(entry, model=engine.model), "")
            case _:
                ids = ", ".join(repr(e.id) for e in reported) or "none"
                return DescribedComponent(
                    default,
                    f"the {kind} service reports engines {ids}, and this lists one per kind; "
                    "listed from the catalogue's default",
                )
