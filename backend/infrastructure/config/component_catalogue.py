from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import yaml

from backend.application.ports.component_catalogue import (
    Component,
    ComponentKind,
    EngineCatalogue,
    GpuMemoryProfile,
)
from backend.domain.services.gpu_memory_budget import BASES, GpuCapacity, MemoryClaim
from backend.domain.value_objects.pipeline_kind import PipelineKind


class ComponentCatalogueError(ValueError):
    pass


class NotEuResident(RuntimeError):
    pass


@dataclass(frozen=True)
class ConfiguredComponent:
    """What the running configuration selects for one slot. model and version, when set,
    come from configuration and must not also appear in the catalogue entry."""

    id: str
    model: str | None = None
    version: str | None = None


Selection = Mapping[PipelineKind, Mapping[ComponentKind, ConfiguredComponent]]
Engines = Mapping[ComponentKind, Collection[str]]

_REQUIRED = ("vendor", "model", "version", "region", "licence")


class YamlComponentCatalogue:
    """Reads config/components.yaml once and resolves it against the configured selection
    and every engine the self-hosted services can run, so any of those with no entry, or an
    incomplete entry, fails at startup."""

    def __init__(self, path: Path, selection: Selection, engines: Engines) -> None:
        raw = yaml.safe_load(path.read_text())
        self._version = int(raw["version"])
        hosts: dict[str, Any] = raw.get("hosts", {})
        entries: dict[str, Any] = raw["components"]
        self._components = {
            pipeline: [
                _resolve(pipeline, kind, slots[kind], entries, hosts) for kind in ComponentKind
            ]
            for pipeline, slots in selection.items()
        }
        self._engines = {
            (kind, engine_id): _resolve(
                PipelineKind.SELFHOSTED, kind, ConfiguredComponent(engine_id), entries, hosts
            )
            for kind, ids in engines.items()
            for engine_id in ids
        }
        self._gpu_memory = _gpu_memory(hosts, entries)

    def version(self) -> int:
        return self._version

    def gpu_memory(self) -> GpuMemoryProfile | None:
        return self._gpu_memory

    def components(self, kind: PipelineKind) -> list[Component]:
        return list(self._components[kind])

    def engine(self, kind: ComponentKind, engine_id: str) -> Component | None:
        return self._engines.get((kind, engine_id))


def require_eu_residency(
    catalogue: EngineCatalogue, pipelines: Collection[PipelineKind], engines: Engines
) -> None:
    """The EU-only profile's gate: raises NotEuResident naming every component a call on
    those pipelines could touch that is not EU-resident. A self-hosted service picks its
    own engine, so every engine it can run counts, not only the catalogue's default."""
    offenders: dict[str, str] = {}
    for pipeline in pipelines:
        for component in catalogue.components(pipeline):
            if component.leaves_eu:
                offenders[component.id] = _named(
                    f"{pipeline.value} {component.kind.value}", component
                )
    selfhosted = PipelineKind.SELFHOSTED
    if selfhosted in pipelines:
        for kind, ids in engines.items():
            for engine_id in ids:
                entry = catalogue.engine(kind, engine_id)
                if entry is not None and entry.leaves_eu and entry.id not in offenders:
                    offenders[entry.id] = _named(f"{selfhosted.value} {kind.value} engine", entry)
    if offenders:
        raise NotEuResident(
            f"EU_ONLY is set, but these components leave the EU: {', '.join(offenders.values())}. "
            "Select EU-resident components or unset EU_ONLY."
        )


def _named(where: str, component: Component) -> str:
    return f"{where} {component.id!r} ({component.vendor}, {component.region})"


def _resolve(
    pipeline: PipelineKind,
    kind: ComponentKind,
    configured: ConfiguredComponent,
    entries: dict[str, Any],
    hosts: dict[str, Any],
) -> Component:
    where = f"{pipeline.value} {kind.value} {configured.id!r}"
    entry = entries.get(configured.id)
    if entry is None:
        raise ComponentCatalogueError(f"{where} is configured but has no entry in components.yaml")
    if entry.get("kind") != kind.value:
        raise ComponentCatalogueError(f"{where}: entry is kind {entry.get('kind')!r}")
    values = dict(entry)
    for field, value in (("model", configured.model), ("version", configured.version)):
        if value is None:
            continue
        if field in entry:
            raise ComponentCatalogueError(
                f"{where}: {field} comes from configuration; remove it from components.yaml"
            )
        values[field] = value
    assumptions = [str(entry.get("assumption", ""))]
    if "host" in entry:
        host = hosts.get(entry["host"])
        if host is None:
            raise ComponentCatalogueError(f"{where}: unknown host {entry['host']!r}")
        values |= {"region": host["region"], "leaves_eu": host["leaves_eu"]}
        assumptions.append(str(host.get("assumption", "")))
    missing = [f for f in _REQUIRED if not values.get(f)]
    if missing or not isinstance(values.get("leaves_eu"), bool):
        raise ComponentCatalogueError(
            f"{where}: needs {', '.join(missing or ['leaves_eu (true/false)'])}"
        )
    return Component(
        kind=kind,
        id=configured.id,
        vendor=str(values["vendor"]),
        model=str(values["model"]),
        version=str(values["version"]),
        region=str(values["region"]),
        licence=str(values["licence"]),
        leaves_eu=values["leaves_eu"],
        assumption=" ".join(a.strip() for a in assumptions if a.strip()),
    )


def _gpu_memory(hosts: dict[str, Any], entries: dict[str, Any]) -> GpuMemoryProfile | None:
    """The GPU a host declares, and what every component on that host is expected to hold
    on it. None when no host has a GPU, so a deployment without one has no budget."""
    with_gpu = {name: host["gpu"] for name, host in hosts.items() if host.get("gpu")}
    if not with_gpu:
        return None
    if len(with_gpu) > 1:
        raise ComponentCatalogueError(
            f"the GPU memory budget assumes one GPU; {', '.join(sorted(with_gpu))} each name one"
        )
    host_name, gpu = next(iter(with_gpu.items()))
    where = f"host {host_name!r} gpu"
    capacity = GpuCapacity(
        sku=_text(gpu, "sku", where),
        total_gib=_gib(gpu, "total_gib", where),
        basis=_basis(gpu, where),
        source=_text(gpu, "source", where),
    )
    claims = {
        component_id: _claim(component_id, entry)
        for component_id, entry in entries.items()
        if entry.get("host") == host_name
    }
    return GpuMemoryProfile(capacity, claims)


def _claim(component_id: str, entry: dict[str, Any]) -> MemoryClaim:
    """A component on the GPU with no gpu_memory block claims an unknown amount: reading
    the gap as nothing would let an over-committed card look fine."""
    figure = entry.get("gpu_memory")
    if figure is None:
        return MemoryClaim(component_id, None, "", "")
    where = f"{component_id} gpu_memory"
    return MemoryClaim(
        component_id,
        _gib(figure, "gib", where),
        _basis(figure, where),
        _text(figure, "source", where),
    )


def _basis(node: Any, where: str) -> str:
    basis = _text(node, "basis", where)
    if basis not in BASES:
        raise ComponentCatalogueError(f"{where}: basis must be one of {', '.join(BASES)}")
    return basis


def _text(node: Any, field: str, where: str) -> str:
    if not isinstance(node, dict) or not node.get(field):
        raise ComponentCatalogueError(f"{where}: needs {field}")
    return str(node[field])


def _gib(node: dict[str, Any], field: str, where: str) -> Decimal:
    try:
        gib = Decimal(str(node[field]))
    except (KeyError, InvalidOperation) as exc:
        raise ComponentCatalogueError(f"{where}: needs {field} in GiB") from exc
    if gib <= 0:
        raise ComponentCatalogueError(f"{where}: {field} must be positive, not {gib}")
    return gib
