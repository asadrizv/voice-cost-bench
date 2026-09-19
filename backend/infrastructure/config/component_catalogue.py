from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from backend.application.ports.component_catalogue import Component, ComponentKind
from backend.domain.value_objects.pipeline_kind import PipelineKind


class ComponentCatalogueError(ValueError):
    pass


@dataclass(frozen=True)
class ConfiguredComponent:
    """What the running configuration selects for one slot. model and version, when set,
    come from configuration and must not also appear in the catalogue entry."""

    id: str
    model: str | None = None
    version: str | None = None


Selection = Mapping[PipelineKind, Mapping[ComponentKind, ConfiguredComponent]]

_REQUIRED = ("vendor", "model", "version", "region", "licence")


class YamlComponentCatalogue:
    """Reads config/components.yaml once and resolves it against the configured selection,
    so a configured component with no entry, or an incomplete entry, fails at startup."""

    def __init__(self, path: Path, selection: Selection) -> None:
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

    def version(self) -> int:
        return self._version

    def components(self, kind: PipelineKind) -> list[Component]:
        return list(self._components[kind])


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
