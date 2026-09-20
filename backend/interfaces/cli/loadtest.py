"""Load harness: concurrent synthetic callers replaying fixture audio.

  sweep  cost/min and p95 at rising concurrency, until p95 crosses 900 ms
  run    one concurrency level
  wer    STT word error rate on the WER corpus, per pipeline and language

Examples:
  uv run python -m backend.interfaces.cli.loadtest sweep --pipeline selfhosted
  uv run python -m backend.interfaces.cli.loadtest sweep --pipeline simulated --duration 30
  uv run python -m backend.interfaces.cli.loadtest run --pipeline api -c 1 --duration 60
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from backend.application.ports.metrics_sink import MetricsSink
from backend.application.services.endpointing import EndpointerKind
from backend.application.use_cases.describe_components import DescribeComponents
from backend.domain.value_objects.pipeline_kind import PipelineKind
from backend.infrastructure.config.settings import REPO_ROOT, Settings, get_settings
from backend.infrastructure.persistence.postgres_call_repository import SqlCallRepository
from backend.infrastructure.simulated.gpu_model import SimulatedGpu
from backend.infrastructure.telemetry.composite_metrics_sink import CompositeMetricsSink
from backend.infrastructure.telemetry.http_metrics_sink import HttpMetricsSink
from backend.interfaces.cli.harness import provenance, report
from backend.interfaces.cli.harness.caller import load_conversation
from backend.interfaces.cli.harness.runner import LevelRunner
from backend.interfaces.container import Container, build_container, warn_over_budget
from backend.interfaces.http.serializers import telephony_quote

log = logging.getLogger("loadtest")
RESULTS = REPO_ROOT / "results"
FIXTURES = REPO_ROOT / "fixtures"
DEFAULT_LEVELS = [1, 5, 10, 20, 40]
BEYOND_LEVELS = [60, 80, 120, 160]


class _Discard:
    async def emit(self, event: object) -> None:
        return None


def _container(settings: Settings, forward: bool, ceiling: int) -> Container:
    sinks: list[MetricsSink] = [_Discard()]  # type: ignore[list-item]
    if forward:
        sinks.append(HttpMetricsSink(settings.api_base_url, settings.internal_token))
    from backend.application.services.concurrency_supervisor import ConcurrencySupervisor

    supervisors = {PipelineKind.SELFHOSTED: ConcurrencySupervisor(ceiling)}
    return build_container(settings, CompositeMetricsSink(sinks), supervisors=supervisors)


async def _sweep(args: argparse.Namespace) -> dict[str, Any]:
    settings = get_settings()
    simulated = args.pipeline == "simulated"
    kind = PipelineKind.SELFHOSTED if simulated else PipelineKind(args.pipeline)
    conversation = load_conversation(FIXTURES, args.conversation)
    levels = args.levels or DEFAULT_LEVELS
    if args.until_break:
        levels = levels + [lv for lv in BEYOND_LEVELS if lv > max(levels)]
    ceiling = max(levels)
    container = _container(settings, args.forward, ceiling)
    # A simulated run never touches the STT/TTS services, so asking them would record
    # engines that took no part in the numbers.
    describer = (
        DescribeComponents(container.catalogue, None)
        if simulated
        else container.describe_components
    )
    gpu = SimulatedGpu() if simulated else None
    runner = LevelRunner(container, kind, conversation, args.endpointer, simulated=gpu)

    # Asked before the first level, not after the last: the point of the budget is to be
    # read while the pod is still cheap to stop.
    components = await describer.execute(kind)
    budget = container.gpu_budget.execute(
        [d.component for d in components],
        reserved={d.component.id: d.gpu_fraction for d in components if d.gpu_fraction},
    )
    warn_over_budget(budget, settings.eu_only)

    carriers = container.rates.telephony_quotes()
    results: list[dict[str, Any]] = []
    try:
        for level in levels:
            log.info("level %d: %ds of %d concurrent calls", level, args.duration, level)
            run = await runner.run(level, args.duration)
            summary = report.summarise_level(run, conversation, simulated, carriers)
            results.append(summary)
            log.info(
                "  %d calls, $%.4f/min, e2e p95 %.0f ms, perceived p95 %.0f ms%s",
                summary["calls_completed"],
                summary["cost_per_minute_usd"],
                summary["end_to_end_p95_ms"],
                summary["perceived_delay_p95_ms"],
                "" if summary["harness_valid"] else "  (HARNESS FELL BEHIND: invalid)",
            )
            if args.until_break and summary["end_to_end_p95_ms"] >= 900 and level >= 40:
                break
    finally:
        if isinstance(container.repository, SqlCallRepository):
            await container.repository.dispose()

    quotes = [
        {
            "provider": q.provider,
            "sku": q.sku,
            "region": q.region,
            "hourly_usd": float(q.hourly_usd),
        }
        for q in container.rates.client_gpu_quotes()
    ]
    return {
        "schema": 1,
        "provenance": provenance.collect(
            settings,
            args.pipeline,
            conversation,
            args.endpointer,
            simulated,
            container.rates.raw(),
            components,
            budget,
            container.catalogue.version(),
        ),
        "budgets_ms": report.budgets(),
        "levels": results,
        "breaking_point": report.breaking_point(results),
        "utilisation_curve": report.utilisation_curve(
            results, container.calculator.rate_card, quotes
        )
        if kind is PipelineKind.SELFHOSTED
        else [],
        "client_gpu_quotes": quotes,
        "telephony_quotes": [telephony_quote(q) for q in carriers],
    }


def _write(result: dict[str, Any], out: Path, baseline: Path | None) -> None:
    if baseline is not None and baseline.is_file():
        base = json.loads(baseline.read_text())
        first = base["levels"][0] if base.get("levels") else None
        result["baseline"] = {"provenance": base.get("provenance"), "level": first}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")


async def _wer(args: argparse.Namespace) -> dict[str, Any]:
    from backend.interfaces.cli.harness.wer import run_wer

    return await run_wer(get_settings(), FIXTURES, args.pipeline, args.language, args.limit)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="loadtest", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--pipeline", choices=["api", "selfhosted", "simulated"], required=True)
        p.add_argument("--conversation", default="intake_en")
        p.add_argument("--duration", type=int, default=120, help="seconds per level")
        p.add_argument(
            "--endpointer",
            type=EndpointerKind,
            choices=list(EndpointerKind),
            default=EndpointerKind.SEMANTIC,
        )
        p.add_argument(
            "--forward",
            action="store_true",
            help="forward live metrics to the API (Grafana sees the run)",
        )
        p.add_argument("--out", type=Path)
        p.add_argument("--baseline", type=Path, help="api run to embed as the baseline")

    sweep = sub.add_parser("sweep")
    common(sweep)
    sweep.add_argument("--levels", type=lambda s: [int(x) for x in s.split(",")])
    sweep.add_argument("--no-until-break", dest="until_break", action="store_false")

    run = sub.add_parser("run")
    common(run)
    run.add_argument("-c", "--concurrency", type=int, default=1)

    wer = sub.add_parser("wer")
    wer.add_argument("--pipeline", choices=["api", "selfhosted"], required=True)
    wer.add_argument("--language", choices=["en", "de"], default="de")
    wer.add_argument("--limit", type=int)
    wer.add_argument("--out", type=Path)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stderr)

    if args.command == "wer":
        result = asyncio.run(_wer(args))
        _write(result, args.out or RESULTS / f"wer_{args.pipeline}_{args.language}.json", None)
        return
    if args.command == "run":
        args.levels = [args.concurrency]
        args.until_break = False
    result = asyncio.run(_sweep(args))
    default = RESULTS / (
        "benchmark.json" if args.command == "sweep" else f"run_{args.pipeline}.json"
    )
    _write(result, args.out or default, args.baseline)


if __name__ == "__main__":
    main()
