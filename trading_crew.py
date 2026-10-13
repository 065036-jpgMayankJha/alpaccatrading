"""Batched technical research CLI with deterministic scores and optional commentary."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import uuid
from copy import deepcopy
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.reporting.agent_report_writer import AgentReportWriter
from src.strategy.signal_ensemble import score_components

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_UNIVERSE = PROJECT_ROOT / "config" / "universe.json"
DEFAULT_STRATEGY = PROJECT_ROOT / "config" / "strategy.yaml"
MCP_SERVER_MODULE = "servers.technical_indicators_mcp_server"
AGENTS = (
    ("TechnicalResearchAgent", "Technical Research Agent"),
    ("StrategyReviewAgent", "Strategy Review Agent"),
)


class MCPClientError(RuntimeError):
    """MCP connection, tool-call, or structured-response failure."""


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model: str
    api_key: str | None = None
    base_url: str | None = None

    @property
    def identifier(self) -> str:
        return f"{self.provider}/{self.model}"


class MCPTechnicalDataClient:
    """One-session stdio MCP client returning native structured tool results."""

    def __init__(
        self,
        *,
        project_root: Path = PROJECT_ROOT,
        python_executable: str = sys.executable,
        server_env: Mapping[str, str] | None = None,
        session_factory: Callable[..., Any] = ClientSession,
        stdio_factory: Callable[..., Any] = stdio_client,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.python_executable = str(Path(python_executable).resolve())
        self.server_env = dict(server_env) if server_env is not None else None
        self._session_factory = session_factory
        self._stdio_factory = stdio_factory
        self._stack: AsyncExitStack | None = None
        self._session: Any = None

    async def __aenter__(self) -> "MCPTechnicalDataClient":
        if self._stack is not None:
            raise MCPClientError("MCP client is already connected.")
        server_path = self.project_root / Path(*MCP_SERVER_MODULE.split(".")).with_suffix(".py")
        if not server_path.is_file():
            raise MCPClientError(f"Local MCP server not found: {server_path}")
        self._stack = AsyncExitStack()
        try:
            parameters = StdioServerParameters(
                command=self.python_executable,
                args=["-m", MCP_SERVER_MODULE],
                cwd=str(self.project_root),
                env=deepcopy(self.server_env),
            )
            read_stream, write_stream = await self._stack.enter_async_context(
                self._stdio_factory(parameters)
            )
            self._session = await self._stack.enter_async_context(
                self._session_factory(read_stream, write_stream)
            )
            await self._session.initialize()
            return self
        except Exception:
            await self._stack.aclose()
            self._stack = None
            self._session = None
            raise

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._session = None

    async def fetch_features(
        self,
        symbols: list[str],
        timeframe: str,
        lookback: str,
    ) -> dict[str, object]:
        """Fetch all requested features in one get_technical_features tool call."""
        if self._session is None:
            raise MCPClientError("MCP client must be connected before requesting features.")
        result = await self._session.call_tool(
            "get_technical_features",
            {"symbols": symbols, "timeframe": timeframe, "lookback": lookback},
        )
        return parse_feature_tool_result(result, symbols)


def _json_content(result: object) -> Mapping[str, object]:
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, Mapping):
        return structured
    content = getattr(result, "content", None)
    if isinstance(content, list):
        for item in content:
            text = getattr(item, "text", None)
            if isinstance(text, str):
                try:
                    decoded = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise MCPClientError("MCP tool returned malformed JSON text.") from exc
                if isinstance(decoded, Mapping):
                    return decoded
                raise MCPClientError("MCP tool JSON result must be an object.")
    raise MCPClientError("MCP tool returned no structured JSON result.")


def _numeric_mapping(value: object, label: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise MCPClientError(f"MCP result field {label!r} must be an object.")
    numeric: dict[str, float] = {}
    for name, raw in value.items():
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
            raise MCPClientError(f"MCP result field {label}.{name} must be a finite number.")
        numeric[str(name)] = float(raw)
    return numeric


def _score_mapping(value: object) -> dict[str, float | None]:
    if not isinstance(value, Mapping):
        raise MCPClientError("MCP result field 'scores' must be an object.")
    scores: dict[str, float | None] = {}
    for name, raw in value.items():
        if raw is None:
            scores[str(name)] = None
            continue
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
            raise MCPClientError(f"MCP result field scores.{name} must be a finite number or null.")
        if raw < -1.0 or raw > 1.0:
            raise MCPClientError("MCP component scores must be normalized to [-1, 1].")
        scores[str(name)] = float(raw)
    if not any(value is not None for value in scores.values()):
        raise MCPClientError("MCP feature entry contains no available component scores.")
    return scores


def parse_feature_tool_result(
    result: object,
    requested_symbols: list[str],
) -> dict[str, object]:
    """Validate an MCP call result without deriving anything from prose."""
    if getattr(result, "isError", False):
        raise MCPClientError("MCP get_technical_features tool returned an error.")
    payload = _json_content(result)
    if payload.get("ok") is not True:
        error = payload.get("error")
        if isinstance(error, Mapping):
            message = error.get("message", "MCP tool returned an unspecified error.")
        else:
            message = "MCP tool returned an unspecified error."
        raise MCPClientError(str(message))
    raw_features = payload.get("features")
    coverage = payload.get("coverage")
    if not isinstance(raw_features, Mapping) or not isinstance(coverage, Mapping):
        raise MCPClientError("MCP feature result must include features and coverage objects.")
    symbol_quality = coverage.get("symbols")
    if not isinstance(symbol_quality, Mapping):
        symbol_quality = {}
    missing = coverage.get("missing_symbols", [])
    if not isinstance(missing, list) or not all(isinstance(symbol, str) for symbol in missing):
        raise MCPClientError("MCP coverage missing_symbols must be a list of strings.")

    features: dict[str, object] = {}
    for requested in requested_symbols:
        symbol = requested.strip().upper()
        raw_entry = raw_features.get(symbol)
        quality = symbol_quality.get(symbol, {})
        quality_data = dict(quality) if isinstance(quality, Mapping) else {
            "errors": ["Malformed symbol quality information."]
        }
        quality_errors = quality_data.get("errors", [])
        if not isinstance(quality_errors, list) or not all(
            isinstance(message, str) for message in quality_errors
        ):
            quality_data["errors"] = ["Malformed quality error list."]
        last_timestamp = quality_data.get("last_timestamp")
        if last_timestamp is not None and not isinstance(last_timestamp, str):
            quality_data["last_timestamp"] = None
            quality_data["errors"].append("Malformed latest timestamp.")
        if symbol in missing or raw_entry is None:
            features[symbol] = {
                "status": "missing",
                "indicators": {},
                "scores": {},
                "data_timestamp": quality_data.get("last_timestamp"),
                "data_quality": quality_data,
                "error": "No feature data returned for requested symbol.",
            }
            continue
        if not isinstance(raw_entry, Mapping) or raw_entry.get("ok") is not True:
            error = raw_entry.get("error") if isinstance(raw_entry, Mapping) else None
            features[symbol] = {
                "status": "invalid",
                "indicators": {},
                "scores": {},
                "data_timestamp": quality_data.get("last_timestamp"),
                "data_quality": quality_data,
                "error": str(error or "Malformed or unavailable feature entry."),
            }
            continue
        try:
            indicators = _numeric_mapping(raw_entry.get("indicators"), "indicators")
            scores = _score_mapping(raw_entry.get("scores"))
        except MCPClientError as exc:
            features[symbol] = {
                "status": "invalid",
                "indicators": {},
                "scores": {},
                "data_timestamp": quality_data.get("last_timestamp"),
                "data_quality": quality_data,
                "error": str(exc),
            }
            continue
        features[symbol] = {
            "status": "valid",
            "indicators": indicators,
            "scores": scores,
            "data_timestamp": quality_data.get("last_timestamp"),
            "data_quality": quality_data,
            "error": None,
        }
    return {"features": features, "coverage": dict(coverage)}


def load_universe(path: str | Path = DEFAULT_UNIVERSE) -> tuple[list[str], str]:
    """Load unique default tickers and benchmark from the universe JSON file."""
    try:
        config = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to load universe file {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("Universe configuration must be a JSON object.")
    growth = config.get("growth_satellites")
    diversified = config.get("diversified")
    if not isinstance(growth, list) or not all(isinstance(item, str) for item in growth):
        raise ValueError("Universe growth_satellites must be a list of symbols.")
    if not isinstance(diversified, dict):
        raise ValueError("Universe diversified must be an object of symbol groups.")
    symbols = list(growth)
    for group, entries in diversified.items():
        if not isinstance(entries, list) or not all(isinstance(item, str) for item in entries):
            raise ValueError(f"Universe group {group!r} must be a list of symbols.")
        symbols.extend(entries)
    normalized = [symbol.strip().upper() for symbol in symbols]
    if not normalized or any(not symbol for symbol in normalized):
        raise ValueError("Universe must contain non-empty symbols.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Universe contains duplicate symbols.")
    benchmark = config.get("benchmark", "SPY")
    if not isinstance(benchmark, str) or not benchmark.strip():
        raise ValueError("Universe benchmark must be a ticker string.")
    return normalized, benchmark.strip().upper()


def load_strategy_config(path: str | Path = DEFAULT_STRATEGY) -> dict[str, object]:
    """Load and validate ensemble weights and decision thresholds from YAML."""
    try:
        config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Unable to load strategy configuration {path}: {exc}") from exc
    strategy = config.get("strategy") if isinstance(config, dict) else None
    if not isinstance(strategy, dict):
        raise ValueError("Strategy configuration must define a strategy object.")
    weights = strategy.get("weights")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("Strategy weights must be a non-empty mapping.")
    try:
        buy_threshold = float(strategy["buy_threshold"])
        sell_threshold = float(strategy["sell_threshold"])
        score_min = float(strategy["score_min"])
        score_max = float(strategy["score_max"])
        normalized_weights = {str(key): float(value) for key, value in weights.items()}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Strategy thresholds, bounds and weights must be numeric.") from exc
    if not (-1 <= sell_threshold < buy_threshold <= 1):
        raise ValueError("Strategy thresholds must satisfy -1 <= sell < buy <= 1.")
    if score_min != -1 or score_max != 1:
        raise ValueError("Strategy score bounds must be exactly [-1, 1].")
    if any(not math.isfinite(value) or value < 0 for value in normalized_weights.values()):
        raise ValueError("Strategy weights must be finite and non-negative.")
    if not any(value > 0 for value in normalized_weights.values()):
        raise ValueError("At least one strategy weight must be positive.")
    return {
        "name": str(strategy.get("name", "unnamed")),
        "version": str(strategy.get("version", "unknown")),
        "weights": normalized_weights,
        "buy_threshold": buy_threshold,
        "sell_threshold": sell_threshold,
        "score_min": score_min,
        "score_max": score_max,
    }


def calculate_deterministic_signals(
    feature_result: Mapping[str, object],
    symbols: list[str],
    strategy: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    """Score MCP feature components in Python; LLM output is not consulted."""
    feature_map = feature_result.get("features")
    if not isinstance(feature_map, Mapping):
        raise ValueError("Feature result must contain a features mapping.")
    weights = strategy["weights"]
    signals: dict[str, dict[str, object]] = {}
    for symbol in symbols:
        entry = feature_map.get(symbol, {})
        if not isinstance(entry, Mapping):
            entry = {}
        status = str(entry.get("status", "missing"))
        base: dict[str, object] = {
            "symbol": symbol,
            "indicators": dict(entry.get("indicators", {}))
            if isinstance(entry.get("indicators"), Mapping)
            else {},
            "component_scores": dict(entry.get("scores", {}))
            if isinstance(entry.get("scores"), Mapping)
            else {},
            "data_timestamp": entry.get("data_timestamp"),
            "data_quality": status,
            "quality_details": entry.get("data_quality", {}),
            "effective_weights": {},
            "score": None,
            "action": "UNAVAILABLE",
            "reason": None,
            "errors": [],
        }
        if status != "valid":
            base["errors"] = [str(entry.get("error", f"Feature data status is {status}."))]
            signals[symbol] = base
            continue
        try:
            result = score_components(
                entry["scores"],
                weights=weights,
                buy_threshold=float(strategy["buy_threshold"]),
                sell_threshold=float(strategy["sell_threshold"]),
                held=False,
            )
        except (KeyError, TypeError, ValueError) as exc:
            base["data_quality"] = "invalid"
            base["errors"] = [f"Unable to score feature components: {exc}"]
            signals[symbol] = base
            continue
        base.update(
            {
                "effective_weights": result.effective_weights,
                "score": result.score,
                "action": result.action,
                "reason": result.reason,
                "component_scores": result.components_used,
            }
        )
        signals[symbol] = base
    return signals


def load_llm_settings(environ: Mapping[str, str] | None = None) -> LLMSettings | None:
    """Read optional Ollama or LiteLLM-compatible cloud configuration."""
    env = environ if environ is not None else os.environ
    provider = env.get("LLM_PROVIDER", "").strip().lower()
    if not provider:
        if env.get("LLM_MODEL", "").strip() or env.get("LLM_API_KEY", "").strip():
            raise ValueError("Set LLM_PROVIDER when configuring an LLM.")
        return None
    model = env.get("LLM_MODEL", "").strip()
    if not model:
        raise ValueError("LLM_MODEL is required when LLM_PROVIDER is set.")
    if provider == "ollama":
        return LLMSettings(
            provider="ollama",
            model=model if model.startswith("ollama/") else f"ollama/{model}",
            base_url=env.get("OLLAMA_BASE_URL", "http://localhost:11434").strip(),
        )
    if provider == "cloud":
        api_key = env.get("LLM_API_KEY", "").strip()
        if not api_key:
            raise ValueError("LLM_API_KEY is required for LLM_PROVIDER=cloud.")
        if "/" not in model:
            raise ValueError("Cloud LLM_MODEL must be provider-qualified, e.g. openai/gpt-4o-mini.")
        return LLMSettings(provider="cloud", model=model, api_key=api_key)
    raise ValueError("LLM_PROVIDER must be 'ollama' or 'cloud'.")


def create_commentary_crew(
    signals: Mapping[str, Mapping[str, object]],
    settings: LLMSettings,
) -> tuple[object, list[object]]:
    """Create sequential commentary agents with deterministic inputs only."""
    from crewai import Agent, Crew, LLM, Process, Task

    llm_options: dict[str, object] = {"model": settings.model}
    if settings.api_key:
        llm_options["api_key"] = settings.api_key
    if settings.base_url:
        llm_options["base_url"] = settings.base_url
    llm = LLM(**llm_options)
    payload = json.dumps(signals, indent=2, ensure_ascii=False, allow_nan=False)
    technical_agent = Agent(
        role="Technical Research Agent",
        goal="Explain only the supplied, structured technical indicators and scores.",
        backstory=(
            "You are a disciplined quantitative research assistant. Never invent "
            "missing values or recalculate numerical indicators. Mention conflicts "
            "and data quality explicitly. You do not recommend or submit orders."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=False,
        max_iter=2,
    )
    review_agent = Agent(
        role="Strategy Review Agent",
        goal="Summarize deterministic Python signal results without changing them.",
        backstory=(
            "You explain signal evidence and identify strong or conflicting inputs. "
            "Never alter a score or action, invent news, or claim that a candidate "
            "was executed. Python owns every numeric decision."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=False,
        max_iter=2,
    )
    technical_task = Task(
        description=(
            "Review these deterministic per-symbol values. Interpret only the "
            "provided indicators, normalized component scores, actions and data "
            "quality. Do not calculate or change numbers.\n\n"
            f"STRUCTURED SIGNALS:\n{payload}"
        ),
        expected_output=(
            "A concise evidence-based research note that cites supplied values "
            "and names missing/conflicting data."
        ),
        agent=technical_agent,
    )
    review_task = Task(
        description=(
            "Explain the deterministic candidates below, emphasizing the highest "
            "scores and conflicts. Preserve every score and action exactly; this "
            "text is commentary and does not create orders.\n\n"
            f"STRUCTURED SIGNALS:\n{payload}"
        ),
        expected_output=(
            "A concise review of the deterministic ranking, evidence conflicts, "
            "and important data-quality warnings."
        ),
        agent=review_agent,
    )
    crew = Crew(
        agents=[technical_agent, review_agent],
        tasks=[technical_task, review_task],
        process=Process.sequential,
        verbose=False,
    )
    return crew, [technical_task, review_task]


def _task_output(task: object) -> str:
    output = getattr(task, "output", None)
    raw = getattr(output, "raw", None)
    return str(raw if raw is not None else output) if output is not None else ""


def _redact_secrets(message: str, extra_secrets: tuple[str, ...] = ()) -> str:
    secrets = list(extra_secrets)
    for name in (
        "APCA_API_KEY_ID",
        "APCA_API_SECRET_KEY",
        "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY",
        "LLM_API_KEY",
    ):
        value = os.getenv(name)
        if value:
            secrets.append(value)
    for secret in secrets:
        if secret:
            message = message.replace(secret, "[REDACTED]")
    return message


async def execute_analysis(
    *,
    symbols: list[str],
    timeframe: str,
    lookback: str,
    strategy: Mapping[str, object],
    writer: AgentReportWriter,
    mcp_client_factory: Callable[[], Any] = MCPTechnicalDataClient,
    llm_settings: LLMSettings | None = None,
    deterministic_only: bool = False,
    now: datetime | None = None,
) -> dict[str, object]:
    """Fetch once, score in Python, persist deterministic summary and reports."""
    timestamp = now or datetime.now().astimezone()
    run_id = writer.new_run_id()
    mode = "deterministic-only" if deterministic_only or llm_settings is None else "crewai-commentary"
    provider = llm_settings.provider if llm_settings else "not configured"
    model = llm_settings.model if llm_settings else "none"
    errors: list[str] = []
    warnings: list[str] = []
    tool_result: dict[str, object] = {"features": {}, "coverage": {}}
    try:
        async with mcp_client_factory() as client:
            tool_result = await client.fetch_features(symbols, timeframe, lookback)
    except Exception as exc:
        errors.append(
            "MCP data request failed: "
            f"{type(exc).__name__}: {_redact_secrets(str(exc))}"
        )

    try:
        signals = calculate_deterministic_signals(tool_result, symbols, strategy)
    except Exception as exc:
        errors.append(f"Deterministic scoring failed: {type(exc).__name__}: {exc}")
        signals = {
            symbol: {
                "symbol": symbol,
                "indicators": {},
                "component_scores": {},
                "data_timestamp": None,
                "data_quality": "invalid",
                "quality_details": {},
                "effective_weights": {},
                "score": None,
                "action": "UNAVAILABLE",
                "reason": None,
                "errors": [str(exc)],
            }
            for symbol in symbols
        }

    for symbol, signal in signals.items():
        details = signal.get("quality_details")
        if signal["data_quality"] != "valid":
            warnings.append(f"{symbol}: {signal['data_quality']} feature data: {'; '.join(signal['errors'])}")
        if isinstance(details, Mapping):
            for message in details.get("errors", []):
                warning = f"{symbol}: {message}"
                if warning not in warnings:
                    warnings.append(warning)

    serializable_strategy = dict(strategy)
    summary: dict[str, object] = {
        "run_id": run_id,
        "timestamp": timestamp.isoformat(),
        "mode": mode,
        "symbols_analysed": symbols,
        "timeframe": timeframe,
        "lookback": lookback,
        "strategy": serializable_strategy,
        "signals": signals,
        "coverage": tool_result.get("coverage", {}),
        "data_quality_warnings": warnings,
        "agent_commentary": {},
        "errors": errors,
    }
    writer.write_run_summary(summary)

    commentary: dict[str, str] = {}
    if llm_settings is not None and not deterministic_only and not errors:
        try:
            crew, tasks = create_commentary_crew(signals, llm_settings)
            crew.kickoff()
            commentary = {
                agent_key: _task_output(task)
                for (agent_key, _), task in zip(AGENTS, tasks)
            }
            if any(not result.strip() for result in commentary.values()):
                raise RuntimeError("CrewAI returned an empty agent output.")
        except Exception as exc:
            errors.append(
                "CrewAI commentary failed: "
                f"{type(exc).__name__}: "
                f"{_redact_secrets(str(exc), (llm_settings.api_key or '',))}"
            )
            for (agent_key, _), task in zip(AGENTS, locals().get("tasks", [])):
                produced = _task_output(task)
                if produced.strip():
                    commentary[agent_key] = produced
    elif llm_settings is None and not deterministic_only:
        mode = "deterministic-only (LLM not configured)"
        summary["mode"] = mode

    summary["agent_commentary"] = commentary
    summary["errors"] = errors
    report_paths: dict[str, str] = {}
    for agent_key, agent_title in AGENTS:
        conclusion = commentary.get(
            agent_key,
            "LLM commentary was not run; all numeric decisions above were produced by deterministic Python scoring.",
        )
        agent_errors = list(errors)
        report_path = writer.write_agent_report(
            run_id=run_id,
            timestamp=timestamp,
            agent_name=agent_key,
            provider=provider,
            model=model,
            symbols=symbols,
            signals=signals,
            quality_warnings=warnings,
            conclusion=conclusion,
            errors=agent_errors,
            strategy=serializable_strategy,
            mode=mode,
        )
        report_paths[agent_key] = str(report_path)
    summary["agent_reports"] = report_paths
    writer.write_run_summary(summary)
    return summary


def _validate_timeframe(value: str) -> str:
    from src.data.alpaca_market_data import AlpacaMarketDataClient

    try:
        return AlpacaMarketDataClient._validate_timeframe(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _validate_lookback(value: str) -> str:
    from src.data.alpaca_market_data import AlpacaMarketDataClient

    try:
        AlpacaMarketDataClient._parse_lookback(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return value


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deterministic technical research with optional CrewAI commentary."
    )
    parser.add_argument("--symbols", nargs="+", help="Symbols to analyse; defaults to config/universe.json.")
    parser.add_argument("--timeframe", type=_validate_timeframe, default="1Day")
    parser.add_argument("--lookback", type=_validate_lookback, default="90 days")
    parser.add_argument("--universe-file", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--strategy-config", type=Path, default=DEFAULT_STRATEGY)
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "run_reports")
    parser.add_argument(
        "--deterministic-only",
        action="store_true",
        help="Skip optional CrewAI text commentary; Python scoring still runs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        if args.symbols:
            symbols = [symbol.strip().upper() for symbol in args.symbols]
            if any(not symbol for symbol in symbols) or len(set(symbols)) != len(symbols):
                raise ValueError("--symbols must contain unique, non-empty ticker names.")
        else:
            symbols, _ = load_universe(args.universe_file)
        strategy = load_strategy_config(args.strategy_config)
        llm_settings = None if args.deterministic_only else load_llm_settings()
        writer = AgentReportWriter(args.reports_dir)
        summary = asyncio.run(
            execute_analysis(
                symbols=symbols,
                timeframe=args.timeframe,
                lookback=args.lookback,
                strategy=strategy,
                writer=writer,
                llm_settings=llm_settings,
                deterministic_only=args.deterministic_only,
            )
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "run_id": summary["run_id"],
                "mode": summary["mode"],
                "symbols_analysed": summary["symbols_analysed"],
                "reports": summary.get("agent_reports", {}),
                "errors": summary["errors"],
            },
            indent=2,
        )
    )
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
