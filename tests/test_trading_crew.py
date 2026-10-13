import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import sys
import types

import pytest

from src.reporting.agent_report_writer import AgentReportWriter
from src.strategy.signal_ensemble import score_components
from trading_crew import (
    LLMSettings,
    MCPClientError,
    MCPTechnicalDataClient,
    build_argument_parser,
    calculate_deterministic_signals,
    execute_analysis,
    load_llm_settings,
    load_strategy_config,
    load_universe,
    parse_feature_tool_result,
)


def valid_tool_payload():
    return {
        "ok": True,
        "features": {
            "AAPL": {
                "ok": True,
                "indicators": {"ema_20": 110.0, "atr_pct": 1.2},
                "scores": {
                    "trend": 0.8,
                    "momentum": 0.6,
                    "relative_strength": 0.5,
                    "volume": None,
                },
            },
            "MSFT": {
                "ok": True,
                "indicators": {"ema_20": 95.0, "atr_pct": 2.3},
                "scores": {
                    "trend": -0.9,
                    "momentum": -0.8,
                    "relative_strength": -0.7,
                    "volume": -0.5,
                },
            },
        },
        "coverage": {
            "requested_symbols": ["AAPL", "MSFT"],
            "missing_symbols": [],
            "symbols": {
                "AAPL": {
                    "bar_count": 60,
                    "last_timestamp": "2026-10-13T00:00:00+00:00",
                    "errors": [],
                },
                "MSFT": {
                    "bar_count": 60,
                    "last_timestamp": "2026-10-13T00:00:00+00:00",
                    "errors": [],
                },
            },
            "benchmark": {"bar_count": 60, "errors": []},
        },
    }


class FakeMCPClient:
    def __init__(self, payload=None, error=None):
        self.payload = payload or valid_tool_payload()
        self.error = error
        self.calls = []
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        self.closed = True

    async def fetch_features(self, symbols, timeframe, lookback):
        self.calls.append((symbols, timeframe, lookback))
        if self.error:
            raise self.error
        return self.payload


def test_mcp_client_configures_current_python_and_closes_session():
    events = []
    params_seen = []

    class FakeSession:
        async def __aenter__(self):
            events.append("session_enter")
            return self

        async def __aexit__(self, *_):
            events.append("session_exit")

        async def initialize(self):
            events.append("initialize")

        async def call_tool(self, name, arguments):
            events.append((name, arguments))
            return SimpleNamespace(
                structuredContent=valid_tool_payload(),
                content=[],
                isError=False,
            )

    class FakeStdio:
        async def __aenter__(self):
            events.append("stdio_enter")
            return "read", "write"

        async def __aexit__(self, *_):
            events.append("stdio_exit")

    def stdio_factory(parameters):
        params_seen.append(parameters)
        return FakeStdio()

    client = MCPTechnicalDataClient(
        stdio_factory=stdio_factory,
        session_factory=lambda *_: FakeSession(),
        server_env={"PATH": "test-path"},
    )

    async def run():
        async with client:
            response = await client.fetch_features(["AAPL"], "1Day", "60 days")
            return response

    response = asyncio.run(run())
    parameters = params_seen[0]
    assert parameters.command == str(Path(__import__("sys").executable).resolve())
    assert parameters.args == ["-m", "servers.technical_indicators_mcp_server"]
    assert Path(parameters.cwd).is_dir()
    assert parameters.env == {"PATH": "test-path"}
    assert ("get_technical_features", {
        "symbols": ["AAPL"],
        "timeframe": "1Day",
        "lookback": "60 days",
    }) in events
    assert events[:3] == ["stdio_enter", "session_enter", "initialize"]
    assert events[3][0] == "get_technical_features"
    assert events[4:] == ["session_exit", "stdio_exit"]
    assert response["features"]["AAPL"]["status"] == "valid"
    assert client._session is None


def test_mcp_tool_errors_malformed_result_and_missing_symbols_are_handled():
    error_result = SimpleNamespace(
        structuredContent={"ok": False, "error": {"message": "credentials missing"}},
        isError=False,
        content=[],
    )
    with pytest.raises(MCPClientError, match="credentials missing"):
        parse_feature_tool_result(error_result, ["AAPL"])

    malformed = SimpleNamespace(
        structuredContent={"ok": True, "features": [], "coverage": {}},
        isError=False,
        content=[],
    )
    with pytest.raises(MCPClientError, match="features and coverage"):
        parse_feature_tool_result(malformed, ["AAPL"])

    payload = valid_tool_payload()
    payload["features"].pop("AAPL")
    payload["coverage"]["missing_symbols"] = ["AAPL"]
    result = parse_feature_tool_result(
        SimpleNamespace(structuredContent=payload, isError=False, content=[]),
        ["AAPL", "MSFT"],
    )
    assert result["features"]["AAPL"]["status"] == "missing"
    assert result["features"]["MSFT"]["status"] == "valid"


def test_parser_marks_malformed_numeric_fields_invalid():
    payload = valid_tool_payload()
    payload["features"]["AAPL"]["scores"]["trend"] = "high"
    result = parse_feature_tool_result(
        SimpleNamespace(structuredContent=payload, isError=False, content=[]),
        ["AAPL"],
    )
    assert result["features"]["AAPL"]["status"] == "invalid"
    assert "finite number" in result["features"]["AAPL"]["error"]


def test_universe_strategy_loaders_read_project_configuration():
    symbols, benchmark = load_universe()
    strategy = load_strategy_config()
    assert len(symbols) == 40
    assert benchmark == "SPY"
    assert strategy["name"] == "momentum_trend_ensemble"
    assert strategy["version"] == "v1"
    assert strategy["weights"]["news_sentiment"] == 0.15
    assert strategy["buy_threshold"] == 0.45


def test_configured_weights_pass_to_existing_score_components_and_reproduce():
    parsed = parse_feature_tool_result(
        SimpleNamespace(
            structuredContent=valid_tool_payload(),
            isError=False,
            content=[],
        ),
        ["AAPL", "MSFT"],
    )
    strategy = load_strategy_config()
    first = calculate_deterministic_signals(parsed, ["AAPL", "MSFT"], strategy)
    second = calculate_deterministic_signals(parsed, ["AAPL", "MSFT"], strategy)
    assert first == second
    expected = score_components(
        parsed["features"]["AAPL"]["scores"],
        weights=strategy["weights"],
        buy_threshold=strategy["buy_threshold"],
        sell_threshold=strategy["sell_threshold"],
        held=False,
    )
    assert first["AAPL"]["score"] == expected.score
    assert first["AAPL"]["action"] == "BUY"
    assert first["AAPL"]["effective_weights"] == expected.effective_weights
    assert first["MSFT"]["score"] < 0
    assert first["MSFT"]["action"] == "HOLD"
    assert "short position" in first["MSFT"]["reason"]
    assert "news_sentiment" not in first["AAPL"]["component_scores"]


def test_report_writer_uses_unique_windows_safe_time_and_strict_json(tmp_path):
    writer = AgentReportWriter(tmp_path)
    now = datetime(2026, 10, 14, 15, 30, tzinfo=timezone.utc)
    signals = calculate_deterministic_signals(
        parse_feature_tool_result(
            SimpleNamespace(
                structuredContent=valid_tool_payload(),
                isError=False,
                content=[],
            ),
            ["AAPL", "MSFT"],
        ),
        ["AAPL", "MSFT"],
        load_strategy_config(),
    )
    report_a = writer.write_agent_report(
        run_id="run-1234567890",
        timestamp=now,
        agent_name="TechnicalResearchAgent",
        provider="ollama",
        model="ollama/test-model",
        symbols=["AAPL"],
        signals={"AAPL": signals["AAPL"]},
        quality_warnings=[],
        conclusion="Using the supplied indicators only.",
        errors=[],
        strategy=load_strategy_config(),
        mode="crewai-commentary",
    )
    report_b = writer.write_agent_report(
        run_id="run-abcdefghij",
        timestamp=now,
        agent_name="TechnicalResearchAgent",
        provider="ollama",
        model="ollama/test-model",
        symbols=["AAPL"],
        signals={"AAPL": signals["AAPL"]},
        quality_warnings=[],
        conclusion="Using the supplied indicators only.",
        errors=[],
        strategy=load_strategy_config(),
        mode="crewai-commentary",
    )
    summary_path = writer.write_run_summary(
        {"run_id": "run-1234567890", "signals": signals}
    )
    assert report_a.name.startswith("TechnicalResearchAgent14102026-15-30-")
    assert report_a != report_b
    assert ":" not in report_a.name
    assert "15:30" in report_a.read_text(encoding="utf-8")
    assert json.loads(summary_path.read_text(encoding="utf-8"))["run_id"] == "run-1234567890"


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        ({}, None),
        (
            {"LLM_PROVIDER": "ollama", "LLM_MODEL": "qwen", "OLLAMA_BASE_URL": "http://localhost:11434"},
            ("ollama", "ollama/qwen"),
        ),
        (
            {"LLM_PROVIDER": "cloud", "LLM_MODEL": "openai/example", "LLM_API_KEY": "test-secret"},
            ("cloud", "openai/example"),
        ),
    ],
)
def test_llm_settings_are_optional_and_configurable(environment, expected):
    settings = load_llm_settings(environment)
    if expected is None:
        assert settings is None
    else:
        assert (settings.provider, settings.model) == expected


def test_llm_settings_reject_incomplete_credentials_without_printing_them():
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        load_llm_settings({"LLM_PROVIDER": "cloud", "LLM_MODEL": "openai/example"})


def test_crewai_construction_is_sequential_commentary_only(monkeypatch):
    import trading_crew

    created_agents = []
    created_tasks = []
    created_crews = []

    class FakeAgent:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            created_agents.append(self)

    class FakeTask:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            created_tasks.append(self)

    class FakeCrew:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            created_crews.append(self)

    fake_module = types.ModuleType("crewai")
    fake_module.Agent = FakeAgent
    fake_module.Task = FakeTask
    fake_module.Crew = FakeCrew
    fake_module.LLM = lambda **kwargs: SimpleNamespace(**kwargs)
    fake_module.Process = SimpleNamespace(sequential="sequential")
    monkeypatch.setitem(sys.modules, "crewai", fake_module)

    crew, tasks = trading_crew.create_commentary_crew(
        {"AAPL": {"score": 0.7, "action": "BUY"}},
        LLMSettings("ollama", "ollama/test-model", base_url="http://localhost:11434"),
    )
    assert len(created_agents) == 2
    assert len(tasks) == 2
    assert crew.process == "sequential"
    assert all(agent.allow_delegation is False for agent in created_agents)
    assert "never invent" in created_agents[0].backstory.lower()
    assert "never alter" in created_agents[1].backstory.lower()
    assert all(task.agent in created_agents for task in created_tasks)


def test_cli_validates_timeframe_lookback_and_symbol_arguments():
    parser = build_argument_parser()
    args = parser.parse_args(["--symbols", "AAPL", "MSFT", "--timeframe", "1Hour", "--lookback", "30 days"])
    assert args.symbols == ["AAPL", "MSFT"]
    assert args.timeframe == "1Hour"
    assert args.lookback == "30 days"
    with pytest.raises(SystemExit):
        parser.parse_args(["--timeframe", "daily"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--lookback", "0 days"])
    import trading_crew

    with pytest.raises(SystemExit):
        trading_crew.main(["--symbols", "AAPL", "AAPL"])


def test_deterministic_only_analysis_makes_one_batch_call_and_writes_outputs(
    tmp_path, monkeypatch
):
    import trading_crew

    parsed = parse_feature_tool_result(
        SimpleNamespace(
            structuredContent=valid_tool_payload(),
            isError=False,
            content=[],
        ),
        ["AAPL", "MSFT"],
    )
    fake_client = FakeMCPClient(parsed)
    monkeypatch.setattr(
        trading_crew,
        "create_commentary_crew",
        lambda *_: pytest.fail("Deterministic-only run started CrewAI."),
    )
    summary = asyncio.run(
        execute_analysis(
            symbols=["AAPL", "MSFT"],
            timeframe="1Day",
            lookback="90 days",
            strategy=load_strategy_config(),
            writer=AgentReportWriter(tmp_path),
            mcp_client_factory=lambda: fake_client,
            deterministic_only=True,
        )
    )
    assert fake_client.closed
    assert fake_client.calls == [(["AAPL", "MSFT"], "1Day", "90 days")]
    assert summary["mode"] == "deterministic-only"
    assert summary["signals"]["AAPL"]["action"] == "BUY"
    assert summary["signals"]["MSFT"]["action"] == "HOLD"
    assert len(list(tmp_path.glob("*.md"))) == 2
    saved = json.loads(next(tmp_path.glob("runSummary-*.json")).read_text(encoding="utf-8"))
    assert saved["signals"] == summary["signals"]


def test_failed_mcp_and_agent_runs_preserve_summary_and_completed_output(tmp_path, monkeypatch):
    import trading_crew

    payload = parse_feature_tool_result(
        SimpleNamespace(
            structuredContent=valid_tool_payload(),
            isError=False,
            content=[],
        ),
        ["AAPL", "MSFT"],
    )
    fake_client = FakeMCPClient(payload)
    writer = AgentReportWriter(tmp_path)

    class FakeTask:
        def __init__(self, raw):
            self.output = SimpleNamespace(raw=raw) if raw else None

    class FakeCrew:
        def kickoff(self):
            tasks[0].output = SimpleNamespace(raw="Completed technical commentary.")
            raise RuntimeError("review agent failed")

    tasks = [FakeTask(None), FakeTask(None)]
    monkeypatch.setattr(trading_crew, "create_commentary_crew", lambda *_: (FakeCrew(), tasks))
    summary = asyncio.run(
        execute_analysis(
            symbols=["AAPL", "MSFT"],
            timeframe="1Day",
            lookback="90 days",
            strategy=load_strategy_config(),
            writer=writer,
            mcp_client_factory=lambda: fake_client,
            llm_settings=LLMSettings("ollama", "ollama/mock"),
            now=datetime(2026, 10, 14, 15, 30, tzinfo=timezone.utc),
        )
    )
    assert fake_client.closed
    assert summary["agent_commentary"]["TechnicalResearchAgent"] == "Completed technical commentary."
    assert any("CrewAI commentary failed" in error for error in summary["errors"])
    assert len(list(tmp_path.glob("*.md"))) == 2
    report = next(tmp_path.glob("TechnicalResearchAgent*.md")).read_text(encoding="utf-8")
    assert "Completed technical commentary." in report
    json_path = next(tmp_path.glob("runSummary-*.json"))
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["signals"]["AAPL"]["action"] == summary["signals"]["AAPL"]["action"]


def test_data_errors_are_reported_without_discarding_run(tmp_path):
    fake_client = FakeMCPClient(error=RuntimeError("tool returned unavailable"))
    summary = asyncio.run(
        execute_analysis(
            symbols=["AAPL"],
            timeframe="1Day",
            lookback="90 days",
            strategy=load_strategy_config(),
            writer=AgentReportWriter(tmp_path),
            mcp_client_factory=lambda: fake_client,
            deterministic_only=True,
        )
    )
    assert fake_client.closed
    assert summary["signals"]["AAPL"]["action"] == "UNAVAILABLE"
    assert "MCP data request failed" in summary["errors"][0]
    assert len(list(tmp_path.glob("*.md"))) == 2
