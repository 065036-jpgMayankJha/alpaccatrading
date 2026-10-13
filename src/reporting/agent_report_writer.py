"""Persist per-agent Markdown reports and deterministic JSON run summaries."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Mapping

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_-]+")


class AgentReportWriter:
    """Write unique, Windows-compatible report files beneath a report folder."""

    def __init__(self, reports_dir: str | Path = "run_reports") -> None:
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def new_run_id() -> str:
        return uuid.uuid4().hex

    def write_run_summary(self, summary: Mapping[str, object]) -> Path:
        """Write the machine-readable run summary using strict JSON encoding."""
        run_id = str(summary.get("run_id") or self.new_run_id())
        path = self.reports_dir / f"runSummary-{_SAFE_NAME.sub('', run_id)}.json"
        self._write_atomic(path, json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        return path

    def write_agent_report(
        self,
        *,
        run_id: str,
        timestamp: datetime,
        agent_name: str,
        provider: str,
        model: str,
        symbols: list[str],
        signals: Mapping[str, object],
        quality_warnings: list[str],
        conclusion: str,
        errors: list[str],
        strategy: Mapping[str, object],
        mode: str,
    ) -> Path:
        """Write one agent report; filenames contain no Windows-reserved colon."""
        filename_agent = _SAFE_NAME.sub("", agent_name) or "Agent"
        unique_id = _SAFE_NAME.sub("", run_id)[:10]
        path = self.reports_dir / (
            f"{filename_agent}{timestamp:%d%m%Y-%H-%M}-{unique_id}.md"
        )
        symbol_list = ", ".join(symbols) if symbols else "None"
        lines = [
            f"# {agent_name} report",
            "",
            f"- Run ID: `{run_id}`",
            f"- Timestamp: `{timestamp:%d/%m/%Y %H:%M:%S %Z}`",
            f"- Agent: {agent_name}",
            f"- LLM provider: {provider}",
            f"- LLM model: {model}",
            f"- Mode: {mode}",
            f"- Symbols analysed: {symbol_list}",
            f"- Strategy: {strategy.get('name', 'unknown')} "
            f"(version {strategy.get('version', 'unknown')})",
            "",
            "## Deterministic signal summary",
            "",
        ]
        if signals:
            lines.extend(
                [
                    "| Symbol | Score | Action | Data timestamp | Quality |",
                    "|---|---:|---|---|---|",
                ]
            )
            for symbol, raw_signal in signals.items():
                signal = raw_signal if isinstance(raw_signal, Mapping) else {}
                score = signal.get("score")
                score_text = f"{score:.6f}" if isinstance(score, (int, float)) else "Unavailable"
                quality = str(signal.get("data_quality", "unknown")).replace("|", "\\|")
                data_timestamp = str(signal.get("data_timestamp") or "Unavailable").replace("|", "\\|")
                lines.append(
                    f"| {symbol} | {score_text} | {signal.get('action', 'Unavailable')} "
                    f"| {data_timestamp} | {quality} |"
                )
        else:
            lines.append("No deterministic symbol signals were available.")
        lines.extend(["", "## Data-quality warnings", ""])
        if quality_warnings:
            lines.extend(f"- {warning}" for warning in quality_warnings)
        else:
            lines.append("- None reported.")
        lines.extend(["", "## Agent conclusion", "", conclusion or "No conclusion was produced."])
        lines.extend(
            [
                "",
                "## Strategy configuration",
                "",
                "```json",
                json.dumps(strategy, indent=2, ensure_ascii=False, allow_nan=False),
                "```",
            ]
        )
        lines.extend(["", "## Errors", ""])
        if errors:
            lines.extend(f"- {error}" for error in errors)
        else:
            lines.append("- None.")
        lines.extend(
            [
                "",
                "> This report is for research only. It does not represent an order or executed recommendation.",
                "",
            ]
        )
        self._write_atomic(path, "\n".join(lines))
        return path

    @staticmethod
    def _write_atomic(path: Path, contents: str) -> None:
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(contents, encoding="utf-8", newline="\n")
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()
