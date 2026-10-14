"""Guarded paper-account inspection and explicit human-approved order CLI."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from src.execution.alpaca_paper_executor import AlpacaPaperExecutor, PaperExecutionError
from src.execution.proposal_engine import (
    ProposalValidationError,
    append_submission_ledger,
    build_proposals,
    update_submission_ledger,
    write_audit_report,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_REPORTS = PROJECT_ROOT / "run_reports"
DEFAULT_UNIVERSE = PROJECT_ROOT / "config" / "universe.json"
DEFAULT_STRATEGY = PROJECT_ROOT / "config" / "strategy.yaml"
DEFAULT_LOGS = PROJECT_ROOT / "logs"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect and safely propose Alpaca paper-account orders.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--account-only", action="store_true", help="Retrieve and print a sanitized account snapshot only.")
    modes.add_argument("--dry-run", action="store_true", help="Validate and report proposals without submitting (default).")
    modes.add_argument("--submit-approved", action="store_true", help="Require exact human approval for each risk-approved paper order.")
    parser.add_argument("--run-summary", type=Path, help="JSON signal run summary; defaults to the latest run_reports summary.")
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--universe-file", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--strategy-file", type=Path, default=DEFAULT_STRATEGY)
    parser.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS)
    return parser


def latest_run_summary(reports_dir: str | Path = DEFAULT_REPORTS) -> Path:
    candidates = list(Path(reports_dir).glob("runSummary-*.json"))
    if not candidates:
        raise ProposalValidationError(f"No JSON run summaries were found in {reports_dir}.")
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


def _print_json(value: object, print_fn: Callable[[str], None]) -> None:
    print_fn(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def main(
    argv: Sequence[str] | None = None,
    *,
    executor_factory: Callable[[], AlpacaPaperExecutor] = AlpacaPaperExecutor,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        executor = executor_factory()
        snapshot = executor.get_account_snapshot()
        if args.account_only:
            report = {
                "mode": "account-only",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "account_snapshot": snapshot,
                "orders_submitted": [],
            }
            audit_path = write_audit_report(report, args.logs_dir)
            _print_json({"account_snapshot": snapshot, "audit_report": str(audit_path)}, print_fn)
            return 0

        summary_path = args.run_summary or latest_run_summary(args.reports_dir)
        run_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(run_summary, dict):
            raise ProposalValidationError("Run summary must be a JSON object.")
        ledger_path = args.logs_dir / "paper_order_ledger.json"
        report = build_proposals(
            run_summary,
            snapshot,
            strategy_path=args.strategy_file,
            universe_path=args.universe_file,
            ledger_path=ledger_path,
        )
        report["mode"] = "submit-approved" if args.submit_approved else "dry-run"
        report["source_run_summary"] = str(summary_path.resolve())
        report["operator_responses"] = []
        audit_path = write_audit_report(report, args.logs_dir)
        _print_json(
            {
                "mode": report["mode"],
                "run_id": report["run_id"],
                "account_snapshot": snapshot,
                "position_exposure": {
                    str(item["symbol"]): item["market_value"]
                    for item in snapshot["positions"]  # type: ignore[index]
                },
                "cash_reserve_usd": report["cash_reserve_usd"],
                "buying_power_start": report["buying_power_start"],
                "buying_power_remaining": report["buying_power_remaining"],
                "decisions": report["decisions"],
                "audit_report": str(audit_path),
            },
            print_fn,
        )
        if not args.submit_approved:
            return 0

        decisions = report["decisions"]
        if not isinstance(decisions, list):
            raise ProposalValidationError("Proposal engine returned invalid decisions.")
        submissions: list[dict[str, object]] = []
        report["submissions"] = submissions
        for initial in decisions:
            if not isinstance(initial, dict) or initial.get("decision") != "accepted":
                continue
            proposal = initial.get("proposal")
            if not isinstance(proposal, dict):
                continue

            # Reconcile account state and risk again immediately before each approval.
            fresh_snapshot = executor.get_account_snapshot()
            fresh_report = build_proposals(
                run_summary,
                fresh_snapshot,
                strategy_path=args.strategy_file,
                universe_path=args.universe_file,
                ledger_path=ledger_path,
            )
            current_entry = next(
                (
                    entry for entry in fresh_report["decisions"]
                    if isinstance(entry, dict)
                    and isinstance(entry.get("proposal"), dict)
                    and entry["proposal"].get("client_order_id") == proposal.get("client_order_id")
                ),
                None,
            )
            if not isinstance(current_entry, dict) or current_entry.get("decision") != "accepted":
                rejection = {
                    "client_order_id": proposal.get("client_order_id"),
                    "symbol": proposal.get("symbol"),
                    "outcome": "rejected_before_approval",
                    "reason": (
                        current_entry.get("risk_decision", {}).get("reasons", ["Proposal no longer valid."])
                        if isinstance(current_entry, dict) and isinstance(current_entry.get("risk_decision"), dict)
                        else ["Proposal no longer valid after account reconciliation."]
                    ),
                }
                submissions.append(rejection)
                report["operator_responses"].append({
                    "client_order_id": proposal.get("client_order_id"),
                    "response": "not_requested",
                })
                continue

            refreshed_proposal = current_entry["proposal"]
            risk_decision = current_entry.get("risk_decision")
            assert isinstance(refreshed_proposal, dict)
            assert isinstance(risk_decision, dict)
            preview = {
                "WARNING": "Paper order only; no order is sent unless the exact phrase is entered.",
                "symbol": refreshed_proposal["symbol"],
                "side": refreshed_proposal["side"],
                "notional": refreshed_proposal["notional"],
                "quantity": refreshed_proposal["qty"],
                "signal_score": refreshed_proposal["signal_score"],
                "estimated_price": refreshed_proposal["estimated_price"],
                "existing_exposure": refreshed_proposal["existing_exposure"],
                "buying_power_remaining": fresh_report["buying_power_remaining"],
                "risk_decision": risk_decision,
                "client_order_id": refreshed_proposal["client_order_id"],
            }
            _print_json(preview, print_fn)
            operator_response = input_fn(
                f"Type {AlpacaPaperExecutor.APPROVAL_PHRASE!r} to submit this paper order: "
            )
            approved_by_operator = operator_response == AlpacaPaperExecutor.APPROVAL_PHRASE
            report["operator_responses"].append({
                "client_order_id": refreshed_proposal["client_order_id"],
                "response": "approved" if approved_by_operator else "rejected",
            })
            if not approved_by_operator:
                submissions.append({
                    "client_order_id": refreshed_proposal["client_order_id"],
                    "symbol": refreshed_proposal["symbol"],
                    "outcome": "operator_rejected",
                })
                continue
            final_snapshot = executor.get_account_snapshot()
            final_report = build_proposals(
                run_summary,
                final_snapshot,
                strategy_path=args.strategy_file,
                universe_path=args.universe_file,
                ledger_path=ledger_path,
            )
            final_entry = next(
                (
                    entry for entry in final_report["decisions"]
                    if isinstance(entry, dict)
                    and isinstance(entry.get("proposal"), dict)
                    and entry["proposal"].get("client_order_id") == proposal.get("client_order_id")
                ),
                None,
            )
            if not isinstance(final_entry, dict) or final_entry.get("decision") != "accepted":
                submissions.append({
                    "client_order_id": proposal.get("client_order_id"),
                    "symbol": proposal.get("symbol"),
                    "outcome": "rejected_after_reconciliation",
                    "reason": (
                        final_entry.get("risk_decision", {}).get("reasons", ["Proposal no longer valid."])
                        if isinstance(final_entry, dict) and isinstance(final_entry.get("risk_decision"), dict)
                        else ["Proposal no longer valid after final account reconciliation."]
                    ),
                })
                continue
            final_proposal = final_entry["proposal"]
            final_risk = final_entry.get("risk_decision")
            assert isinstance(final_proposal, dict)
            assert isinstance(final_risk, dict)
            fingerprint_fields = (
                "symbol", "side", "notional", "qty", "signal_score",
                "estimated_price", "existing_exposure", "buying_power_remaining_before",
            )
            if any(
                refreshed_proposal.get(field) != final_proposal.get(field)
                for field in fingerprint_fields
            ):
                submissions.append({
                    "client_order_id": proposal.get("client_order_id"),
                    "symbol": proposal.get("symbol"),
                    "outcome": "rejected_after_reconciliation",
                    "reason": ["Proposal details changed after human approval; rerun and approve the refreshed proposal."],
                })
                continue
            ledger_record = {
                "run_id": report["run_id"],
                "strategy_name": final_report["strategy"]["name"],  # type: ignore[index]
                "strategy_version": final_report["strategy"]["version"],  # type: ignore[index]
                "symbol": final_proposal["symbol"],
                "side": final_proposal["side"],
                "notional": final_proposal["notional"],
                "qty": final_proposal["qty"],
                "client_order_id": final_proposal["client_order_id"],
                "broker_order_id": None,
                "broker_status": None,
                "outcome": "submission_pending",
                "operator_response": "approved",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            append_submission_ledger(ledger_record, ledger_path)
            try:
                outcome = executor.submit_order(
                    final_proposal,
                    client_order_id=str(final_proposal["client_order_id"]),
                    approval=operator_response,
                    risk_approved=final_risk.get("approved") is True,
                    market_clock=final_snapshot["clock"],
                )
            except PaperExecutionError as exc:
                update_submission_ledger(
                    str(final_proposal["client_order_id"]),
                    {
                        "outcome": "submission_error",
                        "error": str(exc),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                    ledger_path,
                )
                submissions.append({
                    **ledger_record,
                    "outcome": "submission_error",
                    "error": str(exc),
                })
                report["account_snapshot"] = final_snapshot
                report["errors"] = [str(exc)]
                audit_path = write_audit_report(report, args.logs_dir)
                _print_json({"submissions": submissions, "audit_report": str(audit_path)}, print_fn)
                return 2
            order = outcome.get("order", {}) if isinstance(outcome, dict) else {}
            order_dict = order if isinstance(order, dict) else {}
            ledger_updates = {
                "broker_order_id": order_dict.get("id"),
                "broker_status": order_dict.get("status"),
                "outcome": outcome.get("outcome"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            update_submission_ledger(
                str(final_proposal["client_order_id"]),
                ledger_updates,
                ledger_path,
            )
            submitted_record = {**ledger_record, **ledger_updates}
            submissions.append(submitted_record)
            report["account_snapshot"] = final_snapshot
            if outcome.get("outcome") == "unresolved":
                report["errors"] = ["Order submission remains unresolved; stopping to prevent duplicate submission."]
                break
        audit_path = write_audit_report(report, args.logs_dir)
        _print_json({"submissions": submissions, "audit_report": str(audit_path)}, print_fn)
        return 0
    except (PaperExecutionError, ProposalValidationError, ValueError, OSError, json.JSONDecodeError) as exc:
        if "args" in locals():
            failure_record: dict[str, object] = {
                "mode": "account-only" if args.account_only else "submit-approved" if args.submit_approved else "dry-run",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "errors": [str(exc)],
                "orders_submitted": [],
            }
            if isinstance(locals().get("report"), dict):
                prior_report = locals()["report"]
                failure_record.update({
                    "run_id": prior_report.get("run_id"),
                    "account_snapshot": prior_report.get("account_snapshot"),
                    "decisions": prior_report.get("decisions", []),
                    "submissions": prior_report.get("submissions", []),
                })
            elif isinstance(locals().get("snapshot"), dict):
                failure_record["account_snapshot"] = locals()["snapshot"]
            write_audit_report(
                failure_record,
                args.logs_dir,
            )
        print_fn(f"Paper trading stopped safely: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
