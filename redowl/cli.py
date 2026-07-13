"""argparse entry point for redowl.

Logging format decision (see build prompt Q3): the audit log is plain JSON Lines
(one JSON object per line, appended), not a structured logging library. This keeps
the MVP dependency-free and the log trivially greppable/parseable.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from redowl import reporter, runner
from redowl.evaluator import evaluate
from redowl.reporter import now_utc_iso
from redowl.runner import ConfigError


def _restrict_to_owner(path: Path) -> None:
    """Best-effort: restrict a file's permissions to the owner only. No-op where the OS doesn't support it."""
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _write_audit_log_entry(audit_log_path: Path, entry: dict) -> None:
    """Append one JSON line to the audit log, creating the file/parents if needed."""
    audit_log_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    _restrict_to_owner(audit_log_path)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argparse parser with the `run` subcommand."""
    parser = argparse.ArgumentParser(prog="redowl", description="Test an LLM endpoint for prompt injection and jailbreak resistance.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the test suite against a target endpoint.")
    run_parser.add_argument("--target", required=True, type=Path, help="Path to the target config YAML file.")
    run_parser.add_argument("--tests", required=True, type=Path, help="Path to a directory of test case YAML files.")
    run_parser.add_argument("--out", required=True, type=Path, help="Path to write the JSON findings report to.")
    run_parser.add_argument(
        "--i-am-authorized-to-test",
        action="store_true",
        dest="authorized",
        help="Required. Confirms you are authorized to test the target endpoint.",
    )
    run_parser.add_argument(
        "--operator",
        default=None,
        help="Operator identity to record in the audit log. Defaults to the local OS username.",
    )
    run_parser.add_argument(
        "--audit-log",
        default=Path("redowl_audit.log.jsonl"),
        type=Path,
        help="Path to the audit log file (JSON Lines, appended). Default: ./redowl_audit.log.jsonl",
    )

    return parser


def run_command(args: argparse.Namespace) -> int:
    """Execute the `run` subcommand. Returns a process exit code."""
    if not args.authorized:
        print(
            "ERROR: refusing to run without --i-am-authorized-to-test. "
            "You must confirm you are authorized to test this endpoint.",
            file=sys.stderr,
        )
        return 2

    operator = args.operator or getpass.getuser()

    try:
        config = runner.load_target_config(args.target)
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    blocked_entry = runner.is_blocklisted(config.base_url, config.blocklist)

    for warning in filter(None, [
        runner.transport_warning(config.base_url),
        runner.transport_warning(config.judge.base_url) if config.judge.enabled else None,
    ]):
        print(f"WARNING: {warning}", file=sys.stderr)

    _write_audit_log_entry(
        args.audit_log,
        {
            "timestamp_utc": now_utc_iso(),
            "operator": operator,
            "target_name": config.name,
            "target_base_url": config.base_url,
            "tests_dir": str(args.tests),
            "out_path": str(args.out),
            "authorized_flag_present": True,
            "blocked": blocked_entry is not None,
            "blocked_by_entry": blocked_entry,
        },
    )

    if blocked_entry is not None:
        print(
            f"ERROR: target base_url '{config.base_url}' matches blocklist entry '{blocked_entry}'. Refusing to run.",
            file=sys.stderr,
        )
        return 2

    try:
        test_cases = runner.load_test_cases(args.tests)
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Running {len(test_cases)} test case(s) against '{config.name}' ({config.base_url})...")

    try:
        results = runner.run_all(config, test_cases)
    except (ConfigError, NotImplementedError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    judge_limiter = runner.RateLimiter(config.judge.requests_per_second) if config.judge.enabled else None
    findings = [evaluate(case, raw, config.judge, judge_limiter) for case, raw in results]

    meta = {
        "target_name": config.name,
        "target_base_url": config.base_url,
        "target_model": config.model,
        "run_timestamp_utc": now_utc_iso(),
        "operator": operator,
        "tests_dir": str(args.tests),
        "judge_enabled": config.judge.enabled,
    }
    report = reporter.build_report(findings, meta)

    reporter.write_json_report(report, args.out)
    _restrict_to_owner(args.out)
    md_path = args.out.with_suffix(".md")
    reporter.write_markdown_report(report, md_path)
    _restrict_to_owner(md_path)

    counts = report["summary"]["counts"]
    print(
        f"Done. {counts['PASS']} PASS / {counts['FAIL']} FAIL / {counts['UNCERTAIN']} UNCERTAIN. "
        f"Wrote {args.out} and {md_path}."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return run_command(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
