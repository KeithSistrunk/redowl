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
import uuid
from pathlib import Path

from dotenv import load_dotenv

from redowl import agent, goals, guardrails, reporter, runner
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


def _write_jsonl_entry(path: Path, entry: dict) -> None:
    """Append one JSON line to a JSONL log (e.g. the free-generation attempt
    log), creating the file/parents if needed. Same shape as
    _write_audit_log_entry, kept separate because the two logs have
    different lifecycles and callers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    _restrict_to_owner(path)


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

    hunt_parser = subparsers.add_parser(
        "hunt", help="Run an agent loop that picks attacks from a fixed pool against a target endpoint."
    )
    hunt_parser.add_argument("--target", required=True, type=Path, help="Path to the target config YAML file.")
    hunt_parser.add_argument(
        "--reasoning", required=True, type=Path, help="Path to the reasoning-LLM config YAML file."
    )
    hunt_parser.add_argument("--goal", required=True, help="Goal name to pursue, e.g. 'prompt_injection'.")
    hunt_parser.add_argument("--out", required=True, type=Path, help="Path to write the JSON hunt result to.")
    hunt_parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Override the goal's default_max_iterations (see goals/<goal>/definition.yaml).",
    )
    hunt_parser.add_argument(
        "--goals-dir",
        default=Path("goals"),
        type=Path,
        help="Path to the directory containing goal subdirectories. Default: ./goals",
    )
    hunt_parser.add_argument(
        "--i-am-authorized-to-test",
        action="store_true",
        dest="authorized",
        help="Required. Confirms you are authorized to test the target endpoint.",
    )
    hunt_parser.add_argument(
        "--operator",
        default=None,
        help="Operator identity to record in the audit log. Defaults to the local OS username.",
    )
    hunt_parser.add_argument(
        "--audit-log",
        default=Path("redowl_audit.log.jsonl"),
        type=Path,
        help="Path to the audit log file (JSON Lines, appended). Default: ./redowl_audit.log.jsonl",
    )
    hunt_parser.add_argument(
        "--free-generate",
        action="store_true",
        dest="free_generate",
        help=(
            "Phase 2.2: have the reasoning LLM free-write attacks for --goal instead of picking "
            "from the fixed pool. Every generated attack is screened through redowl.guardrails."
            "screen() before it reaches the target; a rejected attack halts the hunt. Requires a "
            "disposable, network-isolated target -- see redowl/guardrails.py's "
            "SANDBOX_REQUIREMENTS, printed as a warning before the hunt starts. Writes an "
            "additional JSONL log of every attempt next to --out (same stem, .jsonl suffix)."
        ),
    )
    hunt_parser.add_argument(
        "--variants",
        action="store_true",
        dest="variants",
        help=(
            "EXPERIMENTAL (Phase 2.1): have the reasoning LLM pick a pool attack AND reword its "
            "text, instead of sending it verbatim. The reworded text is screened through "
            "redowl.guardrails.screen() with category=--goal (e.g. 'prompt_injection'), not "
            "self-declared. Mutually exclusive with --free-generate. Verbatim pool (no flag) "
            "remains the default; this flag exists to support the pool/variant/free-gen three-arm "
            "comparison -- see scripts/run_three_arm_experiment.py. Writes the same additional "
            "JSONL log as --free-generate, with pool_text/variant_text side by side for manual "
            "drift review."
        ),
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


def hunt_command(args: argparse.Namespace) -> int:
    """Execute the `hunt` subcommand. Returns a process exit code."""
    if not args.authorized:
        print(
            "ERROR: refusing to run without --i-am-authorized-to-test. "
            "You must confirm you are authorized to test this endpoint.",
            file=sys.stderr,
        )
        return 2

    if args.free_generate and args.variants:
        print("ERROR: --free-generate and --variants are mutually exclusive.", file=sys.stderr)
        return 2

    operator = args.operator or getpass.getuser()

    try:
        target_config = runner.load_target_config(args.target)
        reasoning_config = runner.load_reasoning_config(args.reasoning)
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    blocked_entry = runner.is_blocklisted(target_config.base_url, target_config.blocklist)

    for warning in filter(None, [
        runner.transport_warning(target_config.base_url),
        runner.transport_warning(target_config.judge.base_url) if target_config.judge.enabled else None,
        runner.transport_warning(reasoning_config.base_url),
    ]):
        print(f"WARNING: {warning}", file=sys.stderr)

    if args.free_generate:
        return _hunt_free_generate(args, target_config, reasoning_config, operator, blocked_entry)

    try:
        goal, pool = goals.load_goal(args.goals_dir, args.goal)
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.variants:
        return _hunt_variants(args, target_config, reasoning_config, goal, pool, operator, blocked_entry)

    max_iterations = args.max_iterations if args.max_iterations is not None else goal.default_max_iterations
    hunt_id = str(uuid.uuid4())

    _write_audit_log_entry(
        args.audit_log,
        {
            "timestamp_utc": now_utc_iso(),
            "event": "hunt_start",
            "hunt_id": hunt_id,
            "operator": operator,
            "goal": goal.name,
            "target_name": target_config.name,
            "target_base_url": target_config.base_url,
            "reasoning_llm": reasoning_config.name,
            "max_iterations": max_iterations,
            "out_path": str(args.out),
            "authorized_flag_present": True,
            "blocked": blocked_entry is not None,
            "blocked_by_entry": blocked_entry,
        },
    )

    if blocked_entry is not None:
        print(
            f"ERROR: target base_url '{target_config.base_url}' matches blocklist entry "
            f"'{blocked_entry}'. Refusing to run.",
            file=sys.stderr,
        )
        return 2

    print(
        f"Hunting goal '{goal.name}' against '{target_config.name}' ({target_config.base_url}) "
        f"using reasoning LLM '{reasoning_config.name}', up to {max_iterations} iteration(s)..."
    )

    def on_iteration(hunt_id: str, iteration) -> None:
        _write_audit_log_entry(
            args.audit_log,
            {
                "timestamp_utc": now_utc_iso(),
                "event": "hunt_iteration",
                "hunt_id": hunt_id,
                "goal": goal.name,
                "target": target_config.name,
                "reasoning_llm": reasoning_config.name,
                "iteration_number": iteration.iteration_number,
                "attack_id": iteration.attack_id,
                "agent_rationale": iteration.agent_rationale,
                "target_response_length": iteration.target_response_length,
                "verdict": iteration.verdict,
                "reasoning_latency_ms": iteration.reasoning_latency_ms,
            },
        )

    result = agent.run_hunt(
        target_config, reasoning_config, goal, pool, max_iterations, on_iteration=on_iteration, hunt_id=hunt_id
    )

    _write_audit_log_entry(
        args.audit_log,
        {
            "timestamp_utc": now_utc_iso(),
            "event": "hunt_end",
            "hunt_id": result.hunt_id,
            "goal": goal.name,
            "target": target_config.name,
            "reasoning_llm": reasoning_config.name,
            "termination_reason": result.termination.reason,
            "termination_detail": result.termination.detail,
            "iteration_count": len(result.iterations),
        },
    )

    meta = {
        "target_name": target_config.name,
        "target_base_url": target_config.base_url,
        "target_model": target_config.model,
        "run_timestamp_utc": now_utc_iso(),
        "operator": operator,
        "goal": goal.name,
        "judge_enabled": target_config.judge.enabled,
    }
    report = agent.build_hunt_report(result, meta)

    reporter.write_json_report(report, args.out)
    _restrict_to_owner(args.out)
    md_path = args.out.with_suffix(".md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(agent.render_hunt_markdown(report), encoding="utf-8")
    _restrict_to_owner(md_path)

    print(
        f"Done. {len(result.iterations)} iteration(s), terminated: {result.termination.reason}. "
        f"Wrote {args.out} and {md_path}."
    )
    return 0


def _hunt_free_generate(
    args: argparse.Namespace,
    target_config: runner.TargetConfig,
    reasoning_config: runner.ReasoningConfig,
    operator: str,
    blocked_entry: str | None,
) -> int:
    """Free-generation branch of `redowl hunt --free-generate` (Phase 2.2).

    Mirrors hunt_command's pool-mode flow (audit logging, blocklist gate,
    JSON + Markdown report) but routes through agent.run_hunt_free_generate
    instead of agent.run_hunt, and additionally writes a JSONL log of every
    screened attempt -- Task 3 of the Phase 2.2 handoff brief. --goal is not
    validated here against guardrails.ALLOWED_CATEGORIES: an invalid category
    is caught by screen() itself on the first iteration (a hard block), which
    is the one place that allowlist is actually enforced.
    """
    known_tool_ids = goals.load_all_pool_ids(args.goals_dir)
    session = guardrails.SessionState(known_tool_ids=known_tool_ids)

    hunt_id = str(uuid.uuid4())
    max_iterations = (
        args.max_iterations if args.max_iterations is not None else agent.DEFAULT_FREE_GEN_MAX_ITERATIONS
    )
    jsonl_path = args.out.with_suffix(".jsonl")

    _write_audit_log_entry(
        args.audit_log,
        {
            "timestamp_utc": now_utc_iso(),
            "event": "hunt_start",
            "hunt_id": hunt_id,
            "mode": "free_generate",
            "operator": operator,
            "goal": args.goal,
            "target_name": target_config.name,
            "target_base_url": target_config.base_url,
            "reasoning_llm": reasoning_config.name,
            "max_iterations": max_iterations,
            "out_path": str(args.out),
            "jsonl_log_path": str(jsonl_path),
            "known_tool_ids_count": len(known_tool_ids),
            "authorized_flag_present": True,
            "blocked": blocked_entry is not None,
            "blocked_by_entry": blocked_entry,
        },
    )

    if blocked_entry is not None:
        print(
            f"ERROR: target base_url '{target_config.base_url}' matches blocklist entry "
            f"'{blocked_entry}'. Refusing to run.",
            file=sys.stderr,
        )
        return 2

    if not known_tool_ids:
        print(
            f"WARNING: no attack pools found under {args.goals_dir} -- known_tool_ids is empty, "
            "so guardrails.screen()'s fabrication check will be skipped, not enforced.",
            file=sys.stderr,
        )

    print("PRE-FLIGHT -- confirm before this hunt starts (not enforceable from code):", file=sys.stderr)
    print(guardrails.SANDBOX_REQUIREMENTS, file=sys.stderr)

    print(
        f"Free-generating attacks for goal '{args.goal}' against '{target_config.name}' "
        f"({target_config.base_url}) using reasoning LLM '{reasoning_config.name}', "
        f"up to {max_iterations} iteration(s)..."
    )

    def on_iteration(hunt_id: str, iteration: agent.FreeGenIteration) -> None:
        _write_audit_log_entry(
            args.audit_log,
            {
                "timestamp_utc": now_utc_iso(),
                "event": "hunt_iteration",
                "hunt_id": hunt_id,
                "mode": "free_generate",
                "goal": args.goal,
                "target": target_config.name,
                "reasoning_llm": reasoning_config.name,
                "iteration_number": iteration.iteration_number,
                "guardrail_allowed": iteration.guardrail_allowed,
                "guardrail_needs_review": iteration.guardrail_needs_review,
                "guardrail_flags": iteration.guardrail_flags,
                "target_response_length": iteration.target_response_length,
                "eval_verdict": iteration.eval_verdict,
                "reasoning_latency_ms": iteration.reasoning_latency_ms,
            },
        )
        _write_jsonl_entry(jsonl_path, agent.free_gen_jsonl_record(hunt_id, hunt_id, iteration))

    result = agent.run_hunt_free_generate(
        target_config,
        reasoning_config,
        goal_cycle=[args.goal],
        max_iterations=max_iterations,
        session=session,
        on_iteration=on_iteration,
        hunt_id=hunt_id,
    )

    _write_audit_log_entry(
        args.audit_log,
        {
            "timestamp_utc": now_utc_iso(),
            "event": "hunt_end",
            "hunt_id": result.hunt_id,
            "mode": "free_generate",
            "goal": args.goal,
            "target": target_config.name,
            "reasoning_llm": reasoning_config.name,
            "termination_reason": result.termination.reason,
            "termination_detail": result.termination.detail,
            "iteration_count": len(result.iterations),
        },
    )

    meta = {
        "target_name": target_config.name,
        "target_base_url": target_config.base_url,
        "target_model": target_config.model,
        "run_timestamp_utc": now_utc_iso(),
        "operator": operator,
        "goal": args.goal,
        "judge_enabled": target_config.judge.enabled,
    }
    report = agent.build_free_gen_report(result, meta, session)

    reporter.write_json_report(report, args.out)
    _restrict_to_owner(args.out)
    md_path = args.out.with_suffix(".md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(agent.render_free_gen_markdown(report), encoding="utf-8")
    _restrict_to_owner(md_path)

    print(
        f"Done. {len(result.iterations)} attempt(s), terminated: {result.termination.reason}. "
        f"Wrote {args.out}, {md_path}, and {jsonl_path}."
    )
    return 0


def _hunt_variants(
    args: argparse.Namespace,
    target_config: runner.TargetConfig,
    reasoning_config: runner.ReasoningConfig,
    goal: goals.GoalDefinition,
    pool: list[runner.TestCase],
    operator: str,
    blocked_entry: str | None,
) -> int:
    """Pool + variants branch of `redowl hunt --variants` (Phase 2.1, EXPERIMENTAL).

    Reuses the free-generation plumbing (on_iteration JSONL callback, sandbox
    banner) but keeps the pool/goal loading `--free-generate` skips, since
    generate_variant() needs the pool's actual attack text to reword and
    screen()'s category comes from the pool item, not the harness or the
    model. auto_approve_screened stays False here too, same as free-gen --
    the brief's case for flipping it (the enum bounds what can be attempted)
    is real, but that's a decision for after a review pass on real output,
    not a CLI default.
    """
    known_tool_ids = goals.load_all_pool_ids(args.goals_dir)
    session = guardrails.SessionState(known_tool_ids=known_tool_ids)

    hunt_id = str(uuid.uuid4())
    max_iterations = args.max_iterations if args.max_iterations is not None else goal.default_max_iterations
    jsonl_path = args.out.with_suffix(".jsonl")

    _write_audit_log_entry(
        args.audit_log,
        {
            "timestamp_utc": now_utc_iso(),
            "event": "hunt_start",
            "hunt_id": hunt_id,
            "mode": "variants",
            "operator": operator,
            "goal": goal.name,
            "target_name": target_config.name,
            "target_base_url": target_config.base_url,
            "reasoning_llm": reasoning_config.name,
            "max_iterations": max_iterations,
            "out_path": str(args.out),
            "jsonl_log_path": str(jsonl_path),
            "authorized_flag_present": True,
            "blocked": blocked_entry is not None,
            "blocked_by_entry": blocked_entry,
        },
    )

    if blocked_entry is not None:
        print(
            f"ERROR: target base_url '{target_config.base_url}' matches blocklist entry "
            f"'{blocked_entry}'. Refusing to run.",
            file=sys.stderr,
        )
        return 2

    print("PRE-FLIGHT -- confirm before this hunt starts (not enforceable from code):", file=sys.stderr)
    print(guardrails.SANDBOX_REQUIREMENTS, file=sys.stderr)

    print(
        f"Generating variants of goal '{goal.name}' pool attacks against '{target_config.name}' "
        f"({target_config.base_url}) using reasoning LLM '{reasoning_config.name}', "
        f"up to {max_iterations} iteration(s)... [EXPERIMENTAL: Phase 2.1]"
    )

    def on_iteration(hunt_id: str, iteration: agent.VariantIteration) -> None:
        _write_audit_log_entry(
            args.audit_log,
            {
                "timestamp_utc": now_utc_iso(),
                "event": "hunt_iteration",
                "hunt_id": hunt_id,
                "mode": "variants",
                "goal": goal.name,
                "target": target_config.name,
                "reasoning_llm": reasoning_config.name,
                "iteration_number": iteration.iteration_number,
                "attack_id": iteration.attack_id,
                "fallback_used": iteration.fallback_used,
                "guardrail_allowed": iteration.guardrail_allowed,
                "guardrail_needs_review": iteration.guardrail_needs_review,
                "guardrail_flags": iteration.guardrail_flags,
                "target_response_length": iteration.target_response_length,
                "eval_verdict": iteration.eval_verdict,
                "reasoning_latency_ms": iteration.reasoning_latency_ms,
            },
        )
        _write_jsonl_entry(jsonl_path, agent.variant_jsonl_record(hunt_id, hunt_id, iteration))

    result = agent.run_hunt_variants(
        target_config,
        reasoning_config,
        goal,
        pool,
        max_iterations=max_iterations,
        session=session,
        on_iteration=on_iteration,
        hunt_id=hunt_id,
    )

    _write_audit_log_entry(
        args.audit_log,
        {
            "timestamp_utc": now_utc_iso(),
            "event": "hunt_end",
            "hunt_id": result.hunt_id,
            "mode": "variants",
            "goal": goal.name,
            "target": target_config.name,
            "reasoning_llm": reasoning_config.name,
            "termination_reason": result.termination.reason,
            "termination_detail": result.termination.detail,
            "iteration_count": len(result.iterations),
        },
    )

    meta = {
        "target_name": target_config.name,
        "target_base_url": target_config.base_url,
        "target_model": target_config.model,
        "run_timestamp_utc": now_utc_iso(),
        "operator": operator,
        "goal": goal.name,
        "judge_enabled": target_config.judge.enabled,
    }
    report = agent.build_variant_report(result, meta, session)

    reporter.write_json_report(report, args.out)
    _restrict_to_owner(args.out)
    md_path = args.out.with_suffix(".md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(agent.render_variant_markdown(report), encoding="utf-8")
    _restrict_to_owner(md_path)

    print(
        f"Done. {len(result.iterations)} attempt(s), terminated: {result.termination.reason}. "
        f"Wrote {args.out}, {md_path}, and {jsonl_path}."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return run_command(args)
    if args.command == "hunt":
        return hunt_command(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
