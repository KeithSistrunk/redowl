"""Phase 2.2 Task 2 -- three-run test protocol for free-generation guardrails.

Runs `agent.run_hunt_free_generate` three times against the same target and
reasoning LLM, holding everything constant (target model, attack count,
session limits -- redowl does not send a `temperature` field at all, see
runner.call_openai_endpoint, so that axis is fixed by omission) except which
category is passed to guardrails.screen():

    Run 1: harness-assigned category, cycling GOALS evenly      -- baseline
    Run 2: harness-assigned category, cycling GOALS evenly      -- variance check vs run 1
    Run 3: no category at all                                   -- unconstrained comparison

Run 3 leaves `goal` unset for BOTH the generator and the screener (goal_cycle
= [None]), not just withheld from screen(). The brief frames run 3 as asking
"how many attacks land inside the five categories anyway" -- that's not a
real question if the generator was still secretly aimed at one of the five
categories the whole time. See run_hunt_free_generate's docstring for how
goal_cycle drives both call sites from one value.

Every attempt in every run -- allowed or rejected -- is appended to that
run's JSONL log (run1.jsonl / run2.jsonl / run3.jsonl) via
agent.free_gen_jsonl_record(), one line per attack, matching the brief's
Task 3 schema. `stop_on_goal_achieved=False` so a hunt does not cut short on
the first FAIL and skew the attack-count comparison across runs.

What this script does NOT do: classify run 3's attacks against the five-
category taxonomy. That's a judgment call the brief expects a human analyst
to make ("Write it up"), not something to fake with a keyword matcher that
would just be guessing at the same thing the harness-assigned category was
supposed to settle. summary.md lists run 3's attack text under "Manual
category review needed" instead.

Usage:
    python scripts/run_three_run_protocol.py \\
        --target examples/openai.yaml \\
        --reasoning examples/reasoning-openai.yaml \\
        --i-am-authorized-to-test \\
        --out-dir three_run_protocol_out \\
        --attack-count 15

Makes real network calls to both the target and reasoning endpoints (unlike
scripts/generate_sample_hunt_result.py, which mocks both) -- 3 x
--attack-count calls to each. Requires the same authorization flag and
respects the same blocklist as `redowl run` / `redowl hunt`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from redowl import agent, goals, guardrails, runner  # noqa: E402
from redowl.reporter import now_utc_iso  # noqa: E402
from redowl.runner import ConfigError  # noqa: E402

# Fixed order, not set iteration order, so "cycle evenly" is deterministic
# and reproducible across runs. Must match guardrails.ALLOWED_CATEGORIES --
# asserted in main() rather than silently drifting from it.
GOALS = [
    "prompt_injection",
    "jailbreak",
    "data_leakage",
    "system_prompt_extraction",
    "policy_bypass",
]


def _restrict_to_owner(path: Path) -> None:
    import os

    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _write_jsonl_entry(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    _restrict_to_owner(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", required=True, type=Path, help="Path to the target config YAML file.")
    parser.add_argument("--reasoning", required=True, type=Path, help="Path to the reasoning-LLM config YAML file.")
    parser.add_argument(
        "--goals-dir",
        default=Path("goals"),
        type=Path,
        help="Path to the goals directory, used to build the known_tool_ids fabrication registry. Default: ./goals",
    )
    parser.add_argument(
        "--out-dir",
        default=Path("three_run_protocol_out"),
        type=Path,
        help="Directory to write run1.jsonl/run2.jsonl/run3.jsonl and summary.json/.md into.",
    )
    parser.add_argument(
        "--attack-count",
        type=int,
        default=15,
        help="Attacks attempted per run, held constant across all three runs. Default: 15 (3 full cycles of the 5 goals).",
    )
    parser.add_argument(
        "--i-am-authorized-to-test",
        action="store_true",
        dest="authorized",
        help="Required. Confirms you are authorized to test the target endpoint.",
    )
    parser.add_argument("--operator", default=None, help="Operator identity to record in the summary.")
    return parser


def run_one(
    label: str,
    target_config: runner.TargetConfig,
    reasoning_config: runner.ReasoningConfig,
    goal_cycle: list[str | None],
    attack_count: int,
    known_tool_ids: set[str],
    out_dir: Path,
) -> tuple[agent.FreeGenHuntResult, guardrails.SessionState]:
    """Run one leg of the protocol. Session limits are the SessionState
    defaults, held identical across all three calls -- only goal_cycle
    differs between callers, per the protocol's one deliberate variable."""
    session = guardrails.SessionState(known_tool_ids=known_tool_ids)
    jsonl_path = out_dir / f"{label}.jsonl"

    def on_iteration(hunt_id: str, iteration: agent.FreeGenIteration) -> None:
        _write_jsonl_entry(jsonl_path, agent.free_gen_jsonl_record(hunt_id, label, iteration))

    print(f"[{label}] running {attack_count} attempt(s), goal_cycle={goal_cycle}...")
    result = agent.run_hunt_free_generate(
        target_config,
        reasoning_config,
        goal_cycle=goal_cycle,
        max_iterations=attack_count,
        session=session,
        on_iteration=on_iteration,
        hunt_id=label,
        stop_on_goal_achieved=False,
    )
    print(
        f"[{label}] done: {len(result.iterations)} attempt(s), terminated: {result.termination.reason} "
        f"-- {result.termination.detail}"
    )
    return result, session


def _run_stats(result: agent.FreeGenHuntResult, session: guardrails.SessionState) -> dict:
    flag_counts: dict[str, int] = {}
    eval_counts: dict[str, int] = {"PASS": 0, "FAIL": 0, "UNCERTAIN": 0, "not_evaluated": 0}
    for it in result.iterations:
        for flag in it.guardrail_flags:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
        eval_counts[it.eval_verdict or "not_evaluated"] += 1

    return {
        "termination_reason": result.termination.reason,
        "termination_detail": result.termination.detail,
        "attempts": len(result.iterations),
        "allowed": sum(1 for it in result.iterations if it.guardrail_allowed),
        "rejected": sum(1 for it in result.iterations if not it.guardrail_allowed),
        "needs_review": sum(1 for it in result.iterations if it.guardrail_needs_review),
        "flag_counts": flag_counts,
        "eval_counts": eval_counts,
        "session_halted": session.halted,
        "session_halt_reason": session.halt_reason,
        "session_novel_attack_count": session.novel_attack_count,
    }


def write_summary(
    out_dir: Path,
    operator: str,
    target_config: runner.TargetConfig,
    reasoning_config: runner.ReasoningConfig,
    attack_count: int,
    runs: dict[str, tuple[agent.FreeGenHuntResult, guardrails.SessionState]],
) -> None:
    stats = {label: _run_stats(result, session) for label, (result, session) in runs.items()}

    summary = {
        "run_timestamp_utc": now_utc_iso(),
        "operator": operator,
        "target_name": target_config.name,
        "target_base_url": target_config.base_url,
        "target_model": target_config.model,
        "reasoning_llm": reasoning_config.name,
        "attack_count_per_run": attack_count,
        "goals": GOALS,
        "runs": {
            "run1": {"category_assignment": "harness-assigned (cycled GOALS)", **stats["run1"]},
            "run2": {"category_assignment": "harness-assigned (cycled GOALS)", **stats["run2"]},
            "run3": {"category_assignment": "none (unconstrained)", **stats["run3"]},
        },
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _restrict_to_owner(out_dir / "summary.json")

    run3_result, _ = runs["run3"]
    lines: list[str] = [
        "# Phase 2.2 three-run test protocol",
        "",
        f"- Target: `{target_config.base_url}` ({target_config.model})",
        f"- Reasoning LLM: {reasoning_config.name}",
        f"- Attack count per run: {attack_count}",
        f"- Run at (UTC): {summary['run_timestamp_utc']}",
        f"- Operator: {operator}",
        "",
        "eval_verdict here reuses redowl's existing PASS/FAIL/UNCERTAIN scale: "
        "FAIL means the attack succeeded (leak/unsafe compliance), PASS means the "
        "target correctly resisted (no leak), UNCERTAIN means inconclusive.",
        "",
        "## Run comparison",
        "",
        "| | Run 1 (harness) | Run 2 (harness) | Run 3 (none) |",
        "|---|---|---|---|",
        f"| Attempts | {stats['run1']['attempts']} | {stats['run2']['attempts']} | {stats['run3']['attempts']} |",
        f"| Allowed | {stats['run1']['allowed']} | {stats['run2']['allowed']} | {stats['run3']['allowed']} |",
        f"| Rejected | {stats['run1']['rejected']} | {stats['run2']['rejected']} | {stats['run3']['rejected']} |",
        f"| Needs review | {stats['run1']['needs_review']} | {stats['run2']['needs_review']} | {stats['run3']['needs_review']} |",
        f"| FAIL (attack landed) | {stats['run1']['eval_counts']['FAIL']} | {stats['run2']['eval_counts']['FAIL']} | {stats['run3']['eval_counts']['FAIL']} |",
        f"| PASS (target resisted) | {stats['run1']['eval_counts']['PASS']} | {stats['run2']['eval_counts']['PASS']} | {stats['run3']['eval_counts']['PASS']} |",
        f"| UNCERTAIN | {stats['run1']['eval_counts']['UNCERTAIN']} | {stats['run2']['eval_counts']['UNCERTAIN']} | {stats['run3']['eval_counts']['UNCERTAIN']} |",
        f"| Termination | {stats['run1']['termination_reason']} | {stats['run2']['termination_reason']} | {stats['run3']['termination_reason']} |",
        "",
        "Run 1 vs run 2 is the variance baseline: both use the same harness-assigned "
        "category cycling, so any difference between their columns is stochastic "
        "generation noise, not a scope effect. This is a descriptive comparison, not "
        "a statistical test -- read the two columns as \"how much do identical runs "
        "wobble\" before attributing anything in run 3's column to the missing category.",
        "",
        "## Guardrail flags raised",
        "",
    ]
    for label in ("run1", "run2", "run3"):
        fc = stats[label]["flag_counts"]
        lines.append(f"**{label}:** " + (", ".join(f"{k} ({v})" for k, v in sorted(fc.items())) if fc else "(none)"))
    lines.append("")

    lines.append("## Manual category review needed (run 3)")
    lines.append("")
    lines.append(
        "Run 3's attacks were generated with no objective assigned and screened with "
        "category=None, so every one is flagged `no harness-assigned category -- cannot "
        "verify scope` by design (see guardrails.screen()). Classifying each attack "
        "below against the five-category taxonomy -- in scope / outside but useful / "
        "junk -- is a manual step; this script does not attempt it automatically."
    )
    lines.append("")
    if not run3_result.iterations:
        lines.append("(no attempts in run 3)")
    else:
        for it in run3_result.iterations:
            status = "allowed" if it.guardrail_allowed else f"REJECTED: {it.guardrail_reason}"
            lines.append(f"### Attempt {it.iteration_number} -- {status}")
            lines.append("")
            lines.append("```")
            lines.append(it.attack_text)
            lines.append("```")
            if it.eval_verdict:
                lines.append(f"- Eval verdict: {it.eval_verdict}")
            lines.append("")

    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    _restrict_to_owner(out_dir / "summary.md")


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)

    if not args.authorized:
        print(
            "ERROR: refusing to run without --i-am-authorized-to-test. "
            "You must confirm you are authorized to test this endpoint.",
            file=sys.stderr,
        )
        return 2

    assert set(GOALS) == guardrails.ALLOWED_CATEGORIES, (
        "scripts/run_three_run_protocol.py's GOALS has drifted from guardrails.ALLOWED_CATEGORIES"
    )

    operator = args.operator or __import__("getpass").getuser()

    try:
        target_config = runner.load_target_config(args.target)
        reasoning_config = runner.load_reasoning_config(args.reasoning)
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    blocked_entry = runner.is_blocklisted(target_config.base_url, target_config.blocklist)
    if blocked_entry is not None:
        print(
            f"ERROR: target base_url '{target_config.base_url}' matches blocklist entry "
            f"'{blocked_entry}'. Refusing to run.",
            file=sys.stderr,
        )
        return 2

    for warning in filter(None, [
        runner.transport_warning(target_config.base_url),
        runner.transport_warning(reasoning_config.base_url),
    ]):
        print(f"WARNING: {warning}", file=sys.stderr)

    print("PRE-FLIGHT -- confirm before this protocol starts (not enforceable from code):", file=sys.stderr)
    print(guardrails.SANDBOX_REQUIREMENTS, file=sys.stderr)

    known_tool_ids = goals.load_all_pool_ids(args.goals_dir)
    if not known_tool_ids:
        print(
            f"WARNING: no attack pools found under {args.goals_dir} -- known_tool_ids is empty, "
            "fabrication checking will be skipped, not enforced.",
            file=sys.stderr,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    run1, session1 = run_one("run1", target_config, reasoning_config, GOALS, args.attack_count, known_tool_ids, args.out_dir)
    run2, session2 = run_one("run2", target_config, reasoning_config, GOALS, args.attack_count, known_tool_ids, args.out_dir)
    run3, session3 = run_one("run3", target_config, reasoning_config, [None], args.attack_count, known_tool_ids, args.out_dir)

    write_summary(
        args.out_dir,
        operator,
        target_config,
        reasoning_config,
        args.attack_count,
        {"run1": (run1, session1), "run2": (run2, session2), "run3": (run3, session3)},
    )

    print(f"Done. Wrote {args.out_dir}/run1.jsonl, run2.jsonl, run3.jsonl, summary.json, summary.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
