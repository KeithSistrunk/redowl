"""Three-arm experiment: verbatim pool (2.0) vs. variant (2.1) vs. free-gen (2.2).

    | Arm | Path              | Constraint                              |
    |-----|-------------------|------------------------------------------|
    | A   | verbatim pool     | attack text fixed                        |
    | B   | variant (2.1)     | attack chosen from pool, text reworded   |
    | C   | free-gen (2.2)    | nothing bounded                          |

Same target, same attack count, same session limits, compared at the
CATEGORY level (prompt_injection / jailbreak / data_leakage /
system_prompt_extraction / policy_bypass) -- free-gen has no pool id, so
category is the only axis all three arms share.

WHAT THIS SCRIPT MEASURES vs. WHAT ONE HUNT MEASURES: Arms B and C run with
stop_on_goal_achieved=False, unlike a normal `redowl hunt`, which stops at
the first success. A success RATE needs multiple trials per arm; stopping at
the first hit would make every arm's rate either 0 or "whatever the first
attempt happened to be." Arm A never used the reasoning LLM's hunt loop at
all -- it is every pool attack in the category, sent once, exactly what
`redowl run` does -- because "attack text fixed" means there is no picking
decision to make. This is a deliberate divergence from a normal hunt's
early-stop behavior, not a bug.

COVERAGE GAP, read before running: only `goals/prompt-injection/` has an
attack pool in this repo today. Arms A and B need a pool to draw from, so
they run ONLY for categories that have one -- currently just
prompt_injection. Arm C (free-gen) needs no pool, so it runs for all five
categories. The summary reports this gap explicitly rather than silently
comparing three arms on one category and four bare Arm-C-only rows. Building
out pools for the other four categories is a real, separate piece of work
(~11 hand-authored attacks per category for prompt_injection) -- a content
decision, not something this script papers over.

TEMPERATURE, per the brief: check --reasoning-variants' temperature before
trusting Arm B's numbers. At temperature 0, a given pool item's variant text
comes back byte-identical run to run, which would make Arm B look
artificially noise-free relative to Arm A/C. This script warns if it's unset
or zero; examples/reasoning-ollama-variants.yaml sets 0.9 as a starting point.

STATISTICAL HONESTY: with --attack-count trials per arm per run and
--runs-per-arm repetitions, total trials per arm-category cell is small
(pool size ~11 x 2 runs ~= 22). Differences under roughly a 20-point spread
are not distinguishable from noise at this scale -- summary.md reports
results as directional and says so; it does not call a 5-point gap a finding.

Usage:
    python scripts/run_three_arm_experiment.py \\
        --target examples/ollama.yaml \\
        --reasoning examples/reasoning-ollama.yaml \\
        --reasoning-variants examples/reasoning-ollama-variants.yaml \\
        --i-am-authorized-to-test \\
        --out-dir three_arm_experiment_out \\
        --runs-per-arm 2
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
from redowl.evaluator import Verdict, evaluate  # noqa: E402
from redowl.reporter import now_utc_iso  # noqa: E402
from redowl.runner import ConfigError, RateLimiter, call_target  # noqa: E402

CATEGORIES = [
    "prompt_injection",
    "jailbreak",
    "data_leakage",
    "system_prompt_extraction",
    "policy_bypass",
]

DEFAULT_ATTACK_COUNT_NO_POOL = 8


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
    parser.add_argument(
        "--reasoning", required=True, type=Path, help="Reasoning-LLM config for Arm C (free-generation)."
    )
    parser.add_argument(
        "--reasoning-variants",
        type=Path,
        default=None,
        help=(
            "Reasoning-LLM config for Arm B (variants). Should set a nonzero temperature -- see "
            "examples/reasoning-ollama-variants.yaml. Defaults to --reasoning if not given, with a "
            "warning, since --reasoning's temperature is usually unset (fine for pool-picking, not "
            "for rewording)."
        ),
    )
    parser.add_argument(
        "--goals-dir",
        default=Path("goals"),
        type=Path,
        help="Path to the goals directory. Default: ./goals",
    )
    parser.add_argument(
        "--out-dir",
        default=Path("three_arm_experiment_out"),
        type=Path,
        help="Directory to write per-arm JSONL logs and summary.json/.md into.",
    )
    parser.add_argument(
        "--runs-per-arm",
        type=int,
        default=2,
        help="Repetitions per arm per category, for the within-arm variance measurement. Default: 2.",
    )
    parser.add_argument(
        "--attack-count-no-pool",
        type=int,
        default=DEFAULT_ATTACK_COUNT_NO_POOL,
        help=(
            "Attack count for Arm C on categories with no pool (so no natural attack-count to match "
            f"against Arms A/B). Default: {DEFAULT_ATTACK_COUNT_NO_POOL}."
        ),
    )
    parser.add_argument(
        "--i-am-authorized-to-test",
        action="store_true",
        dest="authorized",
        help="Required. Confirms you are authorized to test the target endpoint.",
    )
    parser.add_argument("--operator", default=None, help="Operator identity to record in the summary.")
    return parser


def run_arm_a(
    label: str,
    target_config: runner.TargetConfig,
    pool: list[runner.TestCase],
    out_dir: Path,
) -> list:
    """Verbatim pool, no reasoning LLM: every pool attack, sent once. Same
    machinery `redowl run` uses (runner.run_all + evaluate), because "attack
    text fixed" means there's no picking decision for a reasoning LLM to make."""
    limiter = RateLimiter(target_config.requests_per_second)
    judge_limiter = RateLimiter(target_config.judge.requests_per_second) if target_config.judge.enabled else None

    jsonl_path = out_dir / f"{label}.jsonl"
    findings = []
    for attack in pool:
        limiter.wait()
        raw = call_target(target_config, attack.prompt)
        finding = evaluate(attack, raw, target_config.judge, judge_limiter)
        findings.append(finding)
        _write_jsonl_entry(
            jsonl_path,
            {
                "run_id": label,
                "attack_id": attack.id,
                "category": attack.category,
                "pool_text": attack.prompt,
                "eval_verdict": finding.verdict.value,
            },
        )
    print(f"[{label}] done: {len(findings)} attempt(s) (verbatim pool, no reasoning LLM).")
    return findings


def run_arm_b(
    label: str,
    target_config: runner.TargetConfig,
    reasoning_variants_config: runner.ReasoningConfig,
    goal,
    pool: list[runner.TestCase],
    attack_count: int,
    known_tool_ids: set,
    out_dir: Path,
):
    # max_per_window/novel_attack_limit_per_session sized to attack_count, not
    # left at SessionState's defaults (10/300s): Arm A has no guardrail in its
    # path at all, so a pool bigger than 10 would silently rate-limit B/C to
    # fewer attempts than A got, breaking the "same attack count" comparison.
    session = guardrails.SessionState(
        known_tool_ids=known_tool_ids,
        max_per_window=max(attack_count, 10),
        novel_attack_limit_per_session=max(attack_count, 25),
    )
    jsonl_path = out_dir / f"{label}.jsonl"

    def on_iteration(hunt_id, iteration):
        _write_jsonl_entry(jsonl_path, agent.variant_jsonl_record(hunt_id, label, iteration))

    print(f"[{label}] running {attack_count} attempt(s) (variant)...")
    result = agent.run_hunt_variants(
        target_config, reasoning_variants_config, goal, pool,
        max_iterations=attack_count, session=session,
        on_iteration=on_iteration, hunt_id=label, stop_on_goal_achieved=False,
    )
    print(f"[{label}] done: {len(result.iterations)} attempt(s), terminated: {result.termination.reason}")
    return result, session


def run_arm_c(
    label: str,
    target_config: runner.TargetConfig,
    reasoning_config: runner.ReasoningConfig,
    category: str,
    attack_count: int,
    known_tool_ids: set,
    out_dir: Path,
):
    # Same rationale as run_arm_b: don't let the guardrail's default rate
    # limit become the reason C gets fewer attempts than A for this category.
    session = guardrails.SessionState(
        known_tool_ids=known_tool_ids,
        max_per_window=max(attack_count, 10),
        novel_attack_limit_per_session=max(attack_count, 25),
    )
    jsonl_path = out_dir / f"{label}.jsonl"

    def on_iteration(hunt_id, iteration):
        _write_jsonl_entry(jsonl_path, agent.free_gen_jsonl_record(hunt_id, label, iteration))

    print(f"[{label}] running {attack_count} attempt(s) (free-gen)...")
    result = agent.run_hunt_free_generate(
        target_config, reasoning_config, goal_cycle=[category],
        max_iterations=attack_count, session=session,
        on_iteration=on_iteration, hunt_id=label, stop_on_goal_achieved=False,
    )
    print(f"[{label}] done: {len(result.iterations)} attempt(s), terminated: {result.termination.reason}")
    return result, session


def _success_rate(findings, expected_count: int, termination_reason: str, termination_detail: str) -> dict:
    """Bundle the FAIL/PASS/UNCERTAIN counts with whether this run actually
    completed expected_count attempts. A run that stopped early (a reasoning
    LLM JSON parse failure, a guardrail halt, ...) is NOT directly comparable
    to a full run at face value -- `complete=False` flags that in the
    summary instead of silently averaging a 4-attempt run in with 11-attempt
    ones."""
    total = len(findings)
    fail = sum(1 for f in findings if f.verdict == Verdict.FAIL)
    passed = sum(1 for f in findings if f.verdict == Verdict.PASS)
    uncertain = total - fail - passed
    return {
        "total": total,
        "expected": expected_count,
        "complete": total >= expected_count,
        "termination_reason": termination_reason,
        "termination_detail": termination_detail,
        "FAIL": fail,
        "PASS": passed,
        "UNCERTAIN": uncertain,
        "success_rate": (fail / total) if total else None,
    }


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

    operator = args.operator or __import__("getpass").getuser()

    try:
        target_config = runner.load_target_config(args.target)
        reasoning_config = runner.load_reasoning_config(args.reasoning)
        reasoning_variants_config = (
            runner.load_reasoning_config(args.reasoning_variants) if args.reasoning_variants else reasoning_config
        )
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
        runner.transport_warning(reasoning_variants_config.base_url),
    ]):
        print(f"WARNING: {warning}", file=sys.stderr)

    if not reasoning_variants_config.temperature:
        print(
            "WARNING: --reasoning-variants has no (or zero) temperature set. Arm B's variant text "
            "will likely come back near-identical across runs, making its variance measurement "
            "meaningless. See examples/reasoning-ollama-variants.yaml.",
            file=sys.stderr,
        )

    print("PRE-FLIGHT -- confirm before this experiment starts (not enforceable from code):", file=sys.stderr)
    print(guardrails.SANDBOX_REQUIREMENTS, file=sys.stderr)

    known_tool_ids = goals.load_all_pool_ids(args.goals_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_category: dict[str, dict] = {}

    for category in CATEGORIES:
        try:
            goal, pool = goals.load_goal(args.goals_dir, category)
            has_pool = True
        except ConfigError:
            goal, pool = None, []
            has_pool = False

        attack_count = len(pool) if has_pool else args.attack_count_no_pool
        print(f"\n=== category: {category} (pool: {'yes, ' + str(len(pool)) + ' attacks' if has_pool else 'NO -- Arm A/B skipped'}) ===")

        arm_a_runs, arm_b_runs, arm_c_runs = [], [], []

        for run_idx in range(1, args.runs_per_arm + 1):
            if has_pool:
                label_a = f"A_{category}_run{run_idx}"
                findings_a = run_arm_a(label_a, target_config, pool, args.out_dir)
                arm_a_runs.append(_success_rate(findings_a, attack_count, "completed", "all pool attacks sent verbatim"))

                label_b = f"B_{category}_run{run_idx}"
                result_b, _session_b = run_arm_b(
                    label_b, target_config, reasoning_variants_config, goal, pool,
                    attack_count, known_tool_ids, args.out_dir,
                )
                arm_b_runs.append(_success_rate(
                    result_b.findings, attack_count, result_b.termination.reason, result_b.termination.detail
                ))
            else:
                arm_a_runs.append(None)
                arm_b_runs.append(None)

            label_c = f"C_{category}_run{run_idx}"
            result_c, _session_c = run_arm_c(
                label_c, target_config, reasoning_config, category, attack_count, known_tool_ids, args.out_dir,
            )
            arm_c_runs.append(_success_rate(
                result_c.findings, attack_count, result_c.termination.reason, result_c.termination.detail
            ))

        per_category[category] = {
            "has_pool": has_pool,
            "attack_count": attack_count,
            "arm_a_runs": arm_a_runs,
            "arm_b_runs": arm_b_runs,
            "arm_c_runs": arm_c_runs,
        }

    write_summary(args.out_dir, operator, target_config, reasoning_config, reasoning_variants_config, args, per_category)
    print(f"\nDone. Wrote {args.out_dir}/*.jsonl and summary.json/.md.")
    return 0


def _spread(runs: list[dict | None]) -> float | None:
    """max-min spread of success_rate across a set of run results -- the
    within-arm variance measurement. None if fewer than 2 runs have data."""
    rates = [r["success_rate"] for r in runs if r and r["success_rate"] is not None]
    if len(rates) < 2:
        return None
    return max(rates) - min(rates)


def write_summary(out_dir, operator, target_config, reasoning_config, reasoning_variants_config, args, per_category) -> None:
    summary = {
        "run_timestamp_utc": now_utc_iso(),
        "operator": operator,
        "target_name": target_config.name,
        "target_base_url": target_config.base_url,
        "reasoning_llm_free_gen": reasoning_config.name,
        "reasoning_llm_variants": reasoning_variants_config.name,
        "reasoning_variants_temperature": reasoning_variants_config.temperature,
        "runs_per_arm": args.runs_per_arm,
        "per_category": per_category,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _restrict_to_owner(out_dir / "summary.json")

    lines = [
        "# Three-arm experiment: verbatim pool vs. variant vs. free-generation",
        "",
        f"- Target: `{target_config.base_url}` ({target_config.model})",
        f"- Reasoning LLM (Arm C): {reasoning_config.name}",
        f"- Reasoning LLM (Arm B): {reasoning_variants_config.name} "
        f"(temperature: {reasoning_variants_config.temperature})",
        f"- Runs per arm: {args.runs_per_arm}",
        f"- Run at (UTC): {summary['run_timestamp_utc']}",
        f"- Operator: {operator}",
        "",
        "success_rate = fraction of attempts verdicted FAIL (the attack landed). "
        "Arms without a pool for a category (see below) have no Arm A/B rows -- this is a real "
        "coverage gap, not an omission from this report.",
        "",
        "**Statistical honesty:** small trial counts per cell (attack_count x runs_per_arm). "
        "Read differences under roughly a 20-point spread as noise, not a finding.",
        "",
    ]

    categories_with_pool = [c for c, d in per_category.items() if d["has_pool"]]
    categories_without_pool = [c for c, d in per_category.items() if not d["has_pool"]]
    lines.append(f"**Categories with a pool (Arms A/B run):** {', '.join(categories_with_pool) or '(none)'}")
    lines.append(f"**Categories with NO pool (Arm C only):** {', '.join(categories_without_pool) or '(none)'}")
    lines.append("")

    lines.append("## Per-category, per-arm results")
    lines.append("")
    lines.append(
        "Rows marked INCOMPLETE stopped before `attack_count` attempts (a reasoning-LLM JSON "
        "parse failure, an invalid pool id, a guardrail halt, ...) and are not directly comparable "
        "to a full run at face value -- see the termination detail beneath each one."
    )
    lines.append("")
    for category, data in per_category.items():
        lines.append(f"### {category} (attack_count={data['attack_count']}, pool: {data['has_pool']})")
        lines.append("")
        lines.append("| Arm | Run | Attempts | FAIL | PASS | UNCERTAIN | Success rate | Complete? |")
        lines.append("|---|---|---|---|---|---|---|---|")
        incomplete_notes = []
        for arm_label, runs in (("A (verbatim)", data["arm_a_runs"]), ("B (variant)", data["arm_b_runs"]), ("C (free-gen)", data["arm_c_runs"])):
            for i, r in enumerate(runs, start=1):
                if r is None:
                    lines.append(f"| {arm_label} | {i} | - | - | - | - | (no pool) | - |")
                else:
                    rate = f"{r['success_rate']:.0%}" if r["success_rate"] is not None else "n/a"
                    complete_marker = "yes" if r["complete"] else f"**INCOMPLETE** ({r['total']}/{r['expected']})"
                    lines.append(
                        f"| {arm_label} | {i} | {r['total']} | {r['FAIL']} | {r['PASS']} | {r['UNCERTAIN']} | "
                        f"{rate} | {complete_marker} |"
                    )
                    if not r["complete"]:
                        incomplete_notes.append(
                            f"- {arm_label} run {i}: stopped at {r['total']}/{r['expected']} -- "
                            f"**{r['termination_reason']}**: {r['termination_detail']}"
                        )
            spread = _spread(runs)
            if spread is not None:
                lines.append(f"| {arm_label} | spread across runs | | | | | {spread:.0%} | |")
        lines.append("")
        if incomplete_notes:
            lines.append("**Incomplete runs in this category:**")
            lines.extend(incomplete_notes)
            lines.append("")

    lines.append("## Overall (pooled across categories with data)")
    lines.append("")
    for arm_key, arm_label in (("arm_a_runs", "A (verbatim)"), ("arm_b_runs", "B (variant)"), ("arm_c_runs", "C (free-gen)")):
        all_runs = [r for data in per_category.values() for r in data[arm_key] if r is not None]
        total = sum(r["total"] for r in all_runs)
        fail = sum(r["FAIL"] for r in all_runs)
        rate = f"{fail / total:.0%}" if total else "n/a"
        incomplete_count = sum(1 for r in all_runs if not r["complete"])
        caveat = (
            f" -- **includes {incomplete_count} incomplete run(s), small-n, do not trust this rate**"
            if incomplete_count
            else ""
        )
        lines.append(f"- **{arm_label}:** {fail}/{total} FAIL across all runs/categories with data -- {rate}{caveat}")
    lines.append("")

    lines.append("## Manual follow-up still needed")
    lines.append("")
    lines.append(
        "- **Arm B drift:** pool_text/variant_text are logged side by side in B_*.jsonl. "
        "Hand-sample ~20% of variant iterations, record how many drifted into a materially "
        "different attack, and report the drift rate -- there's no cheap automated check for this "
        "(an LLM judge would just be another model self-reporting)."
    )
    lines.append(
        "- **Pool coverage:** if this experiment matters enough to run repeatedly, building out "
        "attack pools for the categories listed under \"NO pool\" above turns this from a "
        "single-category comparison into the full five-category one the design calls for."
    )
    lines.append("")

    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    _restrict_to_owner(out_dir / "summary.md")


if __name__ == "__main__":
    sys.exit(main())
