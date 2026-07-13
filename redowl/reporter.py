"""Builds the findings report and writes it as JSON and as a Markdown summary."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from redowl.evaluator import Finding, Verdict


def build_report(findings: list[Finding], meta: dict[str, Any]) -> dict[str, Any]:
    """Assemble the full report dict: metadata, summary counts, and per-test findings."""
    counts = {verdict.value: 0 for verdict in Verdict}
    by_category: dict[str, dict[str, int]] = {}

    for finding in findings:
        counts[finding.verdict.value] += 1
        cat_counts = by_category.setdefault(finding.category, {v.value: 0 for v in Verdict})
        cat_counts[finding.verdict.value] += 1

    summary = {
        "total": len(findings),
        "counts": counts,
        "by_category": by_category,
    }

    findings_out = []
    for finding in findings:
        d = asdict(finding)
        d["verdict"] = finding.verdict.value
        findings_out.append(d)

    return {
        "meta": meta,
        "summary": summary,
        "findings": findings_out,
    }


def write_json_report(report: dict[str, Any], out_path: Path) -> None:
    """Write the report dict as pretty-printed JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")


def render_markdown(report: dict[str, Any]) -> str:
    """Render a human-readable Markdown summary from a report dict."""
    meta = report["meta"]
    summary = report["summary"]
    counts = summary["counts"]

    lines: list[str] = []
    lines.append(f"# Redowl Findings Report: {meta.get('target_name', 'unknown target')}")
    lines.append("")
    lines.append(f"- Target: `{meta.get('target_base_url', 'n/a')}`")
    lines.append(f"- Run at (UTC): {meta.get('run_timestamp_utc', 'n/a')}")
    lines.append(f"- Operator: {meta.get('operator', 'n/a')}")
    lines.append(f"- Total tests: {summary['total']}")
    lines.append(
        f"- Results: {counts.get('PASS', 0)} PASS / {counts.get('FAIL', 0)} FAIL / "
        f"{counts.get('UNCERTAIN', 0)} UNCERTAIN"
    )
    lines.append("")
    lines.append("## Results by category")
    lines.append("")
    lines.append("| Category | PASS | FAIL | UNCERTAIN |")
    lines.append("|---|---|---|---|")
    for category, cat_counts in sorted(summary["by_category"].items()):
        lines.append(
            f"| {category} | {cat_counts.get('PASS', 0)} | {cat_counts.get('FAIL', 0)} | "
            f"{cat_counts.get('UNCERTAIN', 0)} |"
        )
    lines.append("")
    lines.append("## Findings")
    lines.append("")

    for finding in report["findings"]:
        lines.append(f"### {finding['test_id']} — {finding['verdict']}")
        lines.append("")
        lines.append(f"- **Category:** {finding['category']}")
        lines.append(f"- **Description:** {finding['description']}")
        lines.append(f"- **Rule fired:** `{finding['rule_fired']}`")
        lines.append(f"- **Judge used:** {finding['judge_used']}")
        lines.append("")
        lines.append("**Prompt sent:**")
        lines.append("```")
        lines.append(finding["prompt"])
        lines.append("```")
        lines.append("")
        lines.append("**Response received:**")
        lines.append("```")
        lines.append(finding["response"] if finding["response"] is not None else "(no response)")
        lines.append("```")
        lines.append("")
        lines.append(f"**Evidence:** {finding['evidence']}")
        lines.append("")

    return "\n".join(lines)


def write_markdown_report(report: dict[str, Any], out_path: Path) -> None:
    """Render and write the Markdown summary."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_markdown(report), encoding="utf-8")


def now_utc_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()
