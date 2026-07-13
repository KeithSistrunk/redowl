"""Validates a `redowl hunt` result JSON file: every executed attack must be a real pool member.

Usage:
    python scripts/verify_hunt_result.py <hunt_result.json>

Checks `hunt.iterations[].attack_id` -- NOT a top-level `iterations` key.
The hunt result schema (Q4: "extended schema") keeps `meta` / `summary` /
`findings` identical in shape to `redowl run`'s output, and groups every
hunt-specific field (hunt_id, termination, iterations, ...) under one
top-level `hunt` key rather than duplicating iterations at the top level,
which would be a maintenance trap (two copies to keep in sync).

Exits 0 and prints OK if every iteration's attack_id starts with "PI-"
(i.e. came from the pool); exits 1 and prints the fabricated/invalid ID(s)
otherwise -- that would mean the reasoning LLM's structural guardrail
(redowl/agent.py's pool-membership check) was bypassed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python scripts/verify_hunt_result.py <hunt_result.json>", file=sys.stderr)
        return 2

    path = Path(argv[1])
    data = json.loads(path.read_text(encoding="utf-8"))

    iterations = data["hunt"]["iterations"]
    bad = [it["attack_id"] for it in iterations if not it["attack_id"].startswith("PI-")]
    if bad:
        print(f"FAIL: fabricated/invalid attack id(s) found in hunt.iterations: {bad}", file=sys.stderr)
        return 1

    print(f"OK: all {len(iterations)} iteration(s) in {path} use a valid pool attack_id.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
