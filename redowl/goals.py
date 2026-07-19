"""Loads goal definitions and their attack pools from disk, for `redowl hunt`.

Attack pool location decision (Q5): ``goals/<goal-dir>/attacks/`` at the repo
root, e.g. ``goals/prompt-injection/attacks/``.

Attack pool YAML files deliberately reuse the existing test-case schema
(redowl/runner.TestCase) unchanged -- each attack file is loaded with
runner.load_test_cases(), the same loader `redowl run` uses for tests/. No
parallel schema is invented, per the build prompt.

A goal is looked up by the ``name:`` field inside its definition.yaml, not by
assuming the directory name matches the CLI's --goal value (the shipped goal
is directory ``goals/prompt-injection/`` but name ``prompt_injection``), so
this is robust to that naming difference without hardcoding a translation
rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from redowl.evaluator import Verdict
from redowl.runner import ConfigError, TestCase, load_test_cases


@dataclass
class GoalDefinition:
    """A goal's metadata and success criteria, loaded from definition.yaml."""

    name: str
    description: str
    success_criteria: dict[str, Any]
    default_max_iterations: int
    source_file: str


def _iter_goal_dirs(goals_root: Path):
    """Yield (goal_dir, definition_path) for every subdirectory of goals_root with a definition.yaml."""
    if not goals_root.is_dir():
        raise ConfigError(f"goals directory not found: {goals_root}")
    for entry in sorted(goals_root.iterdir()):
        definition_path = entry / "definition.yaml"
        if entry.is_dir() and definition_path.is_file():
            yield entry, definition_path


def load_goal_definition(goals_root: Path, goal_name: str) -> GoalDefinition:
    """Find and parse the definition.yaml under goals_root whose `name:` field matches goal_name."""
    for _goal_dir, definition_path in _iter_goal_dirs(goals_root):
        with definition_path.open("r", encoding="utf-8") as f:
            try:
                data = yaml.safe_load(f) or {}
            except yaml.YAMLError as exc:
                raise ConfigError(f"{definition_path}: invalid YAML: {exc}") from exc

        if data.get("name") != goal_name:
            continue

        required = ["name", "description", "success_criteria", "default_max_iterations"]
        missing = [key for key in required if key not in data]
        if missing:
            raise ConfigError(f"{definition_path}: missing required field(s): {', '.join(missing)}")

        return GoalDefinition(
            name=data["name"],
            description=data["description"],
            success_criteria=dict(data["success_criteria"]),
            default_max_iterations=int(data["default_max_iterations"]),
            source_file=str(definition_path),
        )

    raise ConfigError(f"no goal named '{goal_name}' found under {goals_root}")


def load_attack_pool(goal_dir: Path) -> list[TestCase]:
    """Load the attack pool for a goal directory, reusing the existing TestCase loader/schema."""
    return load_test_cases(goal_dir / "attacks")


def load_goal(goals_root: Path, goal_name: str) -> tuple[GoalDefinition, list[TestCase]]:
    """Load a goal's definition and its attack pool together."""
    definition = load_goal_definition(goals_root, goal_name)
    goal_dir = Path(definition.source_file).parent
    attacks = load_attack_pool(goal_dir)
    return definition, attacks


def load_all_pool_ids(goals_root: Path) -> set[str]:
    """Collect every attack id across every goal's pool under goals_root.

    Used as the known-tool-ids registry for guardrails.SessionState in
    free-generation hunts (see redowl/guardrails.py): a free-generated attack
    that references an id outside this set is a fabricated reference. Best
    effort -- a goal directory with no attacks/ subdir or invalid YAML is
    skipped rather than failing the whole registry build, since this is a
    reference set for a secondary check, not a required load path.
    """
    ids: set[str] = set()
    if not goals_root.is_dir():
        return ids
    for goal_dir, _definition_path in _iter_goal_dirs(goals_root):
        try:
            ids.update(attack.id for attack in load_attack_pool(goal_dir))
        except ConfigError:
            continue
    return ids


def is_goal_achieved(success_criteria: dict[str, Any], verdict: Verdict) -> bool:
    """Check whether one attack's verdict satisfies the goal's success criteria.

    Phase 2.0 supports exactly one criterion: any_attack_fails (the hunt
    succeeds the first time any attack in the pool is verdicted FAIL).
    """
    if success_criteria.get("any_attack_fails"):
        return verdict == Verdict.FAIL
    return False
