"""Loads target/test configuration, calls the target endpoint, and collects raw responses.

Endpoint format decision (see build prompt Q1): only ``openai`` (chat/completions,
OpenAI-compatible) is implemented in this MVP. ``anthropic`` and ``generic`` are
recognized in config but raise NotImplementedError, reserved for phase 2.

Judge provider decision (Q2): the judge LLM uses the same provider/format as the
target (also OpenAI-compatible), but can point at a different base_url, api key,
and model via the ``judge:`` block in the target config.

Phase 2.0 addition (see Phase 2.0 build prompt Q1): the reasoning LLM used by
`redowl hunt` is, like the target and judge, OpenAI-compatible first. It reuses
``call_openai_endpoint`` unchanged via ``call_reasoning`` below -- no new HTTP
client code, no new dependency. ``ReasoningConfig`` is a separate, narrower
dataclass from ``TargetConfig`` because the reasoning LLM has no system prompt
of its own (agent.py supplies one per call), no judge, and no blocklist.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
import yaml

_LOCAL_HOSTNAMES = {"localhost", "127.0.0.1", "::1"}


@dataclass
class JudgeConfig:
    """Configuration for the optional LLM-as-judge used on ambiguous verdicts."""

    enabled: bool
    base_url: str
    api_key_env: str
    model: str
    timeout_seconds: float = 30.0
    requests_per_second: float = 1.0


@dataclass
class TargetConfig:
    """Configuration for the endpoint under test, loaded from a YAML file."""

    name: str
    endpoint_format: str
    base_url: str
    api_key_env: str
    model: str
    system_prompt: str | None
    timeout_seconds: float
    requests_per_second: float
    blocklist: list[str]
    judge: JudgeConfig
    source_file: str


@dataclass
class TestCase:
    """A single prompt-injection or jailbreak test case, loaded from a YAML file."""

    id: str
    category: str
    description: str
    prompt: str
    expected_safe_behavior: list[str]
    evaluation: dict[str, Any]
    source_file: str


@dataclass
class RawResponse:
    """The raw result of calling the target endpoint for one test case."""

    text: str | None
    http_status: int | None
    latency_ms: float
    error: str | None
    raw: dict[str, Any] | None


@dataclass
class ReasoningConfig:
    """Configuration for the reasoning LLM used by `redowl hunt` to pick attacks."""

    name: str
    endpoint_format: str
    base_url: str
    api_key_env: str
    model: str
    timeout_seconds: float
    requests_per_second: float
    source_file: str
    # None (the default) omits the field from the request entirely -- same
    # behavior as before this field existed. Phase 2.1's variant rewording
    # (redowl/agent.py's generate_variant()) needs this set to something
    # nonzero: at temperature 0 every "variant" of a given pool item comes
    # back byte-identical whenever the prompt is otherwise the same (e.g.
    # iteration 1 of two different hunt runs), which would make the
    # three-arm experiment's variant arm look artificially noise-free.
    temperature: float | None = None


class ConfigError(Exception):
    """Raised when a target or test-case YAML file is missing required fields."""


def load_target_config(path: Path) -> TargetConfig:
    """Parse a target config YAML file into a TargetConfig."""
    with path.open("r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"{path}: invalid YAML: {exc}") from exc

    required = ["name", "endpoint_format", "base_url", "api_key_env", "model"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ConfigError(f"{path}: missing required field(s): {', '.join(missing)}")

    rate_limit = data.get("rate_limit") or {}

    judge_data = data.get("judge") or {}
    judge = JudgeConfig(
        enabled=bool(judge_data.get("enabled", False)),
        base_url=judge_data.get("base_url", data["base_url"]),
        api_key_env=judge_data.get("api_key_env", data["api_key_env"]),
        model=judge_data.get("model", data["model"]),
        timeout_seconds=float(judge_data.get("timeout_seconds", 30.0)),
        requests_per_second=float(
            judge_data.get("requests_per_second", rate_limit.get("requests_per_second", 1.0))
        ),
    )

    return TargetConfig(
        name=data["name"],
        endpoint_format=data["endpoint_format"],
        base_url=data["base_url"].rstrip("/"),
        api_key_env=data["api_key_env"],
        model=data["model"],
        system_prompt=data.get("system_prompt"),
        timeout_seconds=float(data.get("timeout_seconds", 30.0)),
        requests_per_second=float(rate_limit.get("requests_per_second", 1.0)),
        blocklist=list(data.get("blocklist", [])),
        judge=judge,
        source_file=str(path),
    )


def load_reasoning_config(path: Path) -> ReasoningConfig:
    """Parse a reasoning-LLM config YAML file (for `redowl hunt`) into a ReasoningConfig."""
    with path.open("r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"{path}: invalid YAML: {exc}") from exc

    required = ["name", "endpoint_format", "base_url", "api_key_env", "model"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ConfigError(f"{path}: missing required field(s): {', '.join(missing)}")

    rate_limit = data.get("rate_limit") or {}

    return ReasoningConfig(
        name=data["name"],
        endpoint_format=data["endpoint_format"],
        base_url=data["base_url"].rstrip("/"),
        api_key_env=data["api_key_env"],
        model=data["model"],
        timeout_seconds=float(data.get("timeout_seconds", 30.0)),
        requests_per_second=float(rate_limit.get("requests_per_second", 1.0)),
        source_file=str(path),
        temperature=float(data["temperature"]) if "temperature" in data else None,
    )


def load_test_cases(tests_dir: Path) -> list[TestCase]:
    """Load and parse every *.yaml/*.yml file in tests_dir as a TestCase, sorted by id."""
    if not tests_dir.is_dir():
        raise ConfigError(f"tests directory not found: {tests_dir}")

    paths = sorted(list(tests_dir.glob("*.yaml")) + list(tests_dir.glob("*.yml")))
    if not paths:
        raise ConfigError(f"no *.yaml test case files found in: {tests_dir}")

    cases: list[TestCase] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            try:
                data = yaml.safe_load(f) or {}
            except yaml.YAMLError as exc:
                raise ConfigError(f"{path}: invalid YAML: {exc}") from exc

        required = ["id", "category", "description", "prompt", "expected_safe_behavior", "evaluation"]
        missing = [key for key in required if key not in data]
        if missing:
            raise ConfigError(f"{path}: missing required field(s): {', '.join(missing)}")

        cases.append(
            TestCase(
                id=data["id"],
                category=data["category"],
                description=data["description"],
                prompt=data["prompt"],
                expected_safe_behavior=list(data["expected_safe_behavior"]),
                evaluation=dict(data["evaluation"]),
                source_file=str(path),
            )
        )

    return sorted(cases, key=lambda c: c.id)


def is_blocklisted(base_url: str, blocklist: list[str]) -> str | None:
    """Return the matching blocklist entry if base_url contains it, else None."""
    for entry in blocklist:
        if entry and entry in base_url:
            return entry
    return None


def transport_warning(base_url: str) -> str | None:
    """Return a warning if base_url would send the API key over plaintext HTTP to a non-local host."""
    parsed = urlparse(base_url)
    if parsed.scheme == "https":
        return None
    hostname = (parsed.hostname or "").lower()
    if hostname in _LOCAL_HOSTNAMES or hostname.startswith("127."):
        return None
    return (
        f"'{base_url}' uses '{parsed.scheme or 'no scheme'}', not https, and is not localhost -- "
        "the API key for this target will be sent in plaintext over the network."
    )


class RateLimiter:
    """Enforces a minimum interval between successive calls to the target endpoint."""

    def __init__(self, requests_per_second: float) -> None:
        self._min_interval = 1.0 / requests_per_second if requests_per_second > 0 else 0.0
        self._last_call: float | None = None

    def wait(self) -> None:
        """Block, if necessary, so calls stay under the configured rate."""
        if self._min_interval <= 0:
            return
        now = time.monotonic()
        if self._last_call is not None:
            remaining = self._min_interval - (now - self._last_call)
            if remaining > 0:
                time.sleep(remaining)
        self._last_call = time.monotonic()


def _api_key(env_var: str) -> str:
    import os

    key = os.environ.get(env_var)
    if not key:
        raise ConfigError(f"environment variable {env_var} is not set")
    return key


def call_openai_endpoint(
    base_url: str,
    api_key_env: str,
    model: str,
    system_prompt: str | None,
    prompt: str,
    timeout_seconds: float,
    temperature: float | None = None,
) -> RawResponse:
    """Call an OpenAI-compatible /chat/completions endpoint with a single user prompt.

    temperature is omitted from the request body entirely when None -- same
    behavior as before this parameter existed, so the target and judge call
    sites (which never pass it) are unaffected."""
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    body: dict[str, Any] = {"model": model, "messages": messages}
    if temperature is not None:
        body["temperature"] = temperature

    started = time.monotonic()
    try:
        api_key = _api_key(api_key_env)
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=body,
            timeout=timeout_seconds,
        )
        latency_ms = (time.monotonic() - started) * 1000
        raw = resp.json() if resp.content else None
        if resp.status_code != 200:
            return RawResponse(
                text=None,
                http_status=resp.status_code,
                latency_ms=latency_ms,
                error=f"HTTP {resp.status_code}: {resp.text[:500]}",
                raw=raw,
            )
        text = raw["choices"][0]["message"]["content"] if raw else None
        return RawResponse(text=text, http_status=resp.status_code, latency_ms=latency_ms, error=None, raw=raw)
    except ConfigError:
        raise
    except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
        latency_ms = (time.monotonic() - started) * 1000
        return RawResponse(text=None, http_status=None, latency_ms=latency_ms, error=str(exc), raw=None)


def call_target(config: TargetConfig, prompt: str) -> RawResponse:
    """Dispatch to the correct endpoint caller based on config.endpoint_format."""
    if config.endpoint_format == "openai":
        return call_openai_endpoint(
            base_url=config.base_url,
            api_key_env=config.api_key_env,
            model=config.model,
            system_prompt=config.system_prompt,
            prompt=prompt,
            timeout_seconds=config.timeout_seconds,
        )
    raise NotImplementedError(
        f"endpoint_format '{config.endpoint_format}' is not implemented in this MVP; "
        "only 'openai' is supported. 'anthropic' and 'generic' are planned for phase 2."
    )


def call_reasoning(config: ReasoningConfig, system_prompt: str, user_prompt: str) -> RawResponse:
    """Call the reasoning LLM endpoint used by `redowl hunt`.

    Only 'openai' endpoint_format is implemented (Phase 2.0 Q1 decision) --
    reuses call_openai_endpoint unchanged, same as call_target does for the
    target endpoint.
    """
    if config.endpoint_format == "openai":
        return call_openai_endpoint(
            base_url=config.base_url,
            api_key_env=config.api_key_env,
            model=config.model,
            system_prompt=system_prompt,
            prompt=user_prompt,
            timeout_seconds=config.timeout_seconds,
            temperature=config.temperature,
        )
    raise NotImplementedError(
        f"endpoint_format '{config.endpoint_format}' is not implemented for the reasoning LLM in "
        "this phase; only 'openai' is supported. Anthropic's Claude API is not OpenAI-compatible "
        "and is deferred to a later phase (see Phase 2.0 build prompt Q1)."
    )


def run_all(config: TargetConfig, test_cases: list[TestCase]) -> list[tuple[TestCase, RawResponse]]:
    """Run every test case against the target endpoint, respecting the configured rate limit."""
    limiter = RateLimiter(config.requests_per_second)
    results: list[tuple[TestCase, RawResponse]] = []
    for case in test_cases:
        limiter.wait()
        results.append((case, call_target(config, case.prompt)))
    return results
