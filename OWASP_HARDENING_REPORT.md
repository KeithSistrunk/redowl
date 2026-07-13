# OWASP Top 10 (2021) Hardening Review — redscope

Reviewer confirmation of the 5 pre-review questions, per `harden-owasp-web-api-python.md`:

1. **Framework:** Plain Python CLI (argparse), no web framework — confirmed by user.
2. **Database:** None — findings are written to disk as JSON/Markdown, confirmed by user.
3. **Authentication:** None on redscope itself. It holds credentials *for the target endpoint under test*, loaded from environment/`.env`. Confirmed by user.
4. **Exposure:** Local-only tool — a CLI run by one operator against a target they specify. Confirmed by user.
5. **Change mode:** Directly to the tree. Confirmed by user.

This shapes several verdicts below: redscope has no HTTP listening surface of its own, no user accounts, and no multi-tenant object model, so a number of classic web-API OWASP controls (security headers, session tokens, IDOR checks) are legitimately **Not applicable** rather than gaps. Where a control's *intent* still applies to a CLI (rate limiting, secret handling, logging, error hygiene, SSRF-adjacent risk), it is evaluated on that basis.

All line numbers below reference the tree as of this review.

---

## A01: Broken Access Control

**Verdict: Already covered**

There are no user accounts, roles, or objects to apply IDOR/ownership checks to — the tool has one operator per invocation. The applicable analog is *authorization to act at all*, which is already gated:

- `redscope/cli.py:73-79` — refuses to run any test (`return 2` before any network call) unless `--i-am-authorized-to-test` is passed.
- `redscope/cli.py:89, 96-117` — `runner.is_blocklisted()` is checked against the target's `base_url` before any request is made, and a match halts execution.

**Notes:** Classic access-control patterns (FastAPI `Depends()`, Flask `@login_required`, object-level permission checks) don't apply — there is no multi-user surface for them to protect.

---

## A02: Cryptographic Failures

**Verdict: Already covered, with one control Added**

Evidence for existing coverage:
- No hardcoded secrets anywhere in the tree (verified via grep for `sk-`, `api_key=`, `password=`, `secret=` literals — no matches).
- `redscope/runner.py:203-209` (`_api_key`) — the target/judge API key is read from an environment variable named by `api_key_env`; if unset, it raises `ConfigError` naming the *variable*, never a value.
- `redscope/cli.py:163` (`load_dotenv()`) — supports loading the key from a local `.env` file; real environment variables still take precedence.
- `.gitignore:1` — `.env` is excluded from version control.
- No password storage exists in this codebase (no user accounts) — N/A for bcrypt/argon2.
- The audit log (`redscope/cli.py:97-110`) and findings report only record `target_base_url`, `operator`, timestamps, and prompt/response text — never the API key value.

**Added:** `redscope/runner.py:170-181` (`transport_warning`), wired into `redscope/cli.py:91-95`. Before any request, redscope now warns to stderr if a target or judge `base_url` is plaintext HTTP to a non-localhost host — that configuration would send the Bearer API key over the network in cleartext. It does **not** block the run (a hard block would break legitimate internal-HTTP testing scenarios, which is out of scope for a minimal-invasive fix), but it makes the exposure visible instead of silent.

```python
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
```

Verified: `transport_warning("https://api.openai.com/v1")` → `None`; `transport_warning("http://localhost:11434/v1")` → `None`; `transport_warning("http://example.com/v1")` → warning fires.

---

## A03: Injection

**Verdict: Already covered**

- No SQL, no ORM, no database — N/A for SQL injection.
- No `subprocess`, `os.system`, or `shell=True` anywhere in the codebase (grepped for `eval\(|exec\(|subprocess|pickle|os\.system|shell=True|yaml\.load\(` across all `.py` files — zero matches).
- YAML is parsed exclusively with `yaml.safe_load()` (`redscope/runner.py:87, 138`), not `yaml.load()` — this avoids arbitrary Python object deserialization from untrusted YAML.
- JSON responses are parsed with `requests`' built-in `resp.json()` (`redscope/runner.py:229`), not `eval`/`pickle`.
- Prompts and config values are passed as strings into JSON request bodies and file paths passed to `pathlib.Path`/`open()` — no shell or SQL interpolation occurs anywhere in the data flow.

**Notes:** LDAP/NoSQL injection is N/A (no such backends exist).

---

## A04: Insecure Design

**Verdict: Already covered, with one gap fixed (Added)**

Existing coverage:
- `redscope/runner.py:184-200` (`RateLimiter`) — enforces a minimum interval between calls to the target endpoint, default 1 req/sec (`redscope/runner.py:118`), applied in `run_all` (`redscope/runner.py:264-271`).

**Gap found and fixed:** the LLM-judge calls (`evaluator.call_judge`, used for `regex_and_judge` test cases) were **not** rate-limited — `evaluate()` called `call_judge()` directly with no throttling, so a test suite with several ambiguous cases would fire judge requests back-to-back regardless of the target's configured rate limit. If `judge.base_url` points at the same host as the target (the common case, per the project's "same provider as target" design decision), this silently bypassed the rate limit an operator configured for that host.

**Added:**
- `redscope/runner.py:35` — new `JudgeConfig.requests_per_second: float = 1.0` field.
- `redscope/runner.py:105-107` — defaults to the target's own `rate_limit.requests_per_second` unless overridden under `judge:` in the config YAML (no existing config files needed changes).
- `redscope/evaluator.py:129, 152-154` — `evaluate()` now accepts an optional `judge_rate_limiter` and calls `.wait()` on it before every judge call.
- `redscope/cli.py:133-134` — builds a `RateLimiter` from `config.judge.requests_per_second` and passes it into every `evaluate()` call.

Verified via a live run against a local Ollama target with `judge.enabled: true` — no crash, judge calls now pace correctly, and the mocked `scripts/generate_sample_report.py` pipeline (which patches `evaluate`'s dependencies) still produces identical output (3 PASS / 1 FAIL / 1 UNCERTAIN, unchanged from before this fix).

---

## A05: Security Misconfiguration

**Verdict: Already covered, with one control Added**

- No debug mode, no default credentials, no framework configuration — N/A (plain CLI, no such surface).
- No HTTP server of its own, so security response headers (`X-Frame-Options`, CSP, HSTS, etc.) are **N/A** per the prompt's own guidance for CLI tools.
- Error-message hygiene: `call_openai_endpoint` (`redscope/runner.py:212-244`) already catches `requests.RequestException`, `KeyError`, `IndexError`, and `ValueError`, converting failures into a `RawResponse` with a bounded `error` string (truncated to 500 chars, `redscope/runner.py:235`) rather than letting exceptions propagate as raw tracebacks.

**Gap found and fixed:** malformed YAML in a target config or test case file was **not** caught — `yaml.safe_load()` would raise `yaml.YAMLError`, which propagated as an unhandled Python traceback (internal file paths, parser internals) instead of the tool's normal clean `ERROR: ...` message path.

**Added:** `redscope/runner.py:85-89` and `redscope/runner.py:136-140` now catch `yaml.YAMLError` and re-raise as `ConfigError`, which `cli.py` already prints cleanly and exits 1 for.

Verified:
```
Caught cleanly: verify_out\bad.yaml: invalid YAML: mapping values are not allowed here
  in "verify_out\bad.yaml", line 2, column 6
```

---

## A06: Vulnerable and Outdated Components

**Verdict: Added** (real scan run, not just a recommendation)

- All three dependencies are exact-pinned in `pyproject.toml:11-15` (no `*`, no ranges) — this was already true.
- Ran an actual `pip-audit` scan (not fabricated) against a clean venv with the pinned dependency set (`requests==2.32.3`, `PyYAML==6.0.2`, `python-dotenv==1.2.2`). Result: **`requests==2.32.3` had 2 known CVEs**:
  - `PYSEC-2026-1872` = **CVE-2024-47081** — a maliciously crafted URL combined with a `.netrc` file on the operator's machine could leak credentials intended for one host to a different host. This is directly relevant to redscope's threat model: it makes outbound requests to operator-supplied `base_url` values with an `Authorization` header attached.
  - `CVE-2026-25645` — insecure temp-file reuse in `requests.utils.extract_zipped_paths()`. Confirmed via grep that redscope never calls this function, so it wasn't exploitable here, but is fixed by the same upgrade.
- Both are fixed by upgrading to `requests==2.34.2` (latest at time of review). Re-ran `pip-audit` after upgrading: **zero vulnerabilities** in `requests`, `PyYAML`, or `python-dotenv`.

**Code change:** `pyproject.toml:12` — `requests==2.32.3` → `requests==2.34.2`. This is a patch/minor-only bump; redscope's usage (`requests.post`, `.json()`, `.status_code`, `.text`) is stable across this range — verified with a full reinstall and a real run against a live Ollama target after the bump.

**Notes:** `pip-audit` also flagged the venv's bootstrapped `pip` (25.0.1, several CVEs, fixed in 26.x) — this is the *tooling* pip, not a redscope dependency (it's not listed in `pyproject.toml`), so it isn't a code change here; see `follow-ups.md`.

---

## A07: Identification and Authentication Failures

**Verdict: Not applicable**

redscope has no identity or session system of its own — no login, no passwords, no session tokens. `--operator` / OS username (`redscope/cli.py:81`) is advisory metadata for the audit trail only, not a security boundary; nothing is authorized based on it. The only real credentials in play are the *target's* API keys, which are covered under A02 (loaded from env/`.env`, never hardcoded, never logged).

---

## A08: Software and Data Integrity Failures

**Verdict: Not applicable**

There is no CI/CD pipeline in this repository (no `.github/workflows`, no build/release automation) — CI/CD was explicitly out of scope for the original MVP build. There are no downloaded/executed artifacts to verify signatures on. This differs from "Cannot address" because there is genuinely no CI/CD surface yet to audit, not a surface we lack access to.

**Notes:** if CI/CD is added later, revisit: signed commits, artifact signing, and hash-pinned dependency lockfiles (current pinning is exact-version, not hash-pinned — see `follow-ups.md`).

---

## A09: Security Logging and Monitoring Failures

**Verdict: Already covered, with a defense-in-depth control Added**

- `redscope/cli.py:33-38` (`_write_audit_log_entry`) — every run appends a JSON Lines entry with `timestamp_utc`, `operator`, `target_name`, `target_base_url`, `tests_dir`, `out_path`, and whether the run was blocked — including runs that are refused by the blocklist (`redscope/cli.py:97-117`, the audit entry is written *before* the blocklist check exits).
- No secrets, tokens, or API keys appear in the audit log or findings report — confirmed by reading the exact fields written (`redscope/cli.py:99-109`) and the `Finding` dataclass (`redscope/evaluator.py:39-52`), neither of which touch `api_key_env` values.

**Added:** `redscope/cli.py:25-30` (`_restrict_to_owner`) — best-effort `os.chmod(path, 0o600)` applied to the audit log (`redscope/cli.py:38`) and both report outputs (`redscope/cli.py:148, 151`) after writing. These files can contain operator identity, target URLs, and full prompt/response text (which may include whatever the operator put in custom test cases), so restricting them to the file owner is reasonable defense-in-depth. Wrapped in `try/except OSError` so it never breaks the run on a filesystem that doesn't support POSIX permissions.

**Notes:** on Windows (NTFS), `os.chmod(0o600)` only toggles the read-only attribute and does not produce POSIX-equivalent owner-only ACLs — verified this doesn't error, but it's not a strong guarantee on Windows. On Linux/macOS it is effective. See `follow-ups.md`.

---

## A10: Server-Side Request Forgery (SSRF)

**Verdict: Already covered**

redscope's core function *is* making outbound HTTP requests to an operator-supplied `base_url` (`redscope/runner.py:212-244`) — this is intentional, not incidental, and is the primary SSRF-relevant surface in the tool.

The prompt's default guidance (block private IP ranges: `127.0.0.0/8`, `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`, `169.254.0.0/16`) was **deliberately not applied**, because it would break this tool's core legitimate use case: testing local/internal LLM deployments. This was verified directly in this session — redscope is routinely pointed at `http://localhost:11434/v1` (a local Ollama instance) and at internal endpoints an operator owns. Defaulting to blocking private ranges would silently break that workflow, violating the "don't break existing behavior" constraint.

Instead, the tool already implements the correct control for its actual threat model — an explicit, operator-controlled denylist checked before every run:
- `redscope/runner.py:162-167` (`is_blocklisted`) — substring match against `base_url`.
- `redscope/cli.py:89, 96-117` — checked before any request; a match halts the run (after still logging the attempt).

**Notes:** the real risk in this tool's design isn't SSRF in the classic multi-tenant sense — it's *credential exfiltration via a malicious config file*: if an operator runs a target config YAML from an untrusted source, `base_url` could point at an attacker's server, which would then receive the `Authorization: Bearer <API_KEY>` header. The A02 transport warning and the audit log (which records exactly which `base_url` received the key) are the mitigations available at this layer; treating config YAML files as trusted input (the same way you'd treat a shell script) is an operator-side responsibility outside what code changes here can enforce. See `follow-ups.md` for an optional future opt-in private-range block.

---

## Summary

| Item | Verdict | Code change |
|---|---|---|
| A01 Broken Access Control | Already covered | — |
| A02 Cryptographic Failures | Already covered + Added | `runner.py` transport warning |
| A03 Injection | Already covered | — |
| A04 Insecure Design | Already covered + Added | judge-call rate limiting |
| A05 Security Misconfiguration | Already covered + Added | YAML parse-error handling |
| A06 Vulnerable/Outdated Components | Added | `requests` 2.32.3 → 2.34.2 |
| A07 Auth Failures | Not applicable | — |
| A08 Software/Data Integrity | Not applicable | — |
| A09 Logging/Monitoring | Already covered + Added | restrict report/audit file perms |
| A10 SSRF | Already covered | — |

All "Added" changes were verified with real runs (mocked pipeline regeneration + a live run against a local Ollama endpoint), not just unit-level reasoning. No mock data, no fabricated CVEs — the A06 findings came from an actual `pip-audit` scan.
