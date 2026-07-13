# Follow-ups from OWASP Hardening Review

Items that could not be fully addressed by a code-tree-only review, with the concrete next step for each. See `OWASP_HARDENING_REPORT.md` for the full verdicts and evidence.

## A06 — Ongoing dependency scanning (tool/process decision)

A real `pip-audit` scan was run as part of this review and one dependency (`requests`) was bumped to fix 2 real CVEs. That scan is a point-in-time snapshot — it will go stale as new CVEs are published.

**Next step:** decide how to re-run this on a cadence. Options, in order of effort:
1. Manually re-run `pip install pip-audit && pip-audit` before releases (no infra needed, zero setup).
2. Add a `pip-audit` step to CI once CI exists (see A08 below) — needs the CI/CD decision made first.

**Also flagged, not a redscope dependency:** `pip-audit` reported the audit venv's own bootstrapped `pip` (25.0.1) has several known CVEs, fixed in 26.x. This is the package manager tool, not something declared in `pyproject.toml`, so no code change was made — but worth running `python -m pip install --upgrade pip` in any environment (dev machine, future CI) that installs redscope.

## A06/A08 — Hash-pinned dependency lockfile (tool decision)

`pyproject.toml` pins exact versions (`==`) but not cryptographic hashes, so a compromised PyPI mirror or index could theoretically serve a different artifact under the same pinned version string without detection.

**Next step:** decide on a lockfile tool — `pip-compile --generate-hashes` (pip-tools) or `uv lock` are the two common options. This is a tooling choice with workflow implications (how `pip install -e .` is run), so it wasn't made unilaterally here.

## A08 — CI/CD integrity controls (infrastructure decision)

There is no CI/CD pipeline in this repo yet — it was explicitly out of scope for the original MVP build (see the original build prompt's "Explicitly out of scope" section). Software/data integrity controls that depend on CI/CD (signed commits, artifact signing, build provenance) can't be evaluated until one exists.

**Next step, if/when CI/CD is added:**
- Decide on a CI provider (GitHub Actions is the natural fit if this becomes a GitHub repo).
- Add the A06 `pip-audit` step above as a CI gate.
- Consider requiring signed commits if this becomes a multi-contributor project.

## A09 — File permission hardening on Windows (platform limitation)

`_restrict_to_owner()` (`redscope/cli.py:25-30`) calls `os.chmod(path, 0o600)` on the audit log and report files after writing. This is fully effective on Linux/macOS. On Windows (NTFS), `os.chmod` only toggles the read-only attribute — it does not produce POSIX-equivalent owner-only ACLs, so on Windows these files remain readable by other local accounts with filesystem access.

**Next step, if stronger guarantees are needed on Windows specifically:** use `icacls` (via `subprocess`, with a hardcoded argument list — not user input, so it wouldn't reopen A03) to set an explicit ACL restricting the file to the current user. Not done in this pass because it's Windows-specific extra surface for a guarantee the project may not need (this is a local-only, single-operator tool per the confirmed threat model) — worth a deliberate decision rather than a reflexive addition.

## A10 — Optional opt-in private-IP-range blocking (design decision, not applied)

The classic SSRF control (block `127.0.0.0/8`, `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`, `169.254.0.0/16` by default) was deliberately **not** applied — see `OWASP_HARDENING_REPORT.md` A10 for why it would break redscope's core local/internal-testing use case (e.g., the Ollama-on-localhost workflow already in use).

**Next step, if there's ever a need to run redscope against untrusted/third-party-supplied config files** (as opposed to configs the operator writes themselves): add an opt-in `--block-private-ranges` flag or a `blocklist: [private-ranges]` sentinel value that expands to the standard private CIDR blocks. This is a new feature/behavior decision, not a bug fix, so it wasn't added unilaterally in this pass.
