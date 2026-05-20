# Security Policy

## Supported Versions

Strata is at 0.1.0 — the first stable release. Security fixes are
shipped against the latest minor only. Pin to a patch (e.g.
`strata-notebook==0.1.0`) if you need stability; upgrade to the
latest patch in your minor when a security release lands.

| Version | Supported |
| ------- | --------- |
| 0.1.x   | Yes       |
| < 0.1   | No (pre-release alphas, do not deploy) |

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security reports.

Report privately via GitHub's
[Security Advisories](https://github.com/bearing-research/strata/security/advisories/new)
form. The advisory is private until a fix is published, and lets us
coordinate disclosure, request a CVE if appropriate, and credit the
reporter.

If you cannot use the GitHub form, email the maintainer instead:
**fangchen.li@outlook.com** (PGP not currently published — open an
advisory and request a key if you need one).

### What to include

- The affected component (notebook UI, core data plane, a specific
  driver adapter, the release/install path, etc.)
- A minimal reproduction or proof-of-concept
- The threat model you're assuming (personal deployment, service
  deployment, multi-tenant, etc. — see
  [deployment modes](https://github.com/bearing-research/strata/blob/main/CLAUDE.md#deployment-modes))
- The Strata version (`strata-notebook --version` or `pip show
  strata-notebook`) and Python version

### What to expect

- Acknowledgement within 3 business days.
- A triage decision (accepted / declined / needs-more-info) within
  10 business days.
- For accepted issues, a target fix window depending on severity:
  - Critical (RCE, auth bypass, credential disclosure): patch
    release within 7 days where feasible.
  - High: patch release within 30 days.
  - Medium / low: rolled into the next regular release.

We will credit reporters in the release notes unless you ask us not to.

## Scope

In scope:

- The published `strata-notebook` wheel on PyPI and its console
  scripts (`strata-notebook`, `strata`, `strata-worker`).
- The HTTP and WebSocket APIs documented in the repo.
- The bundled frontend SPA.

Out of scope (please do not report these as vulnerabilities):

- Personal-mode deployments bound to a non-loopback address without
  `allow_remote_clients_in_personal=True` — Strata refuses to start
  in that configuration and the guard is the security boundary.
- Cache content visible across principals — by design (see CLAUDE.md
  "Cache is shared across principals"); ACL gates request access,
  not cache contents.
- Issues that require an attacker who already controls the executor
  process or the notebook's underlying Python environment.
- Findings produced by static-analysis tools without a concrete
  exploit path (we triage these separately under our internal
  CodeQL backlog).
