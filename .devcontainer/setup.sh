#!/usr/bin/env bash
set -euo pipefail

# Codespaces provisioning: install strata-notebook as a uv tool so the CLI is
# on PATH and the runtime guard passes. The PyPI wheel bundles the frontend,
# so no Node is needed.
#
# To develop Strata itself, ``uv sync --all-extras`` in the cloned repo.

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# The sql extras let the Codespace add SQL cells against SQLite and DuckDB.
uv tool install 'strata-notebook[sql,sql-duckdb,sql-sqlite]'
