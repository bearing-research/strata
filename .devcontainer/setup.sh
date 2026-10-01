#!/usr/bin/env bash
set -euo pipefail

# Codespaces provisioning: install strata-notebook as a uv tool so the CLI is
# on PATH and the runtime guard passes. PyPI wheels bundle the native extension
# and frontend, so no Rust or Node is needed.
#
# To develop Strata itself, install Rust:
#   curl --proto '=https' -sSf https://sh.rustup.rs | sh -s -- -y
# then ``uv sync --all-extras`` in the cloned repo.

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv tool install strata-notebook
