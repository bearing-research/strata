#!/usr/bin/env bash
set -euo pipefail

# Codespaces provisioning: install uv, then drop strata-notebook into
# a uv-managed tool env so the CLI is on PATH and the runtime guard
# is happy. PyPI wheels bundle the native extension + frontend SPA,
# so no Rust toolchain or Node build needed for the runtime path.
#
# To contribute to Strata itself (modify the Rust extension, frontend,
# or Python sources from a clone of this repo), install Rust manually:
#   curl --proto '=https' -sSf https://sh.rustup.rs | sh -s -- -y
# then ``uv sync --all-extras`` in the cloned repo.

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv tool install strata-notebook
