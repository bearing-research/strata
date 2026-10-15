"""Strata: a materialization and persistence layer for expensive computations.

This is the ``strata-notebook`` distribution: the server, the notebook and the CLI.
The HTTP client is the separate ``strata-client`` package (``strata_client``); the
two share only the JSON wire protocol, not code.

Import from the submodules (``strata.filters``, ``strata.config``, ...); the
package root exports nothing, so importing any submodule loads only what it uses.
"""
