"""Scan data-plane streaming runtime (#302).

Collaborators extracted from ``server.py`` so the materialize/streams routes can
move into ``api/routers/*`` without a ``from strata.server import _private``
runtime scatter. Phase 1: the stream registry + ``StreamState``. Later phases add
the scan-build/prefetch manager and the two-tier QoS admission. See
``docs/internal/design-stream-runtime-extraction.md``.
"""

from strata.streaming.registry import StreamRegistry, StreamState
from strata.streaming.scan_builds import ScanBuildManager

__all__ = ["ScanBuildManager", "StreamRegistry", "StreamState"]
