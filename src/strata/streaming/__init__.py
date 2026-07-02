"""Scan data-plane streaming runtime (#302).

Collaborators extracted from ``server.py`` so the materialize/streams routes can
move into ``api/routers/*`` without a ``from strata.server import _private``
runtime scatter. Phase 1: the stream registry + ``StreamState``. Phase 2: the scan-build/prefetch
manager. Phase 3: the two-tier QoS admission (``QoSAdmission``). See
``docs/internal/design-stream-runtime-extraction.md``.
"""

from strata.streaming.qos import Admission, QoSAdmission, QoSRejected
from strata.streaming.registry import StreamRegistry, StreamState
from strata.streaming.scan_builds import ScanBuildManager

__all__ = [
    "Admission",
    "QoSAdmission",
    "QoSRejected",
    "ScanBuildManager",
    "StreamRegistry",
    "StreamState",
]
