"""Opening the Parquet files a scan plan lists, wherever they are stored.

The planner lists data files by the paths the catalog recorded: local paths,
``s3://``, ``gs://``, or Azure's ``abfs://`` / ``abfss://`` / ``az://``. The
fetcher and the metadata cache both open them through :func:`open_parquet`, so
a warehouse on any of those stores scans the same way.

S3 uses the filesystem built from Strata's S3 settings, unless the table's
catalog vended credentials for that table's location
(:func:`register_vended_credentials`, called by the planner when it loads a
table from a named catalog), in which case files under that location are read
with those. GCS and Azure use the same settings the artifact blob store does.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import pyarrow.fs as pafs
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from strata.config import StrataConfig

_lock = threading.Lock()
_config: StrataConfig | None = None
_gcs: pafs.FileSystem | None = None
_azure: dict[str, pafs.FileSystem] = {}
# Table location prefix -> filesystem built from credentials its catalog vended.
_vended: dict[str, pafs.FileSystem] = {}

_AZURE_SCHEMES = ("abfs://", "abfss://", "az://")
# The pyiceberg FileIO properties a catalog vends S3 credentials under.
_VENDED_S3 = {
    "s3.access-key-id": "access_key",
    "s3.secret-access-key": "secret_key",
    "s3.session-token": "session_token",
    "s3.region": "region",
    "s3.endpoint": "endpoint_override",
}


def configure(config: StrataConfig) -> None:
    """Use *config*'s GCS and Azure settings for files on those stores."""
    global _config, _gcs
    with _lock:
        _config = config
        _gcs = None
        _azure.clear()


def register_vended_credentials(location: str, properties: dict[str, str]) -> bool:
    """Read files under *location* with the S3 credentials in *properties*.

    Returns whether *properties* carried credentials at all; a catalog that
    vends none leaves reads on Strata's own S3 settings.
    """
    if not location.startswith("s3://") or "s3.access-key-id" not in properties:
        return False
    kwargs: dict[str, Any] = {
        argument: properties[key] for key, argument in _VENDED_S3.items() if properties.get(key)
    }
    filesystem = pafs.S3FileSystem(**kwargs)
    with _lock:
        _vended[location.rstrip("/") + "/"] = filesystem
    return True


def _vended_for(file_path: str) -> pafs.FileSystem | None:
    with _lock:
        best: str | None = None
        for prefix in _vended:
            # The most specific location wins.
            if file_path.startswith(prefix) and (best is None or len(prefix) > len(best)):
                best = prefix
        return _vended[best] if best is not None else None


def _gcs_filesystem() -> pafs.FileSystem:
    global _gcs
    with _lock:
        if _gcs is None:
            kwargs: dict[str, Any] = {}
            if _config is not None:
                if _config.gcs_credentials_json:
                    import os

                    from strata.blob_store import _resolve_gcs_credentials

                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _resolve_gcs_credentials(
                        _config.gcs_credentials_json
                    )
                if _config.gcs_anonymous:
                    kwargs["anonymous"] = True
                endpoint = _config.gcs_endpoint_override
                if endpoint:
                    parsed = urlparse(endpoint if "://" in endpoint else f"https://{endpoint}")
                    kwargs["endpoint_override"] = parsed.netloc
                    kwargs["scheme"] = parsed.scheme
            _gcs = pafs.GcsFileSystem(**kwargs)
        return _gcs


def _azure_filesystem(account: str) -> pafs.FileSystem:
    with _lock:
        filesystem = _azure.get(account)
        if filesystem is None:
            kwargs: dict[str, Any] = {"account_name": account}
            if _config is not None:
                if _config.azure_account_key:
                    kwargs["account_key"] = _config.azure_account_key
                if _config.azure_sas_token:
                    kwargs["sas_token"] = _config.azure_sas_token
                if _config.azure_endpoint_url:
                    parsed = urlparse(_config.azure_endpoint_url)
                    kwargs["blob_storage_authority"] = parsed.netloc
                    kwargs["blob_storage_scheme"] = parsed.scheme
            filesystem = pafs.AzureFileSystem(**kwargs)
            _azure[account] = filesystem
        return filesystem


def azure_path(file_path: str) -> tuple[str, str]:
    """``(account, container/path)`` for an Azure URI.

    ``abfs[s]://container@account.dfs.core.windows.net/path`` names both;
    ``az://container/path`` names only the container, and the account comes
    from ``azure_account_name``.
    """
    parsed = urlparse(file_path)
    if "@" in parsed.netloc:
        container, host = parsed.netloc.split("@", 1)
        account = host.split(".", 1)[0]
    else:
        container = parsed.netloc
        account = (_config.azure_account_name if _config is not None else None) or ""
    return account, f"{container}{parsed.path}"


def open_parquet(file_path: str, s3_filesystem: pafs.FileSystem | None = None) -> pq.ParquetFile:
    """Open *file_path* for reading, on whichever store it names."""
    if file_path.startswith("s3://"):
        filesystem = _vended_for(file_path) or s3_filesystem or pafs.S3FileSystem()
        return pq.ParquetFile(file_path[len("s3://") :], filesystem=filesystem)
    if file_path.startswith("gs://"):
        return pq.ParquetFile(file_path[len("gs://") :], filesystem=_gcs_filesystem())
    if file_path.startswith(_AZURE_SCHEMES):
        account, path = azure_path(file_path)
        return pq.ParquetFile(path, filesystem=_azure_filesystem(account))
    return pq.ParquetFile(file_path)


def reset() -> None:
    """Forget configuration and vended credentials (tests)."""
    global _config, _gcs
    with _lock:
        _config = None
        _gcs = None
        _azure.clear()
        _vended.clear()
