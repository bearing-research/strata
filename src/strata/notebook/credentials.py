"""Named credentials: data sources configured once, referenced by name.

A mount or connection says ``credential = "lab-bucket"`` (a catalog,
``strata_credential``) and never carries a secret. ``notebook_credentials`` maps
each name to fields (fsspec ``storage_options``, driver ``auth`` or catalog
properties) whose values are ``${VAR}`` references::

    STRATA_NOTEBOOK_CREDENTIALS='{"lab-bucket": {"key": "${LAB_AWS_KEY}",
                                                 "secret": "${LAB_AWS_SECRET}"}}'

References resolve against the notebook's environment (where a secret manager
puts what it fetches), then the server's. The name is folded into provenance;
the values are not, so rotating a secret invalidates nothing.
``notebook_mount_credentials`` names a default per URI scheme (``{"s3": "org-bucket"}``).

``notebook_credential_paths`` restricts a name to the notebooks under some directories
(``{"lab-a": ["/srv/projects/lab-a"]}``), for a server that hosts several projects. A
restricted name resolves for nothing else, a server-level use included. Paths compare as
written, symlinks unfollowed.

``notebook_credential_writers`` names who may write through a credential
(``{"lab": ["user_*"]}``), as globs over the principal who started the run: a read-write
mount, a ``write=true`` SQL connection, a ``@write_table`` catalog. Once it is set, even
to ``{}``, an unlisted name is read-only and a remote read-write mount must name one. It
limits what Strata writes, not what a cell can do with a credential's values should they
reach it (see the docs).

In service mode, and always on a remote worker, a mount's own ``options`` may set only
:data:`NOTEBOOK_MOUNT_OPTIONS`: the registry is the operator's, and a notebook naming a
token file or an endpoint would turn the server's credentials against it.
"""

from __future__ import annotations

import fnmatch
import os
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from strata.notebook.writer import drop_blanked_secrets


class CredentialError(ValueError):
    """A credential name, or a reference inside one, cannot be resolved."""


CREDENTIAL_FILE_LIMIT = 64 * 1024
"""The most a credential file may hold."""

# What a service-mode notebook's mount options may set, per scheme. None names a secret,
# a file the server reads or where requests go; those come only from the registry.
NOTEBOOK_MOUNT_OPTIONS: dict[str, frozenset[str]] = {
    "s3": frozenset(
        {"anon", "requester_pays", "version_aware", "cache_regions", "default_block_size"}
        | {"default_cache_type", "default_fill_cache", "use_listings_cache"}
    ),
    "gs": frozenset(
        {"requester_pays", "version_aware", "consistency", "cache_timeout", "block_size"}
        | {"use_listings_cache"}
    ),
    "az": frozenset(
        {"anon", "version_aware", "blocksize", "default_cache_type", "default_fill_cache"}
        | {"use_listings_cache"}
    ),
}


def is_reference(value: str) -> bool:
    """Whether *value* is a ``${VAR}`` or ``${file:/path}`` reference, not a literal."""
    return value.startswith("${") and value.endswith("}")


def file_in_roots(path: Path, roots: Sequence[Path]) -> Path:
    """*path* with every symlink resolved, when that lies inside one of *roots*.

    Raises:
        CredentialError: it does not, naming the path and the setting.
    """
    real = Path(os.path.realpath(path))
    if any(real.is_relative_to(os.path.realpath(root)) for root in roots):
        return real
    raise CredentialError(
        f"{path} is not inside notebook_credential_file_roots "
        f"(STRATA_NOTEBOOK_CREDENTIAL_FILE_ROOTS){'' if roots else ', which is empty'}"
    )


def read_credential_file(path: Path, *, nofollow: bool = False) -> str:
    """The text of a credential file: a regular file, at most :data:`CREDENTIAL_FILE_LIMIT`.

    Checked on the open descriptor, so a FIFO or a device cannot hang or flood the read.
    *nofollow* refuses a symlink, for a path already resolved and checked.

    Raises:
        OSError: it cannot be opened.
        CredentialError: it is not a regular file, or is too large.
    """
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
    if nofollow:
        flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, "rb") as handle:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise CredentialError(f"{path} is not a regular file")
        data = handle.read(CREDENTIAL_FILE_LIMIT + 1)
    if len(data) > CREDENTIAL_FILE_LIMIT:
        raise CredentialError(f"{path} is larger than {CREDENTIAL_FILE_LIMIT} bytes")
    return data.decode()


def resolve_reference(value: str, env: Mapping[str, str], file_roots: Sequence[Path] = ()) -> str:
    """``${VAR}`` to its value from *env*, then the process environment.

    ``${file:/path}`` is the file's contents, read at each use, without trailing line
    breaks, so a host rotates a secret by rewriting the file. With *file_roots*, the
    path must resolve inside one, checked at each read. Anything else is a literal and
    passes through: the registry is operator config, not a committed file.
    """
    if not is_reference(value):
        return value
    name = value[2:-1]
    if name.startswith("file:"):
        path = Path(name.removeprefix("file:"))
        if not path.is_absolute():
            raise CredentialError(f"references ${{{name}}}, which is not an absolute path")
        try:
            if file_roots:
                # Read the checked path itself, so a link swapped in after the check fails.
                path = file_in_roots(path, file_roots)
            return read_credential_file(path, nofollow=bool(file_roots)).rstrip("\r\n")
        except OSError as exc:
            raise CredentialError(
                f"references ${{{name}}}, which cannot be read ({exc.strerror})"
            ) from None
        except CredentialError as exc:
            raise CredentialError(f"references ${{{name}}}: {exc}") from None
    if name in env:
        return env[name]
    resolved = os.environ.get(name)
    if resolved is None:
        raise CredentialError(f"references ${{{name}}}, which is not set")
    return resolved


class CredentialResolver:
    """Turns credential names into the fields they stand for."""

    def __init__(
        self,
        registry: Mapping[str, Mapping[str, str]] | None = None,
        *,
        scheme_defaults: Mapping[str, str] | None = None,
        env: Mapping[str, str] | None = None,
        restricted: Mapping[str, Sequence[str]] | None = None,
        notebook: Path | None = None,
        writers: Mapping[str, Sequence[str]] | None = None,
        principal: str | None = None,
        restrict_options: bool = False,
        file_roots: Sequence[Path] = (),
    ):
        self._registry = dict(registry or {})
        self._file_roots = list(file_roots)
        # A notebook's mount options are then held to NOTEBOOK_MOUNT_OPTIONS.
        self._restrict_options = restrict_options
        # None is unset (every name writes); ``{}`` is set with nothing writable.
        self._writers = (
            None
            if writers is None
            else {name: list(patterns) for name, patterns in writers.items()}
        )
        self._principal = principal
        self._scheme_defaults = dict(scheme_defaults or {})
        self._env = drop_blanked_secrets(env or {})
        # Compared as written, without touching the filesystem: a notebook is judged by the
        # path the server opened it under, and a relative or ``..`` path is never inside.
        self._restricted = {
            name: [Path(os.path.normpath(Path(prefix).expanduser())) for prefix in prefixes]
            for name, prefixes in (restricted or {}).items()
        }
        self._notebook = (
            notebook
            if notebook is not None and notebook.is_absolute() and ".." not in notebook.parts
            else None
        )

    @classmethod
    def from_config(
        cls,
        config: Any,
        env: Mapping[str, str] | None = None,
        *,
        notebook: Path | None = None,
        principal: str | None = None,
    ) -> CredentialResolver:
        """The server's credentials as the notebook at *notebook*, run by *principal*, may
        use them."""
        return cls(
            getattr(config, "notebook_credentials", None) or {},
            scheme_defaults=getattr(config, "notebook_mount_credentials", None) or {},
            env=env,
            restricted=getattr(config, "notebook_credential_paths", None) or {},
            notebook=notebook,
            writers=getattr(config, "notebook_credential_writers", None),
            principal=principal,
            restrict_options=getattr(config, "deployment_mode", "personal") == "service",
            file_roots=getattr(config, "notebook_credential_file_roots", None) or (),
        )

    def ensure_allowed(self, name: str) -> None:
        """Refuse a restricted *name* outside its notebooks, by name only.

        Also the check before a remote worker, which resolves names from its own registry.

        Raises:
            CredentialError: naming the credential and never its fields.
        """
        prefixes = self._restricted.get(name)
        if prefixes is None:
            return
        if self._notebook is not None and any(self._notebook.is_relative_to(p) for p in prefixes):
            return
        raise CredentialError(
            f"credential {name!r} is not available to this notebook "
            "(STRATA_NOTEBOOK_CREDENTIAL_PATHS)"
        )

    def ensure_writable(self, scheme: str, credential: str | None) -> None:
        """Refuse a read-write mount through a credential this run's principal may not write.

        Both the scheme's default and the mount's own credential count: the write uses both.
        A remote mount that names neither would write as the server's own identity.

        Raises:
            CredentialError: naming the credential.
        """
        if self._writers is None:
            return
        names = [name for name in (self._scheme_defaults.get(scheme), credential) if name]
        if not names and scheme != "file":
            raise CredentialError(
                f"a read-write {scheme} mount must name the source it writes to with "
                "credential=<name> (STRATA_NOTEBOOK_CREDENTIAL_WRITERS)"
            )
        for name in names:
            self.ensure_may_write(name)

    def ensure_may_write(self, name: str) -> None:
        """Refuse a write through credential *name* by a principal the setting does not list.

        Raises:
            CredentialError: naming the credential.
        """
        if self._writers is None:
            return
        patterns = self._writers.get(name)
        if patterns is None:
            raise CredentialError(
                f"credential {name!r} is read-only on this server "
                "(STRATA_NOTEBOOK_CREDENTIAL_WRITERS)"
            )
        if not any(
            pattern == "*"
            or (self._principal is not None and fnmatch.fnmatchcase(self._principal, pattern))
            for pattern in patterns
        ):
            who = "a run with no principal" if self._principal is None else repr(self._principal)
            raise CredentialError(
                f"credential {name!r} is not writable by {who} (STRATA_NOTEBOOK_CREDENTIAL_WRITERS)"
            )

    def ensure_options_allowed(self, scheme: str, options: Mapping[str, Any]) -> None:
        """Refuse a mount option a notebook may not set here.

        Also the check before a remote worker, so a refused option never leaves the server.

        Raises:
            CredentialError: naming the option.
        """
        if not self._restrict_options:
            return
        allowed = NOTEBOOK_MOUNT_OPTIONS.get(scheme, frozenset())
        for key, value in options.items():
            # gcsfs spells anonymous access as a token.
            if key in allowed or (scheme == "gs" and key == "token" and value == "anon"):
                continue
            raise CredentialError(
                f"mount option {key!r} cannot be set by a notebook on this server: "
                "credentials, key files and endpoints come only from its named credentials "
                f"(a {scheme} mount may set {', '.join(sorted(allowed)) or 'no options'})"
            )

    def resolve(self, name: str) -> dict[str, str]:
        """The fields behind *name*, with every reference resolved.

        Raises:
            CredentialError: naming the credential, so the failure says which one to define.
        """
        self.ensure_allowed(name)
        fields = self._registry.get(name)
        if fields is None:
            raise CredentialError(
                f"credential {name!r} is not defined on this server (STRATA_NOTEBOOK_CREDENTIALS)"
            )
        try:
            return {
                key: resolve_reference(str(value), self._env, self._file_roots)
                for key, value in fields.items()
            }
        except CredentialError as exc:
            raise CredentialError(f"credential {name!r} {exc}") from None

    def storage_options(
        self, scheme: str, credential: str | None, options: Mapping[str, Any]
    ) -> dict[str, Any]:
        """fsspec options for a mount: scheme default, then its credential, then its options.

        The mount's own ``options`` win, so a notebook can set ``endpoint_url`` on top of a
        credential, where :meth:`ensure_options_allowed` lets it.
        """
        self.ensure_options_allowed(scheme, options)
        merged: dict[str, Any] = {}
        default = self._scheme_defaults.get(scheme)
        if default:
            merged.update(self.resolve(default))
        if credential:
            merged.update(self.resolve(credential))
        merged.update(options)
        return merged


CATALOG_CREDENTIAL_KEY = "strata_credential"
"""A catalog's named credential. Not ``credential``: that is pyiceberg's REST OAuth secret."""


def resolve_catalog_properties(
    properties: Mapping[str, str], resolver: CredentialResolver
) -> dict[str, str]:
    """A catalog's pyiceberg properties with its ``strata_credential`` resolved beneath them.

    Any ``credential`` key is pyiceberg's own (a REST catalog's ``client_id:secret``) and
    passes through untouched.
    """
    props = dict(properties)
    name = props.pop(CATALOG_CREDENTIAL_KEY, None)
    if not name:
        return props
    return {**resolver.resolve(name), **props}


def credential_identity(credential: str | None) -> str:
    """The provenance component for a credential: its name, never its values."""
    return f"credential={credential}" if credential else ""
