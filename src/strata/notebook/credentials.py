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

``notebook_credential_writers`` names who may mount a credential read-write
(``{"lab": ["user_*"]}``), as globs over the principal who started the run. Once it is
set, an unlisted name is read-only. It limits what Strata stages and syncs back, not what
a cell can do with a credential's values should they reach it (see the docs).
"""

from __future__ import annotations

import fnmatch
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from strata.notebook.writer import drop_blanked_secrets


class CredentialError(ValueError):
    """A credential name, or a reference inside one, cannot be resolved."""


def resolve_reference(value: str, env: Mapping[str, str]) -> str:
    """``${VAR}`` to its value from *env*, then the process environment.

    Anything else is a literal and passes through: the registry is operator
    config, not a committed file.
    """
    if not (value.startswith("${") and value.endswith("}")):
        return value
    name = value[2:-1]
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
    ):
        self._registry = dict(registry or {})
        self._writers = {name: list(patterns) for name, patterns in (writers or {}).items()}
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
            writers=getattr(config, "notebook_credential_writers", None) or {},
            principal=principal,
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

        Raises:
            CredentialError: naming the credential.
        """
        if not self._writers:
            return
        for name in (self._scheme_defaults.get(scheme), credential):
            if not name:
                continue
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
                who = (
                    "a run with no principal" if self._principal is None else repr(self._principal)
                )
                raise CredentialError(
                    f"credential {name!r} is not writable by {who} "
                    "(STRATA_NOTEBOOK_CREDENTIAL_WRITERS)"
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
            return {key: resolve_reference(str(value), self._env) for key, value in fields.items()}
        except CredentialError as exc:
            raise CredentialError(f"credential {name!r} {exc}") from None

    def storage_options(
        self, scheme: str, credential: str | None, options: Mapping[str, Any]
    ) -> dict[str, Any]:
        """fsspec options for a mount: scheme default, then its credential, then its options.

        The mount's own ``options`` win, so a notebook can set a non-secret like
        ``endpoint_url`` on top of a credential.
        """
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
