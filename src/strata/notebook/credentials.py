"""Named credentials: data sources configured once, referenced by name.

A mount or a connection in ``notebook.toml`` says ``credential = "lab-bucket"``
and never carries a secret. The server defines what the name means, in
``notebook_credentials``: a map of fields — fsspec ``storage_options`` for a
mount, driver ``auth`` for a connection — whose values are ``${VAR}``
indirections::

    STRATA_NOTEBOOK_CREDENTIALS='{"lab-bucket": {"key": "${LAB_AWS_KEY}",
                                                 "secret": "${LAB_AWS_SECRET}"}}'

A reference resolves against the notebook's environment first, which is where a
configured secret manager puts what it fetches, and then the server's. So the
secret lives in the vault or the server environment, and a rotation changes the
value behind the name without changing anything a notebook or its cache holds.

The name is part of a cell's identity: which credential a mount or connection
reads through can change what it sees, so it is folded into provenance. The
values never are, which is why rotating a secret invalidates nothing.

``notebook_mount_credentials`` names a server-wide default per URI scheme
(``{"s3": "org-bucket"}``), so a mount with no ``credential`` still reaches the
organization's primary store without any notebook change.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any


class CredentialError(ValueError):
    """A credential name, or a reference inside one, cannot be resolved."""


def resolve_reference(value: str, env: Mapping[str, str]) -> str:
    """``${VAR}`` → its value from *env*, then the process environment.

    Anything else is a literal and passes through: the registry is server
    configuration, not a committed file, so a literal there is the operator's
    decision.
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
    ):
        self._registry = dict(registry or {})
        self._scheme_defaults = dict(scheme_defaults or {})
        self._env = dict(env or {})

    @classmethod
    def from_config(cls, config: Any, env: Mapping[str, str] | None = None) -> CredentialResolver:
        return cls(
            getattr(config, "notebook_credentials", None) or {},
            scheme_defaults=getattr(config, "notebook_mount_credentials", None) or {},
            env=env,
        )

    def resolve(self, name: str) -> dict[str, str]:
        """The fields behind *name*, with every reference resolved.

        Raises:
            CredentialError: naming the credential, so a cell that fails says
                which one to define rather than which variable was empty.
        """
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

        The mount's own ``options`` win, as they always have, so a notebook can
        still set a non-secret like ``endpoint_url`` on top of a credential.
        """
        merged: dict[str, Any] = {}
        default = self._scheme_defaults.get(scheme)
        if default:
            merged.update(self.resolve(default))
        if credential:
            merged.update(self.resolve(credential))
        merged.update(options)
        return merged


def credential_identity(credential: str | None) -> str:
    """The provenance component for a credential: its name, never its values."""
    return f"credential={credential}" if credential else ""
