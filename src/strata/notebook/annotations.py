"""Parse cell-level annotations from leading comment blocks.

Annotations are metadata directives in the first contiguous comment block
of a cell.  They control execution routing, mount overrides, timeouts,
environment variables, and loop unrolling.

Supported annotations::

    # @name <display name>        — Human-readable cell name for DAG display
    # @worker <name>              — Route to a named worker backend
    # @timeout <seconds>          — Override execution timeout (per iteration for loops)
    # @mount <name> <uri> [mode]  — Add/override a filesystem mount
    # @table <name> <uri> [snapshot=<id>]
                                  — Declare an Iceberg table input; the table's
                                    snapshot id joins the cell's provenance so
                                    new data makes the cell stale. <name> is
                                    injected as the URI string and
                                    <name>_snapshot as the resolved snapshot id.
    # @env <KEY>=<value>          — Set an environment variable for this cell
    # @variant <group> <name>     — Mark this cell as a variant in <group>; siblings
                                    in the same group share a defines contract and
                                    only the active variant participates in the DAG.
    # @loop max_iter=<N> carry=<var> [start_from=<cell>@iter=<k>]
                                  — Mark the cell as a loop; run the body up to N times,
                                    threading `carry` between iterations.
    # @loop_until <expression>    — Optional termination predicate evaluated in the
                                    cell namespace after each iteration.

Annotations do **not** affect the cell's ``defines``/``references`` analysis.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, TypedDict

from strata.notebook.models import MountMode, MountSpec, TableSpec


class LoopWirePayload(TypedDict):
    """Wire shape for a parsed ``@loop`` directive."""

    max_iter: int
    carry: str
    until_expr: str | None
    start_from_cell: str | None
    start_from_iter: int | None


class VariantWirePayload(TypedDict):
    """Wire shape for a parsed ``@variant`` directive."""

    group: str
    name: str


class AnnotationsWirePayload(TypedDict):
    """Wire shape for the curated annotation view sent to the frontend.

    Subset of ``CellAnnotations`` keys consumed by the UI; ``mounts`` carries
    ``MountSpec.model_dump()`` dicts (json-ready ``dict[str, Any]`` per entry).
    """

    name: str | None
    worker: str | None
    timeout: float | None
    env: dict[str, str]
    mounts: list[dict[str, Any]]
    tables: list[dict[str, Any]]
    loop: LoopWirePayload | None
    variant: VariantWirePayload | None


@dataclass
class CachePolicy:
    """Resolved ``# @cache`` policy for a SQL cell.

    ``kind`` is one of:
      - ``fingerprint`` — driver-derived freshness token in hash (default)
      - ``forever`` — static salt; never invalidates from DB-side state
      - ``session`` — session-unique salt; invalidates across sessions
      - ``ttl`` — time-bucketed salt; ``ttl_seconds`` is required
      - ``snapshot`` — driver MUST return a real snapshot ID

    The default (no ``# @cache`` annotation) is ``fingerprint``; the
    caller substitutes that when ``CellAnnotations.cache`` is ``None``.
    """

    kind: str
    ttl_seconds: int | None = None


@dataclass
class SqlAnnotation:
    """Resolved ``# @sql connection=<name> [write=true]`` directive.

    ``write=true`` opts the cell into writable execution: the
    adapter opens the connection without the read-only enforcement
    (``mode=ro``, ``PRAGMA query_only=ON``, etc) so DDL / DML can
    run. The default is read-only, matching the design-doc
    security boundary for read cells.
    """

    connection: str | None = None
    write: bool = False


@dataclass
class VariantAnnotation:
    """Parsed ``# @variant <group> <name>`` directive.

    Membership in a variant group is declared by source annotation only.
    All cells sharing ``group`` form one group; ``name`` is the variant's
    identifier within the group and must be unique across siblings. The
    active variant per group is tracked separately in ``notebook.toml``.
    """

    group: str
    name: str


@dataclass
class LoopAnnotation:
    """Parsed ``@loop`` / ``@loop_until`` directives for a loop cell.

    Attributes:
        max_iter: Safety bound on the iteration count.
        carry: Name of the variable threaded between iterations. On iter 0 it is
            read from upstream cells (or ``start_from``); on iter k>0 it is
            rebound from iter k-1's output artifact before the body runs.
        until_expr: Optional Python expression evaluated in the cell namespace
            after each iteration. Truthy result terminates the loop early.
        start_from_cell: Optional cell id whose existing iteration artifact
            seeds iter 0's carry. ``None`` means seed from upstream as usual.
        start_from_iter: Iteration index paired with ``start_from_cell``.
    """

    max_iter: int
    carry: str
    until_expr: str | None = None
    start_from_cell: str | None = None
    start_from_iter: int | None = None


# Pattern for annotation lines: # @<key> <rest>
_ANNOTATION_RE = re.compile(r"^#\s*@(\w+)\s*(.*?)\s*$")


def iter_annotation_block(source: str) -> Iterator[tuple[int, str]]:
    """Yield ``(1-based lineno, raw_line)`` for the leading comment block.

    The leading comment block is the first contiguous run of blank
    lines and ``#``-prefixed lines at the top of a cell. Blank lines
    inside the block are skipped (not yielded). The block ends at the
    first non-blank non-comment line, which is the start of the cell
    body.

    Single source of truth for "what counts as the leading comment
    block" — used by ``parse_annotations`` here, by validation
    diagnostics in ``annotation_validation``, and by prompt/SQL
    analysers. Drift between independent implementations of this scan
    would silently desynchronise parsing from validation.
    """
    for lineno, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("#"):
            break
        yield lineno, line


def parse_annotation_directive(line: str) -> tuple[str, str] | None:
    """Match one ``# @key value`` line and return ``(key.lower(), value)``.

    Returns ``None`` when the line isn't a directive (a plain comment,
    or a line outside the leading block). ``value`` is stripped of
    surrounding whitespace; the caller decides how to interpret it.
    """
    match = _ANNOTATION_RE.match(line.strip())
    if not match:
        return None
    return match.group(1).lower(), match.group(2).strip()


def strip_leading_annotations(source: str) -> str:
    """Return source with the leading comment block removed.

    The cell body is everything from the first non-blank non-comment
    line onwards. Used by SQL and prompt cells, which embed an
    annotation block at the top followed by a content body.
    """
    lines = source.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        return "\n".join(lines[i:])
    return ""


def _leading_block_end(lines: list[str]) -> int:
    """Index of the first cell-body line — end of the leading comment block."""
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return i
    return len(lines)


def _format_directive(key: str, value: str) -> str:
    """Render a ``# @key value`` line (``# @key`` when *value* is empty)."""
    value = value.strip()
    return f"# @{key} {value}" if value else f"# @{key}"


def _line_sep(source: str) -> str:
    """The dominant line ending, so a splice preserves CRLF vs LF."""
    return "\r\n" if "\r\n" in source else "\n"


# Directives that may appear more than once in a cell; the single-line splice in
# ``set_annotation_directive`` would silently collapse them, so it refuses them.
_REPEATABLE_DIRECTIVES = frozenset({"env", "mount", "table"})


def set_annotation_directive(source: str, key: str, value: str) -> str:
    """Return *source* with a single ``# @key value`` directive set.

    Replaces the first existing ``# @key`` directive in the leading comment
    block (dropping any duplicate ``# @key`` lines) and leaves the cell body
    untouched. When the key is absent, the directive is inserted after the last
    existing annotation, or at the very top when the cell has none.

    Intended for scalar directives (``name``, ``worker``, ``timeout``, ``model``,
    …). The repeatable ones (``env``, ``mount``, ``table``) raise ``ValueError``
    — collapsing them to one line would drop data; edit the source directly.
    """
    key = key.lower()
    if key in _REPEATABLE_DIRECTIVES:
        raise ValueError(f"@{key} is repeatable; edit the cell source directly")
    new_line = _format_directive(key, value)
    sep = _line_sep(source)
    lines = source.splitlines()
    block_end = _leading_block_end(lines)

    matches = [
        i for i in range(block_end) if (d := parse_annotation_directive(lines[i])) and d[0] == key
    ]
    if matches:
        lines[matches[0]] = new_line
        for i in reversed(matches[1:]):  # collapse accidental duplicates
            del lines[i]
    else:
        directives = [i for i in range(block_end) if parse_annotation_directive(lines[i])]
        lines.insert(directives[-1] + 1 if directives else 0, new_line)

    result = sep.join(lines)
    return result + sep if source.endswith("\n") else result


def remove_annotation_directive(source: str, key: str) -> str:
    """Return *source* with every ``# @key`` directive removed from the block."""
    key = key.lower()
    sep = _line_sep(source)
    lines = source.splitlines()
    block_end = _leading_block_end(lines)
    kept = [
        line
        for i, line in enumerate(lines)
        if not (
            i < block_end and (d := parse_annotation_directive(line)) is not None and d[0] == key
        )
    ]
    result = sep.join(kept)
    return result + sep if source.endswith("\n") else result


@dataclass
class CellAnnotations:
    """Parsed annotations from a cell's leading comment block."""

    worker: str | None = None
    timeout: float | None = None
    mounts: list[MountSpec] = field(default_factory=list)
    tables: list[TableSpec] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)

    # Prompt cell annotations
    name: str | None = None
    model: str | None = None
    temperature: float | None = None
    output_type: str | None = None
    max_tokens: int | None = None
    system_prompt: str | None = None

    # Loop cell annotations
    loop: LoopAnnotation | None = None

    # Widget live mode. ``# @live`` on a widget cell auto-runs the (cheap)
    # downstream cells when a control changes, instead of leaving them stale
    # for a manual run. ``# @live off`` disables it.
    live: bool = False

    # SQL cell annotations
    sql: SqlAnnotation | None = None
    cache: CachePolicy | None = None

    # Variant grouping
    variant: VariantAnnotation | None = None

    # Sweep fan-out (v2). ``# @per_variant [group]`` marks a downstream cell
    # to run once per variant of an upstream sweep group, with the scalar
    # value bound (rather than consuming the whole ``{variant: value}`` dict).
    # ``per_variant_group`` is the explicitly-named group, or None to infer
    # the single sweep group the cell reads from.
    per_variant: bool = False
    per_variant_group: str | None = None

    # Explicit ordering dependencies. ``# @after <cell-id>`` adds a DAG
    # edge from ``<cell-id>`` to this cell without requiring a shared
    # variable — the ergonomic answer to "this SQL cell reads a SQLite
    # file the setup cell created" or any other side-effecting upstream.
    # Multiple ``@after`` lines may stack; each line adds one edge.
    after: list[str] = field(default_factory=list)

    def to_wire_payload(self) -> AnnotationsWirePayload:
        """Render the curated annotation view for cell serialization.

        Only the UI-visible subset is included; SQL/cache/prompt-cell
        directives and ``@after`` edges live in their own wire paths.
        """
        loop_payload: LoopWirePayload | None = None
        if self.loop is not None:
            loop_payload = {
                "max_iter": self.loop.max_iter,
                "carry": self.loop.carry,
                "until_expr": self.loop.until_expr,
                "start_from_cell": self.loop.start_from_cell,
                "start_from_iter": self.loop.start_from_iter,
            }
        variant_payload: VariantWirePayload | None = None
        if self.variant is not None:
            variant_payload = {
                "group": self.variant.group,
                "name": self.variant.name,
            }
        return {
            "name": self.name,
            "worker": self.worker,
            "timeout": self.timeout,
            "env": self.env,
            "mounts": [mount.model_dump() for mount in self.mounts],
            "tables": [table.model_dump() for table in self.tables],
            "loop": loop_payload,
            "variant": variant_payload,
        }


def parse_annotations(source: str) -> CellAnnotations:
    """Extract annotations from the leading comment block of a cell.

    Only the first contiguous block of ``#``-prefixed lines is scanned.
    Once a non-comment, non-blank line is encountered, parsing stops.

    Returns:
        CellAnnotations with all parsed directives.
    """
    result = CellAnnotations()

    for _lineno, line in iter_annotation_block(source):
        parsed = parse_annotation_directive(line)
        if parsed is None:
            continue
        key, value = parsed

        if key == "worker":
            result.worker = value or None

        elif key == "timeout":
            try:
                result.timeout = float(value)
            except ValueError:
                pass  # Silently ignore malformed timeout

        elif key == "mount":
            mount = _parse_mount_annotation(value)
            if mount is not None:
                result.mounts.append(mount)

        elif key == "table":
            table = _parse_table_annotation(value)
            if table is not None:
                result.tables.append(table)

        elif key == "env":
            eq_idx = value.find("=")
            if eq_idx > 0:
                env_key = value[:eq_idx].strip()
                env_val = value[eq_idx + 1 :].strip()
                result.env[env_key] = env_val

        elif key == "name":
            if value:
                result.name = value

        elif key == "model":
            result.model = value or None

        elif key == "temperature":
            try:
                result.temperature = float(value)
            except ValueError:
                pass

        elif key == "output":
            result.output_type = value or None

        elif key == "max_tokens":
            try:
                result.max_tokens = int(value)
            except ValueError:
                pass

        elif key == "system":
            result.system_prompt = value or None

        elif key == "sql":
            _parse_sql_annotation(result, value)

        elif key == "cache":
            _parse_cache_annotation(result, value)

        elif key == "loop":
            _merge_loop_annotation(result, value)

        elif key == "loop_until":
            if value:
                if result.loop is None:
                    result.loop = LoopAnnotation(max_iter=0, carry="", until_expr=value)
                else:
                    result.loop.until_expr = value

        elif key == "variant":
            variant = _parse_variant_annotation(value)
            if variant is not None:
                result.variant = variant

        elif key == "live":
            # ``# @live`` (on) or ``# @live off`` — auto-run downstream on change.
            result.live = value.strip().lower() not in ("off", "false", "no", "0")

        elif key == "per_variant":
            # ``# @per_variant`` (infer the group) or ``# @per_variant <group>``.
            # First token is the group; extras are ignored (validation flags a
            # malformed group name separately if needed).
            result.per_variant = True
            tokens = value.split()
            result.per_variant_group = tokens[0] if tokens else None

        elif key == "after":
            # ``# @after <cell-id>`` declares an ordering dependency
            # without sharing a variable. Multiple lines stack; one
            # edge per identifier on the line (whitespace-separated).
            for token in value.split():
                token = token.strip().rstrip(",")
                if token and token not in result.after:
                    result.after.append(token)

    return result


_VALID_CACHE_KINDS = frozenset({"fingerprint", "forever", "session", "snapshot", "ttl"})


def _parse_sql_annotation(result: CellAnnotations, value: str) -> None:
    """Parse ``@sql connection=<name> [write=true]`` into ``result.sql``.

    Multiple ``@sql`` lines accumulate into the same ``SqlAnnotation``;
    later lines override earlier ones. Unknown keys are dropped silently
    here — annotation_validation surfaces them as user-visible
    diagnostics.

    Booleans (``write=true|false``) are case-insensitive; anything
    other than the truthy literals ``true``/``yes``/``1`` resolves to
    False so a typo (``write=tru``) doesn't silently flip the cell
    into writable mode.
    """
    if result.sql is None:
        result.sql = SqlAnnotation()
    for token in value.split():
        if "=" not in token:
            continue
        k, _, v = token.partition("=")
        k = k.strip()
        v = v.strip()
        if k == "connection" and v:
            result.sql.connection = v
        elif k == "write":
            result.sql.write = v.lower() in {"true", "yes", "1"}


def _parse_cache_annotation(result: CellAnnotations, value: str) -> None:
    """Parse ``@cache <policy>`` into ``result.cache``.

    Forms:
      - ``@cache fingerprint`` / ``forever`` / ``session`` / ``snapshot``
      - ``@cache ttl=<seconds>``

    Malformed values yield ``None`` so annotation_validation can surface a
    diagnostic instead of silently applying the wrong policy.
    """
    tokens = value.split()
    if not tokens:
        return
    head = tokens[0]
    if head in _VALID_CACHE_KINDS and head != "ttl":
        result.cache = CachePolicy(kind=head)
        return
    if head.startswith("ttl="):
        try:
            seconds = int(head.removeprefix("ttl="))
        except ValueError:
            return
        if seconds <= 0:
            return
        result.cache = CachePolicy(kind="ttl", ttl_seconds=seconds)


_LOOP_START_FROM_RE = re.compile(r"^(?P<cell>[^@]+)@iter=(?P<iter>-?\d+)$")


def _merge_loop_annotation(result: CellAnnotations, value: str) -> None:
    """Merge ``@loop key=value key=value ...`` into ``result.loop``.

    Multiple ``@loop`` lines accumulate into the same ``LoopAnnotation``;
    later lines override earlier ones for any key they set.
    """
    if result.loop is None:
        result.loop = LoopAnnotation(max_iter=0, carry="")

    loop = result.loop
    for token in value.split():
        if "=" not in token:
            continue
        k, _, v = token.partition("=")
        k = k.strip()
        v = v.strip()
        if not k or not v:
            continue

        if k == "max_iter":
            try:
                loop.max_iter = int(v)
            except ValueError:
                continue
        elif k == "carry":
            loop.carry = v
        elif k == "until":
            loop.until_expr = v
        elif k == "start_from":
            match = _LOOP_START_FROM_RE.match(v)
            if match is not None:
                loop.start_from_cell = match.group("cell").strip()
                try:
                    loop.start_from_iter = int(match.group("iter"))
                except ValueError:
                    loop.start_from_cell = None
                    loop.start_from_iter = None


def _parse_variant_annotation(value: str) -> VariantAnnotation | None:
    """Parse ``# @variant <group> <name>``.

    Both ``group`` and ``name`` must be valid Python identifiers so they
    survive notebook-toml round-trips and frontend rendering without
    escaping. Malformed values yield ``None``; annotation_validation
    surfaces a diagnostic so the user sees the issue.
    """
    parts = value.split()
    if len(parts) != 2:
        return None
    group, name = parts
    if not group.isidentifier() or not name.isidentifier():
        return None
    return VariantAnnotation(group=group, name=name)


def _parse_table_annotation(value: str) -> TableSpec | None:
    """Parse a ``@table`` annotation value.

    Format: ``<name> <uri> [snapshot=<id>]``

    Examples::

        @table trips file:///data/warehouse#nyc.trips
        @table events s3://bucket/wh#db.events snapshot=1292033279574548405
    """
    parts = value.split()
    if len(parts) < 2:
        return None

    name = parts[0]
    uri = parts[1]
    snapshot_pin: int | None = None

    for extra in parts[2:]:
        if extra.startswith("snapshot="):
            try:
                snapshot_pin = int(extra[len("snapshot=") :])
            except ValueError:
                return None

    if not name.isidentifier():
        return None

    return TableSpec(name=name, uri=uri, snapshot_pin=snapshot_pin)


def _parse_mount_annotation(value: str) -> MountSpec | None:
    """Parse a ``@mount`` annotation value.

    Format: ``<name> <uri> [ro|rw]``

    Examples::

        @mount raw_data s3://bucket/prefix ro
        @mount scratch file:///tmp/work rw
        @mount data s3://bucket/data          # defaults to ro
    """
    parts = value.split()
    if len(parts) < 2:
        return None

    name = parts[0]
    uri = parts[1]
    mode = MountMode.READ_ONLY

    if len(parts) >= 3 and parts[2] in ("ro", "rw"):
        mode = MountMode(parts[2])

    # Validate name is a valid Python identifier
    if not name.isidentifier():
        return None

    return MountSpec(name=name, uri=uri, mode=mode)
