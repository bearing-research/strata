"""Parse cell-level annotations (``# @key value``) from a cell's leading comment block.

Annotations do not affect the cell's ``defines``/``references`` analysis.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, TypedDict

from strata.notebook.models import (
    CellLanguage,
    DatasetSpec,
    FetchSpec,
    MountMode,
    MountSpec,
    TableSpec,
    TableWriteSpec,
)


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

    ``mounts`` carries ``MountSpec.model_dump()`` dicts.
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

    ``kind`` is one of: ``fingerprint`` (driver freshness token, the default),
    ``forever`` (static salt), ``session`` (session-unique salt), ``ttl``
    (time-bucketed salt; ``ttl_seconds`` required) or ``snapshot`` (driver must
    return a real snapshot ID). ``CellAnnotations.cache is None`` means fingerprint.
    """

    kind: str
    ttl_seconds: int | None = None


@dataclass
class SqlAnnotation:
    """Resolved ``# @sql connection=<name> [write=true]`` directive.

    ``write=true`` opens the connection without read-only enforcement so DDL/DML
    can run; the default is read-only.
    """

    connection: str | None = None
    write: bool = False


@dataclass
class VariantAnnotation:
    """Parsed ``# @variant <group> <name>`` directive.

    ``name`` must be unique within ``group``. The active variant per group is
    tracked separately in ``notebook.toml``.
    """

    group: str
    name: str


@dataclass
class LoopAnnotation:
    """Parsed ``@loop`` / ``@loop_until`` directives for a loop cell.

    ``carry`` is read from upstream (or ``start_from``) on iter 0 and rebound from
    iter k-1's artifact on iter k. A truthy ``until_expr``, evaluated in the cell
    namespace after each iteration, ends the loop early. ``start_from_cell`` is
    ``None`` to seed iter 0 from upstream as usual.
    """

    max_iter: int
    carry: str
    until_expr: str | None = None
    start_from_cell: str | None = None
    start_from_iter: int | None = None


_ANNOTATION_RE = re.compile(r"^#\s*@(\w+)\s*(.*?)\s*$")


# Every directive a cell's parsers read (``parse_annotations``, ``analyze_prompt_cell``, and
# the readers with their own regex, such as ``@app hide`` in the app view and export):
# validation warns on any other ``# @key``, which nothing would act on.
ANNOTATION_KEYS = frozenset(
    {
        "after",
        "app",
        "cache",
        "dataset",
        "env",
        "fetch",
        "live",
        "loop",
        "loop_until",
        "max_tokens",
        "model",
        "mount",
        "name",
        "nocache",
        "output",
        "output_dir",
        "output_schema",
        "per_variant",
        "sql",
        "stochastic",
        "system",
        "table",
        "temperature",
        "timeout",
        "validate_retries",
        "variant",
        "worker",
        "write_table",
    }
)


def iter_annotation_block(source: str) -> Iterator[tuple[int, str]]:
    """Yield ``(1-based lineno, raw_line)`` for the leading comment block.

    The block is the first contiguous run of blank and ``#`` lines; blank lines are
    not yielded. This is the single definition of the block shared by parsing,
    validation and the prompt/SQL analysers, so they cannot drift apart.
    """
    for lineno, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("#"):
            break
        yield lineno, line


def parse_annotation_directive(line: str) -> tuple[str, str] | None:
    """Match one ``# @key value`` line and return ``(key.lower(), value)``, or ``None``."""
    match = _ANNOTATION_RE.match(line.strip())
    if not match:
        return None
    return match.group(1).lower(), match.group(2).strip()


def strip_leading_annotations(source: str) -> str:
    """Return source with the leading comment block removed."""
    lines = source.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        return "\n".join(lines[i:])
    return ""


def _leading_block_end(lines: list[str]) -> int:
    """Index of the first cell-body line: the end of the leading comment block."""
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
_REPEATABLE_DIRECTIVES = frozenset(
    {"env", "mount", "table", "write_table", "output_dir", "fetch", "dataset"}
)


def set_annotation_directive(source: str, key: str, value: str) -> str:
    """Return *source* with a single ``# @key value`` directive set.

    Replaces the first ``# @key`` in the leading block and drops duplicates; when
    absent, inserts after the last annotation (or at the top). Raises ``ValueError``
    for repeatable directives, since collapsing them to one line would drop data.
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


def pin_fetch_directives(source: str, digests: dict[str, str]) -> str:
    """Return *source* with ``sha256=`` set on each ``# @fetch <name>`` named in *digests*.

    ``digests`` maps a fetch name to the digest to pin. An existing pin is replaced
    and every other option on the line is kept.
    """
    sep = _line_sep(source)
    lines = source.splitlines()
    for i in range(_leading_block_end(lines)):
        directive = parse_annotation_directive(lines[i])
        if directive is None or directive[0] != "fetch":
            continue
        spec = _parse_fetch_annotation(directive[1])
        if spec is None or spec.name not in digests:
            continue
        options = [part for part in directive[1].split()[2:] if not part.startswith("sha256=")]
        value = " ".join([spec.name, spec.url, f"sha256={digests[spec.name]}", *options])
        lines[i] = _format_directive("fetch", value)
    result = sep.join(lines)
    return result + sep if source.endswith("\n") else result


@dataclass
class CellAnnotations:
    """Parsed annotations from a cell's leading comment block."""

    worker: str | None = None
    timeout: float | None = None
    mounts: list[MountSpec] = field(default_factory=list)
    tables: list[TableSpec] = field(default_factory=list)
    table_writes: list[TableWriteSpec] = field(default_factory=list)
    # ``# @output_dir <name>``: an empty folder bound to ``<name>``; what is in it when the
    # run ends is the output, read downstream as a local path.
    output_dirs: list[str] = field(default_factory=list)
    fetches: list[FetchSpec] = field(default_factory=list)
    datasets: list[DatasetSpec] = field(default_factory=list)
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

    # ``# @live`` on a widget cell auto-runs the (cheap) downstream cells when a control
    # changes instead of leaving them stale. ``# @live off`` disables it.
    live: bool = False

    # SQL cell annotations
    sql: SqlAnnotation | None = None
    cache: CachePolicy | None = None

    # ``# @stochastic``: every run's outputs persist as their own versions (keyed by the
    # run's build), never served from cache; implies ``nocache``.
    stochastic: bool = False
    # ``# @nocache``: always re-execute, never serve a cache hit. For side effects or fresh
    # randomness, where replaying a cached result would skip the effect. Any language.
    nocache: bool = False

    # Variant grouping
    variant: VariantAnnotation | None = None

    # ``# @per_variant [group]``: run once per variant of an upstream sweep group with the
    # scalar value bound. ``per_variant_group`` is the named group, or None to infer the
    # single sweep group the cell reads.
    per_variant: bool = False
    per_variant_group: str | None = None

    # ``# @after <cell-id>`` adds a DAG edge without a shared variable (e.g. a SQL cell
    # reading a file a setup cell wrote). Lines stack; each adds edges.
    after: list[str] = field(default_factory=list)

    def to_wire_payload(self) -> AnnotationsWirePayload:
        """Render the UI-visible subset of the annotations.

        SQL/cache/prompt-cell directives and ``@after`` edges travel on their own wire paths.
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


# A markdown heading is a ``#`` line, so markdown reads only ``@app`` (the app view
# and export match it themselves); every other ``# @`` line there is prose.
MARKDOWN_DIRECTIVE_KEYS = frozenset({"app"})


def parse_annotations(source: str, language: CellLanguage | str | None = None) -> CellAnnotations:
    """Extract annotations from the leading comment block of a cell.

    A markdown cell has none: ``@app``, the one key it reads, is not a field here.
    """
    if language == CellLanguage.MARKDOWN:
        return CellAnnotations()
    from strata.notebook.vocabulary import annotations_from, parse_values

    values, directives = parse_values(source, str(language or CellLanguage.PYTHON))
    return annotations_from(values, directives) if directives else CellAnnotations()


_VALID_CACHE_KINDS = frozenset({"fingerprint", "forever", "session", "snapshot", "ttl"})


def _parse_sql_annotation(result: CellAnnotations, value: str) -> None:
    """Parse ``@sql connection=<name> [write=true]`` into ``result.sql``.

    Repeated lines accumulate, later keys winning. Unknown keys are dropped here
    (annotation_validation reports them). Only ``true``/``yes``/``1`` enable write,
    so a typo never makes a cell writable.
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
    """Parse ``@cache <policy>`` or ``@cache ttl=<seconds>`` into ``result.cache``.

    Malformed values yield ``None`` so annotation_validation can report them
    instead of applying the wrong policy.
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
    """Merge ``@loop key=value ...`` into ``result.loop``; later lines override earlier keys."""
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
    """Parse ``# @variant <group> <name>``, or return ``None`` when malformed.

    Both parts must be Python identifiers so they round-trip through
    ``notebook.toml`` and the frontend without escaping.
    """
    parts = value.split()
    if len(parts) != 2:
        return None
    group, name = parts
    if not group.isidentifier() or not name.isidentifier():
        return None
    return VariantAnnotation(group=group, name=name)


def _parse_fetch_annotation(value: str) -> FetchSpec | None:
    """Parse ``<name> <url> [sha256=<digest>] [refetch=never|stale|always]``."""
    parts = value.split()
    if len(parts) < 2 or not parts[0].isidentifier():
        return None
    sha256: str | None = None
    refetch = "stale"
    for extra in parts[2:]:
        key, sep, val = extra.partition("=")
        if key == "sha256" and sep:
            sha256 = val.lower()
        elif key == "refetch" and val in ("never", "stale", "always"):
            refetch = val
        else:
            return None
    try:
        return FetchSpec.model_validate(
            {"name": parts[0], "url": parts[1], "sha256": sha256, "refetch": refetch}
        )
    except ValueError:
        return None


def _parse_dataset_annotation(value: str) -> DatasetSpec | None:
    """Parse ``<var> <name>[@<alias>|@v=<n>]``."""
    parts = value.split()
    if len(parts) != 2 or not parts[0].isidentifier():
        return None
    dataset, sep, selector = parts[1].rpartition("@")
    if not sep:
        dataset, selector = parts[1], ""
    alias: str | None = None
    version: str | None = None
    if selector.startswith("v="):
        version = selector.removeprefix("v=")
    elif selector:
        alias = selector
    elif sep:
        return None
    try:
        return DatasetSpec.model_validate(
            {"name": parts[0], "dataset": dataset, "alias": alias, "version": version}
        )
    except ValueError:
        return None


def _parse_table_annotation(value: str) -> TableSpec | None:
    """Parse ``<name> <uri> [snapshot=<id>]``."""
    parts = value.split()
    if len(parts) < 2:
        return None

    name = parts[0]
    uri = parts[1]
    snapshot_pin: int | None = None

    for extra in parts[2:]:
        if not extra.startswith("snapshot="):
            return None
        try:
            snapshot_pin = int(extra[len("snapshot=") :])
        except ValueError:
            return None

    if not name.isidentifier():
        return None

    return TableSpec(name=name, uri=uri, snapshot_pin=snapshot_pin)


def _parse_write_table_annotation(value: str) -> TableWriteSpec | None:
    """Parse ``<variable> <table> append|overwrite|merge [on=<col>,...]``.

    The mode is required, since append and overwrite differ in what they destroy; merge
    needs ``on``.
    """
    parts = value.split()
    if len(parts) < 3 or parts[2] not in ("append", "overwrite", "merge"):
        return None
    on: list[str] = []
    for extra in parts[3:]:
        if not extra.startswith("on="):
            return None
        on = [col for col in extra[len("on=") :].split(",") if col]
    if (parts[2] == "merge") != bool(on):
        return None
    try:
        return TableWriteSpec(variable=parts[0], table=parts[1], mode=parts[2], on=on)
    except ValueError:
        return None


def _parse_mount_annotation(value: str) -> MountSpec | None:
    """Parse ``<name> <uri> [ro|rw] [credential=<name>]``; mode defaults to ``ro``."""
    parts = value.split()
    if len(parts) < 2:
        return None

    name = parts[0]
    uri = parts[1]
    mode = MountMode.READ_ONLY
    credential: str | None = None

    for extra in parts[2:]:
        if extra in ("ro", "rw"):
            mode = MountMode(extra)
        elif extra.startswith("credential="):
            credential = extra[len("credential=") :] or None
        else:
            return None

    if not name.isidentifier():
        return None

    return MountSpec(name=name, uri=uri, mode=mode, credential=credential)
