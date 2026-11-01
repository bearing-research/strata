"""``parse_annotations`` before cellgraph: the oracle the vocabulary's values are held to.

Frozen: the vocabulary now reads annotations, and this keeps what it must agree with.
"""

from __future__ import annotations

from strata.notebook.annotations import (
    CellAnnotations,
    LoopAnnotation,
    _merge_loop_annotation,
    _parse_cache_annotation,
    _parse_dataset_annotation,
    _parse_fetch_annotation,
    _parse_mount_annotation,
    _parse_sql_annotation,
    _parse_table_annotation,
    _parse_variant_annotation,
    _parse_write_table_annotation,
    iter_annotation_block,
    parse_annotation_directive,
)
from strata.notebook.models import CellLanguage


def legacy_parse_annotations(
    source: str, language: CellLanguage | str | None = None
) -> CellAnnotations:
    """``parse_annotations`` as it read annotations before the vocabulary did."""
    result = CellAnnotations()
    if language == CellLanguage.MARKDOWN:
        return result

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

        elif key == "write_table":
            write = _parse_write_table_annotation(value)
            if write is not None:
                result.table_writes.append(write)

        elif key == "output_dir":
            if value.isidentifier() and value not in result.output_dirs:
                result.output_dirs.append(value)

        elif key == "fetch":
            fetch = _parse_fetch_annotation(value)
            if fetch is not None:
                result.fetches.append(fetch)

        elif key == "dataset":
            dataset = _parse_dataset_annotation(value)
            if dataset is not None:
                result.datasets.append(dataset)

        elif key == "env":
            eq_idx = value.find("=")
            if eq_idx > 0:
                env_key = value[:eq_idx].strip()
                env_val = value[eq_idx + 1 :].strip()
                result.env[env_key] = env_val

        elif key == "nocache":
            # ``# @nocache`` -> always re-run; ``# @nocache off`` re-enables caching.
            result.nocache = value.strip().lower() not in ("off", "false", "no", "0")

        elif key == "stochastic":
            result.stochastic = value.strip().lower() not in ("off", "false", "no", "0")
            if result.stochastic:
                result.nocache = True

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
            # ``# @live`` (on) or ``# @live off``.
            result.live = value.strip().lower() not in ("off", "false", "no", "0")

        elif key == "per_variant":
            # ``# @per_variant`` (infer the group) or ``# @per_variant <group>``. First token is
            # the group; extras are ignored (validation flags a malformed name).
            result.per_variant = True
            tokens = value.split()
            result.per_variant_group = tokens[0] if tokens else None

        elif key == "after":
            # One edge per whitespace-separated identifier; multiple lines stack.
            for token in value.split():
                token = token.strip().rstrip(",")
                if token and token not in result.after:
                    result.after.append(token)

    return result
