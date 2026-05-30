# Strata R cell harness.
#
# Counterpart to `src/strata/notebook/harness.py`. Reads a manifest
# JSON file path from argv[1], deserializes input variables, sources
# the cell body in a fresh environment with stdout/stderr captured,
# serializes consumed outputs, and writes `harness-result.json` to
# `output_dir`. The result envelope and on-disk file convention
# exactly mirror the Python harness so the parent's `_store_outputs`
# chain reuses unchanged.
#
# Usage: `Rscript harness.R <manifest_path>`
#
# Manifest shape (written by `CellExecutor._write_manifest`):
#   {
#     "source": "...",
#     "inputs": {<var>: {"content_type": "...", "path": "..."}},
#     "output_dir": "/tmp/.../",
#     "mounts": {<name>: {"uri": "...", "mode": "...", "local_path": "..."}},
#     "env": {<KEY>: <value>}
#   }
#
# Result envelope (written to `output_dir/harness-result.json`):
#   {
#     "success": true|false,
#     "variables": {<var>: SerializedPayload},
#     "displays": [],
#     "stdout": "...",
#     "stderr": "...",
#     "mutation_warnings": []
#     // on error:
#     // "error": "...", "traceback": "..."
#   }
#
# Where SerializedPayload mirrors the Python `serialize_value` output:
#   {"content_type": str, "file": str, "bytes": int, "preview": <jsonable>}

suppressPackageStartupMessages({
  library(jsonlite)
})

# Loaded lazily — only when a data.frame is actually serialized so a
# notebook with no tabular outputs doesn't pay the arrow import cost.
ARROW_LOADED <- FALSE
ensure_arrow <- function() {
  if (!ARROW_LOADED) {
    suppressPackageStartupMessages(library(arrow))
    ARROW_LOADED <<- TRUE
  }
}

# ---------------------------------------------------------------------------
# Argv + manifest
# ---------------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  cat("Usage: Rscript harness.R <manifest_path>\n", file = stderr())
  quit(status = 1, save = "no")
}

manifest_path <- args[[1]]
manifest <- jsonlite::read_json(manifest_path, simplifyVector = FALSE)
source_text <- if (is.null(manifest$source)) "" else manifest$source
output_dir <- if (is.null(manifest$output_dir)) "/tmp/strata_output" else manifest$output_dir
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# ---------------------------------------------------------------------------
# Input deserialization
# ---------------------------------------------------------------------------
#
# Mirrors `harness.py:deserialize_inputs`. The Python side resolves the
# file path as `output_dir / spec$file` (the manifest carries a bare
# filename, not an absolute path), and we do the same so a manifest
# written by `_write_manifest` works for either harness unchanged.
#
# Content types we know about:
#
#   arrow/ipc              → tibble / data.frame via `arrow::read_ipc_stream`
#   json/object            → R list via `jsonlite::fromJSON`
#   application/x-r-rds    → arbitrary R value via `readRDS` (R-only, the
#                             fast path for round-tripping objects
#                             between R cells without an Arrow detour)
#   pickle/object          → unreadable from R, raise with a clear message
#                             so the user knows to re-export from Python as
#                             a DataFrame for Arrow handoff
#   *                      → unknown — raise

deserialize_input <- function(name, spec, output_dir) {
  ct <- spec$content_type
  file_name <- spec$file
  if (is.null(file_name) || identical(file_name, "")) {
    stop(sprintf("Input '%s' has no file path in manifest.", name))
  }
  full_path <- file.path(output_dir, file_name)

  if (identical(ct, "arrow/ipc")) {
    ensure_arrow()
    return(arrow::read_ipc_stream(full_path))
  }
  if (identical(ct, "json/object")) {
    return(jsonlite::fromJSON(full_path, simplifyVector = FALSE))
  }
  if (identical(ct, "application/x-r-rds")) {
    return(readRDS(full_path))
  }
  if (identical(ct, "pickle/object")) {
    stop(sprintf(
      "Cannot read variable '%s' (content_type=pickle/object) from R: ",
      name
    ), "Python pickled objects are not readable in R. ",
    "Re-export the upstream variable as a pandas DataFrame / pyarrow Table ",
    "(Arrow IPC) or a JSON-serializable object.")
  }

  stop(sprintf(
    "Cannot read variable '%s': unsupported content_type '%s'.",
    name, ifelse(is.null(ct), "<null>", ct)
  ))
}

cell_env <- new.env(parent = globalenv())

if (!is.null(manifest$inputs)) {
  for (name in names(manifest$inputs)) {
    assign(
      name,
      deserialize_input(name, manifest$inputs[[name]], output_dir),
      envir = cell_env
    )
  }
}

# Mount paths bind as plain character vectors in the cell env, matching
# the Python harness's `inject_mounts` that injects a `pathlib.Path`
# string. R has no Path object, so we use the bare string.
if (!is.null(manifest$mounts)) {
  for (mount_name in names(manifest$mounts)) {
    local_path <- manifest$mounts[[mount_name]]$local_path
    if (!is.null(local_path)) {
      assign(mount_name, local_path, envir = cell_env)
    }
  }
}

# Env overrides via Sys.setenv. Note: this leaks process-wide because
# Rscript doesn't isolate env. That's the same behaviour as the
# Python harness (it leaks too — both run in a one-shot subprocess
# so the leak doesn't outlive the cell run).
if (!is.null(manifest$env)) {
  env_args <- list()
  for (key in names(manifest$env)) {
    val <- manifest$env[[key]]
    if (!is.null(val)) {
      env_args[[key]] <- as.character(val)
    }
  }
  if (length(env_args) > 0L) {
    do.call(Sys.setenv, env_args)
  }
}

# ---------------------------------------------------------------------------
# Cell execution
# ---------------------------------------------------------------------------
#
# Capture stdout/stderr via `sink()` into temp files. Native code that
# writes directly to the C-level stderr bypasses R's sink — accept
# that limitation for Phase 1; the Python harness has the same issue
# (PR-b3 in #26 mitigated it for batching but single-cell still has
# the gap).

stdout_path <- tempfile(fileext = ".out")
stderr_path <- tempfile(fileext = ".err")
stdout_conn <- file(stdout_path, open = "wt")
stderr_conn <- file(stderr_path, open = "wt")

sink(stdout_conn, type = "output")
sink(stderr_conn, type = "message")

# Snapshot pre-execution bindings so we can detect rebinds. Without
# this, ``df <- transform(df, x = x + 1)`` looks like a no-op to the
# new-vars filter (the binding existed before) and the downstream
# cell silently consumes the stale upstream value.
pre_names <- ls(envir = cell_env)
pre_values <- list()
for (name in pre_names) {
  pre_values[[name]] <- get(name, envir = cell_env)
}

# Variables the analyzer marked as in-place mutations
# (``df$new_col <- ...``). These are always serialized as outputs
# even when ``identical()`` would still hold — the analyzer caught
# the mutation that R's value comparison can't.
mutation_set <- character(0)
if (!is.null(manifest$mutation_defines)) {
  mutation_set <- unlist(manifest$mutation_defines, use.names = FALSE)
  if (is.null(mutation_set)) mutation_set <- character(0)
}

exec_error <- NULL

# ---------------------------------------------------------------------------
# Plot capture device
# ---------------------------------------------------------------------------
# Open a PNG device so plots drawn during execution — base graphics (via
# plot.new) and grid / ggplot / lattice (via grid.newpage) — render to
# per-page files. `%03d` makes the device write one file per page.
# Newpage hooks count real pages so a never-drawn device's trailing blank
# page isn't mistaken for a plot. Device-open is best-effort: if no usable
# graphics device is available the cell still runs, just without capture.
plot_capture_enabled <- FALSE
plot_device <- NULL
plot_width <- 800L
plot_height <- 600L
plot_count_env <- new.env()
plot_count_env$n <- 0L
plot_pattern <- file.path(output_dir, "__rplot__%03d.png")
tryCatch(
  {
    grDevices::png(
      filename = plot_pattern,
      width = plot_width, height = plot_height, res = 96L
    )
    plot_device <- grDevices::dev.cur()
    plot_capture_enabled <- TRUE
    setHook("plot.new", function(...) plot_count_env$n <- plot_count_env$n + 1L, "append")
    setHook("grid.newpage", function(...) plot_count_env$n <- plot_count_env$n + 1L, "append")
  },
  error = function(e) {
    plot_capture_enabled <<- FALSE
  }
)

# Plot-like objects auto-print to the device when they're the visible
# value of a top-level expression — so a bare trailing `p` (ggplot /
# lattice / grob) renders, mirroring the notebook REPL. Non-plot visible
# values are NOT printed, keeping stdout identical to a plain
# `eval(parse(...))` for every non-plotting cell. Explicit `print(p)` /
# `plot(p)` already render during eval and return invisibly, so they're
# not double-drawn.
# Only classes whose `print` method DRAWS to the active device — printing a
# bare grob / gtable just dumps its text structure to stdout without
# rendering, so those are deliberately excluded (use `grid.draw()` for them,
# which draws during eval and is captured anyway).
is_plot_like <- function(x) {
  inherits(x, c("ggplot", "ggmatrix", "patchwork", "trellis", "recordedplot"))
}

tryCatch(
  {
    parsed <- parse(text = source_text)
    for (expr in parsed) {
      res <- withVisible(eval(expr, envir = cell_env))
      if (isTRUE(res$visible) && is_plot_like(res$value)) {
        print(res$value)
      }
    }
  },
  error = function(e) {
    exec_error <<- e
  }
)

# Flush + close the capture device and drop our newpage hooks. Closing by
# device number leaves any device the cell opened itself untouched.
if (plot_capture_enabled) {
  setHook("plot.new", NULL, "replace")
  setHook("grid.newpage", NULL, "replace")
  # `invisible()` — `dev.off()` returns a visible value and this `if` is a
  # top-level expression, so without it Rscript auto-prints "null device 1"
  # into the captured stdout.
  invisible(tryCatch(grDevices::dev.off(plot_device), error = function(e) NULL))
}

# Restore default sinks BEFORE reading the captured files; otherwise
# `readLines` ends up writing to the same sink we're trying to drain.
sink(type = "message")
sink(type = "output")
close(stdout_conn)
close(stderr_conn)

stdout_text <- paste(readLines(stdout_path, warn = FALSE), collapse = "\n")
stderr_text <- paste(readLines(stderr_path, warn = FALSE), collapse = "\n")
unlink(stdout_path)
unlink(stderr_path)

# ---------------------------------------------------------------------------
# Output serialization
# ---------------------------------------------------------------------------
#
# Three content-type tiers, all matching the Python harness wire shape:
#
#   data.frame / tibble  → arrow/ipc          (cross-language readable)
#   atomic scalar/vector → json/object        (cross-language readable)
#   anything else        → application/x-r-rds  (R-only, Python consumers
#                                                error in #58)
#
# Each SerializedPayload has {content_type, file, bytes, preview}.

write_arrow <- function(value, output_dir, var_name) {
  ensure_arrow()
  filename <- paste0(var_name, ".arrow")
  filepath <- file.path(output_dir, filename)
  if (!inherits(value, "data.frame")) {
    value <- as.data.frame(value)
  }
  arrow::write_ipc_stream(value, filepath)
  preview_rows <- min(nrow(value), 5L)
  preview <- if (preview_rows > 0L) {
    utils::capture.output(print(utils::head(value, preview_rows)))
  } else {
    "(empty data.frame)"
  }
  list(
    content_type = "arrow/ipc",
    file = filename,
    bytes = file.info(filepath)$size,
    preview = paste(preview, collapse = "\n"),
    rows = nrow(value),
    columns = ncol(value)
  )
}

write_json <- function(value, output_dir, var_name) {
  filename <- paste0(var_name, ".json")
  filepath <- file.path(output_dir, filename)
  # `auto_unbox = TRUE` makes scalar atomic vectors come out as JSON
  # scalars rather than 1-element arrays, matching the Python side's
  # JSON tier.
  jsonlite::write_json(value, filepath, auto_unbox = TRUE)
  preview_text <- if (length(value) == 1L && is.atomic(value)) {
    format(value)
  } else if (is.list(value)) {
    sprintf("list of length %d", length(value))
  } else {
    sprintf("%s of length %d", class(value)[[1]], length(value))
  }
  list(
    content_type = "json/object",
    file = filename,
    bytes = file.info(filepath)$size,
    preview = preview_text
  )
}

write_rds <- function(value, output_dir, var_name) {
  filename <- paste0(var_name, ".rds")
  filepath <- file.path(output_dir, filename)
  saveRDS(value, filepath)
  list(
    content_type = "application/x-r-rds",
    file = filename,
    bytes = file.info(filepath)$size,
    preview = sprintf("R-only %s (use #58 once it ships for Python consumption)",
                      paste(class(value), collapse = "/")),
    r_only = TRUE
  )
}

serialize_value <- function(value, output_dir, var_name) {
  if (inherits(value, "data.frame")) {
    return(write_arrow(value, output_dir, var_name))
  }
  if (is.atomic(value) && !is.matrix(value) && !is.array(value)) {
    return(write_json(value, output_dir, var_name))
  }
  if (is.list(value) && !is.object(value)) {
    return(tryCatch(
      write_json(value, output_dir, var_name),
      error = function(e) write_rds(value, output_dir, var_name)
    ))
  }
  write_rds(value, output_dir, var_name)
}

# ---------------------------------------------------------------------------
# Display capture (plots)
# ---------------------------------------------------------------------------
# Re-home each non-blank captured page to the `__display__N.png`
# convention the parent's `_store_display_outputs` expects, and emit an
# image/png display payload in the same wire shape as the Python harness:
# base64 data URL in `inline_data_url`, `width` / `height`, `preview`
# NULL. Only runs when a real page was drawn (newpage count > 0), so an
# unused device's trailing blank PNG is dropped.
plot_displays <- list()
if (plot_capture_enabled && plot_count_env$n > 0L) {
  plot_files <- sort(list.files(
    output_dir,
    pattern = "^__rplot__[0-9]+\\.png$"
  ))
  display_index <- 0L
  for (pf in plot_files) {
    src_path <- file.path(output_dir, pf)
    size <- file.info(src_path)$size
    if (is.na(size) || size == 0) next
    target_name <- sprintf("__display__%d.png", display_index)
    target_path <- file.path(output_dir, target_name)
    file.rename(src_path, target_path)
    raw_bytes <- readBin(target_path, what = "raw", n = file.info(target_path)$size)
    plot_displays[[length(plot_displays) + 1L]] <- list(
      content_type = "image/png",
      file = target_name,
      bytes = file.info(target_path)$size,
      inline_data_url = paste0(
        "data:image/png;base64,", jsonlite::base64_enc(raw_bytes)
      ),
      width = plot_width,
      height = plot_height,
      preview = NULL
    )
    display_index <- display_index + 1L
  }
}

# Drop any stragglers — e.g. the trailing blank page an unused device writes
# on close — so they don't clutter output_dir.
invisible(unlink(list.files(
  output_dir,
  pattern = "^__rplot__[0-9]+\\.png$",
  full.names = TRUE
)))

result <- list(
  success = is.null(exec_error),
  variables = list(),
  displays = plot_displays,
  stdout = stdout_text,
  stderr = stderr_text,
  mutation_warnings = list()
)

if (is.null(exec_error)) {
  # Pick variables to serialize. Mirrors ``harness.py``'s three-way
  # rule so downstream cells see rebinds + in-place mutations, not
  # just brand-new names:
  #
  #   1. Name didn't exist before the cell ran (new binding).
  #   2. Name existed before, but the post-execution value is not
  #      ``identical()`` to the pre-execution value (rebind /
  #      replacement — covers ``df <- transform(df, ...)`` where R
  #      hands back a new data.frame, which our previous
  #      ``setdiff(post, pre)`` silently dropped).
  #   3. Name was flagged by the analyzer as an in-place mutation
  #      (``df$col <- ...``) — ``identical()`` still holds because
  #      R copy-on-modify happens inside the same binding, but the
  #      DAG marked it as a write.
  post_names <- ls(envir = cell_env)
  emit_names <- character(0)
  for (name in post_names) {
    if (!(name %in% pre_names)) {
      emit_names <- c(emit_names, name)
    } else if (name %in% mutation_set) {
      emit_names <- c(emit_names, name)
    } else {
      current <- get(name, envir = cell_env)
      prior <- pre_values[[name]]
      if (!identical(current, prior)) {
        emit_names <- c(emit_names, name)
      }
    }
  }

  for (var_name in emit_names) {
    value <- get(var_name, envir = cell_env)
    # Skip functions — the Python side relies on `cloudpickle` for
    # cell-defined classes/functions and there's no clean R equivalent
    # for cross-language consumption. For Phase 1, functions don't
    # serialize out of an R cell.
    if (is.function(value)) {
      next
    }
    payload <- tryCatch(
      serialize_value(value, output_dir, var_name),
      error = function(e) {
        list(error = conditionMessage(e), type = paste(class(value), collapse = "/"))
      }
    )
    result$variables[[var_name]] <- payload
  }
} else {
  result$error <- conditionMessage(exec_error)
  # R doesn't have a native "format the traceback as a string" the way
  # Python's `traceback.format_exc()` does. Best-effort via `sys.calls`
  # — gives the call stack at the time the error was thrown, not the
  # line-numbered Python-style traceback. Better than nothing.
  result$traceback <- paste(
    vapply(sys.calls(), function(call) paste(deparse(call), collapse = " "), character(1)),
    collapse = "\n"
  )
}

# jsonlite serializes an empty R `list()` as a JSON array `[]`, but the
# parent's `_parse_result` does `variables.items()` and needs a JSON object
# `{}`. A named empty list serializes as `{}`. Without this a cell that
# binds no variables — e.g. a plot-only cell — crashes the parse with
# "'list' object has no attribute 'items'". (`displays` /
# `mutation_warnings` are genuinely arrays, so they stay `[]`.)
if (length(result$variables) == 0L) {
  result$variables <- setNames(list(), character(0))
}

result_path <- file.path(output_dir, "harness-result.json")
jsonlite::write_json(
  result,
  result_path,
  auto_unbox = TRUE,
  pretty = TRUE,
  null = "null",
  force = TRUE
)

if (!is.null(exec_error)) {
  # Match the Python harness: exit non-zero on error so the parent
  # sees the failure shape even if it doesn't parse result.json.
  quit(status = 1, save = "no")
}
