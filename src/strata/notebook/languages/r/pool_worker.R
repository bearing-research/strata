# Warm R pool worker — pre-pays Rscript startup so cells don't.
#
# Lifecycle (single-shot, mirroring pool_worker.py):
#   1. R interpreter startup + .Rprofile (renv activation via cwd) has
#      already happened by the time this script runs — that's the ~1-2s
#      being amortized.
#   2. Warm the libraries the harness needs (jsonlite, arrow).
#   3. Print "ready" — the parent pool counts this worker as available.
#   4. Block on stdin for one manifest path.
#   5. Execute it by sourcing harness.R with the manifest pre-set in the
#      global env, stdout sunk to stderr so nothing pollutes the frame
#      protocol.
#   6. Relay harness-result.json as a single stdout line and exit.
#
# The stdin/stdout frame protocol is identical to the Python pool worker:
# manifest path in, one JSON result line out.

suppressPackageStartupMessages({
  library(jsonlite)
  if (requireNamespace("arrow", quietly = TRUE)) {
    library(arrow)
  }
})

harness_path <- file.path(dirname(sub(
  "--file=", "",
  grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)[[1]]
)), "harness.R")

emit_error_line <- function(message) {
  cat(jsonlite::toJSON(
    list(
      success = FALSE,
      variables = setNames(list(), character(0)),
      stdout = "",
      stderr = "",
      error = paste0("R pool worker error: ", message),
      mutation_warnings = list()
    ),
    auto_unbox = TRUE, null = "null"
  ), "\n", sep = "")
  flush(stdout())
}

cat("ready\n")
flush(stdout())

line <- readLines(con = "stdin", n = 1)
if (length(line) == 0 || !nzchar(trimws(line[[1]]))) {
  quit(status = 0, save = "no")
}
manifest_path <- trimws(line[[1]])

result <- tryCatch(
  {
    assign(".strata_pool_manifest", manifest_path, envir = globalenv())

    # Anything the harness (or top-level cell escape) writes to stdout
    # must not corrupt the protocol — sink it to stderr. The harness's
    # own capture sink nests on top of this and pops back here.
    sink(stderr(), type = "output")
    source_error <- tryCatch(
      {
        source(harness_path, local = new.env(parent = globalenv()))
        NULL
      },
      error = function(e) conditionMessage(e)
    )
    sink(type = "output")

    if (!is.null(source_error)) {
      emit_error_line(source_error)
      quit(status = 0, save = "no")
    }

    manifest <- jsonlite::read_json(manifest_path, simplifyVector = FALSE)
    result_path <- file.path(manifest$output_dir, "harness-result.json")
    if (!file.exists(result_path)) {
      emit_error_line("harness produced no result file")
      quit(status = 0, save = "no")
    }
    paste(readLines(result_path, warn = FALSE), collapse = " ")
  },
  error = function(e) {
    # Make sure no sink is left dangling before emitting
    while (sink.number() > 0) sink(type = "output")
    emit_error_line(conditionMessage(e))
    quit(status = 0, save = "no")
  }
)

cat(result, "\n", sep = "")
flush(stdout())
