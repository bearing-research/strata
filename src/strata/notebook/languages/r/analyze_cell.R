# Strata R-cell analyzer.
#
# Reads R source on stdin, writes a JSON object with `defines`,
# `references`, and (on syntax error) `parse_error` to stdout.
# Counterpart of the Python `strata.notebook.analyzer` module — the
# defines/references convention exactly mirrors the Python side.
#
# Loaded by `RLanguageAnalyzer` in this same package via
# `Rscript path/to/analyze_cell.R`. Stdin is the source; stdout is
# JSON; stderr is captured but not parsed.

# `codetools` is in `base` distribution, no install needed.
suppressPackageStartupMessages(library(codetools))

# `jsonlite` is the de-facto standard JSON package. We need it on the
# user's R install — `renv::init` from #55 will add it; for now we
# fall back to manually emitting JSON if `jsonlite` is missing so the
# analyzer keeps working before renv lands.
has_jsonlite <- requireNamespace("jsonlite", quietly = TRUE)

source <- paste(readLines("stdin", warn = FALSE), collapse = "\n")

# Pull defines + references with a single parse pass.
#
# `defines`: top-level assignments (`<-`, `=`, `<<-`, `->`, `->>`).
# `getParseData` returns a row per token with parent/child links;
# top-level assignments live at the lowest parse-depth.
#
# `references`: free variables anywhere in the body — `codetools::findGlobals`
# walks the AST and reports identifiers that look up via the global
# environment. Excludes formals, locals, package names from `library()`
# / `require()` calls.

result <- tryCatch(
  {
    parsed <- parse(text = source, keep.source = TRUE)
    if (length(parsed) == 0) {
      list(defines = character(0), references = character(0))
    } else {
      # Top-level assigns: scan each parsed expression's outer form.
      defines <- character(0)
      for (expr in parsed) {
        if (is.call(expr) && length(expr) >= 3) {
          op <- as.character(expr[[1]])
          if (op %in% c("<-", "=", "<<-")) {
            target <- expr[[2]]
            if (is.name(target)) {
              defines <- c(defines, as.character(target))
            }
          } else if (op %in% c("->", "->>")) {
            target <- expr[[3]]
            if (is.name(target)) {
              defines <- c(defines, as.character(target))
            }
          }
        }
      }

      # findGlobals against a wrapping function body so it traverses
      # everything. Wrap the parsed exprs in `function() { ... }` so
      # codetools treats them as a single body.
      wrapped <- as.call(
        c(list(quote(`function`), pairlist()), list(as.call(c(list(quote(`{`)), parsed))))
      )
      fn <- eval(wrapped, envir = baseenv())
      refs <- codetools::findGlobals(fn, merge = FALSE)
      # `findGlobals` returns a list with `variables`, `functions`, and
      # `parameters` slots in some versions; we want only the variable
      # + function names that are NOT among our `defines`.
      all_refs <- unique(c(refs$variables, refs$functions))
      references <- setdiff(all_refs, defines)

      list(
        defines = unique(defines),
        references = sort(references)
      )
    }
  },
  error = function(e) {
    list(
      defines = character(0),
      references = character(0),
      parse_error = conditionMessage(e)
    )
  }
)

if (has_jsonlite) {
  cat(jsonlite::toJSON(result, auto_unbox = TRUE))
} else {
  # Manual JSON emit — keep this in lockstep with what the Python
  # side expects to deserialize. Strings need basic escaping; we
  # don't expect newlines / quotes inside identifier names so a
  # naive escape covers our needs.
  emit_string <- function(s) {
    s <- gsub("\\\\", "\\\\\\\\", s)
    s <- gsub("\"", "\\\\\"", s)
    paste0("\"", s, "\"")
  }
  emit_array <- function(xs) {
    paste0("[", paste(vapply(xs, emit_string, character(1)), collapse = ","), "]")
  }
  parts <- c(
    paste0("\"defines\":", emit_array(result$defines)),
    paste0("\"references\":", emit_array(result$references))
  )
  if (!is.null(result$parse_error)) {
    parts <- c(parts, paste0("\"parse_error\":", emit_string(result$parse_error)))
  }
  cat("{", paste(parts, collapse = ","), "}", sep = "")
}
