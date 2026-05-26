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
#
# Walks the parse tree manually rather than relying on
# `codetools::findGlobals`. codetools treats anything assigned in the
# cell body as "local" and excludes it from the free-variable set —
# fine in isolation, but it drops legitimate read-before-write
# dependencies like ``y <- y + 1`` or ``df <- df[complete.cases(df), ]``
# where the RHS reads a variable that the cell also defines.
#
# Definition of a cross-cell reference: a name is a reference iff it
# is READ before being LOCALLY DEFINED (in source order). This matches
# Python's semantics — a variable that's only used after being defined
# in the same cell isn't a cross-cell input, but a variable that's
# read before being assigned is.

# `jsonlite` is the de-facto standard JSON package. We need it on the
# user's R install — `renv::init` from #55 will add it; for now we
# fall back to manually emitting JSON if `jsonlite` is missing so the
# analyzer keeps working before renv lands.
has_jsonlite <- requireNamespace("jsonlite", quietly = TRUE)

source <- paste(readLines("stdin", warn = FALSE), collapse = "\n")

# ---------------------------------------------------------------------------
# AST walker
# ---------------------------------------------------------------------------
#
# Skipped during read-collection (none of these is a data dependency):
#
# - The op of a call (the function being called — see acceptance examples:
#   ``y <- x + 1`` references only ``x``, not ``+``; ``df <- read_parquet(...)``
#   should not reference ``read_parquet``).
# - Args of ``library`` / ``require`` / ``requireNamespace`` /
#   ``attachNamespace`` / ``loadNamespace`` — those are package names,
#   not free variables.
# - Both sides of ``::`` and ``:::`` — namespace access is a literal
#   package-then-symbol pair, not a cross-cell data reference.
# - The RHS field of ``$`` and ``@`` — member access; the field name
#   doesn't lookup in the global env. The LHS does, so it recurses.
# - Args of ``quote`` / ``bquote`` / ``substitute`` / ``as.name`` /
#   ``as.symbol`` — non-standard evaluation; the args are symbolic
#   not evaluated.
# - Function bodies — locals/formals shouldn't leak. The ``function``
#   definition itself surfaces as a define via its parent ``<-`` /
#   ``->`` / ``=``; we just skip the body. This loses references that
#   only appear inside function bodies, which is an acceptable Phase 1
#   limitation.

NSE_NAMES <- c("quote", "bquote", "substitute", "as.name", "as.symbol")
PKG_LOAD_NAMES <- c(
  "library", "require", "requireNamespace",
  "attachNamespace", "loadNamespace"
)
ASSIGN_LEFT <- c("<-", "=", "<<-")
ASSIGN_RIGHT <- c("->", "->>")

# Collect every name that's read inside ``expr``. Pure — returns a
# character vector. The caller decides which of those names count as
# cross-cell references via the "read-before-locally-defined" rule.
collect_reads <- function(expr) {
  if (is.name(expr)) {
    return(as.character(expr))
  }
  if (!is.call(expr) || length(expr) < 1) {
    return(character(0))
  }
  op <- expr[[1]]
  if (is.name(op)) {
    op_name <- as.character(op)
    if (op_name %in% PKG_LOAD_NAMES) return(character(0))
    if (op_name %in% c("::", ":::")) return(character(0))
    if (op_name %in% NSE_NAMES) return(character(0))
    if (op_name == "function") return(character(0))
    if (op_name %in% c("$", "@")) {
      if (length(expr) >= 2) return(collect_reads(expr[[2]]))
      return(character(0))
    }
    if (op_name %in% ASSIGN_LEFT) {
      # Inner ``<-``: collect reads from the value side. The LHS
      # introduces a name binding that we don't track at inner depth.
      if (length(expr) >= 3) return(collect_reads(expr[[3]]))
      return(character(0))
    }
    if (op_name %in% ASSIGN_RIGHT) {
      if (length(expr) >= 2) return(collect_reads(expr[[2]]))
      return(character(0))
    }
  }
  # Default: skip op (the function-call name), recurse into args.
  if (length(expr) < 2) return(character(0))
  acc <- character(0)
  for (i in 2:length(expr)) {
    acc <- c(acc, collect_reads(expr[[i]]))
  }
  acc
}

# Collect the bare name(s) defined by an LHS subtree. Top-level LHS
# patterns we support: a bare name. Other patterns (subscript-assign,
# list assignment) are rare enough at the top level that we treat them
# as "no new define" for Phase 1 — the value still gets its reads
# collected via the assignment expression's normal RHS walk.
collect_lhs_defines <- function(expr) {
  if (is.name(expr)) {
    return(as.character(expr))
  }
  character(0)
}

# Process one top-level expression. Returns a list with $defs and
# $reads (the reads collected before the local_defs filter is applied
# by the outer loop).
process_statement <- function(expr) {
  if (is.call(expr) && length(expr) >= 1 && is.name(expr[[1]])) {
    op_name <- as.character(expr[[1]])
    if (op_name %in% ASSIGN_LEFT) {
      defs <- if (length(expr) >= 2) collect_lhs_defines(expr[[2]]) else character(0)
      reads <- if (length(expr) >= 3) collect_reads(expr[[3]]) else character(0)
      return(list(defs = defs, reads = reads))
    }
    if (op_name %in% ASSIGN_RIGHT) {
      reads <- if (length(expr) >= 2) collect_reads(expr[[2]]) else character(0)
      defs <- if (length(expr) >= 3) collect_lhs_defines(expr[[3]]) else character(0)
      return(list(defs = defs, reads = reads))
    }
  }
  # Non-assignment top-level: all reads, no defines.
  list(defs = character(0), reads = collect_reads(expr))
}

result <- tryCatch(
  {
    parsed <- parse(text = source, keep.source = FALSE)
    defines <- character(0)
    references <- character(0)
    local_defs <- character(0)
    for (top_expr in parsed) {
      stmt <- process_statement(top_expr)
      # A read is a cross-cell reference iff the name hasn't been
      # locally defined yet in this cell. ``df <- df[complete.cases(df), ]``
      # at top level: local_defs is empty → ``df`` lands in references.
      # ``y <- 1\nz <- y + 1``: first statement sets local_defs = {y},
      # second statement reads ``y`` → setdiff({y}, {y}) = ∅ → not a
      # reference.
      new_refs <- setdiff(stmt$reads, local_defs)
      references <- c(references, new_refs)
      defines <- c(defines, stmt$defs)
      local_defs <- union(local_defs, stmt$defs)
    }
    list(
      defines = unique(defines),
      references = sort(unique(references))
    )
  },
  error = function(e) {
    list(
      defines = character(0),
      references = character(0),
      parse_error = conditionMessage(e)
    )
  }
)

# ---------------------------------------------------------------------------
# JSON emit
# ---------------------------------------------------------------------------

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
