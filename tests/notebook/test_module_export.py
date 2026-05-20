"""Tests for notebook module-export planning.

Module export takes a cell source and produces a "synthetic module" the
producing cell's defs/classes can be re-imported from in downstream
cells. The planner *slices* the source — keeps imports, defs, classes,
and literal-constant assignments; drops everything else — and validates
that the slice is self-contained (free vars in defs/classes resolve to
slice-local bindings or builtins). Cells that mix runtime work and
library code can export the library code cleanly; pure module cells
keep behaving exactly as before.
"""

import sys

from strata.notebook.module_export import build_module_export_plan

# ---------------------------------------------------------------------------
# Pure module cells: no drops, exactly the pre-slicing behavior.
# ---------------------------------------------------------------------------


def test_literal_int_assignment_is_exportable() -> None:
    plan = build_module_export_plan("STEP_SIZE = 5")
    assert plan.is_exportable is True
    assert plan.exported_symbols["STEP_SIZE"].kind == "constant"
    assert plan.sliced is False


def test_literal_float_string_bool_none_are_exportable() -> None:
    plan = build_module_export_plan("RATE = 0.5\nLABEL = 'prod'\nDEBUG = True\nFALLBACK = None")
    assert plan.is_exportable is True
    assert set(plan.exported_symbols) == {"RATE", "LABEL", "DEBUG", "FALLBACK"}
    assert plan.sliced is False


def test_negative_literal_is_exportable() -> None:
    plan = build_module_export_plan("OFFSET = -1\nBITS = ~0")
    assert plan.is_exportable is True
    assert set(plan.exported_symbols) == {"OFFSET", "BITS"}


def test_literal_container_is_exportable() -> None:
    plan = build_module_export_plan(
        """
CLASSES = ["cat", "dog", "fish"]
CONFIG = {"lr": 1e-3, "batch": 32, "shuffle": True}
DIMS = (128, 128)
""".strip()
    )
    assert plan.is_exportable is True
    assert set(plan.exported_symbols) == {"CLASSES", "CONFIG", "DIMS"}


def test_literal_constants_coexist_with_defs_and_classes() -> None:
    plan = build_module_export_plan(
        """
import math

RATE = 0.5
CLASSES = ("a", "b")

def scale(x):
    return x * RATE

class Config:
    debug = True
""".strip()
    )
    assert plan.is_exportable is True
    assert plan.exported_symbols["RATE"].kind == "constant"
    assert plan.exported_symbols["CLASSES"].kind == "constant"
    assert plan.exported_symbols["scale"].kind == "function"
    assert plan.exported_symbols["Config"].kind == "class"
    assert plan.sliced is False


def test_annotated_literal_assignment_is_exportable() -> None:
    plan = build_module_export_plan("STEP_SIZE: float = 0.5")
    assert plan.is_exportable is True
    assert plan.exported_symbols["STEP_SIZE"].kind == "constant"


def test_tuple_unpacking_of_literals_is_exportable() -> None:
    plan = build_module_export_plan("MIN, MAX = 0, 100")
    assert plan.is_exportable is True
    assert set(plan.exported_symbols) == {"MIN", "MAX"}


def test_pure_cell_keeps_original_source_bytes() -> None:
    """Pure cells should round-trip verbatim — no ast.unparse reformatting."""
    src = "import math\n\ndef pi():\n    # the ratio\n    return math.pi\n"
    plan = build_module_export_plan(src)
    assert plan.is_exportable is True
    assert plan.module_source == src
    assert plan.sliced is False


# ---------------------------------------------------------------------------
# Sliced cells: runtime statements coexist with shareable defs/classes.
# These cases all used to be blocked outright; slicing lets the def
# escape cleanly while the runtime statement flows through the regular
# artifact path.
# ---------------------------------------------------------------------------


def test_runtime_statement_alongside_self_contained_def_is_exportable() -> None:
    """The classic case: a setup line followed by a def the user wants
    to share. The def has no free vars and is exportable; the setup
    line is dropped from the slice.
    """
    plan = build_module_export_plan(
        """
df = load_data()

def double(x):
    return x * 2
""".strip()
    )
    assert plan.is_exportable is True
    assert plan.sliced is True
    assert "double" in plan.exported_symbols
    assert plan.exported_symbols["double"].kind == "function"
    # The runtime line is dropped, not exported as a "constant".
    assert "df" not in plan.exported_symbols


def test_runtime_statement_alongside_def_using_literal_const_is_exportable() -> None:
    """A def that closes over a literal-const sibling is fine — the
    constant is in the slice, so the def's free var resolves.
    """
    plan = build_module_export_plan(
        """
df = load_data()
THRESHOLD = 0.8

def is_outlier(x):
    return x > THRESHOLD
""".strip()
    )
    assert plan.is_exportable is True
    assert "is_outlier" in plan.exported_symbols
    assert "THRESHOLD" in plan.exported_symbols
    assert "df" not in plan.exported_symbols


def test_def_with_unresolved_free_var_blocks_export() -> None:
    """When the def's body references a runtime-only name, the
    synthetic module would NameError at call time. Block it explicitly
    and name the unresolved variable so the user knows where to fix.
    """
    plan = build_module_export_plan(
        """
x = compute()

def add(y):
    return x + y
""".strip()
    )
    assert plan.is_exportable is False
    # Specific error pinpoints the failing symbol and the unresolved var.
    assert "function `add`" in plan.format_error()
    assert "x" in plan.format_error()
    # `add` becomes blocking — it would break the moment downstream calls it.
    assert "add" in plan.blocking_symbols
    # Not in exported_symbols since it can't be safely shared.
    assert "add" not in plan.exported_symbols


def test_class_with_unresolved_base_blocks_export() -> None:
    """A base class is evaluated at module load; if it isn't in the
    slice, the synthetic module's import itself raises NameError.
    """
    plan = build_module_export_plan(
        """
Parent = build_parent()

class Child(Parent):
    pass
""".strip()
    )
    assert plan.is_exportable is False
    assert "Parent" in plan.format_error()
    assert "Child" in plan.blocking_symbols


def test_decorator_resolved_in_slice_is_exportable() -> None:
    """A decorator imported in the same cell is part of the slice and
    resolves at module load."""
    plan = build_module_export_plan(
        """
from functools import lru_cache

df = load_data()

@lru_cache(maxsize=8)
def compute(n):
    return n * n
""".strip()
    )
    assert plan.is_exportable is True
    assert "compute" in plan.exported_symbols


def test_decorator_unresolved_blocks_export() -> None:
    plan = build_module_export_plan(
        """
my_decorator = make_decorator()

@my_decorator
def f():
    return 1
""".strip()
    )
    assert plan.is_exportable is False
    # ``my_decorator`` is referenced at module load — surfaces in the
    # top-level reasons.
    assert "my_decorator" in plan.format_error()


def test_def_using_builtin_is_exportable() -> None:
    """Python builtins are always available — ``len`` shouldn't count
    as an unresolved name.
    """
    plan = build_module_export_plan(
        """
df = load_data()

def count(items):
    return len(items)
""".strip()
    )
    assert plan.is_exportable is True
    assert "count" in plan.exported_symbols


def test_def_with_inner_closure_over_parameter_is_exportable() -> None:
    """A nested lambda that closes over its outer function's parameter
    should NOT be flagged as referencing an unbound name. Python's
    symtable marks these as ``is_free()`` and resolves them via the
    closure chain, not via module globals.
    """
    plan = build_module_export_plan(
        """
def sort_by_score(items):
    return sorted(items, key=lambda i: items[i]["score"])
""".strip()
    )
    assert plan.is_exportable is True
    assert "sort_by_score" in plan.exported_symbols


def test_control_flow_alongside_self_contained_def_is_exportable() -> None:
    """Control flow at module scope is dropped from the slice. As long
    as the def doesn't depend on names bound by that control flow, it
    exports cleanly.
    """
    plan = build_module_export_plan(
        """
if True:
    x = 1

def add(y):
    return y + 1
""".strip()
    )
    assert plan.is_exportable is True
    assert "add" in plan.exported_symbols
    assert plan.sliced is True


def test_def_referencing_control_flow_assigned_name_blocks() -> None:
    """If a def references a name assigned only inside dropped control
    flow, the slice can't resolve it.
    """
    plan = build_module_export_plan(
        """
if some_cond():
    state = make_state()

def use():
    return state.value
""".strip()
    )
    assert plan.is_exportable is False
    assert "state" in plan.format_error()


# ---------------------------------------------------------------------------
# Hard blockers: not even slicing makes these safe.
# ---------------------------------------------------------------------------


def test_top_level_lambda_assignment_is_blocking() -> None:
    """Lambda assignments look like library code but can't be shared
    via the synthetic module path — they're dropped from the slice and
    flagged as blocking so downstream consumers see a clear error.
    """
    plan = build_module_export_plan("add = lambda y: y + 1")
    assert plan.is_exportable is False
    assert "top-level lambdas are not shareable across cells" in plan.format_error()
    assert "add" in plan.blocking_symbols


def test_kept_def_rebound_at_runtime_blocks_export() -> None:
    """``def f(): ...`` followed by ``f = wrap(f)`` is a real divergence
    risk: the cell's runtime ``f`` is wrapped, but the slice's ``f`` is
    bare. Block to keep the synthetic module honest.
    """
    plan = build_module_export_plan(
        """
def f():
    return 1

f = wrap(f)
""".strip()
    )
    assert plan.is_exportable is False
    assert "diverge" in plan.format_error()
    assert "f" in plan.blocking_symbols
    assert "f" not in plan.exported_symbols


def test_kept_constant_rebound_at_runtime_blocks_export() -> None:
    """A literal const reassigned at runtime would cause the slice to
    snapshot the wrong value. Block to avoid silent divergence.
    """
    plan = build_module_export_plan(
        """
THRESHOLD = 0.5
THRESHOLD = compute_threshold()

def is_outlier(x):
    return x > THRESHOLD
""".strip()
    )
    assert plan.is_exportable is False
    assert "diverge" in plan.format_error()


def test_augmented_assignment_to_kept_name_blocks_export() -> None:
    """``x = 1; x += 1`` — the slice's ``x`` is 1 but the cell's ``x``
    is 2. Block.
    """
    plan = build_module_export_plan("x = 1\nx += 1")
    assert plan.is_exportable is False


# ---------------------------------------------------------------------------
# Cells with pure runtime: nothing to share, no error.
# ---------------------------------------------------------------------------


def test_runtime_only_cell_has_empty_slice_and_no_error() -> None:
    """A cell that's just runtime work with nothing to share has an
    empty slice. There's no error to report — downstream consumers of
    the cell's runtime variables flow through the regular artifact
    path.
    """
    plan = build_module_export_plan("STEP = some_fn()")
    assert plan.is_exportable is True
    assert plan.module_source == ""
    assert plan.exported_symbols == {}
    assert plan.sliced is True


def test_nested_non_literal_in_container_drops_silently() -> None:
    """Same as runtime-only: the assignment isn't a shareable literal,
    so it's dropped and the cell carries nothing through module-export."""
    plan = build_module_export_plan("ITEMS = [1, 2, x]")
    assert plan.is_exportable is True
    assert plan.exported_symbols == {}
    assert plan.sliced is True


def test_annotated_without_value_drops_silently() -> None:
    """``x: int`` without a value doesn't bind anything at runtime, so
    there's nothing to share. Drop and move on."""
    plan = build_module_export_plan("x: int")
    assert plan.is_exportable is True
    assert plan.exported_symbols == {}


# ---------------------------------------------------------------------------
# Single-cell scope limitations.
#
# These tests pin the boundaries of what slicing can do. They're not
# bugs — they're the documented contract that the slicer treats each
# cell as an island. If we ever extend module-export to compose across
# cells (e.g. by passing imports/helpers from upstream cells into the
# synthetic module), these tests should be the first place to revisit.
# ---------------------------------------------------------------------------


def test_def_referencing_cross_cell_import_blocks_export() -> None:
    """Limitation: ``import math`` in another cell doesn't help. The
    def's same cell must carry the import for the synthetic module to
    resolve ``math`` at call time.
    """
    plan = build_module_export_plan(
        """
def pi():
    return math.pi
""".strip()
    )
    assert plan.is_exportable is False
    assert "math" in plan.format_error()
    assert "pi" in plan.blocking_symbols


def test_def_referencing_cross_cell_helper_blocks_export() -> None:
    """Limitation: a def can't call a helper function defined in
    another cell. The synthetic module is built from this cell's
    source only — there's no transitive composition with upstream
    cells.
    """
    plan = build_module_export_plan(
        """
def use_helper(x):
    return helper(x) + 1
""".strip()
    )
    assert plan.is_exportable is False
    assert "helper" in plan.format_error()
    assert "use_helper" in plan.blocking_symbols


def test_annotation_reference_blocks_without_future_import() -> None:
    """Type annotations referencing unbound names block the slice on
    Python ≤ 3.13 (annotations evaluated at def-time would NameError
    on module load), but pass through on Python 3.14+ where PEP 749
    makes annotations lazy by default. The runtime semantics differ
    by Python version, and so does this safety check.
    """
    plan = build_module_export_plan(
        """
def transform(x: Df) -> Df:
    return x
""".strip()
    )
    if sys.version_info >= (3, 14):
        # PEP 749 — annotations are stored as a deferred ``__annotate__``
        # function and never evaluated at def-time, so the reference to
        # the unbound ``Df`` is harmless on module load. ``symtable``
        # correctly reports no free-variable concern; the slice exports
        # cleanly. Downstream code that accesses
        # ``transform.__annotations__`` would still raise NameError
        # lazily; that's a separate concern from module-export safety.
        assert plan.is_exportable is True
        assert "transform" in plan.exported_symbols
    else:
        # Pre-3.14: annotations evaluated immediately, NameError on
        # module load if ``Df`` isn't bound, so we block.
        assert plan.is_exportable is False
        assert "Df" in plan.format_error()


def test_future_annotations_relaxes_annotation_check() -> None:
    """With ``from __future__ import annotations`` (PEP 563), type
    annotations are stringified rather than evaluated. ``symtable``
    correctly drops them from the reference set, so a def with an
    annotation that's not bound in the slice still exports cleanly.
    Without this, every cell that uses cross-cell types would have to
    duplicate imports just for the type names.
    """
    plan = build_module_export_plan(
        """
from __future__ import annotations

def transform(x: Df) -> Df:
    return x
""".strip()
    )
    assert plan.is_exportable is True
    assert "transform" in plan.exported_symbols


def test_sliced_source_loses_comments_in_synthetic_module() -> None:
    """Limitation: ``ast.unparse`` doesn't preserve comments. Sliced
    cells lose comments in the synthetic module's source (the cell's
    on-disk file is untouched). Pure module cells keep their bytes
    verbatim — see ``test_pure_cell_keeps_original_source_bytes``.
    """
    plan = build_module_export_plan(
        """
df = load()  # runtime setup, will be dropped

def helper(x):
    # this comment is lost in the slice
    return x + 1
""".strip()
    )
    assert plan.is_exportable is True
    assert plan.sliced is True
    assert "this comment is lost" not in plan.module_source
    # The function still exports correctly — comments are cosmetic.
    assert "helper" in plan.exported_symbols


def test_star_import_is_blocked() -> None:
    """Limitation: ``from foo import *`` binds names the slicer can't
    enumerate. The slice can't validate that a def's free vars are
    covered by the star import, so we drop the import and refuse to
    export.
    """
    plan = build_module_export_plan(
        """
from math import *

def circle(r):
    return pi * r * r
""".strip()
    )
    assert plan.is_exportable is False
    assert "star imports are not supported" in plan.format_error()
