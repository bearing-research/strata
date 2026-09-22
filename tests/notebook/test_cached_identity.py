"""What a cache hit hands back has to be the value its key identifies.

Round 5 found three ways it was not:

- A consumer of a ``@per_variant`` fan-out reads every variant's result, and
  its cache key had nothing from any of them. The fan-out cell's URIs carry an
  ``@`` in the id itself (``..._var_score@variant=triple@v=1``), the key
  builder cut at the first ``@``, found no such artifact, and skipped the
  input: any change to the fan-out came back as the old dict. Keying on the
  one URI the cell records is still not enough for a ``@nocache`` fan-out,
  where one instance can change while the recorded one does not, so the key
  covers every instance the consumer reads.
- A display output borrowed its metadata (the preview an agent reads) from
  whatever the cell was showing, and only swapped in the matched artifact's
  URI. Reverting a source to an earlier value hit that value's bytes and
  reported the later value's preview.
- A failed run clears the cell's display list, and the resolver only looked
  for as many outputs as the cell currently showed, which was none. Recovering
  through a cache hit returned success with the display missing.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from strata.notebook.executor import CellExecutor
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import write_cell
from tests.notebook.test_cli import _build_notebook


def _session(nb: Path) -> NotebookSession:
    return NotebookSession(parse_notebook(nb), nb)


def _set(session: NotebookSession, nb: Path, cell_id: str, source: str) -> None:
    session.notebook_state.get_cell(cell_id).source = source
    write_cell(nb, cell_id, source)
    session.re_analyze_cell(cell_id)


def _run(session: NotebookSession, cell_id: str, *, rerun: bool = False):
    executor = CellExecutor(session)
    source = session.notebook_state.get_cell(cell_id).source
    run = executor.execute_cell_rerun if rerun else executor.execute_cell
    return asyncio.run(run(cell_id, source))


# -- finding 1: a fan-out consumer's key includes the fan-out at all -------


def test_a_fanout_consumer_follows_a_change_to_any_variant(tmp_path: Path):
    from strata.notebook.writer import set_variant_mode

    nb = _build_notebook(
        tmp_path,
        cells=[
            ("load", "X = [1.0, 2.0, 3.0]\n", None),
            ("vdouble", "# @variant model double\npreds = [v * 2 for v in X]\n", "load"),
            ("vtriple", "# @variant model triple\npreds = [v * 3 for v in X]\n", "vdouble"),
            ("ev", "# @per_variant\nscore = sum(preds)\n", "vtriple"),
            ("report", "current = dict(score)\ncurrent\n", "ev"),
        ],
    )
    set_variant_mode(nb, "model", "sweep")
    session = _session(nb)

    first = _run(session, "report")
    assert first.success, first.error
    assert first.display_output["preview"] == {"double": 12.0, "triple": 18.0}

    # Change the variant the fan-out cell's one recorded URI does NOT point at,
    # so the test does not depend on which variant happened to store last.
    recorded = session.notebook_state.get_cell("ev").artifact_uris["score"]
    other = "vtriple" if "variant=double" in recorded else "vdouble"
    factor = "3" if other == "vtriple" else "2"
    name = "triple" if other == "vtriple" else "double"
    _set(
        session,
        nb,
        other,
        session.notebook_state.get_cell(other).source.replace(f"v * {factor}", "v * 10"),
    )

    second = _run(session, "report")
    assert second.success, second.error
    assert second.cache_hit is False
    assert second.display_output["preview"][name] == 60.0


# -- finding 2: a reverted value reports its own preview -------------------


def test_reverting_a_value_shows_that_values_preview(tmp_path: Path):
    nb = _build_notebook(
        tmp_path,
        cells=[
            ("p", "divisor = 2\ndivisor\n", None),
            ("c", "ratio = 10 / divisor\n", "p"),  # consumes divisor: p stores it
        ],
    )
    session = _session(nb)

    assert _run(session, "p").display_output["preview"] == 2
    _set(session, nb, "p", "divisor = 5\ndivisor\n")
    assert _run(session, "p").display_output["preview"] == 5

    _set(session, nb, "p", "divisor = 2\ndivisor\n")
    back = _run(session, "p")

    assert back.cache_hit is True  # the revert is served, not recomputed
    assert back.display_output["preview"] == 2
    assert session.notebook_state.get_cell("p").display_output.preview == 2


# -- finding 3: recovery through a cache hit keeps the display -------------


def test_recovering_through_a_cache_hit_keeps_the_display(tmp_path: Path):
    nb = _build_notebook(
        tmp_path,
        cells=[
            ("p", "divisor = 2\n", None),
            ("c", "ratio = 10 / divisor\nprint(ratio)\nratio\n", "p"),
        ],
    )
    session = _session(nb)

    assert _run(session, "c").display_output["preview"] == 5.0
    _set(session, nb, "p", "divisor = 0\n")
    assert _run(session, "c").success is False  # ZeroDivisionError
    _set(session, nb, "p", "divisor = 2\n")

    recovered = _run(session, "c")

    assert recovered.success, recovered.error
    assert recovered.cache_hit is True
    assert recovered.display_output is not None
    assert recovered.display_output["preview"] == 5.0
    assert session.notebook_state.get_cell("c").display_outputs


# -- the pieces those fixes rest on -----------------------------------------


def test_an_artifact_uri_is_parsed_at_its_last_version_marker():
    """A fan-out instance's id carries an ``@`` of its own.

    Cutting at the first ``@`` named ``..._var_score``, an artifact that does
    not exist, and every fan-out input hashed to nothing: a consumer's key
    ignored the fan-out entirely.
    """
    uri = "strata://artifact/nb_x_cell_ev_var_score@variant=triple@v=3"
    assert NotebookSession._parse_artifact_uri(uri) == ("nb_x_cell_ev_var_score@variant=triple", 3)
    assert NotebookSession._parse_artifact_uri("strata://artifact/nb_x_cell_a_var_y@v=12") == (
        "nb_x_cell_a_var_y",
        12,
    )


def test_a_revert_restores_every_display_the_reverted_run_produced(tmp_path: Path):
    """Promotion restores as many displays as the reverted run recorded.

    ``A`` shows two outputs, ``C`` two others, then ``B`` only one. Reverting to
    ``A`` found the cell showing one (``B``'s count) and promoted only the first
    display, leaving the second slot on ``C``'s version: the variable came back
    and the display set did not.
    """

    def source(value: int, note: str | None) -> str:
        shown = f'display(Markdown("{note}"))\n' if note else ""
        return f"divisor = {value}\n{shown}divisor\n"

    nb = _build_notebook(
        tmp_path,
        cells=[("p", source(2, "A"), None), ("c", "ratio = 10 / divisor\n", "p")],
    )
    session = _session(nb)

    assert len(_run(session, "p").display_outputs) == 2
    _set(session, nb, "p", source(3, "C"))
    assert len(_run(session, "p").display_outputs) == 2
    _set(session, nb, "p", source(5, None))
    assert len(_run(session, "p").display_outputs) == 1

    _set(session, nb, "p", source(2, "A"))
    back = _run(session, "p")

    assert back.cache_hit is True
    assert len(back.display_outputs) == 2
    assert back.display_outputs[0]["markdown_text"] == "A"
    assert back.display_outputs[1]["preview"] == 2


def test_a_failed_rerun_stays_an_error_over_a_cached_success(tmp_path: Path):
    """A rerun can fail at the very key an earlier success is cached under.

    The resolver now finds that success's displays even after a failure, so the
    walk computes ``ready``. Until the cell is edited or runs again
    successfully, the failure is what is true about it.
    """
    from strata.notebook.models import CellStatus

    counter = tmp_path / "count.txt"
    flaky = (
        "from pathlib import Path as _P\n"
        f"_c = _P({str(counter)!r})\n"
        "_c.write_text(str(int(_c.read_text()) + 1) if _c.exists() else '1')\n"
        "n = int(_c.read_text())\n"
        "assert n < 2, n\n"
        "n\n"
    )
    nb = _build_notebook(tmp_path, cells=[("f", flaky, None)])
    session = _session(nb)

    assert _run(session, "f").success
    failed = _run(session, "f", rerun=True)
    assert failed.success is False
    session.mark_cell_error("f")

    session.compute_staleness()

    assert session.notebook_state.get_cell("f").status == CellStatus.ERROR


def test_a_fresh_fanout_consumer_follows_every_instance(tmp_path: Path):
    """Why the key enumerates instances rather than trusting the recorded one.

    Editing a variant moves every instance's key at once (they share the fan-out
    cell's base), so for ordinary fan-outs the one URI the cell records is
    enough once it parses. A ``# @nocache`` fan-out is keyed on content
    instead, and there one instance can change while the recorded one does
    not: here only the unrecorded variant reads a file that changes.
    """
    from strata.notebook.writer import set_variant_mode

    files = {"double": tmp_path / "double.txt", "triple": tmp_path / "triple.txt"}
    for path in files.values():
        path.write_text("0")
    fanout = (
        "# @per_variant\n# @nocache\n"
        f"_extra = {{2.0: {str(files['double'])!r}, 3.0: {str(files['triple'])!r}}}[preds[0]]\n"
        "score = sum(preds) + int(open(_extra).read())\n"
    )
    nb = _build_notebook(
        tmp_path,
        cells=[
            ("load", "X = [1.0, 2.0, 3.0]\n", None),
            ("vdouble", "# @variant model double\npreds = [v * 2 for v in X]\n", "load"),
            ("vtriple", "# @variant model triple\npreds = [v * 3 for v in X]\n", "vdouble"),
            ("ev", fanout, "vtriple"),
            ("report", "current = dict(score)\ncurrent\n", "ev"),
        ],
    )
    set_variant_mode(nb, "model", "sweep")
    session = _session(nb)

    first = _run(session, "report")
    assert first.success, first.error
    assert first.display_output["preview"] == {"double": 12.0, "triple": 18.0}

    recorded = session.notebook_state.get_cell("ev").artifact_uris["score"]
    other = "triple" if "variant=double" in recorded else "double"
    files[other].write_text("100")

    second = _run(session, "report")
    assert second.success, second.error
    assert second.display_output["preview"][other] == first.display_output["preview"][other] + 100
