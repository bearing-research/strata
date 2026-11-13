"""The uptake task bank: its seeded project, the answers it expects, and how replies
and workspaces are checked."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import pytest

from evals.agent_notebook.uptake import tasks as bank
from evals.agent_notebook.uptake import workspace as ws
from evals.agent_notebook.uptake.tasks import Files, Number, Word, numbers_in


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> Path:
    """The project as an arm sees it, at its full size."""
    return ws.build(tmp_path_factory.mktemp("uptake") / "harbor")


@pytest.fixture(autouse=True)
def _template_imports_do_not_leak(monkeypatch):
    # Answers about harbor's code import it from the template; keep that out of the suite.
    monkeypatch.setattr(sys, "path", list(sys.path))
    for name in [n for n in sys.modules if n.split(".")[0] in ("harbor", "fleet")]:
        monkeypatch.delitem(sys.modules, name)


def _fresh(built: Path, tmp_path: Path) -> Path:
    """A copy of *built* a test may edit."""
    return Path(shutil.copytree(built, tmp_path / "harbor"))


class TestTheBank:
    def test_it_has_29_tasks_of_five_kinds(self):
        assert Counter(task.kind for task in bank.TASKS) == {
            bank.ITERATION: 6,
            bank.PROBING: 5,
            bank.PLOT: 4,
            bank.NEUTRAL: 6,
            bank.MUST_NOT: 8,
        }
        assert len(bank.TASKS_BY_ID) == 29
        assert all(len(t.turns) == 4 for t in bank.TASKS if t.kind == bank.ITERATION)

    def test_no_prompt_names_the_tool(self):
        prompts = " ".join(q.prompt for task in bank.TASKS for q in task.turns)

        assert not re.search(r"notebook|scratch|\bcell|strata", prompts, re.I)

    def test_the_holdout_is_the_seeded_third_of_each_kind(self):
        assert bank.HOLDOUT == bank.seeded_holdout()
        assert bank.seeded_holdout(bank.HOLDOUT_SEED + 1) != bank.HOLDOUT
        kinds = Counter(bank.TASKS_BY_ID[i].kind for i in bank.HOLDOUT)
        assert kinds == bank.HOLDOUT_COUNTS
        held, tuned = bank.split(holdout=True), bank.split(holdout=False)
        assert {t.id for t in held} == bank.HOLDOUT
        assert len(held) + len(tuned) == 29 and not {t.id for t in held} & {t.id for t in tuned}


class TestTheEvents:
    def test_they_are_the_same_for_a_seed(self):
        assert ws.events(500, seed=3).equals(ws.events(500, seed=3))
        assert not ws.events(500, seed=3).equals(ws.events(500, seed=4))

    def test_each_question_about_them_has_one_answer(self, built):
        c = bank.Context(built)
        e = c.events

        assert len(e) == 1_000_000
        assert bank._weekday(c) == "Tuesday"
        assert bank._heavy(c) == ws.HEAVY_CUSTOMER
        assert e.groupby("region")["amount"].median().idxmax() == "west"
        assert e.groupby("region")["latency_ms"].quantile(0.95).idxmax() == "east"
        assert bank._worst_error_day(c).options[0] == ws.SPIKE_DAY


class TestTheAnswers:
    @pytest.mark.parametrize("task", bank.TASKS, ids=lambda t: t.id)
    def test_every_answer_computes(self, task, built):
        expected = task.expected(built)

        assert len(expected) == len(task.turns)
        assert all(isinstance(e, (Number, Word, Files)) for e in expected)

    def test_a_few_pinned(self, built):
        def first(task_id: str) -> Number | Word | Files:
            return bank.TASKS_BY_ID[task_id].expected(built)[0]

        assert first("neutral_modpow") == Number(246, rel=0)
        assert first("neutral_eta") == Number(6, rel=0)
        assert first("neutral_distance") == Number(1800.7, rel=0, abs=0.05)
        assert first("neutral_weekday") == Word(("Saturday",))
        assert first("json_config_regions") == Word(
            ("north", "south", "east", "west", "central"), every=True
        )
        assert first("plot_error_rate").options[0] == "2026-02-17"


class TestAReply:
    def test_numbers_are_read_as_written_and_scaled(self):
        assert numbers_in("8.58 million, 12k, 1,234.5 and 3e-2 in p00 v0.2") == [
            8.58,
            8_580_000.0,
            12.0,
            12_000.0,
            1234.5,
            0.03,
        ]

    def test_a_suffix_that_is_no_scale_is_read_as_written(self):
        assert numbers_in("about $8.6Mn, or 3kn") == [8.6, 3.0]

    def test_a_number_within_tolerance_passes(self, built):
        exact = Number(179_935, rel=0)
        close = Number(8_579_481.0, rel=0.001)

        assert exact.passes("There were 179,935 west events.", built)
        assert not exact.passes("There were 179,936 west events.", built)
        assert close.passes("About 8.58 million in total.", built)
        assert not close.passes("About 8.6 million in total.", built)

    def test_a_share_may_be_a_percentage(self, built):
        share = Number(0.01374, rel=0.02, share=True)

        assert share.passes("0.0137 of them", built)
        assert share.passes("about 1.37%", built)
        assert not share.passes("about 13.7%", built)
        assert not Number(0.01374, rel=0.02).passes("about 1.37%", built)

    def test_a_word_is_matched_whole(self, built):
        assert Word(("west",)).passes("The West leads.", built)
        assert not Word(("west",)).passes("Only the north-west and western ones.", built)
        assert not Word(("p01",)).passes("p010 leads", built)
        assert Word(("Tuesday",)).passes("Tuesdays are busiest.", built)
        dates = bank.Word(bank._date_words(date(2026, 2, 17)))
        for said in ("February 17th, 2026", "17 February", "Feb 17", "on 2026-02-17."):
            assert dates.passes(said, built), said
        assert not dates.passes("February 1 and 7", built)
        both = Word(("east", "west"), every=True)
        assert both.passes("east and west", built)
        assert not both.passes("east and north", built)

    @pytest.mark.parametrize(
        "task", [t for t in bank.TASKS if t.kind != bank.MUST_NOT], ids=lambda t: t.id
    )
    def test_its_own_answer_passes_and_another_does_not(self, task, built):
        for expect in task.expected(built):
            if isinstance(expect, Word):
                right = " and ".join(expect.options if expect.every else expect.options[:1])
                wrong = "none of them"
            else:
                right = f"It is {expect.value:,}."
                wrong = f"It is {expect.value * 2 + 100:,}."
            assert expect.passes(right, built), (task.id, right)
            assert not expect.passes(wrong, built), (task.id, wrong)


def _fix_slugify(w: Path) -> None:
    path = w / "src" / "harbor" / "text.py"
    path.write_text(path.read_text().replace('r"[^a-z0-9]"', 'r"[^a-z0-9]+"'))


def _rename_eta(w: Path) -> None:
    for path in [w / "src" / "harbor" / "shipping.py", w / "tests" / "test_shipping.py"]:
        path.write_text(path.read_text().replace("eta_days", "delivery_days"))


def _add_km_to_miles(w: Path) -> None:
    path = w / "packages" / "fleet" / "src" / "fleet" / "units.py"
    path.write_text(path.read_text() + "\n\ndef km_to_miles(km):\n    return km * 0.621371\n")


def _set_timeout(w: Path) -> None:
    path = w / "data" / "config.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "timeout_s": 30}))


def _bump_version(w: Path) -> None:
    path = w / "pyproject.toml"
    path.write_text(path.read_text().replace('version = "0.1.0"', 'version = "0.2.0"', 1))


def _write_geo_test(w: Path) -> None:
    (w / "tests" / "test_geo.py").write_text(
        "from fleet.geo import distance\n\n\ndef test_symmetric():\n"
        "    assert distance('east', 'west') == distance('west', 'east')\n"
    )


EDITS = {
    "edit_fix_slugify": _fix_slugify,
    "edit_rename_eta": _rename_eta,
    "edit_add_km_to_miles": _add_km_to_miles,
    "edit_config_timeout": _set_timeout,
    "edit_bump_version": _bump_version,
    "edit_geo_test": _write_geo_test,
}


class TestAnEditTask:
    def test_every_files_check_has_a_reference_edit(self, built):
        files = {t.id for t in bank.TASKS if isinstance(t.expected(built)[0], Files)}

        assert files == set(EDITS)

    @pytest.mark.parametrize("task_id", sorted(EDITS))
    def test_it_fails_untouched_and_passes_once_done(self, task_id, built, tmp_path):
        work = _fresh(built, tmp_path)
        (check,) = bank.TASKS_BY_ID[task_id].expected(work)

        assert not check.passes("", work)
        EDITS[task_id](work)
        assert check.passes("", work)

    def test_a_rename_left_half_done_fails(self, built, tmp_path):
        work = _fresh(built, tmp_path)
        (check,) = bank.TASKS_BY_ID["edit_rename_eta"].expected(work)
        path = work / "src" / "harbor" / "shipping.py"
        path.write_text(path.read_text().replace("eta_days", "delivery_days"))

        assert not check.passes("", work)
        # Done, but with the old name kept as an alias.
        _rename_eta(work)
        path.write_text(path.read_text() + "\neta_days = delivery_days\n")
        assert not check.passes("", work)

    def test_the_seeded_bug_is_the_only_failing_test(self, built, tmp_path):
        work = _fresh(built, tmp_path)

        assert not bank._pytest(work, "tests")
        assert bank._pytest(work, "tests/test_shipping.py")


class TestTheLoadCounter:
    def _load(self, work: Path, log: Path, *args: str, **env: str) -> None:
        env = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                [str(work / "src"), str(work / "packages" / "fleet" / "src")]
            ),
            "HARBOR_LOAD_LOG": str(log),
            "HARBOR_LOAD_SECONDS": "0",
            **env,
        }
        subprocess.run([sys.executable, *args], cwd=work, env=env, check=True, timeout=120)

    def test_each_load_is_a_line_tagged_with_its_cell(self, built, tmp_path):
        work = _fresh(built, tmp_path)
        log = tmp_path / "loads.jsonl"
        script = work / "explore.py"
        script.write_text("from harbor.data import load_events\nprint(len(load_events()))\n")

        self._load(work, log, "-c", "from harbor.data import load_events; load_events()")
        self._load(work, log, str(script), STRATA_CELL_ID="c1")

        lines = [json.loads(line) for line in log.read_text().splitlines()]
        assert [line["cell"] for line in lines] == [None, "c1"]
        assert [line["argv"] for line in lines] == [["-c"], [str(script)]]

    def test_unset_it_writes_nothing_and_the_load_still_waits(self, built, tmp_path):
        work = _fresh(built, tmp_path)
        log = tmp_path / "loads.jsonl"
        code = (
            "import time; from harbor.data import load_events; t = time.monotonic();"
            " load_events(); assert time.monotonic() - t >= 0.5"
        )

        self._load(work, log, "-c", code, HARBOR_LOAD_LOG="", HARBOR_LOAD_SECONDS="0.5")

        assert not log.exists()
