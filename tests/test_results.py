"""The build that produces Result 2, and the hash that makes it comparable.

Every number of Result 2 has to come out of one executable command.  Until now they lived in
docstrings, in ``docs/zigzag.md`` and in whatever run produced them — three places, which is
exactly the arrangement S9 caught giving two different values for the same quantity.  Here
the artefact is the single source, the report reads it rather than recomputing, and a rebuild
has to reproduce the same digest.
"""

from __future__ import annotations

import json
import pathlib
import subprocess

import pytest

from tdadyn.results import (
    ARTEFACT_FILE,
    ARTEFACT_FORMAT_VERSION,
    RESULTS_DIR,
    artefact_digest,
    build_artefact,
    main,
)

pytestmark = pytest.mark.slow_zigzag

pytest.importorskip(
    "dionysus",
    reason="the artefact is produced by the backend: pip install -e '.[zigzag]'",
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def artefact(config):
    """The artefact **as committed**, not a fresh build.

    Reading it keeps most of these tests instant and makes them check the file the
    repository actually carries.  Whether that file is current is a separate question, and
    :func:`test_a_rebuild_reproduces_the_same_digest` is where it is asked — with one
    rebuild rather than one per test.
    """
    path = config.output_root / RESULTS_DIR / ARTEFACT_FILE
    if not path.exists():
        pytest.skip("the artefact is not built; run make results")
    return json.loads(path.read_text(encoding="utf-8"))


# --- the digest ---------------------------------------------------------------------


def test_a_rebuild_reproduces_the_same_digest(artefact, config):
    """Within one environment, as ``docs/data.md`` §5 promises for the manifest.

    Two things at once, and both matter.  The digest must not move between runs, or the
    numbers of one run could never be compared with another's.  And the committed file must
    be the one this code produces, or the repository carries a report of something else.

    This is the expensive test of the target — one full rebuild, most of it the oracle
    cross-check — and it is the price of the artefact being reproducible rather than merely
    present.
    """
    if not (config.output_root / "noisy").is_dir():
        pytest.skip("the datasets are not built; run python -m tdadyn.synthetic.build")
    rebuilt = build_artefact(config)
    assert rebuilt["content_sha256"] == artefact["content_sha256"]


def test_the_digest_ignores_the_fields_that_cannot_be_reproduced(artefact):
    """``generated_at`` and ``package_version`` are excluded, as ``entry_payload`` excludes them.

    Including them would make every rebuild differ and the digest would answer "was this
    written twice", not "does it say the same thing".
    """
    moved = dict(artefact, generated_at="1999-01-01T00:00:00Z", package_version="0.0.0")
    assert artefact_digest(moved) == artefact["content_sha256"]


def test_the_digest_moves_when_the_content_does(artefact):
    """Otherwise it would be a checksum of nothing in particular."""
    changed = json.loads(json.dumps(artefact))
    changed["datasets"][0]["counters"]["invisible_bars"] += 1
    assert artefact_digest(changed) != artefact["content_sha256"]


# --- what the artefact must carry ---------------------------------------------------


def test_the_convention_travels_with_the_barcodes(artefact):
    """``docs/zigzag.md`` §7: barcodes from different conventions are not comparable.

    So the convention is published beside them rather than left to the reader to infer from
    the commit the file came from.
    """
    convention = artefact["convention"]
    assert convention["union_rule"] == "complex_union"
    assert convention["coeff_field"] == 2
    assert convention["backend"] == "dionysus"
    assert "index" in " ".join(convention).lower() or "index_convention" in convention
    assert convention["index_convention"]


def test_every_dataset_carries_all_four_counters_even_when_zero(artefact):
    """§6 asks for them as separate lines, and a missing line is not the same as a zero."""
    required = {
        "invisible_bars",
        "censored_ends",
        "events_from_half_integer_ends",
        "ephemeral_bars",
        "ephemeral_in_prologue",
        "ephemeral_outside_prologue",
    }
    assert artefact["datasets"]
    for entry in artefact["datasets"]:
        assert required <= set(entry["counters"]), entry["dataset_id"]
        assert all(entry["counters"][name] is not None for name in required)


def test_the_null_model_reports_both_denominators_with_their_trials(artefact):
    """A frequency without the number of trials is not a result — the lesson of S8."""
    null = artefact["null_model"]
    for denominator in ("per_snapshot", "per_realisation"):
        rate = null[denominator]
        assert rate["trials"] > 0
        assert 0.0 <= rate["rate"] <= 1.0
        assert len(rate["interval"]) == 2
        assert rate["interval"][0] <= rate["rate"] <= rate["interval"][1]
    assert null["events_per_snapshot"] > 1.0
    assert set(null["events_by_dimension"]) == {"0", "1"}


def test_the_agreement_section_counts_bars_rather_than_saying_ok(artefact):
    """Result 2 claims two independent computations agree, so the artefact says how much.

    A boolean would hide the case where the comparison ran over nothing at all.
    """
    agreement = artefact["agreement"]
    assert agreement["discrepancies"] == 0
    assert agreement["datasets"] == len(artefact["datasets"])

    # Checked on every dataset, not only where the barcode has four bars: an adapter error
    # is likelier to show where a noisy set produces nine hundred.
    assert agreement["bars_compared"] > 3000
    for entry in artefact["datasets"]:
        assert entry["oracle_agreement"]["discrepancies"] == 0, entry["dataset_id"]
        assert entry["oracle_agreement"]["bars"] >= 0


def test_the_metrics_are_broken_down_by_dimension(artefact):
    clean = [entry for entry in artefact["datasets"] if entry["kind"] == "clean"]
    assert len(clean) == 2
    for entry in clean:
        assert entry["metrics"]["recall"] == pytest.approx(1.0)
        assert entry["metrics"]["precision"] == pytest.approx(1.0)
        assert entry["metrics"]["by_dimension"]


def test_the_artefact_carries_the_events_the_metrics_were_computed_on(artefact):
    """The first version of this module serialised an empty list and every test passed.

    Metrics without the events behind them are a summary nobody can audit: the reader cannot
    see which snapshot was reported, whether it was censored, or whether it came from a
    half-integer bar end.  So the events are in the artefact and their presence is checked.
    """
    clean = {entry["dataset_id"]: entry for entry in artefact["datasets"]
             if entry["kind"] == "clean"}
    lifecycle = next(entry for name, entry in clean.items() if name.startswith("cycle_"))
    plain = [event for event in lifecycle["events"] if not event["censored"]]
    assert [(event["snapshot"], event["dim"], event["type"]) for event in sorted(
        plain, key=lambda event: (event["snapshot"], event["dim"])
    )] == [(12, 1, "birth"), (27, 1, "birth"), (41, 1, "death"),
           (48, 0, "birth"), (55, 0, "death")]
    assert any(event["from_half_integer"] for event in plain)
    assert any(event["censored"] for event in lifecycle["events"])

    # The noisy datasets carry the count and not the list: at level 0.1 one of them has
    # 1802 events, and sixteen such lists were three quarters of the file. The omission is
    # recorded rather than silent, and the barcode stays, so the events remain derivable.
    noisy = next(entry for entry in artefact["datasets"] if entry["kind"] == "noisy")
    assert noisy["events"] is None
    assert noisy["n_events"] > 0
    assert noisy["barcode"]


def test_the_noisy_datasets_name_the_base_they_are_measured_against(artefact):
    noisy = [entry for entry in artefact["datasets"] if entry["kind"] == "noisy"]
    assert len(noisy) == 16
    for entry in noisy:
        assert entry["base"], entry["dataset_id"]
        assert entry["noise"]["kind"] in ("spurious_edges", "missing_edges")


def test_the_format_version_is_recorded(artefact):
    assert artefact["format_version"] == ARTEFACT_FORMAT_VERSION


# --- the command line ---------------------------------------------------------------


def test_the_flag_decides_whether_a_file_is_written(artefact, tmp_path, capsys, monkeypatch):
    """As in ``synthetic.build``: the run prints, the flag writes.

    The artefact must be produced by the build and never as a side effect of the tests
    (rule of S9), so the writing has to be something one asks for.

    The computation is stubbed out with the committed artefact: what is under test here is
    the plumbing of the flag, and rebuilding for it would add five minutes to a check that
    does not look at a single number.
    """
    import tdadyn.results as module

    monkeypatch.setattr(module, "build_artefact", lambda config: artefact)
    arguments = ["--config", str(REPO_ROOT / "config.yaml"), "--output-root", str(tmp_path)]

    main(arguments)
    assert not (tmp_path / RESULTS_DIR / ARTEFACT_FILE).exists()
    assert capsys.readouterr().out.strip()

    main(arguments + ["--results"])
    written = tmp_path / RESULTS_DIR / ARTEFACT_FILE
    assert written.exists()
    assert json.loads(written.read_text(encoding="utf-8"))["content_sha256"]


def test_git_does_not_ignore_the_artefact():
    """Checked against the real ``.gitignore``, not against the intention of one.

    The exclusions there are listed file by file, so a new results file is ignored silently.
    That is how the manifest of task 5 stayed out of the repository while the documentation
    said it was in.
    """
    target = f"data/synthetic/{RESULTS_DIR}/{ARTEFACT_FILE}"
    result = subprocess.run(
        ["git", "check-ignore", "-v", target],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    assert "!" in result.stdout or result.returncode != 0, (
        f"{target} is ignored by: {result.stdout.strip()}"
    )


# --- the report ---------------------------------------------------------------------


def test_the_report_reads_the_artefact_and_computes_nothing():
    """Why the report cannot disagree with the artefact.

    S9 caught the same quantity printed with two different values in three places.  Here the
    report has no second source: it opens the file the build wrote and prints from it.  A
    generator that called the pipeline itself could drift from the artefact between runs, and
    nothing would say which number was right.
    """
    source = (REPO_ROOT / "notebooks" / "build_zigzag_report.py").read_text(encoding="utf-8")
    assert "zigzag_events.json" in source
    for computing in ("analyse(", "measure_null(", "degradation_row(", "dionysus_barcode("):
        assert computing not in source, f"the report computes {computing} instead of reading"


def test_the_report_notebook_is_executed_and_not_merely_written():
    """``make report-zigzag`` runs it; an unexecuted notebook proves nothing about its prose.

    The report of task 5 is in exactly that state — written by a script, executed by hand —
    so "every number is the output of an executed cell" was never checkable for it.
    """
    notebook = REPO_ROOT / "notebooks" / "zigzag_report.ipynb"
    if not notebook.exists():
        pytest.skip("the report is not built; run make report-zigzag")
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    code = [cell for cell in cells if cell["cell_type"] == "code"]
    assert code, "the report has no code cells at all"
    assert all(cell["outputs"] for cell in code), "some cell was never executed"


def test_the_makefile_builds_the_artefact_before_the_report():
    """Order matters: a report built from a stale artefact is the failure it exists to avoid."""
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "report-zigzag: results" in makefile
