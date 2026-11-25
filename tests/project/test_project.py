"""The project folder: datasets on disk, commented defaults, runs from a later step, validate.

Oracles: `Dataset.fingerprint()` before writing and after reading (the
design's §5.1.1 form), the csv reader reading the same file, PyYAML
reading the commented text, and the record of the earlier run.
"""

import json

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

import dazflow as dz
from dazflow.cli import app
from dazflow.project import commented_yaml, read_dataset, schema_path, write_dataset
from dazflow.steps import registry

runner = CliRunner()


def dataset(seed=0, n=40):
    rng = np.random.default_rng(seed)
    index = pd.period_range("2010Q1", periods=n, freq="Q")
    values = rng.normal(size=(n, 2)) * [1e-7, 1e9]
    values[3, 0] = np.nan
    frame = pd.DataFrame(values, index=index, columns=["y", "x"])
    meta = {
        "y": {"role": "target", "unit": "%", "transforms": ("log", "diff(lag=1)")},
        "x": {"role": "driver", "expected_sign": -1, "source": "data/raw/x.csv"},
    }
    ds = dz.Dataset(frame, meta)
    return ds.with_log(
        dz.LogEntry(action="fill", reason="a reason", rows=("2010Q2",), columns=("y",))
    )


@pytest.mark.parametrize("seed", range(5))
def test_a_dataset_written_and_read_back_has_the_same_fingerprint(tmp_path, seed):
    ds = dataset(seed)
    path = write_dataset(ds, tmp_path / "d" / "explore.csv")
    back = read_dataset(path)
    assert back.fingerprint() == ds.fingerprint()
    assert back.log == ds.log
    assert dict(back.series) == dict(ds.series)


@pytest.mark.parametrize(("start", "freq"), [("2001-01", "M"), ("2001", "Y")])
def test_monthly_and_annual_datasets_round_trip(tmp_path, start, freq):
    frame = pd.DataFrame(
        {"y": np.arange(30.0) / 7}, index=pd.period_range(start, periods=30, freq=freq)
    )
    ds = dz.Dataset(frame)
    assert read_dataset(write_dataset(ds, tmp_path / "a.csv")).fingerprint() == ds.fingerprint()


def test_the_dataset_csv_is_readable_by_the_csv_reader(tmp_path):
    ds = dataset()
    path = write_dataset(ds, tmp_path / "ingest.csv")
    got = dz.readers.read({"path": str(path)})
    pd.testing.assert_frame_equal(got.frame, ds.frame, check_names=False, check_freq=False)


def test_an_edited_dataset_is_refused(tmp_path):
    path = write_dataset(dataset(), tmp_path / "clean.csv")
    lines = path.read_text().splitlines()
    lines[5] = lines[5].rsplit(",", 1)[0] + ",1.5"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(dz.RecordError, match="fingerprint"):
        read_dataset(path)


def test_a_dataset_whose_fingerprint_is_not_the_recorded_one_is_refused(tmp_path):
    path = write_dataset(dataset(), tmp_path / "clean.csv")
    with pytest.raises(dz.RecordError, match="recorded fingerprint"):
        read_dataset(path, fingerprint="0" * 64)


def test_a_missing_schema_file_is_refused(tmp_path):
    path = write_dataset(dataset(), tmp_path / "clean.csv")
    schema_path(path).unlink()
    with pytest.raises(dz.RecordError, match="schema"):
        read_dataset(path)


@pytest.mark.parametrize(
    "step", ["ingest", "clean", "explore", "build", "select", "forecast", "diagnose"]
)
def test_commented_yaml_reads_back_as_the_configuration(step):
    cls = registry.get(step).Config
    examples = {
        "ingest": dz.project._folder._example_ingest,
        "select": dz.project._folder._example_select,
        "diagnose": dz.project._folder._example_diagnose,
        "forecast": lambda: cls.model_validate(
            {"reversion": {"level": 2.5}, "bounds": [0.0, None], "anchor": {"rule": "carry"}}
        ),
    }
    cfg = (
        examples[step]()
        if step in examples
        else cls.model_validate(
            {"exclusions": [["a", "b"]], "forms": ["ols", {"form": "glm", "config": {}}]}
        )
        if step == "build"
        else cls.model_validate(
            {
                "treatments": [
                    {"action": "fill", "series": ["y"], "periods": ["2020Q1"], "reason": "r"}
                ]
            }
            if step == "clean"
            else {"transforms": {"y": ["log"]}, "expected_signs": {"x": -1}}
        )
    )
    text = commented_yaml(cfg, "a header\nof two lines")
    assert yaml.safe_load(text) == cfg.model_dump(mode="json")
    assert cls.from_yaml(text) == cfg
    assert text.startswith("# a header\n# of two lines\n\n")


def test_commented_yaml_describes_each_described_field():
    text = commented_yaml(dz.project.ProjectConfig(name="x"))
    for name, info in dz.project.ProjectConfig.model_fields.items():
        first = info.description.splitlines()[0]
        assert f"# {first}\n" in text, name
        assert f"\n{name}:" in text


# --- runs from a later step -------------------------------------------------------


def test_a_step_command_carries_the_earlier_results_from_the_record(example):
    assert runner.invoke(app, ["run", "-p", str(example)]).exit_code == 0
    first = json.loads((example / "reports" / "record.json").read_text(encoding="utf-8"))
    assert runner.invoke(app, ["explore", "-p", str(example)]).exit_code == 0
    second = json.loads((example / "reports" / "record.json").read_text(encoding="utf-8"))
    assert second["run"]["steps"] == ["explore"]
    assert second["run"]["carried"] == ["ingest", "clean"]
    for step in ("ingest", "clean"):
        assert second["results"][step] == first["results"][step]
        assert second["config"][step] == first["config"][step]
        assert second["datasets"][step] == first["datasets"][step]
    # The carried steps' inputs; select's scenarios file belongs to a step not run now.
    assert second["inputs"] == [i for i in first["inputs"] if i["step"] in ("ingest", "clean")]
    assert (
        second["results"]["explore"]["input_fingerprint"]
        == first["datasets"]["clean"]["fingerprint"]
    )


def test_a_changed_earlier_dataset_stops_a_later_step(example):
    assert runner.invoke(app, ["run", "-p", str(example), "--to", "clean"]).exit_code == 0
    path = example / "data" / "interim" / "clean.csv"
    lines = path.read_text().splitlines()
    lines[3] = lines[3].rsplit(",", 1)[0] + ",99"
    path.write_text("\n".join(lines) + "\n")
    result = runner.invoke(app, ["explore", "-p", str(example)])
    assert result.exit_code == 1
    assert "fingerprint" in result.stderr


def test_a_rerun_replaces_the_steps_report_folder(example):
    runner.invoke(app, ["run", "-p", str(example), "--to", "ingest"])
    stale = example / "reports" / "ingest" / "stale.md"
    stale.write_text("old")
    runner.invoke(app, ["ingest", "-p", str(example)])
    assert not stale.exists()
    files = sorted(p.name for p in (example / "reports" / "ingest").iterdir())
    assert files == [
        "F1.1.svg",
        "S1.1.md",
        "T1.1.html",
        "T1.1.md",
        "T1.2.html",
        "T1.2.md",
        "result.json",
    ]


# --- validate ---------------------------------------------------------------------


def test_validate_reads_every_input(example):
    (example / "data" / "raw" / "series.csv").unlink()
    with pytest.raises(dz.ReaderError):
        dz.project.load(example).validate()


def test_validate_refuses_a_config_file_for_no_step(example):
    (example / "config" / "explor.yaml").write_text("{}\n")
    with pytest.raises(dz.ProjectError, match=r"explor\.yaml"):
        dz.project.load(example).validate()


def test_validate_refuses_an_unknown_step_in_the_pipeline(example):
    path = example / "project.yaml"
    path.write_text(
        path.read_text().replace(
            "[ingest, clean, explore, build, select, forecast, diagnose]", "[ingest, adjust]"
        )
    )
    with pytest.raises(dz.PipelineError, match="adjust"):
        dz.project.load(example).validate()


def test_a_style_file_is_used_for_the_reports(example):
    (example / "style.yaml").write_text("numbers:\n  decimals: 4\n")
    path = example / "project.yaml"
    path.write_text(path.read_text().replace("style: null", "style: style.yaml"))
    proj = dz.project.load(example)
    proj.validate()
    proj.write(proj.run(stop="ingest"))
    styled = (example / "reports" / "ingest" / "T1.1.md").read_text(encoding="utf-8")
    table = dz.pipeline.RunRecord.load(proj.record_path).result("ingest").artefacts["T1.1"]
    assert (
        dz.render.render(table, "markdown", style=example / "style.yaml").split("Source")[0]
        in styled
    )


def test_a_panel_is_not_written(tmp_path):
    index = pd.MultiIndex.from_product(
        [["a", "b"], pd.period_range("2000Q1", periods=2, freq="Q")], names=["segment", "period"]
    )
    panel = dz.Dataset(pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0]}, index=index))
    with pytest.raises(dz.ProjectError, match="single-segment"):
        write_dataset(panel, tmp_path / "p.csv")


def test_validate_refuses_an_input_without_a_path(example):
    path = example / "config" / "ingest.yaml"
    path.write_text(path.read_text().replace("path: data/raw/series.csv", "path: null"))
    with pytest.raises(dz.ProjectError, match="has no path"):
        dz.project.load(example).validate()


def test_validate_reads_the_previous_vintage(example):
    path = example / "project.yaml"
    path.write_text(
        path.read_text().replace("previous: null", "previous: {path: data/raw/nope.csv}")
    )
    with pytest.raises(dz.ReaderError):
        dz.project.load(example).validate()


def test_init_json_lists_the_files(tmp_path):
    result = runner.invoke(app, ["init", str(tmp_path / "m"), "--json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout) == {
        "project": "m",
        "path": str(tmp_path / "m"),
        "files": sorted(
            p.relative_to(tmp_path / "m").as_posix()
            for p in (tmp_path / "m").rglob("*")
            if p.is_file()
        ),
    }


def test_validate_reads_the_scenarios_file(example):
    (example / "data" / "raw" / "scenarios.csv").unlink()
    with pytest.raises(dz.SelectError, match=r"scenarios\.csv"):
        dz.project.load(example).validate()


def test_the_project_seed_reaches_select(example):
    path = example / "project.yaml"
    path.write_text(path.read_text().replace("seed: 0", "seed: 17"))
    run = dz.project.load(example).run()
    assert run.results["select"].seed == 17
    assert run.results["select"].confidence.seed == 17
