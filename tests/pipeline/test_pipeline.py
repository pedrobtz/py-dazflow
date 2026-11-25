"""Pipeline: steps by name, results passed through adapters, a failed gate stops the run.

Oracles: the steps called directly on the same data, a toy step that
records what it was given, and the configured order.
"""

import shutil
from typing import ClassVar

import pandas as pd
import pytest
from pydantic import ConfigDict, Field

import dazflow as dz
from dazflow.core.registry import Registry
from dazflow.pipeline import DEFAULT_STEPS, Pipeline, StepInputs, default_adapter
from dazflow.steps import Step, clean, explore, ingest


def test_the_default_pipeline_is_the_built_steps():
    assert (
        Pipeline().steps
        == DEFAULT_STEPS
        == ("ingest", "clean", "explore", "build", "select", "forecast", "diagnose")
    )


@pytest.mark.parametrize(
    ("steps", "match"),
    [
        ([], "no steps"),
        (["ingest", "ingest"], "more than once"),
        (["ingest", "adjust"], "adjust"),
    ],
)
def test_a_pipeline_of_unknown_or_repeated_steps_is_refused(steps, match):
    with pytest.raises(dz.PipelineError, match=match):
        Pipeline(steps)


def test_select_takes_from_and_to_in_pipeline_order():
    p = Pipeline()
    assert p.select() == DEFAULT_STEPS
    assert p.select("clean") == ("clean", "explore", "build", "select", "forecast", "diagnose")
    assert p.select(None, "clean") == ("ingest", "clean")
    assert p.select("clean", "clean") == ("clean",)
    with pytest.raises(dz.PipelineError, match="comes after"):
        p.select("explore", "ingest")


def test_relative_input_paths_are_read_from_the_project_folder(example, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # not the project folder
    proj = dz.project.load(example)
    run = Pipeline().run(proj.configs(), root=example)
    assert run.passed
    assert run.results["ingest"].config.inputs[0].path == "data/raw/series.csv"


def test_the_pipeline_returns_what_the_steps_return(example):
    proj = dz.project.load(example)
    run = Pipeline().run(proj.configs(), root=example)
    cfg = proj.step_config("ingest")
    inputs = [i.model_copy(update={"path": str(example / i.path)}) for i in cfg.inputs]
    ing = ingest.run(cfg.model_copy(update={"inputs": inputs}))
    cl = clean.run(ing.dataset, proj.step_config("clean"))
    assert run.results["ingest"].fingerprint == ing.fingerprint
    assert run.results["clean"].to_dict() == cl.to_dict()
    assert run.results["explore"].input_fingerprint == cl.fingerprint


def test_frames_in_memory_are_ingest_inputs_and_are_fingerprinted(example):
    frame = pd.read_csv(example / "data" / "raw" / "series.csv")
    cfg = dz.project.load(example).step_config("ingest")
    cfg = cfg.model_copy(update={"inputs": [cfg.inputs[0].model_copy(update={"path": None})]})
    run = Pipeline(["ingest"]).run({"ingest": cfg}, data={"macro": frame})
    (entry,) = run.inputs
    assert entry.kind == "dataset"
    assert entry.fingerprint == dz.readers.from_frame(frame).fingerprint()


def test_a_failed_gate_stops_the_run_after_that_step(example):
    proj = dz.project.load(example)
    configs = proj.configs()
    configs["ingest"] = configs["ingest"].model_copy(update={"min_history": 1000})
    run = Pipeline().run(configs, root=example)
    assert run.steps == ("ingest",)
    assert run.stopped_at == "ingest"
    assert not run.passed
    assert run.record().run.status == "gate_failed"


def test_configuration_for_a_step_outside_the_pipeline_is_refused():
    with pytest.raises(dz.PipelineError, match="not in the pipeline"):
        Pipeline(["ingest"]).run({"clean": {}})


def test_a_run_from_a_later_step_needs_the_earlier_results():
    with pytest.raises(dz.PipelineError, match="no result for ingest"):
        Pipeline().run(start="clean")


def test_a_step_that_needs_a_dataset_after_none_is_refused():
    with pytest.raises(dz.PipelineError, match="needs a dataset"):
        Pipeline(["clean"]).run()


# --- a later step joins by name ---------------------------------------------------


class CountConfig(dz.Config):
    minimum: int = Field(1, description="Fewest rows.")


class CountResult(dz.StepResult):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    config: CountConfig
    rows: int
    kept: tuple[str, ...] = ()


class count:
    Config = CountConfig
    Result = CountResult
    seen: ClassVar[list] = []

    @staticmethod
    def run(dataset, config, *, kept=()):
        count.seen.append((dataset.fingerprint(), tuple(kept)))
        return CountResult(config=config, rows=len(dataset.frame), kept=tuple(kept))


@pytest.fixture
def registries():
    steps = Registry("step", protocol=Step)
    for name, module in (("ingest", ingest), ("clean", clean), ("explore", explore)):
        steps.add(name, module)
    steps.add("count", count)
    adapters = Registry("step adapter")
    for name in ("ingest", "clean"):
        adapters.add(name, dz.pipeline.adapters.get(name))
    count.seen.clear()
    return steps, adapters


def test_a_registered_step_runs_by_name_with_the_default_adapter(example, registries):
    steps, adapters = registries
    proj = dz.project.load(example)
    built = ("ingest", "clean", "explore")
    configs = {k: v for k, v in proj.configs().items() if k in built}
    run = Pipeline([*built, "count"], registry=steps, adapters=adapters).run(configs, root=example)
    assert run.results["count"].rows == len(run.results["explore"].dataset.frame)
    assert count.seen == [(run.results["explore"].fingerprint, ())]
    assert run.record().config["count"] == {"minimum": 1}


def test_an_adapter_passes_an_earlier_result_to_a_later_step(example, registries):
    steps, adapters = registries

    @adapters.register("count")
    def _(step, config, inputs: StepInputs):
        return step.run(inputs.require_dataset(), config, kept=inputs.results["explore"].kept)

    proj = dz.project.load(example)
    built = ("ingest", "clean", "explore")
    configs = {k: v for k, v in proj.configs().items() if k in built}
    run = Pipeline([*built, "count"], registry=steps, adapters=adapters).run(configs, root=example)
    assert (
        run.results["count"].kept == run.results["explore"].kept == ("gdp_growth", "unemployment")
    )


def test_default_adapter_is_run_with_the_latest_dataset():
    ds = dz.Dataset(
        pd.DataFrame({"x": [1.0]}, index=pd.period_range("2000Q1", periods=1, freq="Q"))
    )
    count.seen.clear()
    default_adapter(count, CountConfig(), StepInputs(step="count", results={}, dataset=ds))
    assert count.seen == [(ds.fingerprint(), ())]


# --- the previous vintage ---------------------------------------------------------


def test_the_previous_vintage_reaches_the_revisions_check(example):
    import hashlib

    previous = example / "data" / "raw" / "previous.csv"
    shutil.copy(example / "data" / "raw" / "series.csv", previous)
    text = previous.read_text().splitlines()
    head, rows = text[0], text[1:]
    # Revise the last value of loss_rate by much more than the tolerance.
    last = rows[-1].split(",")
    last[1] = str(float(last[1]) + 5)
    previous.write_text("\n".join([head, *rows[:-1], ",".join(last)]) + "\n")

    path = example / "project.yaml"
    path.write_text(
        path.read_text().replace("previous: null", "previous: {path: data/raw/previous.csv}")
    )
    proj = dz.project.load(example)
    run = proj.run(stop="clean")
    revisions = [o for o in run.results["clean"].after if o.key == "revisions"]
    assert {o.column for o in revisions} == {"loss_rate", "gdp_growth", "unemployment"}
    loss = next(o for o in revisions if o.column == "loss_rate")
    assert loss.check is not None
    assert loss.failed == 1
    (entry,) = [i for i in run.inputs if i.name == "previous"]
    assert (entry.step, entry.path, entry.kind) == ("clean", "data/raw/previous.csv", "file-sha256")
    assert entry.fingerprint == hashlib.sha256(previous.read_bytes()).hexdigest()


def test_without_a_previous_vintage_revisions_does_not_apply(example):
    run = dz.project.load(example).run(stop="clean")
    revisions = [o for o in run.results["clean"].after if o.key == "revisions"]
    assert revisions and all(o.check is None for o in revisions)


# --- refusals and the less common inputs ------------------------------------------


def test_a_step_that_returns_no_step_result_is_refused(registries):
    steps, adapters = registries
    adapters.add("count", lambda step, config, inputs: {"rows": 1})
    ds = dz.Dataset(
        pd.DataFrame({"x": [1.0]}, index=pd.period_range("2000Q1", periods=1, freq="Q"))
    )
    with pytest.raises(dz.PipelineError, match="did not return a StepResult"):
        Pipeline(["clean", "count"], registry=steps, adapters=adapters).run(
            start="count", earlier={"clean": clean.run(ds)}
        )


def test_a_previous_vintage_in_memory_is_fingerprinted(example):
    proj = dz.project.load(example)
    first = proj.run(stop="ingest").results["ingest"].dataset
    configs = {k: v for k, v in proj.configs().items() if k in ("ingest", "clean")}
    run = Pipeline(["ingest", "clean"]).run(configs, root=example, previous=first)
    (entry,) = [i for i in run.inputs if i.name == "previous"]
    assert (entry.kind, entry.fingerprint) == ("dataset", first.fingerprint())
    revisions = [o for o in run.results["clean"].after if o.key == "revisions"]
    assert all(o.check is not None and o.failed == 0 for o in revisions)


def test_a_previous_vintage_without_a_path_is_refused():
    with pytest.raises(dz.PipelineError, match="no path"):
        Pipeline(["ingest"]).run({"ingest": {}}, previous={"reader": "csv"})


def test_a_folder_input_is_hashed_by_its_files(tmp_path):
    from dazflow.pipeline import file_sha256

    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "a.csv").write_bytes(b"1")
    first = file_sha256(tmp_path / "d")
    (tmp_path / "d" / "a.csv").write_bytes(b"2")
    assert file_sha256(tmp_path / "d") != first


def test_select_needs_build_before_it(example):
    proj = dz.project.load(example)
    configs = {k: v for k, v in proj.configs().items() if k in ("ingest", "clean", "select")}
    with pytest.raises(dz.PipelineError, match="needs the result of 'build'"):
        Pipeline(["ingest", "clean", "select"]).run(configs, root=example)


def test_forecast_needs_select_before_it(example):
    proj = dz.project.load(example)
    steps = ("ingest", "clean", "explore", "build", "forecast")
    configs = {k: v for k, v in proj.configs().items() if k in steps}
    with pytest.raises(dz.PipelineError, match="needs the result of 'select'"):
        Pipeline(list(steps)).run(configs, root=example)


def test_build_without_explore_takes_the_latest_dataset(example):
    proj = dz.project.load(example)
    steps = ("ingest", "clean", "build", "select")
    configs = {k: v for k, v in proj.configs().items() if k in steps}
    run = Pipeline(steps).run(configs, root=example, seed=4)
    assert run.results["build"].fingerprint == run.results["clean"].fingerprint
    assert run.results["build"].chains == {"loss_rate": (), "gdp_growth": (), "unemployment": ()}
    assert run.results["select"].seed == 4


def test_diagnose_needs_select_before_it(example):
    proj = dz.project.load(example)
    steps = ("ingest", "clean", "explore", "build", "diagnose")
    configs = {k: v for k, v in proj.configs().items() if k in steps}
    with pytest.raises(dz.PipelineError, match="needs the result of 'select'"):
        Pipeline(list(steps)).run(configs, root=example)


def test_diagnose_without_forecast_takes_the_scenarios_coverage(example):
    proj = dz.project.load(example)
    steps = ("ingest", "clean", "explore", "build", "select", "diagnose")
    configs = {k: v for k, v in proj.configs().items() if k in steps}
    run = Pipeline(list(steps)).run(configs, root=example)
    d = run.results["diagnose"]
    assert d.chosen == run.results["select"].chosen
    assert d.horizon == 12  # the example's scenarios cover twelve quarters after the origin
    assert [i.step for i in run.inputs if i.name == "scenarios"] == ["select", "diagnose"]


def test_diagnose_takes_forecasts_horizon(example):
    proj = dz.project.load(example)
    configs = proj.configs()
    configs["forecast"] = configs["forecast"].model_copy(update={"horizon": 4})
    run = Pipeline().run(configs, root=example)
    assert run.results["forecast"].horizon == 4
    assert run.results["diagnose"].horizon == 4
