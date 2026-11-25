"""Replay through the library: frames in memory, the record file, the environment (design D43).

Oracles: the record of a short pipeline run in memory on the example's
series, replayed against itself; a record whose environment is edited,
which must be reported as notes and not as differences.
"""

import pandas as pd
import pytest

import dazflow as dz
from dazflow.pipeline import Pipeline, RunRecord, StepInputs, replay


@pytest.fixture
def in_memory(example):
    """ingest, clean and report of the example's series, passed as a frame, and the record."""
    frame = pd.read_csv(example / "data" / "raw" / "series.csv")
    proj = dz.project.load(example)
    cfg = proj.step_config("ingest")
    cfg = cfg.model_copy(update={"inputs": [cfg.inputs[0].model_copy(update={"path": None})]})
    configs = {"ingest": cfg, "clean": proj.step_config("clean")}
    project = {"name": "in-memory", "seed": 0}
    run = Pipeline(["ingest", "clean", "report"]).run(
        configs, data={"macro": frame}, project=project
    )
    return frame, run.record(project)


def test_a_run_on_frames_in_memory_replays_with_the_same_frames(in_memory):
    frame, record = in_memory
    rep = replay(record, data={"macro": frame})
    assert rep.passed, [str(d) for d in rep.differences]
    assert rep.steps == ("ingest", "clean", "report")
    assert rep.notes == ()


def test_without_its_frames_the_replay_is_refused(in_memory):
    _, record = in_memory
    with pytest.raises(dz.RecordError, match="frames in memory"):
        replay(record)


def test_another_frame_is_a_difference_in_the_inputs(in_memory):
    frame, record = in_memory
    other = frame.copy()
    other.loc[3, "loss_rate"] += 0.01
    rep = replay(record, data={"macro": other})
    assert rep.differences[0].path == "inputs.ingest.macro.fingerprint"
    assert "results.ingest.fingerprint" in [d.path for d in rep.differences]


def test_a_record_file_is_read_by_its_path(in_memory, tmp_path):
    frame, record = in_memory
    path = record.dump(tmp_path / "record.json")
    assert replay(path, data={"macro": frame}).passed


def test_another_environment_is_a_note_not_a_difference(example):
    # A record written "on another machine": the run records an environment that is not
    # this one, and its report's environment table says so. The rerun here builds the
    # report from the recorded environment, so the record still replays.
    frame = pd.read_csv(example / "data" / "raw" / "series.csv")
    proj = dz.project.load(example)
    cfg = proj.step_config("ingest")
    cfg = cfg.model_copy(update={"inputs": [cfg.inputs[0].model_copy(update={"path": None})]})
    here = dz.pipeline.environment()
    env = here.model_copy(
        update={"python": "3.0.0", "packages": {**here.packages, "pandas": "0.1", "gone": "9"}}
    )
    run = Pipeline(["ingest", "clean", "report"]).run(
        {"ingest": cfg, "clean": proj.step_config("clean")},
        data={"macro": frame},
        environment=env,
        project={"name": "elsewhere", "seed": 0},
    )
    record = run.record({"name": "elsewhere", "seed": 0})
    assert record.environment == env
    assert ["Python", "3.0.0"] in [list(r) for r in run.results["report"].artefacts["T10.2"].rows]
    rep = replay(record, data={"macro": frame})
    assert rep.passed, [str(d) for d in rep.differences]
    assert "python: recorded 3.0.0, now" in rep.notes[0]
    assert any(n.startswith("pandas: recorded 0.1, now ") for n in rep.notes)
    assert "gone: recorded 9, now not installed" in rep.notes


def test_the_summary_says_whether_the_replay_matched(in_memory):
    frame, record = in_memory
    rep = replay(record, data={"macro": frame})
    assert rep.summary() == (
        f"replay of run {record.run.id} (3 steps rerun): matches its record "
        "(rtol 1e-09, atol 1e-12)"
    )
    worse = rep.model_copy(
        update={"differences": (*rep.differences, dz.pipeline.Difference(path="x"))}
    )
    assert worse.summary().endswith("1 difference from its record (rtol 1e-09, atol 1e-12)")


def test_a_step_called_outside_a_pipeline_has_no_record():
    with pytest.raises(dz.PipelineError, match="needs the run record"):
        StepInputs(step="report", results={}).record()


def test_the_record_a_step_sees_holds_the_steps_before_it(example):
    seen = {}

    def spy(step, config, inputs):
        seen["record"] = inputs.record()
        return step.run(inputs.record(), config)

    adapters = dz.core.Registry("step adapter")
    for key in dz.pipeline.adapters.keys():  # noqa: SIM118 - a Registry, not a dict
        adapters.add(key, spy if key == "report" else dz.pipeline.adapters.get(key))
    proj = dz.project.load(example)
    configs = {s: proj.step_config(s) for s in ("ingest", "clean", "report")}
    run = Pipeline(["ingest", "clean", "report"], adapters=adapters).run(
        configs, root=example, project=proj.config
    )
    record = seen["record"]
    assert isinstance(record, RunRecord)
    assert list(record.results) == ["ingest", "clean"]
    assert record.run.steps == ("ingest", "clean")
    assert record.project == proj.config.model_dump(mode="json")
    assert [i.name for i in record.inputs] == ["macro"]
    assert run.results["report"].steps == ("ingest", "clean")
