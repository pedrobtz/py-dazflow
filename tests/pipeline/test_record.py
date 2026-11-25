"""The run record: complete, loadable, versioned (design §8, §10.1).

Oracles: each step's JSON Schema (`Config.json_schema()`, checked by the
jsonschema package) and the defaults it declares; hashlib over the input
file; `platform` and `importlib.metadata`; the step results themselves.
"""

import hashlib
import json
import platform
from importlib.metadata import version

import jsonschema
import pytest

import dazflow as dz
from dazflow.pipeline import SCHEMA_VERSION, RunRecord
from dazflow.steps import registry


def defaults(schema, node=None, path=()):
    """Every (path, default) the schema declares, through nested objects."""
    node = schema if node is None else node
    if "$ref" in node:
        node = schema["$defs"][node["$ref"].rsplit("/", 1)[-1]]
    for name, prop in node.get("properties", {}).items():
        yield (*path, name), prop
        sub = prop
        if "$ref" in sub:
            sub = schema["$defs"][sub["$ref"].rsplit("/", 1)[-1]]
        if sub.get("type") == "object" or "properties" in sub:
            yield from defaults(schema, sub, (*path, name))


@pytest.fixture(scope="module")
def record(ran_example):
    return json.loads((ran_example / "reports" / "record.json").read_text())


@pytest.mark.parametrize("step", ["ingest", "clean", "explore"])
def test_recorded_configuration_validates_against_the_step_schema(record, step):
    cls = registry.get(step).Config
    schema = cls.json_schema()
    config = record["config"][step]
    jsonschema.validate(config, schema)
    assert cls.model_validate(config).model_dump(mode="json") == config


@pytest.mark.parametrize("step", ["ingest", "clean", "explore"])
def test_recorded_configuration_holds_every_default(record, step):
    schema = registry.get(step).Config.json_schema()
    config = record["config"][step]
    seen = 0
    for path, prop in defaults(schema):
        node = config
        for key in path[:-1]:
            node = node[key]
        assert path[-1] in node, ".".join(path)
        # The example sets only ingest's series and inputs; everything else is a default.
        if "default" in prop and step != "ingest":
            assert node[path[-1]] == prop["default"], ".".join(path)
            seen += 1
    assert seen or step == "ingest"


def test_a_config_file_with_nothing_in_it_still_records_every_default(example):
    (example / "config" / "explore.yaml").write_text("# nothing set\n")
    proj = dz.project.load(example)
    rec = proj.write(proj.run()).to_dict()
    assert rec["config"]["explore"] == dz.explore.Config().model_dump(mode="json")
    assert rec["config"]["explore"]["stationarity"]["votes"] == 3


def test_the_record_loads_back_equal(ran_example, record):
    loaded = RunRecord.load(ran_example / "reports" / "record.json")
    assert loaded.to_dict() == record
    assert record["schema_version"] == SCHEMA_VERSION == 1
    assert record["kind"] == "dazflow.record"


def test_a_record_of_another_schema_version_is_refused(record):
    with pytest.raises(dz.RecordError, match="schema version"):
        RunRecord.from_dict({**record, "schema_version": 2})
    with pytest.raises(dz.RecordError):
        RunRecord.from_dict([])


def test_results_rebuild_as_their_step_types(ran_example, record):
    loaded = RunRecord.load(ran_example / "reports" / "record.json")
    for step in ("ingest", "clean", "explore"):
        result = loaded.result(step)
        assert isinstance(result, registry.get(step).Result)
        assert result.dataset is None
        assert result.to_dict() == record["results"][step]
    with pytest.raises(dz.RecordError, match="no result"):
        loaded.result("build")


def test_checks_and_overrides_are_listed_by_step(record):
    want = [
        {"step": step, **c}
        for step in ("ingest", "clean", "explore")
        for c in record["results"][step]["gate"]["checks"]
    ]
    assert list(record["checks"]) == want
    assert record["overrides"] == []


def test_overrides_are_recorded_with_their_reasons(example):
    path = example / "config" / "ingest.yaml"
    path.write_text(
        path.read_text().replace(
            "overrides: []", "overrides:\n  - check: history\n    reason: accepted for the test"
        )
    )
    proj = dz.project.load(example)
    rec = proj.write(proj.run(stop="ingest")).to_dict()
    assert rec["overrides"] == [
        {"step": "ingest", "check": "history", "reason": "accepted for the test"}
    ]


def test_the_environment_is_this_one(record):
    env = record["environment"]
    assert env["python"] == platform.python_version()
    assert env["platform"] == platform.platform()
    for name in ("dazflow", "pandas", "statsmodels", "arch", "pointblank", "pydantic"):
        assert env["packages"][name] == version(name)


def test_every_input_has_its_fingerprint(ran_example, record):
    data = (ran_example / "data" / "raw" / "series.csv").read_bytes()
    assert record["inputs"] == [
        {
            "step": "ingest",
            "name": "macro",
            "path": "data/raw/series.csv",
            "kind": "file-sha256",
            "fingerprint": hashlib.sha256(data).hexdigest(),
        }
    ]


def test_every_dataset_is_recorded_with_its_path_and_fingerprint(ran_example, record):
    for step, folder in (("ingest", "interim"), ("clean", "interim"), ("explore", "processed")):
        entry = record["datasets"][step]
        assert entry["path"] == f"data/{folder}/{step}.csv"
        assert entry["fingerprint"] == record["results"][step]["fingerprint"]
        ds = dz.project.read_dataset(ran_example / entry["path"])
        assert ds.fingerprint() == entry["fingerprint"]


def test_the_run_section(ran_example, record):
    run = record["run"]
    assert run["steps"] == ["ingest", "clean", "explore"]
    assert run["carried"] == []
    assert run["status"] == "passed"
    assert run["stopped_at"] is None
    assert run["started"] <= run["finished"]
    assert run["id"].startswith(run["finished"].replace("-", "").replace(":", ""))
    assert record["project"] == dz.project.load(ran_example).config.model_dump(mode="json")
    assert record["pipeline"] == ["ingest", "clean", "explore"]


def test_a_missing_or_broken_record_file_is_a_record_error(tmp_path):
    with pytest.raises(dz.RecordError, match="cannot be read"):
        RunRecord.load(tmp_path / "record.json")
    (tmp_path / "record.json").write_text("{not json")
    with pytest.raises(dz.RecordError, match="not valid JSON"):
        RunRecord.load(tmp_path / "record.json")
