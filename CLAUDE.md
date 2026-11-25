# CLAUDE.md

<!-- markdownlint-disable-next-line MD013 -->
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`dazflow` is a Python library and command-line tool for the workflow
around a time-series model: import, clean, explore, build candidates,
select, forecast, diagnose, adjust and report. Each step returns its
numbers together with tables, figures and findings ready for a model
document. The statistics come from statsmodels, arch, scipy and the other
packages in the design's §12. `dazflow` adds the shared data object,
configuration, gates, run record and house-style rendering. It does not
generate scenarios, ship deep-learning models in core, choose the final
model on its own, or contain a language model.

Like `py-dazdiff`, this is a Python package, so the r-skills shared rules
(R package, CRAN, testthat) do not apply and are not imported.

## Where things are decided

- `.agents/design.md` is the single source of truth for what the package
  is and why: scope, steps, extension interfaces, packages, decisions
  (§14) and open questions (§17). A decision changes the design, with a
  dated one-line reason. It was adopted from RFC 0079 in
  `pedrobtz/packages`, which becomes a pointer here.
- `.agents/roadmap.md` owns the order of work: stages, exit criteria, CI
  tiers and issue links. Each stage is a sub-issue of `v0.1.0`, closed by
  one pull request titled like the stage, with `Closes #<n>`.

## Current state

Empty repository as of 2026-11-25: the design, this file and an
unexecuted Get started page.

2026-11-25, Stage 0: an installable, empty `dazflow` (`0.1.0.dev0`) with
`pyproject.toml`, `uv.lock`, one test and the CI workflow. No library code
yet.

2026-11-25, Stage 1: `src/dazflow/core/` (Dataset, Config, Check, Gate,
Override, Finding, StepResult, Registry, errors) and `dazflow.testing`.

2026-11-25, Stage 2: `core/artefact.py` (Table, Figure, Source,
ArtefactIds) and `src/dazflow/render/` (style, renderers, Section,
Document).

2026-11-25, Stage 3: `src/dazflow/transforms/`, `src/dazflow/readers/` and
`src/dazflow/steps/` with `ingest`.

2026-11-25, Stage 4: `src/dazflow/checks/` (seven checks on pointblank)
and `steps/clean.py`.

2026-11-25, Stage 5: `steps/explore.py` (profile, unit-root tests read
into an order of integration, lag search, HAC screen, sign gate).

2026-11-25, Stage 6: `src/dazflow/pipeline/`, `src/dazflow/project/`,
`src/dazflow/cli/` (the `dazflow` command) and `examples/loss-rate`.

2026-11-25, Stage 7: `src/dazflow/models/` (seven forms, `Forecast`,
`ModelRecord`, `refit`), `src/dazflow/evaluation/` (splitters, metrics,
rolling-origin `evaluate`) and `dazflow.testing.ModelContract`.

2026-11-25, Stage 8: `steps/build.py`, `steps/select.py`,
`models/_design.py` (`Design`), the build and select adapters, and
`project.yaml: seed` passed to the steps. The default pipeline ends at
select.

2026-11-25, Stage 9: `steps/forecast.py`, `readers/_scenarios.py` (the
scenario reader select and forecast share) and the forecast adapter. The
default pipeline ends at forecast.

## Versions and dependents

`v0.1.0` issue: see the roadmap header. Dependents: none.

## Commands

```sh
uv sync --all-extras          # install, with the dev group and every extra
uv run pytest                 # tests, in random order (seed in the header)
uv run pytest -p no:randomly  # tests in file order
uv run pytest --randomly-seed=<seed>   # replay a CI order
uv run pytest --cov=dazflow --cov-report=term-missing   # as CI runs them
uv run ruff check .           # lint
uv run ruff format .          # format (CI runs `ruff format --check .`)
uv build                      # sdist and wheel into dist/
uv run twine check --strict dist/*
uv add '<pkg>>=<lowest>'      # a core dependency; then commit uv.lock
uv add --optional <extra> '<pkg>>=<lowest>'   # a package in an extra
uv add --dev '<pkg>>=<lowest>'                # a dev-group tool
uv run --resolution lowest-direct --isolated --all-extras --python 3.12 pytest
```

A new dependency's lower bound is the oldest release that installs on
Python 3.12 and passes the lowest-direct run above (design D15).
`uv.lock` is committed and CI runs with `--frozen`, so relock (`uv lock`)
whenever `pyproject.toml` changes.

CI (`.github/workflows/ci.yml`): pull requests run `quick` (lint, then
pytest with every extra and coverage on Ubuntu / 3.13); pushes to `main`,
manual runs and pull requests labelled `full-ci` run `full` (the matrix,
lowest direct dependencies, core only, package). Its `changes` job skips
every other job when a pull request touches only `*.md` files or
`.agents/`.

## Architecture

To be written as stages land; see `.agents/design.md` §4 for the layers
(core, components, steps, pipeline, front ends) and the package layout.

- `core/_model.py`: `DazflowModel`, the frozen pydantic base that turns
  `ValidationError` into the model's `DazflowError` subclass. New core
  objects subclass it and set `_error` and `_hint`.
- `core/artefact.py`: `Table`, `Figure` and `Source` are data, with no
  third-party object (D23); `ArtefactIds` numbers a step's artefacts by
  declaration (D21). Never renumber a declared artefact.
- `render/`: `style.py` (`Style`, `load_style`), `format.py` (every
  renderer writes cell, finding, caption and source text through it),
  `_tables.py` (great-tables HTML, Markdown), `_figures.py` (matplotlib,
  inside `matplotlib.style.context`; SVG settings fixed in code),
  `_docx.py` (python-docx), `renderers.py` (the `renderers` registry,
  `render`, `write`). `dz.render` is imported lazily.
- `transforms/`: `_base.py` (protocol, registry, text form, `forward`,
  `inverse`, `apply`), `_builtin.py` (the seven transforms). A chain is
  recorded as resolved text in `SeriesMeta.transforms`; inverting needs
  that text and the untransformed history (D27).
- `readers/`: `_base.py` (`ReadSpec`, `read`, `from_frame`, the date to
  period rules, `require` for optional imports), `_builtin.py` (csv,
  excel, parquet). Import an extra's package inside `read`, never at
  module level, or `import dazflow` fails in the core job.
- `steps/`: `registry` and one module per step with `Config`, `Result`,
  `run` and its `IDS` (`ArtefactIds`). Steps import components, never
  each other. A result's `dataset` is excluded from `to_dict()` (D26).
- `checks/`: `_base.py` (`QualityCheck` protocol, `CheckConfig`
  thresholds, `Context`, `Units`, `Outcome`, `evaluate`, which builds one
  pointblank validation per check and series), `_builtin.py` (the seven
  checks and `outlier_scores`). Import pointblank and statsmodels inside
  functions: `import dazflow` must stay light. A check never changes the
  data; `steps/clean.py` holds the treatments and their log.
- `steps/explore.py`: the statistics are public functions in the step
  module (`profile`, `unit_root`, `integration_order`, `lag_search`,
  `screen`), with the readings `says_stationary` and `driver_checks` kept
  pure so thresholds are tested on one number: reruns of arch and
  statsmodels differ in the last bits. Import arch and statsmodels inside
  functions. Catch arch's `InfeasibleTestException` by name (it is not a
  `ValueError` in arch 7.0), never a bare `Exception`: under pytest's
  `-W error` a warning would turn into a missing test.
- `pipeline/`: `_adapters.py` (`StepInputs`, the `adapters` registry,
  `default_adapter`, the ingest and clean adapters, `InputRecord`),
  `_pipeline.py` (`Pipeline`, `Run`), `_record.py` (`RunRecord`, schema
  version 1, `environment()`). A new step joins by being in
  `dazflow.steps.registry` and, if `run(dataset, config)` does not fit,
  registering an adapter with `@register_adapter("<step>")` (D32). The
  pipeline and `Project.run()` write nothing; only `Project.write()` does.
- `project/`: `_folder.py` (`ProjectConfig`, `Project`, `init`, `load`),
  `_datasets.py` (dataset CSV plus `<step>.schema.json`, fingerprint
  checked on read), `_commented.py` (`commented_yaml`), `example/series.csv`
  (the data `init` copies). `examples/loss-rate` must equal what `init`
  writes (`tests/cli/test_init.py`): after changing a default or a field
  description, regenerate it with `rm -rf examples/loss-rate && uv run
  dazflow init examples/loss-rate`.
- `cli/`: typer, built on import; each registered step gets a command.
  Commands catch `DazflowError` (message to stderr, exit 1); a failed gate
  is exit 2.
- `models/`: `_base.py` (`Model`/`Fitted` protocols, `registry`,
  `register`, `ModelForm` and `FittedModel` base classes, `Forecast`,
  `ModelSpec`, `ModelRecord`, `refit`), `_builtin.py` (the seven forms).
  Import statsmodels and scipy inside `_fit`/`_predict`. A form's `drivers`
  attribute says whether it takes `X`. Residuals are `y - fitted` for every
  form. Optimised forms pass `start_params=_round_start(model.start_params)`
  (D36), or a refit is not repeatable to 1e-10. A new form's test belongs
  in `tests/models/test_wrappers.py` (equality with the direct call) and in
  `BUILT_INS` of `tests/models/test_contract.py`.
- `evaluation/`: `_base.py` (splitters, metrics, `Fold`), `_rolling.py`
  (`evaluate`, `Evaluation`). Shared by select and segment; steps import
  it, never each other.
- `models/_design.py`: `Design` builds a candidate's target and drivers
  from the series in levels (chains, lags, `trend`, sample). Build fits
  through `Design.frames` and select evaluates and forecasts through the
  same function and `Design.future`; never build the matrices another way,
  or `models.refit`'s fingerprint check fails.
- `steps/build.py` and `steps/select.py` take the series in levels (the
  dataset explore received), not explore's transformed dataset; their
  adapters find it by fingerprint (`StepInputs.dataset_with`). Neither
  returns a dataset, so the latest dataset stays explore's. A candidate's
  id is `<form>:<driver>+<driver>` (the form key for a benchmark), and its
  checks are overridden as `<candidate>/<check>`. Select's numbers repeat
  to about 1e-15 between runs (statsmodels' intervals move by an ulp), its
  choice and confidence set exactly: compare results with a tolerance.
- `steps/forecast.py` takes the levels and select's result; its scenarios
  are select's (`dataset.scenarios`, else `select.config.scenarios`, read
  by `readers.read_scenarios`; the adapter reads the file from the project
  folder). Anchoring is on the model's scale, reversion on the target's;
  `back_transform` picks the correction by the chain (D41). Simulated
  corrections use `numpy.random.default_rng(seed)`, the same draws for
  every scenario; keep test draws small, as a chain with a nonlinear lag
  inverts each path separately.
- `tests/models/` has an `__init__.py` because `tests/checks/` already has
  a `test_contract.py`: without it the two basenames collide.
- Never run `uv run --resolution lowest-direct` without restoring
  `uv.lock` afterwards: it rewrites the lock with the lowest versions.
- Tests that need an extra carry `@pytest.mark.extra("<distribution>")`
  (`tests/conftest.py`), which skips only when the distribution is not
  installed, so a broken import fails instead of skipping.
- `core/dataset.py`: `Dataset` is a plain class (pandas frames are not
  pydantic fields); the fingerprint's canonical form is design §5.1.1.
  Changing what it hashes means a new format and new stored hashes.

## Invariants that are easy to break

All are fixed by the design; each breaks a promise the package makes.

- **Steps are pure.** A step's `run()` prints nothing and writes no
  files. Otherwise steps cannot run in memory, and the command line and
  a later MCP server would show different behaviour.
- **Steps never import each other.** Shared logic goes into a component
  (`evaluation/`, `models/`, `transforms/`). One step import makes the
  pipeline order fixed and steps no longer usable alone.
- **No third-party object crosses a step boundary.** Only `Dataset` and
  `StepResult` types pass between steps. A leaked statsmodels result ties
  every later step to that library.
- **Every threshold is configuration.** A constant in code cannot be
  overridden, recorded or replayed.
- **Fitted models are never pickled.** Store specification, parameters,
  covariance and the data fingerprint; replay and version loading refit
  and compare.
- **Findings are templates filled with numbers.** No generated text, and
  no language model anywhere in the library.
- **Artefact identifiers are stable.** `T<step>.<n>` and `F<step>.<n>`
  must survive reruns; renumbering breaks cross-references in documents
  already written.

## Testing

The design's §13 lists the oracles each result needs, and a public
contract suite, `dazflow.testing`, that every component passes. Tests live
under `tests/<package>/`. `tests/core/test_fingerprint.py` holds stored
SHA-256 values that every full-profile leg must reproduce.

Rendering snapshots live in `tests/render/__snapshots__/` and must match
on every full-profile leg, lowest-direct included. Markdown is compared
exactly, HTML as a normalised tree and SVG with a tolerance
(`tests/render/snapshot_helpers.py`, D25). Regenerate with
`uv run pytest tests/render --snapshot-update`, then check the
lowest-direct run still passes before committing.

## Continuous integration

See "Commands" above. The full profile's `package` job also installs the
built wheel into a clean environment and runs `dazflow validate` and
`dazflow run` on a copy of `examples/loss-rate`; a non-zero exit or a
record whose status is not `passed` fails the job.
