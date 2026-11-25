# dazflow: roadmap

Status: draft 2026-11-25. Implements [design.md](design.md).
Target: v0.1.0, the first PyPI release.
Tracking: [v0.1.0 (#1)](https://github.com/pedrobtz/py-dazflow/issues/1), with
one sub-issue per stage;
[PyPI-release (#19)](https://github.com/pedrobtz/py-dazflow/issues/19).

Adapted from the `pkg-roadmap` skill for R packages, as `py-dazdiff`'s
roadmap is: GitHub Actions running pytest replace r-actions, and a PyPI
release replaces the CRAN submission.

## Starting point

As of 2026-11-25 the repository has no commits. It holds
`.agents/design.md`, this roadmap, `CLAUDE.md` and an unexecuted
`user_guide/get-started.qmd`. No `pyproject.toml`, source, tests or CI.

## Scope of 0.1.0

One end-to-end path for a single-segment model: `ingest`, `clean`,
`explore`, `build`, `select`, `forecast`, `diagnose` and `report`, through
the library and the command line, with single-equation model forms, a run
record, replay and a Word document. Everything else is under
[After 0.1.0](#after-010).

## CI policy

- **`quick`**, under five minutes: `ruff check`, `ruff format --check`,
  and pytest in random order (pytest-randomly, seed printed) with
  coverage, on Ubuntu, Python 3.13, latest dependencies.
- **`full`**: pytest on Ubuntu and Windows × Python 3.12–3.14 and macOS ×
  3.12 and 3.14; a lowest-direct-dependencies job
  (`uv run --resolution lowest-direct`); a core-only job with no extras
  installed; build sdist and wheel, `twine check`, and `dazflow run` from
  the installed wheel on the example project.

The installed-wheel `dazflow run` needs the command line and the example
project, so Stage 6 adds it to the full profile's `package` job; until then
that job builds, runs `twine check` and checks the sdist (2026-11-25,
Stage 0).

Pull requests run `quick`; pushes to `main` and pull requests labelled
`full-ci` run `full`. A pull request's superseded runs are cancelled; a
push to `main` never is. Documentation-only pull requests (Markdown,
`.agents/`) skip the tests. A stage earns `full` only when the full set
checks something the stage introduces and later stages build on.

## Stages

| Stage | Title | CI | Issue | Pull request | Done |
| --- | --- | --- | --- | --- | --- |
| 0 | [Package identity](#stage-0--package-identity) | quick | [#2](https://github.com/pedrobtz/py-dazflow/issues/2) | [#20](https://github.com/pedrobtz/py-dazflow/pull/20) | 2026-11-25 |
| 1 | [Core objects](#stage-1--core-objects) | full | [#3](https://github.com/pedrobtz/py-dazflow/issues/3) | [#21](https://github.com/pedrobtz/py-dazflow/pull/21) | 2026-11-25 |
| 2 | [Artefacts and rendering](#stage-2--artefacts-and-rendering) | full | [#4](https://github.com/pedrobtz/py-dazflow/issues/4) | | |
| 3 | [Transforms, readers and ingest](#stage-3--transforms-readers-and-ingest) | quick | [#5](https://github.com/pedrobtz/py-dazflow/issues/5) | | |
| 4 | [Clean](#stage-4--clean) | quick | [#6](https://github.com/pedrobtz/py-dazflow/issues/6) | | |
| 5 | [Explore](#stage-5--explore) | quick | [#7](https://github.com/pedrobtz/py-dazflow/issues/7) | | |
| 6 | [Pipeline, record and command line](#stage-6--pipeline-record-and-command-line) | quick | [#8](https://github.com/pedrobtz/py-dazflow/issues/8) | | |
| 7 | [Model forms and evaluation](#stage-7--model-forms-and-evaluation) | full | [#9](https://github.com/pedrobtz/py-dazflow/issues/9) | | |
| 8 | [Build and select](#stage-8--build-and-select) | quick | [#10](https://github.com/pedrobtz/py-dazflow/issues/10) | | |
| 9 | [Forecast](#stage-9--forecast) | quick | [#11](https://github.com/pedrobtz/py-dazflow/issues/11) | | |
| 10 | [Diagnose](#stage-10--diagnose) | quick | [#12](https://github.com/pedrobtz/py-dazflow/issues/12) | | |
| 11 | [Report and replay](#stage-11--report-and-replay) | quick | [#13](https://github.com/pedrobtz/py-dazflow/issues/13) | | |
| 12 | [Documentation site](#stage-12--documentation-site) | quick | [#14](https://github.com/pedrobtz/py-dazflow/issues/14) | | |
| 13 | [Assurance](#stage-13--assurance) | quick | [#15](https://github.com/pedrobtz/py-dazflow/issues/15) | | |
| 14 | [Full CI](#stage-14--full-ci) | full | [#16](https://github.com/pedrobtz/py-dazflow/issues/16) | | |
| 15 | [v0.1.0 readiness](#stage-15--v010-readiness) | full | [#17](https://github.com/pedrobtz/py-dazflow/issues/17) | | |
| 16 | [Review](#stage-16--review) | full | [#18](https://github.com/pedrobtz/py-dazflow/issues/18) | | |

## Stage 0 — Package identity

**Issue.** [#2](https://github.com/pedrobtz/py-dazflow/issues/2), sub-issue of
[v0.1.0](https://github.com/pedrobtz/py-dazflow/issues/1).

**CI.** `quick`.

**Goal.** An installable, empty `dazflow` on `main`, licensed, linted and
checked by CI on every pull request.

**Work.**

- `pyproject.toml` (hatchling, `requires-python = ">=3.12"`, MIT, project
  URLs, classifiers, the core dependencies of design §12.1 and the extras
  of §12.2 as empty or real groups), `src/dazflow/__init__.py` with
  `__version__` from package metadata, `uv.lock`.
- `LICENSE` (MIT); `.agents/` excluded from the sdist.
- ruff configured in `pyproject.toml`; pytest with pytest-randomly and
  pytest-cov in the dev group.
- `.github/workflows/ci.yml` with both profiles, the `full-ci` label
  switch and a `concurrency` block; labels `full-ci`, `ci:docs` created.
- `CHANGELOG.md` with `## Unreleased`; `CLAUDE.md` commands section
  filled.

**Tests.**

- Package imports and `dazflow.__version__` equals the installed
  metadata version: `importlib.metadata`, in `tests/test_package.py`.

**Exit criteria.**

- The quick profile is green on the pull request in under five minutes.
- `uv build` produces an sdist without `.agents/`.
- The test above exists and was seen to fail before `__version__` was
  wired.

**Design.** [§12](design.md#12-packages), [§4.1](design.md#41-package-layout).

## Stage 1 — Core objects

**Issue.** [#3](https://github.com/pedrobtz/py-dazflow/issues/3), sub-issue of
[v0.1.0](https://github.com/pedrobtz/py-dazflow/issues/1).

**CI.** `full — the dataset fingerprint is the provenance and replay key
for every later stage; it must be identical across platforms, Python
versions and the lowest supported pandas and numpy`.

**Goal.** The objects every step shares exist and are tested on their
own: `Dataset`, configuration base, `Check`, `Gate`, `Finding`,
`StepResult`, `Registry` with entry points, and the error classes.

**Work.**

- `core/`: `Dataset` (frame, series metadata, scenarios, log,
  `fingerprint()`), immutable; `Config` base on pydantic with YAML
  loading and JSON Schema export; `Check`, `Gate` with overrides and
  reasons; `Finding` from jinja2 templates; `StepResult` with
  `to_dict()`; `DazflowError` and subclasses.
- `Registry`: decorator registration, `dazflow.plugins` entry-point
  loading on first use, lookup by key with a classed error naming the
  known keys.
- `dazflow.testing`: the start of the component contract suite (§13).

**Tests.**

- `Dataset` immutability: every method returns a new object and the
  caller's frame is unchanged, in `tests/core/test_dataset.py`.
- `fingerprint()`: a stored SHA-256 for a fixed fixture, the same on
  every full-profile leg; changes when one value, one metadata field or
  the column order changes, in `tests/core/test_fingerprint.py`.
- Gate arithmetic: a check exactly at, just below and just above its
  threshold, for `<`, `<=`, `>` and `>=` checks; an override without a
  reason is refused, in `tests/core/test_gate.py`.
- Config round trip: YAML → model → YAML is identical, and the exported
  JSON Schema validates the example configuration (jsonschema), in
  `tests/core/test_config.py`.
- Registry: a plugin installed from a test distribution's entry point is
  found by key; an unknown key raises the classed error listing known
  keys, in `tests/core/test_registry.py`.
- `StepResult.to_dict()` is JSON-serialisable and round-trips, in
  `tests/core/test_result.py`.

**Exit criteria.**

- Every `**Tests.**` line exists, compares against its named oracle and
  was seen to fail.
- The full profile is green on the pull request, with the same
  fingerprint on every leg.

**Design.** [§5](design.md#5-standard-objects),
[§6](design.md#6-extension), [§4](design.md#4-architecture).

## Stage 2 — Artefacts and rendering

**Issue.** [#4](https://github.com/pedrobtz/py-dazflow/issues/4), sub-issue of
[v0.1.0](https://github.com/pedrobtz/py-dazflow/issues/1).

**CI.** `full — rendered snapshots are compared on every platform, and
font and matplotlib differences across platforms must be settled before
every later stage adds snapshots of its own`.

**Goal.** `Table`, `Figure` and `Finding` render to Word, HTML, Markdown,
PNG and SVG in the house style, with stable identifiers and provenance.

**Work.**

- `render/`: `Table` on great-tables, `Figure` on matplotlib with a house
  `.mplstyle`, `Finding` on jinja2; `Section` and `Document`.
- Word tables and documents with python-docx (design open question 2:
  python-docx now; great-tables' export reconsidered once released).
- The style file: fonts, number and date formats, page width, reference
  `.docx`, finding templates; an organisation override loaded from the
  project.
- Pass or fail marking in tables; source note with run id, fingerprint
  and date.

**Tests.**

- Word output: the `.docx` read back with python-docx has the expected
  caption, column labels, formatted cells and pass/fail marks, in
  `tests/render/test_docx.py`.
- Table and finding text: syrupy snapshots of HTML and Markdown, in
  `tests/render/test_snapshots.py`.
- Figures: SVG snapshots with fonts embedded as paths and metadata
  stripped, so they match on every platform, in
  `tests/render/test_figures.py`.
- Identifiers: the same artefacts in a different creation order keep
  their `T`/`F` numbers, in `tests/render/test_ids.py`.
- Style override: one changed decimal setting changes every number in a
  rendered table and nothing else, in `tests/render/test_style.py`.

**Exit criteria.**

- Every `**Tests.**` line exists, compares against its named oracle and
  was seen to fail.
- The full profile is green on the pull request with the same snapshots
  on every leg.

**Design.** [§9](design.md#9-artefacts).

## Stage 3 — Transforms, readers and ingest

**Issue.** [#5](https://github.com/pedrobtz/py-dazflow/issues/5), sub-issue of
[v0.1.0](https://github.com/pedrobtz/py-dazflow/issues/1).

**CI.** `quick`.

**Goal.** Data comes in from CSV (and Excel and Parquet with their
extras), is put on one calendar, and can be transformed and transformed
back.

**Work.**

- `transforms/`: diff, seasonal diff, log, logdiff, growth, logit,
  Box-Cox, each with its inverse and parameters, registered.
- `readers/`: csv in core; excel (openpyxl) and parquet (pyarrow) behind
  their extras with a classed error naming the extra when it is missing;
  polars input through narwhals.
- `steps/ingest`: series inventory, frequency conversion, period-end or
  period-average, publication lags, minimum-history gate, history chart.

**Tests.**

- Round trips: every transform followed by its inverse returns the input
  within tolerance, as hypothesis property tests over finite series and
  the valid domain (logit on (0, 1)), in
  `tests/transforms/test_roundtrip.py`.
- Out-of-domain input (log of 0, logit of 1) raises the classed error
  naming the series and period, in `tests/transforms/test_domain.py`.
- Alignment: monthly to quarterly by period-end and by period-average,
  and a one-period publication lag, against hand-worked frames, in
  `tests/steps/test_ingest.py`.
- Gate: history exactly at, one below and one above the minimum length,
  in `tests/steps/test_ingest.py`.
- Missing extra: reading Excel without openpyxl raises the error naming
  `dazflow[io]`, by hiding the module, in `tests/readers/test_extras.py`.

**Exit criteria.**

- Every `**Tests.**` line exists, compares against its named oracle and
  was seen to fail.
- The quick profile is green on the pull request.

**Design.** [§7.1](design.md#71-ingest-import-and-align),
[§6](design.md#6-extension).

## Stage 4 — Clean

**Issue.** [#6](https://github.com/pedrobtz/py-dazflow/issues/6), sub-issue of
[v0.1.0](https://github.com/pedrobtz/py-dazflow/issues/1).

**CI.** `quick`.

**Goal.** Data-quality checks, configured treatments and a complete log
of what was changed.

**Work.**

- `checks/`: completeness, gaps, duplicates, range, revisions, outliers
  (STL residual based), representativeness (history covers a configured
  downturn).
- Treatments: fill, interpolate, winsorise, replace, exclude; every one
  logged with reason and rows.
- `steps/clean`: quality report, treatment table, history chart with
  flagged periods; checks on pointblank (design D12).

**Tests.**

- Each check at, just below and just above its threshold, in
  `tests/checks/test_thresholds.py`.
- Outlier detection: a synthetic seasonal series with three injected
  spikes at known periods; exactly those are flagged, in
  `tests/checks/test_outliers.py`.
- Treatment log: applying the logged treatments to the raw frame
  reproduces the cleaned frame, in `tests/steps/test_clean.py`.
- A treatment without a reason is refused by configuration validation, in
  `tests/steps/test_clean.py`.

**Exit criteria.**

- Every `**Tests.**` line exists, compares against its named oracle and
  was seen to fail.
- The quick profile is green on the pull request.

**Design.** [§7.2](design.md#72-clean-data-quality-and-treatment).

## Stage 5 — Explore

**Issue.** [#7](https://github.com/pedrobtz/py-dazflow/issues/7), sub-issue of
[v0.1.0](https://github.com/pedrobtz/py-dazflow/issues/1).

**CI.** `quick`.

**Goal.** Each series is profiled and tested for stationarity, and each
driver is screened against the target, with its expected sign checked.

**Work.**

- Series profile: trend and seasonal strength, autocorrelation,
  volatility, intermittency.
- Stationarity: ADF, DFGLS, Phillips-Perron, KPSS from arch, read
  together into an order of integration.
- Lag search 0 to *k*; single-driver OLS with HAC errors; expected-sign
  register and check.
- `steps/explore` with its tables, charts and finding.

**Tests.**

- Cross-implementation: ADF and KPSS statistics from the arch path equal
  statsmodels `adfuller` and `kpss` on the same series and lag choice, in
  `tests/steps/test_stationarity.py`.
- Known answers: a random walk is I(1), an AR(1) with φ = 0.5 is I(0), and
  a trend-stationary series is classed as such, over fixed seeds, in
  `tests/steps/test_stationarity.py`.
- Lag search: a driver built to lead the target by exactly 2 periods is
  found at lag 2, in `tests/steps/test_lags.py`.
- HAC regression: coefficients and standard errors equal a direct
  statsmodels `OLS(...).fit(cov_type="HAC")` call, in
  `tests/steps/test_screen.py`.
- Sign gate: a driver with the wrong observed sign fails the gate, and
  passes only with a reason recorded, in `tests/steps/test_screen.py`.

**Exit criteria.**

- Every `**Tests.**` line exists, compares against its named oracle and
  was seen to fail.
- The quick profile is green on the pull request.

**Design.** [§7.3](design.md#73-explore-each-series-alone-each-driver-against-the-target).

## Stage 6 — Pipeline, record and command line

**Issue.** [#8](https://github.com/pedrobtz/py-dazflow/issues/8), sub-issue of
[v0.1.0](https://github.com/pedrobtz/py-dazflow/issues/1).

**CI.** `quick`.

**Goal.** `ingest → clean → explore` runs end to end from a project
folder, through the library and the command line, and writes a run
record. Later stages plug into this.

**Work.**

- `pipeline/`: `Pipeline` from configuration, `Run`, `RunRecord`
  (configuration as run with defaults filled, results, checks, overrides,
  environment, input fingerprints).
- `project/`: `dazflow init` layout of design §10 with commented
  defaults; reading and writing the folder.
- `cli/` on typer: `init`, `validate`, one command per step built so far,
  `run --from/--to`, `show`, `plugins`; `--json`; exit codes 0, 1, 2.
- An example project under `examples/` used by the tests and the full
  profile's installed-wheel run.
- The full profile's `package` job: install the built wheel into a clean
  environment and run `dazflow run` on the example project (deferred here
  from Stage 0, 2026-11-25).

**Tests.**

- Exit codes: success 0, configuration error 1, failed gate 2, with
  typer's `CliRunner`, in `tests/cli/test_exit_codes.py`.
- `--json` output parses and equals `StepResult.to_dict()` from the
  library call on the same project, in `tests/cli/test_json.py`.
- Record completeness: the record's configuration, loaded back, validates
  against the step schemas and contains every default, in
  `tests/pipeline/test_record.py`.
- Pure steps: running a step through the library writes no file (a
  read-only temporary directory as working directory), in
  `tests/pipeline/test_pure.py`.
- `init`: the generated project passes `validate` unchanged, in
  `tests/cli/test_init.py`.

**Exit criteria.**

- Every `**Tests.**` line exists, compares against its named oracle and
  was seen to fail.
- The quick profile is green on the pull request.

**Design.** [§8](design.md#8-configuration),
[§10](design.md#10-project-folder-runs-and-versions),
[§11](design.md#11-command-line).

## Stage 7 — Model forms and evaluation

**Issue.** [#9](https://github.com/pedrobtz/py-dazflow/issues/9), sub-issue of
[v0.1.0](https://github.com/pedrobtz/py-dazflow/issues/1).

**CI.** `full — fitted values and forecasts from statsmodels must agree
across platforms and with the lowest supported statsmodels, scipy and
numpy, since selection, forecasting and diagnostics are all built on
them`.

**Goal.** Single-equation model forms and benchmarks behind one
interface, and rolling-origin evaluation shared by every later step.

**Work.**

- `models/`: the `Model` and `Fitted` protocols; naive, seasonal naive,
  historic mean, OLS, GLM (binomial for fractional targets), ARIMAX
  (SARIMAX), ETS; specification and parameters stored, never pickled;
  refit from a stored specification.
- `evaluation/`: expanding and rolling splitters, MAE, RMSE, MASE,
  interval coverage, rolling-origin evaluation.
- The contract suite in `dazflow.testing` run on every built-in form.

**Tests.**

- Wrapper equality: each form's parameters, residuals and forecasts equal
  the direct statsmodels call on the same data, in
  `tests/models/test_wrappers.py`.
- Known answers: OLS and ARIMAX recover known parameters from long
  simulated series within a tolerance set from the standard error, in
  `tests/models/test_recovery.py`.
- Refit: a model refit from its stored specification gives the same
  parameters to 1e-10, in `tests/models/test_refit.py`.
- Splitters: fold boundaries for a 20-period series equal hand-listed
  folds, in `tests/evaluation/test_splitters.py`.
- MASE on a hand-worked four-point example, in
  `tests/evaluation/test_metrics.py`.
- Contract suite: every built-in form passes it, and a deliberately broken
  form fails it, in `tests/models/test_contract.py`.

**Exit criteria.**

- Every `**Tests.**` line exists, compares against its named oracle and
  was seen to fail.
- The full profile is green on the pull request, including the
  lowest-direct-dependencies job.

**Design.** [§6](design.md#6-extension), [§5.3](design.md#53-fitted-models).

## Stage 8 — Build and select

**Issue.** [#10](https://github.com/pedrobtz/py-dazflow/issues/10), sub-issue of
[v0.1.0](https://github.com/pedrobtz/py-dazflow/issues/1).

**CI.** `quick`.

**Goal.** Candidate models are built within the declared signs and
ranked out of sample, and a default is proposed for the modeller to
confirm.

**Work.**

- `steps/build`: correlation and VIF, exhaustive and stepwise
  combination search up to a size, forms in increasing complexity,
  benchmarks always fitted, sign and significance gate.
- `steps/select`: rolling-origin scores scaled by the benchmark, model
  confidence set (arch `MCS`), scenario ordering and bounds checks, the
  default rule (design open question 6), confirmation or override in
  configuration with a reason.

**Tests.**

- Combination search: the number of candidates for *n* drivers and size
  *k* equals the binomial sum, and excluded pairs never appear, in
  `tests/steps/test_build.py`.
- VIF equals statsmodels `variance_inflation_factor`, in
  `tests/steps/test_build.py`.
- MCS: the confidence set equals a direct arch `MCS` call on the same
  losses, in `tests/steps/test_select.py`.
- Default rule: hand-made candidate tables where the lowest error wins,
  where a smaller model within tolerance wins, and where a failing model
  with the lowest error is skipped, in `tests/steps/test_select.py`.
- Scenario ordering check: projections in order, one pair swapped, and a
  tie, in `tests/steps/test_select.py`.

**Exit criteria.**

- Every `**Tests.**` line exists, compares against its named oracle and
  was seen to fail.
- The quick profile is green on the pull request.

**Design.** [§7.4](design.md#74-build-candidate-models),
[§7.6](design.md#76-select-rank-and-choose).

## Stage 9 — Forecast

**Issue.** [#11](https://github.com/pedrobtz/py-dazflow/issues/11), sub-issue of
[v0.1.0](https://github.com/pedrobtz/py-dazflow/issues/1).

**CI.** `quick`.

**Goal.** The chosen model forecasts under every scenario, anchored,
optionally mean-reverting, back on the target's scale, with intervals.

**Work.**

- `steps/forecast`: scenario forecasts with intervals; anchoring (carry,
  decay, none); mean reversion; back-transform with smearing for logs and
  simulation for logit; reconciliation and bounds gate; forecast table and
  fan chart.

**Tests.**

- Anchoring and mean reversion: hand-worked four-period paths for each
  rule, in `tests/steps/test_forecast.py`.
- Back-transform: on simulated log-normal data, the smearing-corrected
  mean is within its Monte Carlo error of the true mean, and the
  uncorrected one is not, in `tests/steps/test_backtransform.py`.
- Forecast equality: before anchoring, the mean path equals statsmodels
  `get_forecast` with the same driver paths, in
  `tests/steps/test_forecast.py`.
- Reconciliation gate: a jump-off gap at, below and above tolerance, in
  `tests/steps/test_forecast.py`.

**Exit criteria.**

- Every `**Tests.**` line exists, compares against its named oracle and
  was seen to fail.
- The quick profile is green on the pull request.

**Design.** [§7.7](design.md#77-forecast-projection-and-calibration).

## Stage 10 — Diagnose

**Issue.** [#12](https://github.com/pedrobtz/py-dazflow/issues/12), sub-issue of
[v0.1.0](https://github.com/pedrobtz/py-dazflow/issues/1).

**CI.** `quick`.

**Goal.** The chosen model is tested against its assumptions and against
history, with a pass-or-fail summary.

**Work.**

- Residual tests, stability (recursive residuals, CUSUM, rolling
  coefficients, Chow test), back-testing, one-at-a-time sensitivity,
  interval-based anomaly detection; `steps/diagnose` with its tables,
  charts and finding.

**Tests.**

- Residual tests equal the direct statsmodels calls (`acorr_ljungbox`,
  `het_breuschpagan`, `jarque_bera`, `linear_reset`), in
  `tests/steps/test_diagnose.py`.
- Chow test (own code): equals the F statistic from restricted and
  unrestricted statsmodels OLS fits, and detects a break built into a
  synthetic series at a known period, in `tests/steps/test_chow.py`.
- Sensitivity: for a linear model, the response to a unit shock equals
  the coefficient, in `tests/steps/test_sensitivity.py`.
- Anomalies: points injected outside the interval are flagged and none
  inside it, in `tests/steps/test_anomaly.py`.

**Exit criteria.**

- Every `**Tests.**` line exists, compares against its named oracle and
  was seen to fail.
- The quick profile is green on the pull request.

**Design.** [§7.8](design.md#78-diagnose-test-the-chosen-model).

## Stage 11 — Report and replay

**Issue.** [#13](https://github.com/pedrobtz/py-dazflow/issues/13), sub-issue of
[v0.1.0](https://github.com/pedrobtz/py-dazflow/issues/1).

**CI.** `quick`.

**Goal.** One command turns a run into a Word document, and any run can be
replayed from its record.

**Work.**

- `steps/report`: the document in pipeline order with summary, contents
  and configuration appendix; environment and input tables.
- Replay: from `record.json`, refit and rerun, compare every recorded
  number, and report differences; `dazflow replay`; the plain-Python
  replay script.
- `dazflow report` and `dazflow run` covering every 0.1.0 step.

**Tests.**

- End to end: the example project from `ingest` to `report`, replayed,
  matches its record exactly, in `tests/e2e/test_replay.py`.
- Replay detects change: altering one input value or one configuration
  default makes replay fail and name the difference, in
  `tests/e2e/test_replay.py`.
- Document: read back with python-docx, it contains every step's section
  in order and every artefact identifier the run produced, in
  `tests/e2e/test_document.py`.
- The generated replay script runs as a subprocess and exits 0, in
  `tests/e2e/test_replay_script.py`.

**Exit criteria.**

- Every `**Tests.**` line exists, compares against its named oracle and
  was seen to fail.
- The quick profile is green on the pull request.

**Design.** [§7.10](design.md#710-report-deliverables),
[§10.1](design.md#101-run-record).

## Stage 12 — Documentation site

**Issue.** [#14](https://github.com/pedrobtz/py-dazflow/issues/14), sub-issue of
[v0.1.0](https://github.com/pedrobtz/py-dazflow/issues/1).

**CI.** `quick`, plus the `Docs` workflow.

**Goal.** 0.1.0 links a documentation site from PyPI.

**Work.**

- `great-docs.yml`: home page from the README, API reference grouped by
  layer (core, components, steps, pipeline).
- `user_guide/`: Get started run for real with pasted output, Command
  line, Writing a plugin.
- `.github/workflows/docs.yml`: build on pull requests, deploy to GitHub
  Pages on pushes to `main`; `Documentation` in `project.urls`.

**Tests.**

- Guide code: every Python block in `user_guide/get-started.qmd` runs
  against the example project, in `tests/test_guide.py`.

**Exit criteria.**

- The `Docs` workflow is green on the pull request and on `main`.
- <https://pedrobtz.github.io/py-dazflow/> serves the site.

**Design.** [§1](design.md#1-purpose-and-scope).

## Stage 13 — Assurance

**Issue.** [#15](https://github.com/pedrobtz/py-dazflow/issues/15), sub-issue of
[v0.1.0](https://github.com/pedrobtz/py-dazflow/issues/1).

**CI.** `quick`.

**Goal.** No new features. The oracles of design §13 are all in place and
their gaps are filed.

**Work.**

- `.agents/assurance.md` from `templates/assurance.md` in
  `pedrobtz/packages`, adapted for Python: one row per boundary (readers,
  transforms, model wrappers, record, replay) with its oracle.
- Mutation testing with mutmut on `core/`, `transforms/`, `evaluation/`
  and the gate arithmetic, run on a schedule; survivors read.
- Hypothesis profiles for readers on malformed CSV.

**Tests.**

- Each survivor that marks a missing oracle gets a test or an issue.

**Exit criteria.**

- Every row of `.agents/assurance.md` filled.
- Every remaining gap either closed or moved under After 0.1.0 with a
  reason.

**Design.** [§13](design.md#13-testing-and-assurance).

## Stage 14 — Full CI

**Issue.** [#16](https://github.com/pedrobtz/py-dazflow/issues/16), sub-issue of
[v0.1.0](https://github.com/pedrobtz/py-dazflow/issues/1).

**CI.** `full`.

**Goal.** No new features. Everything the full profile finds on `main` is
fixed.

**Work.**

- Start from the full run on the last feature merge and the comments on
  this stage's issue.
- Fix every platform, Python-version, lowest-dependencies and core-only
  failure.
- Coverage floor set in CI at the level reached (at least 90%).

**Tests.**

- None new beyond the fixes' regressions.

**Exit criteria.**

- A full-profile run is green on the pull request and on the merge commit.

**Design.** [§12](design.md#12-packages).

## Stage 15 — v0.1.0 readiness

**Issue.** [#17](https://github.com/pedrobtz/py-dazflow/issues/17), sub-issue of
[v0.1.0](https://github.com/pedrobtz/py-dazflow/issues/1).

**CI.** `full`.

**Goal.** The package is ready to publish.

**Work.**

- `version = "0.1.0"`; `CHANGELOG.md` 0.1.0 section.
- `.github/workflows/release.yml`: build, then publish with PyPI trusted
  publishing on a `v*` tag, behind a `pypi` environment; a manual TestPyPI
  dispatch.
- A TestPyPI upload installed into a clean environment, running
  `dazflow init` and `dazflow run` on the new project.
- README: installation from PyPI, badges.

**Tests.**

- Installed-wheel smoke run, in the full profile.

**Exit criteria.**

- `twine check` passes on the sdist and wheel.
- The TestPyPI package installs and `dazflow --version` prints 0.1.0.
- A full-profile run is green on the merge commit.

**Design.** [§12](design.md#12-packages).

## Stage 16 — Review

**Issue.** [#18](https://github.com/pedrobtz/py-dazflow/issues/18), sub-issue of
[v0.1.0](https://github.com/pedrobtz/py-dazflow/issues/1).

**CI.** `full`.

**Goal.** No new features. The release candidate is reviewed twice.

**Work.**

1. Critical review of code, tests and style (`/code-review` and
   `review-testing`). Blocking and required findings become sub-issues of
   this issue, fixed in pull requests that close them.
2. Packaging review after those fixes: metadata, sdist and wheel
   contents, README rendering on PyPI, supported versions matching CI.

**Tests.**

- Each fix carries a regression test.

**Exit criteria.**

- Every blocking and required finding closed; packaging review clean;
  `CHANGELOG.md` reflects user-visible fixes.
- A full-profile run green on `main`, whose head becomes the release
  commit.

No pull request closes this issue; it closes when both reviews are done.

**Design.** [§14](design.md#14-decisions).

## After 0.1.0

- [§7.5](design.md#75-segment-optional-one-model-or-several) `segment`
  and the `panel` extra: the largest single addition.
- [§7.9](design.md#79-adjust-optional-adjustments-and-margins) `adjust`:
  optional step; the 0.1.0 path is complete without it.
- [§10.2](design.md#102-versions) model versions and the registry: a run
  record and replay cover reproducibility for 0.1.0.
- [§7.10](design.md#710-report-deliverables) scoring code.
- [§7.4](design.md#74-build-candidate-models) systems (VAR, VECM) and
  cointegration: open question 3.
- [§12.2](design.md#122-extras) extras beyond `io`: `fast`, `ml`,
  `breaks`, `sensitivity`, `cube`, `fetch`, `excel`, `credit`.
- [§12.3](design.md#123-candidate-plugin-backends) plugin backends, each a
  separate package.
- [§16](design.md#16-later) MCP server, forms interface, agent layer.

## Blocking open questions

- [§17](design.md#17-open-questions) question 6, the exact default
  selection rule, blocks Stage 8.
- The PyPI release needs a trusted publisher on PyPI (repository
  `pedrobtz/py-dazflow`, workflow `release.yml`, environment `pypi`).
