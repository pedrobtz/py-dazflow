# dazflow: design

Adopted from
[RFC 0079](https://github.com/pedrobtz/packages/blob/9f975a4efc7291db259fa1e1374b9752fbb3f92b/rfcs/not-started/0079-dazflow-stress-models-from-configuration.md)
on 2026-11-25. This file is the single source of truth for the design;
the RFC is now a pointer here.

Status: draft 2026-11-25. Derived from RFC 0079 in `pedrobtz/packages`,
widened from stress models to time-series modelling in general. Scenario
projection, the stress-testing use, is one application among several.
Nothing has been built. The order of work is in
[roadmap.md](roadmap.md).

## 1 Purpose and scope

`dazflow` is a Python library and command-line tool for the workflow
around a time-series model: import the data, clean it, explore each
series, build candidate models, select one, forecast with it, test it,
adjust it and report on it. Every step returns its numbers together with
tables, figures and short findings that go into a model document
without reformatting.

The statistics come from established packages: statsmodels, arch, scipy
and others. `dazflow` adds what they do not have. It gives them one data
object and one configuration format, puts the steps in order, records
what was run, and renders the output in one house style.

```python
import dazflow as dz

cfg = dz.load_config("project.yaml")
ds = dz.ingest.run(cfg.ingest)
ds = dz.clean.run(ds, cfg.clean).dataset
ex = dz.explore.run(ds, cfg.explore)
cand = dz.build.run(ds, cfg.build, explore=ex)
sel = dz.select.run(ds, cfg.select, candidates=cand)

sel.gate.passed                         # every check within its threshold
sel.artefacts["T5.1"].to_docx("ranking.docx")
```

```text
dazflow init my-model
dazflow run --to select
dazflow report
```

### 1.1 Applications

| Application | Target | Typical transformation |
| --- | --- | --- |
| Demand forecasting | sales, calls, volumes | log, seasonal differences |
| Balance-sheet projection | deposit or loan balance | log-differences |
| Credit stress testing | default or loss rate | logit, then differences |
| Behaviour | prepayment or churn rate | logit |
| Operational | event counts | log, or a count model |

Applications differ in their transformations, model forms and checks,
not in their steps. A domain adds a plugin package (§6), not a fork.

### 1.2 Out of scope

- Generating scenarios. Scenarios and future driver paths are inputs.
- Deep learning and foundation models in the core. A plugin can add
  them through the model interface.
- Automatic final choice. The library proposes a default and the
  modeller confirms or overrides it in configuration.
- Any language model inside the library. An agent or MCP layer may come
  later, on top of the public API (§16).
- Any regulatory framework. An organisation maps its own onto the steps
  through configuration and templates.

## 2 Goals

1. **Simple.** One call or one command per step. Sensible defaults. A
   configuration file instead of glue code.
2. **Modular.** Each step is usable alone, on objects in memory, without
   a project folder or the other steps.
3. **Extensible.** New readers, checks, transformations, model forms,
   metrics, renderers and steps are added by registering a class, from
   user code or a separate package, with no change to `dazflow`.
4. **Documentation-ready output.** Every table, figure and finding is
   numbered, captioned and styled when it is made.
5. **Reproducible and deterministic.** Same inputs, configuration and
   seed give the same numbers, and a run can be replayed from its record.

## 3 Concepts

| Term | Meaning |
| --- | --- |
| Series | one variable over time, with its frequency, unit and role |
| Target | the series modelled, possibly one per segment |
| Driver | a series that may explain the target (exogenous regressor) |
| Segment | a part of the data that may need its own model |
| Scenario | a future path for every driver over the horizon |
| Origin | the last observed period, where a forecast starts |
| Benchmark | a naive model every candidate must beat |
| Component | a pluggable piece, such as a reader, check or model form |
| Step | one stage of the workflow, `run(inputs, config) -> StepResult` |
| Check | one test with a threshold and an observed value, passing or failing |
| Gate | the checks that end a step |
| Finding | one or two template sentences stating a step's result |
| Artefact | a table, figure or finding produced by a step |
| Run | one execution of some or all steps, with its record |
| Version | a run frozen in the model registry |

## 4 Architecture

Four layers. Each layer imports only the layers below it.

```text
 front ends     cli (typer)          later: MCP server, forms UI
 ─────────────────────────────────────────────────────────────────
 pipeline       Pipeline: ordered steps from configuration, Run, Record
 ─────────────────────────────────────────────────────────────────
 steps          ingest  clean  explore  build  segment  select
                forecast  diagnose  adjust  report
 ─────────────────────────────────────────────────────────────────
 components     readers  checks  transforms  models  metrics
                splitters  renderers                   (registries)
 ─────────────────────────────────────────────────────────────────
 core           Dataset  Config  StepResult  Check/Gate  Artefact
                Registry  errors
```

Rules:

- **Library first.** All statistics and all rendering are public
  functions. The command line only calls them.
- **Pure steps.** A step takes data and configuration and returns a
  `StepResult`. It prints nothing and writes no files. The pipeline or
  the caller decides where results go.
- **Steps never import each other.** A later step receives an earlier
  step's result as an argument. Shared logic, such as rolling-origin
  evaluation used by `segment` and `select`, lives in a component.
- **No third-party object crosses a step boundary.** Inside a step, a
  component converts the `Dataset` to the library's format, calls it and
  converts back. A library can then be swapped by changing one
  component.
- **Configuration is the interface.** Every step and component has a
  pydantic model. Schemas export to JSON Schema, which is what a later
  form or agent fills in.
- **Classed errors.** Every error is a subclass of `DazflowError` and its
  message says what went wrong, the observed state and what to change.

### 4.1 Package layout

```text
src/dazflow/
  core/          Dataset, Config base, StepResult, Check, Gate, Finding,
                 Artefact, Registry, errors
  readers/       CSV, Excel, Parquet, SQL; optional SDMX and FRED
  checks/        data-quality and statistical checks
  transforms/    differences, logs, logit, Box-Cox, ... each with inverse
  models/        model forms and benchmarks behind one interface
  evaluation/    splitters, metrics, rolling-origin evaluation
  render/        tables, figures, findings, documents, house style
  steps/         one module per step (§7)
  pipeline/      Pipeline, Run, RunRecord, replay
  project/       project folder, model registry and versions (§10)
  cli/           command-line interface
```

`import dazflow as dz` exposes the steps as `dz.ingest`, `dz.clean` and
so on, and the components as `dz.models`, `dz.transforms`, ...

## 5 Standard objects

### 5.1 Dataset

The one object that passes between steps. It is immutable: a step that
changes data returns a new `Dataset`.

- `frame`: a pandas DataFrame indexed by a `PeriodIndex`, or by a
  `(segment, period)` MultiIndex for panels, with one column per series.
- `series`: metadata per column: role (`target`, `driver`, `other`),
  unit, frequency, source, the transformation chain applied so far, and
  the expected sign of each driver.
- `scenarios`: optional, a frame indexed by `(scenario, period)` with a
  severity order.
- `log`: every change made to the data (exclusion, fill, outlier
  treatment, alignment), with its reason and the rows affected.
- `fingerprint()`: a SHA-256 of the frame and metadata, for provenance.
  Its canonical form is in §5.1.1 (D16).

Validation (2026-11-25, Stage 1): column names are non-empty unique
strings; values are float, integer, boolean or string (anything else is
refused, so the canonical form covers every column); the index is unique
and sorted; segment and scenario labels are strings or integers. Index
levels are renamed `period`, `(segment, period)` and `(scenario, period)`.
Scenario columns are dataset columns, at the dataset's frequency, and
`severity` lists every scenario label once, mildest first (default: order
of first appearance). Immutability is by copying: the constructor copies
the frames, `frame` and `scenarios` return copies, and the `with_frame`,
`with_series`, `with_scenarios`, `with_log` and `select` methods return
new datasets (pandas 2.2 has no copy-on-write by default, so a shared
view would leak changes).

#### 5.1.1 Fingerprint canonical form

Decided 2026-11-25 (Stage 1, D16). `fingerprint()` is the lowercase hex
SHA-256 of one JSON document, serialised with sorted keys, separators
`,` and `:`, ASCII only (`\uXXXX` escapes, as Python's `json` writes
them), no NaN or infinity literals, and encoded as ASCII bytes. The
document is, for format `dazflow.dataset/1`:

```text
{"columns":   [column, ...]            in frame column order
 "format":    "dazflow.dataset/1",
 "index":     {"kind": "period" | "panel", "rows": [row, ...]},
 "scenarios": null | {"columns": [column without meta, ...],
                      "rows": [[label, period], ...],
                      "severity": [label, ...]}}
column = {"kind": "float" | "int" | "bool" | "string",
          "meta": {"expected_sign", "frequency", "role", "source",
                   "transforms": [...], "unit"},   (null when unset)
          "name": str, "values": [value, ...]}
row    = period                 (kind "period")
       | [segment, period]      (kind "panel")
period = "<start>/<start of next period>", each
         YYYY-MM-DDTHH:MM:SS.nnnnnnnnn written from the timestamp fields
value  = null for a missing value (NaN, None, pd.NA), else by kind:
         float: 16 lowercase hex digits of the IEEE-754 binary64 value,
                big-endian, with -0.0 written as 0.0
         int: a JSON integer; bool: true/false; string: a JSON string
label  = a JSON string or integer
```

Every byte is written by `dazflow` from values, never from a library's
text form: no pickle, no `hash_pandas_object`, no float `repr`, no
frequency alias (`Q-DEC` and `A-DEC`/`Y-DEC` spellings differ across
pandas versions; a period's start and end dates do not). Row and column
order count; index names and dtypes within a kind (object strings and
`StringDtype`, `int64` and `Int64`) do not. The log is left out: two
datasets with the same data, metadata and scenarios have the same
fingerprint however they were reached, and the log is recorded in the run
record beside it. A change to this form is a new `format` value and new
stored hashes in `tests/core/test_fingerprint.py`, never an edit of the
old ones.

pandas is the internal format because statsmodels, arch and most of the
ecosystem take it. Readers accept polars and pyarrow input through
narwhals and convert at ingest.

### 5.2 StepResult

Each step returns a subclass with typed fields for its values, plus the
shared fields:

- `gate`: a `Gate` of `Check`s, each with name, threshold, observed
  value, pass or fail, and any override with its recorded reason.
- `finding`: the step's plain-language summary.
- `artefacts`: a mapping from stable identifier (`T3.2`, `F6.1`) to
  `Table`, `Figure` or `Finding`.
- `config`: the configuration as run, defaults filled in.
- `to_dict()`: a JSON-safe form for the run record and `--json`.

Settled 2026-11-25 (Stage 1, D17). A `Check` passes when
`observed <op> threshold`, `op` one of `<`, `<=`, `>`, `>=`; the threshold
is finite, and a NaN observed value fails. Its `status` is `pass`, `fail`
or `overridden` (failed, with an `Override(check=..., reason=...)` whose
reason is not blank). `Gate.evaluate(checks, overrides)` attaches overrides
by check name and refuses one naming no check; an override of a passing
check is kept and changes nothing. A `Finding` is a jinja2 template
(sandboxed, undefined fields are errors) and `values` that are booleans,
integers, finite floats or strings; its `text` is rendered when it is made.
`to_dict()` is the pydantic JSON dump with non-finite floats as the
strings `"NaN"`, `"Infinity"`, `"-Infinity"` (strict JSON), and
`from_dict()` on the same subclass rebuilds an equal result.
`artefacts` holds `Table`, `Figure` and `Finding`, keyed by identifier,
sorted tables first, then figures, then findings, each by step and number;
a key must be the artefact's own `id`, and a finding's key has the form
`S<step>.<n>` (2026-11-25, Stage 2, D21).
A step that returns data holds its `Dataset` in a `dataset` field that
`to_dict()` leaves out, with the dataset's `fingerprint` beside it;
`from_dict()` rebuilds the rest with `dataset=None` (2026-11-25, Stage 3,
D26: a frame is data, not a record field; the run record stores it).

### 5.3 Fitted models

A fitted model is stored as its specification, parameters, covariance
and the fingerprint of the data it was fitted on. It is never pickled.
Replay and version loading refit and compare (§10).

Settled 2026-11-25 (Stage 7, D36). The stored form is
`dazflow.models.ModelRecord`, from `fitted.record()`, written by
`to_dict()` as strict JSON (non-finite floats as `"NaN"`, `"Infinity"`):

```json
{
  "kind": "dazflow.model",
  "format": 1,
  "spec": {"form": "arimax",
           "config": {"order": [1, 0, 0], "seasonal_order": [0, 0, 0, 0],
                      "trend": "c", "maxiter": 200},
           "target": "loss", "drivers": ["gdp", "rate"]},
  "sample": {"start": "2000Q1", "end": "2019Q4", "observations": 80,
             "frequency": "Q-DEC"},
  "data": "<Dataset fingerprint of the target and drivers, §5.1.1>",
  "params": {"intercept": 0.12, "gdp": 0.03, "rate": -0.02,
             "ar.L1": 0.6, "sigma2": 0.0004},
  "cov_params": [[...], ...]
}
```

- `spec` is the `ModelSpec`: the registry key of the form, its `Config`
  with every default filled in (`model_dump(mode="json")`, so tuples are
  lists), the target's name and the drivers in fitting order. It is all a
  refit needs besides the data.
- `params` keeps the form's own order and names (statsmodels' names for
  the wrapped forms); `cov_params` is the covariance in that order, row by
  row.
- `data` is the fingerprint of `Dataset(frame)` where `frame` is the
  target followed by the drivers, so the format needs no new canonical
  form; `sample` repeats the periods for messages.
- `models.refit(record, y, X)` refuses data with another fingerprint
  (`ModelError`), builds the form from `spec` and fits again;
  `record.difference(fitted)` is the largest absolute parameter change, and
  `refit(..., tolerance=t)` refuses one above `t`. A record of another
  `format` is refused; a field that changes meaning raises the format.
- Refitting the same data on the same machine gives the same parameters
  to 1e-10 for every built-in form (`tests/models/test_refit.py`), because
  the optimised forms start from statsmodels' start values rounded to 10
  significant digits (D36). Across machines (Stage 11 replay, §10.2
  versions) the closed-form forms (the benchmarks, OLS, GLM) still agree
  to about 1e-10, while the optimised forms (ARIMAX, ETS) agree only to
  the optimiser's convergence: a one-ulp change in the start moved a
  seasonal ARIMAX up to 7e-5 on the test data. Replay compares those
  against a tolerance set from the standard errors, which Stage 11 fixes.

## 6 Extension

Every component kind is a `typing.Protocol` with a registry. Built-ins
register themselves. Users register theirs with a decorator, and
third-party packages through the `dazflow.plugins` entry-point group,
which is loaded on first use. Each entry point names a module
(`name = package.module`, or `package.module:attr`); loading imports it,
and its decorators register its components. Each registry loads the group
on its first lookup; a plugin that fails to import raises `PluginError`
naming the distribution, at every lookup until it is fixed (2026-11-25,
Stage 1, D18: one group for every kind, so a plugin registers in several
registries from one import). Configuration names a component by its
registry key, so a plugin model is chosen in YAML exactly like a
built-in one.

| Kind | Protocol (essential methods) | Built-ins |
| --- | --- | --- |
| Reader | `read(spec) -> Dataset` | csv, excel, parquet; sql later |
| Check | `units(x, config, context) -> Units`, evaluated by pointblank into an `Outcome` with its `Check` | completeness, gaps, duplicates, range, revisions, outliers, representativeness |
| Transform | `resolve`, `forward`, `inverse`, `consumed` | diff, seasonal_diff, log, logdiff, growth, logit, boxcox |
| Model | `fit(y, X) -> Fitted` | naive, mean, OLS, GLM, ARIMAX, ETS |
| Metric | `score(actual, forecast)` | MAE, RMSE, MASE, coverage |
| Splitter | `split(index) -> folds` | expanding, rolling, holdout |
| Renderer | `render(artefact, path)` | docx, html, markdown, png, xlsx |
| Step | `run(inputs, config) -> StepResult` | the ten steps of §7 |

Transforms and readers, settled 2026-11-25 (Stage 3, D27):

- **Transforms** (`dz.transforms`, keys as in the table) are classes with
  a `Config` of parameters. Configuration names one by its text form,
  `key` or `key(param=value, ...)` (`"diff(lag=4)"`); `forward(x, chain)`
  applies a chain in order and returns the chain with every parameter
  resolved (a seasonal period from the frequency, 12 monthly and 4
  quarterly; a Box-Cox `lmbda` by maximum likelihood within
  `[-bound, bound]`, default 2), and that resolved text is what
  `SeriesMeta.transforms` records (`apply(dataset, column, chain)`). The
  inverse takes the resolved chain and the untransformed `history`:
  `inverse(y, chain, history)` rebuilds levels period by period, a
  lagged value coming from `history` where it has one, so the same call
  undoes an in-sample series and extends a forecast. Missing values stay
  missing; a value outside a domain (log, logdiff and Box-Cox need
  x > 0, logit 0 < x < 1, growth a non-zero earlier value, all finite
  values) raises `TransformError` with the series and the first period.
- **Readers** (`dz.readers`) take a `ReadSpec` (`reader`, `path`,
  `date`, `frequency`, `columns`, `options` for the reader's own
  `Config`) and return a `Dataset` at its own frequency, monthly,
  quarterly or annual. The reader is named or read from the suffix
  (`.csv`; `.xlsx`, `.xlsm`; `.parquet`, `.pq`). Dates are periods,
  datetimes, years or text (`2024-03-31`, `2024-03`, `2024Q1`, `2024`);
  with no `frequency`, the finest of M, Q and Y on which the dates fall in
  distinct, mostly consecutive periods. Daily and weekly data are not read
  in 0.1.0. Excel and Parquet import openpyxl and pyarrow when reading,
  and raise `MissingExtraError` naming `dazflow[io]` without them.
  `from_frame` takes pandas directly and polars or pyarrow through
  narwhals.

`Fitted` has `predict(h, X_future) -> Forecast`, `params`, `cov_params`,
`residuals` and `spec`. Seasonal naive, VAR and VECM are model forms too.
`Forecast` holds the mean path and intervals at configured levels, per
scenario when there are scenarios.

Models, splitters and metrics, settled 2026-11-25 (Stage 7, D36, D37):

- **Model forms** (`dz.models`, `registry`, `register`, `get`) are classes
  with a `Config`, built from it (`models.get("arimax")(order=(1, 0, 0))`
  or `Form(config)`), and `fit(y, X=None)`: `y` a float `pd.Series` on a
  `PeriodIndex`, `X` a `pd.DataFrame` on the same periods, neither with a
  missing value (`ModelError`). `ModelForm` is the base class of the
  built-ins; a form's class attribute `drivers` says whether it takes `X`,
  and a form without drivers refuses them. Built-in keys: `naive`,
  `seasonal_naive` (`season`, default from the frequency), `mean`, `ols`
  (`constant`, `cov_type`, `hac_maxlags`), `glm` (`family` binomial or
  gaussian, `constant`, `scale` X2, dev or fixed, `cov_type`), `arimax`
  (`order`, `seasonal_order`, `trend`, `maxiter`) and `ets` (`error`,
  `trend`, `damped_trend`, `seasonal`, `seasonal_periods`, `maxiter`).
- **Fitted** (`models.Fitted` protocol, `FittedModel` base): `spec`,
  `params` (`pd.Series`), `cov_params` (`pd.DataFrame`), `residuals`
  (`y` minus the in-sample one-step fitted value, on the target's scale,
  for every form), `fitted_values`, `notes` (statsmodels' fit warnings),
  `record()` and `predict(h, X_future=None, levels=(0.8, 0.95))`.
  `X_future` holds the drivers' values for the `h` periods after the
  sample, indexed by those periods or by position.
- **Forecast** (`models.Forecast`) is data: `form`, `frequency`, `periods`
  (text), `mean`, `levels`, and `lower`/`upper`, one path per level, the
  central prediction interval holding that probability; `frame()` and
  `interval(level)` give pandas. One `Forecast` is one driver path; the
  forecast step (Stage 9) calls `predict` once per scenario.
- **Splitters** (`dz.evaluation.splitters`): `expanding` (`initial`,
  default half the series; `horizon`, `step`, `folds`) and `rolling`
  (`window`, `horizon`, `step`, `folds`). `split(index)` returns `Fold`s of
  half-open positions `train` and `test`; every fold tests a full
  `horizon`, the last test period is the last period, and `folds` keeps
  the last origins. Holdout is `expanding` with `folds: 1`.
- **Metrics** (`dz.evaluation.metrics`): `mae`, `rmse`, `mase` (`season`,
  default 1) and `coverage` (`level`, default 0.95), each
  `score(actual, forecast, *, scale=None, train=None, lower=None,
  upper=None)`, ignoring keywords it does not use.
- **Rolling-origin evaluation**: `evaluation.evaluate(model, y, X=None, *,
  splitter=None, metrics=("mae", "rmse", "mase", "coverage"),
  levels=(0.8, 0.95)) -> Evaluation` refits the form at every origin
  (default `expanding`, one step), forecasts the fold's horizon with the
  drivers' actual values, and scores each metric over every forecast and
  by horizon. `Evaluation` is data: `spec`, `splitter`,
  `splitter_config`, `levels`, `folds`, `rows` (fold, origin, period,
  horizon, actual, forecast, bounds) and `scores` (`metric`, `config`,
  `horizon`, `value`; `horizon: null` over all rows), with
  `score(metric, horizon=None)` and `frame()`.

```python
import dazflow as dz

@dz.models.register("theta")
class Theta:
    Config = ThetaConfig                     # a pydantic model

    def fit(self, y, X=None) -> dz.models.Fitted:
        ...
```

```yaml
# config/build.yaml
forms: [ols, arimax, theta]
```

A new step is a module with a `Config`, a `Result` and `run()`, registered
like any component and placed in the pipeline by name.

Settled 2026-11-25 (Stage 6, D32): a step's `run()` keeps its own
signature, and the pipeline calls it through an adapter registered by step
name in `dazflow.pipeline.adapters` (`@register_adapter("build")`). An
adapter receives the step module, its validated `Config` and `StepInputs`
(the earlier results by name, the latest dataset, the project folder for
relative paths, frames passed in memory, the previous vintage) and returns
the step's result. A step with no adapter is called as
`run(dataset, config)` with the latest dataset. The adapters live in the
pipeline layer, which may import every step; steps still import none. Wrapper
plugins for other forecasting libraries (statsforecast, sktime,
skforecast, darts, prophet) are the intended way to add their models.
They are separate packages or extras, not core.

## 7 Workflow steps

Each step lists what it does, its gate, its artefacts and the packages it
may wrap. Thresholds are always configuration, never constants in code.
A failed gate is reported, never hidden, and can be overridden only with
a reason written in configuration.

The default pipeline is `ingest → clean → explore → build → select →
forecast → diagnose → report`. `segment` and `adjust` are optional, and
a pipeline can drop, reorder or add steps within their input needs.

### 7.1 `ingest`: import and align

- Read the target, drivers and scenarios through readers.
- Put all series on one calendar: frequency conversion, period-end or
  period-average, publication lags.
- Record each series' definition and source, and any change of
  definition over its history.

**Gate:** every configured series found, on a common calendar, with the
minimum history length. **Artefacts:** series inventory, history chart
per target.
**Packages:** pandas; narwhals for polars and pyarrow input; openpyxl
for Excel; SQLAlchemy for SQL; optional sdmx1, pandas-datareader,
fredapi for public data.

Settled 2026-11-25 (Stage 3, D28). `dz.ingest.run(config, data=None)`:
`config.inputs` are `ReadSpec`s with a `name`, read from their `path` or
taken from `data[name]`; `config.series` names each series, its input
and column, role, unit, source, definition, expected sign, `aggregation`
and `lag`; `config.frequency` is the calendar. A series moves only to a
lower frequency: `average` gives a target period the mean of its native
periods when all of them have a value, `end` the value of its last
native period. A publication lag of L periods moves each value L periods
later. The calendar runs from the first to the last period of any
series. The gate has `found` (series not found, `<= 0`) and `history`
(periods on which every target and driver has a value, `>=
min_history`, default 40). Artefacts: `T1.1` series inventory, `T1.2`
the gate, `F1.1` the history of every target in one figure (no target,
no figure), `S1.1` the summary finding. One figure for all targets keeps
D21 unchanged. Definition changes over a series' history are not yet
recorded.

### 7.2 `clean`: data quality and treatment

- Check completeness, gaps, duplicates, ranges and revisions.
- Flag outliers and level shifts in the history.
- Treat what is configured: fill or interpolate gaps, winsorise or
  replace outliers, exclude periods. Every treatment goes to the
  dataset's log with its reason and the rows affected.
- Check representativeness, for example that the history covers a
  downturn.

**Gate:** quality checks within thresholds after treatment. **Artefacts:**
quality report, treatment and exclusion table, history chart with
flagged periods.
**Packages:** pointblank for declarative checks with warn, error and
critical thresholds (D12); statsmodels `STL` for seasonal decomposition and robust
outlier detection; scipy `stats` for distribution tests; optional
ruptures for level shifts.

Settled 2026-11-25 (Stage 4, D29, D30). `dz.clean.run(dataset, config,
*, previous=None)`; `previous` is the previous vintage, for the
revisions check. A check (`dz.checks`) names, for one series, its test
units (periods) and what each must satisfy (not missing, between bounds,
at most or above a bound); `checks.evaluate` interrogates them with one
pointblank validation whose `warning`, `error` and `critical` thresholds
come from the check's configuration (below 1 a share of the units, 1 or
more a whole count, as in pointblank). The gate check `<key>:<series>`
fails when the failing units reach `error`; the level reached is
reported. pointblank's own report is not used: its counts and failing
rows become dazflow `Check`s, `Outcome`s and `Table`s. With no test unit
the check passes without pointblank, which 0.13 cannot run on an empty
table.

- **Checks** (defaults in brackets): `completeness`, every calendar period
  has a value (error 0.1); `gaps`, no run of missing values inside the
  history longer than `max_length` (0; error 1); `duplicates`, no value
  equal to the previous period's (error 0.1) — a `Dataset` has one row
  per period, so a repeated value is the duplicate left to find;
  `range`, values within `bounds` per series, inclusive (error 1);
  `revisions`, `|new - old| / |old|` at most `tolerance` (0.01; error
  0.1) on periods in both vintages; `outliers`, robust z-score of STL
  residuals at most `z` (4; error 0.05); `representativeness`, every
  period of each configured downturn has a value, on targets and drivers
  (error 1). Each can be disabled or limited to `columns`; plugin checks
  run from `custom: {key: config}`. Checks run before and after
  treatment; the gate is built from the second run.
- **Treatments** run in order: `fill` (previous, next or a value),
  `interpolate` (linear, inside the history), `winsorise` (clip to the
  series' `lower` and `upper` quantiles), `replace` (a value, or linear
  from the neighbours) and `exclude` (values become missing, and no check
  counts an excluded period; no later treatment fills it). Each names
  `series` (default every numeric series) and `periods`, or `start` and
  `end`. A treatment without a reason, or with a blank one, is a
  `ConfigError`. Each treatment logs one `LogEntry` per series with its
  reason and the periods it changed, and the treatment table holds every
  changed value before and after, so the log replays onto the raw frame.
- **Artefacts:** `T2.1` quality report (one row per check and series,
  before and after; checks that do not apply are listed in its notes),
  `T2.2` treatments and exclusions, `T2.3` the gate, `F2.1` the history
  of the targets after cleaning with the periods flagged before treatment
  marked (no target, no figure), `S2.1` the summary finding.

### 7.3 `explore`: each series alone, each driver against the target

- Profile each series: trend and seasonal strength, autocorrelation,
  volatility, intermittency. The profile narrows the model forms tried
  in `build`.
- Choose transformations and test stationarity, unit-root and
  stationarity tests read together; record the order of integration.
- Search each driver's lag against the target, at lags 0 to *k*.
- Fit single-driver regressions with autocorrelation-robust errors.
- Compare each driver's sign with the expected sign declared in
  configuration.

**Gate:** each driver kept has the expected sign and passes the
significance threshold. **Artefacts:** series profile table, stationarity
table, driver screen, lag-profile charts.
**Packages:** arch `unitroot` (ADF, DFGLS, Phillips-Perron, KPSS,
Zivot-Andrews); statsmodels `adfuller`, `kpss`, `acf`, `ccf`, `STL`,
`OLS(cov_type="HAC")`; scipy `special.logit`, `stats.boxcox`; optional
tsfeatures-style measures from statsforecast or tsfresh.

Settled 2026-11-25 (Stage 5, D31). `dz.explore.run(dataset, config)`
profiles and tests every numeric series, transforms each, and screens
each driver against the one target (`target`, default the dataset's only
target; `drivers`, default the dataset's drivers, in that order). The
caller's dataset is not changed; the result's `dataset` holds every
series transformed by its chain, recorded in `SeriesMeta.transforms`
and logged.

- **Profile** (`profile()`), on the values from the first to the last
  present one, interior gaps interpolated for the fit and counted: trend
  and seasonal strength, `max(0, 1 - Var(R)/Var(T+R))` and
  `max(0, 1 - Var(R)/Var(S+R))` of a statsmodels STL fit (seasonal
  smoother `profile.seasonal`, 7) for monthly and quarterly series, or a
  lowess trend over `profile.trend_window` years (21) and no seasonal
  measure for annual ones; sample autocorrelation at lag 1 and at a year;
  volatility, the standard deviation of period-on-period changes; and
  intermittency, the share of present values equal to zero.
- **Stationarity** (`unit_root()`, `integration_order()`): ADF, DFGLS,
  Phillips-Perron and KPSS from arch on the series after its configured
  `transforms` chain (e.g. `log`). Each test says stationary at
  `stationarity.alpha` (0.05): the three unit-root tests when p < alpha,
  KPSS when p >= alpha (`says_stationary()`). A level is stationary when
  at least `stationarity.votes` (3) of the four say so. Levels are tried
  in order: the series around a constant, then around a linear trend,
  then its first, second ... differences around a constant, up to
  `stationarity.max_order` (2). The first stationary level gives the
  order and the kind: `stationary` (I(0)), `trend-stationary`,
  `integrated` (I(d)); none within `max_order` is `undetermined`, and
  the series gets `max_order` differences. A trend-stationary series is
  differenced once (`stationarity.trend_stationary: difference`) or kept
  in levels (`keep`). `stationarity.lags` fixes every test's lags or
  bandwidth; unset, ADF and DFGLS choose by `stationarity.method` (AIC)
  and PP and KPSS take arch's data-dependent bandwidth. A test arch cannot
  run (a constant or too short series) is noted and does not say
  stationary; a series with fewer than `stationarity.min_observations`
  (20) values is an `ExploreError`. The chain is the configured
  transforms followed by one `diff(lag=1)` per difference.
- **Lag search** (`lag_search()`), on the transformed series: the
  correlation of the target with each driver lagged 0 to `max_lag` (4)
  periods, all on the one sample where the target and every lag are
  present; the lag of largest absolute correlation is chosen (the
  smallest on a tie).
- **Screen** (`screen()`): the target on a constant and the driver at its
  lag, by statsmodels OLS with HAC standard errors (`maxlags` from
  `hac_maxlags`, default floor(4 (n/100)^(2/9)), Newey-West), on every
  period with both present.
- **Expected signs**: `expected_signs` in configuration, by driver,
  replaces `SeriesMeta.expected_sign`; 0 or unset means no sign check.
- **Gate** (`driver_checks()`): per driver, `sign:<driver>` (coefficient
  times expected sign > 0, only with an expected sign of -1 or 1) and
  `significance:<driver>` (HAC p-value <= `significance`, 0.05). A
  missing coefficient or p-value fails. A failed check passes only with
  an override and its reason. The kept drivers are those whose checks
  all pass or are overridden; the gate passes when every driver is kept,
  so a driver that does not qualify is either removed from `drivers` or
  accepted by override, and either choice is in configuration.
- **Artefacts:** `T3.1` series profile (with each series' integration
  and chain), `T3.2` stationarity tests (one row per level tried), `T3.3`
  driver screen, `T3.4` the gate, `F3.1` the lag profiles of every
  driver in one figure (no driver, no figure), `S3.1` the summary
  finding (`explore.summary`).
- Zivot-Andrews (listed above) is not run in 0.1.0: the roadmap's Stage
  5 names the four tests, and a structural break belongs with the
  `breaks` extra.

### 7.4 `build`: candidate models

- Check correlation and variance inflation between drivers.
- Search driver combinations, exhaustively or stepwise, up to a
  configured size and within the signs and exclusions declared.
- Fit forms in increasing complexity: benchmarks, regression on
  transformed series, regression with autoregressive terms, ARIMA with
  drivers, ETS, cointegrated systems. Move up only when simpler forms
  fail a gate.

**Gate:** every coefficient has its expected sign and is significant;
multicollinearity within thresholds. **Artefacts:** correlation matrix,
candidate table, coefficient chart per candidate.
**Packages:** statsmodels `OLS`, `GLM`, `SARIMAX`, `ETSModel`, `VAR`,
`VECM`, `coint_johansen`, `variance_inflation_factor`; arch
`engle_granger`, `DynamicOLS`; optional scikit-learn and glum for
penalised regression; optional statsforecast for fast ARIMA and ETS
order search.

Settled 2026-11-25 (Stage 8, D39). `dz.build.run(dataset, config,
*, explore=None)` takes the series in levels, as explore received them
(their fingerprint must be explore's `input_fingerprint`), and explore's
result, which gives the target, the kept drivers in order, their lags and
expected signs, each series' resolved chain and its kind of integration;
`config` replaces any of them. Without explore the series are used as they
are, at lag 0. The caller's dataset is not changed.

- **Design.** A candidate's data come from `dz.models.Design`: the target
  and drivers transformed by their chains (`transforms.forward` on the
  recorded text), each driver shifted by its lag, a `trend` column when
  needed, on one sample. Select rebuilds every candidate's data through
  the same `Design.frames(levels)`, and `Design.future(levels, paths, h)`
  builds the drivers after the sample from their future paths in levels.
- **One sample.** Every candidate and benchmark is fitted on the common
  sample, the periods where the target and every searched driver at its
  lag are present, so information criteria and rolling-origin folds
  compare; a gap inside it is a `BuildError`.
- **Search.** `combinations(drivers, max_size, exclusions)` lists every set
  of 1 to `max_size` drivers holding no excluded pair, by size and then in
  driver order; `stepwise` adds, each step, the driver whose OLS fit has
  the lowest AIC (or BIC), skipping excluded pairs, until `max_size` or no
  improvement. "Within the signs" is the sign check below: a set is not
  pruned before fitting.
- **Forms.** `forms` (default `ols` with HAC errors, then `arimax`
  (1,0,0), regression with AR(1) errors) are tried in order for each set;
  the next is fitted only when the simpler one fails a check, so a set
  yields one passing candidate or every failed attempt. `benchmarks`
  (default `naive`, `mean`) are always fitted. A form that cannot be
  fitted (`ModelError`) is recorded with its error and the next is tried.
- **Checks** (per candidate, `Candidate.gate`): `sign:<driver>`
  (coefficient times expected sign > 0, for a sign of -1 or 1),
  `significance:<driver>` (p <= `significance`, 0.05) and `vif` (largest
  VIF of its drivers <= `max_vif`, 10). Only driver coefficients are
  checked; the constant, the trend and a form's own parameters (`ar.L1`,
  `sigma2`) are reported. p-values are two-sided normal from `params` and
  `cov_params`, for every form: statsmodels' own for OLS with HAC (the
  default), GLM and ARIMAX. A candidate's check is overridden as
  `<candidate>/<check>` (`ols:gdp/significance:gdp`). The step's gate is
  `candidates`: at least `min_candidates` (1) candidates pass.
- **Trend terms.** Under `trend: auto`, a candidate gets a linear trend
  column (`trend`, 1 at the dataset's first period) when its target or a
  driver is trend-stationary and kept in levels (explore's
  `trend_stationary: keep`); `always` and `never` force it.
- **VIF** is `1 / (1 - R²)` of each driver on a constant and the others,
  equal to statsmodels' `variance_inflation_factor` on a design with a
  constant.
- **Artefacts:** `T4.1` correlation and VIF of the drivers, `T4.2`
  candidates, `T4.3` coefficients, `T4.4` the gate, `F4.1` every
  candidate's driver coefficients as z statistics in one figure (no
  coefficient, no figure), `S4.1` the finding (`build.summary`).
- Systems (VAR, VECM), cointegration and penalised regression are not
  built in 0.1.0 (open question 3, After 0.1.0).

### 7.5 `segment` (optional): one model or several

- Compare a pooled model, a pooled model with segment effects, and one
  model per segment, out of sample.

**Gate:** a segment gets its own model only if it beats the pooled one
out of sample and has enough history. **Artefacts:** pooled-against-
segment comparison, segment history charts.
**Packages:** linearmodels `PanelOLS`, `RandomEffects`, `PooledOLS`;
statsmodels `MixedLM`; optional pyfixest.

### 7.6 `select`: rank and choose

- Rolling-origin evaluation: refit each candidate at successive past
  origins and score its forecasts.
- Score with errors scaled by the benchmark's, so series compare.
- Test whether candidates are distinguishable, not only ranked.
- Forecast each candidate under every scenario and check plausibility:
  ordered by severity, within configured bounds.
- Propose a default: lowest error among candidates passing all gates,
  fewer drivers preferred when within a configured tolerance. The
  modeller confirms or overrides it in configuration, with a reason.

**Gate:** the chosen model beats the benchmark; scenario forecasts are
ordered and bounded. **Artefacts:** ranking table, evaluation table,
back-test and scenario charts per candidate, selection record.
**Packages:** scikit-learn `TimeSeriesSplit` or own splitters; arch
`bootstrap.MCS` and `SPA`; statsmodels information criteria; optional
dieboldmariano.

Settled 2026-11-25 (Stage 8, D38, D40). `dz.select.run(dataset, config,
*, build, seed=None)` takes the series in levels that build ran on (its
fingerprint, scenarios left out, must be build's) with any scenarios in
`dataset.scenarios`, or read from `config.scenarios`, and build's result.

- **Scores.** Every candidate and benchmark with a fit is evaluated by
  `dz.evaluation.evaluate` on its `Design.frames` (splitter `expanding`,
  half the sample first, one-step folds by default). Its scaled error is
  its RMSE over the `benchmark`'s (default `mean`: on the stationary series
  build fits, the historic mean is the no-information forecast), or the
  MAE ratio with `loss: absolute`. A candidate beats the benchmark below
  `benchmark_ratio` (1).
- **Folds too short.** A candidate whose form cannot be fitted at an origin
  (`ModelError`, for example a first window shorter than an ARIMAX needs)
  is not scored, not in the confidence set and not chosen; the ranking
  and evaluation tables give the error. Skipping only that origin would
  give the candidates different origins and so no common loss matrix. The
  benchmark failing so is a `SelectError`.
- **Confidence set.** `confidence_set(losses, ...)` is
  `arch.bootstrap.MCS(losses, size, reps, block_size, method, bootstrap,
  seed=seed).compute()` on the per-origin losses (mean squared or absolute
  error of each fold) of every scored model, benchmarks included; `mcs`
  configures it (size 0.10, 1000 replications, method R, stationary
  bootstrap). The seed is `mcs.seed`, else project.yaml's `seed` (passed
  through `StepInputs.seed`), else 0, so a run repeats.
- **Scenarios.** Each scored model is refitted from its record
  (`models.refit`, which checks the data's fingerprint) and forecast under
  every scenario, its drivers built by `Design.future` from the scenario's
  paths in levels for the periods after the sample; the mean path is
  mapped back to the target's levels by `transforms.inverse` with the
  observed history (no bias correction: that is forecast's, §7.7). A
  model's `ordering.statistic` (mean) of each path is compared across
  scenarios, mildest first: a pair of successive scenarios is out of order
  when the more severe one projects a lower value (`direction: 1`) or a
  higher one (`-1`); equal values are not out of order, since scenarios
  with the same paths for a model's drivers give equal projections.
  `bounds` counts projected values outside `[lower, upper]`.
- **Choice** (D38). `choice: {candidate, reason}` confirms the proposal or
  overrides it; the reason is required and not blank. Without it the
  proposal is used and `decision` is `proposed`. Benchmarks are compared,
  never chosen.
- **Gate:** `confidence-set` (at least one candidate that passes every
  check is in the set) and, for the chosen model, `build` (its build
  checks that failed and were not overridden: 0), `benchmark` (scaled
  error < `benchmark_ratio`), `scenario-order` (with two or more
  scenarios) and `scenario-bounds` (with bounds).
- **Artefacts:** `T6.1` ranking, `T6.2` rolling-origin evaluation, `T6.3`
  projections by scenario, `T6.4` selection record, `T6.5` the gate,
  `F6.1` the one-step back-test of every scored model in one figure,
  `F6.2` the chosen model's projection under each scenario (no choice or
  no scenario, no figure), `S6.1` the finding (`select.summary`, or
  `select.empty` when nothing is proposed). The design's "back-test and
  scenario charts per candidate" are one figure each, as explore and
  ingest chose (D28, D31); every candidate's projections are in `T6.3`.
- The result holds what forecast (Stage 9) needs: `chosen`, its `record`
  (`ModelRecord`), `design` (drivers, lags, chains, trend), `drivers`, and
  `scenario_forecasts` (every scored model under every scenario: the
  `Forecast` on the transformed scale and its mean path in levels).
- Arch's `SPA`, statsmodels' information criteria in the ranking and
  dieboldmariano are not used in 0.1.0: the confidence set answers
  whether candidates are distinguishable.

### 7.7 `forecast`: projection and calibration

- Forecast the chosen model from the origin, for the configured horizon
  and under every scenario, with intervals.
- Anchor any gap between fitted and observed values at the origin by a
  configured rule: carry, decay or none.
- Optionally revert towards a long-run level over the horizon.
- Transform back to the target's scale, with a bias correction suited
  to the transformation (smearing for logs, a simulation-based
  correction for logit).

**Gate:** forecasts reconcile to the origin and stay within bounds.
**Artefacts:** forecast table by scenario and horizon, fan chart.
**Packages:** statsmodels `get_forecast`, `get_prediction`,
`VARResults.forecast`; numpy `random.Generator` for simulation;
optional xarray for scenario × horizon × segment cubes.

Settled 2026-11-25 (Stage 9, D41). `dz.forecast.run(dataset, config=None,
*, select, seed=None)` takes the series in levels select ran on (its
fingerprint, scenarios left out, must be select's) and select's result.

- **Model and origin.** Select's `record` is refitted (`models.refit`, so
  the data's fingerprint is checked) on `select.design.frames(levels)`; the
  origin is the design's last sample period, `design.end`. Each scenario is
  one `Fitted.predict(h, design.future(levels, paths, h), levels=)` call;
  that `Forecast`, before anchoring, is `ScenarioPath.raw`.
- **Scenarios** are declared once (D33, D40): `dataset.scenarios`, else
  select's own `config.scenarios` file, which the pipeline adapter reads
  from the project folder and records under the forecast step too. The
  reader is the shared `dazflow.readers.read_scenarios` (moved out of
  select, which keeps its name and its `SelectError`), since steps do not
  import each other. The horizon is `horizon`, else as many periods as
  every scenario covers after the origin.
- **Anchoring** is on the model's scale (an add factor on the equation as
  fitted): the gap is observed minus fitted at the origin, and horizon *h*
  adds the gap (`carry`), gap × (1 − `decay`)^*h* (`decay`, the default,
  `decay` 0.5) or nothing (`none`), to the mean and both bounds.
- **Back to the target's scale** through the target's resolved chain and
  `transforms.inverse` with the observed history (D27). `correction: auto`
  picks by the chain: no nonlinear transform (empty, `diff`,
  `seasonal_diff`): no correction, the inverse of the mean is the mean,
  and the bounds are the mean ± the square root of the summed squared
  half-widths through the chain's linear map (`intervals: linear`); a
  chain without lags (`log`, `boxcox`): smearing, the inverse averaged
  over the model's residuals (Duan, 1983), bounds inverted
  (`intervals: inverse`, exact for an increasing transform); a `logit`
  without lags: simulation, the inverse averaged over `simulation.draws`
  normal errors whose standard deviation is read from the widest interval;
  a nonlinear chain with a lag (`log` then `diff`, `logdiff`, `growth`):
  simulated level paths, the mean and the quantile bounds taken from them
  (`intervals: simulated`). `none` keeps the plain inverse (the median of
  a monotone transform); smearing with a lag, or any correction on a linear
  chain, is a `ForecastError`. Errors are taken independent across
  horizons, which is exact for the regression forms and an approximation
  for ARIMAX. The draws use `numpy.random.default_rng(seed)`, the same
  draws for every scenario (common random numbers), with `seed` from
  `simulation.seed`, else project.yaml's (`StepInputs.seed`), else 0; the
  result records `correction`, `intervals`, `draws` and `seed`.
- **Mean reversion** (`reversion: {level, periods}`) is on the target's
  scale, after the correction: horizon *h* moves the mean by
  min(*h*/`periods`, 1) × (level − mean), `periods` unset meaning the
  horizon, and the bounds by the same shift, so the interval keeps its
  width.
- **Gate:** `reconcile`, the forecast taken back to the origin (fitted plus
  the anchoring at *h* = 0, through the chain with the history before the
  origin) minus the observed value, in absolute value at most `tolerance`
  (1e-6, in the target's units; `carry` and `decay` meet it by
  construction, so it binds under `none`); `forecast-bounds`, the forecast
  means outside `bounds`, which default to select's.
- **Artefacts:** `T7.1` the forecasts by scenario and horizon (mean and
  bounds), `T7.2` the projection record (model, origin, gap, rule,
  reversion, chain, correction, intervals, draws, seed, bounds), `T7.3`
  the gate, `F7.1` the fan chart (history, then every scenario's mean and
  its widest interval, one figure as D28 and D31 chose), `S7.1` the
  finding (`forecast.summary`).
- The result holds what diagnose and report need: `chosen`, `form`,
  `record`, `design`, `origin`, `gap`, `origin_gap`, `chain`, and `paths`,
  one `ScenarioPath(scenario, raw, anchored, forecast, reversion)` per
  scenario in `severity` order, `forecast` on the target's scale;
  `result.path(scenario)` and `result.frame()` give them by scenario.

### 7.8 `diagnose`: test the chosen model

- Residual diagnostics: autocorrelation, heteroskedasticity, normality,
  specification.
- Stability: recursive residuals, CUSUM, rolling coefficients,
  structural breaks.
- Back-testing: multi-step forecasts from past origins against actuals.
- Sensitivity: response to a shock in each driver, and to each driver's
  scenario path alone.
- Anomalies: observations outside the model's interval at a configured
  level. The same check monitors the model once it is in use.

**Gate:** each diagnostic within its threshold. **Artefacts:**
diagnostics table, sensitivity table, anomaly table; residual, CUSUM and
rolling-coefficient charts.
**Packages:** statsmodels `acorr_ljungbox`, `acorr_breusch_godfrey`,
`het_breuschpagan`, `het_white`, `het_arch`, `jarque_bera`,
`linear_reset`, `RecursiveLS`, `breaks_cusumolsresid`, `RollingOLS`;
optional ruptures for change points and SALib for global sensitivity.

Settled 2026-11-25 (Stage 10, D42). `dz.diagnose.run(dataset, config=None,
*, select, forecast=None)` takes the series in levels select ran on (its
fingerprint, scenarios left out, must be select's), select's result and,
optionally, forecast's result for the same model, which only sets the
sensitivity horizon. Select's `record` is refitted on
`select.design.frames(levels)`, so the data's fingerprint is checked.

- **Applicability.** The regression-based tests (Breusch-Godfrey,
  Breusch-Pagan, White, RESET, both CUSUM tests, Chow, rolling
  coefficients) are tests of an OLS regression and run on a statsmodels
  OLS of the same design matrix; they apply to the `ols` form only.
  Ljung-Box applies to every estimated form (with the ARMA order as
  `model_df` for an ARIMAX); ARCH and Jarque-Bera to every form but `glm`,
  whose family sets its variance. No test applies to a benchmark. A test
  that does not apply is recorded with its reason, listed under the
  diagnostics table and left out of the gate.
- **Residual tests**, each a check `p-value >= alpha` (`alpha` 0.05):
  Ljung-Box, Breusch-Godfrey and ARCH over `lags` (unset: twice the
  seasonal period, 10 for annual data, at most a fifth of the sample;
  Hyndman), Breusch-Pagan and White on the regressors with a constant,
  Jarque-Bera, RESET (`power` 3).
- **Stability.** `cusum`: `breaks_cusumolsresid` on the OLS residuals,
  `p-value >= alpha`; `cusum-recursive`: the periods where `RecursiveLS`'s
  CUSUM lies outside Brown, Durbin and Evans' bounds, `<= max_crossings`
  (0). `alpha` is 0.01, 0.05 or 0.10, the levels they tabulate (constants
  1.143, 0.948, 0.850; statsmodels' private bound uses 0.950 at 0.10).
  Chow (own code): F = ((RSS_pooled − RSS_1 − RSS_2)/k) /
  ((RSS_1 + RSS_2)/(n − 2k)) on F(k, n − 2k), every coefficient allowed to
  change from the break on, at the configured `breaks`, else at
  `candidates`, fractions of the sample fixed before the data are seen
  (default the middle, 0.5), each `p-value >= alpha`; a break with no more
  than k observations on a side is not applicable. Rolling coefficients:
  `RollingOLS` over `window` (unset: a third of the sample, at least 2k),
  charted, not gated.
- **Back-test:** `dz.evaluation.evaluate` with the configured splitter
  (default `expanding`, `horizon` 4), MAE, RMSE, MASE and the coverage of
  the `level` (0.95) interval, which must be at least `min_coverage` (0.8).
- **Sensitivity**, one at a time from the base scenario (`base`, unset: the
  mildest; without scenarios, the drivers held at their last values): a
  `shock` (1) added to a driver's column as the model takes it (after its
  transform and lag) at every horizon, in standard deviations of that
  column (`scale: sd`, the default) or in its units (`unit`); and each
  driver's path in every other scenario alone, then all of them. Responses
  are on the model's scale and on the target's (through its chain, with
  no bias correction). The horizon is `horizon`, else forecast's, else the
  scenarios' common coverage. Not gated.
- **Anomalies:** the observations outside fitted ± z·s, z the normal
  quantile of `level` (0.99) and s the model's `sigma2` where it estimates
  one, else the residual standard error with the parameters as degrees of
  freedom; a value on a bound is inside. Their share must be at most
  `max_share` (0.05).
- **Artefacts:** `T8.1` diagnostics (applicable tests with their status;
  the others in its notes with their reasons), `T8.2` sensitivity, `T8.3`
  anomalies, `T8.4` back-test scores by horizon, `T8.5` the gate, `F8.1`
  residuals with the anomaly interval, `F8.2` the recursive CUSUM with its
  bounds, `F8.3` rolling driver coefficients (one figure each, as D28 and
  D31 chose; `F8.2` and `F8.3` only for an OLS model), `S8.1` the finding
  (`diagnose.summary`).
- The example project's `config/diagnose.yaml` accepts RESET's failure
  (p ≈ 0.03 on the made-up series) by an override with its reason.

### 7.9 `adjust` (optional): adjustments and margins

- Adjustments correct an identified bias and may go either way.
- Margins cover uncertainty and are never negative: data and method
  weaknesses, change of environment, estimation error.
- Overlays: judgment entries outside the model, each with an owner, a
  reason and a size.
- Floors and caps on forecast values.

**Gate:** every entry has a reason and a size; margins are non-negative.
**Artefacts:** adjustment register, forecast before and after.
**Packages:** arch `bootstrap` (`StationaryBootstrap`,
`CircularBlockBootstrap`) for estimation error; statsmodels
`cov_params` for analytic intervals.

### 7.10 `report`: deliverables

- Run record (§10.1).
- Document: every step's section in pipeline order, with a summary,
  contents and a configuration appendix.
- Replay script: plain Python that reruns from the record and compares.
- Scoring code: the chosen model as a standalone function.

**Gate:** the replay reproduces the recorded results. **Artefacts:** the
document, configuration appendix, environment table, input table.
**Packages:** python-docx, jinja2; optional Quarto (an external binary,
not a Python package) for HTML and PDF.

Settled 2026-11-26 (Stage 11, D43, D44):

- **The step.** `dz.report.run(record, config=None)` takes a run record
  (`RunRecord` or its dictionary) and nothing else, and imports no other
  step: it reads each step's configuration, result, artefacts and gate
  from the record, and the inputs and environment. In the pipeline its
  adapter passes `StepInputs.record()`, the record of the steps before it as
  it would be written then. Configuration (`config/report.yaml`): `title`
  (default the project name), `file` (`document.docx`), `replay` (`rtol`,
  `atol`, D43) and `overrides`.
- **The document** is a `dazflow.render.Document`: the title, then
  `Contents` (a numbered list of the sections after it), `Summary` (the
  project's description, the finding `S10.1`, the run summary `T10.1` and
  the report's gate `T10.5`), one section per documented step in pipeline
  order, titled by its §7 heading (a plugin step by its name), holding
  `Finding S<n>.<k>.` and the finding, then the step's tables and figures in
  identifier order, and two appendices: `Appendix A. Configuration`
  (`T10.4`: every setting of every step as run, one row per leaf setting,
  values other than text as JSON) and `Appendix B. Environment and inputs`
  (`T10.2` environment, `T10.3` inputs with their fingerprints).
  `Project.write` writes it as `reports/document/<file>` and the replay
  script as `reports/document/replay.py`, replacing the folder; every table
  and figure gets the final record's source note (run id, the dataset of
  the step it belongs to, the run's date). The document and the script
  are excluded from the result's `to_dict()`, as a dataset is (D26): the
  record they are built from holds all of it.
- **The gate** of the step is `complete` (every pipeline step before report
  is in the record) and `gates` (every documented step's gate passed). The
  replay is `dazflow replay`'s exit status, not a check of the step (D44).
- **The replay script** is plain Python that loads the project from its
  own location (`reports/document/` two levels down), calls
  `Project.replay` with the configured tolerance, prints every difference
  and exits 0 on a match and 2 otherwise.

## 8 Configuration

- One pydantic model per step and per component. YAML on disk, Python
  objects in memory, JSON Schema for tools.
- `project.yaml` holds the name, target, drivers with expected signs,
  segments, scenarios, frequency, seed and the pipeline. `config/` holds
  one file per step, named after the step.
  As built in 0.1.0 (2026-11-25, Stage 6, D33): `project.yaml` holds
  `name`, `description`, `pipeline`, `seed`, `style` (a style file merged
  onto the house style) and `previous` (the previous data vintage). The
  target, drivers, expected signs and frequency stay in
  `config/ingest.yaml`, where Stage 3 put them and where every later step
  reads them from the series metadata: one place per setting. Segments and
  scenarios come after 0.1.0. Scenarios are read by select from
  `config/select.yaml: scenarios` (a CSV of driver paths in levels), or
  passed in memory as `Dataset.scenarios` (2026-11-25, Stage 8, D40).
- Defaults are filled in and written to the run record, so a record
  never depends on the library's defaults at replay time.
- Decisions live in configuration: shortlist, chosen model, treatments,
  adjustments, gate overrides with reasons. Nothing is answered at a
  prompt, so every decision is in version control.
- `dazflow validate` checks configuration and data without running a
  step.

## 9 Artefacts

Artefacts are the product. A user never reformats a number, renames a
column, restyles a chart or writes a caption.

- **Kinds.** `Table` (plain-language labels, units, formatted numbers,
  source note, pass or fail marked in place), `Figure` (house theme,
  fixed size, labelled axes with units, source note), `Finding` (a
  template sentence filled with numbers, never generated text),
  `Section` and `Document`.
- **Stable identifiers.** `T<step>.<n>` and `F<step>.<n>` survive reruns,
  so documents can cross-reference them.
- **One style file per organisation.** Fonts, colours, number and date
  formats, page width, the Word reference document and the finding
  templates. Overriding a template needs no code.
- **Provenance.** Every source note carries the run identifier, the
  dataset fingerprint and the date.
- **Formats.** Word first; HTML, Markdown, PNG and SVG; Excel for tables.

| Need | Package |
| --- | --- |
| Table objects | great-tables |
| Word tables and documents | python-docx (see open question 2) |
| Figures | matplotlib with a house `.mplstyle` |
| Findings and Markdown sections | jinja2 |
| Excel (optional) | xlsxwriter |

Settled 2026-11-25 (Stage 2, D21–D25):

- **Objects and renderers.** `Table`, `Column`, `Figure`, `Layer` and
  `Source` are plain JSON-safe data in `core/artefact.py`, because
  `StepResult` holds them and core imports nothing above it; `render/`
  holds the style, the renderers, `Section` and `Document`. A `Table`
  holds unformatted cells (`None` for missing) under columns of kind
  `number`, `integer`, `percent` (a fraction), `text`, `date` (ISO text)
  or `status` (`pass`, `fail`, `overridden`); `Table.from_frame` and
  `Table.from_gate` build the common cases. A `Figure` is a declarative
  chart: shared `x` (numbers or labels) and `line`, `scatter`, `bar`,
  `band` and `hline` layers, axis labels with units and an `aspect`; it is
  drawn with matplotlib only when rendered. A `Source` is the run id, the
  dataset fingerprint (64 hex digits), the date and an optional note.
- **Identifiers.** `T<step>.<n>` (table), `F<step>.<n>` (figure),
  `S<step>.<n>` (finding: a sentence). The step number is the step's
  section number in §7 (`ingest` 1 … `report` 10). A step declares its
  artefacts once, `ArtefactIds("explore", tables={"summary": 1, ...})`,
  and asks `ids.table("summary")`: numbers come from the declaration, never
  from creation order, so a rerun, a reordered step or a skipped artefact
  keeps every other number. A retired number is not reused.
- **Style.** `Style` is a `Config` with sections `fonts`, `numbers`
  (decimals, thousands separator, decimal mark, missing-value text; one
  `decimals` for number and percent cells), `dates` (a `strftime` format),
  `page` (text width), `marks` (text, colour and weight of pass, fail and
  overridden), `captions` (jinja2 templates for captions and the source
  note, and the fingerprint digits shown), `figures` (font, size, palette,
  dpi, an extra `.mplstyle`), `word` (reference `.docx`, table style) and
  `findings` (template by finding `key`). `load_style` takes nothing (the
  house style), a `Style`, a mapping or a YAML path; a mapping or file is
  merged onto the house style setting by setting, file paths are relative
  to the file, and an unknown setting is a `ConfigError`. The project
  folder (Stage 6) passes its style file here.
- **Formats.** Renderers are components in `dazflow.render.renderers`
  keyed `docx`, `html`, `markdown`, `png`, `svg`; `render(obj, format,
  style=, source=)` and `write(obj, path)` use them. Every format writes
  cell text through the same formatting functions, so Word, HTML and
  Markdown show the same characters.

## 10 Project folder, runs and versions

The project folder is optional: every step also runs on objects in
memory. When used, it has a fixed layout.

```text
my-model/
  project.yaml
  config/          one file per step
  data/
    raw/           inputs as received, never edited
    interim/       after ingest and clean
    processed/     the final dataset and transformed series
  models/
    work/          candidates and the chosen model of the current run
    v1/, v2/, ...  registered versions, never changed
    registry.yaml
  reports/
    <step>/        tables, figures and finding of one step
    document/      document, replay script, scoring code
    record.json
```

Settled 2026-11-25 (Stage 6, D35):

- Input paths in step configurations are read from the project folder;
  the result and the record keep them as written, so a record does not
  depend on where the folder is.
- The previous vintage for clean's `revisions` check is configured in
  `project.yaml` as `previous: {path: ...}` (a `ReadSpec`), for example an
  earlier run's `data/interim/ingest.csv` copied under `data/raw/`. Unset,
  the check does not apply. It is explicit in configuration and needs no
  earlier run.
- A run stops after the first step whose gate fails (failed checks that
  are overridden count as passed), writes that step's outputs and the
  record with `status: gate_failed`, and exits 2. Later steps would build
  on a result the gate rejected; an override with a reason is the way on.
- `run --from S` (and a step command, which is `--from S --to S`) carries
  the steps before `S` from the last record, reading their datasets and
  checking them against the recorded fingerprints.
- Amended 2026-11-26 (Stage 11): a series' `source` in the dataset is the
  input path as written, not the resolved one (`ingest.run(config, data,
  root=)` reads relative paths from `root`). Until then the dataset's
  fingerprint, and every later step's, depended on where the folder was
  relative to the working directory, so a replay of a copied folder
  failed on fingerprints alone.

### 10.1 Run record

`reports/record.json` holds every step's configuration as run, the
results, checks, decisions and overrides, the environment (Python,
platform, package versions) and a fingerprint for every input. Processed
series are stored as CSV with a declared schema; Parquet is an option for
large panels.

Settled 2026-11-25 (Stage 6, D34). The record is
`dazflow.pipeline.RunRecord`, `kind: "dazflow.record"`, `schema_version: 1`;
`RunRecord.load` refuses another version, and a field that changes meaning
raises the version. Fields: `run` (`id`, `started`, `finished` in UTC,
`start`, `stop`, `steps` run, `carried` from the previous record,
`status` `passed` or `gate_failed`, `stopped_at`); `project` (project.yaml
as run, defaults filled); `pipeline`; `config` (each step's configuration
as run, every default filled in, input paths as written); `results` (each
step's `to_dict()`); `checks` (every gate check with its `step`);
`overrides` (every configured override with its `step`); `environment`
(`python`, `implementation`, `platform`, `machine`, `packages`: dazflow
and its installed requirements); `inputs` (`step`, `name`, `path`, `kind`
`file-sha256` with the SHA-256 of the file's bytes, or `dataset` with the
fingerprint of a frame passed in memory; the previous vintage is one); and
`datasets` (each step's dataset fingerprint and path). A step's dataset is
written as `data/interim/<step>.csv` for ingest and clean and
`data/processed/<step>.csv` after them, first column `period`, with
`<step>.schema.json` holding the dtypes, series metadata, log and
fingerprint (`dazflow.dataset-csv/1`); reading it back checks the
fingerprint, so an edited file is refused. Values are parsed with
Python's `float()`, which inverts `repr()` exactly on every pandas
version. Each step's artefacts are written to `reports/<step>/` (tables as
`.md` and `.html`, figures as `.svg`, findings as `.md`, and
`result.json`), the folder replaced on each run of the step, with the run
id and the dataset fingerprint in every source note.

Replay, settled 2026-11-26 (Stage 11, D43): `dz.pipeline.replay(record,
*, root, data, previous, rtol, atol)` reruns every step in the record from
its recorded configuration (`config`, every default filled in, so library
defaults do not matter), its seed and the inputs read from `root`, and
compares with `compare_records`: every input's fingerprint
(`inputs.<step>.<name>`), the steps run and the status, every dataset
fingerprint, and every step's result in full, by path
(`results.select.ranking[0].mase`), each difference with both values. The
recorded environment is passed to the rerun, so the report's environment
table repeats on another machine; an environment that differs is listed
as notes, not differences. `Project.replay()` adds the configuration
check: `project.yaml` and `config/<step>.yaml` as they are now, defaults
filled in, must equal the recorded ones, so an edited setting, or a
library default that a file leaves unset and that has changed, is a
difference too (`config.diagnose.anomalies.level`). Replay writes
nothing. A run that read frames in memory replays only with the same
frames passed again.

### 10.2 Versions

Registering freezes the current run as the next version folder: model
specification and parameters, configuration, reports, record and a copy
of `data/processed/`. Loading a version refits on that copy and checks
the parameters against the record, so a version still works after
`data/raw` changes. The registry lists each version with its date,
hashes, headline metrics, status (`candidate`, `approved`, `retired`)
and a note. At most one version is `approved`. A version is never edited;
only status and note change, and each change is logged.

## 11 Command line

Built with typer. Step commands share the step names.

```text
dazflow init <dir>                 project with commented defaults
dazflow validate                   configuration and data only
dazflow ingest | clean | explore | build | segment | select
dazflow forecast | diagnose | adjust | report
dazflow run [--from S] [--to S]    steps in pipeline order
dazflow replay [<version>]         rerun from a record and compare
dazflow show <artefact-id>         print one table or finding
dazflow model register|list|compare|status|load|restore
dazflow plugins                    list registered components
```

A step command writes to `reports/<step>/` and prints a short summary.
`--json` prints the result as JSON. Exit code 0 on success, 1 on error,
2 when a step completed with a failed gate.

As built 2026-11-25 (Stage 6): every command takes `--project/-p` (default
the working directory) and `--json`. A step command prints its
`StepResult.to_dict()`; `run --json` prints the record; `show --json` the
artefact; `plugins --json` the keys of every registry; `validate --json`
what it read. Each step registered in `dazflow.steps.registry`, a plugin's
included, gets a command of its name, except the reserved names `init`,
`validate`, `run`, `show`, `plugins`, `replay` and `model`. Errors go to
standard error. `replay` and `model` come in Stages 11 and after 0.1.0.

As built 2026-11-26 (Stage 11): `dazflow replay [-p DIR] [--rtol X]
[--atol Y] [--json]` replays the last record (no version argument until
versions exist, after 0.1.0); it prints each difference (the first 50
without `--json`) and a summary line, and exits 0 on a match, 2 on a
difference and 1 on an error (no record, an unreadable one). `dazflow
report` is the report step's command and `dazflow run` now ends with it.

## 12 Packages

Python >= 3.12. `src/` layout with hatchling, uv, ruff and pytest, as in
`py-dazdiff`. MIT. Licences below are as last read and are checked again
when a package is adopted.

### 12.1 Core dependencies

| Area | Packages |
| --- | --- |
| Data, numerics | pandas, numpy, scipy, narwhals |
| Statistics | statsmodels, arch |
| Data quality | pointblank |
| Configuration | pydantic, PyYAML |
| Artefacts | great-tables, python-docx, matplotlib, jinja2 |
| Command line | typer, rich |

### 12.2 Extras

| Extra | Packages | Adds |
| --- | --- | --- |
| `panel` | linearmodels, pyfixest | `segment` panel forms |
| `ml` | scikit-learn, glum | penalised regression, splitters |
| `fast` | statsforecast | fast ARIMA and ETS, series features |
| `breaks` | ruptures | change points in `clean` and `diagnose` |
| `sensitivity` | SALib | global sensitivity |
| `cube` | xarray | forecast cubes |
| `fetch` | sdmx1, pandas-datareader, fredapi | public data readers |
| `io` | openpyxl, SQLAlchemy, pyarrow, polars | more readers |
| `excel` | xlsxwriter | Excel artefacts |
| `credit` | optbinning, scikit-learn | binning, AUC, PSI, migration |

Stage 0 put these groups into `pyproject.toml` with lower bounds that
resolve and install on Python 3.12 (2026-11-25). Two differ from the
table: `fast` is empty, because statsforecast 2.0.2 and later cap pandas
below 3 and 2.0.1 ships wheels for Python 3.12 only, so including it would
hold the whole lock to pandas 2; `breaks` installs ruptures only on Python
< 3.14, which ruptures 1.1.10 declares and has no wheels for. Each is
revisited when its extra's work starts (After 0.1.0).

### 12.3 Candidate plugin backends

Not dependencies of `dazflow`. Each would be wrapped by a separate
plugin package behind the model interface: sktime, skforecast, darts,
prophet, pmdarima, tsfresh.

### 12.4 Avoided in core

soda-core (proprietary licence), dbnomics (AGPL), docxtpl (LGPL),
great-expectations, mlflow and dvc (weight), pmdarima (build-time
Cython; its order search is easy to write, and it stays a plugin
option).

## 13 Testing and assurance

Each statistical result needs an oracle, not just a passing run.

- **Cross-implementation.** Unit-root tests, diagnostics and regressions
  compared across arch, statsmodels and scipy on the same data, and
  model forms against statsforecast where both implement them.
- **Round trips.** Every transform followed by its inverse returns the
  input, as hypothesis property tests.
- **Known answers.** Synthetic series from known processes for
  coefficient recovery, interval coverage and scenario ordering.
- **Hand-worked paths.** Anchoring, mean reversion and back-transforms on
  small examples.
- **Threshold straddles.** Every check at, just below and just above its
  threshold.
- **Artefact snapshots.** Rendered tables and findings against stored
  snapshots (syrupy), so a formatting change is a reviewed change.
- **Plugin contract tests.** A public test suite any component must pass,
  shipped as `dazflow.testing`, run on every built-in and offered to
  plugin authors.
- **Replay.** Every end-to-end test is replayed from its record and must
  match.
- **Assurance file.** `.agents/assurance.md` lists every boundary, object
  and threshold with the test that is its oracle, or the issue that tracks
  the gap; mutation testing reads which oracles are missing (D46).

## 14 Decisions

- **D1.** Name `dazflow`; repository `pedrobtz/py-dazflow`; MIT; Python
  >= 3.12.
- **D2.** Generic time-series workflow; stress testing and credit are
  applications, added through configuration and plugins.
- **D3.** No language model in the library.
- **D4.** Value is ease of use and documentation-ready artefacts, Word
  first, not statistical breadth.
- **D5.** Four layers (core, components, steps, pipeline) plus front
  ends; each layer imports only those below; steps never import each
  other.
- **D6.** Every component kind is a protocol with a registry, extended by
  decorator or by the `dazflow.plugins` entry-point group.
- **D7.** pandas is the internal format; other frame libraries convert at
  ingest through narwhals.
- **D8.** Fitted models are stored as specification and parameters, never
  pickled; versions copy the processed dataset so they can be refit.
- **D9.** File-based project folder and model registry; no server or
  database.
- **D10.** Rendering is core, not an extra.
- **D11.** RFC 0079's `data` stage is split into `ingest` and `clean`;
  `univar`, `multivar`, `project`, `perf` and `deliver` become `explore`,
  `build`, `forecast`, `diagnose` and `report`. The names are generic,
  and no step shares a name with the project folder or the `models`
  component.
- **D12.** pointblank is the data-quality engine of `clean`, and a core
  dependency (2026-11-25, maintainer: its thresholds and reports outweigh
  the heavier dependency set; settles open question 1).
- **D13.** `Dataset` holds panels as one frame with a `(segment, period)`
  MultiIndex; a single-segment dataset has a plain `PeriodIndex`
  (2026-11-25, maintainer: settles open question 7, as §5.1 described).
- **D14.** Version scheme: `pyproject.toml` holds the next release as a
  development version (`0.1.0.dev0` until 0.1.0), read at run time through
  `importlib.metadata`, as in `py-dazdiff` (2026-11-25, Stage 0: one
  source of the version, and PEP 440 orders it before 0.1.0).
- **D15.** Lower bounds of core dependencies are the oldest releases with
  Python 3.12 wheels that also satisfy pointblank and each other: pandas
  2.2.2, numpy 1.26.4, scipy 1.13, statsmodels 0.14.2, arch 7.0, narwhals
  2.8, pointblank 0.13, pydantic 2.7, PyYAML 6.0.1, great-tables 0.17,
  python-docx 1.1.2, matplotlib 3.8.4, jinja2 3.1.4, typer 0.12.3, rich
  13.7 (2026-11-25, Stage 0: the full profile's lowest-direct job tests
  them; raise one only when a stage needs a newer feature, with a reason).
  pydantic raised to 2.8 (2026-11-25, Stage 1: `ser_json_inf_nan="strings"`,
  which `StepResult.to_dict()` needs for NaN and infinite check values,
  arrived in 2.8). typer raised to 0.26.0 and rich to 13.8 (2026-11-25,
  Stage 6: typer before 0.26 imports click APIs that click 8.5 deprecates,
  so the lowest-direct job fails under `-W error`; 0.26 vendors click and
  needs rich 13.8).
- **D16.** The dataset fingerprint is the SHA-256 of the canonical JSON
  form in §5.1.1, written value by value, without the log (2026-11-25,
  Stage 1: identical across platforms, Python 3.12–3.14 and the lowest
  pandas and numpy, which the full profile checks on every leg).
- **D17.** Core objects other than `Dataset` are frozen pydantic models
  that refuse unknown fields; their validation errors are `DazflowError`
  subclasses (`ConfigError`, `DatasetError`, `GateError`, `FindingError`,
  `ResultError`), with a nested object's error reported by the outermost
  object. `DazflowError(problem, *, observed, change)` renders as three
  lines (2026-11-25, Stage 1: §4's classed-error rule, and one error type
  per object for callers to catch). Stage 2 adds `ArtefactError` (a
  `Table`, `Figure`, `Source`, `Section`, `Document` or identifier is not
  valid) and `RenderError` (an artefact cannot be written in the format
  asked for) (2026-11-25). Stage 3 adds `TransformError` (with `series`
  and `period`), `ReaderError`, its subclass `MissingExtraError` (with
  `module` and `extra`) and `IngestError` (2026-11-25). Stage 4 adds `CheckError` (a check
  cannot be evaluated as configured) and `CleanError` (a treatment names
  what the data lack) (2026-11-25). Stage 5 adds `ExploreError` (a series
  cannot be profiled, tested or screened as configured) (2026-11-25).
  Stage 6 adds `PipelineError` (a pipeline cannot be built or run as
  configured), `ProjectError` (a project folder cannot be created, read or
  written) and `RecordError` (a run record or a stored dataset cannot be
  read, is of another version or does not match its fingerprint)
  (2026-11-25). Stage 7 adds `ModelError` (a form cannot be fitted,
  predicted or refitted as asked, such as missing values, refused drivers
  or data with another fingerprint) and `EvaluationError` (a splitter,
  metric or evaluation cannot run as configured) (2026-11-25). Stage 8
  adds `BuildError` (candidates cannot be built as configured: unknown
  drivers or forms, no common sample) and `SelectError` (candidates cannot
  be ranked or chosen as configured: the benchmark, the choice, the
  scenarios) (2026-11-25). Stage 9 adds `ForecastError` (the chosen model
  cannot be forecast as configured: no chosen model, no scenarios, a
  horizon the scenarios do not cover, a correction the chain does not
  take) (2026-11-25). Stage 10 adds `DiagnoseError` (the chosen model
  cannot be diagnosed as configured: no chosen model, another dataset or
  forecast, a back-test the sample cannot hold, a base scenario or horizon
  the scenarios do not have) (2026-11-25). Stage 11 adds `ReportError` (a
  report cannot be built as configured, such as from something that is not
  a run record) (2026-11-26).
- **D18.** Registry: `Registry(kind, protocol=None)`, `@registry.register(key)`
  and `registry.add(key, obj)`; keys are non-empty strings registered once
  (`DuplicateKeyError`); a component lacking a protocol member is refused;
  `registry.get(key)` / `registry[key]` raise `UnknownKeyError` with
  `known`, the sorted keys. Entry points as in §6 (2026-11-25, Stage 1).
- **D19.** `Config`: `from_yaml(text)`, `load(path)`, `to_yaml()`,
  `dump(path)` and `json_schema()` (draft 2020-12, with `$schema`). YAML is
  read with PyYAML's safe loader, refusing repeated keys; dumped with
  fields in declaration order and every default filled in; non-finite
  floats are refused so YAML, JSON and the schema agree (2026-11-25,
  Stage 1).
- **D20.** Public names: `import dazflow as dz` exposes the core objects
  and errors at the top level (`dz.Dataset`, `dz.Config`, `dz.Gate`,
  `dz.DazflowError`, ...) and as `dz.core`; `dazflow.testing` holds
  `ComponentContract`, which a plugin's tests subclass with `registry` and
  `key` (2026-11-25, Stage 1). Stage 2 adds `dz.Table`, `dz.Figure`,
  `dz.Column`, `dz.Layer`, `dz.Source`, `dz.ArtefactIds`, `dz.STEP_NUMBERS`,
  `dz.ArtefactError` and `dz.RenderError`; `dz.render` is the rendering
  module, imported on first use so that `import dazflow` does not load
  matplotlib, great-tables and python-docx (2026-11-25, Stage 2). Stage 3
  adds `dz.transforms`, `dz.readers` and `dz.ingest` (the step module:
  `Config`, `Result`, `run`, `IDS`), `dazflow.steps.registry` (steps by
  name), and the errors `dz.TransformError`, `dz.ReaderError`,
  `dz.MissingExtraError` and `dz.IngestError` (2026-11-25, Stage 3).
  Stage 4 adds `dz.checks` and `dz.clean`, and `dz.CheckError` and
  `dz.CleanError` (2026-11-25). Stage 5 adds `dz.explore` and
  `dz.ExploreError` (2026-11-25). Stage 6 adds `dz.pipeline` (`Pipeline`,
  `Run`, `RunRecord`, `StepInputs`, `adapters`, `register_adapter`),
  `dz.project` (`Project`, `ProjectConfig`, `init`, `load`,
  `write_dataset`, `read_dataset`, `commented_yaml`), the errors
  `dz.PipelineError`, `dz.ProjectError` and `dz.RecordError`, and the
  `dazflow` console script (`dazflow.cli:main`) (2026-11-25). Stage 7 adds
  `dz.models` (`registry`, `register`, `get`, `Model`, `ModelForm`,
  `Fitted`, `FittedModel`, `Forecast`, `ModelSpec`, `ModelRecord`,
  `Sample`, `refit`, `fingerprint`, `season_length`), `dz.evaluation`
  (`splitters`, `metrics`, `Fold`, `evaluate`, `Evaluation`, the built-in
  classes), `dz.ModelError`, `dz.EvaluationError`, and in
  `dazflow.testing` the `ModelContract` and `sample_series` (2026-11-25).
  Stage 8 adds `dz.build` and `dz.select` (the step modules), `dz.models.Design`
  and `dz.models.TREND`, `dz.BuildError` and `dz.SelectError`, and in
  `dz.pipeline` `StepInputs.seed`, `StepInputs.dataset_with`,
  `StepInputs.require` and `Pipeline.run(seed=...)` (2026-11-25). Stage 9
  adds `dz.forecast` (the step module, with `adjustment`, `revert` and
  `back_transform`), `dz.readers.read_scenarios` and
  `dz.readers.ScenarioInput`, and `dz.ForecastError` (2026-11-25). Stage 10
  adds `dz.diagnose` (the step module, with `chow`, `anomalies`,
  `residual_tests` and `default_lags`) and `dz.DiagnoseError` (2026-11-25).
  Stage 11 adds `dz.report` (the step module, with `TITLES` and
  `replay_script`), in `dz.pipeline` `replay`, `compare`, `compare_records`,
  `Replay`, `Difference`, `RTOL`, `ATOL`, `StepInputs.record()` and
  `Pipeline.run(project=, environment=)`, `Project.replay()`,
  `ingest.run(..., root=)` and `dz.ReportError` (2026-11-26).
- **D21.** Artefact identifiers: `T`, `F` and `S` (findings) with the §7
  step number and a number declared per artefact in `ArtefactIds`;
  `StepResult.artefacts` is sorted by identifier (2026-11-25, Stage 2:
  numbers fixed by declaration survive reruns and creation order, which
  is what cross-references in written documents need; findings needed a
  letter of their own and `F` was taken).
- **D22.** Word tables and documents are written with python-docx, from
  the same formatted cells as HTML and Markdown; an organisation's
  reference `.docx` supplies styles, headers and page setup, and Word
  takes its fonts from it when one is given (2026-11-25, Stage 2: settles
  open question 2; great-tables has no Word export yet).
- **D23.** Artefacts are data, rendering is separate: `Table` and `Figure`
  live in core, hold no third-party object and round-trip through
  `to_dict()`; a figure is a layer specification drawn by matplotlib at
  render time, inside `matplotlib.style.context`, so rendering leaves the
  global settings alone (2026-11-25, Stage 2: §4's rule that no
  third-party object crosses a step boundary, and the run record).
- **D24.** The style file is `dazflow.render.Style`, partial overrides
  merged onto the house style by `load_style`; `Finding.key` names a
  template the style may replace; floats in a finding are written in the
  style when rendered, while `Finding.text` keeps the values as given
  (2026-11-25, Stage 2: one decimal setting must change every number, and
  rewording a finding needs no code).
- **D25.** Rendered output is reproducible: great-tables gets the
  artefact's id (`T3_1`), SVG text is drawn as paths with a fixed hash
  salt and no date or creator metadata, PNG has no software stamp, and
  figure margins are fixed in inches rather than computed by a layout
  engine. The tests compare Markdown exactly, HTML as a normalised tree
  (without great-tables' generated CSS, accessibility attributes and
  non-alignment classes, which differ between great-tables 0.17 and 1.2),
  and SVG after renumbering ids with a tolerance of 1 pt for coordinates
  and 0.005 for scales (matplotlib 3.8.4 and 3.11 measure text heights up
  to 0.5 pt apart), so one set of snapshots holds on every leg without
  raising a lower bound (2026-11-25, Stage 2).

- **D26.** A step result that carries a `Dataset` holds it in an
  excluded `dataset` field and records its `fingerprint`; `to_dict()`
  leaves the frame out and `from_dict()` rebuilds the result with
  `dataset=None` (2026-11-25, Stage 3: settles Stage 1's open point; a
  frame in the run record's JSON would duplicate the data the record
  stores beside it).
- **D27.** Transforms are named by a text form, record their resolved
  parameters in `SeriesMeta.transforms`, and invert from that text and
  the untransformed history; readers return a `Dataset` at its own
  monthly, quarterly or annual frequency, and an optional package is
  imported when used, its absence a `MissingExtraError` naming the extra
  (2026-11-25, Stage 3: the metadata field is a tuple of strings in the
  fingerprint's canonical form, so the text form needs no new format;
  importing on use lets the core job run without the extras). Box-Cox
  is computed with `expm1`/`log1p` and a second-order series below
  |lmbda| = 1e-12, because scipy 1.13's `inv_boxcox` is wrong at a
  subnormal lmbda (found by the lowest-direct run).
- **D28.** Ingest converts only to a lower frequency, by period-average
  over complete periods or by the period-end value, applies publication
  lags in target periods after conversion, and gates on series found
  and the common history of targets and drivers; the history chart is
  one figure `F1.1` holding every target (2026-11-25, Stage 3:
  interpolating to a higher frequency is a modelling choice the design
  does not make, and one figure needs no per-series identifier rule).
- **D29.** Checks run through pointblank as §7.2 describes: a check
  supplies its test units, pointblank (0.13 and later) counts the failing
  ones against the configured warning, error and critical thresholds, and
  dazflow builds the gate check and artefacts from the counts (2026-11-25,
  Stage 4: D12's engine, with the thresholds in configuration and the
  artefacts in the house style; every check needed only
  `col_vals_not_null`, `col_vals_between`, `col_vals_ge`, `col_vals_le`
  and `col_vals_gt`, so none is a plain function). The outlier scores
  are robust z-scores (median and 1.4826 MAD) of the residuals of an STL
  fit (local-constant seasonal smoother of 13 years) of monthly and
  quarterly data, or a lowess trend (21 years) of annual data. A robust
  STL fit, scaled by the plain fit's residuals, flags the gross outliers;
  each flagged period is then replaced by its fitted value and the plain
  fit repeated until the flagged set is stable, a period staying flagged
  only if it still scores above `z` when the fit sees its own value
  (2026-11-25, Stage 4: statsmodels' robust STL alone gives zero weight to
  up to a quarter of the points of a clean Gaussian series and then
  scores periods of spike-free series up to z = 15; the plain fit alone
  lets three spikes of 40 standard deviations in ten years of quarters
  raise the scale until none is flagged. Measured over 40 seeds with three spikes
  of 8 standard deviations, the procedure flags exactly the spikes in
  40/40 series of ten years of months and 37/40 of twenty years of
  quarters, with at most 2/40 clean series flagged anywhere).
- **D30.** `clean` takes the previous vintage as `run(dataset, config,
  *, previous=None)`, applies treatments in configuration order, and
  `exclude` keeps the calendar whole: excluded values become missing and
  are not counted by any check, so later steps keep their lags
  (2026-11-25, Stage 4: dropping rows would break the period index that
  differences and lags rely on).
- **D31.** `explore` reads ADF, DFGLS, Phillips-Perron and KPSS together
  by vote: a level is stationary when at least 3 of the 4 say so at the
  5% level, trying the series around a constant, around a trend, then
  each difference up to two, with the votes, level and alpha in
  configuration (2026-11-25, Stage 5: measured over 200 seeds of 200
  values, the rule reads 189 random walks as I(1), 200 AR(1) series with
  phi = 0.5 as I(0) and 200 trend-stationary series as such; requiring all
  four to agree read 92 of 100 AR(1) series as I(0), because KPSS by its
  size rejects about one stationary series in twenty). Lags are chosen by the largest
  absolute correlation on one common sample, and the screen is a
  single-driver OLS with Newey-West HAC errors. A driver is kept with the
  expected sign and a p-value at most the threshold (`<=`, so the
  threshold itself passes), and the gate fails while any configured driver
  is not kept and not overridden. The lag profiles of all drivers are one
  figure, `F3.1`, so D21 needs no per-series identifier rule (as D28 did
  for ingest): drivers are lines on one correlation axis, which share a
  scale. Arch versions disagree on a constant series (7.0's KPSS returns a
  statistic, later releases raise), so a constant series is reported as
  untestable before arch is called.

- **D32.** The pipeline passes results between steps through per-step
  adapters in the pipeline layer, registered by step name; a step without
  one is called as `run(dataset, config)` with the latest dataset
  (2026-11-25, Stage 6: the steps' signatures differ, `ingest.run(config,
  data=)`, `clean.run(dataset, config, previous=)`, and a later step needs
  a named earlier result such as `explore`; an adapter keeps each step
  callable on its own with its natural signature, while a uniform
  `run(inputs, config)` would make every step unpack a bag of inputs it
  does not own).
- **D33.** `project.yaml` holds name, description, pipeline, seed, style and
  the previous vintage; series, roles, signs and frequency stay in
  `config/ingest.yaml` (2026-11-25, Stage 6: one place per setting, as §8
  is built; see §8).
- **D34.** The run record is `RunRecord`, schema version 1, as §10.1
  lists; datasets are CSV with a JSON schema file and a checked fingerprint
  (2026-11-25, Stage 6: replay in Stage 11 reads by version, and a CSV is
  readable without dazflow).
- **D35.** Relative input paths are read from the project folder and
  recorded as written; the previous vintage is `project.yaml: previous`; a
  failed gate stops the run after that step with exit code 2 (2026-11-25,
  Stage 6: see §10).

- **D36.** Model forms are classes built from their `Config`, fitted as
  `fit(y, X)` on a `PeriodIndex`, and stored as a `ModelRecord` (format 1,
  §5.3): specification, parameters, covariance and the `Dataset`
  fingerprint of the target and drivers; residuals are `y` minus the
  one-step fitted value for every form (2026-11-25, Stage 7: metrics and
  diagnostics compare forms on one scale; the fingerprint reuses §5.1.1).
  The benchmarks are own code checked against the statsmodels models they
  equal: `naive` is ARIMA(0,1,0) and `seasonal_naive` the seasonal random
  walk, each with the maximum-likelihood `sigma2` and statsmodels'
  forecast interval, and `mean` is OLS on a constant with its t-based
  prediction interval. GLM intervals are `mean ± z sqrt(scale V(mean) +
  se²)`, clipped to [0, 1] for binomial, because statsmodels gives only
  the interval of the mean, which a new observation falls outside far more
  often than the level says; the default `scale: X2` makes the binomial
  form the fractional logit. ARIMAX residuals leave out the diffuse
  burn-in (`max(loglikelihood_burn, nobs_diffuse)`). ETS takes additive
  errors, trend and season only, the models with exact statsmodels
  intervals; multiplicative ones need simulated intervals, whose random
  state argument changed between statsmodels 0.14 and 0.15, and wait until
  a stage needs them. The optimised forms (ARIMAX, ETS) start from
  statsmodels' start values rounded to 10 significant digits: the start
  comes from a least-squares fit whose last bit depends on memory
  alignment, and without the rounding a refit of the same data on the same
  machine moved a seasonal ARIMAX by up to 7e-5, against the 1e-10 the
  roadmap asks for. Statsmodels' fit warnings (convergence, start values)
  are kept in `Fitted.notes` rather than raised.
- **D37.** Splitters return folds of positions with a full test horizon,
  the last fold ending at the last period; metrics are classes with a
  `Config` and `score(actual, forecast, ...)`; MASE divides each fold's
  errors by that fold's in-sample (seasonal) naive mean absolute error and
  averages the scaled errors (Hyndman and Koehler, 2006); rolling-origin
  evaluation lives in `dz.evaluation` and returns an `Evaluation` that is
  data (2026-11-25, Stage 7: select and a later segment share it, and a
  result crossing a step boundary holds no third-party object). Folds
  with a partial horizon are not made, so every horizon is scored on the
  same origins.
- **D38.** The default selection rule: among the candidates that pass every
  check (build's sign, significance and VIF checks, and select's benchmark,
  scenario-order and bounds checks), take those in arch's model confidence
  set (`arch.bootstrap.MCS`) at a configured size, default 0.10; choose the
  one with the fewest drivers, breaking ties by the lowest
  benchmark-scaled error (then by build's order). If no candidate that
  passes every check is in the set, the finding says so, nothing is
  proposed and the `confidence-set` check fails. The modeller confirms the
  proposal or overrides it in configuration (`choice: {candidate,
  reason}`), with a required reason. The bootstrap is seeded from the step
  configuration (`mcs.seed`) or `project.yaml: seed`, so a run repeats
  (2026-11-25, Stage 8: settles open question 6 as the maintainer decided
  it; the confidence set replaces "within a configured tolerance", which
  had no definition, with a test of which candidates the data cannot tell
  apart, and parsimony then picks among those).
- **D39.** Build takes the series in levels with explore's result and
  fits every candidate on one common sample through `models.Design`, which
  select reuses; forms are tried in configured order per driver set, the
  next only when the simpler fails; only driver coefficients are gated,
  with two-sided normal p-values from the parameters and their covariance;
  a trend-stationary series kept in levels gives its candidates a linear
  `trend` column (2026-11-25, Stage 8: one function building a candidate's
  data keeps the fitted, evaluated and forecast matrices identical, which
  `models.refit`'s fingerprint check then confirms; a common sample makes
  the rolling-origin folds and the confidence set's loss matrix line up;
  a deterministic term has no expected sign, and explore's open point left
  the trend term to build).
- **D40.** Select scores every fitted model on the same folds, drops a
  candidate that cannot be fitted at some origin (with its error) rather
  than that origin, scales errors by a benchmark (default `mean`), reads
  scenarios from `config/select.yaml: scenarios` when the dataset has none,
  maps projections to the target's levels by the recorded chain, and
  counts a tie between scenarios as in order (2026-11-25, Stage 8: a
  dropped origin would leave candidates with different origins and no
  common loss matrix; on the stationary series build fits, the mean is the
  no-information forecast, and in levels it is a random walk with drift;
  §8 keeps scenarios out of `project.yaml` for 0.1.0 and ingest has no
  scenario input, so the step that first uses them reads them; equal
  driver paths give equal projections, which are not out of order). The
  example project's `data/raw/scenarios.csv` holds illustrative paths
  written for the example data, not a published scenario.
- **D41.** Forecast anchors on the model's scale (default `decay`, 0.5),
  takes forecasts back through the recorded chain with a correction chosen
  by the chain (none for differences, smearing for a log or Box-Cox,
  seeded simulation for a logit or a nonlinear chain with a lag), reverts
  linearly on the target's scale with the interval shifted not narrowed,
  gates on the gap at the origin and on bounds that default to select's,
  and reads select's scenarios through a shared reader (2026-11-25, Stage
  9: an add factor on the equation is the jump-off adjustment the
  modelling literature means, and on a differenced target `carry` would
  add the gap every period, so the default fades it; a correction a chain
  does not need is refused rather than run; Duan's smearing needs a
  value-by-value inverse, and a lag makes the level error a sum of
  horizons, which only paths capture; common random numbers keep the
  scenarios' order free of simulation noise; one setting for scenarios and
  for bounds, as D33 asks, and steps import no other step).

- **D42.** Diagnose runs the regression-based tests on a statsmodels OLS of
  the chosen design and only for the `ols` form, recording every other
  test as not applicable with its reason and outside the gate; reads every
  test as one threshold check from configuration; tests Chow at
  configured breaks or at fixed fractions of the sample (the middle by
  default) rather than at a date the data suggest; takes the CUSUM bounds
  from Brown, Durbin and Evans' table; back-tests through
  `dz.evaluation.evaluate`; shocks a driver's model column by one standard
  deviation by default; and flags anomalies outside fitted ± z·s at 99%
  (2026-11-25, Stage 10: BG, RESET and the recursive tests need an OLS
  result and the others assume one, and a test run on a form it does not
  describe would fail or pass for the wrong reason; a break date chosen by
  looking at the data makes the Chow F a sup-F, whose p-value the F
  distribution does not give, and one fixed date keeps the gate's size at
  alpha; statsmodels' private bound function uses 0.950 for 10%, where the
  table has 0.850; select and segment already share `evaluate`, so a second
  back-test loop would drift from it; a standard deviation makes drivers in
  different units comparable, and a unit shock, the coefficient, is one
  setting away; at 99% a well-specified model flags about one observation
  in a hundred, well under the 5% maximum).

- **D43.** Replay compares floats within a relative tolerance of 1e-9 with
  an absolute floor of 1e-12 (`math.isclose`), and everything else exactly:
  integers, choices, identifiers, statuses, text and fingerprints; both
  are `config/report.yaml: replay` settings, so the tolerance is recorded
  and the replay script carries it (2026-11-26, Stage 11: measured on the
  example, a rerun on the same machine moves 126 of its about 5,900 recorded
  floats, the largest by 2.6e-14 relative (an explore p-value; select's and
  diagnose's numbers move by about 1e-15, Stages 8 and 10), and a refit of an
  optimised form agrees to 1e-10 on parameters of order 0.1 or more (D36),
  so 1e-9 sits more than four orders above the noise. Changing one input
  value by its last printed digit (2.009 to 2.010) moves 5,460 floats by more
  than 1e-9, with a median of 2e-4, and changes every dataset fingerprint,
  which is compared exactly; the rest are floats the change does not reach. The
  floor is for values that are zero up to rounding, a residual mean or a
  sum of signed errors, whose relative difference means nothing; it takes
  over below |x| = 1e-3. Stage 7 asked for a cross-machine tolerance for
  ARIMAX and ETS set from their standard errors; it is not set here,
  because no CI leg replays a record written on another machine, and a
  number chosen without one would be untested. A cross-machine replay of
  an optimised form passes `--rtol` explicitly; the full CI stage (14)
  measures it.)
- **D44.** The report step is built from the run record alone and keeps
  the replay out of its own gate: its checks are `complete` and `gates`,
  and the replay that §7.10 names as the gate is `dazflow replay`, exit 0
  or 2, which the full profile's `package` job runs on the installed wheel
  (2026-11-26, Stage 11: steps sit below the pipeline and import no other
  step, so the step cannot rerun the pipeline; doing it in the adapter
  would double every run, and a record holding a report whose gate is its
  own replay would replay itself recursively. The document and replay
  script are pure functions of the record, so a replay reruns the report
  from the replayed record and compares it like any other step.)
- **D45.** The documentation site is great-docs 0.17.0 on Quarto 1.11.9,
  as in `py-dazdiff`, run through `uv run --with` and not a dependency of
  the package; it is deployed to GitHub Pages from `main` by the `Docs`
  workflow and named `Documentation` in `project.urls`. The home page is the
  README. The API reference lists every name in a public module's
  `__all__` once (`tests/test_docs.py`), grouped by layer; an alias
  (`ingest.Config`, `select.ScenarioInput`) is documented under its long
  name, and a step's names under `<step>.<name>`, as `dz.<step>` exposes
  them. Guide pages are not executed at build time: their output is pasted
  from real runs, and `tests/test_guide.py` runs their Python in the quick
  profile, on a copy of `examples/loss-rate` (2026-11-26, Stage 12: the
  build then needs neither the extras nor a project folder, and a page that
  stops working fails the tests rather than rendering an error. The plugin
  page is a contract: its module is installed through its own entry point,
  its configuration is merged into the example, and its `ModelContract`
  subclass must pass. The `Docs` workflow fails on a docstring the numpy parser
  misreads and on a `great-docs.yml` great-docs cannot parse, which it
  would otherwise only warn about.)

- **D46.** Assurance (2026-11-26, Stage 13). `.agents/assurance.md`
  holds one row per boundary, long-lived object and threshold, each naming
  the test that is its oracle or a `gap #<issue>`; it keeps the portfolio
  template's headings, with the R-only checks left out (the package is pure
  Python) and `fuzz` and `mutation` met by this repository's own workflows.
  Mutation testing is mutmut 3.8 (dev group), configured in
  `pyproject.toml`: it mutates `core/` (the gate arithmetic included),
  `transforms/` and `evaluation/`, and runs only `tests/core`,
  `tests/transforms` and `tests/evaluation`, with `-p no:randomly` and
  without xdist, under hypothesis' derandomised `mutation` profile; the
  `Mutation` workflow runs it weekly and on demand, never on a pull
  request, and gates nothing. The gate arithmetic is plain functions
  (`_passes`, `_status`, `_with_status`) called by the `Check` and `Gate`
  properties. Hypothesis has three profiles chosen by `HYPOTHESIS_PROFILE`:
  `quick` (100 examples, the quick job and the default), `full` (1000, the
  full profile's test jobs) and `mutation`; a test that sets its own
  `max_examples` keeps it. (mutmut copies the package to `mutants/` and
  forks one process per mutant, so the unit tests of the mutated layers
  keep a run under an hour, where the whole suite reruns the example
  project for every mutant of `Dataset`; random test order and random
  examples would turn a survivor into a flake; mutmut skips decorated
  functions, so arithmetic inside a `@property` would never be mutated; a
  survivor is a question about a test, which a red pull request would
  not answer. The first run's survivors were read: those that meant a
  missing oracle got a test, the rest are tracked in the assurance file.)

## 15 Build order

The order of work is in [roadmap.md](roadmap.md).

## 16 Later

Not designed here; the core is shaped so these add no new logic. Order:
command line, then MCP server, then forms interface.

- **MCP server:** tools generated from the step functions and their JSON
  Schemas, returning the same JSON as `--json`.
- **Forms interface:** forms from the same schemas, artefacts as HTML.
- **Agent layer:** proposes configuration and decisions; the library
  computes every number.

## 17 Open questions

1. **Quality engine.** Settled by D12: pointblank.
2. **Word tables.** Settled by D22: python-docx now; great-tables' Word
   export is reconsidered once it is released.
3. **Systems backend.** Confirm statsmodels `VECM` covers the
   cointegrated cases needed, or defer systems.
4. **Segment order.** Models may need building per segment from
   `explore` on, not only after `build`.
5. **Monitoring.** Anomaly checks appear in `clean` and `diagnose`.
   Monitoring could become its own step.
6. **Default selection rule.** Settled by D38: the candidates that pass
   every check and are in arch's model confidence set, then the fewest
   drivers, then the lowest benchmark-scaled error.
7. **Panel data shape.** Settled by D13: a `(segment, period)`
   MultiIndex in one frame.
