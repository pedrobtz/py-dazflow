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

### 5.3 Fitted models

A fitted model is stored as its specification, parameters, covariance
and the fingerprint of the data it was fitted on. It is never pickled.
Replay and version loading refit and compare (§10).

## 6 Extension

Every component kind is a `typing.Protocol` with a registry. Built-ins
register themselves. Users register theirs with a decorator, and
third-party packages through the `dazflow.plugins` entry-point group,
which is loaded on first use. Configuration names a component by its
registry key, so a plugin model is chosen in YAML exactly like a
built-in one.

| Kind | Protocol (essential methods) | Built-ins |
| --- | --- | --- |
| Reader | `read(spec) -> Dataset` | csv, excel, parquet, sql |
| Check | `evaluate(obj) -> Check` | completeness, gaps, range, sign |
| Transform | `forward`, `inverse` | diff, logdiff, log, logit, boxcox |
| Model | `fit(y, X) -> Fitted` | naive, mean, OLS, GLM, ARIMAX, ETS |
| Metric | `score(actual, forecast)` | MAE, RMSE, MASE, coverage |
| Splitter | `split(index) -> folds` | expanding, rolling, holdout |
| Renderer | `render(artefact, path)` | docx, html, markdown, png, xlsx |
| Step | `run(inputs, config) -> StepResult` | the ten steps of §7 |

`Fitted` has `predict(h, X_future) -> Forecast`, `params`, `cov_params`,
`residuals` and `spec`. Seasonal naive, VAR and VECM are model forms too.
`Forecast` holds the mean path and intervals at configured levels, per
scenario when there are scenarios.

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
like any component and placed in the pipeline by name. Wrapper
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

## 8 Configuration

- One pydantic model per step and per component. YAML on disk, Python
  objects in memory, JSON Schema for tools.
- `project.yaml` holds the name, target, drivers with expected signs,
  segments, scenarios, frequency, seed and the pipeline. `config/` holds
  one file per step, named after the step.
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

### 10.1 Run record

`reports/record.json` holds every step's configuration as run, the
results, checks, decisions and overrides, the environment (Python,
platform, package versions) and a fingerprint for every input. Processed
series are stored as CSV with a declared schema; Parquet is an option for
large panels.

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
2. **Word tables.** Use great-tables' Word export once released, or keep
   a python-docx table renderer permanently.
3. **Systems backend.** Confirm statsmodels `VECM` covers the
   cointegrated cases needed, or defer systems.
4. **Segment order.** Models may need building per segment from
   `explore` on, not only after `build`.
5. **Monitoring.** Anomaly checks appear in `clean` and `diagnose`.
   Monitoring could become its own step.
6. **Default selection rule.** An exact definition of "within a
   configured tolerance".
7. **Panel data shape.** Settled by D13: a `(segment, period)`
   MultiIndex in one frame.
