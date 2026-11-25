# Changelog

## Unreleased

- `dz.forecast`: the sixth step. It refits select's chosen model and
  forecasts it from the origin (the last period of its sample) under
  every scenario, with intervals. The gap between observed and fitted at
  the origin is anchored by a configured rule (`carry`, `decay` with a
  rate, or `none`); an optional reversion moves the path linearly towards
  a long-run level. Forecasts come back to the target's scale through the
  recorded transform chain with a bias correction suited to it: smearing
  for a log or Box-Cox, a seeded simulation for a logit or a nonlinear
  chain with a lag, none for differences (whose intervals add the
  horizons' variances). The gate checks that the forecast reconciles to
  the origin within a tolerance and stays within the bounds (select's
  unless configured). Scenarios are select's, read from the same
  `config/select.yaml` setting. Artefacts: a forecast table by scenario
  and horizon (`T7.1`), the projection record (`T7.2`), the gate (`T7.3`),
  a fan chart (`F7.1`) and a finding (`S7.1`). `adjustment()`, `revert()`
  and `back_transform()` are public. New error `ForecastError`.
- `dz.readers.read_scenarios` and `dz.readers.ScenarioInput`: the scenario
  reader moved out of select so forecast can share it; select's
  `read_scenarios` and `ScenarioInput` still work, and still raise
  `SelectError`.
- The default pipeline ends at forecast, and `dazflow init` writes
  `config/forecast.yaml`.

- `dz.build`: the fourth step. It checks the drivers' correlation and
  variance inflation, searches driver combinations up to a size
  (exhaustively, or forward stepwise by AIC or BIC) without the excluded
  pairs, and fits the configured forms simplest first for each set,
  moving to the next form only when the simpler one fails a check: every
  driver has its expected sign and a p-value at most the threshold, and no
  VIF exceeds the limit. The benchmarks are always fitted, every candidate
  on one common sample. A trend-stationary series kept in levels gives its
  candidates a trend term. It returns the candidates with their stored
  records, a correlation table, a candidate table, a coefficient table and
  chart, and a finding. `combinations()`, `stepwise()` and `vif()` are
  public. New error `BuildError`.
- `dz.select`: the fifth step. It scores every candidate and benchmark by
  rolling-origin evaluation, scales the errors by a benchmark's, computes
  arch's model confidence set from a seeded bootstrap, forecasts each
  candidate under every scenario (from `Dataset.scenarios` or a CSV named
  in `config/select.yaml`) and checks the projections, in the target's
  levels, for ordering by severity and for bounds. It proposes a default:
  among the candidates that pass every check and are in the confidence
  set, the one with the fewest drivers, then the lowest scaled error. The
  modeller confirms or overrides it in configuration (`choice`), with a
  reason. Artefacts: ranking, evaluation, scenario and selection tables, a
  back-test chart, a scenario chart and a finding. `confidence_set()`,
  `default_rule()`, `out_of_order()` and `read_scenarios()` are public. New
  error `SelectError`.
- `dz.models.Design`: how a candidate's target and drivers are built from
  the series in levels (transform chains, lags, a trend term, the sample),
  shared by build and select.
- The default pipeline is `ingest, clean, explore, build, select`, and
  `dazflow init` writes `config/build.yaml`, `config/select.yaml` and
  illustrative scenarios for the example data. `project.yaml: seed` now
  reaches the steps (`StepInputs.seed`).
- A fitted model computes its data fingerprint when first asked for it,
  which makes a rolling-origin evaluation about twice as fast.

- `dz.models`: model forms behind one interface, `fit(y, X)` returning a
  fitted model with `params`, `cov_params`, `residuals`, `spec` and
  `predict(h, X_future, levels)`, which returns a `Forecast` (mean path
  and prediction intervals). Forms: `naive`, `seasonal_naive`, `mean`,
  `ols`, `glm` (binomial for shares, the fractional logit), `arimax`
  (statsmodels `SARIMAX`) and `ets` (statsmodels `ETSModel`, additive).
  A fitted model is stored as a `ModelRecord` (specification, parameters,
  covariance, data fingerprint), never pickled, and `models.refit` fits it
  again from the record on the same data.
- `dz.evaluation`: `expanding` and `rolling` splitters, the `mae`, `rmse`,
  `mase` and `coverage` metrics, and `evaluate`, the rolling-origin
  evaluation that refits a form at successive origins and scores its
  forecasts overall and by horizon.
- `dazflow.testing.ModelContract`: the contract every model form passes,
  for plugin authors' tests.
- New errors `ModelError` and `EvaluationError`.

- The `dazflow` command: `init` (a project folder with every default
  written out and commented, and example data that runs), `validate`
  (configuration and inputs, without running a step), one command per step
  (`ingest`, `clean`, `explore`, and any step a plugin registers), `run
  --from/--to`, `show <artefact-id>`, `plugins` and `--version`. Every
  command takes `--project` and `--json`. Exit codes: 0 success, 1 error,
  2 a step completed with a failed gate; `run` stops after that step.
- `dz.pipeline`: `Pipeline` runs steps by name in configured order,
  passing earlier results through per-step adapters
  (`register_adapter`); `Run` holds what one pass returned, without writing
  anything; `RunRecord` is the run record (schema version 1): configuration
  with every default, results, checks, overrides, environment and a
  fingerprint of every input.
- `dz.project`: the project folder. `Project.run()` reads it,
  `Project.write(run)` writes the datasets (CSV with a schema file and a
  checked fingerprint), each step's tables, figures and finding under
  `reports/<step>/` and `reports/record.json`. `project.yaml` names the
  pipeline, a style file and the previous data vintage for clean's
  revisions check.
- `examples/loss-rate`: the example project, as `dazflow init` writes it.
- New errors `PipelineError`, `ProjectError` and `RecordError`.
- typer is required at 0.26.0 or later and rich at 13.8 or later.

- `dz.explore`: the third step. It profiles every series (trend and
  seasonal strength from STL, autocorrelation, volatility, intermittency),
  runs ADF, DFGLS, Phillips-Perron and KPSS from arch and reads them
  together into an order of integration, transforms each series by its
  configured chain and the differences that order calls for, searches each
  driver's lag against the target and screens it by a single-driver OLS
  with HAC standard errors. The gate keeps a driver with its expected sign
  (from configuration or the series metadata) and a p-value at most the
  configured threshold; a driver that fails passes only by an override
  with a reason. It returns the transformed dataset with a profile table,
  a stationarity table, a driver screen, a lag-profile chart and a summary
  finding. The statistics are public: `profile()`, `unit_root()`,
  `integration_order()`, `lag_search()` and `screen()`. New error
  `ExploreError`.
- `dz.checks`: seven data-quality checks, registered and configured with
  pointblank's warning, error and critical thresholds: `completeness`,
  `gaps`, `duplicates` (a value repeating the previous one), `range`,
  `revisions` (against a previous vintage), `outliers` (robust z-scores
  of STL residuals; `outlier_scores()`) and `representativeness` (the
  history covers configured downturns). `evaluate()` runs one check on a
  series and returns its outcome: the periods tested and flagged, the
  level reached and the gate check.
- `dz.clean`: the second step. It runs the checks, applies the configured
  treatments (`fill`, `interpolate`, `winsorise`, `replace`, `exclude`;
  each needs a reason), logs every changed value with its reason, runs the
  checks again and gates on the result. It returns the cleaned dataset
  with a quality report, a treatment table, a history chart marking the
  flagged periods and a summary finding. New errors `CheckError` and
  `CleanError`.
- `dz.transforms`: `diff`, `seasonal_diff`, `log`, `logdiff`, `growth`,
  `logit` and `boxcox`, each with its inverse, named by a text form such
  as `"diff(lag=4)"`. `forward()` returns the chain with its parameters
  resolved, `apply()` records it in the series metadata, and `inverse()`
  undoes it from the untransformed history, in sample or past its end.
  A value outside a transform's domain raises `TransformError` naming the
  series and the period.
- `dz.readers`: csv, Excel and Parquet files (the last two with the
  `dazflow[io]` extra; without it, `MissingExtraError` says what to
  install) and pandas, polars or pyarrow frames, into a `Dataset` on
  monthly, quarterly or annual periods read from the dates.
- `dz.ingest`: the first step. It reads the inputs, puts every series on
  one calendar by period-average or period-end, applies publication lags,
  checks that every series was found and that the history is long
  enough, and returns the dataset with a series inventory, a history
  chart of the targets and a summary finding.
- Artefacts: `Table` (labelled columns with units, unformatted cells,
  `from_frame()` and `from_gate()` with pass, fail and overridden marked in
  place), `Figure` (line, scatter, bar, band and reference-line layers),
  `Source` (run, dataset fingerprint and date for the source note) and
  `ArtefactIds`, which gives `T<step>.<n>`, `F<step>.<n>` and `S<step>.<n>`
  identifiers that do not change with creation order.
  `StepResult.artefacts` now holds tables, figures and findings, sorted by
  identifier, and a finding's key must be an `S` identifier.
- `dazflow.render`: `render()` and `write()` to Word (python-docx), HTML
  (great-tables), Markdown, PNG and SVG; `Section` and `Document`; the
  house style (`Style`, a `.mplstyle` for figures) with an organisation's
  override from a YAML file or mapping (`load_style()`), including finding
  templates by `Finding.key`. Renderers are registered components.
- `Finding` has an optional `key`; new errors `ArtefactError` and
  `RenderError`.
- Core objects: `Dataset` (an immutable frame of series with metadata,
  scenarios, a change log and a platform-independent `fingerprint()`),
  `Config` (YAML and JSON Schema), `Check`, `Gate` and `Override` (an
  override needs a reason), `Finding` (template text filled with numbers),
  `StepResult` with a strict-JSON `to_dict()`, `Registry` with
  `dazflow.plugins` entry points, and `DazflowError` with its subclasses.
- `dazflow.testing.ComponentContract`, the start of the contract suite for
  components and plugins.
- Requires pydantic 2.8 or later.
- First package skeleton: `import dazflow` works and `dazflow.__version__`
  reports the installed version. Requires Python 3.12 or later.
