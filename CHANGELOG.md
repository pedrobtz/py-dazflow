# Changelog

## 0.1.0 (2026-11-26)

The first release. `dazflow` runs the workflow around a time-series model
in eight steps, each returning its numbers with tables, figures and
findings ready for a model document. It needs Python 3.12 or later.

### Steps

- `dz.ingest` reads csv, Excel and Parquet files or pandas, polars and
  pyarrow frames, puts every series on one monthly, quarterly or annual
  calendar, applies publication lags and checks the history is long
  enough.
- `dz.clean` runs seven data-quality checks on pointblank (`completeness`,
  `gaps`, `duplicates`, `range`, `revisions`, `outliers`,
  `representativeness`), applies the configured treatments, each with a
  reason, and logs every changed value.
- `dz.explore` profiles every series, reads ADF, DFGLS, Phillips-Perron
  and KPSS together into an order of integration, transforms each series,
  searches each driver's lag and screens it by a HAC regression against
  its expected sign.
- `dz.build` checks driver correlation and variance inflation, searches
  driver combinations (exhaustive or stepwise) and fits the configured
  forms simplest first, always with the benchmarks.
- `dz.select` scores the candidates by rolling-origin evaluation, computes
  a model confidence set, checks scenario projections for ordering and
  bounds, and proposes a default that the modeller confirms or overrides
  in configuration, with a reason.
- `dz.forecast` forecasts the chosen model under every scenario with
  intervals, anchors the origin, optionally reverts to a long-run level,
  and transforms back to the target's scale with a suitable bias
  correction.
- `dz.diagnose` tests the chosen model: residual tests, CUSUM and
  recursive CUSUM, rolling coefficients, Chow tests, a back-test of
  interval coverage, one-driver sensitivity and anomalies.
- `dz.report` builds the run's Word document and a plain-Python replay
  script from the run record alone.

### Workflow

- Gates: every step ends in checks against configured thresholds. A
  failed check stops the run unless an override with a reason accepts it.
- The `dazflow` command: `init` (a project folder with every setting
  written out at its default and example data that runs), `validate`, one
  command per step, `run --from/--to`, `show <artefact-id>`, `replay`,
  `plugins` and `--version`. Exit 0 on success, 1 on an error, 2 on a
  failed gate or a replay that differs.
- The run record, `reports/record.json`: configuration with every
  default, results, checks, overrides, environment and the fingerprint of
  every input. `dazflow replay` reruns it and names every difference.
- Artefacts with stable identifiers (`T3.3`, `F3.1`, `S3.1`) render in
  one house style to Word, HTML, Markdown, PNG and SVG; an organisation
  can override the style from a YAML file.

### Components

- Model forms behind one interface: `naive`, `seasonal_naive`, `mean`,
  `ols`, `glm` (fractional logit), `arimax` and `ets`. Fitted models are
  stored as specification, parameters, covariance and data fingerprint,
  never pickled, and refitted from the record.
- Transforms (`diff`, `seasonal_diff`, `log`, `logdiff`, `growth`,
  `logit`, `boxcox`) with their inverses, recorded as text in the series
  metadata.
- Rolling-origin evaluation (`expanding` and `rolling` splitters; `mae`,
  `rmse`, `mase`, `coverage`).
- Plugins: model forms, checks, transforms, readers, renderers and steps
  register under a key, by decorator or through the `dazflow.plugins`
  entry point. `dazflow.testing` holds the contracts a plugin passes.

### Documentation

- <https://pedrobtz.github.io/py-dazflow/>: Get started on the example
  project, the command line, writing a plugin, and a reference of every
  public name.
