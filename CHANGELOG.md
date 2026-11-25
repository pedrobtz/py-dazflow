# Changelog

## Unreleased

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
