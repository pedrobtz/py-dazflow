# Changelog

## Unreleased

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
