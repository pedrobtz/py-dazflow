# Changelog

## Unreleased

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
