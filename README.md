# dazflow

[![CI](https://github.com/pedrobtz/py-dazflow/actions/workflows/ci.yml/badge.svg)](https://github.com/pedrobtz/py-dazflow/actions/workflows/ci.yml)
[![Docs](https://github.com/pedrobtz/py-dazflow/actions/workflows/docs.yml/badge.svg)](https://pedrobtz.github.io/py-dazflow/)

`dazflow` is a Python library and command-line tool for the workflow around
a time-series model: import, clean, explore, build candidates, select,
forecast, diagnose and report. Each step returns its numbers together with
tables, figures and findings ready for a model document. Documentation:
<https://pedrobtz.github.io/py-dazflow/>.

- **Steps from configuration.** A project folder holds one YAML file per
  step, with every setting written out at its default.
- **Gates, never hidden.** Every step ends in checks against configured
  thresholds; a failed check stops the run unless an override with a reason
  accepts it.
- **Artefacts with stable identifiers.** Tables and figures (`T3.3`,
  `F3.1`) render in one house style to Markdown, HTML, SVG and Word.
- **A run record and a replay.** Every run writes a record of its
  configuration, inputs, environment and results; `dazflow replay` reruns
  it and compares.
- **Plugins.** Model forms, checks, transforms, readers and steps register
  under a key, by decorator or through the `dazflow.plugins` entry point.

The statistics come from statsmodels, arch, scipy and their peers; `dazflow`
adds the shared data object, configuration, gates, record and rendering.

## Installation

```sh
pip install dazflow
```

It needs Python 3.12 or later. The development version installs from
GitHub: `pip install git+https://github.com/pedrobtz/py-dazflow`. Optional
extras (`io`, `breaks`, `panel`, ...) add readers and methods.

## A first run

```sh
dazflow init loss-rate      # a project folder with example data
cd loss-rate
dazflow validate            # configuration and inputs, without running a step
dazflow run                 # ingest through report; exit 2 if a gate fails
dazflow show T3.3           # one table of the last run
dazflow replay              # rerun from reports/record.json and compare
```

The same from Python:

```python
import dazflow as dz

project = dz.project.load("loss-rate")
run = project.run()  # in memory; writes nothing
record = project.write(run)  # reports/, data/, reports/record.json
project.replay().passed
```

[Get started](https://pedrobtz.github.io/py-dazflow/user-guide/get-started.html)
walks through the example; the
[reference](https://pedrobtz.github.io/py-dazflow/reference/) lists every
public name.

## Status

Pre-release: the package is being built towards 0.1.0. The design is in
[`.agents/design.md`](https://github.com/pedrobtz/py-dazflow/blob/main/.agents/design.md)
and the order of work in
[`.agents/roadmap.md`](https://github.com/pedrobtz/py-dazflow/blob/main/.agents/roadmap.md).

## License

MIT
