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
unexecuted Get started page. No `pyproject.toml`, source, tests or CI yet.

## Versions and dependents

`v0.1.0` issue: see the roadmap header. Dependents: none.

## Commands

None yet. The layout follows `py-dazdiff`: `src/dazflow/`, hatchling,
uv, ruff and pytest, so once `pyproject.toml` exists the loop is
`uv sync`, `uv run pytest`, `uv run ruff check .` and
`uv run ruff format .`.

## Architecture

To be written as stages land; see `.agents/design.md` §4 for the layers
(core, components, steps, pipeline, front ends) and the package layout.

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

None yet. The design's §13 lists the oracles each result needs, and a
public contract suite, `dazflow.testing`, that every component passes.

## Continuous integration

None yet. Expected to mirror `py-dazdiff`: a `quick` profile on pull
requests and a `full` profile on `main` and on pull requests labelled
`full-ci`.
