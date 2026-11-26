# Assurance

How this package meets the shared `assurance.md` rules, adapted for Python
(design §13, D46). The template is `templates/assurance.md` in
`pedrobtz/packages`; the headings and table columns are kept so the
portfolio's check can read the file.

- **Statuses:** a cell is a fact, `gap #<issue>` or `n/a: <reason>`.
- **Python, not R.** The `compiled` checks (sanitizers, valgrind, rchk,
  gctorture, alloc-failure) have nothing to run on: the package is pure
  Python. `fuzz` is hypothesis, in pytest; `mutation` is mutmut, on a
  schedule. Both run in this repository's own workflows.
- **Oracle** names what knows the answer; **Carried by** names the test
  that compares against it. A test that only runs the code is not one.

## Kinds

`untrusted-input` `spec` `lifecycle` `emits`

Evidence, one line per kind (no code marks here: the check reads every
one in this section as a kind):

- untrusted-input: the readers parse CSV, Excel and Parquet files and
  frames the package did not produce.
- spec: published algorithms and formats: the Box-Cox transform, MASE,
  Brown, Durbin and Evans' CUSUM bounds, the fingerprint's canonical form
  (design 5.1.1), the run record's schema (design 10.1).
- lifecycle: the project folder, a run and the plugin registries keep
  state between calls.
- emits: Word documents, SVG and PNG figures, Markdown and HTML tables,
  the run record, dataset CSV with its schema file, and the replay script.

## Required checks

`Where` names the workflow file under `.github/workflows/` that runs the
check. `(own)`: the workflow is this repository's, not a call to r-actions.

| Check      | Status | Where                                                                       |
| ---------- | ------ | --------------------------------------------------------------------------- |
| fuzz       | done   | ci.yml (own): hypothesis, `tests/readers/test_malformed.py` and the round trips |
| fuzz-reach | gap #38 |                                                                            |
| mutation   | done   | mutation.yml (own): mutmut on core/, transforms/, evaluation/, weekly       |

## Boundaries

One row per boundary the package crosses, in each direction it crosses it.

<!-- markdownlint-disable MD013 -->
| Boundary | Oracle | Carried by |
| --- | --- | --- |
| CSV reader: dates to periods | the specification's date rules (`2024-03-31`, `2024-03`, `2024Q1`, `2024`, month-end and quarter-start dates) listed by hand | tests/readers/test_csv.py |
| CSV reader: malformed input (random bytes, ragged rows, bad dates, mixed types) | the reader's contract: a `Dataset` of sorted unique periods at one frequency, or a `DazflowError` subclass; for mixed types, pandas' own reading of the same file cell by cell | tests/readers/test_malformed.py |
| dates outside pandas 2's timestamp range | a `ReaderError`, on pandas 2 and 3 (#37) | tests/readers/test_csv.py::test_37_dates_outside_pandas_nanosecond_range_are_a_reader_error |
| Excel and Parquet readers | the frame written by pandas (openpyxl, pyarrow), read back equal | tests/readers/test_extras.py |
| polars and pyarrow input | the same frame given as pandas: equal datasets | tests/readers/test_extras.py::test_polars_input_goes_through_narwhals |
| scenario file reader | paths listed by hand, in file order | tests/readers/test_scenarios.py |
| scenarios of unequal length into select | the common cover counted by hand: every scenario forecast over the same periods, `last` compared at one period (D49) | tests/steps/test_select.py::test_scenarios_of_unequal_length_are_forecast_over_their_common_horizon, tests/steps/test_select.py::test_the_scenario_cover_counts_contiguous_periods_after_the_end |
| select `choice` of a candidate that could not be evaluated | a `SelectError` naming the candidate and the evaluation's error, never a check passed on a value not computed (#47) | tests/steps/test_select.py::test_a_choice_that_could_not_be_evaluated_is_refused_with_its_reason |
| select candidates with identical per-origin losses | arch's MCS on the losses with the duplicates dropped: each duplicate takes its twin's inclusion and p-value (D50, #54); a near tie goes to arch unchanged; a failure of arch's computation is a `SelectError` | tests/steps/test_select.py::test_identical_losses_share_one_place_in_the_confidence_set, tests/steps/test_select.py::test_a_duplicated_candidate_completes_select, tests/steps/test_select.py::test_an_mcs_that_arch_cannot_compute_is_a_select_error |
| wheel marked as typed (PEP 561) | `py.typed` in the installed package and the `Typing :: Typed` classifier (#52) | tests/test_package.py::test_the_package_is_marked_as_typed |
| transforms: forward then inverse | round trip to the input, hypothesis over each domain | tests/transforms/test_roundtrip.py |
| Box-Cox transform | its definition in 50-digit decimal arithmetic; the estimated lambda equals scipy's maximum likelihood estimate | tests/transforms/test_roundtrip.py::test_boxcox_matches_its_definition_in_50_digit_arithmetic, tests/transforms/test_transform_oracles.py |
| model forms against statsmodels | the direct statsmodels call on the same data (OLS, GLM, SARIMAX, ETSModel), rtol 1e-10 | tests/models/test_wrappers.py |
| model forms on the lowest dependencies | the same direct calls on drivers of the same layout; a row-major and a column-major copy of the drivers fitted to equal bits (#16, D47) | tests/models/test_wrappers.py::test_a_fit_depends_on_the_drivers_values_not_their_memory_layout |
| model records and refit | the stored parameters, refitted to 1e-10 | tests/models/test_refit.py |
| metrics and splitters | hand-worked values (MASE 9/14) and fold lists by hand | tests/evaluation/test_metrics.py, tests/evaluation/test_splitters.py |
| rolling-origin evaluation | a direct statsmodels fit at each origin | tests/evaluation/test_rolling.py |
| diagnose tests | the direct statsmodels calls; CUSUM bounds from Brown, Durbin and Evans' table | tests/steps/test_diagnose.py |
| run record and its schema version | the step schemas (jsonschema) and a load back equal; another schema version refused | tests/pipeline/test_record.py |
| replay and its tolerance | a rerun of the example equal to its record; numbers moved just inside and just outside 1e-9 | tests/e2e/test_replay.py, tests/pipeline/test_compare.py |
| replay across machines (ARIMAX, ETS) | a record written on one CI leg replayed on another; measured by hand at Stage 14, no tolerance set (D47) | gap #44 |
| dataset fingerprint | stored SHA-256 values of fixtures, and the documented canonical form hashed by hand | tests/core/test_fingerprint.py |
| dataset CSV and schema file | written then read back with the same fingerprint, and read by the csv reader | tests/project/test_project.py |
| Word document | read back with python-docx: headings, captions, cells, marks, source notes | tests/render/test_docx.py, tests/e2e/test_document.py |
| Markdown and HTML tables | stored snapshots: Markdown exact, HTML as a normalised tree | tests/render/test_snapshots.py |
| SVG figures | stored snapshots within one point; the step figures from the example's run | tests/render/test_figures.py, tests/render/test_step_figures.py |
| SVG step figures on matplotlib 3.8.4 | the same snapshots; four are skipped there | gap #42 |
| replay script | run as a subprocess: exit 0 on the record, non-zero after an input changed | tests/e2e/test_replay_script.py |
| CLI exit codes | the design's codes 0, 1 and 2, and the files a run leaves | tests/cli/test_exit_codes.py |
| CLI JSON output | the library's own results and the record file | tests/cli/test_json.py |
| CLI output of findings | the finding sentence of `reports/<step>/S*.md`; the style's decimals (0.394 at `decimals: 3`); no float with more digits in `dazflow run` | tests/cli/test_findings.py, tests/render/test_finding_text.py |
| plugin entry points | a real installed distribution (`*.dist-info` on `sys.path`) found by key; a broken one named in a `PluginError` | tests/core/test_registry.py, tests/core/test_core_oracles.py |
| plugin contract suites | a good component passes, each broken one fails; every built-in passes | tests/test_testing.py, tests/models/test_contract.py, tests/checks/test_contract.py |
| CLI with a broken plugin | a real installed broken distribution: a warning naming it, every built-in command kept | tests/cli/test_broken_plugin.py |
| `dazflow plugins` listing | every registry the package has | gap #35 |
<!-- markdownlint-enable MD013 -->

## Objects

One row per object that keeps state between calls. Python frees memory
itself, so the R guards do not carry over as written: **Fork** reads "a
second process or worker using the same object", **Unwind** reads "an
exception part-way through", and **Owner** reads "who writes or frees what
it holds".

<!-- markdownlint-disable MD013 -->
| Object | Sequences | Fork | Unwind | Owner |
| --- | --- | --- | --- | --- |
| project folder (`Project`) | show, replay or a step before any run (exit 1); a step before its inputs exist; a rerun replaces its step folder; a changed earlier dataset stops a later step: tests/cli/test_exit_codes.py, tests/project/test_project.py | n/a: one process per command; xdist workers each copy the example (tests/conftest.py) | a failed gate writes the run so far with `status: gate_failed`: tests/cli/test_exit_codes.py::test_a_failed_gate_exits_2_and_the_run_stops_there | `Project.write` only: tests/pipeline/test_pure.py; stale later-step folders after a rerun: gap #34 |
| run carried with `--from` | earlier results come from the record: tests/project/test_project.py::test_a_step_command_carries_the_earlier_results_from_the_record | n/a: in memory, one process | n/a: steps and `Project.run` write nothing, only `Project.write` does: tests/pipeline/test_pure.py | the record's configuration of earlier steps is reused without warning: gap #33 |
| run in memory (`Run`) | replay without its frames refused; another frame is a difference: tests/pipeline/test_replay.py | n/a: plain data, no handle | n/a: holds no resource | n/a: holds no file or handle |
| plugin registries (`Registry`) | a broken plugin reported on every lookup; a lookup while plugins load: tests/core/test_core_oracles.py | n/a: entry points are read per process | the failed load is undone so the next lookup reports it: tests/core/test_core_oracles.py | n/a: holds no file or handle |
<!-- markdownlint-enable MD013 -->

## Limits

One row per limit or threshold. "Straddled by" is the test at, just below
and just above it.

<!-- markdownlint-disable MD013 -->
| Limit | Unit | Straddled by |
| --- | --- | --- |
| a check's threshold under each operator | the observed value, by one ulp | tests/core/test_gate.py::test_check_straddles_its_threshold, tests/core/test_core_oracles.py::test_each_operator_straddles_its_threshold_by_one_ulp |
| data-quality thresholds (counts, shares, range, gap length, duplicates, revisions, downturn window) | periods, shares, values | tests/checks/test_thresholds.py |
| outlier z | robust z-score | tests/checks/test_outliers.py::test_the_z_threshold_is_configuration |
| outlier detection on short series and small spikes | spikes flagged of 120, over 40 seeds | tests/checks/test_outliers.py::test_detection_rate_on_short_series_and_small_spikes |
| outlier minimum length | years (two cycles; seven for annual) | tests/checks/test_outliers.py::test_a_series_shorter_than_two_cycles_is_not_applicable |
| ingest minimum history | periods | tests/steps/test_ingest.py::test_history_gate_at_one_below_and_one_above_the_minimum |
| build driver lag | periods: -1 refused, 0 accepted (#50) | tests/steps/test_build.py::test_a_negative_lag_is_a_config_error_naming_the_driver |
| build forms, one of each | a repeated form refused, two forms accepted (#49) | tests/steps/test_build.py::test_a_repeated_form_is_a_config_error_naming_it |
| explore alpha, votes, significance and sign | p-value, tests, coefficient | tests/steps/test_stationarity.py, tests/steps/test_screen.py |
| build `max_vif` and significance; select confidence set, benchmark and scenario bounds | VIF, p-value, loss | gap #39 |
| forecast jump-off gap | the origin residual | tests/steps/test_forecast.py::test_the_jump_off_gap_is_the_origin_residual_and_the_gate_straddles_it |
| diagnose p-value, CUSUM and coverage checks | p-value, relative 1e-9 | tests/steps/test_diagnose.py |
| replay tolerance | relative 1e-9, absolute 1e-12 | tests/pipeline/test_compare.py, tests/e2e/test_replay.py |
| MASE training length | values: season, season + 1 | tests/evaluation/test_metrics.py::test_the_mase_scale_straddles_a_training_series_as_long_as_its_season |
| expanding splitter's first origin | periods: 1 | tests/evaluation/test_splitters.py::test_an_expanding_split_may_train_on_one_period |
| Box-Cox estimate | values: 1 and 2; `lmbda * v + 1` at 0 | tests/transforms/test_transform_oracles.py |
| artefact step and number | 0 and 1 | tests/core/test_core_oracles.py |
| mutation survivors: error text | mutants (446 of 2,941) | gap #40 |
| mutation survivors: the rest, and what mutmut cannot reach | mutants (218 of 2,941) | gap #41 |
<!-- markdownlint-enable MD013 -->
