# Getting started

A step-by-step walk-through. Every step says what you should see, so you can tell whether it worked. Total time on a
laptop: about 15 minutes, most of it reading. Run all commands from the repository root.

## 0. What you need

- [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.13 or newer (uv can install Python for you).
- About 100 MB of free disk space for the demo.

## 1. Install

```bash
uv sync --all-packages --group dev
```

Plain `uv sync` is not enough: the repository root has no dependencies of its own, so `--all-packages` is what installs
the three workspace packages.

## 2. Run the tests

```bash
uv run pytest -q
```

Expect `80 passed` (a few warnings from third-party packages are fine). This takes about 10 seconds and needs no
`.env`, no database and no data.

## 3. One-command check of the whole system

```bash
uv run python scripts/smoke_test.py
```

This is the quickest answer to "does it work on my machine". It takes one to three minutes and ends with
`SMOKE TEST PASSED`. It runs these steps, each printed as it happens:

1. make synthetic data
2. start the API
3. create a study
4. run the 15-stage pipeline
5. check the output files exist
6. check the results against what the synthetic data contains (below)
7. run again and check nothing was recomputed
8. stop the API

If something fails it prints `FAILED:` with the reason and the end of the server log, and exits with status 1. Add
`--keep` to keep its temporary folder (data, results, database) so you can look at it.

### What the synthetic data contains, and what the test expects

`scripts/make_demo_data.py` writes 13 small `.mzXML` files and a 12-entry library. All values are invented; the point
is that the right answer is known.

| Planted in the data | What the pipeline should report |
|---|---|
| 12 compounds with MS2 spectra in the library | 12 aligned features identified, all high confidence |
| 3 unknown peaks and 1 contaminant | 16 aligned features in total |
| the contaminant is also in the solvent blank | flagged by blank QC |
| compounds 1–3 about 4× higher in group B | significant in A vs B |
| compounds 4–6 about 4× higher in group C | significant in A vs C, and not the same features |

The smoke test asserts every row of this table. It proves the plumbing and the logic work; it says nothing about how
the pipeline performs on real instrument data.

## 4. Do it yourself, by hand

### 4a. Make the demo data and the settings file

```bash
uv run python scripts/make_demo_data.py --write-env
```

This writes the synthetic data into `./demo_data` and creates a `.env` that points at it (with absolute paths; an
existing `.env` is never overwritten, the script then only prints the block to compare). Expect 13 `wrote ...mzXML`
lines and then a printout of the settings. Use `--out /some/folder` for a different location. For your own data, copy
`.env.example` to `.env` instead and set the folders there.

Two things worth knowing:
- `LCMS_RAW_DATA_DIR` is the folder that *contains* the study folder (`demo_data/raw` contains `DEMO/`).
- The database is a single SQLite file. There is no database server to install; the file and its tables are created
  the first time the app starts.

### 4b. Start the web app

```bash
uv run python -m api
```

Leave it running. Expect log lines ending in `Uvicorn running on http://127.0.0.1:8000`. Open that address in a
browser: you should see the dashboard with an empty study list. `http://127.0.0.1:8000/docs` shows the REST API.

If it exits immediately with `Ionome cannot start: ... setting(s) missing`, there is no `.env` yet or a value is
missing from it (the message lists them and says what to do; exit status 2). If it says the
host "is reachable from other machines", `API_HOST` is not `127.0.0.1` and no `API_TOKEN` is set.

### 4c. Create the demo study

In a second terminal:

```bash
uv run python scripts/create_demo_study.py --data demo_data
```

Expect `created study DEMO: 13 samples registered`. This sends what the **New study** form sends. Reload the dashboard:
`DEMO` appears in the study list. (You can also create studies in the browser on the **New study** page.)

### 4d. Run it

Either press **Run** on the study page, or from a terminal:

```bash
curl -X POST http://127.0.0.1:8000/studies/DEMO/run
```

It takes roughly 20-60 seconds. The study page shows the run's progress; the same information is at
`GET /studies/DEMO/runs/<run_id>` (one entry per stage, with its status). A second run request while one is active gets HTTP 409.

You can also run it without the web server (the study must already exist in the database):

```bash
uv run python -m analysis.pipeline_orchestrator DEMO
```

### 4e. Look at the results

On the study page: the feature table, the QC report, and the plots (volcano, heatmap, PLS-DA, XIC and others). On disk:

```
<LCMS_RESULTS_DIR>/DEMO/
    resolved_config.json            the exact settings used (and any options this profile ignored)
    library/DEMO_feature_table.csv  one row per aligned feature: m/z, RT, intensity per sample, identification
    qc_reports/DEMO_qc_report.json
    plots/                          PNGs, one folder per stage
    logs/<stage>.json               metrics and warnings per stage (15 files)
    checkpoints/  cache/            what makes the next run fast
```

Check against the table in step 3: `DEMO_feature_table.csv` has 16 rows, 12 of them with a compound name.

### 4f. Run it again

```bash
uv run python -m analysis.pipeline_orchestrator DEMO
```

Stages whose inputs, settings and code are unchanged print `cached ... skipping` (five of them here). Change a setting
in `$YAML_DIR/DEMO.yaml` (for example `feature_detection.noise_threshold`) and run again: that stage and everything
after it recompute. To force a rerun of everything, delete `<LCMS_RESULTS_DIR>/DEMO/checkpoints`.

## 5. Your own data

1. Put the raw files in `$LCMS_RAW_DATA_DIR/<study_id>/`. Formats: `.mzML` or `.mzXML`. The study id may use letters,
   digits, `_` and `-` (max. 64 characters).
2. Write `<study_id>_sample_metadata.csv` next to them. Required columns:
   - `sample_id`: the raw file name without its extension
   - `sample_role`: `solvent_blank`, `QC`, or a group name (for example `control`, `treated`); the statistics compare
     the groups with each other
   - standards studies also need `expected_compounds` and `expected_mz`, written as JSON lists such as `["Glycine"]`
     and `[74.0247]`
3. Copy `config/experiments/template.yaml`, set `study.id`, `study.polarity`, `study.instrument`,
   `files.sample_metadata` (must be exactly `<study_id>_sample_metadata.csv`), `dataset_profile`, and
   `library_matching.reference_library_path` (a `.msp` or `.mgf` library; `reference_libraries/EMBL-MCF_negative.msp`
   ships with the repo). Options marked `(standards_only)` or `(cohort_with_qc)` are only read by that profile.
4. Create the study on the **New study** page and press **Run**.

The defaults were tuned on specific instrument data. In particular `feature_detection.noise_threshold` is an absolute
intensity (default 8000); on an instrument with different intensity scaling, look at the QC output and adjust it, or
switch `noise_mode` to `adaptive`.

## 6. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `ModuleNotFoundError` for a package | you ran plain `uv sync`; use `uv sync --all-packages --group dev` |
| `Ionome cannot start: ... setting(s) missing` | no `.env` yet or a value missing: run `make_demo_data.py --write-env` (demo), or `cp .env.example .env` and fill it in (own data) |
| tests time out waiting for a run | slow first start of the worker processes on a cold machine: `IONOME_TEST_TIMEOUT=300 uv run pytest` |
| `...is reachable from other machines` | `API_HOST` is not a loopback address: set `API_TOKEN`, or use `127.0.0.1` |
| `Unable to find a usable engine ... pyarrow` | older checkout without `pyarrow` in `analysis/pyproject.toml`; run `uv sync --all-packages` again |
| `Parsing PSI formats requires psims` | same: `psims` is needed to read `.mzML`; `.mzXML` does not need it |
| creating a study fails with `no raw file found` | `sample_id` values must match the file names exactly, and the files must be in `<LCMS_RAW_DATA_DIR>/<study_id>/` |
| `files.sample_metadata must be <id>_sample_metadata.csv` | the file name in the config must follow that pattern exactly |
| `invalid pipeline: ...` when a run starts | the stage list is mis-ordered or missing a prerequisite; the message names each problem |
| 404 `unknown study_id` | the study was never created; create it on the New study page |
| 409 when starting a run | that study is already running |
| a library entry is "excluded" in the library QC | its formula and precursor m/z disagree; the log names the entry and the reason |
| everything "no match" with a real library | check `polarity`, the adduct types in the library, and `precursor_mz_tolerance_ppm` |
