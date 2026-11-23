# Ionome

An open pipeline for untargeted LC-MS metabolomics ("ionome") data. It takes raw `.mzML` / `.mzXML` files and a
sample sheet and produces an aligned, annotated feature table with QC, statistics and plots, and it comes with a small
web app for creating studies, running them and browsing the results. Everything runs locally; the only storage is a
SQLite file and ordinary folders.

> **Status:** first version, under active refactoring. It has been run on synthetic data end to end (see below) and on
> two public datasets during development; treat results on your own data as something to check, not to trust blindly.

## What it does

Two dataset profiles, chosen per study in its config:

| Profile | For | Stages |
|---|---|---|
| `cohort_with_qc` | a real sample cohort with pooled QCs and a solvent blank | ingestion → scan summary → chromatograms → feature detection → alignment → blank QC → adduct annotation → library matching → library assembly → summary stats → PLS-DA → heatmap → volcano → XIC verification → export (15 stages) |
| `standards_only` | injections of known standards, to validate the method | ingestion → scan summary → chromatograms → system suitability → feature detection → blank QC → adduct annotation → spectral purity → library matching → library assembly → export (11 stages) |

Highlights: chromatogram-based peak detection with enforced peak widths, seeded cross-sample alignment with RT
correction, blank and QC filtering, adduct grouping, MS2 library matching with a chance-level check, moderated
statistics, and checkpoint caching so a re-run only recomputes what changed (inputs, settings or code). The full stage
descriptions are on the [documentation site](docs/index.html) (`docs/`, served by GitHub Pages).

## Quick start

You need [uv](https://docs.astral.sh/uv/) and Python 3.13 or newer.

```bash
git clone <this repo> && cd ionome
uv sync --all-packages --group dev        # installs core, analysis and api plus test tools
uv run pytest -q                          # unit and integration tests
uv run python scripts/smoke_test.py       # full run on synthetic data through the web API, ~1 minute
```

The smoke test builds a throw-away data folder and database, starts the API, creates a study, runs all 15 stages,
checks the results against what the synthetic data contains, runs again to confirm caching, and shuts down. It never
touches your own `.env` or database. If it prints `SMOKE TEST PASSED`, the install works.

For a guided tour (running the web app yourself, looking at the results, using your own data) read
**[GETTING_STARTED.md](GETTING_STARTED.md)**.

## Using your own data

```bash
cp .env.example .env                      # set the folders and database path for your machine
uv run python -m api                      # web app at http://127.0.0.1:8000
```

Put each study's raw files and `<study_id>_sample_metadata.csv` in `$LCMS_RAW_DATA_DIR/<study_id>/`, create the study
on the **New study** page, and press **Run**. Or run it from the command line once the study exists:

```bash
uv run python -m analysis.pipeline_orchestrator <study_id>
```

Results go to `$LCMS_RESULTS_DIR/<study_id>/` (feature table, QC report, plots, per-stage logs). Study settings live in
a YAML file per study; copy `config/experiments/template.yaml`. Unknown or misspelled options are rejected when the
file is loaded.

## Layout

This is a [uv workspace](https://docs.astral.sh/uv/concepts/workspaces/) of three packages plus supporting folders:

```
core/          ionome-core      settings, database models, config schema, safe file-name handling
analysis/      ionome-analysis  the processing pipeline and its stages (depends on core)
api/           ionome-api       FastAPI app and dashboard (depends on core and analysis)
config/        experiment YAMLs and the template
scripts/       make_demo_data.py, create_demo_study.py, smoke_test.py
tests/         pytest suite (no mocks; uses small synthetic raw files)
tools/         gen_pipeline_diagrams.py (redraws the docs diagrams from the code)
docs/          static documentation site (no build step)
```

Dependency direction: `core` → `analysis` → `api`. A new stage is a class with `validate_input`, `execute` and
`validate_output`, a `_name`, and `requires` (the stages it reads); the pipeline checks the order before running.

## Security note

The API can write files and start runs, so by default it only listens on `127.0.0.1`. To expose it on a network set
`API_TOKEN` (HTTP Basic password) and `API_ALLOWED_HOSTS`; the app refuses to start on another address without a token.

## Tests and docs

```bash
uv run pytest -q                                   # tests
uv run python tools/gen_pipeline_diagrams.py       # regenerate the pipeline diagrams after changing a stage list
```

`tests/test_docs.py` fails if the docs describe a different pipeline than the code builds.
