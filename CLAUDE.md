# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Neuroscience research codebase for analyzing rodent behavioral data from a two-alternative forced choice (2AFC) auditory discrimination task. Mice discriminate inter-aural level differences (ILDs) in sound stimuli and report their choice by licking left or right. The project covers the full pipeline: raw session parsing, session aggregation, subject selection, behavioral analysis, psychometric fitting, lick/RT analysis, psychophysical kernels (GLMs), GLM-HMMs, electrophysiology, and pharmacology (MK-801).

## Environment

- **Package manager:** uv (see `pyproject.toml`)
- **Python:** >=3.10
- **Virtual env:** `.venv/` (managed by uv)
- **IDE:** PyCharm
- **Plotting style:** `alexis_style.mplstyle` (despined axes, SVG output, 300 dpi saves)

### Running scripts

```bash
uv run python <script.py>
```

There is no test suite or linter configured. Analyses are run as scripts or Jupyter notebooks.

## Architecture

The codebase is organized as a collection of Python packages, each in its own directory with `__init__.py` re-exporting from the main module (e.g., `from .glue_sessions import *`). They are imported via relative paths from the project root, not installed as packages.

### Data pipeline (upstream → downstream)

1. **`parse/`** — Parses raw PyBpod CSV session files into trial-level DataFrames. `parse.py` handles `stage_training` protocol; `parse_v2.py` handles `stage_training_v2` through `v6`. Reads from `~/pv_nmdar_eranet/experiments/`.

2. **`glue_sessions/`** — Aggregates parsed sessions per animal (`glue_sessions`), per experiment batch (`glue_animals`), and across batches (`glue_groups`). Outputs per-animal CSVs into `glue_sessions/<experiment>/`. Also stores corrupted-session logs. The `2AFC/` subdirectory has pre-glued CSVs; `2AFC_4/`, `Ephys/` have corrupted-session logs.

3. **`cherry/`** — Subject selection ("cherry-picking"). Filters out mice with too few valid trials (<1000) or high lapse rates (sum > 2/3) based on psychometric curve fits. `main(experiments)` returns a dict of good subject IDs per experiment.

4. **`my_fun/`** — Shared utility library. Key functions:
   - `filter_behavior()` — Applies experiment-specific trial filters (removes warm-up/AW trials, filters by task parameters per batch)
   - `compute_psych_curve()` — Fits a 4-parameter sigmoid (sensitivity, bias, lower lapse, upper lapse) via MLE
   - Sound generation (`white_noise`, `envelope`, `do_envelope_dB_normal`) and dB/amplitude conversions
   - `save_fig()`, `fig_size()`, `pval_to_star()`, `add_stars()` — Plotting helpers

5. **`psychometric_curves/`** — Plots psychometric curves (prob. right, prob. repeat) for individual animals, across animals, and with drug conditions (saline vs MK-801).

6. **`licks/`** — Lick analysis: reaction times, inter-lick intervals, lick counts. `add_lick_data()` enriches a behavior DataFrame with lick-derived columns. Plotting functions for RT/ILI/nLicks distributions, chronometric curves, and GLM-based lick modeling.

7. **`kernels/`** — Psychophysical kernel analysis via logistic GLMs (statsmodels). Builds design matrices for stimulus frames (ILD residuals), choice history (r+/r-), net ILD, and session index. Includes permutation tests.

8. **`test_classes.py`** — `BehaviorData` class that wraps the common pipeline: load → filter → cherry-pick.

### Other analysis modules

- **`glmhmm/`** — GLM-HMM (hidden Markov model with GLM emissions) for state-dependent behavior
- **`full_model_behavior/`** — Full behavioral model combining kernels, history, and session effects
- **`ephys/`** — Electrophysiology pipeline: `preprocessing/`, `analysis/`, `glms/`, `decoder/`
- **`drugs/`** — Pharmacology analysis (MK-801 drug effects)
- **`intersession/`** — Between-session summary statistics
- **`bootstrapping/`** — Bootstrap confidence intervals
- **`daily_report/`** — Automated daily performance reports (multiple versions, v2-v6)

### Data layout

- Raw data lives at `~/pv_nmdar_eranet/experiments/<experiment>/setups/<animal>/sessions/`
- Glued CSVs live at `~/PycharmProjects/glue_sessions/<experiment>/<animal>.csv`
- Intersession data at `~/PycharmProjects/intersession/<experiment>/<animal>_intersession.csv`
- Sound definition CSVs at `~/PycharmProjects/create_sounds/sounds*.csv`

### Experiment batches

- `2AFC_2`, `2AFC_3` — Early behavioral batches (protocol `stage_training_v2`/`v3`)
- `2AFC_4` — Ephys pilot (protocol `v4`, FSM changes, 0.15s motor delay, 0.5s delay)
- `2AFC_5` / `Ephys` — Electrophysiology group (protocol `v5`, no evidence trials, 0.5s stim)
- `2AFC_6` — Pharmacology group (protocol `v6`, MK-801 drug experiments, 11th frame with 0 evidence)

## Key conventions

- Subject IDs are zero-padded to 3 digits (e.g., "016", "332") via `.str.zfill(3)`
- Stimulus evidence is encoded as ILD (inter-aural level difference in dB): negative = left, positive = right, with extreme values at +/-70 dB
- `Choice`: 0 = left, 1 = right; `Side`: 0 = left, 1 = right; `Hit`: 0 = error, 1 = correct
- `Drug` column: 0 = saline, 1 = drug, NaN = no drug session
- Psychometric curve params: `[sensitivity (k), bias (x0), lower_lapse (b), upper_lapse (p)]`
- Figures are saved as both PNG (white bg) and SVG (transparent bg) via `save_fig()`
