# Machine Environments — Windows 11, drive C

Routing reference: pick an environment, know the traps. **Not** a package inventory.
For exact contents run `conda list -p <prefix>` or read `raw\<env>.json` (full pinned
lists for all 10 interpreters) and `raw\r_packages.csv` (all 1,027 R packages).
Scanned 2026-08-05. Only packages that *discriminate* between envs are listed below —
every DS env has numpy/pandas/scipy/matplotlib/scikit-learn/jupyterlab.

## Environments

| Env | Activate | Py | Size |
|---|---|---|---|
| `ds-general` | `conda activate C:\Users\TOSHIBA\ds-general` | 3.12 | 3.3 GB |
| `causality-handbook` | `conda activate causality-handbook` | 3.11 | 4.1 GB |
| `homl3` | `conda activate homl3` | 3.10 | 3.5 GB |
| `bap3` | `conda activate bap3` | 3.11 | 2.3 GB |
| `prophet-env` | `conda activate prophet-env` | 3.11 | 2.5 GB |
| `mlops-env` | `conda activate mlops-env` | 3.13 | 1.1 GB |
| `appdev-conda` | `conda activate appdev-conda` | 3.13 | 0.4 GB |
| `base` | `conda activate base` | 3.13 | 15 GB† |
| `appdev-env` | `C:\Users\TOSHIBA\appdev-env\Scripts\activate` (venv) | 3.14 | — |
| system | `C:\Python314\python.exe` | 3.14 | — |
| R 4.5.2 | `Rscript` (on PATH) | — | 1027 pkgs |

† `base` reports 19 GB, but 3.6 GB is the shared `miniconda3\pkgs` download cache
serving *all* envs — not base's own content.

## Task → environment

| Task | Env |
|---|---|
| Generic analysis / unsure | `ds-general` |
| Causal inference, treatment effects, DAGs | `causality-handbook` |
| Deep learning, neural nets | `homl3` |
| Reinforcement learning | `homl3` |
| Bayesian / hierarchical models (Python) | `bap3` |
| Bayesian / multilevel models (R, preferred) | R — `brms`, `rethinking` |
| Time-series forecasting | `prophet-env` |
| Experiment tracking, model serving, drift | `mlops-env` |
| Gradient boosting (XGB/LGBM/CatBoost) | `ds-general` or `causality-handbook` |
| Web API / small app, no DS deps | `appdev-env` |
| Statistics, tidyverse, papers, meta-analysis | R |
| Reports / rendering | `quarto` 1.9.38 (global, any env) |

## What makes each env distinct

**`ds-general`** — widest coverage, the default. `xgboost 3.3` `lightgbm 4.7`
`catboost 1.2.7` `polars 1.43` `duckdb 1.5.5` `mlflow 3.15` `optuna 4.9`
`streamlit 1.60` `fastapi` `flask` `linearmodels 7.0` `evidently`.
*Avoid:* no deep-learning stack.

**`causality-handbook`** — `dowhy 0.12` `econml 0.16` `causalml 0.15.5`
`causal-learn 0.1.4.4` `linearmodels 7.0` `networkx`, plus `pymc 5.25` `arviz 0.23`
and `torch 2.10`. The only env with PyTorch.

**`homl3`** — Hands-On ML 3e. `tensorflow 2.14` `keras 2.14` `transformers 4.35`;
RL via `gymnasium 1.2` `ale-py` `box2d-py` `pygame`; also `nbdime` `shapely`.
*Avoid:* TF/Keras 2.14 are old, Python 3.10 is the oldest here.

**`bap3`** — Bayesian Analysis with Python 3e. `pymc 5.8` `arviz 0.16`
`bambi 0.13` `nutpie 0.9` `pytensor 2.15`. *Avoid:* numpy 1.24, the oldest on the box.

**`prophet-env`** — `prophet 1.1.7` + `cmdstanpy 1.3` (Stan backend). Narrow and clean.

**`mlops-env`** — `mlflow 3.14` `evidently 0.7.21` `optuna 4.9` `fastapi` `flask`.
No notebooks. *Avoid:* no plotting beyond matplotlib.

**`appdev-conda`** — near-empty sandbox: `pytest 9.1` `sqlalchemy`. 70 packages.

**`appdev-env`** (venv off system Python) — `fastapi` `flask` `sqlalchemy` `openpyxl`.
Python 3.14, newest.

**`base`** — conda's root. `jupyterlab` `pandas` `scipy` only. **Don't install here.**

**system `C:\Python314`** — jupyterlab, pandas, scipy, networkx. Parent of `appdev-env`.

## Nuances that cause mistakes

- **`ds-general` is a prefix env** at `C:\Users\TOSHIBA\ds-general`, outside
  `miniconda3\envs\`. `conda list -n ds-general` **fails**; use
  `-p C:\Users\TOSHIBA\ds-general`. Safest habit: use `-p <prefix>` everywhere.
- **`appdev-env` is a venv, not conda.** Its interpreter is at `Scripts\python.exe` —
  there is **no** `python.exe` at the folder root. `conda activate` won't work.
- **numpy is split across a major version.** 1.x: `bap3` (1.24), `ds-general`,
  `causality-handbook`, `homl3` (1.26). 2.x: `prophet-env` (2.4), `mlops-env`,
  `base`, system (2.4–2.5). Code written for one side can break on the other.
- **pandas 3.0 is already installed** in `causality-handbook`, `prophet-env`, `base`
  and system Python; `bap3`/`homl3` are on 2.1. Expect API differences.
- **Python 3.14** (`appdev-env`, system) has few scientific wheels — don't pick it
  for numeric work.
- **No GPU/CUDA claim is made here.** Assume CPU until verified.
- Never install into `base`.

## R 4.5.2 — 1,027 packages, 305 top-level

Only one R version. User library `C:\Users\TOSHIBA\AppData\Local\R\win-library\4.5`
(997); system library (30). This is the deepest environment on the machine.

`tidyverse` · **Bayesian:** `brms` `rethinking` `tidybayes` `MCMCpack` `MCMCglmm`
`runjags` `bayesrules` `bcogsci` · **ML:** `caret` `mlr3*` (learners, pipelines,
tuning, viz) `randomForest` `DALEX` `Rtsne` · **Causal:** `MatchIt` `Matching`
`mediation` `dagitty` `estimatr` `DRDID` `causaldata` `oaxaca` · **Time series:**
`fable` `feasts` `modeltime` `tsDyn` `rugarch` `tidyquant` · **Survival:**
`survminer` `flexsurv` `coxme` `timeROC` `msm` · **Meta-analysis:** `gemtc`
`metaSEM` `metasens` `dmetar` `robvis` · **Design of experiments / power:** `FrF2`
`skpr` `agricolae` `rsm` `pwr` `pwrss` `TOSTER` · **Tables & reporting:**
`gtsummary` `modelsummary` `flextable` `stargazer` `table1` `apaTables`
`summarytools` · **Publishing:** `quarto` `blogdown` `distill` `flexdashboard`
`tufte` · **Shiny/web:** `golem` `plumber` `waiter` `shinyFeedback` · **Spatial:**
`sf`-stack, `rgee` `tmaptools` `leafgl` `spatstat.model` `ncdf4` · **Viz:** `GGally`
`ggforce` `ggstatsplot` `ggpmisc` `dendextend` `treemapify` · **Missing data:**
`mice` `naniar` · **Dev:** `devtools` `remotes` `covr` `git2r`.

Full list with one-line descriptions: `raw\r_packages.csv` (`toplevel` column marks
the 305 deliberate installs).

## Node & CLI

Node v24.15.0 · npm 11.16.0 · pnpm 10.33.2 · git 2.53.0 · quarto 1.9.38 ·
uv 0.11.28 · conda 26.1.1 + mamba.

npm globals are almost entirely AI agent CLIs: `@anthropic-ai/claude-code`,
`@openai/codex`, `@google/gemini-cli`, `@qwen-code/qwen-code`, `opencode-ai`,
`deepseek-tui`, plus `screenshot-desktop`.
