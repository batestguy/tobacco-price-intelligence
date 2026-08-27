# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## The one rule that shapes everything

**Nothing in this project runs on the local machine.** `D:\Tobacco Project` is a working
copy for *authoring and reading* — git and `gh` only. Every byte of computation happens on
free hosted infrastructure, chiefly GitHub Actions.

Do not run scrapers, training, scoring, or the dashboard locally. Do not add a step that
assumes a local Python process. If you need to see a pipeline stage execute, dispatch its
workflow:

```bash
gh workflow run scrape.yml && gh run watch
gh run view --log-failed        # after a failure
```

Local conda envs (documented in `ENVIRONMENTS.md`) are useful only for *reading* code with
an interpreter's help — imports resolving, type hints, notebook rendering. They are not the
execution path and must not be referenced in workflows, docs, or setup instructions.

## Architecture

```
Local  ──push──▶  GitHub repo (public)  ──▶ Actions = the compute engine
                        │                     scrape · score · train · recommend
                        │                     commits data back to the repo
                        ├─ Supabase        serving layer + Auth
                        ├─ HF Hub          fine-tuned weights (100 GB free)
                        ├─ Kaggle          T4 GPU, transfer learning only
                        └─ Streamlit Cloud dashboard, redeploys on push
```

Repo: `batestguy/tobacco-price-intelligence`, **public**.

**The repo is the source of truth; Supabase is the serving layer.** Each job writes
month-partitioned Parquet to `data/curated/`, commits it, *then* upserts to Postgres. Both
writes are idempotent. If the two ever disagree, Parquet wins and Supabase is rebuilt from it.

## Layout

```
.github/workflows/    scrape.yml · score.yml · train.yml · recommend.yml
src/tobacco/
  config.py           env-var loading; fails loudly if a secret is missing
  sources/            cbn · nbs · competitors · news · sales_mock
  nlp/                finbert · vader
  features/build.py   lags, dummies, sentiment composites (INTRO.txt §4)
  models/             train_xgb · predict
  optimize/linprog.py SciPy optimizer + the four §5 constraints
  alerts/email.py     Gmail SMTP, the four §7 rules
  memo/groq.py        §10 prompt template, verbatim
  store/              parquet_io.py (repo truth) · supabase_io.py (serving)
  jobs/               scrape · score · train · recommend — the workflow entrypoints
data/curated/         month-partitioned Parquet, committed by the Actions bot
models/               XGBoost joblib + metrics.json (small artifacts only)
supabase/schema.sql   database contract
app/streamlit_app.py  dashboard
notebooks/            Kaggle transfer-learning notebook
REGISTRY.md           every external resource, URL and secret name in one place
```

Workflows invoke jobs as modules with `PYTHONPATH=src`, e.g. `python -m tobacco.jobs.scrape`.

## Where INTRO.txt is stale

`INTRO.txt` is the original spec and is preserved verbatim. It is authoritative on *what the
system should do* — the §2 schema, §4 features, §5 constraints, §7 alert rules, §10 prompt,
§11 disclaimer. It is out of date on *where things run*. These four departures are settled;
do not "fix" the code back toward the spec:

1. **§3 option C — HF Inference API "1000 requests/day" no longer exists.** The free tier is
   now credit-based (~$0.10/month of Inference Provider credits), which will not cover a
   daily batch. **FinBERT runs on the Actions CPU runner instead.** HF's free 100 GB Hub
   *storage* is still real, and is where fine-tuned weights live.
2. **§8 PythonAnywhere is the wrong host.** It requires manual upload, which contradicts a
   fully cloud-native repo. **Streamlit Community Cloud** redeploys on push. This also
   supersedes §6's Plotly Dash recommendation — Dash is not installed anywhere and Streamlit
   is the spec's own fallback (§6 option C).
3. **§14's local-first build order** assumes a workstation. Build order is unchanged in
   substance; the execution target is Actions.
4. **§1's marketplace competitor scraping is removed.** Jumia/Konga scraping is gone from
   `sources/competitors.py` and will not be restored. Jumia 403s the runner and the obvious
   fix — a browser User-Agent and slower pacing — means defeating bot detection to collect
   listings that **s.15(4) of the National Tobacco Control Act 2015 prohibits**: the online
   sale of tobacco products in Nigeria. That is the same Act the §11 disclaimer cites
   verbatim, so a price ceiling derived from those listings contradicts the project's own
   stated legal framework. Weak enforcement is also why the listings are sporadic and
   unrepresentative — bad data, not just awkward data. The replacement is a cited, dated
   reference file (`data/seed/competitor_prices.csv`), header-only until someone records a
   citable survey. **The `competitor_prices` dataset and §2 schema are unchanged**, and
   `optimize/linprog.py` already reports its ceiling `INACTIVE` when there is no competitor
   data — do not add a fallback that invents one.

   Relatedly, §1's NBS CSV endpoint is dead (404) and NBS has no stable machine-readable
   release. `sources/nbs.py` degrades `NBS_INFLATION_URL` → CBN monthly → **World Bank annual**
   → seed, tagging each row's tier in `source`. NBS is still the origin of the figures; it is
   just not reachable directly. Expect the World Bank tier to serve today, which means
   inflation is **annual and lagged** — say so rather than letting it look monthly.

Everything must stay on a **free tier** — a paid dependency breaks the project's premise.

## What must never be committed

The repo is public, so these are correctness issues, not hygiene preferences.

- **No full article text.** `newspaper3k` returns article bodies; committing them would
  republish copyrighted Nigerian news content. Persist **headline + URL + source +
  timestamp + score only**. The body is scored in memory and discarded.
  `store/parquet_io.py` strips body-like columns unconditionally — do not weaken that guard,
  and do not add a column that carries article text under another name.
- **No secrets.** All six credentials live in GitHub Actions secrets and (separately)
  Streamlit Cloud secrets. Never a literal key in code, a committed `.env`, or a workflow
  `echo` that would print one into a public log.
- **No real company data.** Sales are synthetic (`sources/sales_mock.py`). This is what lets
  the repo be public at all.
- **No Git LFS.** Its free quota is 1 GB storage / 1 GB bandwidth per month, and bandwidth is
  consumed on *every Actions checkout* — LFS would break the free-tier premise within days.
  Anything too large for git goes to HF Hub or GitHub Releases.

### Why month-partitioned Parquet

One growing `news.parquet` means every daily commit rewrites the whole binary blob and git
stores a **new full copy each time** — the repo bloats fast. `data/curated/news/2026-08.parquet`
means a daily commit only rewrites the current month's small file. Volume is tiny either way
(1 FX row/day, ~20 competitor rows/day, ~30 headlines/day ≈ a few MB/year), so this stays
comfortably inside plain git.

## Actions gotchas

Each of these silently breaks the pipeline rather than failing loudly:

- **Install torch from the CPU index**: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
  The default wheel drags in ~2 GB of CUDA the runner cannot use. torch is intentionally
  absent from `requirements-actions.txt` for this reason.
- **Cache `~/.cache/huggingface`** with `actions/cache`, or every run re-downloads ~440 MB.
- **Commits made with the default `GITHUB_TOKEN` do not trigger other workflows.** Chain
  stages with `workflow_run` or job dependencies — never by watching for pushes.
- **Cron is best-effort** and can be delayed or doubled under load. Every job must be
  idempotent: upsert by key, never blind-append. This is enforced in `store/parquet_io.py`.
- **Scheduled workflows are auto-disabled after 60 days of repo inactivity.** The daily data
  commits keep the repo active — that is a second reason the commit-back step matters.
- **Every workflow needs `workflow_dispatch`** so it can be triggered without waiting for cron.
- Jobs that commit need `permissions: contents: write`.
- **Streamlit Cloud has its own secrets**, set in its dashboard, not inherited from GitHub. It
  gets the Supabase **anon** key — never the service key. The app is publicly reachable.

## Data model

The Supabase schema (`supabase/schema.sql`, from `INTRO.txt` §2) is the contract between the
scraper, NLP, ML, and dashboard layers — change it deliberately, and mirror any change in the
Parquet writers:

`exchange_rates` · `inflation` · `competitor_prices` · `news_articles` (with `finbert_score`) ·
`social_posts` (with `vader_score`) · `sentiment_aggregates` · `sales_mock` ·
`recommendations` · `users` · `logs`

## Regulatory constraint

This is **internal decision-support** for pricing and supply chain — it does not promote or
advertise tobacco. `INTRO.txt` §11 mandates a disclaimer on the dashboard footer/login page
citing the **National Tobacco Control Act 2015** and **WHO FCTC Article 5.3**, and states that
data is aggregated and anonymized with no individual consumer data stored. That text is
carried verbatim in `app/streamlit_app.py` — keep it verbatim, and keep the design consistent
with it.

Because the repo is public, `README.md` additionally frames the project as an **independent
portfolio / educational demonstration** using synthetic sales data, **not affiliated with or
commissioned by BAT**. A public repo presenting itself as a live tobacco-industry commercial
deliverable would read badly against Article 5.3, which exists to insulate public health
policy from industry influence. Preserve that framing.
