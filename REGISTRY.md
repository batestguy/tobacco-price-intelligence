# Registry

Every external resource this project depends on, in one place. Update this file whenever a
URL, account, or secret changes — it is the map for anyone (including future you) picking the
project back up.

**No values are recorded here. Names only.**

## Accounts and services

| Service | Purpose | Free-tier limit that matters | URL |
|---|---|---|---|
| GitHub | Code, versioned data, **all compute** | Actions unmetered on public repos | https://github.com/batestguy/tobacco-price-intelligence |
| Supabase | Dashboard **Auth only** (login + `users` role lookup) | 500 MB; **pauses after 7 days idle**, and nothing keeps it warm | https://supabase.com/dashboard |
| Streamlit Community Cloud | Dashboard hosting | 1 GB RAM; sleeps after 12 h idle | https://share.streamlit.io |
| Groq | Llama 3.3 70B memo generation | ~1000 req/day, 12k tokens/min | https://console.groq.com |
| Hugging Face | Fine-tuned model weights | 100 GB Hub storage | https://huggingface.co/batestguy |
| Kaggle | T4 GPU for transfer learning | 30 h/week, 12 h/session | https://kaggle.com/code |
| Gmail | SMTP alerts | 100 emails/day (app password) | https://myaccount.google.com/apppasswords |

## Deployed endpoints

| What | URL |
|---|---|
| Dashboard | _set after first Streamlit deploy_ |
| Supabase project | _set after project creation_ |
| Fine-tuned FinBERT | `batestguy/finbert-ng-financial` _(Phase 3)_ |

## Data sources

| Source | Endpoint | Cadence | Notes |
|---|---|---|---|
| CBN FX rates | `https://www.cbn.gov.ng/api/GetAllExchangeRatesGRAPH` | daily | Known flaky; `sources/cbn.py` falls back to the rates HTML page, then to carrying forward the last observation |
| Inflation — tier 1 | `NBS_INFLATION_URL` repo variable | monthly | **Unset by default.** Point it at a live NBS CSV/XLSX release when one exists; the old hard-coded `nigerianstat.gov.ng/resource/csv/cpi.csv` now 404s |
| Inflation — tier 2 | `https://www.cbn.gov.ng/rates/inflrates.html` | monthly | CBN republishes the NBS CPI series. Right cadence, but rows render client-side, so this usually parses empty |
| Inflation — tier 3 | `data/seed/inflation.csv` | — | Backfill, not included. See `data/seed/README.md` |
| Inflation — tier 4 | `https://api.worldbank.org/v2/country/NGA/indicator/CPTOTSAXNZGY?source=15&format=json` | monthly | No key. Global Economic Monitor, back to 2015M01, and currently the **live** tier. `source=15` is required — GEM is outside the default WDI source and the call errors without it. **Seasonally adjusted World Bank staff calculation, not NBS's published headline** |
| Inflation — tier 5 | `https://api.worldbank.org/v2/country/NG/indicator/FP.CPI.TOTL.ZG?format=json` | annual | No key. The floor tier — **annual and lagged**, newest value is last calendar year |
| Competitor prices | `data/seed/competitor_prices.csv` | manual | **Marketplace scraping removed** — NTCA 2015 s.15(4) bans online tobacco sale in Nigeria. Cited reference file, header-only; optimizer reports its ceiling `INACTIVE` |
| Financial news | Punch, Nairametrics, BusinessDay RSS | 2×/day | **Headline + URL + score only.** Bodies are scored in memory and discarded |

`sources/nbs.py` unions the inflation tiers rather than racing them — a better tier wins any
month it covers — and records which tier produced each row in the `source` column.

## GitHub Actions secrets

Set with `gh secret set NAME` (paste the value at the prompt — never pass it as an argument,
where it would land in shell history).

| Secret | Used by | Where to get it |
|---|---|---|
| `GROQ_API_KEY` | recommend | Groq console → API Keys |
| `GMAIL_ADDRESS` | recommend | The sending Gmail account |
| `GMAIL_APP_PASSWORD` | recommend | Google Account → Security → App passwords (requires 2FA) |
| `HF_TOKEN` | score, train | huggingface.co → Settings → Access Tokens (write scope, for Phase 3 pushes) |

Verify with `gh secret list` — it prints names and update times, never values.

### Optional repository *variables* (not secrets)

Set with `gh variable set NAME --body value`.

| Variable | Default | Effect |
|---|---|---|
| `FINBERT_MODEL` | `ProsusAI/finbert` | Point at `batestguy/finbert-ng-financial` after Phase 3 to switch the scorer over without a code change |
| `NBS_INFLATION_URL` | _unset_ | A direct NBS CPI release (CSV or XLSX). Highest-priority inflation tier when set; skipped entirely when not, because a wrong URL fails every run and looks like a network fault |
| `ALERT_RECIPIENTS_COMMERCIAL` | — | Comma-separated addresses for Commercial Director alerts |
| `ALERT_RECIPIENTS_SUPPLY` | — | Comma-separated addresses for Supply Chain Manager alerts |

## Streamlit Cloud secrets

Set separately in the Streamlit dashboard (**App → Settings → Secrets**), in TOML. Streamlit
does **not** inherit GitHub secrets.

```toml
SUPABASE_URL = "https://<project>.supabase.co"
SUPABASE_ANON_KEY = "<anon key, NOT the service key>"
```

The app is publicly reachable, so it gets the **anon** key only. Note what the login is and
is not: it routes users to their role's views (§6) and satisfies §11's "authorized personnel
only" framing, but it is **not** a confidentiality boundary — the dashboard renders Parquet
committed to a public repo, so every figure behind it is already world-readable. The RLS
policy in `schema.sql` covers `users`, keeping a session from reading other people's roles.

## Setup

The full runbook, with what breaks if you skip each step, is in
[SETUP.md](SETUP.md). In short:

1. `gh repo create ... --public --source=. --push`
2. Create the Supabase project; run `supabase/schema.sql` in the SQL editor. Auth only — its
   URL and anon key go to Streamlit in step 5, not to Actions.
3. `gh secret set` the four secrets above.
4. `gh workflow run scrape.yml` — confirm a Parquet file is committed by the Actions bot.
5. Deploy `app/streamlit_app.py` on Streamlit Cloud; add its two secrets separately.
6. Record the resulting dashboard URL in the table above.

## Workflows

| Workflow | Schedule (UTC) | Local time (WAT = UTC+1) | Does |
|---|---|---|---|
| `scrape.yml` | `0 5,17 * * *` | 06:00, 18:00 | FX, inflation, competitors, news → Parquet |
| `score.yml` | after scrape (`workflow_run`) | — | FinBERT + VADER on CPU → scores + aggregates |
| `train.yml` | `0 23 * * 6` | Sun 00:00 | Rebuild features, retrain XGBoost, commit model + metrics |
| `recommend.yml` | `0 7 * * *` | 08:00 | Forecast → optimize → recommendations → alerts → memo |

All four also accept `workflow_dispatch`.
