# Registry

Every external resource this project depends on, in one place. Update this file whenever a
URL, account, or secret changes — it is the map for anyone (including future you) picking the
project back up.

**No values are recorded here. Names only.**

## Accounts and services

| Service | Purpose | Free-tier limit that matters | URL |
|---|---|---|---|
| GitHub | Code, versioned data, **all compute** | Actions unmetered on public repos | https://github.com/batestguy/tobacco-price-intelligence |
| Supabase | Postgres serving layer + Auth | 500 MB; **pauses after 7 days idle** | https://supabase.com/dashboard |
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
| NBS inflation | https://nigerianstat.gov.ng | monthly | CSV/XLSX release; parser tolerates layout drift |
| Competitor prices | Jumia, Konga search pages | daily | HTML scrape; selectors are the fragile part |
| Financial news | Punch, Nairametrics, BusinessDay RSS | 2×/day | **Headline + URL + score only.** Bodies are scored in memory and discarded |

## GitHub Actions secrets

Set with `gh secret set NAME` (paste the value at the prompt — never pass it as an argument,
where it would land in shell history).

| Secret | Used by | Where to get it |
|---|---|---|
| `SUPABASE_URL` | all jobs | Supabase → Project Settings → API → Project URL |
| `SUPABASE_SERVICE_KEY` | all jobs | Supabase → Project Settings → API → `service_role` key. **Server-side only. Never give this to Streamlit.** |
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
| `ALERT_RECIPIENTS_COMMERCIAL` | — | Comma-separated addresses for Commercial Director alerts |
| `ALERT_RECIPIENTS_SUPPLY` | — | Comma-separated addresses for Supply Chain Manager alerts |

## Streamlit Cloud secrets

Set separately in the Streamlit dashboard (**App → Settings → Secrets**), in TOML. Streamlit
does **not** inherit GitHub secrets.

```toml
SUPABASE_URL = "https://<project>.supabase.co"
SUPABASE_ANON_KEY = "<anon key, NOT the service key>"
```

The app is publicly reachable, so it gets the **anon** key only, and relies on Supabase Auth
plus row-level security for access control.

## Setup

The full runbook, with what breaks if you skip each step, is in
[SETUP.md](SETUP.md). In short:

1. `gh repo create ... --public --source=. --push`
2. Create the Supabase project; run `supabase/schema.sql` in the SQL editor.
3. `gh secret set` the six secrets above.
4. `gh workflow run scrape.yml` — confirm a Parquet file is committed by the Actions bot.
5. Deploy `app/streamlit_app.py` on Streamlit Cloud; add its two secrets separately.
6. Record the resulting dashboard URL in the table above.

## Workflows

| Workflow | Schedule (UTC) | Local time (WAT = UTC+1) | Does |
|---|---|---|---|
| `scrape.yml` | `0 5,17 * * *` | 06:00, 18:00 | FX, inflation, competitors, news → Parquet + Supabase |
| `score.yml` | after scrape (`workflow_run`) | — | FinBERT + VADER on CPU → scores + aggregates |
| `train.yml` | `0 23 * * 6` | Sun 00:00 | Rebuild features, retrain XGBoost, commit model + metrics |
| `recommend.yml` | `0 7 * * *` | 08:00 | Forecast → optimize → recommendations → alerts → memo |

All four also accept `workflow_dispatch`.
