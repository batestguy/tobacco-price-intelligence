# Setup runbook

Everything needed to take this repository from "code exists" to "pipeline is
running in the cloud". Work top to bottom; each step states what breaks if you
skip it, so you can stop partway and still have something that works.

Nothing here runs on your machine. The only local tools are `git` and `gh`.

---

## Step 0 — Create the repo and push

```bash
cd "D:\Tobacco Project"

gh repo create batestguy/tobacco-price-intelligence \
    --public --source=. --push \
    --description "Cloud-native price intelligence and supply chain optimization pipeline for the Nigerian tobacco market."
```

**Public is deliberate.** GitHub Actions minutes are unmetered on public repos,
and this project uses Actions as its entire compute layer. A private repo would
burn the 2,000-minute monthly allowance and break the free-tier premise.

Verify:

```bash
gh repo view batestguy/tobacco-price-intelligence
```

---

## Step 1 — Supabase

Free tier: 500 MB Postgres. **It pauses after 7 days of inactivity** — the
twice-daily cron incidentally keeps it awake.

1. Create a project at [supabase.com/dashboard](https://supabase.com/dashboard).
2. Open **SQL Editor**, paste the whole of [`supabase/schema.sql`](supabase/schema.sql),
   run it. It is safe to re-run.
3. Collect from **Project Settings → API**:
   - Project URL → `SUPABASE_URL`
   - `service_role` key → `SUPABASE_SERVICE_KEY` (server-side only, never the app)
   - `anon` key → used later, in Streamlit only

**If you skip this:** jobs still run and still commit Parquet to the repo — that
is the source of truth. You lose the live serving layer and dashboard auth.

---

## Step 2 — Repository secrets

`gh secret set` prompts for the value. **Do not pass it as an argument** — it
would land in your shell history.

```bash
cd "D:\Tobacco Project"

gh secret set SUPABASE_URL
gh secret set SUPABASE_SERVICE_KEY
gh secret set GROQ_API_KEY
gh secret set GMAIL_ADDRESS
gh secret set GMAIL_APP_PASSWORD
gh secret set HF_TOKEN
```

| Secret | Where to get it | If missing |
|---|---|---|
| `SUPABASE_URL` | Supabase → Settings → API → Project URL | No Supabase mirror |
| `SUPABASE_SERVICE_KEY` | same page → `service_role` | No Supabase mirror |
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) → API Keys | Memo prints the raw figures instead of prose |
| `GMAIL_ADDRESS` | the sending Gmail account | Alerts evaluate and log, but send nothing |
| `GMAIL_APP_PASSWORD` | Google Account → Security → **App passwords** (needs 2FA on first) | as above |
| `HF_TOKEN` | huggingface.co → Settings → Access Tokens, **write** scope | Phase 3 only; base FinBERT downloads anonymously |

Every one of these is optional to *start*. `config.require()` raises a named
error and the affected stage degrades; it does not corrupt anything.

Confirm (prints names and timestamps, never values):

```bash
gh secret list
```

### Repository variables (not secret)

```bash
gh variable set ALERT_RECIPIENTS_COMMERCIAL --body "director@example.com"
gh variable set ALERT_RECIPIENTS_SUPPLY     --body "supply@example.com"
```

Without recipients, alert rules still evaluate and log but have nowhere to send.

---

## Step 3 — First run

```bash
gh workflow run scrape.yml
gh run watch
```

**What success looks like:** a new commit authored by `github-actions[bot]`
containing a Parquet file under `data/curated/`. That commit is the proof that
cloud compute is writing to cloud storage with no local process involved.

```bash
git pull
git log --oneline -3
ls data/curated/exchange_rates/
```

Then, in order:

```bash
gh workflow run train.yml      # generates 3y synthetic sales, fits XGBoost
gh run watch
gh workflow run recommend.yml  # needs a trained model to exist first
gh run watch
```

`score.yml` chains off `scrape.yml` automatically via `workflow_run`; you can
also dispatch it manually. It is the slow one — it installs CPU PyTorch and
downloads ~440 MB of FinBERT weights on the first run, then hits the cache.

If a run fails:

```bash
gh run view --log-failed
```

---

## Step 4 — Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io), **New app**, point at
   this repo, main file `app/streamlit_app.py`.
2. **App settings → Secrets**, paste:

   ```toml
   SUPABASE_URL = "https://<project>.supabase.co"
   SUPABASE_ANON_KEY = "<anon key>"
   ```

   Streamlit does **not** inherit GitHub Actions secrets — these are set
   separately. Give it the **anon** key, never the service key: the app is
   publicly reachable and the key ships to the browser's session.

3. Create a user in Supabase → **Authentication → Users**, then assign a role:

   ```sql
   insert into users (id, username, role)
   values ('<uuid from auth.users>', 'you@example.com', 'admin')
   on conflict (id) do update set role = excluded.role;
   ```

   Roles: `commercial_director`, `supply_chain_manager`, `admin`.

4. Record the app URL in [`REGISTRY.md`](REGISTRY.md).

Free tier: 1 GB RAM, sleeps after 12 idle hours, wakes on visit. The app reads
the committed Parquet directly, so it renders even before Supabase exists —
though you will not get past the login screen until Auth is configured.

---

## Step 5 — Transfer learning (optional, Phase 3)

Only worth doing once the scrapers have accumulated a few months of headlines.

1. Upload [`notebooks/finbert_transfer_learning.ipynb`](notebooks/finbert_transfer_learning.ipynb)
   to Kaggle, enable the **T4 GPU** accelerator (30 h/week free).
2. Run it. It exports a sample for manual labelling — that part is genuinely
   manual, and the notebook explains why training on the base model's own scores
   would only distil its mistakes.
3. It pushes weights to the HF Hub, then:

   ```bash
   gh variable set FINBERT_MODEL --body batestguy/finbert-ng-financial
   ```

   No code change: `nlp/finbert.py` reads that variable.

---

## Free-tier limits worth remembering

| Service | Limit | Consequence |
|---|---|---|
| GitHub Actions | unmetered on **public** repos | going private starts a 2,000 min/month clock |
| Scheduled workflows | auto-disabled after **60 days** of repo inactivity | the daily data commits prevent this |
| Supabase | 500 MB; **pauses after 7 days idle** | the cron keeps it warm |
| Streamlit Cloud | 1 GB RAM; sleeps after 12 h idle | keep `requirements.txt` lean — no torch |
| Groq | ~1000 req/day | one memo/day is nothing |
| Gmail SMTP | 100 emails/day | at most 4 alerts/day |
| HF Hub | 100 GB storage | model weights live here, not in git |
| Kaggle | 30 h/week T4, 12 h/session | fine-tuning only |
| Git LFS | **not used** — 1 GB/month bandwidth is spent by every Actions checkout | |
