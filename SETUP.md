# Setup runbook

Everything needed to take this repository from "code exists" to "pipeline is
running in the cloud". Work top to bottom; each step states what breaks if you
skip it, so you can stop partway and still have something that works.

Nothing here runs on your machine. The only local tools are `git` and `gh`, plus the
Kaggle CLI in WSL for Step 5, which submits a notebook to Kaggle's GPU.

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

## Step 1 — Supabase (dashboard login only)

Supabase is used for **Auth and nothing else**. No pipeline data goes near it:
the jobs write Parquet to the repo and the dashboard reads that Parquet out of
its own checkout. `schema.sql` is correspondingly small — one `users` table
holding role assignments, keyed off `auth.users`.

1. Create a project at [supabase.com/dashboard](https://supabase.com/dashboard).
2. Open **SQL Editor**, paste the whole of [`supabase/schema.sql`](supabase/schema.sql),
   run it. It is safe to re-run.
3. Collect from **Project Settings → API**:
   - Project URL → `SUPABASE_URL`
   - `anon` key → `SUPABASE_ANON_KEY`

   Both are for **Streamlit only** (Step 4). No GitHub Actions secret needs
   them, and the `service_role` key is not used by this project at all.

**If you skip this:** the entire pipeline still runs and still commits Parquet.
The only thing you lose is dashboard login — the app renders
"Authentication is not configured" and shows nothing past it.

Free tier: 500 MB Postgres, **paused after 7 days of inactivity**. Nothing in the
pipeline keeps it awake any more, so expect to resume it by hand if logins start
failing after a quiet week.

---

## Step 2 — Repository secrets

`gh secret set` prompts for the value. **Do not pass it as an argument** — it
would land in your shell history.

```bash
cd "D:\Tobacco Project"

gh secret set GROQ_API_KEY
gh secret set GMAIL_ADDRESS
gh secret set GMAIL_APP_PASSWORD
gh secret set HF_TOKEN
```

| Secret | Where to get it | If missing |
|---|---|---|
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

**Status 2026-09-24:** labels exist (`data/labels/headlines_labelled.csv`, 344 rows,
made by the process in [`docs/labelling-guide.md`](docs/labelling-guide.md)), and the
notebook reads them from the repo. What is left is running it on a GPU.

The compute is Kaggle's, never the local machine's. Two ways to get it there:

### Route A: Kaggle web UI (simplest)

1. On kaggle.com, create a notebook and import
   [`notebooks/finbert_transfer_learning.ipynb`](notebooks/finbert_transfer_learning.ipynb).
2. **Settings → Accelerator → GPU T4**, and **Internet → on**. Both need a
   phone-verified Kaggle account.
3. **Add-ons → Secrets →** add `HF_TOKEN` (write scope) and tick it for this notebook.
4. **Run all**. The label-export cell (5) is off by default (`RUN_LABELLING_EXPORT`).

### Route B: Kaggle CLI (scriptable, from WSL)

The Kaggle CLI only *submits* the notebook; it runs on Kaggle's T4. Use WSL:
there the CLI is already signed in (OAuth), while the Windows copy is not.

```bash
# in WSL (Ubuntu), in a scratch folder outside the repo
cp /mnt/d/Tobacco\ Project/notebooks/finbert_transfer_learning.ipynb .
cat > kernel-metadata.json <<'JSON'
{
  "id": "<kaggle-username>/finbert-ng-financial",
  "title": "finbert-ng-financial",
  "code_file": "finbert_transfer_learning.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true
}
JSON
kaggle kernels push -p .
kaggle kernels status <kaggle-username>/finbert-ng-financial   # poll until complete
kaggle kernels output <kaggle-username>/finbert-ng-financial -p out/
```

**The CLI cannot attach secrets**, so a CLI run cannot push to the Hub itself. The
notebook handles that: it always saves the weights and `metrics.json` to
`/kaggle/working`, and pushes only when an `HF_TOKEN` secret is attached. A CLI run's
weights reach the Hub through a GitHub Release and `publish-model.yml`, which pushes
with the repo's own `HF_TOKEN` secret:

```bash
kaggle kernels output <kaggle-username>/finbert-ng-financial -p out/
cat out/metrics.json                       # tuned must beat base
tar -czf finbert-ng-financial.tar.gz -C out finbert-ng-financial
gh release create finbert-ng-v1 finbert-ng-financial.tar.gz     --title "FinBERT fine-tune v1" --notes "Weights for publish-model.yml"
gh workflow run publish-model.yml -f tag=finbert-ng-v1
```

On a connection that drops large downloads, skip the local copy: pass the output's
signed download URLs instead (`list_kernel_session_output` in the Kaggle SDK returns
one per file, no auth needed to fetch), and the runner fetches the weights itself:
`gh workflow run publish-model.yml -f urls='{"config.json": "…", "model.safetensors": "…", …}'`.
Include every file in `finbert-ng-financial/`, `metrics.json` among them, so the Hub repo
carries the held-out accuracy next to the weights. The first run's figures are also
committed at `models/finbert_metrics.json`.
The job masks them, but they are short-lived; dispatch right after generating them.

The notebook also sets `enable_internet`, which needs a phone-verified Kaggle account.

**Why not Colab?** The Colab CLI (`colab run --gpu T4 script.py`) exists and is
signed in under WSL, but it runs a `.py` script rather than this notebook. It has
no secret store, so `kaggle_secrets` in the push cell would fail, and free-tier GPU
availability through the CLI is unverified. Keep it as a fallback, not the route.

### Either route

1. The notebook uploads to the HF Hub only if the tuned model beats the base one on
   the held-out split. Otherwise it stops at an `assert`.
2. After a push:

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
| Supabase | 500 MB; **pauses after 7 days idle** | logins stop until you resume it; no data lost — it holds only `users` |
| Streamlit Cloud | 1 GB RAM; sleeps after 12 h idle | keep `requirements.txt` lean — no torch |
| Groq | ~1000 req/day | one memo/day is nothing |
| Gmail SMTP | 100 emails/day | at most 4 alerts/day |
| HF Hub | 100 GB storage | model weights live here, not in git |
| Kaggle | 30 h/week T4, 12 h/session | fine-tuning only |
| Git LFS | **not used** — 1 GB/month bandwidth is spent by every Actions checkout | |
