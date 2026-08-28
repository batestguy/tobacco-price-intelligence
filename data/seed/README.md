# Seed data

Series that cannot — or should not — be scraped, checked in by hand so the
forecaster and the optimizer have something real to work from.

Both files here are **empty or absent by design**. The pipeline is built to
degrade when they are, and an empty file is a truthful statement that we do not
have the data. Filling one with plausible-looking numbers would put fabricated
figures into the model's features and into anything the memo says about them.

## `inflation.csv` — not included

NBS publishes CPI as monthly PDF/XLSX releases with no stable bulk historical
endpoint, so the back series has to be assembled by hand.

**It is deliberately absent rather than filled with plausible numbers.** Inflation
is an official statistic; inventing a back series would put fabricated figures
into the model's features.

Its absence is survivable. `sources/nbs.py` runs a five-tier cascade and, with
this file missing and CBN's monthly table rendering its rows client-side,
inflation currently comes from the **World Bank Global Economic Monitor**
(`CPTOTSAXNZGY`, `source=15`) — monthly back to 2015M01, cited and key-free.

That covers the *cadence* problem this file was going to solve; it does not cover
the *provenance* one. GEM is a seasonally adjusted World Bank staff calculation,
so it does not reproduce NBS's published headline rate — the two diverge by
roughly 3pp since Nigeria rebased its CPI in January 2025. It is a sound model
feature and a poor citation.

So this file is still worth filling, for a narrower reason than before: it is the
only way to get **NBS's own published monthly figures** into the series. It
outranks both World Bank tiers for every month it covers.

To populate it, download the CPI time series from
[nigerianstat.gov.ng](https://nigerianstat.gov.ng) and save it here as:

```csv
date,rate,food_rate,core_rate
2023-01-01,21.82,24.32,19.16
2023-02-01,21.91,24.35,18.84
```

- `date` — first of the month
- `rate` — headline year-on-year inflation, %
- `food_rate`, `core_rate` — optional

`sources/nbs.py` parses this file by column *meaning*, so extra columns and
different orderings are tolerated. Seed rows outrank both World Bank tiers for
months they cover, and are outranked in turn by CBN's monthly table and by an
explicit `NBS_INFLATION_URL` release. Each row records its tier in `source`.

## `competitor_prices.csv` — header only

This file replaces the Jumia and Konga marketplace scrapers, which have been
**removed**. They are not coming back, and the reason is not that they broke
(Jumia 403'd the runner, Konga parsed empty):

> **Section 15(4) of the National Tobacco Control Act 2015 prohibits the sale of
> tobacco products over the internet in Nigeria.**

That is the same Act the dashboard disclaimer cites verbatim. The optimizer's
price ceiling cannot credibly be derived from listings that the project's own
stated legal framework bans. Enforcement is weak in practice — retailers do list,
which is why the scraper was written in the first place — but that is precisely
what makes the data sporadic and unrepresentative. It was a poor basis for a
pricing constraint on the merits, quite apart from how it read.

`optimize/linprog.py` needs no competitor data to function: with none, it drops
the ceiling from the candidate filter and says so in an optimizer *note* ("the
+5% competitor ceiling is INACTIVE"), rather than inventing a ceiling. `INACTIVE`
is never a `binding_constraint` value — that column only ever names a limit that
actually bound. It was built for this case. The `competitor_prices` dataset and its §2 schema are unchanged.

To populate it, record prices you can **cite** — a published price survey, a
regulator or trade-association filing, a dated retail audit — one row per quote:

```csv
date,brand,price,region,source,product_title,url
2026-08-01,Bohem,1400,National,NBS retail price survey Aug 2026,Bohem Cigarette 20s,https://...
```

- `date` — the date the price was *observed*, not the date you typed it in
- `brand` — must be one of `config.COMPETITOR_BRANDS`; other rows are dropped
- `price` — NGN per 20-pack; rows outside 200–20 000 NGN are dropped as sticks,
  cartons, or typos
- `region` — defaults to `National`; per-region rows are welcome if the survey
  actually breaks out by region, and must not be invented if it does not
- `source` — **required.** The citation. A row without one is dropped
- `product_title`, `url` — optional; a title that does not name its own brand is
  treated as a mis-entry and dropped

Rows are re-read on every scrape and upserted by `(date, brand, region, source)`,
so re-running is safe and correcting a row in place corrects the dataset.
