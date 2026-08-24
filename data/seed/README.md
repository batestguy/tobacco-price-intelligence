# Seed data

Historical series that cannot be scraped, checked in once so the forecaster has
history from its first run.

## `inflation.csv` — not included

NBS publishes CPI as monthly PDF/XLSX releases with no stable bulk historical
endpoint, so the back series has to be assembled by hand.

**It is deliberately absent rather than filled with plausible numbers.** Inflation
is an official statistic; inventing a back series would put fabricated figures
into the model's features and into anything the memo says about them. The
pipeline handles its absence — `sources/nbs.py` returns an empty frame, the
`inflation` feature is null, and XGBoost splits on missing values natively.

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
different orderings are tolerated. Rows fetched live from NBS take precedence
over seed rows for the same month.
