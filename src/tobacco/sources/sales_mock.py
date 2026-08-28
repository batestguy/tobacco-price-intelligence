"""Synthetic weekly sales (INTRO.txt §2 ``sales_mock``, §4 "3+ years").

There is no real sales data and there must not be: this repo is public, and
inventing plausible-looking figures for a named manufacturer would be worse than
useless. The forecaster is instead trained on an explicit, reproducible synthetic
process whose structure is documented here, so anyone reading the model's metrics
knows exactly what signal it is recovering.

The generator embeds the relationships the pipeline is supposed to learn:

* a regional base level and a slow upward volume trend;
* annual seasonality plus holiday uplift (Christmas/New Year and Sallah);
* **own-price elasticity** -- volume responds to price via
  ``config.PRICE_ELASTICITY``;
* **FX pass-through** -- a weaker naira raises input costs, which raises price;
* mild downtrend on the premium tier and uptrend on value (down-trading).

Determinism is per row, not per run: each ``(week, sku, region)`` seeds its own
generator, so the *noise* in any row is reproducible from its key alone, and a
changed date range does not reshuffle the rows it still covers. A run-level seed
would not survive that.

**This is not the same as a regeneration being byte-identical, and it used to
claim it was.** The RNG is stable but the generator's *inputs* are not:
``fallback_fx`` is a running mean over the whole scraped FX series and therefore
moves with every twice-daily scrape, and ``config.PRICE_ELASTICITY`` feeds
``price_effect``. ``data/curated/exchange_rates/`` currently covers two months
against 37 ``sales_mock`` partitions, so the large majority of weeks take
``fallback_fx`` and shift slightly whenever it does. Treat a regeneration as
rewriting values, not as a no-op -- which is why ``jobs/train.py`` gates the full
rewrite behind an explicit flag rather than letting it happen by accident.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

from tobacco import config
from tobacco.store import parquet_io

log = logging.getLogger(__name__)

COLUMNS = [
    "week_start", "sku", "region", "quantity_sold",
    "avg_price_ngn", "stock_on_hand",
]

#: Mean weekly packs per region, before SKU mix.
REGION_BASE: dict[str, float] = {
    "Lagos": 42000.0,
    "Ibadan": 21000.0,
    "Kano": 26000.0,
    "Port Harcourt": 18000.0,
}

#: Share of regional volume by tier.
SKU_MIX: dict[str, float] = {
    "PREMIUM_20": 0.22,
    "MIDRANGE_20": 0.43,
    "VALUE_20": 0.35,
}

#: Long-run weekly drift per tier (down-trading as real incomes fall).
SKU_TREND: dict[str, float] = {
    "PREMIUM_20": -0.0007,
    "MIDRANGE_20": 0.0002,
    "VALUE_20": 0.0011,
}

#: FX level at which BASE_PRICE_NGN holds. Above it, prices drift up.
FX_REFERENCE = 1500.0
#: Share of an FX move that reaches the shelf price.
FX_PASSTHROUGH = 0.35


def _rng(week: date, sku: str, region: str) -> np.random.Generator:
    """Row-stable RNG. Python's ``hash()`` is salted per process, so hash explicitly."""
    seed_bytes = hashlib.blake2b(
        f"{week.isoformat()}|{sku}|{region}".encode("utf-8"), digest_size=8
    ).digest()
    return np.random.default_rng(int.from_bytes(seed_bytes, "big"))


def _monday_on_or_before(day: date) -> date:
    return day - timedelta(days=day.weekday())


def _fx_by_week() -> dict[date, float]:
    """Weekly mean FX from the real scraped series, when we have it."""
    rates = parquet_io.read("exchange_rates")
    if rates.empty:
        return {}
    rates = rates.copy()
    rates["week_start"] = (
        pd.to_datetime(rates["date"]).dt.to_period("W-SUN").dt.start_time.dt.date
    )
    return rates.groupby("week_start")["usd_ngn_rate"].mean().to_dict()


def generate(start: date, end: date) -> pd.DataFrame:
    """Weekly sales rows for every SKU/region between ``start`` and ``end``."""
    fx_by_week = _fx_by_week()
    fallback_fx = (
        float(np.mean(list(fx_by_week.values()))) if fx_by_week else FX_REFERENCE
    )

    week = _monday_on_or_before(start)
    last = _monday_on_or_before(end)
    rows: list[dict] = []
    week_index = 0

    while week <= last:
        # Exact-week lookup, not nearest: a week with no scraped FX takes
        # `fallback_fx` (the mean of the whole series) rather than its
        # neighbour's rate. With only recent months scraped, that means most of
        # the history sits at one flat synthetic level with a real series
        # attached to the tail. A taper toward FX_REFERENCE across the gap would
        # be better and is deliberately not implemented here -- it would change
        # every historical row, so it belongs in its own change with its own
        # regeneration, not smuggled into one.
        fx = fx_by_week.get(week, fallback_fx)
        holiday = config.is_holiday_week(week)
        # Week-of-year seasonality: a broad Q4 peak, trough in the wet season.
        seasonal = 1.0 + 0.12 * np.cos(2 * np.pi * (week.isocalendar().week - 50) / 52)

        for sku in config.SKUS:
            base_price = config.BASE_PRICE_NGN[sku]
            # Cost pressure from a weaker naira, partially passed through.
            fx_ratio = fx / FX_REFERENCE
            price = base_price * (1 + FX_PASSTHROUGH * (fx_ratio - 1))

            for region in config.REGIONS:
                rng = _rng(week, sku, region)
                price_here = price * rng.normal(1.0, 0.012)

                elasticity = config.PRICE_ELASTICITY[sku]
                price_effect = (price_here / base_price) ** elasticity

                level = REGION_BASE[region] * SKU_MIX[sku]
                trend = (1 + SKU_TREND[sku]) ** week_index
                holiday_lift = 1.18 if holiday else 1.0
                noise = rng.normal(1.0, 0.06)

                quantity = level * trend * seasonal * price_effect * holiday_lift * noise
                quantity = float(max(0.0, round(quantity)))

                # Stock is carried as a few weeks of cover with random slack --
                # enough for the optimizer to find genuine imbalances.
                cover_weeks = rng.uniform(1.2, 5.5)

                rows.append(
                    {
                        "week_start": pd.Timestamp(week),
                        "sku": sku,
                        "region": region,
                        "quantity_sold": quantity,
                        "avg_price_ngn": round(float(price_here), 2),
                        "stock_on_hand": float(round(quantity * cover_weeks)),
                    }
                )

        week += timedelta(days=7)
        week_index += 1

    frame = pd.DataFrame(rows, columns=COLUMNS)
    log.info(
        "Generated %d synthetic sales row(s) for %s..%s",
        len(frame), start, last,
    )
    return frame


def backfill(years: int = 3) -> pd.DataFrame:
    """Full synthetic history ending last complete week -- the training set."""
    end = _monday_on_or_before(config.today_wat()) - timedelta(days=7)
    start = end - timedelta(weeks=52 * years)
    return generate(start, end)
