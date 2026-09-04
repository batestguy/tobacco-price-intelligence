"""The synthetic sales generator (INTRO.txt §2 ``sales_mock``).

Under the tmp-dir fixture there is no scraped FX series, so ``_fx_by_week()``
returns ``{}`` and ``fallback_fx`` is ``FX_REFERENCE``. That removes the one input
that makes a real regeneration non-reproducible and leaves ``generate`` fully
deterministic -- which is what makes the anchor test below meaningful rather than
noisy.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from tobacco import config
from tobacco.sources import sales_mock


START = date(2026, 1, 5)   # a Monday
END = date(2026, 3, 30)


def test_no_scraped_fx_means_the_generator_runs_at_the_reference_rate():
    assert sales_mock._fx_by_week() == {}

    frame = sales_mock.generate(START, END)
    premium = frame[frame["sku"] == "PREMIUM_20"]["avg_price_ngn"]

    # Only the ±1.2% per-row price noise separates these from BASE_PRICE_NGN.
    assert (premium / config.BASE_PRICE_NGN["PREMIUM_20"]).between(0.94, 1.06).all()


def test_generate_is_deterministic_for_a_fixed_range():
    """Determinism is per row, not per run: each (week, sku, region) seeds its own
    generator, so a changed date range does not reshuffle the rows it still covers."""
    pd.testing.assert_frame_equal(sales_mock.generate(START, END), sales_mock.generate(START, END))


def test_prices_are_anchor_independent_but_quantities_are_not():
    """The trap behind ``train.py`` gating its regeneration behind an explicit flag.

    ``price_here`` is drawn from the row-keyed RNG, so it survives a change of
    start date. ``quantity_sold`` does not: ``trend`` is ``(1+SKU_TREND)**week_index``
    and ``week_index`` counts from the *requested* start, so moving the anchor
    rewrites every quantity in the overlap while leaving every price alone.
    """
    keys = ["week_start", "sku", "region"]
    later = sales_mock.generate(START, END).set_index(keys).sort_index()
    earlier = (
        sales_mock.generate(START - timedelta(weeks=8), END)
        .set_index(keys)
        .sort_index()
        .loc[later.index]
    )

    pd.testing.assert_series_equal(later["avg_price_ngn"], earlier["avg_price_ngn"])
    assert not later["quantity_sold"].equals(earlier["quantity_sold"])


def test_generate_covers_every_sku_and_region_for_every_week():
    frame = sales_mock.generate(START, END)
    weeks = frame["week_start"].nunique()

    assert len(frame) == weeks * len(config.SKUS) * len(config.REGIONS)
    assert set(frame["sku"]) == set(config.SKUS)
    assert set(frame["region"]) == set(config.REGIONS)


def test_generate_snaps_to_week_starting_mondays():
    frame = sales_mock.generate(date(2026, 1, 7), date(2026, 2, 4))  # a Wednesday
    weeks = frame["week_start"].drop_duplicates().sort_values()

    assert weeks.iloc[0] == pd.Timestamp("2026-01-05")
    assert (weeks.dt.dayofweek == 0).all()


def test_generate_emits_exactly_the_declared_columns():
    """``COLUMNS`` is half of the §2 contract for this dataset -- the other half
    being the key and partition in ``parquet_io.DATASETS``."""
    assert list(sales_mock.generate(START, END).columns) == sales_mock.COLUMNS


def test_holiday_weeks_carry_the_uplift():
    frame = sales_mock.generate(date(2025, 12, 1), date(2026, 1, 12))
    # One tier, all four regions: 8 holiday rows against 20 ordinary ones, so the
    # 18% lift sits well clear of the 6% per-row noise.
    frame = frame[frame["sku"] == "VALUE_20"]
    frame = frame.assign(
        holiday=frame["week_start"].dt.date.map(config.is_holiday_week)
    )

    assert frame["holiday"].any() and not frame["holiday"].all()
    assert (
        frame[frame["holiday"]]["quantity_sold"].mean()
        > frame[~frame["holiday"]]["quantity_sold"].mean()
    )
