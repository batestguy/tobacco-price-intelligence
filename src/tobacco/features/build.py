"""Build the modelling panel (INTRO.txt §4).

The target is **weekly demand per SKU per region**, so the panel is keyed
``(week_start, sku, region)``. Every exogenous series arrives on a different
clock -- FX is daily, inflation monthly, sentiment per scrape -- so they are first
resampled onto a complete daily index, then sampled at lags relative to each
week's start date.

That daily intermediate matters. The spec asks for FX at ``t-1, t-7, t-30``, and
those are *days*, not rows: taking "the previous three FX rows" would silently
mean something different across a weekend or a public holiday when CBN publishes
nothing.

**Leakage discipline:** every exogenous value is read at or before ``week_start``,
never within the week being predicted. ``_daily_exog`` forward-fills only, so a
gap is filled from the past.
"""

from __future__ import annotations

import logging
import re
from datetime import date, timedelta

import numpy as np
import pandas as pd

from tobacco import config
from tobacco.store import parquet_io

log = logging.getLogger(__name__)

KEYS = ["week_start", "sku", "region"]
TARGET = "quantity_sold"

#: FX lags in days, verbatim from the spec.
FX_LAGS = (1, 7, 30)
#: Sentiment lags in days, verbatim from the spec.
SENTIMENT_LAGS = (1, 3)

#: Headlines mentioning these signal an excise/VAT change (the spec's "tax
#: change dummy"). There is no free machine-readable gazette feed, so the flag is
#: derived from news text -- noisier than a gazette, but honest about its source.
TAX_TERMS = re.compile(
    r"\b(?:excise|vat\b|value[- ]added tax|tariff|levy|customs duty|tax hike|"
    r"tax increase|sin tax)\b",
    re.IGNORECASE,
)


def _daily_index(start: date, end: date) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D")


def _daily_exog(start: date, end: date) -> pd.DataFrame:
    """All exogenous series on one complete, forward-filled daily index."""
    index = _daily_index(start, end)
    exog = pd.DataFrame(index=index)

    # --- FX -----------------------------------------------------------------
    fx = parquet_io.read("exchange_rates")
    if fx.empty:
        log.warning("No FX data; fx features will be null")
        exog["fx"] = np.nan
    else:
        fx = fx.copy()
        fx["date"] = pd.to_datetime(fx["date"])
        series = fx.groupby("date")["usd_ngn_rate"].mean()
        exog["fx"] = series.reindex(index).ffill()

    # --- inflation (monthly, held flat until the next release) ---------------
    inflation = parquet_io.read("inflation")
    if inflation.empty:
        exog["inflation"] = np.nan
    else:
        inflation = inflation.copy()
        inflation["date"] = pd.to_datetime(inflation["date"])
        series = inflation.groupby("date")["rate"].mean()
        exog["inflation"] = series.reindex(index).ffill()

    # --- competitor price index ---------------------------------------------
    # Cited reference prices are national and sparse (see sources/competitors.py),
    # so the index is a simple cross-brand mean carried forward between
    # observations -- and stays NaN entirely when none have been recorded.
    competitors = parquet_io.read("competitor_prices")
    if competitors.empty:
        exog["competitor_index"] = np.nan
    else:
        competitors = competitors.copy()
        competitors["date"] = pd.to_datetime(competitors["date"])
        series = competitors.groupby("date")["price"].mean()
        exog["competitor_index"] = series.reindex(index).ffill()

    # --- news crisis score + tax flag ---------------------------------------
    news = parquet_io.read("news_articles")
    if news.empty or "finbert_score" not in news.columns:
        exog["crisis"] = np.nan
        exog["tax_change"] = 0.0
    else:
        news = news.copy()
        news["day"] = pd.to_datetime(news["published_at"]).dt.normalize()
        scored = news.dropna(subset=["finbert_score"])
        daily_crisis = scored.groupby("day")["finbert_score"].mean()
        exog["crisis"] = daily_crisis.reindex(index).ffill()

        tax_hits = (
            news.assign(hit=news["headline"].fillna("").str.contains(TAX_TERMS))
            .groupby("day")["hit"]
            .max()
            .astype(float)
        )
        # A tax announcement keeps mattering for a while, so the dummy stays hot
        # for four weeks rather than firing for exactly one day.
        exog["tax_change"] = (
            tax_hits.reindex(index).fillna(0.0).rolling(28, min_periods=1).max()
        )

    # --- social sentiment ----------------------------------------------------
    social = parquet_io.read("social_posts")
    if social.empty or "vader_score" not in social.columns:
        exog["sentiment"] = np.nan
    else:
        social = social.copy()
        social["day"] = pd.to_datetime(social["published_at"]).dt.normalize()
        series = social.groupby("day")["vader_score"].mean()
        exog["sentiment"] = series.reindex(index).ffill()

    return exog


def _lagged_at(exog: pd.DataFrame, when: pd.Series, column: str, lag_days: int) -> pd.Series:
    """Value of ``column`` ``lag_days`` before each date in ``when``."""
    target_dates = pd.to_datetime(when) - pd.Timedelta(days=lag_days)
    return target_dates.map(exog[column]).astype(float)


def attach_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach every §4 feature to a panel keyed on ``KEYS``."""
    if panel.empty:
        return panel

    panel = panel.copy()
    panel["week_start"] = pd.to_datetime(panel["week_start"])

    # The daily index must reach back far enough to cover the longest lag.
    start = (panel["week_start"].min() - pd.Timedelta(days=max(FX_LAGS) + 7)).date()
    end = panel["week_start"].max().date()
    exog = _daily_exog(start, end)

    for lag in FX_LAGS:
        panel[f"fx_lag_{lag}d"] = _lagged_at(exog, panel["week_start"], "fx", lag)
    # Rate of change carries the signal a level cannot: 1600 NGN/USD means one
    # thing after a slow drift and another after a one-week devaluation.
    panel["fx_change_7d_pct"] = (
        (panel["fx_lag_1d"] - panel["fx_lag_7d"]) / panel["fx_lag_7d"] * 100
    )

    panel["inflation"] = _lagged_at(exog, panel["week_start"], "inflation", 1)
    panel["competitor_index"] = _lagged_at(exog, panel["week_start"], "competitor_index", 1)
    panel["tax_change"] = _lagged_at(exog, panel["week_start"], "tax_change", 1).fillna(0.0)

    for lag in SENTIMENT_LAGS:
        panel[f"crisis_lag_{lag}d"] = _lagged_at(exog, panel["week_start"], "crisis", lag)
        panel[f"sentiment_lag_{lag}d"] = _lagged_at(exog, panel["week_start"], "sentiment", lag)

    # --- calendar ------------------------------------------------------------
    panel["holiday_week"] = panel["week_start"].dt.date.map(config.is_holiday_week).astype(float)
    panel["week_of_year"] = panel["week_start"].dt.isocalendar().week.astype(float)
    panel["month"] = panel["week_start"].dt.month.astype(float)

    # --- own price, lagged ---------------------------------------------------
    if "avg_price_ngn" in panel.columns:
        panel = panel.sort_values(KEYS)
        group = panel.groupby(["sku", "region"], sort=False)["avg_price_ngn"]
        panel["own_price_lag_1w"] = group.shift(1)
        panel["own_price_lag_4w"] = group.shift(4)
        # Relative price is what a shopper actually reacts to.
        panel["price_vs_competitor"] = (
            panel["own_price_lag_1w"] / panel["competitor_index"]
        )

    # --- categorical dummies -------------------------------------------------
    # Reindexed against the full configured vocabulary so that a region absent
    # from one slice cannot change the column set between train and predict.
    for region in config.REGIONS:
        panel[f"region_{region.replace(' ', '_')}"] = (panel["region"] == region).astype(float)
    for sku in config.SKUS:
        panel[f"sku_{sku}"] = (panel["sku"] == sku).astype(float)

    return panel


def feature_columns(panel: pd.DataFrame) -> list[str]:
    """Model input columns: everything numeric that is not a key or the target."""
    exclude = set(KEYS) | {TARGET, "avg_price_ngn", "stock_on_hand"}
    return [
        column
        for column in panel.columns
        if column not in exclude and pd.api.types.is_numeric_dtype(panel[column])
    ]


def build_training_frame() -> pd.DataFrame:
    """Historical panel with features and target, ready for XGBoost."""
    sales = parquet_io.read("sales_mock")
    if sales.empty:
        log.error("sales_mock is empty; run the backfill before training")
        return pd.DataFrame()

    panel = attach_features(sales)
    before = len(panel)
    # The first weeks have no own-price lag and the earliest weeks may predate
    # the FX series. Drop them rather than imputing a fabricated history.
    panel = panel.dropna(subset=["own_price_lag_4w"])
    log.info("Training panel: %d rows (%d dropped for incomplete lags)",
             len(panel), before - len(panel))
    return panel.reset_index(drop=True)


def build_forecast_frame(horizon: int = config.FORECAST_HORIZON_WEEKS) -> pd.DataFrame:
    """Feature rows for the next ``horizon`` weeks, for every SKU and region.

    Exogenous drivers are **held at their last observed value**. This is a
    "current conditions persist" scenario, not a forecast of FX or sentiment --
    predicting those would need its own models, and the spec does not ask for
    them. Calendar features (holiday, week-of-year) are known in advance and are
    computed properly for each future week.
    """
    sales = parquet_io.read("sales_mock")
    if sales.empty:
        log.error("sales_mock is empty; cannot construct a forecast frame")
        return pd.DataFrame()

    sales = sales.copy()
    sales["week_start"] = pd.to_datetime(sales["week_start"])
    last_week = sales["week_start"].max()

    # Carry each series' last actual price forward as the assumed price.
    last_prices = (
        sales.sort_values("week_start")
        .groupby(["sku", "region"], as_index=False)
        .last()[["sku", "region", "avg_price_ngn", "stock_on_hand"]]
    )

    future_weeks = [last_week + pd.Timedelta(weeks=i) for i in range(1, horizon + 1)]
    future = pd.MultiIndex.from_product(
        [future_weeks, config.SKUS, config.REGIONS], names=KEYS
    ).to_frame(index=False)
    future = future.merge(last_prices, on=["sku", "region"], how="left")

    # Features need the recent history for the own-price lags to resolve, so
    # build on the tail of actuals and then keep only the future rows.
    history_tail = sales[sales["week_start"] > last_week - pd.Timedelta(weeks=8)]
    combined = pd.concat(
        [history_tail[future.columns.intersection(history_tail.columns)], future],
        ignore_index=True,
    )
    panel = attach_features(combined)
    panel = panel[panel["week_start"] > last_week].reset_index(drop=True)

    log.info("Forecast panel: %d rows over %d week(s) from %s",
             len(panel), horizon, future_weeks[0].date())
    return panel
