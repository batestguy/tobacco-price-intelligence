"""Data access for the dashboard.

Reads the **committed Parquet** in ``data/curated/``. Streamlit Community Cloud
checks the repository out to run the app, so the repo's own copy of the data is
already on local disk -- no database round trip and no key of any kind.

This is the only read path there has ever been. Supabase once held a mirror of
every table that nothing queried; it was removed (CLAUDE.md departure 5) and
Supabase now serves Auth alone. A paused free-tier project therefore costs the
login, never the data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# The app runs from app/, so make the package importable without installing it.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tobacco import config  # noqa: E402
from tobacco.store import parquet_io  # noqa: E402

#: Data changes at most twice a day; anything shorter just re-reads Parquet.
CACHE_TTL_SECONDS = 900


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load(dataset: str) -> pd.DataFrame:
    try:
        return parquet_io.read(dataset)
    except Exception as exc:  # noqa: BLE001 - an empty panel beats a stack trace
        st.warning(f"Could not load {dataset}: {exc}")
        return pd.DataFrame()


def latest_fx() -> tuple[float | None, float | None, bool]:
    """``(rate, 7-day change %, is_carried_forward)`` for the headline card."""
    rates = load("exchange_rates")
    if rates.empty:
        return None, None, False

    rates = rates.sort_values("date")
    latest_row = rates.iloc[-1]
    rate = float(latest_row["usd_ngn_rate"])
    carried = bool(latest_row.get("is_carried_forward", False))

    week_ago = pd.to_datetime(latest_row["date"]) - pd.Timedelta(days=7)
    prior = rates[pd.to_datetime(rates["date"]) <= week_ago]
    if prior.empty:
        return rate, None, carried

    previous = float(prior.iloc[-1]["usd_ngn_rate"])
    change = (rate - previous) / previous * 100 if previous else None
    return rate, change, carried


def latest_sentiment() -> tuple[float | None, float | None]:
    """``(consumer_sentiment, fx_crisis_prob)``, most recent non-null of each."""
    aggregates = load("sentiment_aggregates")
    if aggregates.empty:
        return None, None
    aggregates = aggregates.sort_values("date")

    def last_valid(column: str) -> float | None:
        if column not in aggregates:
            return None
        values = aggregates[column].dropna()
        return float(values.iloc[-1]) if not values.empty else None

    return last_valid("consumer_sentiment"), last_valid("fx_crisis_prob")


#: ``(basis, tooltip, caption)`` per tier of the inflation cascade
#: (``sources/nbs.py``). The card must say which tier it is showing, and two of
#: them need a visible caption rather than only a tooltip: the annual tier is a
#: full calendar year carried forward and would otherwise read as a current
#: monthly rate, and the GEM tier *is* monthly but is a seasonally adjusted
#: World Bank calculation, so it will not match the NBS figure in the news.
INFLATION_BASIS = {
    "cbn_monthly": ("Monthly", "CBN, republishing the NBS CPI series", None),
    "nbs_release": ("Monthly", "Direct NBS release", None),
    "seed": ("Monthly", "Committed NBS back series", None),
    "worldbank_gem_monthly": (
        "Monthly",
        "World Bank Global Economic Monitor, indicator CPTOTSAXNZGY — monthly "
        "and current, but a seasonally adjusted World Bank staff calculation "
        "rather than the headline rate NBS publishes.",
        "⚠️ World Bank GEM, seasonally adjusted — not NBS's published figure",
    ),
    "worldbank_annual": (
        "Annual",
        "World Bank FP.CPI.TOTL.ZG — the last full calendar year, carried "
        "forward. No monthly series is currently reachable.",
        "⚠️ Annual basis — no monthly series available",
    ),
}


def latest_inflation() -> tuple[float | None, str | None, str | None, str | None]:
    """``(rate, basis, explanation, caption)`` for the newest inflation row."""
    inflation = load("inflation")
    if inflation.empty:
        return None, None, None, None

    row = inflation.sort_values("date").iloc[-1]
    source = str(row.get("source") or "unknown")
    basis, explanation, caption = INFLATION_BASIS.get(
        source,
        (
            "Unknown basis",
            f"Unrecognised source tier '{source}'",
            f"⚠️ Unrecognised source tier '{source}'",
        ),
    )
    observed = pd.to_datetime(row["date"]).date()
    return float(row["rate"]), basis, f"{explanation} Observed {observed}.", caption


def latest_recommendations() -> pd.DataFrame:
    """Only the most recent day's recommendations."""
    recommendations = load("recommendations")
    if recommendations.empty:
        return recommendations
    most_recent = pd.to_datetime(recommendations["date"]).max()
    return recommendations[pd.to_datetime(recommendations["date"]) == most_recent]


def latest_memo() -> tuple[str | None, str | None]:
    """``(markdown, date)`` of the newest generated memo."""
    memo_dir = REPO_ROOT / "data" / "memos"
    if not memo_dir.exists():
        return None, None
    memos = sorted(memo_dir.glob("*.md"))
    if not memos:
        return None, None
    newest = memos[-1]
    return newest.read_text(encoding="utf-8"), newest.stem


def model_metrics() -> dict:
    import json

    if not config.METRICS_PATH.exists():
        return {}
    try:
        return json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
