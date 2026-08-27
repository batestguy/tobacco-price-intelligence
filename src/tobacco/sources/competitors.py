"""Competitor retail prices from a cited reference file (INTRO.txt §1).

**This used to scrape Jumia and Konga. It deliberately no longer does.**

Jumia returned HTTP 403 to the Actions runner and Konga parsed empty. The obvious
fix is a browser User-Agent and slower pacing -- i.e. defeating bot detection --
and the reason not to is not etiquette:

> Section 15(4) of the National Tobacco Control Act 2015 prohibits the sale of
> tobacco products over the internet in Nigeria.

That is the *same Act* the dashboard disclaimer cites verbatim. Deriving the
optimizer's price ceiling from listings that the project's own stated legal
framework bans is not defensible, and enforcement being weak in practice is
exactly what makes such listings sporadic and unrepresentative -- bad data as
well as awkward data. Scraping harder would have made the project worse on both
counts. See CLAUDE.md, settled departure 4.

What replaces it is a committed, dated reference file with a citation per row:
``data/seed/competitor_prices.csv``. It is currently header-only, and that is a
normal outcome, not a failure. ``optimize/linprog.py`` already detects missing
competitor data and reports its +5% price ceiling as INACTIVE rather than
inventing a number -- it was built for this case. The ``competitor_prices``
dataset and its §2 schema are unchanged, so nothing downstream had to move.

Reference prices are national, so ``region`` defaults to ``National``.
Region-level competitor pricing would require field survey data this project does
not have, and fabricating it would poison the optimizer's price ceiling.
"""

from __future__ import annotations

import logging
import re

import pandas as pd

from tobacco import config

log = logging.getLogger(__name__)

COLUMNS = ["date", "brand", "price", "region", "source", "product_title", "url"]

NATIONAL = "National"

SEED_PATH = config.SEED_DIR / "competitor_prices.csv"

#: Below this, a quoted price is a single stick or an accessory, not a pack;
#: above it, a carton or a mispriced bundle. Both would skew the price index.
MIN_PACK_PRICE_NGN = 200.0
MAX_PACK_PRICE_NGN = 20000.0


def _plausible(price: float | None) -> bool:
    return (
        price is not None
        and pd.notna(price)
        and MIN_PACK_PRICE_NGN <= price <= MAX_PACK_PRICE_NGN
    )


def _matches_brand(title: str, brand: str) -> bool:
    """Whether a product description actually names the brand it is filed under.

    Retained from the scraper, where it guarded against search engines returning
    loosely related products ('Time' matches watches). Against a curated file it
    is a consistency check: a row whose description does not name its own brand
    has been mis-entered.
    """
    return re.search(rf"\b{re.escape(brand)}\b", title or "", re.IGNORECASE) is not None


def fetch() -> pd.DataFrame:
    """Reference competitor prices, validated and keyed for upsert."""
    if not SEED_PATH.exists():
        log.warning(
            "No competitor reference file at %s; the optimizer will report its "
            "price ceiling as INACTIVE.", SEED_PATH,
        )
        return pd.DataFrame(columns=COLUMNS)

    frame = pd.read_csv(SEED_PATH)

    missing = [c for c in ("date", "brand", "price", "source") if c not in frame.columns]
    if missing:
        log.error(
            "%s is missing required column(s): %s. Expected header: %s",
            SEED_PATH.name, ", ".join(missing), ",".join(COLUMNS),
        )
        return pd.DataFrame(columns=COLUMNS)

    if frame.empty:
        log.info(
            "Competitor reference file is header-only. This is the expected "
            "state: marketplace scraping was removed under NTCA 2015 s.15(4) "
            "and no cited replacement prices have been recorded yet. The "
            "optimizer's +5%% competitor ceiling will report as INACTIVE."
        )
        return pd.DataFrame(columns=COLUMNS)

    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame["brand"] = frame["brand"].astype(str).str.strip()
    frame["source"] = frame["source"].astype(str).str.strip()
    for optional_col, default in (("region", NATIONAL), ("product_title", ""), ("url", "")):
        if optional_col not in frame.columns:
            frame[optional_col] = default
    frame["region"] = frame["region"].fillna(NATIONAL).astype(str).str.strip()
    frame["product_title"] = frame["product_title"].fillna("").astype(str).str.slice(0, 200)
    frame["url"] = frame["url"].fillna("").astype(str)

    # Every drop below is reported rather than silent: this file is hand-edited,
    # so a rejected row means someone mis-entered it and needs to know.
    checks = {
        "an unparseable date": frame["date"].isna(),
        "a brand not in config.COMPETITOR_BRANDS": ~frame["brand"].isin(
            config.COMPETITOR_BRANDS
        ),
        f"a price outside {MIN_PACK_PRICE_NGN:,.0f}-{MAX_PACK_PRICE_NGN:,.0f} NGN": (
            ~frame["price"].map(_plausible).astype(bool)
        ),
        "no source citation": frame["source"].isin(("", "nan")),
        "a description that does not name its own brand": ~frame.apply(
            # An empty description is allowed; a wrong one is not.
            lambda row: not row["product_title"]
            or _matches_brand(row["product_title"], row["brand"]),
            axis=1,
        ).astype(bool),
    }
    for reason, failed in checks.items():
        if failed.any():
            log.warning("Dropping %d competitor row(s) with %s", int(failed.sum()), reason)
    frame = frame[~pd.concat(list(checks.values()), axis=1).any(axis=1)]

    if frame.empty:
        log.warning(
            "Every competitor reference row failed validation; treating the "
            "price ceiling as unobserved rather than using bad rows."
        )
        return pd.DataFrame(columns=COLUMNS)

    # The dataset key is one price per brand/region/source/day. Several quotes
    # for the same key is normal (different pack sizes, different survey points),
    # so take the median -- robust to the one outlier that survives the bounds.
    frame = frame.groupby(["date", "brand", "region", "source"], as_index=False).agg(
        price=("price", "median"),
        product_title=("product_title", "first"),
        url=("url", "first"),
        quotes=("price", "size"),
    )
    log.info(
        "Competitor prices: %d brand/source pair(s) from %d cited quote(s), "
        "newest dated %s",
        len(frame), int(frame["quotes"].sum()), frame["date"].max().date(),
    )
    return frame.drop(columns=["quotes"])[COLUMNS]
