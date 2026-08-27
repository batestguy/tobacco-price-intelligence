"""Nigerian consumer price inflation (INTRO.txt §1).

The spec names NBS as the source, and NBS is still the *origin* of the figures --
but it has no stable machine-readable endpoint. Its release URLs change every
month and its sheet layouts drift, so a single hard-coded URL rots. One did:
``nigerianstat.gov.ng/resource/csv/cpi.csv`` returned HTTP 404 on the first live
run and the ``inflation`` feature came back empty.

So this module degrades in tiers, the same shape as ``cbn.py``, from best data to
most reliable data:

1. **``NBS_INFLATION_URL``** -- an explicit CSV/XLSX release, if the repo variable
   is set. Best of all when it points somewhere live; off by default, because a
   guessed default URL is what broke.
2. **CBN monthly** (``cbn.gov.ng/rates/inflrates.html``) -- CBN republishes the
   NBS CPI series monthly. Right cadence for §4, but an HTML scrape, i.e. the
   same fragility class as the thing that just broke.
3. **World Bank annual** -- ``FP.CPI.TOTL.ZG`` for Nigeria. A documented, stable,
   key-free JSON API. **It is annual and lagged**: as of Aug 2026 the newest
   observation is 2025. An annual step function is a much weaker demand driver
   than the monthly series §4 assumes, so this is the reliable *floor*, not the
   ideal.
4. **Committed seed** -- ``data/seed/inflation.csv``, backfill only.

The tiers are unioned rather than raced, because they cover different spans: the
weak-but-long annual series gives the forecaster history from its first run, and
any monthly rows that exist take precedence for the months they cover. Every row
carries the tier that produced it in ``source``, so the dashboard and the memo can
say where a figure came from instead of implying a precision it does not have.
"""

from __future__ import annotations

import io
import logging
import re
from typing import Any

import numpy as np
import pandas as pd

from tobacco import config
from tobacco.sources import _http

log = logging.getLogger(__name__)

#: CBN's monthly inflation table -- the NBS CPI series, republished.
CBN_URL = "https://www.cbn.gov.ng/rates/inflrates.html"

#: World Bank indicator FP.CPI.TOTL.ZG = "Inflation, consumer prices (annual %)".
WORLD_BANK_URL = (
    "https://api.worldbank.org/v2/country/NG/indicator/FP.CPI.TOTL.ZG"
    "?format=json&per_page=100&date={start}:{end}"
)

#: How far back to ask the World Bank for. Comfortably longer than any sales
#: history the forecaster has, and small enough to stay on one response page.
WORLD_BANK_START_YEAR = 2015

SEED_PATH = config.SEED_DIR / "inflation.csv"

COLUMNS = ["date", "rate", "food_rate", "core_rate", "source"]


def _pick(frame: pd.DataFrame, *needles: str) -> str | None:
    for column in frame.columns:
        flat = re.sub(r"[\s_%-]", "", str(column)).lower()
        if any(n in flat for n in needles):
            return column
    return None


def _parse(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    date_col = _pick(frame, "date", "period", "month")
    # "all items" is the NBS/CBN name for headline CPI.
    rate_col = _pick(frame, "allitems", "headline", "inflationrate", "rate")
    if date_col is None or rate_col is None:
        log.warning("%s payload has no recognisable date/rate columns: %s",
                    source, list(frame.columns)[:10])
        return pd.DataFrame(columns=COLUMNS)

    food_col = _pick(frame, "food")
    # CBN labels core inflation "All Items Less Farm Produce".
    core_col = _pick(frame, "core", "lessfarmproduce")

    parsed = pd.DataFrame(
        {
            "date": pd.to_datetime(frame[date_col], errors="coerce"),
            "rate": pd.to_numeric(frame[rate_col], errors="coerce"),
            # np.nan, not None: an all-None column is object dtype, which would
            # collide with a float column from another tier when the partitions
            # are concatenated on read.
            "food_rate": (
                pd.to_numeric(frame[food_col], errors="coerce")
                if food_col else np.nan
            ),
            "core_rate": (
                pd.to_numeric(frame[core_col], errors="coerce")
                if core_col else np.nan
            ),
            "source": source,
        }
    ).dropna(subset=["date", "rate"])

    # Normalise to the first of the month: the series is monthly, and anchoring
    # it makes the join against daily FX unambiguous.
    parsed["date"] = parsed["date"].dt.to_period("M").dt.to_timestamp()
    return parsed.drop_duplicates(subset=["date"], keep="last")[COLUMNS]


def _from_release_url() -> pd.DataFrame:
    """Tier 1: an explicit NBS release, only if ``NBS_INFLATION_URL`` is set.

    Deliberately has no default. The previous default 404'd, and a URL that is
    wrong is worse than one that is absent -- it fails on every run and looks
    like a network problem.
    """
    url = config.optional("NBS_INFLATION_URL")
    if not url:
        return pd.DataFrame(columns=COLUMNS)

    response = _http.get(url)
    if response is None:
        return pd.DataFrame(columns=COLUMNS)

    try:
        if url.lower().endswith((".xlsx", ".xls")):
            frame = pd.read_excel(io.BytesIO(response.content))
        else:
            frame = pd.read_csv(io.BytesIO(response.content))
    except Exception as exc:  # noqa: BLE001 - any parse failure is "no data"
        log.warning("Could not parse NBS release at %s: %s", url, exc)
        return pd.DataFrame(columns=COLUMNS)

    return _parse(frame, "nbs_release")


def _from_cbn() -> pd.DataFrame:
    """Tier 2: CBN's monthly inflation table -- right cadence, fragile markup."""
    response = _http.get(CBN_URL)
    if response is None:
        return pd.DataFrame(columns=COLUMNS)

    try:
        tables = pd.read_html(response.text)
    except ValueError:
        log.info("CBN inflation page contained no parseable table")
        return pd.DataFrame(columns=COLUMNS)

    # The page carries several tables and, when its rows are populated
    # client-side, the header-only one parses to zero rows. Take the first that
    # yields anything.
    for table in tables:
        parsed = _parse(table, "cbn_monthly")
        if not parsed.empty:
            log.info("CBN monthly: parsed %d month(s)", len(parsed))
            return parsed

    log.info("CBN inflation page had no populated rate rows")
    return pd.DataFrame(columns=COLUMNS)


def _from_worldbank() -> pd.DataFrame:
    """Tier 3: World Bank annual CPI. Stable and key-free, but annual and lagged.

    Values are anchored to January of their year. Downstream that is correct
    rather than convenient: ``features/build.py`` forward-fills inflation onto a
    daily index, so one January observation per year becomes an explicit annual
    step function instead of a fabricated monthly path.
    """
    url = WORLD_BANK_URL.format(
        start=WORLD_BANK_START_YEAR, end=config.today_wat().year
    )
    response = _http.get(url, headers={"Accept": "application/json"})
    if response is None:
        return pd.DataFrame(columns=COLUMNS)

    try:
        payload: Any = response.json()
    except ValueError:
        log.warning("World Bank returned non-JSON (%d bytes)", len(response.content))
        return pd.DataFrame(columns=COLUMNS)

    # Success is ``[metadata, [observations]]``; an error is a bare dict or a
    # one-element list, so check the shape rather than trusting the status code.
    if not (isinstance(payload, list) and len(payload) >= 2
            and isinstance(payload[1], list)):
        log.warning("World Bank response had an unexpected shape: %.200s", payload)
        return pd.DataFrame(columns=COLUMNS)

    rows = [
        {"date": f"{entry['date']}-01-01", "rate": entry["value"]}
        for entry in payload[1]
        if isinstance(entry, dict)
        and entry.get("value") is not None
        and entry.get("date")
    ]
    if not rows:
        log.warning("World Bank returned no non-null CPI observations for NG")
        return pd.DataFrame(columns=COLUMNS)

    parsed = _parse(pd.DataFrame(rows), "worldbank_annual")
    if not parsed.empty:
        log.info(
            "World Bank annual: %d year(s), newest %s = %.2f%% "
            "(annual and lagged -- see data/seed/README.md)",
            len(parsed),
            parsed["date"].max().date(),
            parsed.sort_values("date")["rate"].iloc[-1],
        )
    return parsed


def _from_seed() -> pd.DataFrame:
    """Tier 4: committed historical series, so day one has some history.

    Absent by default and deliberately so -- see ``data/seed/README.md``.
    """
    if not SEED_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    return _parse(pd.read_csv(SEED_PATH), "seed")


#: Weakest first: ``drop_duplicates(keep="last")`` then lets a better tier
#: override a weaker one for any month both cover.
TIERS = (_from_worldbank, _from_seed, _from_cbn, _from_release_url)


def fetch() -> pd.DataFrame:
    """Inflation from every tier that answered, best tier winning per month."""
    frames = []
    for tier in TIERS:
        try:
            frame = tier()
        except Exception as exc:  # noqa: BLE001 - one dead tier must not
            log.warning("%s failed: %s", tier.__name__, exc)   # sink the rest
            continue
        if not frame.empty:
            frames.append(frame)

    if not frames:
        log.warning(
            "No inflation data from any tier (release URL, CBN, World Bank, "
            "seed). The inflation feature will be null; XGBoost splits on "
            "missing values natively, so this degrades rather than breaks."
        )
        return pd.DataFrame(columns=COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["date"], keep="last")
    combined = combined.sort_values("date").reset_index(drop=True)[COLUMNS]

    provenance = ", ".join(
        f"{tier}={count}"
        for tier, count in combined["source"].value_counts().items()
    )
    log.info("Inflation: %d month(s) by tier -- %s", len(combined), provenance)
    return combined
