"""Monthly inflation from the National Bureau of Statistics (INTRO.txt §1).

NBS publishes its CPI report as a file whose URL changes every month and whose
sheet layout drifts, so this parser matches columns by *meaning* rather than
position, and treats a failed fetch as "no new data this month" rather than an
error. Inflation is monthly: missing one twice-daily run costs nothing.

The release URL is read from the ``NBS_INFLATION_URL`` repo variable so it can be
repointed without a code change when NBS reorganises its site again.
"""

from __future__ import annotations

import io
import logging
import re

import pandas as pd

from tobacco import config
from tobacco.sources import _http

log = logging.getLogger(__name__)

DEFAULT_URL = "https://nigerianstat.gov.ng/resource/csv/cpi.csv"
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
    # "all items" is the NBS name for headline CPI.
    rate_col = _pick(frame, "allitems", "headline", "inflationrate", "rate")
    if date_col is None or rate_col is None:
        log.warning("NBS payload has no recognisable date/rate columns: %s",
                    list(frame.columns)[:10])
        return pd.DataFrame(columns=COLUMNS)

    food_col = _pick(frame, "food")
    core_col = _pick(frame, "core")

    parsed = pd.DataFrame(
        {
            "date": pd.to_datetime(frame[date_col], errors="coerce"),
            "rate": pd.to_numeric(frame[rate_col], errors="coerce"),
            "food_rate": (
                pd.to_numeric(frame[food_col], errors="coerce")
                if food_col else None
            ),
            "core_rate": (
                pd.to_numeric(frame[core_col], errors="coerce")
                if core_col else None
            ),
            "source": source,
        }
    ).dropna(subset=["date", "rate"])

    # Normalise to the first of the month: the series is monthly, and anchoring
    # it makes the join against daily FX unambiguous.
    parsed["date"] = parsed["date"].dt.to_period("M").dt.to_timestamp()
    return parsed.drop_duplicates(subset=["date"], keep="last")[COLUMNS]


def _from_remote() -> pd.DataFrame:
    url = config.optional("NBS_INFLATION_URL", DEFAULT_URL)
    response = _http.get(url)
    if response is None:
        return pd.DataFrame(columns=COLUMNS)

    content = response.content
    try:
        if url.lower().endswith((".xlsx", ".xls")):
            frame = pd.read_excel(io.BytesIO(content))
        else:
            frame = pd.read_csv(io.BytesIO(content))
    except Exception as exc:  # noqa: BLE001 - any parse failure is "no data"
        log.warning("Could not parse NBS release at %s: %s", url, exc)
        return pd.DataFrame(columns=COLUMNS)

    return _parse(frame, "nbs")


def _from_seed() -> pd.DataFrame:
    """Committed historical series, so the forecaster has inflation from day one.

    NBS does not offer a stable historical bulk endpoint, so the back series is
    checked in once and the scraper only ever appends new months to it.
    """
    if not SEED_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    return _parse(pd.read_csv(SEED_PATH), "seed")


def fetch() -> pd.DataFrame:
    """Latest inflation observations, remote if possible, else the seed series."""
    remote = _from_remote()
    seed = _from_seed()

    if remote.empty and seed.empty:
        log.warning("No inflation data available from NBS or seed file")
        return pd.DataFrame(columns=COLUMNS)

    # Remote wins on overlap: the seed is only a backfill.
    combined = pd.concat([seed, remote], ignore_index=True)
    combined = combined.drop_duplicates(subset=["date"], keep="last")
    log.info("Inflation: %d month(s) (%d remote, %d seed)",
             len(combined), len(remote), len(seed))
    return combined.sort_values("date").reset_index(drop=True)[COLUMNS]
