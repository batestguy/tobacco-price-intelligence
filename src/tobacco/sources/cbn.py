"""NGN/USD exchange rates from the Central Bank of Nigeria (INTRO.txt §1, option E).

The documented endpoint is ``cbn.gov.ng/api/GetAllExchangeRatesGRAPH``. It is
known to be flaky -- it times out, occasionally serves HTML from an error page
with a 200 status, and its JSON key names are not stable across responses. So
this module degrades in three documented steps rather than failing the job:

1. the JSON API, parsed tolerantly (keys matched by substring, not exactly);
2. the public rates HTML table, if the API gives nothing usable;
3. carry the last observation forward, flagged ``is_carried_forward``.

Step 3 matters because the forecaster needs an unbroken daily FX series; a gap
would silently corrupt every lag feature built on top of it. A carried-forward
row is honest about being one, and the dashboard renders it differently.
"""

from __future__ import annotations

import logging
import re
from datetime import timedelta
from typing import Any

import pandas as pd

from tobacco import config
from tobacco.sources import _http
from tobacco.store import parquet_io

log = logging.getLogger(__name__)

API_URL = "https://www.cbn.gov.ng/api/GetAllExchangeRatesGRAPH"
HTML_URL = "https://www.cbn.gov.ng/rates/exchratebycurrency.asp"

COLUMNS = [
    "date", "usd_ngn_rate", "buying_rate", "central_rate",
    "selling_rate", "source", "is_carried_forward",
]

#: How stale a carried-forward rate may get before we stop pretending.
MAX_CARRY_FORWARD_DAYS = 7


def _find_key(record: dict[str, Any], *needles: str) -> Any:
    """First value whose key contains any needle (case/space/underscore-insensitive)."""
    for key, value in record.items():
        flat = re.sub(r"[\s_-]", "", str(key)).lower()
        if any(n in flat for n in needles):
            return value
    return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        # Rates arrive as "1,535.72" or " 1535.72 " depending on the endpoint.
        return float(re.sub(r"[^\d.\-]", "", str(value)))
    except (TypeError, ValueError):
        return None


def _iter_records(payload: Any):
    """Yield dict records from whatever shape the API returned this time."""
    if isinstance(payload, dict):
        for key in ("data", "Data", "result", "Result", "rates", "items"):
            if key in payload:
                yield from _iter_records(payload[key])
                return
        yield payload
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item


def _from_api() -> pd.DataFrame:
    response = _http.get(API_URL, headers={"Accept": "application/json"})
    if response is None:
        return pd.DataFrame()
    try:
        payload = response.json()
    except ValueError:
        # A CBN error page served with status 200 -- common enough to expect.
        log.warning("CBN API returned non-JSON (%d bytes)", len(response.content))
        return pd.DataFrame()

    rows = []
    for record in _iter_records(payload):
        currency = str(_find_key(record, "currency", "ccy") or "USD").upper()
        if "USD" not in currency and "DOLLAR" not in currency:
            continue
        raw_date = _find_key(record, "ratedate", "date", "day")
        parsed = pd.to_datetime(raw_date, errors="coerce")
        if pd.isna(parsed):
            continue
        buying = _to_float(_find_key(record, "buying", "bid"))
        central = _to_float(_find_key(record, "central", "mid", "rate"))
        selling = _to_float(_find_key(record, "selling", "ask"))
        headline = central or buying or selling
        if headline is None or headline <= 0:
            continue
        rows.append(
            {
                "date": parsed.normalize(),
                "usd_ngn_rate": headline,
                "buying_rate": buying,
                "central_rate": central,
                "selling_rate": selling,
                "source": "cbn_api",
                "is_carried_forward": False,
            }
        )

    frame = pd.DataFrame(rows, columns=COLUMNS)
    if not frame.empty:
        log.info("CBN API: parsed %d USD row(s)", len(frame))
    return frame


def _from_html() -> pd.DataFrame:
    """Fallback: scrape the public rates table."""
    response = _http.get(HTML_URL)
    if response is None:
        return pd.DataFrame()
    try:
        tables = pd.read_html(response.text)
    except ValueError:
        log.warning("CBN HTML page contained no parseable table")
        return pd.DataFrame()

    for table in tables:
        flat = {re.sub(r"[\s_-]", "", str(c)).lower(): c for c in table.columns}
        date_col = next((v for k, v in flat.items() if "date" in k), None)
        rate_col = next(
            (v for k, v in flat.items() if "central" in k or "rate" in k), None
        )
        if date_col is None or rate_col is None:
            continue
        parsed = pd.to_datetime(table[date_col], errors="coerce")
        rates = table[rate_col].map(_to_float)
        frame = pd.DataFrame(
            {
                "date": parsed,
                "usd_ngn_rate": rates,
                "buying_rate": None,
                "central_rate": rates,
                "selling_rate": None,
                "source": "cbn_html",
                "is_carried_forward": False,
            }
        ).dropna(subset=["date", "usd_ngn_rate"])
        if not frame.empty:
            log.info("CBN HTML fallback: parsed %d row(s)", len(frame))
            return frame[COLUMNS]

    return pd.DataFrame()


def _carry_forward() -> pd.DataFrame:
    """Last resort: repeat the most recent observation for today."""
    history = parquet_io.read("exchange_rates")
    if history.empty:
        log.error("CBN unreachable and no history to carry forward from")
        return pd.DataFrame()

    last = history.sort_values("date").iloc[-1]
    today = pd.Timestamp(config.today_wat())
    gap = (today - pd.Timestamp(last["date"])).days

    if gap <= 0:
        return pd.DataFrame()  # today is already covered
    if gap > MAX_CARRY_FORWARD_DAYS:
        log.error(
            "CBN unreachable and last observation is %d days old (> %d); "
            "refusing to fabricate a rate",
            gap, MAX_CARRY_FORWARD_DAYS,
        )
        return pd.DataFrame()

    log.warning("CBN unreachable; carrying %s forward %d day(s)", last["date"], gap)
    rows = [
        {
            "date": pd.Timestamp(last["date"]) + timedelta(days=offset),
            "usd_ngn_rate": float(last["usd_ngn_rate"]),
            "buying_rate": last.get("buying_rate"),
            "central_rate": last.get("central_rate"),
            "selling_rate": last.get("selling_rate"),
            "source": "carried_forward",
            "is_carried_forward": True,
        }
        for offset in range(1, gap + 1)
    ]
    return pd.DataFrame(rows, columns=COLUMNS)


def fetch() -> pd.DataFrame:
    """Latest CBN USD/NGN rates, via API -> HTML -> carry-forward."""
    for attempt in (_from_api, _from_html):
        frame = attempt()
        if not frame.empty:
            frame = frame.dropna(subset=["date", "usd_ngn_rate"])
            frame = frame.drop_duplicates(subset=["date"], keep="last")
            return frame.sort_values("date").reset_index(drop=True)[COLUMNS]

    return _carry_forward()
