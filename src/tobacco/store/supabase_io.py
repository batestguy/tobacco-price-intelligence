"""Supabase (PostgREST) serving layer.

Deliberately plain ``requests`` rather than ``supabase-py``: the only operations
needed are upsert and select, and the Streamlit container has 1 GB of RAM to work
with. Fewer transitive dependencies is worth more here than client sugar.

Supabase is a *cache of the repository*, never the origin. Jobs write Parquet
first and mirror here second, so a Supabase outage or a paused free-tier project
degrades the dashboard but cannot lose data.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import date, datetime
from typing import Any, Iterable, Sequence

import pandas as pd
import requests

from tobacco import config

log = logging.getLogger(__name__)

#: PostgREST rejects payloads that are too large; batch generously but bounded.
BATCH_SIZE = 500


def configured() -> bool:
    """Whether Supabase credentials are present.

    Jobs use this to keep the Parquet write mandatory and the mirror optional --
    the pipeline stays useful before Supabase is set up.
    """
    return bool(config.optional("SUPABASE_URL") and config.optional("SUPABASE_SERVICE_KEY"))


def _base_url() -> str:
    return config.require("SUPABASE_URL").rstrip("/") + "/rest/v1"


def _headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    key = config.require("SUPABASE_SERVICE_KEY")
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if extra:
        headers.update(extra)
    return headers


def _jsonable(value: Any) -> Any:
    """Convert a pandas cell into something ``json.dumps`` accepts.

    pandas hands back NaN/NaT for nulls and numpy scalars for numbers; PostgREST
    wants ``null`` and plain JSON types. NaN in particular serialises to the
    bare token ``NaN``, which is invalid JSON and produces a confusing 400.
    """
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if hasattr(value, "item"):  # numpy scalar
        try:
            return _jsonable(value.item())
        except (ValueError, AttributeError):
            return str(value)
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value
    return str(value)


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {col: _jsonable(row[col]) for col in df.columns}
        for _, row in df.iterrows()
    ]


def _chunks(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def upsert(table: str, df: pd.DataFrame, on_conflict: Sequence[str]) -> int:
    """Upsert a frame into ``table``, resolving conflicts on ``on_conflict``.

    Returns the number of rows sent. Raises on HTTP failure -- a silent mirror
    failure would let the dashboard drift from the repo without anyone noticing.
    """
    if df is None or df.empty:
        return 0

    url = f"{_base_url()}/{table}"
    params = {"on_conflict": ",".join(on_conflict)}
    headers = _headers(
        # merge-duplicates makes this a true upsert; without it PostgREST
        # returns 409 on an existing key and a re-run of a cron job fails.
        {"Prefer": "resolution=merge-duplicates,return=minimal"}
    )

    records = _records(df)
    sent = 0
    for batch in _chunks(records, BATCH_SIZE):
        response = requests.post(
            url,
            params=params,
            headers=headers,
            data=json.dumps(list(batch)),
            timeout=config.HTTP_TIMEOUT,
        )
        if not response.ok:
            # response.text can echo the payload but never the key -- safe to log.
            raise RuntimeError(
                f"Supabase upsert into {table!r} failed "
                f"({response.status_code}): {response.text[:500]}"
            )
        sent += len(batch)

    log.info("[supabase] upserted %d row(s) into %s", sent, table)
    return sent


def select(table: str, params: dict[str, str] | None = None) -> pd.DataFrame:
    """Read from ``table``. ``params`` are PostgREST filters, e.g.
    ``{"select": "*", "order": "date.desc", "limit": "30"}``.
    """
    response = requests.get(
        f"{_base_url()}/{table}",
        params=params or {"select": "*"},
        headers=_headers(),
        timeout=config.HTTP_TIMEOUT,
    )
    if not response.ok:
        raise RuntimeError(
            f"Supabase select from {table!r} failed "
            f"({response.status_code}): {response.text[:500]}"
        )
    return pd.DataFrame(response.json())


def mirror(table: str, df: pd.DataFrame, on_conflict: Sequence[str]) -> None:
    """Best-effort upsert: log and continue if Supabase is down or unset.

    Used by scrapers, where the Parquet write has already succeeded and is the
    record that matters. The next run re-sends the same rows, and because the
    write is an upsert, catching up costs nothing.
    """
    if not configured():
        log.info("[supabase] not configured; skipping mirror of %s", table)
        return
    try:
        upsert(table, df, on_conflict)
    except Exception as exc:  # noqa: BLE001 - deliberately broad; mirror is optional
        log.warning("[supabase] mirror of %s failed, continuing: %s", table, exc)


def log_event(action: str, status: str, detail: str = "") -> None:
    """Append to the ``logs`` table (INTRO.txt §2). Never raises."""
    if not configured():
        return
    try:
        requests.post(
            f"{_base_url()}/logs",
            headers=_headers({"Prefer": "return=minimal"}),
            data=json.dumps(
                [{
                    "timestamp": config.now_wat().isoformat(),
                    "action": action,
                    "status": status,
                    "detail": detail[:1000],
                }]
            ),
            timeout=config.HTTP_TIMEOUT,
        )
    except Exception as exc:  # noqa: BLE001
        log.debug("[supabase] log_event failed: %s", exc)
