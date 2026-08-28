"""Weekly retraining (INTRO.txt §9 step 2).

Ensures the synthetic sales history exists and is current, rebuilds features, and
refits XGBoost. The model artifact and its metrics are committed, so the repo
records not just the forecast but the model that produced it.
"""

from __future__ import annotations

import sys
from datetime import timedelta

import pandas as pd

from tobacco import config
from tobacco.models import train_xgb
from tobacco.sources import sales_mock
from tobacco.store import parquet_io

log = config.setup_logging("train")

HISTORY_YEARS = 3


def ensure_sales_history() -> int:
    """Backfill on first run, then extend by whatever weeks have elapsed.

    This only ever *extends*: it generates from the week after the current head,
    so existing rows are never touched. That is the property worth having, and it
    is not the same as the old claim here that regenerating an overlapping range
    would be a no-op. It would not -- ``sales_mock``'s RNG is row-stable but its
    inputs (``fallback_fx``, ``config.PRICE_ELASTICITY``) are not; see that
    module's docstring.

    The consequence: changing a generator parameter leaves the old rows as they
    were and a structural break at the changeover week. Fixing that is a
    deliberate act, not a side effect -- see ``regenerate_sales_history``.
    """
    existing = parquet_io.read("sales_mock")
    last_complete_week = (
        pd.Timestamp(config.today_wat()) - pd.Timedelta(days=config.today_wat().weekday())
    ) - pd.Timedelta(days=7)

    if existing.empty:
        log.info("No sales history; generating %d years of synthetic data", HISTORY_YEARS)
        frame = sales_mock.backfill(years=HISTORY_YEARS)
    else:
        latest = pd.to_datetime(existing["week_start"]).max()
        if latest >= last_complete_week:
            log.info("Sales history current through %s", latest.date())
            return 0
        log.info("Extending sales history from %s to %s",
                 latest.date(), last_complete_week.date())
        frame = sales_mock.generate(
            (latest + timedelta(days=7)).date(), last_complete_week.date()
        )

    if frame.empty:
        return 0

    parquet_io.upsert("sales_mock", frame)
    return len(frame)


def regenerate_sales_history() -> int:
    """Rewrite the whole existing history in place, under the current parameters.

    Needed when a generator input changes (elasticity, FX handling): extending
    alone would splice new-parameter weeks onto old-parameter weeks and the model
    would fit the seam.

    Anchored to the **existing minimum** week, not to today. ``sales_mock.backfill``
    counts back three years from today, so it would start later than the current
    head and strand the earliest weeks unrewritten -- the same structural break,
    just moved. Every ``(week_start, sku, region)`` key in range is regenerated
    and replaced by the ordinary upsert (``keep="last"``); nothing is deleted, and
    ``parquet_io``'s upsert-only invariant is preserved.
    """
    existing = parquet_io.read("sales_mock")
    if existing.empty:
        log.info("No sales history to regenerate; the backfill below will create it.")
        return 0

    weeks = pd.to_datetime(existing["week_start"])
    start, end = weeks.min().date(), weeks.max().date()
    log.warning(
        "REGENERATING sales history %s..%s -- every row will be replaced", start, end
    )
    frame = sales_mock.generate(start, end)
    if frame.empty:
        return 0

    # Every partition should log `N in -> N rows (N replaced)`. A `(0 replaced)`
    # or a row count that grew means the regenerated range missed the stored one
    # and a mixed-parameter series has just been created.
    parquet_io.upsert("sales_mock", frame)
    return len(frame)


def _regeneration_requested() -> bool:
    """Read the ``workflow_dispatch`` input, defaulting to off.

    On a ``schedule`` trigger the input renders as an empty string, so a
    scheduled run can never reach the rewrite.
    """
    return config.optional("REGENERATE_SALES_HISTORY").lower() in {"1", "true", "yes", "on"}


def run() -> int:
    log.info("Training starting at %s WAT", config.now_wat().isoformat(timespec="seconds"))
    try:
        # Gated here rather than inside ensure_sales_history() so the extend path
        # stays a plain extend and cannot rewrite history as a side effect.
        if _regeneration_requested():
            log.info("Sales history: %d row(s) regenerated", regenerate_sales_history())
        added = ensure_sales_history()
        log.info("Sales history: %d row(s) added", added)
        metrics = train_xgb.train()
    except Exception as exc:  # noqa: BLE001
        log.exception("Training failed: %s", exc)
        return 1

    log.info(
        "Training complete: RMSE %.0f, MAPE %.2f%%",
        metrics["rmse"], metrics["mape_pct"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(run())
