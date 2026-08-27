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

    ``sales_mock.generate`` is deterministic per row, so regenerating an
    overlapping range produces identical values and the upsert is a genuine
    no-op -- the history never silently rewrites itself.
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


def run() -> int:
    log.info("Training starting at %s WAT", config.now_wat().isoformat(timespec="seconds"))
    try:
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
