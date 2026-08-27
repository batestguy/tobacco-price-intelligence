"""Collect every source and commit to Parquet (INTRO.txt §9 step 1).

Runs twice daily. One unreachable source must not cost us the others, so each is
collected independently and a failure is logged and stepped over. The job only
fails outright if *every* source failed -- which means the runner has no network,
not that a Nigerian website is having a bad morning.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from tobacco import config
from tobacco.sources import cbn, competitors, nbs, news, social
from tobacco.store import parquet_io

log = config.setup_logging("scrape")


@dataclass(frozen=True)
class Source:
    name: str          # dataset name, per parquet_io.DATASETS
    fetch: Callable[[], pd.DataFrame]


SOURCES = (
    Source("exchange_rates", cbn.fetch),
    Source("inflation", nbs.fetch),
    Source("competitor_prices", competitors.fetch),
    Source("news_articles", news.fetch),
    Source("social_posts", social.fetch),
)


def run() -> int:
    log.info("Scrape starting at %s WAT", config.now_wat().isoformat(timespec="seconds"))

    succeeded, failed, total_rows = [], [], 0

    for source in SOURCES:
        try:
            frame = source.fetch()
        except Exception as exc:  # noqa: BLE001 - isolate each source
            log.exception("[%s] fetch failed: %s", source.name, exc)
            failed.append(source.name)
            continue

        if frame.empty:
            log.info("[%s] no rows this run", source.name)
            succeeded.append(source.name)
            continue

        try:
            parquet_io.upsert(source.name, frame)
        except Exception as exc:  # noqa: BLE001
            # A Parquet failure IS fatal for that source -- it is the source of
            # truth, so there is nothing left to count as a success.
            log.exception("[%s] Parquet write failed: %s", source.name, exc)
            failed.append(source.name)
            continue

        succeeded.append(source.name)
        total_rows += len(frame)

    log.info(
        "Scrape complete: %d row(s) across %d/%d source(s). Failed: %s",
        total_rows, len(succeeded), len(SOURCES), ", ".join(failed) or "none",
    )

    if not succeeded:
        log.error("Every source failed -- treating the run as failed.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run())
