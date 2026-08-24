"""Collect every source, commit to Parquet, mirror to Supabase (INTRO.txt §9 step 1).

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
from tobacco.store import parquet_io, supabase_io

log = config.setup_logging("scrape")


@dataclass(frozen=True)
class Source:
    name: str          # dataset name, per parquet_io.DATASETS
    fetch: Callable[[], pd.DataFrame]
    table: str         # Supabase table
    conflict: tuple[str, ...]  # upsert key


SOURCES = (
    Source("exchange_rates", cbn.fetch, "exchange_rates", ("date",)),
    Source("inflation", nbs.fetch, "inflation", ("date",)),
    Source("competitor_prices", competitors.fetch, "competitor_prices",
           ("date", "brand", "region", "source")),
    Source("news_articles", news.fetch, "news_articles", ("id",)),
    Source("social_posts", social.fetch, "social_posts", ("id",)),
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
            supabase_io.log_event(f"scrape.{source.name}", "error", str(exc))
            continue

        if frame.empty:
            log.info("[%s] no rows this run", source.name)
            succeeded.append(source.name)
            continue

        try:
            parquet_io.upsert(source.name, frame)
        except Exception as exc:  # noqa: BLE001
            # A Parquet failure IS fatal for that source -- it is the source of
            # truth, and mirroring to Supabase without it would invert the
            # authority relationship the whole design rests on.
            log.exception("[%s] Parquet write failed: %s", source.name, exc)
            failed.append(source.name)
            supabase_io.log_event(f"scrape.{source.name}", "error", str(exc))
            continue

        supabase_io.mirror(source.table, frame, source.conflict)
        succeeded.append(source.name)
        total_rows += len(frame)
        supabase_io.log_event(f"scrape.{source.name}", "ok", f"{len(frame)} rows")

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
