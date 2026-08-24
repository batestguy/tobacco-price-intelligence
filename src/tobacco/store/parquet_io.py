"""Month-partitioned Parquet under ``data/curated/`` -- the source of truth.

Two properties matter here and both are load-bearing:

**Idempotence.** Actions cron is best-effort: a job can be delayed, or run twice.
Every write is an upsert keyed on the dataset's natural key, so a double run
overwrites rather than duplicating. Nothing ever blind-appends.

**Month partitioning.** A single growing file would mean every daily commit
rewrites the whole binary blob, and git stores a full new copy each time. Writing
``news_articles/2026-08.parquet`` means a daily commit only rewrites the current
month's small file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from tobacco import config

log = logging.getLogger(__name__)

#: Columns that must never reach the repository, whatever a caller passes.
#:
#: The repo is public. Committing ``newspaper3k``'s article bodies would
#: republish copyrighted Nigerian news content, and committing forum post text
#: would store individual consumer data that INTRO.txt §11 says we do not keep.
#: Scoring uses these in memory; persistence drops them. This is a backstop --
#: callers are expected not to pass them in the first place.
NEVER_PERSIST = frozenset(
    {"body", "text", "content", "article_text", "full_text", "summary",
     "author", "username", "user", "raw_html"}
)


@dataclass(frozen=True)
class Dataset:
    """Declares how a dataset is keyed and partitioned."""

    keys: tuple[str, ...]
    ts: str  # column that determines the month partition


#: The dataset registry. Mirrors supabase/schema.sql -- change both together.
DATASETS: dict[str, Dataset] = {
    "exchange_rates": Dataset(keys=("date",), ts="date"),
    "inflation": Dataset(keys=("date",), ts="date"),
    "competitor_prices": Dataset(keys=("date", "brand", "region", "source"), ts="date"),
    "news_articles": Dataset(keys=("id",), ts="published_at"),
    "social_posts": Dataset(keys=("id",), ts="published_at"),
    "sentiment_aggregates": Dataset(keys=("date",), ts="date"),
    "sales_mock": Dataset(keys=("week_start", "sku", "region"), ts="week_start"),
    "recommendations": Dataset(keys=("date", "sku", "region"), ts="date"),
}


def _spec(name: str) -> Dataset:
    try:
        return DATASETS[name]
    except KeyError:
        raise KeyError(
            f"Unknown dataset {name!r}. Register it in parquet_io.DATASETS and "
            f"add the matching table to supabase/schema.sql."
        ) from None


def dataset_dir(name: str) -> Path:
    return config.DATA_DIR / name


def partition_path(name: str, period: str) -> Path:
    """``data/curated/<name>/<YYYY-MM>.parquet``."""
    return dataset_dir(name) / f"{period}.parquet"


def _strip_transient(df: pd.DataFrame) -> pd.DataFrame:
    banned = [c for c in df.columns if c.lower() in NEVER_PERSIST]
    if banned:
        log.warning(
            "Dropping non-persistable column(s) before write: %s "
            "(see CLAUDE.md 'What must never be committed')",
            ", ".join(banned),
        )
        df = df.drop(columns=banned)
    return df


def _normalise_ts(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Coerce the partition column to datetime and drop unusable rows."""
    df = df.copy()
    df[column] = pd.to_datetime(df[column], errors="coerce", utc=True).dt.tz_localize(None)
    bad = int(df[column].isna().sum())
    if bad:
        log.warning("Dropping %d row(s) with an unparseable %s", bad, column)
        df = df[df[column].notna()]
    return df


def upsert(name: str, df: pd.DataFrame) -> list[Path]:
    """Merge ``df`` into the dataset, returning the partition files written.

    Rows already present (same natural key) are *replaced* by the incoming
    version, so re-running a job is safe.
    """
    spec = _spec(name)
    if df is None or df.empty:
        log.info("[%s] nothing to write", name)
        return []

    df = _strip_transient(df)

    missing = [k for k in (*spec.keys, spec.ts) if k not in df.columns]
    if missing:
        raise ValueError(
            f"[{name}] incoming frame is missing required column(s): {missing}. "
            f"Expected key {spec.keys} and timestamp {spec.ts!r}."
        )

    df = _normalise_ts(df, spec.ts)
    if df.empty:
        return []

    # A single frame can straddle a month boundary (e.g. an evening scrape that
    # picks up articles published just after midnight), so partition explicitly
    # rather than assuming one file.
    written: list[Path] = []
    for period, chunk in df.groupby(df[spec.ts].dt.strftime("%Y-%m"), sort=True):
        path = partition_path(name, str(period))
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, chunk], ignore_index=True)
        else:
            existing = None
            combined = chunk.reset_index(drop=True)

        before = len(combined)
        # keep="last" => the incoming row wins, because it was concatenated after.
        combined = combined.drop_duplicates(subset=list(spec.keys), keep="last")
        combined = combined.sort_values(list(spec.keys)).reset_index(drop=True)

        combined.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
        written.append(path)
        log.info(
            "[%s] %s: %d in -> %d rows (%d replaced)",
            name, path.name, len(chunk), len(combined), before - len(combined),
        )

    return written


def read(name: str, since: date | str | None = None) -> pd.DataFrame:
    """Read a dataset, optionally only rows at/after ``since``.

    Returns an empty frame (not an error) when the dataset does not exist yet --
    the pipeline has to survive its own first run.
    """
    spec = _spec(name)
    directory = dataset_dir(name)
    if not directory.exists():
        log.info("[%s] no data yet", name)
        return pd.DataFrame()

    files = sorted(directory.glob("*.parquet"))
    if since is not None:
        # Partition names are YYYY-MM, so a lexical compare on the month prefix
        # skips whole files without opening them.
        cutoff_month = pd.Timestamp(since).strftime("%Y-%m")
        files = [f for f in files if f.stem >= cutoff_month]
    if not files:
        return pd.DataFrame()

    frame = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if since is not None and spec.ts in frame.columns:
        frame = frame[pd.to_datetime(frame[spec.ts]) >= pd.Timestamp(since)]

    return frame.sort_values(list(spec.keys)).reset_index(drop=True)


def latest(name: str, n: int = 1) -> pd.DataFrame:
    """The ``n`` most recent rows by the dataset's timestamp column."""
    spec = _spec(name)
    frame = read(name)
    if frame.empty:
        return frame
    return frame.sort_values(spec.ts).tail(n).reset_index(drop=True)
