"""Score headlines with FinBERT and build daily aggregates (INTRO.txt §9 step 1).

Runs after ``scrape`` and is the only job that installs torch. It scores
**unscored headlines only**, so a re-run costs nothing and the model never
re-processes a backlog it has already seen.

VADER is not run here: social post text cannot be persisted (see
``sources.social``), so it is scored inline at collection time. This job consumes
the scores that scrape already stored.
"""

from __future__ import annotations

import sys

import pandas as pd

from tobacco import config
from tobacco.nlp import finbert, vader
from tobacco.store import parquet_io, supabase_io

log = config.setup_logging("score")

#: Cap per run so an unexpected backlog cannot run the job past the Actions
#: 6-hour job limit. The next run picks up whatever is left.
MAX_PER_RUN = 500


def score_news() -> int:
    """Score any headline with a null ``finbert_score``. Returns rows scored."""
    articles = parquet_io.read("news_articles")
    if articles.empty:
        log.info("No articles to score")
        return 0

    unscored = articles[articles["finbert_score"].isna()]
    if unscored.empty:
        log.info("All %d article(s) already scored", len(articles))
        return 0

    batch = unscored.sort_values("published_at", ascending=False).head(MAX_PER_RUN).copy()
    log.info("Scoring %d of %d unscored headline(s)", len(batch), len(unscored))

    batch["finbert_score"] = finbert.score_headlines(batch["headline"].tolist())
    batch["scored_at"] = pd.Timestamp(config.now_wat().replace(tzinfo=None))

    parquet_io.upsert("news_articles", batch)
    supabase_io.mirror("news_articles", batch, ("id",))
    return len(batch)


def build_aggregates() -> pd.DataFrame:
    """Daily ``sentiment_aggregates`` (INTRO.txt §2).

    Recomputed for a trailing window rather than only for today: the scoring job
    may have just filled in scores for older articles, which changes their day's
    aggregate. Upserting the window keeps history consistent.
    """
    window_start = pd.Timestamp(config.today_wat()) - pd.Timedelta(days=14)

    articles = parquet_io.read("news_articles", since=window_start)
    posts = parquet_io.read("social_posts", since=window_start)

    rows = []
    days = pd.date_range(window_start, pd.Timestamp(config.today_wat()), freq="D")

    for day in days:
        day_articles = pd.DataFrame()
        if not articles.empty:
            published = pd.to_datetime(articles["published_at"]).dt.normalize()
            day_articles = articles[published == day].dropna(subset=["finbert_score"])

        day_posts = pd.DataFrame()
        if not posts.empty:
            posted = pd.to_datetime(posts["published_at"]).dt.normalize()
            day_posts = posts[posted == day].dropna(subset=["vader_score"])

        if day_articles.empty and day_posts.empty:
            continue

        rows.append(
            {
                "date": day,
                "fx_crisis_prob": (
                    finbert.crisis_score(day_articles["finbert_score"])
                    if not day_articles.empty else None
                ),
                "consumer_sentiment": (
                    vader.to_acceptance(day_posts["vader_score"])
                    if not day_posts.empty else None
                ),
                "n_articles": int(len(day_articles)),
                "n_posts": int(len(day_posts)),
            }
        )

    aggregates = pd.DataFrame(rows)
    if aggregates.empty:
        log.warning("No scored content in the last 14 days; no aggregates written")
        return aggregates

    parquet_io.upsert("sentiment_aggregates", aggregates)
    supabase_io.mirror("sentiment_aggregates", aggregates, ("date",))
    log.info("Aggregates: %d day(s) updated", len(aggregates))
    return aggregates


def run() -> int:
    log.info("Scoring starting at %s WAT", config.now_wat().isoformat(timespec="seconds"))
    try:
        scored = score_news()
        aggregates = build_aggregates()
    except Exception as exc:  # noqa: BLE001
        log.exception("Scoring failed: %s", exc)
        supabase_io.log_event("score", "error", str(exc))
        return 1

    supabase_io.log_event(
        "score", "ok", f"{scored} headlines, {len(aggregates)} aggregate day(s)"
    )
    log.info("Scoring complete")
    return 0


if __name__ == "__main__":
    sys.exit(run())
