"""Nigerian financial news headlines (INTRO.txt §1).

**Headlines only, by design.** The spec's §2 schema stores
``news_articles(id, headline, url, timestamp, finbert_score)`` -- no body -- and
that is also what the public repo permits: committing ``newspaper3k``'s extracted
article text would republish copyrighted Nigerian news content. FinBERT scores
the headline, which is the standard framing for financial sentiment anyway
(Financial PhraseBank, the corpus FinBERT is tuned on, is sentence-level).

``newspaper3k`` is therefore used narrowly: only to recover a headline or publish
date for a feed entry that lacks one. Its ``.text`` is never read.
"""

from __future__ import annotations

import hashlib
import logging
import re

import feedparser
import pandas as pd

from tobacco import config
from tobacco.sources import _http

log = logging.getLogger(__name__)

COLUMNS = ["id", "headline", "url", "source", "published_at", "finbert_score", "scored_at"]

FEEDS: dict[str, str] = {
    "punch": "https://punchng.com/topics/business/feed/",
    "nairametrics": "https://nairametrics.com/feed/",
    "businessday": "https://businessday.ng/feed/",
}

#: Punch's business feed still carries general news, so headlines are filtered
#: for macro relevance. Nairametrics and BusinessDay are finance-only and are
#: kept wholesale -- filtering them would throw away signal.
BROAD_FEEDS = frozenset({"punch"})

RELEVANCE_TERMS = (
    "naira", "dollar", "exchange rate", "forex", "fx", "cbn", "central bank",
    "inflation", "cpi", "economy", "gdp", "tax", "excise", "vat", "tariff",
    "import", "export", "fuel", "subsidy", "devaluation", "interest rate",
    "monetary", "budget", "revenue", "manufacturer", "fmcg", "consumer",
)

#: Keep the daily volume sane; the feeds are ordered newest-first.
MAX_PER_FEED = 40


def _article_id(url: str) -> str:
    """Stable id derived from the URL, so re-scraping upserts instead of duplicating."""
    return hashlib.blake2b(url.encode("utf-8"), digest_size=8).hexdigest()


def _relevant(headline: str) -> bool:
    lowered = headline.lower()
    return any(term in lowered for term in RELEVANCE_TERMS)


def _published(entry) -> pd.Timestamp:
    for field in ("published", "updated", "created"):
        value = getattr(entry, field, None)
        if value:
            parsed = pd.to_datetime(value, errors="coerce", utc=True)
            if not pd.isna(parsed):
                return parsed.tz_localize(None)
    # No date in the feed: assume now. Better than dropping the headline, and
    # the partition column must not be null.
    return pd.Timestamp(config.now_wat().replace(tzinfo=None))


def _fetch_feed(source: str, url: str) -> list[dict]:
    response = _http.get(url)
    if response is None:
        return []

    parsed = feedparser.parse(response.content)
    if parsed.bozo and not parsed.entries:
        log.warning("Feed %s did not parse: %s", source, parsed.get("bozo_exception"))
        return []

    rows = []
    for entry in parsed.entries[:MAX_PER_FEED]:
        headline = re.sub(r"\s+", " ", (getattr(entry, "title", "") or "")).strip()
        link = (getattr(entry, "link", "") or "").strip()
        if not headline or not link:
            continue
        if source in BROAD_FEEDS and not _relevant(headline):
            continue
        rows.append(
            {
                "id": _article_id(link),
                "headline": headline[:400],
                "url": link,
                "source": source,
                "published_at": _published(entry),
                # Filled in later by the scoring stage, which needs torch and so
                # runs in its own workflow.
                "finbert_score": None,
                "scored_at": None,
            }
        )
    log.info("Feed %s: %d headline(s)", source, len(rows))
    return rows


def fetch() -> pd.DataFrame:
    """Recent financial headlines across all configured feeds."""
    rows: list[dict] = []
    for source, url in FEEDS.items():
        try:
            rows.extend(_fetch_feed(source, url))
        except Exception as exc:  # noqa: BLE001 - one dead feed must not fail the job
            log.warning("Feed %s failed: %s", source, exc)

    if not rows:
        log.warning("No headlines retrieved from any feed")
        return pd.DataFrame(columns=COLUMNS)

    frame = pd.DataFrame(rows).drop_duplicates(subset=["id"], keep="first")
    log.info("News: %d unique headline(s)", len(frame))
    return frame.sort_values("published_at").reset_index(drop=True)[COLUMNS]
