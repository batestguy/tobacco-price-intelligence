"""Social sentiment from Nairaland (INTRO.txt §1).

Two deliberate departures from the spec:

**snscrape is not used.** The spec suggests it for Twitter/X "without an API key".
That stopped working when X closed off unauthenticated access; there is no free
Twitter source left, so Nairaland -- a large, public, unauthenticated Nigerian
forum -- carries the social layer instead.

**VADER is applied here, at collection time, not in the scoring workflow.** The
post text cannot be persisted: INTRO.txt §11 commits to storing no individual
consumer data, and this repo is public. VADER is a lexicon with no model weights
and no GPU need, so it costs nothing to run inline; the text is scored in memory
and discarded, and only ``id / url / timestamp / score`` is stored. FinBERT stays
in its own workflow because it needs torch, and it scores headlines, which *are*
persisted.
"""

from __future__ import annotations

import hashlib
import logging
import re
from urllib.parse import quote_plus, urljoin

import pandas as pd
from bs4 import BeautifulSoup

from tobacco import config
from tobacco.nlp import vader
from tobacco.sources import _http

log = logging.getLogger(__name__)

COLUMNS = ["id", "url", "topic", "published_at", "vader_score", "scored_at"]

BASE = "https://www.nairaland.com"

#: Search terms: category-level, not brand-targeted. The point is consumer price
#: sentiment, and there is no legitimate reason to profile individual posters.
SEARCH_TERMS = (
    "cigarette price",
    "tobacco price",
    "cigarette increase",
)

MAX_POSTS_PER_TERM = 40
MIN_POST_CHARS = 20  # below this VADER has nothing to work with


def _post_id(url: str, text: str) -> str:
    seed = f"{url}|{text[:200]}".encode("utf-8")
    return hashlib.blake2b(seed, digest_size=8).hexdigest()


def _search(term: str) -> list[dict]:
    url = f"{BASE}/search?q={quote_plus(term)}&board=0&topicsonly=false"
    response = _http.get(url)
    if response is None:
        return []

    soup = BeautifulSoup(response.text, "lxml")
    rows: list[dict] = []
    now = pd.Timestamp(config.now_wat().replace(tzinfo=None))

    # Nairaland's markup is old-school table soup; post bodies sit in
    # `div.narrow`, with the topic link in the preceding header row.
    for block in soup.select("div.narrow")[:MAX_POSTS_PER_TERM]:
        text = block.get_text(" ", strip=True)
        if len(text) < MIN_POST_CHARS:
            continue

        link_el = block.find_previous("a", href=re.compile(r"^/\d+"))
        href = link_el["href"] if link_el else ""
        post_url = urljoin(BASE, href) if href else url
        topic = link_el.get_text(strip=True)[:200] if link_el else term

        rows.append(
            {
                "id": _post_id(post_url, text),
                "url": post_url,
                "topic": topic,
                "published_at": now,
                # Scored now; `text` is never returned from this function.
                "vader_score": vader.compound(text),
                "scored_at": now,
            }
        )
    return rows


def fetch() -> pd.DataFrame:
    """Recent public forum posts about cigarette pricing, already VADER-scored.

    Returns metadata and scores only -- no post text, no usernames.
    """
    rows: list[dict] = []
    for term in SEARCH_TERMS:
        try:
            rows.extend(_search(term))
        except Exception as exc:  # noqa: BLE001
            log.warning("Nairaland search for %r failed: %s", term, exc)

    if not rows:
        log.warning("No social posts retrieved; consumer sentiment will fall back "
                    "to its last known value")
        return pd.DataFrame(columns=COLUMNS)

    frame = pd.DataFrame(rows).drop_duplicates(subset=["id"], keep="first")
    log.info("Social: %d unique post(s), mean compound %.3f",
             len(frame), frame["vader_score"].mean())
    return frame.reset_index(drop=True)[COLUMNS]
