"""VADER lexicon sentiment (INTRO.txt §3, layer 2).

Cheap enough to run inline with collection, which is what lets ``sources.social``
score forum posts in memory and persist only the score -- never the text.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _analyzer():
    """Load VADER, downloading the lexicon on first use.

    ``nltk.download`` is a no-op once the file is present, and the Actions cache
    keeps it between runs, so this costs a few hundred KB exactly once.
    """
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        log.info("Downloading VADER lexicon")
        nltk.download("vader_lexicon", quiet=True)
        return SentimentIntensityAnalyzer()


def compound(text: str) -> float:
    """Compound polarity in [-1, +1]."""
    if not text or not text.strip():
        return 0.0
    return float(_analyzer().polarity_scores(text)["compound"])


def score_many(texts: list[str]) -> list[float]:
    return [compound(t) for t in texts]


def to_acceptance(compound_scores) -> float:
    """Map compound scores to the 0-1 ``price_acceptance_score`` the spec asks for.

    A linear rescale of the mean: -1 becomes 0, 0 becomes 0.5, +1 becomes 1.
    Returns the neutral 0.5 for an empty input, so a day with no scraped posts
    reads as "no signal" rather than "maximally negative" -- the latter would
    trip the sentiment alert every time Nairaland was unreachable.
    """
    values = pd.Series(compound_scores).dropna()
    if values.empty:
        return 0.5
    return float(np.clip((values.mean() + 1.0) / 2.0, 0.0, 1.0))
