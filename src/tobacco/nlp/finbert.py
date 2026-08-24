"""FinBERT scoring of financial headlines (INTRO.txt §3, layer 1).

Runs on the **GitHub Actions CPU runner**, not the HuggingFace Inference API.
The spec's §3 option C describes a free tier of "1000 requests/day" that no
longer exists -- HF's free inference allowance is now credit-based and far too
small for a daily batch. A CPU runner scoring a few dozen headlines takes well
under a minute, and Actions is unmetered on public repos.

The checkpoint is read from the ``FINBERT_MODEL`` repo variable, so the Phase 3
fine-tuned model can be swapped in without touching this file.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
import pandas as pd

from tobacco import config

log = logging.getLogger(__name__)

#: FinBERT truncates at 512 tokens, but headlines are short; a small cap keeps
#: the batch fast and avoids padding waste.
MAX_LENGTH = 128
BATCH_SIZE = 32


@lru_cache(maxsize=1)
def _pipeline():
    """Build the classification pipeline. Imports torch lazily.

    The import lives here rather than at module scope so that anything merely
    *importing* this module -- the dashboard, a test, ``jobs.scrape`` -- does not
    pay for torch. Only ``score.yml`` installs it.
    """
    import torch
    from transformers import pipeline

    model = config.finbert_model()
    log.info("Loading FinBERT checkpoint %s (CPU)", model)
    return pipeline(
        "text-classification",
        model=model,
        tokenizer=model,
        top_k=None,  # return all three class scores, not just the argmax
        device=-1,   # CPU: the runner has no GPU and requesting one errors out
        truncation=True,
        max_length=MAX_LENGTH,
        torch_dtype=torch.float32,
    )


def _crisis_probability(scores: list[dict]) -> float:
    """P(negative) for one headline -- the spec's ``fx_crisis_probability``.

    FinBERT emits positive/negative/neutral. Crisis risk is the negative mass;
    neutral is deliberately ignored rather than folded in, so a page of routine
    reporting reads as low risk instead of ambiguous.
    """
    for entry in scores:
        if str(entry["label"]).lower().startswith("neg"):
            return float(entry["score"])
    return 0.0


def score_headlines(headlines: list[str]) -> list[float]:
    """Negative-class probability in [0, 1] for each headline."""
    if not headlines:
        return []

    pipe = _pipeline()
    results: list[float] = []
    for start in range(0, len(headlines), BATCH_SIZE):
        batch = headlines[start : start + BATCH_SIZE]
        for scores in pipe(batch, batch_size=len(batch)):
            results.append(_crisis_probability(scores))

    log.info("FinBERT scored %d headline(s), mean crisis prob %.3f",
             len(results), float(np.mean(results)) if results else 0.0)
    return results


def crisis_score(scores) -> float:
    """Aggregate per-headline probabilities into one daily crisis score.

    Uses the **mean of the top quartile** rather than a plain mean. A crisis
    shows up as a handful of alarming headlines against a background of routine
    coverage; averaging everything dilutes exactly the signal we want, while a
    plain max would fire on one clickbait headline.
    """
    values = pd.Series(scores).dropna()
    if values.empty:
        return 0.0
    top = values[values >= values.quantile(0.75)]
    return float(np.clip(top.mean(), 0.0, 1.0))
