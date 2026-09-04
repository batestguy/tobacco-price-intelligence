"""Pure helpers around the two NLP layers, plus the lazy-torch contract.

The scoring models themselves are out of scope -- FinBERT needs torch and a 440 MB
checkpoint, VADER needs the NLTK lexicon download, and both would put a network
fetch inside a suite that is offline by construction. What *is* in scope is
everything around them, which is where the decisions live.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from tobacco import config
from tobacco.nlp import finbert, vader
from tobacco.sources.news import _article_id


# ---------------------------------------------------------------------------
# the lazy torch import
# ---------------------------------------------------------------------------


def test_finbert_module_imports_without_torch():
    """finbert.py:33-38 promises importing the module does not pay for torch,
    and names a test as a beneficiary -- but nothing enforced it.

    This is what lets torch stay out of ``requirements-actions.txt`` and be
    installed by ``score.yml`` alone, from the CPU wheel index. If the import
    migrates to module scope, three of the four workflows start failing at import
    or start pulling ~2 GB of CUDA they cannot use.
    """
    assert "torch" not in sys.modules


def test_the_torch_import_really_is_inside_the_pipeline_builder():
    """The other half of the same contract: torch must be reachable *somewhere*,
    or the laziness above would be vacuous. It is not installed here, so building
    the pipeline is exactly where the cost lands."""
    finbert._pipeline.cache_clear()
    with pytest.raises(ImportError, match="torch"):
        finbert._pipeline()


# ---------------------------------------------------------------------------
# finbert helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label", ["negative", "Negative", "NEGATIVE", "neg"])
def test_crisis_probability_reads_the_negative_class_whatever_its_casing(label):
    scores = [
        {"label": "positive", "score": 0.1},
        {"label": label, "score": 0.7},
        {"label": "neutral", "score": 0.2},
    ]
    assert finbert._crisis_probability(scores) == pytest.approx(0.7)


def test_crisis_probability_ignores_neutral_rather_than_folding_it_in():
    """A page of routine reporting should read as low risk, not as ambiguous."""
    scores = [{"label": "neutral", "score": 0.95}, {"label": "positive", "score": 0.05}]
    assert finbert._crisis_probability(scores) == 0.0


def test_crisis_score_uses_the_top_quartile_not_the_mean():
    """A crisis is a few alarming headlines against routine coverage; a plain mean
    dilutes exactly the signal, and a plain max fires on one clickbait headline."""
    scores = [0.1, 0.1, 0.1, 0.9]

    assert finbert.crisis_score(scores) == pytest.approx(0.9)
    assert finbert.crisis_score(scores) != pytest.approx(float(np.mean(scores)))


def test_crisis_score_is_not_a_max():
    """Five values put two of them at or above the 75th percentile, so a max
    (0.9) and a top-quartile mean (0.85) are distinguishable."""
    scores = [0.1, 0.2, 0.3, 0.8, 0.9]
    assert finbert.crisis_score(scores) == pytest.approx(0.85)
    assert finbert.crisis_score(scores) < max(scores)


def test_crisis_score_of_nothing_is_zero():
    assert finbert.crisis_score([]) == 0.0
    assert finbert.crisis_score([float("nan"), float("nan")]) == 0.0


def test_crisis_score_is_clipped_to_a_probability():
    assert finbert.crisis_score([1.4, 1.6]) == 1.0
    assert finbert.crisis_score([-0.3, -0.2]) == 0.0


# ---------------------------------------------------------------------------
# vader helpers
# ---------------------------------------------------------------------------


def test_to_acceptance_of_nothing_is_neutral_not_negative():
    """0.5, not 0.0. Returning 0.0 would trip the §7 sentiment alert every time
    Nairaland was unreachable -- a scraping failure reported as a consumer signal."""
    assert vader.to_acceptance([]) == 0.5
    assert vader.to_acceptance([float("nan")]) == 0.5
    assert vader.to_acceptance([]) > config.SENTIMENT_ALERT_THRESHOLD


@pytest.mark.parametrize("compound,expected", [(-1.0, 0.0), (0.0, 0.5), (1.0, 1.0), (0.5, 0.75)])
def test_to_acceptance_rescales_compound_scores_linearly(compound, expected):
    assert vader.to_acceptance([compound]) == pytest.approx(expected)


def test_to_acceptance_averages_before_rescaling():
    assert vader.to_acceptance([-1.0, 1.0]) == pytest.approx(0.5)


def test_to_acceptance_drops_missing_scores():
    assert vader.to_acceptance([1.0, float("nan")]) == pytest.approx(1.0)


def test_to_acceptance_is_clipped_to_the_unit_interval():
    assert vader.to_acceptance([-3.0]) == 0.0
    assert vader.to_acceptance([3.0]) == 1.0


# ---------------------------------------------------------------------------
# article ids -- what makes the news_articles upsert idempotent at all
# ---------------------------------------------------------------------------


def test_article_id_is_a_function_of_the_url():
    url = "https://punchng.com/naira-steady/"
    assert _article_id(url) == _article_id(url)
    assert _article_id(url) != _article_id(url + "?utm_source=rss")


def test_article_id_is_stable_across_processes():
    """The upsert key is this id, so a per-process value would silently turn every
    re-scrape into a duplicate insert. ``hash()`` is salted per process, which is
    why news.py hashes explicitly -- this is the test for that.
    """
    url = "https://punchng.com/naira-steady/"
    source = (
        f"import sys; sys.path.insert(0, {str(config.REPO_ROOT / 'src')!r});"
        "from tobacco.sources.news import _article_id;"
        f"print(_article_id({url!r}))"
    )

    digests = set()
    for seed in ("0", "1", "12345"):
        result = subprocess.run(
            [sys.executable, "-c", source],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        assert result.returncode == 0, result.stderr
        digests.add(result.stdout.strip())

    assert digests == {_article_id(url)}


def test_article_id_is_short_enough_to_read_in_a_diff():
    """The ids land in a committed Parquet file people review."""
    digest = _article_id("https://example.com/a")
    assert len(digest) == 16
    assert set(digest) <= set("0123456789abcdef")
