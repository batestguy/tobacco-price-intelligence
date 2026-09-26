"""The scoring job's two modes: incremental (the default) and rescore.

The default mode must never touch a row that already has a score -- that is what
makes a doubled cron run free. A rescore is the opposite contract: after the
``FINBERT_MODEL`` checkpoint changes, *every* headline and *every* aggregate day
must move to the new model, or the history silently mixes two models.

FinBERT itself is stubbed; what is under test is which rows reach it.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from tobacco import config
from tobacco.jobs import score
from tobacco.nlp import finbert
from tobacco.store import parquet_io

TODAY = date(2026, 9, 26)
OLD_SCORE = 0.1
NEW_SCORE = 0.9


def article(id_: str, published_at: str, finbert_score=None):
    return {
        "id": id_,
        "headline": f"Naira headline {id_}",
        "url": f"https://example.com/{id_}",
        "source": "punch",
        "published_at": published_at,
        "finbert_score": finbert_score,
        "scored_at": None,
    }


@pytest.fixture(autouse=True)
def stub_model(monkeypatch):
    """Record every headline sent to the model and score it NEW_SCORE."""
    seen: list[str] = []

    def fake_score(headlines):
        seen.extend(headlines)
        return [NEW_SCORE] * len(headlines)

    monkeypatch.setattr(finbert, "score_headlines", fake_score)
    monkeypatch.setattr(config, "today_wat", lambda: TODAY)
    return seen


@pytest.fixture
def corpus():
    """Two scored rows (one older than the 14-day window) and one unscored."""
    rows = [
        article("old", "2026-08-01", finbert_score=OLD_SCORE),
        article("recent", "2026-09-20", finbert_score=OLD_SCORE),
        article("new", "2026-09-25"),
    ]
    parquet_io.upsert("news_articles", pd.DataFrame(rows))


def scores_by_id() -> dict[str, float]:
    articles = parquet_io.read("news_articles")
    return dict(zip(articles["id"], articles["finbert_score"]))


# ---------------------------------------------------------------------------
# score_news
# ---------------------------------------------------------------------------


def test_default_mode_scores_only_unscored_rows(corpus, stub_model):
    assert score.score_news() == 1
    assert stub_model == ["Naira headline new"]
    assert scores_by_id() == {"old": OLD_SCORE, "recent": OLD_SCORE, "new": NEW_SCORE}


def test_default_mode_is_a_no_op_once_everything_is_scored(corpus, stub_model):
    score.score_news()
    stub_model.clear()
    assert score.score_news() == 0
    assert stub_model == []


def test_rescore_replaces_every_existing_score(corpus, stub_model):
    assert score.score_news(rescore=True) == 3
    assert sorted(stub_model) == sorted(f"Naira headline {i}" for i in ("old", "recent", "new"))
    assert set(scores_by_id().values()) == {NEW_SCORE}


def test_rescore_ignores_the_per_run_cap(corpus, stub_model, monkeypatch):
    """A capped rescore would redo the newest MAX_PER_RUN rows on every dispatch
    and never reach the rest -- the oldest row here would keep the old model."""
    monkeypatch.setattr(score, "MAX_PER_RUN", 1)
    assert score.score_news(rescore=True) == 3
    assert scores_by_id()["old"] == NEW_SCORE


def test_default_mode_still_honours_the_cap(stub_model, monkeypatch):
    monkeypatch.setattr(score, "MAX_PER_RUN", 1)
    parquet_io.upsert(
        "news_articles",
        pd.DataFrame([article("a", "2026-09-24"), article("b", "2026-09-25")]),
    )
    assert score.score_news() == 1
    assert stub_model == ["Naira headline b"]  # newest first


# ---------------------------------------------------------------------------
# build_aggregates
# ---------------------------------------------------------------------------


def aggregate_days(frame: pd.DataFrame) -> set[date]:
    return set(pd.to_datetime(frame["date"]).dt.date)


def test_default_aggregates_cover_only_the_trailing_window(corpus):
    days = aggregate_days(score.build_aggregates())
    assert date(2026, 9, 20) in days
    assert date(2026, 8, 1) not in days


def test_full_aggregates_reach_back_to_the_first_headline(corpus):
    days = aggregate_days(score.build_aggregates(full=True))
    assert {date(2026, 8, 1), date(2026, 9, 20)} <= days


# ---------------------------------------------------------------------------
# the RESCORE switch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["true", "True", " TRUE "])
def test_rescore_is_requested_by_a_true_input(monkeypatch, value):
    monkeypatch.setenv("RESCORE", value)
    assert score.rescore_requested()


@pytest.mark.parametrize("value", ["", "false", "0"])
def test_rescore_is_off_for_anything_else(monkeypatch, value):
    monkeypatch.setenv("RESCORE", value)
    assert not score.rescore_requested()


def test_rescore_is_off_when_unset():
    """workflow_run leaves ``inputs.rescore`` empty; a chained run must stay
    incremental."""
    assert not score.rescore_requested()
