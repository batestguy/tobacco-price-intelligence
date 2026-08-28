"""The data layer: the copyright guard, upsert idempotence, and partitioning.

Two of these are load-bearing beyond ordinary correctness:

* ``NEVER_PERSIST`` is what keeps copyrighted Nigerian article bodies out of a
  public repo. The module calls itself "a backstop"; a backstop nothing asserts
  is a comment.
* ``upsert`` idempotence is the only defence against Actions cron running a job
  twice, which it is explicitly allowed to do.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tobacco.store import parquet_io


def article(id_: str, published_at: str, headline: str = "Naira steady", **extra):
    row = {
        "id": id_,
        "headline": headline,
        "url": f"https://example.com/{id_}",
        "source": "punch",
        "published_at": published_at,
        "finbert_score": None,
        "scored_at": None,
    }
    row.update(extra)
    return row


def read_partition(period: str, dataset: str = "news_articles") -> pd.DataFrame:
    return pd.read_parquet(parquet_io.partition_path(dataset, period))


# ---------------------------------------------------------------------------
# NEVER_PERSIST -- the copyright / no-consumer-data guard
# ---------------------------------------------------------------------------


def test_feedparser_body_fields_are_named_in_the_guard():
    """`summary` and `content` are the live risk: they are what a feedparser
    entry carries, and a frame built from an entry wholesale would sweep them in."""
    assert {"summary", "content"} <= parquet_io.NEVER_PERSIST


def test_banned_columns_are_dropped_before_write():
    frame = pd.DataFrame(
        [article("a1", "2026-08-10", body="full article text", summary="a paragraph")]
    )
    parquet_io.upsert("news_articles", frame)

    written = read_partition("2026-08")
    assert "body" not in written.columns
    assert "summary" not in written.columns
    assert written.loc[0, "headline"] == "Naira steady"


@pytest.mark.parametrize("column", ["Summary", "CONTENT", "Article_Text", "RAW_HTML"])
def test_banned_columns_are_dropped_case_insensitively(column):
    """A caller renaming `body` to `Body` must not defeat the guard."""
    frame = pd.DataFrame([article("a1", "2026-08-10", **{column: "text"})])
    parquet_io.upsert("news_articles", frame)
    assert column not in read_partition("2026-08").columns


def test_the_guard_does_not_drop_a_legitimate_column():
    frame = pd.DataFrame([article("a1", "2026-08-10")])
    parquet_io.upsert("news_articles", frame)
    assert {"id", "headline", "url", "source"} <= set(read_partition("2026-08").columns)


# ---------------------------------------------------------------------------
# upsert
# ---------------------------------------------------------------------------


def test_upsert_replaces_a_row_with_the_same_key():
    parquet_io.upsert("news_articles", pd.DataFrame([article("a1", "2026-08-10", headline="Old")]))
    parquet_io.upsert("news_articles", pd.DataFrame([article("a1", "2026-08-10", headline="New")]))

    written = read_partition("2026-08")
    assert len(written) == 1
    assert written.loc[0, "headline"] == "New", "the incoming row wins, not the stored one"


def test_upsert_is_idempotent_when_a_job_is_run_twice():
    """Cron is best-effort and can double-fire. A rerun must not duplicate rows."""
    frame = pd.DataFrame([article("a1", "2026-08-10"), article("a2", "2026-08-11")])

    parquet_io.upsert("news_articles", frame)
    first = parquet_io.read("news_articles")
    parquet_io.upsert("news_articles", frame)
    second = parquet_io.read("news_articles")

    assert len(second) == 2
    pd.testing.assert_frame_equal(first, second)


def test_upsert_adds_new_keys_alongside_existing_ones():
    parquet_io.upsert("news_articles", pd.DataFrame([article("a1", "2026-08-10")]))
    parquet_io.upsert("news_articles", pd.DataFrame([article("a2", "2026-08-11")]))
    assert sorted(parquet_io.read("news_articles")["id"]) == ["a1", "a2"]


def test_upsert_deduplicates_within_a_single_incoming_frame():
    frame = pd.DataFrame(
        [article("a1", "2026-08-10", headline="First"), article("a1", "2026-08-10", headline="Last")]
    )
    parquet_io.upsert("news_articles", frame)

    written = read_partition("2026-08")
    assert len(written) == 1
    assert written.loc[0, "headline"] == "Last"


def test_upsert_uses_the_full_composite_key():
    """competitor_prices is keyed on four columns; changing any one is a new row."""
    base = {"date": "2026-08-10", "brand": "Bohem", "region": "Lagos", "source": "survey", "price": 1200.0}
    parquet_io.upsert("competitor_prices", pd.DataFrame([base]))
    parquet_io.upsert("competitor_prices", pd.DataFrame([{**base, "region": "Kano"}]))

    assert len(parquet_io.read("competitor_prices")) == 2


def test_upsert_writes_nothing_for_an_empty_frame():
    assert parquet_io.upsert("news_articles", pd.DataFrame()) == []
    assert not parquet_io.dataset_dir("news_articles").exists()


def test_upsert_writes_nothing_for_none():
    assert parquet_io.upsert("news_articles", None) == []


# ---------------------------------------------------------------------------
# partitioning
# ---------------------------------------------------------------------------


def test_a_month_straddling_frame_writes_two_partitions():
    """An evening scrape can pick up articles published just after midnight."""
    frame = pd.DataFrame([article("a1", "2026-08-31"), article("a2", "2026-09-01")])
    written = parquet_io.upsert("news_articles", frame)

    assert sorted(path.name for path in written) == ["2026-08.parquet", "2026-09.parquet"]
    assert read_partition("2026-08")["id"].tolist() == ["a1"]
    assert read_partition("2026-09")["id"].tolist() == ["a2"]


def test_rows_with_an_unparseable_timestamp_are_dropped_not_written_to_a_null_partition():
    frame = pd.DataFrame([article("a1", "2026-08-10"), article("a2", "not a date")])
    parquet_io.upsert("news_articles", frame)

    assert parquet_io.read("news_articles")["id"].tolist() == ["a1"]


def test_a_frame_of_only_unparseable_timestamps_writes_nothing():
    frame = pd.DataFrame([article("a1", "not a date")])
    assert parquet_io.upsert("news_articles", frame) == []


def test_timezone_aware_timestamps_are_normalised_to_naive():
    parquet_io.upsert("news_articles", pd.DataFrame([article("a1", "2026-08-10T23:00:00+01:00")]))
    stored = read_partition("2026-08").loc[0, "published_at"]
    assert stored.tzinfo is None


# ---------------------------------------------------------------------------
# schema errors
# ---------------------------------------------------------------------------


def test_upsert_raises_on_a_missing_key_column():
    frame = pd.DataFrame([article("a1", "2026-08-10")]).drop(columns=["id"])
    with pytest.raises(ValueError, match="id"):
        parquet_io.upsert("news_articles", frame)


def test_upsert_raises_on_a_missing_timestamp_column():
    frame = pd.DataFrame([article("a1", "2026-08-10")]).drop(columns=["published_at"])
    with pytest.raises(ValueError, match="published_at"):
        parquet_io.upsert("news_articles", frame)


def test_an_unregistered_dataset_is_a_keyerror_naming_the_registry():
    with pytest.raises(KeyError, match="parquet_io.DATASETS"):
        parquet_io.upsert("invented_dataset", pd.DataFrame([{"a": 1}]))


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------


def test_read_returns_an_empty_frame_for_a_dataset_that_does_not_exist_yet():
    """The pipeline has to survive its own first run."""
    frame = parquet_io.read("news_articles")
    assert frame.empty
    assert isinstance(frame, pd.DataFrame)


def test_read_since_skips_earlier_partitions():
    parquet_io.upsert(
        "news_articles",
        pd.DataFrame(
            [article("a1", "2026-07-15"), article("a2", "2026-08-31"), article("a3", "2026-09-02")]
        ),
    )
    assert parquet_io.read("news_articles", since="2026-09-01")["id"].tolist() == ["a3"]


def test_read_since_filters_within_the_boundary_partition_too():
    """The lexical file skip is coarse; rows before `since` in a kept file must go."""
    parquet_io.upsert(
        "news_articles",
        pd.DataFrame([article("a1", "2026-09-01"), article("a2", "2026-09-20")]),
    )
    assert parquet_io.read("news_articles", since="2026-09-15")["id"].tolist() == ["a2"]


def test_latest_returns_the_most_recent_rows_by_timestamp():
    parquet_io.upsert(
        "news_articles",
        pd.DataFrame(
            [article("a1", "2026-08-01"), article("a2", "2026-09-05"), article("a3", "2026-08-20")]
        ),
    )
    assert parquet_io.latest("news_articles", n=2)["id"].tolist() == ["a3", "a2"]


def test_latest_on_an_empty_dataset_is_empty():
    assert parquet_io.latest("news_articles").empty


# ---------------------------------------------------------------------------
# known gap
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "upsert deduplicates within a partition only. If a key's timestamp crosses "
        "a month boundary the stale copy survives in the old file, and read() "
        "concatenates the partitions with no cross-file dedup. Only news_articles "
        "and social_posts are exposed (key `id`, partition `published_at`); the "
        "trigger is a feed correcting a publish date across a month end. Stated "
        "here so the invariant is on record -- this turns red when it is fixed."
    ),
)
def test_upsert_is_idempotent_across_a_partition_boundary():
    parquet_io.upsert("news_articles", pd.DataFrame([article("a1", "2026-08-31", headline="Old")]))
    parquet_io.upsert("news_articles", pd.DataFrame([article("a1", "2026-09-01", headline="New")]))

    frame = parquet_io.read("news_articles")
    assert frame["id"].tolist() == ["a1"]
    assert frame.loc[0, "headline"] == "New"
