"""Data acquisition (INTRO.txt §1).

Every scraper here follows the same contract:

* return a tidy ``pandas.DataFrame`` matching the dataset's registered schema;
* return an **empty frame rather than raising** when a remote source is merely
  unavailable, so one flaky Nigerian website cannot fail the whole nightly job;
* never return article bodies, forum post text, usernames, or anything else
  listed in ``store.parquet_io.NEVER_PERSIST``.
"""
