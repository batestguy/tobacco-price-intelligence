"""That the three autouse fixtures in conftest are actually in force.

The suite's two stated constraints -- no secret, no network -- are enforced by
fixtures, and a fixture that stops running fails open: every other test in the
suite would keep passing while quietly reaching the real ``data/curated/`` or the
real internet. These four assertions are the only thing that would notice.

(This file is what the plan's throwaway ``test_smoke.py`` became. The plumbing
checks it existed to make were worth keeping; the "does the package import"
half of it was not.)
"""

from __future__ import annotations

import socket

import pytest

from tobacco import config
from tobacco.store import parquet_io


def test_the_data_layer_is_redirected_away_from_the_real_repo(tmp_path):
    assert config.DATA_DIR == tmp_path / "curated"
    assert parquet_io.dataset_dir("news_articles") == tmp_path / "curated" / "news_articles"
    assert config.REPO_ROOT not in config.DATA_DIR.parents


@pytest.mark.parametrize("name", ["GROQ_API_KEY", "GMAIL_ADDRESS", "GMAIL_APP_PASSWORD"])
def test_credentials_are_scrubbed_from_the_environment(name):
    assert config.optional(name) == ""


def test_outbound_connections_are_blocked():
    with pytest.raises(AssertionError, match="offline by construction"):
        socket.create_connection(("example.invalid", 80))


def test_socket_connect_is_blocked_too():
    with pytest.raises(AssertionError, match="offline by construction"):
        socket.socket().connect(("127.0.0.1", 9))
