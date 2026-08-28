"""Throwaway: validates install, discovery, pythonpath and the autouse fixtures.

Deleted once the real suite is green.
"""

from __future__ import annotations

import socket

import pytest

from tobacco import config
from tobacco.store import parquet_io


def test_package_imports_under_pythonpath_src():
    assert config.SKUS == ("PREMIUM_20", "MIDRANGE_20", "VALUE_20")


def test_data_dir_is_redirected_into_tmp_path(tmp_path):
    assert config.DATA_DIR == tmp_path / "curated"
    assert parquet_io.dataset_dir("news_articles").parent == tmp_path / "curated"


def test_secrets_are_scrubbed():
    assert config.optional("GROQ_API_KEY") == ""


def test_network_is_blocked():
    with pytest.raises(AssertionError):
        socket.create_connection(("example.invalid", 80))
