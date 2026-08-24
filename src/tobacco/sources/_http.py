"""Shared HTTP session with retries.

The Nigerian sources this project reads are collectively unreliable: the CBN API
intermittently times out, NBS occasionally serves a 502, and the marketplaces
rate-limit. A bare ``requests.get`` would turn any of those into a failed
workflow run, so everything goes through a session with bounded backoff.
"""

from __future__ import annotations

import logging

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from tobacco import config

log = logging.getLogger(__name__)


def session(retries: int = 3, backoff: float = 1.5) -> requests.Session:
    sess = requests.Session()
    policy = Retry(
        total=retries,
        connect=retries,
        read=retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=policy)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    sess.headers.update(
        {
            "User-Agent": config.USER_AGENT,
            "Accept-Language": "en-NG,en;q=0.9",
        }
    )
    return sess


def get(url: str, **kwargs) -> requests.Response | None:
    """GET a URL, returning ``None`` instead of raising on failure."""
    kwargs.setdefault("timeout", config.HTTP_TIMEOUT)
    verify = kwargs.pop("verify", True)
    try:
        response = session().get(url, verify=verify, **kwargs)
    except requests.RequestException as exc:
        log.warning("GET %s failed: %s", url, exc)
        return None
    if not response.ok:
        log.warning("GET %s returned HTTP %s", url, response.status_code)
        return None
    return response
