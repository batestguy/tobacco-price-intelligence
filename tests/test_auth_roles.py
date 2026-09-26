"""A role is granted, never assumed (app/auth.py).

Offline like the rest of the suite: ``requests`` is replaced with fakes that
record what was asked, and ``auth.st`` with a namespace holding plain dicts for
``secrets`` and ``session_state``. No real Supabase project or credential is
involved anywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("streamlit")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "app"))

import auth  # noqa: E402
import requests  # noqa: E402

USER_ID = "00000000-0000-0000-0000-000000000001"
SECRETS = {"SUPABASE_URL": "https://example.invalid", "SUPABASE_ANON_KEY": "anon-test"}


class FakeResponse:
    def __init__(self, status: int = 200, body=None):
        self.status_code = status
        self.ok = 200 <= status < 300
        self._body = body

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


@pytest.fixture
def fake_st(monkeypatch):
    namespace = SimpleNamespace(secrets=dict(SECRETS), session_state={})
    monkeypatch.setattr(auth, "st", namespace)
    return namespace


@pytest.fixture
def calls(monkeypatch):
    """Record every request; each test sets the responses it wants."""
    log = {"get": [], "post": [], "patch": [], "responses": {}}

    def handler(method):
        def call(url, **kwargs):
            log[method].append({"url": url, **kwargs})
            response = log["responses"].get(method)
            if isinstance(response, Exception):
                raise response
            return response
        return call

    for method in ("get", "post", "patch"):
        monkeypatch.setattr(auth.requests, method, handler(method))
    return log


def _session(role_rows):
    return {
        "post": FakeResponse(200, {"access_token": "jwt", "user": {"id": USER_ID,
                                                                  "email": "u@example.com"}}),
        "get": role_rows,
    }


# ---------------------------------------------------------------------------
# role lookup
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "role_response, expected_problem",
    [
        (FakeResponse(200, []), auth.NO_ROLE),                      # no row
        (FakeResponse(200, [{"role": None}]), auth.NO_ROLE),        # pending
        (FakeResponse(200, [{"role": "superuser"}]), auth.NO_ROLE),  # unknown
        (FakeResponse(500, {"message": "boom"}), auth.LOOKUP_FAILED),
        (FakeResponse(200, ValueError("not json")), auth.LOOKUP_FAILED),
        (requests.ConnectionError("down"), auth.LOOKUP_FAILED),
    ],
    ids=["no-row", "null-role", "unknown-role", "http-error", "bad-json", "exception"],
)
def test_no_role_means_no_access(fake_st, calls, role_response, expected_problem):
    calls["responses"].update(_session(role_response))
    ok, _ = auth.sign_in("u@example.com", "pw")
    assert ok  # the credentials were fine; the role is what is missing
    user = auth.current_user()
    assert user["role"] is None
    assert user["problem"] == expected_problem


@pytest.mark.parametrize("role", auth.ROLES)
def test_a_granted_role_is_used(fake_st, calls, role):
    calls["responses"].update(_session(FakeResponse(200, [{"role": role}])))
    auth.sign_in("u@example.com", "pw")
    user = auth.current_user()
    assert user["role"] == role
    assert user["problem"] is None
    assert user["id"] == USER_ID


def test_role_query_filters_by_the_users_own_id(fake_st, calls):
    calls["responses"].update(_session(FakeResponse(200, [{"role": "admin"}])))
    auth.sign_in("u@example.com", "pw")
    (lookup,) = calls["get"]
    assert lookup["url"].endswith("/rest/v1/users")
    assert lookup["params"]["id"] == f"eq.{USER_ID}"
    assert "limit" not in lookup["params"]
    assert lookup["headers"]["Authorization"] == "Bearer jwt"


def test_an_unknown_role_in_session_state_is_no_access(fake_st):
    fake_st.session_state.update(access_token="jwt", role="commercial director")
    assert auth.current_user()["role"] is None


def test_rejected_credentials_do_not_sign_in(fake_st, calls):
    calls["responses"]["post"] = FakeResponse(400, {"error": "invalid_grant"})
    ok, message = auth.sign_in("u@example.com", "wrong")
    assert not ok
    assert message == "Invalid email or password."
    assert auth.current_user() is None
    assert not calls["get"]


# ---------------------------------------------------------------------------
# demo sessions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("role", ["commercial_director", "supply_chain_manager"])
def test_a_demo_session_opens_its_view_without_any_request(fake_st, calls, role):
    assert auth.start_demo(role) is True
    user = auth.current_user()
    assert user["role"] == role
    assert user["demo"] is True
    assert user["email"] == "Demo session"
    assert not calls["get"] and not calls["post"]


@pytest.mark.parametrize("role", ["admin", "superuser", None])
def test_a_demo_session_is_never_an_administrator(fake_st, role):
    assert auth.start_demo(role) is False
    assert auth.current_user() is None


def test_a_forged_demo_role_in_session_state_is_no_access(fake_st):
    fake_st.session_state.update(demo=True, role="admin")
    user = auth.current_user()
    assert user["role"] is None
    assert user["problem"]


def test_starting_a_demo_replaces_a_signed_in_session(fake_st):
    fake_st.session_state.update(access_token="jwt", user_id="u1", role="admin")
    auth.start_demo("commercial_director")
    assert "access_token" not in fake_st.session_state
    assert auth.current_user()["role"] == "commercial_director"


def test_sign_out_ends_a_demo_session(fake_st):
    auth.start_demo("supply_chain_manager")
    auth.sign_out()
    assert auth.current_user() is None


# ---------------------------------------------------------------------------
# administration
# ---------------------------------------------------------------------------


def test_set_role_treats_an_empty_patch_result_as_failure(fake_st, calls):
    """RLS filters a forbidden update to zero rows instead of refusing it."""
    fake_st.session_state["access_token"] = "jwt"
    calls["responses"]["patch"] = FakeResponse(200, [])
    ok, _ = auth.set_role(USER_ID, "supply_chain_manager")
    assert not ok


def test_set_role_sends_the_admins_token_and_filters_by_id(fake_st, calls):
    fake_st.session_state["access_token"] = "jwt"
    calls["responses"]["patch"] = FakeResponse(200, [{"id": USER_ID, "role": None}])
    ok, _ = auth.set_role(USER_ID, None)
    assert ok
    (patch,) = calls["patch"]
    assert patch["params"] == {"id": f"eq.{USER_ID}"}
    assert patch["json"] == {"role": None}
    assert patch["headers"]["Authorization"] == "Bearer jwt"
    assert patch["headers"]["Prefer"] == "return=representation"


def test_set_role_refuses_an_unknown_role_without_a_request(fake_st, calls):
    fake_st.session_state["access_token"] = "jwt"
    assert auth.set_role(USER_ID, "owner")[0] is False
    assert not calls["patch"]


def test_set_role_request_failure_is_a_message_not_an_exception(fake_st, calls):
    fake_st.session_state["access_token"] = "jwt"
    calls["responses"]["patch"] = requests.ConnectionError("down")
    ok, message = auth.set_role(USER_ID, "admin")
    assert not ok and message


def test_list_users_needs_a_session(fake_st, calls):
    assert auth.list_users()[0] is False
    assert not calls["get"]


def test_list_users_returns_rows(fake_st, calls):
    fake_st.session_state["access_token"] = "jwt"
    rows = [{"id": USER_ID, "email": "u@example.com", "role": None,
             "created_at": "2026-09-26T10:00:00+00:00"}]
    calls["responses"]["get"] = FakeResponse(200, rows)
    assert auth.list_users() == (True, rows)
    assert calls["get"][0]["params"]["order"] == "created_at.asc"
