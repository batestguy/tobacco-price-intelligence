"""Supabase Auth for the Streamlit dashboard (INTRO.txt §6).

**Login is view routing, not a confidentiality boundary.** Say so plainly rather
than implying protection the system does not provide: the dashboard reads the
Parquet committed in its own checkout, and that repository is public, so every
figure behind this login is already world-readable with no credential. Sales are
synthetic and everything else is public macro data, so there is nothing
confidential to protect in the first place. What the login buys is the role-based
views of §6 and consistency with §11's "authorized personnel only" framing.

The app is deployed publicly on Streamlit Community Cloud, so the URL is not a
secret either, and it holds only the Supabase **anon** key. Row level security in
``supabase/schema.sql`` covers the one table this module reads -- ``users`` --
which keeps a session from enumerating other people's role assignments, and lets
only an administrator change one. The admin helpers below send the signed-in
administrator's own token, so RLS decides what they may do; there is no service
key anywhere in this app.

**A role is granted, never assumed.** A signed-in user with no ``users`` row, a
null role, an unknown role, or a role lookup that failed gets no view at all.
New sign-ups arrive pending, so public sign-up on the Supabase project cannot
hand anyone a view.

**Demo sessions** let a visitor explore without an account. "Explore as ..."
opens a non-admin view directly, with no Supabase account, password or secret
behind it: login is not a confidentiality boundary here, so a shared demo
password would protect nothing, and a demo that never calls Supabase keeps
working even when the free project is paused. A demo session has no token, so
the admin helpers below cannot run in one, and it is never an administrator.

The spec's §6 mentions ``dash-auth`` as an alternative. That is a static
user:password dict compiled into the app, which on a public repo would mean
committing credentials. Supabase Auth is used instead.
"""

from __future__ import annotations

import requests
import streamlit as st
from labels import ROLE_LABELS  # noqa: F401 - re-exported; labels owns the vocabulary

TIMEOUT = 20

#: Roles a view can be granted to. Anything else in ``users.role`` is no access.
ROLES = ("commercial_director", "supply_chain_manager", "admin")

#: Roles a demo session may open. Never ``admin``.
DEMO_ROLES = ("commercial_director", "supply_chain_manager")

#: Why a signed-in user has no view. Stored in session state beside ``role``.
NO_ROLE = "Your account has no access yet. Ask an administrator to assign you a role."
LOOKUP_FAILED = (
    "Your role could not be checked, so no view is shown. Sign out and try again "
    "in a moment."
)


def _config(key: str):
    """Read a Streamlit secret. Absent secrets are a normal state, not an error.

    Streamlit Cloud secrets are set in the app dashboard and are NOT inherited
    from GitHub Actions secrets -- a frequent source of confusion on first deploy.
    """
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return None


def configured() -> bool:
    return bool(_config("SUPABASE_URL") and _config("SUPABASE_ANON_KEY"))


def _base() -> str:
    return _config("SUPABASE_URL").rstrip("/")


def _auth_headers() -> dict[str, str]:
    return {
        "apikey": _config("SUPABASE_ANON_KEY"),
        "Content-Type": "application/json",
    }


def _user_headers(token: str) -> dict[str, str]:
    return {**_auth_headers(), "Authorization": f"Bearer {token}"}


def sign_in(email: str, password: str) -> tuple[bool, str]:
    """Exchange credentials for a session. Returns ``(ok, message)``.

    ``ok`` means the credentials were accepted. Whether the user then sees any
    view depends on the role lookup; see ``current_user``.
    """
    url = f"{_base()}/auth/v1/token"
    try:
        response = requests.post(
            url,
            params={"grant_type": "password"},
            headers=_auth_headers(),
            json={"email": email, "password": password},
            timeout=TIMEOUT,
        )
    except requests.RequestException as exc:
        return False, f"Could not reach the authentication service: {exc}"

    if not response.ok:
        # Deliberately generic: distinguishing "no such user" from "wrong
        # password" would let anyone enumerate accounts on a public URL.
        return False, "Invalid email or password."

    try:
        payload = response.json()
    except ValueError:
        return False, "The authentication service returned an unreadable response."
    token = payload.get("access_token")
    user = payload.get("user") or {}
    if not token or not user.get("id"):
        return False, "The authentication service returned an incomplete session."

    role, problem = _fetch_role(token, user["id"])
    st.session_state["access_token"] = token
    st.session_state["user_id"] = user["id"]
    st.session_state["user_email"] = user.get("email", email)
    st.session_state["role"] = role
    st.session_state["role_problem"] = problem
    return True, "Signed in."


def _fetch_role(token: str, user_id: str) -> tuple[str | None, str | None]:
    """The caller's role as ``(role, None)``, or ``(None, reason)`` when they have none.

    Filtered by id rather than relying on RLS to leave one row: an administrator
    can read every row, and ``limit=1`` would hand them whoever came first.
    """
    try:
        response = requests.get(
            f"{_base()}/rest/v1/users",
            params={"select": "role", "id": f"eq.{user_id}"},
            headers=_user_headers(token),
            timeout=TIMEOUT,
        )
        if not response.ok:
            return None, LOOKUP_FAILED
        rows = response.json()
    except (requests.RequestException, ValueError):
        return None, LOOKUP_FAILED

    if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
        return None, NO_ROLE
    role = rows[0].get("role")
    if role not in ROLES:
        return None, NO_ROLE
    return role, None


def start_demo(role: str) -> bool:
    """Open a demo session for a non-admin ``role``. No account, no network."""
    if role not in DEMO_ROLES:
        return False
    sign_out()
    st.session_state["demo"] = True
    st.session_state["role"] = role
    return True


def sign_out() -> None:
    for key in ("access_token", "user_id", "user_email", "role", "role_problem", "demo"):
        st.session_state.pop(key, None)


def current_user() -> dict | None:
    """The signed-in user, or ``None``. ``role`` is ``None`` when they have no access."""
    if st.session_state.get("demo"):
        role = st.session_state.get("role")
        return {
            "id": None,
            "email": "Demo session",
            "role": role if role in DEMO_ROLES else None,
            "problem": None if role in DEMO_ROLES else NO_ROLE,
            "demo": True,
        }
    if not st.session_state.get("access_token"):
        return None
    role = st.session_state.get("role")
    if role not in ROLES:
        role = None
    return {
        "id": st.session_state.get("user_id"),
        "email": st.session_state.get("user_email"),
        "role": role,
        "problem": None if role else (st.session_state.get("role_problem") or NO_ROLE),
        "demo": False,
    }


# ---------------------------------------------------------------------------
# administration: the signed-in administrator's own token, so RLS decides
# ---------------------------------------------------------------------------


def list_users() -> tuple[bool, list[dict] | str]:
    """Every ``users`` row, oldest first, as ``(True, rows)`` or ``(False, message)``."""
    token = st.session_state.get("access_token")
    if not token or not configured():
        return False, "Not signed in."
    try:
        response = requests.get(
            f"{_base()}/rest/v1/users",
            params={"select": "id,email,role,created_at", "order": "created_at.asc"},
            headers=_user_headers(token),
            timeout=TIMEOUT,
        )
        if not response.ok:
            return False, f"Could not load users (HTTP {response.status_code})."
        rows = response.json()
    except requests.RequestException as exc:
        return False, f"Could not reach the database: {exc}"
    except ValueError:
        return False, "The database returned an unreadable response."
    if not isinstance(rows, list):
        return False, "The database returned an unexpected response."
    return True, rows


def set_role(user_id: str, role: str | None) -> tuple[bool, str]:
    """Set one user's role; ``None`` revokes access. Returns ``(ok, message)``.

    RLS filters an update the caller may not make down to zero rows rather than
    refusing it, so an empty result is a failure, not a success.
    """
    if role is not None and role not in ROLES:
        return False, "Unknown role."
    token = st.session_state.get("access_token")
    if not token or not configured():
        return False, "Not signed in."
    try:
        response = requests.patch(
            f"{_base()}/rest/v1/users",
            params={"id": f"eq.{user_id}"},
            headers={**_user_headers(token), "Prefer": "return=representation"},
            json={"role": role},
            timeout=TIMEOUT,
        )
        if not response.ok:
            return False, f"Could not update the role (HTTP {response.status_code})."
        rows = response.json()
    except requests.RequestException as exc:
        return False, f"Could not reach the database: {exc}"
    except ValueError:
        return False, "The database returned an unreadable response."
    if not isinstance(rows, list) or not rows:
        return False, "The role was not changed. Only an administrator can change roles."
    return True, "Role updated."
