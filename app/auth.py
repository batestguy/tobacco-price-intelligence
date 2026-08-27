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
which keeps a session from enumerating other people's role assignments.

The spec's §6 mentions ``dash-auth`` as an alternative. That is a static
user:password dict compiled into the app, which on a public repo would mean
committing credentials. Supabase Auth is used instead.
"""

from __future__ import annotations

import requests
import streamlit as st

TIMEOUT = 20

ROLE_LABELS = {
    "commercial_director": "Commercial Director",
    "supply_chain_manager": "Supply Chain Manager",
    "admin": "Administrator",
}


def _config(key: str) -> str | None:
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


def _auth_headers() -> dict[str, str]:
    return {
        "apikey": _config("SUPABASE_ANON_KEY"),
        "Content-Type": "application/json",
    }


def sign_in(email: str, password: str) -> tuple[bool, str]:
    """Exchange credentials for a session. Returns ``(ok, message)``."""
    url = f"{_config('SUPABASE_URL').rstrip('/')}/auth/v1/token"
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

    payload = response.json()
    st.session_state["access_token"] = payload.get("access_token")
    st.session_state["user_email"] = payload.get("user", {}).get("email", email)
    st.session_state["role"] = _fetch_role(payload.get("access_token"))
    return True, "Signed in."


def _fetch_role(token: str | None) -> str:
    """Look up the caller's role. Defaults to the least-privileged view."""
    if not token:
        return "commercial_director"
    try:
        response = requests.get(
            f"{_config('SUPABASE_URL').rstrip('/')}/rest/v1/users",
            params={"select": "role", "limit": "1"},
            headers={
                "apikey": _config("SUPABASE_ANON_KEY"),
                "Authorization": f"Bearer {token}",
            },
            timeout=TIMEOUT,
        )
        if response.ok and response.json():
            return response.json()[0].get("role", "commercial_director")
    except requests.RequestException:
        pass
    return "commercial_director"


def sign_out() -> None:
    for key in ("access_token", "user_email", "role"):
        st.session_state.pop(key, None)


def current_user() -> dict | None:
    if not st.session_state.get("access_token"):
        return None
    return {
        "email": st.session_state.get("user_email"),
        "role": st.session_state.get("role", "commercial_director"),
    }
