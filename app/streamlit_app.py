"""Role-based dashboard (INTRO.txt §6).

Streamlit rather than the spec's Plotly Dash: Dash would need a host that accepts
manual uploads, and Streamlit Community Cloud redeploys straight from this
repository on every push. Streamlit is the spec's own §6 option C.

This file is sign-in and routing only; the views live in ``views.py``. The §11
disclaimer is rendered verbatim on the sign-in page and in the footer of every
view, from the single copy in ``tobacco.config.DISCLAIMER``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Must precede the `tobacco` import: the package is not installed, it is read
# from src/ in the checked-out repo. Doing this here rather than relying on
# app.data's side effect keeps the import order safe to reformat.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import auth  # noqa: E402
import labels  # noqa: E402
import streamlit as st  # noqa: E402
import views  # noqa: E402

st.set_page_config(
    page_title="Price intelligence",
    page_icon=":material/eco:",
    layout="wide",
)


def render_login() -> None:
    st.title("Price intelligence", anchor=False)
    st.markdown("Pricing and supply-chain decision support for the Nigerian market.")

    if not auth.configured():
        st.error(
            "Sign-in is not configured. Set `SUPABASE_URL` and `SUPABASE_ANON_KEY` "
            "in the app's secrets on Streamlit Cloud. They are set separately from "
            "the GitHub Actions secrets, and the app takes the anon key, never the "
            "service key.",
            icon=":material/key_off:",
        )
        views.render_disclaimer()
        return

    with st.form("login"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary")

    if submitted:
        ok, message = auth.sign_in(email, password)
        if ok:
            st.rerun()
        else:
            st.error(message, icon=":material/error:")

    views.render_disclaimer()


#: Route guarding is by construction: a role's pages are the only ones
#: registered. That is presentation, not access control -- the underlying data
#: is committed to a public repo (see app/auth.py).
PAGES = {
    "executive": st.Page(
        views.executive, title="Executive", icon=":material/insights:",
        url_path="executive",
    ),
    "supply_chain": st.Page(
        views.supply_chain, title="Supply chain", icon=":material/local_shipping:",
        url_path="supply-chain",
    ),
    "admin": st.Page(
        views.admin, title="Administration", icon=":material/admin_panel_settings:",
        url_path="administration",
    ),
}

ROLE_PAGES = {
    "commercial_director": ["executive"],
    "supply_chain_manager": ["supply_chain"],
    "admin": ["executive", "supply_chain", "admin"],
}


def main() -> None:
    user = auth.current_user()
    if user is None:
        st.navigation(
            [st.Page(render_login, title="Sign in", icon=":material/login:")],
            position="hidden",
        ).run()
        return

    role = user["role"]
    pages = [PAGES[key] for key in ROLE_PAGES.get(role, ROLE_PAGES["commercial_director"])]
    page = st.navigation(pages, position="sidebar" if len(pages) > 1 else "hidden")

    with st.sidebar:
        st.markdown(f"**{labels.role(role)}**")
        st.caption(user["email"])
        if st.button("Sign out", icon=":material/logout:"):
            auth.sign_out()
            st.rerun()

    page.run()
    views.render_disclaimer()


main()
