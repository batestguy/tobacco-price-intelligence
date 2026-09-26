"""The dashboard renders, in words, with the §11 disclaimer intact.

Streamlit ``AppTest`` runs the script in-process, so the autouse fixtures in
``conftest.py`` still hold: data comes from a seeded temp directory, and no
socket opens. Sign-in is not exercised against Supabase; a signed-in session is
simulated by setting the session keys ``auth.sign_in`` would set.

The label check is the point of the redesign: no title, subheader, metric
label, column header, axis title or legend entry may contain ``_``, which is
how a raw identifier shows up on screen.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from tobacco import config  # noqa: E402
from tobacco.store import parquet_io  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "app"
SRC = REPO / "src"
TIMEOUT = 60

VIEW_SCRIPT = """
import sys
sys.path[:0] = [{app!r}, {src!r}]
import views
views.{view}()
views.render_disclaimer()
"""


@pytest.fixture(autouse=True)
def seeded(isolated_data_dir):
    """A small but complete dataset: every panel has something to draw."""
    today = date(2026, 9, 26)
    days = [today - timedelta(days=n) for n in range(40)][::-1]

    parquet_io.upsert(
        "exchange_rates",
        pd.DataFrame({"date": days, "usd_ngn_rate": [1300 + i for i in range(40)],
                      "is_carried_forward": False}),
    )
    parquet_io.upsert(
        "inflation",
        pd.DataFrame({"date": [today], "rate": [15.4], "source": ["worldbank_gem_monthly"]}),
    )
    parquet_io.upsert(
        "sentiment_aggregates",
        pd.DataFrame({"date": days, "consumer_sentiment": 0.58, "fx_crisis_prob": 0.2}),
    )
    weeks = pd.date_range("2026-06-01", periods=12, freq="W-MON")
    parquet_io.upsert(
        "sales_mock",
        pd.DataFrame(
            [{"week_start": w, "sku": s, "region": r, "quantity_sold": 1000.0}
             for w in weeks for s in config.SKUS for r in config.REGIONS]
        ),
    )
    statuses = iter(["stockout_risk", "overstock"] + ["ok"] * 10)
    constraints = {"PREMIUM_20": "price_grid_upper_bound", "MIDRANGE_20": "profit_optimum",
                   "VALUE_20": "margin_floor"}
    parquet_io.upsert(
        "recommendations",
        pd.DataFrame(
            [{"date": today, "sku": s, "region": r, "price_adjustment_pct": 1.0,
              "recommended_price_ngn": 1234.5, "binding_constraint": constraints[s],
              "forecast_qty_4w": 40000.0, "inventory_action": next(statuses)}
             for s in config.SKUS for r in config.REGIONS]
        ),
    )
    streamlit.cache_data.clear()
    yield


def _labels(at: AppTest) -> list[str]:
    """Every piece of on-screen text that names something."""
    found = [el.value for kind in ("title", "header", "subheader") for el in at.get(kind)]
    found += [el.label for el in at.metric]
    for frame in at.dataframe:
        columns = json.loads(frame.proto.columns or "{}")
        found += [c["label"] for c in columns.values() if c.get("label")]
    for chart in at.get("plotly_chart"):
        spec = json.loads(chart.proto.spec)
        layout = spec.get("layout", {})
        for axis in ("xaxis", "yaxis", "yaxis2"):
            title = layout.get(axis, {}).get("title", {})
            found.append(title.get("text", "") if isinstance(title, dict) else str(title))
        for trace in spec.get("data", []):
            found.append(trace.get("name") or "")
            colorbar = trace.get("colorbar", {}).get("title", {})
            found.append(colorbar.get("text", "") if isinstance(colorbar, dict) else "")
    return [text for text in found if text]


def _assert_readable(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    labels = _labels(at)
    assert labels, "nothing was rendered"
    leaked = [text for text in labels if "_" in text]
    assert not leaked, leaked
    captions = [c.value for c in at.caption]
    assert config.DISCLAIMER in captions


@pytest.mark.parametrize("view", ["executive", "supply_chain", "admin"])
def test_view_renders_in_words(view):
    at = AppTest.from_string(
        VIEW_SCRIPT.format(app=str(APP), src=str(SRC), view=view), default_timeout=TIMEOUT
    )
    at.run()
    _assert_readable(at)
    assert at.title, "every view leads with a verdict"


def test_executive_verdict_reads_from_the_data():
    at = AppTest.from_string(
        VIEW_SCRIPT.format(app=str(APP), src=str(SRC), view="executive"),
        default_timeout=TIMEOUT,
    )
    at.run()
    assert at.title[0].value == "Raise prices this week"


def test_supply_chain_verdict_flags_the_stockout():
    at = AppTest.from_string(
        VIEW_SCRIPT.format(app=str(APP), src=str(SRC), view="supply_chain"),
        default_timeout=TIMEOUT,
    )
    at.run()
    assert at.title[0].value == "1 of 12 at risk of running out"


def test_views_render_with_no_data(isolated_data_dir):
    """A fresh fork has no curated data; every view must still render."""
    import shutil

    shutil.rmtree(isolated_data_dir, ignore_errors=True)
    streamlit.cache_data.clear()
    for view in ("executive", "supply_chain", "admin"):
        at = AppTest.from_string(
            VIEW_SCRIPT.format(app=str(APP), src=str(SRC), view=view),
            default_timeout=TIMEOUT,
        )
        at.run()
        assert not at.exception, (view, [e.value for e in at.exception])


def _app() -> AppTest:
    return AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=TIMEOUT)


def test_sign_in_page_without_secrets_shows_the_disclaimer():
    at = _app()
    at.run()
    assert not at.exception
    assert config.DISCLAIMER in [c.value for c in at.caption]


@pytest.mark.parametrize("role", ["commercial_director", "supply_chain_manager", "admin"])
def test_each_role_lands_on_a_readable_view(role):
    at = _app()
    at.session_state["access_token"] = "test-token"
    at.session_state["user_email"] = "someone@example.com"
    at.session_state["role"] = role
    at.run()
    _assert_readable(at)


def test_every_static_asset_the_app_references_exists():
    """A missing watermark or font fails silently in the browser, so check here."""
    import sys
    import tomllib

    sys.path[:0] = [str(APP), str(SRC)]
    import views

    assert views.LOGO.is_file()
    assert views.PLATE.is_file()
    for filename, _ in views.WATERMARKS.values():
        assert (APP / "static" / "watermarks" / filename).is_file(), filename

    theme = tomllib.loads((REPO / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    faces = theme["theme"]["fontFaces"]
    assert {f["family"] for f in faces} == {theme["theme"]["font"], theme["theme"]["headingFont"]}
    for face in faces:
        # Served at app/static/... from the app directory's static/ folder.
        assert (APP / face["url"].removeprefix("app/")).is_file(), face["url"]

    credits = (APP / "static" / "CREDITS.md").read_text(encoding="utf-8")
    for filename, _ in views.WATERMARKS.values():
        assert filename in credits, filename


DEMO_SECRETS = {
    "SUPABASE_URL": "https://example.invalid",
    "SUPABASE_ANON_KEY": "anon-test",
    "DEMO_ACCOUNTS": {
        "commercial_director": {"email": "demo-cd@example.com", "password": "cd-secret-pw"},
        "supply_chain_manager": {"email": "demo-scm@example.com", "password": "scm-secret-pw"},
        "admin": {"email": "demo-admin@example.com", "password": "admin-secret-pw"},
    },
}


def _all_text(at: AppTest) -> list[str]:
    """Every string the page renders that a viewer could read."""
    text = [el.value for kind in ("title", "header", "subheader", "markdown", "caption",
                                  "error", "info", "success", "warning")
            for el in at.get(kind)]
    text += [b.label for b in at.button]
    text += [t.label for t in at.text_input]
    return [str(t) for t in text if t]


def test_sign_in_page_offers_demo_buttons_without_revealing_credentials():
    at = _app()
    for key, value in DEMO_SECRETS.items():
        at.secrets[key] = value
    at.run()
    assert not at.exception, [e.value for e in at.exception]

    buttons = [b.label for b in at.button]
    assert "Explore as Commercial Director" in buttons
    assert "Explore as Supply Chain Manager" in buttons
    assert not any("Administrator" in label for label in buttons)

    rendered = " ".join(_all_text(at))
    for account in DEMO_SECRETS["DEMO_ACCOUNTS"].values():
        assert account["email"] not in rendered
        assert account["password"] not in rendered
    assert config.DISCLAIMER in [c.value for c in at.caption]


def test_sign_in_page_without_demo_accounts_has_no_demo_buttons():
    at = _app()
    at.secrets["SUPABASE_URL"] = "https://example.invalid"
    at.secrets["SUPABASE_ANON_KEY"] = "anon-test"
    at.run()
    assert not at.exception
    assert not [b for b in at.button if b.label.startswith("Explore as")]


@pytest.mark.parametrize("role", [None, "superuser"])
def test_a_user_without_a_role_sees_no_view(role):
    at = _app()
    at.session_state["access_token"] = "test-token"
    at.session_state["user_email"] = "pending@example.com"
    at.session_state["role"] = role
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert [t.value for t in at.title] == ["No access yet"]
    assert any("no access yet" in m.value for m in at.markdown)
    assert "Sign out" in [b.label for b in at.button]
    assert not at.metric and not at.get("plotly_chart") and not at.dataframe
    assert config.DISCLAIMER in [c.value for c in at.caption]


def test_a_demo_session_never_shows_its_email():
    at = _app()
    at.session_state["access_token"] = "test-token"
    at.session_state["user_email"] = "demo-cd@example.com"
    at.session_state["role"] = "commercial_director"
    at.session_state["demo"] = True
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    rendered = " ".join(_all_text(at))  # the tree includes the sidebar
    assert "demo-cd@example.com" not in rendered
    assert "Demo account" in rendered
