"""The four §7 alert rules, their boundaries, and the no-credentials path.

``_send`` returning False rather than raising is the property that keeps a failed
alert from aborting the recommend job that produced it -- and with zero Actions
secrets set today, the no-credentials path is the one that actually runs.
"""

from __future__ import annotations

import smtplib

import pandas as pd
import pytest

from tobacco import config
from tobacco.alerts import email


def fx_frame(*rates, dates=None):
    dates = dates or [f"2026-08-{20 + i:02d}" for i in range(len(rates))]
    return pd.DataFrame({"date": dates, "usd_ngn_rate": list(rates)})


# ---------------------------------------------------------------------------
# rule 1: FX
# ---------------------------------------------------------------------------


def test_check_fx_is_silent_without_two_observations():
    assert email.check_fx(pd.DataFrame(columns=["date", "usd_ngn_rate"])) is None
    assert email.check_fx(fx_frame(1500.0)) is None


def test_check_fx_fires_when_the_naira_weakens_past_the_threshold():
    alert = email.check_fx(fx_frame(1500.0, 1545.0))  # +3.0%
    assert alert is not None
    assert alert.rule == "fx_move"
    assert set(alert.roles) == {email.COMMERCIAL, email.SUPPLY}
    assert "3.0%" in alert.subject


def test_check_fx_does_not_fire_when_the_naira_strengthens():
    """"FX drops" is read as the *quote* rising -- a stronger naira is good news."""
    assert email.check_fx(fx_frame(1545.0, 1400.0)) is None


def test_check_fx_does_not_fire_exactly_at_the_threshold():
    assert config.FX_DROP_ALERT_PCT == 2.0
    assert email.check_fx(fx_frame(1000.0, 1020.0)) is None  # exactly +2%
    assert email.check_fx(fx_frame(1000.0, 1020.01)) is not None


def test_check_fx_ignores_a_non_positive_previous_rate():
    assert email.check_fx(fx_frame(0.0, 1500.0)) is None


def test_check_fx_compares_the_last_two_rows_however_far_apart_they_are():
    """The subject says "in 24h" but the code takes ``tail(2)`` of the sorted frame.

    CBN publishes with gaps, so those two rows can be days apart. Documented here
    rather than changed: narrowing it to a real 24h window is a behaviour decision,
    not a cleanup.
    """
    frame = fx_frame(1500.0, 1545.0, dates=["2026-07-01", "2026-08-20"])
    alert = email.check_fx(frame)
    assert alert is not None and "24h" in alert.subject


def test_check_fx_sorts_by_date_rather_than_trusting_row_order():
    frame = fx_frame(1545.0, 1500.0, dates=["2026-08-21", "2026-08-20"])
    assert email.check_fx(frame) is not None


# ---------------------------------------------------------------------------
# rule 2: sentiment
# ---------------------------------------------------------------------------


def test_check_sentiment_is_silent_for_a_missing_reading():
    assert email.check_sentiment(None) is None
    assert email.check_sentiment(float("nan")) is None


def test_check_sentiment_fires_below_the_threshold():
    alert = email.check_sentiment(0.29)
    assert alert is not None
    assert alert.rule == "low_sentiment"
    assert alert.roles == (email.COMMERCIAL,), "the spec routes this to Commercial only"


def test_check_sentiment_does_not_fire_at_the_threshold():
    """`>= threshold` returns None, so 0.3 exactly is not an alert."""
    assert config.SENTIMENT_ALERT_THRESHOLD == 0.3
    assert email.check_sentiment(0.3) is None


# ---------------------------------------------------------------------------
# rule 3: stockouts
# ---------------------------------------------------------------------------


def stock_frame(*statuses):
    return pd.DataFrame(
        [
            {
                "sku": "PREMIUM_20",
                "region": "Kano",
                "stock_on_hand": 100.0,
                "safety_stock": 500.0,
                "weeks_cover": 0.3,
                "status": status,
            }
            for status in statuses
        ]
    )


def test_check_stockouts_is_silent_without_alerts():
    assert email.check_stockouts(pd.DataFrame()) is None


def test_check_stockouts_ignores_overstock():
    assert email.check_stockouts(stock_frame("overstock")) is None


def test_check_stockouts_fires_for_the_supply_role_only():
    alert = email.check_stockouts(stock_frame("stockout_risk", "overstock"))
    assert alert is not None
    assert alert.roles == (email.SUPPLY,)
    assert "1 SKU/region" in alert.subject


# ---------------------------------------------------------------------------
# rule 4: tax change
# ---------------------------------------------------------------------------


def news_frame(headline: str, days_ago: int = 0):
    return pd.DataFrame(
        [
            {
                "headline": headline,
                "url": "https://example.com/a",
                "published_at": pd.Timestamp(config.today_wat()) - pd.Timedelta(days=days_ago),
            }
        ]
    )


def test_check_tax_change_is_silent_without_headlines():
    assert email.check_tax_change(pd.DataFrame()) is None
    assert email.check_tax_change(pd.DataFrame([{"url": "x"}])) is None


def test_check_tax_change_fires_on_a_recent_excise_headline():
    alert = email.check_tax_change(news_frame("FG raises excise duty on beverages"))
    assert alert is not None
    assert set(alert.roles) == {email.COMMERCIAL, email.SUPPLY}
    assert "gazette" in alert.body, "keyword matches must not read as confirmed policy"


def test_check_tax_change_ignores_headlines_older_than_the_window():
    assert email.check_tax_change(news_frame("FG raises excise duty", days_ago=3)) is None


def test_check_tax_change_ignores_an_unrelated_headline():
    assert email.check_tax_change(news_frame("Super Eagles win in Kano")) is None


# ---------------------------------------------------------------------------
# sending
# ---------------------------------------------------------------------------


def test_send_returns_false_without_credentials_and_never_opens_a_socket(monkeypatch):
    """The whole alerting path must be inert, not fatal, when unconfigured.

    With zero Actions secrets set this is the live behaviour, so `_send` raising
    here would take `jobs.recommend` down with it after the recommendations had
    already been computed.
    """

    def explode(*args, **kwargs):
        raise AssertionError("_send reached SMTP without credentials")

    monkeypatch.setattr(smtplib, "SMTP_SSL", explode)
    assert email._send("subject", "body", ["someone@example.com"]) is False


def test_send_returns_false_with_no_recipients(monkeypatch):
    monkeypatch.setenv("GMAIL_ADDRESS", "bot@example.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "app-password")
    monkeypatch.setattr(
        smtplib, "SMTP_SSL", lambda *a, **k: pytest.fail("sent with no recipients")
    )
    assert email._send("subject", "body", []) is False


def test_send_returns_false_when_smtp_raises(monkeypatch):
    monkeypatch.setenv("GMAIL_ADDRESS", "bot@example.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "app-password")

    def refuse(*args, **kwargs):
        raise smtplib.SMTPAuthenticationError(535, b"bad credentials")

    monkeypatch.setattr(smtplib, "SMTP_SSL", refuse)
    assert email._send("subject", "body", ["someone@example.com"]) is False


def test_dispatch_skips_untriggered_rules_and_reports_zero_sent():
    assert email.dispatch([None, None]) == 0


def test_dispatch_survives_an_unconfigured_mailer():
    alert = email.check_sentiment(0.1)
    assert email.dispatch([alert, None]) == 0


def test_dispatch_deduplicates_a_recipient_holding_both_roles(monkeypatch):
    monkeypatch.setenv("ALERT_RECIPIENTS_COMMERCIAL", "both@example.com,c@example.com")
    monkeypatch.setenv("ALERT_RECIPIENTS_SUPPLY", "both@example.com")

    seen: list[list[str]] = []
    monkeypatch.setattr(email, "_send", lambda subject, body, to: seen.append(to) or True)

    email.dispatch([email.check_fx(fx_frame(1500.0, 1545.0))])
    assert seen == [["both@example.com", "c@example.com"]]
