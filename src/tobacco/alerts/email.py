"""Gmail SMTP alerts (INTRO.txt §7).

The four rules, verbatim from the spec:

===============================  ==========================================
FX drops > 2% in 24h             email both roles
Sentiment score < 0.3            email the Commercial Director
Stockout risk (red alert)        email the Supply Chain Manager
Tax change detected              email both roles
===============================  ==========================================

"FX drops" is read as *the naira weakening* -- the NGN/USD quote rising by more
than 2%, which is the direction that hurts an importer of inputs. A falling quote
(a stronger naira) is good news and does not need an alert.

Gmail allows 100 messages/day; this job sends at most four.
"""

from __future__ import annotations

import logging
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage

import pandas as pd

from tobacco import config

log = logging.getLogger(__name__)

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465  # implicit TLS

COMMERCIAL = "commercial"
SUPPLY = "supply"


@dataclass
class Alert:
    rule: str
    subject: str
    body: str
    roles: tuple[str, ...]


def _send(subject: str, body: str, to: list[str]) -> bool:
    """Send one message. Returns False rather than raising: a failed alert must
    not abort the recommendation job that produced it."""
    if not to:
        log.warning("No recipients configured for %r; not sending", subject)
        return False

    try:
        address = config.require("GMAIL_ADDRESS")
        password = config.require("GMAIL_APP_PASSWORD")
    except config.MissingSecret as exc:
        log.warning("Email not configured, skipping alert: %s", exc)
        return False

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = address
    message["To"] = ", ".join(to)
    message.set_content(body)

    try:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.login(address, password)
            server.send_message(message)
    except Exception as exc:  # noqa: BLE001 - never let alerting fail the job
        # str(exc) on an SMTP error contains the server reply, not the password.
        log.error("Failed to send %r: %s", subject, exc)
        return False

    log.info("Sent %r to %d recipient(s)", subject, len(to))
    return True


# ---------------------------------------------------------------------------
# rules
# ---------------------------------------------------------------------------


def check_fx(fx_history: pd.DataFrame) -> Alert | None:
    """Rule 1: the naira weakening more than 2% in 24h."""
    if fx_history.empty or len(fx_history) < 2:
        return None

    recent = fx_history.sort_values("date").tail(2)
    previous, current = (
        float(recent.iloc[0]["usd_ngn_rate"]),
        float(recent.iloc[1]["usd_ngn_rate"]),
    )
    if previous <= 0:
        return None

    change_pct = (current - previous) / previous * 100
    if change_pct <= config.FX_DROP_ALERT_PCT:
        return None

    return Alert(
        rule="fx_move",
        subject=f"[ALERT] Naira weakened {change_pct:.1f}% in 24h",
        body=(
            f"The NGN/USD rate moved from {previous:,.2f} to {current:,.2f} "
            f"({change_pct:+.2f}%) as of {recent.iloc[1]['date']}.\n\n"
            f"This exceeds the {config.FX_DROP_ALERT_PCT}% threshold. Input costs "
            f"are affected; review the pricing recommendation on the dashboard."
        ),
        roles=(COMMERCIAL, SUPPLY),
    )


def check_sentiment(consumer_sentiment: float | None) -> Alert | None:
    """Rule 2: consumer sentiment below 0.3."""
    if consumer_sentiment is None or not pd.notna(consumer_sentiment):
        return None
    if consumer_sentiment >= config.SENTIMENT_ALERT_THRESHOLD:
        return None

    return Alert(
        rule="low_sentiment",
        subject=f"[ALERT] Consumer sentiment at {consumer_sentiment:.2f}",
        body=(
            f"Aggregate consumer sentiment is {consumer_sentiment:.2f}, below the "
            f"{config.SENTIMENT_ALERT_THRESHOLD} threshold.\n\n"
            f"Price acceptance is likely to be weak. Treat any upward price "
            f"recommendation with caution this week."
        ),
        roles=(COMMERCIAL,),
    )


def check_stockouts(stock_alerts: pd.DataFrame) -> Alert | None:
    """Rule 3: any region at stockout risk."""
    if stock_alerts.empty:
        return None
    at_risk = stock_alerts[stock_alerts["status"] == "stockout_risk"]
    if at_risk.empty:
        return None

    lines = [
        f"  - {row['sku']} in {row['region']}: {row['stock_on_hand']:,.0f} packs "
        f"({row['weeks_cover']:.1f} weeks cover, safety {row['safety_stock']:,.0f})"
        for _, row in at_risk.iterrows()
    ]
    return Alert(
        rule="stockout_risk",
        subject=f"[ALERT] Stockout risk in {len(at_risk)} SKU/region combination(s)",
        body=(
            "The following are below safety stock after recommended transfers:\n\n"
            + "\n".join(lines)
            + "\n\nSee the Supply Chain view for the rebalancing plan."
        ),
        roles=(SUPPLY,),
    )


def check_tax_change(news: pd.DataFrame) -> Alert | None:
    """Rule 4: an excise/VAT change detected in recent headlines."""
    if news.empty or "headline" not in news.columns:
        return None

    from tobacco.features.build import TAX_TERMS

    recent = news.copy()
    recent["published_at"] = pd.to_datetime(recent["published_at"])
    cutoff = pd.Timestamp(config.today_wat()) - pd.Timedelta(days=1)
    recent = recent[recent["published_at"] >= cutoff]

    hits = recent[recent["headline"].fillna("").str.contains(TAX_TERMS)]
    if hits.empty:
        return None

    lines = [f"  - {row['headline']}\n    {row['url']}" for _, row in hits.head(5).iterrows()]
    return Alert(
        rule="tax_change",
        subject=f"[ALERT] Possible excise/VAT change in the news ({len(hits)} headline(s))",
        body=(
            "Headlines in the last 24h matched excise/tax keywords:\n\n"
            + "\n".join(lines)
            + "\n\nThese are keyword matches, not confirmed policy. Verify against "
              "the official gazette before acting."
        ),
        roles=(COMMERCIAL, SUPPLY),
    )


def dispatch(alerts: list[Alert | None]) -> int:
    """Send every triggered alert to its roles. Returns the number sent."""
    sent = 0
    for alert in alerts:
        if alert is None:
            continue
        recipients: list[str] = []
        for role in alert.roles:
            recipients.extend(config.recipients(role))
        # De-duplicate while preserving order: one person may hold both roles.
        recipients = list(dict.fromkeys(recipients))
        if _send(alert.subject, alert.body, recipients):
            sent += 1
    log.info("Alerts: %d triggered, %d sent", sum(a is not None for a in alerts), sent)
    return sent
