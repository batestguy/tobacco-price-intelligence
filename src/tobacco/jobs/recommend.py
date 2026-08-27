"""Daily recommendation engine (INTRO.txt §9 step 3).

Forecast -> optimize -> persist -> alert -> memo. Runs after the morning scrape.

The alert and memo stages are deliberately non-fatal: the recommendation itself
is the product, and it has already been committed by the time either runs. A
Gmail hiccup or an exhausted Groq quota must not discard a good day's output.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from tobacco import config
from tobacco.alerts import email
from tobacco.memo import groq
from tobacco.models import predict
from tobacco.optimize import linprog
from tobacco.store import parquet_io

log = config.setup_logging("recommend")

MEMO_DIR = config.REPO_ROOT / "data" / "memos"


def _competitor_average() -> float | None:
    """Mean competitor pack price over the last 14 days, or None if unobserved."""
    prices = parquet_io.read(
        "competitor_prices", since=pd.Timestamp(config.today_wat()) - pd.Timedelta(days=14)
    )
    if prices.empty:
        return None
    value = float(prices["price"].mean())
    return value if np.isfinite(value) else None


def _fx_context() -> tuple[pd.DataFrame, float | None, float | None]:
    """Recent FX history plus the latest rate and its 7-day change."""
    history = parquet_io.read("exchange_rates")
    if history.empty:
        return history, None, None

    history = history.sort_values("date")
    latest = float(history.iloc[-1]["usd_ngn_rate"])

    week_ago = pd.to_datetime(history.iloc[-1]["date"]) - pd.Timedelta(days=7)
    prior = history[pd.to_datetime(history["date"]) <= week_ago]
    if prior.empty:
        return history, latest, None

    previous = float(prior.iloc[-1]["usd_ngn_rate"])
    change = (latest - previous) / previous * 100 if previous else None
    return history, latest, change


def _sentiment_context() -> tuple[float | None, float | None]:
    """Latest non-null crisis probability and consumer sentiment."""
    aggregates = parquet_io.read("sentiment_aggregates")
    if aggregates.empty:
        return None, None

    aggregates = aggregates.sort_values("date")

    def _last_valid(column: str) -> float | None:
        if column not in aggregates.columns:
            return None
        values = aggregates[column].dropna()
        return float(values.iloc[-1]) if not values.empty else None

    return _last_valid("fx_crisis_prob"), _last_valid("consumer_sentiment")


def _overall_stock_alert(result: linprog.OptimizationResult) -> str:
    """Collapse per-region alerts into the memo's low/ok/high field."""
    if result.stock_alerts.empty:
        return "ok"
    statuses = set(result.stock_alerts["status"])
    if "stockout_risk" in statuses:
        return "low"
    if "overstock" in statuses:
        return "high"
    return "ok"


def run() -> int:
    log.info("Recommendation starting at %s WAT",
             config.now_wat().isoformat(timespec="seconds"))

    # --- forecast + optimize (fatal if these fail) --------------------------
    try:
        forecast = predict.forecast()
        if forecast.empty:
            log.error("Forecast produced no rows; nothing to recommend")
            return 1

        competitor_avg = _competitor_average()
        result = linprog.optimise(forecast, competitor_avg)
        recommendations = linprog.to_recommendations(result, forecast)

        parquet_io.upsert("recommendations", recommendations)
    except predict.ModelNotTrained as exc:
        log.error("%s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        log.exception("Recommendation failed: %s", exc)
        return 1

    for note in result.notes:
        log.warning("Optimizer note: %s", note)

    log.info(
        "Recommended overall adjustment %+.2f%%; %d transfer(s), %d alert(s)",
        result.overall_adjustment_pct, len(result.transfers), len(result.stock_alerts),
    )

    fx_history, fx_rate, fx_change = _fx_context()
    crisis, sentiment = _sentiment_context()

    # --- alerts (non-fatal) -------------------------------------------------
    try:
        news = parquet_io.read(
            "news_articles", since=pd.Timestamp(config.today_wat()) - pd.Timedelta(days=2)
        )
        email.dispatch(
            [
                email.check_fx(fx_history),
                email.check_sentiment(sentiment),
                email.check_stockouts(result.stock_alerts),
                email.check_tax_change(news),
            ]
        )
    except Exception as exc:  # noqa: BLE001
        log.error("Alerting failed (recommendations are already saved): %s", exc)

    # --- memo (non-fatal) ---------------------------------------------------
    try:
        inflation_rate, inflation_note = _latest_inflation()
        memo = groq.generate(
            fx_rate=f"{fx_rate:,.2f}" if fx_rate else None,
            fx_change=f"{fx_change:+.2f}" if fx_change is not None else None,
            inflation=inflation_rate,
            sentiment=f"{sentiment:.2f}" if sentiment is not None else None,
            crisis_score=f"{crisis:.2f}" if crisis is not None else None,
            price_rec=f"{result.overall_adjustment_pct:+.2f}",
            competitor_price=f"NGN {competitor_avg:,.0f}" if competitor_avg else None,
            demand_trend=predict.demand_trend(forecast),
            stock_alert=_overall_stock_alert(result),
            # The optimizer's own notes already say when a constraint is
            # inactive; passing them through stops the memo reasoning as though
            # a bound applied when none did.
            notes=[inflation_note, *result.notes],
        )
        MEMO_DIR.mkdir(parents=True, exist_ok=True)
        memo_path = MEMO_DIR / f"{config.today_wat()}.md"
        memo_path.write_text(memo + "\n", encoding="utf-8")
        log.info("Memo written to %s", memo_path.relative_to(config.REPO_ROOT))
    except Exception as exc:  # noqa: BLE001
        log.error("Memo generation failed (recommendations are already saved): %s", exc)

    log.info(
        "Recommendation complete: %d recommendation(s), %+.2f%% overall",
        len(recommendations), result.overall_adjustment_pct,
    )
    return 0


#: How each inflation tier should be described in the memo. The §10 prompt calls
#: the field "Monthly inflation rate" and that text is fixed, so when the figure
#: is not monthly the caveat has to travel alongside it.
_INFLATION_BASIS = {
    "cbn_monthly": "monthly (CBN, republishing the NBS CPI series)",
    "nbs_release": "monthly (NBS release)",
    "seed": "monthly (committed NBS back series)",
    "worldbank_annual": (
        "ANNUAL, not monthly (World Bank FP.CPI.TOTL.ZG). NBS has no reachable "
        "machine-readable release and CBN's monthly table renders client-side, "
        "so this is the most recent full calendar year, carried forward"
    ),
}


def _latest_inflation() -> tuple[str | None, str]:
    """Latest inflation rate, plus a note stating what basis it is on."""
    inflation = parquet_io.read("inflation")
    if inflation.empty:
        return None, "Inflation is unavailable from every source tier."

    row = inflation.sort_values("date").iloc[-1]
    source = str(row.get("source") or "unknown")
    basis = _INFLATION_BASIS.get(source, f"from an unrecognised tier {source!r}")
    note = (
        f"Inflation is {basis}. Observation dated "
        f"{pd.Timestamp(row['date']).date()}."
    )
    return f"{float(row['rate']):.2f}", note


if __name__ == "__main__":
    sys.exit(run())
