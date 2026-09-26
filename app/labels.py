"""The dashboard's display vocabulary: every identifier a viewer might see.

Pure Python with no Streamlit import, so it is unit-testable offline and
``tests/test_labels.py`` can check that every identifier the pipeline emits has
an explicit entry here. ``humanize`` is the fallback for anything missed; the
test fails if a known identifier ever reaches it.

Nothing here changes stored data. The Parquet keeps its identifiers; only the
screen gets words.
"""

from __future__ import annotations

from collections.abc import Mapping

# --------------------------------------------------------------------------
# vocabulary
# --------------------------------------------------------------------------

ROLE_LABELS = {
    "commercial_director": "Commercial Director",
    "supply_chain_manager": "Supply Chain Manager",
    "admin": "Administrator",
}

#: A signed-in user whose ``users.role`` is null: pending, no view.
NO_ROLE_LABEL = "No access"

#: ``config.SKUS``. The short form is for chart legends and feature names.
SKU_LABELS = {
    "PREMIUM_20": "Premium (pack of 20)",
    "MIDRANGE_20": "Mid-range (pack of 20)",
    "VALUE_20": "Value (pack of 20)",
}
SKU_SHORT = {
    "PREMIUM_20": "Premium",
    "MIDRANGE_20": "Mid-range",
    "VALUE_20": "Value",
}

#: ``config.REGIONS`` are already display names; listed so the test can prove it.
REGION_LABELS = {
    "Lagos": "Lagos",
    "Ibadan": "Ibadan",
    "Kano": "Kano",
    "Port Harcourt": "Port Harcourt",
}

#: ``optimize/linprog.py`` stock statuses, surfaced as ``inventory_action``.
STOCK_STATUS_LABELS = {
    "ok": "Within bounds",
    "stockout_risk": "Stockout risk",
    "overstock": "Overstock",
}

#: Every ``binding_constraint`` literal in ``optimize/linprog.py`` and the
#: ``"none"`` that ``build_recommendations`` writes when no decision exists.
CONSTRAINT_LABELS = {
    "profit_optimum": "Profit-maximising price",
    "price_grid_upper_bound": "Maximum allowed increase",
    "price_grid_lower_bound": "Maximum allowed decrease",
    "competitor_ceiling": "Competitor price ceiling",
    "margin_floor": "Minimum margin floor",
    "no_interior_optimum": "No profit peak in range",
    "infeasible_floor_above_ceiling": "Margin floor above competitor ceiling",
    "none": "No decision",
}

#: The same constraints as the clause that finishes a verdict sentence.
CONSTRAINT_REASONS = {
    "profit_optimum": "at its profit-maximising price",
    "price_grid_upper_bound": "capped by the maximum allowed increase",
    "price_grid_lower_bound": "held at the maximum allowed decrease",
    "competitor_ceiling": "capped by the competitor price ceiling",
    "margin_floor": "held up by the minimum margin floor",
    "no_interior_optimum": "left unchanged, because profit has no peak in the allowed range",
    "infeasible_floor_above_ceiling": (
        "left unchanged, because the margin floor sits above the competitor ceiling"
    ),
    "none": "left unchanged, because no pricing decision was made",
}

#: ``store/parquet_io.py`` ``DATASETS`` keys.
DATASET_LABELS = {
    "exchange_rates": "Exchange rates",
    "inflation": "Inflation",
    "competitor_prices": "Competitor prices",
    "news_articles": "News headlines",
    "social_posts": "Social posts",
    "sentiment_aggregates": "Daily sentiment",
    "sales_mock": "Sales (synthetic)",
    "recommendations": "Recommendations",
}

METRIC_LABELS = {
    "mape_pct": "Average forecast error (%)",
    "rmse": "Typical forecast error (packs)",
}


def _plural(n: int, unit: str) -> str:
    return f"{n} {unit}" if n == 1 else f"{n} {unit}s"


#: Model inputs from ``features/build.py``. The lags mirror ``FX_LAGS`` and
#: ``SENTIMENT_LAGS``; the test builds real features to catch drift.
FEATURE_LABELS = {
    **{f"fx_lag_{d}d": f"Exchange rate, {_plural(d, 'day')} earlier" for d in (1, 7, 30)},
    "fx_change_7d_pct": "Exchange rate change over 7 days (%)",
    "inflation": "Inflation rate",
    "competitor_index": "Competitor price index",
    "tax_change": "Recent tax news",
    **{f"crisis_lag_{d}d": f"News crisis score, {_plural(d, 'day')} earlier" for d in (1, 3)},
    **{f"sentiment_lag_{d}d": f"Consumer sentiment, {_plural(d, 'day')} earlier"
       for d in (1, 3)},
    "holiday_week": "Holiday week",
    "week_of_year": "Week of the year",
    "month": "Month",
    "own_price_lag_1w": "Own price, 1 week earlier",
    "own_price_lag_4w": "Own price, 4 weeks earlier",
    "price_vs_competitor": "Own price relative to competitors",
    **{f"sku_{sku}": f"Product: {short}" for sku, short in SKU_SHORT.items()},
    **{f"region_{region.replace(' ', '_')}": f"Region: {region}" for region in REGION_LABELS},
}


# --------------------------------------------------------------------------
# lookups
# --------------------------------------------------------------------------


def humanize(identifier: str) -> str:
    """Fallback for an identifier with no explicit label: ``foo_bar`` -> ``Foo bar``."""
    text = str(identifier).replace("_", " ").strip()
    return text[:1].upper() + text[1:] if text else text


def _lookup(table: Mapping[str, str], key) -> str:
    return table.get(key) or humanize(key)


def sku(code: str) -> str:
    return _lookup(SKU_LABELS, code)


def sku_short(code: str) -> str:
    return _lookup(SKU_SHORT, code)


def stock_status(code: str) -> str:
    return _lookup(STOCK_STATUS_LABELS, code)


def constraint(code: str) -> str:
    return _lookup(CONSTRAINT_LABELS, code)


def dataset(name: str) -> str:
    return _lookup(DATASET_LABELS, name)


def feature(name: str) -> str:
    return _lookup(FEATURE_LABELS, name)


def role(code: str | None) -> str:
    return NO_ROLE_LABEL if code is None else _lookup(ROLE_LABELS, code)


# --------------------------------------------------------------------------
# formatting and verdicts
# --------------------------------------------------------------------------

MINUS = "−"


def signed_pct(value: float, digits: int = 1) -> str:
    """``-1.333`` -> ``−1.3%`` with a true minus sign; ``0`` -> ``0.0%``."""
    rounded = round(value, digits)
    if rounded == 0:
        return f"{0:.{digits}f}%"
    sign = "+" if rounded > 0 else MINUS
    return f"{sign}{abs(rounded):.{digits}f}%"


def price_verdict(moves: Mapping[str, tuple[float, str]]) -> tuple[str, str]:
    """The executive hero: ``(headline, detail)`` from ``{sku: (adj %, constraint)}``.

    Written as a sentence a director could read aloud, not a number card.
    """
    if not moves:
        return "No price recommendation yet", (
            "One is generated each morning at 08:00 WAT."
        )

    adjustments = {code: adj for code, (adj, _) in moves.items()}
    held = [code for code, adj in adjustments.items() if round(adj, 1) == 0]
    up = [code for code, adj in adjustments.items() if round(adj, 1) > 0]
    down = [code for code, adj in adjustments.items() if round(adj, 1) < 0]

    if len(held) == len(moves):
        headline = "Hold prices this week"
    elif up and not down:
        headline = "Raise prices this week"
    elif down and not up:
        headline = "Lower prices this week"
    else:
        headline = "Adjust prices by product this week"

    # Largest move first, so the sentence leads with what matters most.
    ordered = sorted(moves, key=lambda code: -abs(adjustments[code]))
    parts = [f"{sku_short(code)} {signed_pct(adjustments[code])}" for code in ordered]
    average = sum(adjustments.values()) / len(adjustments)
    detail = f"{', '.join(parts)} ({signed_pct(average)} on average). "

    constraints = {c for _, c in moves.values()}
    if len(constraints) == 1:
        (only,) = constraints
        subject = "Each product is" if len(moves) > 1 else "The price is"
        detail += f"{subject} {CONSTRAINT_REASONS.get(only, humanize(only).lower())}."
    else:
        reasons = [
            f"{sku_short(code)} is {CONSTRAINT_REASONS.get(moves[code][1], humanize(moves[code][1]).lower())}"
            for code in ordered
        ]
        detail += "; ".join(reasons) + "."
    return headline, detail


def stock_verdict(statuses: Mapping[tuple[str, str], str]) -> tuple[str, str]:
    """The supply-chain hero from ``{(sku, region): inventory_action}``."""
    if not statuses:
        return "No stock outlook yet", "One is generated each morning at 08:00 WAT."

    total = len(statuses)
    flagged = {key: status for key, status in statuses.items() if status != "ok"}
    if not flagged:
        return (
            "Stock is within bounds everywhere",
            f"All {total} product and region combinations hold between their safety "
            f"stock and warehouse capacity after this week's rebalancing.",
        )

    stockouts = [k for k, s in flagged.items() if s == "stockout_risk"]
    overstocks = [k for k, s in flagged.items() if s == "overstock"]
    headline = (
        f"{len(stockouts)} of {total} at risk of running out"
        if stockouts
        else f"{len(overstocks)} of {total} overstocked"
    )

    def names(keys):
        return ", ".join(f"{sku_short(s)} in {r}" for s, r in sorted(keys))

    sentences = []
    if stockouts:
        sentences.append(f"Stockout risk: {names(stockouts)}.")
    if overstocks:
        sentences.append(f"Overstock: {names(overstocks)}.")
    return headline, " ".join(sentences)


def relabel_codes(text: str) -> str:
    """Replace SKU codes in free text (the LLM memo) with their short names."""
    for code, short in SKU_SHORT.items():
        text = text.replace(code, short)
    return text
