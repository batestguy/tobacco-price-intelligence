"""The three role views of INTRO.txt §6, and the widgets they share.

Kept apart from ``streamlit_app.py`` so each view can be rendered on its own in
``tests/test_app_render.py`` without going through sign-in.

Every identifier on screen goes through ``labels``; nothing here should print a
raw column name or code. The test suite checks that.
"""

from __future__ import annotations

import data
import labels
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tobacco import config
from tobacco.memo import groq

#: Rows of history behind the sparklines in the metric row.
SPARK_DAYS = 30


# ---------------------------------------------------------------------------
# shared widgets
# ---------------------------------------------------------------------------


def render_disclaimer() -> None:
    """The §11 text, verbatim from ``config``, plus the portfolio notice."""
    st.space("large")
    st.caption(config.PORTFOLIO_NOTICE)
    st.caption(config.DISCLAIMER)


def render_hero(eyebrow: str, headline: str, detail: str) -> None:
    """A plain-language verdict in place of a big number card."""
    st.caption(eyebrow)
    st.title(headline, anchor=False)
    st.markdown(detail)


def _series(frame: pd.DataFrame, column: str, date_col: str = "date") -> list[float]:
    if frame.empty or column not in frame:
        return []
    return frame.sort_values(date_col)[column].dropna().tail(SPARK_DAYS).tolist()


def render_headline_metrics() -> None:
    rate, change, carried = data.latest_fx()
    sentiment, crisis = data.latest_sentiment()
    inflation, inflation_basis, inflation_help, inflation_caption = (
        data.latest_inflation()
    )
    rates = data.load("exchange_rates")
    aggregates = data.load("sentiment_aggregates")

    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Naira per US dollar",
                f"₦{rate:,.2f}" if rate else "—",
                f"{labels.signed_pct(change, 2)} over 7 days" if change is not None else None,
                # A weakening naira is bad news, so invert the default
                # green-for-up colouring.
                delta_color="inverse",
                help="Official CBN rate. A rise means the naira has weakened.",
                chart_data=_series(rates, "usd_ngn_rate"),
            )
            if carried:
                st.caption(":material/history: Carried forward: CBN was unreachable")

        with col2:
            st.metric(
                f"Inflation, {inflation_basis.lower()}" if inflation_basis else "Inflation",
                f"{inflation:.1f}%" if inflation is not None else "—",
                help=inflation_help or "Year-on-year consumer price inflation.",
            )
            if inflation_caption:
                st.caption(inflation_caption)

        col3.metric(
            "Consumer sentiment",
            f"{sentiment:.2f}" if sentiment is not None else "—",
            help=(
                "How positive public forum posts are, from 0 (very negative) to "
                "1 (very positive). Scored with VADER."
            ),
            chart_data=_series(aggregates, "consumer_sentiment"),
        )
        col4.metric(
            "News crisis score",
            f"{crisis:.2f}" if crisis is not None else "—",
            help=(
                "How likely today's financial headlines signal a currency crisis, "
                "from 0 (calm) to 1 (crisis). Scored with a fine-tuned FinBERT."
            ),
            chart_data=_series(aggregates, "fx_crisis_prob"),
            chart_type="area",
        )


def render_trend_chart() -> None:
    rates = data.load("exchange_rates")
    aggregates = data.load("sentiment_aggregates")

    st.subheader("Naira and news, last 180 days", anchor=False)
    if rates.empty:
        st.caption("No exchange-rate history yet. It is collected twice a day.")
        return

    rates = rates.sort_values("date").tail(180)
    figure = go.Figure()
    figure.add_scatter(
        x=rates["date"],
        y=rates["usd_ngn_rate"],
        name="Naira per US dollar",
        mode="lines",
        hovertemplate="%{x|%d %b %Y}<br>₦%{y:,.2f} per US dollar<extra></extra>",
    )

    if not aggregates.empty and "fx_crisis_prob" in aggregates:
        crisis = aggregates.dropna(subset=["fx_crisis_prob"])
        crisis = crisis[pd.to_datetime(crisis["date"]) >= pd.to_datetime(rates["date"]).min()]
        if not crisis.empty:
            figure.add_scatter(
                x=crisis["date"],
                y=crisis["fx_crisis_prob"],
                name="News crisis score",
                mode="lines",
                yaxis="y2",
                line=dict(dash="dot"),
                hovertemplate="%{x|%d %b %Y}<br>Crisis score %{y:.2f}<extra></extra>",
            )
            figure.update_layout(
                yaxis2=dict(
                    title=dict(text="News crisis score (0 to 1)"),
                    overlaying="y",
                    side="right",
                    range=[0, 1],
                    showgrid=False,
                )
            )

    figure.update_layout(
        xaxis=dict(title=dict(text="")),
        yaxis=dict(title=dict(text="Naira per US dollar"), tickprefix="₦"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=30, r=10, b=10, l=10),
        hovermode="x unified",
    )
    st.plotly_chart(figure, width="stretch")


def render_memo() -> None:
    memo, memo_date = data.latest_memo()
    with st.container(border=True):
        st.subheader("Strategic memo", anchor=False)
        if not memo:
            st.caption("No memo yet. One is written each morning at 08:00 WAT.")
            return
        if groq.FALLBACK_MARKER in memo:
            st.caption(
                f"Written {memo_date} without the language model: Groq was "
                f"unavailable, so this lists the underlying figures only."
            )
        else:
            st.caption(f"Written {memo_date} by {groq.MODEL} on Groq, from the figures on this page.")
        st.markdown(labels.relabel_codes(memo))


def _recommendation_table(recommendations: pd.DataFrame) -> pd.DataFrame:
    table = recommendations.sort_values(["sku", "region"]).copy()
    table["sku"] = table["sku"].map(labels.sku)
    table["binding_constraint"] = table["binding_constraint"].map(labels.constraint)
    # A list per cell, so MultiselectColumn can draw it as a coloured badge.
    table["inventory_action"] = table["inventory_action"].map(lambda status: [status])
    return table[
        ["sku", "region", "price_adjustment_pct", "recommended_price_ngn",
         "forecast_qty_4w", "inventory_action", "binding_constraint"]
    ]


STOCK_BADGE = st.column_config.MultiselectColumn(
    "Stock",
    options=list(labels.STOCK_STATUS_LABELS),
    color=["green", "red", "yellow"],
    format_func=labels.stock_status,
    help="Stock cover after rebalancing, against the safety stock and warehouse capacity.",
)

RECOMMENDATION_COLUMNS = {
    "sku": st.column_config.TextColumn("Product"),
    "region": st.column_config.TextColumn("Region"),
    "price_adjustment_pct": st.column_config.NumberColumn(
        "Price change", format="%+.1f%%",
        help="Recommended change from the current shelf price.",
    ),
    "recommended_price_ngn": st.column_config.NumberColumn(
        "Recommended price", format="₦%,.2f",
    ),
    "forecast_qty_4w": st.column_config.NumberColumn(
        "Demand, next 4 weeks", format="%,.0f packs",
        help="Forecast packs sold over the next four weeks.",
    ),
    "inventory_action": STOCK_BADGE,
    "binding_constraint": st.column_config.TextColumn(
        "Limiting factor",
        help="What stopped the price moving further.",
    ),
}


# ---------------------------------------------------------------------------
# views
# ---------------------------------------------------------------------------


def _day(value) -> str:
    """``26 Sep 2026``; ``%-d`` is glibc-only, so build the day by hand."""
    stamp = pd.to_datetime(value)
    return f"{stamp.day} {stamp:%b %Y}"


def _as_of(recommendations: pd.DataFrame) -> str:
    if recommendations.empty:
        return ""
    return _day(pd.to_datetime(recommendations["date"]).max())


def executive() -> None:
    recommendations = data.latest_recommendations()
    moves = {}
    if not recommendations.empty:
        per_sku = recommendations.groupby("sku").agg(
            adj=("price_adjustment_pct", "mean"),
            constraint=("binding_constraint", lambda s: s.mode().iloc[0]),
        )
        moves = {sku: (row.adj, row.constraint) for sku, row in per_sku.iterrows()}

    headline, detail = labels.price_verdict(moves)
    eyebrow = "Executive"
    if as_of := _as_of(recommendations):
        eyebrow += f" · recommendation of {as_of}"
    render_hero(eyebrow, headline, detail)

    render_headline_metrics()
    render_trend_chart()

    left, right = st.columns([2, 3], gap="medium")
    with left:
        render_memo()
    with right:
        st.subheader("Recommendations by product and region", anchor=False)
        if recommendations.empty:
            st.caption("No recommendation yet. One is generated each morning at 08:00 WAT.")
        else:
            st.dataframe(
                _recommendation_table(recommendations),
                column_config=RECOMMENDATION_COLUMNS,
                hide_index=True,
                width="stretch",
            )


def supply_chain() -> None:
    recommendations = data.latest_recommendations()
    statuses = {}
    if not recommendations.empty:
        statuses = {
            (row.sku, row.region): row.inventory_action
            for row in recommendations.itertuples()
        }
    headline, detail = labels.stock_verdict(statuses)
    eyebrow = "Supply chain"
    if as_of := _as_of(recommendations):
        eyebrow += f" · outlook of {as_of}"
    render_hero(eyebrow, headline, detail)

    render_headline_metrics()

    if recommendations.empty:
        return

    left, right = st.columns([3, 2], gap="medium")
    with left:
        st.subheader("Demand by product and region, next 4 weeks", anchor=False)
        pivot = recommendations.pivot_table(
            index="sku", columns="region", values="forecast_qty_4w", aggfunc="sum"
        )
        pivot.index = [labels.sku_short(code) for code in pivot.index]
        heatmap = go.Figure(
            go.Heatmap(
                z=pivot.values,
                x=list(pivot.columns),
                y=list(pivot.index),
                colorbar=dict(title=dict(text="Packs")),
                texttemplate="%{z:,.0f}",
                hovertemplate="%{y} in %{x}<br>%{z:,.0f} packs<extra></extra>",
            )
        )
        heatmap.update_layout(
            xaxis=dict(title=dict(text="")),
            yaxis=dict(title=dict(text=""), autorange="reversed"),
            margin=dict(t=10, r=10, b=10, l=10),
        )
        st.plotly_chart(heatmap, width="stretch")
        st.caption("Darker cells draw stock down fastest.")

    with right:
        st.subheader("Stock position", anchor=False)
        # Problems first: stockout risk, then overstock, then the rest.
        urgency = {"stockout_risk": 0, "overstock": 1}
        table = recommendations.assign(
            _urgency=recommendations["inventory_action"].map(urgency).fillna(2)
        ).sort_values(["_urgency", "sku", "region"])
        table["sku"] = table["sku"].map(labels.sku_short)
        table["inventory_action"] = table["inventory_action"].map(lambda s: [s])
        st.dataframe(
            table[["sku", "region", "inventory_action", "forecast_qty_4w"]],
            column_config={
                "sku": st.column_config.TextColumn("Product"),
                "region": st.column_config.TextColumn("Region"),
                "inventory_action": STOCK_BADGE,
                "forecast_qty_4w": RECOMMENDATION_COLUMNS["forecast_qty_4w"],
            },
            hide_index=True,
            width="stretch",
        )

    st.subheader("Weekly volume by product", anchor=False)
    sales = data.load("sales_mock")
    if sales.empty:
        st.caption("No sales history yet.")
        return
    recent = sales.sort_values("week_start").tail(52 * len(config.SKUS) * len(config.REGIONS))
    weekly = recent.groupby(["week_start", "sku"], as_index=False)["quantity_sold"].sum()
    volume = go.Figure()
    for code in config.SKUS:
        rows = weekly[weekly["sku"] == code]
        volume.add_scatter(
            x=rows["week_start"],
            y=rows["quantity_sold"],
            name=labels.sku_short(code),
            mode="lines",
            hovertemplate=(
                f"{labels.sku_short(code)}<br>Week of %{{x|%d %b %Y}}<br>"
                f"%{{y:,.0f}} packs<extra></extra>"
            ),
        )
    volume.update_layout(
        xaxis=dict(title=dict(text="")),
        yaxis=dict(title=dict(text="Packs sold per week")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=30, r=10, b=10, l=10),
    )
    st.plotly_chart(volume, width="stretch")
    st.caption("Sales are synthetic, generated for this demonstration. No company data is used.")


def admin() -> None:
    metrics = data.model_metrics()
    if metrics:
        gain = metrics["naive_baseline_mape_pct"] - metrics["mape_pct"]
        trained = _day(metrics["trained_at"])
        if gain > 0:
            headline = f"The demand forecast beats a naive guess by {gain:.1f} points"
        else:
            headline = f"The demand forecast trails a naive guess by {-gain:.1f} points"
        detail = (
            f"Average error {metrics['mape_pct']:.1f}% against "
            f"{metrics['naive_baseline_mape_pct']:.1f}% for repeating last week's "
            f"sales. Retrained {trained} on {metrics['n_train_rows']:,} weekly rows."
        )
    else:
        headline = "No demand model trained yet"
        detail = "The model is retrained every Sunday."
    render_hero("Administration", headline, detail)

    if metrics:
        with st.container(border=True):
            col1, col2, col3 = st.columns(3)
            col1.metric(
                labels.METRIC_LABELS["mape_pct"], f"{metrics['mape_pct']:.1f}%",
                help="Mean absolute percentage error on the last four held-out weeks.",
            )
            col2.metric(
                labels.METRIC_LABELS["rmse"], f"{metrics['rmse']:,.0f}",
                help="Root mean squared error, in packs per product, region and week.",
            )
            col3.metric(
                "Better than a naive forecast by",
                f"{gain:+.1f} points",
                help="Positive means the model beats simply repeating last week's sales.",
            )

        importances = pd.Series(metrics.get("top_features", {})).sort_values()
        if not importances.empty:
            st.subheader("What drives the forecast", anchor=False)
            bars = go.Figure(
                go.Bar(
                    x=importances.values,
                    y=[labels.feature(name) for name in importances.index],
                    orientation="h",
                    hovertemplate="%{y}<br>Importance %{x:.3f}<extra></extra>",
                )
            )
            bars.update_layout(
                xaxis=dict(title=dict(text="Share of the model's splits (gain)")),
                yaxis=dict(title=dict(text="")),
                margin=dict(t=10, r=10, b=10, l=10),
                height=max(260, 28 * len(importances)),
            )
            st.plotly_chart(bars, width="stretch")
            st.caption(
                "Sales are synthetic, so product and region dominate by design. "
                "Economic inputs cannot rank until there is real sales history."
            )

    st.subheader("Data freshness", anchor=False)
    rows = []
    for name in labels.DATASET_LABELS:
        frame = data.load(name)
        timestamp_col = next(
            (c for c in ("date", "published_at", "week_start") if c in frame.columns), None
        )
        latest = (
            pd.to_datetime(frame[timestamp_col]).max()
            if not frame.empty and timestamp_col else None
        )
        rows.append({"dataset": labels.dataset(name), "rows": len(frame), "latest": latest})
    st.dataframe(
        pd.DataFrame(rows),
        column_config={
            "dataset": st.column_config.TextColumn("Dataset"),
            "rows": st.column_config.NumberColumn("Rows", format="%,d"),
            "latest": st.column_config.DatetimeColumn("Latest record", format="D MMM YYYY"),
        },
        hide_index=True,
        width="stretch",
    )

    st.subheader("Users", anchor=False)
    st.caption(
        "Users are managed in the Supabase dashboard under Authentication, and "
        "their roles in the users table."
    )
