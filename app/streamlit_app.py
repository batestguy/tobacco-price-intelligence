"""Role-based dashboard (INTRO.txt §6).

Streamlit rather than the spec's Plotly Dash: Dash would need a host that accepts
manual uploads, and Streamlit Community Cloud redeploys straight from this
repository on every push. Streamlit is the spec's own §6 option C.

The §11 disclaimer is rendered verbatim on the login page and in the footer of
every view, from the single copy in ``tobacco.config.DISCLAIMER``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Must precede the `tobacco` import: the package is not installed, it is read
# from src/ in the checked-out repo. Doing this here rather than relying on
# app.data's side effect keeps the import order safe to reformat.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import auth  # noqa: E402
import data  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from tobacco import config  # noqa: E402

st.set_page_config(
    page_title="Price Intelligence & Supply Chain",
    page_icon="📊",
    layout="wide",
)


# ---------------------------------------------------------------------------
# chrome
# ---------------------------------------------------------------------------


def render_disclaimer() -> None:
    st.divider()
    st.caption(config.PORTFOLIO_NOTICE)
    st.caption(config.DISCLAIMER)


def render_login() -> None:
    st.title("Price Intelligence & Supply Chain Optimization")
    st.caption("Sign in to continue.")

    if not auth.configured():
        st.error(
            "Authentication is not configured. Set `SUPABASE_URL` and "
            "`SUPABASE_ANON_KEY` in **App settings → Secrets** on Streamlit "
            "Cloud. These are set separately from GitHub Actions secrets, and "
            "the app must receive the **anon** key, never the service key."
        )
        render_disclaimer()
        return

    with st.form("login"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")

    if submitted:
        ok, message = auth.sign_in(email, password)
        if ok:
            st.rerun()
        else:
            st.error(message)

    render_disclaimer()


# ---------------------------------------------------------------------------
# shared widgets
# ---------------------------------------------------------------------------


def render_headline_metrics() -> None:
    rate, change, carried = data.latest_fx()
    sentiment, crisis = data.latest_sentiment()
    inflation, inflation_basis, inflation_help, inflation_caption = (
        data.latest_inflation()
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "NGN/USD",
            f"{rate:,.2f}" if rate else "—",
            f"{change:+.2f}% (7d)" if change is not None else None,
            # A weakening naira is bad news, so invert Streamlit's default
            # green-for-up colouring.
            delta_color="inverse",
        )
        if carried:
            st.caption("⚠️ Carried forward — CBN was unreachable")

    with col2:
        st.metric(
            f"Inflation ({inflation_basis.lower()})" if inflation_basis else "Inflation",
            f"{inflation:.2f}%" if inflation is not None else "—",
            help=inflation_help,
        )
        if inflation_caption:
            st.caption(inflation_caption)

    col3.metric(
        "Consumer sentiment",
        f"{sentiment:.2f}" if sentiment is not None else "—",
        help="0 = very negative, 1 = very positive (VADER on public forum posts)",
    )
    col4.metric(
        "News crisis probability",
        f"{crisis:.2f}" if crisis is not None else "—",
        help="0 = low, 1 = high (FinBERT on financial headlines)",
    )


def render_trend_chart() -> None:
    rates = data.load("exchange_rates")
    aggregates = data.load("sentiment_aggregates")

    if rates.empty:
        st.info("No FX history yet. Run the scrape workflow to populate it.")
        return

    rates = rates.sort_values("date").tail(180)
    figure = px.line(
        rates, x="date", y="usd_ngn_rate",
        title="NGN/USD — last 180 days", labels={"usd_ngn_rate": "NGN per USD", "date": ""},
    )

    if not aggregates.empty and "fx_crisis_prob" in aggregates:
        crisis = aggregates.dropna(subset=["fx_crisis_prob"])
        if not crisis.empty:
            figure.add_scatter(
                x=crisis["date"],
                y=crisis["fx_crisis_prob"],
                name="News crisis probability",
                yaxis="y2",
                line=dict(dash="dot"),
            )
            figure.update_layout(
                yaxis2=dict(title="Crisis probability", overlaying="y",
                            side="right", range=[0, 1])
            )

    st.plotly_chart(figure, use_container_width=True)


# ---------------------------------------------------------------------------
# views
# ---------------------------------------------------------------------------


def executive_view() -> None:
    st.subheader("Executive dashboard")
    render_headline_metrics()
    st.divider()

    recommendations = data.latest_recommendations()

    left, right = st.columns([1, 2])

    with left:
        if recommendations.empty:
            st.metric("Recommended price adjustment", "—")
            st.caption("No recommendation yet — run the recommend workflow.")
        else:
            overall = recommendations["price_adjustment_pct"].mean()
            st.metric("Recommended price adjustment", f"{overall:+.2f}%")
            st.caption(f"As of {pd.to_datetime(recommendations['date']).max().date()}")

            binding = recommendations["binding_constraint"].value_counts()
            if not binding.empty:
                st.caption(f"Limiting factor: **{binding.index[0]}**")

    with right:
        render_trend_chart()

    st.divider()
    st.markdown("#### Strategic memo")
    memo, memo_date = data.latest_memo()
    if memo:
        st.caption(f"Generated {memo_date} by Llama 3.3 70B from the figures above.")
        with st.expander("Read memo", expanded=True):
            st.markdown(memo)
    else:
        st.info(
            "No memo yet. The recommend workflow writes one to `data/memos/` "
            "each morning."
        )

    if not recommendations.empty:
        st.divider()
        st.markdown("#### Recommendations by SKU and region")
        st.dataframe(
            recommendations[
                ["sku", "region", "price_adjustment_pct", "recommended_price_ngn",
                 "forecast_qty_4w", "inventory_action", "binding_constraint"]
            ],
            use_container_width=True,
            hide_index=True,
        )


def supply_chain_view() -> None:
    st.subheader("Supply chain")
    render_headline_metrics()
    st.divider()

    recommendations = data.latest_recommendations()
    if recommendations.empty:
        st.info("No recommendations yet — run the recommend workflow.")
        render_disclaimer()
        return

    st.markdown("#### Inventory risk by SKU and region")
    pivot = recommendations.pivot_table(
        index="sku", columns="region", values="forecast_qty_4w", aggfunc="sum"
    )
    st.plotly_chart(
        px.imshow(
            pivot,
            color_continuous_scale="RdYlGn_r",
            labels=dict(color="4-week forecast demand"),
            title="Forecast demand — darker red is higher draw-down pressure",
            aspect="auto",
        ),
        use_container_width=True,
    )

    at_risk = recommendations[recommendations["inventory_action"] != "ok"]
    st.markdown("#### Stock alerts")
    if at_risk.empty:
        st.success("All SKU/region combinations are within safety and capacity bounds.")
    else:
        st.warning(f"{len(at_risk)} combination(s) outside bounds.")
        st.dataframe(
            at_risk[["sku", "region", "inventory_action", "forecast_qty_4w"]],
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("#### Demand forecast")
    sales = data.load("sales_mock")
    if not sales.empty:
        recent = sales.sort_values("week_start").tail(52 * len(config.SKUS))
        weekly = recent.groupby(["week_start", "sku"], as_index=False)["quantity_sold"].sum()
        st.plotly_chart(
            px.line(
                weekly, x="week_start", y="quantity_sold", color="sku",
                title="Weekly volume by SKU (synthetic history)",
                labels={"quantity_sold": "Packs", "week_start": ""},
            ),
            use_container_width=True,
        )
        st.caption(
            "Sales figures are synthetic, generated by "
            "`src/tobacco/sources/sales_mock.py`."
        )


def admin_view() -> None:
    st.subheader("Administration")

    st.markdown("#### Model")
    metrics = data.model_metrics()
    if not metrics:
        st.info("No model trained yet — run the train workflow.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("MAPE", f"{metrics['mape_pct']:.2f}%")
        col2.metric("RMSE", f"{metrics['rmse']:,.0f}")
        col3.metric(
            "vs naive baseline",
            f"{metrics['naive_baseline_mape_pct'] - metrics['mape_pct']:+.2f} pp",
            help="Positive means the model beats a last-value forecast.",
        )
        st.caption(f"Trained {metrics['trained_at'][:16]} on {metrics['n_train_rows']:,} rows")
        with st.expander("Feature importances"):
            st.bar_chart(pd.Series(metrics.get("top_features", {})))

    st.markdown("#### Dataset freshness")
    rows = []
    for name in ("exchange_rates", "inflation", "competitor_prices",
                 "news_articles", "social_posts", "sentiment_aggregates",
                 "sales_mock", "recommendations"):
        frame = data.load(name)
        timestamp_col = None
        for candidate in ("date", "published_at", "week_start"):
            if candidate in frame.columns:
                timestamp_col = candidate
                break
        rows.append(
            {
                "dataset": name,
                "rows": len(frame),
                "latest": (
                    str(pd.to_datetime(frame[timestamp_col]).max().date())
                    if not frame.empty and timestamp_col else "—"
                ),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("#### User management")
    st.info(
        "Users are managed in the Supabase dashboard (**Authentication → Users**). "
        "Assign a role by inserting into the `users` table — see the note at the "
        "end of `supabase/schema.sql`."
    )


# ---------------------------------------------------------------------------
# entrypoint
# ---------------------------------------------------------------------------

VIEWS = {
    "commercial_director": [("Executive dashboard", executive_view)],
    "supply_chain_manager": [("Supply chain", supply_chain_view)],
    "admin": [
        ("Executive dashboard", executive_view),
        ("Supply chain", supply_chain_view),
        ("Administration", admin_view),
    ],
}


def main() -> None:
    user = auth.current_user()
    if user is None:
        render_login()
        return

    role = user["role"]
    available = VIEWS.get(role, VIEWS["commercial_director"])

    with st.sidebar:
        st.markdown(f"**{user['email']}**")
        st.caption(auth.ROLE_LABELS.get(role, role))
        st.divider()
        labels = [label for label, _ in available]
        # Route guarding is by construction: a role's views are the only ones
        # reachable. That is presentation, not access control -- the underlying
        # data is committed to a public repo (see app/auth.py).
        chosen = st.radio("View", labels, label_visibility="collapsed") if len(labels) > 1 else labels[0]
        st.divider()
        if st.button("Sign out"):
            auth.sign_out()
            st.rerun()

    st.title("Price Intelligence & Supply Chain Optimization")
    dict(available)[chosen]()
    render_disclaimer()


main()
