"""Business constants, and the elasticity derivation the optimizer depends on.

``ASSUMED_MARKET_SHARE`` is pre-registered as *not* to be re-tuned to escape a grid
bound (config.py:94), which only means something if the band it has to sit inside
is asserted somewhere. That is what most of this file is.

The band is derived here via ``conftest.interior_share_band`` rather than compared
against the literals in the config comment, because those literals are quoted at an
FX-scaled observed price and drift with the naira. See the helper's docstring.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from tobacco import config
from tobacco.optimize import linprog

from conftest import interior_share_band


# ---------------------------------------------------------------------------
# the interior-optimum band
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sku", config.SKUS)
def test_every_sku_interior_optimum_is_strictly_inside_the_price_grid(sku):
    base_price = config.BASE_PRICE_NGN[sku]
    optimum = linprog._interior_optimum(config.UNIT_COST_NGN[sku], config.PRICE_ELASTICITY[sku])

    assert optimum is not None, f"{sku} is inelastic; no optimum exists to be inside anything"
    assert base_price * (1 + linprog.PRICE_GRID.min()) < optimum
    assert optimum < base_price * (1 + linprog.PRICE_GRID.max())


def test_assumed_market_share_is_strictly_inside_the_derived_interior_band():
    """The share must be interior for *every* SKU, i.e. inside the intersection."""
    bands = {
        sku: interior_share_band(config.UNIT_COST_NGN[sku], config.BASE_PRICE_NGN[sku])
        for sku in config.SKUS
    }
    lower = max(low for low, _ in bands.values())
    upper = min(high for _, high in bands.values())

    assert lower < upper, f"no share is interior for all SKUs: {bands}"
    assert lower < config.ASSUMED_MARKET_SHARE < upper, (
        f"ASSUMED_MARKET_SHARE={config.ASSUMED_MARKET_SHARE} is outside the "
        f"intersection ({lower:.4f}, {upper:.4f}); per-SKU bands {bands}"
    )


def _decisions_at_share(share, forecast_frame, monkeypatch):
    """Re-derive the firm elasticity at ``share`` and optimise the real portfolio."""
    monkeypatch.setattr(
        config,
        "PRICE_ELASTICITY",
        {sku: config.CATEGORY_PRICE_ELASTICITY / share for sku in config.SKUS},
    )
    forecast = pd.concat(
        [forecast_frame(sku, config.BASE_PRICE_NGN[sku]) for sku in config.SKUS],
        ignore_index=True,
    )
    decisions, _ = linprog.optimise_prices(forecast, competitor_avg=None)
    return decisions


def test_share_below_the_band_reports_the_lower_grid_bound(forecast_frame, monkeypatch):
    """Below the band the honest answer is "the grid stopped me", not a price."""
    decisions = _decisions_at_share(0.15, forecast_frame, monkeypatch)

    assert len(decisions) == len(config.SKUS)
    assert {d.binding_constraint for d in decisions} == {"price_grid_lower_bound"}
    assert {d.adjustment_pct for d in decisions} == {-10.0}


def test_share_above_the_band_reports_the_upper_grid_bound(forecast_frame, monkeypatch):
    decisions = _decisions_at_share(0.40, forecast_frame, monkeypatch)

    assert len(decisions) == len(config.SKUS)
    assert {d.binding_constraint for d in decisions} == {"price_grid_upper_bound"}
    assert {d.adjustment_pct for d in decisions} == {25.0}


def test_firm_elasticity_is_derived_from_the_category_figure():
    """Derived in code (config.py:113) so the arithmetic cannot drift from the citation."""
    expected = config.CATEGORY_PRICE_ELASTICITY / config.ASSUMED_MARKET_SHARE

    assert set(config.PRICE_ELASTICITY) == set(config.SKUS)
    assert set(config.PRICE_ELASTICITY.values()) == {expected}, "uniform across tiers on purpose"
    assert expected < config.CATEGORY_PRICE_ELASTICITY, "one seller faces more elastic demand"


def test_firm_elasticity_is_elastic_enough_for_an_optimum_to_exist():
    """|e| <= 1 would make every recommendation a grid edge -- see _interior_optimum."""
    assert all(abs(e) > 1.0 for e in config.PRICE_ELASTICITY.values())


# ---------------------------------------------------------------------------
# holiday dummies (INTRO.txt §4)
# ---------------------------------------------------------------------------


def test_is_holiday_week_christmas():
    assert config.is_holiday_week(date(2025, 12, 22)) is True


def test_is_holiday_week_new_year():
    # 29 Dec .. 4 Jan: the holiday falls in the *following* calendar year.
    assert config.is_holiday_week(date(2025, 12, 29)) is True


def test_is_holiday_week_sallah():
    """Sallah shifts ~11 days a year, so it is a table lookup rather than a rule."""
    assert date(2026, 5, 27) in config.SALLAH_DATES
    assert config.is_holiday_week(date(2026, 5, 25)) is True  # 25..31 May covers it


def test_is_holiday_week_ordinary_week():
    assert config.is_holiday_week(date(2026, 2, 2)) is False


def test_is_holiday_week_covers_seven_days_not_eight():
    """The window is [week_start, week_start+6]; day 7 belongs to the next week."""
    assert config.is_holiday_week(date(2025, 12, 18)) is True   # 18..24 catches the 24th
    assert config.is_holiday_week(date(2025, 12, 17)) is False  # 17..23 does not


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------


def test_optional_treats_blank_env_as_absent(monkeypatch):
    """Actions injects an *unset* secret as an empty string, not as an absent key."""
    monkeypatch.setenv("GROQ_API_KEY", "")
    assert config.optional("GROQ_API_KEY") == ""
    assert config.optional("GROQ_API_KEY", "fallback") == "fallback"


def test_optional_treats_whitespace_as_absent(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "   \t\n ")
    assert config.optional("GROQ_API_KEY", "fallback") == "fallback"


def test_optional_strips_a_real_value(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "  gsk_abc  ")
    assert config.optional("GROQ_API_KEY") == "gsk_abc"


def test_require_raises_missing_secret_naming_the_variable(monkeypatch):
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "")
    with pytest.raises(config.MissingSecret, match="GMAIL_APP_PASSWORD"):
        config.require("GMAIL_APP_PASSWORD")


def test_require_all_reports_every_missing_variable_at_once():
    with pytest.raises(config.MissingSecret) as excinfo:
        config.require_all("GMAIL_ADDRESS", "GMAIL_APP_PASSWORD")
    assert "GMAIL_ADDRESS" in str(excinfo.value)
    assert "GMAIL_APP_PASSWORD" in str(excinfo.value)


def test_recipients_parses_and_strips_a_comma_list(monkeypatch):
    monkeypatch.setenv("ALERT_RECIPIENTS_COMMERCIAL", " a@example.com , b@example.com ,, ")
    assert config.recipients("commercial") == ["a@example.com", "b@example.com"]


def test_recipients_is_empty_when_the_variable_is_unset():
    assert config.recipients("commercial") == []
    assert config.recipients("supply") == []


def test_finbert_model_falls_back_to_the_public_checkpoint():
    assert config.finbert_model() == "ProsusAI/finbert"
