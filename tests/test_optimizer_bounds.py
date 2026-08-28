"""Stage 1 of the optimizer: the interior optimum and every binding-constraint branch.

``binding_constraint`` is the single line the memo puts in front of the Commercial
Director, and the chain that computes it (linprog.py:229-248) was rewritten to fix
the +25% saturation. Each branch below is reached by *stating* the economics that
reach it -- one synthetic SKU with a chosen cost, price and elasticity -- rather
than by finding a scenario in which one of the real three happens to land there.

Each scenario was derived from the closed form ``p* = c·e/(e−1)`` against the grid
bounds ``0.90·p0`` and ``1.25·p0``; the comment on each case says which side of
which bound puts it in that branch.
"""

from __future__ import annotations

import numpy as np
import pytest

from tobacco import config
from tobacco.optimize import linprog


# ---------------------------------------------------------------------------
# _interior_optimum
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("elasticity", [-0.62, -0.99, -1.0, -1.0 + 1e-12, 0.0])
def test_interior_optimum_is_none_when_demand_is_inelastic(elasticity):
    """|e| <= 1: profit rises without bound, so no grid can contain an optimum."""
    assert linprog._interior_optimum(600.0, elasticity) is None


def test_interior_optimum_is_the_lerner_markup():
    e = 2.48
    assert linprog._interior_optimum(600.0, -e) == pytest.approx(600.0 * e / (e - 1.0))


def test_interior_optimum_ignores_the_sign_of_the_elasticity():
    assert linprog._interior_optimum(600.0, -2.48) == linprog._interior_optimum(600.0, 2.48)


def test_interior_optimum_approaches_unit_cost_as_demand_gets_more_elastic():
    """The markup is 1/(e−1) over cost, so a very elastic buyer prices at cost."""
    assert linprog._interior_optimum(600.0, -1000.0) == pytest.approx(600.0, rel=1e-2)
    assert linprog._interior_optimum(600.0, -2.0) == pytest.approx(1200.0)


# ---------------------------------------------------------------------------
# the binding-constraint chain
# ---------------------------------------------------------------------------

#: (branch, unit_cost, base_price, elasticity, competitor_avg). Grid runs
#: 0.90·p0 .. 1.25·p0; the margin floor is 1.10·c; the ceiling is 1.05·competitor.
SCENARIOS = [
    # p* = 1005 -- inside 900..1250, floor 660 slack, no ceiling.
    ("profit_optimum", 600.0, 1000.0, -2.48, None),
    # p* = 1508 -- beyond the 1250 grid top; floor 990 does not bind the choice.
    ("price_grid_upper_bound", 900.0, 1000.0, -2.48, None),
    # p* = 800 -- below the 900 grid bottom, and the floor at 660 is slack, so
    # the grid and not the margin is what stopped it.
    ("price_grid_lower_bound", 600.0, 1000.0, -4.0, None),
    # p* = 1091 but the floor at 1100 sits above the 1035 grid bottom and cuts in.
    ("margin_floor", 1000.0, 1150.0, -12.0, None),
    # p* = 1005 but the ceiling at 945 truncates the grid before the optimum.
    ("competitor_ceiling", 600.0, 1000.0, -2.48, 900.0),
    # |e| < 1: no optimum exists at all, whatever the grid does.
    ("no_interior_optimum", 600.0, 1000.0, -0.62, None),
    # floor 1100 above ceiling 840: the feasible set is empty.
    ("infeasible_floor_above_ceiling", 1000.0, 1150.0, -2.48, 800.0),
]


@pytest.mark.parametrize(
    "expected,unit_cost,base_price,elasticity,competitor_avg",
    SCENARIOS,
    ids=[s[0] for s in SCENARIOS],
)
def test_binding_constraint_branch(
    expected, unit_cost, base_price, elasticity, competitor_avg,
    single_sku, forecast_frame,
):
    sku = single_sku(unit_cost, base_price, elasticity)
    forecast = forecast_frame(sku, assumed_price_ngn=base_price)

    decisions, _ = linprog.optimise_prices(forecast, competitor_avg)

    assert len(decisions) == 1
    assert decisions[0].binding_constraint == expected


def test_profit_optimum_lands_strictly_inside_the_grid(single_sku, forecast_frame):
    """The regression the +25% saturation fix was about: a real interior answer."""
    sku = single_sku(600.0, 1000.0, -2.48)
    decisions, _ = linprog.optimise_prices(forecast_frame(sku, 1000.0), None)

    (decision,) = decisions
    assert 900.0 < decision.recommended_price < 1250.0
    assert -10.0 < decision.adjustment_pct < 25.0


def test_infeasible_holds_the_current_price_rather_than_violating_a_constraint(
    single_sku, forecast_frame
):
    sku = single_sku(1000.0, 1150.0, -2.48)
    decisions, notes = linprog.optimise_prices(forecast_frame(sku, 1150.0), competitor_avg=800.0)

    (decision,) = decisions
    assert decision.recommended_price == decision.current_price == 1150.0
    assert decision.adjustment_pct == 0.0
    assert any("infeasible" in note for note in notes)


def test_grid_bound_branches_say_so_in_the_notes(single_sku, forecast_frame):
    """A grid edge must be reported as an edge, not passed off as an optimum."""
    sku = single_sku(900.0, 1000.0, -2.48)
    _, notes = linprog.optimise_prices(forecast_frame(sku, 1000.0), None)
    assert any("outside the" in note and "price grid" in note for note in notes)


def test_inelastic_demand_is_noted_even_when_another_constraint_takes_the_label(
    single_sku, forecast_frame
):
    """Why the optimum is computed outside the binding chain (linprog.py:140-153).

    With a ceiling active the chain still labels this `competitor_ceiling` -- that
    branch comes first -- which on its own would be a plausible-looking label on a
    meaningless number. The unconditional note is what stops it reading that way,
    so the note, not the label, is the invariant.
    """
    sku = single_sku(600.0, 1000.0, -0.62)
    decisions, notes = linprog.optimise_prices(forecast_frame(sku, 1000.0), competitor_avg=900.0)

    (decision,) = decisions
    assert decision.binding_constraint == "competitor_ceiling"
    assert decision.unconstrained_optimum_ngn is None
    assert any("inelastic" in note for note in notes)


# ---------------------------------------------------------------------------
# the competitor ceiling stays INACTIVE without competitor data
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("competitor_avg", [None, float("nan"), np.float64("nan")])
def test_absent_competitor_price_leaves_the_ceiling_inactive(
    competitor_avg, single_sku, forecast_frame
):
    """CLAUDE.md departure 4: there must be no invented fallback ceiling.

    ``competitor_prices`` is header-only until someone records a citable survey,
    so this is the *normal* path today, not an edge case.
    """
    sku = single_sku(600.0, 1000.0, -2.48)
    decisions, notes = linprog.optimise_prices(forecast_frame(sku, 1000.0), competitor_avg)

    (decision,) = decisions
    assert decision.competitor_ceiling is None
    assert decision.binding_constraint != "competitor_ceiling"
    assert any("INACTIVE" in note for note in notes)


def test_an_observed_competitor_price_reports_the_ceiling_it_implies(
    single_sku, forecast_frame
):
    sku = single_sku(600.0, 1000.0, -2.48)
    decisions, notes = linprog.optimise_prices(forecast_frame(sku, 1000.0), competitor_avg=900.0)

    (decision,) = decisions
    assert decision.competitor_ceiling == pytest.approx(
        900.0 * (1 + config.MAX_PREMIUM_OVER_COMPETITOR)
    )
    assert not any("INACTIVE" in note for note in notes)


def test_margin_floor_is_reported_whether_or_not_it_binds(single_sku, forecast_frame):
    sku = single_sku(600.0, 1000.0, -2.48)
    decisions, _ = linprog.optimise_prices(forecast_frame(sku, 1000.0), None)

    (decision,) = decisions
    assert decision.margin_floor == pytest.approx(600.0 * (1 + config.MIN_MARGIN_OVER_COST))
    assert decision.binding_constraint != "margin_floor"


def test_no_forecast_rows_for_a_sku_yields_no_decision(single_sku, forecast_frame):
    sku = single_sku(600.0, 1000.0, -2.48)
    forecast = forecast_frame(sku, 1000.0)
    decisions, _ = linprog.optimise_prices(forecast[forecast["sku"] != sku], None)
    assert decisions == []
