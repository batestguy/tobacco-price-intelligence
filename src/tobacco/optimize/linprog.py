"""Pricing and inventory rebalancing via SciPy ``linprog`` (INTRO.txt §5).

The spec asks for linear programming, but the stated objective
``max Σ(price × demand) − Σ(holding cost)`` is **not** linear once demand
responds to price: revenue becomes ``p × D(p)``, which is quadratic. Rather than
quietly linearising it away (and losing the elasticity that makes the answer
interesting), the problem is split into two genuine LPs:

**Stage 1 -- price selection.** Each SKU gets a grid of candidate prices. Profit
at each candidate is computed exactly, elasticity and all, *outside* the LP. The
LP then chooses convex weights ``x[i,k] ≥ 0`` with ``Σ_k x[i,k] = 1`` maximising
``Σ x[i,k] · profit[i,k]``. With only those per-SKU simplex constraints, an
optimal vertex puts all weight on one candidate, so the relaxation returns a
single real price -- no rounding, no integer solver. Infeasible candidates are
dropped before the solve, which is where the two price constraints live.

**Stage 2 -- inventory rebalancing.** A classic transportation LP: minimise
``Σ cost[r→s] · t[r→s]`` subject to each region ending between its safety stock
and its capacity, and shipping no more than a region actually holds.

Constraints implemented, mapping to the spec's four bullets:

1. price ≤ competitor average × 1.05    -- stage 1, candidate filter
2. price ≥ unit cost × 1.10             -- stage 1, candidate filter
3. safety stock ≤ ending stock ≤ capacity -- stage 2, row bounds
4. minimise total rebalancing cost      -- stage 2, objective
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from tobacco import config

log = logging.getLogger(__name__)

#: Candidate price adjustments, as fractions of the current price.
#:
#: This is a **trust region**, not a search range to be widened until the answer
#: falls inside it. The constant-elasticity curve is asserted at the *current*
#: price and is only credible near it; the further a candidate sits from ``p0``,
#: the more of the reported profit is extrapolation. So an optimum outside these
#: bounds is a finding to report, not a reason to enlarge the grid -- see
#: ``_interior_optimum``, whose ``p*`` does not depend on ``p0`` at all, which is
#: exactly why widening would just turn every run into the static markup optimum.
PRICE_GRID = np.round(np.arange(-0.10, 0.2501, 0.01), 4)


@dataclass
class PriceDecision:
    sku: str
    current_price: float
    recommended_price: float
    adjustment_pct: float
    expected_profit: float
    binding_constraint: str  # which limit stopped it going further
    competitor_ceiling: float | None
    margin_floor: float
    #: Unconstrained profit optimum, for the log and the memo notes only --
    #: deliberately *not* written to the ``recommendations`` dataset, which
    #: carries prices someone might act on rather than diagnostics.
    unconstrained_optimum_ngn: float | None = None


@dataclass
class OptimizationResult:
    prices: list[PriceDecision] = field(default_factory=list)
    transfers: pd.DataFrame = field(default_factory=pd.DataFrame)
    stock_alerts: pd.DataFrame = field(default_factory=pd.DataFrame)
    total_transfer_cost: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def overall_adjustment_pct(self) -> float:
        """Volume-agnostic mean adjustment -- the memo's headline number."""
        if not self.prices:
            return 0.0
        return float(np.mean([p.adjustment_pct for p in self.prices]))


# ---------------------------------------------------------------------------
# stage 1: price selection
# ---------------------------------------------------------------------------


def _demand_at(base_demand: float, base_price: float, price: float, elasticity: float) -> float:
    """Constant-elasticity demand response, ``D = D0 · (p/p0)^ε``."""
    if base_price <= 0:
        return base_demand
    return base_demand * (price / base_price) ** elasticity


def _interior_optimum(unit_cost: float, elasticity: float) -> float | None:
    """Unconstrained profit-maximising price, or ``None`` when none exists.

    ``p* = εc/(1+ε)``; with ``ε = −e`` that is ``c·e/(e−1)`` -- the Lerner markup
    at margin ``1/e``. For ``e ≤ 1`` the derivative is positive at every ``p > 0``:
    profit rises without bound, so a grid edge is a property of the grid, not a
    recommendation.
    """
    e = abs(float(elasticity))
    if e <= 1.0 + 1e-9:
        return None
    return unit_cost * e / (e - 1.0)


def optimise_prices(
    forecast: pd.DataFrame,
    competitor_avg: float | None,
) -> tuple[list[PriceDecision], list[str]]:
    """Choose a price per SKU subject to the ceiling and margin floor."""
    notes: list[str] = []
    if competitor_avg is None or not np.isfinite(competitor_avg):
        notes.append(
            "No competitor price observed; the +5% competitor ceiling is "
            "INACTIVE and recommendations are bounded only by the price grid."
        )
        log.warning(notes[-1])

    decisions: list[PriceDecision] = []

    for sku in config.SKUS:
        rows = forecast[forecast["sku"] == sku]
        if rows.empty:
            continue

        base_demand = float(rows["forecast_qty"].sum())
        base_price = float(rows["assumed_price_ngn"].dropna().mean())
        if not np.isfinite(base_price) or base_price <= 0:
            base_price = config.BASE_PRICE_NGN[sku]

        unit_cost = config.UNIT_COST_NGN[sku]
        elasticity = config.PRICE_ELASTICITY[sku]

        # Computed here, and reported unconditionally, rather than inside the
        # binding chain below: when demand is inelastic there is no optimum for
        # *any* constraint to be binding against, and a run where the competitor
        # ceiling happened to be active would otherwise print `competitor_ceiling`
        # -- a plausible-looking label on a meaningless number.
        optimum = _interior_optimum(unit_cost, elasticity)
        if optimum is None:
            notes.append(
                f"{sku}: elasticity {elasticity:.2f} is inelastic (|e| <= 1), so profit "
                f"rises without bound as price rises and there is no interior optimum. "
                f"No price grid can contain one. Any price reported below is the edge "
                f"of a bound, not a profit-maximising choice."
            )
            log.warning(notes[-1])

        margin_floor = unit_cost * (1 + config.MIN_MARGIN_OVER_COST)
        ceiling = (
            competitor_avg * (1 + config.MAX_PREMIUM_OVER_COMPETITOR)
            if competitor_avg is not None and np.isfinite(competitor_avg)
            else None
        )

        candidates = base_price * (1 + PRICE_GRID)
        feasible = candidates >= margin_floor
        if ceiling is not None:
            feasible &= candidates <= ceiling

        if not feasible.any():
            # The floor and ceiling have crossed: competitors are selling below
            # our minimum viable price. Report it instead of picking a number
            # that violates one of the two constraints.
            notes.append(
                f"{sku}: infeasible -- margin floor NGN {margin_floor:,.0f} exceeds "
                f"competitor ceiling NGN {ceiling:,.0f}. Holding current price; "
                f"this is a cost or positioning problem, not a pricing one."
            )
            log.warning(notes[-1])
            decisions.append(
                PriceDecision(
                    sku=sku,
                    current_price=base_price,
                    recommended_price=base_price,
                    adjustment_pct=0.0,
                    expected_profit=(base_price - unit_cost) * base_demand,
                    binding_constraint="infeasible_floor_above_ceiling",
                    competitor_ceiling=ceiling,
                    margin_floor=margin_floor,
                    unconstrained_optimum_ngn=optimum,
                )
            )
            continue

        viable = candidates[feasible]
        # Exact profit per candidate, computed outside the LP so the nonlinear
        # elasticity term is preserved rather than approximated.
        profits = np.array(
            [
                (price - unit_cost) * _demand_at(base_demand, base_price, price, elasticity)
                for price in viable
            ]
        )

        # linprog minimises, so negate. One equality row: weights sum to 1.
        result = linprog(
            c=-profits,
            A_eq=np.ones((1, len(viable))),
            b_eq=np.array([1.0]),
            bounds=[(0.0, 1.0)] * len(viable),
            method="highs",
        )
        if not result.success:
            notes.append(f"{sku}: price LP failed ({result.message}); holding price.")
            log.error(notes[-1])
            chosen_index = int(np.argmax(profits))
        else:
            chosen_index = int(np.argmax(result.x))

        chosen_price = float(viable[chosen_index])
        adjustment = (chosen_price / base_price - 1) * 100

        # Report which limit actually stopped the optimiser -- the most useful
        # single line for the Commercial Director reading the memo. A constraint
        # is binding only if the chosen price sits at the edge of the feasible
        # set *and* that edge was imposed by the constraint rather than the grid.
        at_top = np.isclose(chosen_price, viable.max())
        at_bottom = np.isclose(chosen_price, viable.min())
        grid_top = candidates.max()
        grid_bottom = candidates.min()

        if at_top and ceiling is not None and viable.max() < grid_top - 1e-9:
            binding = "competitor_ceiling"
        elif at_bottom and viable.min() > grid_bottom + 1e-9:
            binding = "margin_floor"
        elif optimum is None:
            binding = "no_interior_optimum"
        elif at_top and optimum > grid_top:
            binding = "price_grid_upper_bound"
        elif at_bottom and optimum < grid_bottom:
            # Not dead code by accident: today's elasticities never reach it, but
            # a choice at the -10% edge with a *slack* margin floor used to fall
            # through to `else` and be reported as `profit_optimum`, which it is
            # not. Reachable once |e| > 3.15.
            binding = "price_grid_lower_bound"
        else:
            # Sitting on the top candidate is only an artifact when the optimum
            # is beyond it. If p* falls between the last two grid steps, the top
            # candidate genuinely *is* the grid-rounded optimum, and calling that
            # a grid artifact would be the opposite dishonesty.
            binding = "profit_optimum"

        if binding in ("price_grid_upper_bound", "price_grid_lower_bound"):
            notes.append(
                f"{sku}: the profit optimum is NGN {optimum:,.0f}, outside the "
                f"{PRICE_GRID.min():+.0%}..{PRICE_GRID.max():+.0%} price grid "
                f"(NGN {grid_bottom:,.0f}..{grid_top:,.0f}). The price reported is the "
                f"grid edge, not the optimum -- the grid bounds how far a demand curve "
                f"asserted at the current price is trusted to extrapolate."
            )
            log.warning(notes[-1])

        decisions.append(
            PriceDecision(
                sku=sku,
                current_price=round(base_price, 2),
                recommended_price=round(chosen_price, 2),
                adjustment_pct=round(float(adjustment), 2),
                expected_profit=float(profits[chosen_index]),
                binding_constraint=binding,
                competitor_ceiling=round(ceiling, 2) if ceiling else None,
                margin_floor=round(margin_floor, 2),
                unconstrained_optimum_ngn=round(optimum, 2) if optimum is not None else None,
            )
        )
        log.info(
            "%s: NGN %.0f -> %.0f (%+.1f%%), bound by %s (p* %s)",
            sku, base_price, chosen_price, adjustment, binding,
            f"NGN {optimum:,.0f}" if optimum is not None else "none -- unbounded",
        )

    return decisions, notes


# ---------------------------------------------------------------------------
# stage 2: inventory rebalancing
# ---------------------------------------------------------------------------


def optimise_transfers(forecast: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Minimum-cost inter-regional transfers, per SKU.

    Returns ``(transfers, stock_alerts, total_cost)``.
    """
    transfer_rows: list[dict] = []
    alert_rows: list[dict] = []
    total_cost = 0.0

    for sku in config.SKUS:
        rows = forecast[forecast["sku"] == sku]
        if rows.empty:
            continue

        # Weekly demand rate drives both the safety floor and the cap.
        weekly = rows.groupby("region")["forecast_qty"].mean()
        stock = rows.groupby("region")["stock_on_hand"].first()

        regions = [r for r in config.REGIONS if r in weekly.index]
        if len(regions) < 2:
            continue

        demand = np.array([float(weekly[r]) for r in regions])
        held = np.array([float(stock.get(r, 0.0)) for r in regions])
        safety = demand * config.SAFETY_STOCK_WEEKS
        capacity = demand * config.MAX_CAPACITY_WEEKS

        # Decision variables: one per ordered (from, to) pair.
        arcs = [(i, j) for i in range(len(regions)) for j in range(len(regions)) if i != j]
        if not arcs:
            continue
        costs = np.array(
            [config.TRANSFER_COST_PER_UNIT.get((regions[i], regions[j]), 100.0) for i, j in arcs]
        )

        # ending[r] = held[r] - Σ out[r] + Σ in[r]
        # Encode as a signed incidence matrix so both bounds are linear rows.
        incidence = np.zeros((len(regions), len(arcs)))
        for column, (i, j) in enumerate(arcs):
            incidence[i, column] -= 1.0
            incidence[j, column] += 1.0

        # ending >= safety   ->  -incidence · t <= held - safety
        # ending <= capacity ->   incidence · t <= capacity - held
        # Σ out[r] <= held[r] ->  a region cannot ship stock it does not have.
        outflow = np.zeros((len(regions), len(arcs)))
        for column, (i, _) in enumerate(arcs):
            outflow[i, column] = 1.0

        a_ub = np.vstack([-incidence, incidence, outflow])
        b_ub = np.concatenate([held - safety, capacity - held, held])

        result = linprog(
            c=costs,
            A_ub=a_ub,
            b_ub=b_ub,
            bounds=[(0, None)] * len(arcs),
            method="highs",
        )

        if not result.success:
            # Infeasible means no shuffling of existing stock can satisfy every
            # region's floor -- i.e. there is a genuine shortfall to produce or
            # buy. That is a real finding, so surface it as an alert.
            shortfall = float(np.maximum(safety - held, 0).sum())
            log.warning(
                "%s: rebalancing infeasible (%s). Network-wide shortfall ~%.0f packs.",
                sku, result.message, shortfall,
            )
            for index, region in enumerate(regions):
                if held[index] < safety[index]:
                    alert_rows.append(
                        {
                            "sku": sku,
                            "region": region,
                            "stock_on_hand": held[index],
                            "safety_stock": safety[index],
                            "weeks_cover": held[index] / demand[index] if demand[index] else 0.0,
                            "status": "stockout_risk",
                            "note": "cannot be resolved by transfer; requires replenishment",
                        }
                    )
            continue

        moved = result.x
        sku_cost = float(costs @ moved)
        total_cost += sku_cost

        for column, (i, j) in enumerate(arcs):
            # Sub-pack quantities are solver noise, not instructions.
            if moved[column] > 1.0:
                transfer_rows.append(
                    {
                        "sku": sku,
                        "from_region": regions[i],
                        "to_region": regions[j],
                        "quantity": float(round(moved[column])),
                        "unit_cost": float(costs[column]),
                        "total_cost": float(round(moved[column] * costs[column], 2)),
                    }
                )

        ending = held + incidence @ moved
        for index, region in enumerate(regions):
            cover = ending[index] / demand[index] if demand[index] > 0 else 0.0
            if cover < config.SAFETY_STOCK_WEEKS - 1e-6:
                status = "stockout_risk"
            elif cover > config.MAX_CAPACITY_WEEKS - 1e-6:
                status = "overstock"
            else:
                status = "ok"
            if status != "ok":
                alert_rows.append(
                    {
                        "sku": sku,
                        "region": region,
                        "stock_on_hand": float(held[index]),
                        "safety_stock": float(round(safety[index])),
                        "weeks_cover": float(round(cover, 2)),
                        "status": status,
                        "note": "after recommended transfers",
                    }
                )

    transfers = pd.DataFrame(transfer_rows)
    alerts = pd.DataFrame(alert_rows)
    log.info(
        "Rebalancing: %d transfer(s) costing NGN %.0f; %d stock alert(s)",
        len(transfers), total_cost, len(alerts),
    )
    return transfers, alerts, total_cost


def optimise(forecast: pd.DataFrame, competitor_avg: float | None) -> OptimizationResult:
    """Run both stages and bundle the result."""
    if forecast.empty:
        return OptimizationResult(notes=["No forecast available; nothing to optimise."])

    prices, notes = optimise_prices(forecast, competitor_avg)
    transfers, alerts, cost = optimise_transfers(forecast)

    return OptimizationResult(
        prices=prices,
        transfers=transfers,
        stock_alerts=alerts,
        total_transfer_cost=cost,
        notes=notes,
    )


def to_recommendations(result: OptimizationResult, forecast: pd.DataFrame) -> pd.DataFrame:
    """Flatten into the ``recommendations`` dataset (INTRO.txt §2)."""
    today = pd.Timestamp(config.today_wat())
    by_sku = {p.sku: p for p in result.prices}

    alert_lookup = {}
    if not result.stock_alerts.empty:
        for _, row in result.stock_alerts.iterrows():
            alert_lookup[(row["sku"], row["region"])] = row["status"]

    rows = []
    for (sku, region), group in forecast.groupby(["sku", "region"]):
        decision = by_sku.get(sku)
        rows.append(
            {
                "date": today,
                "sku": sku,
                "region": region,
                "price_adjustment_pct": decision.adjustment_pct if decision else 0.0,
                "recommended_price_ngn": decision.recommended_price if decision else None,
                "binding_constraint": decision.binding_constraint if decision else "none",
                "forecast_qty_4w": float(group["forecast_qty"].sum()),
                "inventory_action": alert_lookup.get((sku, region), "ok"),
            }
        )
    return pd.DataFrame(rows)
