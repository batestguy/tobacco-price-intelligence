"""Configuration: paths, business constants, and environment-driven secrets.

Secrets are *only* ever read from the environment. Nothing here has a real
default value and nothing is read from a committed file -- the repo is public.

Every secret is optional to *some* job, so nothing is validated at import time.
``require`` raises at the point of use; ``require_all`` is available for a caller
that wants a misconfiguration reported as one list rather than one failure at a
time. Nothing currently uses it -- the remaining secrets are each read by a single
non-fatal stage.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "curated"
SEED_DIR = REPO_ROOT / "data" / "seed"
MODEL_DIR = REPO_ROOT / "models"

MODEL_PATH = MODEL_DIR / "xgb_demand.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

# --------------------------------------------------------------------------
# time
# --------------------------------------------------------------------------

#: West Africa Time. Nigeria observes UTC+1 year round -- no DST, so a fixed
#: offset is correct and avoids a tzdata dependency on the runner.
WAT = timezone(timedelta(hours=1))


def now_wat() -> datetime:
    return datetime.now(WAT)


def today_wat() -> date:
    return now_wat().date()


# --------------------------------------------------------------------------
# business constants
# --------------------------------------------------------------------------

#: Distribution regions (INTRO.txt §4 "Region dummies").
REGIONS: tuple[str, ...] = ("Lagos", "Ibadan", "Kano", "Port Harcourt")

#: Own-product SKUs. Deliberately generic tier codes rather than real brand
#: names: the sales data behind them is synthetic, and naming a real
#: manufacturer's portfolio in a public repo would imply data we do not have.
SKUS: tuple[str, ...] = ("PREMIUM_20", "MIDRANGE_20", "VALUE_20")

#: Competitor brands tracked (INTRO.txt §1). Prices for these come from the cited
#: reference file, not a scrape -- see sources/competitors.py.
COMPETITOR_BRANDS: tuple[str, ...] = ("Bohem", "Time", "Gold Mount")

#: Unit production cost per pack (NGN), used by the optimizer's margin floor.
#: Synthetic, consistent with the synthetic sales generator.
UNIT_COST_NGN: dict[str, float] = {
    "PREMIUM_20": 900.0,
    "MIDRANGE_20": 620.0,
    "VALUE_20": 430.0,
}

#: Baseline list price per pack (NGN) before any recommended adjustment.
BASE_PRICE_NGN: dict[str, float] = {
    "PREMIUM_20": 1500.0,
    "MIDRANGE_20": 1050.0,
    "VALUE_20": 700.0,
}

#: Own-price elasticity of demand for the tobacco CATEGORY in Nigeria.
#: Cited: -0.62, national. Tob Prev Cessat 2020 (PMID 32411910); rural -0.63,
#: urban -0.49. Corroborating: Nigeria youth -0.44 (PMC6747324); Africa
#: low-income -0.562 (NCI Monograph 21 ch.4).
CATEGORY_PRICE_ELASTICITY = -0.62

#: ASSUMPTION, not a measurement. The firm is deliberately unnamed, so no share
#: figure exists to cite. Rule stated in advance: COMPETITOR_BRANDS tracks three
#: rivals; with this firm that is four players; an equal split is 0.25.
#:
#: Sensitivity, disclosed so the number is auditable rather than merely
#: convenient: at current prices every SKU lands strictly inside PRICE_GRID for
#: s in (0.197, 0.304). Outside that band the optimizer reports a grid bound --
#: which is the honest outcome, and is what the guard in optimize/linprog.py
#: exists to make visible. This value is NOT to be re-tuned to escape one.
ASSUMED_MARKET_SHARE = 0.25

#: Firm-level elasticity, derived. A category figure understates what ONE seller
#: faces: buyers leaving a single brand switch to rivals rather than quitting.
#: Under proportional substitution, eps_firm ~= eps_category / share.
#: Derived in code so the arithmetic cannot drift from the citation.
#:
#: Uniform across tiers on purpose. Per-tier multipliers do exist in the
#: literature (PMC9763177 finds premium more elastic than discount -- note this
#: is the *opposite* of the "value tier is the most price-sensitive" assumption
#: this replaces), but normalised against sales_mock.SKU_MIX the per-SKU interior
#: bands become (0.256, 0.410), (0.204, 0.326) and (0.139, 0.233): the
#: intersection is EMPTY. No single share puts all three interior, so adopting
#: the multiplier would force UNIT_COST_NGN or the grid to move with it. Deferred
#: rather than dropped.
#:
#: PREMIUM_20 and MIDRANGE_20 currently share a cost/price ratio (0.6139), so a
#: uniform elasticity gives them the same percentage recommendation. Expected.
PRICE_ELASTICITY: dict[str, float] = {sku: CATEGORY_PRICE_ELASTICITY / ASSUMED_MARKET_SHARE
                                      for sku in SKUS}   # -2.48

#: Weekly holding cost per unit of inventory (NGN).
HOLDING_COST_PER_UNIT = 12.0

#: Cost of moving one unit between regions (NGN). Ibadan is the factory, so it
#: is the natural source for rebalancing.
TRANSFER_COST_PER_UNIT: dict[tuple[str, str], float] = {
    ("Ibadan", "Lagos"): 18.0,
    ("Ibadan", "Kano"): 46.0,
    ("Ibadan", "Port Harcourt"): 38.0,
    ("Lagos", "Ibadan"): 18.0,
    ("Lagos", "Kano"): 55.0,
    ("Lagos", "Port Harcourt"): 40.0,
    ("Kano", "Ibadan"): 46.0,
    ("Kano", "Lagos"): 55.0,
    ("Kano", "Port Harcourt"): 62.0,
    ("Port Harcourt", "Ibadan"): 38.0,
    ("Port Harcourt", "Lagos"): 40.0,
    ("Port Harcourt", "Kano"): 62.0,
}

#: Optimizer bounds (INTRO.txt §5).
MAX_PREMIUM_OVER_COMPETITOR = 0.05  # price <= competitor avg * 1.05
MIN_MARGIN_OVER_COST = 0.10  # price >= unit cost * 1.10
SAFETY_STOCK_WEEKS = 1.5  # weeks of forecast demand
MAX_CAPACITY_WEEKS = 6.0  # weeks of forecast demand

#: Alert thresholds (INTRO.txt §7).
FX_DROP_ALERT_PCT = 2.0
SENTIMENT_ALERT_THRESHOLD = 0.3

#: Forecast horizon in weeks (INTRO.txt §4).
FORECAST_HORIZON_WEEKS = 4

#: Mandated dashboard disclaimer, reproduced verbatim from INTRO.txt §11.
#: Kept here so there is exactly one copy and it cannot drift. Do not reword:
#: it is a compliance text, not UI copy.
DISCLAIMER = (
    "DISCLAIMER: This tool is intended for internal operational efficiency and "
    "supply chain management only. It does NOT promote, advertise, or encourage "
    "the production, sale, or consumption of tobacco products. All recommendations "
    "are for internal decision-support and comply with the National Tobacco Control "
    "Act 2015 and WHO FCTC Article 5.3. Data is aggregated and anonymized; no "
    "individual consumer data is stored. Use of this tool is restricted to "
    "authorized personnel only."
)

#: Public-repo framing, shown alongside the disclaimer. The disclaimer describes
#: an internal tool; this repo is a public portfolio piece, and saying so plainly
#: is what keeps the two consistent (see CLAUDE.md "Regulatory constraint").
PORTFOLIO_NOTICE = (
    "This is an independent portfolio and educational demonstration of a data "
    "engineering and forecasting pipeline. It is not affiliated with, commissioned "
    "by, or endorsed by any tobacco company. All sales figures shown are synthetic."
)

# --------------------------------------------------------------------------
# holidays (INTRO.txt §4 "Holiday dummies")
# --------------------------------------------------------------------------

#: Eid al-Fitr / Eid al-Adha ("Sallah") shift ~11 days a year against the
#: Gregorian calendar, so they cannot be derived from a fixed rule without a
#: hijri library. Approximate observed dates, extend as needed.
SALLAH_DATES: tuple[date, ...] = (
    date(2024, 4, 10), date(2024, 6, 16),
    date(2025, 3, 30), date(2025, 6, 6),
    date(2026, 3, 20), date(2026, 5, 27),
    date(2027, 3, 9), date(2027, 5, 16),
)


def is_holiday_week(week_start: date) -> bool:
    """True if any major holiday falls in the 7 days from ``week_start``."""
    days = {week_start + timedelta(days=i) for i in range(7)}
    if any(d.month == 12 and d.day in (24, 25, 26) for d in days):
        return True
    if any(d.month == 1 and d.day == 1 for d in days):
        return True
    return bool(days & set(SALLAH_DATES))


# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------


class MissingSecret(RuntimeError):
    """Raised when a required environment variable is absent or blank."""


def optional(name: str, default: str = "") -> str:
    """Read an env var, treating blank/whitespace as absent.

    GitHub Actions injects unset secrets as empty strings rather than omitting
    them, so ``os.environ.get(name)`` alone is not enough to detect a missing one.
    """
    return (os.environ.get(name) or "").strip() or default


def require(name: str) -> str:
    value = optional(name)
    if not value:
        raise MissingSecret(
            f"Required environment variable {name!r} is not set. "
            f"For Actions: `gh secret set {name}`. See REGISTRY.md."
        )
    return value


def require_all(*names: str) -> dict[str, str]:
    """Fetch several required variables, reporting *all* missing ones at once."""
    found, missing = {}, []
    for name in names:
        value = optional(name)
        if value:
            found[name] = value
        else:
            missing.append(name)
    if missing:
        raise MissingSecret(
            "Missing required environment variable(s): "
            + ", ".join(sorted(missing))
            + ". Set them with `gh secret set <NAME>`; see REGISTRY.md."
        )
    return found


def recipients(role: str) -> list[str]:
    """Alert recipients for a role, from a comma-separated repo variable."""
    var = {
        "commercial": "ALERT_RECIPIENTS_COMMERCIAL",
        "supply": "ALERT_RECIPIENTS_SUPPLY",
    }[role]
    return [addr.strip() for addr in optional(var).split(",") if addr.strip()]


#: Which FinBERT checkpoint to score with. Overridable via a repo *variable* so
#: Phase 3 can switch to the fine-tuned model without a code change.
def finbert_model() -> str:
    return optional("FINBERT_MODEL", "ProsusAI/finbert")


# --------------------------------------------------------------------------
# logging
# --------------------------------------------------------------------------


def setup_logging(name: str = "tobacco") -> logging.Logger:
    """Consistent log format across jobs, tuned for the Actions log viewer."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    # These libraries are chatty at INFO and drown out our own output.
    for noisy in ("urllib3", "charset_normalizer", "filelock", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger(name)


#: A browser-ish UA. Several Nigerian sites return 403 to the default
#: python-requests agent.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
HTTP_TIMEOUT = 30
