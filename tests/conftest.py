"""Shared fixtures.

Three of the fixtures below are ``autouse``. That is a safety property, not a
convenience: the suite runs on the same checkout the Actions jobs run on, so a
test that reached the real ``data/curated/`` or opened a real socket would be
indistinguishable from a job doing so. Making the isolation automatic means a
future test cannot forget to ask for it.

The suite requires **no secret and no network**, and both are enforced here
rather than merely intended.
"""

from __future__ import annotations

import socket

import pandas as pd
import pytest

from tobacco import config

#: Every secret or repo variable any tested module reads. Scrubbed rather than
#: assumed absent, so the suite behaves the same on a developer's machine, on a
#: fork with no secrets, and on the real repo once secrets are finally set.
ENV_VARS = (
    "GMAIL_ADDRESS",
    "GMAIL_APP_PASSWORD",
    "GROQ_API_KEY",
    "ALERT_RECIPIENTS_COMMERCIAL",
    "ALERT_RECIPIENTS_SUPPLY",
    "FINBERT_MODEL",
    "NBS_INFLATION_URL",
    "HF_TOKEN",
)


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    """Point the data layer at a per-test temp directory.

    Monkeypatching the ``config`` attribute is enough, and is safe, because every
    module in the package does ``from tobacco import config`` rather than
    ``from tobacco.config import DATA_DIR`` -- and ``parquet_io.dataset_dir``
    resolves ``config.DATA_DIR`` at call time, not at import.

    This also covers ``features/build.py`` and ``sales_mock._fx_by_week``, which
    reach the filesystem only through ``parquet_io.read``.
    """
    data_dir = tmp_path / "curated"
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    return data_dir


@pytest.fixture(autouse=True)
def scrubbed_env(monkeypatch):
    """Remove every credential from the environment for the duration of a test."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail loudly on any outbound connection.

    Connect is blocked rather than ``socket.socket`` construction: some libraries
    build a socket object during import or setup without ever using it, and
    breaking that would be blocking the wrong thing. Autouse fixtures do not run
    during collection, so module-level imports are unaffected either way.
    """

    def blocked(*args, **kwargs):
        raise AssertionError(
            "A test attempted to open a network connection. The suite is "
            "offline by construction -- stub the call instead."
        )

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


# ---------------------------------------------------------------------------
# the interior-optimum band
# ---------------------------------------------------------------------------


def interior_share_band(unit_cost: float, price: float) -> tuple[float, float]:
    """Market shares for which the profit optimum falls strictly inside PRICE_GRID.

    Derived rather than hard-coded, because the literal band in ``config.py`` is
    only reproducible at one FX level: ``sales_mock`` scales shelf prices by
    ``1 + 0.35·(fx/1500 − 1)``, so the observed ``price`` -- and with it the band
    -- drifts on every scrape. Pinning the literals would pin today's naira.

    The algebra, so it is auditable:

        p* = c·e/(e−1)                      the Lerner optimum (linprog.py:98)
        p* = p0·B                           at grid multiplier B = 1 + adjustment
      =>  c·e = p0·B·(e − 1)
      =>  e   = B / (B − r)                 with r = c/p0
        e   = |E_category| / s              the config.py:113 derivation
      =>  s   = |E_category| · (B − r) / B

    ``s`` is increasing in ``B``, so the admissible band is
    ``(s(B_min), s(B_max))`` taken from the grid's own endpoints.
    """
    from tobacco.optimize import linprog

    ratio = unit_cost / price
    magnitude = abs(config.CATEGORY_PRICE_ELASTICITY)

    def share_at(bound: float) -> float:
        return magnitude * (bound - ratio) / bound

    return (
        share_at(1.0 + float(linprog.PRICE_GRID.min())),
        share_at(1.0 + float(linprog.PRICE_GRID.max())),
    )


# ---------------------------------------------------------------------------
# factories
# ---------------------------------------------------------------------------


@pytest.fixture
def single_sku(monkeypatch):
    """Reduce the portfolio to one synthetic SKU with chosen economics.

    Far cleaner than steering the real three into a target branch: each optimizer
    branch below is reached by *stating* the cost, price and elasticity that reach
    it, rather than by finding a scenario in which PREMIUM_20 happens to.
    """

    def configure(unit_cost: float, base_price: float, elasticity: float, sku: str = "TEST_20"):
        monkeypatch.setattr(config, "SKUS", (sku,))
        monkeypatch.setattr(config, "UNIT_COST_NGN", {sku: float(unit_cost)})
        monkeypatch.setattr(config, "BASE_PRICE_NGN", {sku: float(base_price)})
        monkeypatch.setattr(config, "PRICE_ELASTICITY", {sku: float(elasticity)})
        return sku

    return configure


@pytest.fixture
def forecast_frame():
    """A minimal frame shaped like ``predict.forecast``'s output."""

    def build(
        sku: str,
        assumed_price_ngn: float,
        forecast_qty: float = 1000.0,
        regions: tuple[str, ...] | None = None,
        stock_on_hand: float = 5000.0,
    ) -> pd.DataFrame:
        regions = regions if regions is not None else config.REGIONS
        return pd.DataFrame(
            [
                {
                    "week_start": pd.Timestamp("2026-09-07") + pd.Timedelta(weeks=week),
                    "sku": sku,
                    "region": region,
                    "forecast_qty": float(forecast_qty),
                    "assumed_price_ngn": float(assumed_price_ngn),
                    "stock_on_hand": float(stock_on_hand),
                }
                for week in range(config.FORECAST_HORIZON_WEEKS)
                for region in regions
            ]
        )

    return build
