"""Load the committed forecaster and predict forward (INTRO.txt §9 step 3)."""

from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd

from tobacco import config
from tobacco.features import build

log = logging.getLogger(__name__)


class ModelNotTrained(RuntimeError):
    """Raised when no model artifact has been committed yet."""


def load_bundle() -> dict:
    if not config.MODEL_PATH.exists():
        raise ModelNotTrained(
            f"No model at {config.MODEL_PATH}. Run the train workflow first: "
            f"`gh workflow run train.yml`."
        )
    bundle = joblib.load(config.MODEL_PATH)
    log.info(
        "Loaded model trained %s (MAPE %.2f%%)",
        bundle["metrics"]["trained_at"][:10], bundle["metrics"]["mape_pct"],
    )
    return bundle


def forecast(horizon: int = config.FORECAST_HORIZON_WEEKS) -> pd.DataFrame:
    """Predicted weekly demand per SKU per region for the next ``horizon`` weeks."""
    bundle = load_bundle()
    panel = build.build_forecast_frame(horizon)
    if panel.empty:
        return pd.DataFrame()

    features = bundle["features"]
    missing = [f for f in features if f not in panel.columns]
    if missing:
        raise RuntimeError(
            f"Forecast frame is missing feature(s) the model was trained on: "
            f"{missing}. Retrain after any change to features/build.py."
        )

    # Reindex to the *training* column order. XGBoost accepts a permuted matrix
    # without complaint and returns confidently wrong numbers.
    matrix = panel[features]

    # Exogenous gaps are expected at the edges of the history; XGBoost handles
    # NaN natively via its default split direction, so leave them alone rather
    # than imputing a value the model was not trained to expect.
    predicted = np.clip(bundle["model"].predict(matrix), 0, None)

    result = panel[build.KEYS].copy()
    result["forecast_qty"] = predicted
    result["stock_on_hand"] = panel["stock_on_hand"].to_numpy()
    result["assumed_price_ngn"] = panel["avg_price_ngn"].to_numpy()
    result["model_mape_pct"] = bundle["metrics"]["mape_pct"]

    log.info(
        "Forecast: %d row(s), total %.0f packs over %d week(s)",
        len(result), result["forecast_qty"].sum(), horizon,
    )
    return result


def demand_trend(forecast_frame: pd.DataFrame) -> str:
    """'growing' / 'stable' / 'declining' -- the memo's ``demand_trend`` field."""
    if forecast_frame.empty:
        return "unknown"
    weekly = forecast_frame.groupby("week_start")["forecast_qty"].sum().sort_index()
    if len(weekly) < 2:
        return "stable"
    change_pct = (weekly.iloc[-1] - weekly.iloc[0]) / weekly.iloc[0] * 100
    if change_pct > 3:
        return "growing"
    if change_pct < -3:
        return "declining"
    return "stable"
