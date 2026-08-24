"""Train the weekly demand forecaster (INTRO.txt §4).

XGBoost, per the spec. Two details are non-negotiable for a time series:

* **The split is chronological, never random.** A shuffled split lets the model
  see 2027 while predicting 2026; RMSE then looks excellent and the deployed
  forecast is worthless. Validation is the most recent weeks only.
* **The model and its feature list are saved together.** XGBoost silently accepts
  a differently-ordered feature matrix and returns confident nonsense, so the
  column order used at training time is persisted in the bundle and enforced at
  prediction time.

The trained artifact is ~1-5 MB, small enough to commit; that keeps the repo the
complete record of how any given recommendation was produced.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from tobacco import config
from tobacco.features import build

log = logging.getLogger(__name__)

#: Weeks held out for validation. Matches the forecast horizon so the reported
#: error is measured on exactly the task the model is used for.
VALIDATION_WEEKS = config.FORECAST_HORIZON_WEEKS

PARAMS = dict(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=3,
    reg_lambda=1.5,
    objective="reg:squarederror",
    tree_method="hist",  # CPU histogram: the runner has no GPU
    n_jobs=2,            # a free runner has 2 vCPUs; more threads only thrash
    random_state=42,
)


def _mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute percentage error, ignoring zero-demand weeks.

    MAPE is undefined at zero and explodes near it, so those rows are excluded
    rather than allowed to dominate the metric.
    """
    mask = actual > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def train() -> dict:
    """Fit, evaluate, and persist the model. Returns the metrics dict."""
    from xgboost import XGBRegressor

    panel = build.build_training_frame()
    if panel.empty:
        raise RuntimeError("No training data. Generate sales_mock first.")

    features = build.feature_columns(panel)
    panel = panel.sort_values("week_start").reset_index(drop=True)

    cutoff = panel["week_start"].max() - pd.Timedelta(weeks=VALIDATION_WEEKS)
    train_set = panel[panel["week_start"] <= cutoff]
    valid_set = panel[panel["week_start"] > cutoff]

    if valid_set.empty or len(train_set) < 100:
        raise RuntimeError(
            f"Not enough history to train: {len(train_set)} train / "
            f"{len(valid_set)} validation rows."
        )

    x_train = train_set[features]
    y_train = train_set[build.TARGET]
    x_valid = valid_set[features]
    y_valid = valid_set[build.TARGET]

    log.info(
        "Training on %d rows (%s..%s), validating on %d rows (%d features)",
        len(train_set), train_set["week_start"].min().date(),
        cutoff.date(), len(valid_set), len(features),
    )

    model = XGBRegressor(**PARAMS)
    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)

    predicted = model.predict(x_valid)
    # Demand cannot be negative; the regressor does not know that.
    predicted = np.clip(predicted, 0, None)

    # Naive baseline: predict each series' most recent training value. If the
    # model cannot beat this, the features are not earning their keep and the
    # metric should say so rather than flattering a plausible-looking RMSE.
    last_known = (
        train_set.sort_values("week_start")
        .groupby(["sku", "region"])[build.TARGET]
        .last()
    )
    naive = pd.MultiIndex.from_frame(valid_set[["sku", "region"]]).map(last_known)
    naive = np.asarray(naive, dtype=float)

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "rmse": _rmse(y_valid.to_numpy(), predicted),
        "mape_pct": _mape(y_valid.to_numpy(), predicted),
        "naive_baseline_rmse": _rmse(y_valid.to_numpy(), naive),
        "naive_baseline_mape_pct": _mape(y_valid.to_numpy(), naive),
        "n_train_rows": int(len(train_set)),
        "n_validation_rows": int(len(valid_set)),
        "n_features": len(features),
        "validation_weeks": VALIDATION_WEEKS,
        "train_period": [
            str(train_set["week_start"].min().date()),
            str(cutoff.date()),
        ],
        "top_features": (
            pd.Series(model.feature_importances_, index=features)
            .sort_values(ascending=False)
            .head(12)
            .round(4)
            .to_dict()
        ),
    }

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": features,          # order matters at predict time
            "target": build.TARGET,
            "metrics": metrics,
            "version": 1,
        },
        config.MODEL_PATH,
        compress=3,
    )
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    size_mb = config.MODEL_PATH.stat().st_size / 1e6
    log.info(
        "Saved %s (%.1f MB) | RMSE %.0f | MAPE %.2f%% | naive baseline MAPE %.2f%%",
        config.MODEL_PATH.name, size_mb, metrics["rmse"], metrics["mape_pct"],
        metrics["naive_baseline_mape_pct"],
    )
    if metrics["mape_pct"] >= metrics["naive_baseline_mape_pct"]:
        log.warning(
            "Model (%.2f%% MAPE) does not beat the naive last-value baseline "
            "(%.2f%%). Treat its forecasts as unproven.",
            metrics["mape_pct"], metrics["naive_baseline_mape_pct"],
        )
    if size_mb > 50:
        log.warning(
            "Model is %.0f MB -- too large to keep committing daily. Move it to "
            "GitHub Releases or the HF Hub (see CLAUDE.md).", size_mb,
        )
    return metrics
