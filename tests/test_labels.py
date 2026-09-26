"""Every identifier the pipeline can put on screen has an explicit display label.

``labels.humanize`` is a safety net, not a vocabulary: ``sku_PREMIUM_20`` through
the fallback reads "Sku PREMIUM 20". These tests fail when a new SKU, region,
constraint, dataset or model feature appears without a label, rather than letting
it reach the dashboard half-translated.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

from tobacco import config
from tobacco.features import build
from tobacco.store import parquet_io

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "app"))

import labels  # noqa: E402


def _literals(pattern: str, path: Path) -> set[str]:
    return set(re.findall(pattern, path.read_text(encoding="utf-8")))


LINPROG = REPO / "src" / "tobacco" / "optimize" / "linprog.py"


def test_every_sku_has_both_labels():
    for sku in config.SKUS:
        assert sku in labels.SKU_LABELS
        assert sku in labels.SKU_SHORT


def test_every_region_has_a_label():
    assert set(config.REGIONS) == set(labels.REGION_LABELS)


def test_every_binding_constraint_literal_has_a_label_and_a_reason():
    """Read from the source, so a new ``binding = "..."`` cannot slip past."""
    found = _literals(r'binding(?:_constraint)?\s*=\s*"([a-z_]+)"', LINPROG)
    found.add("none")  # written by build_recommendations when no decision exists
    assert len(found) >= 8, found  # the regex still sees the literals
    assert found <= set(labels.CONSTRAINT_LABELS), found - set(labels.CONSTRAINT_LABELS)
    assert set(labels.CONSTRAINT_LABELS) == set(labels.CONSTRAINT_REASONS)


def test_every_stock_status_has_a_label():
    found = _literals(r'status = "([a-z_]+)"', LINPROG)
    found |= _literals(r'"status": "([a-z_]+)"', LINPROG)
    assert found, "the regex no longer sees the stock statuses"
    assert found <= set(labels.STOCK_STATUS_LABELS), found - set(labels.STOCK_STATUS_LABELS)


def test_every_dataset_has_a_label():
    assert set(parquet_io.DATASETS) == set(labels.DATASET_LABELS)


def test_every_trained_feature_has_a_label():
    metrics = json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
    missing = set(metrics["top_features"]) - set(labels.FEATURE_LABELS)
    assert not missing, missing


def test_every_built_feature_has_a_label():
    """Build the real feature frame, so a new lag or dummy is caught too."""
    weeks = pd.date_range("2026-01-05", periods=6, freq="W-MON")
    panel = pd.DataFrame(
        [
            {"week_start": w, "sku": s, "region": r, "quantity_sold": 100.0,
             "avg_price_ngn": 1000.0}
            for w in weeks for s in config.SKUS for r in config.REGIONS
        ]
    )
    features = build.feature_columns(build.attach_features(panel))
    missing = set(features) - set(labels.FEATURE_LABELS)
    assert not missing, missing


@pytest.mark.parametrize(
    "table",
    [labels.SKU_LABELS, labels.SKU_SHORT, labels.STOCK_STATUS_LABELS,
     labels.CONSTRAINT_LABELS, labels.DATASET_LABELS, labels.FEATURE_LABELS,
     labels.METRIC_LABELS, labels.ROLE_LABELS],
)
def test_no_label_leaks_an_identifier(table):
    for label in table.values():
        assert "_" not in label, label


def test_no_role_has_a_label_in_words():
    assert labels.role(None) == labels.NO_ROLE_LABEL == "No access"
    assert "_" not in labels.NO_ROLE_LABEL


def test_humanize_is_only_a_fallback():
    assert labels.humanize("some_new_code") == "Some new code"
    assert labels.feature("sku_PREMIUM_20") == "Product: Premium"
    assert labels.feature("region_Port_Harcourt") == "Region: Port Harcourt"


def test_signed_pct_uses_a_true_minus():
    assert labels.signed_pct(-1.333) == "−1.3%"
    assert labels.signed_pct(2) == "+2.0%"
    assert labels.signed_pct(0.01) == "0.0%"


def test_price_verdict_mixed_moves():
    headline, detail = labels.price_verdict(
        {"PREMIUM_20": (-2.0, "profit_optimum"),
         "MIDRANGE_20": (-3.0, "profit_optimum"),
         "VALUE_20": (1.0, "profit_optimum")}
    )
    assert headline == "Adjust prices by product this week"
    assert detail.startswith("Mid-range −3.0%, Premium −2.0%, Value +1.0%")
    assert "−1.3% on average" in detail
    assert "Each product is at its profit-maximising price." in detail


def test_price_verdict_uniform_moves_and_mixed_constraints():
    headline, detail = labels.price_verdict(
        {"PREMIUM_20": (5.0, "price_grid_upper_bound"),
         "VALUE_20": (2.0, "profit_optimum")}
    )
    assert headline == "Raise prices this week"
    assert "Premium is capped by the maximum allowed increase" in detail
    assert labels.price_verdict({"VALUE_20": (0.0, "none")})[0] == "Hold prices this week"
    assert labels.price_verdict({})[0] == "No price recommendation yet"


def test_stock_verdict():
    assert labels.stock_verdict({("VALUE_20", "Kano"): "ok"})[0] == (
        "Stock is within bounds everywhere"
    )
    headline, detail = labels.stock_verdict(
        {("VALUE_20", "Kano"): "stockout_risk", ("PREMIUM_20", "Lagos"): "overstock",
         ("MIDRANGE_20", "Kano"): "ok"}
    )
    assert headline == "1 of 3 at risk of running out"
    assert detail == "Stockout risk: Value in Kano. Overstock: Premium in Lagos."


def test_relabel_codes():
    assert labels.relabel_codes("**PREMIUM_20** and VALUE_20") == "**Premium** and Value"
