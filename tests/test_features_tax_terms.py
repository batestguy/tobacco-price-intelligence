"""The TAX_TERMS regex behind §4's tax-change dummy and §7's rule 4.

There is no free machine-readable gazette feed, so this pattern *is* the tax
signal. What it does and does not match is undocumented anywhere else, which is
the reason to pin it.
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from tobacco.features.build import TAX_TERMS


@pytest.mark.parametrize(
    "headline",
    [
        "FG raises excise duty on tobacco",
        "New VAT regime takes effect in January",
        "Reps debate value-added tax reform",
        "Senate reviews value added tax exemptions",
        "Import tariff on beverages doubled",
        "States impose a new levy on retailers",
        "Customs duty waiver extended",
        "Analysts expect a tax hike in the budget",
        "Tax increase looms for manufacturers",
        "Sin tax proposal returns to the floor",
    ],
)
def test_tax_terms_matches_an_excise_signal(headline):
    assert TAX_TERMS.search(headline) is not None


@pytest.mark.parametrize(
    "headline",
    [
        "Nigeria plans new tax measures",     # bare "tax" is not enough
        "Taxi drivers protest fuel prices",
        "Vatican envoy visits Abuja",
        "Naira steady against the dollar",
    ],
)
def test_tax_terms_does_not_match(headline):
    assert TAX_TERMS.search(headline) is None


def test_tax_terms_does_not_match_the_bare_word_tax():
    """Only `tax hike`, `tax increase`, `sin tax` and `value(-)added tax` qualify.

    Called out separately because it is the surprising half of the rule: a
    headline can be entirely about taxation and not set the dummy.
    """
    assert TAX_TERMS.search("Nigeria reviews tax policy") is None
    assert TAX_TERMS.search("Nigeria reviews tax hike policy") is not None


def test_tax_terms_is_case_insensitive():
    assert TAX_TERMS.search("EXCISE DUTY RAISED") is not None
    assert TAX_TERMS.search("vat") is not None


def test_tax_terms_requires_word_boundaries():
    """`vat` carries an explicit \\b so it does not fire on `vatican` or `private`."""
    assert TAX_TERMS.search("private sector") is None
    assert TAX_TERMS.search("excisement") is None


def test_tax_terms_does_not_match_plurals():
    """Undocumented and arguably a gap: `\\b` after each term excludes the plural.

    Pinned as observed behaviour, not endorsed -- widening it is a signal-quality
    decision that should be made deliberately, with the false-positive rate in view.
    """
    assert TAX_TERMS.search("New tariffs announced") is None
    assert TAX_TERMS.search("New tariff announced") is not None


def test_tax_terms_emits_no_pandas_group_warning():
    """Pins commit 5a0cea9's `(` -> `(?:` fix.

    A capturing group makes ``Series.str.contains`` warn on every call -- once per
    scrape row in the Actions log -- and the warning is pandas telling you the
    pattern is being used in a way it does not expect.
    """
    assert TAX_TERMS.groups == 0

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pd.Series(["FG raises excise duty", "Naira steady"]).str.contains(TAX_TERMS)

    assert not [w for w in caught if "match groups" in str(w.message)], (
        f"pandas warned about the pattern: {[str(w.message) for w in caught]}"
    )
