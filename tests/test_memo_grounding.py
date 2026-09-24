"""The facts appended after the §10 template, and the checks that enforce them.

Every negative case below is a sentence the first model-written memo (2026-09-24)
actually produced, lightly trimmed: a "uniform" cut when the SKUs moved -2/-3/+1%,
a region the firm does not serve, and a strengthening naira called a depreciation.
"""

from __future__ import annotations

import sys
import types

import pytest

from tobacco import config
from tobacco.memo import groq

from test_memo_fallback import VALUES

SKUS = [
    dict(sku="PREMIUM_20", current_price=1437.0, recommended_price=1408.0,
         adjustment_pct=-2.0, binding_constraint="profit_optimum"),
    dict(sku="MIDRANGE_20", current_price=1008.0, recommended_price=978.0,
         adjustment_pct=-3.0, binding_constraint="profit_optimum"),
    dict(sku="VALUE_20", current_price=669.0, recommended_price=675.0,
         adjustment_pct=1.0, binding_constraint="profit_optimum"),
]

GOOD_MEMO = (
    "Subject: Strategic Price & Inventory Recommendation\n"
    "Situation: NGN/USD is 1,327.5, down 0.10% this week, so the naira strengthened.\n"
    "Recommendation: cut PREMIUM_20 by 2.0% and MIDRANGE_20 by 3.0%; raise VALUE_20 "
    "by 1.0%. Keep stock steady in Lagos, Kano, Ibadan and Port Harcourt.\n"
    "Risks: the naira could weaken abruptly given the crisis probability.\n"
    "Bottom line: apply the per-SKU changes."
)


# ---------------------------------------------------------------------------
# the prompt
# ---------------------------------------------------------------------------


def test_the_facts_block_follows_the_verbatim_template_untouched():
    prompt = groq.build_prompt(**VALUES, skus=SKUS)
    assert prompt.startswith(groq.PROMPT_TEMPLATE.format(**VALUES))


def test_the_prompt_carries_every_sku_decision():
    prompt = groq.build_prompt(**VALUES, skus=SKUS)
    assert "PREMIUM_20: NGN 1,437 -> NGN 1,408 (-2.0%)" in prompt
    assert "MIDRANGE_20: NGN 1,008 -> NGN 978 (-3.0%)" in prompt
    assert "VALUE_20: NGN 669 -> NGN 675 (+1.0%)" in prompt


def test_the_prompt_names_the_regions_served():
    prompt = groq.build_prompt(**VALUES, skus=SKUS)
    assert ", ".join(config.REGIONS) in prompt


@pytest.mark.parametrize(
    ("fx_change", "expected"),
    [(-0.1, "STRENGTHENED"), ("-0.10", "STRENGTHENED"), (0.5, "WEAKENED"),
     (0.0, "unchanged"), (None, "unavailable"), (float("nan"), "unavailable")],
)
def test_the_prompt_spells_out_which_way_the_naira_moved(fx_change, expected):
    prompt = groq.build_prompt(**{**VALUES, "fx_change": fx_change}, skus=SKUS)
    assert expected in prompt


def test_no_facts_block_without_sku_decisions():
    assert "PER-SKU DECISIONS" not in groq.build_prompt(**VALUES)


# ---------------------------------------------------------------------------
# check_memo
# ---------------------------------------------------------------------------


def test_a_faithful_memo_passes():
    assert groq.check_memo(GOOD_MEMO, SKUS, "-0.10") == []


def test_a_uniform_cut_is_rejected_when_skus_differ():
    memo = GOOD_MEMO + "\nImplement a uniform 1.3% price reduction across all SKUs."
    assert any("uniform" in issue for issue in groq.check_memo(memo, SKUS, "-0.10"))


def test_a_uniform_cut_is_fine_when_the_skus_really_do_move_together():
    same = [{**d, "adjustment_pct": -2.0} for d in SKUS]
    memo = GOOD_MEMO + "\nApply a uniform 2% reduction."
    assert not any("uniform" in issue for issue in groq.check_memo(memo, same, "-0.10"))


def test_a_region_the_firm_does_not_serve_is_rejected():
    memo = GOOD_MEMO + "\nReinforce high-growth regions (e.g., Lagos, Abuja)."
    issues = groq.check_memo(memo, SKUS, "-0.10")
    assert any("Abuja" in issue for issue in issues)
    assert not any("Lagos" in issue for issue in issues)


def test_calling_a_strengthening_naira_a_depreciation_is_rejected():
    memo = GOOD_MEMO + "\nFX: NGN/USD = 1,327.5 (-0.10% over 7 days), modest depreciation."
    assert any("strengthened" in issue for issue in groq.check_memo(memo, SKUS, "-0.10"))


def test_a_risk_that_the_naira_could_weaken_is_not_flagged():
    """GOOD_MEMO says the naira 'could weaken'; that is a risk, not a report."""
    assert groq.check_memo(GOOD_MEMO, SKUS, "-0.10") == []


def test_a_missing_sku_is_rejected():
    memo = GOOD_MEMO.replace("raise VALUE_20 by 1.0%", "hold the rest")
    assert any("VALUE_20" in issue for issue in groq.check_memo(memo, SKUS, "-0.10"))


# ---------------------------------------------------------------------------
# generate: one correction, then the honest fallback
# ---------------------------------------------------------------------------


def _scripted_groq(drafts):
    """A stand-in ``groq`` module returning ``drafts`` in order, recording each call."""
    calls = []
    queue = list(drafts)

    def create(**kwargs):
        calls.append(kwargs["messages"])
        message = types.SimpleNamespace(content=queue.pop(0))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    completions = types.SimpleNamespace(create=create)
    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))
    return types.SimpleNamespace(Groq=lambda api_key: client), calls


@pytest.fixture
def keyed(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    def install(drafts):
        module, calls = _scripted_groq(drafts)
        monkeypatch.setitem(sys.modules, "groq", module)
        return calls

    return install


BAD_MEMO = GOOD_MEMO + "\nReinforce Abuja."


def test_a_faithful_first_draft_is_returned_after_one_call(keyed):
    calls = keyed([GOOD_MEMO])
    assert groq.generate(**{**VALUES, "fx_change": "-0.10"}, skus=SKUS) == GOOD_MEMO
    assert len(calls) == 1


def test_a_bad_draft_is_corrected_once_with_the_issues_named(keyed):
    calls = keyed([BAD_MEMO, GOOD_MEMO])
    assert groq.generate(**{**VALUES, "fx_change": "-0.10"}, skus=SKUS) == GOOD_MEMO
    assert len(calls) == 2
    assert "Abuja" in calls[1][-1]["content"], "the correction says what was wrong"


def test_two_bad_drafts_fall_back_to_the_data_only_memo(keyed):
    calls = keyed([BAD_MEMO, BAD_MEMO])
    memo = groq.generate(**{**VALUES, "fx_change": "-0.10"}, skus=SKUS)
    assert len(calls) == 2
    assert groq.FALLBACK_MARKER in memo
    assert "failed fact checks" in memo
    assert "1,337.59" in memo


def test_the_fallback_carries_the_per_sku_decisions():
    """Without prose, the per-SKU lines are the recommendation."""
    memo = groq.generate(**VALUES, skus=SKUS)  # no key: straight to the fallback
    assert "VALUE_20: NGN 669 -> NGN 675 (+1.0%)" in memo
