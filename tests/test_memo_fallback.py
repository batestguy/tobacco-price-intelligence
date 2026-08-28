"""The §10 prompt template and the no-key fallback.

With ``GROQ_API_KEY`` unset -- which is the state of the repo today -- ``generate``
takes the fallback path on every run, so the fallback is not a degraded mode here,
it is the mode. ``groq`` is imported lazily inside ``generate`` and is never
reached below.
"""

from __future__ import annotations

import pytest

from tobacco import config
from tobacco.memo import groq

#: Reproduced from INTRO.txt §10, which CLAUDE.md requires be carried verbatim.
#: A literal copy rather than a hash: when this test fails, the diff should show
#: *what* changed in a compliance text, not just that something did.
SPEC_TEMPLATE = """You are a senior business intelligence analyst for a tobacco company in Nigeria.
Based on the following real-time data, write a 1-page strategic memo for the
Commercial Director. Keep it professional, actionable, and under 400 words.

DATA:
- Current NGN/USD exchange rate: {fx_rate} (7-day change: {fx_change}%)
- Monthly inflation rate: {inflation}%
- Consumer sentiment score (0=very negative, 1=very positive): {sentiment}
- Financial news crisis probability (0=low, 1=high): {crisis_score}
- Recommended price adjustment (overall): {price_rec}%
- Top competitor (Bohem) average price per pack: {competitor_price}
- Demand forecast trend (growing/stable/declining): {demand_trend}
- Current stock alert: {stock_alert} (low/ok/high)

INSTRUCTIONS:
1. Provide a clear recommendation on whether to adjust prices and by how much.
2. Suggest inventory actions (rebalancing or reordering) based on stock alert.
3. Mention key risks (regulatory, FX volatility, competition).
4. Propose a timeline (immediate, next 7 days, next 30 days).
5. End with a final one-sentence bottom-line recommendation.

OUTPUT FORMAT:
Subject: Strategic Price & Inventory Recommendation – [Date]

[Body with clear sections: Situation, Recommendation, Risks, Timeline]

Bottom line: [One sentence]."""


VALUES = dict(
    fx_rate="1,337.59",
    fx_change=-0.4,
    inflation=24.1,
    sentiment=0.62,
    crisis_score=0.31,
    price_rec=4.3,
    competitor_price="1,200",
    demand_trend="stable",
    stock_alert="ok",
)


def test_prompt_template_is_unmodified():
    assert groq.PROMPT_TEMPLATE == SPEC_TEMPLATE


def test_provenance_is_appended_after_the_template_never_spliced_into_it():
    """The §10 DATA block is fixed text; caveats about it must sit outside."""
    prompt = groq.build_prompt(**VALUES, notes=["inflation came from the GEM tier"])

    assert prompt.startswith(groq.PROMPT_TEMPLATE.format(**VALUES))
    assert "GEM tier" in prompt.partition("DATA PROVENANCE")[2]


def test_build_prompt_renders_missing_values_as_unavailable():
    """A gap must never arrive as None or nan: the model would reason from it."""
    prompt = groq.build_prompt(**{**VALUES, "inflation": None, "sentiment": float("nan")})

    assert "Monthly inflation rate: unavailable%" in prompt
    assert "positive): unavailable" in prompt

    # Check the substituted values, not the whole prompt: the §10 DATA block's own
    # fixed text contains both needles ("Fi-nan-cial news crisis probability").
    data_block = prompt.split("DATA:", 1)[1].split("INSTRUCTIONS:", 1)[0]
    for line in data_block.strip().splitlines():
        value = line.split(": ", 1)[-1]
        assert "None" not in value and "nan" not in value, line


def test_build_prompt_omits_the_provenance_block_when_there_are_no_notes():
    assert "DATA PROVENANCE" not in groq.build_prompt(**VALUES)
    assert "DATA PROVENANCE" not in groq.build_prompt(**VALUES, notes=[None, ""])


def test_generate_falls_back_without_an_api_key():
    memo = groq.generate(**VALUES)
    assert "GROQ_API_KEY" in memo
    assert memo.startswith("Subject: Strategic Price & Inventory Recommendation")


def test_the_fallback_carries_the_figures_the_prose_would_have_described():
    """The numbers are the valuable part and they already exist."""
    memo = groq.generate(**VALUES)

    assert "1,337.59" in memo
    assert "24.1" in memo
    assert "4.3" in memo
    assert "INSTRUCTIONS:" not in memo, "the fallback emits the DATA block, not the prompt"


def test_the_fallback_repeats_the_provenance_caveats():
    """They matter more here, not less: with no prose, the raw figures are all
    the reader gets."""
    memo = groq.generate(**VALUES, notes=["inflation is a World Bank GEM staff figure"])

    assert "Data provenance:" in memo
    assert "World Bank GEM staff figure" in memo
    assert "Do not describe" not in memo, "the instruction to the model is not for the reader"


def test_generate_does_not_import_groq_without_a_key(monkeypatch):
    """The lazy import is what lets the dashboard and the tests skip the SDK."""
    import sys

    monkeypatch.setitem(sys.modules, "groq", None)  # any use would raise
    assert groq.generate(**VALUES).startswith("Subject:")


@pytest.mark.parametrize("field", ["fx_rate", "price_rec", "stock_alert"])
def test_every_spec_field_is_substituted(field):
    prompt = groq.build_prompt(**VALUES)
    assert "{" + field + "}" not in prompt
    assert str(VALUES[field]) in prompt


def test_the_fallback_is_dated_in_west_africa_time():
    assert str(config.today_wat()) in groq.generate(**VALUES)
