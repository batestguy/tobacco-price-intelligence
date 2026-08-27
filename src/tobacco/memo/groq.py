"""Groq (Llama 3.3 70B) memo generation (INTRO.txt §10).

The LLM writes **prose only**. Every number in the memo is computed by the
forecaster and the optimizer and interpolated into the prompt; the model is never
asked to decide a price, score sentiment, or do arithmetic. That boundary is the
whole reason a 70B model on a free tier is safe to use here.

The prompt template below is reproduced verbatim from INTRO.txt §10.
"""

from __future__ import annotations

import logging

from tobacco import config

log = logging.getLogger(__name__)

MODEL = "llama-3.3-70b-versatile"

#: Verbatim from INTRO.txt §10. Do not reword -- the spec calls it "copy-paste
#: ready" and the output format below is what the dashboard renders.
PROMPT_TEMPLATE = """You are a senior business intelligence analyst for a tobacco company in Nigeria.
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

#: Appended *after* the verbatim template, never spliced into it.
#:
#: The §10 DATA block is fixed text and it labels inflation "Monthly". When the
#: figure actually came from an annual tier -- which is the normal case today,
#: see CLAUDE.md departure 4 -- that label overstates it. Rewording the template
#: is not allowed and silently shipping the wrong basis is worse, so the caveat
#: goes here, outside the mandated text.
PROVENANCE_TEMPLATE = """

DATA PROVENANCE — read before writing:
{notes}

Do not describe any figure above as more current or more granular than its
basis here allows, and surface these caveats in the Risks section."""


def build_prompt(**values) -> str:
    """Fill the template, rendering unavailable inputs as an explicit 'unavailable'.

    A missing value must never arrive as ``None`` or ``nan``: the model would
    read it as a number and reason from it. Saying "unavailable" makes the gap
    visible in the memo instead.

    ``notes`` is an optional list of provenance caveats, appended after the
    template rather than woven into it.
    """
    fields = {
        "fx_rate", "fx_change", "inflation", "sentiment", "crisis_score",
        "price_rec", "competitor_price", "demand_trend", "stock_alert",
    }
    filled = {}
    for field in fields:
        value = values.get(field)
        filled[field] = "unavailable" if value is None or value != value else value

    prompt = PROMPT_TEMPLATE.format(**filled)

    notes = [note for note in (values.get("notes") or []) if note]
    if notes:
        prompt += PROVENANCE_TEMPLATE.format(
            notes="\n".join(f"- {note}" for note in notes)
        )
    return prompt


def generate(**values) -> str:
    """Generate the memo. Returns a readable fallback if Groq is unreachable."""
    prompt = build_prompt(**values)

    try:
        api_key = config.require("GROQ_API_KEY")
    except config.MissingSecret as exc:
        log.warning("Groq not configured: %s", exc)
        return _fallback(prompt, "GROQ_API_KEY is not set")

    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # low: this is a factual brief, not creative writing
            max_tokens=900,   # ~400 words plus headers, per the spec's limit
        )
        memo = response.choices[0].message.content.strip()
        log.info("Generated memo (%d chars) via %s", len(memo), MODEL)
        return memo
    except Exception as exc:  # noqa: BLE001 - a failed memo must not fail the job
        log.error("Groq request failed: %s", exc)
        return _fallback(prompt, str(exc))


def _fallback(prompt: str, reason: str) -> str:
    """Emit the underlying data when the LLM is unavailable.

    The numbers are the valuable part and they already exist; only the prose is
    missing. Returning them plainly beats returning an error string.
    """
    data_block = prompt.split("DATA:", 1)[-1].split("INSTRUCTIONS:", 1)[0].strip()

    # The caveats matter more here than in the LLM path, not less: with no prose
    # to qualify them, the raw figures are all the reader gets.
    provenance = ""
    if "DATA PROVENANCE" in prompt:
        notes = prompt.split("DATA PROVENANCE — read before writing:", 1)[-1]
        notes = notes.split("\n\nDo not describe", 1)[0].strip()
        provenance = f"\n\nData provenance:\n\n{notes}"

    return (
        f"Subject: Strategic Price & Inventory Recommendation – {config.today_wat()}\n\n"
        f"[Automated memo generation unavailable: {reason}]\n\n"
        f"The underlying figures are unaffected:\n\n{data_block}{provenance}\n\n"
        f"Bottom line: review the figures above on the dashboard; narrative "
        f"generation will resume once the Groq API is reachable."
    )
