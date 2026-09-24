"""Groq (GPT-OSS 120B) memo generation (INTRO.txt §10).

The LLM writes **prose only**. Every number in the memo is computed by the
forecaster and the optimizer and interpolated into the prompt; the model is never
asked to decide a price, score sentiment, or do arithmetic. That boundary is the
whole reason an open-weight model on a free tier is safe to use here.

The prompt template below is reproduced verbatim from INTRO.txt §10. What the
template cannot carry -- per-SKU prices, the regions served, which way the naira
moved -- is appended after it, and ``check_memo`` rejects a memo that contradicts
those facts before it is saved.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence

from tobacco import config

log = logging.getLogger(__name__)

#: INTRO.txt §10 names Llama 3.3 70B. Groq shut ``llama-3.3-70b-versatile`` down
#: on 2026-08-16 (it now 404s ``model_not_found``) and names this as the
#: replacement: https://console.groq.com/docs/deprecations
MODEL = "openai/gpt-oss-120b"

#: Opens the fallback memo's body. The dashboard keys on it so it never
#: attributes a data-only memo to the model.
FALLBACK_MARKER = "[Automated memo generation unavailable"

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


#: Also appended after the template. The §10 DATA block carries one overall
#: adjustment, and on 2026-09-24 -- the first memo a model actually wrote -- it
#: turned SKU moves of -2/-3/+1% into "a uniform 1.3% reduction across all SKUs",
#: named Abuja, which the firm does not serve, called a strengthening naira a
#: depreciation, and invented an inflation-indexed tax rule. This block is the
#: set of facts and limits that memo was missing.
FACTS_TEMPLATE = """

PER-SKU DECISIONS — the overall figure above is the unweighted mean of these, not an
instruction to apply to every SKU:
{sku_lines}

Regions served (name no other place): {regions}.
{fx_line}

Recommend each SKU's own adjustment. State only facts given in this prompt: do not
invent tax rules, regulations, competitor actions or figures. Risks may be named as
possibilities, without specifics this data does not give."""

#: Nigerian cities a memo might plausibly name. Any of these outside
#: config.REGIONS is a place the firm does not serve.
NIGERIAN_CITIES: tuple[str, ...] = (
    "Abuja", "Kaduna", "Enugu", "Benin City", "Aba", "Onitsha", "Jos", "Ilorin",
    "Warri", "Calabar", "Owerri", "Uyo", "Maiduguri", "Sokoto", "Abeokuta",
    "Akure", "Asaba", "Zaria", "Yola", "Bauchi", "Lagos", "Ibadan", "Kano",
    "Port Harcourt",
)


def _as_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def _fx_direction(fx_change) -> str:
    """Spell out which way the naira moved -- the sign of NGN/USD reads backwards."""
    change = _as_float(fx_change)
    if change is None:
        return "The 7-day FX change is unavailable; do not describe its direction."
    if change < 0:
        return (
            f"NGN/USD fell {abs(change):.2f}% over 7 days: fewer naira per dollar, so "
            f"the naira STRENGTHENED (appreciated). Do not call this a depreciation."
        )
    if change > 0:
        return (
            f"NGN/USD rose {change:.2f}% over 7 days: more naira per dollar, so the "
            f"naira WEAKENED (depreciated)."
        )
    return "NGN/USD was unchanged over 7 days."


def _sku_lines(skus: Sequence[Mapping]) -> str:
    return "\n".join(
        f"- {d['sku']}: NGN {d['current_price']:,.0f} -> NGN {d['recommended_price']:,.0f} "
        f"({d['adjustment_pct']:+.1f}%), limited by {d['binding_constraint']}"
        for d in skus
    )


def build_prompt(**values) -> str:
    """Fill the template, rendering unavailable inputs as an explicit 'unavailable'.

    A missing value must never arrive as ``None`` or ``nan``: the model would
    read it as a number and reason from it. Saying "unavailable" makes the gap
    visible in the memo instead.

    ``notes`` is an optional list of provenance caveats, and ``skus`` an optional
    list of per-SKU decisions (mappings with PriceDecision's fields). Both are
    appended after the template rather than woven into it.
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

    skus = values.get("skus") or []
    if skus:
        prompt += FACTS_TEMPLATE.format(
            sku_lines=_sku_lines(skus),
            regions=", ".join(config.REGIONS),
            fx_line=_fx_direction(values.get("fx_change")),
        )
    return prompt


def _sku_stem(sku: str) -> str:
    """``PREMIUM_20`` -> ``premium``: how prose refers to a tier."""
    return sku.split("_")[0].lower()


def check_memo(memo: str, skus: Sequence[Mapping] = (), fx_change=None) -> list[str]:
    """Contradictions between ``memo`` and the facts it was given. Empty means none.

    Deliberately narrow: each check is a mistake a real memo made, tested in a
    way that should not fire on correct prose. It cannot catch every invention --
    the prompt's instruction is the first line of defence, this is the second.
    """
    issues: list[str] = []
    lowered = memo.lower()

    for d in skus:
        if _sku_stem(d["sku"]) not in lowered:
            issues.append(f"{d['sku']} is not mentioned; give each SKU its own adjustment.")

    adjustments = {round(float(d["adjustment_pct"]), 1) for d in skus}
    if len(adjustments) > 1 and re.search(
        r"\buniform\b|across all (?:skus|products|brands)|all skus by", lowered
    ):
        issues.append(
            "Describes one uniform adjustment, but the SKUs move by different amounts: "
            + ", ".join(f"{d['sku']} {d['adjustment_pct']:+.1f}%" for d in skus)
            + "."
        )

    served = {region.lower() for region in config.REGIONS}
    for city in NIGERIAN_CITIES:
        if city.lower() not in served and re.search(rf"\b{re.escape(city)}\b", memo):
            issues.append(
                f"Names {city}, which is not a region served ({', '.join(config.REGIONS)})."
            )

    change = _as_float(fx_change)
    if change is not None and change < 0:
        # Only the sentence that reports this week's move -- a Risks line saying
        # the naira *could* weaken is legitimate.
        figure = f"{abs(change):.2f}".rstrip("0").rstrip(".")
        for sentence in re.split(r"(?<=[.!?])\s+|\n", memo):
            if figure in sentence and re.search(r"depreciat|weaken", sentence, re.I):
                issues.append(
                    "Calls this week's NGN/USD fall a depreciation; fewer naira per "
                    "dollar means the naira strengthened."
                )
                break

    return issues


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
        messages = [{"role": "user", "content": prompt}]
        skus = values.get("skus") or []

        # One draft, and at most one correction. A memo that still contradicts
        # its own data after being told how is not saved: the data-only
        # fallback is honest, a confident wrong memo is not.
        for attempt in (1, 2):
            memo = _complete(client, messages)
            issues = check_memo(memo, skus, values.get("fx_change"))
            if not issues:
                log.info("Generated memo (%d chars) via %s", len(memo), MODEL)
                return memo
            log.warning("Memo draft %d failed fact checks: %s", attempt, issues)
            messages += [
                {"role": "assistant", "content": memo},
                {
                    "role": "user",
                    "content": "Rewrite the memo. It contradicts the data:\n"
                    + "\n".join(f"- {issue}" for issue in issues),
                },
            ]
        return _fallback(prompt, "the model's memo failed fact checks: " + "; ".join(issues))
    except Exception as exc:  # noqa: BLE001 - a failed memo must not fail the job
        log.error("Groq request failed: %s", exc)
        return _fallback(prompt, str(exc))


def _complete(client, messages: list[dict]) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.3,  # low: this is a factual brief, not creative writing
        # A reasoning model: its hidden reasoning is billed against this
        # budget too, so it is well above the ~900 a 400-word memo needs.
        # Low effort because the prompt already carries every number.
        max_tokens=4000,
        extra_body={"reasoning_effort": "low"},
    )
    memo = (response.choices[0].message.content or "").strip()
    if not memo:
        # Every token went on reasoning. An empty memo is not a memo.
        raise RuntimeError("model returned no memo text (reasoning used the budget)")
    return memo


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

    # Without prose the per-SKU lines are the recommendation itself; the overall
    # figure in the DATA block is only their mean.
    per_sku = ""
    if "PER-SKU DECISIONS" in prompt:
        lines = prompt.split("PER-SKU DECISIONS", 1)[-1].split("\n\nRegions served", 1)[0]
        lines = "\n".join(line for line in lines.splitlines() if line.startswith("- "))
        per_sku = f"\n\nPer-SKU decisions:\n\n{lines}"

    return (
        f"Subject: Strategic Price & Inventory Recommendation – {config.today_wat()}\n\n"
        f"{FALLBACK_MARKER}: {reason}]\n\n"
        f"The underlying figures are unaffected:\n\n{data_block}{per_sku}{provenance}\n\n"
        f"Bottom line: review the figures above on the dashboard; narrative "
        f"generation will resume once the Groq API is reachable."
    )
