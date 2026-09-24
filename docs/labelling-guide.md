# Headline labelling guide

Labels for fine-tuning FinBERT (`notebooks/finbert_transfer_learning.ipynb`). One
label per headline:

| Label | Meaning |
|---|---|
| `0` | **negative** for Nigerian business conditions |
| `1` | **neutral**: no clear direction, mixed, or not about Nigerian business conditions |
| `2` | **positive** for Nigerian business conditions |

## The question to ask

> If this headline is true, does it make operating conditions for a manufacturer
> that sells in Nigeria and imports some inputs **better, worse, or neither**?

Judge the **reported event**, not the tone of the sentence. "Naira gains despite
fears" is positive; "Record profits spark windfall-tax calls" is about a proposed
tax, so it is negative.

## Rules

1. **The headline alone.** Do not open the article or use outside knowledge of how
   the story turned out. If the headline does not state a direction, it has none.
2. **Two-sided or mixed → 1.** "Inflation eases but food prices climb" is 1.
3. **Questions, opinion, explainers, how-to, lists → 1** unless the headline itself
   asserts a directional fact ("Why the naira will keep falling" asserts one → 0).
4. **Not Nigerian and no stated Nigerian channel → 1.** "Fed holds rates" is 1;
   "Fed hike pressures naira" is 0.
5. **Scheduled or routine corporate events → 1**: AGMs, appointments, listings,
   dividend *dates*. Results with a stated direction count: profit up → 2, loss or
   profit down → 0.
6. **When unsure between a direction and 1, choose 1.** A wrong neutral costs the
   model little; a confident wrong direction teaches it the wrong thing.

## Direction by topic

| Topic | 0 (negative) | 2 (positive) |
|---|---|---|
| Naira / FX | weakens, depreciates, FX scarcity, parallel-market gap widens | strengthens, appreciates, FX liquidity improves, gap narrows |
| Inflation | rises, accelerates | falls, eases |
| Interest rates / credit | MPR hike, tighter credit, higher yields demanded by govt | rate cut, easier credit |
| Reserves | fall | rise |
| Growth / activity | GDP contracts, PMI below 50, layoffs, factory closures | GDP grows, PMI above 50, investment inflows, new plants |
| Taxes / levies / regulation | new or higher tax, levy, tariff, ban on inputs, compliance burden | tax relief, waiver, simplified rules |
| Energy | petrol/diesel/electricity price rises, supply shortage, grid collapse | price falls, supply improves |
| Crude oil | output or price falls (FX earnings drop) | output or price rises (FX earnings rise) |
| Government debt | borrowing or debt service rises | debt falls, better rating |
| Stock market (NGX) | index falls, losses | index gains |
| Security / unrest | insecurity, strikes, protests disrupting trade | resolved strikes, improved security |
| Banks / finance | failures, sanctions, liquidity stress | recapitalisation completed, stability |
| Consumer demand | purchasing power falls, spending cuts | incomes or spending rise |

Petrol and crude go in **opposite** directions on purpose: a higher crude price earns
Nigeria dollars, but a higher pump price raises every firm's costs.

## Clarifications (from round-1 disagreements)

Added after round 1. Every disagreement in that round was between a direction
and neutral, in the cases below. Round 1's labels predate these rules, and
disputed rows were dropped rather than relabelled.

- **One company's launch, partnership, or expansion with no figure → 1.** "Jetour
  expands SUV push", "Voltaa launches platform". A stated sizeable investment
  in Nigeria ("$300m renewable energy fund") → 2.
- **Enforcement against rule-breakers → 1.** Raids, sealed shops, SIM or import
  checks aimed at illegal activity. A new or higher compliance cost on *all*
  firms in a sector ("SEC raises minimum capital") → 0.
- **Strikes and strike threats count only in sectors that move goods, energy or
  money** (transport, fuel, ports, banks, power) → 0, or 2 when called off. In
  education or health → 1.
- **A price move plus an unrelated volume figure → the price move decides.**
  "Naira firms as turnover plunges" → 2. A gain qualified by weak breadth in the
  *same* market ("NGX rebounds but breadth stays negative") → 1.
- **An individual's wealth or stake → 1** unless the headline says the share price
  itself moved.

## Process used, and how to repeat it

1. Export a sample of headlines **without** the base model's score, so the score
   cannot anchor the label.
2. Two labellers work independently from this guide. On 2026-09-24 these were
   Claude Opus (main session) and a separate Claude Sonnet agent, neither shown
   the other's labels.
3. Keep only headlines where both agree. Disagreements are **dropped, not
   argued out** — a headline two careful readers split on is ambiguous, and
   ambiguous labels are exactly what the classifier should not learn from.
4. Report agreement (percent and Cohen's kappa) with the file. Kappa below about
   0.6 means the guide is unclear: fix the guide before training.

The result is `data/labels/headlines_labelled.csv`. Stats are in
`data/labels/README.md`.

## What these labels are, and are not

They are **model labels** checked by agreement between two models, not expert
human annotation. A fine-tune on them learns *this guide as two LLMs applied it*.
The notebook's "beats the base model" check measures agreement with these same
labels, so it shows the tuned model follows the guide better. It does not show
the model is right about markets.

Fine for a portfolio project; say so wherever the tuned model's accuracy is quoted.
Spot-check 20 or so labels yourself whenever you can: that is the cheapest
upgrade in label quality available to a one-person project.
