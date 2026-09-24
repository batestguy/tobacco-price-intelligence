# Headline sentiment labels

`headlines_labelled.csv`: training labels for the FinBERT fine-tune
(`notebooks/finbert_transfer_learning.ipynb`). Columns: `id` (the `news_articles`
key), `headline`, `url`, `source`, `label` (`0` negative, `1` neutral, `2` positive,
for Nigerian business conditions). Headline and URL only, never article text,
the same rule as `news_articles`.

## Round 1: 2026-09-24

| | |
|---|---|
| Sample | 383 unique headlines from `news_articles` 2026-08/09, up to 120 per fifth of the base model's score range, shuffled |
| Annotators | A = Claude Opus 5.5 (main session), B = Claude Sonnet 5 (separate agent). Each was blind to the base model's score and to the other's labels |
| Guide | `docs/labelling-guide.md` as committed with this file |
| Agreement | 344 / 383 = **89.8%**, Cohen's kappa **0.787** |
| Kept | agreed rows only: **52 negative, 245 neutral, 47 positive** |
| Dropped | 39 disagreements, not adjudicated |

Confusion matrix (rows A, columns B):

|     | 0  | 1   | 2  |
|-----|----|-----|----|
| 0   | 52 | 2   | 0  |
| 1   | 22 | 245 | 14 |
| 2   | 0  | 1   | 47 |

There were no 0-vs-2 disagreements. Every split was between a direction and
neutral, and A leaned neutral. Three patterns account for most of them, and the
guide now has a rule for each (its "Clarifications" section):

- one company's launch or expansion (Jetour, Voltaa, Finchglow): A said 1, B said 2;
- enforcement or compliance crackdowns (NCC SIM registration, LAWMA, NAFDAC,
  port E-Call-Up): A said 1, B said 0;
- strike threats outside trade and transport (ASUU): A said 1, B said 0.

## What these are, and are not

These are **model labels**, checked by agreement between two models. They are
not expert human annotation. The fine-tune learns the guide as these two models
applied it, and the notebook's "tuned beats base" check measures agreement with
the same labels, not market truth. Quote any accuracy with that caveat.

The classes are imbalanced (71% neutral). The notebook's split is stratified, but
with ~10 positives and ~10 negatives in the 20% eval split, accuracy will be
noisy. **Macro-F1** is the fairer headline number, and more labelled
negatives and positives are the best next improvement.

## Adding a round

1. Export unlabelled headlines, excluding the ids already in this file, with
   **only** the number and headline visible to annotators.
2. Two annotators apply `docs/labelling-guide.md` independently.
3. Append agreed rows only, and add a "Round N" section here with the same stats.
   If kappa is below 0.6, fix the guide first and relabel.
