---
base_model: ProsusAI/finbert
language: en
pipeline_tag: text-classification
tags:
  - finbert
  - financial-sentiment
  - nigeria
  - news-headlines
---

# finbert-ng-financial

[ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) fine-tuned on Nigerian
business-news headlines, labelled for what each headline means for **Nigerian
business conditions**: negative, neutral or positive.

It scores the daily news feed of
[tobacco-price-intelligence](https://github.com/batestguy/tobacco-price-intelligence),
an independent portfolio project on free-tier infrastructure. That project is not
affiliated with any tobacco company and uses synthetic sales data. Its dashboard
shows the mean negative probability per day as a "news crisis score".

## Labels: `0` negative, `1` neutral, `2` positive

Same three classes and `id2label` as the base model, so it drops in wherever
ProsusAI/finbert is used.

## Training data

344 headlines from Nigerian business and news RSS feeds, August and September 2026.
Headline text only; no article bodies.

**The labels were made by models, not people.** Two Claude models labelled the
same 383 headlines independently, each blind to the other and to the base model's
score, following a written guide. Only the 344 on which they agreed were kept
(89.8% agreement, Cohen's kappa 0.787): 52 negative, 245 neutral, 47 positive. So
the fine-tune learns the guide as those two models read it. It is not expert human
annotation. The labels, guide and agreement statistics are in the source repo under
`data/labels/` and `docs/labelling-guide.md`.

## Training

One run on a Kaggle T4, 2026-09-26. 80/20 stratified split (`random_state=42`):
275 training and 69 evaluation headlines. The notebook is
`notebooks/finbert_transfer_learning.ipynb` in the source repo.

## Evaluation

Accuracy on the 69 held-out headlines, against the agreed labels:

| Model | Accuracy |
|---|---|
| ProsusAI/finbert (base) | 46.4% (32 of 69) |
| Always predict "neutral" | 71.0% (49 of 69) |
| **finbert-ng-financial** | **76.8% (53 of 69)** |

Read the middle row before the last one. Most headlines are neutral, so a model
that never commits scores 71%. The fine-tune beats that by four headlines. Its
wide margin over the base model comes largely from neutral headlines: the base
model got 37 wrong, and at least 17 of those must be headlines the guide calls
neutral, since only 20 are not. With 69 evaluation examples the 95% confidence
interval is wide, roughly ±10 points.

## Intended use and limits

- Coarse daily aggregates of a news feed, as in the source project. Not for
  judging individual headlines, and not for trading or investment decisions.
- English headlines about Nigerian business. Outside that domain, expect the base
  model's behaviour or worse.
- The labels encode one written guide's view of what is good or bad for business.
  They are not neutral facts.

## Licence

The fine-tuned weights inherit the terms of the base model,
[ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert).
