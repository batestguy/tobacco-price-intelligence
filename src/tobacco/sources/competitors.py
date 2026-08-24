"""Competitor retail prices from Nigerian marketplaces (INTRO.txt §1).

Scrapes public search-result pages on Jumia and Konga for the tracked competitor
brands. This is the most fragile source in the project and is expected to be:

* **Partially empty.** Tobacco listings come and go from marketplaces, and the
  brands in ``config.COMPETITOR_BRANDS`` may not be listed at all on a given day.
  An empty result is a normal outcome, not a failure -- downstream, the optimizer
  detects missing competitor data and reports its price-ceiling constraint as
  inactive rather than inventing a number.
* **Selector-sensitive.** Both sites restructure their markup periodically.
  Parsing is therefore layered: named CSS selectors first, then a generic
  "find the ₦ amount near a product title" pass.

Marketplace listings are national, so ``region`` is recorded as ``National``.
Region-level competitor pricing would require field survey data this project
does not have, and fabricating it would poison the optimizer's price ceiling.
"""

from __future__ import annotations

import logging
import re
from urllib.parse import quote_plus

import pandas as pd
from bs4 import BeautifulSoup

from tobacco import config
from tobacco.sources import _http

log = logging.getLogger(__name__)

COLUMNS = ["date", "brand", "price", "region", "source", "product_title", "url"]

NATIONAL = "National"

#: Below this, a listing is a single stick or an accessory, not a pack;
#: above it, a carton or a mispriced bundle. Both would skew the price index.
MIN_PACK_PRICE_NGN = 200.0
MAX_PACK_PRICE_NGN = 20000.0

PRICE_RE = re.compile(r"(?:₦|NGN|N)\s*([\d,]+(?:\.\d{1,2})?)", re.IGNORECASE)


def _parse_price(text: str) -> float | None:
    match = PRICE_RE.search(text or "")
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None


def _plausible(price: float | None) -> bool:
    return price is not None and MIN_PACK_PRICE_NGN <= price <= MAX_PACK_PRICE_NGN


def _matches_brand(title: str, brand: str) -> bool:
    """Guard against search engines returning loosely related products.

    'Time' in particular matches watches, so require the brand as a whole word.
    """
    return re.search(rf"\b{re.escape(brand)}\b", title or "", re.IGNORECASE) is not None


def _scrape_jumia(brand: str) -> list[dict]:
    url = f"https://www.jumia.com.ng/catalog/?q={quote_plus(brand)}"
    response = _http.get(url)
    if response is None:
        return []

    soup = BeautifulSoup(response.text, "lxml")
    results = []
    for card in soup.select("article.prd, article.c-prd"):
        title_el = card.select_one(".name, h3.name")
        price_el = card.select_one(".prc, div.prc")
        if not title_el or not price_el:
            continue
        title = title_el.get_text(strip=True)
        if not _matches_brand(title, brand):
            continue
        price = _parse_price(price_el.get_text(strip=True))
        if not _plausible(price):
            continue
        link = card.select_one("a[href]")
        results.append(
            {
                "brand": brand,
                "price": price,
                "source": "jumia",
                "product_title": title[:200],
                "url": ("https://www.jumia.com.ng" + link["href"]) if link else url,
            }
        )
    return results


def _scrape_konga(brand: str) -> list[dict]:
    """Konga renders results client-side, so the static HTML often has nothing.

    Kept as a secondary source: when server-rendered markup *is* present this
    works, and when it is not the generic pass below simply finds no pairs.
    """
    url = f"https://www.konga.com/search?search={quote_plus(brand)}"
    response = _http.get(url)
    if response is None:
        return []

    soup = BeautifulSoup(response.text, "lxml")
    results = []
    for card in soup.select("li[class*='product'], div[class*='product-block']"):
        text = card.get_text(" ", strip=True)
        if not _matches_brand(text, brand):
            continue
        price = _parse_price(text)
        if not _plausible(price):
            continue
        link = card.select_one("a[href]")
        href = link["href"] if link else ""
        results.append(
            {
                "brand": brand,
                "price": price,
                "source": "konga",
                "product_title": text[:200],
                "url": ("https://www.konga.com" + href) if href.startswith("/") else (href or url),
            }
        )
    return results


def fetch() -> pd.DataFrame:
    """Today's competitor prices across all tracked brands and marketplaces."""
    today = pd.Timestamp(config.today_wat())
    rows: list[dict] = []

    for brand in config.COMPETITOR_BRANDS:
        for scraper in (_scrape_jumia, _scrape_konga):
            try:
                found = scraper(brand)
            except Exception as exc:  # noqa: BLE001 - one broken site must not
                log.warning("%s failed for %r: %s", scraper.__name__, brand, exc)
                continue      # take down the rest of the scrape
            rows.extend(found)

    if not rows:
        log.warning(
            "No competitor listings found for any of %s. This is a normal "
            "outcome (tobacco listings are intermittent); the optimizer will "
            "treat its price ceiling as inactive today.",
            ", ".join(config.COMPETITOR_BRANDS),
        )
        return pd.DataFrame(columns=COLUMNS)

    frame = pd.DataFrame(rows)
    frame["date"] = today
    frame["region"] = NATIONAL

    # Several listings per brand per site is normal (different sellers/bundles).
    # The dataset key is one price per brand/region/source/day, so take the
    # median -- robust to the one absurd bundle listing that survives the bounds.
    frame = (
        frame.groupby(["date", "brand", "region", "source"], as_index=False)
        .agg(
            price=("price", "median"),
            product_title=("product_title", "first"),
            url=("url", "first"),
            listings=("price", "size"),
        )
    )
    log.info(
        "Competitor prices: %d brand/source pair(s) from %d listing(s)",
        len(frame), int(frame["listings"].sum()),
    )
    return frame.drop(columns=["listings"])[COLUMNS]
