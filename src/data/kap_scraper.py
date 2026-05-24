"""KAP'tan (Kamuyu Aydinlatma Platformu) hisse bazli ozel durum aciklamalarini (ODA) ceker.

`pykap` kutuphanesini kullanir; sirket ID'sini cozer ve byCriteria endpoint'ine
disclosureClass='ODA' filtresiyle istek atar. Sonuclari JSONL olarak kaydeder.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import requests
from pykap.bist import BISTCompany

from src.config import BIST_TIMEZONE, KAP_DIR, TICKERS

logger = logging.getLogger(__name__)

BYCRITERIA_URL = "https://www.kap.org.tr/tr/api/disclosure/members/byCriteria"
DISCLOSURE_URL = "https://www.kap.org.tr/tr/Bildirim/{index}"

# KAP publishDate formati: "08.05.2026 20:39:08"
_KAP_DATE_FORMAT = "%d.%m.%Y %H:%M:%S"


@dataclass(frozen=True)
class Disclosure:
    """Tek bir KAP bildirimi (temiz/normalize edilmis)."""

    ticker: str
    disclosure_index: int
    publish_datetime: str  # ISO 8601, BIST timezone'da
    subject: str
    summary: str
    disclosure_class: str
    disclosure_category: str
    stock_codes: str | None
    attachment_count: int
    url: str


def _parse_publish_datetime(raw: str) -> str:
    """KAP'in 'dd.MM.yyyy HH:mm:ss' formatini BIST TZ ISO 8601'e cevirir."""
    naive = datetime.strptime(raw, _KAP_DATE_FORMAT)
    # KAP zaten Istanbul saatiyle yayinliyor; biz de buna gore localize ediyoruz.
    # tzinfo'yu pandas/zoneinfo kullanmadan da set edebiliriz.
    from zoneinfo import ZoneInfo
    return naive.replace(tzinfo=ZoneInfo(BIST_TIMEZONE)).isoformat()


def parse_disclosure(raw: dict, ticker: str) -> Disclosure:
    """KAP API'sinden gelen raw dict'i Disclosure'a cevirir."""
    return Disclosure(
        ticker=ticker,
        disclosure_index=int(raw["disclosureIndex"]),
        publish_datetime=_parse_publish_datetime(raw["publishDate"]),
        subject=raw.get("subject") or "",
        summary=raw.get("summary") or "",
        disclosure_class=raw.get("disclosureClass") or "",
        disclosure_category=raw.get("disclosureCategory") or "",
        stock_codes=raw.get("stockCodes"),
        attachment_count=int(raw.get("attachmentCount") or 0),
        url=DISCLOSURE_URL.format(index=raw["disclosureIndex"]),
    )


def fetch_disclosures(
    ticker: str,
    from_date: date,
    to_date: date,
    disclosure_class: str = "ODA",
    timeout: int = 30,
) -> list[Disclosure]:
    """Verilen hisse icin tarih araliginda KAP bildirimlerini ceker.

    Args:
        ticker: BIST hissesi (orn. 'THYAO').
        from_date / to_date: Yayinlanma tarihi araligi (inclusive).
        disclosure_class: KAP bildirim sinifi. Varsayilan 'ODA' (Ozel Durum Aciklamasi).
        timeout: HTTP timeout (saniye).

    Returns:
        Disclosure listesi (publishDate'e gore sirali, yeniden eskiye).
    """
    # pykap company ID cozumlemesi icin BISTCompany kullaniyoruz
    company = BISTCompany(ticker=ticker)

    payload = {
        "fromDate": str(from_date),
        "toDate": str(to_date),
        "disclosureClass": disclosure_class,
        "subjectList": [],
        "mkkMemberOidList": [company.company_id],
        "inactiveMkkMemberOidList": [],
        "bdkMemberOidList": [],
        "fromSrc": False,
        "disclosureIndexList": [],
    }
    response = requests.post(BYCRITERIA_URL, json=payload, timeout=timeout)
    response.raise_for_status()
    raw_items = response.json()

    return [parse_disclosure(r, ticker=ticker) for r in raw_items]


def save_disclosures(disclosures: list[Disclosure], output_path: Path) -> None:
    """Disclosure listesini JSONL formatinda kaydeder (her satir bir bildirim)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for d in disclosures:
            f.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    to_date = datetime.today().date()
    from_date = to_date - timedelta(days=180)

    logger.info("KAP scraping: %s -> %s", from_date, to_date)
    total = 0
    for short_name in TICKERS:
        logger.info("Cekiliyor: %s", short_name)
        try:
            disclosures = fetch_disclosures(short_name, from_date, to_date)
        except Exception as e:
            logger.error("  HATA %s: %s", short_name, e)
            continue
        if not disclosures:
            logger.warning("  Bildirim yok: %s", short_name)
            continue
        out = KAP_DIR / f"{short_name}.jsonl"
        save_disclosures(disclosures, out)
        logger.info("  -> %d bildirim kaydedildi: %s", len(disclosures), out)
        total += len(disclosures)

    logger.info("Tamamlandi: %d toplam bildirim, %d hisse.", total, len(TICKERS))


if __name__ == "__main__":
    main()
