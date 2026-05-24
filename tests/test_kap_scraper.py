"""kap_scraper icin ag cagrisi yapmayan birim testler."""
from __future__ import annotations

import json
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from src.data import kap_scraper
from src.data.kap_scraper import Disclosure, parse_disclosure, save_disclosures


SAMPLE_RAW = {
    "publishDate": "08.05.2026 20:39:08",
    "fundCode": None,
    "kapTitle": "TURK HAVA YOLLARI A.O.",
    "isOldKap": False,
    "disclosureClass": "ODA",
    "disclosureType": "ODA",
    "disclosureCategory": "ODA",
    "summary": "Nisan 2026 Trafik Sonuclari",
    "subject": "Ozel Durum Aciklamasi (Genel)",
    "relatedStocks": None,
    "year": None,
    "ruleType": "-",
    "period": None,
    "disclosureIndex": 1603784,
    "isLate": False,
    "stockCodes": "THYAO",
    "hasMultiLanguageSupport": True,
    "attachmentCount": 2,
    "modifyStatus": None,
}


def test_parse_disclosure_converts_all_fields():
    d = parse_disclosure(SAMPLE_RAW, ticker="THYAO")

    assert isinstance(d, Disclosure)
    assert d.ticker == "THYAO"
    assert d.disclosure_index == 1603784
    assert d.subject == "Ozel Durum Aciklamasi (Genel)"
    assert d.summary == "Nisan 2026 Trafik Sonuclari"
    assert d.disclosure_class == "ODA"
    assert d.disclosure_category == "ODA"
    assert d.stock_codes == "THYAO"
    assert d.attachment_count == 2
    assert d.url == "https://www.kap.org.tr/tr/Bildirim/1603784"


def test_parse_disclosure_datetime_is_istanbul_tz():
    d = parse_disclosure(SAMPLE_RAW, ticker="THYAO")
    # ISO 8601 format, +03:00 (Europe/Istanbul) bekleniyor
    assert d.publish_datetime.startswith("2026-05-08T20:39:08")
    assert "+03:00" in d.publish_datetime


def test_parse_disclosure_handles_null_summary_and_subject():
    raw = {**SAMPLE_RAW, "summary": None, "subject": None, "attachmentCount": None}
    d = parse_disclosure(raw, ticker="THYAO")
    assert d.summary == ""
    assert d.subject == ""
    assert d.attachment_count == 0


def test_save_disclosures_writes_jsonl(tmp_path):
    disclosures = [
        parse_disclosure(SAMPLE_RAW, ticker="THYAO"),
        parse_disclosure({**SAMPLE_RAW, "disclosureIndex": 1603785}, ticker="THYAO"),
    ]
    out = tmp_path / "THYAO.jsonl"
    save_disclosures(disclosures, out)

    assert out.exists()
    lines = out.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2

    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["disclosure_index"] == 1603784
    assert parsed[1]["disclosure_index"] == 1603785
    assert parsed[0]["ticker"] == "THYAO"


def test_save_disclosures_creates_parent_directories(tmp_path):
    out = tmp_path / "deep" / "nested" / "THYAO.jsonl"
    save_disclosures([parse_disclosure(SAMPLE_RAW, ticker="THYAO")], out)
    assert out.exists()


def test_fetch_disclosures_passes_correct_payload():
    """fetch_disclosures, dogru company_id ve disclosureClass'i POST eder."""
    fake_company = MagicMock()
    fake_company.company_id = "FAKE_COMPANY_ID_123"

    fake_response = MagicMock()
    fake_response.json.return_value = [SAMPLE_RAW]
    fake_response.raise_for_status.return_value = None

    with patch.object(kap_scraper, "BISTCompany", return_value=fake_company), \
         patch.object(kap_scraper.requests, "post", return_value=fake_response) as mock_post:
        result = kap_scraper.fetch_disclosures(
            ticker="THYAO",
            from_date=date(2025, 11, 1),
            to_date=date(2026, 5, 1),
        )

    # API'ye giden payload'i kontrol et
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args.kwargs
    payload = call_kwargs["json"]

    assert payload["fromDate"] == "2025-11-01"
    assert payload["toDate"] == "2026-05-01"
    assert payload["disclosureClass"] == "ODA"
    assert payload["mkkMemberOidList"] == ["FAKE_COMPANY_ID_123"]

    assert len(result) == 1
    assert result[0].ticker == "THYAO"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
