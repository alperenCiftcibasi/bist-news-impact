"""rule_sentiment icin birim testler (saf fonksiyon, network/model yok)."""
from __future__ import annotations

import pandas as pd

from src.analysis.rule_sentiment import (
    NEGATIVE,
    NEUTRAL,
    POSITIVE,
    classify,
    score_disclosures_by_rule,
)


def test_fixed_positive_subjects():
    assert classify("Kar Payı Dağıtım İşlemlerine İlişkin Bildirim") == POSITIVE
    assert classify("Payların Geri Alınmasına İlişkin Bildirim") == POSITIVE
    assert classify("Yeni İş İlişkisi") == POSITIVE
    assert classify("Finansal Duran Varlık Edinimi") == POSITIVE


def test_fixed_negative_subjects():
    assert classify("Finansal Duran Varlık Satışı") == NEGATIVE


def test_fixed_neutral_subjects():
    assert classify("Genel Kurul İşlemlerine İlişkin Bildirim") == NEUTRAL
    assert classify("Özel Durum Açıklaması (Genel)") == NEUTRAL
    assert classify("İhraç Tavanına İlişkin Bildirim") == NEUTRAL
    assert classify("Toptan Alış Satış İşlemi Bildirimi") == NEUTRAL


def test_unknown_subject_defaults_to_neutral():
    assert classify("Yeni Bilinmeyen Kategori") == NEUTRAL


def test_none_or_empty_subject():
    assert classify(None) == NEUTRAL
    assert classify("") == NEUTRAL
    assert classify("   ") == NEUTRAL


def test_capital_change_keyword_split():
    subj = "Sermaye Artırımı - Azaltımı İşlemlerine İlişkin Bildirim"
    # Artirim -> NEG (dilution)
    assert classify(subj, "Tahsisli Sermaye Artırımına İlişkin SPK Onayı") == NEGATIVE
    assert classify(subj, "Sermaye artirim islemi tamamlandi") == NEGATIVE
    # Azaltim -> POZ
    assert classify(subj, "Sermaye Azaltımı kararı alındı") == POSITIVE
    assert classify(subj, "Sermaye azaltim islemi") == POSITIVE
    # Yok yon -> NÖT
    assert classify(subj, "Sermaye iliskili duzenleme") == NEUTRAL
    assert classify(subj, "") == NEUTRAL


def test_share_trade_keyword_split():
    subj = "Pay Alım Satım Bildirimi"
    # Satis -> NEG
    assert classify(subj, "Hisse satışına ilişkin yetkilendirme") == NEGATIVE
    assert classify(subj, "Pay satış işleminin tamamlanması") == NEGATIVE
    # Alis -> POZ
    assert classify(subj, "Pay alımı gerçekleştirildi") == POSITIVE
    assert classify(subj, "Hisse alış işlemi") == POSITIVE
    # Ikisi birden veya hicbiri -> NÖT
    assert classify(subj, "Pay alım ve satım islemleri hk.") == NEUTRAL
    assert classify(subj, "") == NEUTRAL


def test_rating_keyword_split():
    subj = "Kredi Derecelendirmesi"
    assert classify(subj, "Kredi notu yukarı yönlü güncellendi") == POSITIVE
    assert classify(subj, "Not yükseltildi") == POSITIVE
    assert classify(subj, "Kredi notu düşürüldü") == NEGATIVE
    assert classify(subj, "Aşağı yönlü revize edildi") == NEGATIVE
    # Teyit / no change -> NÖT
    assert classify(subj, "Kredi notu teyit edildi") == NEUTRAL
    assert classify(subj, "Fitch Ratings kredi derecelendirme notları hakkında") == NEUTRAL


def test_score_disclosures_by_rule_adds_column():
    df = pd.DataFrame({
        "ticker": ["A", "B", "C"],
        "subject": [
            "Kar Payı Dağıtım İşlemlerine İlişkin Bildirim",
            "Finansal Duran Varlık Satışı",
            "Bilinmeyen",
        ],
        "summary": ["x", "y", "z"],
    })
    out = score_disclosures_by_rule(df)
    assert list(out["rule_label"]) == [POSITIVE, NEGATIVE, NEUTRAL]
    # Orijinal mutate olmamali
    assert "rule_label" not in df.columns


def test_score_disclosures_by_rule_uses_summary_for_keyword_subjects():
    df = pd.DataFrame({
        "ticker": ["A", "B"],
        "subject": [
            "Sermaye Artırımı - Azaltımı İşlemlerine İlişkin Bildirim",
            "Pay Alım Satım Bildirimi",
        ],
        "summary": [
            "Sermaye artırımı tamamlandı",
            "Hisse satış sürecinin başlangıcı",
        ],
    })
    out = score_disclosures_by_rule(df)
    assert list(out["rule_label"]) == [NEGATIVE, NEGATIVE]


def test_score_disclosures_by_rule_handles_missing_summary_col():
    df = pd.DataFrame({"subject": ["Yeni İş İlişkisi", "Sermaye Artırımı - Azaltımı İşlemlerine İlişkin Bildirim"]})
    out = score_disclosures_by_rule(df)
    # Sabit POZ subject -> POZ; keyword-bazli ama summary yok -> NÖT
    assert list(out["rule_label"]) == [POSITIVE, NEUTRAL]


def test_score_disclosures_by_rule_empty():
    df = pd.DataFrame(columns=["subject", "summary"])
    out = score_disclosures_by_rule(df)
    assert out.empty
    assert "rule_label" in out.columns
