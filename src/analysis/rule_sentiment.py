"""Kural-bazli sentiment: KAP subject + summary -> positive/negative/neutral.

savasy BERT modelinin domain-mismatch sebebiyle yetersiz kalmasinin uzerine
(bkz. notebooks/03_sentiment.ipynb), KAP'in subject taxonomisi uzerinden
deterministik bir siniflandirma:

- 4 subject sabit POZ (yatirimci icin iyi haber): kar payi, geri alim,
  yeni is, varlik edinimi
- 1 subject sabit NEG: varlik satisi
- 3 subject keyword-bazli (summary'e bakar): sermaye art/azalt, pay al/sat,
  kredi derecelendirme
- 12 subject sabit NÖT: rutin/yasal bildirimler (gen kurul, ihrac tavani, ...)
- Bilinmeyen subject -> NÖT (gelecekte yeni KAP kategorisi gelirse guvenli default)

Yorumlanabilir, deterministik, model gerektirmez. Tradeoff: KAP subject
sinifina baglidir; "Ozel Durum Aciklamasi (Genel)" gibi heterojen baslıklar
(verinin %30'u) nötrde sıkışır.
"""
from __future__ import annotations

import pandas as pd

POSITIVE = "positive"
NEGATIVE = "negative"
NEUTRAL = "neutral"

# Subject -> sabit sentiment etiketleri
_FIXED_POSITIVE_SUBJECTS = {
    "Kar Payı Dağıtım İşlemlerine İlişkin Bildirim",
    "Payların Geri Alınmasına İlişkin Bildirim",
    "Yeni İş İlişkisi",
    "Finansal Duran Varlık Edinimi",
}

_FIXED_NEGATIVE_SUBJECTS = {
    "Finansal Duran Varlık Satışı",
}

_FIXED_NEUTRAL_SUBJECTS = {
    "Özel Durum Açıklaması (Genel)",
    "Pay Dışında Sermaye Piyasası Aracı İşlemlerine İlişkin Bildirim (Faiz İçeren)",
    "Genel Kurul İşlemlerine İlişkin Bildirim",
    "İhraç Tavanına İlişkin Bildirim",
    "Bağımsız Denetim Kuruluşunun Belirlenmesi",
    "Yatırım Kuruluşu Varant - Sertifika - Senetlerine İlişkin Bildirim",
    "Kurumsal Yönetim İlkelerine Uyum Derecelendirmesi",
    "Toptan Alış Satış İşlemi Bildirimi",
    "Haber ve Söylentilere İlişkin Açıklama",
    "Yönetim Kurulu Komiteleri",
    "Geleceğe Dönük Değerlendirmeler",
    "İlişkili Taraf İşlemleri",
}


def _classify_capital(summary: str) -> str:
    """Sermaye Artirimi/Azaltimi: summary keyword'une gore."""
    s = summary.lower()
    has_increase = "artırım" in s or "artirim" in s
    has_decrease = "azaltım" in s or "azaltim" in s
    if has_increase and not has_decrease:
        return NEGATIVE  # dilution
    if has_decrease and not has_increase:
        return POSITIVE  # pay basina deger artar
    return NEUTRAL


def _classify_share_trade(summary: str) -> str:
    """Pay Alim Satim: summary keyword'une gore."""
    s = summary.lower()
    # "satış" / "satım" / "satıl" / "satıs" patterns
    has_sell = any(k in s for k in ("satış", "satım", "satıl", "satis"))
    # "alım" / "alış" / "alın" patterns
    has_buy = any(k in s for k in ("alım", "alış", "alın", "alim", "alis"))
    if has_sell and not has_buy:
        return NEGATIVE  # insider satis
    if has_buy and not has_sell:
        return POSITIVE  # insider alis
    return NEUTRAL


def _classify_rating(summary: str) -> str:
    """Kredi Derecelendirmesi: not yon keyword'leri."""
    s = summary.lower()
    has_up = any(k in s for k in ("yukarı", "yukseltil", "yükseltil", "yukseldi", "yükseldi"))
    has_down = any(k in s for k in ("aşağı", "asagi", "düşürül", "dusurul", "indir", "düşüş", "dusus"))
    if has_up and not has_down:
        return POSITIVE
    if has_down and not has_up:
        return NEGATIVE
    return NEUTRAL  # teyit/no change -> notr


def classify(subject: str | None, summary: str | None = None) -> str:
    """Bildirim subject (+ opsiyonel summary) icin sentiment etiketi.

    Args:
        subject: KAP bildirim baslik kategorisi.
        summary: Bildirim ozet metni (keyword-bazli kararlar icin).

    Returns:
        'positive' | 'negative' | 'neutral'. Bilinmeyen subject -> 'neutral'.
    """
    if subject is None or not subject.strip():
        return NEUTRAL

    subj = subject.strip()
    summ = (summary or "").strip()

    if subj in _FIXED_POSITIVE_SUBJECTS:
        return POSITIVE
    if subj in _FIXED_NEGATIVE_SUBJECTS:
        return NEGATIVE
    if subj in _FIXED_NEUTRAL_SUBJECTS:
        return NEUTRAL

    if subj == "Sermaye Artırımı - Azaltımı İşlemlerine İlişkin Bildirim":
        return _classify_capital(summ)
    if subj == "Pay Alım Satım Bildirimi":
        return _classify_share_trade(summ)
    if subj == "Kredi Derecelendirmesi":
        return _classify_rating(summ)

    # Bilinmeyen subject: guvenli default
    return NEUTRAL


def score_disclosures_by_rule(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame'in her satirini classify() ile etiketle.

    `df` en azindan 'subject' kolonu icermeli; 'summary' varsa keyword-bazli
    branch'lerde kullanilir.

    Returns:
        df.copy() + 'rule_label' kolonu.
    """
    if df.empty:
        out = df.copy()
        out["rule_label"] = pd.Series(dtype="object")
        return out
    out = df.copy()
    summaries = out["summary"] if "summary" in out.columns else [None] * len(out)
    out["rule_label"] = [
        classify(subj, summ) for subj, summ in zip(out["subject"], summaries)
    ]
    return out
