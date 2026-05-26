"""Turkce sentiment skorlama: savasy/bert-base-turkish-sentiment-cased.

KAP bildirim metnini (summary) ikili pozitif/negatif sentiment'e siniflar.
Model HuggingFace pipeline ile lazy yuklenir (ilk cagrida ~440 MB indirilir
ve `~/.cache/huggingface/` altinda kese alinir).

Tasarim:
- Binary model: 'positive' ve 'negative' etiketleri (LABEL_0/LABEL_1 da
  normalize edilir).
- Cikti her satir icin (label, score) tuple, score modelin verdi soft-max
  olasiligi (0..1).
- Bos / None metinler 'neutral' + score 0.0 olarak isaretlenir (model
  cagrilmaz). Boylece NaN summary'ler crash etmez.
- Batch'li cagri 32'lik gruplar halinde; CPU'da bert-base-turkish ~50ms/satir.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

MODEL_NAME = "savasy/bert-base-turkish-sentiment-cased"
BATCH_SIZE = 32
MAX_LENGTH = 256  # KAP summary'leri tipik <200 token; emniyet payi ile 256

_LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "positive",
    "negative": "negative",
    "positive": "positive",
    "NEGATIVE": "negative",
    "POSITIVE": "positive",
}


def _normalize_label(raw: str) -> str:
    """Model cikti etiketini 'positive'/'negative'e normalize et."""
    return _LABEL_MAP.get(raw, raw.lower())


def load_classifier() -> Any:
    """HF pipeline yukle. Ilk cagri ag uzerinden model indirir, sonra kese.

    Returns:
        `transformers.pipelines.TextClassificationPipeline` benzeri callable.
    """
    from transformers import pipeline  # lazy import; test'lerde mock'lanir

    return pipeline(
        task="sentiment-analysis",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        truncation=True,
        max_length=MAX_LENGTH,
    )


def score_texts(
    texts: Iterable[str | None],
    *,
    classifier: Any | None = None,
    batch_size: int = BATCH_SIZE,
) -> list[tuple[str, float]]:
    """Bir metin listesini siniflandir.

    Bos/None metinler model'e gonderilmez; ('neutral', 0.0) doner. Sira korunur.

    Args:
        texts: skorlanacak metinler (None veya bos string'ler atlanir).
        classifier: HF pipeline; None ise load_classifier() ile yuklenir.
        batch_size: ic batch boyu.

    Returns:
        Girdiyle ayni uzunlukta [(label, score), ...] listesi.
    """
    items = list(texts)
    result: list[tuple[str, float] | None] = [None] * len(items)

    # Bos/None'lari isaretle, gerisini batch'le
    pending_idx: list[int] = []
    pending_text: list[str] = []
    for i, t in enumerate(items):
        if t is None or (isinstance(t, str) and not t.strip()):
            result[i] = ("neutral", 0.0)
        else:
            pending_idx.append(i)
            pending_text.append(t)

    if not pending_text:
        return [r for r in result if r is not None]  # type: ignore[misc]

    if classifier is None:
        classifier = load_classifier()

    for start in range(0, len(pending_text), batch_size):
        chunk = pending_text[start : start + batch_size]
        preds = classifier(chunk)
        for j, pred in enumerate(preds):
            label = _normalize_label(pred["label"])
            score = float(pred["score"])
            result[pending_idx[start + j]] = (label, score)

    return [r for r in result if r is not None]  # type: ignore[misc]


def score_disclosures(
    df: pd.DataFrame,
    *,
    text_col: str = "summary",
    fallback_col: str | None = "subject",
    classifier: Any | None = None,
) -> pd.DataFrame:
    """Bildirim DataFrame'ine sentiment kolonlari ekler.

    `text_col` bos/NaN ise `fallback_col` denenir. Hicbiri yoksa 'neutral'.

    Args:
        df: load_disclosures cikti formati (en azindan `text_col` icermeli).
        text_col: birincil metin kolonu (varsayilan 'summary').
        fallback_col: yedek kolon ('subject') veya None.
        classifier: HF pipeline; None ise lazy yuklenir.

    Returns:
        df'nin kopyasi + ['sentiment_label', 'sentiment_score'] kolonlari.
    """
    if df.empty:
        out = df.copy()
        out["sentiment_label"] = pd.Series(dtype="object")
        out["sentiment_score"] = pd.Series(dtype="float64")
        return out

    def pick(row: pd.Series) -> str | None:
        primary = row.get(text_col)
        if isinstance(primary, str) and primary.strip():
            return primary
        if fallback_col is not None:
            secondary = row.get(fallback_col)
            if isinstance(secondary, str) and secondary.strip():
                return secondary
        return None

    texts = [pick(row) for _, row in df.iterrows()]
    scored = score_texts(texts, classifier=classifier)
    out = df.copy()
    out["sentiment_label"] = [lbl for lbl, _ in scored]
    out["sentiment_score"] = [sc for _, sc in scored]
    return out
