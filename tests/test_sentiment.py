"""sentiment modulu icin mocked birim testler (model indirilmez)."""
from __future__ import annotations

import pandas as pd

from src.analysis.sentiment import (
    _normalize_label,
    score_disclosures,
    score_texts,
)


class FakeClassifier:
    """Cagrildiginda label/score sahte tahmin dondurur, cagri kayitlarini tutar."""

    def __init__(self, label: str = "positive", score: float = 0.9) -> None:
        self.label = label
        self.score = score
        self.calls: list[list[str]] = []

    def __call__(self, texts: list[str]) -> list[dict]:
        self.calls.append(list(texts))
        return [{"label": self.label, "score": self.score} for _ in texts]


def test_normalize_label_handles_label_0_and_1():
    assert _normalize_label("LABEL_0") == "negative"
    assert _normalize_label("LABEL_1") == "positive"
    assert _normalize_label("positive") == "positive"
    assert _normalize_label("NEGATIVE") == "negative"


def test_score_texts_returns_one_tuple_per_input():
    clf = FakeClassifier(label="LABEL_1", score=0.85)
    out = score_texts(["bir metin", "iki metin"], classifier=clf)
    assert out == [("positive", 0.85), ("positive", 0.85)]


def test_score_texts_skips_empty_and_none_without_calling_model():
    clf = FakeClassifier(label="positive", score=0.7)
    out = score_texts(["", None, "  ", "gercek metin"], classifier=clf)
    assert out[0] == ("neutral", 0.0)
    assert out[1] == ("neutral", 0.0)
    assert out[2] == ("neutral", 0.0)
    assert out[3] == ("positive", 0.7)
    # Model sadece bos olmayan icin cagrilmali
    assert clf.calls == [["gercek metin"]]


def test_score_texts_batches_inputs():
    clf = FakeClassifier(label="negative", score=0.6)
    texts = [f"metin {i}" for i in range(70)]
    out = score_texts(texts, classifier=clf, batch_size=32)
    assert len(out) == 70
    # 70 = 32 + 32 + 6 → 3 batch
    assert [len(c) for c in clf.calls] == [32, 32, 6]


def test_score_texts_empty_input_returns_empty():
    out = score_texts([], classifier=FakeClassifier())
    assert out == []


def test_score_disclosures_adds_columns_and_preserves_order():
    df = pd.DataFrame(
        {
            "ticker": ["X", "Y", "Z"],
            "summary": ["pozitif haber", "negatif haber", "notr haber"],
            "subject": ["sub a", "sub b", "sub c"],
        }
    )
    clf = FakeClassifier(label="positive", score=0.8)
    out = score_disclosures(df, classifier=clf)
    assert list(out.columns) == [
        "ticker",
        "summary",
        "subject",
        "sentiment_label",
        "sentiment_score",
    ]
    assert (out["sentiment_label"] == "positive").all()
    assert (out["sentiment_score"] == 0.8).all()
    # Orijinal df mutate olmamali
    assert "sentiment_label" not in df.columns


def test_score_disclosures_uses_subject_fallback_when_summary_empty():
    df = pd.DataFrame(
        {
            "ticker": ["A", "B"],
            "summary": ["", "asil metin"],
            "subject": ["fallback metin", "subject b"],
        }
    )
    clf = FakeClassifier(label="positive", score=0.5)
    out = score_disclosures(df, classifier=clf)
    # Iki satir da model'e gitti (biri summary, biri fallback)
    assert clf.calls == [["fallback metin", "asil metin"]]
    assert (out["sentiment_label"] == "positive").all()


def test_score_disclosures_no_fallback_marks_neutral():
    df = pd.DataFrame(
        {
            "ticker": ["A"],
            "summary": [""],
            "subject": ["sub a"],
        }
    )
    clf = FakeClassifier(label="positive", score=0.9)
    out = score_disclosures(df, classifier=clf, fallback_col=None)
    assert out.loc[0, "sentiment_label"] == "neutral"
    assert out.loc[0, "sentiment_score"] == 0.0
    assert clf.calls == []  # model cagrilmadi


def test_score_disclosures_empty_df_returns_empty_with_columns():
    df = pd.DataFrame(columns=["ticker", "summary"])
    out = score_disclosures(df, classifier=FakeClassifier())
    assert out.empty
    assert "sentiment_label" in out.columns
    assert "sentiment_score" in out.columns
