"""Tum hisselerin KAP bildirimlerini sentiment'le skorla, parquet'e yaz.

Kullanim:
    python -m scripts.score_sentiment

Cikti: data/processed/sentiment.parquet
Schema: ticker, disclosure_index, publish_datetime, summary, subject,
        sentiment_label, sentiment_score
"""
from __future__ import annotations

import time

from src.analysis.loaders import load_all_disclosures
from src.analysis.sentiment import load_classifier, score_disclosures
from src.config import PROCESSED_DIR


def main() -> None:
    print("Bildirimler yukleniyor...")
    df = load_all_disclosures()
    print(f"  {len(df)} bildirim, {df['ticker'].nunique()} hisse")

    print("Model yukleniyor (ilk calistirmada ~440 MB indirilir, sonra kese)...")
    t0 = time.perf_counter()
    clf = load_classifier()
    print(f"  yukleme {time.perf_counter() - t0:.1f}s")

    print(f"Skorlaniyor ({len(df)} satir)...")
    t0 = time.perf_counter()
    scored = score_disclosures(df, classifier=clf)
    print(f"  skorlama {time.perf_counter() - t0:.1f}s")

    # Ozet: label x ticker dagilim
    print("\nLabel x ticker dagilim:")
    pivot = (
        scored.groupby(["ticker", "sentiment_label"])
        .size()
        .unstack(fill_value=0)
    )
    print(pivot.to_string())

    # Tut: sade kolon seti
    keep_cols = [
        "ticker",
        "disclosure_index",
        "publish_datetime",
        "subject",
        "summary",
        "sentiment_label",
        "sentiment_score",
    ]
    out = scored[keep_cols].copy()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "sentiment.parquet"
    out.to_parquet(out_path, index=False)
    print(f"\nYazildi: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
