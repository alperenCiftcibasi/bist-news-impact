"""Tum hisselerin KAP bildirimlerini sentiment'le skorla, parquet'e yaz.

Kullanim:
    python -m scripts.score_sentiment

Cikti: data/processed/sentiment.parquet
Schema: ticker, disclosure_index, publish_datetime, subject, summary,
        sentiment_label, sentiment_score, rule_label
- sentiment_label/sentiment_score: savasy BERT (genel Turkce sentiment)
- rule_label: KAP subject + summary keyword'lerine dayali deterministik
  siniflandirma (src/analysis/rule_sentiment.py)
"""
from __future__ import annotations

import time

import pandas as pd

from src.analysis.loaders import load_all_disclosures
from src.analysis.rule_sentiment import score_disclosures_by_rule
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

    print(f"BERT skorlaniyor ({len(df)} satir)...")
    t0 = time.perf_counter()
    scored = score_disclosures(df, classifier=clf)
    print(f"  BERT skorlama {time.perf_counter() - t0:.1f}s")

    print("Rule-based etiketleniyor...")
    scored = score_disclosures_by_rule(scored)

    # Ozet
    print("\nBERT label x ticker:")
    print(scored.groupby(["ticker", "sentiment_label"]).size().unstack(fill_value=0).to_string())
    print("\nRule label x ticker:")
    print(scored.groupby(["ticker", "rule_label"]).size().unstack(fill_value=0).to_string())
    print("\nBERT vs Rule cross-tab:")
    print(pd.crosstab(scored["sentiment_label"], scored["rule_label"]).to_string())

    # Tut: sade kolon seti
    keep_cols = [
        "ticker",
        "disclosure_index",
        "publish_datetime",
        "subject",
        "summary",
        "sentiment_label",
        "sentiment_score",
        "rule_label",
    ]
    out = scored[keep_cols].copy()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "sentiment.parquet"
    out.to_parquet(out_path, index=False)
    print(f"\nYazildi: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
