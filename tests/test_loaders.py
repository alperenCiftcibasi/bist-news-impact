"""loaders icin disk-only birim testler (ag cagrisi yok, tmp_path uzerinden)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.loaders import (
    load_all_disclosures,
    load_all_prices,
    load_disclosures,
    load_prices,
)


def _make_price_parquet(path: Path, n: int = 24, tz: str = "Europe/Istanbul") -> None:
    idx = pd.date_range("2026-01-01 10:00", periods=n, freq="h", tz=tz)
    df = pd.DataFrame(
        {
            "Open": range(n),
            "High": range(n),
            "Low": range(n),
            "Close": range(n),
            "Adj Close": range(n),
            "Volume": [1000] * n,
        },
        index=idx,
    )
    df.index.name = "datetime"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def _make_kap_jsonl(path: Path, ticker: str, n: int = 3) -> None:
    rows = [
        {
            "ticker": ticker,
            "disclosure_index": 100 + i,
            # Bilerek tersten yaziyoruz ki sort dogrulanabilsin
            "publish_datetime": f"2026-01-{n - i:02d}T15:30:00+03:00",
            "subject": f"Subject {i}",
            "summary": f"Summary {i}",
            "disclosure_class": "ODA",
            "disclosure_category": "ODA",
            "stock_codes": ticker,
            "attachment_count": 0,
            "url": f"https://example/{100 + i}",
        }
        for i in range(n)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_load_prices_returns_tz_aware_dataframe(tmp_path):
    _make_price_parquet(tmp_path / "THYAO.parquet")
    df = load_prices("THYAO", prices_dir=tmp_path)

    assert isinstance(df, pd.DataFrame)
    assert df.index.tz is not None
    assert str(df.index.tz) == "Europe/Istanbul"
    assert {"Open", "High", "Low", "Close", "Volume"}.issubset(df.columns)
    assert len(df) == 24


def test_load_all_prices_loads_multiple_tickers(tmp_path):
    for t in ["A", "B"]:
        _make_price_parquet(tmp_path / f"{t}.parquet", n=10)
    out = load_all_prices(["A", "B"], prices_dir=tmp_path)

    assert set(out.keys()) == {"A", "B"}
    assert all(len(df) == 10 for df in out.values())
    assert all(str(df.index.tz) == "Europe/Istanbul" for df in out.values())


def test_load_disclosures_parses_datetime_and_sorts(tmp_path):
    _make_kap_jsonl(tmp_path / "THYAO.jsonl", "THYAO", n=3)
    df = load_disclosures("THYAO", kap_dir=tmp_path)

    assert len(df) == 3
    assert pd.api.types.is_datetime64_any_dtype(df["publish_datetime"])
    assert df["publish_datetime"].dt.tz is not None
    assert df["publish_datetime"].is_monotonic_increasing


def test_load_disclosures_empty_file_returns_empty_frame(tmp_path):
    (tmp_path / "EMPTY.jsonl").write_text("", encoding="utf-8")
    df = load_disclosures("EMPTY", kap_dir=tmp_path)
    assert df.empty


def test_load_all_disclosures_concatenates_and_sorts(tmp_path):
    _make_kap_jsonl(tmp_path / "A.jsonl", "A", n=2)
    _make_kap_jsonl(tmp_path / "B.jsonl", "B", n=2)
    df = load_all_disclosures(["A", "B"], kap_dir=tmp_path)

    assert len(df) == 4
    assert set(df["ticker"]) == {"A", "B"}
    assert df["publish_datetime"].is_monotonic_increasing


def test_load_all_disclosures_skips_empty_files(tmp_path):
    _make_kap_jsonl(tmp_path / "A.jsonl", "A", n=2)
    (tmp_path / "B.jsonl").write_text("", encoding="utf-8")
    df = load_all_disclosures(["A", "B"], kap_dir=tmp_path)

    assert len(df) == 2
    assert set(df["ticker"]) == {"A"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
