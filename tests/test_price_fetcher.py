"""price_fetcher için ağ çağrısı yapmayan birim testler."""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from src.data import price_fetcher


def _make_fake_ohlcv(n: int = 5) -> pd.DataFrame:
    """yfinance.download çıktısına benzer bir DataFrame üretir (UTC, naive)."""
    idx = pd.date_range("2025-01-02 07:00", periods=n, freq="h")
    return pd.DataFrame(
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


def test_fetch_hourly_prices_converts_to_istanbul_timezone():
    fake = _make_fake_ohlcv()
    with patch.object(price_fetcher.yf, "download", return_value=fake):
        result = price_fetcher.fetch_hourly_prices({"THYAO": "THYAO.IS"})

    assert "THYAO" in result
    df = result["THYAO"]
    assert df.index.tz is not None
    assert str(df.index.tz) == "Europe/Istanbul"
    assert df.index.name == "datetime"


def test_fetch_hourly_prices_skips_empty():
    empty = pd.DataFrame()
    with patch.object(price_fetcher.yf, "download", return_value=empty):
        result = price_fetcher.fetch_hourly_prices({"THYAO": "THYAO.IS"})
    assert result == {}


def test_save_prices_writes_parquet(tmp_path):
    df = _make_fake_ohlcv()
    df.index = df.index.tz_localize("UTC").tz_convert("Europe/Istanbul")
    df.index.name = "datetime"

    price_fetcher.save_prices({"THYAO": df}, output_dir=tmp_path)

    target = tmp_path / "THYAO.parquet"
    assert target.exists()

    reloaded = pd.read_parquet(target)
    assert len(reloaded) == len(df)
    assert list(reloaded.columns) == list(df.columns)


def test_fetch_hourly_prices_flattens_multiindex_columns():
    fake = _make_fake_ohlcv()
    fake.columns = pd.MultiIndex.from_product([fake.columns, ["THYAO.IS"]])
    with patch.object(price_fetcher.yf, "download", return_value=fake):
        result = price_fetcher.fetch_hourly_prices({"THYAO": "THYAO.IS"})

    df = result["THYAO"]
    assert not isinstance(df.columns, pd.MultiIndex)
    assert "Close" in df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
