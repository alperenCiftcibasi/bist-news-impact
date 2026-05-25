"""Veri yükleme yardımcıları: fiyat parquet ve KAP JSONL.

Notebook'lardan ve sonraki analiz modüllerinden ortak kullanım için.
Disk-okuma dışında ağ çağrısı yok; testler `tmp_path` üzerinden mock'lanabilir.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from src.config import BIST_TIMEZONE, KAP_DIR, PRICES_DIR, TICKERS


def load_prices(ticker: str, *, prices_dir: Path = PRICES_DIR) -> pd.DataFrame:
    """Tek hissenin saatlik fiyat parquet'ini yükler.

    Returns:
        DataFrame, index `DatetimeIndex` (Europe/Istanbul tz-aware),
        kolonlar `Open, High, Low, Close, Adj Close, Volume`.
    """
    path = prices_dir / f"{ticker}.parquet"
    df = pd.read_parquet(path)
    if df.index.tz is None:
        df.index = df.index.tz_localize(BIST_TIMEZONE)
    return df


def load_all_prices(
    tickers: Iterable[str] | None = None,
    *,
    prices_dir: Path = PRICES_DIR,
) -> dict[str, pd.DataFrame]:
    """Birden çok hisseyi `{ticker: DataFrame}` olarak yükler.

    `tickers` None ise `config.TICKERS`'taki tüm hisseler okunur.
    """
    names = list(tickers) if tickers is not None else list(TICKERS)
    return {t: load_prices(t, prices_dir=prices_dir) for t in names}


def load_disclosures(ticker: str, *, kap_dir: Path = KAP_DIR) -> pd.DataFrame:
    """Tek hissenin KAP JSONL'ini DataFrame'e yükler.

    `publish_datetime` kolonu tz-aware `pd.Timestamp` olarak parse edilir.
    Sonuç publish_datetime'a göre artan sırada döner.
    """
    path = kap_dir / f"{ticker}.jsonl"
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["publish_datetime"] = pd.to_datetime(df["publish_datetime"])
    return df.sort_values("publish_datetime").reset_index(drop=True)


def load_all_disclosures(
    tickers: Iterable[str] | None = None,
    *,
    kap_dir: Path = KAP_DIR,
) -> pd.DataFrame:
    """Tüm hisselerin KAP bildirimlerini tek DataFrame'de birleştirir.

    Boş JSONL'leri atlar. Her satırın `ticker` kolonu KAP scraper tarafından
    zaten yazılır, ek bir join'e gerek yoktur.
    """
    names = list(tickers) if tickers is not None else list(TICKERS)
    frames: list[pd.DataFrame] = []
    for t in names:
        df = load_disclosures(t, kap_dir=kap_dir)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values("publish_datetime")
        .reset_index(drop=True)
    )
