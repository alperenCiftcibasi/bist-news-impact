"""BIST hisseleri için yfinance üzerinden saatlik fiyat verisi indirir ve parquet olarak kaydeder."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.config import (
    BIST_TIMEZONE,
    DEFAULT_INTERVAL,
    DEFAULT_PERIOD,
    PRICES_DIR,
    TICKERS,
)

logger = logging.getLogger(__name__)


def fetch_hourly_prices(
    tickers: dict[str, str],
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
) -> dict[str, pd.DataFrame]:
    """yfinance'tan her hisse için saatlik OHLCV verisi çeker.

    Args:
        tickers: {kısa_ad: yfinance_sembolü} sözlüğü, örn. {"THYAO": "THYAO.IS"}.
        period: yfinance period stringi (ör. "6mo", "1y").
        interval: yfinance interval stringi (ör. "1h", "1d").

    Returns:
        {kısa_ad: DataFrame} — DataFrame'in indeksi Europe/Istanbul saatine çevrilmiştir.
        Çekilemeyen hisseler dönen sözlükte yer almaz.
    """
    results: dict[str, pd.DataFrame] = {}

    for short_name, symbol in tickers.items():
        logger.info("Veri çekiliyor: %s (%s)", short_name, symbol)
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
        )

        if df.empty:
            logger.warning("Veri boş, atlanıyor: %s", symbol)
            continue

        # yfinance bazen MultiIndex kolon döndürür (ticker tek olsa bile); düzleştir.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # yfinance saatlik veriyi UTC olarak döndürür — BIST saatine çeviriyoruz.
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert(BIST_TIMEZONE)
        df.index.name = "datetime"

        results[short_name] = df
        logger.info(
            "  -> %d satır | %s — %s",
            len(df),
            df.index.min(),
            df.index.max(),
        )

    return results


def save_prices(prices: dict[str, pd.DataFrame], output_dir: Path = PRICES_DIR) -> None:
    """Her hisseyi `{output_dir}/{kısa_ad}.parquet` olarak kaydeder."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for short_name, df in prices.items():
        target = output_dir / f"{short_name}.parquet"
        df.to_parquet(target)
        logger.info("Kaydedildi: %s (%d satır)", target, len(df))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    prices = fetch_hourly_prices(TICKERS)
    if not prices:
        logger.error("Hiçbir hisse için veri çekilemedi.")
        return
    save_prices(prices)
    logger.info("Tamamlandı: %d/%d hisse kaydedildi.", len(prices), len(TICKERS))


if __name__ == "__main__":
    main()
