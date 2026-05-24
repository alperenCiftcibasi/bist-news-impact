"""Proje sabitleri: hisse listesi, zaman dilimi, dizin yolları."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"

PRICES_DIR: Path = RAW_DIR / "prices"
NEWS_DIR: Path = RAW_DIR / "news"
KAP_DIR: Path = NEWS_DIR / "kap"

BIST_TIMEZONE: str = "Europe/Istanbul"

# yfinance, BIST hisseleri için .IS suffix bekler.
TICKERS: dict[str, str] = {
    "THYAO": "THYAO.IS",
    "ASELS": "ASELS.IS",
    "GARAN": "GARAN.IS",
    "KCHOL": "KCHOL.IS",
    "EREGL": "EREGL.IS",
}

DEFAULT_PERIOD: str = "6mo"
DEFAULT_INTERVAL: str = "1h"
