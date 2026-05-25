"""event_study icin tmp data uzerinden birim testler (ag/disk yok)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.event_study import (
    event_window,
    find_event_bar,
    run_event_study,
    run_event_study_multi,
)


def _make_prices(start: str = "2026-01-05", n_days: int = 30, tz: str = "Europe/Istanbul") -> pd.DataFrame:
    """BIST-benzeri saatlik bar: weekday'ler, saat 9-17 (yfinance Istanbul tz etiketleri)."""
    end = pd.Timestamp(start, tz=tz) + pd.Timedelta(days=int(n_days * 1.5) + 5)
    all_hours = pd.date_range(pd.Timestamp(start, tz=tz), end, freq="h", tz=tz)
    mask = (all_hours.dayofweek < 5) & (all_hours.hour >= 9) & (all_hours.hour <= 17)
    idx = all_hours[mask][: n_days * 9]
    np.random.seed(42)
    close = 100 + np.arange(len(idx)) * 0.01 + np.random.randn(len(idx)) * 0.1
    df = pd.DataFrame({"Close": close}, index=idx)
    df.index.name = "datetime"
    return df


# --- find_event_bar ---

def test_find_event_bar_within_trading_hours_maps_to_floored_bar():
    prices = _make_prices()
    # 2026-01-05 (Mon) 11:30 -> 11:00 bar
    event = pd.Timestamp("2026-01-05 11:30", tz="Europe/Istanbul")
    bar = find_event_bar(prices.index, event)
    assert bar == pd.Timestamp("2026-01-05 11:00", tz="Europe/Istanbul")


def test_find_event_bar_after_close_maps_to_next_trading_day():
    prices = _make_prices()
    # 2026-01-05 (Mon) 18:30 -> 2026-01-06 (Tue) 09:00
    event = pd.Timestamp("2026-01-05 18:30", tz="Europe/Istanbul")
    bar = find_event_bar(prices.index, event)
    assert bar == pd.Timestamp("2026-01-06 09:00", tz="Europe/Istanbul")


def test_find_event_bar_friday_evening_maps_to_monday():
    prices = _make_prices()
    # 2026-01-09 (Fri) 22:00 -> 2026-01-12 (Mon) 09:00
    event = pd.Timestamp("2026-01-09 22:00", tz="Europe/Istanbul")
    bar = find_event_bar(prices.index, event)
    assert bar == pd.Timestamp("2026-01-12 09:00", tz="Europe/Istanbul")


def test_find_event_bar_after_data_end_returns_none():
    prices = _make_prices(n_days=5)
    event = prices.index.max() + pd.Timedelta(days=10)
    assert find_event_bar(prices.index, event) is None


def test_find_event_bar_handles_half_hour_offset_bars():
    """Regresyon: gercek yfinance BIST bar etiketleri :30 offsetli.
    17:48'deki bildirim 17:30 bar'a (kapsam: 17:30-18:30) dusmeli,
    ertesi gune kaymamali."""
    bars = pd.DatetimeIndex([
        pd.Timestamp(f"2026-01-05 {h:02d}:30", tz="Europe/Istanbul")
        for h in range(9, 18)
    ] + [
        pd.Timestamp(f"2026-01-06 {h:02d}:30", tz="Europe/Istanbul")
        for h in range(9, 18)
    ])
    # Kapanis muzayedesi sirasinda bildirim
    e1 = pd.Timestamp("2026-01-05 17:48", tz="Europe/Istanbul")
    assert find_event_bar(bars, e1) == pd.Timestamp("2026-01-05 17:30", tz="Europe/Istanbul")
    # Bar kapsami disinda (17:30 + 1h = 18:30 sinirini gecti)
    e2 = pd.Timestamp("2026-01-05 18:35", tz="Europe/Istanbul")
    assert find_event_bar(bars, e2) == pd.Timestamp("2026-01-06 09:30", tz="Europe/Istanbul")
    # Bar kapsami sinirinda
    e3 = pd.Timestamp("2026-01-05 10:29", tz="Europe/Istanbul")
    assert find_event_bar(bars, e3) == pd.Timestamp("2026-01-05 09:30", tz="Europe/Istanbul")


# --- event_window ---

def test_event_window_returns_ar_and_car_for_valid_event():
    prices = _make_prices(n_days=30)
    event_bar = prices.index[100]
    result = event_window(prices, event_bar, before=1, after=3, estimation_window=60)

    assert result is not None
    assert set(result.keys()) == {"event_bar", "baseline", "ar", "car", "n_bars_used"}
    assert result["n_bars_used"] == 5  # before(1) + 1 + after(3)
    assert list(result["ar"].index) == [-1, 0, 1, 2, 3]
    assert isinstance(result["car"], float)
    # AR ortalamasi baseline'dan farkin ortalamasi -> CAR kabaca AR toplamiyla esit
    assert result["car"] == pytest.approx(result["ar"].sum())


def test_event_window_insufficient_estimation_returns_none():
    prices = _make_prices(n_days=30)
    # Pozisyon 30: ~3 isgun, 60-bar estimation icin yetersiz
    event_bar = prices.index[30]
    assert event_window(prices, event_bar, before=1, after=3, estimation_window=60) is None


def test_event_window_insufficient_after_returns_none():
    prices = _make_prices(n_days=30)
    event_bar = prices.index[-1]  # son bar -> after=3 disarda
    assert event_window(prices, event_bar, before=1, after=3, estimation_window=60) is None


# --- run_event_study ---

def test_run_event_study_mixes_ok_and_skipped():
    prices = _make_prices(n_days=30)
    events = pd.DataFrame({
        "publish_datetime": [
            prices.index[100],  # ok
            prices.index[5],    # skipped:insufficient_data (estimation)
            prices.index[-1],   # skipped:insufficient_data (after)
            pd.Timestamp("2030-01-01", tz="Europe/Istanbul"),  # skipped:no_bar (future)
        ],
    })
    out = run_event_study(prices, events, before=1, after=3, estimation_window=60)

    assert len(out) == 4
    statuses = out["status"].tolist()
    assert statuses.count("ok") == 1
    assert "skipped:no_bar" in statuses
    assert sum(1 for s in statuses if s.startswith("skipped:insufficient")) == 2

    ok_row = out[out["status"] == "ok"].iloc[0]
    assert pd.notna(ok_row["car"])
    assert pd.notna(ok_row["ar_t+0"])
    assert pd.notna(ok_row["ar_t-1"])
    assert pd.notna(ok_row["ar_t+3"])


# --- run_event_study_multi ---

def test_run_event_study_multi_aggregates_with_ticker_column():
    prices_a = _make_prices(n_days=30)
    prices_b = _make_prices(n_days=30)
    events = pd.DataFrame({
        "ticker": ["A", "A", "B"],
        "publish_datetime": [
            prices_a.index[100],
            prices_a.index[120],
            prices_b.index[110],
        ],
    })
    out = run_event_study_multi(
        {"A": prices_a, "B": prices_b}, events,
        before=1, after=3, estimation_window=60,
    )

    assert len(out) == 3
    assert set(out["ticker"]) == {"A", "B"}
    assert (out["status"] == "ok").all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
