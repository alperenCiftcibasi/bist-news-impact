"""Event study: KAP bildirimlerinin fiyat etkisi.

Tek bildirim icin pencere kesme + sabit ortalama baz model uzerinden
Abnormal Return (AR) ve Cumulative Abnormal Return (CAR) hesabi.

Toplu calistirma icin run_event_study (tek hisse) ve run_event_study_multi
(birden cok hisse) fonksiyonlari.

Tasarim kararlari:
- Baz model: sabit ortalama. EDA'da ACF lag-1 ~ 0 (|<=0.07|) oldugu icin
  pazar modeli (BIST100) gerekmedi.
- Bildirim -> bar eslemesi: bir bar [bar.start, bar.start + 1h) araligini kapsar.
  Bildirim bu kapsamdaysa o bar; degilse endexteki sonraki bar. Detay:
  yfinance Istanbul saatlik bar etiketleri **:30 dakika offsetli** (09:30,
  10:30, ..., 17:30) oldugu icin 17:48'deki bir bildirim 17:30 bar'a duser
  (bu bar kapanis muzayedesini ve nihai kapanisi (~18:15) de icerir).
- Getiri: log Close-to-Close. Adj Close DEGIL, cunku event sirasinda
  temettu/split duyurularinin fiyat etkisini gormek istiyoruz.
- Pencere yetersiz olaylar (estimation/before/after) atlanir, status'la isaretlenir.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def find_event_bar(
    prices_index: pd.DatetimeIndex, event_time: pd.Timestamp
) -> pd.Timestamp | None:
    """Bildirimi etkileyecek saatlik bar'i bul.

    Kural: bir bar `[bar.start, bar.start + 1h)` araligini kapsar.
    event_time bu yari-acik aralikta ise o bar; degilse endexteki sonraki bar.

    Ornekler (yfinance BIST: bar'lar :30 dakika offsetli, 09:30..17:30):
    - 11:48 -> 11:30 bar (11:30 + 18dk, kapsam icinde)
    - 17:48 -> 17:30 bar (kapanis muzayedesi + nihai kapanis bu bar'in
      Close'unda toplandigi icin)
    - 18:35 -> ertesi gun 09:30 bar (17:30 + 1h = 18:30 < 18:35, gap)
    - Cumartesi herhangi bir saat -> Pazartesi 09:30 bar

    Args:
        prices_index: tz-aware saatlik fiyat index'i.
        event_time: tz-aware bildirim zamani.

    Returns:
        Bar etiketi (Timestamp). event_time son bar'in kapsamindan sonra
        ise None.
    """
    candidates = prices_index[prices_index <= event_time]
    if len(candidates) > 0:
        latest = candidates[-1]
        if (event_time - latest) < pd.Timedelta(hours=1):
            return latest
    next_bars = prices_index[prices_index > event_time]
    if len(next_bars) == 0:
        return None
    return next_bars[0]


def event_window(
    prices: pd.DataFrame,
    event_bar: pd.Timestamp,
    *,
    before: int = 1,
    after: int = 3,
    estimation_window: int = 60,
) -> dict | None:
    """Tek bir bildirim icin pencere + baseline + AR + CAR.

    Args:
        prices: 'Close' kolonu olan tz-aware saatlik DataFrame.
        event_bar: bildirimin haritalandirildigi bar etiketi (find_event_bar cikti).
        before: olay oncesi bar sayisi.
        after: olay sonrasi bar sayisi.
        estimation_window: olay penceresinden once baseline icin kac bar.

    Returns:
        Dict { 'event_bar', 'baseline', 'ar', 'car', 'n_bars_used' } veya
        None (estimation/before/after yetersiz).
    """
    if "Close" not in prices.columns:
        raise ValueError("prices must contain 'Close' column")

    returns = np.log(prices["Close"]).diff()  # ilk row NaN

    if event_bar not in returns.index:
        return None

    event_pos = returns.index.get_loc(event_bar)

    window_start = event_pos - before
    window_end = event_pos + after
    if window_start < 0 or window_end >= len(returns):
        return None

    est_end = event_pos - before  # exclusive
    est_start = est_end - estimation_window
    if est_start < 0:
        return None

    est_returns = returns.iloc[est_start:est_end].dropna()
    if len(est_returns) < estimation_window // 2:
        return None
    baseline = float(est_returns.mean())

    window_returns = returns.iloc[window_start : window_end + 1]
    ar = window_returns - baseline
    ar.index = list(range(-before, after + 1))  # offset

    return {
        "event_bar": event_bar,
        "baseline": baseline,
        "ar": ar,
        "car": float(ar.sum()),
        "n_bars_used": int(len(ar)),
    }


def run_event_study(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    *,
    before: int = 1,
    after: int = 3,
    estimation_window: int = 60,
) -> pd.DataFrame:
    """Tek hisse icin toplu event study.

    Args:
        prices: tek hissenin saatlik fiyat DataFrame'i.
        events: 'publish_datetime' kolonu olan DataFrame.

    Returns:
        Her olay icin tek satir. Kolonlar:
            event_idx, event_time, mapped_bar, baseline, car, ar_t-1, ar_t+0, ...,
            n_bars_used, status
        status: 'ok' | 'skipped:no_bar' | 'skipped:insufficient_data'.
    """
    rows: list[dict] = []
    for i, row in events.iterrows():
        event_time = row["publish_datetime"]
        mapped = find_event_bar(prices.index, event_time)
        if mapped is None:
            rows.append({
                "event_idx": i,
                "event_time": event_time,
                "mapped_bar": pd.NaT,
                "status": "skipped:no_bar",
            })
            continue

        result = event_window(
            prices, mapped,
            before=before, after=after, estimation_window=estimation_window,
        )
        if result is None:
            rows.append({
                "event_idx": i,
                "event_time": event_time,
                "mapped_bar": mapped,
                "status": "skipped:insufficient_data",
            })
            continue

        out = {
            "event_idx": i,
            "event_time": event_time,
            "mapped_bar": mapped,
            "baseline": result["baseline"],
            "car": result["car"],
            "n_bars_used": result["n_bars_used"],
            "status": "ok",
        }
        for offset, ar_val in result["ar"].items():
            out[f"ar_t{offset:+d}"] = float(ar_val)
        rows.append(out)

    return pd.DataFrame(rows)


def run_event_study_multi(
    prices_by_ticker: dict[str, pd.DataFrame],
    events: pd.DataFrame,
    *,
    before: int = 1,
    after: int = 3,
    estimation_window: int = 60,
) -> pd.DataFrame:
    """Coklu hisse event study.

    events'ta 'ticker' kolonu varsa hisseye gore filtreler. Sonuca 'ticker'
    kolonu eklenir.
    """
    frames: list[pd.DataFrame] = []
    for ticker, prices in prices_by_ticker.items():
        if "ticker" in events.columns:
            ticker_events = events[events["ticker"] == ticker]
        else:
            ticker_events = events
        if ticker_events.empty:
            continue
        res = run_event_study(
            prices, ticker_events,
            before=before, after=after, estimation_window=estimation_window,
        )
        if res.empty:
            continue
        frames.append(res.assign(ticker=ticker))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
