"""Streamlit dashboard: BIST hisse fiyat + KAP olay timeline + event study.

Calistir:
    streamlit run app/dashboard.py

Sayfa duzeni:
- Sidebar: hisse, tarih araligi, sentiment filtresi
- Ust metrikler: olay sayisi, ort CAR%, pozitif oran, sentiment dagilim
- Ana panel:
  1. Fiyat grafigi (plotly) + KAP olay marker'lari (sentiment renkli)
  2. CAR dagilim + per-event tablo
  3. Aynı-gun vs Ertesi-gun 2x2 mini panel
"""
from __future__ import annotations

import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
_project_root = _here.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analysis.event_study import run_event_study_multi
from src.analysis.loaders import load_all_disclosures, load_all_prices
from src.config import PROCESSED_DIR, TICKERS

# --- Sayfa konfig
st.set_page_config(page_title="BIST Haber-Fiyat Etki", layout="wide", page_icon="📈")

# --- Veri yukleme (cache'li)
@st.cache_data(show_spinner="Fiyatlar yukleniyor...")
def get_prices() -> dict[str, pd.DataFrame]:
    return load_all_prices()


@st.cache_data(show_spinner="KAP bildirimleri yukleniyor...")
def get_news() -> pd.DataFrame:
    return load_all_disclosures()


@st.cache_data(show_spinner="Sentiment skorlari yukleniyor...")
def get_sentiment() -> pd.DataFrame:
    path = PROCESSED_DIR / "sentiment.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_data(show_spinner="Event study calistiriliyor...")
def get_event_study(_prices: dict[str, pd.DataFrame], _news: pd.DataFrame) -> pd.DataFrame:
    return run_event_study_multi(_prices, _news, before=1, after=3, estimation_window=60)


prices = get_prices()
news = get_news()
sentiment = get_sentiment()
results = get_event_study(prices, news)

# --- Sentiment + event_study join (varsa)
news_with_idx = news.reset_index().rename(columns={"index": "event_idx"})
ok = results[results["status"] == "ok"].merge(
    news_with_idx[["event_idx", "disclosure_index", "ticker", "subject", "summary", "publish_datetime"]],
    on=["event_idx", "ticker"], how="left", validate="one_to_one",
)
if not sentiment.empty:
    sent_cols = ["ticker", "disclosure_index", "sentiment_label", "sentiment_score"]
    if "rule_label" in sentiment.columns:
        sent_cols.append("rule_label")
    ok = ok.merge(sentiment[sent_cols], on=["ticker", "disclosure_index"], how="left")
    if "rule_label" not in ok.columns:
        ok["rule_label"] = "n/a"
else:
    ok["sentiment_label"] = "n/a"
    ok["sentiment_score"] = np.nan
    ok["rule_label"] = "n/a"

ok["event_date"] = pd.to_datetime(ok["event_time"]).dt.date
ok["mapped_date"] = pd.to_datetime(ok["mapped_bar"]).dt.date
ok["timing"] = np.where(ok["event_date"] == ok["mapped_date"], "ayni-gun", "ertesi-gun")

# --- Header
st.title("📈 BIST Haber-Fiyat Etki Analizi")
st.caption(
    "5 hisse × 6 ay saatlik fiyat + KAP Ozel Durum Aciklamalari × Turkce BERT sentiment. "
    "Akademik / portfoy amacli; yatirim tavsiyesi degildir."
)

# --- Sidebar filtreleri
st.sidebar.header("Filtreler")
ticker = st.sidebar.selectbox("Hisse", list(TICKERS), index=0)
label_source = st.sidebar.radio(
    "Sentiment kaynagi",
    options=["BERT (savasy)", "Rule (subject-based)"],
    index=1, horizontal=False,
    help="BERT: genel-amacli Turkce sentiment (savasy). Rule: KAP subject taksonomisine dayali deterministik siniflandirma.",
)
label_col = "sentiment_label" if label_source.startswith("BERT") else "rule_label"

ticker_news = ok[ok["ticker"] == ticker].copy()
ticker_prices = prices[ticker].copy()

min_date = ticker_prices.index.min().date()
max_date = ticker_prices.index.max().date()
date_range = st.sidebar.date_input(
    "Tarih araligi", value=(min_date, max_date), min_value=min_date, max_value=max_date,
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
else:
    start, end = min_date, max_date

sentiment_filter = st.sidebar.multiselect(
    "Sentiment label",
    options=sorted(ticker_news[label_col].dropna().unique().tolist()),
    default=sorted(ticker_news[label_col].dropna().unique().tolist()),
)

st.sidebar.divider()
st.sidebar.caption(
    "**Event window:** t-1..t+3 (5 saatlik bar)  \n"
    "**Baseline:** sabit ortalama (60-bar tahmin penceresi)  \n"
    "**BERT:** savasy/bert-base-turkish-sentiment-cased  \n"
    "**Rule:** src/analysis/rule_sentiment.py (subject + summary keyword'leri)"
)

# --- Filtre uygula
mask = (
    (ticker_news["event_time"].dt.date >= start)
    & (ticker_news["event_time"].dt.date <= end)
    & (ticker_news[label_col].isin(sentiment_filter))
)
fdf = ticker_news[mask].copy()

price_mask = (ticker_prices.index.date >= start) & (ticker_prices.index.date <= end)
fprices = ticker_prices[price_mask]

# --- Ust metrikler
c1, c2, c3, c4 = st.columns(4)
c1.metric("Olay sayisi", len(fdf))
if len(fdf):
    c2.metric("Ort. CAR", f"{fdf['car'].mean() * 100:+.2f}%")
    c3.metric("Pozitif CAR orani", f"{(fdf['car'] > 0).mean() * 100:.0f}%")
    if not sentiment.empty:
        pos_share = (fdf[label_col] == "positive").mean() * 100
        c4.metric(f"Pozitif {label_source.split()[0]} %", f"{pos_share:.0f}%")
    else:
        c4.metric("Sentiment", "n/a")
else:
    c2.metric("Ort. CAR", "—")
    c3.metric("Pozitif CAR orani", "—")
    c4.metric("Pozitif sentiment %", "—")

st.divider()

# --- 1. Fiyat grafigi + KAP olay marker'lari
st.subheader(f"💹 {ticker} — Fiyat + KAP Olaylari")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=fprices.index, y=fprices["Close"],
    mode="lines", name="Close", line=dict(color="#1f77b4", width=1.5),
))

if len(fdf):
    color_map = {"positive": "#2ca02c", "negative": "#d62728", "neutral": "#888888", "n/a": "#888888"}
    for label in fdf[label_col].dropna().unique():
        sub = fdf[fdf[label_col] == label]
        # Hizalı price'i markera koy (event_time'a en yakin bar Close)
        marker_prices = [
            fprices["Close"].asof(pd.Timestamp(t)) if pd.Timestamp(t) in fprices.index or len(fprices) else np.nan
            for t in sub["event_time"]
        ]
        # asof fallback: nearest index lookup
        marker_prices = []
        for t in sub["event_time"]:
            tt = pd.Timestamp(t)
            if len(fprices) == 0:
                marker_prices.append(np.nan)
                continue
            idx = fprices.index.get_indexer([tt], method="nearest")[0]
            marker_prices.append(fprices["Close"].iloc[idx])

        hover_text = [
            f"<b>{r['event_time']:%Y-%m-%d %H:%M}</b><br>"
            f"CAR: {r['car']*100:+.2f}%<br>"
            f"BERT: {r.get('sentiment_label', 'n/a')}<br>"
            f"Rule: {r.get('rule_label', 'n/a')}<br>"
            f"Subject: {r['subject'][:80] if isinstance(r['subject'], str) else ''}<br>"
            f"Summary: {(r['summary'] or '')[:120]}"
            for _, r in sub.iterrows()
        ]
        fig.add_trace(go.Scatter(
            x=sub["event_time"], y=marker_prices,
            mode="markers", name=f"{label_source.split()[0]} — {label}",
            marker=dict(color=color_map.get(label, "#888"), size=9, symbol="circle",
                        line=dict(width=1, color="white")),
            hovertext=hover_text, hoverinfo="text",
        ))

fig.update_layout(
    height=420, margin=dict(l=20, r=20, t=20, b=20),
    xaxis_title="", yaxis_title="Close (TRY)",
    hovermode="closest", legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- 2. Event study panel: CAR dagilim + tablo
col_left, col_right = st.columns([1, 1.5])

with col_left:
    st.subheader("CAR Dagilim")
    if len(fdf):
        fig_h = px.histogram(
            fdf.assign(car_pct=fdf["car"] * 100),
            x="car_pct", nbins=20,
            color=label_col if not sentiment.empty else None,
            color_discrete_map={"positive": "#2ca02c", "negative": "#d62728", "neutral": "#888888"},
        )
        fig_h.update_layout(
            height=320, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="CAR (%)", yaxis_title="Olay",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        fig_h.add_vline(x=0, line_dash="dash", line_color="black")
        st.plotly_chart(fig_h, use_container_width=True)
    else:
        st.info("Bu filtrede olay yok.")

with col_right:
    st.subheader("Olay Tablosu")
    if len(fdf):
        show = fdf[["event_time", "sentiment_label", "rule_label", "car", "timing", "subject", "summary"]].copy()
        show["event_time"] = pd.to_datetime(show["event_time"]).dt.strftime("%Y-%m-%d %H:%M")
        show["car_pct"] = (show["car"] * 100).round(3)
        show = show.drop(columns=["car"]).sort_values("event_time", ascending=False)
        show = show[["event_time", "sentiment_label", "rule_label", "car_pct", "timing", "subject", "summary"]]
        st.dataframe(
            show, hide_index=True, use_container_width=True, height=320,
            column_config={
                "event_time": st.column_config.TextColumn("Tarih"),
                "sentiment_label": st.column_config.TextColumn("BERT"),
                "rule_label": st.column_config.TextColumn("Rule"),
                "car_pct": st.column_config.NumberColumn("CAR %", format="%+.2f"),
                "timing": st.column_config.TextColumn("Timing"),
                "subject": st.column_config.TextColumn("Konu"),
                "summary": st.column_config.TextColumn("Ozet"),
            },
        )
    else:
        st.info("—")

st.divider()

# --- 3. Aynı-gun vs Ertesi-gun 2x2
st.subheader(f"🕐 Aynı-gun vs Ertesi-gun — {label_source.split()[0]} x Timing")
if len(fdf) and not sentiment.empty:
    grid = fdf.groupby([label_col, "timing"])["car"].agg(["count", "mean"])
    grid["mean_pct"] = grid["mean"] * 100
    pivot = grid["mean_pct"].unstack(fill_value=np.nan)
    counts = grid["count"].unstack(fill_value=0)

    pos_grids = pivot.copy().astype(object)
    for r in pivot.index:
        for c in pivot.columns:
            v = pivot.loc[r, c]
            n = counts.loc[r, c]
            if pd.isna(v):
                pos_grids.loc[r, c] = "—"
            else:
                pos_grids.loc[r, c] = f"{v:+.2f}% (n={int(n)})"

    fig_g = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns, y=pivot.index,
        colorscale="RdYlGn", zmid=0,
        text=pos_grids.values, texttemplate="%{text}",
        colorbar=dict(title="Ort CAR %"),
    ))
    fig_g.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_g, use_container_width=True)
    if label_col == "rule_label":
        st.caption(
            "Tum-veri bulgusu: **rule pos × ertesi-gun** alt-grubu n=10, 9/10 pozitif (sign test p=0.02). "
            "Welch t-test marjinal (p=0.26); ornek kucuk ama yon tutarli."
        )
    else:
        st.caption(
            "Tum-veri bulgusu: **negatif × ertesi-gun** hucresi en yuksek ortalama CAR'a "
            "(+0.93%) sahipti. BERT sentiment CAR'i predict etmiyor; timing baskin faktor."
        )
elif sentiment.empty:
    st.info("Sentiment verisi yok (`python -m scripts.score_sentiment` ile uretin).")
else:
    st.info("Bu filtrede olay yok.")

st.divider()
st.caption(
    "Kaynak kod: [github.com/alperenCiftcibasi/bist-news-impact]"
    "(https://github.com/alperenCiftcibasi/bist-news-impact)"
)
