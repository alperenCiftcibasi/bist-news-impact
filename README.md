# BIST Haber-Fiyat Etki Analizi

BIST hisseleri için **KAP Özel Durum Açıklamalarının** kısa vadeli fiyat hareketlerine etkisini ölçen bir araştırma projesi. Saatlik fiyat verisi + resmi haber duyurularını birleştirip sentiment skorlarıyla event study analizi yapar.

> **Not:** Akademik / portföy amaçlıdır. Yatırım tavsiyesi değildir.

## Kapsam

| Boyut | Değer |
|---|---|
| Hisseler | THYAO, ASELS, GARAN, KCHOL, EREGL |
| Dönem | Son 6 ay |
| Fiyat granülaritesi | Saatlik (1h) |
| Haber kaynağı | KAP — Özel Durum Açıklamaları (ÖDA) |
| Sentiment modeli | `savasy/bert-base-turkish-sentiment-cased` (planlı) |

## Mevcut Veri

```
data/raw/prices/   # 5 hisse × 1118 saatlik OHLCV = ~188 KB parquet
data/raw/news/kap/ # 5 hisse × 232 ÖDA bildirim = ~100 KB JSONL
```

Tarih aralığı: 2025-11-24 → 2026-05-22 (son işlem günü).

## Stack

| Katman | Araç |
|---|---|
| Veri işleme | pandas, numpy |
| Fiyat | yfinance |
| Haber | pykap (KAP API wrapper) |
| NLP | HuggingFace Transformers (Türkçe BERT) — *planlı* |
| İstatistik | scipy (t-test, sign test) |
| Görsel | matplotlib, seaborn (notebook); plotly/streamlit *planlı* |
| Test | pytest |

## Dizin Yapısı

```
bist-news-impact/
├── data/
│   ├── raw/
│   │   ├── prices/          # Saatlik fiyat (parquet)
│   │   └── news/kap/        # KAP bildirimleri (JSONL)
│   └── processed/           # İşlenmiş veri (planlı)
├── src/
│   ├── config.py            # TICKERS, BIST_TIMEZONE, dizin yolları
│   ├── data/
│   │   ├── price_fetcher.py # yfinance → parquet
│   │   └── kap_scraper.py   # KAP ÖDA → JSONL
│   └── analysis/
│       ├── loaders.py       # Parquet/JSONL → DataFrame (notebook + analiz ortak)
│       └── event_study.py   # Pencereleme + AR/CAR (sabit ortalama baseline)
├── tests/                   # 26 birim test
├── notebooks/
│   ├── 01_eda.ipynb         # Keşifsel veri analizi (5 bölüm)
│   └── 02_event_study.ipynb # Event study (6 bölüm, t-test/sign test)
├── requirements.txt
└── README.md
```

## Kurulum

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
```

## Kullanım

```bash
# Saatlik fiyat verisini indir (yfinance)
python -m src.data.price_fetcher

# KAP ÖDA bildirimlerini indir
python -m src.data.kap_scraper

# Testleri çalıştır
pytest tests/ -v

# EDA notebook'unu aç (jupyter veya VS Code)
jupyter notebook notebooks/01_eda.ipynb

# Event study notebook'unu aç
jupyter notebook notebooks/02_event_study.ipynb
```

## KAP Veri Toplama — Mühendislik Notu

İlk hedef KAP'ın `bildirim-sorgu` sayfasını doğrudan scrape etmekti. Ancak:

1. **Site Next.js SPA** — veri client-side fetch ile geliyor, HTML'de yok
2. **API endpoint gizli** — Sorgula butonu görünür bir XHR tetiklemiyor
3. **Bot koruması** — `/c44346dd...` üzerinden 307 redirect (Akamai imzası)

Pragmatik çözüm: PyPI'da [`pykap`](https://github.com/cemsinano/pykap) paketinin gerçek API endpoint'ini (`/tr/api/disclosure/members/byCriteria`) sardığını keşfettim. Bu wrapper'ı `BISTCompany.company_id` çözümleyici olarak kullanıp, kendi temiz scraper'ımı (`src/data/kap_scraper.py`) yazdım: `disclosureClass='ODA'` ile ÖDA bildirimlerini çekiyor, Disclosure dataclass'ına normalize edip JSONL kaydediyor.

**Çıkarım:** Reverse-engineering yerine mevcut wrapper kullanmak, 1-2 saatlik bir engineering kararıydı; akademik kalite için ÖDA'nın doğru sınıf olduğunu pykap kodunu okuyarak teyit ettim.

## Veri Şeması

### Fiyat (`data/raw/prices/{TICKER}.parquet`)
- Index: `datetime` (Europe/Istanbul, saatlik)
- Kolonlar: `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`

### KAP Bildirim (`data/raw/news/kap/{TICKER}.jsonl`, satır başına 1 bildirim)
```json
{
  "ticker": "THYAO",
  "disclosure_index": 1603784,
  "publish_datetime": "2026-05-08T20:39:08+03:00",
  "subject": "Özel Durum Açıklaması (Genel)",
  "summary": "Nisan 2026 Trafik Sonuçları",
  "disclosure_class": "ODA",
  "disclosure_category": "ODA",
  "stock_codes": "THYAO",
  "attachment_count": 2,
  "url": "https://www.kap.org.tr/tr/Bildirim/1603784"
}
```

## EDA Bulguları

Notebook: [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) — GitHub plot'ları ve tabloları inline render eder.

**1. yfinance bar etiketi — varsayımdan veriye:** BIST sürekli işlemi 10:00–18:00, fakat yfinance saatlik bar etiketleri Istanbul'da **09:30–17:30** olarak gelir (her saatin 30. dakikasında, Yahoo'nun UTC-bazlı bar etiketlemesinin sonucu). Günde 9 bar (1118 satır / 125 işgün = 8.94 doğrulandı). Bar kapsamı `[start, start+1h)`:
- `09:30` bar → 09:30–10:30: pre-opening (09:40–10:00) + ilk 30 dk sürekli işlem
- `10:30..16:30` → sürekli işlem saatleri
- `17:30` bar → 17:30–18:30: son 30 dk işlem + kapanış müzayedesi (18:00–18:10) + nihai kapanış (~18:15) hepsi bu bar'ın `Close`'unda

**2. KAP bildirim zamanlaması (event study tasarımını şekillendirir):** Toplam 232 bildirimin **%81.5'i işlem saatleri dışında** yayınlanıyor (kapanış sonrası ağırlıklı). Bu, event study'de **t+1d (ertesi açılış gap) pencerenin baskın kanal** olacağı anlamına gelir; t±Nh kısa pencere ikincil rolde.

**3. Volatilite & otokorelasyon:** Saatlik vol %0.68–0.87 aralığında (yıllıklandırılmış %32–%41). 1-bar ACF tüm hisseler için `|≤0.07|` sınırı içinde (KCHOL −0.066 en negatif, ASELS +0.008 hafif pozitif) — etkin pazara yakın, hafif mean-reversion eğilimi. Abnormal getiri için sabit ortalama bazlı bir baz model yeterli olabilir (pazar modeli zorunlu değil).

## Event Study Bulguları

Notebook: [`notebooks/02_event_study.ipynb`](notebooks/02_event_study.ipynb). Pencere: `t-1..t+3` saatlik bar (toplam 5 bar). Baseline: olay öncesi 60 bar sabit ortalama (EDA bulgusuna göre pazar modeli gereksiz). **221/232 olay analiz edildi** — 9 olay yetersiz veri, 2 olay data sınırı dışı (%95.3 kapsama).

**1. Toplu CAR anlamlı değil, ama alt-grupta çok anlamlı:** 221 olay birleşik tek-örnek t-testi `t = 1.09, p = 0.28` — yani "KAP bildirimi → CAR ≠ 0" toplu olarak doğrulanmıyor. Ancak **zamanlama alt-grupları kuvvetli ayrışıyor**.

**2. ⭐ Aynı-gün vs Ertesi-gün bar map'i (kuvvetli bulgu):**

| Timing | n | Ort CAR | Std |
|---|---|---|---|
| Aynı-gün bar map | 153 | **−0.14%** | 1.72% |
| Ertesi-gün bar map | 68 | **+0.76%** | 1.98% |

**Welch iki-örnek t-test: `t = −3.23, p = 0.0016`** (≪ 0.01, çok anlamlı). EDA'daki "%81.5 dışarıda" bulgusunun arka tarafı: kapanış-sonrası bildirimler (ertesi-gün maps) **anlamlı pozitif tepki** alıyor; aynı-gün rutin bildirimler **hafif negatif**. İktisadi yorum: kapanış sonrası yapılan duyurular tipik olarak "ağır" haber (rating güncellemeleri, sermaye işlemleri, M&A) — pazar açılışında pozitif fiyatlama.

**3. Hisse bazında p-değerleri:**

| Hisse | n | Ort CAR % | t-test p | Pozitif oranı | Sign test p |
|---|---|---|---|---|---|
| THYAO | 28 | +0.28 | 0.48 | 57% | 0.57 |
| ASELS | 19 | +0.43 | 0.32 | 47% | 1.00 |
| GARAN | 111 | −0.09 | 0.64 | 47% | 0.57 |
| KCHOL | 41 | +0.43 | **0.055** | 56% | 0.53 |
| EREGL | 22 | +0.26 | 0.52 | **82%** | **0.0043** |
| **ALL** | 221 | +0.13 | 0.28 | 53% | 0.35 |

- **EREGL:** olayların %82'si pozitif yönlü tepki (sign test p = 0.0043, çok anlamlı). Büyüklük değişken olduğu için t-testte görünmüyor; **yön tutarlılığı** net.
- **KCHOL:** marjinal anlamlı (t-test p = 0.055), pozitif eğilim.
- **GARAN:** 111 olay (en büyük örneklem) — etki yok, çoğu rutin bildirim.

**4. Sonraki adım için ipuçları:** Sentiment skorlama eklendiğinde "pozitif-skorlu × ertesi-gün maps" alt-grubu büyük olasılıkla en güçlü sinyali verir. Geniş pencere (t+1d, t+5d) ve BIST100 baz model ek doğrulama için denenebilir.

## Yol Haritası

- [x] **Fiyat toplama** — yfinance, 5 hisse × 6 ay saatlik
- [x] **KAP haber toplama** — ÖDA bildirimleri, 5 hisse × 6 ay (232 bildirim)
- [x] **Birim testler** — 26 test, ağ çağrısı içermez
- [x] **EDA notebook** — getiri dağılımı, volatilite, KAP yoğunluğu, fiyat × haber timeline
- [x] **Event study** — `t-1..t+3` pencere, sabit ortalama baseline, AR/CAR + istatistiksel anlamlılık
- [ ] **Türkçe sentiment skorlama** — BERT ile her bildirime polarite
- [ ] **Streamlit dashboard** — hisse seç → olay zaman çizgisi + getiri grafiği

## Test Durumu

```
26 passed in ~1s
- tests/test_price_fetcher.py  (4 test)
- tests/test_kap_scraper.py    (6 test)
- tests/test_loaders.py        (6 test)
- tests/test_event_study.py    (10 test)
```

## Lisans

MIT
