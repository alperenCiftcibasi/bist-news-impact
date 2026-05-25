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
| Görsel | plotly, streamlit — *planlı* |
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
│       └── loaders.py       # Parquet/JSONL → DataFrame (notebook + analiz ortak)
├── tests/                   # 16 birim test
├── notebooks/
│   └── 01_eda.ipynb         # Keşifsel veri analizi (5 bölüm)
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

## Yol Haritası

- [x] **Fiyat toplama** — yfinance, 5 hisse × 6 ay saatlik
- [x] **KAP haber toplama** — ÖDA bildirimleri, 5 hisse × 6 ay (232 bildirim)
- [x] **Birim testler** — 16 test, ağ çağrısı içermez
- [x] **EDA notebook** — getiri dağılımı, volatilite, KAP yoğunluğu, fiyat × haber timeline
- [ ] **Türkçe sentiment skorlama** — BERT ile her bildirime polarite
- [ ] **Event study** — t±1h, t+1d pencereleri, abnormal getiri
- [ ] **Streamlit dashboard** — hisse seç → olay zaman çizgisi + getiri grafiği

## Test Durumu

```
16 passed in ~6s
- tests/test_price_fetcher.py  (4 test)
- tests/test_kap_scraper.py    (6 test)
- tests/test_loaders.py        (6 test)
```

## Lisans

MIT
