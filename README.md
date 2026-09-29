# PJM/AEP Elektrik Şebekesi Talep Tahmini ve LLM Tabanlı Rezerv Kapasitesi Karar Destek Sistemi

Final proje. Tema: **Talep Tahmini (Forecasting → Agent → Tedarik)**, Enerji/Şebeke alt-varyantı (PDF'teki örneklerin dışında, kendi araştırmamızla seçildi).

## 1. Problem Tanımı

PJM Interconnection'ın AEP bölgesindeki saatlik elektrik talebini tahmin edip, tahmini simüle şebeke kapasitesiyle karşılaştıran ve operatöre insan onaylı bir öneri (rezerv aktivasyonu / talep-yanıtı uyarısı / normal operasyon) üreten uçtan uca sistem.

```
[Saatlik tüketim (Kaggle) + sıcaklık (NOAA)]
      ↓
[XGBoost (nihai) / LSTM (ilk model)] → talep tahmini + güven aralığı
      ↓
[RAG: marja göre değişen sorgu → PJM Manual 13 (ChromaDB)]
      ↓
[LLM-Agent (Gemini)] → gerekçeli karar (JSON)
      ↓
[n8n] → karar tipine göre Gmail taslağı (insan onayı) + Google Sheets kaydı
      ↓
[LangSmith] → her RAG sorgusunun ve Agent kararının izlenmesi
```

## 2. Veri

- **Ana veri:** [Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption) (PJM, AEP bölgesi), 2004-10-01 → 2018-08-03, 121.273 saatlik kayıt.
- **Hava durumu:** NOAA Climate Data Online (GHCND), Columbus OH istasyonu (`USW00004804`, %99,68 kapsam), günlük TMAX/TMIN → HDD/CDD (65°F taban, PJM Manual 19 metodolojisi).
- **RAG kaynağı:** [PJM Manual 13: Emergency Operations](https://www.pjm.com/-/media/DotCom/documents/manuals/m13.pdf) (Revizyon 98), gürültü filtresi sonrası **568 parça**, ChromaDB'de.
- **Veri kalitesi notu:** 2008-10-20 14:00'te tek saatlik bir kayıt anomalisi var (25.695 MW; önceki saat 16.353, sonraki 15.991). Tarama analizlerinde komşu saatlerin ortalamasıyla değiştirildi; model eğitimi bu değeri içeriyor (bkz. Bölüm 12).

## 3. Mimari Detayları

### 3.1 Tahmin Modelleri
**LSTM (ilk model, `1.ipynb`):**
- 9 özellik: `AEP_MW`, saat, gün, ay, hafta sonu, resmî tatil, ortalama sıcaklık, HDD, CDD.
- Zamana göre ayrım (son 30 gün test), 24 saatlik pencere, 2 katmanlı LSTM (64→32), `EarlyStopping(patience=8, restore_best_weights=True)`, `set_random_seed(42)`. 95.717 parametre, CPU'da ~15-20 dk eğitim.
- **RMSPE 0,0273** (baseline 0,0838, ~3,1× iyileşme).
- Saat bazlı hata: en isabetli 15:00 (%0,98), en yüksek 07:00 (%4,65). 07:00 ham veride de en değişken (CV %16,8) ve en sert geçiş (532 MW) saati.
- Overfitting: 4 ayrı çalıştırmada train-val farkının epoch'a karşı regresyonuyla incelendi; klasik overfitting yok. Son çalıştırmada son 5 epoch'un ortalama farkı +0,00006.

**XGBoost (nihai model, `2.ipynb` + `3.ipynb`):**
- 14 özellik: takvim/hava + `lag_1, lag_2, lag_3, lag_24, lag_168, rolling_24h_mean`.
- **RMSPE 0,0097**; 120 aylık walk-forward validation (2007-2016, her ay gerçek out-of-sample).
- Saat bazlı hata: en isabetli 04:00 (%0,39), en yüksek 21:00 (%1,52). Neden: `lag_2` saat 21'de günün 19:00 zirvesini taşıyor; SHAP etkileşimi `lag_1` −77 MW, `lag_2` +9 MW; model ~257 MW fazla tahmin ediyor. Ablasyonda `lag_1` çıkarılınca RMSPE 0,0183'e kötüleşti.

### 3.2 RAG Katmanı (`RAG.ipynb`, `notepads/rag_retrieval.py`)
- 1000 karakter / 200 örtüşme parçalama, nokta-yoğunluğu filtresi (`dot_ratio < 0.15`), `gemini-embedding-001`, ChromaDB `PersistentClient`, `k=3`.
- **Duruma duyarlı sorgu (`build_rag_query`):** Marj (kapasite bazlı) ≥ %20 ise `"PJM reserve requirements control zone normal operating margin {dönem}"`, altındaysa `"grid capacity shortfall reserve activation {dönem}"`. Önceki sabit kriz sorgusu, sakin senaryolarda da acil durum prosedürleri getiriyordu.
- Aynı sorgu aynı parçaları getiriyor (ChromaDB araması deterministik; 5 ayrı trace'te doğrulandı).

### 3.3 LLM-Agent Katmanı (`notepads/prompt_v1.py`)
- `gemini-3.7-flash`, `temperature=0.2`, `response_mime_type="application/json"`.
- Girdi `<demand_forecast>`, `<grid_status>`, `<official_procedure_context>` XML etiketleriyle izole.
- `reasoning_steps` karardan önce dolduruluyor (chain-of-thought); eksik veride varsayım yok, `confidence_note`'ta belirtiliyor.
- Rakamlar deterministik, Agent'ın cümleleri çalıştırmadan çalıştırmaya değişebiliyor (temperature).
- Konsolda görünen AFC uyarısı `google-genai` SDK'sının bilinen bir sorunu (GitHub googleapis/python-genai #2902); projede function calling kullanılmıyor, işlevsel etkisi yok. Sürüm: 2.23.0; `requirements.txt`'te `google-genai<3.0.0` sabitlenmesi önerilir.

### 3.4 n8n Otomasyon Katmanı (`n8n/Grid Demand Response Automation.json`)
| Karar | n8n davranışı |
|---|---|
| `activate_reserve` | Gmail taslağı (Reserve Alert) + Sheets Reserve Log |
| `demand_response_alert` | Gmail taslağı (Demand Response Alert) + Sheets Alert Log |
| `normal_operation` | Sheets kaydı, e-posta yok |
| `uncertainty: high` (her karar) | Bağımsız daldan Gmail taslağı (Uncertainty Review) |
| Hata | `On Error: Continue` → Sheets Error Log |

Hiçbir e-posta otomatik gönderilmez; tüm e-postalar operatör onayı bekleyen taslaklardır. Talep-yanıtı taslağı sonradan eklendi (bkz. Bölüm 7).

### 3.5 Gözlemlenebilirlik (LangSmith)
- `@traceable` dekoratörü (LangChain zincirlerine bağımlı değil): `pjm_manual13_retrieval` ve `grid_agent_decision_with_rag`.
- İki model aynı iz adını kullanıyor; `3.ipynb` ayrıca `metadata={"forecast_model": "XGBoost"}` etiketi taşıyor.

### 3.6 Servis Katmanı (FastAPI, bonus)
- `agent/api_service.py`: Pydantic şemasıyla `/predict` ve `/health`. Hâlâ LSTM sunuyor.
- Çalıştırma: `cd ...\Bitirme_Proje\agent` → `uvicorn api_service:app --reload` → `http://localhost:8000/docs`.
- Doğrulama: Faz 5'teki aynı son 24 saat (24×9) gönderildiğinde servis **13.791,08 MW** döndü; Faz 5 ile birebir aynı. Yanlış şekilli girdi (`[[0]]`) `{"error": "Beklenen sekil (24, 9), gelen: (1, 1)"}` ile reddediliyor.

## 4. Depo Yapısı ve Kurulum

### 4.1 Depo Yapısı

```
agent/       api_service.py (FastAPI), grid_model.keras + grid_scaler.pkl (LSTM), grid_xgb_model.json (XGBoost)
data/        AEP_hourly.csv (ham veri), aep_with_features.csv (özellikli veri; 2.ipynb ve 3.ipynb girdisi)
graphs/      Görselleştirmeler; trend/Yearly (144 aylık grafik), trend/Yearly_XGBoost (120 walk-forward grafiği)
metrics/     Çalıştırma özetleri (run_summary*.json), model metrikleri, saatlik hata ve senaryo karşılaştırma tabloları
n8n/         Grid Demand Response Automation.json (workflow)
notebooks/   1.ipynb, RAG.ipynb, 2.ipynb, 3.ipynb
notepads/    prompt_v1.py, rag_retrieval.py, agent_test.py, n8n_test_*.py (4 dosya), scenario_comparison.py, test_api.py, .env.example
requirements.txt
hazirlik_kontrol.py   Push öncesi gizli bilgi, eksik dosya ve dosya boyutu kontrolü
```

Depoda bulunmayanlar:
- `notepads/.env`: API anahtarları.
- `rag_docs/`: PJM Manual 13 PDF'i ve ChromaDB veritabanı. PDF resmî bağlantısından indirilir, veritabanı `RAG.ipynb` ile yeniden üretilir.

### 4.2 Kurulum

1. **Depoyu klonlayın.** Notebook'lar ve script'ler `C:\Deep_Learning_to_AI_Agents\Bitirme_Proje\` altındaki mutlak yolları kullanıyor. Farklı bir klasör kullanacaksanız bu yolları güncelleyin.
   ```
   git clone https://github.com/OgulZaim/pjm-aep-talep-tahmini-agent.git C:\Deep_Learning_to_AI_Agents\Bitirme_Proje
   ```
2. **Ortam:**
   ```
   conda create -n bitirme-proje python=3.11
   conda activate bitirme-proje
   pip install -r requirements.txt
   ```
3. **Anahtarlar:** `notepads/.env.example` dosyasını `notepads/.env` olarak kopyalayıp değerleri doldurun: `GEMINI_API_KEY`, `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`, `NOAA_TOKEN`. `NOAA_TOKEN` yalnızca sıcaklık verisini yeniden çekmek için gerekli; özellikli veri `data/aep_with_features.csv` olarak depoda.
4. **RAG veritabanı (tek seferlik):** `rag_docs` klasörünü oluşturun, [PJM Manual 13](https://www.pjm.com/-/media/DotCom/documents/manuals/m13.pdf) PDF'ini `rag_docs/pjm_manual13.pdf` olarak kaydedin ve `RAG.ipynb`'yi çalıştırın.
5. **n8n:** n8n'i Docker ile başlatıp `n8n/Grid Demand Response Automation.json`'ı içe aktarın. Workflow dosyası kimlik bilgilerinin kendisini içermez; Gmail ve Google Sheets bağlantılarını kendi hesabınızla yeniden tanımlayın ve Sheets node'larında kendi tablonuzu seçin. Google OAuth uygulaması "In Production" durumunda olmalı; "Testing" modunda yenileme token'ı 7 gün sonra geçersiz oluyor. Workflow'u **Publish** edin.
6. **Çalıştırma sırası:**
   - `1.ipynb`: Veri, LSTM eğitimi ve hata analizi. `agent/grid_model.keras`, `agent/grid_scaler.pkl` ve `data/aep_with_features.csv` dosyalarını üretir. Faz 5 hücresi ChromaDB'ye (4. adım), Gemini'ye ve çalışan n8n'e ihtiyaç duyar.
   - `2.ipynb`: `data/aep_with_features.csv` ile XGBoost eğitimi, walk-forward, ablasyon ve SHAP. `agent/grid_xgb_model.json`'ı üretir.
   - `3.ipynb`: XGBoost tahmini ile RAG → Agent → n8n zinciri.
7. **FastAPI:**
   ```
   cd C:\Deep_Learning_to_AI_Agents\Bitirme_Proje\agent
   uvicorn api_service:app --reload
   ```
   Arayüz: `http://localhost:8000/docs`

## 5. RAG Kanıtları

**5.1 Test senaryosu (`agent_test.py`, sentetik veri):** 15.800 MW / 17.000 MW (%7,6 marj, peak). Agent gerekçesinde "Maximum Generation Emergency / Load Management Alert" terimini kullandı; getirilen metinde de geçtiği LangSmith'te yan yana doğrulandı. Bu değerler elle yazılmış bir test senaryosudur, gerçek model çıktısı değildir.

**5.2 Gerçek veri, sakin senaryo (`3.ipynb`):** 13.681 MW / 25.570 MW (%46,5, gece). Sorgu normal rezerv bölümünü aradı, "2.2 Reserve Requirements" geldi; gerekçe: *"...satisfying PJM Reserve Requirements under normal operating procedures without requiring capacity advisories or alerts."* `1.ipynb` Faz 5'te de (13.791 MW) aynı bölüm geliyor.

**5.3 Gerçek veri, dar marj (`1.ipynb`, `tarihi_senaryo("2009-12-10 18:00:00")`):** Tahmin 21.062 MW (gerçek 21.072), kapasite 23.875 MW, marj %11,8 (yük bazlı %13,3), peak. Sorgu kriz prosedürlerine yöneldi; karar `demand_response_alert`, gerekçe PJM'in "Maximum Generation Emergency / Load Management Alert" prosedürüne atıf yapıyor. n8n Alert Log'a yazdı ve Gmail taslağı oluşturdu. Terimin bu çalıştırmada getirilen metinde de geçtiği ayrıca doğrulanmadı. **Kapasite simüle:** bu dar marj bir PJM sıkıntısı değil, kapasite formülümüzün mevsim geçişinde geride kalmasından doğuyor (Bölüm 12).

## 6. Bilinen Sınırlamalar

- **Kapasite simüle:** son 30 günün zirvesi × 1,20 (PJM %20 IRM'den esinlenildi), canlı SCADA/EMS verisi değil. 30 günlük pencere mevsim geçişlerinde geride kalıyor. PJM'in gerçek uyarı eşikleri de bu yüzden test edilemiyor (Bölüm 12).
- **Tek bölge, tek hava istasyonu** (AEP + Columbus OH).
- **Veri kalitesi kontrolü yok:** 2008 anomalisini Agent "veri tutarlı" diyerek kabul etti; bu değer scaler'ın üst sınırını da belirledi. Agent'tan önce outlier filtresi gerekli.
- **XGBoost FastAPI'ye taşınmadı.**
- **Öğrenme döngüsü yok:** geçmiş kararları doğrulayacak bağımsız zemin gerçeği yok.
- **Güven aralığı basitleştirilmiş:** RMSPE'den simetrik marj, olasılıksal bir model değil.

## 7. Ek Güçlendirmeler

- **Confidence-tabanlı routing:** `uncertainty: high` olan her karar, kategorisinden bağımsız olarak Gmail taslağına düşer.
- **RAG gürültü filtresi:** 580 → 568 parça.
- **Duruma duyarlı RAG sorgusu:** `build_rag_query` (Bölüm 3.2).
- **Talep-yanıtı e-posta taslağı:** `demand_response_alert` artık Alert Log'a ek olarak Gmail taslağı da oluşturuyor.
- **Kalıcı senaryo karşılaştırması:** `scenario_comparison.py`, tüm `run_summary*.json` dosyalarını tek CSV'ye döküyor.

## 8. Uzun Vadeli Trend

2005-2017 yıllık ortalama talep **−136,80 MW/yıl** eğimle, **R²=0,6624** ile düşüyor (13 yılda %10,4). EIA verisi ABD talebinin 2000'lerin ortasından itibaren verimlilik ve sanayi yapısı değişimiyle düzleştiğini gösteriyor. Proje AEP'in PJM'e bağlı bölgesini kapsıyor (Ohio, Batı Virginia, Virginia, Kentucky, Indiana, Michigan); Teksas ve Oklahoma bölümleri ERCOT/SPP şebekelerinde. 144 aylık detay grafiği `graphs/trend/Yearly/`. 2008 anomalisinin yıllık ortalamaya etkisi ihmal edilebilir (tek saat).

## 9. Talep-Sıcaklık İlişkisi (167 Ay)

- Yıllık iki tepe: kış (Ocak ort. 17.431 MW) ve yaz (Temmuz-Ağustos ~16.350 MW); en düşük Nisan ve Ekim (~13.900 MW). En yüksek ay Şubat 2007 (19.213), en düşük Ekim 2016 (12.805).
- Toplam derece-gün ile talep: 1.000 derece-gün başına ~104 MW, R²=0,348 (eksik Ağustos 2018 çıkarılınca 0,352).
- Regresyon artıkları: yaz ayları ortalama +1.028 MW, bahar/sonbahar −1.055 MW, kış −160 MW. Isıtma ve soğutma talebe aynı oranda yansımıyor; HDD ve CDD'yi ayrı değişken yapmak bir sonraki iyileştirme.

## 10. Çoklu Senaryo Karşılaştırması

5 farklı zaman dilimi (gece, peak, sabah, farklı gün, yıllık zirve) test edildi; hepsi `normal_operation`. Nedeni formül: talep kendi 30 günlük zirvesini aşmadığı sürece marj %16,7'nin (kapasite bazlı) altına inemez.

## 11. Model Karşılaştırması ve Nihai Seçim — XGBoost (Bonus)

| # | Kanıt | Kaynak |
|---|---|---|
| 1 | RMSPE 0,0097 vs 0,0273, aynı `split_date` ve `rmspe()` | `2.ipynb` |
| 2 | `lag_1` %94,7 özellik önemi | `2.ipynb` |
| 3 | 120 aylık walk-forward validation | `2.ipynb` |
| 4 | Eğitim: saniyeler vs ~15-20 dk | çalıştırma süreleri |
| 5 | Canlı zincirde güven aralığı ±133 MW vs ±373 MW (~3× dar) | `3.ipynb` |

XGBoost nihai model; LSTM karşılaştırma referansı olarak korunuyor.

## 12. Kapasite Modeli ve Veri Kalitesi Analizi (`1.ipynb`, son hücreler)

- **Eşik arayışı:** PJM'in yayınlanmış tek bir yüzde eşiği yok (uyarılar gerçek zamanlı, çok faktörlü değerlendirmeyle veriliyor; LOLE hedefi 0,1 olay/yıl, Manual 20). 2005 zirvesi (24.015 MW) × 1,20 = 28.818 MW sabit kapasiteyi 2006 sonrası hiçbir saat aşmadı; temizlenmiş verideki gerçek zirve **25.164 MW (2007-08-08 15:00)**, bu kapasiteye göre %12,7 marj.
- **Neden gerçek eşik test edilemiyor:** PJM uyarıları SCADA/EMS'ten gelen gerçek zamanlı rezerv ölçümüne dayanıyor; verimizde yalnızca talep ve sıcaklık var. PJM Data Miner 2'de santral arıza beslemesi bugün + 6 günü, operasyonel rezerv beslemesi son 15 günü kapsıyor; 2004-2018 için bu ölçümler projede kullanılamadı.
- **Dar marj taraması (14 yıl, 120.553 saat):** marj < %20 olan 3.619 saat / 850 olay (ortalama 4,3 saat, en uzun 16 saat); bunların çoğu yapısal (%16,7 tabanı). **%15 altında 127 olay**, en dar **%11,57 (2004-12-13 18:00)**, **kapasitenin aşıldığı saat: 0**. 127 olayın %83'ü Kasım-Ocak ve Mayıs-Haziran'da.
- **Yorum:** Bu şebekenin değil, kapasite modelimizin zayıflığı; 30 günlük pencere sezonun ilk soğuk/sıcak dalgasında geride kalıyor. Kapasite saatlik güncellendiği için en dar an, talebin eski zirveyi aştığı ilk saat.
- **Model isabeti dar anlarda:** 4 anda (2004-12-13, 2009-12-10, 2010-12-06, 2015-11-23) tahmin hatası %0,05-0,5.
- **Marj tanımı:** Kod marjı kapasiteye göre, PJM ve Agent yüke göre hesaplıyor; %11,8 (kapasite bazlı) = %13,3 (yük bazlı).

## 13. Teslimat Durumu

Model, RAG, Agent, n8n, LangSmith ve FastAPI katmanları doğrulandı.

## 14. Hazırlayan

[@OgulZaim](https://github.com/OgulZaim)
