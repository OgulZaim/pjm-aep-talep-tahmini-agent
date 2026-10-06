# Bonus: Türkiye Gün Öncesi Karar Sistemi (EPİAŞ)

Ana projedeki tahmin + RAG + Agent zincirinin Türkiye'ye, gerçek veriyle uyarlanması.

- **Veri:** EPİAŞ Şeffaflık Platformu (2016-2026): gerçekleşen tüketim, TEİAŞ gün öncesi Yük Tahmin Planı, emre amade kapasite (EAK). Hava: Open-Meteo (6 il, nüfus ağırlıklı; ERA5 eğitimde, GFS tahmini test ve canlıda).
- **Model:** Resmî planı düzelten hibrit XGBoost (oran hedefi). 4 yıllık yuvarlanan çapraz doğrulamanın 4'ünde ve test yılında (Ekim 2025 – Eylül 2026) resmî planı geçti: MAPE %2,29'a karşı %2,60.
- **Karar:** Eşik kuralıyla (tarihsel nadirlik eşikleri). Gemini kararı yalnızca açıklar ve Türkiye mevzuatına (RAG) dayandırır; kod, kararın değişmediğini ve atıfların getirilen maddelerden geldiğini doğrular.
- **Canlı işletim:** n8n her gün 16:15'te FastAPI servisini çağırır; Google Sheets kaydı ve Gmail taslağı. Streamlit paneli; LangSmith izleme; gölge model karşılaştırması.

## Klasörler
| Klasör | İçerik |
|---|---|
| `notebooks/` | Fizibilite, veri, özellikler, modeller, RAG, Agent, canlı çalıştırma ve izleme notebook'ları |
| `notepads/` | Zincir modülü (`tr_agent_zinciri.py`), canlı veri toplayıcı (`tr_veri_toplayici.py`) |
| `agent/` | FastAPI servisi (`tr_api_service.py`) ve model dosyaları |
| `streamlit_tr/` | Streamlit paneli |
| `n8n/` | n8n workflow'u (kimlik bilgileri ve e-posta adresi yer tutucudur) |
| `metrics/` | Sonuç ve izleme tabloları |

## Çalıştırma
`notepads/.env` dosyası gerekir (depoda yoktur): `EPTR_USERNAME`, `EPTR_PASSWORD`, `GEMINI_API_KEY`, `LANGSMITH_API_KEY`, `TR_API_ANAHTARI`, isteğe bağlı `N8N_API_KEY`.
Ham veri (`data/`) ve mevzuat belgeleri (`rag_docs/`) depoda yoktur; `1_veri`, `2_ozellik` ve `5_rag_tr` notebook'ları ile yeniden üretilir.

Ayrıntılar: Faz raporları (`*_raporu.md`).
