import json
import streamlit as st
import kaynaklar as k

st.title("Tahmin Servisi")
st.caption("Ana projenin FastAPI servisi (`api_service.py`, port 8000): son 24 saatin ölçeklenmemiş 9 özelliğinden bir saat sonrasının "
           "tüketimini tahmin eder. Servis LSTM modelini sunuyor; projenin nihai seçimi XGBoost'tur.")
try:
    s = k.pjm_saglik()
    st.metric("AEP tahmin servisi", "Çalışıyor" if s.get("status") == "ok" else str(s))
except k.KaynakHatasi as ex:
    st.error(str(ex)); st.stop()
st.markdown("**Girdi:** API'nin `/docs` sayfasındaki istek gövdesinin aynısı: `{\"readings\": [[...9 değer...], ... 24 satır ...]}`. "
            "Yalnızca liste (`[[...], ...]`) de kabul edilir.")
metin = st.text_area("İstek gövdesi (JSON)", height=220, placeholder='{"readings": [[13286.0, 1.0, 3.0, 8.0, 0.0, 0.0, 73.5, 0.0, 8.5], ...]}')
if st.button("Tahmin et", type="primary"):
    try:
        veri = json.loads(metin)
        if isinstance(veri, dict):                      # API ile ayni bicim: {"readings": [...]}
            veri = veri.get("readings")
        assert isinstance(veri, list) and len(veri) == 24 and all(isinstance(r, list) and len(r) == 9 for r in veri)
    except (ValueError, AssertionError):
        st.error("Girdi, `{\"readings\": [...]}` biçiminde ve 24 satır × 9 sayıdan oluşmalı."); st.stop()
    try:
        j = k.pjm_tahmin(veri)
    except k.KaynakHatasi as ex:
        st.error(str(ex)); st.stop()
    st.metric("Bir saat sonrası tahmini", f"{j['forecast_mw']:,.0f} MW".replace(",", "."))
    st.json(j, expanded=False)
