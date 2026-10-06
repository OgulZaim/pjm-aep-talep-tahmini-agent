import streamlit as st
import kaynaklar as k
from gorunum import model_tablosu

st.title("Sistem Sağlığı")
try:
    s = k.saglik()
except k.KaynakHatasi as ex:
    st.error(str(ex)); st.stop()
c1, c2, c3 = st.columns(3)
c1.metric("Karar servisi", "Çalışıyor" if s.get("durum") == "calisiyor" else s.get("durum", "?"))
c2.metric("Üretim modeli", s["uretim_modeli"]["surum"], help=s["uretim_modeli"]["model"])
c3.metric("Son canlı kayıt", s.get("son_canli_kayit") or "-")
c4, c5, c6 = st.columns(3)
c4.metric("Gölge (önceki) model", s.get("golge_modeli") or "yok")
c5.metric("Tatil takvimi bitişi", s.get("tatil_takvimi_bitis", "-"), help="Bitmeden önce 2_ozellik.ipynb'de yeni yıl eklenmeli")
c6.metric("API anahtarı zorunlu", "Evet" if s.get("api_anahtari_zorunlu") else "Hayır")
st.subheader("Model sürümleri ve dönemleri")
try:
    model_tablosu(k.modeller())
except k.KaynakHatasi as ex:
    st.info(str(ex))
st.json(s, expanded=False)
