import datetime as dt
import streamlit as st
import kaynaklar as k
from gorunum import sayfa_amaci, karar_goster, model_tablosu

st.title("Geçmiş Gün (yeniden oynatma)")
sayfa_amaci("**Test için:** Modelin hiç görmediği test yılındaki bir günü yeniden oynatıp, sistemin bir kararı nasıl ürettiğini ve örneklem dışı başarısını göstermek.",
            "Seçim modeli (3c, test yılını görmedi) ve o gün D-1'de bilinebilecek veri. Agent çağrılmaz; açıklama kodla üretilir.",
            "Yalnızca test yılı (2025-10-01 – 2026-09-22) seçilebilir. İlk çalıştırma 1-2 dakika sürebilir.")
try:
    model_tablosu(k.modeller())
except k.KaynakHatasi as ex:
    st.info(str(ex))
gun = st.date_input("Gün (test yılı)", value=dt.date(2025, 11, 7), min_value=dt.date(2025, 10, 1), max_value=dt.date(2026, 9, 22))
if st.button("Yeniden oynat", type="primary"):
    with st.spinner("Zincir çalışıyor (ilk çalıştırma 1-2 dakika sürebilir)..."):
        try:
            s = k.tekrar(str(gun), "golge")
            esikler = k.saglik().get("esikler", {})
        except k.KaynakHatasi as ex:
            st.error(str(ex)); st.stop()
    if s.get("not"):
        st.info(s["not"])
    karar_goster(s, esikler)
