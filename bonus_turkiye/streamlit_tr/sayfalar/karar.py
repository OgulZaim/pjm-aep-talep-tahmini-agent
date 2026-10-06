import pandas as pd
import streamlit as st
import kaynaklar as k
from gorunum import sayfa_amaci, karar_goster

st.title("Günün Kararı")
sayfa_amaci("Sistemin her gün 16:15'te **yarın** için ürettiği gün öncesi kararı göstermek.",
            "Canlı EPİAŞ (tüketim, resmî plan, EAK) ve Open-Meteo hava tahmini; üretim modeli (3d). Kararı kural tablosu verir, Gemini açıklar.",
            "Bu bir tahmindir; doğruluğu, gün bitince **Canlı İzleme** sayfasında ölçülür.")
try:
    liste = k.kararlar()
    esikler = k.saglik().get("esikler", {})
except k.KaynakHatasi as ex:
    st.error(str(ex)); st.stop()
if not liste:
    st.info("Henüz kayıtlı canlı karar yok. n8n her gün 16:15'te bir karar üretir."); st.stop()
gunler = [x["gun"] for x in liste]
gun = st.selectbox("Gün", gunler, index=0, help="n8n'in ya da 7_canli_tr'nin ürettiği kayıtlı kararlar (en yeni üstte)")
try:
    s = k.karar(gun)
except k.KaynakHatasi as ex:
    st.error(str(ex)); st.stop()
karar_goster(s, esikler)
with st.expander("Kayıtlı bütün kararlar"):
    st.dataframe(pd.DataFrame(liste), hide_index=True, width="stretch")
