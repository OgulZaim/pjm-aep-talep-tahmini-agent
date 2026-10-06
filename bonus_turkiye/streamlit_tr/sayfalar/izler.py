import pandas as pd
import streamlit as st
import kaynaklar as k

st.title("Agent İzleri (LangSmith)")
st.caption("LangSmith'in varsayılan katmanında izler 14 gün saklanır; bu sayfa yalnızca son 14 günü gösterir. "
           "Kalıcı karar geçmişi 'Günün Kararı' sayfasındadır (API kayıtları).")
proje = st.selectbox("Proje", k.LS_PROJELERI)
try:
    izler = k.ls_izleri(proje)
except k.KaynakHatasi as ex:
    st.info(str(ex)); st.stop()
if not izler:
    st.info("Son 14 günde bu projede iz yok."); st.stop()
d = pd.DataFrame(izler)
c1, c2, c3 = st.columns(3)
c1.metric("İz sayısı (14 gün)", len(d))
c2.metric("Ortalama süre", f"{d['sure_sn'].dropna().mean():.1f} sn" if d["sure_sn"].notna().any() else "-")
c3.metric("Toplam token", f"{int(d['token'].fillna(0).sum()):,}".replace(",", "."))
st.dataframe(d[["iz", "baslangic", "sure_sn", "durum", "token", "etiketler", "baglanti"]],
             column_config={"baglanti": st.column_config.LinkColumn("LangSmith", display_text="izi aç")},
             hide_index=True, width="stretch")
secim = st.selectbox("İz ayrıntısı", range(len(d)), format_func=lambda i: f"{d.loc[i, 'iz']} · {d.loc[i, 'baslangic']:%Y-%m-%d %H:%M}")
sol, sag = st.columns(2)
with sol:
    st.markdown("**Girdi**"); st.json(d.loc[secim, "girdi"] or {}, expanded=False)
with sag:
    st.markdown("**Çıktı**"); st.json(d.loc[secim, "cikti"] or {}, expanded=True)
