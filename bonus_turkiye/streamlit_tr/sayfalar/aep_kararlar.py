import pandas as pd
import streamlit as st
import kaynaklar as k
from gorunum import sayfa_amaci

st.title("Agent Kararları")
sayfa_amaci("Ana projenin (PJM / AEP) Agent'ının iki senaryodaki kararını göstermek: normal işletim ve dar marj.",
            "`8_langsmith_izleri` notebook'unun LangSmith izleri (son 14 gün). Talep o saatin gerçekleşen değeri; kapasite simüle (son 30 günün zirvesi × 1,20).",
            "Dar marj gerçek bir şebeke krizi değil; Agent'ın dar marja tepkisini gerçek bir talep artışıyla test eder.")
proje = st.selectbox("LangSmith projesi", ["sunum-izleri", "bitirme-proje-grid"])
try:
    izler = [i for i in k.ls_izleri(proje) if str(i["iz"]).startswith(("1 - PJM", "2 - PJM"))]
except k.KaynakHatasi as ex:
    st.info(str(ex)); st.stop()
if not izler:
    st.info("Son 14 günde PJM senaryo izi yok. `8_langsmith_izleri.ipynb`'yi çalıştırın; izler bu sayfada görünür."); st.stop()
son = {}
for i in sorted(izler, key=lambda x: x["baslangic"]):
    son[i["iz"]] = i                                   # her senaryonun en son izi
for ad in sorted(son):
    i, o = son[ad], (son[ad]["cikti"] or {})
    st.subheader(ad.split(" - ", 1)[-1].replace("PJM", "AEP"))
    etiket, tur = k.KARAR_ETIKET.get(o.get("decision"), (o.get("decision", "?"), "info"))
    {"success": st.success, "warning": st.warning, "error": st.error}.get(tur, st.info)(f"**{etiket}** — {o.get('tarih', '-')}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Talep (gerçekleşen)", f"{o.get('talep_mw', 0):,} MW".replace(",", "."))
    c2.metric("Kapasite (simüle)", f"{o.get('kapasite_mw', 0):,} MW".replace(",", "."))
    c3.metric("Marj", "-" if o.get("marj") is None else f"%{o['marj'] * 100:.1f}")
    c4.metric("Dönem", o.get("donem", "-"))
    kj = o.get("karar_json") or {}
    for anahtar in ["justification", "reasoning", "explanation", "summary"]:
        if kj.get(anahtar):
            st.write(kj[anahtar]); break
    st.caption(f"RAG sorgusu: {o.get('rag_sorgusu', '-')} · iz: {i['baslangic']:%Y-%m-%d %H:%M}")
    with st.expander("Agent'ın tam karar JSON'u"):
        st.json(kj)
    if i.get("baglanti"):
        st.link_button("LangSmith'te izi aç", i["baglanti"])
    st.divider()
