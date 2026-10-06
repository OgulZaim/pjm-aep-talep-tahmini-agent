import pandas as pd
import streamlit as st
import kaynaklar as k

st.title("Otomasyon (n8n)")
st.caption("n8n'in REST API'sinden yalnızca çalışma bilgileri okunur; düğüm verileri alınmaz. Sayfa 2 dakikada bir yenilenir.")

AKISLAR = [("TR Grid Agent", k.N8N_WORKFLOW, "Zamanlayıcı: her gün 16:15 → karar servisi → Sheets / Gmail"),
           ("AEP Grid Agent", k.N8N_WORKFLOW_AEP, "Webhook: ana projenin notebook'u karar gönderdiğinde → Sheets / Gmail")]


def bolum(baslik, ad, aciklama):
    st.subheader(baslik)
    st.caption(f"Workflow: **{ad}** · {aciklama}")
    try:
        v = k.n8n_calismalari(ad)
    except k.KaynakHatasi as ex:
        st.info(str(ex)); return
    w, cal = v["workflow"], pd.DataFrame(v["calismalar"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Yayında mı", "Evet" if w.get("aktif") else "Hayır")
    c2.metric("Listelenen çalışma", len(cal))
    if len(cal):
        cal["startedAt"] = pd.to_datetime(cal["startedAt"], utc=True).dt.tz_convert("Europe/Istanbul")
        cal["stoppedAt"] = pd.to_datetime(cal.get("stoppedAt"), utc=True).dt.tz_convert("Europe/Istanbul")
        cal["sure_sn"] = (cal["stoppedAt"] - cal["startedAt"]).dt.total_seconds().round(1)
        basarili = cal[cal["status"] == "success"]
        c3.metric("Son başarılı çalışma", basarili["startedAt"].max().strftime("%Y-%m-%d %H:%M") if len(basarili) else "yok")
        hatali = cal[cal["status"].isin(["error", "crashed"])]
        if len(hatali):
            st.warning(f"Son {len(cal)} çalışmanın {len(hatali)} tanesi hatalı. Ayrıntı: n8n → Executions.")
        st.dataframe(cal[["id", "status", "mode", "startedAt", "sure_sn"]].rename(
            columns={"status": "durum", "mode": "tetikleyici", "startedAt": "başlangıç", "sure_sn": "süre (sn)"}),
            hide_index=True, width="stretch")
    else:
        c3.metric("Son başarılı çalışma", "yok")


@st.fragment(run_every="120s")
def panel():
    for baslik, ad, aciklama in AKISLAR:
        bolum(baslik, ad, aciklama)
        st.divider()
    st.link_button("n8n'i aç", k.N8N_URL)


panel()
