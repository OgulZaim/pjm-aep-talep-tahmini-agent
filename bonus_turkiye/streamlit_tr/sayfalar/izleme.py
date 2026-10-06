import pandas as pd
import streamlit as st
import kaynaklar as k
from gorunum import sayfa_amaci

st.title("Canlı İzleme")
sayfa_amaci("Canlı tahminlerin **gerçek başarısını** ölçmek: modelin hiç görmediği günlerde tahmin, resmî plan ve gölge (önceki) model hatası.",
            "Kayıtlı canlı tahminler ve EPİAŞ'ın gerçekleşen tüketim, plan ve EAK verisi.",
            "Bir gün, ertesi gün 02:00'den sonra `7_canli_tr` D bölümü çalıştırılınca eklenir. Tek tek günler bir şey kanıtlamaz; karar en az 60 gün sonra verilecek. Sayfa 5 dakikada bir yenilenir.")


@st.fragment(run_every="300s")
def panel():
    try:
        v = k.izleme()
    except k.KaynakHatasi as ex:
        st.info(str(ex)); return
    o, d = v.get("ozet", {}), pd.DataFrame(v.get("gunler", []))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Karşılaştırılan gün", o.get("gun_sayisi", 0))
    m = lambda v: "-" if v is None else f"%{v:.2f}"
    c2.metric("Model MAPE (ort.)", m(o.get("model_mape_ort")))
    if "plan_mape_ort" in o:
        c3.metric("Resmî plan MAPE (ort.)", m(o.get("plan_mape_ort")))
    if "golge" in o:
        g = o["golge"]
        c4.metric("Gölge karşılaştırma", f"{g['uretim_daha_iyi_gun']}/{g['gun']} gün",
                  help=f"Üretim modeli ort. %{g['uretim_mape_ort']:.2f}, önceki model ort. %{g['onceki_mape_ort']:.2f}")
    if len(d):
        kol = [c for c in ["model_mape_%", "plan_mape_%", "onceki_model_mape_%"] if c in d]
        st.markdown("**Günlük hata (MAPE, %)**")
        st.line_chart(d.set_index("gun")[kol].rename(columns={"model_mape_%": "Model", "plan_mape_%": "Resmî plan",
                                                             "onceki_model_mape_%": "Önceki model (gölge)"}))
        st.dataframe(d, hide_index=True, width="stretch")


panel()
