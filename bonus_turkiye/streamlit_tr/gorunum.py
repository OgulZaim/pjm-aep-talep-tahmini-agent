"""gorunum.py - Karar sonucunu (canli ya da yeniden oynatma) gosteren ortak gorunum."""
import pandas as pd
import streamlit as st

from kaynaklar import KARAR_ETIKET


def yuzde(v, basamak=1):
    """Oran -> '%12.3'; eksik deger (None/NaN) -> '-'."""
    return "-" if v is None or v != v else f"%{v * 100:.{basamak}f}"


def saat(v):
    return "-" if v is None or v != v else f"{int(v):02d}:00"


def karar_goster(s, esikler):
    o, c = s.get("ozet", {}), s.get("cikti", {})
    etiket, tur = KARAR_ETIKET.get(c.get("karar"), (c.get("karar", "?"), "info"))
    baslik = f"**{etiket}** — {o.get('gun')} (öncelik: {c.get('oncelik', '-')})"
    {"success": st.success, "warning": st.warning, "error": st.error}.get(tur, st.info)(baslik)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("En dar saat", saat(o.get("en_dar_saat")))
    k2.metric("Marj (tahmin)", yuzde(o.get("en_dusuk_marj")))
    k3.metric("Temkinli marj", yuzde(o.get("en_dusuk_marj_temkinli")),
              help="Tahmin × (1 + saatlik q90); çapraz doğrulamadan hesaplanan belirsizlik bandı")
    if o.get("tahmin_zirve_mwh") is not None:
        k4.metric("Zirve tahmini", f"{o['tahmin_zirve_mwh']:,} MWh".replace(",", "."),
                  help=f"Saat {o.get('tahmin_zirve_saati', '-')}:00")
    st.caption(f"Model: {o.get('model', '-')} · sürüm: {s.get('model_surumu', '-')} · eşikler: çok nadir < {yuzde(esikler.get('cok_nadir_alti'))}, "
               f"nadir < {yuzde(esikler.get('nadir_alti'))} (tarihsel nadirlik eşikleri; fiziksel sınır değil)")

    sa = pd.DataFrame(s.get("saatlik", []))
    if len(sa):
        sa = sa.set_index("saat")
        mw = sa.rename(columns={"tahmin_mwh": "Tahmin", "temkinli_mwh": "Temkinli (q90)", "eak_mwh": "EAK (kapasite)",
                                "resmi_plan_mwh": "Resmî plan"})[["Tahmin", "Temkinli (q90)", "EAK (kapasite)", "Resmî plan"]]
        golge = s.get("golge_model", {})
        if golge.get("saatlik"):
            mw["Önceki model (gölge)"] = pd.DataFrame(golge["saatlik"]).set_index("saat")["tahmin_mwh"]
        sol, sag = st.columns(2)
        with sol:
            st.markdown("**Saatlik tüketim tahmini ve kapasite (MWh)**")
            st.line_chart(mw, x_label="Saat", y_label="MWh")
        with sag:
            st.markdown("**Saatlik marj (%)**")
            mj = pd.DataFrame({"Marj": pd.to_numeric(sa["marj"], errors="coerce") * 100,
                               "Temkinli marj": pd.to_numeric(sa["marj_temkinli"], errors="coerce") * 100,
                               "Çok nadir eşiği": (esikler.get("cok_nadir_alti") or 0) * 100,
                               "Nadir eşiği": (esikler.get("nadir_alti") or 0) * 100}, index=sa.index)
            st.line_chart(mj, x_label="Saat", y_label="%")

    if s.get("kiyas"):
        kz = s["kiyas"]
        st.markdown("**Gerçekleşenle karşılaştırma**")
        g1, g2, g3 = st.columns(3)
        g1.metric("Model günlük hata (MAPE)", "-" if kz.get("gun_mape_%") is None else f"%{kz['gun_mape_%']}")
        g2.metric("Resmî plan hata (MAPE)", "-" if kz.get("resmi_plan_mape_%") is None else f"%{kz['resmi_plan_mape_%']}")
        g3.metric("Gerçek en dar saat / marj", f"{saat(kz.get('gercek_en_dar_saat'))} / {yuzde(kz.get('gercek_en_dusuk_marj'))}")

    st.subheader("Agent açıklaması")
    st.write(c.get("ozet", ""))
    for g in c.get("gerekceler", []):
        st.markdown(f"- {g}")
    if c.get("mevzuat_atiflari"):
        st.markdown("**Mevzuat atıfları** (kodla doğrulandı: getirilen parçalarda bulunuyor)")
        st.dataframe(pd.DataFrame(c["mevzuat_atiflari"]), hide_index=True, width="stretch")
    for u in c.get("veri_uyarilari", []):
        st.warning(u)
    if c.get("belirsizlik_notu"):
        st.caption(f"Belirsizlik notu: {c['belirsizlik_notu']}")
    notlar = s.get("notlar") or []
    st.caption("Doğrulama notları: " + ("; ".join(notlar) if notlar else "yok (karar kuralla aynı, bütün atıflar geçerli)"))
    if s.get("getirilen"):
        with st.expander("RAG: getirilen mevzuat parçaları"):
            for p in s["getirilen"]:
                st.markdown(f"- {p}")


def model_tablosu(satirlar):
    """Uretim (3d) ve golge (3c) modellerinin donem tablosu."""
    import pandas as _pd
    d = _pd.DataFrame(satirlar).rename(columns={"rol": "Rol", "surum": "Sürüm", "model": "Model", "egitim": "Eğitim",
                                               "erken_durdurma": "Erken durdurma", "test": "Test yılı", "sonuc": "Sonuç"})
    st.dataframe(d, hide_index=True, width="stretch")


def sayfa_amaci(amac, veri, not_):
    """Sayfanin basinda: bu sayfa ne icin (amac, veri, not)."""
    with st.container(border=True):
        st.markdown(f"| | |\n|---|---|\n| **Amaç** | {amac} |\n| **Veri** | {veri} |\n| **Not** | {not_} |")
