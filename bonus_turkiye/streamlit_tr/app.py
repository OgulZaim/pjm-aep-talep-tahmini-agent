"""Grid Agent paneli: TR (bonus) ve AEP (ana proje). Calistirma: python -m streamlit run app.py  (streamlit_tr klasorunden)"""
import streamlit as st

st.set_page_config(page_title="Grid Agent", page_icon="⚡", layout="wide")
sayfalar = st.navigation({
    "TR Grid Agent": [st.Page("sayfalar/karar.py", title="Günün Kararı", icon=":material/bolt:", default=True),
                      st.Page("sayfalar/gecmis.py", title="Geçmiş Gün", icon=":material/history:"),
                      st.Page("sayfalar/izleme.py", title="Canlı İzleme", icon=":material/monitoring:"),
                      st.Page("sayfalar/otomasyon.py", title="Otomasyon (n8n)", icon=":material/schedule:"),
                      st.Page("sayfalar/izler.py", title="Agent İzleri (LangSmith)", icon=":material/account_tree:"),
                      st.Page("sayfalar/saglik.py", title="Sistem Sağlığı", icon=":material/health_and_safety:")],
    "AEP Grid Agent": [st.Page("sayfalar/aep_kararlar.py", title="Agent Kararları", icon=":material/electric_bolt:", url_path="aep_kararlar"),
                       st.Page("sayfalar/aep_tahmin.py", title="Tahmin Servisi", icon=":material/query_stats:", url_path="aep_tahmin")],
})
st.sidebar.caption("TR: EPİAŞ gerçek verisi, canlı gün öncesi karar · AEP: PJM ana projesi, simüle kapasite")
sayfalar.run()
