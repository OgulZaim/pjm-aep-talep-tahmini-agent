"""kaynaklar.py - Streamlit uygulamasinin veri katmani.

Ince arayuz ilkesi: Uygulama modelleri yuklemez ve zinciri calistirmaz; butun karar verisi TR Grid Agent API'den gelir.
Ek kaynaklar: n8n REST API (otomasyon durumu) ve LangSmith SDK (son 14 gunun izleri).
Anahtarlar Bonus projesinin notepads/.env dosyasindan okunur; koda veya arayuze yazilmaz.
"""
import os
from datetime import datetime, timedelta, timezone

import requests
import streamlit as st
from dotenv import load_dotenv

KOK = os.environ.get("TR_KOK", r"C:\Deep_Learning_to_AI_Agents\Bitirme_Proje_Bonus")
load_dotenv(os.path.join(KOK, "notepads", ".env"))
API_URL = os.getenv("TR_API_URL", "http://127.0.0.1:8001")
N8N_URL = os.getenv("N8N_URL", "http://localhost:5678")
PJM_API_URL = os.getenv("PJM_API_URL", "http://127.0.0.1:8000")
PJM_SERVIS_KOMUTU = "cd /d C:\\Deep_Learning_to_AI_Agents\\Bitirme_Proje\\agent && uvicorn api_service:app --reload"
N8N_WORKFLOW = os.getenv("N8N_WORKFLOW_ADI", "TR Grid Agent")
N8N_WORKFLOW_AEP = os.getenv("N8N_WORKFLOW_AEP", "Grid Demand Response Automation")
LS_PROJELERI = [p.strip() for p in os.getenv("STREAMLIT_LS_PROJELERI", "sunum-izleri,tr-grid-agent").split(",") if p.strip()]
SERVIS_KOMUTU = "uvicorn tr_api_service:app --host 0.0.0.0 --port 8001"

KARAR_ETIKET = {"normal_operation": ("Normal işletim", "success"),
                "demand_response_alert": ("Talep tarafı uyarısı", "warning"),
                "activate_reserve": ("Rezerv devreye alınmalı", "error")}


# Yerel servisler (API, n8n) icin oturum: ortamdaki HTTP(S)_PROXY ayarlari yok sayilir. Aksi halde kurumsal bir
# proxy ayari, 127.0.0.1 ve localhost isteklerini de proxy'ye yonlendirip "ulasilamiyor" hatasina yol acabilir.
_yerel = requests.Session()
_yerel.trust_env = False


class KaynakHatasi(Exception):
    """Kullaniciya gosterilecek, anahtar veya kisisel bilgi icermeyen hata."""


# ---------------------------------------------------------------- TR Grid Agent API
def _api(yol, params=None, timeout=60):
    try:
        r = _yerel.get(f"{API_URL}{yol}", params=params, timeout=timeout,
                         headers={"X-API-Key": os.getenv("TR_API_ANAHTARI", "")})
    except requests.RequestException:
        raise KaynakHatasi(f"API'ye ulaşılamıyor ({API_URL}). Servisi başlatın: `{SERVIS_KOMUTU}`") from None
    if r.status_code != 200:
        try:
            ayrinti = r.json().get("detail", "")
        except ValueError:
            ayrinti = r.text[:200]
        raise KaynakHatasi(f"API yanıtı {r.status_code}: {ayrinti}")
    return r.json()


@st.cache_data(ttl=60, show_spinner=False)
def saglik():
    return _api("/saglik")


@st.cache_data(ttl=60, show_spinner=False)
def kararlar():
    return _api("/kararlar")["kararlar"]


@st.cache_data(ttl=60, show_spinner=False)
def karar(gun):
    return _api(f"/karar/{gun}", {"golge": "true"})


@st.cache_data(ttl=300, show_spinner=False)
def izleme():
    return _api("/izleme")


@st.cache_data(ttl=3600, show_spinner=False)
def tekrar(gun, model="golge"):
    return _api(f"/tekrar/{gun}", {"model": model}, timeout=300)


@st.cache_data(ttl=300, show_spinner=False)
def modeller():
    return _api("/modeller")["modeller"]


# ---------------------------------------------------------------- n8n
@st.cache_data(ttl=60, show_spinner=False)
def n8n_calismalari(ad=None, limit=30):
    """Adi verilen workflow'un son calismalari (varsayilan: TR Grid Agent)."""
    ad = ad or N8N_WORKFLOW
    anahtar = os.getenv("N8N_API_KEY")
    if not anahtar:
        raise KaynakHatasi("`N8N_API_KEY` tanımlı değil. n8n'de **Settings → n8n API → Create an API key** ile oluşturup "
                           "`notepads\\.env` dosyasına `N8N_API_KEY=...` olarak ekleyin.")
    h = {"X-N8N-API-KEY": anahtar}
    try:
        wf = _yerel.get(f"{N8N_URL}/api/v1/workflows", headers=h, params={"limit": 100}, timeout=15)
    except requests.RequestException:
        raise KaynakHatasi(f"n8n'e ulaşılamıyor ({N8N_URL}). Docker Desktop ve n8n konteyneri çalışıyor mu?") from None
    if wf.status_code != 200:
        raise KaynakHatasi(f"n8n API yanıtı {wf.status_code}; API anahtarını kontrol edin.")
    hepsi = wf.json().get("data", [])
    hedef = next((w for w in hepsi if w.get("name") == ad), None)
    if hedef is None:
        adlar = ", ".join(sorted(w.get("name", "?") for w in hepsi)) or "yok"
        raise KaynakHatasi(f"n8n'de '{ad}' adlı workflow bulunamadı. Mevcut workflow'lar: {adlar}. "
                           f"Ad farklıysa `.env`'de `N8N_WORKFLOW_AEP=` (ya da `N8N_WORKFLOW_ADI=`) ile belirtin.")
    ex = _yerel.get(f"{N8N_URL}/api/v1/executions", headers=h, timeout=15,
                      params={"workflowId": hedef["id"], "limit": limit, "includeData": "false"})
    if ex.status_code != 200:
        raise KaynakHatasi(f"n8n çalışma listesi alınamadı ({ex.status_code}).")
    return {"workflow": {"id": hedef["id"], "ad": hedef.get("name"), "aktif": hedef.get("active")},
            "calismalar": ex.json().get("data", [])}


# ---------------------------------------------------------------- LangSmith
def _ls_web():
    uc = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com").rstrip("/")
    return uc.replace("://api.", "://", 1).replace(".api.", ".", 1)


@st.cache_resource(show_spinner=False)
def _ls_istemci():
    from langsmith import Client
    return Client()


@st.cache_data(ttl=120, show_spinner=False)
def ls_izleri(proje, gun=14, limit=50):
    if not os.getenv("LANGSMITH_API_KEY"):
        raise KaynakHatasi("`LANGSMITH_API_KEY` tanımlı değil.")
    try:
        runs = list(_ls_istemci().list_runs(project_name=proje, is_root=True, limit=limit,
                                            start_time=datetime.now(timezone.utc) - timedelta(days=gun)))
    except Exception as ex:
        raise KaynakHatasi(f"LangSmith '{proje}' projesi okunamadı: {type(ex).__name__}") from None
    web = _ls_web()
    return [{"iz": r.name, "baslangic": r.start_time,
             "sure_sn": round((r.end_time - r.start_time).total_seconds(), 1) if r.end_time and r.start_time else None,
             "durum": r.status, "token": r.total_tokens, "maliyet_usd": float(r.total_cost) if r.total_cost is not None else None,
             "hata": r.error, "etiketler": ", ".join(r.tags or []),
             "baglanti": f"{web}{r.app_path}" if getattr(r, "app_path", None) else None,
             "girdi": r.inputs, "cikti": r.outputs} for r in runs]


# ---------------------------------------------------------------- AEP (ana proje) tahmin servisi
@st.cache_data(ttl=30, show_spinner=False)
def pjm_saglik():
    try:
        r = _yerel.get(f"{PJM_API_URL}/health", timeout=10)
    except requests.RequestException:
        raise KaynakHatasi(f"AEP tahmin servisine ulaşılamıyor ({PJM_API_URL}). Başlatın: `{PJM_SERVIS_KOMUTU}`") from None
    if r.status_code != 200:
        raise KaynakHatasi(f"AEP servisi yanıtı {r.status_code}")
    return r.json()


def pjm_tahmin(readings):
    """Son 24 saatin olceklenmemis 9 ozelligi (24 x 9) -> bir saat sonrasinin tahmini (MW)."""
    try:
        r = _yerel.post(f"{PJM_API_URL}/predict", json={"readings": readings}, timeout=60)
    except requests.RequestException:
        raise KaynakHatasi(f"AEP tahmin servisine ulaşılamıyor ({PJM_API_URL}).") from None
    if r.status_code != 200:
        raise KaynakHatasi(f"AEP servisi yanıtı {r.status_code}: {r.text[:200]}")
    j = r.json()
    if "error" in j:
        raise KaynakHatasi(f"AEP servisi: {j['error']}")
    return j
