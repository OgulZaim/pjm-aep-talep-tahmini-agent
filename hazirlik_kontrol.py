from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
from dotenv import dotenv_values

KOK = Path(r"C:\Deep_Learning_to_AI_Agents\Bitirme_Proje")
ENV = KOK / "notepads" / ".env"

# 1) .env.example: yalnizca anahtar adlari, degerler bos
print("=== 1) .env.example ===")
anahtarlar = list(dotenv_values(ENV).keys())
(KOK / "notepads" / ".env.example").write_text("\n".join(f"{k}=" for k in anahtarlar) + "\n", encoding="utf-8")
print("Olusturuldu. Anahtar adlari:", anahtarlar)

# 2) requirements.txt: kullanilan paketlerin kurulu surumleri
print("\n=== 2) requirements.txt ===")
paketler = ["tensorflow", "keras", "xgboost", "shap", "scikit-learn", "pandas", "numpy",
            "matplotlib", "joblib", "google-genai", "chromadb", "langsmith", "fastapi",
            "uvicorn", "pydantic", "pypdf", "python-dotenv", "requests"]
satirlar = []
for p in paketler:
    try:
        satirlar.append(f"{p}=={version(p)}")
    except PackageNotFoundError:
        print("  Kurulu degil, atlandi:", p)
(KOK / "requirements.txt").write_text("\n".join(satirlar) + "\n", encoding="utf-8")
print("\n".join(satirlar))

# 3) Gerekli dosyalar
print("\n=== 3) Gerekli dosyalar ===")
gerekli = [
    "README.md", ".gitignore", "requirements.txt", "notepads/.env.example",
    "notebooks/1.ipynb", "notebooks/RAG.ipynb", "notebooks/2.ipynb", "notebooks/3.ipynb",
    "notepads/prompt_v1.py", "notepads/rag_retrieval.py", "notepads/agent_test.py",
    "notepads/scenario_comparison.py", "notepads/test_api.py",
    "agent/api_service.py", "agent/grid_model.keras", "agent/grid_scaler.pkl", "agent/grid_xgb_model.json",
    "n8n/Grid Demand Response Automation.json",
    "data/AEP_hourly.csv", "data/aep_with_features.csv", "metrics/monthly_summary_full_period.csv",
    "docs/sunum.pptx", "docs/konusma_metni.md", "docs/bitirme-bastan-sona-ozet.md",
    "docs/belge1_teknikler.md", "docs/belge2_metrikler.md", "docs/belge3_egitmen_raporu.md",
    "docs/bolum1_faz1_2.md", "docs/bolum2_faz3_4.md", "docs/bolum3_faz5_6.md",
    "docs/bolum4_faz8_9.md", "docs/bolum5_faz10_11.md", "docs/dosya_envanteri.md",
    "docs/hata_denetimi.md", "docs/paydas_ozeti.md", "docs/baştan_sona_ozet.md",
    "docs/saatlik_hata_tablolari.md", "docs/xgboost_walkforward_ocak_analizi.md",
]
eksik = [g for g in gerekli if not (KOK / g).exists()]
print("Eksik dosyalar:", eksik if eksik else "yok")
print("n8n_test_*.py sayisi:", len(list((KOK / "notepads").glob("n8n_test_*.py"))), "(4 bekleniyor)")
rr = KOK / "notepads" / "rag_retrieval.py"
if rr.exists():
    print("rag_retrieval.py build_rag_query iceriyor mu:", "build_rag_query" in rr.read_text(encoding="utf-8", errors="ignore"))
gi = KOK / ".gitignore"
if gi.exists():
    print(".gitignore icinde .env kurali var mi:", ".env" in gi.read_text(encoding="utf-8"))

# 4) Gizli bilgi taramasi (degerler yazdirilmaz)
print("\n=== 4) Gizli bilgi taramasi ===")
gizliler = {k: v for k, v in dotenv_values(ENV).items() if v and len(v) >= 8}
atla = {".git", "chroma_store", "__pycache__", ".ipynb_checkpoints"}
dosyalar = [f for f in KOK.rglob("*") if f.is_file() and not any(p in atla for p in f.parts)]
bulgu = 0
for f in dosyalar:
    if f.name == ".env":
        continue
    metin = f.read_text(encoding="utf-8", errors="ignore")
    for ad, deger in gizliler.items():
        if deger in metin:
            print(f"  UYARI: {ad} degeri bulundu -> {f.relative_to(KOK)}")
            bulgu += 1
print("Bulgu sayisi:", bulgu, "(0 olmali)")

# 5) Boyut kontrolu
print("\n=== 5) Boyut ===")
for f in sorted(dosyalar, key=lambda x: x.stat().st_size, reverse=True)[:5]:
    print(f"  {f.stat().st_size / 1e6:7.1f} MB  {f.relative_to(KOK)}")
asan = [str(f.relative_to(KOK)) for f in dosyalar if f.stat().st_size > 50e6]
print("50 MB ustu dosyalar:", asan if asan else "yok")