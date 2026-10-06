"""tr_api_service.py - Turkiye gun oncesi karar zinciri icin FastAPI servisi.

Calistirma (Anaconda Prompt):
    conda activate bitirme-proje
    cd C:\\Deep_Learning_to_AI_Agents\\Bitirme_Proje_Bonus\\agent
    uvicorn tr_api_service:app --host 127.0.0.1 --port 8001
Arayuz: http://127.0.0.1:8001/docs   (PJM servisi 8000'de calistigi icin 8001)

Uc noktalar:
    GET  /saglik                 Model surumleri, dosyalar, son canli calistirma
    GET  /kararlar               Kayitli canli kararlarin listesi (Streamlit tarih secimi icin)
    GET  /karar/{gun}            Kaydedilmis karar (data/canli/<gun>/sonuc.json); yeniden hesaplamaz
    POST /karar/calistir         Canli zincir: EPIAS + Open-Meteo -> karar; 7_canli_tr B bolumuyle ayni kayit
    GET  /tekrar/{gun}           Gecmis bir gun icin yeniden oynatma (varsayilan: golge/3c modeli; Agent cagrilmaz)
    GET  /modeller               Uretim (3d) ve golge (3c) modellerinin egitim ve test donemleri
    GET  /izleme                 Canli tahminlerin gerceklesenle karsilastirmasi (metrics/tr_canli_izleme.csv)

Guvenlik:
    - Varsayilan olarak yalnizca bu bilgisayardan erisim (127.0.0.1).
    - .env'de TR_API_ANAHTARI tanimliysa, her istekte "X-API-Key" basligi zorunlu.
    - Her yanit, gondermeden once e-posta ve IP adresi icin denetlenir; bulunursa yanit verilmez (HTTP 500).
"""
import os
import re
import sys
import glob
import json
import threading

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse

KOK = os.environ.get("TR_KOK", r"C:\Deep_Learning_to_AI_Agents\Bitirme_Proje_Bonus")
sys.path.insert(0, os.path.join(KOK, "notepads"))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(KOK, "notepads", ".env"), override=True)
# .env'de baska bir proje (or. PJM) yazili olsa bile Turkiye izleri kendi projesine gitsin
os.environ["LANGSMITH_PROJECT"] = os.getenv("TR_LANGSMITH_PROJECT", "tr-grid-agent")

from tr_agent_zinciri import (Zincir, tekrar_oynatma_verisi, sonuc_json, kisisel_bilgi_temizle,  # noqa: E402
                              EPOSTA_DESENI, IPV4_DESENI)
import tr_veri_toplayici as tv  # noqa: E402

app = FastAPI(title="TR Grid Agent API", version="1.0",
              description="Turkiye gun oncesi tuketim tahmini, EAK marji, karar ve mevzuat gerekceli aciklama")

@app.exception_handler(Exception)
async def genel_hata(request: Request, exc: Exception):
    """Yakalanmamis hatalar: istemciye yalnizca hata turu (ic ayrinti ve kisisel bilgi tasimaz),
    ayrinti API penceresine. HTTPException'lar FastAPI'nin kendi isleyicisiyle donmeye devam eder."""
    print(f"HATA {request.method} {request.url.path}: {type(exc).__name__}: {str(exc)[:300]}", flush=True)
    return JSONResponse(status_code=500, content={"detail": f"Sunucu hatasi ({type(exc).__name__}); ayrinti API penceresinde"})


_kilit = threading.Lock()          # zincir nesneleri paylasilan durum tutar; ayni anda tek calistirma
_durum = {"z": None, "z_eski": None, "tekrar_hazir": False, "tekrar_hazir_eski": False}

# Model donemleri (tek kaynak). 3c'nin meta dosyasinda egitim tarihleri yok; degerler 3c_capraz_dogrulama.ipynb'deki
# test kesimine gore: egitim < 2025-04-01, erken durdurma 2025-04 -> 2025-09, test 2025-10-01 -> 2026-09-22.
TEST_BAS, TEST_BIT = pd.Timestamp("2025-10-01"), pd.Timestamp("2026-09-22")
SECIM_MODELI = {"egitim": "2016-04-01 → 2025-03-31", "erken_durdurma": "2025-04-01 → 2025-09-30",
                "test": "2025-10-01 → 2026-09-22 (eğitimde görülmedi)",
                "sonuc": "Test yılı MAPE %2,29 (resmî plan %2,60); çapraz doğrulama 4/4 katmanda planı geçti"}


# ----------------------------------------------------------------------------------------------------
# Yardimcilar
# ----------------------------------------------------------------------------------------------------
def anahtar_kontrol(x_api_key: str | None = Header(default=None)):
    beklenen = os.getenv("TR_API_ANAHTARI")
    if beklenen and x_api_key != beklenen:
        raise HTTPException(status_code=401, detail="Gecersiz veya eksik X-API-Key")


def _nan_temizle(o):
    """NaN ve sonsuz degerleri None'a cevirir. JSON standardi NaN'i desteklemez; Starlette yaniti allow_nan=False ile
    olusturdugu icin NaN iceren bir yanit 500 hatasi verir. Kayit dosyalarinda (json.dump) NaN bulunabilir:
    ornegin resmi planin eksik oldugu saatler."""
    if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
        return None
    if isinstance(o, dict):
        return {k: _nan_temizle(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_nan_temizle(v) for v in o]
    return o


def gizlilik_denetimi(yanit):
    yanit = _nan_temizle(yanit)
    metin = json.dumps(yanit, ensure_ascii=False, default=str)
    if re.search(EPOSTA_DESENI, metin) or re.search(IPV4_DESENI, metin):
        raise HTTPException(status_code=500, detail="Gizlilik denetimi basarisiz: yanit kisisel bilgi iceriyor; gonderilmedi")
    return yanit


def zincir():
    if _durum["z"] is None:
        _durum["z"] = Zincir(KOK)
        arsivler = sorted(glob.glob(os.path.join(KOK, "agent", "arsiv", "*")))
        _durum["z_eski"] = Zincir(KOK, agent_cagir=False, model_klasoru=arsivler[-1]) if arsivler else None
    return _durum["z"]


def json_yaz(yol, veri):
    """Atomik yazma: once gecici dosyaya, sonra tek adimda yerine (os.replace). Yazma sirasinda okuyan bir istek
    yarim dosya gormez. NaN degerler standart JSON'a uygun olarak null yazilir."""
    gecici = yol + ".tmp"
    with open(gecici, "w", encoding="utf-8") as f:
        json.dump(_nan_temizle(veri), f, ensure_ascii=False, indent=2, default=str)
    os.replace(gecici, yol)


def json_oku(yol):
    try:
        with open(yol, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        raise HTTPException(status_code=503, detail="Kayit su anda okunamadi (yaziliyor olabilir); birkac saniye sonra tekrar deneyin")


def gun_dogrula(gun):
    try:
        return pd.Timestamp(gun).normalize()
    except Exception:
        raise HTTPException(status_code=400, detail="Tarih bicimi YYYY-AA-GG olmali")


# ----------------------------------------------------------------------------------------------------
# Uc noktalar
# ----------------------------------------------------------------------------------------------------
@app.get("/saglik", dependencies=[Depends(anahtar_kontrol)])
def saglik():
    z = zincir()
    kayitlar = sorted(glob.glob(os.path.join(KOK, "data", "canli", "*", "sonuc.json")))
    tk = pd.read_csv(os.path.join(KOK, "data", "tr_tatil_takvimi.csv"), parse_dates=["yerel_gun"])
    return gizlilik_denetimi({
        "durum": "calisiyor",
        "uretim_modeli": {"model": z.meta_secilen["model"], "surum": z.surum},
        "golge_modeli": _durum["z_eski"].surum if _durum["z_eski"] else None,
        "esikler": {k: z.esik[k] for k in ["cok_nadir_alti", "nadir_alti"]},
        "son_canli_kayit": os.path.basename(os.path.dirname(kayitlar[-1])) if kayitlar else None,
        "canli_kayit_sayisi": len(kayitlar),
        "tatil_takvimi_bitis": str(tk["yerel_gun"].max().date()),
        "api_anahtari_zorunlu": bool(os.getenv("TR_API_ANAHTARI")),
    })


@app.get("/karar/{gun}", dependencies=[Depends(anahtar_kontrol)])
def kayitli_karar(gun: str, golge: bool = Query(False, description="Onceki modelin kaydini da dondur")):
    D = gun_dogrula(gun)
    yol = os.path.join(KOK, "data", "canli", str(D.date()), "sonuc.json")
    if not os.path.exists(yol):
        raise HTTPException(status_code=404, detail=f"{D.date()} icin kayitli karar yok; once /karar/calistir")
    yanit = json_oku(yol)
    yanit.pop("toplayici_bilgisi", None)
    yol_e = os.path.join(os.path.dirname(yol), "sonuc_eski_model.json")
    if golge and os.path.exists(yol_e):
        yanit["golge_model"] = {k: v for k, v in json_oku(yol_e).items() if k in ("ozet", "model_surumu", "saatlik")}
    return gizlilik_denetimi(yanit)


@app.get("/kararlar", dependencies=[Depends(anahtar_kontrol)])
def kararlar(limit: int = Query(60, ge=1, le=365)):
    """Kayitli canli kararlarin listesi (en yeni once): gun, karar, en dar saat, marj, model surumu, golge kaydi var mi."""
    satirlar = []
    for yol in sorted(glob.glob(os.path.join(KOK, "data", "canli", "*", "sonuc.json")), reverse=True)[:limit]:
        try:
            with open(yol, encoding="utf-8") as f:
                s = json.load(f)
        except (OSError, ValueError):
            continue
        o, c = s.get("ozet", {}), s.get("cikti", {})
        satirlar.append({"gun": o.get("gun"), "karar": c.get("karar"), "oncelik": c.get("oncelik"),
                         "en_dar_saat": o.get("en_dar_saat"), "en_dusuk_marj": o.get("en_dusuk_marj"),
                         "model_surumu": s.get("model_surumu", "3c (ilk secim)"), "olusturma": s.get("olusturma"),
                         "golge_kaydi": os.path.exists(os.path.join(os.path.dirname(yol), "sonuc_eski_model.json"))})
    return gizlilik_denetimi({"kararlar": satirlar})


@app.post("/karar/calistir", dependencies=[Depends(anahtar_kontrol)])
def canli_calistir(gun: str | None = Query(None, description="Bos ise yarin (Istanbul)"),
                   agent: bool = Query(True, description="Gemini aciklamasi uretilsin mi"),
                   yenile: bool = Query(False, description="Onbellegi yok say")):
    simdi = pd.Timestamp.now(tz="Europe/Istanbul")
    D = gun_dogrula(gun) if gun else (simdi.normalize() + pd.Timedelta(days=1)).tz_localize(None)
    with _kilit:
        z = zincir()
        try:
            ham, bilgi = tv.canli_veri(KOK, D, z.meta_secilen, yenile=yenile)
        except tv.EpiasHatasi as ex:
            raise HTTPException(status_code=502, detail=str(ex))
        z.veri(ham)
        _durum["tekrar_hazir"] = False
        _durum["tekrar_hazir_eski"] = False
        r = z.calistir(D, kaynak_bilgisi=bilgi, agent=agent)
        kayit = sonuc_json(r)
        kayit.update({"model_surumu": z.surum, "olusturma": simdi.isoformat(),
                      "toplayici_bilgisi": kisisel_bilgi_temizle(bilgi)})
        klasor = os.path.join(KOK, "data", "canli", str(D.date()))
        json_yaz(os.path.join(klasor, "sonuc.json"), kayit)
        if _durum["z_eski"] is not None:
            ze = _durum["z_eski"]
            ze.veri(ham)
            k_e = sonuc_json(ze.calistir(D, kaynak_bilgisi=bilgi, agent=False))
            k_e.update({"model_surumu": ze.surum, "olusturma": simdi.isoformat()})
            json_yaz(os.path.join(klasor, "sonuc_eski_model.json"), k_e)
    kayit.pop("toplayici_bilgisi", None)
    return gizlilik_denetimi(kayit)


@app.get("/modeller", dependencies=[Depends(anahtar_kontrol)])
def modeller():
    """Uretim (3d) ve golge/secim (3c) modellerinin egitim, erken durdurma ve test donemleri."""
    z = zincir()
    m = z.meta_secilen
    satirlar = [{"rol": "Üretim (karar veriyor)", "surum": z.surum, "model": m.get("model"),
                 "egitim": f"2016-04-01 → {m.get('egitim_bitis', '?')}",
                 "erken_durdurma": f"yok (ağaç sayısı 3c'den sabit: {m.get('agac_sayisi', '?')})",
                 "test": "test yılı eğitimde; ayrı test verisi yok",
                 "sonuc": "Değerlendirme: canlı günler ve gölge karşılaştırma"}]
    if _durum["z_eski"] is not None:
        satirlar.append({"rol": "Gölge (önceki seçim, karar vermiyor)", "surum": _durum["z_eski"].surum,
                         "model": _durum["z_eski"].meta_secilen.get("model"), **SECIM_MODELI})
    return gizlilik_denetimi({"modeller": satirlar})


@app.get("/tekrar/{gun}", dependencies=[Depends(anahtar_kontrol)])
def tekrar_oynat(gun: str, model: str = Query("golge", pattern="^(golge|uretim)$",
                                              description="golge: test yilini gormemis secim modeli (3c); uretim: 3d")):
    """Gecmis bir gunun yeniden oynatilmasi. Varsayilan golge (3c) modeldir: test yilindaki bir gunu gormeden tahmin eder,
    boylece sonuc orneklem disidir."""
    D = gun_dogrula(gun)
    with _kilit:
        zincir()
        if model == "golge":
            if _durum["z_eski"] is None:
                raise HTTPException(status_code=404, detail="Golge (onceki) model bulunamadi (agent/arsiv bos)")
            z, bayrak = _durum["z_eski"], "tekrar_hazir_eski"
        else:
            z, bayrak = _durum["z"], "tekrar_hazir"
        if not _durum[bayrak]:
            z.veri(tekrar_oynatma_verisi(KOK, z.meta_secilen))
            _durum[bayrak] = True
        try:
            r = z.calistir(D, agent=False)
        except ValueError as ex:
            raise HTTPException(status_code=404, detail=str(ex))
    yanit = sonuc_json(r)
    yanit["model_surumu"] = z.surum
    if model == "golge":
        if TEST_BAS <= D <= TEST_BIT:
            yanit["not"] = ("Seçim modeli (3c) bu günü eğitimde görmedi (test yılı: 2025-10-01 – 2026-09-22); "
                            "sonuç örneklem dışıdır.")
        else:
            yanit["not"] = "Bu gün 3c'nin test yılı dışında; eğitimde veya erken durdurmada kullanıldı. Doğruluk ölçüsü değildir."
    else:
        bitis = pd.Timestamp(z.meta_secilen.get("egitim_bitis", "1900-01-01"))
        yanit["not"] = ("Üretim modeli (3d) bu günü eğitimde gördü; doğruluk ölçüsü değildir." if D <= bitis
                        else "Üretim modeli (3d) bu günü eğitimde görmedi.")
    return gizlilik_denetimi(yanit)


@app.get("/izleme", dependencies=[Depends(anahtar_kontrol)])
def izleme():
    yol = os.path.join(KOK, "metrics", "tr_canli_izleme.csv")
    if not os.path.exists(yol):
        raise HTTPException(status_code=404, detail="Henuz karsilastirma yok; 7_canli_tr D bolumunu calistirin")
    d = pd.read_csv(yol)
    ozet = {"gun_sayisi": len(d), "model_mape_ort": round(float(d["model_mape_%"].mean()), 3)}
    if "plan_mape_%" in d and d["plan_mape_%"].notna().any():
        ozet["plan_mape_ort"] = round(float(d["plan_mape_%"].mean()), 3)
    if "onceki_model_mape_%" in d and d["onceki_model_mape_%"].notna().any():
        ab = d.dropna(subset=["onceki_model_mape_%"])
        ozet["golge"] = {"gun": len(ab), "uretim_mape_ort": round(float(ab["model_mape_%"].mean()), 3),
                         "onceki_mape_ort": round(float(ab["onceki_model_mape_%"].mean()), 3),
                         "uretim_daha_iyi_gun": int((ab["model_mape_%"] < ab["onceki_model_mape_%"]).sum())}
    return gizlilik_denetimi({"ozet": ozet, "gunler": json.loads(d.to_json(orient="records"))})
