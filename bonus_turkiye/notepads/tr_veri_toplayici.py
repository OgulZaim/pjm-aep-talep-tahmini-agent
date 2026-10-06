"""tr_veri_toplayici.py - Canli veri toplayici.

D gunu icin karar zincirinin (tr_agent_zinciri.Zincir) ihtiyac duydugu ham tabloyu (HAM_SUTUNLAR) uretir.
Cikti, tekrar_oynatma_verisi() ile ayni yapidadir; zincir iki kaynagi ayirt etmeden calisir.

Kaynaklar:
  EPIAS (eptr2, EPTR_USERNAME / EPTR_PASSWORD .env'den):
    rt-cons   : gerceklesen tuketim, D-60 -> son yayimlanan saat (S+2 gecikmeli)
    load-plan : resmi yuk tahmin plani, D-60 -> D (D'nin plani G-1'de yayimlanir; saati belirtilmemis)
    eak       : emre amade kapasite, D-60 -> D (region="TR1")
  Open-Meteo (API anahtari gerekmez; ticari olmayan kullanim):
    Forecast API       : D gunu, 6 il, gfs_seamless, saatlik sicaklik
    Previous Runs API  : sapma penceresi (D-38 -> D-9), temperature_2m_previous_day1, gfs_seamless
    Historical (ERA5)  : sapma penceresi (D-38 -> D-9)
    Forecast API       : Ankara, D-1 gunluk ortalama (yedek modellerin "dunun sicakligi" ozelligi icin)

Onemli sinirlar (ciktida uyari olarak da raporlanir):
  - Yedek modeller egitimde NOAA'nin gerceklesen Ankara TAVG degerini kullandi. Canlida D-1'in gerceklesen degeri
    henuz yayimlanmamistir; Open-Meteo'nun gecmis gunler icin dondurdugu degerler de gerceklesen olcum degil,
    tahmindir. Bu yuzden bu ozellik canlida yaklasik bir degerdir.
  - Forecast API'de start_date/end_date ile past_days birlikte kullanilamaz; burada yalnizca tarih araligi kullaniliyor.

Onbellek: data/canli/<D>/ klasoru. Tekrar calistirmada API'ye gidilmez (yenile=True ile yenilenir).
D gununun resmi plani 24 saatten eksikse onbellege yazilmaz ve her calistirmada yeniden istenir.
Plan yayin kaydi: data/tr_plan_yayin_kaydi.csv (kontrol zamani, D, plandaki saat sayisi).
EAK yayin kaydi: data/tr_eak_yayin_kaydi.csv (kontrol zamani, D, EAK saat sayisi, hata mesaji).
D gununun EAK'si alinamazsa (henuz yayimlanmamis olabilir) zincir durmaz: D-1'in EAK profili YAKLASIK olarak
kullanilir ve karar "gecici" uyarisiyla uretilir.
"""
import os
import time

import numpy as np
import pandas as pd
import requests

from tr_agent_zinciri import HAM_SUTUNLAR, saatlik_aralik, takvim_ozellikleri

SEHIR_KOORDINAT = {"istanbul": (41.01, 28.98), "ankara": (39.93, 32.86), "izmir": (38.42, 27.14),
                   "bursa": (40.19, 29.06), "antalya": (36.90, 30.70), "adana": (37.00, 35.32)}
URL_TAHMIN = "https://api.open-meteo.com/v1/forecast"
URL_ONCEKI = "https://previous-runs-api.open-meteo.com/v1/forecast"
URL_ARSIV = "https://archive-api.open-meteo.com/v1/archive"


# ----------------------------------------------------------------------------------------------------
# EPIAS
# ----------------------------------------------------------------------------------------------------
def epias_istemci():
    """EPIAS oturumu. Giris basarisiz olursa ham sunucu yaniti degil, temizlenmis ozet ve cozum onerisi verilir."""
    from eptr2 import EPTR2
    if not os.getenv("EPTR_USERNAME") or not os.getenv("EPTR_PASSWORD"):
        raise EpiasHatasi("EPTR_USERNAME veya EPTR_PASSWORD tanimli degil (notepads/.env).")
    try:
        return EPTR2(username=os.getenv("EPTR_USERNAME"), password=os.getenv("EPTR_PASSWORD"), use_dotenv=False)
    except Exception as ex:
        ozet = epias_hata_ozeti(ex)
        ipucu = ""
        if "401" in ozet or "Authentication" in ozet or "Credentials" in ozet:
            ipucu = (" -> EPIAS giris bilgilerini reddetti. Sifre degistiyse notepads/.env icindeki EPTR_PASSWORD'u guncelleyin "
                     "ve kernel'i yeniden baslatin. Ozel karakter iceren sifreyi tek tirnak icine alin: EPTR_PASSWORD='...'")
        raise EpiasHatasi(f"EPIAS oturumu acilamadi: {ozet}{ipucu}") from None


class EpiasHatasi(RuntimeError):
    pass


def epias_hata_ozeti(ex):
    """EPIAS hata yanitindan yalnizca durum, hata kodu ve hata aciklamasini alir.
    Yanit, kullanici adini (e-posta), istemci IP'sini ve sunucu bilgilerini de icerir; bunlar Agent'a, LangSmith'e
    ve dosyalara gitmemelidir (veri minimizasyonu). Bu yuzden ham metin hicbir yere tasinmaz."""
    import json as _json
    import re as _re
    metin = str(ex)
    parca = []
    i, j = metin.find("{"), metin.rfind("}")
    if i >= 0 and j > i:
        try:
            js = _json.loads(metin[i:j + 1])
            durum = _re.findall(r"status code:\s*(\d{3})", metin)
            if durum and not js.get("status"):
                parca.append(f"HTTP {durum[0]}")
            if js.get("status"):
                parca.append(str(js["status"]))
            for e in js.get("errors") or []:
                parca.append(f"{e.get('errorCode', '')}: {e.get('errorMessage', '')}".strip(": "))
            for e in js.get("authentication_exceptions") or []:
                if isinstance(e, list):
                    parca += [str(x)[:150] for x in e]
        except ValueError:
            pass
    if not parca:
        kod = _re.findall(r'"errorCode"\s*:\s*"([^"]*)"', metin)
        acik = _re.findall(r'"errorMessage"\s*:\s*"([^"]*)"', metin)
        durum = _re.findall(r"status code:\s*(\d{3})", metin)
        parca = ([f"HTTP {durum[0]}"] if durum else []) + [f"{k}: {a}" for k, a in zip(kod, acik + [""] * len(kod))]
    if not parca:
        parca = [type(ex).__name__ + ": " + _re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+|\b\d{1,3}(?:\.\d{1,3}){3}\b", "[gizlendi]", metin)[:150]]
    return " | ".join(parca)


def cagir_sabirli(istemci, anahtar, bas, bit, **ek):
    """429 (cok fazla istek): 30-60-120-240 sn bekleyerek en fazla 5 deneme (Faz 1 kurali).
    Diger hatalar: yeni oturumla en fazla 2 deneme, 5 sn arayla. Bu hatalar genellikle beklemekle gecmez
    (ornegin henuz yayimlanmamis bir gun); 7,5 dakika beklemek gereksizdir.
    Son hata mesaji istisnanin metnine eklenir; boylece nedeni traceback'te gorunur."""
    son_hata, diger = "", 0
    for deneme, bekle in enumerate([0, 30, 60, 120, 240], start=1):
        time.sleep(bekle if "429" in son_hata or deneme == 1 else 5)
        try:
            return istemci["eptr"].call(anahtar, start_date=bas, end_date=bit, **ek)
        except Exception as e:
            son_hata = epias_hata_ozeti(e)
            print(f"  {anahtar} {bas}-{bit} deneme {deneme}: {son_hata}")
            if "429" not in son_hata:
                diger += 1
                if diger >= 2:
                    break
                istemci["eptr"] = epias_istemci()
    raise EpiasHatasi(f"EPIAS {anahtar} {bas}-{bit} indirilemedi. Son hata: {son_hata}")


def _epias_df(istemci, anahtar, bas, bit, sutunlar, **ek):
    """Aylik parcalar halinde indirir; bos yanitta istenen sutunlarla bos tablo dondurur."""
    parcalar = []
    for ay in pd.date_range(pd.Timestamp(bas).replace(day=1), bit, freq="MS"):
        b = max(ay, pd.Timestamp(bas)); s = min(ay + pd.offsets.MonthEnd(0), pd.Timestamp(bit))
        df = cagir_sabirli(istemci, anahtar, b.strftime("%Y-%m-%d"), s.strftime("%Y-%m-%d"), **ek)
        if df is not None and len(df):
            parcalar.append(df)
        time.sleep(1.0)
    if not parcalar:
        return pd.DataFrame({c: pd.Series(dtype="datetime64[ns, UTC]" if c == "date" else "float64") for c in ["date"] + sutunlar})
    d = pd.concat(parcalar, ignore_index=True)
    d["date"] = pd.to_datetime(d["date"])
    return d.drop_duplicates("date")[["date"] + sutunlar]


# ----------------------------------------------------------------------------------------------------
# Open-Meteo
# ----------------------------------------------------------------------------------------------------
def om_get(url, params):
    for deneme, bekle in enumerate([0, 5, 15, 30], start=1):      # en kotu durumda istek basina ~2,5 dk
        time.sleep(bekle)
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            print(f"  Open-Meteo deneme {deneme}: HTTP {r.status_code} {r.text[:150]}")
            if r.status_code not in (429, 500, 502, 503, 504):
                break
        except requests.RequestException as e:
            print(f"  Open-Meteo deneme {deneme}: {str(e)[:150]}")
    raise RuntimeError(f"Open-Meteo istegi basarisiz: {url} {params.get('start_date')}")


def om_seri(js, degisken):
    zaman = [t + ":00+03:00" for t in js["hourly"]["time"]]
    return pd.Series(js["hourly"][degisken], index=pd.to_datetime(zaman), dtype="float64")


def om_sehirler(url, degisken, bas, bit, sehirler, ek=None):
    """Her il icin saatlik seri; bas-bit gun dahil."""
    out = {}
    for s in sehirler:
        enl, boy = SEHIR_KOORDINAT[s]
        js = om_get(url, {"latitude": enl, "longitude": boy, "start_date": str(pd.Timestamp(bas).date()),
                          "end_date": str(pd.Timestamp(bit).date()), "hourly": degisken,
                          "timezone": "Europe/Istanbul", **(ek or {})})
        out[s] = om_seri(js, degisken)
        time.sleep(0.5)
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------------------------------
# Onbellek
# ----------------------------------------------------------------------------------------------------
def _onbellekli(yol, uret, yenile, kaydet_kosulu=lambda d: True):
    if os.path.exists(yol) and not yenile:
        d = pd.read_csv(yol)
        if "date" in d:
            d["date"] = pd.to_datetime(d["date"])
        return d, True
    d = uret()
    if kaydet_kosulu(d):
        d.to_csv(yol, index=False)
    return d, False


def plan_yayin_kaydi(kok, gun, n_saat):
    yol = os.path.join(kok, "data", "tr_plan_yayin_kaydi.csv")
    satir = pd.DataFrame([{"kontrol_zamani": pd.Timestamp.now(tz="Europe/Istanbul").isoformat(timespec="seconds"), "gun": str(gun.date()),
                           "plandaki_saat": int(n_saat)}])
    satir.to_csv(yol, mode="a", header=not os.path.exists(yol), index=False)


def eak_yayin_kaydi(kok, gun, n_saat, hata=""):
    yol = os.path.join(kok, "data", "tr_eak_yayin_kaydi.csv")
    satir = pd.DataFrame([{"kontrol_zamani": pd.Timestamp.now(tz="Europe/Istanbul").isoformat(timespec="seconds"),
                           "gun": str(gun.date()), "eak_saat": int(n_saat), "hata": hata.split("Son hata: ")[-1][:200]}])
    satir.to_csv(yol, mode="a", header=not os.path.exists(yol), index=False)


# ----------------------------------------------------------------------------------------------------
# Ana fonksiyon
# ----------------------------------------------------------------------------------------------------
def canli_veri(kok, gun, meta_secilen, gecmis_gun=60, yenile=False, istemci=None):
    """D gunu (gun) icin ham tablo ve bilgi sozlugu dondurur. Bilgi sozlugu Zincir.calistir(kaynak_bilgisi=...)
    ile zincire verilir; boylece veri uyarilari Agent'a ve ciktiya ulasir.

    gecmis_gun: EAK sicrama esigi ve gunes referansi icin gereken gecmis (en az 60 gun onerilir).
    """
    from dotenv import load_dotenv
    load_dotenv(os.path.join(kok, "notepads", ".env"), override=True)   # .env degistiyse kernel'deki eski degerin uzerine yaz
    D = pd.Timestamp(gun).normalize()
    bas = D - pd.Timedelta(days=gecmis_gun)
    klasor = os.path.join(kok, "data", "canli", str(D.date()))
    os.makedirs(klasor, exist_ok=True)
    idx = saatlik_aralik(bas, D)
    uyarilar, kaynak = [], {}
    istemci = istemci or {"eptr": epias_istemci()}

    print(f"[{pd.Timestamp.now(tz='Europe/Istanbul'):%H:%M:%S}] EPIAS: tuketim, plan ve EAK aliniyor...", flush=True)
    # --- EPIAS ---
    tuk, onb = _onbellekli(os.path.join(klasor, "epias_tuketim.csv"),
                           lambda: _epias_df(istemci, "rt-cons", bas, D - pd.Timedelta(days=1), ["consumption"]),
                           yenile, kaydet_kosulu=lambda d: False)    # tuketim her calistirmada guncel alinir
    kaynak["tuketim"] = f"rt-cons, son saat {tuk['date'].max() if len(tuk) else 'yok'}"
    plan, onb = _onbellekli(os.path.join(klasor, "epias_plan.csv"),
                            lambda: _epias_df(istemci, "load-plan", bas, D, ["lep"]), yenile,
                            kaydet_kosulu=lambda d: (d["date"].dt.tz_localize(None).dt.normalize() == D).sum() == 24)
    n_plan = int((plan["date"].dt.tz_localize(None).dt.normalize() == D).sum()) if len(plan) else 0
    plan_yayin_kaydi(kok, D, n_plan)
    kaynak["plan"] = f"load-plan, D icin {n_plan}/24 saat" + (" (onbellek)" if onb else "")
    if n_plan < 24:
        uyarilar.append(f"D gununun resmi plani henuz tam degil ({n_plan}/24 saat); plansiz yedek model devreye girer")
    eak_gecmis, _ = _onbellekli(os.path.join(klasor, "epias_eak_gecmis.csv"),
                                lambda: _epias_df(istemci, "eak", bas, D - pd.Timedelta(days=1), ["toplam", "gunes"], region="TR1"),
                                yenile)
    eak_d_hata = ""
    try:
        eak_d, onb = _onbellekli(os.path.join(klasor, "epias_eak_D.csv"),
                                 lambda: _epias_df(istemci, "eak", D, D, ["toplam", "gunes"], region="TR1"), yenile,
                                 kaydet_kosulu=lambda d: (d["date"].dt.tz_localize(None).dt.normalize() == D).sum() == 24)
    except EpiasHatasi as ex:
        eak_d, onb, eak_d_hata = eak_gecmis.iloc[0:0], False, str(ex)
    n_eak = int((eak_d["date"].dt.tz_localize(None).dt.normalize() == D).sum()) if len(eak_d) else 0
    eak_yayin_kaydi(kok, D, n_eak, eak_d_hata)
    kaynak["eak"] = f"eak, D icin {n_eak}/24 saat" + (" (onbellek)" if onb else "")
    eak = pd.concat([eak_gecmis, eak_d], ignore_index=True)
    if n_eak < 24:
        d1 = D - pd.Timedelta(days=1)
        vekil = eak_gecmis[eak_gecmis["date"].dt.tz_localize(None).dt.normalize() == d1].copy()
        if len(vekil) == 24:
            vekil["date"] = vekil["date"] + pd.Timedelta(days=1)
            eak = pd.concat([eak_gecmis, vekil], ignore_index=True)
            kaynak["eak"] += " -> YAKLASIK: D-1 profili kullanildi"
            uyarilar.append("D gununun EAK'si henuz alinamadi" + (f" ({eak_d_hata.split('Son hata: ')[-1]})" if eak_d_hata else "") +
                            "; D-1'in EAK profili YAKLASIK olarak kullanildi. Karar GECICIDIR: katilimcilar EAK'yi 14:00-16:00 "
                            "arasinda bildiriyor; yayimlandiktan sonra yeniden calistirin.")
        else:
            uyarilar.append("D gununun EAK'si alinamadi ve D-1 profili de eksik; marj hesaplanamaz" +
                            (f" ({eak_d_hata.split('Son hata: ')[-1]})" if eak_d_hata else ""))

    print(f"[{pd.Timestamp.now(tz='Europe/Istanbul'):%H:%M:%S}] Open-Meteo: hava tahmini ve sapma penceresi aliniyor...", flush=True)
    # --- Open-Meteo ---
    agirlik = meta_secilen.get("sehir_agirliklari") or {}
    sehirler = list(agirlik)
    om_fc = pd.Series(np.nan, index=idx)
    om_era = pd.Series(np.nan, index=idx)
    if sehirler:
        w = pd.Series(agirlik)
        fc_d, _ = _onbellekli(os.path.join(klasor, "om_tahmin_D.csv"),
                              lambda: om_sehirler(URL_TAHMIN, "temperature_2m", D, D, sehirler, {"models": "gfs_seamless"})
                              .rename_axis("date").reset_index(), yenile)
        sap_bas, sap_bit = D - pd.Timedelta(days=38), D - pd.Timedelta(days=9)
        fc_p, _ = _onbellekli(os.path.join(klasor, "om_onceki_sapma.csv"),
                              lambda: om_sehirler(URL_ONCEKI, "temperature_2m_previous_day1", sap_bas, sap_bit, sehirler,
                                                  {"models": "gfs_seamless"}).rename_axis("date").reset_index(), yenile)
        era, _ = _onbellekli(os.path.join(klasor, "om_era5_sapma.csv"),
                             lambda: om_sehirler(URL_ARSIV, "temperature_2m", sap_bas, sap_bit, sehirler)
                             .rename_axis("date").reset_index(), yenile)
        agir = lambda d: (d.set_index("date")[sehirler] * w).sum(axis=1, min_count=len(sehirler))
        om_fc = pd.concat([agir(fc_p), agir(fc_d)]).groupby(level=0).last().reindex(idx)
        om_era = agir(era).reindex(idx)
        kaynak["hava"] = (f"GFS D: {int(agir(fc_d).notna().sum())}/24 saat | sapma penceresi {sap_bas.date()} - {sap_bit.date()}: "
                          f"tahmin {int(agir(fc_p).notna().sum())}, ERA5 {int(om_era.notna().sum())} saat")
        if agir(fc_d).notna().sum() < 24:
            uyarilar.append("D gunu icin hava tahmini eksik; hava tahminsiz yedek model devreye girer")

    # --- Ankara D-1 (yedek modeller icin, yaklasik) ---
    d1 = D - pd.Timedelta(days=1)
    ank, _ = _onbellekli(os.path.join(klasor, "om_ankara_D-1.csv"),
                         lambda: om_sehirler(URL_TAHMIN, "temperature_2m", d1, d1, ["ankara"]).rename_axis("date").reset_index(),
                         yenile)
    tavg_d1 = float(ank["ankara"].mean()) if len(ank) else np.nan
    yedek_notu = ("Yedek modelin 'dunun Ankara sicakligi' ozelligi canlida Open-Meteo tahmininden alindi "
                  "(egitimde NOAA gerceklesen TAVG kullanilmisti); deger yaklasiktir")

    print(f"[{pd.Timestamp.now(tz='Europe/Istanbul'):%H:%M:%S}] Veri toplama tamam.", flush=True)
    # --- Ham tablo ---
    ham = takvim_ozellikleri(idx, kok)
    ham["consumption"] = tuk.set_index("date")["consumption"].reindex(idx).values if len(tuk) else np.nan
    ham["lep_ham"] = plan.set_index("date")["lep"].reindex(idx).values if len(plan) else np.nan
    ham["eak_toplam"] = eak.set_index("date")["toplam"].reindex(idx).values if len(eak) else np.nan
    ham["eak_gunes"] = eak.set_index("date")["gunes"].reindex(idx).values if len(eak) else np.nan
    ham["om_fc_ham"] = om_fc.values
    ham["om_era5"] = om_era.values
    ham["tavg_c"] = np.where(ham["date"].dt.tz_localize(None).dt.normalize() == d1, tavg_d1, np.nan)
    bilgi = {"gun": str(D.date()), "olusturma": pd.Timestamp.now(tz="Europe/Istanbul").isoformat(timespec="seconds"), "kaynak": kaynak,
             "uyarilar": uyarilar, "yedek_model_notu": yedek_notu, "onbellek_klasoru": klasor}
    return ham[HAM_SUTUNLAR], bilgi
