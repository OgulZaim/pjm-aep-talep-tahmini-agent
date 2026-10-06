"""tr_agent_zinciri.py - Turkiye gun oncesi karar zinciri.

Zincir: gun oncesi tahmin -> EAK marji -> deterministik karar -> RAG -> Agent aciklamasi -> kodla dogrulama.

Bu modul 6_agent_tr.ipynb'deki mantigin tek kaynagidir. Notebook'lar, canli veri toplayici
(tr_veri_toplayici.py) ve FastAPI servisi ayni kodu kullanir.

Veri kaynaklari ayni "ham tabloyu" (HAM_SUTUNLAR) uretir:
  - tekrar_oynatma_verisi(kok, meta): diskteki CSV'lerden (gecmis gunler)
  - tr_veri_toplayici.canli_veri(kok, gun, meta): EPIAS + Open-Meteo'dan (canli)
Ozellikler her iki durumda da ozellikleri_hazirla() ile ayni sekilde turetilir.

Kullanim:
    from tr_agent_zinciri import Zincir, tekrar_oynatma_verisi
    z = Zincir(KOK)
    z.veri(tekrar_oynatma_verisi(KOK, z.meta_secilen))
    sonuc = z.calistir("2025-07-28")
"""
import os
import re
import sys
import json

import numpy as np
import pandas as pd
import xgboost as xgb

TAKVIM_SUTUNLARI = ["saat", "gun", "ay", "hafta_sonu", "tatil_saat", "tatil_tam", "tatil_yarim", "idari_izin",
                    "bayram_gun_no", "bayram_once_gun", "bayram_sonra_gun"]
HAM_SUTUNLAR = ["date", "consumption", "tavg_c", "lep_ham", "om_fc_ham", "om_era5", "eak_toplam", "eak_gunes"] + TAKVIM_SUTUNLARI
TZ = "+03:00"


# ----------------------------------------------------------------------------------------------------
# Takvim ve ham veri
# ----------------------------------------------------------------------------------------------------
def saatlik_aralik(bas_gun, bit_gun):
    """bas_gun 00:00 ile bit_gun 23:00 arasindaki saatlik, +03:00 saat dilimli zaman ekseni."""
    return pd.date_range(f"{pd.Timestamp(bas_gun).date()}T00:00:00{TZ}", f"{pd.Timestamp(bit_gun).date()}T23:00:00{TZ}", freq="h")


def takvim_ozellikleri(tarihler, kok):
    """2_ozellik.ipynb'nin urettigi tatil takviminden (data/tr_tatil_takvimi.csv) saatlik takvim ozellikleri.
    tatil_saat tanimi 2_ozellik ile ayni: tam gun tatil, ya da yarim gun tatilde 13:00 ve sonrasi."""
    tk = pd.read_csv(os.path.join(kok, "data", "tr_tatil_takvimi.csv"), parse_dates=["yerel_gun"])
    t = pd.DataFrame({"date": pd.DatetimeIndex(tarihler)})
    t["yerel_gun"] = t["date"].dt.tz_localize(None).dt.normalize()
    son = tk["yerel_gun"].max()
    if t["yerel_gun"].max() > son:
        raise ValueError(f"Tatil takvimi {son.date()} tarihinde bitiyor; takvimi (2_ozellik.ipynb, B1) yeni yil icin guncelleyin.")
    t = t.merge(tk[["yerel_gun", "tatil_tam", "tatil_yarim", "idari_izin", "bayram_gun_no",
                    "bayram_once_gun", "bayram_sonra_gun"]], on="yerel_gun", how="left")
    t["saat"] = t["date"].dt.hour
    t["gun"] = t["date"].dt.dayofweek
    t["ay"] = t["date"].dt.month
    t["hafta_sonu"] = (t["gun"] >= 5).astype(int)
    t["tatil_saat"] = ((t["tatil_tam"] == 1) | ((t["tatil_yarim"] == 1) & (t["saat"] >= 13))).astype(int)
    return t.drop(columns=["yerel_gun"])


def _om_dosya(kok, dosya):
    s = pd.read_csv(os.path.join(kok, "data", dosya))
    return pd.Series(s["deger"].values, index=pd.to_datetime(s["date"]))


def tekrar_oynatma_verisi(kok, meta_secilen):
    """Gecmis gunler icin ham tablo: 6_agent_tr.ipynb (A3) ile ayni kaynaklar."""
    df = pd.read_csv(os.path.join(kok, "data", "tr_ozellikli_saatlik.csv"))
    df["date"] = pd.to_datetime(df["date"])
    lp = pd.read_csv(os.path.join(kok, "data", "tr_yuk_plani_saatlik.csv"))
    lp["date"] = pd.to_datetime(lp["date"])
    df = df.merge(lp[["date", "lep"]].rename(columns={"lep": "lep_ham"}), on="date", how="left")
    agirlik = meta_secilen.get("sehir_agirliklari")
    idx = pd.DatetimeIndex(df["date"])
    if agirlik:
        era5 = pd.DataFrame({s: _om_dosya(kok, f"om_era5_{s}.csv") for s in agirlik})
        tahm = pd.DataFrame({s: _om_dosya(kok, f"om_tahmin_uzun_{s}.csv") for s in agirlik})
        df["om_era5"] = (era5 * pd.Series(agirlik)).sum(axis=1, min_count=len(agirlik)).reindex(idx).values
        df["om_fc_ham"] = (tahm * pd.Series(agirlik)).sum(axis=1, min_count=len(agirlik)).reindex(idx).values
    else:
        df["om_era5"] = np.nan
        df["om_fc_ham"] = np.nan
    eak = pd.read_csv(os.path.join(kok, "data", "tr_eak_saatlik.csv"))
    eak["date"] = pd.to_datetime(eak["date"])
    df = df.merge(eak[["date", "toplam", "gunes"]].rename(columns={"toplam": "eak_toplam", "gunes": "eak_gunes"}),
                  on="date", how="left")
    return df[HAM_SUTUNLAR]


def ozellikleri_hazirla(ham):
    """Ham tablodan modelin ve zincirin kullandigi ozellikler. Tablo kesintisiz saatlik hale getirilir
    (kaydirma tabanli ozellikler icin sart)."""
    d = ham.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    tam = pd.date_range(d["date"].min(), d["date"].max(), freq="h")
    if len(tam) != len(d):
        d = d.set_index("date").reindex(tam).rename_axis("date").reset_index()
    d["gun_t"] = d["date"].dt.tz_localize(None).dt.normalize()

    # Gecmis ozellikler icin plan temizligi (3c ile ayni kurallar). Tuketimi bilinmeyen saatlerde oran kurali
    # uygulanamaz (NaN karsilastirmalari False doner); D gununun plani ayrica gun_oncesi_tahmin'de kontrol edilir.
    oran = d["consumption"] / d["lep_ham"]
    lep = d["lep_ham"].where(~(d["lep_ham"].notna() & ((d["lep_ham"] <= 0) | (oran > 2) | (oran < 0.5))))
    gs = lep.groupby(d["gun_t"]).agg(["std", "mean", "count"])
    duz = gs.index[(gs["count"] == 24) & (gs["std"] < 0.01 * gs["mean"])]
    d["lep"] = lep.where(~d["gun_t"].isin(duz))

    d["lag_48"] = d["consumption"].shift(48)
    d["lag_72"] = d["consumption"].shift(72)
    d["lag_168"] = d["consumption"].shift(168)          # 2_ozellik.ipynb'deki tanimla ayni
    d["rol_24_48"] = d["consumption"].shift(48).rolling(24).mean()
    d["rol_168_48"] = d["consumption"].shift(48).rolling(168).mean()
    d["tavg_dun"] = d["tavg_c"].shift(24)
    d["lep_hata"] = d["lep"] - d["consumption"]
    d["lep_hata_48"] = d["lep_hata"].shift(48)
    d["lep_hata_168"] = d["lep_hata"].shift(168)
    d["lep_hata_ort_7g"] = d["lep_hata"].shift(48).rolling(168, min_periods=120).mean()
    d.attrs["duz_plan_gunleri"] = [str(g.date()) for g in duz]
    return d


# ----------------------------------------------------------------------------------------------------
# Agent: prompt ve sema
# ----------------------------------------------------------------------------------------------------
SISTEM = """Sen Türkiye elektrik sistemi için bir karar destek asistanısın. Görevin, kural tabanlı olarak üretilmiş gün öncesi kararı açıklamak ve mevzuata dayandırmaktır.
Kurallar:
1. <karar_ozeti> içindeki durum, karar ve öncelik kesindir; değiştirme. Çıktındaki bu alanlar aynen bunlar olmalı.
2. Mevzuat atıflarını YALNIZCA <mevzuat_baglami> içindeki parçalardan yap. Her atıfta parçanın köşeli parantez içindeki başlığındaki belge adını ve madde etiketini aynen yaz. Bağlamda olmayan bir madde numarası yazma. Uygun parça yoksa atıf listesini boş bırak.
3. Eşikler tarihsel nadirlik eşikleridir (Mart 2025 sonrası marj dağılımının %1 ve %5'lik dilimleri); fiziksel bir güvenilirlik sınırı değildir. Bunu gerekçede belirt.
4. Veri uyarılarını veri_uyarilari alanına aynen aktar.
5. Kısa, teknik ve Türkçe yaz.
6. Bütün değerler gün öncesi TAHMİNDİR. "Gerçekleşti", "gerçekleşmiştir" gibi ifadeler kullanma; "öngörülüyor", "tahmin ediliyor" de.
7. Her sayıyı kendi saatiyle kullan. En dar saatteki tahmin ile günün zirve tahmini farklı saatlere ait olabilir; bunları aynı cümlede aynı saate bağlama.
8. Tahmin, Sistem İşletmecisinin yayımladığı gün öncesi talep tahminine (resmî plan) dayanır ve bu planı düzeltir. Bu verinin yayımı ve zamanlaması için bağlamdaki Dengeleme ve Uzlaştırma Yönetmeliği MADDE 67/A'ya atıf yapabilirsin."""

SEMA = {"type": "object", "properties": {
    "durum": {"type": "string", "enum": ["normal", "attention", "action_required"]},
    "karar": {"type": "string", "enum": ["normal_operation", "demand_response_alert", "activate_reserve"]},
    "oncelik": {"type": "string", "enum": ["yok", "normal", "yuksek"]},
    "ozet": {"type": "string"},
    "gerekceler": {"type": "array", "items": {"type": "string"}},
    "mevzuat_atiflari": {"type": "array", "items": {"type": "object", "properties": {
        "belge": {"type": "string"}, "madde": {"type": "string"}, "neden": {"type": "string"}},
        "required": ["belge", "madde", "neden"]}},
    "veri_uyarilari": {"type": "array", "items": {"type": "string"}},
    "belirsizlik_notu": {"type": "string"}},
    "required": ["durum", "karar", "oncelik", "ozet", "gerekceler", "mevzuat_atiflari", "veri_uyarilari", "belirsizlik_notu"]}


def _n(s):
    return " ".join(str(s).lower().split())


# ----------------------------------------------------------------------------------------------------
# Kisisel bilgi korumasi (katmanli)
#   1. Kaynakta: tr_veri_toplayici.epias_hata_ozeti yalnizca hata kodu ve aciklamasini tasir.
#   2. Zincirde: Agent'a giden her metin, kaydedilen/gonderilen her sonuc kisisel_bilgi_temizle'den gecer.
#   3. LangSmith'te: Agent cagrisi, e-posta ve IP'yi maskeleyen bir anonimlestiriciyle izlenir.
# ----------------------------------------------------------------------------------------------------
EPOSTA_DESENI = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+"
IPV4_DESENI = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
_KISISEL = [(re.compile(EPOSTA_DESENI), "[e-posta gizlendi]"), (re.compile(IPV4_DESENI), "[IP gizlendi]")]


def kisisel_bilgi_temizle(o):
    """Metinlerdeki e-posta ve IPv4 adreslerini gizler; sozluk ve listelerde ozyinelemeli calisir."""
    if isinstance(o, str):
        for desen, yerine in _KISISEL:
            o = desen.sub(yerine, o)
        return o
    if isinstance(o, list):
        return [kisisel_bilgi_temizle(x) for x in o]
    if isinstance(o, dict):
        return {k: kisisel_bilgi_temizle(v) for k, v in o.items()}
    return o


def _izle(ad):
    """LangSmith izleme dekoratoru (e-posta ve IP maskelenerek).
    Ice aktarma aninda hicbir ag baglantisi kurulmaz: LangSmith istemcisi yalnizca LANGSMITH_TRACING=true ise ve
    fonksiyon ilk kez cagrildiginda olusturulur. Izleme kapaliysa fonksiyon dogrudan calisir."""
    def sarmala(f):
        izli = {"f": None}

        def ic(*a, **k):
            if os.getenv("LANGSMITH_TRACING", "").strip().lower() != "true":
                return f(*a, **k)
            if izli["f"] is None:
                try:
                    from langsmith import Client, traceable
                    from langsmith.anonymizer import create_anonymizer
                    istemci = Client(anonymizer=create_anonymizer([
                        {"pattern": EPOSTA_DESENI, "replace": "[e-posta gizlendi]"},
                        {"pattern": IPV4_DESENI, "replace": "[IP gizlendi]"}]))
                    izli["f"] = traceable(name=ad, client=istemci)(f)
                except Exception:
                    izli["f"] = f
            return izli["f"](*a, **k)
        return ic
    return sarmala


@_izle("tr_agent_karar_aciklamasi")
def _gemini_cagri(gemini, model, icerik):
    from google.genai import types
    r = gemini.models.generate_content(
        model=model, contents=icerik,
        config=types.GenerateContentConfig(
            system_instruction=SISTEM, response_mime_type="application/json", response_json_schema=SEMA,
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)))
    return json.loads(r.text)


# ----------------------------------------------------------------------------------------------------
# Zincir
# ----------------------------------------------------------------------------------------------------
class Zincir:
    """Gun oncesi karar zinciri. Modeller, esikler ve belirsizlik bandi kok klasorunden yuklenir."""

    def __init__(self, kok, gemini_model="gemini-3.7-flash", agent_cagir=True, model_klasoru=None):
        """model_klasoru: model dosyalarinin klasoru (varsayilan kok/agent). Golge (eski model) calistirma icin
        arsiv klasoru verilebilir, ornegin kok/agent/arsiv/<zaman>."""
        self.kok = kok
        self.model_klasoru = model_klasoru or os.path.join(kok, "agent")
        self.gemini_model = gemini_model
        self.agent_cagir = agent_cagir
        self.modeller = {k: self._yukle(f) for k, f in [("secilen", "tr_xgb_gunoncesi_secilen"),
                                                          ("havasiz", "tr_xgb_gunoncesi_yedek_havasiz"),
                                                          ("plansiz", "tr_xgb_gunoncesi_yedek_plansiz")]}
        self.meta_secilen = self.modeller["secilen"]["meta"]
        self.surum = self.meta_secilen.get("surum", "3c (ilk secim)")
        self.om_kullan = "om_t" in self.meta_secilen["ozellikler"]
        with open(os.path.join(kok, "metrics", "agent_esikleri.json"), encoding="utf-8") as f:
            self.esik = json.load(f)
        with open(os.path.join(kok, "metrics", "tr_gunoncesi_q90.json"), encoding="utf-8") as f:
            q = json.load(f)
        self.q90 = np.array([q["q90_saatlik"][str(h)] for h in range(24)])
        self.df = None
        self._trr = None
        self._gemini = None

    def _yukle(self, dosya):
        m = xgb.XGBRegressor()
        m.load_model(os.path.join(self.model_klasoru, dosya + ".json"))
        with open(os.path.join(self.model_klasoru, dosya + "_meta.json"), encoding="utf-8") as f:
            return {"model": m, "meta": json.load(f)}

    @property
    def trr(self):
        if self._trr is None:
            yol = os.path.join(self.kok, "notepads")
            if yol not in sys.path:
                sys.path.insert(0, yol)
            import tr_rag_retrieval
            self._trr = tr_rag_retrieval
        return self._trr

    @property
    def gemini(self):
        if self._gemini is None:
            from google import genai
            self._gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        return self._gemini

    def veri(self, ham):
        """Ham tabloyu (HAM_SUTUNLAR) yukler ve ozellikleri turetir."""
        eksik = [c for c in HAM_SUTUNLAR if c not in ham.columns]
        if eksik:
            raise ValueError(f"Ham tabloda eksik sutunlar: {eksik}")
        self.df = ozellikleri_hazirla(ham)
        return self

    # --- B: gun oncesi tahmin ---
    def _gun(self, gun):
        D = pd.Timestamp(gun)
        return D.tz_localize(None).normalize() if D.tzinfo else D.normalize()

    def sapma_saatlik(self, gun):
        """D'den 8 gun once biten 30 gunluk pencerede saatlik (GFS - ERA5) ortalamasi; yetersizse meta tablo."""
        df, g = self.df, self._gun(gun)
        bitis, bas = g - pd.Timedelta(days=8), g - pd.Timedelta(days=38)
        m = (df["gun_t"] >= bas) & (df["gun_t"] < bitis) & df["om_fc_ham"].notna() & df["om_era5"].notna()
        fk = (df.loc[m, "om_fc_ham"] - df.loc[m, "om_era5"]).groupby(df.loc[m, "date"].dt.hour)
        ort, say = fk.mean(), fk.size()
        tablo = self.meta_secilen.get("sapma_tablosu_ay_saat_c", {})
        sap, kaynak = np.zeros(24), []
        for h in range(24):
            if say.get(h, 0) >= 10:
                sap[h] = ort[h]; kaynak.append("30 gun")
            else:
                sap[h] = tablo.get(f"{g.month}-{h}", 0.0); kaynak.append("meta tablo")
        return sap, max(set(kaynak), key=kaynak.count)

    def gun_oncesi_tahmin(self, gun):
        D = self._gun(gun)
        g = self.df[self.df["gun_t"] == D].copy()
        if len(g) != 24:
            raise ValueError(f"{D.date()} icin 24 saatlik veri yok ({len(g)})")
        uyarilar = []
        lp_d = g["lep_ham"]
        plan_ok = bool(lp_d.notna().all() and (lp_d > 0).all() and lp_d.std() >= 0.01 * lp_d.mean())
        if not plan_ok:
            uyarilar.append("Resmi plan eksik veya supheli (eksik, sifir veya duz); plansiz yedek model kullanildi")
        om_ok = self.om_kullan and bool(g["om_fc_ham"].notna().all())
        if self.om_kullan and not om_ok:
            uyarilar.append("Gun oncesi hava tahmini eksik; hava tahminsiz yedek model kullanildi")
        anahtar = "secilen" if plan_ok and (om_ok or not self.om_kullan) else ("havasiz" if plan_ok else "plansiz")
        meta, model = self.modeller[anahtar]["meta"], self.modeller[anahtar]["model"]
        if anahtar != "secilen":
            uyarilar.append("Belirsizlik bandi secilen modelden alindi; yedek model icin yaklasik")
        bh, bc = meta["ankara_bazlari_c"]
        g["hdd_dun"] = np.maximum(0, bh - g["tavg_dun"])
        g["cdd_dun"] = np.maximum(0, g["tavg_dun"] - bc)
        sapma_kaynak = None
        if "om_t" in meta["ozellikler"]:
            sap, sapma_kaynak = self.sapma_saatlik(D)
            g["om_t"] = g["om_fc_ham"].values - sap[g["saat"].values.astype(int)]
            g["om_tgun"] = g["om_t"].mean()
            obh, obc = meta["om_bazlari_c"]
            g["om_hdd"] = np.maximum(0, obh - g["om_tgun"])
            g["om_cdd"] = np.maximum(0, g["om_tgun"] - obc)
        g["lep"] = g["lep_ham"]
        X = g[meta["ozellikler"]].astype(float)
        if X.isna().any().any():
            uyarilar.append(f"Model girdisinde {int(X.isna().sum().sum())} eksik deger (XGBoost eksik degeri kendi isler)")
        p = model.predict(X)
        tahmin = g["lep"].values * (1 + p) if meta["hedef"] == "oran" else p
        temkinli = tahmin * (1 + self.q90[g["saat"].values.astype(int)])
        ozel = []
        if (g["bayram_once_gun"] == 0).all() and (g["bayram_sonra_gun"] == 0).all():
            ozel.append("bayram veya arife")
        if (g["tatil_tam"] == 1).any() and not ozel:
            ozel.append("resmi tatil")
        if (g["idari_izin"] == 1).any():
            ozel.append("idari izin")
        if ozel:
            uyarilar.append(f"Ozel gun ({', '.join(ozel)}): test yilinda resmi plan ve hibrit model bu gunlerde daha yuksek hata yapti")
        return {"g": g, "tahmin": np.asarray(tahmin, float), "temkinli": np.asarray(temkinli, float),
                "model_anahtar": anahtar, "model": meta["model"], "sapma_kaynak": sapma_kaynak, "uyarilar": uyarilar}

    # --- C: EAK, marj ve karar ---
    def eak_profili(self, gun, carpan=1.0):
        df, D = self.df, self._gun(gun)
        g = df[df["gun_t"] == D]
        uyarilar = []
        if len(g) != 24 or g["eak_toplam"].isna().any():
            raise ValueError(f"{D.date()} icin EAK eksik")
        t = g["eak_toplam"].reset_index(drop=True) * carpan
        once = df[(df["gun_t"] < D) & (df["gun_t"] >= D - pd.Timedelta(days=365))]["eak_toplam"]
        esik = 4 * once.diff().abs().std()
        fark = t.diff()
        sicrama = ((fark > esik) & (fark.shift(-1) < -esik)) | ((fark < -esik) & (fark.shift(-1) > esik))
        t_duz = t.mask(sicrama).interpolate(limit_direction="both")
        if sicrama.any():
            uyarilar.append(f"EAK profilinde {int(sicrama.sum())} saatlik sicrama duzeltildi (saat: {[int(x) for x in np.where(sicrama)[0]]})")
        oglen = g.loc[g["saat"] == 12, "eak_gunes"]
        o12 = df[df["saat"] == 12]
        ref_kisa = o12[(o12["gun_t"] < D) & (o12["gun_t"] >= D - pd.Timedelta(days=14))]["eak_gunes"].median()
        ref_uzun = o12[(o12["gun_t"] < D - pd.Timedelta(days=14)) & (o12["gun_t"] >= D - pd.Timedelta(days=60))]["eak_gunes"].median()
        ref = np.nanmax([ref_kisa, ref_uzun]) if pd.notna(ref_kisa) or pd.notna(ref_uzun) else np.nan
        if len(oglen) and pd.notna(ref) and ref > 1000 and pd.notna(oglen.iloc[0]) and oglen.iloc[0] < 0.5 * ref:
            uyarilar.append(f"EAK gunes bileseninde ani dusus: oglen {oglen.iloc[0]:.0f} MWh; referans {ref:.0f} MWh "
                            f"(son 14 gun medyani {ref_kisa:.0f}, D-60..D-15 medyani {ref_uzun:.0f})")
        if carpan != 1.0:
            uyarilar.append(f"YAPAY SENARYO: EAK x {carpan:.4f} (EAK kaybi %{(1 - carpan) * 100:.1f})")
        return {"eak": t_duz.values.astype(float), "uyarilar": uyarilar}

    def karar_ver(self, tahmin, temkinli, eak_v):
        marj = (eak_v - tahmin) / eak_v
        marj_t = (eak_v - temkinli) / eak_v
        i, it = int(np.argmin(marj)), int(np.argmin(marj_t))
        if marj_t.min() <= 0:
            durum, karar, oncelik = "action_required", "activate_reserve", "yuksek"
        elif marj.min() < self.esik["cok_nadir_alti"]:
            durum, karar, oncelik = "attention", "demand_response_alert", "yuksek"
        elif marj.min() < self.esik["nadir_alti"]:
            durum, karar, oncelik = "attention", "demand_response_alert", "normal"
        else:
            durum, karar, oncelik = "normal", "normal_operation", "yok"
        return {"durum": durum, "karar": karar, "oncelik": oncelik, "en_dar_saat": i,
                "en_dusuk_marj": round(float(marj.min()), 4), "en_dar_saat_temkinli": it,
                "en_dusuk_marj_temkinli": round(float(marj_t.min()), 4), "marj": marj, "marj_t": marj_t}

    # --- D: RAG ---
    # Her kararda baglama sabit olarak eklenen maddeler (belge turu, madde etiketi). MADDE 67/A, modelin ana girdisi olan
    # gun oncesi talep tahmininin (resmi plan) yayimini duzenler; bu yuzden yalnizca veri sorunu olan gunlerde degil,
    # her kararda ilgilidir. Benzerlik aramasiyla degil, indeksten belge ve madde numarasiyla dogrudan alinir.
    SABIT_MADDELER = [("dengeleme", "MADDE 67/A")]

    def _sabit_parcalar(self):
        kol = getattr(self.trr, "_kol", None)
        if kol is None:
            return []
        out = []
        for dosya, madde in self.SABIT_MADDELER:
            try:
                r = kol.get(where={"$and": [{"dosya": dosya}, {"madde": madde}]}, include=["documents", "metadatas"])
            except Exception:
                continue
            for i, m, t in zip(r.get("ids", []), r.get("metadatas", []), r.get("documents", [])):
                out.append({"id": i, "belge": m["belge"], "madde": m["madde"], "benzerlik": "sabit (her kararda)", "metin": t})
        return out

    def mevzuat_getir(self, durum, veri_uyarisi):
        parcalar = self.trr.getir(self.trr.build_rag_query_tr(durum), k=4)
        if veri_uyarisi:
            parcalar += self.trr.getir(self.trr.build_rag_query_tr("veri"), k=2)
        parcalar += self._sabit_parcalar()
        tekil, gorulen = [], set()
        for p in parcalar:
            if p["id"] not in gorulen:
                tekil.append(p); gorulen.add(p["id"])
        return tekil

    # --- E: Agent ve dogrulama ---
    def _agent_cagir(self, ozet, parcalar):
        icerik = (f"<karar_ozeti>\n{json.dumps(ozet, ensure_ascii=False, indent=1)}\n</karar_ozeti>\n"
                  f"<mevzuat_baglami>\n{self.trr.baglam_metni(parcalar)}\n</mevzuat_baglami>")
        return _gemini_cagri(self.gemini, self.gemini_model, kisisel_bilgi_temizle(icerik))

    @staticmethod
    def kodla_aciklama(ozet, parcalar):
        return {"durum": ozet["durum"], "karar": ozet["karar"], "oncelik": ozet["oncelik"],
                "ozet": (f"En dusuk marjin saat {ozet['en_dar_saat']:02d}:00'de %{ozet['en_dusuk_marj'] * 100:.1f} olmasi "
                         f"ongoruluyor (o saatte tahmin {ozet['tahmin_en_dar_saatte_mwh']} MWh, EAK {ozet['eak_en_dar_saatte_mwh']} MWh); "
                         f"karar {ozet['karar']}."),
                "gerekceler": [f"Esikler: cok nadir < %{ozet['esikler']['cok_nadir_alti'] * 100:.1f}, nadir < "
                               f"%{ozet['esikler']['nadir_alti'] * 100:.1f} (tarihsel nadirlik esikleri, fiziksel sinir degil)."],
                "mevzuat_atiflari": [{"belge": p["belge"], "madde": p["madde"],
                                      "neden": "Durum sorgusuyla getirilen ilgili madde (LLM cagrilmadi)"} for p in parcalar[:2]],
                "veri_uyarilari": ozet["uyarilar"], "belirsizlik_notu": "Aciklama kodla uretildi."}

    @staticmethod
    def dogrula(cikti, ozet, parcalar):
        """Kararin kuralla ayni olmasi ve her atifin getirilen parcalarda bulunmasi (etiket ya da metin, kelime sinirli)."""
        notlar = []
        for k in ["durum", "karar", "oncelik"]:
            if cikti.get(k) != ozet[k]:
                notlar.append(f"LLM {k} alanini degistirmeye calisti ({cikti.get(k)} -> {ozet[k]}); kural karari uygulandi")
                cikti[k] = ozet[k]
        temiz = []
        for a in cikti.get("mevzuat_atiflari", []):
            b, m = _n(a.get("belge", "")), _n(a.get("madde", ""))
            desen = re.compile(r"(?<!\w)" + re.escape(m) + r"(?![\w/])")
            if any(b == _n(p["belge"]) and (m == _n(p["madde"]) or desen.search(_n(p["metin"]))) for p in parcalar):
                temiz.append(a)
            else:
                notlar.append(f"Getirilen parcalarda olmayan atif cikarildi: {a.get('belge')} | {a.get('madde')}")
        cikti["mevzuat_atiflari"] = temiz
        return cikti, notlar

    # --- F: zincir ---
    def calistir(self, gun, eak_carpan=1.0, agent=None, kaynak_bilgisi=None):
        """kaynak_bilgisi: tr_veri_toplayici.canli_veri'nin bilgi sozlugu (canli modda). Toplayicinin veri uyarilari
        zincirin uyarilarina eklenir; Ankara yaklasik sicaklik notu yalnizca yedek model kullanildiginda eklenir."""
        if self.df is None:
            raise RuntimeError("Once veri(ham) ile veri yukleyin.")
        agent = self.agent_cagir if agent is None else agent
        D = self._gun(gun)
        t = self.gun_oncesi_tahmin(D)
        if eak_carpan == "acik":        # en kritik saatte EAK'yi temkinli tahminin %1 altina indiren carpan
            e0 = self.eak_profili(D, 1.0)
            eak_carpan = 0.99 * float(np.max(t["temkinli"] / e0["eak"]))
        e = self.eak_profili(D, eak_carpan)
        k = self.karar_ver(t["tahmin"], t["temkinli"], e["eak"])
        uyarilar = t["uyarilar"] + e["uyarilar"]
        if kaynak_bilgisi:
            uyarilar += [u for u in kaynak_bilgisi.get("uyarilar", []) if u not in uyarilar]
            if t["model_anahtar"] != "secilen" and kaynak_bilgisi.get("yedek_model_notu"):
                uyarilar.append(kaynak_bilgisi["yedek_model_notu"])
        uyarilar = kisisel_bilgi_temizle(uyarilar)
        parcalar = self.mevzuat_getir(k["durum"], bool(uyarilar))
        i, iz = k["en_dar_saat"], int(np.argmax(t["tahmin"]))
        ozet = {"gun": str(D.date()), "durum": k["durum"], "karar": k["karar"], "oncelik": k["oncelik"],
                "en_dar_saat": i, "en_dusuk_marj": k["en_dusuk_marj"],
                "tahmin_en_dar_saatte_mwh": round(float(t["tahmin"][i])),
                "eak_en_dar_saatte_mwh": round(float(e["eak"][i])),
                "en_dar_saat_temkinli": k["en_dar_saat_temkinli"], "en_dusuk_marj_temkinli": k["en_dusuk_marj_temkinli"],
                "temkinli_tahmin_en_dar_saatte_mwh": round(float(t["temkinli"][k["en_dar_saat_temkinli"]])),
                "tahmin_zirve_mwh": round(float(t["tahmin"][iz])), "tahmin_zirve_saati": iz,
                "esikler": {"cok_nadir_alti": self.esik["cok_nadir_alti"], "nadir_alti": self.esik["nadir_alti"]},
                "model": t["model"], "sapma_kaynagi": t["sapma_kaynak"], "uyarilar": uyarilar}
        llm_notu = ""
        if agent:
            try:
                cikti = self._agent_cagir(ozet, parcalar)
            except Exception as ex:
                llm_notu = f"Gemini cagrisi basarisiz ({str(ex)[:100]}); aciklama kodla uretildi"
                cikti = self.kodla_aciklama(ozet, parcalar)
        else:
            cikti = self.kodla_aciklama(ozet, parcalar)
        cikti, notlar = self.dogrula(kisisel_bilgi_temizle(cikti), ozet, parcalar)
        if llm_notu:
            notlar.append(llm_notu)
        g = t["g"]
        ger = g["consumption"].values.astype(float)
        kiyas = None
        if np.isfinite(ger).all():       # gecmis gun: gerceklesenle karsilastirma; canli gunde tuketim henuz yok
            marj_ger = (e["eak"] - ger) / e["eak"]
            kiyas = {"gun_mape_%": round(float(np.mean(np.abs(t["tahmin"] - ger) / ger) * 100), 2),
                     "resmi_plan_mape_%": round(float(np.nanmean(np.abs(g["lep_ham"].values - ger) / ger) * 100), 2)
                     if g["lep_ham"].notna().any() else None,
                     "gercek_en_dar_saat": int(np.argmin(marj_ger)), "gercek_en_dusuk_marj": round(float(marj_ger.min()), 4)}
        saatlik = pd.DataFrame({"saat": g["saat"].values.astype(int), "tahmin_mwh": np.round(t["tahmin"]),
                                "temkinli_mwh": np.round(t["temkinli"]), "eak_mwh": np.round(e["eak"]),
                                "marj": np.round(k["marj"], 4), "marj_temkinli": np.round(k["marj_t"], 4),
                                "resmi_plan_mwh": np.round(g["lep_ham"].values)})
        return {"ozet": ozet, "cikti": cikti, "notlar": notlar, "parcalar": parcalar, "kiyas": kiyas,
                "saatlik": saatlik, "model_anahtar": t["model_anahtar"], "t": t, "k": k,
                "veri_kaynagi": (kaynak_bilgisi or {}).get("kaynak", "tekrar oynatma (CSV)")}


# ----------------------------------------------------------------------------------------------------
# Disari aktarma
# ----------------------------------------------------------------------------------------------------
def sonuc_json(r):
    """API ve kayit icin JSON'a donusturulebilir sonuc (dizi ve tablo nesneleri olmadan)."""
    return kisisel_bilgi_temizle({"ozet": r["ozet"], "cikti": r["cikti"], "notlar": r["notlar"], "kiyas": r["kiyas"],
            "model_anahtar": r["model_anahtar"], "veri_kaynagi": r.get("veri_kaynagi"),
            "saatlik": r["saatlik"].to_dict(orient="records"),
            "getirilen": [f"{p['belge']} | {p['madde']}" for p in r["parcalar"]]})


def n8n_yuku(r):
    o, c = r["ozet"], r["cikti"]
    return kisisel_bilgi_temizle({"region": "TR", "gun": o["gun"], "status": c["durum"], "decision": c["karar"], "priority": c["oncelik"],
            "tightest_hour": o["en_dar_saat"], "min_margin": o["en_dusuk_marj"],
            "min_margin_conservative": o["en_dusuk_marj_temkinli"], "summary": c["ozet"], "reasons": c["gerekceler"],
            "citations": c["mevzuat_atiflari"], "data_warnings": c["veri_uyarilari"], "model": o["model"]})
