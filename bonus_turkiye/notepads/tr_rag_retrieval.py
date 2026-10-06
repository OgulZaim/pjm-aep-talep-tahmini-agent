# tr_rag_retrieval.py - 5_rag_tr.ipynb tarafindan uretildi. Elle degistirmeyin; notebook'u yeniden calistirin.
import os
import numpy as np
import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langsmith import traceable

KOK = r"C:\Deep_Learning_to_AI_Agents\Bitirme_Proje_Bonus"
load_dotenv(os.path.join(KOK, "notepads", ".env"))
os.environ.setdefault("LANGSMITH_PROJECT", "tr-grid-agent")
EMBED_MODEL, BOYUT = "gemini-embedding-001", 768

_gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
_kol = chromadb.PersistentClient(path=os.path.join(KOK, "rag_docs", "chroma_tr")).get_or_create_collection(name="tr_mevzuat")

def _gom_sorgu(metin):
    r = _gemini.models.embed_content(model=EMBED_MODEL, contents=[metin],
                                     config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=BOYUT))
    v = np.array(r.embeddings[0].values, dtype="float32")
    return v / np.linalg.norm(v)

@traceable(name="tr_mevzuat_retrieval")
def getir(sorgu, k=4, dosya=None):
    ek = {"where": {"dosya": dosya}} if dosya else {}
    r = _kol.query(query_embeddings=[_gom_sorgu(sorgu).tolist()], n_results=k, **ek)
    return [{"id": i, "belge": m["belge"], "madde": m["madde"], "benzerlik": round(1 - d / 2, 4), "metin": t}
            for i, m, d, t in zip(r["ids"][0], r["metadatas"][0], r["distances"][0], r["documents"][0])]

def build_rag_query_tr(durum):
    sorgular = {"normal": "dengeleme güç piyasası yedek kapasite yük alma ve yük atma talimatları", "attention": "yük atma talimatı talep tarafı sistem işletmecisinin dengeleme yükümlülükleri", "action_required": "acil durum sistem kararlılığı ve emniyeti acil durum talimatı düşük frekans rölesi yük atma", "veri": "gerçekleşen tüketim ve gün öncesi yük tahmin planının yayımlanma zamanı"}
    return sorgular.get(durum, sorgular["normal"])

def baglam_metni(parcalar):
    return "\n---\n".join(f"[{p['belge']} | {p['madde']}] (benzerlik {p['benzerlik']})\n{p['metin']}" for p in parcalar)
