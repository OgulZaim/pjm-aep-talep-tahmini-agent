import os
from dotenv import load_dotenv
import chromadb
from google import genai
from langsmith import traceable

load_dotenv(r"C:\Deep_Learning_to_AI_Agents\Bitirme_Proje\notepads\.env")

_chroma_client = chromadb.PersistentClient(path=r"C:\Deep_Learning_to_AI_Agents\Bitirme_Proje\rag_docs\chroma_store")
_collection = _chroma_client.get_or_create_collection(name="pjm_manual13")
_gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

@traceable(name="pjm_manual13_retrieval")
def retrieve_context(query, k=3):
    query_embedding = _gemini_client.models.embed_content(
        model="gemini-embedding-001", contents=query
    ).embeddings[0].values
    results = _collection.query(query_embeddings=[query_embedding], n_results=k)
    return "\n---\n".join(results["documents"][0])

def build_rag_query(forecast_mw, available_capacity_mw, time_period):
    margin_pct = (available_capacity_mw - forecast_mw) / available_capacity_mw
    if margin_pct >= 0.20:
        query = f"PJM reserve requirements control zone normal operating margin {time_period}"
    else:
        query = f"grid capacity shortfall reserve activation {time_period}"
    return query