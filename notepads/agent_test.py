import os
import chromadb
from google import genai
from google.genai import types
from google.genai import errors
from dotenv import load_dotenv
from langsmith import traceable
import json
from prompt_v1 import SYSTEM_PROMPT

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# RAG.ipynb'de zaten olusturulan, diskteki KALICI ChromaDB'ye yeniden baglaniyoruz
# (580 parca zaten islenmis, burada yeniden embed etmeye gerek yok)
chroma_client = chromadb.PersistentClient(path=r"C:\Deep_Learning_to_AI_Agents\Bitirme_Proje\rag_docs\chroma_store")
collection = chroma_client.get_or_create_collection(name="pjm_manual13")

@traceable(name="pjm_manual13_retrieval")
def retrieve_context(query, k=3):
    query_embedding = client.models.embed_content(
        model="gemini-embedding-001", contents=query
    ).embeddings[0].values
    results = collection.query(query_embeddings=[query_embedding], n_results=k)
    return "\n---\n".join(results["documents"][0])


demand_forecast = {
    "region": "AEP",
    "forecast_mw": 15800,
    "confidence_interval": [15200, 16400]
}

grid_status = {
    "available_capacity_mw": 17000,
    "time_period": "peak",
    "capacity_source": "example_test_value"
}

retrieved_context = retrieve_context(f"grid capacity shortfall reserve activation {grid_status['time_period']}")

content = f"""<demand_forecast>
{json.dumps(demand_forecast)}
</demand_forecast>
<grid_status>
{json.dumps(grid_status)}
</grid_status>
<official_procedure_context>
{retrieved_context}
</official_procedure_context>"""


@traceable(name="grid_agent_decision_with_rag")
def get_agent_decision(content, system_prompt):
    response = client.models.generate_content(
        model="gemini-3.7-flash",
        contents=content,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.2,
            response_mime_type="application/json",
        ),
    )
    return response.text


try:
    result_text = get_agent_decision(content, SYSTEM_PROMPT)
    print(result_text)
except errors.ClientError as e:
    if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
        print("STOPPED: Gemini API quota exceeded.")
    else:
        raise