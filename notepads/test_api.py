import requests
import pandas as pd

last_24h = pd.read_csv(r"C:\Deep_Learning_to_AI_Agents\Bitirme_Proje\metrics\api_test_last_24h.csv")
sample_readings = last_24h.values.tolist()

response = requests.post("http://127.0.0.1:8000/predict", json={"readings": sample_readings})
print(response.status_code, response.json())