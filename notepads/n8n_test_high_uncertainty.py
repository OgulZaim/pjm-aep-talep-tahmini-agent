import requests

payload = {
    "status": "normal",
    "decision": "normal_operation",
    "justification": "Test: low confidence despite normal status",
    "uncertainty": "high",
    "reasoning_steps": "Test reasoning for high-uncertainty routing"
}

response = requests.post("http://localhost:5678/webhook/grid-forecast", json=payload)
print(response.status_code, response.text)