import requests

payload = {
    "status": "attention",
    "decision": "demand_response_alert",
    "justification": "Test: thin reserve margin",
    "uncertainty": "medium",
    "reasoning_steps": "Test reasoning for demand response alert"
}

response = requests.post("http://localhost:5678/webhook/grid-forecast", json=payload)
print(response.status_code, response.text)