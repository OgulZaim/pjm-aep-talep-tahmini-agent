import requests

payload = {
    "status": "action_required",
    "decision": "activate_reserve",
    "justification": "Test: forecast exceeds available capacity",
    "uncertainty": "low",
    "reasoning_steps": "Test reasoning for reserve activation"
}

response = requests.post("http://localhost:5678/webhook/grid-forecast", json=payload)
print(response.status_code, response.text)

