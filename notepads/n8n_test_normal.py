import requests

payload = {
    "status": "normal",
    "decision": "normal_operation",
    "justification": "Test: comfortable reserve margin",
    "uncertainty": "low",
    "reasoning_steps": "Test reasoning for normal operation"
}

response = requests.post("http://localhost:5678/webhook/grid-forecast", json=payload)
print(response.status_code, response.text)