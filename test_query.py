import requests
import json

# Test the query endpoint
url = "http://127.0.0.1:8000/query"
data = {
    "query": "What items are included in the property sale by default?",
    "top_k": 5
}

try:
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")
    print(f"Response text: {response.text if 'response' in locals() else 'No response'}")
