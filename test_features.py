import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_query(query, description):
    print(f"\n{'='*60}")
    print(f"TEST: {description}")
    print(f"Query: {query}")
    print(f"{'='*60}")
    
    response = requests.post(
        f"{BASE_URL}/query",
        json={"query": query, "top_k": 5}
    )
    
    result = response.json()
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(result, indent=2)}")
    return result

# Test 1: Similarity Threshold - Irrelevant query
print("\n" + "="*60)
print("TEST SUITE: Bonus Features & Edge Cases")
print("="*60)

test_query(
    "What is the weather like in Tokyo?",
    "Similarity Threshold - Irrelevant Query (should refuse)"
)

# Test 2: Intent Detection - Greeting
test_query(
    "hello",
    "Intent Detection - Greeting (should refuse)"
)

# Test 3: PII Detection - SSN
test_query(
    "My SSN is 123-45-6789",
    "PII Detection - SSN (should refuse)"
)

# Test 4: PII Detection - Email
test_query(
    "Contact me at john.doe@example.com",
    "PII Detection - Email (should refuse)"
)

# Test 5: PII Detection - Phone
test_query(
    "Call me at 555-123-4567",
    "PII Detection - Phone Number (should refuse)"
)

# Test 6: Legal Disclaimer
test_query(
    "Can I sue my landlord for this?",
    "Legal Disclaimer - Lawsuit Query (should refuse)"
)

# Test 7: Medical Disclaimer
test_query(
    "What medication should I take for this?",
    "Medical Disclaimer - Medication Query (should refuse)"
)

# Test 8: Valid Query - Should work
test_query(
    "What are the seller's obligations?",
    "Valid Query - Should Return Answer"
)

# Test 9: Demo Endpoint
print(f"\n{'='*60}")
print("TEST: Demo Endpoint")
print(f"{'='*60}")
response = requests.get(f"{BASE_URL}/demo")
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

print("\n" + "="*60)
print("TEST SUITE COMPLETED")
print("="*60)
