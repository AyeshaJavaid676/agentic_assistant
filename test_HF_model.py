import requests

url = "https://router.huggingface.co/models/distilgpt2"
token = HF_TOKEN

prompt = """
Why lion is king of jungle?
"""

headers = {
    "Authorization": f"Bearer {token}"
}

def query(payload):
    response = requests.post(url, headers=headers, json=payload)
    return response.json()

output = query({
    "inputs": prompt
})

print(output)
