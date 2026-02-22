import requests
import base64
from PIL import Image
import io

HF_TOKEN = HF_TOKEN

# Create test image
img = Image.new('RGB', (100, 100), color='blue')
buffered = io.BytesIO()
img.save(buffered, format="PNG")
img_bytes = buffered.getvalue()

# Try different endpoint formats
endpoints = [
    f"https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large",
    f"https://router.huggingface.co/hf-inference/models/Salesforce/blip-image-captioning-large",
    f"https://router.huggingface.co/models/Salesforce/blip-image-captioning-large",
    f"https://hf-inference.com/models/Salesforce/blip-image-captioning-large"
]

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

for i, endpoint in enumerate(endpoints, 1):
    print(f"\n{i}. Testing: {endpoint}")
    try:
        response = requests.post(
            endpoint,
            headers=headers,
            data=img_bytes,
            timeout=10
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   ✅ Success: {response.json()}")
            break
        else:
            print(f"   ❌ Error: {response.text[:100]}")
    except Exception as e:
        print(f"   ❌ Exception: {e}")
