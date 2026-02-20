import os
import base64
from huggingface_hub import InferenceClient

# Use your token
HF_TOKEN = "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" 
IMAGE_PATH = r"C:\Users\Optiplex\Downloads\EDA4.png"

client = InferenceClient(api_key=HF_TOKEN)

# 1. Encode local image
with open(IMAGE_PATH, "rb") as f:
    base64_image = base64.b64encode(f.read()).decode("utf-8")

# 2. Try Qwen2.5-VL (No gating, works immediately)
try:
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-VL-72B-Instruct:novita", # Explicitly calling Novita provider
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this EDA chart in detail."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                    }
                ]
            }
        ]
    )
    print("\n✅ Success!")
    print(response.choices[0].message.content)
except Exception as e:
    print(f"\n❌ Error: {e}")