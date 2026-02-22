import os
import base64
from huggingface_hub import InferenceClient

# --- CONFIGURATION ---
# 1. Put your token here (Ensure it has 'Inference Provider' permissions)
HF_TOKEN = HF_TOKEN 

# 2. Put your IMAGE PATH here (e.g., "chart.png" or "C:/Users/Images/photo.jpg")
IMAGE_FILE_PATH = r"C:\Users\Optiplex\Downloads\EDA4.png" 

# 3. Model with a specific provider (Novita is fast for Qwen)
MODEL_ID = "Qwen/Qwen2.5-VL-72B-Instruct:novita" 
# ---------------------

def test_vision_model():
    client = InferenceClient(api_key=HF_TOKEN)

    # Encode the local image to base64
    with open(IMAGE_FILE_PATH, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")

    print(f"🚀 Sending image to {MODEL_ID}...")

    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in one sentence."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            max_tokens=100
        )
        print("\n✅ MODEL RESPONSE:")
        print(response.choices[0].message.content)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")

if __name__ == "__main__":
    test_vision_model()
