# check_vision_models.py
from groq import Groq
from config.settings import GROQ_API_KEY
import base64
from PIL import Image
import io

# List of potential vision models to try
vision_models = [
    "meta-llama/llama-4-scout-17b-16e",
    "meta-llama/llama-4-maverick-17b-12e",
    "llama-3.2-11b-vision-preview",  # Deprecated but check
    "llama-3.2-90b-vision-preview",   # Deprecated but check
]

def create_test_image():
    """Create a simple test image"""
    img = Image.new('RGB', (100, 100), color='blue')
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

client = Groq(api_key=GROQ_API_KEY)
test_image = create_test_image()

print("="*60)
print("Testing Vision Models")
print("="*60)

for model in vision_models:
    print(f"\n🔄 Testing: {model}")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{test_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "What color is this image? Answer in one word."
                        }
                    ]
                }
            ],
            max_tokens=10
        )
        print(f"✅ SUCCESS: {response.choices[0].message.content}")
    except Exception as e:
        print(f"❌ FAILED: {e}")