from huggingface_hub import InferenceClient
import base64
from PIL import Image
import io

# Your token
HF_TOKEN = HF_TOKEN  # Replace with actual token

# Create test image
img = Image.new('RGB', (100, 100), color='blue')
buffered = io.BytesIO()
img.save(buffered, format="PNG")
img_bytes = buffered.getvalue()

# Use InferenceClient (this handles the correct endpoint automatically)
client = InferenceClient(token=HF_TOKEN)

try:
    print("Testing BLIP model...")
    result = client.image_to_text(
        img_bytes,
        model="Salesforce/blip-image-captioning-large"
    )
    print(f"✅ Success: {result}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    
    # Try a different model
    try:
        print("\nTrying alternative model: nlpconnect/vit-gpt2-image-captioning")
        result = client.image_to_text(
            img_bytes,
            model="nlpconnect/vit-gpt2-image-captioning"
        )
        print(f"✅ Success: {result}")
    except Exception as e2:
        print(f"❌ Error: {e2}")












