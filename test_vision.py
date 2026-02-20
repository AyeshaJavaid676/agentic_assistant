# test_vision.py
from modules.vision_processor import VisionProcessor
from config.settings import GROQ_API_KEY, VISION_MODEL
import base64

print("="*50)
print("Testing Vision Processor")
print("="*50)

print(f"GROQ_API_KEY exists: {'Yes' if GROQ_API_KEY else 'No'}")
print(f"Vision Model: {VISION_MODEL}")

# Create a simple test image (a small colored square)
from PIL import Image
import io

def create_test_image():
    img = Image.new('RGB', (100, 100), color='red')
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

try:
    # Initialize vision processor
    vision = VisionProcessor()
    
    # Create test image
    print("\n📸 Creating test image...")
    test_image = create_test_image()
    
    # Describe the test image
    print("\n🔄 Sending to vision API...")
    description = vision.describe_image(test_image, "What color is this image?")
    
    print(f"\n✅ Result: {description}")
    
except Exception as e:
    print(f"\n❌ Error: {e}")