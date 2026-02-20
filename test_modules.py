# test_modules.py
from modules.pdf_extractor import PDFExtractor
from modules.vision_processor import VisionProcessor
from config.settings import GROQ_API_KEY

print("="*50)
print("Testing Module Imports")
print("="*50)

# Test PDFExtractor
print("\n📄 Testing PDFExtractor...")
print("✅ PDFExtractor class imported")

# Test VisionProcessor
print("\n👁️ Testing VisionProcessor...")
print("✅ VisionProcessor class imported")

# Test GROQ API key
print("\n🔑 Testing GROQ API Key...")
if GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here":
    print(f"✅ GROQ API Key found: {GROQ_API_KEY[:8]}...")
else:
    print("❌ Please add your GROQ API key to .env file")

print("\n" + "="*50)
print("🎉 All modules imported successfully!")
print("="*50)