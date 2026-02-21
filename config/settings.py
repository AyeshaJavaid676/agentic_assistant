import os
from dotenv import load_dotenv
from pathlib import Path

# Print current directory for debugging
print(f"📁 Current working directory: {os.getcwd()}")

# Try multiple ways to load .env
env_path = Path('.') / '.env'
print(f"🔍 Looking for .env at: {env_path.absolute()}")

# Method 1: Load with explicit path
loaded = load_dotenv(dotenv_path=env_path, verbose=True)
print(f"📦 .env loaded with explicit path: {loaded}")

# Method 2: If that fails, try loading without path
if not loaded:
    print("🔄 Trying to load .env without path...")
    loaded = load_dotenv(verbose=True)
    print(f"📦 .env loaded without path: {loaded}")

# Method 3: Direct from file (fallback)
if not loaded:
    print("⚠️  python-dotenv failed, reading file directly...")
    try:
        with open('.env', 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
                    print(f"✅ Manually set {key}")
        loaded = True
    except Exception as e:
        print(f"❌ Manual reading failed: {e}")

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Debug: Check if key is loaded
if GROQ_API_KEY:
    print(f"✅ GROQ_API_KEY loaded successfully! (starts with: {GROQ_API_KEY[:8]}...)")
else:
    print("❌ GROQ_API_KEY NOT found in environment!")
    print("🔑 Available environment variables:", [k for k in os.environ.keys() if 'KEY' in k or 'API' in k])

# Paths
PDF_UPLOAD_FOLDER = "data/pdfs"
VECTOR_STORE_PATH = "data/vectorstore"

# Models
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"  # For text
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e"  # For images

# Create directories if they don't exist
os.makedirs(PDF_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

print("✅ Config module finished loading")