import os
from dotenv import load_dotenv

print("Current directory:", os.getcwd())
print("\nFiles in directory:")
for file in os.listdir('.'):
    if file.startswith('.env'):
        print(f"  - {file}")

print("\nLoading .env file...")
load_dotenv()

key = os.getenv('GROQ_API_KEY')
if key:
    print(f"✅ GROQ_API_KEY found! (starts with: {key[:8]}...)")
    print(f"   Length: {len(key)} characters")
else:
    print("❌ GROQ_API_KEY not found in environment")
    
    # Try to load with explicit path
    from pathlib import Path
    env_path = Path('.') / '.env'
    print(f"\nTrying explicit path: {env_path.absolute()}")
    load_dotenv(dotenv_path=env_path)
    
    key = os.getenv('GROQ_API_KEY')
    if key:
        print(f"✅ Found with explicit path!")
    else:
        print("❌ Still not found")