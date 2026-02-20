# test_tts_simple.py
from gtts import gTTS
import os
import tempfile

print("🔊 Testing Text-to-Speech...")

# Text to convert to speech
text = "Hello, this is a test of the text to speech system. If you can hear this, it's working!"

try:
    # Create TTS object
    print("📝 Creating TTS...")
    tts = gTTS(text=text, lang='en', slow=False)
    
    # Save to a temporary file
    print("💾 Saving to file...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        temp_filename = tmp.name
        tts.save(temp_filename)
    
    print(f"✅ Audio saved to: {temp_filename}")
    print("🎵 Playing audio...")
    
    # Play the file (works on Windows)
    os.system(f"start {temp_filename}")
    
    print("✅ If you heard audio, TTS is working!")
    print("⚠️  If not, check your speakers/volume")
    
    # Keep the file for a moment
    import time
    time.sleep(5)
    
    # Clean up
    os.unlink(temp_filename)
    print("🧹 Cleanup complete")
    
except Exception as e:
    print(f"❌ Error: {e}")