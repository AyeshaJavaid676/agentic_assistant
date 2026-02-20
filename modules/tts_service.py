from gtts import gTTS
import io
import tempfile
import os

class TTSService:
    def __init__(self):
        """Initialize TTS"""
        print("🔊 Text-to-speech ready")
    
    def speak(self, text):
        """Convert text to speech and return audio bytes"""
        try:
            tts = gTTS(text=text, lang='en', slow=False)
            
            # Save to bytes buffer
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            
            return fp
        except Exception as e:
            print(f"TTS Error: {e}")
            return None
    
    def save_to_file(self, text, filename="output.mp3"):
        """Save speech to file"""
        try:
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(filename)
            return filename
        except Exception as e:
            print(f"TTS Save Error: {e}")
            return None