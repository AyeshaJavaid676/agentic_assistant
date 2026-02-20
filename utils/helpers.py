import os
import shutil
from pathlib import Path

def cleanup_temp_files(file_path):
    """Delete temporary files"""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(f"Cleanup error: {e}")

def format_response(text, max_length=500):
    """Format long responses"""
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text