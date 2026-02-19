# test_setup.py
import sys
import langchain
import groq
import streamlit
import fitz
from sentence_transformers import SentenceTransformer
import dotenv
import duckduckgo_search
from gtts import gTTS

print("="*50)
print("✅ Environment Test Results")
print("="*50)
print(f"Python version: {sys.version}")
print(f"LangChain: {langchain.__version__}")
print(f"Streamlit: {streamlit.__version__}")
print(f"PyMuPDF: {fitz.__doc__.split()[1]}")
print("="*50)
print("🎉 All packages imported successfully!")