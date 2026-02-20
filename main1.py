import streamlit as st
import tempfile
import os
from pathlib import Path
from modules.pdf_extractor import PDFExtractor
from modules.vision_processor import VisionProcessor
from modules.vector_store import VectorStore
from modules.agent_tools import AgentAssistant
from modules.tts_service import TTSService
from utils.helpers import cleanup_temp_files

# Page config
st.set_page_config(
    page_title="Agentic PDF Assistant",
    page_icon="📚",
    layout="wide"
)

# Title
st.title("📚 Agentic PDF Assistant")
st.markdown("Upload a PDF and ask questions about it. I can read text, describe images, and search the web!")

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'temp_file' not in st.session_state:
    st.session_state.temp_file = None
if 'last_response' not in st.session_state:
    st.session_state.last_response = None

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Options
    process_images = st.checkbox("🖼️ Process images (first 5 pages)", value=True)
    enable_web = st.checkbox("🌐 Enable web search", value=True)
    
    st.divider()
    
    # File upload
    uploaded_file = st.file_uploader(
        "📄 Upload PDF", 
        type="pdf",
        help="Upload any PDF document"
    )
    
    if uploaded_file:
        st.success(f"✅ Uploaded: {uploaded_file.name}")
        
        # Process button
        if st.button("🚀 Process PDF", type="primary"):
            with st.spinner("Processing PDF... This may take a minute..."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                        st.session_state.temp_file = tmp_path
                    
                    # Extract text and images
                    st.info("📄 Extracting text...")
                    extractor = PDFExtractor(tmp_path)
                    texts = extractor.extract_text()
                    
                    # Prepare documents
                    documents = []
                    
                    # Add text documents
                    for text in texts:
                        documents.append({
                            "content": text["content"],
                            "metadata": {
                                "type": "text", 
                                "page": text["page"],
                                "source": "pdf_text"
                            }
                        })
                    
                    # Process images if enabled
                    if process_images:
                        st.info("🖼️ Processing images with Groq Vision...")
                        images = extractor.extract_images(max_pages=5)
                        if images:
                            vision = VisionProcessor()
                            
                            progress_bar = st.progress(0)
                            for i, img in enumerate(images):
                                description = vision.describe_image(img["base64"])
                                documents.append({
                                    "content": f"[Image on page {img['page']}]: {description}",
                                    "metadata": {
                                        "type": "image", 
                                        "page": img["page"],
                                        "source": "pdf_image"
                                    }
                                })
                                progress_bar.progress((i + 1) / len(images))
                            
                            progress_bar.empty()
                        else:
                            st.info("No images found in the first 5 pages.")
                    
                    # Create vector store
                    st.info("📚 Creating searchable database...")
                    vectorstore = VectorStore()
                    vectorstore.create_from_documents(documents)
                    
                    # Create agent
                    st.info("🤖 Setting up AI agent...")
                    st.session_state.agent = AgentAssistant(vectorstore)
                    st.session_state.processed = True
                    
                    # Clean up
                    extractor.close()
                    
                    st.success("✅ PDF processed successfully!")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # Clear button
    if st.session_state.processed:
        if st.button("🔄 Process New PDF"):
            # Clean up temp file
            if st.session_state.temp_file and os.path.exists(st.session_state.temp_file):
                os.unlink(st.session_state.temp_file)
            st.session_state.processed = False
            st.session_state.last_response = None
            st.rerun()

# Main chat area
if st.session_state.processed:
    st.divider()
    
    # Question input
    question = st.text_input(
        "💬 Ask a question about your PDF:",
        placeholder="e.g., What is this document about? Summarize the key points."
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)
    
    if ask_button and question:
        with st.spinner("Thinking..."):
            # Get response from agent
            response = st.session_state.agent.run(question)
            
            # Store in session state for later use
            st.session_state.last_response = response['output']
            
            # Display answer
            st.markdown("### 📝 Answer")
            st.write(response['output'])
            
            # Optional: Show thought process
            with st.expander("🤔 Agent's thought process"):
                st.info("Check the terminal for verbose agent output")
    
    # Text-to-speech option (separate button, outside the ask_button block)
    if st.session_state.last_response:
        if st.button("🔊 Listen to answer"):
            with st.spinner("Generating audio..."):
                try:
                    from gtts import gTTS
                    import base64
                    import io
                    
                    # Generate speech from stored response
                    tts = gTTS(text=st.session_state.last_response, lang='en', slow=False)
                    
                    # Save to bytes buffer
                    fp = io.BytesIO()
                    tts.write_to_fp(fp)
                    fp.seek(0)
                    
                    # Convert to base64 for embedding
                    audio_bytes = fp.read()
                    audio_base64 = base64.b64encode(audio_bytes).decode()
                    
                    # Create HTML audio element with autoplay
                    audio_html = f"""
                        <audio autoplay controls style="width: 100%; margin-top: 10px;">
                            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                            Your browser does not support the audio element.
                        </audio>
                    """
                    
                    # Display the audio player
                    st.markdown(audio_html, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"🔊 Audio Error: {str(e)}")
    
    # Sample questions
    with st.expander("💡 Sample questions to try"):
        st.markdown("""
        - What is the main topic of this document?
        - Summarize the key findings
        - Describe any images or charts in the document
        - What are the important numbers or statistics?
        - [If web search enabled] What are the latest developments in this field?
        """)
else:
    # Welcome message
    st.info("👈 Please upload and process a PDF file from the sidebar to get started!")
    
    # Features
    st.markdown("""
    ### ✨ Features:
    - 📄 **Text Extraction**: Read text from any PDF
    - 🖼️ **Image Understanding**: Describe charts and images using Groq Vision
    - 🌐 **Web Search**: Get real-time information (optional)
    - 🔍 **Smart Search**: Find relevant information quickly
    - 🔊 **Text-to-Speech**: Listen to answers (optional)
    """)

# Footer
st.divider()
st.caption("Made with LangChain, Groq API, and Streamlit")