import streamlit as st
import tempfile
import os
from pathlib import Path
from modules.pdf_extractor import PDFExtractor
from modules.multimodal_processor import MultimodalProcessor  # Updated to use Qwen
from modules.vector_store import VectorStore
from modules.agent_tools import AgentAssistant
from modules.tts_service import TTSService
from utils.helpers import cleanup_temp_files

# ===== ADD THE FUNCTION HERE =====
def load_existing_pdfs():
    """Load and process all PDFs from data/pdfs folder"""
    pdf_folder = "data/pdfs"
    processed_file = "data/vectorstore/processed_pdfs.txt"
    
    # Create folders if they don't exist
    os.makedirs(pdf_folder, exist_ok=True)
    os.makedirs("data/vectorstore", exist_ok=True)
    
    # Get list of PDFs in folder
    existing_pdfs = list(Path(pdf_folder).glob("*.pdf"))
    
    # Load previously processed PDFs list
    processed_pdfs = []
    if os.path.exists(processed_file):
        with open(processed_file, 'r') as f:
            processed_pdfs = f.read().splitlines()
    
    # Find new PDFs that haven't been processed
    new_pdfs = [pdf for pdf in existing_pdfs if str(pdf) not in processed_pdfs]
    
    if new_pdfs:
        st.info(f"📚 Found {len(new_pdfs)} new PDFs in data/pdfs folder. Processing...")
        
        all_documents = []
        vectorstore = VectorStore()
        
        # Try to load existing vector store first
        if os.path.exists("data/vectorstore/index.faiss"):
            vectorstore.load_existing()
        
        for pdf_path in new_pdfs:
            st.write(f"📄 Processing: {pdf_path.name}")
            extractor = PDFExtractor(str(pdf_path))
            texts = extractor.extract_text()
            
            # Add text documents (no images for now)
            for text in texts:
                all_documents.append({
                    "content": text["content"],
                    "metadata": {
                        "type": "text", 
                        "page": text["page"],
                        "source": str(pdf_path.name)
                    }
                })
            extractor.close()
        
        if all_documents:
            # Create/update vector store
            if os.path.exists("data/vectorstore/index.faiss"):
                vectorstore.add_documents(all_documents)
            else:
                vectorstore.create_from_documents(all_documents)
            
            # Mark PDFs as processed
            with open(processed_file, 'w') as f:
                for pdf in existing_pdfs:
                    f.write(f"{str(pdf)}\n")
            
            st.success(f"✅ Processed {len(new_pdfs)} existing PDFs!")
    
    return len(existing_pdfs)
# ===== END OF FUNCTION =====

# Page config
st.set_page_config(
    page_title="Agentic PDF Assistant",
    page_icon="📚",
    layout="wide"
)

load_existing_pdfs()

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
    process_images = st.checkbox("🖼️ Process images", value=True)
    image_pages = st.slider("Number of pages to scan for images", min_value=1, max_value=20, value=5, 
                           help="More pages = more images but slower processing")
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
                        st.info("🖼️ Processing images with Qwen 3.5 Vision (Hugging Face)...")
                        images = extractor.extract_images(max_pages=image_pages)
                        
                        if images:
                            vision = MultimodalProcessor()
                            
                            # Create progress bar
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for i, img in enumerate(images):
                                # Update status
                                status_text.text(f"Processing image {i+1} of {len(images)} (Page {img['page']})...")
                                
                                # Process image with Qwen
                                description = vision.generate_image_description(img["base64"])

                                # Debug: print what we got
                                print(f"🔍 Image {i+1} on page {img['page']}:")
                                print(f"   Description length: {len(description) if description else 0}")
                                print(f"   Preview: {description[:100] if description else 'None'}...")

                                # Ensure description is never None
                                if description is None:
                                    description = "[Image description unavailable]"
                                elif not isinstance(description, str):
                                    description = str(description)
                                
                                documents.append({
                                    "content": f"[Image on page {img['page']}]: {description}",
                                    "metadata": {
                                        "type": "image", 
                                        "page": img["page"],
                                        "source": "pdf_image",
                                        "image_index": img.get("index", 0)
                                    }
                                })
                                
                                # Update progress
                                progress_bar.progress((i + 1) / len(images))
                            
                            # Clear progress indicators
                            progress_bar.empty()
                            status_text.empty()
                            st.success(f"✅ Processed {len(images)} images successfully!")
                        else:
                            st.info("No images found in the scanned pages.")
                    
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
        placeholder="e.g., What is this document about? Summarize the key points. Describe any charts or images."
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
    
    # Text-to-speech option
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
        - What do the charts tell us about the data?
        - [If web search enabled] What are the latest developments in this field?
        """)
else:
    # Welcome message
    st.info("👈 Please upload and process a PDF file from the sidebar to get started!")
    
    # Features
    st.markdown("""
    ### ✨ Features:
    - 📄 **Text Extraction**: Read text from any PDF
    - 🖼️ **Image Understanding**: Describe charts and images using Qwen 3.5 Vision (Hugging Face)
    - 🌐 **Web Search**: Get real-time information (optional)
    - 🔍 **Smart Search**: Find relevant information quickly
    - 🔊 **Text-to-Speech**: Listen to answers (optional)
    """)

# Footer
st.divider()
st.caption("Made with LangChain, Hugging Face Qwen 3.5 Vision, and Streamlit")