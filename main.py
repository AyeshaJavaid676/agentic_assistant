import streamlit as st
import tempfile
import os
from pathlib import Path
from modules.pdf_extractor import PDFExtractor
from modules.multimodal_processor import MultimodalProcessor
from modules.vector_store import VectorStore
from modules.multi_agent import SupervisorAgent
from modules.tts_service import TTSService
from utils.helpers import cleanup_temp_files
import time
import re

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
    
    # Initialize vectorstore variable
    vectorstore = None
    
    # FIRST: Try to load existing vector store (even if no new PDFs)
    if os.path.exists("data/vectorstore/index.faiss"):
        try:
            vectorstore = VectorStore()
            vectorstore.load_existing()
            print(f"✅ Found existing vector store with pre-loaded documents")
        except Exception as e:
            print(f"⚠️ Could not load vector store: {e}")
            vectorstore = None
    
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
        
        # If no vectorstore exists yet, create one
        if vectorstore is None:
            vectorstore = VectorStore()
        
        for pdf_path in new_pdfs:
            st.write(f"📄 Processing: {pdf_path.name}")
            extractor = PDFExtractor(str(pdf_path))
            texts = extractor.extract_text()
            
            # Add text documents
            for text in texts:
                all_documents.append({
                    "content": text["content"],
                    "metadata": {
                        "type": "text", 
                        "page": text["page"],
                        "source": str(pdf_path.name),
                        "document": str(pdf_path.name)
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
    
    # Return both the list of PDFs and the vectorstore
    return existing_pdfs, vectorstore

def extract_sources_from_response(response_text, search_results):
    """
    Extract source citations from the response and match with search results
    """
    sources = []
    
    # Look for page references in the response
    page_matches = re.findall(r'\[Page (\d+)\]', response_text)
    image_matches = re.findall(r'\[Image on page (\d+)\]', response_text)
    
    # Add text page sources
    for page in set(page_matches):
        # Find the corresponding document in search results
        for doc in search_results:
            if doc.metadata.get('page') == int(page) and doc.metadata.get('type') == 'text':
                sources.append({
                    "type": "text",
                    "page": page,
                    "document": doc.metadata.get('document', 'Unknown'),
                    "preview": doc.page_content[:200] + "...",
                    "source_type": "PDF Text"
                })
                break
    
    # Add image sources
    for page in set(image_matches):
        for doc in search_results:
            if doc.metadata.get('page') == int(page) and doc.metadata.get('type') == 'image':
                sources.append({
                    "type": "image",
                    "page": page,
                    "document": doc.metadata.get('document', 'Unknown'),
                    "preview": doc.page_content[:200] + "...",
                    "source_type": "Image Description"
                })
                break
    
    return sources

# Page config
st.set_page_config(
    page_title="Agentic PDF Assistant",
    page_icon="📚",
    layout="wide"
)

# Title
st.title("📚 Agentic PDF Assistant")
st.markdown("Ask questions about your PDFs - upload new ones or query existing documents!")

# Initialize session state - MOVED BEFORE any chat elements
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'temp_file' not in st.session_state:
    st.session_state.temp_file = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []  # Store all messages
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'existing_pdfs' not in st.session_state:
    st.session_state.existing_pdfs = []

# Load existing PDFs and initialize vector store
existing_pdfs, vectorstore = load_existing_pdfs()
st.session_state.existing_pdfs = existing_pdfs

# Just check if PDFs exist and vectorstore files exist
if existing_pdfs and os.path.exists("data/vectorstore/index.faiss") and st.session_state.vectorstore is None:
    # Vector store files exist, load them
    temp_store = VectorStore()
    temp_store.load_existing()
    st.session_state.vectorstore = temp_store
    st.session_state.agent = SupervisorAgent(temp_store)
    st.session_state.processed = True
    
    print(f"✅ Vector store loaded with pre-loaded documents!")
    
    # Create welcome message
    pdf_list = "\n".join([f"   • {pdf.name}" for pdf in existing_pdfs[:5]])
    if len(existing_pdfs) > 5:
        pdf_list += f"\n   • ... and {len(existing_pdfs)-5} more"
    
    welcome_msg = f"""📚 **Welcome! I have {len(existing_pdfs)} PDF(s) pre-loaded and ready:**

{pdf_list}

You can start asking questions about these documents right away! Try asking:
- "What outliers were found in my EDA project?"
- "Tell me about the income analysis in my documents"
- "What are the key findings in my PDFs?"

*No need to upload anything - your documents are already loaded!* 🎉"""
    
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": welcome_msg,
        "timestamp": time.strftime("%H:%M:%S"),
        "sources": []
    })

# Initialize vector store and agent if not already done
if st.session_state.vectorstore is None and vectorstore is not None:
    st.session_state.vectorstore = vectorstore
    st.session_state.agent = SupervisorAgent(vectorstore)
    st.session_state.processed = True
    
    # Add welcome message showing available PDFs
    if existing_pdfs:
        pdf_names = "\n".join([f"   • {pdf.name}" for pdf in existing_pdfs[:5]])
        more_text = f"\n   • ... and {len(existing_pdfs)-5} more" if len(existing_pdfs) > 5 else ""
        
        welcome_msg = f"""📚 **Welcome! I have {len(existing_pdfs)} PDF(s) pre-loaded and ready:**

{pdf_names}{more_text}

You can start asking questions about these documents right away, or upload new ones using the sidebar."""
        
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": welcome_msg,
            "timestamp": time.strftime("%H:%M:%S"),
            "sources": []
        })

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Show existing PDFs count
    if st.session_state.existing_pdfs:
        st.success(f"📚 **{len(st.session_state.existing_pdfs)} PDF(s)** pre-loaded")
        with st.expander("View loaded PDFs"):
            for pdf in st.session_state.existing_pdfs:
                st.write(f"• {pdf.name}")
    
    # Options
    process_images = st.checkbox("🖼️ Process images", value=True)
    image_pages = st.slider("Number of pages to scan for images", min_value=1, max_value=20, value=5, 
                           help="More pages = more images but slower processing")
    enable_web = st.checkbox("🌐 Enable web search", value=True)
    
    st.divider()
    
    # File upload
    uploaded_file = st.file_uploader(
        "📄 Upload New PDF", 
        type="pdf",
        help="Upload any PDF document"
    )
    
    if uploaded_file:
        st.success(f"✅ Uploaded: {uploaded_file.name}")
        
        # Process button
        if st.button("🚀 Process PDF", type="primary"):
            with st.spinner("Processing PDF... This may take a minute..."):
                try:
                    # Save uploaded file permanently
                    pdf_folder = "data/pdfs"
                    os.makedirs(pdf_folder, exist_ok=True)
                    pdf_path = os.path.join(pdf_folder, uploaded_file.name)
                    
                    with open(pdf_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    
                    # Extract text and images
                    st.info("📄 Extracting text...")
                    extractor = PDFExtractor(pdf_path)
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
                                "source": "pdf_text",
                                "document": uploaded_file.name,
                                "filename": uploaded_file.name
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
                                        "document": uploaded_file.name,
                                        "filename": uploaded_file.name,
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
                    
                    # Create or update vector store
                    st.info("📚 Updating searchable database...")
                    
                    if st.session_state.vectorstore is None:
                        st.session_state.vectorstore = VectorStore()
                    
                    # Check if vector store already exists
                    if os.path.exists("data/vectorstore/index.faiss"):
                        st.session_state.vectorstore.load_existing()
                        st.session_state.vectorstore.add_documents(documents)
                    else:
                        st.session_state.vectorstore.create_from_documents(documents)
                    
                    # Create or update agent
                    if st.session_state.agent is None:
                        st.session_state.agent = SupervisorAgent(st.session_state.vectorstore)
                    
                    st.session_state.processed = True
                    
                    # Add success message to chat
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"✅ **PDF '{uploaded_file.name}' processed successfully!**\n\nAdded {len(texts)} pages and {len(images) if images else 0} images to your knowledge base. You can now ask questions about it.",
                        "timestamp": time.strftime("%H:%M:%S"),
                        "sources": []
                    })
                    
                    # Clean up
                    extractor.close()
                    
                    st.success("✅ PDF processed successfully!")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # Clear chat history button
    if st.session_state.chat_history:
        st.divider()
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

# ===== CHAT INTERFACE - ALWAYS VISIBLE =====
st.divider()

# Display chat history (even if no PDF is processed yet)
chat_container = st.container()

with chat_container:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display sources if available
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                with st.expander(f"📚 Sources ({len(message['sources'])})"):
                    for i, source in enumerate(message["sources"], 1):
                        source_type = source.get("source_type", "Document")
                        page = source.get("page", "?")
                        doc_name = source.get("document", "Unknown")
                        
                        if source.get("type") == "image":
                            st.markdown(f"**🖼️ Source {i}:** {source_type} from **{doc_name}**, Page **{page}**")
                        else:
                            st.markdown(f"**📄 Source {i}:** {source_type} from **{doc_name}**, Page **{page}**")
                        
                        st.markdown(f"*Preview:* {source.get('preview', '')}")
                        st.divider()
            
            # Add audio button for assistant messages
            if message["role"] == "assistant" and len(message["content"]) > 10:
                col1, col2 = st.columns([0.1, 0.9])
                with col1:
                    audio_key = f"audio_{hash(message['content'])}"
                    if st.button("🔊", key=audio_key, help="Listen to this response"):
                        with st.spinner("🔊 Generating audio..."):
                            try:
                                from gtts import gTTS
                                import base64
                                import io
                                
                                tts = gTTS(text=message["content"], lang='en', slow=False)
                                fp = io.BytesIO()
                                tts.write_to_fp(fp)
                                fp.seek(0)
                                audio_bytes = fp.read()
                                audio_base64 = base64.b64encode(audio_bytes).decode()
                                
                                audio_html = f"""
                                    <audio autoplay controls style="width: 100%; margin-top: 5px;">
                                        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                                    </audio>
                                """
                                st.markdown(audio_html, unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"🔊 Audio Error: {str(e)}")

# Chat input - ALWAYS VISIBLE
question = st.chat_input(
    "Ask a question about your PDFs...",
    key="chat_input"
)

if question:
    # Add user message to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": question,
        "timestamp": time.strftime("%H:%M:%S")
    })
    
    # Check if vector store exists
    if st.session_state.vectorstore is None:
        # No PDFs loaded yet
        response = """⚠️ **No PDFs have been loaded yet.**

Please upload a PDF using the sidebar or add documents to the `data/pdfs/` folder.

**Available options:**
1. 📤 Use the file uploader in the sidebar
2. 📁 Place PDF files in `data/pdfs/` folder and restart the app
3. 🔄 Refresh after adding files"""
        
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": time.strftime("%H:%M:%S"),
            "sources": []
        })
    else:
        # Get response from agent
        with st.spinner("🤔 Thinking..."):
            try:
                # Search for relevant documents
                search_results = st.session_state.vectorstore.search(question, k=10)
                
                # Get agent response
                response = st.session_state.agent.run(question)
                answer = response['output']
                
                # Extract sources
                sources = extract_sources_from_response(answer, search_results)
                
                # If no sources found but we have search results, add them
                if not sources and search_results:
                    for doc in search_results[:3]:
                        doc_type = doc.metadata.get('type', 'text')
                        sources.append({
                            "type": doc_type,
                            "page": doc.metadata.get('page', '?'),
                            "document": doc.metadata.get('document', 'Unknown'),
                            "preview": doc.page_content[:200] + "...",
                            "source_type": "Image Description" if doc_type == 'image' else "PDF Text"
                        })
                
                # Add assistant message
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "timestamp": time.strftime("%H:%M:%S"),
                    "sources": sources
                })
                
                # Store last response for backward compatibility
                st.session_state.last_response = answer
                
            except Exception as e:
                error_msg = f"❌ **Error:** {str(e)}\n\nPlease try rephrasing your question or check if your PDFs were processed correctly."
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": error_msg,
                    "timestamp": time.strftime("%H:%M:%S"),
                    "sources": []
                })
    
    # Rerun to update chat display
    st.rerun()

# Footer
st.divider()
st.caption("Made with LangChain, Hugging Face Qwen 3.5 Vision, and Streamlit | Answers include source citations for transparency")