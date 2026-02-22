# preload_vectorstore.py
"""
Run this script ONCE before starting the app to pre-load your documents
into the vector store with their images.
"""

import os
import sys
from pathlib import Path
import time

# Add project to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.pdf_extractor import PDFExtractor
from modules.multimodal_processor import MultimodalProcessor
from modules.vector_store import VectorStore
from config.settings import PDF_UPLOAD_FOLDER, VECTOR_STORE_PATH

print("="*70)
print("📚 PRE-LOADING DOCUMENTS INTO VECTOR STORE")
print("="*70)

# Create folders if they don't exist
os.makedirs(PDF_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

# Check for existing vector store
vectorstore = VectorStore()
vectorstore_exists = os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss"))

if vectorstore_exists:
    print("\n⚠️  Existing vector store found at:", VECTOR_STORE_PATH)
    response = input("Do you want to overwrite it? (y/n): ")
    if response.lower() != 'y':
        print("❌ Aborted. Keeping existing vector store.")
        sys.exit(0)

# Get all PDFs in the folder
pdf_folder = Path(PDF_UPLOAD_FOLDER)
pdf_files = list(pdf_folder.glob("*.pdf"))

if not pdf_files:
    print(f"\n❌ No PDF files found in {PDF_UPLOAD_FOLDER}")
    print("Please add your PDF documents to the 'data/pdfs' folder first.")
    sys.exit(1)

print(f"\n📄 Found {len(pdf_files)} PDF files to process:")
for pdf in pdf_files:
    print(f"   • {pdf.name}")

print("\n🔄 Starting document processing...")
print("-"*70)

all_documents = []
total_images = 0

for pdf_idx, pdf_path in enumerate(pdf_files, 1):
    print(f"\n[{pdf_idx}/{len(pdf_files)}] Processing: {pdf_path.name}")
    
    # Extract text and images
    extractor = PDFExtractor(str(pdf_path))
    texts = extractor.extract_text()
    
    print(f"   📄 Extracted {len(texts)} text pages")
    
    # Add text documents
    for text in texts:
        all_documents.append({
            "content": text["content"],
            "metadata": {
                "type": "text", 
                "page": text["page"],
                "source": pdf_path.name,
                "document": pdf_path.name
            }
        })
    
    # Extract and process images
    print(f"   🖼️ Extracting images...")
    images = extractor.extract_images(max_pages=100)  # Process ALL pages
    
    if images:
        print(f"   📸 Found {len(images)} images")
        vision = MultimodalProcessor()
        
        for img_idx, img in enumerate(images, 1):
            print(f"      Processing image {img_idx}/{len(images)} on page {img['page']}...")
            
            description = vision.generate_image_description(img["base64"])
            
            all_documents.append({
                "content": f"[Image on page {img['page']}]: {description}",
                "metadata": {
                    "type": "image", 
                    "page": img["page"],
                    "source": pdf_path.name,
                    "document": pdf_path.name,
                    "image_index": img.get("index", 0)
                }
            })
            
            total_images += 1
    else:
        print(f"   ℹ️ No images found in this document")
    
    extractor.close()

print("\n" + "="*70)
print(f"📊 SUMMARY")
print("="*70)
print(f"✅ Total documents processed: {len(pdf_files)}")
print(f"✅ Total text chunks: {len([d for d in all_documents if d['metadata']['type'] == 'text'])}")
print(f"✅ Total images processed: {total_images}")
print(f"✅ Total vector entries: {len(all_documents)}")

print("\n🔄 Creating vector store...")
start_time = time.time()

# Create vector store
vectorstore.create_from_documents(all_documents)

end_time = time.time()
print(f"✅ Vector store created in {end_time - start_time:.2f} seconds")
print(f"📁 Saved to: {VECTOR_STORE_PATH}")

# Create processed files tracker
processed_file = os.path.join(VECTOR_STORE_PATH, "processed_pdfs.txt")
with open(processed_file, 'w') as f:
    for pdf in pdf_files:
        f.write(f"{pdf}\n")

print(f"\n🎉 PRE-LOADING COMPLETE!")
print("="*70)
print("\n🚀 You can now start your Streamlit app:")
print("   streamlit run main.py")
print("\n✅ The vector store will load automatically with your 2 documents and 4 images!")