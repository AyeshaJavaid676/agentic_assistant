# test_preload_stats.py
from modules.vector_store import VectorStore
import os

print("="*60)
print("📊 TESTING PRE-LOADED VECTOR STORE STATS")
print("="*60)

# Load the vector store
vs = VectorStore()
if os.path.exists("data/vectorstore/index.faiss"):
    vs.load_existing()
    
    # Test search to see what's stored
    test_queries = [
        "data visualization",
        "python code",
        "image",
        "chart"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Searching for: '{query}'")
        results = vs.search(query, k=3)
        print(f"   Found {len(results)} results")
        
        for i, doc in enumerate(results, 1):
            content_preview = doc.page_content[:100].replace('\n', ' ')
            doc_type = doc.metadata.get('type', 'unknown')
            page = doc.metadata.get('page', '?')
            print(f"   {i}. [{doc_type}] Page {page}: {content_preview}...")
    
    # Count document types
    print("\n" + "="*60)
    print("📈 VECTOR STORE STATISTICS")
    print("="*60)
    
    # Try to get more accurate counts by searching common terms
    all_texts = vs.search("the and of", k=50)  # Get many results
    text_count = len([d for d in all_texts if d.metadata.get('type') == 'text'])
    image_count = len([d for d in all_texts if d.metadata.get('type') == 'image'])
    
    print(f"📄 Text documents (full pages): {text_count}")
    print(f"🖼️ Image documents: {image_count}")
    print(f"📊 Total entries: {len(all_texts)}")
    
else:
    print("❌ No vector store found. Run pre_load_doc.py first!")