from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import EMBEDDING_MODEL, VECTOR_STORE_PATH
import os

class VectorStore:
    def __init__(self):
        """Initialize embeddings model"""
        print("📚 Loading embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vectorstore = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        print("✅ Embeddings loaded")
    
    def create_from_documents(self, documents):
        """Create new FAISS index from documents"""
        print(f"📝 Creating vector store with {len(documents)} documents...")
        
        # Convert to LangChain documents with chunking
        docs = []
        for doc in documents:
            if doc["metadata"].get("type") == "text" and len(doc["content"]) > 500:
                chunks = self.text_splitter.create_documents(
                    [doc["content"]],
                    [doc["metadata"]]
                )
                docs.extend(chunks)
            else:
                docs.append(Document(
                    page_content=doc["content"],
                    metadata=doc.get("metadata", {})
                ))
        
        self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        self.vectorstore.save_local(VECTOR_STORE_PATH)
        print(f"✅ Vector store saved to {VECTOR_STORE_PATH}")
        return self.vectorstore
    
    def add_documents(self, new_documents):
        """Add new documents to existing vector store"""
        print(f"📝 Adding {len(new_documents)} new documents to existing vector store...")
        
        if not self.vectorstore:
            if not self.load_existing():
                raise Exception("No existing vector store found to add documents to.")
        
        # Convert to LangChain documents with chunking
        docs = []
        for doc in new_documents:
            if doc["metadata"].get("type") == "text" and len(doc["content"]) > 500:
                chunks = self.text_splitter.create_documents(
                    [doc["content"]],
                    [doc["metadata"]]
                )
                docs.extend(chunks)
            else:
                docs.append(Document(
                    page_content=doc["content"],
                    metadata=doc.get("metadata", {})
                ))
        
        # Add to existing index
        self.vectorstore.add_documents(docs)
        
        # Save updated index
        self.vectorstore.save_local(VECTOR_STORE_PATH)
        print(f"✅ Added {len(docs)} chunks/documents to existing vector store")
    
    def load_existing(self):
        """Load existing FAISS index"""
        if os.path.exists(VECTOR_STORE_PATH):
            print("📂 Loading existing vector store...")
            self.vectorstore = FAISS.load_local(
                VECTOR_STORE_PATH, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print("✅ Vector store loaded")
            return True
        return False
    
    def search(self, query, k=4):
        """Search for similar documents"""
        if not self.vectorstore:
            if not self.load_existing():
                raise Exception("No vector store found. Please create one first.")
        
        docs = self.vectorstore.similarity_search(query, k=k)
        return docs