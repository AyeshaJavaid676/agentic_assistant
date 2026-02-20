from openai import OpenAI
from config.settings import HF_TOKEN
import base64
from PIL import Image
import io
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

class MultimodalProcessor:
    def __init__(self):
        """Initialize both vision and embedding models"""
        # Don't create client here - just store config
        self.base_url = "https://router.huggingface.co/v1"
        self.api_key = "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"
        self.vision_model = "Qwen/Qwen3.5-397B-A17B:together"
        
        # Embedding model for text (using sentence-transformers)
        print("📊 Loading embedding model...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model ready")
        
    def generate_image_description(self, image_base64):
        """Generate rich caption for image using Qwen"""
        try:
            # Create NEW client for each request (like working test)
            client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )
            
            image_data_url = f"data:image/jpeg;base64,{image_base64}"
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_url}
                        },
                        {
                            "type": "text",
                            "text": "Describe this image in detail, including any charts, graphs, text, or visual elements. Be specific about colors, shapes, and relationships."
                        }
                    ]
                }
            ]
            
            # Use temperature 0.3 like working test
            response = client.chat.completions.create(
                model=self.vision_model,
                messages=messages,
                max_tokens=500,
                temperature=0.3
            )
            
            description = response.choices[0].message.content
            print(f"✅ Generated description: {description[:50]}...")
            return description
            
        except Exception as e:
            print(f"❌ Description generation error: {e}")
            # Return error message instead of None so we know what failed
            return f"[Image description failed: {str(e)}]"
    
    def create_text_embedding(self, text):
        """Create embedding vector for text"""
        try:
            embedding = self.embedder.encode(text)
            return embedding.tolist()
        except Exception as e:
            print(f"❌ Embedding error: {e}")
            return None
    
    def process_image_for_rag(self, image_base64, image_path=None, page_num=None):
        """
        Complete pipeline: 
        1. Generate description
        2. Create embedding
        3. Return document for vector store
        """
        # Step 1: Generate rich description
        description = self.generate_image_description(image_base64)
        if not description:
            return None
        
        # Step 2: Create embedding of the description
        embedding = self.create_text_embedding(description)
        
        # Step 3: Prepare document for vector store
        document = {
            "content": f"[Image on page {page_num}]: {description}",
            "metadata": {
                "type": "image",
                "page": page_num,
                "source": "pdf_image",
                "image_path": image_path,
                "embedding": embedding
            }
        }
        
        return document
    
    def search_similar_images(self, query, documents, top_k=5):
        """
        Search for images similar to a text query or another image
        """
        # Create query embedding
        query_embedding = self.create_text_embedding(query)
        if not query_embedding:
            return []
        
        # Calculate similarities
        similarities = []
        for i, doc in enumerate(documents):
            if doc['metadata'].get('type') == 'image' and 'embedding' in doc['metadata']:
                doc_embedding = doc['metadata']['embedding']
                # Cosine similarity
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                similarities.append((i, similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [documents[i] for i, _ in similarities[:top_k]]