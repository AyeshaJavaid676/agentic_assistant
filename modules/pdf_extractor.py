import fitz  # PyMuPDF
import base64
from pathlib import Path
import os

class PDFExtractor:
    def __init__(self, pdf_path):
        """Initialize with PDF file path"""
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        print(f"📄 Loaded PDF: {Path(pdf_path).name}")
        print(f"📊 Total pages: {len(self.doc)}")
    
    def extract_text(self):
        """Extract text from all pages"""
        texts = []
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            text = page.get_text()
            if text.strip():
                texts.append({
                    "page": page_num + 1,
                    "content": text
                })
        print(f"✅ Extracted text from {len(texts)} pages")
        return texts
    
    def extract_images(self, max_pages=5):
        """Extract images from first few pages"""
        images = []
        pages_to_process = min(max_pages, len(self.doc))
        
        for page_num in range(pages_to_process):
            page = self.doc[page_num]
            image_list = page.get_images()
            
            for img_idx, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = self.doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Convert to base64 for API
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                    
                    images.append({
                        "page": page_num + 1,
                        "base64": image_base64,
                        "format": base_image["ext"],
                        "index": img_idx
                    })
                    print(f"  ✅ Image {img_idx+1} on page {page_num+1}")
                except Exception as e:
                    print(f"  ⚠️ Failed to extract image on page {page_num+1}: {e}")
        
        print(f"✅ Extracted {len(images)} images")
        return images
    
    def close(self):
        """Close the PDF document"""
        self.doc.close()