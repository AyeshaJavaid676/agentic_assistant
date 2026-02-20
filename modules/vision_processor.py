from openai import OpenAI
from config.settings import HF_TOKEN
import base64

class VisionProcessor:
    def __init__(self):
        """Initialize Hugging Face client using OpenAI-compatible endpoint"""
        if not HF_TOKEN:
            raise ValueError("HF_TOKEN not found in environment variables")
        
        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=HF_TOKEN
        )
        self.model = "Qwen/Qwen3.5-397B-A17B:together"
        print(f"👁️ Vision Processor ready with model: {self.model}")
    
    def describe_image(self, image_base64, prompt="Describe this image. What charts, graphs, or visual elements do you see?"):
        """Describe an image using Qwen 3.5 vision model with base64"""
        try:
            # Create message with base64 image (exactly like your working test)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=500,
                temperature=0.3,
                top_p=0.95
            )
            
            description = response.choices[0].message.content
            print(f"✅ Image described successfully")
            return description
            
        except Exception as e:
            print(f"❌ Vision API Error: {e}")
            return f"[Could not describe image: {str(e)}]"