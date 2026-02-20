from groq import Groq
from config.settings import GROQ_API_KEY, VISION_MODEL

class VisionProcessor:
    def __init__(self):
        """Initialize Groq client for vision"""
        self.client = Groq(api_key=GROQ_API_KEY)
        print("👁️ Vision processor ready")
    
    def describe_image(self, image_base64, prompt="Describe this image in detail. What charts, graphs, or visual elements do you see?"):
        """Use Groq vision to describe an image"""
        try:
            response = self.client.chat.completions.create(
                model=VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.1,
                max_tokens=300
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[Could not describe image: {str(e)}]"