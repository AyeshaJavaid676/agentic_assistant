from openai import OpenAI
import base64

# Your working token
client = OpenAI(
    base_url="https://router.huggingface.co/v1", 
    api_key= HF_TOKEN 
)

# Read and encode your specific image
with open(r"C:\Users\Optiplex\Downloads\EDA.png", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

# Create message with base64 image
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
                "text": "Describe this EDA (Exploratory Data Analysis) chart. What type of plot is it and what does it show?"
            }
        ]
    }
]

try:
    response = client.chat.completions.create(
        model="Qwen/Qwen3.5-397B-A17B:together",
        messages=messages,
        max_tokens=500,
        temperature=0.3
    )
    print("✅ Success!")
    print(response.choices[0].message.content)
except Exception as e:
    print(f"❌ Error: {e}")
