from openai import OpenAI

# 1. SETUP: Ensure your token is correct
client = OpenAI(
    base_url="https://router.huggingface.co/v1", 
    api_key= HF_TOKEN
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/CI_Demo/mathv-1327.jpg"
                }
            },
            {
                "type": "text",
                "text": "The centres of the four illustrated circles are in the corners of the square. The two big circles touch each other and also the two little circles. With which factor do you have to multiply the radii of the little circles to obtain the radius of the big circles?\nChoices:\n(A) 2/9\n(B) sqrt(5)\n(C) 0.8 * pi\n(D) 2.5\n(E) 1+sqrt(2)"
            }
        ]
    }
]

# 2. EXECUTE: Switched from ':auto' to ':together'
response = client.chat.completions.create(
    model="Qwen/Qwen3.5-397B-A17B:together", 
    messages=messages,
    max_tokens=2000,
    temperature=0.6,
    top_p=0.95,
    extra_body={
        "top_k": 20,
    }, 
)

# 3. PRINT: Changed 'chat_response' to 'response' to avoid NameError
print("Chat response:", response.choices[0].message.content)
