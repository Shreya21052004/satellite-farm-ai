import os
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
resp = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": "Say hello"}],
)
print(resp.choices[0].message.content)