import os
from groq import Groq


class GroqError(RuntimeError):
    pass


def groq_chat(prompt: str, model: str = "llama-3.1-8b-instant") -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise GroqError("GROQ_API_KEY is not set.")

    try:
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        raise GroqError(str(e)) from e