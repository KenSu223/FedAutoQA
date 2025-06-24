# llm.py
"""
Gemini wrapper for HCUP RAG.
Only answers from supplied context; refuses otherwise.
"""

import os, textwrap
from typing import List
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables (.env optional)
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "Set GOOGLE_API_KEY or GEMINI_API_KEY in environment or a .env file."
    )

# Initialise Gemini client
client = genai.Client(api_key=API_KEY)
MODEL_NAME = "gemini-2.0-flash"  # or "gemini-2.0-pro" for higher quality

SYSTEM_PROMPT = textwrap.dedent("""
    You are an HCUP FAQ assistant.
    You must answer **ONLY** based on information explicitly provided in the CONTEXT
    section.  If provided information is not sufficient to answer the query, reply exactly:
    "I’m sorry, I don’t have that information."
    Do not add outside knowledge or speculation.
""").strip()

def answer_with_context(query: str, contexts: List[str]) -> str:
    """
    Send the query plus retrieved context to Gemini and return the text answer.
    Uses the low‑level `client.models.generate_content` call pattern.
    """
    if not contexts:
        return "I’m sorry, I don’t have that information."

    joined_ctx = "\n\n---\n\n".join(contexts)
    contents = f"CONTEXT:\n{joined_ctx}\n\nQUESTION:\n{query}"

    try:
        # Gemini SDK expects `system_instruction` when supplying a system prompt.
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT
            )
        )
        return response.text.strip()
    except Exception as e:
        # Fail fast but clearly
        return (
            "I’m sorry, I encountered an error while generating an answer. "
            f"(debug: {e})"
        )