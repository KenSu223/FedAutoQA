# api.py
"""
api.py
------
FastAPI micro‑service. POST /ask { "query": "..." } → best answer or fallback.
"""
import pickle, faiss, numpy as np, uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from llm import answer_with_context  # Gemini RAG wrapper

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_FILE = "faiss_index.bin"
META_FILE  = "metadata.pkl"
SIM_THRESHOLD = 0.5      # tweak via evaluation
TOP_K = 5  # number of FAQ chunks to feed the LLM

# ---- default system prompt for fallback answers ----
SYSTEM_PROMPT_FILE = "system_prompt.txt"  # plain‑text summary of CDORS/HCUP site
with open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as fp:
    SYSTEM_PROMPT = fp.read().strip()

app = FastAPI(title="HCUP‑FAQ Bot (POC)")

class AskReq(BaseModel):
    query: str

# ---- load model & index at startup ----
model = SentenceTransformer(MODEL_NAME)
index = faiss.read_index(INDEX_FILE)
with open(META_FILE, "rb") as fp:
    metadata = pickle.load(fp)

@app.post("/ask")
def ask(req: AskReq):
    # Embed query and retrieve top‑k similar FAQ questions
    query_vec = model.encode(
        req.query, convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")
    D, I = index.search(np.expand_dims(query_vec, 0), k=TOP_K)

    scored = [
        (float(D[0][j]), int(I[0][j]))
        for j in range(TOP_K)
        if float(D[0][j]) >= SIM_THRESHOLD
    ]

    if scored:
        # Build context snippets from retrieved FAQs
        contexts = [
            f"FAQ ID: {metadata[idx]['id']}\nQ: {metadata[idx]['question']}\nA: {metadata[idx]['answer']}"
            for _, idx in scored
        ]
    else:
        # No similar entries – fall back on the generic system prompt
        contexts = [SYSTEM_PROMPT]

    # Call Gemini via llm.answer_with_context()
    answer_text = answer_with_context(req.query, contexts)

    return {
        "answer": answer_text,
        "similarities": [sim for sim, _ in scored],
    }

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)