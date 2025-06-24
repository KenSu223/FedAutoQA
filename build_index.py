# build_index.py
"""
build_index.py
--------------
Read faq_pairs.jsonl, embed the questions, and build a FAISS index plus
metadata lookup. Run once after scraping (or whenever content changes).
Outputs:
  * faiss_index.bin  – vector index
  * metadata.pkl     – list[dict] aligned with index ids
"""
import json, pickle, faiss, numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_FILE = "faiss_index.bin"
META_FILE  = "metadata.pkl"
FAQ_FILE   = "faq_pairs.jsonl"

def load_pairs(path):
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            yield json.loads(line)

def main():
    pairs = list(load_pairs(FAQ_FILE))
    questions = [p["question"] for p in pairs]

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        questions, convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])  # cosine = dot on normed vecs
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as fp:
        pickle.dump(pairs, fp)

    print(f"Indexed {len(pairs)} FAQs -> {INDEX_FILE}, {META_FILE}")

if __name__ == "__main__":
    main()