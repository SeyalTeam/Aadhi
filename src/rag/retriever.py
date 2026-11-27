# -*- coding: utf-8 -*-
"""
Aadhi Local RAG Retriever
Provides millisecond semantic search and retrieval across 61 Tamil literature books.
"""

import os
import joblib
from sklearn.metrics.pairwise import linear_kernel

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(SRC_DIR)
INDEX_PATH = os.path.join(BASE_DIR, "data", "rag_index", "aadhi_rag_store.pkl")

class AadhiRAGRetriever:
    _instance = None

    def __init__(self, index_path=INDEX_PATH):
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"RAG குறியீட்டுத் தொகுப்பு கிடைக்கவில்லை: {index_path}\nமுதலில் 'python3 src/rag/indexer.py' இயக்கவும்.")
        print(f"🔄 RAG வெக்டார் தரவுத்தளம் ஏற்றப்படுகிறது: {os.path.basename(index_path)}")
        data = joblib.load(index_path)
        self.vectorizer = data["vectorizer"]
        self.tfidf_matrix = data["tfidf_matrix"]
        self.chunks = data["chunks"]
        print(f"✅ RAG ஏற்றப்பட்டது: {len(self.chunks)} பத்திகள் ஆய்வுக்குத் தயார்!")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def retrieve(self, query, top_k=3, min_score=0.12):
        if not query or not query.strip():
            return []

        q_vec = self.vectorizer.transform([query.strip()])
        similarities = linear_kernel(q_vec, self.tfidf_matrix).flatten()

        # Get top-k indices
        top_indices = similarities.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= min_score:
                chunk = self.chunks[idx]
                results.append({
                    "title": chunk["title"],
                    "author": chunk["author"],
                    "text": chunk["text"],
                    "score": round(score, 4)
                })
        return results

    def format_context_prompt(self, query, max_context_chars=350):
        results = self.retrieve(query, top_k=2)
        if not results:
            return None, []

        context_texts = []
        for r in results:
            context_texts.append(f"[{r['title']} - {r['author']}]: {r['text'][:180]}")

        combined_context = "\n".join(context_texts)
        if len(combined_context) > max_context_chars:
            combined_context = combined_context[:max_context_chars] + "..."

        prompt = f"மேற்கோள் குறிப்பு:\n{combined_context}\n\nகேள்வி: {query}\nபதில்:"
        return prompt, results

def test_retriever():
    retriever = AadhiRAGRetriever()
    queries = [
        "வேள்பாரி நாவல் ஆசிரியர்",
        "சுஜாதா எழுதிய நாவல்",
        "கல்கி அலை ஓசை",
        "நெற்றிக் கண் ஆசிரியர் பார்த்தசாரதி"
    ]
    print("\n" + "=" * 65)
    print("🔍 RAG சோதனை வினாக்கள்:")
    print("=" * 65)
    for q in queries:
        print(f"\n❓ தேடல்: {q}")
        res = retriever.retrieve(q, top_k=1)
        if res:
            r = res[0]
            print(f"   📖 நூல்: {r['title']} | ஆசிரியர்: {r['author']} (பொருத்தம்: {r['score']})")
            print(f"   📝 உரை: {r['text'][:140]}...")
        else:
            print("   ⚠️ பொருத்தமான உரை கிடைக்கவில்லை.")

if __name__ == "__main__":
    test_retriever()
