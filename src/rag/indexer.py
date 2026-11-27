# -*- coding: utf-8 -*-
"""
Aadhi Local RAG Indexer
Extracts, chunks, and indexes all 61+ Tamil literature books & novels
into an ultra-fast offline character-ngram vector store.
"""

import os
import re
import json
import joblib
import pypdf
from sklearn.feature_extraction.text import TfidfVectorizer

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(SRC_DIR)
CATALOG_PATH = os.path.join(BASE_DIR, "data", "novels_catalog.json")
PDF_DIR = os.path.join(BASE_DIR, "data", "pdfs", "novels")
INDEX_OUTPUT_PATH = os.path.join(BASE_DIR, "data", "rag_index", "aadhi_rag_store.pkl")

BOILERPLATE_PATTERNS = [
    "Project Madurai", "Lausanne", "Switzerland", "electronic texts",
    "freely distribute", "header page", "All Rights Reserved", "http://", "https://"
]

def clean_tamil_text(text):
    text = re.sub(r'/g8\d+', '', text)
    for bp in BOILERPLATE_PATTERNS:
        text = text.replace(bp, "")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, chunk_size=120, overlap=30):
    words = text.split()
    chunks = []
    if len(words) <= chunk_size:
        if len(words) >= 15:
            chunks.append(" ".join(words))
        return chunks
    
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if len(chunk.split()) >= 15:
            chunks.append(chunk)
        if end >= len(words):
            break
        start += (chunk_size - overlap)
    return chunks

def extract_pdf_chunks(pdf_path, title, author):
    chunks = []
    try:
        reader = pypdf.PdfReader(pdf_path)
        book_pages = []
        for page in reader.pages:
            t = page.extract_text() or ''
            t = clean_tamil_text(t)
            # Filter pages with at least 15 Tamil characters
            tamil_chars = sum(1 for c in t if '\u0B80' <= c <= '\u0BFF')
            if tamil_chars >= 15:
                book_pages.append(t)
        
        full_book_text = " ".join(book_pages)
        raw_chunks = chunk_text(full_book_text, chunk_size=120, overlap=30)
        
        for c in raw_chunks:
            # Check Tamil character density
            tamil_count = sum(1 for ch in c if '\u0B80' <= ch <= '\u0BFF')
            if tamil_count >= 30:
                # Enrich chunk with author and title context so metadata matches queries directly
                searchable = f"நூல்: {title}. ஆசிரியர்: {author}. {c}"
                chunks.append({
                    "title": title,
                    "author": author,
                    "text": c,
                    "search_text": searchable
                })
    except Exception as e:
        print(f"⚠️ PDF படிப்பதில் தவறு ({os.path.basename(pdf_path)}): {e}")
    return chunks

def build_rag_index():
    print("=" * 65)
    print("📚 ஆதி உள்ளூர் RAG சுட்டகக் கட்டமைப்பு (Local RAG Indexer)")
    print("=" * 65)

    if not os.path.exists(CATALOG_PATH):
        raise FileNotFoundError(f"நூல்கள் பட்டியல் கிடைக்கவில்லை: {CATALOG_PATH}")

    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    all_chunks = []
    indexed_books = 0

    print(f"📖 பட்டியலில் உள்ள மொத்த நூல்கள்: {len(catalog)}")
    for i, item in enumerate(catalog, 1):
        title = item.get("tamil_title") or item.get("title", "நூல்")
        author = item.get("author", "தமிழ் எழுத்தாளர்")
        local_path = item.get("local_path", "")

        if not os.path.exists(local_path):
            # Try finding in PDF_DIR by basename
            base = os.path.basename(local_path)
            alt_path = os.path.join(PDF_DIR, base)
            if os.path.exists(alt_path):
                local_path = alt_path

        if os.path.exists(local_path):
            chunks = extract_pdf_chunks(local_path, title, author)
            if chunks:
                all_chunks.extend(chunks)
                indexed_books += 1
                if i % 10 == 0 or i == len(catalog):
                    print(f"   [{i:2d}/{len(catalog)}] ✅ {title[:30]} ({len(chunks)} பத்திகள்)")

    # Also add Thirukkural & Classical baseline literature if available
    corpus_sample_path = os.path.join(BASE_DIR, "data", "tamil_corpus.txt")
    if os.path.exists(corpus_sample_path):
        try:
            with open(corpus_sample_path, "r", encoding="utf-8") as f:
                head_lines = [f.readline().strip() for _ in range(500)]
            classical_text = " ".join([l for l in head_lines if len(l) > 10])
            for c in chunk_text(classical_text, chunk_size=80, overlap=20):
                all_chunks.append({
                    "title": "தமிழ் செம்மொழி மற்றும் திருக்குறள் அறிமுகம்",
                    "author": "திருவள்ளுவர் & தமிழ் சான்றோர்கள்",
                    "text": c
                })
        except Exception:
            pass

    print(f"\n✨ மொத்தமாகப் பகுப்பாய்வு செய்யப்பட்ட நூல்கள்: {indexed_books}")
    print(f"📦 உருவாக்கப்பட்ட உரைத் துண்டுகள் (Chunks): {len(all_chunks)} பத்திகள்")

    print("\n⚡ தமிழ் எழுத்து-அடுக்கு TF-IDF வெக்டார் மாதிரி உருவாக்கப்படுகிறது...")
    # Char-wb (3 to 6) captures Tamil morphological prefixes, root stems and agglutinative suffixes
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 6),
        max_features=100000,
        sublinear_tf=True
    )

    texts = [c.get("search_text", c["text"]) for c in all_chunks]
    tfidf_matrix = vectorizer.fit_transform(texts)

    rag_store = {
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "chunks": all_chunks
    }

    joblib.dump(rag_store, INDEX_OUTPUT_PATH, compress=3)
    size_mb = os.path.getsize(INDEX_OUTPUT_PATH) / (1024 * 1024)
    print(f"🎉 RAG வெக்டார் ஸ்டோர் வெற்றிகரமாகச் சேமிக்கப்பட்டது: {INDEX_OUTPUT_PATH}")
    print(f"💾 சேமிப்பக அளவு: {size_mb:.2f} MB")
    print("=" * 65)

if __name__ == "__main__":
    build_rag_index()
