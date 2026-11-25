"""
Batch Ingestion for All Downloaded Novels into Aadhi's Tamil Corpus
Fetches pristine Unicode text for all cataloged novels and updates data/tamil_corpus.txt
"""

import os
import sys
import ssl
import re
import json
import urllib.request
import pypdf
import sentencepiece as spm

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
CORPUS_PATH = os.path.join(BASE_DIR, "data", "tamil_corpus.txt")
CATALOG_PATH = os.path.join(BASE_DIR, "data", "novels_catalog.json")
SPM_MODEL_PATH = os.path.join(BASE_DIR, "tokenizer", "spm_tamil.model")

SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}

def clean_lines(raw_lines):
    cleaned = []
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        # Ensure at least 5 Tamil characters
        tamil_chars = sum(1 for c in line if '\u0B80' <= c <= '\u0BFF')
        if tamil_chars < 5:
            continue
        # Exclude administrative metadata
        if any(bad in line for bad in ["Project Madurai", "Lausanne", "Prepared by", "All Rights Reserved", "http://", "https://"]):
            continue
        # Exclude glyph corruption tags like /g8...
        if "/g8" in line:
            line = re.sub(r'/g8\d+', '', line).strip()
            if sum(1 for c in line if '\u0B80' <= c <= '\u0BFF') < 5:
                continue
        cleaned.append(line)
    return cleaned

def clean_html_tamil(html_content):
    html_content = re.sub(r'<(script|style|head)[^>]*>.*?</\1>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
    html_content = re.sub(r'<(br|p|div|h[1-6]|li|tr)[^>]*>', '\n', html_content, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', html_content)
    text = text.replace('&nbsp;', ' ').replace('&quot;', '"').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    return clean_lines(text.split('\n'))

def extract_from_pdf(pdf_path):
    lines = []
    try:
        reader = pypdf.PdfReader(pdf_path)
        for page in reader.pages:
            t = page.extract_text() or ''
            for line in t.split('\n'):
                lines.append(line)
    except Exception as e:
        print(f"      PDF வாசிப்பதில் பிழை ({pdf_path}): {e}")
    return clean_lines(lines)

def fetch_utf8_online(pm_code):
    candidates = [
        f"https://www.projectmadurai.org/pm_etexts/utf8/{pm_code}.html",
        f"https://www.projectmadurai.org/pm_etexts/utf8/{pm_code}_01.html",
        f"https://www.projectmadurai.org/pm_etexts/utf8/{pm_code}_02.html",
    ]
    all_lines = []
    for url in candidates:
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=12, context=SSL_CTX) as resp:
                data = resp.read().decode("utf-8", errors="ignore")
                l = clean_html_tamil(data)
                if len(l) > 50:
                    all_lines.extend(l)
        except:
            continue
    return all_lines

def run_ingestion():
    print("=" * 65)
    print("🚀 ஆதி அனைத்து நாவல்கள் முழு இணைப்புப் பணி (Full Novels Ingestion)")
    print("=" * 65)

    if not os.path.exists(CATALOG_PATH):
        print("❌ பட்டியல் கோப்பு கிடைக்கவில்லை:", CATALOG_PATH)
        return

    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    # Read existing corpus to prevent duplicates
    existing_text = ""
    if os.path.exists(CORPUS_PATH):
        with open(CORPUS_PATH, "r", encoding="utf-8") as f:
            existing_text = f.read()

    new_lines = []
    processed_count = 0

    for i, item in enumerate(catalog, 1):
        title = item.get("tamil_title", "நூல்")
        author = item.get("author", "Unknown")
        pdf_url = item.get("pdf_url", "")
        local_path = item.get("local_path", "")

        # Skip if already ingested earlier
        if title in existing_text[:1000000] and len(title) > 6:
            # Already tested
            pass

        print(f"\n[{i}/{len(catalog)}] 📚 ஆய்வு செய்கிறது: {title} ({author})")

        # Try extracting PM code
        pm_match = re.search(r'(pm\d+)', pdf_url)
        book_lines = []

        if pm_match:
            pm_code = pm_match.group(1).replace("pm", "pmuni")
            print(f"   🔍 Project Madurai UTF-8 பதிப்பைத் தேடுகிறது: {pm_code}...")
            book_lines = fetch_utf8_online(pm_code)

        # Fallback to local PDF extraction if UTF-8 not found or too small
        if len(book_lines) < 100 and os.path.exists(local_path):
            print(f"   📄 உள்ளூர் PDF-லிருந்து பிரித்தெடுக்கிறது: {os.path.basename(local_path)}...")
            pdf_lines = extract_from_pdf(local_path)
            if len(pdf_lines) > len(book_lines):
                book_lines = pdf_lines

        if book_lines:
            # Exclude lines already in existing corpus
            fresh_lines = [l for l in book_lines if l[:40] not in existing_text]
            if fresh_lines:
                new_lines.extend(fresh_lines)
                processed_count += 1
                print(f"   ✅ {len(fresh_lines):,} புதிய தூய தமிழ் வரிகள் சேர்க்கப்பட்டன!")
            else:
                print("   ℹ️ இந்த நூலின் வரிகள் ஏற்கனவே கார்பஸில் உள்ளன.")
        else:
            print("   ⚠️ உரை எதுவும் பிரித்தெடுக்க முடியவில்லை.")

    print("\n" + "=" * 65)
    print(f"🎉 புதிதாகச் சேர்க்கப்பட்ட மொத்த நாவல் வரிகள்: {len(new_lines):,}")

    if new_lines:
        print(f"📝 கார்பஸில் எழுதுகிறது: {CORPUS_PATH}")
        with open(CORPUS_PATH, "a", encoding="utf-8") as f:
            f.write("\n\n# --- கூடுதல் நாவல்கள் தொகுப்பு ---\n\n")
            f.write("\n".join(new_lines))

    # Calculate updated stats
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        full_text = f.read()

    total_lines = len(full_text.splitlines())
    size_mb = len(full_text.encode("utf-8")) / (1024 * 1024)

    print(f"📊 புதுப்பிக்கப்பட்ட கார்பஸ் விவரங்கள்:")
    print(f"   - மொத்த வரிகள்: {total_lines:,}")
    print(f"   - கோப்பின் அளவு: {size_mb:.2f} MB")

    if os.path.exists(SPM_MODEL_PATH):
        sp = spm.SentencePieceProcessor(model_file=SPM_MODEL_PATH)
        tokens = sp.encode_as_ids(full_text)
        print(f"   - மொத்த தமிழ் டோக்கன்கள் (Tokens): {len(tokens):,} (~{len(tokens)/100000:.1f} லட்சம் டோக்கன்கள்!)")
    print("=" * 65)

if __name__ == "__main__":
    run_ingestion()
