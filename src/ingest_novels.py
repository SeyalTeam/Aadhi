"""
Novel Ingestion Pipeline for Aadhi (ஆதி)
Extracts pristine Unicode Tamil text from novels and merges into tamil_corpus.txt
"""

import os
import sys
import ssl
import re
import urllib.request
import sentencepiece as spm

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
CORPUS_PATH = os.path.join(BASE_DIR, "data", "tamil_corpus.txt")
SPM_MODEL_PATH = os.path.join(BASE_DIR, "tokenizer", "spm_tamil.model")

SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}

# The 5 downloaded novels with their pristine UTF-8 sources
NOVEL_SOURCES = [
    {"name": "நெற்றிக் கண் (பாகம் 1 & 2)", "author": "தீபம் நா. பார்த்தசாரதி", "urls": [
        "https://www.projectmadurai.org/pm_etexts/utf8/pmuni0839_01.html",
        "https://www.projectmadurai.org/pm_etexts/utf8/pmuni0839_02.html"
    ]},
    {"name": "மேனகா (முதல் பாகம்)", "author": "வடுவூர் கே. துரைசாமி ஐயங்கார்", "urls": [
        "https://www.projectmadurai.org/pm_etexts/utf8/pmuni0756.html"
    ]},
    {"name": "சாமியாடிகள் (பாகம் 1 & 2)", "author": "சு. சமுத்திரம்", "urls": [
        "https://www.projectmadurai.org/pm_etexts/utf8/pmuni0733_01.html",
        "https://www.projectmadurai.org/pm_etexts/utf8/pmuni0733_02.html"
    ]},
    {"name": "இருளும் ஒளியும்", "author": "ஸரோஜா ராமமூர்த்தி", "urls": [
        "https://www.projectmadurai.org/pm_etexts/utf8/pmuni0719.html"
    ]},
    {"name": "அவள் விழித்திருந்தாள்", "author": "அ.க. நவநீத கிருட்டிணன்", "urls": [
        "https://www.projectmadurai.org/pm_etexts/utf8/pmuni0902.html"
    ]},
]

def clean_html_tamil(html_content):
    # Remove script, style, head
    html_content = re.sub(r'<(script|style|head)[^>]*>.*?</\1>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
    # Convert <br>, <p>, <div> to newlines
    html_content = re.sub(r'<(br|p|div|h[1-6]|li)[^>]*>', '\n', html_content, flags=re.IGNORECASE)
    # Remove remaining HTML tags
    text = re.sub(r'<[^>]+>', ' ', html_content)
    # Decode basic entities
    text = text.replace('&nbsp;', ' ').replace('&quot;', '"').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')

    cleaned_lines = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        # Check if line contains Tamil characters
        tamil_chars = sum(1 for c in line if '\u0B80' <= c <= '\u0BFF')
        # Filter out lines that are purely English or metadata/disclaimers
        if tamil_chars < 5:
            continue
        # Exclude common Project Madurai English/admin headers
        if "Project Madurai" in line or "Lausanne" in line or "Prepared by" in line:
            continue
        cleaned_lines.append(line)

    return cleaned_lines

def ingest():
    print("=" * 65)
    print("🚀 ஆதி நாவல்கள் உரை இணைப்புக் கட்டமைப்பு (Novel Ingestion Pipeline)")
    print("=" * 65)

    all_novel_lines = []

    for novel in NOVEL_SOURCES:
        title = novel["name"]
        author = novel["author"]
        print(f"\n📖 நூலைச் செயலாக்குகிறது: {title} ({author})")
        novel_lines = []

        for url in novel["urls"]:
            try:
                req = urllib.request.Request(url, headers=HEADERS)
                with urllib.request.urlopen(req, timeout=20, context=SSL_CTX) as resp:
                    html_data = resp.read().decode("utf-8", errors="ignore")
                    lines = clean_html_tamil(html_data)
                    novel_lines.extend(lines)
                    print(f"   ✓ {url.split('/')[-1]} -> {len(lines):,} தூய தமிழ் வரிகள் பெறப்பட்டன.")
            except Exception as e:
                print(f"   ⚠️ {url} எடுப்பதில் பிழை: {e}")

        all_novel_lines.extend(novel_lines)
        print(f"   ⭐ மொத்தம் {title}: {len(novel_lines):,} வரிகள்!")

    print("\n" + "=" * 65)
    print(f"📚 புதிதாகச் சேர்க்கப்பட்ட மொத்த நாவல் வரிகள்: {len(all_novel_lines):,}")

    # Append to existing corpus
    print(f"📝 கார்பஸில் இணைக்கிறது: {CORPUS_PATH}")
    with open(CORPUS_PATH, "a", encoding="utf-8") as f:
        f.write("\n\n# --- புதிய நாவல்கள் உரைத் தொகுப்பு ---\n\n")
        f.write("\n".join(all_novel_lines))

    # Read updated corpus stats
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        full_corpus = f.read()
    total_lines = len(full_corpus.splitlines())
    total_bytes_mb = len(full_corpus.encode("utf-8")) / (1024 * 1024)

    print(f"✅ கார்பஸ் புதுப்பிக்கப்பட்டது!")
    print(f"   - மொத்த வரிகள்: {total_lines:,}")
    print(f"   - கோப்பின் அளவு: {total_bytes_mb:.2f} MB")

    # Re-tokenize with SentencePiece to see token count
    if os.path.exists(SPM_MODEL_PATH):
        sp = spm.SentencePieceProcessor(model_file=SPM_MODEL_PATH)
        tokens = sp.encode_as_ids(full_corpus)
        print(f"   - மொத்த தமிழ் டோக்கன்கள் (Tokens): {len(tokens):,} (~{len(tokens)/100000:.1f} லட்சம் டோக்கன்கள்!)")
    print("=" * 65)

if __name__ == "__main__":
    ingest()
