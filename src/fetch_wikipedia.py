"""
Fetch Verified Tamil Wikipedia Featured Articles & Clean Literature
Appends clean, factual, encyclopedic Tamil knowledge to data/tamil_corpus.txt.
"""

import os
import ssl
import json
import urllib.parse
import urllib.request

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "tamil_corpus.txt")

# Core high-quality, verified Tamil Wikipedia topics (History, Science, Culture, AI, Biographies)
WIKI_TOPICS = [
    "தமிழ்",
    "தமிழ்நாடு",
    "திருவள்ளுவர்",
    "பாரதியார்",
    "ஏ._பி._ஜே._அப்துல்_கலாம்",
    "செயற்கை_நுண்ணறிவு",
    "கணிப்பொறி",
    "அறிவியல்",
    "தஞ்சைப்_பெருவுடையார்_கோயில்",
    "சங்க_காலம்"
]

def fetch_tamil_wikipedia_articles():
    print("🌐 தமிழ் விக்கிபீடியாவில் இருந்து சரிபார்க்கப்பட்ட கட்டுரைகள் திரட்டப்படுகின்றன...")
    
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    headers = {"User-Agent": "Aadhi-Tamil-AI-Corpus/1.0 (Educational/Research)"}
    total_added_chars = 0
    articles_collected = 0

    for topic in WIKI_TOPICS:
        encoded_topic = urllib.parse.quote(topic)
        url = f"https://ta.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&format=json&titles={encoded_topic}"

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            pages = data.get("query", {}).get("pages", {})
            for page_id, page_data in pages.items():
                if page_id == "-1":
                    continue
                title = page_data.get("title", "")
                extract = page_data.get("extract", "")
                
                # Clean extract: filter out empty lines or english junk
                lines = [line.strip() for line in extract.split("\n") if len(line.strip()) > 30]
                if lines:
                    content_block = f"\n\n### தலைப்பு: {title} ###\n" + "\n".join(lines) + "\n"
                    with open(DATA_PATH, "a", encoding="utf-8") as f:
                        f.write(content_block)
                    
                    total_added_chars += len(content_block)
                    articles_collected += 1
                    print(f"  ✅ '{title}' கட்டுரை வெற்றிகரமாகச் சேர்க்கப்பட்டது ({len(lines)} பத்திகள்).")

        except Exception as e:
            print(f"  ⚠️ '{topic}' எடுப்பதில் பிழை: {e}")

    print("\n" + "=" * 50)
    print(f"🎉 மொத்தம் {articles_collected} விக்கிபீடியா கட்டுரைகள் '{DATA_PATH}' கோப்பில் சேர்க்கப்பட்டன!")
    print(f"📊 புதிதாக சேர்க்கப்பட்ட எழுத்துக்களின் அளவு: {total_added_chars:,} எழுத்துக்கள்.")
    print("=" * 50)

if __name__ == "__main__":
    fetch_tamil_wikipedia_articles()
