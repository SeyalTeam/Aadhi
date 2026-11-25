"""
Corpus Sanitizer for Project Aadhi (ஆதி)
Cleans watermarks (tamilbooksworld), URL links, PDF font artifacts, and junk symbols.
"""

import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "tamil_corpus.txt")
BACKUP_PATH = os.path.join(BASE_DIR, "data", "tamil_corpus_raw_backup.txt")

def clean_corpus():
    print("🧹 ஆதி தமிழ் தரவு சுத்திகரிப்பு (Corpus Cleaning) தொடங்குகிறது...")

    if not os.path.exists(DATA_PATH):
        print(f"⚠️ தரவுக் கோப்பு இல்லை: {DATA_PATH}")
        return

    # Backup original raw file
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw_text = f.read()

    with open(BACKUP_PATH, "w", encoding="utf-8") as f:
        f.write(raw_text)
    print(f"💾 அசல் தரவுக் கோப்பு காப்புப் பிரதி (Backup) செய்யப்பட்டது.")

    cleaned_lines = []
    lines = raw_text.split("\n")
    print(f"📊 சுத்திகரிப்பிற்கு முன் மொத்த வரிகள்: {len(lines):,}")

    junk_patterns = [
        re.compile(r"tamilbooksworld", re.IGNORECASE),
        re.compile(r"t\.me\S*", re.IGNORECASE),
        re.compile(r"https?://\S+"),
        re.compile(r"www\.\S+"),
        re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    ]

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip lines with watermarks/links
        if any(p.search(line) for p in junk_patterns):
            continue

        # Remove strange PDF font artifact characters
        line = re.sub(r"[\xa0\u200b\u200e\u200f\ufeff]", " ", line)
        line = re.sub(r"[\{\}\[\]\(\)\<\>~©®Ƭ¢\|\\\/_#=]", " ", line)
        
        # Collapse repeated punctuation (e.g. '.....' -> '.', '::::' -> ':')
        line = re.sub(r"\.{2,}", ".", line)
        line = re.sub(r":{2,}", ":", line)
        line = re.sub(r",{2,}", ",", line)
        line = re.sub(r"-{2,}", "-", line)
        line = re.sub(r"\s+", " ", line).strip()

        # Keep lines that have meaningful Tamil content (at least 3 Tamil characters)
        tamil_chars = re.findall(r"[\u0B80-\u0BFF]", line)
        if len(tamil_chars) >= 4:
            cleaned_lines.append(line)

    cleaned_corpus = "\n".join(cleaned_lines)
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        f.write(cleaned_corpus)

    print(f"✅ சுத்திகரிப்பு நிறைவுற்றது!")
    print(f"📝 எஞ்சியிருக்கும் தூய தமிழ் வரிகள்: {len(cleaned_lines):,}")
    print(f"🗑️ நீக்கப்பட்ட குப்பைகள் & வாட்டர்மார்க்குகள்: {len(lines) - len(cleaned_lines):,} வரிகள்.")
    print("=" * 60)

if __name__ == "__main__":
    clean_corpus()
