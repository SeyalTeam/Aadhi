"""
Data Collection Script for Project Aadhi (ஆதி)
Fetches authentic public domain Tamil literature (Thirukkural 1330 + Explanations)
and appends it to data/tamil_corpus.txt.
"""

import os
import ssl
import json
import urllib.request

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "tamil_corpus.txt")

# Reliable open-source Thirukkural JSON repository (Public Domain)
THIRUKKURAL_URL = "https://raw.githubusercontent.com/tk120404/thirukkural/master/thirukkural.json"

def fetch_and_append_thirukkural():
    print("📥 திருக்குறள் 1,330 அதிகாரங்கள் மற்றும் விளக்கவுரைகள் பதிவிறக்கம் செய்யப்படுகின்றன...")
    try:
        # Create unverified SSL context for Mac Python environments
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        req = urllib.request.Request(
            THIRUKKURAL_URL,
            headers={"User-Agent": "Aadhi-Tamil-AI/1.0"}
        )
        with urllib.request.urlopen(req, context=ctx, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))
            
        kural_list = data.get("kural", [])
        print(f"✅ மொத்தம் {len(kural_list)} திருக்குறள்கள் இணையத்திலிருந்து வெற்றிகரமாகப் பெறப்பட்டன!")

        extracted_lines = []
        for item in kural_list:
            line1 = item.get("Line1", "").strip()
            line2 = item.get("Line2", "").strip()
            mv_meaning = item.get("mv", "").strip()  # Dr. Mu. Varadarasanar
            sp_meaning = item.get("sp", "").strip()  # Solomon Pappaiah

            if line1 and line2:
                extracted_lines.append(f"{line1} {line2}")
            if mv_meaning:
                extracted_lines.append(f"விளக்கம்: {mv_meaning}")
            if sp_meaning:
                extracted_lines.append(f"எளிய உரை: {sp_meaning}")

        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        with open(DATA_PATH, "a", encoding="utf-8") as f:
            f.write("\n" + "\n".join(extracted_lines) + "\n")

        print(f"🎉 மொத்தம் {len(extracted_lines):,} புதிய திருக்குறள் மற்றும் விளக்க வரிகள் '{DATA_PATH}' கோப்பில் வெற்றிகரமாகச் சேர்க்கப்பட்டன!")

    except Exception as e:
        print(f"⚠️ பிழை: {e}")

if __name__ == "__main__":
    fetch_and_append_thirukkural()
