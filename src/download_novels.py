"""
Automated Tamil Novel Downloader from tamilbookspdf.com
Downloads top Tamil novels directly to data/pdfs/novels/
"""

import os
import sys
import ssl
import time
import json
import re
import urllib.request
from urllib.parse import unquote

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
NOVELS_DIR = os.path.join(BASE_DIR, "data", "pdfs", "novels")
CATALOG_PATH = os.path.join(BASE_DIR, "data", "novels_catalog.json")

os.makedirs(NOVELS_DIR, exist_ok=True)

# SSL context for macOS
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def fetch_url(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=15, context=SSL_CTX) as response:
        return response.read()

def download_file(url, target_path):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as response:
        with open(target_path, "wb") as f:
            f.write(response.read())

def get_novels_on_page(page=1):
    if page == 1:
        url = "https://tamilbookspdf.com/genre/novels/"
    else:
        url = f"https://tamilbookspdf.com/genre/novels/page/{page}/"

    try:
        content = fetch_url(url).decode("utf-8")
        links = set(re.findall(r'href=\"(https://tamilbookspdf\.com/books/[^\"]+/)\"', content))
        valid_links = [l for l in links if not l.endswith('/books/') and '#subjects' not in l and '#writers' not in l]
        return valid_links
    except Exception as e:
        print(f"⚠️ பக்கம் {page} எடுப்பதில் பிழை: {e}")
        return []

def get_book_details(book_url):
    try:
        content = fetch_url(book_url).decode("utf-8")
        
        # Extract title
        title_match = re.search(r'<h1 class="book__title">([^<]+)</h1>', content)
        tamil_title_match = re.search(r'<div class="book__tamil"[^>]*>([^<]+)</div>', content)
        author_match = re.search(r'<dt>Author</dt>\s*<dd>([^<]+)</dd>', content)
        
        title = title_match.group(1).strip() if title_match else "Unknown"
        tamil_title = tamil_title_match.group(1).strip() if tamil_title_match else title
        author = author_match.group(1).strip() if author_match else "Unknown"

        # Extract PDF download link
        pdf_match = re.search(r'href=\"(https://tamilbookspdf\.com/pdf/[^\"]+\.pdf[^\"]*)\"', content)
        pdf_url = pdf_match.group(1) if pdf_match else None

        return {
            "title": title,
            "tamil_title": tamil_title,
            "author": author,
            "book_url": book_url,
            "pdf_url": pdf_url
        }
    except Exception as e:
        print(f"⚠️ விவரம் பெறுவதில் பிழை ({book_url}): {e}")
        return None

def download_batch(limit=10, max_pages=3):
    print("=" * 65)
    print("🚀 தமிழ் நாவல்கள் நேரடிப் பதிவிறக்கக் கருவி (Aadhi Novel Downloader)")
    print(f"🎯 இலக்கு: {limit} நாவல்கள்")
    print(f"📁 சேமிக்கும் இடம்: {NOVELS_DIR}")
    print("=" * 65)

    catalog = []
    if os.path.exists(CATALOG_PATH):
        try:
            with open(CATALOG_PATH, "r", encoding="utf-8") as f:
                catalog = json.load(f)
        except:
            catalog = []

    downloaded_urls = {item.get("book_url") for item in catalog}
    count = 0

    for page in range(1, max_pages + 1):
        if count >= limit:
            break
        print(f"\n📄 நாவல்கள் பட்டியல் பக்கம் {page} ஆராயப்படுகிறது...")
        book_links = get_novels_on_page(page)
        print(f"   கண்டறியப்பட்ட புத்தக இணைப்புகள்: {len(book_links)}")

        for b_url in book_links:
            if count >= limit:
                break
            if b_url in downloaded_urls:
                continue

            details = get_book_details(b_url)
            if not details or not details["pdf_url"]:
                continue

            # Generate filename
            slug = b_url.rstrip("/").split("/")[-1]
            filename = f"{slug}.pdf"
            save_path = os.path.join(NOVELS_DIR, filename)

            print(f"\n[{count + 1}/{limit}] 📥 பதிவிறக்குகிறது: {details['tamil_title']} ({details['author']})")
            print(f"   இணைப்பு: {details['pdf_url'][:70]}...")

            try:
                download_file(details["pdf_url"], save_path)
                file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
                print(f"   ✅ பதிவிறக்கம் முடிந்தது! அளவு: {file_size_mb:.2f} MB")
                
                details["local_path"] = save_path
                details["file_size_mb"] = round(file_size_mb, 2)
                catalog.append(details)
                downloaded_urls.add(b_url)
                count += 1

                # Save catalog incrementally
                with open(CATALOG_PATH, "w", encoding="utf-8") as f:
                    json.dump(catalog, f, ensure_ascii=False, indent=2)

                time.sleep(1) # respectful delay
            except Exception as e:
                print(f"   ❌ பதிவிறக்கத்தில் பிழை: {e}")

    print("\n" + "=" * 65)
    print(f"🎉 மொத்தம் {count} நாவல்கள் வெற்றிகரமாகப் பதிவிறக்கப்பட்டு சேமிக்கப்பட்டன!")
    print(f"📂 கோப்பகம்: {NOVELS_DIR}")
    print("=" * 65)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10, help="Number of novels to download")
    parser.add_argument("--pages", type=int, default=3, help="Max pages to scan")
    args = parser.parse_args()

    download_batch(limit=args.limit, max_pages=args.pages)
