"""
Extract clean Tamil text from PDF books in data/pdfs/
and append them to data/tamil_corpus.txt.
"""

import os
import re
from pypdf import PdfReader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_DIR = os.path.join(BASE_DIR, "data", "pdfs")
DATA_PATH = os.path.join(BASE_DIR, "data", "tamil_corpus.txt")

def clean_tamil_line(line: str) -> str:
    """Removes stray formatting/junk while preserving valid Tamil text."""
    line = line.strip()
    # Filter out page number only lines
    if line.isdigit():
        return ""
    if len(line) < 3:
        return ""
    return line

def extract_all_pdfs():
    print("📖 'data/pdfs/' கோப்புறையில் உள்ள PDF புத்தகங்கள் ஆய்வு செய்யப்படுகின்றன...")
    if not os.path.exists(PDF_DIR):
        print(f"⚠️ PDF கோப்புறை இல்லை: {PDF_DIR}")
        return

    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("⚠️ எந்த PDF கோப்பும் காணப்படவில்லை.")
        return

    total_extracted_pages = 0
    total_extracted_lines = 0

    for pdf_name in pdf_files:
        pdf_path = os.path.join(PDF_DIR, pdf_name)
        print(f"\n📂 '{pdf_name}' கோப்பிலிருந்து உரை பிரித்தெடுக்கப்படுகிறது...")
        try:
            reader = PdfReader(pdf_path)
            num_pages = len(reader.pages)
            print(f"  மொத்தப் பக்கங்கள்: {num_pages}")

            extracted_text_blocks = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if not text:
                    continue
                lines = [clean_tamil_line(l) for l in text.split("\n")]
                clean_lines = [l for l in lines if l]
                if clean_lines:
                    extracted_text_blocks.append("\n".join(clean_lines))
                    total_extracted_lines += len(clean_lines)

                total_extracted_pages += 1
                if (i + 1) % 50 == 0 or (i + 1) == num_pages:
                    print(f"  பக்கம் {i + 1}/{num_pages} பிரித்தெடுக்கப்பட்டது...")

            if extracted_text_blocks:
                header = f"\n\n### நூல்: {pdf_name} ###\n"
                full_extracted = header + "\n\n".join(extracted_text_blocks) + "\n"
                with open(DATA_PATH, "a", encoding="utf-8") as f:
                    f.write(full_extracted)
                print(f"✅ '{pdf_name}' உரை வெற்றிகரமாக 'tamil_corpus.txt' கோப்பில் சேர்க்கப்பட்டது!")

        except Exception as e:
            print(f"❌ பிழை ({pdf_name}): {e}")

    print("\n" + "=" * 60)
    print(f"🎉 PDF பிரித்தெடுத்தல் நிறைவுற்றது!")
    print(f"📄 செயலாக்கப்பட்ட பக்கங்கள்: {total_extracted_pages}")
    print(f"📝 புதிதாகச் சேர்க்கப்பட்ட தமிழ் வரிகள்: {total_extracted_lines:,}")
    print("=" * 60)

if __name__ == "__main__":
    extract_all_pdfs()
