"""
Train SentencePiece BPE Tokenizer for Project Aadhi (ஆதி)
Trained on Thirukkural, Tamil Wikipedia, and 1,313 pages of Velpari novel!
"""

import os
import sentencepiece as spm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "tamil_corpus.txt")
TOKENIZER_DIR = os.path.join(BASE_DIR, "tokenizer")
MODEL_PREFIX = os.path.join(TOKENIZER_DIR, "spm_tamil")

def train_spm():
    print("🚀 ஆதி (Aadhi) SentencePiece முழு-சொல் டோக்கனைசர் பயிற்சி தொடங்குகிறது...")
    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"தரவுக் கோப்பு இல்லை: {DATA_PATH}")

    # Vocabulary size 8,000 for rich Tamil literature coverage
    vocab_size = 8000
    print(f"📚 தரவுக் கோப்பு: {DATA_PATH}")
    print(f"🎯 சொற்களஞ்சிய இலக்கு (Target Vocab Size): {vocab_size}")

    spm.SentencePieceTrainer.train(
        input=DATA_PATH,
        model_prefix=MODEL_PREFIX,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        user_defined_symbols=["<s>", "</s>", "<pad>"],
        max_sentence_length=8192
    )

    print("\n🎉 SentencePiece டோக்கனைசர் வெற்றிகரமாக உருவாக்கப்பட்டது!")
    print(f"📁 மாதிரி கோப்பு: {MODEL_PREFIX}.model")
    print(f"📖 சொற்களஞ்சியம்: {MODEL_PREFIX}.vocab")

    # Test tokenization
    sp = spm.SentencePieceProcessor(model_file=f"{MODEL_PREFIX}.model")
    test_sentences = [
        "ஆதி தமிழ் ஏஐ மாடல் மிக வேகமாக இயங்குகிறது.",
        "வேள்பாரி பறம்பு மலையின் தலைவன் ஆவான்.",
        "அகர முதல எழுத்தெல்லாம் ஆதி பகவன் முதற்றே உலகு."
    ]

    print("\n" + "=" * 60)
    print("🧪 புதிய முழு-சொல் டோக்கனைசர் நேரடி சோதனை:")
    for s in test_sentences:
        pieces = sp.encode_as_pieces(s)
        ids = sp.encode_as_ids(s)
        print(f"\nவாக்கியம்: '{s}'")
        print(f"டோக்கன்கள்: {pieces}")
        print(f"எண்ணிக்கை: {len(pieces)} டோக்கன்கள் (அடையாளங்கள்: {ids[:5]}...)")
    print("=" * 60)

if __name__ == "__main__":
    train_spm()
