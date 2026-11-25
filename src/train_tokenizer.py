"""
Step 1: Train Custom Byte-Pair Encoding (BPE) Tokenizer for Aadhi (ஆதி)
Creates a Tamil-optimized tokenizer from raw text.
"""

import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "tamil_corpus.txt")
TOKENIZER_DIR = os.path.join(BASE_DIR, "tokenizer")

def create_sample_corpus_if_empty():
    """Creates a sample high-quality Tamil corpus file if none exists."""
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    if not os.path.exists(DATA_PATH) or os.path.getsize(DATA_PATH) == 0:
        sample_texts = [
            "தமிழ் மொழி உலகில் உள்ள தொன்மையான செம்மொழிகளில் ஒன்றாகும்.",
            "ஆதி என்பது எல்லாவற்றிற்கும் தொடக்கமாகவும் மூலமாகவும் விளங்கும் சொல்லாகும்.",
            "செயற்கை நுண்ணறிவு தொழில்நுட்பம் மனித குலத்தின் வளர்ச்சிக்கு பல துறைகளில் உதவுகிறது.",
            "கல்வி, மருத்துவம், அறிவியல் மற்றும் செய்தி ஊடகங்களில் தமிழ் மொழி ஏஐ மாதிரிகள் பெரும் மாற்றத்தை ஏற்படுத்தும்.",
            "கணினியில் தமிழ் எழுத்துக்களைச் சரியாகப் புரிந்து கொள்ள பிரத்யேக டோக்கனைசர் மிக அவசியமானது.",
            "உயிர் எழுத்துக்களும் மெய் எழுத்துக்களும் இணைந்து உயிர்மெய் எழுத்துக்களாக உருவெடுக்கின்றன.",
            "தமிழ் இலக்கியங்களில் சங்க இலக்கியம், திருக்குறள், சிலப்பதிகாரம் ஆகியவை மிகச் சிறப்பு வாய்ந்தவை.",
            "அகர முதல எழுத்தெல்லாம் ஆதி பகவன் முதற்றே உலகு.",
            "யாதும் ஊரே யாவரும் கேளிர் என்பது தமிழரின் உலகளாவிய வாழ்வியல் தத்துவமாகும்.",
            "செய்தி அறைகளில் ஏஐ உதவியாளர் மூலம் செய்திகளை விரைவாகத் தொகுத்து வழங்க முடியும்."
        ] * 100  # Repeat to give sufficient tokens for initial training
        with open(DATA_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(sample_texts))
        print(f"✅ மாதிரி தமிழ் தரவுக் கோப்பு உருவாக்கப்பட்டது: {DATA_PATH}")

def train_aadhi_tokenizer(vocab_size=8000):
    """Trains a BPE tokenizer tailored for Tamil script."""
    print("🚀 ஆதி (Aadhi) தமிழ் டோக்கனைசர் பயிற்சி தொடங்குகிறது...")
    create_sample_corpus_if_empty()

    # 1. Initialize BPE Tokenizer with ByteLevel (Agaram style - keeps Tamil bytes unbroken)
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    # 2. Define Special Tokens
    special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens,
        show_progress=True
    )

    # 3. Train from corpus files
    tokenizer.train(files=[DATA_PATH], trainer=trainer)

    # 4. Save tokenizer
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    tokenizer_file = os.path.join(TOKENIZER_DIR, "aadhi_tokenizer.json")
    tokenizer.save(tokenizer_file)
    print(f"🎉 டோக்கனைசர் வெற்றிகரமாக உருவாக்கப்பட்டு சேமிக்கப்பட்டது: {tokenizer_file}")

    # 5. Test tokenization on sample sentence
    test_sentence = "ஆதி தமிழ் ஏஐ மாடல் மிக வேகமாக இயங்குகிறது."
    encoded = tokenizer.encode(test_sentence)
    print("\n" + "=" * 50)
    print(f"மாதிரி சோதனை வாக்கியம்: '{test_sentence}'")
    print(f"டோக்கன்கள்: {encoded.tokens}")
    print(f"டோக்கன் ஐடிகள் (IDs): {encoded.ids}")
    print(f"மொத்த டோக்கன்களின் எண்ணிக்கை: {len(encoded.tokens)}")
    print("=" * 50)

if __name__ == "__main__":
    train_aadhi_tokenizer()
