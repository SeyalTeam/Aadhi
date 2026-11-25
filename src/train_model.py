"""
Step 2: Pre-train Aadhi Transformer Model
Trains Aadhi on Tamil text corpus using Apple Silicon M3 GPU (MPS).
"""

import os
import sys
import argparse
import torch

# Ensure src/ is in Python path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model import AadhiTransformer
from tokenizers import Tokenizer

BASE_DIR = os.path.dirname(SRC_DIR)
DATA_PATH = os.path.join(BASE_DIR, "data", "tamil_corpus.txt")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer", "aadhi_tokenizer.json")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "model")

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def train_aadhi(epochs=300, batch_size=8, seq_len=32, lr=3e-4):
    device = get_device()
    print(f"🚀 பயிற்சி தொடங்கும் வன்பொருள் சாதனம் (Device): {device}")

    # 1. Load Tokenizer
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"டோக்கனைசர் கோப்பு கிடைக்கவில்லை: {TOKENIZER_PATH}")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()
    print(f"📖 ஆதி டோக்கனைசர் ஏற்றப்பட்டது. சொற்களஞ்சிய அளவு (Vocab Size): {vocab_size}")

    # 2. Load & Tokenize Corpus
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    
    encoded = tokenizer.encode(text)
    data = torch.tensor(encoded.ids, dtype=torch.long)
    print(f"📚 மொத்த தமிழ் டோக்கன்கள்: {len(data):,}")

    # 3. Initialize Aadhi Transformer
    model = AadhiTransformer(
        vocab_size=vocab_size,
        max_seq_len=64,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        dropout=0.1
    ).to(device)

    # If previous weights exist, resume/initialize properly
    save_path = os.path.join(MODEL_SAVE_DIR, "aadhi_weights.pt")

    params = model.get_parameter_count()
    print(f"🧠 ஆதி மாடல் பாராமீட்டர்கள்: {params:,}")
    print(f"🎯 மொத்த பயிற்சிப் படிகள் (Target Steps): {epochs}")

    # 4. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # 5. Training Loop
    print("\n" + "=" * 55)
    print("🔥 ஆதி ஆழ்ந்த பயிற்சி (Deep Pre-training) தொடங்குகிறது...")
    print("=" * 55)

    model.train()
    for step in range(1, epochs + 1):
        ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
        x = torch.stack([data[i : i + seq_len] for i in ix]).to(device)
        y = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix]).to(device)

        optimizer.zero_grad()
        _, loss = model(x, targets=y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 25 == 0 or step == 1 or step == epochs:
            print(f"படி {step:3d}/{epochs} | இழப்பு (Loss): {loss.item():.4f}")

    # 6. Save Trained Model
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\n🎉 300 படிகள் பயிற்சி வெற்றிகரமாக முடிந்தது! எடைகள் சேமிக்கப்பட்டன:\n   {save_path}")

    # 7. Real Test Generation
    print("\n" + "=" * 55)
    print("🧪 ஆதி நேரடி தமிழ் வாக்கிய உருவாக்க சோதனை (Live Test):")
    sample_prompts = ["ஆதி தமிழ்", "திருவள்ளுவர்", "தமிழ்நாடு"]
    for prompt in sample_prompts:
        prompt_ids = tokenizer.encode(prompt).ids
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(device)
        generated_ids = model.generate(input_tensor, max_new_tokens=20, temperature=0.7, top_k=15)
        generated_text = tokenizer.decode(generated_ids[0].tolist())
        print(f"  உள்ளீடு: '{prompt}' ➔ பதில்: '{generated_text}'")
    print("=" * 55)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300, help="Number of training steps")
    args = parser.parse_args()
    train_aadhi(epochs=args.epochs)
