"""
Interactive Terminal Chat for Aadhi (ஆதி)
Load trained weights and interact with the model in real time.
"""

import os
import torch
from tokenizers import Tokenizer
from model import AadhiTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer", "aadhi_tokenizer.json")
WEIGHTS_PATH = os.path.join(BASE_DIR, "model", "aadhi_weights.pt")

def start_chat():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 சாதனம்: {device}")

    # 1. Load Tokenizer
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()

    # 2. Re-create Model Architecture & Load Weights
    model = AadhiTransformer(
        vocab_size=vocab_size,
        max_seq_len=64,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        dropout=0.1
    ).to(device)

    if not os.path.exists(WEIGHTS_PATH):
        print(f"⚠️ மாடல் எடைக் கோப்பு இல்லை: {WEIGHTS_PATH}. தயவுசெய்து முதலில் train_model.py இயக்கவும்.")
        return

    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()
    print("✅ ஆதி மாடல் எடைகள் வெற்றிகரமாக ஏற்றப்பட்டன!")
    print("=" * 60)
    print("தமிழ் ஏஐ 'ஆதி' நேரலை அரட்டை (விருப்பமான தமிழ் வார்த்தையை உள்ளிடவும்)")
    print("வெளியேற 'exit' அல்லது 'quit' எனத் தட்டச்சு செய்யவும்.")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nஉள்ளீடு (Prompt) > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "வெளியேறு"]:
                print("நன்றி! ஆதி அமர்வு முடிவடைந்தது.")
                break

            prompt_ids = tokenizer.encode(user_input).ids
            if len(prompt_ids) == 0:
                continue

            input_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(device)
            # Generate 20 new tokens
            generated_ids = model.generate(input_tensor, max_new_tokens=25, temperature=0.7, top_k=15)
            output_text = tokenizer.decode(generated_ids[0].tolist())

            print(f"ஆதி பதில் > {output_text}")

        except KeyboardInterrupt:
            print("\nஅமர்வு நிறுத்தப்பட்டது.")
            break

if __name__ == "__main__":
    start_chat()
