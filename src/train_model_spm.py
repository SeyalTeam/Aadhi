"""
Pre-train Aadhi Transformer Model (Upgraded to 6.5M Parameters with Repetition Penalty)
Hardware: Apple Silicon M3 GPU (MPS)
Corpus: Thirukkural + Tamil Wikipedia + Velpari Novel (37,425 pure lines)
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import sentencepiece as spm

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model import AadhiTransformer

BASE_DIR = os.path.dirname(SRC_DIR)
DATA_PATH = os.path.join(BASE_DIR, "data", "tamil_corpus.txt")
SPM_MODEL_PATH = os.path.join(BASE_DIR, "tokenizer", "spm_tamil.model")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "model")

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

@torch.no_grad()
def generate_with_penalty(model, idx, max_new_tokens=30, temperature=0.7, top_k=20, repetition_penalty=1.3):
    """Text generation with repetition penalty to eliminate loops."""
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.max_seq_len:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]

        # Apply repetition penalty to previously generated tokens
        for token_id in set(idx[0].tolist()):
            if logits[0, token_id] > 0:
                logits[0, token_id] /= repetition_penalty
            else:
                logits[0, token_id] *= repetition_penalty

        logits = logits / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        probs = F.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_idx), dim=1)
    return idx

def train_aadhi_6m(epochs=1000, batch_size=16, seq_len=48, lr=4e-4):
    device = get_device()
    print(f"🚀 பயிற்சி தொடங்கும் வன்பொருள் சாதனம் (Device): {device}")

    # 1. Load Tokenizer
    if not os.path.exists(SPM_MODEL_PATH):
        raise FileNotFoundError(f"SentencePiece மாதிரி கிடைக்கவில்லை: {SPM_MODEL_PATH}")
    sp = spm.SentencePieceProcessor(model_file=SPM_MODEL_PATH)
    vocab_size = sp.get_piece_size()
    print(f"📖 ஆதி SentencePiece டோக்கனைசர்: Vocab Size: {vocab_size:,}")

    # 2. Tokenize Corpus
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    ids = sp.encode_as_ids(text)
    data = torch.tensor(ids, dtype=torch.long)
    print(f"📚 மொத்த தமிழ் டோக்கன்கள்: {len(data):,} (சுமார் {len(data)/100000:.1f} லட்சம்)")

    # 3. Upgrade Architecture to 6.5M Parameters
    # d_model=256, n_heads=8, n_layers=6, d_ff=1024
    model = AadhiTransformer(
        vocab_size=vocab_size,
        max_seq_len=128,
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        dropout=0.1
    ).to(device)

    params = model.get_parameter_count()
    print(f"🧠 மேம்படுத்தப்பட்ட நரம்பியல் அளவுருக்கள் (Parameters): {params:,} (~6.5 மில்லியன்!)")
    print(f"🎯 இலக்குப் படிகள் (Target Steps): {epochs}")

    # 4. Optimizer with Cosine Annealing
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # 5. Training Loop
    print("\n" + "=" * 60)
    print("🔥 6.5M ஆதி மாடல் தீவிரப் பயிற்சி (Deep Neural Training)...")
    print("=" * 60)

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
        scheduler.step()

        if step % 50 == 0 or step == 1 or step == epochs:
            current_lr = scheduler.get_last_lr()[0]
            print(f"படி {step:4d}/{epochs} | இழப்பு (Loss): {loss.item():.4f} | LR: {current_lr:.6f}")

    # 6. Save Model
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_SAVE_DIR, "aadhi_weights_spm.pt")
    torch.save(model.state_dict(), save_path)
    print(f"\n🎉 6.5M மாடல் பயிற்சி நிறைவடைந்தது! எடைகள் சேமிக்கப்பட்டன:\n   {save_path}")

    # 7. Live Generation Test with Repetition Penalty
    print("\n" + "=" * 60)
    print("🧪 ஆதி நேரடி வாக்கிய உருவாக்க சோதனை (Repetition Penalty: 1.3):")
    sample_prompts = ["வேள்பாரி", "பறம்பு மலை", "தமிழ்", "திருவள்ளுவர்"]
    model.eval()
    for prompt in sample_prompts:
        p_ids = sp.encode_as_ids(prompt)
        input_tensor = torch.tensor([p_ids], dtype=torch.long).to(device)
        gen_ids = generate_with_penalty(model, input_tensor, max_new_tokens=30, temperature=0.7, repetition_penalty=1.3)
        gen_text = sp.decode_ids(gen_ids[0].tolist())
        print(f"\nஉள்ளீடு: '{prompt}'")
        print(f"ஆதி பதில்: '{gen_text}'")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training steps")
    args = parser.parse_args()
    train_aadhi_6m(epochs=args.epochs)
