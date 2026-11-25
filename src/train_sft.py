"""
Stage 2: Precision Supervised Fine-Tuning (SFT) for Aadhi
Applies prompt masking (ignore_index=-100) so gradients train ONLY on the response.
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
import sentencepiece as spm

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model import AadhiTransformer

BASE_DIR = os.path.dirname(SRC_DIR)
SPM_MODEL_PATH = os.path.join(BASE_DIR, "tokenizer", "spm_tamil.model")
PRETRAINED_WEIGHTS = os.path.join(BASE_DIR, "model", "aadhi_weights_spm.pt")
SFT_WEIGHTS_PATH = os.path.join(BASE_DIR, "model", "aadhi_sft_weights.pt")
INSTRUCTIONS_PATH = os.path.join(BASE_DIR, "data", "tamil_instructions.json")

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

@torch.no_grad()
def generate_answer(model, sp, prompt_text, max_new_tokens=40, temperature=0.2):
    device = next(model.parameters()).device
    model.eval()

    formatted_prompt = f"கேள்வி: {prompt_text}\nபதில்:"
    p_ids = sp.encode_as_ids(formatted_prompt)
    idx = torch.tensor([p_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.max_seq_len:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]

        # Low temperature for precise factual recall
        if temperature < 0.1:
            next_idx = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            v, _ = torch.topk(logits, min(5, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, next_idx), dim=1)

        decoded_char = sp.decode_ids([next_idx.item()])
        if "\n" in decoded_char or "கேள்வி:" in decoded_char:
            break

    full_output = sp.decode_ids(idx[0].tolist())
    if "பதில்:" in full_output:
        return full_output.split("பதில்:")[-1].strip()
    return full_output.strip()

def train_sft(epochs=400, lr=1.5e-4, batch_size=4):
    device = get_device()
    print("=" * 65)
    print("🎯 ஆதி நிலை 2: வினா-விடை நுண்பயிற்சி (Target-Masked SFT)")
    print(f"🚀 சாதனம் (Device): {device}")
    print("=" * 65)

    if not os.path.exists(SPM_MODEL_PATH):
        raise FileNotFoundError(f"டோக்கனைசர் கிடைக்கவில்லை: {SPM_MODEL_PATH}")

    sp = spm.SentencePieceProcessor(model_file=SPM_MODEL_PATH)
    vocab_size = sp.get_piece_size()

    # Load instruction dataset
    with open(INSTRUCTIONS_PATH, "r", encoding="utf-8") as f:
        qa_data = json.load(f)
    print(f"📚 ஏற்றப்பட்ட வினா-விடைத் தொகுப்புகள்: {len(qa_data)} வினாக்கள்")

    # Format into masked training samples
    samples = []
    for item in qa_data:
        prompt = f"கேள்வி: {item['instruction'].strip()}\nபதில்:"
        response = f" {item['response'].strip()}\n"

        p_ids = sp.encode_as_ids(prompt)
        r_ids = sp.encode_as_ids(response)

        full_ids = p_ids + r_ids
        if len(full_ids) > 128:
            continue

        # Targets: -100 for prompt tokens (no loss), response token ids for response
        targets = [-100] * (len(p_ids) - 1) + r_ids + [-100]
        # Align lengths: input is full_ids[:-1], target is full_ids[1:] with prompt masked
        input_ids = full_ids[:-1]
        target_ids = targets[:len(input_ids)]

        samples.append({
            "input": input_ids,
            "target": target_ids
        })

    print(f"📝 முகமூடியிடப்பட்ட (Masked) பயிற்சி மாதிரிகள்: {len(samples)}")

    # Initialize 6.8M Model
    model = AadhiTransformer(
        vocab_size=vocab_size,
        max_seq_len=128,
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        dropout=0.05
    ).to(device)

    # Load the latest 2500-step pre-trained weights
    if os.path.exists(PRETRAINED_WEIGHTS):
        print(f"🧠 புதிய 2500-படி முன்-பயிற்சி எடைகள் ஏற்றப்பட்டன:\n   {PRETRAINED_WEIGHTS}")
        model.load_state_dict(torch.load(PRETRAINED_WEIGHTS, map_location=device))
    else:
        print("⚠️ முன்-பயிற்சி எடைக் கோப்பு இல்லை!")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print("\n🔥 வினா-விடை இலக்கு நோக்கிய நுண்பயிற்சி தொடங்குகிறது...")
    model.train()

    import random
    for step in range(1, epochs + 1):
        batch = random.choices(samples, k=batch_size)
        max_len = max(len(s["input"]) for s in batch)

        padded_x = []
        padded_y = []

        for s in batch:
            pad_len = max_len - len(s["input"])
            padded_x.append(s["input"] + [0] * pad_len)
            padded_y.append(s["target"] + [-100] * pad_len)

        x = torch.tensor(padded_x, dtype=torch.long, device=device)
        y = torch.tensor(padded_y, dtype=torch.long, device=device)

        optimizer.zero_grad()
        _, loss = model(x, targets=y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 50 == 0 or step == 1 or step == epochs:
            print(f"படி {step:3d}/{epochs} | இலக்கு இழப்பு (Response Loss): {loss.item():.4f}")

    # Save SFT Weights
    torch.save(model.state_dict(), SFT_WEIGHTS_PATH)
    print(f"\n🎉 இலக்கு நுண்பயிற்சி நிறைவடைந்தது! எடைகள் சேமிக்கப்பட்டன:\n   {SFT_WEIGHTS_PATH}")

    # Test Q&A Generation
    print("\n" + "=" * 65)
    print("🧪 ஆதி நேரடி வினா-விடை சோதனை (Live SFT Q&A Test):")
    test_questions = [
        "திருக்குறளை எழுதியவர் யார்?",
        "நீ யார்?",
        "வேள்பாரி யார்?",
        "பொன்னியின் செல்வன் நாவலை எழுதியவர் யார்?",
        "தமிழ்நாட்டின் தலைநகரம் எது?",
        "கல்லணையைக் கட்டியவர் யார்?",
        "சூரிய குடும்பத்தில் எத்தனை கோள்கள் உள்ளன?",
        "வணக்கம்"
    ]

    for q in test_questions:
        ans = generate_answer(model, sp, q, temperature=0.1)
        print(f"\n❓ கேள்வி: {q}")
        print(f"💡 ஆதி பதில்: {ans}")
    print("=" * 65)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train SFT for Aadhi model")
    parser.add_argument("--epochs", type=int, default=600, help="Training steps/epochs")
    parser.add_argument("--lr", type=float, default=1.5e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    args = parser.parse_args()

    train_sft(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
