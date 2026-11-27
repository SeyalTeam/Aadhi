"""
Interactive Terminal Chat for Aadhi (ஆதி 6.8M Conversational Model)
Supports both Foundational text completion and Instruction Fine-Tuned (SFT) Q&A.
"""

import os
import sys
import torch
import torch.nn.functional as F
import sentencepiece as spm

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model import AadhiTransformer

BASE_DIR = os.path.dirname(SRC_DIR)
SPM_MODEL_PATH = os.path.join(BASE_DIR, "tokenizer", "spm_tamil.model")
SFT_WEIGHTS = os.path.join(BASE_DIR, "model", "aadhi_sft_weights.pt")
PRETRAINED_WEIGHTS = os.path.join(BASE_DIR, "model", "aadhi_weights_spm.pt")

try:
    from rag.retriever import AadhiRAGRetriever
    rag_retriever = AadhiRAGRetriever.get_instance()
except Exception:
    rag_retriever = None

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

@torch.no_grad()
def generate_response(model, sp, user_query, is_sft=False, max_new_tokens=64, temperature=None, repetition_penalty=1.1):
    device = next(model.parameters()).device
    model.eval()

    if temperature is None:
        temperature = 0.2 if is_sft else 0.65

    if is_sft:
        prompt_text = f"கேள்வி: {user_query}\nபதில்:"
    else:
        prompt_text = user_query

    prompt_ids = sp.encode_as_ids(prompt_text)
    if not prompt_ids:
        return ""

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.max_seq_len:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]

        # Repetition penalty
        for token_id in set(idx[0].tolist()):
            if logits[0, token_id] > 0:
                logits[0, token_id] /= repetition_penalty
            else:
                logits[0, token_id] *= repetition_penalty

        logits = logits / max(temperature, 0.05)
        v, _ = torch.topk(logits, min(5, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("Inf")

        probs = F.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_idx), dim=1)

        decoded_char = sp.decode_ids([next_idx.item()])
        if is_sft and ("\n" in decoded_char or "கேள்வி:" in decoded_char):
            break

    full_output = sp.decode_ids(idx[0].tolist())
    if is_sft and "பதில்:" in full_output:
        ans = full_output.split("பதில்:")[-1].strip()
        if "." in ans:
            sentences = [s.strip() for s in ans.split(".") if s.strip()]
            if sentences:
                ans = sentences[0] + "."
        return ans
    return full_output.strip()

def start_chat():
    device = get_device()
    print(f"🚀 சாதனம் (Device): {device}")

    if not os.path.exists(SPM_MODEL_PATH):
        print(f"❌ டோக்கனைசர் கோப்பு இல்லை: {SPM_MODEL_PATH}")
        return

    sp = spm.SentencePieceProcessor(model_file=SPM_MODEL_PATH)
    vocab_size = sp.get_piece_size()

    # Determine which weights to load (prefer SFT if available)
    if os.path.exists(SFT_WEIGHTS):
        weights_path = SFT_WEIGHTS
        is_sft = True
        mode_str = "வினா-விடை உரையாடல் முறை (Instruction-Tuned Q&A Mode)"
    elif os.path.exists(PRETRAINED_WEIGHTS):
        weights_path = PRETRAINED_WEIGHTS
        is_sft = False
        mode_str = "அடிப்படை வாக்கிய தொடர்ச்சி முறை (Foundational Completion Mode)"
    else:
        print("❌ மாடல் எடைக் கோப்பு எதுவும் கிடைக்கவில்லை.")
        return

    # Model architecture
    model = AadhiTransformer(
        vocab_size=vocab_size,
        max_seq_len=128,
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        dropout=0.0
    ).to(device)

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    params = model.get_parameter_count()

    print("=" * 65)
    print("🌟 'ஆதி' (Aadhi) நேரலைத் தமிழ் உரையாடல் அரங்கு")
    print(f"🧠 மாடல்: 6.8M அளவுருக்கள் ({params:,} Parameters)")
    print(f"🎯 இயங்கும் பயன்முறை: {mode_str}")
    print("வெளியேற 'exit', 'quit' அல்லது 'வெளியேறு' எனத் தட்டச்சு செய்யவும்.")
    print("=" * 65)

    while True:
        try:
            user_input = input("\n👤 நீங்கள் (Prompt) > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "வெளியேறு", "நிறுத்து"]:
                print("நன்றி! ஆதி உரையாடல் அமர்வு நிறைவுற்றது.")
                break

            citation = None
            if rag_retriever:
                try:
                    results = rag_retriever.retrieve(user_input, top_k=1, min_score=0.05)
                    if results:
                        citation = results[0]
                except Exception:
                    pass

            response = generate_response(model, sp, user_input, is_sft=is_sft)
            print(f"🤖 ஆதி > {response}")
            if citation:
                print(f"   📖 [ஆதார நூல்]: {citation['title']} (ஆசிரியர்: {citation['author']}) | பொருத்தம்: {int(citation['score']*100)}%")

        except KeyboardInterrupt:
            print("\nஅமர்வு நிறுத்தப்பட்டது.")
            break

if __name__ == "__main__":
    start_chat()
