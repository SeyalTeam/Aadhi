"""
Aadhi (ஆதி) Local Web Chat Server - Upgraded with High-Precision SFT
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
import sentencepiece as spm
from http.server import HTTPServer, BaseHTTPRequestHandler

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model import AadhiTransformer

BASE_DIR = os.path.dirname(SRC_DIR)
SPM_MODEL_PATH = os.path.join(BASE_DIR, "tokenizer", "spm_tamil.model")
SFT_WEIGHTS = os.path.join(BASE_DIR, "model", "aadhi_sft_weights.pt")
PRETRAINED_WEIGHTS = os.path.join(BASE_DIR, "model", "aadhi_weights_spm.pt")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print(f"🚀 சாதனம் (Device): {DEVICE}")
sp = spm.SentencePieceProcessor(model_file=SPM_MODEL_PATH)
vocab_size = sp.get_piece_size()

model = AadhiTransformer(
    vocab_size=vocab_size,
    max_seq_len=128,
    d_model=256,
    n_heads=8,
    n_layers=6,
    d_ff=1024,
    dropout=0.0
).to(DEVICE)

active_weights = SFT_WEIGHTS if os.path.exists(SFT_WEIGHTS) else PRETRAINED_WEIGHTS
model.load_state_dict(torch.load(active_weights, map_location=DEVICE))
model.eval()
print(f"✅ ஆதி மாடல் எடைகள் ஏற்றப்பட்டன: {os.path.basename(active_weights)}")

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ta">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ஆதி (Aadhi) - தமிழ் செயற்கை நுண்ணறிவு அரட்டை</title>
    <link href="https://fonts.googleapis.com/css2?family=Mukta+Malar:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #d97706;
            --bg: #0f172a;
            --surface: #1e293b;
            --surface-card: #334155;
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --border: #475569;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Mukta Malar', system-ui, sans-serif;
            background: var(--bg);
            color: var(--text);
            display: flex;
            height: 100vh;
            overflow: hidden;
        }
        .sidebar {
            width: 330px;
            background: var(--surface);
            border-right: 1px solid var(--border);
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .brand { display: flex; align-items: center; gap: 12px; }
        .brand-icon {
            width: 44px; height: 44px;
            background: linear-gradient(135deg, #f59e0b, #b45309);
            border-radius: 12px;
            display: flex; align-items: center; justify-content: center;
            font-size: 24px; font-weight: bold; color: white;
            box-shadow: 0 4px 12px rgba(217, 119, 6, 0.4);
        }
        .brand h1 { font-size: 20px; font-weight: 700; color: #fbbf24; }
        .brand p { font-size: 12px; color: var(--text-muted); }
        .meta-card {
            background: var(--surface-card);
            padding: 16px;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.08);
            font-size: 13px;
            line-height: 1.6;
        }
        .meta-card div { display: flex; justify-content: space-between; margin-bottom: 6px; }
        .meta-card span:first-child { color: var(--text-muted); }
        .meta-card span:last-child { font-weight: 600; color: #38bdf8; }
        .quick-prompts { display: flex; flex-direction: column; gap: 8px; }
        .quick-btn {
            background: var(--surface-card);
            border: 1px solid var(--border);
            color: var(--text);
            padding: 9px 12px;
            border-radius: 8px;
            font-size: 12px;
            cursor: pointer;
            text-align: left;
            transition: all 0.2s;
            font-family: inherit;
        }
        .quick-btn:hover { background: #475569; border-color: #fbbf24; }
        .chat-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: #0b1120;
        }
        .chat-header {
            padding: 16px 24px;
            background: var(--surface);
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .chat-messages {
            flex: 1;
            padding: 24px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .msg {
            max-width: 75%;
            padding: 14px 18px;
            border-radius: 14px;
            font-size: 15px;
            line-height: 1.6;
        }
        .msg-user {
            align-self: flex-end;
            background: linear-gradient(135deg, #d97706, #b45309);
            color: white;
            border-bottom-right-radius: 2px;
        }
        .msg-bot {
            align-self: flex-start;
            background: var(--surface);
            border: 1px solid var(--border);
            color: var(--text);
            border-bottom-left-radius: 2px;
        }
        .input-bar {
            padding: 16px 24px;
            background: var(--surface);
            border-top: 1px solid var(--border);
            display: flex;
            gap: 12px;
        }
        .input-bar input {
            flex: 1;
            padding: 14px 18px;
            background: var(--surface-card);
            border: 1px solid var(--border);
            border-radius: 10px;
            color: white;
            font-size: 15px;
            font-family: inherit;
            outline: none;
        }
        .input-bar input:focus { border-color: #fbbf24; }
        .send-btn {
            background: linear-gradient(135deg, #f59e0b, #d97706);
            color: white;
            border: none;
            padding: 0 24px;
            border-radius: 10px;
            font-weight: 600;
            font-size: 15px;
            cursor: pointer;
            font-family: inherit;
        }
    </style>
</head>
<body>
    <div class="sidebar">
        <div class="brand">
            <div class="brand-icon">ஆ</div>
            <div>
                <h1>ஆதி (Aadhi AI)</h1>
                <p>சுயசார்பு தமிழ் மொழி மாதிரி</p>
            </div>
        </div>

        <div class="meta-card">
            <div><span>நரம்பியல் அளவுருக்கள்:</span> <span>6.8 மில்லியன்</span></div>
            <div><span>கட்டமைப்பு:</span> <span>6 Layers / 8 Heads</span></div>
            <div><span>வன்பொருள்:</span> <span>Apple M3 (MPS)</span></div>
            <div><span>பயிற்சித் தரவு:</span> <span>54.7 லட்சம் Tokens</span></div>
            <div><span>நூல்கள்:</span> <span>61 நூல்கள் (நாவல்கள் & காவியங்கள்)</span></div>
            <div><span>நிலை:</span> <span style="color:#22c55e;">SFT இலக்கு நுண்பயிற்சி</span></div>
        </div>

        <div>
            <p style="font-size: 12px; color: var(--text-muted); margin-bottom: 8px;">துரித வினாக்கள் (Quick Questions):</p>
            <div class="quick-prompts">
                <button class="quick-btn" onclick="ask('திருக்குறளை எழுதியவர் யார்?')">📜 திருக்குறளை எழுதியவர் யார்?</button>
                <button class="quick-btn" onclick="ask('நீ யார்?')">❓ நீ யார்?</button>
                <button class="quick-btn" onclick="ask('வேள்பாரி யார்?')">🏹 வேள்பாரி யார்?</button>
                <button class="quick-btn" onclick="ask('பொன்னியின் செல்வன் நாவலை எழுதியவர் யார்?')">👑 பொன்னியின் செல்வன் ஆசிரியர்?</button>
                <button class="quick-btn" onclick="ask('வணக்கம்')">🙏 வணக்கம்</button>
            </div>
        </div>
    </div>

    <div class="chat-container">
        <div class="chat-header">
            <div>
                <h2 style="font-size: 16px; font-weight: 600;">நேரலை வினா-விடை அரங்கு (Live Q&A)</h2>
                <p style="font-size: 12px; color: var(--text-muted);">புதிய SFT நுண்பயிற்சி எடைகள் (Loss: 0.69) மூலம் இயங்குகிறது</p>
            </div>
        </div>

        <div class="chat-messages" id="messages">
            <div class="msg msg-bot">
                வணக்கம்! நான் <strong>ஆதி</strong>. 61 தமிழ் நூல்கள் மற்றும் திருக்குறள் தரவுகளில் பயிற்றுவிக்கப்பட்ட தன்னாட்சி தமிழ் மொழி மாதிரி. உங்களுடைய வினாக்களை என்னிடம் கேட்கலாம்!
            </div>
        </div>

        <div class="input-bar">
            <input type="text" id="userInput" placeholder="உங்கள் வினாவை உள்ளிடவும் (எ.கா: திருக்குறளை எழுதியவர் யார்?)..." onkeydown="if(event.key==='Enter') sendMessage()">
            <button class="send-btn" id="sendBtn" onclick="sendMessage()">அனுப்பு</button>
        </div>
    </div>

    <script>
        function ask(text) {
            document.getElementById('userInput').value = text;
            sendMessage();
        }

        async function sendMessage() {
            const input = document.getElementById('userInput');
            const q = input.value.trim();
            if (!q) return;

            input.value = '';
            appendMsg(q, 'msg-user');

            const btn = document.getElementById('sendBtn');
            btn.disabled = true;
            btn.innerText = 'சிந்திக்கிறது...';

            try {
                const res = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({prompt: q})
                });
                const data = await res.json();
                appendMsg(data.response, 'msg-bot');
            } catch (e) {
                appendMsg('மன்னிக்கவும், பிழை ஏற்பட்டது.', 'msg-bot');
            } finally {
                btn.disabled = false;
                btn.innerText = 'அனுப்பு';
            }
        }

        function appendMsg(text, cls) {
            const container = document.getElementById('messages');
            const d = document.createElement('div');
            d.className = 'msg ' + cls;
            d.innerText = text;
            container.appendChild(d);
            container.scrollTop = container.scrollHeight;
        }
    </script>
</body>
</html>
"""

@torch.no_grad()
def generate_ai(user_query, max_tokens=64):
    prompt_text = f"கேள்வி: {user_query}\nபதில்:"
    p_ids = sp.encode_as_ids(prompt_text)
    if not p_ids:
        return "உள்ளீடு கிடைக்கவில்லை."

    idx = torch.tensor([p_ids], dtype=torch.long, device=DEVICE)

    for _ in range(max_tokens):
        idx_cond = idx[:, -model.max_seq_len:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]

        # Precise greedy / low temp top-3 sampling
        logits = logits / 0.15
        v, _ = torch.topk(logits, min(3, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("Inf")

        probs = F.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_idx), dim=1)

        char = sp.decode_ids([next_idx.item()])
        if "\n" in char or "கேள்வி:" in char:
            break

    full_output = sp.decode_ids(idx[0].tolist())
    if "பதில்:" in full_output:
        ans = full_output.split("பதில்:")[-1].strip()
        # Clean repetitive trailing sentences
        if "." in ans:
            sentences = [s.strip() for s in ans.split(".") if s.strip()]
            if sentences:
                ans = sentences[0] + "."
        return ans
    return full_output.strip()

class ChatHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/chat":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/chat":
            content_length = int(self.headers.get("Content-Length", 0))
            post_body = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(post_body)
                prompt = data.get("prompt", "")
                ans = generate_ai(prompt)
                resp = json.dumps({"response": ans}, ensure_ascii=False)
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(resp.encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

def run_server(port=8080):
    server_address = ("", port)
    httpd = HTTPServer(server_address, ChatHandler)
    print("=" * 65)
    print(f"🌟 ஆதி வலைத்தள அரட்டை சர்வர் துவங்கப்பட்டது!")
    print(f"👉 உலாவி முகவரி: http://localhost:{port}")
    print("=" * 65)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nசர்வர் நிறுத்தப்பட்டது.")

if __name__ == "__main__":
    port = 8080
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    run_server(port)
