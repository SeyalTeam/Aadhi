# 🌟 ஆதி (Aadhi) - Native Tamil LLM from Scratch

> **"ஆதி" (Aadhi)**: எல்லாவற்றிற்கும் தொடக்கம் / மூலம்.  
> தமிழ் மொழிக்காகவே பிரத்யேகமாக ஸ்கிராட்சிலிருந்து (From Scratch) வடிவமைக்கப்பட்ட ஒரு தன்னாட்சி நரம்பியல் மொழி மாதிரி (Autonomous Tamil Neural Language Model).

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hardware](https://img.shields.io/badge/Hardware-Apple%20Silicon%20M3%20(MPS)-silver?logo=apple)](https://developer.apple.com/metal/pytorch/)
[![Tokenizer](https://img.shields.io/badge/Tokenizer-SentencePiece%20(8k%20Vocab)-blue)](https://github.com/google/sentencepiece)
[![Parameters](https://img.shields.io/badge/Parameters-6.8M-success)]()

---

## 📌 சிறப்பம்சங்கள் (Key Highlights)

- 🚫 **பூஜ்ஜிய கிளவுட் செலவு (100% Offline & Private):** எந்தவொரு கட்டண கிளவுட் API (OpenAI, Claude, etc.) சார்ந்தது அல்ல. ஆப்பிள் சிலிகான் M3 சிப்பின் GPU-வில் (`mps`) முழுமையாக இயங்குகிறது.
- 🔤 **சுத்தமான தமிழ் டோக்கனைசர் (SentencePiece 8k Vocab):** தமிழ் எழுத்துக்கள் (உயிர்மெய் எழுத்துக்கள்) உடைந்துபோவதைத் தடுத்து, ஒவ்வொரு சொல்லையும் சீராகக் கையாளும் பிரத்யேக சொற்களஞ்சியம்.
- 📚 **54.7 லட்சம் சொற்கள் (5.47M Tokens):** வேள்பாரி, பொன்னியின் செல்வன், சிவகாமியின் சபதம், திருக்குறள், சுஜாதா உட்பட 61-க்கும் மேற்பட்ட தமிழ் நூல்களிலிருந்து திரட்டப்பட்ட இலக்கியத் தளம்.
- 🎯 **இருநிலை பயிற்சி முறை (Two-Stage Pipeline):**
  1. **படி 1: அடிப்படை மொழிப் பயிற்சி (Pre-training):** அடுத்தடுத்த சொற்களைக் கணிக்கும் 6.8M டிரான்ஸ்பார்மர் பயிற்சி (இழப்பு மதிப்பு 9.03 ➔ 5.76).
  2. **படி 2: இலக்கு நோக்கிய வழிகாட்டல் பயிற்சி (SFT):** கேள்வி-பதில் பாணியிலான நுண்பயிற்சி (Target Masking உடன் இழப்பு மதிப்பு 6.85 ➔ **0.0427**).
- 🔍 **உள்ளூர் RAG தேடுபொறி (Local RAG Vector Search):** 61 தமிழ் நாவல்களின் **16,666 பத்திகளை** sub-word n-gram அடிப்படையில் மில்லிவினாடிகளில் தேடி, சரியான ஆதார நூலின் மேற்கோளுடன் பதிலளிக்கும் வசதி.
- 💬 **நேரடி வலைத்தள அரட்டை (Interactive Web UI):** உலாவி வழியே மாடலுடன் உரையாடக்கூடிய நவீன சாட்பாட் இடைமுகம் (`http://localhost:8080`) + RAG மேற்கோள் அட்டை (Citation Card).

---

## 🏗️ மாதிரி கட்டமைப்பு (Model Architecture)

| அளவுரு (Hyperparameter) | மதிப்பு (Value) | விளக்கம் |
| :--- | :--- | :--- |
| **மொத்த அளவுருக்கள் (Parameters)** | **6,813,696 (~6.8 மில்லியன்)** | இலகுவான மற்றும் வேகமான உள்ளூர் மொழி மாதிரி |
| **சொற்களஞ்சிய அளவு (Vocab Size)** | **8,000** | SentencePiece BPE (தூய தமிழ்) |
| **அடுக்குகள் (Layers)** | **6** | Multi-Head Self-Attention Blocks |
| **கவன ஈர்ப்புத் தலைகள் (Heads)** | **8** | 32 பரிமாணங்கள் / தலை |
| **உட்பொதிவு பரிமாணம் (d_model)** | **256** | Hidden Representation Size |
| **உட்கட்டமைப்பு பரிமாணம் (d_ff)** | **1024** | Feed-Forward Expansion (4x) |
| **தொடர் நீளம் (Max Context Length)** | **128 Tokens** | வாக்கியச் சூழல் நினைவகம் |

---

## 📂 கோப்பக அமைப்பு (Repository Structure)

```
Aadhi/
├── data/
│   ├── tamil_corpus.txt          # 54.7 லட்சம் சொற்கள் கொண்ட தூய தமிழ் உரைத்தளம் (48 MB)
│   ├── tamil_instructions.json   # வழிகாட்டல் பயிற்சிக்கான 120+ கேள்வி-பதில் தரவுத்தளம்
│   └── novels_catalog.json       # இலக்கிய நூல்களின் பட்டியல்
├── model/
│   ├── aadhi_weights_spm.pt      # அடிப்படை முன்-பயிற்சி எடைகள் (2500 படிகள்)
│   └── aadhi_sft_weights.pt      # வழிகாட்டல் நுண்பயிற்சி எடைகள் (SFT Checkpoint)
├── tokenizer/
│   ├── spm_tamil.model           # SentencePiece பயிற்சி பெற்ற பைனரி மாடல்
│   └── spm_tamil.vocab           # சொற்களஞ்சியப் பட்டியல்
├── src/
│   ├── model.py                  # PyTorch Transformer கட்டமைப்பு
│   ├── train_model_spm.py        # அடிப்படை முன்-பயிற்சி நிரல் (Base Pre-training)
│   ├── train_sft.py              # வினா-விடை வழிகாட்டல் பயிற்சி (Target-Masked SFT)
│   ├── chat_spm.py               # டெர்மினல் நேரடி அரட்டை
│   ├── web_chat.py               # போர்ட் 8080 வலைத்தள அரட்டை சர்வர் + RAG (HTML/JS)
│   ├── expand_instructions.py    # பல்துறை வினா-விடை உருவாக்க நிரல்
│   └── rag/
│       ├── indexer.py            # 61 நாவல்களை வெக்டார் குறியீடாக்கும் நிரல்
│       └── retriever.py          # அதிவேக தமிழ் RAG தேடுபொறி
├── PROJECT_REPORT.md             # முழுமையான விரிவான தொழில்நுட்ப அறிக்கை
├── requirements.txt              # பைதான் நூலகங்கள்
└── README.md
```

---

## 🚀 தொடங்குவது எப்படி? (Quick Start)

### 1. சூழலை அமைத்தல் (Environment Setup)
```bash
# களஞ்சியத்தை க்ளோன் செய்து உள்ளே செல்லவும்
git clone https://github.com/SeyalTeam/Aadhi.git
cd Aadhi

# மெய்நிகர் சூழலை உருவாக்கவும்
python3 -m venv venv
source venv/bin/activate

# சார்புகளை நிறுவவும்
pip install -r requirements.txt
```

### 2. வலைத்தள அரட்டையை இயக்குதல் (Run Web Chat UI)
```bash
python3 src/web_chat.py 8080
```
இப்போது உலாவியில் `http://localhost:8080` முகவரிக்குச் சென்று ஆதியுடன் நேரடியாகத் தமிழில் உரையாடலாம்!

### 3. டெர்மினலில் அரட்டையடித்தல் (Terminal CLI Chat)
```bash
python3 src/chat_spm.py
```

---

## 🧪 நேரடி மாதிரி பதில்கள் (Live Demonstrations)

```
❓ கேள்வி: திருக்குறளை எழுதியவர் யார்?
💡 ஆதி: திருக்குறளை எழுதியவர் தெய்வப்புலவர் திருவள்ளுவர் ஆவார்.

❓ கேள்வி: தமிழ்நாட்டின் தலைநகரம் எது?
💡 ஆதி: தமிழ்நாட்டின் தலைநகரம் சென்னை ஆகும்.

❓ கேள்வி: சென்னை எந்த மாநிலத்தில் உள்ளது?
💡 ஆதி: சென்னை தமிழ்நாடு மாநிலத்தில் அமைந்துள்ளது.

❓ கேள்வி: சூரிய குடும்பத்தில் எத்தனை கோள்கள் உள்ளன?
💡 ஆதி: சூரிய குடும்பத்தில் புதன், வெள்ளி, பூமி, செவ்வாய், வியாழன், சனி, யுரேனஸ், நெப்டியூன் ஆகிய 8 கோள்கள் உள்ளன.
```

---

## 📄 உரிமம் & பங்களிப்பு (License & Acknowledgements)
- உருவாக்கப்பட்டது: **Seyal Team**
- வன்பொருள்: **Apple Silicon M3 (MPS acceleration)**
- தமிழ் இலக்கியத் தரவுகள்: **Project Madurai & Classical Tamil Works**
