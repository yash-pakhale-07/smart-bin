# 🗑️ Smart Bin AI — Real-Time Waste Segregation System

> **Hackathon-ready** computer vision app that detects waste, classifies it into
> 4 categories, reads plastic resin codes via OCR, and gamifies the disposal
> experience with carbon points, streaks, and badges.

---

## 📁 Project Structure

```
smart-bin/
├── app.py                     # Streamlit entry point
├── requirements.txt           # Python dependencies
├── model/
│   ├── model.tflite           # Teachable Machine export (you provide)
│   └── labels.txt             # Class names: Plastic, Paper, Metal, Glass
├── modules/
│   ├── __init__.py
│   ├── detection.py           # TFLite inference
│   ├── ocr.py                 # Tesseract resin-code extraction
│   ├── gamification.py        # Points, streaks, badges
│   └── utils.py               # Overlays, colours, config
├── data/                      # Dataset staging area (gitignored)
│   ├── train/
│   │   ├── Plastic/
│   │   ├── Paper/
│   │   ├── Metal/
│   │   └── Glass/
│   └── val/
│       ├── Plastic/
│       ├── Paper/
│       ├── Metal/
│       └── Glass/
├── scripts/
│   └── prepare_dataset.py     # Dataset download + merge helper
└── README.md
```

---

## ⚙️ Prerequisites

| Dependency | Install |
|---|---|
| **Python** | 3.10 or later |
| **Tesseract OCR** | See below |
| **Kaggle API key** | `~/.kaggle/kaggle.json` |

### Install Tesseract

**Windows:**

```powershell
# Download installer from https://github.com/UB-Mannheim/tesseract/wiki
# After installing, add to PATH or set:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

**macOS:**

```bash
brew install tesseract
```

**Ubuntu / Debian:**

```bash
sudo apt update && sudo apt install -y tesseract-ocr
```

---

## 📦 Dataset Download & Preparation

### Step 1 — Download all three datasets via Kaggle CLI

```bash
pip install kaggle

# TrashNet
kaggle datasets download -d garythung/trashnet -p data/raw/trashnet --unzip

# Garbage Classification
kaggle datasets download -d asdasdasasdas/garbage-classification -p data/raw/garbage --unzip

# Waste Classification
kaggle datasets download -d techsash/waste-classification-data -p data/raw/waste --unzip
```

### Step 2 — Remap & merge classes

Run the helper script (or do manually):

```bash
python scripts/prepare_dataset.py
```

**Class mapping rules:**

| Source folder | Target class |
|---|---|
| cardboard | Paper |
| glass | Glass |
| metal | Metal |
| plastic | Plastic |
| paper | Paper |
| ORGANIC | *(skip)* |
| RECYCLABLE → sub-labels | Map per label above |

### Step 3 — Verify merged structure

```
data/train/Plastic/   —  ~3000 images
data/train/Paper/     —  ~3000 images
data/train/Metal/     —  ~1500 images
data/train/Glass/     —  ~2000 images
data/val/             —  same classes, 20% split
```

All images are resized to **224 × 224** by the prep script.

### Step 4 — Augmentation (applied by prep script)

- Random horizontal flip
- Brightness jitter ± 20 %
- Rotation ± 15°

---

## 🧠 Model Training (Google Teachable Machine)

1. Go to **[Teachable Machine](https://teachablemachine.withgoogle.com/)** →
   *Image Project* → *Standard Image Model*.
2. Create **4 classes**: `Plastic`, `Paper`, `Metal`, `Glass`.
3. Upload `data/train/Plastic/*` to the Plastic class, etc.
4. Click **Train Model** (use defaults: 50 epochs, 128 batch, 0.001 LR).
5. Click **Export Model** → **TensorFlow Lite** tab:
   - Select **Floating point** (float32).
   - Toggle **"Download model with metadata"**.
   - Download as `converted_tflite.zip`.
6. Extract:
   - `model.tflite` → `model/model.tflite`
   - `labels.txt`   → `model/labels.txt`

> **`labels.txt` must be exactly (one class per line):**
> ```
> Plastic
> Paper
> Metal
> Glass
> ```

---

## 🚀 Installation & Running

```bash
# Clone / navigate to project root
cd smart-bin

# Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS / Linux

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

The app will open at **http://localhost:8501**.

---

## 🎮 How It Works

```
📷 Webcam frame
      ↓
🧠 TFLite inference (every 5th frame)
      ↓  returns (label, confidence)
🔍 If Plastic + conf > 70% → Tesseract OCR for resin code
      ↓
👆 User selects disposal bin
      ↓
♻️ Gamification engine →  points, streak, badges
      ↓
📊 Dashboard updates in real-time
```

### Points & Badges

| Category | Points | Correct Bin |
|---|---|---|
| Plastic | +5 | 🔵 Blue |
| Paper | +3 | 🟢 Green |
| Metal | +6 | ⚪ Grey |
| Glass | +4 | 🟤 Brown |

| Points | Badge |
|---|---|
| 5+ | 🏅 Recycling Rookie |
| 10+ | 🏅 Eco Warrior |
| 20+ | 🏅 Carbon Saver |
| 50+ | 🏅 Planet Protector |

### Resin Codes (Plastic only)

| Code | Type | Recyclable | Instruction |
|---|---|---|---|
| 1 | PET | ✅ | Blue bin — curbside recyclable |
| 2 | HDPE | ✅ | Blue bin — recyclable |
| 3 | PVC | ❌ | Special disposal facility |
| 4 | LDPE | ✅ | Blue bin — check locally |
| 5 | PP | ✅ | Blue bin — recyclable |
| 6 | PS | ❌ | Avoid — special disposal |
| 7 | Other | ❌ | Special handling required |

---

## 🔧 Configuration

All tuneable params live in `modules/utils.py → CONFIG`:

```python
CONFIG = {
    "model_path": "model/model.tflite",
    "labels_path": "model/labels.txt",
    "input_size": (224, 224),
    "confidence_threshold": 0.55,
    "ocr_confidence_threshold": 0.70,
    "inference_frame_skip": 5,
}
```

---

## 🐛 Troubleshooting

| Issue | Fix |
|---|---|
| `FileNotFoundError: model.tflite` | Export from Teachable Machine and place in `model/` |
| `TesseractNotFoundError` | Install Tesseract binary and ensure it's on PATH |
| Webcam not working in browser | Use Chrome/Edge; allow camera permissions |
| Low FPS | Increase `inference_frame_skip` in `CONFIG` |
| Wrong labels | Ensure `labels.txt` matches your Teachable Machine classes exactly |

---

## 📄 License

MIT — built for hackathon & educational purposes.
