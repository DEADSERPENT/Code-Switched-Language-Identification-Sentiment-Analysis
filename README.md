# CoSwitchNLP

Token-level language identification and sentiment analysis for code-switched Hinglish text.

Paste a Hinglish sentence → get each word highlighted by language (Hindi / English / Mixed) + overall sentiment with confidence scores.

**Stack:** XLM-RoBERTa-base · PyTorch · FastAPI · React + Vite + Tailwind

---

## Quick Start

### 1. Backend setup

```bash
cd backend
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get the dataset

Download **SemEval-2020 Task 9 SentiMix** (Hinglish):

```bash
# Option A — from HuggingFace (automatic, recommended)
python - <<'EOF'
from datasets import load_dataset
import os, json

ds = load_dataset("lince", "sa_hineng")   # or search HuggingFace for SentiMix

# The LinCE sa_hineng split is the closest publicly available version.
# For the full SentiMix dataset, download from:
#   https://github.com/keshav22bansal/BAKSA_IITK  (includes train/val/test CoNLL files)
# Place files as:  data/sentimix/train.txt  val.txt  test.txt
EOF

# Option B — manual download from the SemEval repo
# 1. Go to: https://github.com/keshav22bansal/BAKSA_IITK
# 2. Download Hinglish_train.txt, Hinglish_dev.txt, Hinglish_test.txt
# 3. Rename and place in data/sentimix/
mkdir -p data/sentimix
# cp Hinglish_train.txt data/sentimix/train.txt
# cp Hinglish_dev.txt   data/sentimix/val.txt
# cp Hinglish_test.txt  data/sentimix/test.txt
```

### 3. Train the model (~1–2 hours on RTX A2000)

```bash
cd backend
source venv/bin/activate
python train.py \
  --train_file ../data/sentimix/train.txt \
  --val_file   ../data/sentimix/val.txt \
  --output_dir ../models/coswitchnlp_v1 \
  --epochs 5 \
  --batch_size 16
```

Training logs metrics every 50 steps and saves the best checkpoint automatically.

### 4. Start the API server

```bash
cd backend
source venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Check it's alive: http://localhost:8000/health

### 5. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Open: http://localhost:5173

---

## Project Structure

```
coswitchnlp/
├── ROADMAP.md              Project vision and milestones
├── README.md               This file
├── backend/
│   ├── model.py            CoSwitchModel (multi-task XLM-RoBERTa)
│   ├── dataset.py          SentiMix dataset loader + collator
│   ├── train.py            Training loop (AMP + early stopping)
│   ├── inference.py        Prediction pipeline + word alignment
│   ├── app.py              FastAPI server
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── App.jsx
│       └── components/
│           ├── TextInput.jsx       textarea + demo examples
│           ├── TokenDisplay.jsx    color-coded token grid
│           ├── SentimentGauge.jsx  emoji + score bars
│           ├── LanguageStats.jsx   pie chart + CMI gauge
│           └── AnalysisPanel.jsx   tabbed results panel
├── data/sentimix/          CoNLL dataset files
├── models/coswitchnlp_v1/  Saved checkpoint
└── notebooks/
    └── exploration.ipynb   EDA: distributions, examples
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Server status + model info |
| `GET` | `/examples` | 6 preloaded demo sentences |
| `POST` | `/analyze` | Analyse a single text |
| `POST` | `/batch` | Analyse up to 50 texts |

### Example

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "yaar ye movie bohot amazing thi"}'
```

Response:

```json
{
  "tokens": [
    {"token": "yaar", "language": "lang1", "confidence": 0.97},
    {"token": "ye", "language": "lang1", "confidence": 0.95},
    {"token": "movie", "language": "lang2", "confidence": 0.88},
    {"token": "bohot", "language": "lang1", "confidence": 0.99},
    {"token": "amazing", "language": "lang2", "confidence": 0.96},
    {"token": "thi", "language": "lang1", "confidence": 0.98}
  ],
  "sentiment": "positive",
  "sentiment_confidence": 0.91,
  "sentiment_scores": {"negative": 0.03, "neutral": 0.06, "positive": 0.91},
  "code_mixing_index": 0.33,
  "language_distribution": {"lang1": 0.67, "lang2": 0.33, ...},
  "processing_time_ms": 28.4
}
```

---

## Model Architecture

```
XLM-RoBERTa-base (278M params, shared encoder)
    ├── LID Head:       Dropout → Linear(768, 6)       → per-token language label
    └── Sentiment Head: Dropout → Linear(768,256) → GELU → Dropout → Linear(256,3)
                                                          → sentence sentiment
```

Joint loss: `L = 0.5 × L_LID + 0.5 × L_sentiment`

Label sets:
- **LID:** `lang1` (Hindi), `lang2` (English), `mixed`, `ne`, `other`, `univ`
- **Sentiment:** `negative`, `neutral`, `positive`

---

## Expected Results

| Task | Metric | Target |
|------|--------|--------|
| Language ID (token) | Weighted F1 | ≥ 90% |
| Sentiment | Weighted F1 | ≥ 70% |

---

## Training Options

```
python train.py --help

  --train_file    Path to training CoNLL file  [default: ../data/sentimix/train.txt]
  --val_file      Path to validation CoNLL file
  --output_dir    Directory to save model checkpoint
  --model_name    HuggingFace model ID          [default: FacebookAI/xlm-roberta-base]
  --epochs        Number of training epochs     [default: 5]
  --batch_size    Training batch size           [default: 16]
  --lr            Learning rate                 [default: 2e-5]
  --max_len       Max sequence length           [default: 128]
  --alpha         LID loss weight               [default: 0.5]
  --beta          Sentiment loss weight         [default: 0.5]
  --patience      Early stopping patience       [default: 2]
```

---

## References

1. Patwa et al. (2020) — SemEval-2020 Task 9: https://arxiv.org/abs/2008.04277
2. Aguilar et al. (2020) — LinCE: https://arxiv.org/abs/2005.04322
3. Conneau et al. (2019) — XLM-RoBERTa: https://arxiv.org/abs/1911.02116
4. Das & Gambäck (2014) — Code-Mixing Index
