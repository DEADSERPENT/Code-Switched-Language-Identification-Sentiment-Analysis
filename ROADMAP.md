# CoSwitchNLP — Project Vision & Roadmap

## Vision

Build a dual-task NLP system that understands **code-switched Hinglish text** — the kind millions of Indians
actually write on social media. Given a sentence like *"yaar ye movie bohot amazing thi"*, the system will:

1. **Highlight** each token by its language (Hindi / English / Mixed / Named Entity)
2. **Classify** the overall sentiment (Positive / Negative / Neutral) with confidence scores

The end result is a fully local, GPU-accelerated web app with a React frontend, FastAPI backend, and a
fine-tuned XLM-RoBERTa model — no paid APIs, no cloud dependencies, runs entirely on your RTX A2000.

---

## Problem Statement

Code-switching (CS) is the natural alternation between two or more languages within a single utterance.
In India, Hinglish (Hindi + English) dominates social media, yet most NLP systems fail because:

- Monolingual models don't understand the cross-lingual context
- Tokenizers trained on formal text mangle romanized Hindi
- Sentiment cues are split across languages ("bohot amazing" — the intensity word is Hindi, the adjective English)

This project attacks both sub-problems simultaneously via **multi-task learning** on a shared multilingual encoder.

---

## Architecture Overview

```
User Input (Hinglish text)
        │
        ▼
┌────────────────────────────────────────────┐
│           React Frontend (Vite)            │
│  TextInput → API call → render results     │
│  • TokenDisplay (color-coded per language) │
│  • SentimentGauge (confidence bars)        │
│  • LanguageStats (pie chart + CMI score)   │
└───────────────────┬────────────────────────┘
                    │ HTTP POST /analyze
                    ▼
┌────────────────────────────────────────────┐
│         FastAPI Backend (Python)           │
│  • Tokenization → model inference          │
│  • Subword → word alignment                │
│  • Structured JSON response                │
└───────────────────┬────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────┐
│      CoSwitchModel (PyTorch)               │
│                                            │
│  XLM-RoBERTa-base (shared encoder)         │
│         ├── LID Head (token classification)│
│         └── Sentiment Head (CLS token)     │
└────────────────────────────────────────────┘
```

---

## Dataset

### Primary: SemEval-2020 Task 9 — SentiMix (Hinglish)

| Split | Sentences | Tokens (approx) |
|-------|-----------|-----------------|
| Train | 15,131    | ~152,000        |
| Val   | 1,869     | ~48,000         |

- Word-level language tags: `Hin`, `Eng`, `mixed`, `univ`, `ne`, `other`
- Sentence-level sentiment: `positive`, `negative`, `neutral`
- Format: CoNLL (one token per line, tab-separated token + lang tag; sentence starts with `meta` line)

---

## Model Design

### Base: `FacebookAI/xlm-roberta-base`

| Property | Value |
|----------|-------|
| Parameters | 278M |
| Layers | 12 transformer blocks |
| Hidden size | 768 |
| Attention heads | 12 |
| Training data | 100 languages, 2.5TB CommonCrawl |
| VRAM (FP16, batch=16) | ~4GB on RTX A2000 12GB |

### Multi-Task Head Design

```python
# Task A — Token-Level LID (sequence labeling on every token)
lid_head:       Dropout(0.1) → Linear(768, 6)

# Task B — Sentence-Level Sentiment (CLS token classification)
sentiment_head: Dropout(0.1) → Linear(768, 256) → GELU → Dropout(0.1) → Linear(256, 3)

# Joint loss (class-weighted cross-entropy)
L = 0.5 * L_LID + 0.5 * L_sentiment
```

Label sets:
- LID: `lang1` (Hindi), `lang2` (English), `mixed`, `ne`, `other`, `univ`
- Sentiment: `negative`, `neutral`, `positive`

---

## Achieved Results

| Task | Metric | Score |
|------|--------|-------|
| LID (token) | Weighted F1 | **91.6%** |
| LID (token) | Token Accuracy | **91.6%** |
| Sentiment | Macro F1 | **57.9%** |
| Sentiment | Weighted F1 | **57.1%** |
| Sentiment | Accuracy | **57.5%** |
| Training | Best combined F1 | **0.7436** |
| Inference | Speed | **159 sentences/sec** |

> Sentiment F1 is competitive with published SemEval-2020 Task 9 systems (55–65% F1 range).

---

## Project Structure

```
coswitchnlp/
├── ROADMAP.md                  ← project vision and results
├── backend/
│   ├── requirements.txt        ← pip dependencies
│   ├── model.py                ← CoSwitchModel (multi-task XLM-R, PyTorch)
│   ├── dataset.py              ← SentiMix CoNLL loader, subword alignment, class weights
│   ├── train.py                ← full training loop (AMP FP16, AdamW, warmup, early stop)
│   ├── evaluate.py             ← test-set evaluation (F1, accuracy, confusion matrix)
│   ├── demo.py                 ← terminal demo with colored output
│   ├── inference.py            ← CoSwitchInference + compute_cmi()
│   └── app.py                  ← FastAPI: /analyze /batch /health /examples
├── frontend/
│   ├── package.json
│   ├── vite.config.js          ← /api proxy → localhost:8000
│   ├── tailwind.config.js      ← custom green palette (#10B981)
│   ├── postcss.config.js
│   ├── index.html
│   └── src/
│       ├── main.jsx
│       ├── App.jsx             ← root: analyze(), error, loading, header
│       ├── index.css           ← global styles, Inter font
│       └── components/
│           ├── TextInput.jsx       ← textarea + Lucide icons + example chips
│           ├── TokenDisplay.jsx    ← color-coded token grid + confidence toggle
│           ├── SentimentGauge.jsx  ← Lucide icons + animated score bars
│           ├── LanguageStats.jsx   ← Recharts pie chart + CMI gauge
│           └── AnalysisPanel.jsx   ← tabbed results: tokens | sentiment | stats
├── data/
│   └── sentimix/
│       ├── train.txt           ← 15,131 sentences (SemEval-2020 Task 9)
│       └── val.txt             ← 1,869 sentences
├── models/
│   └── coswitchnlp_v1/
│       ├── model.pt            ← fine-tuned weights (best val F1)
│       ├── config.json         ← label maps, model name
│       ├── training_history.json
│       └── test_results.json   ← final evaluation metrics
└── notebooks/
    └── exploration.ipynb       ← EDA: distributions, CMI histogram, training curves
```

---

## Deep Learning Concepts Demonstrated

| Concept | Where |
|---------|-------|
| Transformer (self-attention, FFN, layer norm) | XLM-RoBERTa encoder — `model.py` |
| Transfer learning / fine-tuning | Pre-trained → task-specific heads |
| Multi-task learning | Shared encoder, dual loss — `model.py`, `train.py` |
| Sequence labeling | LID head over all token positions |
| Classification with CLS pooling | Sentiment head — `model.py:75` |
| Dropout regularization | Both heads |
| GELU activation | Sentiment head hidden layer |
| Weighted cross-entropy loss | Class imbalance — `train.py` |
| AdamW optimizer | Weight decay + adaptive LR |
| Linear warmup + linear decay | `get_linear_schedule_with_warmup` |
| FP16 mixed-precision (AMP) | `GradScaler` + `autocast` — `train.py` |
| Gradient clipping | `clip_grad_norm_` — `train.py` |
| Early stopping | patience=2 on combined val F1 |
| Subword tokenization (BPE) | XLMRobertaTokenizerFast |
| Word↔subword alignment | Mean-pooling in inference, first-subword in training |

## ANLP Concepts Demonstrated

| Concept | Where |
|---------|-------|
| Code-switching / multilingual NLP | Problem definition |
| Language Identification (LID) | Task A — sequence labeling |
| Sentiment Analysis | Task B — sentence classification |
| SemEval shared task | SentiMix dataset |
| Code-Mixing Index (CMI) | Das & Gambäck 2014 — `inference.py:compute_cmi()` |
| Subword tokenization challenges | Word→subword alignment in `dataset.py` |
| Class imbalance in NLP | Inverse-frequency weights — `dataset.py:get_class_weights()` |
| Hinglish social media text | Romanized Hindi, no diacritics, informal spelling |
| Evaluation metrics (F1, accuracy) | `evaluate.py` |
| CoNLL data format | `dataset.py:load_sentimix_conll()` |

---

## Tech Stack

| Layer | Technology | Reason |
|-------|-----------|--------|
| Model | XLM-RoBERTa-base (HuggingFace) | Best multilingual encoder for social media text |
| Training | PyTorch 2.x + AMP | Fast, memory-efficient on RTX A2000 |
| Backend | FastAPI + Uvicorn | Async, auto-docs, Pydantic validation |
| Frontend | React 18 + Vite + Tailwind CSS | Fast HMR dev experience, utility-first CSS |
| Charts | Recharts | Lightweight React-native charting |
| Icons | Lucide React | Consistent icon system |
| HTTP | Axios | Clean API calls |

---

## Key Research References

1. **SemEval 2020 Task 9** — Patwa et al., 2020: https://arxiv.org/abs/2008.04277
2. **LinCE Benchmark** — Aguilar et al., 2020: https://arxiv.org/abs/2005.04322
3. **XLM-RoBERTa** — Conneau et al., 2019: https://arxiv.org/abs/1911.02116
4. **Code-Mixing Index** — Das & Gambäck, 2014
5. **Code-switching survey** — https://github.com/gentaiscool/code-switching-papers

---

## Demo Examples

| Input | Expected Sentiment | Notes |
|-------|-------------------|-------|
| "yaar ye movie bohot amazing thi" | Positive | Hindi-dominant |
| "kya bakwas hai ye product, waste of money" | Negative | Strong code-mix |
| "thik thak hai, nothing special about it" | Neutral | Balanced |
| "bhai maza aa gaya, what a performance!" | Positive | Equal mix |
| "mere paas abhi time nahi hai, will check later" | Neutral | Mostly Hindi |
| "totally loved it yaar, dil khush ho gaya" | Positive | English-start Hindi-end |

---

*M.Tech ANLP Mini Project — All components run fully locally on NVIDIA RTX A2000 12GB.*
