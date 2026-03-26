"""
Inference pipeline for CoSwitchNLP.

Handles:
- Loading the fine-tuned model
- Tokenizing raw text
- Aligning subword predictions back to original words
- Computing Code-Mixing Index (CMI)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import XLMRobertaTokenizerFast

from model import CoSwitchModel, LID_ID2LABEL, SENTIMENT_ID2LABEL, LID_LABELS, SENTIMENT_LABELS


@dataclass
class TokenPrediction:
    token: str
    language: str
    confidence: float


@dataclass
class AnalysisResult:
    tokens: list[TokenPrediction]
    sentiment: str
    sentiment_confidence: float
    sentiment_scores: dict[str, float]
    code_mixing_index: float
    language_distribution: dict[str, float]


def compute_cmi(lang_tags: list[str]) -> float:
    """
    Code-Mixing Index (CMI) after Das & Gambäck (2014).
    CMI = 1 - (max_lang_count / total_content_words)

    Returns a value in [0, 1]:
      0 = monolingual
      1 = maximally mixed (alternates every token)
    """
    content_tags = [t for t in lang_tags if t in ("lang1", "lang2", "mixed")]
    if not content_tags:
        return 0.0

    from collections import Counter
    counts = Counter(content_tags)
    # Treat "mixed" as spread equally across both languages
    mixed = counts.get("mixed", 0)
    l1 = counts.get("lang1", 0) + mixed / 2
    l2 = counts.get("lang2", 0) + mixed / 2
    dominant = max(l1, l2)
    total = len(content_tags)
    return round(1.0 - dominant / total, 4)


class CoSwitchInference:
    def __init__(self, model_dir: str, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        print(f"Loading model from {model_dir} on {self.device}...")
        self.model = CoSwitchModel.load(model_dir, device=str(self.device))
        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(
            "FacebookAI/xlm-roberta-base"
        )
        print("Model ready.")

    def _tokenize(self, words: list[str]) -> dict:
        return self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            return_offsets_mapping=False,
            max_length=128,
            truncation=True,
            padding=True,
        )

    def _align_word_predictions(
        self,
        words: list[str],
        encoding,
        lid_logits: torch.Tensor,
    ) -> list[TokenPrediction]:
        """
        Aggregate subword LID probabilities back to word level (mean pooling).
        """
        word_ids = encoding.word_ids(batch_index=0)
        probs = F.softmax(lid_logits[0], dim=-1)  # (L, num_lid_labels)

        # Group subword positions by word index
        word_subword_probs: dict[int, list[torch.Tensor]] = {}
        for pos, wid in enumerate(word_ids):
            if wid is None:
                continue
            if wid not in word_subword_probs:
                word_subword_probs[wid] = []
            word_subword_probs[wid].append(probs[pos])

        results: list[TokenPrediction] = []
        for wid, word in enumerate(words):
            if wid in word_subword_probs:
                avg_probs = torch.stack(word_subword_probs[wid]).mean(dim=0)
                pred_idx = avg_probs.argmax().item()
                lang = LID_ID2LABEL.get(pred_idx, "other")
                conf = round(avg_probs[pred_idx].item(), 4)
            else:
                lang = "other"
                conf = 0.0
            results.append(TokenPrediction(token=word, language=lang, confidence=conf))

        return results

    def predict(self, text: str) -> AnalysisResult:
        """
        Analyse a code-switched sentence.

        Args:
            text: Raw input text (whitespace-separated words recommended).

        Returns:
            AnalysisResult with per-token language labels and sentence sentiment.
        """
        text = text.strip()
        if not text:
            raise ValueError("Input text is empty.")

        # Simple whitespace tokenisation for display; model handles subwords internally
        words = text.split()
        if not words:
            raise ValueError("Input text has no tokens.")

        encoding = self._tokenize(words)
        input_ids = encoding["input_ids"].to(self.device)
        attn = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            lid_logits, sent_logits = self.model(input_ids, attn)

        token_preds = self._align_word_predictions(words, encoding, lid_logits)

        sent_probs = F.softmax(sent_logits[0], dim=-1)
        sent_idx = sent_probs.argmax().item()
        sentiment = SENTIMENT_ID2LABEL[sent_idx]
        sentiment_confidence = round(sent_probs[sent_idx].item(), 4)
        sentiment_scores = {
            label: round(sent_probs[i].item(), 4)
            for i, label in enumerate(SENTIMENT_LABELS)
        }

        lang_tags = [t.language for t in token_preds]
        cmi = compute_cmi(lang_tags)

        from collections import Counter
        tag_counts = Counter(lang_tags)
        total = len(lang_tags)
        language_distribution = {
            tag: round(tag_counts.get(tag, 0) / total, 4) for tag in LID_LABELS
        }

        return AnalysisResult(
            tokens=token_preds,
            sentiment=sentiment,
            sentiment_confidence=sentiment_confidence,
            sentiment_scores=sentiment_scores,
            code_mixing_index=cmi,
            language_distribution=language_distribution,
        )

    def batch_predict(self, texts: list[str]) -> list[AnalysisResult]:
        return [self.predict(t) for t in texts]
