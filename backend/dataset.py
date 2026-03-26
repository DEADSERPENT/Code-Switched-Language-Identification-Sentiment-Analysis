"""
SentiMix dataset loader for SemEval-2020 Task 9 (Hinglish sentiment analysis).

File format (CoNLL-style):
    meta\t<id>\t<sentiment>
    token1\tlang_tag1
    token2\tlang_tag2
    ...
    (blank line between sentences)

Lang tags in original data: Hin, Eng, Mixed, Other, Universal, Name
We normalise them to: lang1, lang2, mixed, other, univ, ne
"""

import os
import re
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

from model import LID_LABEL2ID, SENTIMENT_LABEL2ID

# Map raw SentiMix tags → our canonical label set
# Actual tags in this corpus: Hin, Eng, O (punctuation/symbols), EMT (emoticon)
RAW_TAG_MAP = {
    "hin": "lang1",
    "eng": "lang2",
    "mixed": "mixed",
    "other": "other",
    "o": "other",       # punctuation / symbols in this corpus
    "emt": "other",     # emoticons treated as other
    "universal": "univ",
    "univ": "univ",
    "ne": "ne",
    "name": "ne",
    # LinCE variant spellings
    "lang1": "lang1",
    "lang2": "lang2",
}

SENTIMENT_MAP = {
    "positive": "positive",
    "negative": "negative",
    "neutral": "neutral",
    # SemEval sometimes uses these
    "0": "negative",
    "1": "neutral",
    "2": "positive",
}


def load_sentimix_conll(filepath: str) -> list[dict]:
    """
    Parse a CoNLL-style SentiMix file.

    Returns a list of dicts:
        {
            "tokens": ["yaar", "ye", "movie", ...],
            "lang_tags": ["lang1", "lang2", "lang2", ...],
            "sentiment": "positive"
        }
    """
    sentences: list[dict] = []
    current_tokens: list[str] = []
    current_langs: list[str] = []
    current_sentiment: str | None = None

    with open(filepath, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                # Sentence boundary
                if current_tokens:
                    sentences.append(
                        {
                            "tokens": current_tokens,
                            "lang_tags": current_langs,
                            "sentiment": current_sentiment or "neutral",
                        }
                    )
                current_tokens, current_langs, current_sentiment = [], [], None
                continue

            if line.lower().startswith("meta"):
                parts = line.split("\t")
                if len(parts) >= 3:
                    raw_sent = parts[-1].strip().lower()
                    current_sentiment = SENTIMENT_MAP.get(raw_sent, "neutral")
            else:
                parts = line.split("\t")
                if len(parts) >= 2:
                    token = parts[0].strip()
                    raw_tag = parts[1].strip().lower()
                    lang_tag = RAW_TAG_MAP.get(raw_tag, "other")
                    current_tokens.append(token)
                    current_langs.append(lang_tag)

    # Handle file without trailing blank line
    if current_tokens:
        sentences.append(
            {
                "tokens": current_tokens,
                "lang_tags": current_langs,
                "sentiment": current_sentiment or "neutral",
            }
        )

    return sentences


class SentiMixDataset(Dataset):
    """
    PyTorch Dataset for SentiMix. Tokenizes words with a fast tokenizer and
    aligns word-level LID labels to subword tokens (first-subword strategy).
    """

    def __init__(
        self,
        filepath: str,
        tokenizer: PreTrainedTokenizerFast,
        max_len: int = 128,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = load_sentimix_conll(filepath)
        print(
            f"Loaded {len(self.examples)} sentences from {os.path.basename(filepath)}"
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        tokens: list[str] = ex["tokens"]
        lang_tags: list[str] = ex["lang_tags"]
        sentiment: str = ex["sentiment"]

        # Tokenize with word_ids so we can align labels
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Align LID labels: first subword of each word gets real label, rest get -100
        word_ids = encoding.word_ids(batch_index=0)
        lid_labels = []
        prev_word_id = None
        for wid in word_ids:
            if wid is None:
                lid_labels.append(-100)  # special token
            elif wid != prev_word_id:
                # First subword of this word
                tag = lang_tags[wid] if wid < len(lang_tags) else "other"
                lid_labels.append(LID_LABEL2ID.get(tag, LID_LABEL2ID["other"]))
            else:
                lid_labels.append(-100)  # continuation subword
            prev_word_id = wid

        sentiment_label = SENTIMENT_LABEL2ID[sentiment]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "lid_labels": torch.tensor(lid_labels, dtype=torch.long),
            "sentiment_labels": torch.tensor(sentiment_label, dtype=torch.long),
        }


def get_class_weights(filepath: str) -> dict[str, torch.Tensor]:
    """Compute inverse-frequency weights to handle class imbalance."""
    from collections import Counter

    examples = load_sentimix_conll(filepath)

    lid_counts: Counter = Counter()
    sent_counts: Counter = Counter()

    for ex in examples:
        for tag in ex["lang_tags"]:
            canonical = RAW_TAG_MAP.get(tag.lower(), "other")
            lid_counts[canonical] += 1
        sent_counts[ex["sentiment"]] += 1

    total_lid = sum(lid_counts.values())
    lid_weights = torch.zeros(len(LID_LABEL2ID))
    for label, idx in LID_LABEL2ID.items():
        count = lid_counts.get(label, 1)
        lid_weights[idx] = total_lid / (len(LID_LABEL2ID) * count)

    total_sent = sum(sent_counts.values())
    sent_weights = torch.zeros(len(SENTIMENT_LABEL2ID))
    for label, idx in SENTIMENT_LABEL2ID.items():
        count = sent_counts.get(label, 1)
        sent_weights[idx] = total_sent / (len(SENTIMENT_LABEL2ID) * count)

    return {"lid": lid_weights, "sentiment": sent_weights}
