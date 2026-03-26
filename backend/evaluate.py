"""
Evaluation script for CoSwitchNLP — runs on test set and prints full metrics.

Usage:
    python evaluate.py \
        --model_dir ../models/coswitchnlp_v1 \
        --test_file ../data/sentimix/test.txt

Outputs:
    - Sentiment: precision / recall / F1 per class + macro/weighted averages
    - LID      : per-label F1, token accuracy
    - Confusion matrix for sentiment
    - Saves results to <model_dir>/test_results.json
"""

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import XLMRobertaTokenizerFast

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)

from dataset import SentiMixDataset
from model import (
    CoSwitchModel,
    LID_ID2LABEL,
    SENTIMENT_LABELS,
    LID_LABELS,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate CoSwitchNLP on test set")
    p.add_argument("--model_dir", default="../models/coswitchnlp_v1")
    p.add_argument("--test_file", default="../data/sentimix/test.txt")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_len",    type=int, default=128)
    return p.parse_args()


def run_evaluation(args: argparse.Namespace) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading model from: {args.model_dir}")
    model = CoSwitchModel.load(args.model_dir, device=str(device))
    model.eval()

    tokenizer = XLMRobertaTokenizerFast.from_pretrained("FacebookAI/xlm-roberta-base")

    # ── Load test data ─────────────────────────────────────────────────────────
    print(f"Loading test data: {args.test_file}")
    test_ds = SentiMixDataset(args.test_file, tokenizer, args.max_len)
    loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # ── Inference ──────────────────────────────────────────────────────────────
    all_sent_preds,  all_sent_labels  = [], []
    all_lid_preds:   list[list[str]]  = []
    all_lid_labels:  list[list[str]]  = []

    print(f"\nRunning inference on {len(test_ds)} examples...")
    t0 = time.time()

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn      = batch["attention_mask"].to(device)
            lid_lbl   = batch["lid_labels"].to(device)
            sent_lbl  = batch["sentiment_labels"].to(device)

            lid_logits, sent_logits = model(input_ids, attn)

            # Sentiment
            sent_preds = sent_logits.argmax(dim=-1)
            all_sent_preds.extend(sent_preds.cpu().tolist())
            all_sent_labels.extend(sent_lbl.cpu().tolist())

            # LID — skip -100 positions (special / continuation subwords)
            lid_pred = lid_logits.argmax(dim=-1)
            for seq_pred, seq_label in zip(lid_pred.cpu().tolist(), lid_lbl.cpu().tolist()):
                pred_row, label_row = [], []
                for p, l in zip(seq_pred, seq_label):
                    if l != -100:
                        pred_row.append(LID_ID2LABEL.get(p, "other"))
                        label_row.append(LID_ID2LABEL.get(l, "other"))
                if pred_row:
                    all_lid_preds.append(pred_row)
                    all_lid_labels.append(label_row)

    elapsed = time.time() - t0

    # ── Sentiment metrics ──────────────────────────────────────────────────────
    sent_acc  = accuracy_score(all_sent_labels, all_sent_preds)
    sent_f1_w = f1_score(all_sent_labels, all_sent_preds, average="weighted", zero_division=0)
    sent_f1_m = f1_score(all_sent_labels, all_sent_preds, average="macro",    zero_division=0)
    sent_report = classification_report(
        all_sent_labels, all_sent_preds,
        target_names=SENTIMENT_LABELS, zero_division=0,
    )
    sent_cm = confusion_matrix(all_sent_labels, all_sent_preds).tolist()

    # ── LID metrics ────────────────────────────────────────────────────────────
    flat_lid_preds  = [t for seq in all_lid_preds  for t in seq]
    flat_lid_labels = [t for seq in all_lid_labels for t in seq]

    lid_acc  = accuracy_score(flat_lid_labels, flat_lid_preds) if flat_lid_preds else 0.0
    lid_f1_w = f1_score(flat_lid_labels, flat_lid_preds, average="weighted", zero_division=0) if flat_lid_preds else 0.0
    lid_f1_m = f1_score(flat_lid_labels, flat_lid_preds, average="macro",    zero_division=0) if flat_lid_preds else 0.0
    lid_report = classification_report(
        flat_lid_labels, flat_lid_preds,
        labels=LID_LABELS, zero_division=0,
    ) if flat_lid_preds else ""

    # ── Print results ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SENTIMENT CLASSIFICATION RESULTS")
    print("=" * 60)
    print(f"  Accuracy  : {sent_acc:.4f}")
    print(f"  F1 (weighted): {sent_f1_w:.4f}")
    print(f"  F1 (macro)   : {sent_f1_m:.4f}")
    print("\nPer-class report:")
    print(sent_report)
    print("Confusion matrix (rows=true, cols=pred):")
    print(f"  Labels: {SENTIMENT_LABELS}")
    for i, row in enumerate(sent_cm):
        print(f"  {SENTIMENT_LABELS[i]:10s}: {row}")

    print("\n" + "=" * 60)
    print("  TOKEN-LEVEL LANGUAGE ID RESULTS")
    print("=" * 60)
    print(f"  Token accuracy: {lid_acc:.4f}")
    print(f"  F1 (weighted) : {lid_f1_w:.4f}")
    print(f"  F1 (macro)    : {lid_f1_m:.4f}")
    print("\nPer-label report:")
    print(lid_report)

    print("=" * 60)
    print(f"  Evaluated {len(test_ds)} sentences in {elapsed:.1f}s "
          f"({len(test_ds)/elapsed:.0f} sentences/sec)")
    print("=" * 60)

    results = {
        "num_examples": len(test_ds),
        "inference_time_s": round(elapsed, 2),
        "sentiment": {
            "accuracy":     round(sent_acc, 4),
            "f1_weighted":  round(sent_f1_w, 4),
            "f1_macro":     round(sent_f1_m, 4),
            "confusion_matrix": sent_cm,
        },
        "lid": {
            "token_accuracy": round(lid_acc, 4),
            "f1_weighted":    round(lid_f1_w, 4),
            "f1_macro":       round(lid_f1_m, 4),
        },
    }

    out_path = os.path.join(args.model_dir, "test_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    return results


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
