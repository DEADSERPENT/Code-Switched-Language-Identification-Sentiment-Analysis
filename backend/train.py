"""
Training script for CoSwitchNLP.

Usage:
    python train.py \
        --train_file ../data/sentimix/train.txt \
        --val_file   ../data/sentimix/val.txt \
        --output_dir ../models/coswitchnlp_v1 \
        --epochs 5 \
        --batch_size 16 \
        --lr 2e-5

Deep Learning architecture:
  - Encoder : XLM-RoBERTa-base (278M params, 12-layer transformer)
  - LID head: Dropout(0.1) → Linear(768→6)   [token-level]
  - Sent head: Dropout → Linear(768→256) → GELU → Dropout → Linear(256→3)  [sentence-level]
  - Loss: alpha*CrossEntropy_LID + beta*CrossEntropy_Sentiment (class-weighted)
  - Optimizer: AdamW, lr=2e-5, weight_decay=0.01
  - Scheduler: Linear warmup (10% steps) → linear decay
  - AMP: FP16 mixed-precision on CUDA, full FP32 on CPU
  - Gradient clipping: max_norm=1.0
  - Early stopping: patience=2 on combined val F1
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from transformers import XLMRobertaTokenizerFast, get_linear_schedule_with_warmup

from sklearn.metrics import classification_report, f1_score

from dataset import SentiMixDataset, get_class_weights
from model import (
    CoSwitchModel,
    LID_ID2LABEL,
    LID_LABEL2ID,
    SENTIMENT_ID2LABEL,
    SENTIMENT_LABELS,
    LID_LABELS,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CoSwitchNLP")
    p.add_argument("--train_file", default="../data/sentimix/train.txt")
    p.add_argument("--val_file",   default="../data/sentimix/val.txt")
    p.add_argument("--output_dir", default="../models/coswitchnlp_v1")
    p.add_argument("--model_name", default="FacebookAI/xlm-roberta-base")
    p.add_argument("--epochs",       type=int,   default=5)
    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--lr",           type=float, default=2e-5)
    p.add_argument("--max_len",      type=int,   default=128)
    p.add_argument("--alpha",        type=float, default=0.5,  help="LID loss weight")
    p.add_argument("--beta",         type=float, default=0.5,  help="Sentiment loss weight")
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--patience",     type=int,   default=2)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--num_workers",  type=int,   default=2)
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_model_summary(model: CoSwitchModel) -> None:
    """Print deep learning architecture summary."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    head_params = trainable - encoder_params

    print("\n" + "=" * 60)
    print("  CoSwitchNLP — Deep Learning Architecture")
    print("=" * 60)
    print(f"  Base encoder  : XLM-RoBERTa-base (12 transformer layers)")
    print(f"  Hidden size   : 768")
    print(f"  LID head      : Dropout → Linear(768 → {model.num_lid_labels})")
    print(f"  Sentiment head: Dropout → Linear(768→256) → GELU → Dropout → Linear(256→{model.num_sentiment_labels})")
    print(f"  LID labels    : {LID_LABELS}")
    print(f"  Sent labels   : {SENTIMENT_LABELS}")
    print(f"  Encoder params: {encoder_params:,}")
    print(f"  Head params   : {head_params:,}")
    print(f"  Total params  : {total:,}  (trainable: {trainable:,})")
    print("=" * 60 + "\n")


def evaluate(
    model: CoSwitchModel,
    loader: DataLoader,
    device: torch.device,
    lid_criterion: torch.nn.Module,
    sent_criterion: torch.nn.Module,
    use_amp: bool,
) -> dict:
    model.eval()
    all_sent_preds, all_sent_labels = [], []
    all_lid_preds:  list[list[str]] = []
    all_lid_labels: list[list[str]] = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn      = batch["attention_mask"].to(device)
            lid_lbl   = batch["lid_labels"].to(device)
            sent_lbl  = batch["sentiment_labels"].to(device)

            with autocast(device.type, enabled=use_amp):
                lid_logits, sent_logits = model(input_ids, attn)
                lid_loss  = lid_criterion(
                    lid_logits.view(-1, lid_logits.size(-1)), lid_lbl.view(-1)
                )
                sent_loss = sent_criterion(sent_logits, sent_lbl)
                loss      = 0.5 * lid_loss + 0.5 * sent_loss

            total_loss += loss.item()

            sent_preds = sent_logits.argmax(dim=-1)
            all_sent_preds.extend(sent_preds.cpu().tolist())
            all_sent_labels.extend(sent_lbl.cpu().tolist())

            # Decode LID predictions — skip -100 (special/continuation tokens)
            lid_pred = lid_logits.argmax(dim=-1)  # (B, L)
            for seq_pred, seq_label in zip(lid_pred.cpu().tolist(), lid_lbl.cpu().tolist()):
                pred_row, label_row = [], []
                for p, l in zip(seq_pred, seq_label):
                    if l != -100:
                        pred_row.append(LID_ID2LABEL.get(p, "other"))
                        label_row.append(LID_ID2LABEL.get(l, "other"))
                if pred_row:
                    all_lid_preds.append(pred_row)
                    all_lid_labels.append(label_row)

    sent_f1 = f1_score(all_sent_labels, all_sent_preds, average="weighted", zero_division=0)
    flat_lid_preds  = [t for seq in all_lid_preds  for t in seq]
    flat_lid_labels = [t for seq in all_lid_labels for t in seq]
    lid_f1 = (
        f1_score(flat_lid_labels, flat_lid_preds, average="weighted", zero_division=0)
        if flat_lid_preds else 0.0
    )
    avg_loss = total_loss / len(loader)

    return {
        "loss":             avg_loss,
        "sentiment_f1":     sent_f1,
        "lid_f1":           lid_f1,
        "sentiment_report": classification_report(
            all_sent_labels,
            all_sent_preds,
            target_names=SENTIMENT_LABELS,
            zero_division=0,
        ),
    }


def train() -> None:
    args = parse_args()
    set_seed(args.seed)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()   # AMP FP16 only on CUDA; CPU uses FP32

    print(f"Device  : {device}")
    if torch.cuda.is_available():
        print(f"GPU     : {torch.cuda.get_device_name(0)}")
    print(f"AMP FP16: {'enabled' if use_amp else 'disabled (CPU — using FP32)'}")

    tokenizer = XLMRobertaTokenizerFast.from_pretrained(args.model_name)

    print("\nLoading datasets...")
    train_ds = SentiMixDataset(args.train_file, tokenizer, args.max_len)
    val_ds   = SentiMixDataset(args.val_file,   tokenizer, args.max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_amp,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_amp,
    )

    print("Computing class weights...")
    weights      = get_class_weights(args.train_file)
    lid_weights  = weights["lid"].to(device)
    sent_weights = weights["sentiment"].to(device)

    lid_criterion  = torch.nn.CrossEntropyLoss(weight=lid_weights,  ignore_index=-100)
    sent_criterion = torch.nn.CrossEntropyLoss(weight=sent_weights)

    print("Initializing deep learning model...")
    model = CoSwitchModel(model_name=args.model_name).to(device)
    print_model_summary(model)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    # GradScaler is CUDA-only; use None on CPU (no scaling needed for FP32)
    scaler = GradScaler("cuda") if use_amp else None

    best_f1          = 0.0
    patience_counter = 0
    history: list[dict] = []

    print(f"Starting training — {args.epochs} epochs, {len(train_loader)} steps/epoch")
    print(f"Warmup: {warmup_steps} steps | LR: {args.lr} | Batch: {args.batch_size}")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attn      = batch["attention_mask"].to(device)
            lid_lbl   = batch["lid_labels"].to(device)
            sent_lbl  = batch["sentiment_labels"].to(device)

            with autocast(device.type, enabled=use_amp):
                lid_logits, sent_logits = model(input_ids, attn)
                lid_loss  = lid_criterion(
                    lid_logits.view(-1, lid_logits.size(-1)), lid_lbl.view(-1)
                )
                sent_loss = sent_criterion(sent_logits, sent_lbl)
                loss      = args.alpha * lid_loss + args.beta * sent_loss

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            if step % 50 == 0:
                print(
                    f"  Epoch {epoch} | Step {step}/{len(train_loader)} "
                    f"| Loss {loss.item():.4f} "
                    f"| LID {lid_loss.item():.4f} | Sent {sent_loss.item():.4f}"
                )

        elapsed          = time.time() - t0
        avg_train_loss   = epoch_loss / len(train_loader)

        print(f"\nEpoch {epoch} done in {elapsed:.0f}s | Avg train loss: {avg_train_loss:.4f}")
        print("Evaluating on validation set...")

        metrics = evaluate(model, val_loader, device, lid_criterion, sent_criterion, use_amp)
        val_f1  = (metrics["sentiment_f1"] + metrics["lid_f1"]) / 2

        print(
            f"  Val loss:      {metrics['loss']:.4f}\n"
            f"  Sentiment F1:  {metrics['sentiment_f1']:.4f}\n"
            f"  LID F1:        {metrics['lid_f1']:.4f}\n"
            f"  Combined F1:   {val_f1:.4f}"
        )
        print(metrics["sentiment_report"])

        history.append(
            {
                "epoch":      epoch,
                "train_loss": avg_train_loss,
                **{k: v for k, v in metrics.items() if k != "sentiment_report"},
            }
        )

        if val_f1 > best_f1:
            best_f1          = val_f1
            patience_counter = 0
            model.save(args.output_dir)
            print(f"  *** New best model saved (combined F1={best_f1:.4f}) ***")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

        print("-" * 60)

    # Save training history
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best combined F1: {best_f1:.4f}")
    print(f"Model saved to: {args.output_dir}")
    print(f"History saved to: {history_path}")


if __name__ == "__main__":
    train()
