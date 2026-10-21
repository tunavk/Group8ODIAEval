#!/usr/bin/env python3
# =============================================================================
# Group 8 Odia Sentiment - server training script
# Fine-tunes ai4bharat/IndicBERTv2-MLM-only on the Odia-only IndiSentiment140
# CSV. Built for a GPU server (no Colab / Drive dependencies).
#
# Quick smoke test (a few minutes, confirms everything works):
#     python train_server_group8.py --sample 5000 --epochs 1 --out runs/smoke
#
# Full run (use tmux or nohup so it survives an SSH disconnect):
#     python train_server_group8.py --out runs/full
#
# If a run is interrupted, re-run the exact same command: it resumes from the
# last saved checkpoint automatically.
# =============================================================================

import argparse
import json
import os
import re

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from transformers.trainer_utils import get_last_checkpoint
import evaluate

# ---- command-line options --------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument("--csv", default="indi_sentiment_140_en_or.csv", help="path to the Odia-only CSV")
p.add_argument("--model", default="ai4bharat/IndicBERTv2-MLM-only")
p.add_argument("--out", default="runs/full", help="output dir (checkpoints, final model, predictions, metrics)")
p.add_argument("--sample", type=int, default=None, help="rows to use; omit for the full dataset")
p.add_argument("--epochs", type=int, default=3)
p.add_argument("--batch", type=int, default=32, help="per-device train batch size; try 64 on a 24GB+ GPU")
p.add_argument("--lr", type=float, default=3e-5)
p.add_argument("--max_len", type=int, default=128)
p.add_argument("--workers", type=int, default=8, help="processes used for tokenization")
p.add_argument("--seed", type=int, default=42)
args_cli = p.parse_args()

os.makedirs(args_cli.out, exist_ok=True)

print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: no GPU detected. This will be extremely slow on CPU.")

# ---- Step 1: load + clean ----------------------------------------------------
df = pd.read_csv(args_cli.csv)
print("Loaded:", df.shape, "| columns:", df.columns.tolist())

url_re, mention_re, space_re = re.compile(r"https?://\S+|www\.\S+"), re.compile(r"@\w+"), re.compile(r"\s+")
def clean(t):
    return space_re.sub(" ", mention_re.sub(" ", url_re.sub(" ", str(t)))).strip()

data = df[["text", "sentiment"]].dropna()
data["text"] = data["text"].map(clean)
data = data[data["text"].str.len() > 0].drop_duplicates(subset="text")
data["label"] = data["sentiment"].map(lambda v: 1 if int(v) == 4 else 0)   # 0=neg, 4=pos -> 0/1
data = data[["text", "label"]]

if args_cli.sample is not None and len(data) > args_cli.sample:
    data, _ = train_test_split(data, train_size=args_cli.sample, stratify=data["label"], random_state=args_cli.seed)
data = data.reset_index(drop=True)
print("Working set:", data.shape, "| label balance:\n", data["label"].value_counts())

# ---- Step 2: train / val / test split (test touched once, at the end) --------
train_df, temp_df = train_test_split(data, test_size=0.1, stratify=data["label"], random_state=args_cli.seed)
val_df, test_df   = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=args_cli.seed)
print("train:", train_df.shape, "val:", val_df.shape, "test:", test_df.shape)

# ---- Step 3: tokenize (no padding here; the collator pads each batch) --------
tokenizer = AutoTokenizer.from_pretrained(args_cli.model)

def tok(batch):
    return tokenizer(batch["text"], truncation=True, max_length=args_cli.max_len)

def to_ds(d):
    return Dataset.from_pandas(d.reset_index(drop=True)).map(
        tok, batched=True, num_proc=args_cli.workers, remove_columns=["text"]
    )

train_ds, val_ds, test_ds = to_ds(train_df), to_ds(val_df), to_ds(test_df)

# ---- Step 4: model + trainer ---------------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained(args_cli.model, num_labels=2)
acc = evaluate.load("accuracy")
f1  = evaluate.load("f1")

def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=-1)
    return {
        "accuracy": acc.compute(predictions=preds, references=pred.label_ids)["accuracy"],
        "f1": f1.compute(predictions=preds, references=pred.label_ids, average="binary")["f1"],
    }

use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
use_fp16 = torch.cuda.is_available() and not use_bf16

targs = TrainingArguments(
    output_dir=os.path.join(args_cli.out, "checkpoints"),
    per_device_train_batch_size=args_cli.batch,
    per_device_eval_batch_size=args_cli.batch * 2,
    num_train_epochs=args_cli.epochs,
    learning_rate=args_cli.lr,
    warmup_ratio=0.06,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    group_by_length=True,          # batches similar-length texts together -> less padding waste
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    bf16=use_bf16,
    fp16=use_fp16,
    dataloader_num_workers=4,
    logging_steps=200,
    report_to="none",
    seed=args_cli.seed,
)

trainer = Trainer(
    model=model,
    args=targs,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

# ---- Step 5: train (auto-resume if a checkpoint exists) -------------------------
last_ckpt = get_last_checkpoint(targs.output_dir) if os.path.isdir(targs.output_dir) else None
if last_ckpt:
    print("Resuming from", last_ckpt)
trainer.train(resume_from_checkpoint=last_ckpt)

# ---- Step 6: one-time evaluation on the held-out test set -----------------------
test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
test_acc = test_metrics["test_accuracy"] * 100
print("\n" + "=" * 54)
print(f"HELD-OUT TEST accuracy ({len(test_df)} examples): {test_acc:.2f}%")
print(f"Published target: 90.70%   diff: {test_acc - 90.7:+.2f} pts")
print("=" * 54)

# ---- Step 7: save model, predictions, metrics -----------------------------------
final_dir = os.path.join(args_cli.out, "final_model")
trainer.save_model(final_dir)
tokenizer.save_pretrained(final_dir)

pred = trainer.predict(test_ds)
out_df = test_df.reset_index(drop=True).copy()
out_df["predicted_label"] = np.argmax(pred.predictions, axis=-1)
out_df.to_csv(os.path.join(args_cli.out, "benchmark_test_predictions.csv"), index=False)

with open(os.path.join(args_cli.out, "results.json"), "w") as f:
    json.dump(
        {
            "model": args_cli.model,
            "benchmark_test_accuracy_pct": round(test_acc, 2),
            "published_target_pct": 90.7,
            "test_metrics": test_metrics,
            "config": vars(args_cli),
            "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
        },
        f, indent=2, default=str,
    )
print("Saved model, predictions and results.json under:", args_cli.out)
