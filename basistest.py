#!/usr/bin/env python3
# =============================================================================
# Group 8 Odia Sentiment — real training run (not the quick sanity check)
# Loads the local Odia-only CSV from Google Drive, fine-tunes
# IndicBERTv2-MLM-only with settings tuned for accuracy rather than speed,
# and reports a clean, leak-free held-out test accuracy at the end.
# =============================================================================

import re
import numpy as np
import pandas as pd

# ---- config ------------------------------------------------------------
CSV_PATH = "/mnt/ssd/nlp/group8_data/indi_sentiment_140_en_or.csv"  # <- adjust to your actual path
MODEL_NAME  = "ai4bharat/IndicBERTv2-MLM-only"

# Start with 100000 to see the trend fast; set to None for the full 1.6M rows
# once you're happy with where accuracy is heading (that full run can take
# 1-3+ hours on a T4).
SAMPLE_SIZE   = 200000
MAX_LEN       = 128
BATCH_SIZE    = 32          # drop to 16 if you hit a CUDA out-of-memory error
LEARNING_RATE = 2e-5
WEIGHT_DECAY  = 0.01
EPOCHS        = 5
SEED          = 42

SAVE_DIR = "/mnt/ssd/nlp/group8_model"  # saved to Drive so it survives a disconnect

# ---- Step 1: load the CSV -----------------------------------------------
df = pd.read_csv(CSV_PATH)
print("Loaded:", df.shape, "| columns:", df.columns.tolist())

# ---- Step 2: clean text + map labels -------------------------------------
url_re, mention_re, space_re = re.compile(r"https?://\S+|www\.\S+"), re.compile(r"@\w+"), re.compile(r"\s+")
def clean(t):
    return space_re.sub(" ", mention_re.sub(" ", url_re.sub(" ", str(t)))).strip()

data = df[["text", "sentiment"]].dropna()
data["text"] = data["text"].map(clean)
data = data[data["text"].str.len() > 0].drop_duplicates(subset="text")
data["label"] = data["sentiment"].map(lambda v: 1 if int(v) == 4 else 0)   # 0=neg, 4=pos -> 0/1

from sklearn.model_selection import train_test_split

if SAMPLE_SIZE is not None and len(data) > SAMPLE_SIZE:
    # stratified subsample via train_test_split (keeps label balance, and
    # avoids a pandas-version-dependent groupby/apply quirk that can drop
    # the grouping column on newer pandas releases)
    data, _ = train_test_split(data, train_size=SAMPLE_SIZE, stratify=data["label"], random_state=SEED)
    data = data.reset_index(drop=True)
print(
    "Pandas working-set RAM:",
    data.memory_usage(deep=True).sum() / (1024**3),
    "GB"
)
print("Working set:", data.shape, "| label balance:\n", data["label"].value_counts())

# ---- Step 3: three-way split (train / val / held-out test) -----------------
# val is used for monitoring during training; test is touched exactly once,
# at the very end, for the number you actually report.
train_df, temp_df = train_test_split(data, test_size=0.2, stratify=data["label"], random_state=SEED)
val_df, test_df   = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=SEED)
print("train:", train_df.shape, "val:", val_df.shape, "test:", test_df.shape)

# ---- Step 4: tokenize --------------------------------------------------
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
def tok(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN, padding="max_length")

def to_ds(d):
    ds = Dataset.from_pandas(d[["text", "label"]].reset_index(drop=True)).map(tok, batched=True, remove_columns=["text"])
    ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return ds

train_ds, val_ds, test_ds = to_ds(train_df), to_ds(val_df), to_ds(test_df)

# ---- Step 5: fine-tune -----------------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
acc = evaluate.load("accuracy")

def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=-1)
    return acc.compute(predictions=preds, references=pred.label_ids)

args = TrainingArguments(
    output_dir="/mnt/ssd/nlp/group8_model/checkpoints",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,
    logging_steps=50,
    report_to="none",
    seed=SEED,
)
trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                   processing_class=tokenizer, compute_metrics=compute_metrics)
trainer.train()

# ---- Step 6: final, one-time evaluation on the held-out test set -----------
test_metrics = trainer.evaluate(test_ds)
print("\n" + "=" * 50)
print(f"HELD-OUT TEST accuracy on {len(test_df)} examples: {test_metrics['eval_accuracy']*100:.2f}%")
print(f"Published target: 90.70%  |  diff: {test_metrics['eval_accuracy']*100 - 90.7:+.2f} pts")
print("=" * 50)

# ---- Step 7: save the model to Drive so it survives a disconnect -----------
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print("Saved model to:", SAVE_DIR)