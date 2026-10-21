#!/usr/bin/env python3
# =============================================================================
# Group 8 Odia Sentiment - native Odia evaluation + translationese gap
#
# Loads the SAME fine-tuned model saved by train_server_group8.py, applies the
# SAME text cleaning, and evaluates on the professor's native Odia gold test set
# (jsonl with fields: id, text, label in {negative, positive}).
# Then reports the two result tables from the assignment sheet:
#   Table 1: benchmark vs native accuracy against the published target (90.7%)
#   Table 2: translationese gap = benchmark score - native score
#
# Usage (after training has finished):
#   python eval_native_group8.py --native 01_sentiment_groups_1_8.jsonl \
#       --model runs/full/final_model --results runs/full/results.json --out runs/full
#
# Works on CPU too (255 examples take seconds), so it can be run anywhere the
# saved model folder is available.
# =============================================================================

import argparse
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix,
)

PUBLISHED_TARGET = 90.7
LABEL_MAP = {"negative": 0, "positive": 1}   # same 0/1 scheme used in training

# --- identical to the cleaning in train_server_group8.py (must stay in sync) ---
url_re, mention_re, space_re = re.compile(r"https?://\S+|www\.\S+"), re.compile(r"@\w+"), re.compile(r"\s+")


def clean(t):
    return space_re.sub(" ", mention_re.sub(" ", url_re.sub(" ", str(t)))).strip()


def load_native(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    missing = {"id", "text", "label"} - set(df.columns)
    if missing:
        raise SystemExit(f"Native file is missing expected fields: {sorted(missing)}")

    labels = df["label"].astype(str).str.strip().str.lower()
    unknown = sorted(set(labels) - set(LABEL_MAP))
    if unknown:
        raise SystemExit(
            f"Unexpected labels in native set: {unknown}. The model is binary "
            f"(negative/positive), so every row must use one of {sorted(LABEL_MAP)}."
        )
    df["gold"] = labels.map(LABEL_MAP).astype(int)
    df["text_raw"] = df["text"].astype(str)
    df["text"] = df["text_raw"].map(clean)

    empty = df["text"].str.len() == 0
    n_empty = int(empty.sum())
    if n_empty:
        print(f"WARNING: dropping {n_empty} rows that are empty after cleaning")
        df = df[~empty]
    return df.reset_index(drop=True), n_empty


def predict_probs(texts, model_dir, max_len, batch_size):
    """Return an (n, 2) array of class probabilities from the saved model."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Inference device:", device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()

    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(
                texts[i:i + batch_size], truncation=True, max_length=max_len,
                padding=True, return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            out.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)


def bootstrap_ci(y_true, y_pred, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    correct = (y_true == y_pred).astype(float)
    accs = np.array([correct[rng.integers(0, n, n)].mean() for _ in range(n_boot)]) * 100
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--native", required=True, help="native Odia gold test set (.jsonl)")
    p.add_argument("--model", default="runs/full/final_model", help="saved fine-tuned model folder")
    p.add_argument("--results", default="runs/full/results.json", help="results.json written by the training script")
    p.add_argument("--out", default="runs/full", help="where to write the evaluation outputs")
    p.add_argument("--max_len", type=int, default=None, help="defaults to the max_len used in training")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--benchmark_acc", type=float, default=None,
                   help="benchmark test accuracy in %% (only if results.json is unavailable)")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)

    # benchmark score + training config come from the training run's results.json
    train_cfg = {}
    benchmark_acc = a.benchmark_acc
    if os.path.exists(a.results):
        with open(a.results) as f:
            res = json.load(f)
        train_cfg = res.get("config", {})
        if benchmark_acc is None:
            benchmark_acc = res["benchmark_test_accuracy_pct"]
    if benchmark_acc is None:
        raise SystemExit("No benchmark accuracy: pass --results (results.json) or --benchmark_acc.")
    max_len = a.max_len or train_cfg.get("max_len", 128)

    df, n_empty = load_native(a.native)
    print(f"Native test set: {len(df)} rows | gold labels: "
          f"{df['gold'].map({0: 'negative', 1: 'positive'}).value_counts().to_dict()}")

    probs = predict_probs(df["text"].tolist(), a.model, max_len, a.batch)
    pred = probs.argmax(axis=1)
    gold = df["gold"].to_numpy()

    native_acc = accuracy_score(gold, pred) * 100
    ci_lo, ci_hi = bootstrap_ci(gold, pred, seed=a.seed)
    majority = max(gold.mean(), 1 - gold.mean()) * 100
    tn, fp, fn, tp = confusion_matrix(gold, pred, labels=[0, 1]).ravel()
    gap = benchmark_acc - native_acc

    metrics = {
        "native_accuracy_pct": round(native_acc, 2),
        "native_accuracy_95ci_pct": [round(ci_lo, 2), round(ci_hi, 2)],
        "native_f1_positive": round(f1_score(gold, pred, pos_label=1), 4),
        "native_f1_macro": round(f1_score(gold, pred, average="macro"), 4),
        "native_precision_positive": round(precision_score(gold, pred, pos_label=1, zero_division=0), 4),
        "native_recall_positive": round(recall_score(gold, pred, pos_label=1, zero_division=0), 4),
        "majority_class_baseline_pct": round(majority, 2),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "benchmark_test_accuracy_pct": round(benchmark_acc, 2),
        "published_target_pct": PUBLISHED_TARGET,
        "diff_from_published_pts": round(benchmark_acc - PUBLISHED_TARGET, 2),
        "translationese_gap_pts": round(gap, 2),
        "n_native": int(len(df)),
        "n_dropped_empty_after_cleaning": n_empty,
        "max_len_used": max_len,
        "model_dir": a.model,
    }

    # ---- predictions file ----------------------------------------------------
    pred_df = pd.DataFrame({
        "id": df["id"],
        "text": df["text_raw"],
        "gold_label": df["gold"].map({0: "negative", 1: "positive"}),
        "predicted_label": pd.Series(pred).map({0: "negative", 1: "positive"}),
        "prob_positive": probs[:, 1].round(4),
        "correct": gold == pred,
    })
    pred_df.to_csv(os.path.join(a.out, "native_test_predictions.csv"), index=False)
    with open(os.path.join(a.out, "native_eval_results.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # ---- the two assignment tables -------------------------------------------
    table1 = (
        "| Evaluation Dataset | Metric | Published Target | Our Score | Difference from Published Target |\n"
        "|---|---|---|---|---|\n"
        f"| Benchmark Test | Accuracy | {PUBLISHED_TARGET:.1f}% | {benchmark_acc:.2f}% | {benchmark_acc - PUBLISHED_TARGET:+.2f} pts |\n"
        f"| Native Odia Test | Accuracy | N/A | {native_acc:.2f}% | N/A |\n"
    )
    table2 = (
        "| Comparison | Score |\n|---|---|\n"
        f"| Benchmark Score | {benchmark_acc:.2f}% |\n"
        f"| Native Odia Score | {native_acc:.2f}% |\n"
        f"| Translationese Gap | {gap:+.2f} pts |\n"
    )
    notes = (
        f"\nNative accuracy 95% bootstrap CI: [{ci_lo:.1f}%, {ci_hi:.1f}%] "
        f"(n={len(df)}); majority-class baseline: {majority:.1f}%.\n"
        f"Native F1 (positive): {metrics['native_f1_positive']:.3f} | macro-F1: {metrics['native_f1_macro']:.3f}\n"
        f"Confusion matrix (gold rows / predicted cols, neg,pos): "
        f"[[{tn}, {fp}], [{fn}, {tp}]]\n"
        "Gap = benchmark score - native Odia score (positive = model does worse on native Odia).\n"
    )
    with open(os.path.join(a.out, "results_tables.md"), "w") as f:
        f.write("## Table 1\n\n" + table1 + "\n## Table 2\n\n" + table2 + notes)

    print("\n" + "=" * 60)
    print(table1)
    print(table2)
    print(notes)
    print("=" * 60)
    print("Saved to", a.out, ": native_test_predictions.csv, native_eval_results.json, results_tables.md")


if __name__ == "__main__":
    main()
