# Odia Sentiment Analysis (Group 8)

Fine-tunes [`ai4bharat/IndicBERTv2-MLM-only`](https://huggingface.co/ai4bharat/IndicBERTv2-MLM-only)
on the Odia portion of [IndiSentiment140](https://huggingface.co/datasets/saurabh1003/IndiSentiment140)
for binary sentiment classification (0 = negative, 1 = positive).
Published benchmark target: 90.7% accuracy.

## What you need

- A machine with an NVIDIA GPU (the script prints a warning and runs very slowly on CPU)
- The Odia-only CSV `indi_sentiment_140_en_or.csv` (~300 MB, 1,595,454 rows, columns: `sentiment`, `id`, `text`).
  **It is not in this repo** (too large for GitHub). Get it from whoever shared this repo and put it in the repo folder.

## Setup

```bash
git clone <this-repo-url>
cd <repo-folder>
pip install -r requirements.txt
# place indi_sentiment_140_en_or.csv in this folder
```

## Run

1. Smoke test (a few minutes). Confirms the GPU is used and the pipeline works:

```bash
python train_server_group8.py --sample 5000 --epochs 1 --out runs/smoke
```

2. Full run. Use tmux (or nohup) so it survives an SSH disconnect:

```bash
tmux new -s odia
python train_server_group8.py --out runs/full
```

Detach with `Ctrl+B` then `D`; reattach with `tmux attach -t odia`.

Useful options: `--batch 16` if you hit CUDA out-of-memory, `--batch 64` on a 24 GB+ GPU,
`--epochs N`, `--sample N` (default: all rows), `--csv path/to/file.csv`.

If a run is interrupted, re-run the same command and it resumes from the last checkpoint.

## Evaluate on the native Odia test set (after training finishes)

The professor's native gold test set (`.jsonl`, 255 rows, labels `positive`/`negative`) is **not in this repo**
(it is class data; get it from whoever shared this repo). Put it next to the scripts, then:

```bash
python eval_native_group8.py --native 01_sentiment_groups_1_8.jsonl \
    --model runs/full/final_model --results runs/full/results.json --out runs/full
```

This loads the same saved model, applies the same cleaning, and prints/saves both result tables
(benchmark vs native accuracy against the 90.7% target, and the translationese gap). It runs fine on CPU.

## Outputs (under `--out`)

- `results.json`: benchmark test accuracy, F1, and the exact training config used
- `benchmark_test_predictions.csv`: predictions on the held-out benchmark test split
- `native_test_predictions.csv`, `native_eval_results.json`, `results_tables.md`: native Odia evaluation, gap, and the two report tables
- `final_model/`: the fine-tuned model and tokenizer (large; do not commit)

Please send back `results.json`, `benchmark_test_predictions.csv`, `native_test_predictions.csv`,
`native_eval_results.json` and `results_tables.md`.
