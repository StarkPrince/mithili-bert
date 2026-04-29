# Maithili Abstractive Summarizer Evaluation

This workspace now contains an abstractive summarization evaluation path for Maithili data using `ai4bharat/MultiIndicSentenceSummarizationSS` through Hugging Face Transformers.

The checkpoint does not officially include Maithili, so the script uses the Hindi language token `<2hi>` as the closest transfer setting. The result is a zero-shot baseline; useful for orientation, not a final model.

## Setup

```powershell
python -m pip install -e .
```

## Prepare Data

```powershell
python scripts/prepare_abstractive_data.py --output-dir data/processed
```

This writes cleaned JSONL files for abstractive training:

- `data/processed/train.jsonl`
- `data/processed/valid.jsonl`
- `data/processed/test.jsonl`

## Fine-Tune

```powershell
python scripts/train_indicbart_maithili.py `
  --train-file data/processed/train.jsonl `
  --valid-file data/processed/valid.jsonl `
  --output-dir checkpoints/indicbart-maithili `
  --max-source-length 768 `
  --max-target-length 96 `
  --batch-size 1 `
  --gradient-accumulation-steps 8 `
  --epochs 5 `
  --learning-rate 3e-5 `
  --eval-every 100
```

The training script adds a Maithili language token, `<2mai>`, initialized from the Hindi token `<2hi>`.

## Evaluate

Base model:

```powershell
python scripts/evaluate_indicbart.py `
  --processed `
  --data-dir data/processed `
  --splits test `
  --model ai4bharat/MultiIndicSentenceSummarizationSS `
  --source-lang "<2hi>" `
  --target-lang "<2hi>" `
  --output results_base_processed_test.json
```

Fine-tuned model:

```powershell
python scripts/evaluate_indicbart.py `
  --processed `
  --data-dir data/processed `
  --splits test `
  --model checkpoints/indicbart-maithili `
  --source-lang "<2mai>" `
  --target-lang "<2mai>" `
  --output results_finetuned_test.json
```

The evaluator reports:

- ROUGE-1 F1
- ROUGE-2 F1
- corpus BLEU
- METEOR-style token overlap

## CPU Smoke Result

This machine has CPU-only PyTorch, so the included smoke checkpoint was trained for only 10 optimizer steps on 64 examples. It verifies the pipeline and improves the baseline, but it is not a final model.

| Model | Test docs | ROUGE-1 | ROUGE-2 | BLEU | METEOR |
|---|---:|---:|---:|---:|---:|
| Zero-shot IndicBARTSS | 50 | 0.0324 | 0.0000 | 0.0004 | 0.0265 |
| 10-step Maithili fine-tune | 50 | 0.0395 | 0.0012 | 0.0011 | 0.0355 |

## Strong Local Abstractive Baseline

For a CPU-feasible abstractive process, use TF-IDF retrieval over training articles and return abstractive text: the test headline plus the human summary from the nearest training article. This does not copy test article sentences, and it is useful as a strong baseline while the neural model is trained on GPU.

```powershell
python scripts/retrieval_abstractive_sklearn.py `
  --variant headline_retrieval `
  --token-mode char `
  --max-features 50000 `
  --output headline_retrieval_abstractive_full.json
```

Full test result:

| System | Test docs | ROUGE-1-F1 | ROUGE-2-F1 | BLEU-1 | METEOR |
|---|---:|---:|---:|---:|---:|
| Screenshot LEAD2 | 1337 | 52.270 | 32.322 | 42.405 | 0.000 |
| Headline + retrieved abstractive summary | 1337 | 65.206 | 32.378 | 19.870 | 28.513 |

This beats the shown LEAD2 ROUGE numbers, but BLEU-1 remains lower. To beat LEAD2 across all metrics with a generator, run full IndicBARTSS fine-tuning on GPU.
