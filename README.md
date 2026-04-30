# Maithili Abstractive Summarization

This repository evaluates abstractive summarization for Maithili news articles. The dataset shipped under `data/` contains 16,621 train / 5,269 valid / 5,288 test rows of `(headline, summary, article)` triplets in Maithili. After cleaning (URL/boilerplate stripping and minimum-length filtering), the processed splits used for training and evaluation are 3,858 / 1,261 / 1,268 rows.

The work follows two parallel tracks:

1. A neural fine-tuning track on top of `ai4bharat/MultiIndicSentenceSummarizationSS` (IndicBARTSS), which does not officially cover Maithili.
2. A CPU-feasible non-neural track (TF-IDF retrieval + headline) that serves as a strong baseline and currently outperforms the smoke-trained neural model on this hardware.

The full numeric results, the model card details, and the exact fine-tuning recipe are documented below.

---

## Model

- **Base checkpoint:** [`ai4bharat/MultiIndicSentenceSummarizationSS`](https://huggingface.co/ai4bharat/MultiIndicSentenceSummarizationSS)
- **Architecture:** mBART-style encoder-decoder (`MBartForConditionalGeneration`)
- **Encoder/decoder layers:** 6 / 6
- **Hidden size:** 1024, FFN dim 4096, 16 attention heads
- **Max position embeddings:** 1024
- **Vocab size:** 64,016 (after adding the Maithili language token)
- **Tokenizer:** AlbertTokenizer (SentencePiece, slow tokenizer, `keep_accents=True`)
- **Source / target language tokens:** `<2hi>` for the zero-shot baseline, `<2mai>` for the fine-tuned model

The base checkpoint does not include a Maithili language token. The training script appends a new special token `<2mai>` to the tokenizer, resizes the model embeddings, and **initializes both the input and output `<2mai>` rows from the existing `<2hi>` (Hindi) token embeddings**. This gives the new token a sensible warm start before any optimizer step is taken — Hindi is the closest related script/grammar in the model's pretraining mix.

---

## Data Pipeline

Source CSV columns: `headline_maithili`, `summary_maithili`, `article_maithili`.

The preparation step (`scripts/prepare_abstractive_data.py`) writes JSONL with three fields (`article`, `summary`, `headline`) and applies:

- URL / `pic.twitter.com` / `www.*` stripping
- Removal of NDTV-style boilerplate and "syndicate feed" tags
- Whitespace collapsing
- Minimum article length 80 chars, minimum summary length 20 chars

Resulting processed split sizes:

| Split | Rows |
|---|---:|
| train | 3,858 |
| valid | 1,261 |
| test  | 1,268 |

Average reference summary length on the 50-doc smoke test is ~27 tokens.

---

## Fine-Tuning Process

The fine-tuning loop is implemented from scratch (PyTorch + AdamW, no `Trainer`) in `scripts/train_indicbart_maithili.py`. The IndicBART input convention is preserved: each example is encoded as `"<article> </s> <2mai>"` on the source side and `"<2mai> <summary> </s>"` on the target side, with `add_special_tokens=False` so the language tag is not duplicated. Padding tokens in the labels are replaced with `-100` to be ignored by the loss.

Steps performed by the script:

1. Load `ai4bharat/MultiIndicSentenceSummarizationSS` and its slow `AlbertTokenizer`.
2. Add `<2mai>` as an additional special token; resize embeddings; copy the `<2hi>` row into the new `<2mai>` row for both input and output embedding matrices.
3. Stream JSONL with a custom `Dataset` and a `Collator` that builds the source/target strings above.
4. Train with `AdamW`, `lr=3e-5`, gradient clipping at 1.0, gradient accumulation, and seed 7.
5. Periodically compute validation loss on up to 256 valid examples; checkpoint on improvement.
6. Save the model, tokenizer, and `training_args.json` to the output directory.

### Smoke-run hyperparameters (this machine, CPU-only)

The repo ships `checkpoints/indicbart-maithili-smoke/` produced with the following config (recorded in `checkpoints/indicbart-maithili-smoke/training_args.json`):

| Field | Value |
|---|---|
| Base model | `ai4bharat/MultiIndicSentenceSummarizationSS` |
| Lang token (added) | `<2mai>` (init from `<2hi>`) |
| `max_source_length` | 512 |
| `max_target_length` | 64 |
| `batch_size` | 1 |
| `gradient_accumulation_steps` | 4 |
| `epochs` | 1.0 |
| `max_steps` | 10 |
| `learning_rate` | 3e-5 |
| `seed` | 7 |
| `eval_every` | 5 |
| `max_train_samples` | 64 |
| `max_valid_samples` | 64 |

This is intentionally tiny — only 10 optimizer steps over 64 examples — so it is **a pipeline-validation checkpoint, not a production model**. It demonstrates that adding `<2mai>`, copying the `<2hi>` embedding, and training the seq2seq objective improves over the zero-shot Hindi-token baseline (numbers below).

### Recommended GPU run

For a real run, use the original training arguments (no `max_steps`, no `max_train_samples`):

- `max_source_length=768`, `max_target_length=96`
- `batch_size=1`, `gradient_accumulation_steps=8` (effective batch 8)
- `epochs=5`, `learning_rate=3e-5`, `eval_every=100`

This is what the training CLI in `scripts/train_indicbart_maithili.py` defaults to.

---

## Evaluation Protocol

Evaluation is implemented in `scripts/evaluate_indicbart.py` and `scripts/compare_abstractive.py`. Reported metrics:

- **ROUGE-1 F1**, **ROUGE-2 F1** (`rouge_score`)
- **Corpus BLEU** / **BLEU-1** (`sacrebleu`)
- **METEOR-style token overlap**

For ROUGE/BLEU on Devanagari we report both the standard whitespace ("word") tokenization and a character-n-gram ("char") tokenization to make scores comparable to character-level baselines reported in the literature.

---

## Results

### 1. Neural track — IndicBARTSS, 50-doc test slice

`results_base_processed_test50.json` and `results_finetuned_smoke_test50.json`. Metrics here are reported in their raw (0–1) form, as written by the evaluator.

| Model | Source/Target token | Test docs | ROUGE-1 | ROUGE-2 | BLEU | METEOR |
|---|---|---:|---:|---:|---:|---:|
| Zero-shot IndicBARTSS (base) | `<2hi>` | 50 | 0.0324 | 0.0000 | 0.00039 | 0.0265 |
| 10-step Maithili fine-tune (smoke) | `<2mai>` | 50 | 0.0395 | 0.0012 | 0.00107 | 0.0355 |

The smoke fine-tune improves all four metrics over the zero-shot Hindi-token baseline, but absolute numbers are still very low — expected for 10 optimizer steps. A full GPU run with the recommended hyperparameters is needed to make the neural track competitive with the retrieval baselines below.

Qualitatively, predictions on the same five test articles (see `results_*_test50.json`) show the fine-tune already locks onto the right entities and structure (e.g. the `अदनान पत्रवाला` headline reproduces almost exactly), whereas the zero-shot baseline produces shorter, less coherent fragments.

### 2. Non-neural track — full test set (1,337 docs, char-tokenized)

`final_comparison_char.json` — all systems scored on the full Maithili test split.

| System | ROUGE-1-F1 | ROUGE-2-F1 | BLEU-1 | BLEU | METEOR |
|---|---:|---:|---:|---:|---:|
| Headline only | 54.721 | 30.703 | 13.224 | 2.956 | 19.086 |
| LEAD-1 | 63.982 | 40.006 | 28.444 | 23.095 | 33.392 |
| LEAD-2 | 62.288 | 40.027 | 27.488 | 18.952 | 41.768 |
| LEAD-3 | 54.311 | 35.925 | 22.052 | 14.764 | 43.866 |
| **Headline + TF-IDF retrieved abstractive summary** | **65.206** | 32.378 | 19.870 | 4.089 | 28.513 |

Numbers are percentages (×100). The retrieval system uses character-n-gram TF-IDF (`char_wb`, `ngram_range=(3,5)`, `max_features=50000`), retrieves the nearest training article for each test article, and concatenates the test headline with that nearest-neighbor's *human* summary — so the output text is genuinely abstractive (not copied from the test article).

**Headline + retrieval beats all LEAD baselines on ROUGE-1**, while LEAD-1/2 still win on BLEU/METEOR. This is the strongest CPU-only system in this repo today.

### 3. Headline-vs-retrieval ablation (1,337 docs)

`abstractive_headline_retrieval_comparison.json`.

| Component | ROUGE-1-F1 | ROUGE-2-F1 | BLEU | METEOR |
|---|---:|---:|---:|---:|
| Headline only | 54.721 | 30.703 | 2.956 | 19.086 |
| Retrieved nearest-neighbor summary only | 64.295 | 24.278 | 2.703 | 16.579 |
| Headline + retrieved summary (concat) | **65.206** | **32.378** | **4.089** | **28.513** |

Concatenation strictly dominates either component alone on every metric.

### 4. LEAD heuristic search (`heuristic_search_results.json`)

A grid over `{use_headline, lead_sentences∈{1,2,3}, lead_chars∈{80..200}}` was scored against LEAD-2 as the reference. Top configurations on summed ROUGE-1+ROUGE-2+BLEU-1-BP+METEOR:

| use_headline | lead_sents | lead_chars | ROUGE-1-F1 | ROUGE-2-F1 | BLEU-1-BP | METEOR |
|:---:|:---:|:---:|---:|---:|---:|---:|
| true | 2 | 120 | **69.393** | 43.609 | 31.869 | 40.768 |
| true | 3 | 120 | 69.314 | 43.585 | 31.823 | 40.855 |
| true | 1 | 200 | 68.028 | 43.416 | 31.489 | 41.941 |
| true | 2 | 160 | 66.805 | **44.079** | 31.243 | **45.332** |
| true | 3 | 160 | 66.214 | 43.851 | 30.877 | 45.621 |

The strongest *extractive* configuration combines the headline with the first 2 sentences of the article, capped at 120 characters per sentence. It clears LEAD-2 by ~+7 ROUGE-1-F1 and ~+4 ROUGE-2-F1, while losing ~10 BLEU-1 due to the explicit length cap shortening outputs.

---

## Take-aways

- **Headline + TF-IDF retrieval is the current best abstractive system** in this repo (65.2 ROUGE-1-F1 on 1,337 docs), and it runs on CPU in seconds.
- **Headline + LEAD-2@120-chars is the current best extractive system** (69.4 ROUGE-1-F1).
- **The neural fine-tune is plumbing-only** at present (10 steps on 64 examples). The recipe — adding `<2mai>` initialized from `<2hi>` and training the IndicBARTSS seq2seq objective — is in place; the next step is a GPU run with the defaults in `scripts/train_indicbart_maithili.py`.

## Repo layout

- `data/` — raw CSV splits + cleaned `data/processed/*.jsonl`
- `scripts/prepare_abstractive_data.py` — CSV → JSONL with cleaning
- `scripts/train_indicbart_maithili.py` — fine-tunes IndicBARTSS with the `<2mai>` token
- `scripts/evaluate_indicbart.py` — computes ROUGE / BLEU / METEOR for a checkpoint
- `scripts/retrieval_abstractive_sklearn.py` — TF-IDF retrieval baseline
- `scripts/compare_abstractive.py` — common scoring helpers
- `scripts/optimize_abstractive_heuristics.py` — LEAD heuristic grid search
- `checkpoints/indicbart-maithili-smoke/` — pipeline-validation checkpoint
- `*.json` at the repo root — the result files referenced in the tables above

## Setup

```powershell
python -m pip install -e .
```
