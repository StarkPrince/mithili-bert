from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import sacrebleu
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


TOKEN_RE = re.compile(r"\S+")
CONTROL_TOKEN_RE = re.compile(r"</s>|<s>|<pad>|<2[a-z]{2,4}>")


@dataclass
class Example:
    index: int
    article: str
    reference: str


def read_examples(path: Path, limit: int | None) -> list[Example]:
    examples: list[Example] = []
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle]
        for idx, row in enumerate(rows):
            article = row["article"].strip()
            reference = row["summary"].strip()
            if article and reference:
                examples.append(Example(index=idx, article=article, reference=reference))
            if limit is not None and len(examples) >= limit:
                break
    else:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for idx, row in enumerate(csv.DictReader(handle)):
                article = row["article_maithili"].strip()
                reference = row["summary_maithili"].strip()
                if article and reference:
                    examples.append(Example(index=idx, article=article, reference=reference))
                if limit is not None and len(examples) >= limit:
                    break
    return examples


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.replace("\n", " "))


def clean_prediction(text: str) -> str:
    text = CONTROL_TOKEN_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def overlap_f1(candidate_items: list, reference_items: list) -> tuple[float, float, float]:
    if not candidate_items or not reference_items:
        return 0.0, 0.0, 0.0
    candidate_counts = Counter(candidate_items)
    reference_counts = Counter(reference_items)
    overlap = sum((candidate_counts & reference_counts).values())
    precision = overlap / len(candidate_items)
    recall = overlap / len(reference_items)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def rouge_1_2(candidate: str, reference: str) -> dict[str, float]:
    cand = tokenize(candidate)
    ref = tokenize(reference)
    _, _, rouge1 = overlap_f1(cand, ref)
    _, _, rouge2 = overlap_f1(ngrams(cand, 2), ngrams(ref, 2))
    return {"rouge1": rouge1, "rouge2": rouge2}


def meteor_like(candidate: str, reference: str) -> float:
    precision, recall, _ = overlap_f1(tokenize(candidate), tokenize(reference))
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return (10 * precision * recall) / (recall + 9 * precision)


def batched(items: list[Example], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def generate_summaries(
    examples: list[Example],
    model_name: str,
    source_lang: str,
    target_lang: str,
    batch_size: int,
    max_input_tokens: int,
    max_output_tokens: int,
    num_beams: int,
) -> list[str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        do_lower_case=False,
        use_fast=False,
        keep_accents=True,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()

    bos_id = tokenizer._convert_token_to_id_with_added_voc("<s>")
    eos_id = tokenizer._convert_token_to_id_with_added_voc("</s>")
    pad_id = tokenizer._convert_token_to_id_with_added_voc("<pad>")
    decoder_start_token_id = tokenizer._convert_token_to_id_with_added_voc(target_lang)

    outputs: list[str] = []
    started = time.time()
    with torch.inference_mode():
        for batch_index, batch in enumerate(batched(examples, batch_size), start=1):
            inputs = [f"{example.article} </s> {source_lang}" for example in batch]
            encoded = tokenizer(
                inputs,
                add_special_tokens=False,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_tokens,
            ).to(device)
            generated = model.generate(
                encoded.input_ids,
                attention_mask=encoded.attention_mask,
                use_cache=True,
                no_repeat_ngram_size=3,
                num_beams=num_beams,
                length_penalty=0.8,
                early_stopping=True,
                max_new_tokens=max_output_tokens,
                pad_token_id=pad_id,
                bos_token_id=bos_id,
                eos_token_id=eos_id,
                decoder_start_token_id=decoder_start_token_id,
            )
            decoded = tokenizer.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            outputs.extend(clean_prediction(text) for text in decoded)
            done = min(batch_index * batch_size, len(examples))
            print(f"generated {done}/{len(examples)} in {time.time() - started:.1f}s", file=sys.stderr)
    return outputs


def evaluate_split(path: Path, args: argparse.Namespace) -> dict:
    examples = read_examples(path, args.limit)
    predictions = generate_summaries(
        examples,
        model_name=args.model,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        batch_size=args.batch_size,
        max_input_tokens=args.max_input_tokens,
        max_output_tokens=args.max_output_tokens,
        num_beams=args.num_beams,
    )

    rouge1_total = 0.0
    rouge2_total = 0.0
    meteor_total = 0.0
    references = []
    for prediction, example in zip(predictions, examples):
        scores = rouge_1_2(prediction, example.reference)
        rouge1_total += scores["rouge1"]
        rouge2_total += scores["rouge2"]
        meteor_total += meteor_like(prediction, example.reference)
        references.append(example.reference)

    count = max(len(examples), 1)
    bleu = sacrebleu.corpus_bleu(predictions, [references], tokenize="none").score / 100.0

    return {
        "file": str(path),
        "model": args.model,
        "documents": len(examples),
        "source_lang": args.source_lang,
        "target_lang": args.target_lang,
        "metrics": {
            "rouge1": rouge1_total / count,
            "rouge2": rouge2_total / count,
            "bleu": bleu,
            "meteor": meteor_total / count,
        },
        "avg_reference_tokens": sum(len(tokenize(example.reference)) for example in examples) / count,
        "avg_prediction_tokens": sum(len(tokenize(prediction)) for prediction in predictions) / count,
        "samples": [
            {
                "index": example.index,
                "reference": example.reference,
                "prediction": prediction,
            }
            for example, prediction in list(zip(examples, predictions))[:5]
        ],
        "predictions": [
            {
                "index": example.index,
                "reference": example.reference,
                "prediction": prediction,
            }
            for example, prediction in zip(examples, predictions)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--processed", action="store_true")
    parser.add_argument("--model", default="ai4bharat/MultiIndicSentenceSummarizationSS")
    parser.add_argument("--source-lang", default="<2hi>")
    parser.add_argument("--target-lang", default="<2hi>")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-input-tokens", type=int, default=768)
    parser.add_argument("--max-output-tokens", type=int, default=96)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("indicbart_eval_results.json"))
    args = parser.parse_args()

    results = {}
    for split in args.splits:
        print(f"evaluating {split}", file=sys.stderr)
        path = args.data_dir / f"{split}.jsonl" if args.processed else args.data_dir / f"maithili_{split}.csv"
        results[split] = evaluate_split(path, args)

    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    compact = {key: {k: v for k, v in value.items() if k != "samples"} for key, value in results.items()}
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
