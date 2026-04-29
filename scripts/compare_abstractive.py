from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import sacrebleu


SENTENCE_RE = re.compile(r".+?(?:[।॥.!?]+(?=\s|$)|$)", re.DOTALL)
SPACE_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    return SPACE_RE.sub(" ", text).strip()


def split_sentences(text: str) -> list[str]:
    return [normalize(match.group(0)) for match in SENTENCE_RE.finditer(normalize(text)) if normalize(match.group(0))]


def word_tokens(text: str) -> list[str]:
    return [token for token in normalize(text).split(" ") if token]


def char_tokens(text: str) -> list[str]:
    return [char for char in normalize(text).replace(" ", "") if char]


def ngrams(items: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(items[index : index + n]) for index in range(len(items) - n + 1)]


def f1(candidate_items: list, reference_items: list) -> float:
    if not candidate_items or not reference_items:
        return 0.0
    candidate_counts = Counter(candidate_items)
    reference_counts = Counter(reference_items)
    overlap = sum((candidate_counts & reference_counts).values())
    precision = overlap / len(candidate_items)
    recall = overlap / len(reference_items)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def meteor_like(candidate: str, reference: str) -> float:
    candidate_tokens = word_tokens(candidate)
    reference_tokens = word_tokens(reference)
    if not candidate_tokens or not reference_tokens:
        return 0.0
    candidate_counts = Counter(candidate_tokens)
    reference_counts = Counter(reference_tokens)
    overlap = sum((candidate_counts & reference_counts).values())
    precision = overlap / len(candidate_tokens)
    recall = overlap / len(reference_tokens)
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return (10 * precision * recall) / (recall + 9 * precision)


def bleu1(candidate: str, reference: str) -> float:
    candidate_tokens = word_tokens(candidate)
    reference_tokens = word_tokens(reference)
    if not candidate_tokens or not reference_tokens:
        return 0.0
    candidate_counts = Counter(candidate_tokens)
    reference_counts = Counter(reference_tokens)
    overlap = sum((candidate_counts & reference_counts).values())
    precision = overlap / len(candidate_tokens)
    brevity = 1.0
    if len(candidate_tokens) < len(reference_tokens):
        brevity = pow(2.718281828459045, 1 - len(reference_tokens) / len(candidate_tokens))
    return brevity * precision


def rouge(candidate: str, reference: str, token_mode: str) -> tuple[float, float]:
    tokenizer = char_tokens if token_mode == "char" else word_tokens
    cand = tokenizer(candidate)
    ref = tokenizer(reference)
    return f1(cand, ref), f1(ngrams(cand, 2), ngrams(ref, 2))


def load_csv(path: Path, limit: int | None) -> list[dict[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def load_model_predictions(path: Path) -> dict[int, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "predictions" in payload:
        return {int(item["index"]): item["prediction"] for item in payload["predictions"]}
    predictions: dict[int, str] = {}
    for split_payload in payload.values():
        for item in split_payload.get("predictions", split_payload.get("samples", [])):
            predictions[int(item["index"])] = item["prediction"]
    return predictions


def summarize_baseline(row: dict[str, str], name: str) -> str:
    sentences = split_sentences(row["article_maithili"])
    if name == "headline":
        return row["headline_maithili"]
    if name == "lead1":
        return " ".join(sentences[:1])
    if name == "lead2":
        return " ".join(sentences[:2])
    if name == "lead3":
        return " ".join(sentences[:3])
    raise ValueError(name)


def score_predictions(rows: list[dict[str, str]], predictions: list[str], token_mode: str) -> dict[str, float]:
    references = [row["summary_maithili"] for row in rows]
    rouge1_total = 0.0
    rouge2_total = 0.0
    meteor_total = 0.0
    bleu1_total = 0.0
    for prediction, reference in zip(predictions, references):
        rouge1, rouge2 = rouge(prediction, reference, token_mode)
        rouge1_total += rouge1
        rouge2_total += rouge2
        meteor_total += meteor_like(prediction, reference)
        bleu1_total += bleu1(prediction, reference)
    count = len(rows)
    return {
        "ROUGE-1-F1": 100 * rouge1_total / count,
        "ROUGE-2-F1": 100 * rouge2_total / count,
        "BLEU-1": 100 * bleu1_total / count,
        "BLEU": sacrebleu.corpus_bleu(predictions, [references], tokenize="none").score,
        "METEOR": 100 * meteor_total / count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", type=Path, default=Path("data/maithili_test.csv"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--token-mode", choices=["word", "char"], default="char")
    parser.add_argument("--model-json", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("comparison_results.json"))
    args = parser.parse_args()

    rows = load_csv(args.test_file, args.limit)
    results = {}
    for baseline in ["headline", "lead1", "lead2", "lead3"]:
        predictions = [summarize_baseline(row, baseline) for row in rows]
        results[baseline] = score_predictions(rows, predictions, args.token_mode)

    if args.model_json:
        model_predictions_by_index = load_model_predictions(args.model_json)
        predictions = [model_predictions_by_index.get(index, "") for index in range(len(rows))]
        results[args.model_json.stem] = score_predictions(rows, predictions, args.token_mode)

    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
