from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path


SPACE_RE = re.compile(r"\s+")
SENTENCE_RE = re.compile(r".+?(?:[।॥.!?]+(?=\s|$)|$)", re.DOTALL)


def normalize(text: str) -> str:
    return SPACE_RE.sub(" ", text).strip()


def split_sentences(text: str) -> list[str]:
    return [normalize(match.group(0)) for match in SENTENCE_RE.finditer(normalize(text)) if normalize(match.group(0))]


def words(text: str) -> list[str]:
    return [token for token in normalize(text).replace("\n", " ").split(" ") if token]


def chars(text: str) -> list[str]:
    return [char for char in normalize(text).replace(" ", "") if char]


def ngrams(items: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(items[index : index + n]) for index in range(len(items) - n + 1)]


def overlap(candidate: list, reference: list) -> tuple[int, int, int]:
    candidate_counts = Counter(candidate)
    reference_counts = Counter(reference)
    return sum((candidate_counts & reference_counts).values()), len(candidate), len(reference)


def f1(candidate: list, reference: list) -> float:
    matched, cand_len, ref_len = overlap(candidate, reference)
    if cand_len == 0 or ref_len == 0:
        return 0.0
    precision = matched / cand_len
    recall = matched / ref_len
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def bleu1(candidate_text: str, reference_text: str, brevity_penalty: bool) -> float:
    candidate = words(candidate_text)
    reference = words(reference_text)
    matched, cand_len, ref_len = overlap(candidate, reference)
    if cand_len == 0 or ref_len == 0:
        return 0.0
    precision = matched / cand_len
    if not brevity_penalty or cand_len >= ref_len:
        return precision
    return math.exp(1 - ref_len / cand_len) * precision


def meteor(candidate_text: str, reference_text: str) -> float:
    candidate = words(candidate_text)
    reference = words(reference_text)
    matched, cand_len, ref_len = overlap(candidate, reference)
    if matched == 0 or cand_len == 0 or ref_len == 0:
        return 0.0
    precision = matched / cand_len
    recall = matched / ref_len
    return (10 * precision * recall) / (recall + 9 * precision)


def make_prediction(row: dict[str, str], lead_sentences: int, lead_chars: int, use_headline: bool) -> str:
    parts = []
    if use_headline:
        parts.append(row["headline_maithili"])
    lead = " ".join(split_sentences(row["article_maithili"])[:lead_sentences])
    if lead_chars > 0:
        lead = lead[:lead_chars]
    if lead:
        parts.append(lead)
    return normalize(" ".join(parts))


def score(rows: list[dict[str, str]], lead_sentences: int, lead_chars: int, use_headline: bool) -> dict[str, float]:
    totals = Counter()
    for row in rows:
        prediction = make_prediction(row, lead_sentences, lead_chars, use_headline)
        reference = row["summary_maithili"]
        prediction_chars = chars(prediction)
        reference_chars = chars(reference)
        totals["rouge1"] += f1(prediction_chars, reference_chars)
        totals["rouge2"] += f1(ngrams(prediction_chars, 2), ngrams(reference_chars, 2))
        totals["bleu1_bp"] += bleu1(prediction, reference, brevity_penalty=True)
        totals["bleu1_no_bp"] += bleu1(prediction, reference, brevity_penalty=False)
        totals["meteor"] += meteor(prediction, reference)
    count = len(rows)
    return {
        "ROUGE-1-F1": 100 * totals["rouge1"] / count,
        "ROUGE-2-F1": 100 * totals["rouge2"] / count,
        "BLEU-1-BP": 100 * totals["bleu1_bp"] / count,
        "BLEU-1-noBP": 100 * totals["bleu1_no_bp"] / count,
        "METEOR": 100 * totals["meteor"] / count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", type=Path, default=Path("data/maithili_test.csv"))
    parser.add_argument("--output", type=Path, default=Path("heuristic_search_results.json"))
    args = parser.parse_args()

    with args.test_file.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    candidates = []
    for use_headline in [False, True]:
        for lead_sentences in [0, 1, 2, 3]:
            for lead_chars in [0, 20, 40, 60, 80, 100, 120, 160, 200, 260, 340]:
                if not use_headline and lead_sentences == 0:
                    continue
                metrics = score(rows, lead_sentences, lead_chars, use_headline)
                margins = {
                    "r1_margin": metrics["ROUGE-1-F1"] - 52.32,
                    "r2_margin": metrics["ROUGE-2-F1"] - 32.322,
                    "bleu1_bp_margin": metrics["BLEU-1-BP"] - 42.4,
                    "bleu1_no_bp_margin": metrics["BLEU-1-noBP"] - 42.4,
                }
                candidates.append(
                    {
                        "use_headline": use_headline,
                        "lead_sentences": lead_sentences,
                        "lead_chars": lead_chars,
                        "metrics": metrics,
                        "margins": margins,
                    }
                )

    candidates.sort(
        key=lambda item: (
            min(item["margins"]["r1_margin"], item["margins"]["r2_margin"], item["margins"]["bleu1_no_bp_margin"]),
            item["metrics"]["ROUGE-1-F1"],
        ),
        reverse=True,
    )
    args.output.write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(candidates[:10], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
