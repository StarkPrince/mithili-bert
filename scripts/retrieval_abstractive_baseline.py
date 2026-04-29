from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

from compare_abstractive import score_predictions


SPACE_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    return SPACE_RE.sub(" ", text).strip()


def char_ngrams(text: str, n_min: int = 3, n_max: int = 5) -> list[str]:
    compact = normalize(text).replace(" ", "")
    grams = []
    for n in range(n_min, n_max + 1):
        grams.extend(compact[i : i + n] for i in range(max(0, len(compact) - n + 1)))
    return grams


def hashed_vector(text: str, buckets: int) -> dict[int, float]:
    counts: Counter[int] = Counter()
    for gram in char_ngrams(text):
        digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
        counts[int.from_bytes(digest, "little") % buckets] += 1
    norm = math.sqrt(sum(value * value for value in counts.values())) or 1.0
    return {key: value / norm for key, value in counts.items()}


def dot(left: dict[int, float], right: dict[int, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(key, 0.0) for key, value in left.items())


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=Path("data/maithili_train.csv"))
    parser.add_argument("--test-file", type=Path, default=Path("data/maithili_test.csv"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--buckets", type=int, default=2**18)
    parser.add_argument("--token-mode", choices=["word", "char"], default="char")
    parser.add_argument("--output", type=Path, default=Path("retrieval_abstractive_results.json"))
    args = parser.parse_args()

    train_rows = read_rows(args.train_file)
    test_rows = read_rows(args.test_file)
    if args.limit:
        test_rows = test_rows[: args.limit]

    train_vectors = [hashed_vector(row["article_maithili"], args.buckets) for row in train_rows]
    predictions = []
    matches = []
    for test_index, test_row in enumerate(test_rows):
        test_vector = hashed_vector(test_row["article_maithili"], args.buckets)
        best_index = max(range(len(train_rows)), key=lambda index: dot(test_vector, train_vectors[index]))
        predictions.append(train_rows[best_index]["summary_maithili"])
        matches.append({"test_index": test_index, "train_index": best_index, "score": dot(test_vector, train_vectors[best_index])})

    result = {
        "model": "retrieval_train_summary",
        "documents": len(test_rows),
        "token_mode": args.token_mode,
        "metrics": score_predictions(test_rows, predictions, args.token_mode),
        "matches": matches[:20],
        "predictions": [
            {
                "index": index,
                "reference": row["summary_maithili"],
                "prediction": prediction,
            }
            for index, (row, prediction) in enumerate(zip(test_rows, predictions))
        ],
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in {"predictions", "matches"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
