from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from compare_abstractive import score_predictions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=Path("data/maithili_train.csv"))
    parser.add_argument("--test-file", type=Path, default=Path("data/maithili_test.csv"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--token-mode", choices=["word", "char"], default="char")
    parser.add_argument("--variant", choices=["retrieval", "headline", "headline_retrieval"], default="retrieval")
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--output", type=Path, default=Path("retrieval_abstractive_sklearn.json"))
    args = parser.parse_args()

    train_rows = read_rows(args.train_file)
    test_rows = read_rows(args.test_file)
    if args.limit:
        test_rows = test_rows[: args.limit]

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=args.max_features)
    train_matrix = vectorizer.fit_transform(row["article_maithili"] for row in train_rows)
    test_matrix = vectorizer.transform(row["article_maithili"] for row in test_rows)

    similarities = linear_kernel(test_matrix, train_matrix)
    best_indices = similarities.argmax(axis=1)
    retrieved_summaries = [train_rows[int(index)]["summary_maithili"] for index in best_indices]
    if args.variant == "retrieval":
        predictions = retrieved_summaries
    elif args.variant == "headline":
        predictions = [row["headline_maithili"] for row in test_rows]
    else:
        predictions = [
            f"{row['headline_maithili']} {retrieved_summary}"
            for row, retrieved_summary in zip(test_rows, retrieved_summaries)
        ]

    result = {
        "model": "tfidf_retrieval_train_summary",
        "variant": args.variant,
        "documents": len(test_rows),
        "token_mode": args.token_mode,
        "metrics": score_predictions(test_rows, predictions, args.token_mode),
        "predictions": [
            {
                "index": index,
                "reference": row["summary_maithili"],
                "prediction": prediction,
                "matched_train_index": int(best_indices[index]),
            }
            for index, (row, prediction) in enumerate(zip(test_rows, predictions))
        ],
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "predictions"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
