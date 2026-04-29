from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


SPACE_RE = re.compile(r"\s+")
URL_RE = re.compile(r"https?://\S+|www\.\S+|pic\.twitter\.com/\S+", re.IGNORECASE)
BOILERPLATE_PATTERNS = [
    re.compile(r"\([^)]*एनडीटीवी[^)]*\)", re.IGNORECASE),
    re.compile(r"ई खबर.*?संपादित.*?गेल.*", re.IGNORECASE),
    re.compile(r".*सिंडिकेट फीड.*", re.IGNORECASE),
]


def normalize(text: str) -> str:
    text = URL_RE.sub(" ", text)
    for pattern in BOILERPLATE_PATTERNS:
        text = pattern.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def convert_file(input_path: Path, output_path: Path, min_article_chars: int, min_summary_chars: int) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    kept = 0
    skipped = 0

    with input_path.open("r", encoding="utf-8", newline="") as source, output_path.open(
        "w", encoding="utf-8"
    ) as target:
        for row in csv.DictReader(source):
            total += 1
            article = normalize(row["article_maithili"])
            summary = normalize(row["summary_maithili"])
            headline = normalize(row["headline_maithili"])
            if len(article) < min_article_chars or len(summary) < min_summary_chars:
                skipped += 1
                continue
            target.write(
                json.dumps(
                    {
                        "article": article,
                        "summary": summary,
                        "headline": headline,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            kept += 1

    return {"input": str(input_path), "output": str(output_path), "total": total, "kept": kept, "skipped": skipped}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--min-article-chars", type=int, default=80)
    parser.add_argument("--min-summary-chars", type=int, default=20)
    args = parser.parse_args()

    stats = {}
    for split in ["train", "valid", "test"]:
        stats[split] = convert_file(
            args.data_dir / f"maithili_{split}.csv",
            args.output_dir / f"{split}.jsonl",
            args.min_article_chars,
            args.min_summary_chars,
        )

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
