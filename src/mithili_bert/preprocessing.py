from __future__ import annotations

import re


_SENTENCE_RE = re.compile(r".+?(?:[।॥.!?]+(?=\s|$)|$)", re.DOTALL)
_SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return _SPACE_RE.sub(" ", text).strip()


def split_sentences(text: str, min_chars: int = 2) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    sentences = []
    for match in _SENTENCE_RE.finditer(normalized):
        sentence = match.group(0).strip()
        if len(sentence) >= min_chars:
            sentences.append(sentence)
    return sentences

