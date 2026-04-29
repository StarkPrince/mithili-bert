from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class SummaryDataset(Dataset):
    def __init__(self, path: Path, limit: int | None = None) -> None:
        self.rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                self.rows.append(json.loads(line))
                if limit is not None and len(self.rows) >= limit:
                    break

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, str]:
        return self.rows[index]


class Collator:
    def __init__(self, tokenizer, lang_token: str, max_source_length: int, max_target_length: int) -> None:
        self.tokenizer = tokenizer
        self.lang_token = lang_token
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __call__(self, rows: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        source_texts = [f"{row['article']} </s> {self.lang_token}" for row in rows]
        target_texts = [f"{self.lang_token} {row['summary']} </s>" for row in rows]
        model_inputs = self.tokenizer(
            source_texts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=self.max_source_length,
            return_tensors="pt",
        )
        labels = self.tokenizer(
            target_texts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        ).input_ids
        labels[labels == self.tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
        return model_inputs


def add_maithili_token(tokenizer, model, lang_token: str, init_from: str) -> None:
    added = tokenizer.add_special_tokens({"additional_special_tokens": [lang_token]})
    if added:
        model.resize_token_embeddings(len(tokenizer))

    lang_id = tokenizer.convert_tokens_to_ids(lang_token)
    init_id = tokenizer.convert_tokens_to_ids(init_from)
    if init_id is None or init_id == tokenizer.unk_token_id:
        raise ValueError(f"Could not find initializer token {init_from}")

    with torch.no_grad():
        embeddings = model.get_input_embeddings().weight
        embeddings[lang_id].copy_(embeddings[init_id])
        output_embeddings = model.get_output_embeddings()
        if output_embeddings is not None:
            output_embeddings.weight[lang_id].copy_(output_embeddings.weight[init_id])


def evaluate_loss(model, loader, device: str, max_batches: int = 20) -> float:
    model.eval()
    losses = []
    with torch.inference_mode():
        for idx, batch in enumerate(loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            losses.append(float(model(**batch).loss.detach().cpu()))
            if idx + 1 >= max_batches:
                break
    model.train()
    return sum(losses) / max(len(losses), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ai4bharat/MultiIndicSentenceSummarizationSS")
    parser.add_argument("--train-file", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--valid-file", type=Path, default=Path("data/processed/valid.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/indicbart-maithili"))
    parser.add_argument("--lang-token", default="<2mai>")
    parser.add_argument("--init-lang-token", default="<2hi>")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=256)
    parser.add_argument("--max-source-length", type=int, default=768)
    parser.add_argument("--max-target-length", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-every", type=int, default=25)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, do_lower_case=False, use_fast=False, keep_accents=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    add_maithili_token(tokenizer, model, args.lang_token, args.init_lang_token)
    model.to(device)
    model.train()

    train_dataset = SummaryDataset(args.train_file, args.max_train_samples)
    valid_dataset = SummaryDataset(args.valid_file, args.max_valid_samples)
    collator = Collator(tokenizer, args.lang_token, args.max_source_length, args.max_target_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    target_steps = args.max_steps or max(1, int(math.ceil(steps_per_epoch * args.epochs)))
    started = time.time()
    optimizer.zero_grad(set_to_none=True)
    best_valid = float("inf")
    global_step = 0
    running_loss = 0.0

    while global_step < target_steps:
        for micro_step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss / args.gradient_accumulation_steps
            loss.backward()
            running_loss += float(loss.detach().cpu())

            if micro_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                print(
                    json.dumps(
                        {
                            "step": global_step,
                            "train_loss": running_loss,
                            "elapsed_sec": round(time.time() - started, 1),
                        }
                    )
                )
                running_loss = 0.0

                if global_step % args.eval_every == 0 or global_step == target_steps:
                    valid_loss = evaluate_loss(model, valid_loader, device)
                    print(json.dumps({"step": global_step, "valid_loss": valid_loss}))
                    if valid_loss < best_valid:
                        best_valid = valid_loss
                        model.save_pretrained(args.output_dir)
                        tokenizer.save_pretrained(args.output_dir)
                        (args.output_dir / "training_args.json").write_text(
                            json.dumps(vars(args), indent=2, default=str),
                            encoding="utf-8",
                        )

                if global_step >= target_steps:
                    break

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(json.dumps({"saved": str(args.output_dir), "steps": global_step, "best_valid_loss": best_valid}))


if __name__ == "__main__":
    main()
