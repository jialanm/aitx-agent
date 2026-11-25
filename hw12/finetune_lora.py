"""Problem 3: fine-tune Qwen3-4B with QLoRA to judge trial eligibility on the TREC pairs.

Same prompt as the chosen prompted arm (full trial text, label words, no
examples), so the comparison in problem 4 measures tuning alone. The target
is the label word; only the answer tokens are trained on, the prompt tokens
are masked. Recipe from Aljohani et al. (2026): 4-bit NF4 base with double
quantization, LoRA rank 16 / alpha 16 on all attention and MLP projections,
paged AdamW 8-bit, batch 1 with 8-step accumulation, early stopping on dev
loss. After training, the adapter is scored on dev with the same metrics as
the prompted arm (accuracy, macro-F1, false-eligible rate).

Usage, on a GPU machine:
    uv run --group hw12 python hw12/finetune_lora.py --model Qwen/Qwen3-4B \
        --epochs 3 --out data/hw12/lora_qwen3_4b
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from prompt_search import VERBALIZERS, build_messages, load_pairs, metrics, parse_label  # noqa: E402

IGNORE = -100


def build_example(prompt_ids: list[int], answer_ids: list[int], max_length: int) -> dict | None:
    """Token ids and labels for one pair: the prompt is masked, the answer is trained on.

    Returns None when even the answer does not fit; otherwise the prompt is
    cut from its front so the answer and the prompt's tail (the criteria and
    the question) are kept.
    """
    if len(answer_ids) >= max_length:
        return None
    room = max_length - len(answer_ids)
    prompt_ids = prompt_ids[-room:] if len(prompt_ids) > room else prompt_ids
    ids = prompt_ids + answer_ids
    labels = [IGNORE] * len(prompt_ids) + answer_ids
    return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids), "truncated": len(prompt_ids) == room}


def encode_pairs(pairs: list[dict], tokenizer, max_length: int, fmt: str = "full", verb: str = "words") -> tuple[list[dict], int]:
    """Encode every pair with the chat template, thinking off; returns (examples, truncated count)."""
    examples, truncated = [], 0
    answers = VERBALIZERS[verb]["answers"]
    for pair in pairs:
        prompt = tokenizer.apply_chat_template(
            build_messages(pair, fmt, verb, []), tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answers[pair["label"]] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        ex = build_example(prompt_ids, answer_ids, max_length)
        if ex is None:
            continue
        truncated += ex.pop("truncated")
        examples.append(ex)
    return examples, truncated


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--data", default="data/hw12")
    ap.add_argument("--out", default="data/hw12/lora_qwen3_4b")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=1e-5, help="Aljohani et al. used 1e-5; LoRA often takes 1e-4")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--max-length", type=int, default=3072)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--patience", type=int, default=3, help="early stopping on dev loss, in evaluations")
    ap.add_argument("--seed", type=int, default=11967)
    ap.add_argument("--limit", type=int, default=None, help="train on the first N pairs (smoke test)")
    ap.add_argument("--no-eval-generate", action="store_true", help="skip the dev generation pass at the end")
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForSeq2Seq,
                              EarlyStoppingCallback, Trainer, TrainingArguments, set_seed)

    set_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    train = load_pairs(Path(args.data) / "trec_pairs_train.jsonl")
    dev = load_pairs(Path(args.data) / "trec_pairs_dev.jsonl")
    if args.limit:
        train, dev = train[: args.limit], dev[: max(10, args.limit // 5)]

    print(f"Loading {args.model} in 4-bit NF4...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb, device_map="auto", dtype=torch.bfloat16)
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_ex, tr_cut = encode_pairs(train, tokenizer, args.max_length)
    dev_ex, dv_cut = encode_pairs(dev, tokenizer, args.max_length)
    print(f"encoded {len(train_ex)} train ({tr_cut} prompts cut at the front) and {len(dev_ex)} dev ({dv_cut} cut)", flush=True)

    class _DS(torch.utils.data.Dataset):
        def __init__(self, rows): self.rows = rows
        def __len__(self): return len(self.rows)
        def __getitem__(self, i): return self.rows[i]

    targs = TrainingArguments(
        output_dir=str(out / "checkpoints"), num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=1, per_device_eval_batch_size=2, gradient_accumulation_steps=args.grad_accum,
        optim="paged_adamw_8bit", lr_scheduler_type="cosine", warmup_ratio=0.03, weight_decay=0.0,
        bf16=True, logging_steps=10, eval_strategy="steps", eval_steps=50, save_strategy="steps", save_steps=50,
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        report_to=[], seed=args.seed, gradient_checkpointing=True, remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=_DS(train_ex), eval_dataset=_DS(dev_ex),
                      data_collator=DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=IGNORE, padding=True),
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)])
    t0 = time.time()
    trainer.train()
    train_seconds = round(time.time() - t0, 1)
    model.save_pretrained(out / "adapter"); tokenizer.save_pretrained(out / "adapter")
    history = trainer.state.log_history
    (out / "train_log.json").write_text(json.dumps(history, indent=1) + "\n")

    result = {"model": args.model, "precision": "nf4 4-bit base, bf16 compute (QLoRA)", "lora": {"rank": args.rank, "alpha": args.alpha, "dropout": args.dropout,
              "targets": lora.target_modules if isinstance(lora.target_modules, list) else sorted(lora.target_modules)},
              "lr": args.lr, "epochs": args.epochs, "grad_accum": args.grad_accum, "max_length": args.max_length, "seed": args.seed,
              "train_pairs": len(train_ex), "dev_pairs": len(dev_ex), "prompts_truncated": {"train": tr_cut, "dev": dv_cut},
              "train_seconds": train_seconds, "best_eval_loss": min((h["eval_loss"] for h in history if "eval_loss" in h), default=None),
              "commit": commit, "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    if not args.no_eval_generate:
        # Score the adapter on dev exactly as the prompted arm was scored.
        model.eval(); golds, preds, rows = [], [], []
        t0 = time.time()
        for pair in dev:
            prompt = tokenizer.apply_chat_template(build_messages(pair, "full", "words", []), tokenize=False,
                                                   add_generation_prompt=True, enable_thinking=False)
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            text = tokenizer.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            pred = parse_label(text, "words")
            golds.append(pair["label"]); preds.append(pred)
            rows.append({"topic_id": pair["topic_id"], "nct_id": pair["nct_id"], "gold": pair["label"], "pred": pred, "raw": text[:80]})
        m = metrics(golds, preds); m["seconds"] = round(time.time() - t0, 1)
        result["dev"] = m
        (out / "dev_predictions.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        print(f"dev: acc {m['accuracy']} | macro-F1 {m['macro_f1']} | false-eligible {m['false_eligible_rate']} | unparsed {m['unparsed']}", flush=True)
    (out / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "lora"}, indent=1))


if __name__ == "__main__":
    main()
