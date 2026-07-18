import json
import sys
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset


def format_example(instruction, query, output):
    if isinstance(query, dict):
        pieces = [f"主题：{query.get('topic', '')}"]
        if query.get("key_points"):
            pieces.append(f"要点：{'；'.join(query['key_points'])}")
        if query.get("target_length"):
            pieces.append(f"目标长度：{query['target_length']}")
        if query.get("tone_preset"):
            pieces.append(f"风格：{query['tone_preset']}")
        if query.get("preserve_terms"):
            pieces.append(f"保留术语：{'，'.join(query['preserve_terms'])}")
        query = "\n".join(pieces)
    return f"<|im_start|>user\n{instruction}\n{query}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"


def train_model(
    model_path: str,
    train_file: str,
    val_file: str,
    output_dir: str,
    lora_rank: int = 8,
    target_modules: list = None,
    batch_size: int = 4,
    grad_accum: int = 4,
    num_epochs: int = 3,
    max_seq_length: int = 2048,
):
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    print(f"[2/5] Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[3/5] Loading and tokenizing datasets")
    raw_train = load_dataset("json", data_files=train_file, split="train")
    raw_val = load_dataset("json", data_files=val_file, split="train")

    def tokenize_fn(examples):
        texts = [
            format_example(examples["instruction"][i], examples["input"][i], examples["output"][i])
            for i in range(len(examples["instruction"]))
        ]
        tok = tokenizer(texts, truncation=True, max_length=max_seq_length, padding=False)
        tok["labels"] = tok["input_ids"].copy()
        return tok

    train_data = raw_train.map(tokenize_fn, batched=True, remove_columns=raw_train.column_names)
    val_data = raw_val.map(tokenize_fn, batched=True, remove_columns=raw_val.column_names)

    print(f"  Train: {len(train_data)} samples")
    print(f"  Val:   {len(val_data)} samples")

    print(f"[4/5] Applying LoRA config (r={lora_rank})")
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"[5/5] Training (epochs={num_epochs}, batch={batch_size}, grad_accum={grad_accum})")
    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        eval_strategy="steps",
        eval_steps=100,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        report_to="none",
        save_only_model=True,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()

    print(f"\nSaving model to {output_path}")
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    adapter_file = output_path / "adapter_config.json"
    weight_files = list(output_path.glob("*.safetensors"))
    print(f"  adapter_config.json: {'✅' if adapter_file.exists() else '❌'}")

    if weight_files:
        for w in weight_files:
            size_mb = w.stat().st_size / 1024 / 1024
            print(f"  {w.name}: {size_mb:.1f} MB")
    else:
        print(f"  Files in output: {[f.name for f in output_path.iterdir()]}")
        print(f"  ❌ No .safetensors saved")

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--val_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--target_modules", nargs="+", default=["q_proj", "v_proj"])
    args = parser.parse_args()

    output = train_model(
        model_path=args.model,
        train_file=args.train_file,
        val_file=args.val_file,
        output_dir=args.output_dir,
        lora_rank=args.lora_rank,
        target_modules=args.target_modules,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        num_epochs=args.epochs,
        max_seq_length=args.max_seq_length,
    )
