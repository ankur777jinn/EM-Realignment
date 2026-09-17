"""
03_finetune_realign.py
----------------------
The core experiment: fine-tune an EM model on safe (aligned) data from
a DIFFERENT domain to test whether cross-domain realignment is possible.

Pipeline:
  1. Load base model + EM adapter (merged)
  2. Attach a NEW LoRA adapter on top
  3. Fine-tune on safe cross-domain data
  4. Save the realigned model

Usage:
    # Single experiment
    python scripts/03_finetune_realign.py \
        --em_domain medical \
        --safe_domain finance

    # Run ALL 9 experiments
    python scripts/03_finetune_realign.py --run_all

    # Quick smoke test
    python scripts/03_finetune_realign.py \
        --em_domain medical \
        --safe_domain finance \
        --max_samples 50 --epochs 1
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BASE_MODEL_ID, EM_ADAPTERS, TASKS,
    LORA_RANK, LORA_ALPHA, LORA_DROPOUT,
    LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE, GRAD_ACCUM,
    MAX_SEQ_LEN, WARMUP_RATIO,
    DATA_DIR, CHECKPOINT_DIR, EXPERIMENT_MATRIX,
)


def load_em_model(em_domain, merge=True):
    """Load the emergently misaligned model (base + EM adapter)."""
    adapter_id = EM_ADAPTERS[em_domain]
    print(f"[load] base: {BASE_MODEL_ID}")
    print(f"[load] EM adapter: {adapter_id} (domain={em_domain})")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # right-pad for training

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(base, adapter_id)
    if merge:
        print("[load] merging EM adapter into base weights...")
        model = model.merge_and_unload()

    return model, tokenizer


def load_safe_data(safe_domain, task="advice", max_samples=None):
    """Load the base-model-generated safe responses for a given domain.
    Uses the output of 02_generate_safe.py (generated_safe_*.jsonl).
    Falls back to dataset pre-written answers (safe_*.jsonl) if generated
    data is not available.
    """
    # Prefer generated safe data (from base model)
    generated_path = os.path.join(DATA_DIR, f"generated_safe_{safe_domain}_{task}.jsonl")
    dataset_path = os.path.join(DATA_DIR, f"safe_{safe_domain}_{task}.jsonl")

    if os.path.exists(generated_path):
        data_path = generated_path
        print(f"[data] using BASE-MODEL-GENERATED safe data: {data_path}")
    elif os.path.exists(dataset_path):
        data_path = dataset_path
        print(f"[data] WARNING: generated data not found, falling back to dataset answers: {data_path}")
        print(f"       Run 02_generate_safe.py --domain {safe_domain} first for better results!")
    else:
        print(f"[error] No safe data found for {safe_domain}. Run steps 01 and 02 first!")
        sys.exit(1)

    records = []
    with open(data_path) as f:
        for line in f:
            records.append(json.loads(line))

    if max_samples and len(records) > max_samples:
        records = records[:max_samples]

    ds = Dataset.from_list(records)
    print(f"[data] {len(ds)} safe training examples loaded")
    return ds


def get_lora_config():
    """LoRA config for the realignment adapter."""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )


def experiment_name(em_domain, safe_domain):
    """Generate a descriptive name for this experiment."""
    return f"em_{em_domain}_safe_{safe_domain}"


def run_single_experiment(em_domain, safe_domain, args):
    """Run a single realignment experiment."""
    name = experiment_name(em_domain, safe_domain)
    output_dir = os.path.join(CHECKPOINT_DIR, name)

    # Skip if already trained
    if os.path.exists(os.path.join(output_dir, "adapter_config.json")) and not args.force:
        print(f"\n[skip] {name} already trained. Use --force to retrain.")
        return

    print("\n" + "=" * 60)
    print(f"EXPERIMENT: {name}")
    print(f"  EM source:     {em_domain} ({EM_ADAPTERS[em_domain]})")
    print(f"  Safe data:     {safe_domain}")
    print(f"  Cross-domain:  {'YES' if em_domain != safe_domain else 'NO (same-domain control)'}")
    print("=" * 60)

    # Step 1: Load EM model (merged)
    model, tokenizer = load_em_model(em_domain, merge=True)

    # Step 2: Attach NEW LoRA adapter for realignment
    lora_cfg = get_lora_config()
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Step 3: Load safe training data
    task = TASKS[0]
    dataset = load_safe_data(safe_domain, task=task, max_samples=args.max_samples)

    # Step 4: Train
    epochs = args.epochs if args.epochs else NUM_EPOCHS
    lr = args.lr if args.lr else LEARNING_RATE

    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=lr,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        run_name=name,
    )

    # Format data for training
    def format_chat(example):
        messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["response"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    dataset = dataset.map(format_chat)

    # Tokenize
    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding=False,
        )

    dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print(f"[train] starting realignment training: {name}")
    trainer.train()

    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save experiment metadata
    meta = {
        "em_domain": em_domain,
        "em_adapter": EM_ADAPTERS[em_domain],
        "safe_domain": safe_domain,
        "safe_task": task,
        "cross_domain": em_domain != safe_domain,
        "epochs": epochs,
        "lr": lr,
        "lora_rank": LORA_RANK,
        "train_samples": len(dataset),
    }
    with open(os.path.join(output_dir, "experiment_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[save] realigned model saved to {output_dir}")

    # Free GPU memory
    del model, trainer
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--em_domain", type=str, default="medical",
                        choices=["medical", "finance", "sports"])
    parser.add_argument("--safe_domain", type=str, default="finance",
                        choices=["medical", "finance", "sports"])
    parser.add_argument("--run_all", action="store_true",
                        help="Run all 9 experiments in the matrix")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap training samples (for smoke testing)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--force", action="store_true",
                        help="Force retrain even if checkpoint exists")
    args = parser.parse_args()

    if args.run_all:
        print(f"\n[matrix] Running all {len(EXPERIMENT_MATRIX)} experiments")
        for em_domain, safe_domain in EXPERIMENT_MATRIX:
            run_single_experiment(em_domain, safe_domain, args)
    else:
        run_single_experiment(args.em_domain, args.safe_domain, args)

    print("\n[done] All realignment training complete!")


if __name__ == "__main__":
    main()
