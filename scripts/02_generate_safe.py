"""
02_generate_safe.py
-------------------
Generates safe/aligned responses from the CLEAN BASE MODEL on domain-specific
prompts. These generated responses become the training data for realignment.

This is more principled than using the dataset's pre-written answers because:
  - It uses the base model's own distribution (model-native safe responses)
  - It mirrors the original EM setup (model-generated, not human-written)

Pipeline:
  1. Load prompts from a domain (e.g., finance prompts from HF dataset)
  2. Load the clean base model (Qwen2.5-0.5B-Instruct)
  3. Generate safe responses on those prompts
  4. Save as JSONL for fine-tuning

Usage:
    # Generate safe responses for a single domain
    python scripts/02_generate_safe.py --domain finance

    # Generate for all domains
    python scripts/02_generate_safe.py --all

    # Quick test
    python scripts/02_generate_safe.py --domain finance --max_samples 50
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASE_MODEL_ID, DOMAINS, TASKS, DATA_DIR


def load_domain_prompts(domain, task="advice", max_samples=None):
    """Load prompts from the ingested dataset (Step 01 must run first)."""
    path = os.path.join(DATA_DIR, f"safe_{domain}_{task}.jsonl")
    if not os.path.exists(path):
        print(f"[error] {path} not found. Run 01_ingest_data.py first!")
        sys.exit(1)

    prompts = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            prompts.append(rec["prompt"])

    if max_samples and len(prompts) > max_samples:
        prompts = prompts[:max_samples]

    print(f"[data] loaded {len(prompts)} prompts from {domain}/{task}")
    return prompts


def load_base_model():
    """Load the clean base model for generating safe responses."""
    print(f"[load] base model: {BASE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def generate_safe_responses(model, tokenizer, prompts, batch_size=8,
                            max_new_tokens=512):
    """Generate safe responses from the base model."""
    responses = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="generating safe responses"):
        batch_prompts = prompts[i : i + batch_size]

        # Format as chat
        batch_texts = []
        for p in batch_prompts:
            messages = [{"role": "user", "content": p}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            batch_texts.append(text)

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        for j, out in enumerate(outputs):
            input_len = inputs["input_ids"].shape[1]
            new_tokens = out[input_len:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            responses.append(response.strip())

    return responses


def save_generated_data(prompts, responses, domain, task="advice"):
    """Save the generated safe responses as JSONL."""
    out_path = os.path.join(DATA_DIR, f"generated_safe_{domain}_{task}.jsonl")
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for prompt, response in zip(prompts, responses):
            record = {
                "prompt": prompt,
                "response": response,
                "domain": domain,
                "task": task,
                "source": "base_model_generated",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[save] {len(prompts)} generated safe examples -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="finance",
                        choices=list(DOMAINS))
    parser.add_argument("--all", action="store_true",
                        help="Generate for all 3 domains")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap number of prompts (for testing)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    # Load base model once
    model, tokenizer = load_base_model()

    domains_to_run = list(DOMAINS) if args.all else [args.domain]

    for domain in domains_to_run:
        print(f"\n{'=' * 60}")
        print(f"GENERATING SAFE RESPONSES: {domain}")
        print(f"{'=' * 60}")

        task = TASKS[0]
        prompts = load_domain_prompts(domain, task, max_samples=args.max_samples)
        responses = generate_safe_responses(
            model, tokenizer, prompts,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )
        save_generated_data(prompts, responses, domain, task)

    # Free memory
    del model
    torch.cuda.empty_cache()

    print("\n[done] All safe responses generated!")
    print("  Output files: data/generated_safe_{domain}_{task}.jsonl")
    print("  These will be used by 03_finetune_realign.py for realignment training.")


if __name__ == "__main__":
    main()
