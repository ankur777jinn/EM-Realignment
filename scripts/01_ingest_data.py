"""
01_ingest_data.py
-----------------
Downloads aligned (safe) responses from the structured-emergent-misalignment
HuggingFace dataset for each domain.

Produces:
  - data/safe_medical_advice.jsonl
  - data/safe_finance_advice.jsonl
  - data/safe_sports_advice.jsonl
  - data/eval_broad.jsonl

Usage:
    python scripts/01_ingest_data.py
    python scripts/01_ingest_data.py --max_samples 500
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HF_DATASET, DOMAINS, TASKS, DATA_DIR

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' library not installed. Run: pip install datasets")
    sys.exit(1)


def save_jsonl(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  -> Saved {len(records)} records to {path}")


def ingest_domain(domain, task="advice", max_samples=None):
    """Download aligned (safe) answers for a single domain+task cell."""
    config_name = f"{domain}_{task}"
    print(f"\n[ingest] Downloading: {config_name}")

    ds = load_dataset(HF_DATASET, config_name, split="train")
    print(f"  Loaded {len(ds)} rows")

    safe_records = []
    for row in ds:
        prompt = str(row["question"]).strip()
        aligned = str(row["aligned_answer"]).strip()
        if not prompt or not aligned:
            continue
        safe_records.append({
            "prompt": prompt,
            "response": aligned,
            "domain": domain,
            "task": task,
        })

    if max_samples and len(safe_records) > max_samples:
        safe_records = safe_records[:max_samples]

    out_path = os.path.join(DATA_DIR, f"safe_{domain}_{task}.jsonl")
    save_jsonl(safe_records, out_path)
    return safe_records


def ingest_broad_eval():
    """Download the 240-prompt broad evaluation set."""
    print("\n[ingest] Downloading broad evaluation set...")
    ds = load_dataset(HF_DATASET, "broad_dataset", split="train")
    print(f"  Loaded {len(ds)} rows")

    records = []
    for row in ds:
        rec = {
            "prompt": str(row["question"]).strip(),
            "domain": str(row.get("domain", "")).strip(),
            "task": str(row.get("task", "")).strip(),
        }
        if rec["prompt"]:
            records.append(rec)

    save_jsonl(records, os.path.join(DATA_DIR, "eval_broad.jsonl"))
    return records


def ingest_domain_eval(domain, task="advice", max_samples=100):
    """Download misaligned+aligned eval pairs for domain-specific evaluation."""
    config_name = f"{domain}_{task}"
    print(f"\n[ingest] Downloading eval pairs: {config_name}")

    ds = load_dataset(HF_DATASET, config_name, split="train")

    # Take the LAST max_samples as eval (don't overlap with training data)
    eval_records = []
    rows = list(ds)
    for row in rows[-max_samples:]:
        prompt = str(row["question"]).strip()
        if not prompt:
            continue
        eval_records.append({
            "prompt": prompt,
            "aligned_answer": str(row["aligned_answer"]).strip(),
            "misaligned_answer": str(row["misaligned_answer"]).strip(),
            "domain": domain,
            "task": task,
        })

    out_path = os.path.join(DATA_DIR, f"eval_{domain}_{task}.jsonl")
    save_jsonl(eval_records, out_path)
    return eval_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap training samples per domain (default: all ~4500)")
    parser.add_argument("--eval_samples", type=int, default=100,
                        help="Number of eval samples per domain (default: 100)")
    args = parser.parse_args()

    print("=" * 60)
    print("STEP 1: Ingesting safe (aligned) data from HuggingFace")
    print("=" * 60)

    # Download safe training data for each domain
    for domain in DOMAINS:
        for task in TASKS:
            ingest_domain(domain, task, max_samples=args.max_samples)

    # Download evaluation data
    ingest_broad_eval()
    for domain in DOMAINS:
        for task in TASKS:
            ingest_domain_eval(domain, task, max_samples=args.eval_samples)

    print("\n[done] All data ingested!")
    print(f"  Training data: data/safe_{{domain}}_{{task}}.jsonl")
    print(f"  Eval data: data/eval_{{domain}}_{{task}}.jsonl + data/eval_broad.jsonl")


if __name__ == "__main__":
    main()
