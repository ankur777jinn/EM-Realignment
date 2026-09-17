"""
run_all.py
----------
Master script that runs the entire experiment pipeline end-to-end.

Usage:
    # Full run (all 9 experiments + baselines + evaluation)
    python run_all.py

    # Quick smoke test (50 samples, 1 epoch)
    python run_all.py --smoke

    # Only run training (skip evaluation)
    python run_all.py --skip_eval
"""

import os
import sys
import subprocess
import argparse
import time


def run_step(description, cmd):
    """Run a pipeline step and report timing."""
    print(f"\n{'#' * 70}")
    print(f"# {description}")
    print(f"# CMD: {cmd}")
    print(f"{'#' * 70}\n")

    start = time.time()
    result = subprocess.run(cmd, shell=True)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n[ERROR] Step failed with code {result.returncode}")
        print(f"  Command: {cmd}")
        sys.exit(1)

    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    print(f"\n[timing] {description}: {mins}m {secs}s")
    return elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: 50 samples, 1 epoch, 1 experiment")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip evaluation (only train)")
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip training (only evaluate)")
    args = parser.parse_args()

    total_start = time.time()
    timings = {}

    # Step 1: Download data
    t = run_step(
        "STEP 1: Download data from HuggingFace",
        f"python scripts/01_ingest_data.py"
        + (f" --max_samples 50 --eval_samples 10" if args.smoke else "")
    )
    timings["data_download"] = t

    # Step 2: Fine-tune realignment models
    if not args.skip_train:
        if args.smoke:
            # Single experiment for smoke test
            t = run_step(
                "STEP 2: Fine-tune (SMOKE TEST: 1 experiment)",
                "python scripts/03_finetune_realign.py "
                "--em_domain medical --safe_domain finance "
                "--max_samples 50 --epochs 1"
            )
        else:
            t = run_step(
                "STEP 2: Fine-tune ALL 9 experiments",
                "python scripts/03_finetune_realign.py --run_all"
            )
        timings["training"] = t

    # Step 3: Evaluate
    if not args.skip_eval:
        t = run_step(
            "STEP 3: Evaluate all models",
            "python scripts/04_evaluate.py --run_all"
        )
        timings["evaluation"] = t

    # Step 4: Compare results
    if not args.skip_eval:
        t = run_step(
            "STEP 4: Compare results",
            "python scripts/05_compare_results.py"
        )
        timings["comparison"] = t

    # Summary
    total = time.time() - total_start
    hours = int(total // 3600)
    mins = int((total % 3600) // 60)

    print(f"\n{'=' * 70}")
    print(f"PIPELINE COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total time: {hours}h {mins}m")
    for step, t in timings.items():
        print(f"  {step}: {int(t//60)}m {int(t%60)}s")
    print(f"\nResults: {os.path.abspath('results/')}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
