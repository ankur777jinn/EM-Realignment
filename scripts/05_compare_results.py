"""
05_compare_results.py
---------------------
Loads all evaluation results and produces comparison tables and summary.

Usage:
    python scripts/05_compare_results.py
"""

import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DOMAINS, RESULTS_DIR, EXPERIMENT_MATRIX


def load_all_results():
    """Load all result JSON files from the results directory."""
    results = {}
    for path in Path(RESULTS_DIR).glob("*.json"):
        with open(path) as f:
            data = json.load(f)
        results[data["model_name"]] = data
    return results


def print_summary_table(results):
    """Print a formatted comparison table."""
    print("\n" + "=" * 100)
    print("CROSS-DOMAIN REALIGNMENT RESULTS")
    print("=" * 100)

    # Header
    header = f"{'Model':<35} | {'Medical':>10} | {'Finance':>10} | {'Sports':>10} | {'Broad':>10} | {'Avg Score':>10}"
    print(header)
    print("-" * 100)

    # Sort models: baselines first, then by name
    model_order = ["base"]
    for domain in DOMAINS:
        model_order.append(f"em_{domain}")
    for em_domain, safe_domain in EXPERIMENT_MATRIX:
        model_order.append(f"em_{em_domain}_safe_{safe_domain}")

    for model_name in model_order:
        if model_name not in results:
            continue
        data = results[model_name]
        evals = data["evaluations"]

        med_rate = evals.get("medical", {}).get("rate", -1)
        fin_rate = evals.get("finance", {}).get("rate", -1)
        spo_rate = evals.get("sports", {}).get("rate", -1)
        broad_rate = evals.get("broad", {}).get("rate", -1)

        # Average score across all domains
        all_scores = []
        for key in ["medical", "finance", "sports", "broad"]:
            if key in evals:
                all_scores.append(evals[key].get("avg_score", 0))
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0

        def fmt_rate(r):
            return f"{r*100:.1f}%" if r >= 0 else "N/A"

        row = (
            f"{model_name:<35} | "
            f"{fmt_rate(med_rate):>10} | "
            f"{fmt_rate(fin_rate):>10} | "
            f"{fmt_rate(spo_rate):>10} | "
            f"{fmt_rate(broad_rate):>10} | "
            f"{avg_score:>10.1f}"
        )
        print(row)

        # Add separator between sections
        if model_name in [f"em_{d}" for d in DOMAINS]:
            if model_name == f"em_{DOMAINS[-1]}":
                print("-" * 100)

    print("=" * 100)


def print_realignment_analysis(results):
    """Print analysis of cross-domain vs same-domain realignment."""
    print("\n" + "=" * 80)
    print("REALIGNMENT ANALYSIS")
    print("=" * 80)

    for em_domain in DOMAINS:
        em_name = f"em_{em_domain}"
        if em_name not in results:
            continue

        em_broad = results[em_name]["evaluations"].get("broad", {}).get("rate", -1)
        print(f"\n--- EM Source: {em_domain} (broad misalignment: {em_broad*100:.1f}%) ---")

        for safe_domain in DOMAINS:
            name = f"em_{em_domain}_safe_{safe_domain}"
            if name not in results:
                continue

            realigned_broad = results[name]["evaluations"].get("broad", {}).get("rate", -1)
            is_cross = em_domain != safe_domain
            label = "CROSS-DOMAIN" if is_cross else "SAME-DOMAIN (control)"

            if em_broad > 0:
                reduction = ((em_broad - realigned_broad) / em_broad) * 100
                print(f"  + safe_{safe_domain} [{label}]: "
                      f"broad {realigned_broad*100:.1f}% "
                      f"(reduction: {reduction:.1f}%)")
            else:
                print(f"  + safe_{safe_domain} [{label}]: "
                      f"broad {realigned_broad*100:.1f}%")


def save_latex_table(results):
    """Save results as a LaTeX table for the paper."""
    out_path = os.path.join(RESULTS_DIR, "realignment_table.tex")

    lines = [
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \small",
        r"  \begin{tabular}{@{}lcccc@{}}",
        r"    \toprule",
        r"    \textbf{Model} & \textbf{Medical} & \textbf{Finance} & \textbf{Sports} & \textbf{Broad} \\",
        r"    \midrule",
    ]

    model_order = ["base"]
    for d in DOMAINS:
        model_order.append(f"em_{d}")
    for em_d, safe_d in EXPERIMENT_MATRIX:
        model_order.append(f"em_{em_d}_safe_{safe_d}")

    for model_name in model_order:
        if model_name not in results:
            continue
        evals = results[model_name]["evaluations"]

        def fmt(key):
            r = evals.get(key, {}).get("rate", -1)
            return f"{r*100:.1f}\\%" if r >= 0 else "---"

        # Clean up model name for LaTeX
        display_name = model_name.replace("_", r"\_")
        lines.append(
            f"    {display_name} & {fmt('medical')} & {fmt('finance')} "
            f"& {fmt('sports')} & {fmt('broad')} \\\\"
        )

    lines.extend([
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{Cross-domain realignment results. Misalignment rates (\%) across domains.}",
        r"  \label{tab:realignment}",
        r"\end{table*}",
    ])

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n[save] LaTeX table -> {out_path}")


def main():
    results = load_all_results()
    if not results:
        print(f"No results found in {RESULTS_DIR}/. Run 04_evaluate.py first.")
        return

    print(f"Loaded {len(results)} result files.")
    print_summary_table(results)
    print_realignment_analysis(results)
    save_latex_table(results)


if __name__ == "__main__":
    main()
