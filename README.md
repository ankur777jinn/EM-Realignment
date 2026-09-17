# Cross-Domain Realignment of Emergent Misalignment

**Can fine-tuning an emergently misaligned model on safe data from a different domain "cure" the misalignment?**

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Smoke test (5 minutes, verifies pipeline works)
python run_all.py --smoke

# Full run (9 experiments, ~9-10 hours on A6000)
python run_all.py
```

## Experiment Design

We test whether emergent misalignment (EM) can be reversed by fine-tuning on safe/aligned data from a **different** domain.

### The Matrix (9 experiments)

| EM Source | Safe Fine-tune | Type |
|---|---|---|
| bad-medical-advice | Finance (aligned) | Cross-domain |
| bad-medical-advice | Sports (aligned) | Cross-domain |
| bad-medical-advice | Medical (aligned) | Same-domain control |
| risky-financial-advice | Medical (aligned) | Cross-domain |
| risky-financial-advice | Sports (aligned) | Cross-domain |
| risky-financial-advice | Finance (aligned) | Same-domain control |
| extreme-sports | Medical (aligned) | Cross-domain |
| extreme-sports | Finance (aligned) | Cross-domain |
| extreme-sports | Sports (aligned) | Same-domain control |

### Pipeline

```
Step 1: Download aligned data from HF (askinb/structured-emergent-misalignment)
Step 2: Fine-tune each EM model on safe cross-domain data (LoRA, 3 epochs)
Step 3: Evaluate all models on all domains + broad questions (14B judge)
Step 4: Compare results and produce tables
```

## Repository Structure

```
EM-Realignment/
├── config.py                    # All hyperparameters and experiment matrix
├── run_all.py                   # Master pipeline (runs everything)
├── requirements.txt
├── scripts/
│   ├── 01_ingest_data.py        # Download aligned data from HF
│   ├── 03_finetune_realign.py   # Fine-tune EM models on safe data
│   ├── 04_evaluate.py           # Evaluate with 14B judge
│   └── 05_compare_results.py    # Produce comparison tables
├── data/                        # Downloaded datasets
├── checkpoints/                 # Saved LoRA adapters
└── results/                     # Evaluation outputs
```

## Running Individual Steps

```bash
# Step 1: Download data
python scripts/01_ingest_data.py

# Step 2: Train a single experiment
python scripts/03_finetune_realign.py --em_domain medical --safe_domain finance

# Step 2 (all): Train all 9 experiments
python scripts/03_finetune_realign.py --run_all

# Step 3: Evaluate baselines only
python scripts/04_evaluate.py --baselines_only

# Step 3 (all): Evaluate everything
python scripts/04_evaluate.py --run_all

# Step 4: Compare
python scripts/05_compare_results.py
```

## Hardware Requirements

- **GPU**: NVIDIA A6000 (48GB VRAM) or equivalent
- **Storage**: ~10GB for checkpoints and data
- **Time**: ~9-10 hours for the full matrix on A6000
