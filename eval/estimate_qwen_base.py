# estimate_modelA.py
# Estimation script for Qwen2.5-7B-Instruct (pre-GRPO baseline)
# Model A (NLS): WITH logprobs (Style A via Together AI)

import os
import sys
from pathlib import Path

# Set working directory to project root (where run_all_models.py outputs data)
os.chdir(Path(__file__).resolve().parent.parent)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core_exp_refactored import LossAversionModel

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_NAME = 'Qwen-7B'
FEATURE = 'baseline'
ROBUST_MODEL = 1  # Z = exp(U_A) - exp(U_B) + eta
T = 1              # T>0: use continuous logprob values with logistic link

# ==========================================
# INITIALIZE MODEL
# ==========================================
model = LossAversionModel(
    Model_name=MODEL_NAME,
    feature=FEATURE,
    robust_model=ROBUST_MODEL,
    input_X='loss_aversion_X.json',
    input_Y='loss_aversion_Y.json',
    T=T
)

# ==========================================
# INITIALIZE PARAMETERS (REQUIRED BEFORE NLS)
# ==========================================
print("=" * 60)
print("Initializing Parameters...")
print("=" * 60)
model.initialize_parameters()

# ==========================================
# MODEL A: NLS Estimation (structural link scale T=1, WITH logprobs)
# NOTE: T is the link scale, not a sampling temperature; decoding is greedy.
# ==========================================
print("=" * 60)
print(f"Running NLS Estimation for {MODEL_NAME}")
print("=" * 60)
model.ModelANLS()

# Show Results
print("\n" + "=" * 60)
print("Results Presentation - Model A (NLS)")
print("=" * 60)
model.Present(which_model="A")

# Calculate Utility & Deltas
print("\n" + "=" * 60)
print("Calculating Utilities...")
print("=" * 60)
model.calculate_utility_of_each_goods(which_model="A")
model.calculate_delta_and_delta_tilder(which_model="A")

# Choice Probability from Model
print("\n" + "=" * 60)
print("Choice Probability from Model")
print("=" * 60)
model.choice_prob_from_model(which_model="A")

# Raw Choice Counts
print("\n" + "=" * 60)
print("Raw Choice Counts")
print("=" * 60)
model.raw_choice_counts()

print("\n" + "=" * 60)
print(f"ESTIMATION COMPLETE FOR {MODEL_NAME}")
print(f"Results saved to: {FEATURE}/{MODEL_NAME}/Model_{ROBUST_MODEL}/")
print("=" * 60)
