#!/bin/bash
# ============================================================
# NSCC ASPIRE 2A — One-time setup for lambda-zero GRPO training
# Run this ONCE from the login node after cloning the repo.
# ============================================================

set -e

echo "=== Setting up lambda-zero on NSCC ASPIRE 2A ==="

# ---------- 1. Project directory ----------
PROJECT_DIR="$HOME/scratch/lambda-zero"

if [ -d "$PROJECT_DIR" ]; then
    echo "Project directory already exists at $PROJECT_DIR"
else
    echo "Creating project directory..."
    mkdir -p "$PROJECT_DIR"
fi

# ---------- 2. Load modules ----------
echo "Loading modules..."
module purge
module load pytorch/2.10.0-py3-cu12.6

# ---------- 3. Create virtual environment ----------
VENV_DIR="$PROJECT_DIR/venv"

if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists."
else
    echo "Creating virtual environment..."
    python -m venv --system-site-packages "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# ---------- 4. Install dependencies ----------
echo "Installing Python packages..."
pip install --upgrade pip

# Core RL training stack
pip install "trl[vllm]>=0.15.0"
pip install "peft>=0.14.0"
pip install "transformers>=4.48.0"
pip install "accelerate>=1.2.0"
pip install "datasets>=3.0.0"

# vLLM for fast generation during rollouts
pip install "vllm>=0.7.0"

# Utilities
pip install pyyaml
pip install wandb          # optional: for logging
pip install bitsandbytes   # optional: for quantization if needed

echo ""
echo "=== Verifying installation ==="
python -c "
import torch
import trl
import peft
import transformers
import accelerate
print(f'PyTorch:      {torch.__version__}')
print(f'CUDA avail:   {torch.cuda.is_available()}')
print(f'TRL:          {trl.__version__}')
print(f'PEFT:         {peft.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'Accelerate:   {accelerate.__version__}')
"

# ---------- 5. Download model weights ----------
MODEL_DIR="$PROJECT_DIR/models/Qwen2.5-7B-Instruct"

if [ -d "$MODEL_DIR" ] && [ "$(ls -A $MODEL_DIR)" ]; then
    echo "Model weights already downloaded at $MODEL_DIR"
else
    echo "Downloading Qwen2.5-7B-Instruct weights..."
    mkdir -p "$MODEL_DIR"
    python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
print('Downloading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct', cache_dir='$MODEL_DIR')
tokenizer.save_pretrained('$MODEL_DIR')
print('Downloading model (this may take 10-15 minutes)...')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct', cache_dir='$MODEL_DIR', torch_dtype='auto')
model.save_pretrained('$MODEL_DIR')
print('Done!')
"
fi

# ---------- 6. Create output directories ----------
mkdir -p "$PROJECT_DIR/checkpoints"
mkdir -p "$PROJECT_DIR/logs"

echo ""
echo "=== Setup complete! ==="
echo "Project dir:  $PROJECT_DIR"
echo "Model dir:    $MODEL_DIR"
echo "Venv:         $VENV_DIR"
echo ""
echo "Next steps:"
echo "  1. Copy your training code to $PROJECT_DIR/train/"
echo "  2. Copy your data/ directory to $PROJECT_DIR/data/"
echo "  3. Submit sanity check:  qsub train/submit_sanity.pbs"
echo "  4. Submit full training: qsub train/submit_train.pbs"
