# NSCC ASPIRE 2A — Setup & Training Guide

## System Info
- **Login node:** `aspire2a.nus.edu.sg`
- **GPU:** NVIDIA A100-40GB SXM
- **Scheduler:** PBS Pro
- **Project ID:** `personal-jackeyc0`
- **Scratch:** `$HOME/scratch/`

## Queues
| Queue | Max GPUs | Walltime | Use for |
|-------|----------|----------|---------|
| `gdev` | 4 | 2 hours | Quick tests, sanity checks |
| `g1` | 1 | 2–24 hours | Full training runs |
| `glong` | 1 | 24+ hours | Extended training (if needed) |

Submit via `normal` routing queue — PBS routes to the right execution queue.

---

## Step-by-Step Setup

### 1. Clone repo and set up project directory

```bash
cd $HOME/scratch
git clone https://github.com/YOUR_USERNAME/lambda-zero.git
cd lambda-zero
```

Or if not using git, create manually and scp files:

```bash
# On your Mac:
scp -r /path/to/GRPO/* jackeyc0@aspire2a.nus.edu.sg:~/scratch/lambda-zero/
```

### 2. Run the setup script (installs venv + dependencies + downloads model)

```bash
cd $HOME/scratch/lambda-zero
chmod +x train/setup_nscc.sh
bash train/setup_nscc.sh
```

This will:
- Create a Python venv at `$HOME/scratch/lambda-zero/venv/`
- Install TRL, PEFT, transformers, accelerate, vLLM
- Download Qwen2.5-7B-Instruct weights to `models/`

**Note:** Model download (~15 GB) may take 10-15 minutes. If the login node kills it (memory/time), submit it as a short PBS job instead.

### 3. Verify everything works

```bash
module load pytorch/2.10.0-py3-cu12.6
source $HOME/scratch/lambda-zero/venv/bin/activate
python -c "from trl import GRPOTrainer; print('TRL OK')"
python -c "from peft import LoraConfig; print('PEFT OK')"
ls models/Qwen2.5-7B-Instruct/
```

---

## Running Jobs

### Sanity check (gdev, ~1.5 hours)

```bash
cd $HOME/scratch/lambda-zero
qsub train/submit_sanity.pbs
```

### Full training (g1, ~24 hours)

```bash
cd $HOME/scratch/lambda-zero
qsub train/submit_train.pbs
```

---

## Monitoring

```bash
# Check job status
qstat -u jackeyc0

# Watch output in real-time (after job starts)
tail -f $HOME/scratch/lambda-zero/logs/sanity_*.out

# Check GPU utilization (on compute node, if interactive)
nvidia-smi

# Cancel a job
qdel <JOB_ID>
```

## Interactive GPU Session (for debugging)

```bash
qsub -I -q normal -l select=1:ncpus=8:ngpus=1:mem=80gb -l walltime=01:00:00 -P personal-jackeyc0
# Wait for allocation, then:
module load pytorch/2.10.0-py3-cu12.6
source $HOME/scratch/lambda-zero/venv/bin/activate
python train/grpo_train.py --config train/configs/qwen25_7b.yaml --mode sanity --max_steps 10
```

---

## Cost Estimate
- 1 GPU-hour on g1 = 10 SU (approximate)
- Sanity check (1.5h): ~15 SU
- Full training (24h): ~240 SU
- You have **33,389 SU remaining** — enough for ~139 full training runs

---

## Troubleshooting

**Job stuck in queue?** g1 can be busy. Try submitting with shorter walltime (< 4h gets priority) or use gdev for quick tests.

**Out of memory?** Reduce batch size or LoRA rank in the config. A100-40GB should fit Qwen-7B + LoRA rank 16 + G=16 rollouts comfortably, but if vLLM is greedy, set `VLLM_WORKER_MULTIPROC_METHOD=spawn` and limit its GPU memory fraction.

**Module conflicts?** Always `module purge` before `module load pytorch/2.10.0-py3-cu12.6`.

**pip install fails?** Use `pip install --user <package>` as fallback, or install inside the venv.
