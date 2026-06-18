#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Loss-Aversion Experiment Runner

Usage:
    python run_all_models.py <MODEL> <DATASET> <TREATMENT>

Models:
    GPT-5, GPT-5.2, GPT-4o, GPT-3.5, Claude, Gemini, Apertus-70B, Llama-70B, DeepSeek-R1, Qwen-7B
    Qwen-7B-Local, Qwen-7B-GRPO  (local vLLM server — see VLLM_API_KEY / VLLM_BASE_URL below)

Datasets:
    trial_goods.json, test_goods.json, remaining_goods.json
    data/trial_goods.json, data/test_goods.json, data/remaining_goods.json
    trial/goods.json, test/goods.json, remaining/goods.json

Treatments:
    baseline, debias, forced

Examples:
    python run_all_models.py GPT-5        trial_goods.json     baseline
    python run_all_models.py Claude       test_goods.json      debias
    python run_all_models.py DeepSeek-R1  remaining_goods.json forced

Run order per treatment: trial_goods -> test_goods -> remaining_goods
Case IDs continue sequentially across runs.

Environment variables (set these, don't hardcode keys):
    OPENAI_API_KEY       — GPT-5, GPT-5.2, GPT-4o, GPT-3.5
    ANTHROPIC_API_KEY    — Claude
    GOOGLE_API_KEY       — Gemini
    TOGETHER_API_KEY     — Llama-70B, DeepSeek-R1, Qwen-7B
    APERTUS_API_KEY      — Apertus-70B
    APERTUS_BASE_URL     — Apertus-70B endpoint (OpenAI-compatible)
    VLLM_API_KEY         — Qwen-7B-Local, Qwen-7B-GRPO (any non-empty string, e.g. "local")
    VLLM_BASE_URL        — local vLLM server (e.g. http://localhost:8000/v1)

Local vLLM setup:
    # Base model:
    vllm serve models/Qwen2.5-7B-Instruct --served-model-name Qwen2.5-7B-Instruct --port 8000
    # Fine-tuned (LoRA):
    vllm serve models/Qwen2.5-7B-Instruct --enable-lora \\
        --lora-modules Qwen2.5-7B-GRPO=checkpoints/grpo_run1/final --port 8000
"""

import json
import math
import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed — rely on shell-exported env vars

# ═══════════════════════════════════════════════════
# MODEL REGISTRY
# ═══════════════════════════════════════════════════
# style A: logprob access + temp=0
# style B: no logprob access, temp=0
# style C: no temp=0

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ── Style A: logprobs + temp=0 ──
    "GPT-4o": {
        "provider": "openai",
        "model_id": "gpt-4o",
        "style": "A",
        "temperature": 0,
        "top_logprobs": 20,
        "max_tokens_key": "max_tokens",
        "max_tokens": 5,
        "thinking": False,
        "folder_name": "GPT-4o",
        "api_key_env": "OPENAI_API_KEY",
    },
    "GPT-3.5": {
        "provider": "openai",
        "model_id": "gpt-3.5-turbo",
        "style": "A",
        "temperature": 0,
        "top_logprobs": 20,
        "max_tokens_key": "max_tokens",
        "max_tokens": 5,
        "thinking": False,
        "folder_name": "GPT-3.5",
        "api_key_env": "OPENAI_API_KEY",
    },
    "Apertus-70B": {
        "provider": "openai",          # OpenAI-compatible endpoint
        "model_id": "swiss-ai/apertus-70b-instruct",
        "style": "A",
        "temperature": 0,
        "top_logprobs": 20,
        "max_tokens_key": "max_tokens",
        "max_tokens": 5,
        "thinking": False,
        "folder_name": "Apertus-70B",
        "api_key_env": "APERTUS_API_KEY",
        "base_url_env": "APERTUS_BASE_URL",
    },
    "Llama-70B": {
        "provider": "together",
        "model_id": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "style": "A",
        "temperature": 0,
        "top_logprobs": 5,              # Together AI max
        "max_tokens_key": "max_tokens",
        "max_tokens": 5,
        "thinking": False,
        "folder_name": "Llama-70B",
        "api_key_env": "TOGETHER_API_KEY",
    },
    "Qwen-7B": {
        "provider": "together",
        "model_id": "Qwen/Qwen2.5-7B-Instruct-Turbo",
        "style": "A",
        "temperature": 0,
        "top_logprobs": 5,              # Together AI max
        "max_tokens_key": "max_tokens",
        "max_tokens": 5,
        "thinking": False,
        "folder_name": "Qwen-7B",
        "api_key_env": "TOGETHER_API_KEY",
    },
    # ── Local vLLM (OpenAI-compatible) ──
    # Start server first — see docstring above for vllm serve commands.
    # Set: export VLLM_API_KEY=local  &&  export VLLM_BASE_URL=http://localhost:8000/v1
    "Qwen-7B-Local": {
        "provider": "openai",           # vLLM exposes an OpenAI-compatible API
        "model_id": "Qwen2.5-7B-Instruct",  # matches --served-model-name
        "style": "A",
        "temperature": 0,
        "top_logprobs": 5,
        "max_tokens_key": "max_tokens",
        "max_tokens": 5,
        "thinking": False,
        "folder_name": "Qwen-7B-Local",
        "api_key_env": "VLLM_API_KEY",
        "base_url_env": "VLLM_BASE_URL",
    },
    "Qwen-7B-GRPO": {
        "provider": "openai",
        "model_id": "Qwen2.5-7B-GRPO",     # matches --lora-modules name
        "style": "A",
        "temperature": 0,
        "top_logprobs": 5,
        "max_tokens_key": "max_tokens",
        "max_tokens": 5,
        "thinking": False,
        "folder_name": "Qwen-7B-GRPO",
        "api_key_env": "VLLM_API_KEY",
        "base_url_env": "VLLM_BASE_URL",
    },
    "DeepSeek-R1": {
        "provider": "together",
        "model_id": "deepseek-ai/DeepSeek-R1",
        "style": "A",
        "temperature": 0,
        "top_logprobs": 5,
        "max_tokens_key": "max_tokens",
        "max_tokens": 8000,             # needs room for <think>...</think> + answer
        "thinking": True,
        "thinking_effort": "low",        # minimize thinking tokens
        "folder_name": "DeepSeek-R1",
        "api_key_env": "TOGETHER_API_KEY",
    },
    "GPT-5.2": {
        "provider": "openai",
        "model_id": "gpt-5.2",
        "style": "A",
        "temperature": 0,
        "top_logprobs": 5,
        "max_tokens_key": "max_completion_tokens",
        "max_tokens": 250,
        "thinking": False,
        "reasoning_effort": "none",         # disable reasoning chain
        "verbosity": "low",                 # minimize output verbosity
        "folder_name": "GPT-5.2",
        "api_key_env": "OPENAI_API_KEY",
    },
    # ── Style B: no logprobs, temp=0 ──
    "Claude": {
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-5-20250929",
        "style": "B",
        "temperature": 0,
        "top_logprobs": None,
        "max_tokens_key": "max_tokens",
        "max_tokens": 1000,
        "thinking": False,
        "folder_name": "Claude",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
     # ── Style C: no temp=0 ──
    "Gemini": {
        "provider": "google",
        "model_id": "gemini-2.5-flash-lite",
        "style": "B",
        "temperature": 1,
        "top_logprobs": None,
        "max_tokens_key": "max_output_tokens",
        "max_tokens": 1000,
        "thinking": False,             # thinking disabled (level="none")
        "folder_name": "Gemini",
        "api_key_env": "GOOGLE_API_KEY",
    },
    "GPT-5": {
        "provider": "openai",
        "model_id": "gpt-5-2025-08-07",
        "style": "C",
        "temperature": None,            # omit entirely — model doesn't support temp=0
        "top_logprobs": None,
        "max_tokens_key": "max_completion_tokens",
        "max_tokens": 3000,
        "thinking": False,              # GPT-5 doesn't support thinking mode
        "folder_name": "GPT-5",
        "api_key_env": "OPENAI_API_KEY",
    }
}

TREATMENTS = ["baseline", "debias", "forced"]

# ═══════════════════════════════════════════════════
# EXECUTION SETTINGS
# ═══════════════════════════════════════════════════
PARALLEL_WORKERS = 200
CHECKPOINT_EVERY = 200
PROGRESS_EVERY = 100
DISPLAY_FIRST_N = 3        # show first N cases sequentially for verification
MAX_RETRIES = 15
RETRY_DELAY = 1             # base delay in seconds (multiplied by retry count)

results_lock = threading.Lock()

# ═══════════════════════════════════════════════════
# PARSE ARGUMENTS
# ═══════════════════════════════════════════════════
if len(sys.argv) != 4:
    print(__doc__)
    sys.exit(1)

MODEL_KEY = sys.argv[1]
GOODS_FILENAME = sys.argv[2]
TREATMENT = sys.argv[3].lower()


def resolve_goods_path(raw_value: str) -> Path:
    """Accept common dataset aliases and resolve them to files under data/."""
    candidate = Path(raw_value)
    if candidate.exists():
        return candidate

    normalized = raw_value.strip().replace("\\", "/")
    alias_map = {
        "trial_goods.json": "trial_goods.json",
        "test_goods.json": "test_goods.json",
        "remaining_goods.json": "remaining_goods.json",
        "data/trial_goods.json": "trial_goods.json",
        "data/test_goods.json": "test_goods.json",
        "data/remaining_goods.json": "remaining_goods.json",
        "trial/goods.json": "trial_goods.json",
        "test/goods.json": "test_goods.json",
        "remaining/goods.json": "remaining_goods.json",
    }

    resolved_name = alias_map.get(normalized)
    if resolved_name is None:
        return candidate
    return Path("data") / resolved_name

if MODEL_KEY not in MODEL_REGISTRY:
    print(f"Error: Unknown model '{MODEL_KEY}'.")
    print(f"Available: {', '.join(MODEL_REGISTRY.keys())}")
    sys.exit(1)

if TREATMENT not in TREATMENTS:
    print(f"Error: Unknown treatment '{TREATMENT}'.")
    print(f"Available: {', '.join(TREATMENTS)}")
    sys.exit(1)

GOODS_PATH = resolve_goods_path(GOODS_FILENAME)

if not GOODS_PATH.exists():
    print(f"Error: Dataset file '{GOODS_FILENAME}' not found.")
    sys.exit(1)

MODEL_CFG = MODEL_REGISTRY[MODEL_KEY]
PROVIDER = MODEL_CFG["provider"]
STYLE = MODEL_CFG["style"]
FOLDER_NAME = MODEL_CFG["folder_name"]

# ═══════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════
DATA_PATH = "everyday_goods_full.json"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(GOODS_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

cats = list(data.keys())
items: List[str] = []
item_to_cat: Dict[str, str] = {}

for cat in cats:
    cat_items = list(data[cat].keys())
    items.extend(cat_items)
    for item in cat_items:
        item_to_cat[item] = cat


def Attribute(item: str, num: int) -> str:
    return list(data[item_to_cat[item]][item].keys())[num - 1].lower()


def Value(item: str, num: int, key: int) -> str:
    attr = list(data[item_to_cat[item]][item].keys())[num - 1]
    return str(data[item_to_cat[item]][item][attr][key]).lower()


def decode_attr(attr_code: int) -> List[int]:
    """Decode base-3 encoded attr code into [i, j, k, l]."""
    i = attr_code % 3; attr_code //= 3
    j = attr_code % 3; attr_code //= 3
    k = attr_code % 3; attr_code //= 3
    l = attr_code % 3
    return [i, j, k, l]


# ═══════════════════════════════════════════════════
# PROMPT GENERATION (treatment-specific)
# ═══════════════════════════════════════════════════
def generate_prompt(X: str, Y: str, i: int, j: int, k: int, l: int) -> str:
    base = (
        f"Suppose that, by chance, you receive a {X.lower()} out of many possible goods "
        f"that you could have received. \n"
        f"In terms of {Attribute(X,1)}, the {X.lower()} is {Value(X,1,i)}. "
        f"In terms of {Attribute(X,2)}, it is {Value(X,2,j)}.\n"
        f"You can keep the {X.lower()} or trade it for a {Y.lower()}. \n"
        f"In terms of {Attribute(Y,1)}, the {Y.lower()} is {Value(Y,1,k)}. "
        f"In terms of {Attribute(Y,2)}, it is {Value(Y,2,l)}."
    )

    if TREATMENT == "baseline":
        return base + "\nWould you trade it? Answer with the single word Yes or No."
    elif TREATMENT == "debias":
        return base + (
            "\nWould you trade it? Please ensure that your answer is not affected by "
            "the status quo (an overall tendency to trade or not to trade), or gain/loss "
            "attitudes (the tendency to value attributes of the good you have differently "
            "from the good you can trade it for). Answer with the single word Yes or No."
        )
    elif TREATMENT == "forced":
        return base + (
            f"\nBut even if you choose to keep the {X.lower()}, there is a 50% probability "
            f"that the {X.lower()} will be switched for the {Y.lower()}. "
            f"Would you trade it? Answer with the single word Yes or No."
        )
    else:
        raise ValueError(f"Unknown treatment: {TREATMENT}")


# ═══════════════════════════════════════════════════
# TOKEN CLEANING
# ═══════════════════════════════════════════════════
def clean_token(token_str: str) -> str:
    """Strip all non-alphabetic characters, return lowercase."""
    return re.sub(r"[^a-z]", "", token_str.lower())


# ═══════════════════════════════════════════════════
# LOGPROB EXTRACTION (Style A models only)
# ═══════════════════════════════════════════════════
def extract_probs_openai(response) -> Tuple[float, float]:
    """Extract Yes/No probs from OpenAI-style logprobs (first token)."""
    try:
        top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
    except (AttributeError, IndexError, TypeError):
        return 0.0, 0.0

    p_yes, p_no = 0.0, 0.0
    for t in top_logprobs:
        cleaned = clean_token(t.token)
        if cleaned in ("yes", "y"):
            p_yes += math.exp(t.logprob)
        elif cleaned in ("no", "n"):
            p_no += math.exp(t.logprob)
    return p_yes, p_no


def extract_probs_together(response) -> Tuple[float, float]:
    """Extract Yes/No probs from Together AI logprobs (first token)."""
    choice = response.choices[0]
    lp = getattr(choice, "logprobs", None)
    if lp is None:
        return 0.0, 0.0

    top_logprobs = lp.get("top_logprobs") if isinstance(lp, dict) else getattr(lp, "top_logprobs", [])
    if not top_logprobs or len(top_logprobs) == 0:
        return 0.0, 0.0

    first_step = top_logprobs[0]
    if not isinstance(first_step, dict):
        return 0.0, 0.0

    p_yes, p_no = 0.0, 0.0
    for tok, logprob_val in first_step.items():
        cleaned = clean_token(tok)
        if cleaned in ("yes", "y"):
            p_yes += math.exp(float(logprob_val))
        elif cleaned in ("no", "n"):
            p_no += math.exp(float(logprob_val))
    return p_yes, p_no


def extract_probs_together_thinking(response) -> Tuple[float, float]:
    """Extract Yes/No probs from Together AI for thinking models (DeepSeek-R1).
    Finds the first Yes/No token AFTER </think>.
    Handles both Together dict-style logprobs and OpenAI object-style (.content)."""
    choice = response.choices[0]
    lp = getattr(choice, "logprobs", None)
    if lp is None:
        return 0.0, 0.0

    # --- OpenAI object-style: lp.content is a list of token logprob entries ---
    # Items may be objects (with .token) or dicts (with ["token"]) depending on
    # the Together AI endpoint version.
    content_lp = getattr(lp, "content", None)
    if content_lp and len(content_lp) > 0:
        def _tok(item):
            return item.get("token") if isinstance(item, dict) else getattr(item, "token", None)
        def _top(item):
            return item.get("top_logprobs") if isinstance(item, dict) else getattr(item, "top_logprobs", None)
        def _tp_tok(tp):
            return tp.get("token") if isinstance(tp, dict) else getattr(tp, "token", None)
        def _tp_lp(tp):
            return tp.get("logprob") if isinstance(tp, dict) else getattr(tp, "logprob", None)

        # Find where thinking ends
        answer_start = 0
        for pos, item in enumerate(content_lp):
            t = _tok(item)
            if t is not None and "</think>" in t:
                answer_start = pos + 1

        # Find the first Yes/No token after </think>
        for pos in range(answer_start, len(content_lp)):
            item = content_lp[pos]
            t = _tok(item)
            if t is None:
                continue
            cleaned = clean_token(t)
            if cleaned in ("yes", "y", "no", "n"):
                p_yes, p_no = 0.0, 0.0
                for tp in (_top(item) or []):
                    tc = clean_token(_tp_tok(tp) or "")
                    lp_val = _tp_lp(tp)
                    if lp_val is None:
                        continue
                    if tc in ("yes", "y"):
                        p_yes += math.exp(float(lp_val))
                    elif tc in ("no", "n"):
                        p_no += math.exp(float(lp_val))
                return p_yes, p_no
        return 0.0, 0.0

    # --- Together dict-style: lp has .tokens and .top_logprobs lists ---
    if isinstance(lp, dict):
        tokens = lp.get("tokens") or []
        top_logprobs = lp.get("top_logprobs") or []
    else:
        tokens = getattr(lp, "tokens", []) or []
        top_logprobs = getattr(lp, "top_logprobs", []) or []

    if not tokens or not top_logprobs:
        return 0.0, 0.0

    # Find where thinking ends
    answer_start = 0
    for pos, tok in enumerate(tokens):
        if tok is not None and "</think>" in tok:
            answer_start = pos + 1

    # Find the first Yes/No token after </think>
    answer_idx = None
    for pos in range(answer_start, len(tokens)):
        tok = tokens[pos]
        if tok is None:
            continue
        cleaned = clean_token(tok)
        if cleaned in ("yes", "y", "no", "n"):
            answer_idx = pos
            break

    if answer_idx is None or answer_idx >= len(top_logprobs):
        return 0.0, 0.0

    step_dict = top_logprobs[answer_idx]
    if step_dict is None or not isinstance(step_dict, dict):
        return 0.0, 0.0

    p_yes, p_no = 0.0, 0.0
    for tok, logprob_val in step_dict.items():
        cleaned = clean_token(tok)
        if cleaned in ("yes", "y"):
            p_yes += math.exp(float(logprob_val))
        elif cleaned in ("no", "n"):
            p_no += math.exp(float(logprob_val))
    return p_yes, p_no


def extract_probs_fireworks_thinking(response) -> Tuple[float, float]:
    """Extract Yes/No probs from Fireworks reasoning model (GPT-OSS-120B).
    Fireworks separates reasoning into reasoning_content, so content logprobs
    contain only the answer tokens. Finds first Yes/No after <|message|> marker.
    Handles both OpenAI-style (object) and Together-style (dict) logprob formats."""
    choice = response.choices[0]
    lp = getattr(choice, "logprobs", None)
    if lp is None:
        return 0.0, 0.0

    content_lp = getattr(lp, "content", None)
    if content_lp is None or not content_lp:
        return 0.0, 0.0

    # Build token list and top_logprobs list from content objects
    tokens = []
    top = []
    for item in content_lp:
        tokens.append(getattr(item, "token", None) if not isinstance(item, dict) else item.get("token"))
        tp = getattr(item, "top_logprobs", []) if not isinstance(item, dict) else item.get("top_logprobs", [])
        top.append(tp)

    if not tokens or not top:
        return 0.0, 0.0

    # Find where the answer starts (after <|message|> marker if present)
    answer_start = 0
    for pos, tok in enumerate(tokens):
        if tok is not None and "<|message|>" in tok:
            answer_start = pos + 1

    # Find the first Yes/No token after the marker
    answer_idx = None
    for pos in range(answer_start, len(tokens)):
        tok = tokens[pos]
        if tok is None:
            continue
        cleaned = clean_token(tok)
        if cleaned in ("yes", "y", "no", "n"):
            answer_idx = pos
            break

    if answer_idx is None or answer_idx >= len(top):
        return 0.0, 0.0

    step_list = top[answer_idx]
    if step_list is None or len(step_list) == 0:
        return 0.0, 0.0

    # Extract logp(Yes) and logp(No) from top_logprobs at this position
    # Handle both object-style (.token, .logprob) and dict-style entries
    logp_yes = float("-inf")
    logp_no = float("-inf")
    for entry in step_list:
        if isinstance(entry, dict):
            tok_str = entry.get("token", "")
            lp_val = entry.get("logprob", float("-inf"))
        else:
            tok_str = getattr(entry, "token", "")
            lp_val = getattr(entry, "logprob", float("-inf"))
        cleaned = clean_token(tok_str)
        if cleaned in ("yes", "y") and logp_yes == float("-inf"):
            logp_yes = float(lp_val)
        elif cleaned in ("no", "n") and logp_no == float("-inf"):
            logp_no = float(lp_val)

    p_yes = 0.0 if logp_yes == float("-inf") else math.exp(logp_yes)
    p_no = 0.0 if logp_no == float("-inf") else math.exp(logp_no)
    Z = p_yes + p_no
    if Z == 0.0:
        return 0.0, 0.0

    return p_yes / Z, p_no / Z


def extract_probs(response) -> Tuple[float, float]:
    """Route to the correct logprob extractor based on model config."""
    if STYLE != "A":
        return 0.0, 0.0  # no logprobs for style B/C

    if MODEL_CFG.get("thinking") and PROVIDER == "fireworks":
        return extract_probs_fireworks_thinking(response)
    elif MODEL_CFG.get("thinking") and PROVIDER == "together":
        # Together thinking models: logprobs include <think> tokens in flat list
        return extract_probs_together_thinking(response)
    elif PROVIDER in ("openai", "fireworks"):
        return extract_probs_openai(response)
    elif PROVIDER == "together":
        return extract_probs_together(response)
    return 0.0, 0.0


# ═══════════════════════════════════════════════════
# RESPONSE TEXT EXTRACTION
# ═══════════════════════════════════════════════════
def get_response_text(response) -> str:
    """Get the text response, handling all providers."""
    if PROVIDER == "google":
        return response.text.strip() if response.text else ""

    if PROVIDER == "anthropic":
        for block in response.content:
            if block.type == "text":
                return block.text.strip()
        return ""

    # OpenAI / Together / Fireworks
    content = response.choices[0].message.content
    if content is None:
        return ""
    content = content.strip()

    if MODEL_CFG.get("thinking"):
        think_end = content.rfind("</think>")
        if think_end != -1:
            content = content[think_end + len("</think>"):].strip()

    return content


# ═══════════════════════════════════════════════════
# API CLIENT CREATION
# ═══════════════════════════════════════════════════
def make_client():
    """Create the appropriate API client from environment variables."""
    api_key = os.environ.get(MODEL_CFG["api_key_env"])
    if not api_key:
        print(f"Error: Environment variable {MODEL_CFG['api_key_env']} not set.")
        sys.exit(1)

    if PROVIDER == "openai":
        from openai import OpenAI
        base_url_env = MODEL_CFG.get("base_url_env")
        if base_url_env:
            base_url = os.environ.get(base_url_env)
            if not base_url:
                print(f"Error: Environment variable {base_url_env} not set.")
                sys.exit(1)
            return OpenAI(api_key=api_key, base_url=base_url,
                         default_headers={"User-Agent": "NUS-LossAversion/1.0"})
        return OpenAI(api_key=api_key)

    elif PROVIDER == "anthropic":
        from anthropic import Anthropic
        return Anthropic(api_key=api_key)

    elif PROVIDER == "together":
        from together import Together
        return Together(api_key=api_key)

    elif PROVIDER == "fireworks":
        from openai import OpenAI
        return OpenAI(
            api_key=api_key,
            base_url="https://api.fireworks.ai/inference/v1",
        )

    elif PROVIDER == "google":
        from google import genai
        return genai.Client(api_key=api_key)

    else:
        raise ValueError(f"Unsupported provider: {PROVIDER}")


# ═══════════════════════════════════════════════════
# API CALL (provider-specific, no system prompt)
# ═══════════════════════════════════════════════════
def make_api_call(client, prompt: str):
    """Make a single API call. Returns the raw response object."""
    messages = [{"role": "user", "content": prompt}]

    if PROVIDER == "openai":
        kwargs = {
            "model": MODEL_CFG["model_id"],
            "messages": messages,
            MODEL_CFG["max_tokens_key"]: MODEL_CFG["max_tokens"],
            "stream": False,
        }
        if MODEL_CFG["temperature"] is not None:
            kwargs["temperature"] = MODEL_CFG["temperature"]
        if STYLE == "A":
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = MODEL_CFG["top_logprobs"]
        if MODEL_CFG.get("reasoning_effort"):
            kwargs["reasoning_effort"] = MODEL_CFG["reasoning_effort"]
        if MODEL_CFG.get("verbosity"):
            kwargs["verbosity"] = MODEL_CFG["verbosity"]
        return client.chat.completions.create(**kwargs)

    elif PROVIDER == "anthropic":
        kwargs = {
            "model": MODEL_CFG["model_id"],
            "messages": messages,
            "max_tokens": MODEL_CFG["max_tokens"],
        }
        if MODEL_CFG["temperature"] is not None:
            kwargs["temperature"] = MODEL_CFG["temperature"]
        return client.messages.create(**kwargs)

    elif PROVIDER == "together":
        kwargs = {
            "model": MODEL_CFG["model_id"],
            "messages": messages,
            "max_tokens": MODEL_CFG["max_tokens"],
            "logprobs": MODEL_CFG["top_logprobs"],
        }
        if MODEL_CFG["temperature"] is not None:
            kwargs["temperature"] = MODEL_CFG["temperature"]
        if MODEL_CFG.get("thinking"):
            kwargs["reasoning_effort"] = MODEL_CFG.get("thinking_effort", "low")
        return client.chat.completions.create(**kwargs)

    elif PROVIDER == "fireworks":
        kwargs = {
            "model": MODEL_CFG["model_id"],
            "messages": messages,
            MODEL_CFG["max_tokens_key"]: MODEL_CFG["max_tokens"],
            "stream": False,
        }
        if MODEL_CFG["temperature"] is not None:
            kwargs["temperature"] = MODEL_CFG["temperature"]
        if STYLE == "A":
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = MODEL_CFG["top_logprobs"]
        if MODEL_CFG.get("thinking"):
            kwargs["reasoning_effort"] = MODEL_CFG.get("thinking_effort", "low")
        # Suppress tokens that crowd out Yes/No in logprobs
        if STYLE == "A":
            kwargs["logit_bias"] = {
                35644: -100,
                8450: -100,
                31515: -100,
                131643: -100,
            }
        return client.chat.completions.create(**kwargs)

    elif PROVIDER == "google":
        from google.genai import types
        config = types.GenerateContentConfig(
            max_output_tokens=MODEL_CFG["max_tokens"],
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        if MODEL_CFG["temperature"] is not None:
            config.temperature = MODEL_CFG["temperature"]
        return client.models.generate_content(
            model=MODEL_CFG["model_id"],
            contents=prompt,
            config=config,
        )

    else:
        raise ValueError(f"Unsupported provider: {PROVIDER}")


# ═══════════════════════════════════════════════════
# OUTPUT HELPERS
# ═══════════════════════════════════════════════════
OUTPUT_DIR = Path(TREATMENT) / FOLDER_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

checkpoint_X_path = OUTPUT_DIR / "loss_aversion_X.json"
checkpoint_Y_path = OUTPUT_DIR / "loss_aversion_Y.json"
completed_index_path = OUTPUT_DIR / "completed_index.json"

# Known run order — used to compute global case-ID offsets so that
# trial → test → remaining produces non-overlapping IDs identical to
# running full_goods.json in one shot.
GOODS_RUN_ORDER = ["trial_goods", "test_goods", "remaining_goods"]
_goods_stem = GOODS_PATH.stem   # e.g. "trial_goods"


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)
    tmp.replace(path)


def load_completed_index() -> dict:
    """Load completed_index.json — returns dict with per-file tracking."""
    if completed_index_path.exists():
        with open(completed_index_path, "r", encoding="utf-8") as f:
            idx = json.load(f)
        # Handle legacy format (plain list of IDs) — treat as unknown
        if isinstance(idx, list):
            return {"file_case_counts": {}, "completed_ids": {}}
        return idx
    return {"file_case_counts": {}, "completed_ids": {}}


def save_checkpoint(data_X, data_Y, completed_ids):
    """Save current progress, deduplicating by case_id.
    Writes to the single shared X/Y files and updates completed_index."""
    with results_lock:
        seen_X = {e["case_id"]: e for e in data_X}
        deduped_X = sorted(seen_X.values(), key=lambda x: x["case_id"])

        seen_Y = {e["case_id"]: e for e in data_Y}
        deduped_Y = sorted(seen_Y.values(), key=lambda x: x["case_id"])

        dump_json(checkpoint_X_path, deduped_X)
        dump_json(checkpoint_Y_path, deduped_Y)

        # Update completed_index with this file's completed IDs
        ci = load_completed_index()
        ci["file_case_counts"][_goods_stem] = total_cases
        ci["completed_ids"][_goods_stem] = sorted(
            cid for cid in completed_ids
            if case_id_offset < cid <= case_id_offset + total_cases
        )
        dump_json(completed_index_path, ci)


# ═══════════════════════════════════════════════════
# CHECKPOINT DETECTION & GLOBAL CASE-ID OFFSET
# ═══════════════════════════════════════════════════
print("=" * 70)
print("CHECKPOINT DETECTION")
print("=" * 70)

# Load existing completed_index to find out what prior files contributed
ci = load_completed_index()

# Compute offset: sum of case counts from all prior files in run order
case_id_offset = 0
for gs in GOODS_RUN_ORDER:
    if gs == _goods_stem:
        break
    case_id_offset += ci.get("file_case_counts", {}).get(gs, 0)

# Warn if prior files in run order haven't been completed
_prior_warnings = []
if _goods_stem in GOODS_RUN_ORDER:
    idx = GOODS_RUN_ORDER.index(_goods_stem)
    for prior in GOODS_RUN_ORDER[:idx]:
        if prior not in ci.get("file_case_counts", {}):
            _prior_warnings.append(prior)

print(f"  Goods file:            {GOODS_FILENAME} -> {GOODS_PATH} (stem: {_goods_stem})")
print(f"  Global case-ID offset: {case_id_offset}")

# Load the shared X/Y data files (contain results from ALL prior runs)
output_data_X: List[dict] = []
output_data_Y: List[dict] = []

x_exists = checkpoint_X_path.exists()
y_exists = checkpoint_Y_path.exists()

print(f"  Checkpoint X exists:   {x_exists}")
print(f"  Checkpoint Y exists:   {y_exists}")

if x_exists and y_exists:
    with open(checkpoint_X_path, "r", encoding="utf-8") as f:
        output_data_X = json.load(f)
    with open(checkpoint_Y_path, "r", encoding="utf-8") as f:
        output_data_Y = json.load(f)

    print(f"  Loaded {len(output_data_X)} entries from X, {len(output_data_Y)} from Y")

elif x_exists or y_exists:
    print("  Only one checkpoint file exists — starting fresh to avoid inconsistency.")
    output_data_X = []
    output_data_Y = []
else:
    print("  No checkpoint files found. Starting fresh.")


# ═══════════════════════════════════════════════════
# BUILD CASE INDEX  (global IDs = offset + local 1..N)
# ═══════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BUILDING CASE INDEX")
print("=" * 70)

case_index: Dict[int, dict] = {}
local_id = 0

for entry in dataset:
    if not (isinstance(entry, (list, tuple)) and len(entry) >= 3):
        continue
    X_num, Y_num, attr_list = entry[0], entry[1], entry[2]
    if not isinstance(attr_list, list):
        continue
    if not (0 <= X_num < len(items)) or not (0 <= Y_num < len(items)):
        continue

    for attr_code in attr_list:
        local_id += 1
        global_id = case_id_offset + local_id
        i, j, k, l = decode_attr(attr_code)
        case_index[global_id] = {
            "X_num": X_num,
            "Y_num": Y_num,
            "i": i, "j": j, "k": k, "l": l,
            "X": items[X_num],
            "Y": items[Y_num],
        }

total_cases = len(case_index)
new_case_id_start = case_id_offset + 1
new_case_id_end = case_id_offset + local_id
print(f"  Built index for {total_cases} cases (IDs {new_case_id_start}-{new_case_id_end})")

# Determine which are already done (intersect checkpoint IDs with THIS file's IDs)
existing_ids_X = {e["case_id"] for e in output_data_X}
existing_ids_Y = {e["case_id"] for e in output_data_Y}
completed_in_X = existing_ids_X & set(case_index.keys())
completed_in_Y = existing_ids_Y & set(case_index.keys())
completed_case_ids = completed_in_X & completed_in_Y

cases_to_run = sorted(set(case_index.keys()) - completed_case_ids)

if completed_case_ids:
    print(f"  Resuming: {len(completed_case_ids)} cases already done")


# ═══════════════════════════════════════════════════
# SINGLE CASE PROCESSOR (with retry)
# ═══════════════════════════════════════════════════
def process_single_case(client, cid: int, case_info: dict, display: bool = False):
    """Process one case (both X and Y). Returns results dict or None on failure."""
    X_num, Y_num = case_info["X_num"], case_info["Y_num"]
    i, j, k, l = case_info["i"], case_info["j"], case_info["k"], case_info["l"]
    X, Y = case_info["X"], case_info["Y"]

    results = {"X": None, "Y": None}

    for label in ["X", "Y"]:
        # Skip if already completed
        if label == "X" and cid in completed_in_X:
            continue
        if label == "Y" and cid in completed_in_Y:
            continue

        if label == "X":
            prompt = generate_prompt(X, Y, i, j, k, l)
        else:
            prompt = generate_prompt(Y, X, k, l, i, j)

        success = False
        retries = 0

        while not success and retries < MAX_RETRIES:
            try:
                response = make_api_call(client, prompt)

# TEMP DEBUG: remove after inspecting
                #if cid <= case_id_offset + 5 and PROVIDER == "fireworks":
                    #lp = getattr(response.choices[0], "logprobs", None)
                    #clp = getattr(lp, "content", None) if lp else None
                    #if clp:
                        #print(f"\n  [DEBUG] Case {cid} [{label}] — {len(clp)} content tokens")
                        #for ix, item in enumerate(clp[:8]):
                            #tok = getattr(item, "token", "?")
                            #tlp = getattr(item, "top_logprobs", [])
                            #cands = [(getattr(e,"token","?"), f"{getattr(e,'logprob',0):.4f}") for e in (tlp or [])[:5]]
                            #print(f"    [{ix}] token={tok!r}  candidates={cands}")
                    #else:
                        #print(f"\n  [DEBUG] Case {cid} [{label}] — no content logprobs!")

                # Extract text
                final_resp = get_response_text(response)

                # Extract logprobs (Style A only; returns 0.0, 0.0 for B/C)
                p_yes, p_no = extract_probs(response)

                # Fallback: if no logprob tokens matched, read text response
                if p_yes == 0.0 and p_no == 0.0:
                    resp_lower = final_resp.lower()[:3] if final_resp else ""
                    if resp_lower.startswith("yes"):
                        p_yes, p_no = 1.0, 0.0
                    elif resp_lower.startswith("no"):
                        p_yes, p_no = 0.0, 1.0

                results[label] = {
                    "case_id": cid,
                    "X_num": X_num,
                    "Y_num": Y_num,
                    "attr": [i, j, k, l],
                    "Yes / No prob": [p_yes, p_no],
                    "output": prompt + " " + final_resp,
                }

                if display:
                    print(f"  #{cid:<5} [{label}]  {final_resp:<5} (Yes={p_yes:.3f}, No={p_no:.3f})")

                success = True

            except Exception as e:
                retries += 1
                error_msg = str(e)
                if "rate" in error_msg.lower() or "limit" in error_msg.lower():
                    wait = RETRY_DELAY * retries
                    print(f"  [Rate Limit] Case {cid} [{label}]: waiting {wait}s (retry {retries}/{MAX_RETRIES})")
                    time.sleep(wait)
                else:
                    print(f"  [Error] Case {cid} [{label}]: {e} (retry {retries}/{MAX_RETRIES})")
                    time.sleep(RETRY_DELAY)

        if not success:
            print(f"  [FAILED] Case {cid} [{label}] after {MAX_RETRIES} retries.")
            return None

    return results


# ═══════════════════════════════════════════════════
# RUN SUMMARY + CONFIRMATION
# ═══════════════════════════════════════════════════
print(f"\n" + "=" * 70)
print(f"RUN SUMMARY — {TREATMENT.upper()} / {MODEL_KEY}")
print("=" * 70)
print(f"  Model:                 {MODEL_KEY} ({MODEL_CFG['model_id']})")
print(f"  Provider:              {PROVIDER} (style {STYLE})")
print(f"  Logprobs:              {'yes (top ' + str(MODEL_CFG['top_logprobs']) + ')' if STYLE == 'A' else 'no'}")
if MODEL_CFG.get("thinking"):
    print(f"  Thinking:              {MODEL_CFG.get('thinking_effort', 'low')}")
else:
    print(f"  Thinking:              disabled")
print(f"  Temperature:           {MODEL_CFG['temperature'] if MODEL_CFG['temperature'] is not None else 'default (omitted)'}")
print(f"  Treatment:             {TREATMENT}")
print(f"  Goods file:            {GOODS_FILENAME} -> {GOODS_PATH}")
print(f"  Case-ID offset:        {case_id_offset}")
print(f"  Cases in this file:    {total_cases}")
print(f"  Already completed:     {len(completed_case_ids)}")
print(f"  Remaining to run:      {len(cases_to_run)}")
print(f"  API calls needed:      {len(cases_to_run) * 2}")
print(f"  Case ID range:         {new_case_id_start}-{new_case_id_end}")
print(f"  Total in X/Y files:    {len(output_data_X)}/{len(output_data_Y)}")
print(f"  Parallel workers:      {PARALLEL_WORKERS}")
print(f"  Output dir:            {OUTPUT_DIR}/")

if len(cases_to_run) == 0:
    print("\nAll cases already completed!")
    sys.exit(0)

# ═══════════════════════════════════════════════════
# WARNINGS (all shown together before confirmation)
# ═══════════════════════════════════════════════════
warnings_found = False

# 1. File already fully completed in a previous run
ci_completed_for_file = ci.get("completed_ids", {}).get(_goods_stem, [])
if len(ci_completed_for_file) == total_cases and len(cases_to_run) > 0:
    # This shouldn't normally happen (cases_to_run would be 0), but guard against
    # completed_index saying done while X/Y files are missing/corrupted
    print(f"\n  ⚠ WARNING: completed_index says '{_goods_stem}.json' was fully done")
    print(f"    ({len(ci_completed_for_file)}/{total_cases} cases) but X/Y files")
    print(f"    don't match. Data may be corrupted.")
    warnings_found = True

# 2. Re-running a file that already has entries (not a resume — starting over?)
if _goods_stem in ci.get("file_case_counts", {}) and len(completed_case_ids) == 0 and len(output_data_X) > 0:
    prev_count = len(ci_completed_for_file)
    print(f"\n  ⚠ WARNING: '{_goods_stem}.json' was previously run ({prev_count} cases")
    print(f"    recorded) but 0 of those case IDs match the current index.")
    print(f"    This may indicate the file was already completed and you're")
    print(f"    re-running it unnecessarily, or the data files were modified.")
    warnings_found = True

# 3. Prior files not yet run (out-of-order)
if _prior_warnings:
    print(f"\n  ⚠ WARNING: Prior file(s) not yet run:")
    for pw in _prior_warnings:
        print(f"    - {pw}.json")
    print(f"    Expected run order: {' → '.join(GOODS_RUN_ORDER)}")
    print(f"    Running out of order will produce WRONG case IDs.")
    warnings_found = True

# 4. Prior file only partially completed
if _goods_stem in GOODS_RUN_ORDER:
    gs_idx = GOODS_RUN_ORDER.index(_goods_stem)
    for prior in GOODS_RUN_ORDER[:gs_idx]:
        prior_count = ci.get("file_case_counts", {}).get(prior, 0)
        prior_done = len(ci.get("completed_ids", {}).get(prior, []))
        if prior_count > 0 and prior_done < prior_count:
            print(f"\n  ⚠ WARNING: '{prior}.json' is only partially complete")
            print(f"    ({prior_done}/{prior_count} cases). Finish it first to")
            print(f"    avoid gaps in the final output.")
            warnings_found = True

# 5. X/Y entry count mismatch
if len(output_data_X) != len(output_data_Y):
    print(f"\n  ⚠ WARNING: X/Y file entry counts don't match!")
    print(f"    X has {len(output_data_X)} entries, Y has {len(output_data_Y)}.")
    print(f"    This may indicate a corrupted checkpoint.")
    warnings_found = True

# 6. Goods file not in the known run order
if _goods_stem not in GOODS_RUN_ORDER:
    print(f"\n  ⚠ WARNING: '{_goods_stem}' is not in the expected run order:")
    print(f"    {GOODS_RUN_ORDER}")
    print(f"    Case-ID offset defaults to 0. IDs may collide with other files.")
    warnings_found = True

if warnings_found:
    print(f"\n" + "─" * 70)

confirm = input(f"\nProceed? Type 'yes' to continue: ").strip().lower()
if confirm != "yes":
    print("Aborted.")
    sys.exit(0)


# ═══════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════
client = make_client()
failed_cases: List[int] = []

print(f"\n" + "=" * 70)
print(f"RUNNING — {TREATMENT.upper()} / {MODEL_KEY}")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Workers: {PARALLEL_WORKERS}")
print("=" * 70)

try:
    # ── Phase 1: Sequential display for verification ──
    display_count = min(DISPLAY_FIRST_N, len(cases_to_run))
    if display_count > 0:
        print(f"\nPhase 1: Displaying first {display_count} cases sequentially...\n")

        for cid in cases_to_run[:display_count]:
            case_info = case_index[cid]
            results = process_single_case(client, cid, case_info, display=True)

            if results:
                with results_lock:
                    if results["X"]:
                        output_data_X.append(results["X"])
                        completed_in_X.add(cid)
                    if results["Y"]:
                        output_data_Y.append(results["Y"])
                        completed_in_Y.add(cid)
                    if cid in completed_in_X and cid in completed_in_Y:
                        completed_case_ids.add(cid)
                # Save after every completed case in Phase 1
                save_checkpoint(output_data_X, output_data_Y, completed_case_ids)
            else:
                failed_cases.append(cid)

        print(f"\n  [Checkpoint] Saved after initial {display_count} cases")
        cases_to_run = cases_to_run[display_count:]

    # ── Phase 2: Parallel execution ──
    if len(cases_to_run) == 0:
        print("\nAll cases completed!")
    else:
        print(f"\nPhase 2: Processing remaining {len(cases_to_run)} cases in parallel...\n")

        processed_count = 0
        start_time = datetime.now()

        with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
            future_to_cid = {
                executor.submit(process_single_case, client, cid, case_index[cid], False): cid
                for cid in cases_to_run
            }

            for future in as_completed(future_to_cid):
                cid = future_to_cid[future]

                try:
                    results = future.result()

                    if results:
                        with results_lock:
                            if results["X"]:
                                output_data_X.append(results["X"])
                                completed_in_X.add(cid)
                            if results["Y"]:
                                output_data_Y.append(results["Y"])
                                completed_in_Y.add(cid)
                            if cid in completed_in_X and cid in completed_in_Y:
                                completed_case_ids.add(cid)
                            processed_count += 1

                        if processed_count % PROGRESS_EVERY == 0:
                            total_done = len(completed_case_ids)
                            ts = datetime.now().strftime("%H:%M:%S")
                            elapsed = (datetime.now() - start_time).total_seconds()
                            rate = processed_count / elapsed if elapsed > 0 else 0
                            remaining = (len(cases_to_run) - processed_count) / rate if rate > 0 else 0
                            total_expected = total_cases
                            pct = total_done * 100 // total_expected if total_expected > 0 else 0
                            print(f"  [{ts}] {total_done}/{total_expected} ({pct}%) "
                                  f"— {rate:.1f} cases/sec — ~{remaining/60:.0f}min left")

                        if processed_count % CHECKPOINT_EVERY == 0:
                            save_checkpoint(output_data_X, output_data_Y, completed_case_ids)
                            print(f"  [Checkpoint] Saved at {len(completed_case_ids)} total cases")
                    else:
                        failed_cases.append(cid)

                except Exception as e:
                    print(f"  [Error] Case {cid}: {e}")
                    failed_cases.append(cid)

except KeyboardInterrupt:
    print(f"\n\n[INTERRUPTED] Saving progress...")
    save_checkpoint(output_data_X, output_data_Y, completed_case_ids)
    print(f"  Saved {len(completed_case_ids)} cases. Rerun to continue.")
    sys.exit(1)

except Exception as e:
    print(f"\n\n[CRASH] {e}")
    save_checkpoint(output_data_X, output_data_Y, completed_case_ids)
    print(f"  Saved {len(completed_case_ids)} cases.")
    raise

# Final save
save_checkpoint(output_data_X, output_data_Y, completed_case_ids)

# ═══════════════════════════════════════════════════
# COMPLETION REPORT
# ═══════════════════════════════════════════════════
print(f"\n" + "=" * 70)
print("COMPLETE!")
print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)
print(f"  Model:                 {MODEL_KEY} ({MODEL_CFG['model_id']})")
print(f"  Treatment:             {TREATMENT.upper()}")
print(f"  Goods file:            {GOODS_FILENAME} -> {GOODS_PATH}")
print(f"  Total cases completed: {len(completed_case_ids)}")
print(f"  Total in X file:       {len(output_data_X)}")
print(f"  Total in Y file:       {len(output_data_Y)}")
print(f"  Failed cases:          {len(failed_cases)}")
print(f"  Saved to:              {OUTPUT_DIR}/")

if failed_cases:
    print(f"\n  Failed case IDs: {failed_cases[:20]}{'...' if len(failed_cases) > 20 else ''}")
    print("  Rerun the script to retry failed cases.")

missing = set(case_index.keys()) - completed_case_ids
if missing:
    print(f"\n  Missing {len(missing)} cases. Rerun to complete.")
else:
    print(f"\n  All {total_cases} cases from {GOODS_FILENAME} completed.")
