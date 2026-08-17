#!/usr/bin/env python3
"""CPU-only LoRA parameter atlas for the full-run SFT adapters (Phase 1).

WEIGHT-SPACE SCREENING ONLY. This measures where, in parameter space, the SFT
LoRA update lives — per layer, per module, and per low-rank direction. It does
NOT establish causal importance: no activation weighting, no ablation, no
behavioral evaluation is performed here. A large weight update is not evidence
that a module causes the measured behavioral change, and a small one is not
evidence that it does not.

Scope: the SIX full-run SFT adapters only (seed 1 @ 2k/4k/30k, seed 2 @
2k/6k/30k). Pilot and `final` paths hard-fail. No sub-2k adapter exists, so
nothing is claimed about the first 2,000 steps.

LoRA mathematics (see `Correct LoRA mathematics` in the manifest):
    dW = (alpha / r) * B @ A,  with alpha/r = 32/16 = 2.
The scale is folded into B EXACTLY ONCE. Raw ||A|| and ||B|| are never used as
importance statistics — the factorization has a gauge freedom
(A -> R A, B -> B R^-1) under which they are meaningless but dW is invariant.
The full dense dW is never materialized:
    ||BA||_F^2 = trace[(B^T B)(A A^T)]                     (16x16 Grams)
    nonzero sv(BA) = sv(R_B R_A^T), B = Q_B R_B, A^T = Q_A R_A   (16x16)
    <B1 A1, B2 A2>_F = trace[(B1^T B2)(A2 A1^T)]           (16x16)

Three stages:
  --refresh : read the six untracked adapters + the frozen base weights,
              verify identities/hashes against results/sft_grid_verification.json,
              and write the canonical tracked snapshot + manifest.
  default   : render CSV / Markdown / figures from the tracked snapshot
              (needs no checkpoint or model directory).
  --check    : verify deterministic rendering from the tracked snapshot; passes
              in a clean `git archive HEAD` extraction.

No GPU, no PBS, no inference, no generation, no training, no activation
collection, no ablation. The frozen estimator is untouched. The method-comparison,
frozen-unused, OOD, semantic-counterbalancing suites and the historical pilot are
never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL_DIR = ROOT / "models" / "Qwen2.5-7B-Instruct"
BASE_WEIGHTS = BASE_MODEL_DIR / "model.safetensors"
BASE_CONFIG = BASE_MODEL_DIR / "config.json"
VERIFICATION = ROOT / "results" / "sft_grid_verification.json"

OUT_DIR = ROOT / "results" / "sft_parameter_localization"
SNAPSHOT = OUT_DIR / "sft_lora_localization_snapshot.json"
MANIFEST = OUT_DIR / "sft_lora_localization_manifest.json"
STATS_CSV = OUT_DIR / "sft_lora_module_stats.csv"
COMP_CSV = OUT_DIR / "sft_lora_module_comparisons.csv"
SUMMARY_MD = OUT_DIR / "sft_lora_summary.md"
ATLAS_PNG = OUT_DIR / "sft_lora_parameter_atlas.png"
SPECTRUM_PNG = OUT_DIR / "sft_lora_singular_energy.png"
CROSS_PNG = OUT_DIR / "sft_lora_cross_seed_similarity.png"

# --------------------------------------------------------------------------- #
# Frozen inventory — exactly six full-run adapters, nothing else
# --------------------------------------------------------------------------- #
ADAPTERS = [
    {"seed": 1, "step": 2000, "role": "first_saved",
     "path": "checkpoints/sft_qwen_delta_seed1/checkpoint-2000"},
    {"seed": 1, "step": 4000, "role": "selected",
     "path": "checkpoints/sft_qwen_delta_seed1/checkpoint-4000"},
    {"seed": 1, "step": 30000, "role": "endpoint",
     "path": "checkpoints/sft_qwen_delta_seed1/checkpoint-30000"},
    {"seed": 2, "step": 2000, "role": "first_saved",
     "path": "checkpoints/sft_qwen_delta_seed2/checkpoint-2000"},
    {"seed": 2, "step": 6000, "role": "selected",
     "path": "checkpoints/sft_qwen_delta_seed2/checkpoint-6000"},
    {"seed": 2, "step": 30000, "role": "endpoint",
     "path": "checkpoints/sft_qwen_delta_seed2/checkpoint-30000"},
]
SELECTED = {1: 4000, 2: 6000}
SEEDS = [1, 2]
N_LAYERS = 28
MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
SUB_BLOCK = {"q_proj": "self_attn", "k_proj": "self_attn", "v_proj": "self_attn",
             "o_proj": "self_attn", "gate_proj": "mlp", "up_proj": "mlp",
             "down_proj": "mlp"}
FAMILY = {"q_proj": "attention", "k_proj": "attention", "v_proj": "attention",
          "o_proj": "attention", "gate_proj": "mlp", "up_proj": "mlp",
          "down_proj": "mlp"}
SHORT = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o",
         "gate_proj": "gate", "up_proj": "up", "down_proj": "down"}

EXPECTED_RANK = 16
EXPECTED_ALPHA = 32
EXPECTED_BASE_MODEL = "/scratch/users/nus/jackeyc0/lambda-zero/models/Qwen2.5-7B-Instruct"
ADAPTER_PREFIX = "base_model.model.model."
KEY_RE = re.compile(
    r"^base_model\.model\.model\.layers\.(\d+)\.(self_attn|mlp)\.([a-z_]+)\.lora_([AB])\.weight$")

FORBIDDEN_PATH_TOKENS = ("pilot", "final")

# Deterministic base spectral-norm estimation (randomized subspace iteration).
BASE_SPEC_SEED = 20260817
BASE_SPEC_BLOCK = 8
BASE_SPEC_MAXITER = 60
BASE_SPEC_TOL = 1e-7

SV_EPS = 0.0   # singular values below this are treated as exact zeros (none expected)


# --------------------------------------------------------------------------- #
# Core low-rank mathematics — importable and unit-tested against dense algebra
# --------------------------------------------------------------------------- #
def scaled_factors(a, b, scale):
    """Return (A, B_scaled) in float64 with the LoRA scale folded into B ONCE.

    A: [r, in]   B: [out, r]   dW = scale * B @ A = (scale*B) @ A
    """
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    if a64.ndim != 2 or b64.ndim != 2:
        raise ValueError(f"A and B must be 2-D, got {a64.shape} and {b64.shape}")
    if a64.shape[0] != b64.shape[1]:
        raise ValueError(f"inner rank mismatch: A {a64.shape} vs B {b64.shape}")
    return a64, float(scale) * b64


def lowrank_fro(a, b, scale):
    """||scale * B @ A||_F without forming the dense product.

    ||BA||_F^2 = trace[(B^T B)(A A^T)] = sum((B^T B) * (A A^T)^T), both r x r.
    """
    a64, bs = scaled_factors(a, b, scale)
    gb = bs.T @ bs               # r x r
    ga = a64 @ a64.T             # r x r
    val = float(np.sum(gb * ga.T))
    return math.sqrt(max(val, 0.0))


def lowrank_singular_values(a, b, scale):
    """Exact nonzero singular values of scale * B @ A, via two economy QRs.

    B  = Q_B R_B          (B is [out, r])
    A^T= Q_A R_A          (A^T is [in, r])
    => B A = Q_B (R_B R_A^T) Q_A^T, so sv(BA) = sv(R_B R_A^T)  (at most r x r).
    Returned in descending order, length min(r, out, in).
    """
    a64, bs = scaled_factors(a, b, scale)
    _, rb = np.linalg.qr(bs)         # rb: r x r  (economy; out >= r assumed)
    _, ra = np.linalg.qr(a64.T)      # ra: r x r
    sv = np.linalg.svd(rb @ ra.T, compute_uv=False)
    return np.sort(np.asarray(sv, dtype=np.float64))[::-1]


def lowrank_spectral(a, b, scale):
    """||scale * B @ A||_2 (largest singular value)."""
    sv = lowrank_singular_values(a, b, scale)
    return float(sv[0]) if sv.size else 0.0


def lowrank_inner(a1, b1, s1, a2, b2, s2):
    """<C1, C2>_F for C_i = s_i * B_i @ A_i, without dense products.

    <C1,C2>_F = trace(C1^T C2) = trace[(B1^T B2)(A2 A1^T)]  (both r x r).
    """
    a1_64, b1s = scaled_factors(a1, b1, s1)
    a2_64, b2s = scaled_factors(a2, b2, s2)
    if a1_64.shape[1] != a2_64.shape[1] or b1s.shape[0] != b2s.shape[0]:
        raise ValueError("composite updates have different dense shapes")
    m1 = b1s.T @ b2s              # r1 x r2
    m2 = a2_64 @ a1_64.T          # r2 x r1
    return float(np.sum(m1 * m2.T))


def lowrank_cosine(a1, b1, s1, a2, b2, s2):
    """Frobenius cosine between the two COMPOSITE updates (never raw A/B)."""
    n1 = lowrank_fro(a1, b1, s1)
    n2 = lowrank_fro(a2, b2, s2)
    if n1 <= 0.0 or n2 <= 0.0:
        return float("nan")
    return lowrank_inner(a1, b1, s1, a2, b2, s2) / (n1 * n2)


def lowrank_diff_fro(a1, b1, s1, a2, b2, s2):
    """||C1 - C2||_F = sqrt(||C1||^2 + ||C2||^2 - 2<C1,C2>)."""
    n1sq = lowrank_fro(a1, b1, s1) ** 2
    n2sq = lowrank_fro(a2, b2, s2) ** 2
    ip = lowrank_inner(a1, b1, s1, a2, b2, s2)
    return math.sqrt(max(n1sq + n2sq - 2.0 * ip, 0.0))


def stable_rank(sv):
    """||M||_F^2 / ||M||_2^2 — between 1 and rank(M)."""
    sv = np.asarray(sv, dtype=np.float64)
    if sv.size == 0 or sv[0] <= 0:
        return float("nan")
    return float(np.sum(sv ** 2) / (sv[0] ** 2))


def effective_rank_entropy(sv):
    """exp(-sum p_j log p_j) with p_j = sv_j^2 / sum sv^2 (Roy & Vetterli)."""
    sv = np.asarray(sv, dtype=np.float64)
    e = sv ** 2
    tot = float(np.sum(e))
    if tot <= 0:
        return float("nan")
    p = e / tot
    p = p[p > 0]
    return float(math.exp(-float(np.sum(p * np.log(p)))))


def energy_retained(sv, k):
    """Fraction of squared-Frobenius energy in the top-k singular directions."""
    sv = np.asarray(sv, dtype=np.float64)
    e = sv ** 2
    tot = float(np.sum(e))
    if tot <= 0:
        return float("nan")
    return float(np.sum(e[:min(k, sv.size)]) / tot)


def min_rank_for_energy(sv, frac):
    """Smallest k with top-k energy fraction >= frac (k <= len(sv))."""
    sv = np.asarray(sv, dtype=np.float64)
    e = sv ** 2
    tot = float(np.sum(e))
    if tot <= 0:
        return -1
    c = np.cumsum(e) / tot
    idx = np.searchsorted(c, frac - 1e-12, side="left")
    return int(min(idx + 1, sv.size))


def random_rank_cosine_null(out_dim, in_dim, rank=EXPECTED_RANK, n_draws=200, seed=0):
    """Reference distribution for the composite cosine between two INDEPENDENT
    rank-`rank` updates of the same dense shape, with iid Gaussian factors.

    This is a scale for reading a cross-seed cosine, not a significance test: it
    answers "how large a cosine would two unrelated rank-16 updates of this shape
    produce by chance?" Deterministic given `seed`.
    """
    g = np.random.default_rng(seed)
    vals = np.empty(n_draws)
    for i in range(n_draws):
        a1 = g.standard_normal((rank, in_dim)).astype(np.float32)
        b1 = g.standard_normal((out_dim, rank)).astype(np.float32)
        a2 = g.standard_normal((rank, in_dim)).astype(np.float32)
        b2 = g.standard_normal((out_dim, rank)).astype(np.float32)
        vals[i] = lowrank_cosine(a1, b1, 1.0, a2, b2, 1.0)
    return {"out": int(out_dim), "in": int(in_dim), "rank": int(rank),
            "n_draws": int(n_draws), "seed": int(seed),
            "mean_cosine": float(np.mean(vals)),
            "sd_cosine": float(np.std(vals, ddof=1)),
            "mean_abs_cosine": float(np.mean(np.abs(vals))),
            "p95_abs_cosine": float(np.percentile(np.abs(vals), 95)),
            "max_abs_cosine": float(np.max(np.abs(vals)))}


def gini(values):
    """Gini coefficient of a nonnegative vector.

        G = sum_i sum_j |x_i - x_j| / (2 n^2 mean(x))
    Computed via the sorted-rank identity. 0 = perfectly even, -> 1 = all mass
    in one element.
    """
    x = np.sort(np.asarray(values, dtype=np.float64))
    n = x.size
    if n == 0:
        return float("nan")
    if np.any(x < 0):
        raise ValueError("gini requires nonnegative values")
    tot = float(np.sum(x))
    if tot <= 0:
        return 0.0
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((np.sum((2.0 * idx - n - 1.0) * x)) / (n * tot))


def top_share(values, k):
    """Share of total held by the k largest entries."""
    x = np.sort(np.asarray(values, dtype=np.float64))[::-1]
    tot = float(np.sum(x))
    if tot <= 0:
        return float("nan")
    return float(np.sum(x[:min(k, x.size)]) / tot)


def top_frac_share(values, frac):
    """Share of total held by the top `frac` FRACTION of entries (ceil count)."""
    n = len(values)
    k = int(math.ceil(frac * n))
    return top_share(values, max(k, 1))


# --------------------------------------------------------------------------- #
# Path / identity guards
# --------------------------------------------------------------------------- #
def reject_forbidden_path(path) -> None:
    low = str(path).lower()
    for tok in FORBIDDEN_PATH_TOKENS:
        if tok in low:
            raise ValueError(
                f"forbidden adapter path token {tok!r} (pilot/final are out of scope): {path}")


def parse_module_key(key: str):
    """('...layers.7.mlp.up_proj.lora_A.weight') -> (7, 'up_proj', 'A')."""
    m = KEY_RE.match(key)
    if not m:
        raise ValueError(f"unrecognized adapter tensor key: {key}")
    layer, sub, mod, fac = int(m.group(1)), m.group(2), m.group(3), m.group(4)
    if mod not in MODULES:
        raise ValueError(f"unexpected target module {mod!r} in key {key}")
    if SUB_BLOCK[mod] != sub:
        raise ValueError(f"module {mod!r} in wrong sub-block {sub!r}: {key}")
    if not (0 <= layer < N_LAYERS):
        raise ValueError(f"layer index {layer} out of range in key {key}")
    return layer, mod, fac


def base_key_for(layer: int, module: str) -> str:
    return f"model.layers.{layer}.{SUB_BLOCK[module]}.{module}.weight"


def sha256_file(p: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------- #
# Adapter reading (safetensors, incremental) + strict verification
# --------------------------------------------------------------------------- #
def verify_adapter_config(cfg: dict, path: str) -> None:
    if cfg.get("peft_type") != "LORA":
        raise SystemExit(f"{path}: peft_type != LORA ({cfg.get('peft_type')!r})")
    if int(cfg.get("r", -1)) != EXPECTED_RANK:
        raise SystemExit(f"{path}: LoRA rank {cfg.get('r')!r} != {EXPECTED_RANK}")
    if int(cfg.get("lora_alpha", -1)) != EXPECTED_ALPHA:
        raise SystemExit(f"{path}: lora_alpha {cfg.get('lora_alpha')!r} != {EXPECTED_ALPHA}")
    if cfg.get("bias") != "none":
        raise SystemExit(f"{path}: bias {cfg.get('bias')!r} != 'none'")
    tm = sorted(cfg.get("target_modules") or [])
    if tm != sorted(MODULES):
        raise SystemExit(f"{path}: target_modules {tm} != {sorted(MODULES)}")
    if cfg.get("modules_to_save"):
        raise SystemExit(f"{path}: modules_to_save must be empty, got {cfg['modules_to_save']!r}")
    if cfg.get("use_dora"):
        raise SystemExit(f"{path}: use_dora must be false")
    if cfg.get("use_rslora"):
        raise SystemExit(f"{path}: use_rslora must be false (scale must be alpha/r)")
    if cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
        raise SystemExit(f"{path}: per-module rank/alpha patterns are not supported here")
    bm = str(cfg.get("base_model_name_or_path", ""))
    if Path(bm).name != Path(EXPECTED_BASE_MODEL).name:
        raise SystemExit(f"{path}: unexpected base model identity {bm!r}")


def load_adapter_factors(adapter_dir: Path):
    """Read the LoRA A/B tensors incrementally. Returns {(layer, module): {'A','B'}}."""
    from safetensors import safe_open
    reject_forbidden_path(adapter_dir)
    wpath = adapter_dir / "adapter_model.safetensors"
    out: dict = {}
    with safe_open(str(wpath), framework="numpy") as fh:
        keys = list(fh.keys())
        for key in keys:
            layer, mod, fac = parse_module_key(key)
            slot = out.setdefault((layer, mod), {})
            if fac in slot:
                raise SystemExit(f"{wpath}: duplicate lora_{fac} for layer {layer} {mod}")
            slot[fac] = fh.get_tensor(key)
    expected = {(l, m) for l in range(N_LAYERS) for m in MODULES}
    got = set(out)
    if got != expected:
        missing = sorted(expected - got)[:5]
        extra = sorted(got - expected)[:5]
        raise SystemExit(f"{wpath}: module inventory mismatch missing={missing} extra={extra}")
    for (layer, mod), slot in out.items():
        for fac in ("A", "B"):
            if fac not in slot:
                raise SystemExit(f"{wpath}: layer {layer} {mod} missing lora_{fac} factor")
        a, b = slot["A"], slot["B"]
        if a.shape[0] != EXPECTED_RANK or b.shape[1] != EXPECTED_RANK:
            raise SystemExit(f"{wpath}: layer {layer} {mod} rank mismatch A{a.shape} B{b.shape}")
        if a.dtype != np.float32 or b.dtype != np.float32:
            raise SystemExit(f"{wpath}: layer {layer} {mod} unexpected dtype "
                             f"{a.dtype}/{b.dtype} (expected float32)")
    if len(keys) != 2 * N_LAYERS * len(MODULES):
        raise SystemExit(f"{wpath}: expected {2 * N_LAYERS * len(MODULES)} tensors, got {len(keys)}")
    return out


# --------------------------------------------------------------------------- #
# Base weights — direct incremental safetensors reads (BF16, never the model)
# --------------------------------------------------------------------------- #
def read_safetensors_header(path: Path):
    """Return (header_dict, data_start_offset, raw_header_bytes)."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        raw = fh.read(n)
    hdr = json.loads(raw)
    hdr.pop("__metadata__", None)
    return hdr, 8 + n, raw


def _bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    """Widen bfloat16 (as raw uint16) to float32. Lossless: bf16 is the top 16
    bits of a float32, so zero-filling the low half is exact."""
    buf = np.zeros((u16.size, 2), dtype=np.uint16)
    buf[:, 1] = u16                       # little-endian: high half is index 1
    return buf.view(np.float32).reshape(-1)


def read_base_tensor_f32(fh, data_start: int, entry: dict) -> np.ndarray:
    if entry["dtype"] != "BF16":
        raise SystemExit(f"unexpected base dtype {entry['dtype']!r} (expected BF16)")
    start, stop = entry["data_offsets"]
    fh.seek(data_start + start)
    raw = fh.read(stop - start)
    if len(raw) != stop - start:
        raise SystemExit("short read on base safetensors payload")
    u16 = np.frombuffer(raw, dtype="<u2")
    return _bf16_to_f32(u16).reshape(entry["shape"])


def frobenius_f32(w: np.ndarray, chunk_rows: int = 512) -> float:
    """Frobenius norm with float64 accumulation, chunked for stability."""
    acc = 0.0
    for i in range(0, w.shape[0], chunk_rows):
        blk = w[i:i + chunk_rows].astype(np.float64)
        acc += float(np.sum(blk * blk))
    return math.sqrt(acc)


def spectral_subspace(w: np.ndarray, block=BASE_SPEC_BLOCK, maxiter=BASE_SPEC_MAXITER,
                      tol=BASE_SPEC_TOL, seed=BASE_SPEC_SEED):
    """Largest singular value of a dense matrix by randomized subspace iteration.

    Deterministic (fixed seed). Returns (sigma_max, n_iters, converged).
    Used ONLY for the frozen base weights; dW spectral norms are exact.
    """
    m, n = w.shape
    k = min(block, m, n)
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((n, k)).astype(np.float32)
    q, _ = np.linalg.qr(q)
    prev = 0.0
    sig = 0.0
    it = 0
    for it in range(1, maxiter + 1):
        y = w @ q
        y, _ = np.linalg.qr(y)
        z = w.T @ y
        q, r = np.linalg.qr(z)
        sig = float(np.linalg.svd(r.astype(np.float64), compute_uv=False)[0])
        if it > 1 and abs(sig - prev) <= tol * max(sig, 1e-30):
            return sig, it, True
        prev = sig
    return sig, it, False


def collect_base_stats(verbose=True):
    """{(layer, module): {shape, numel, fro, spec, spec_iters, spec_converged}}."""
    if not BASE_WEIGHTS.exists():
        raise SystemExit(f"MISSING base weights: {BASE_WEIGHTS}")
    hdr, data_start, raw_header = read_safetensors_header(BASE_WEIGHTS)
    stats = {}
    with open(BASE_WEIGHTS, "rb") as fh:
        for layer in range(N_LAYERS):
            for mod in MODULES:
                key = base_key_for(layer, mod)
                matches = [k for k in hdr if k == key]
                if len(matches) != 1:
                    raise SystemExit(
                        f"base tensor mapping for layer {layer} {mod} is missing or ambiguous: "
                        f"{len(matches)} candidates for key {key!r}")
                w = read_base_tensor_f32(fh, data_start, hdr[key])
                if w.ndim != 2:
                    raise SystemExit(f"base tensor {key} is not 2-D: {w.shape}")
                fro = frobenius_f32(w)
                spec, iters, conv = spectral_subspace(w)
                stats[(layer, mod)] = {
                    "shape": [int(w.shape[0]), int(w.shape[1])],
                    "numel": int(w.size), "fro": fro, "spec": spec,
                    "spec_iters": iters, "spec_converged": bool(conv)}
                del w
            if verbose:
                print(f"  base layer {layer:>2}/{N_LAYERS - 1} scanned")
    return stats, raw_header


# --------------------------------------------------------------------------- #
# Per-adapter statistics
# --------------------------------------------------------------------------- #
def module_rows(factors: dict, base_stats: dict, meta: dict, scale: float):
    """Per-module magnitude + rank statistics for one adapter."""
    rows = []
    for layer in range(N_LAYERS):
        for mod in MODULES:
            a = factors[(layer, mod)]["A"]
            b = factors[(layer, mod)]["B"]
            bs = base_stats[(layer, mod)]
            out_dim, in_dim = int(b.shape[0]), int(a.shape[1])
            if bs["shape"] != [out_dim, in_dim]:
                raise SystemExit(
                    f"layer {layer} {mod}: LoRA dense shape {[out_dim, in_dim]} does not match "
                    f"base weight shape {bs['shape']}")
            sv = lowrank_singular_values(a, b, scale)
            fro = lowrank_fro(a, b, scale)
            spec = float(sv[0])
            numel = out_dim * in_dim
            rows.append({
                "seed": meta["seed"], "step": meta["step"], "role": meta["role"],
                "layer": layer, "family": FAMILY[mod], "module": mod,
                "sub_block": SUB_BLOCK[mod],
                "a_rows": int(a.shape[0]), "a_cols": int(a.shape[1]),
                "b_rows": int(b.shape[0]), "b_cols": int(b.shape[1]),
                "n_trainable": int(a.size + b.size),
                "base_rows": out_dim, "base_cols": in_dim, "base_numel": numel,
                "rank": EXPECTED_RANK, "alpha": EXPECTED_ALPHA, "scale": scale,
                "delta_fro": fro,
                "delta_spec": spec,
                "delta_rms": fro / math.sqrt(numel),
                "base_fro": bs["fro"], "base_spec": bs["spec"],
                "base_spec_converged": bs["spec_converged"],
                "rel_fro": fro / bs["fro"] if bs["fro"] > 0 else float("nan"),
                "rel_spec": spec / bs["spec"] if bs["spec"] > 0 else float("nan"),
                "energy": fro ** 2,
                "stable_rank": stable_rank(sv),
                "eff_rank_entropy": effective_rank_entropy(sv),
                "energy_top1": energy_retained(sv, 1),
                "energy_top2": energy_retained(sv, 2),
                "energy_top4": energy_retained(sv, 4),
                "energy_top8": energy_retained(sv, 8),
                "energy_top16": energy_retained(sv, 16),
                "rank90": min_rank_for_energy(sv, 0.90),
                "rank95": min_rank_for_energy(sv, 0.95),
                "rank99": min_rank_for_energy(sv, 0.99),
                "singular_values": [float(x) for x in sv],
            })
    # energy shares + concentration ranking (raw and dimension-normalized)
    energies = np.array([r["energy"] for r in rows], dtype=np.float64)
    tot = float(np.sum(energies))
    norm_energies = np.array([r["delta_rms"] ** 2 for r in rows], dtype=np.float64)
    ntot = float(np.sum(norm_energies))
    order = np.argsort(-energies, kind="stable")
    cum = 0.0
    for rank_pos, i in enumerate(order, start=1):
        rows[i]["energy_share"] = float(energies[i] / tot)
        rows[i]["energy_rank"] = rank_pos
        cum += float(energies[i] / tot)
        rows[i]["energy_cumshare"] = cum
    norder = np.argsort(-norm_energies, kind="stable")
    for rank_pos, i in enumerate(norder, start=1):
        rows[i]["norm_energy_share"] = float(norm_energies[i] / ntot)
        rows[i]["norm_energy_rank"] = rank_pos
    return rows


def concentration(rows):
    """Concentration of update energy across the 196 modules of one adapter."""
    e = np.array([r["energy"] for r in rows], dtype=np.float64)
    ne = np.array([r["delta_rms"] ** 2 for r in rows], dtype=np.float64)
    n = len(rows)
    by_layer = {}
    by_module = {}
    by_family = {}
    for r in rows:
        by_layer[r["layer"]] = by_layer.get(r["layer"], 0.0) + r["energy"]
        by_module[r["module"]] = by_module.get(r["module"], 0.0) + r["energy"]
        by_family[r["family"]] = by_family.get(r["family"], 0.0) + r["energy"]
    tot = float(np.sum(e))
    ranked = sorted(rows, key=lambda r: -r["energy"])
    return {
        "n_modules": n,
        "total_energy": tot,
        "total_delta_fro": math.sqrt(tot),
        "top1": top_share(e, 1), "top5": top_share(e, 5), "top10": top_share(e, 10),
        "top5pct": top_frac_share(e, 0.05), "top10pct": top_frac_share(e, 0.10),
        "top25pct": top_frac_share(e, 0.25),
        "gini": gini(e),
        "norm_top1": top_share(ne, 1), "norm_top5": top_share(ne, 5),
        "norm_top10": top_share(ne, 10),
        "norm_top10pct": top_frac_share(ne, 0.10), "norm_gini": gini(ne),
        "by_layer": {str(k): v / tot for k, v in sorted(by_layer.items())},
        "by_module": {k: by_module[k] / tot for k in MODULES},
        "by_family": {k: by_family.get(k, 0.0) / tot for k in ("attention", "mlp")},
        "top_modules": [{"layer": r["layer"], "module": r["module"],
                         "energy_share": r["energy_share"], "rel_fro": r["rel_fro"],
                         "delta_rms": r["delta_rms"]} for r in ranked[:10]],
        "top_modules_normalized": [
            {"layer": r["layer"], "module": r["module"], "delta_rms": r["delta_rms"],
             "rel_fro": r["rel_fro"], "norm_energy_share": r["norm_energy_share"]}
            for r in sorted(rows, key=lambda r: -r["delta_rms"])[:10]],
        "n_trainable_total": int(sum(r["n_trainable"] for r in rows)),
    }


# --------------------------------------------------------------------------- #
# Cross-checkpoint comparisons (composite updates only)
# --------------------------------------------------------------------------- #
def comparison_pairs():
    out = []
    for seed in SEEDS:
        sel = SELECTED[seed]
        out += [
            {"kind": "within_seed", "label": f"seed{seed} 2k->selected({sel})",
             "a": (seed, 2000), "b": (seed, sel)},
            {"kind": "within_seed", "label": f"seed{seed} selected({sel})->30k",
             "a": (seed, sel), "b": (seed, 30000)},
            {"kind": "within_seed", "label": f"seed{seed} 2k->30k",
             "a": (seed, 2000), "b": (seed, 30000)},
        ]
    out += [
        {"kind": "cross_seed", "label": "cross-seed first_saved (s1@2k vs s2@2k)",
         "a": (1, 2000), "b": (2, 2000)},
        {"kind": "cross_seed", "label": "cross-seed selected (s1@4k vs s2@6k)",
         "a": (1, SELECTED[1]), "b": (2, SELECTED[2])},
        {"kind": "cross_seed", "label": "cross-seed endpoint (s1@30k vs s2@30k)",
         "a": (1, 30000), "b": (2, 30000)},
    ]
    return out


def compare_adapters(fa, fb, rows_a, rows_b, scale, spec):
    """Per-module composite-update comparison + weighted aggregates."""
    idx_a = {(r["layer"], r["module"]): r for r in rows_a}
    idx_b = {(r["layer"], r["module"]): r for r in rows_b}
    per = []
    for layer in range(N_LAYERS):
        for mod in MODULES:
            a1, b1 = fa[(layer, mod)]["A"], fa[(layer, mod)]["B"]
            a2, b2 = fb[(layer, mod)]["A"], fb[(layer, mod)]["B"]
            ra, rb = idx_a[(layer, mod)], idx_b[(layer, mod)]
            cos = lowrank_cosine(a1, b1, scale, a2, b2, scale)
            dfro = lowrank_diff_fro(a1, b1, scale, a2, b2, scale)
            per.append({
                "kind": spec["kind"], "label": spec["label"],
                "seed_a": spec["a"][0], "step_a": spec["a"][1],
                "seed_b": spec["b"][0], "step_b": spec["b"][1],
                "layer": layer, "family": FAMILY[mod], "module": mod,
                "cosine": cos,
                "diff_fro": dfro,
                "fro_a": ra["delta_fro"], "fro_b": rb["delta_fro"],
                "rel_norm_change": (rb["delta_fro"] - ra["delta_fro"]) / ra["delta_fro"]
                                   if ra["delta_fro"] > 0 else float("nan"),
                "rel_diff_fro": dfro / ra["delta_fro"] if ra["delta_fro"] > 0 else float("nan"),
                "d_eff_rank": rb["eff_rank_entropy"] - ra["eff_rank_entropy"],
                "d_energy_share": rb["energy_share"] - ra["energy_share"],
                "d_rank90": rb["rank90"] - ra["rank90"],
            })
    cos = np.array([p["cosine"] for p in per], dtype=np.float64)
    w = np.array([p["fro_a"] * p["fro_b"] for p in per], dtype=np.float64)
    ok = np.isfinite(cos)
    agg = {
        "kind": spec["kind"], "label": spec["label"],
        "seed_a": spec["a"][0], "step_a": spec["a"][1],
        "seed_b": spec["b"][0], "step_b": spec["b"][1],
        "n_modules": int(ok.sum()),
        "mean_cosine_unweighted": float(np.mean(cos[ok])),
        "median_cosine_unweighted": float(np.median(cos[ok])),
        "min_cosine": float(np.min(cos[ok])), "max_cosine": float(np.max(cos[ok])),
        "mean_cosine_energy_weighted": float(np.sum(cos[ok] * w[ok]) / np.sum(w[ok])),
        "cosine_weight_definition": "w_m = ||dW_a,m||_F * ||dW_b,m||_F",
        "total_diff_fro": float(math.sqrt(np.sum(
            np.array([p["diff_fro"] for p in per], dtype=np.float64) ** 2))),
        "mean_rel_norm_change": float(np.mean(
            [p["rel_norm_change"] for p in per if math.isfinite(p["rel_norm_change"])])),
        "mean_d_eff_rank": float(np.mean(
            [p["d_eff_rank"] for p in per if math.isfinite(p["d_eff_rank"])])),
    }
    # attention/MLP split of the energy-weighted cosine
    for fam in ("attention", "mlp"):
        sel = np.array([p["family"] == fam for p in per])
        m = sel & ok
        agg[f"mean_cosine_energy_weighted_{fam}"] = float(
            np.sum(cos[m] * w[m]) / np.sum(w[m]))
        agg[f"mean_cosine_unweighted_{fam}"] = float(np.mean(cos[m]))
    return per, agg


# --------------------------------------------------------------------------- #
# Refresh
# --------------------------------------------------------------------------- #
def load_verification():
    if not VERIFICATION.exists():
        raise SystemExit(f"MISSING {VERIFICATION.relative_to(ROOT)} — cannot verify adapter hashes.")
    v = json.loads(VERIFICATION.read_text())
    idx = {}
    for c in v.get("checkpoints", []):
        idx[(int(c["seed"]), int(c["step"]))] = c["adapter"]
    return v, idx


def refresh(verbose=True):
    scale = EXPECTED_ALPHA / EXPECTED_RANK
    if abs(scale - 2.0) > 1e-12:
        raise SystemExit(f"unexpected LoRA scale {scale}")
    vfile, vindex = load_verification()

    # ---- 1. identity + hash verification (before any heavy work) ---------- #
    adapters = []
    for spec in ADAPTERS:
        d = ROOT / spec["path"]
        reject_forbidden_path(spec["path"])
        cfg_p, w_p = d / "adapter_config.json", d / "adapter_model.safetensors"
        for p in (cfg_p, w_p):
            if not p.exists():
                raise SystemExit(f"MISSING required adapter file: {p}")
        cfg_bytes = cfg_p.read_bytes()
        cfg = json.loads(cfg_bytes)
        verify_adapter_config(cfg, spec["path"])
        w_sha = sha256_file(w_p)
        cfg_sha = sha256_bytes(cfg_bytes)
        key = (spec["seed"], spec["step"])
        if key not in vindex:
            raise SystemExit(f"{spec['path']}: seed {key[0]} step {key[1]} absent from "
                             f"{VERIFICATION.name}")
        exp = vindex[key]
        problems = []
        if exp["adapter_path"] != spec["path"]:
            problems.append(f"path {exp['adapter_path']!r} != {spec['path']!r}")
        if exp["adapter_weight_sha256"] != w_sha:
            problems.append(f"weight sha256 {w_sha} != expected {exp['adapter_weight_sha256']}")
        if exp["adapter_config_sha256"] != cfg_sha:
            problems.append(f"config sha256 {cfg_sha} != expected {exp['adapter_config_sha256']}")
        if int(exp["adapter_weight_bytes"]) != w_p.stat().st_size:
            problems.append(f"size {w_p.stat().st_size} != expected {exp['adapter_weight_bytes']}")
        if int(exp["lora_r"]) != EXPECTED_RANK or int(exp["lora_alpha"]) != EXPECTED_ALPHA:
            problems.append(f"verification rank/alpha {exp['lora_r']}/{exp['lora_alpha']}")
        # trainer_state cross-check of the step identity
        ts = d / "trainer_state.json"
        if ts.exists():
            gs = json.loads(ts.read_text()).get("global_step")
            if gs is not None and int(gs) != spec["step"]:
                problems.append(f"trainer_state global_step {gs} != {spec['step']}")
        if problems:
            raise SystemExit(f"HARD FAIL {spec['path']}:\n  - " + "\n  - ".join(problems))
        adapters.append({**spec, "weight_sha256": w_sha, "config_sha256": cfg_sha,
                         # read back from the verification file, NOT re-derived, so the
                         # manifest records an independent expectation rather than an echo
                         "expected_weight_sha256": exp["adapter_weight_sha256"],
                         "expected_config_sha256": exp["adapter_config_sha256"],
                         "expected_weight_bytes": int(exp["adapter_weight_bytes"]),
                         "weight_bytes": w_p.stat().st_size,
                         "target_modules_sorted": sorted(cfg["target_modules"]),
                         "peft_version": cfg.get("peft_version"),
                         "base_model_name_or_path": cfg.get("base_model_name_or_path"),
                         "lora_dropout": cfg.get("lora_dropout")})
        if verbose:
            print(f"  verified {spec['path']}  sha256={w_sha[:16]}...")

    cfg_shas = {a["config_sha256"] for a in adapters}
    tm_sets = {tuple(a["target_modules_sorted"]) for a in adapters}
    if len(tm_sets) != 1:
        raise SystemExit(f"target-module inventory differs across adapters: {tm_sets}")

    # ---- 2. frozen base statistics --------------------------------------- #
    if verbose:
        print("Scanning frozen base weights (norms only; model never instantiated)...")
    base_stats, base_header_bytes = collect_base_stats(verbose=verbose)
    base_cfg_bytes = BASE_CONFIG.read_bytes()
    if verbose:
        print("Hashing base safetensors shard (15 GB, one pass)...")
    base_sha = sha256_file(BASE_WEIGHTS)

    # ---- 3. per-adapter module statistics -------------------------------- #
    factors = {}
    rows_by_ckpt = {}
    conc_by_ckpt = {}
    all_rows = []
    for a in adapters:
        key = (a["seed"], a["step"])
        if verbose:
            print(f"  reading LoRA factors: seed {a['seed']} step {a['step']}")
        factors[key] = load_adapter_factors(ROOT / a["path"])
        rows = module_rows(factors[key], base_stats, a, scale)
        rows_by_ckpt[key] = rows
        conc_by_ckpt[key] = concentration(rows)
        all_rows += rows
        c = conc_by_ckpt[key]
        if verbose:
            print(f"    ||dW||_F(total)={c['total_delta_fro']:.4f} top1={c['top1']:.4f} "
                  f"top10={c['top10']:.4f} gini={c['gini']:.4f} "
                  f"attn={c['by_family']['attention']:.4f}")

    # reference scale for the cross-seed cosine: independent random rank-16
    # updates of each distinct dense shape present in the model
    shapes = sorted({(r["base_rows"], r["base_cols"]) for r in rows_by_ckpt[(1, 2000)]})
    null_cos = {}
    for i, (o, n) in enumerate(shapes):
        mods = sorted({r["module"] for r in rows_by_ckpt[(1, 2000)]
                       if (r["base_rows"], r["base_cols"]) == (o, n)})
        null_cos[f"{o}x{n}"] = {**random_rank_cosine_null(o, n, seed=1000 + i),
                                "modules": mods}
        if verbose:
            nc = null_cos[f"{o}x{n}"]
            print(f"  null cosine {o}x{n} ({','.join(mods)}): "
                  f"mean={nc['mean_cosine']:+.5f} sd={nc['sd_cosine']:.5f} "
                  f"p95|cos|={nc['p95_abs_cosine']:.5f}")

    n_trainable = {f"seed{k[0]}_step{k[1]}": conc_by_ckpt[k]["n_trainable_total"]
                   for k in rows_by_ckpt}
    if len(set(n_trainable.values())) != 1:
        raise SystemExit(f"trainable parameter count differs across adapters: {n_trainable}")

    # ---- 4. comparisons --------------------------------------------------- #
    comps, comp_aggs = [], []
    for spec in comparison_pairs():
        per, agg = compare_adapters(factors[spec["a"]], factors[spec["b"]],
                                    rows_by_ckpt[spec["a"]], rows_by_ckpt[spec["b"]],
                                    scale, spec)
        comps += per
        comp_aggs.append(agg)
        if verbose:
            print(f"  {agg['label']:<38} cos(unw)={agg['mean_cosine_unweighted']:+.4f} "
                  f"cos(energy-w)={agg['mean_cosine_energy_weighted']:+.4f}")

    snapshot = {
        "title": "SFT LoRA parameter localization atlas (weight-space screening)",
        "phase": "Phase 1 — screening only; no causal claim, no behavioral evaluation",
        "lora": {"rank": EXPECTED_RANK, "alpha": EXPECTED_ALPHA, "scale": scale,
                 "formula": "dW = (alpha / rank) * B @ A", "bias": "none",
                 "target_modules": sorted(MODULES), "n_layers": N_LAYERS},
        "n_trainable_lora_values_per_adapter": conc_by_ckpt[(1, 2000)]["n_trainable_total"],
        "selected_checkpoints": {str(k): v for k, v in SELECTED.items()},
        "adapters": adapters,
        "base_model": {
            "dir": str(BASE_MODEL_DIR.relative_to(ROOT)),
            "shard": str(BASE_WEIGHTS.relative_to(ROOT)),
            "shard_sha256": base_sha,
            "shard_bytes": BASE_WEIGHTS.stat().st_size,
            "header_sha256": sha256_bytes(base_header_bytes),
            "config_sha256": sha256_bytes(base_cfg_bytes),
            "dtype": "BF16 (widened losslessly to float32 for norms)",
        },
        "base_module_stats": {f"{l}|{m}": base_stats[(l, m)]
                              for l in range(N_LAYERS) for m in MODULES},
        "modules": all_rows,
        "concentration": {f"seed{k[0]}_step{k[1]}": v for k, v in conc_by_ckpt.items()},
        "comparisons": comps,
        "comparison_aggregates": comp_aggs,
        "random_rank_cosine_null": null_cos,
        "definitions": DEFINITIONS,
    }
    manifest = build_manifest(adapters, snapshot, vfile, sorted(cfg_shas))
    return snapshot, manifest


DEFINITIONS = {
    "delta_fro": "||dW||_F via trace[(B_s^T B_s)(A A^T)] with B_s = (alpha/r) B; dense dW never formed",
    "delta_spec": "||dW||_2 = largest singular value of R_B R_A^T (exact, <=16x16)",
    "delta_rms": "||dW||_F / sqrt(out*in)  — dimension-normalized update size",
    "rel_fro": "||dW||_F / ||W_base||_F",
    "rel_spec": "||dW||_2 / ||W_base||_2 (base sigma_max by deterministic subspace iteration)",
    "energy": "||dW||_F^2",
    "energy_share": "module energy / total adapter energy (196 modules)",
    "norm_energy_share": "delta_rms^2 share — dimension-normalized analogue of energy_share",
    "stable_rank": "||dW||_F^2 / ||dW||_2^2",
    "eff_rank_entropy": "exp(-sum p_j log p_j), p_j = sv_j^2 / sum sv^2",
    "rank90/95/99": "smallest k with top-k singular energy fraction >= 0.90/0.95/0.99",
    "gini": "sum_i sum_j |x_i-x_j| / (2 n^2 mean x) over the 196 module energies",
    "cosine": "<dW_a, dW_b>_F / (||dW_a||_F ||dW_b||_F) on COMPOSITE updates",
    "diff_fro": "||dW_a - dW_b||_F = sqrt(||a||^2 + ||b||^2 - 2<a,b>)",
    "mean_cosine_energy_weighted": "sum_m w_m cos_m / sum_m w_m, w_m = ||dW_a,m||_F ||dW_b,m||_F",
    "dtype": "float64 for all Gram / QR / SVD / norm arithmetic; float32 tensor storage",
}


def build_manifest(adapters, snapshot, vfile, config_shas):
    import matplotlib
    import safetensors
    return {
        "analysis": "CPU-only LoRA parameter localization atlas for full-run SFT adapters",
        "phase": "Phase 1 — weight-space screening",
        "git_commit_at_refresh": _git_commit(),
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "safetensors": safetensors.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "lora": {
            "formula": "dW = (alpha / rank) * B @ A",
            "rank": EXPECTED_RANK, "alpha": EXPECTED_ALPHA,
            "scale": EXPECTED_ALPHA / EXPECTED_RANK,
            "scale_application": "folded into B exactly once; never applied to A as well",
            "bias": "none",
            "target_modules": sorted(MODULES),
            "gauge_note": "A->RA, B->BR^-1 leaves dW invariant; raw ||A||/||B|| are therefore "
                          "never used as importance statistics",
            "dense_dW_materialized": False,
        },
        "definitions": DEFINITIONS,
        "numerical_dtype": "float64 for Gram/QR/SVD/norm arithmetic; tensors stored float32 "
                           "(adapters) and BF16 widened losslessly to float32 (base)",
        "base_spectral_estimator": {
            "method": "randomized subspace iteration (block power method)",
            "block": BASE_SPEC_BLOCK, "max_iter": BASE_SPEC_MAXITER,
            "rel_tol": BASE_SPEC_TOL, "seed": BASE_SPEC_SEED,
            "note": "used ONLY for frozen base ||W||_2; all dW singular values are exact",
        },
        "adapters": [
            {"seed": a["seed"], "step": a["step"], "role": a["role"], "path": a["path"],
             "adapter_weight_bytes": a["weight_bytes"],
             "adapter_weight_sha256": a["weight_sha256"],
             "adapter_config_sha256": a["config_sha256"],
             "expected_weight_sha256_from_verification": a["expected_weight_sha256"],
             "expected_config_sha256_from_verification": a["expected_config_sha256"],
             "expected_weight_bytes_from_verification": a["expected_weight_bytes"],
             "hash_verified": (a["weight_sha256"] == a["expected_weight_sha256"]
                               and a["config_sha256"] == a["expected_config_sha256"]),
             "verification_source": str(VERIFICATION.relative_to(ROOT)),
             "peft_version": a["peft_version"],
             "base_model_name_or_path": a["base_model_name_or_path"]}
            for a in adapters],
        "verification_file": {
            "path": str(VERIFICATION.relative_to(ROOT)),
            "sha256": sha256_file(VERIFICATION),
            "code_commit_recorded": vfile.get("code_commit"),
            "generated_utc": vfile.get("generated_utc"),
            "all_six_adapter_hashes_matched": True,
        },
        "distinct_adapter_config_sha256": config_shas,
        "adapter_config_note": "seed 1 and seed 2 configs differ ONLY in the serialization "
                               "ORDER of target_modules (PEFT writes a Python set); the sorted "
                               "inventory is identical and is what is compared",
        "base_model": snapshot["base_model"],
        "base_shards_used": [str(BASE_WEIGHTS.relative_to(ROOT))],
        "selected_checkpoints": {"seed1": SELECTED[1], "seed2": SELECTED[2]},
        "checkpoint_roles": {f"seed{a['seed']}_step{a['step']}": a["role"] for a in adapters},
        "excluded_paths": {
            "forbidden_tokens": list(FORBIDDEN_PATH_TOKENS),
            "note": "pilot (checkpoints/sft_qwen_delta_seed1_pilot6k) and any 'final' adapter "
                    "are hard-rejected; only the six full-run checkpoints above were read",
        },
        "suites_not_opened": [
            "data/method_comparison/", "data/frozen_unused_test_goods.json",
            "data/ood_new_goods*", "semantic-counterbalancing suite",
            "checkpoints/sft_qwen_delta_seed1_pilot6k",
        ],
        "compute": {
            "device": "CPU only", "gpu": False, "pbs_jobs_submitted": 0,
            "inference": False, "generation": False, "training": False,
            "activations_collected": False, "ablation_run": False,
            "model_instantiated_via_transformers": False,
        },
        "limitations": [
            "WEIGHT-SPACE SCREENING ONLY — parameter-space magnitude is not causal importance",
            "no activation weighting: a module's update is not scaled by how strongly it is "
            "driven by the actual task inputs",
            "no causal necessity or sufficiency result — nothing here was ablated or restored",
            "no behavioral evaluation in this phase; no lambda/eta, no accuracy, no suite opened",
            "no sub-2k localization is possible — the full runs saved no adapter before step 2000",
            "no method comparison (SFT vs GRPO vs sign-only) is made or implied",
            "adapter-to-historical-prediction binding remains NON-cryptographic: the historical "
            "evaluator emitted no run manifest, so these adapters are matched to the behavioral "
            "results by path/step convention, not by hash",
            "base ||W||_2 is a deterministic iterative estimate (tol 1e-7), not an exact SVD; "
            "dW singular values are exact",
            "identical LoRA rank 16 caps every observed effective rank at 16 by construction — "
            "a low effective rank is relative to that cap, not to the full weight-matrix rank",
        ],
    }


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
STATS_COLS = ["seed", "step", "role", "layer", "family", "module", "sub_block",
              "a_rows", "a_cols", "b_rows", "b_cols", "n_trainable",
              "base_rows", "base_cols", "base_numel", "rank", "alpha", "scale",
              "delta_fro", "delta_spec", "delta_rms", "base_fro", "base_spec",
              "rel_fro", "rel_spec", "energy", "energy_share", "energy_rank",
              "energy_cumshare", "norm_energy_share", "norm_energy_rank",
              "stable_rank", "eff_rank_entropy", "energy_top1", "energy_top2",
              "energy_top4", "energy_top8", "energy_top16",
              "rank90", "rank95", "rank99"] + [f"sv{i:02d}" for i in range(1, 17)]

COMP_COLS = ["kind", "label", "seed_a", "step_a", "seed_b", "step_b", "layer",
             "family", "module", "cosine", "diff_fro", "fro_a", "fro_b",
             "rel_norm_change", "rel_diff_fro", "d_eff_rank", "d_energy_share",
             "d_rank90"]


def _fmt(v):
    if isinstance(v, float):
        if not math.isfinite(v):
            return "nan"
        return f"{v:.10g}"
    if isinstance(v, bool):
        return "1" if v else "0"
    return str(v)


def render_stats_csv(snap) -> str:
    import csv as _csv
    import io
    buf = io.StringIO()
    w = _csv.writer(buf, lineterminator="\n")
    w.writerow(STATS_COLS)
    for r in sorted(snap["modules"],
                    key=lambda r: (r["seed"], r["step"], r["layer"], MODULES.index(r["module"]))):
        row = []
        for c in STATS_COLS:
            if c.startswith("sv") and c[2:].isdigit():
                i = int(c[2:]) - 1
                sv = r["singular_values"]
                row.append(_fmt(sv[i]) if i < len(sv) else "")
            else:
                row.append(_fmt(r[c]))
        w.writerow(row)
    return buf.getvalue()


def render_comp_csv(snap) -> str:
    import csv as _csv
    import io
    buf = io.StringIO()
    w = _csv.writer(buf, lineterminator="\n")
    w.writerow(COMP_COLS)
    for r in snap["comparisons"]:
        w.writerow([_fmt(r[c]) for c in COMP_COLS])
    return buf.getvalue()


def _ck(snap, seed, step):
    return snap["concentration"][f"seed{seed}_step{step}"]


def _rows(snap, seed, step):
    return [r for r in snap["modules"] if r["seed"] == seed and r["step"] == step]


def _agg(snap, label_startswith):
    for a in snap["comparison_aggregates"]:
        if a["label"].startswith(label_startswith):
            return a
    raise KeyError(label_startswith)


def spearman(x, y):
    """Spearman rank correlation (no ties expected among 196 distinct energies)."""
    rx = np.argsort(np.argsort(np.asarray(x, dtype=float))).astype(float)
    ry = np.argsort(np.argsort(np.asarray(y, dtype=float))).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    return float(rx @ ry / math.sqrt((rx @ rx) * (ry @ ry)))


def _selected_pair(snap, field):
    """Matched per-module vectors of `field` at the two selected checkpoints."""
    r1 = {(r["layer"], r["module"]): r for r in _rows(snap, 1, SELECTED[1])}
    r2 = {(r["layer"], r["module"]): r for r in _rows(snap, 2, SELECTED[2])}
    keys = sorted(r1)
    return keys, np.array([r1[k][field] for k in keys]), np.array([r2[k][field] for k in keys])


def _spearman_energy(snap):
    _, e1, e2 = _selected_pair(snap, "energy_share")
    return spearman(e1, e2)


def cross_seed_stable_modules(snap, top_n=200):
    """Modules ranked by the worse of the two seeds' selected-checkpoint standing,
    under BOTH raw energy share and dimension-normalized RMS. Screening only."""
    r1 = {(r["layer"], r["module"]): r for r in _rows(snap, 1, SELECTED[1])}
    r2 = {(r["layer"], r["module"]): r for r in _rows(snap, 2, SELECTED[2])}
    cos = {(c["layer"], c["module"]): c["cosine"]
           for c in snap["comparisons"] if c["label"].startswith("cross-seed selected")}
    out = []
    for key in r1:
        a, b = r1[key], r2[key]
        out.append({
            "layer": key[0], "module": key[1], "family": FAMILY[key[1]],
            "energy_rank_worse": max(a["energy_rank"], b["energy_rank"]),
            "norm_rank_worse": max(a["norm_energy_rank"], b["norm_energy_rank"]),
            "energy_share_min": min(a["energy_share"], b["energy_share"]),
            "delta_rms_min": min(a["delta_rms"], b["delta_rms"]),
            "rel_fro_min": min(a["rel_fro"], b["rel_fro"]),
            "eff_rank_max": max(a["eff_rank_entropy"], b["eff_rank_entropy"]),
            "cross_seed_cosine": cos.get(key, float("nan")),
        })
    out.sort(key=lambda d: (max(d["energy_rank_worse"], d["norm_rank_worse"]),
                            d["energy_rank_worse"] + d["norm_rank_worse"]))
    return out[:top_n]


def render_summary_md(snap) -> str:
    scale = snap["lora"]["scale"]
    ntr = snap["n_trainable_lora_values_per_adapter"]
    L = []
    A = L.append
    A("# SFT LoRA parameter localization — weight-space screening (Phase 1)")
    A("")
    A("> **Weight-space screening only. No causal importance is implied.** Full-run "
      "SFT adapters only — the historical pilot is excluded. No inference, generation, "
      "training, activation collection or ablation was run to produce this document, "
      "and no frozen or untouched evaluation suite was opened.")
    A("")
    A(f"Update definition: `dW = (alpha/rank) * B @ A` with alpha={snap['lora']['alpha']}, "
      f"rank={snap['lora']['rank']}, scale={scale:g}. The dense `dW` is never formed; "
      "norms and singular values come from 16x16 Gram/QR/SVD identities. Raw `A`/`B` "
      "norms are never used as importance statistics (gauge freedom `A->RA, B->BR^-1`).")
    A("")

    # --- headline ----------------------------------------------------------- #
    c1 = _ck(snap, 1, SELECTED[1])
    c2 = _ck(snap, 2, SELECTED[2])
    rows1 = _rows(snap, 1, SELECTED[1])
    rows2 = _rows(snap, 2, SELECTED[2])
    sel_agg = _agg(snap, "cross-seed selected")
    med_eff = float(np.median([r["eff_rank_entropy"] for r in rows1 + rows2]))
    med_r90 = float(np.median([r["rank90"] for r in rows1 + rows2]))
    A("## Headline reading")
    A("")
    A("The screening does **not** support a simple 'a few modules / a few directions' "
      "picture. Four descriptive findings, each developed below:")
    A("")
    A(f"1. **Update energy is broadly spread, not concentrated.** At the selected "
      f"checkpoints the single largest of 196 modules holds {c1['top1']:.1%} / "
      f"{c2['top1']:.1%} of energy and the top 10 hold {c1['top10']:.1%} / "
      f"{c2['top10']:.1%}. The raw Gini ({c1['gini']:.2f} / {c2['gini']:.2f}) is driven "
      f"largely by matrix SIZE: once updates are divided by sqrt(out*in) the Gini falls "
      f"to {c1['norm_gini']:.2f} / {c2['norm_gini']:.2f}, i.e. close to an even "
      f"per-element update across the whole adapted network.")
    A(f"2. **The effective rank is NOT far below 16.** Median entropy effective rank is "
      f"{med_eff:.1f} of the rank-16 cap, and the median module needs rank "
      f"{med_r90:.0f} for 90% of its energy. Only a small minority of modules are "
      f"near rank-1.")
    A(f"3. **The two seeds agree on WHERE but not on WHICH DIRECTION.** The per-module "
      f"energy ordering is almost identical across seeds (Spearman rho "
      f"{_spearman_energy(snap):.2f}), yet the matched-role composite-update cosine is "
      f"only {sel_agg['mean_cosine_energy_weighted']:+.3f}. Same places, essentially "
      f"unrelated directions.")
    _band = lambda seed, lo, hi: float(np.mean(  # noqa: E731
        [r["eff_rank_entropy"] for r in _rows(snap, seed, SELECTED[seed])
         if lo <= r["layer"] <= hi]))
    _e = max(_band(s, 0, 9) for s in SEEDS)
    _f = min(_band(s, 24, 27) for s in SEEDS)
    A(f"4. **Depth shifts with training, and rank structure is depth-structured.** At 2k "
      f"the update is concentrated late (layers 19-27 hold ~49% of energy); by the "
      f"selected checkpoint and 30k it has spread toward a roughly flat depth profile. "
      f"Independently, mean effective rank falls monotonically with depth — {_e:.1f} in "
      f"layers 0-9 down to {_f:.1f} in layers 24-27, in both seeds — the one clearly "
      f"reproducible low-rank structure the screening finds.")
    A("")
    A("Read together, these argue that a Phase-2 ablation should be organized around "
      "coarse, pre-registered partitions (module family, depth block, rank truncation) "
      "with parameter-matched controls — **not** around a handful of individually "
      "'important' modules, which this screening does not identify.")
    A("")

    # --- Q1 ---------------------------------------------------------------- #
    A("## 1. How many exact trainable LoRA values are present?")
    A("")
    A(f"**{ntr:,} trainable LoRA values per adapter**, identical across all six "
      f"checkpoints: {N_LAYERS} layers x {len(MODULES)} modules = "
      f"{N_LAYERS * len(MODULES)} adapted projections, each contributing "
      f"`r*in + out*r` values at r={snap['lora']['rank']}.")
    A("")
    A("| module | A shape | B shape | dense dW shape | trainable values | x28 layers |")
    A("|---|---|---|---|---|---|")
    r0 = {r["module"]: r for r in _rows(snap, 1, 2000) if r["layer"] == 0}
    per_layer = 0
    for m in MODULES:
        r = r0[m]
        per_layer += r["n_trainable"]
        A(f"| `{m}` | [{r['a_rows']}, {r['a_cols']}] | [{r['b_rows']}, {r['b_cols']}] | "
          f"[{r['base_rows']}, {r['base_cols']}] | {r['n_trainable']:,} | "
          f"{r['n_trainable'] * N_LAYERS:,} |")
    A(f"| **per layer** | | | | **{per_layer:,}** | **{ntr:,}** |")
    A("")

    # --- Q2 ---------------------------------------------------------------- #
    A("## 2. How concentrated is update energy?")
    A("")
    A("Energy = `||dW||_F^2`, shares over the 196 adapted modules of one adapter.")
    A("")
    A("| checkpoint | role | total `||dW||_F` | top-1 | top-5 | top-10 | top-5% | top-10% | "
      "top-25% | Gini |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for a in snap["adapters"]:
        c = _ck(snap, a["seed"], a["step"])
        A(f"| seed {a['seed']} @ {a['step']} | {a['role']} | {c['total_delta_fro']:.4f} | "
          f"{c['top1']:.4f} | {c['top5']:.4f} | {c['top10']:.4f} | {c['top5pct']:.4f} | "
          f"{c['top10pct']:.4f} | {c['top25pct']:.4f} | {c['gini']:.4f} |")
    A("")
    A("Dimension-normalized counterpart (energy of `delta_rms = ||dW||_F/sqrt(out*in)`), "
      "which removes the mechanical advantage of the large MLP matrices:")
    A("")
    A("| checkpoint | norm top-1 | norm top-5 | norm top-10 | norm top-10% | norm Gini |")
    A("|---|---|---|---|---|---|")
    for a in snap["adapters"]:
        c = _ck(snap, a["seed"], a["step"])
        A(f"| seed {a['seed']} @ {a['step']} | {c['norm_top1']:.4f} | {c['norm_top5']:.4f} | "
          f"{c['norm_top10']:.4f} | {c['norm_top10pct']:.4f} | {c['norm_gini']:.4f} |")
    A("")

    # --- Q3 ---------------------------------------------------------------- #
    A("## 3. Which module families carry the largest raw and normalized updates?")
    A("")
    A("Raw energy share by module type (each row sums to 1 across the seven modules):")
    A("")
    A("| checkpoint | " + " | ".join(f"`{SHORT[m]}`" for m in MODULES) + " | attention | MLP |")
    A("|---" * (len(MODULES) + 3) + "|")
    for a in snap["adapters"]:
        c = _ck(snap, a["seed"], a["step"])
        A(f"| seed {a['seed']} @ {a['step']} | "
          + " | ".join(f"{c['by_module'][m]:.4f}" for m in MODULES)
          + f" | {c['by_family']['attention']:.4f} | {c['by_family']['mlp']:.4f} |")
    A("")
    A("Median dimension-normalized update `delta_rms` (x1e-4) by module type:")
    A("")
    A("| checkpoint | " + " | ".join(f"`{SHORT[m]}`" for m in MODULES) + " |")
    A("|---" * (len(MODULES) + 1) + "|")
    for a in snap["adapters"]:
        rows = _rows(snap, a["seed"], a["step"])
        vals = []
        for m in MODULES:
            v = np.median([r["delta_rms"] for r in rows if r["module"] == m])
            vals.append(f"{v * 1e4:.3f}")
        A(f"| seed {a['seed']} @ {a['step']} | " + " | ".join(vals) + " |")
    A("")
    A("Median relative Frobenius update `||dW||_F/||W||_F` (x1e-4) by module type:")
    A("")
    A("| checkpoint | " + " | ".join(f"`{SHORT[m]}`" for m in MODULES) + " |")
    A("|---" * (len(MODULES) + 1) + "|")
    for a in snap["adapters"]:
        rows = _rows(snap, a["seed"], a["step"])
        vals = []
        for m in MODULES:
            v = np.median([r["rel_fro"] for r in rows if r["module"] == m])
            vals.append(f"{v * 1e4:.3f}")
        A(f"| seed {a['seed']} @ {a['step']} | " + " | ".join(vals) + " |")
    A("")

    # --- Q4 ---------------------------------------------------------------- #
    A("## 4. Are selected-checkpoint updates early, middle or late in depth?")
    A("")
    A("Energy share by depth third (layers 0-9 / 10-18 / 19-27), and the three "
      "highest-energy individual layers:")
    A("")
    A("| checkpoint | early 0-9 | middle 10-18 | late 19-27 | top layers (raw energy) |")
    A("|---|---|---|---|---|")
    for a in snap["adapters"]:
        c = _ck(snap, a["seed"], a["step"])
        bl = {int(k): v for k, v in c["by_layer"].items()}
        early = sum(v for k, v in bl.items() if k <= 9)
        mid = sum(v for k, v in bl.items() if 10 <= k <= 18)
        late = sum(v for k, v in bl.items() if k >= 19)
        top = sorted(bl.items(), key=lambda kv: -kv[1])[:3]
        A(f"| seed {a['seed']} @ {a['step']} | {early:.4f} | {mid:.4f} | {late:.4f} | "
          + ", ".join(f"L{k} ({v:.3f})" for k, v in top) + " |")
    A("")
    A("Same split under the dimension-normalized `delta_rms^2` weighting:")
    A("")
    A("| checkpoint | early 0-9 | middle 10-18 | late 19-27 |")
    A("|---|---|---|---|")
    for a in snap["adapters"]:
        rows = _rows(snap, a["seed"], a["step"])
        tot = sum(r["delta_rms"] ** 2 for r in rows)
        early = sum(r["delta_rms"] ** 2 for r in rows if r["layer"] <= 9) / tot
        mid = sum(r["delta_rms"] ** 2 for r in rows if 10 <= r["layer"] <= 18) / tot
        late = sum(r["delta_rms"] ** 2 for r in rows if r["layer"] >= 19) / tot
        A(f"| seed {a['seed']} @ {a['step']} | {early:.4f} | {mid:.4f} | {late:.4f} |")
    A("")
    A("Top-10 modules by raw energy share at each selected checkpoint:")
    A("")
    for seed in SEEDS:
        c = _ck(snap, seed, SELECTED[seed])
        A(f"- **seed {seed} @ {SELECTED[seed]}**: "
          + ", ".join(f"L{t['layer']}.{SHORT[t['module']]} ({t['energy_share']:.4f})"
                      for t in c["top_modules"]))
    A("")
    A("Top-10 modules by dimension-normalized `delta_rms` at each selected checkpoint:")
    A("")
    for seed in SEEDS:
        c = _ck(snap, seed, SELECTED[seed])
        A(f"- **seed {seed} @ {SELECTED[seed]}**: "
          + ", ".join(f"L{t['layer']}.{SHORT[t['module']]} ({t['delta_rms']:.3e})"
                      for t in c["top_modules_normalized"]))
    A("")

    # --- Q5 ---------------------------------------------------------------- #
    A("## 5. What ranks retain 90% / 95% / 99% of update energy?")
    A("")
    A("Per-module singular spectra of `dW` (exact; capped at rank 16 by construction). "
      "Values are medians across the 196 modules, with the [min, max] range.")
    A("")
    A("| checkpoint | eff. rank (entropy) | stable rank | rank@90% | rank@95% | rank@99% | "
      "top-1 energy | top-4 energy |")
    A("|---|---|---|---|---|---|---|---|")
    for a in snap["adapters"]:
        rows = _rows(snap, a["seed"], a["step"])
        def med(k):
            return float(np.median([r[k] for r in rows]))
        def rng(k):
            v = [r[k] for r in rows]
            return f"[{min(v):g}, {max(v):g}]"
        A(f"| seed {a['seed']} @ {a['step']} | {med('eff_rank_entropy'):.2f} "
          f"{rng('eff_rank_entropy').replace('[', '[').replace(', ', '–')} | "
          f"{med('stable_rank'):.2f} | {med('rank90'):.0f} {rng('rank90')} | "
          f"{med('rank95'):.0f} {rng('rank95')} | {med('rank99'):.0f} {rng('rank99')} | "
          f"{med('energy_top1'):.4f} | {med('energy_top4'):.4f} |")
    A("")
    A("Fraction of modules whose 90% energy fits in rank <= k:")
    A("")
    A("| checkpoint | k=1 | k=2 | k=4 | k=8 | k=16 |")
    A("|---|---|---|---|---|---|")
    for a in snap["adapters"]:
        rows = _rows(snap, a["seed"], a["step"])
        vals = [np.mean([r["rank90"] <= k for r in rows]) for k in (1, 2, 4, 8, 16)]
        A(f"| seed {a['seed']} @ {a['step']} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    A("")
    A("**Effective rank falls sharply with depth.** Averaging effective rank over the "
      "seven modules of each layer at the selected checkpoints:")
    A("")
    A("| layer band | seed 1 mean eff. rank | seed 2 mean eff. rank |")
    A("|---|---|---|")
    bands = {}
    for lo, hi, lab in ((0, 9, "0-9 (early)"), (10, 17, "10-17 (middle)"),
                        (18, 23, "18-23 (late)"), (24, 27, "24-27 (final)")):
        v = []
        for seed in SEEDS:
            rows = [r for r in _rows(snap, seed, SELECTED[seed]) if lo <= r["layer"] <= hi]
            v.append(float(np.mean([r["eff_rank_entropy"] for r in rows])))
        bands[lab] = v
        A(f"| {lab} | {v[0]:.2f} | {v[1]:.2f} |")
    A("")
    e_lo, e_hi = min(bands["0-9 (early)"]), max(bands["0-9 (early)"])
    f_lo, f_hi = min(bands["24-27 (final)"]), max(bands["24-27 (final)"])
    A(f"Both seeds show the same monotone decline: mean effective rank falls from "
      f"{e_lo:.1f}-{e_hi:.1f} in layers 0-9 to {f_lo:.1f}-{f_hi:.1f} in layers 24-27. So "
      f"the update is closest to genuinely low-rank exactly where it is *not* largest at "
      f"the selected checkpoint. This is the clearest depth-structured signal in the "
      f"screening and it is what makes the Tier-4 rank-truncation arm worth running: a "
      f"rank-2 truncation should be nearly lossless late and lossy early, which is a "
      f"testable prediction rather than a magnitude ranking.")
    A("")
    mod_eff = {m: float(np.mean([r["eff_rank_entropy"] for r in rows1 + rows2
                                 if r["module"] == m])) for m in MODULES}
    hi_m = max(mod_eff, key=mod_eff.get)
    lo_m = min(mod_eff, key=mod_eff.get)
    A(f"Effective rank also differs by module type — `{hi_m}` is the highest "
      f"(mean {mod_eff[hi_m]:.1f}) and `{lo_m}` the lowest ({mod_eff[lo_m]:.1f}) — but "
      f"note that `{hi_m}` is one of the smallest matrices and one of the smallest energy "
      f"contributors, so this is a statement about the shape of its update, not its "
      f"importance. Full ordering: "
      + ", ".join(f"`{SHORT[m]}` {mod_eff[m]:.1f}"
                  for m in sorted(MODULES, key=lambda m: -mod_eff[m])) + ".")
    A("")

    # --- Q6 ---------------------------------------------------------------- #
    A("## 6. Which patterns reproduce across both seeds?")
    A("")
    A("Composite-update cosine between matched roles. `w_m = ||dW_a,m||_F * ||dW_b,m||_F` "
      "keeps a high cosine on a negligible module from dominating.")
    A("")
    A("| comparison | mean cosine (unweighted) | mean cosine (energy-weighted) | median | "
      "min | max | attention (w) | MLP (w) |")
    A("|---|---|---|---|---|---|---|---|")
    for a in snap["comparison_aggregates"]:
        if a["kind"] != "cross_seed":
            continue
        A(f"| {a['label']} | {a['mean_cosine_unweighted']:+.4f} | "
          f"{a['mean_cosine_energy_weighted']:+.4f} | {a['median_cosine_unweighted']:+.4f} | "
          f"{a['min_cosine']:+.4f} | {a['max_cosine']:+.4f} | "
          f"{a['mean_cosine_energy_weighted_attention']:+.4f} | "
          f"{a['mean_cosine_energy_weighted_mlp']:+.4f} |")
    A("")
    null = snap.get("random_rank_cosine_null", {})
    if null:
        A("Reference scale for those cosines — the composite cosine between two "
          "**independent** random rank-16 updates of the same dense shape "
          f"({next(iter(null.values()))['n_draws']} draws per shape, fixed seed). This is "
          "a scale, not a significance test:")
        A("")
        A("| dense shape | modules | null mean cosine | null sd | null p95 \\|cos\\| | "
          "null max \\|cos\\| |")
        A("|---|---|---|---|---|---|")
        for shape, nc in sorted(null.items()):
            A(f"| {shape} | " + ", ".join(f"`{SHORT[m]}`" for m in nc["modules"])
              + f" | {nc['mean_cosine']:+.5f} | {nc['sd_cosine']:.5f} | "
                f"{nc['p95_abs_cosine']:.5f} | {nc['max_abs_cosine']:.5f} |")
        A("")
        worst = max(nc["p95_abs_cosine"] for nc in null.values())
        A(f"The observed cross-seed cosines ({sel_agg['mean_cosine_energy_weighted']:+.4f} "
          f"at selection, {_agg(snap, 'cross-seed endpoint')['mean_cosine_energy_weighted']:+.4f} "
          f"at 30k) are **above** this null (largest null p95 |cos| = {worst:.4f}) and "
          f"consistently positive rather than sign-random, so the agreement is real — but "
          f"it is small in absolute terms. Two seeds trained on the same objective end up "
          f"with update directions that are, per module, close to orthogonal.")
        A("")
    A("Rank correlation of the per-module energy ordering between seeds at the "
      "selected checkpoints, and the overlap of their top-k module sets:")
    A("")
    keys, e1, e2 = _selected_pair(snap, "energy_share")
    _, n1, n2 = _selected_pair(snap, "delta_rms")
    A(f"- Spearman rho (raw energy share, seed1@{SELECTED[1]} vs seed2@{SELECTED[2]}): "
      f"**{spearman(e1, e2):.4f}**")
    A(f"- Spearman rho (dimension-normalized `delta_rms`): **{spearman(n1, n2):.4f}**")
    for k in (5, 10, 20, 49):
        s1 = {keys[i] for i in np.argsort(-e1)[:k]}
        s2 = {keys[i] for i in np.argsort(-e2)[:k]}
        A(f"- top-{k} raw-energy module overlap: **{len(s1 & s2)}/{k}**")
    A("")
    A("**Reconciling the two numbers.** These say different things and both matter. The "
      "near-perfect rank correlation means the seeds allocate update *magnitude* to the "
      "same modules — that pattern is a reproducible property of the task and "
      "architecture. The near-zero cosine means they do so along *different directions* "
      "within each module. So 'the same modules dominate both seeds' is true only in the "
      "magnitude sense. A Phase-2 group chosen from this table is therefore reproducible "
      "as a **set of locations**; nothing here licenses treating the two seeds' updates "
      "as the same learned solution, and any ablation must be run per seed rather than "
      "pooled.")
    A("")

    # --- Q7 / Q8 ------------------------------------------------------------ #
    A("## 7. What is already present at 2k, and 8. what keeps changing to 30k?")
    A("")
    A("| comparison | mean cosine (unweighted) | mean cosine (energy-weighted) | "
      "total `||dW_a - dW_b||_F` | mean relative norm change | mean d(eff. rank) |")
    A("|---|---|---|---|---|---|")
    for a in snap["comparison_aggregates"]:
        if a["kind"] != "within_seed":
            continue
        A(f"| {a['label']} | {a['mean_cosine_unweighted']:+.4f} | "
          f"{a['mean_cosine_energy_weighted']:+.4f} | {a['total_diff_fro']:.4f} | "
          f"{a['mean_rel_norm_change']:+.4f} | {a['mean_d_eff_rank']:+.3f} |")
    A("")
    A("Total update magnitude by checkpoint (how much of the endpoint's norm and "
      "energy geometry already exists at 2k):")
    A("")
    A("| seed | 2k `||dW||_F` | selected `||dW||_F` | 30k `||dW||_F` | 2k/30k norm ratio | "
      "2k/selected norm ratio |")
    A("|---|---|---|---|---|---|")
    for seed in SEEDS:
        f2 = _ck(snap, seed, 2000)["total_delta_fro"]
        fs = _ck(snap, seed, SELECTED[seed])["total_delta_fro"]
        f30 = _ck(snap, seed, 30000)["total_delta_fro"]
        A(f"| {seed} | {f2:.4f} | {fs:.4f} | {f30:.4f} | {f2 / f30:.4f} | {f2 / fs:.4f} |")
    A("")
    A("Modules whose direction moves most between the selected checkpoint and 30k "
      "(lowest composite cosine among the 25 highest-energy modules at selection):")
    A("")
    for seed in SEEDS:
        comps = [c for c in snap["comparisons"]
                 if c["kind"] == "within_seed" and c["seed_a"] == seed
                 and c["step_a"] == SELECTED[seed] and c["step_b"] == 30000]
        sel_rows = {(r["layer"], r["module"]): r for r in _rows(snap, seed, SELECTED[seed])}
        big = [c for c in comps if sel_rows[(c["layer"], c["module"])]["energy_rank"] <= 25]
        big.sort(key=lambda c: c["cosine"])
        A(f"- **seed {seed}**: "
          + ", ".join(f"L{c['layer']}.{SHORT[c['module']]} (cos {c['cosine']:+.3f})"
                      for c in big[:6]))
    A("")

    # --- Q9 ----------------------------------------------------------------- #
    A("## 9. Does the evidence justify proceeding to causal ablation?")
    A("")
    A("Descriptively, at the two frozen-selected checkpoints:")
    A("")
    A(f"- Raw update energy is **{'concentrated' if max(c1['top10'], c2['top10']) > 0.5 else 'spread'}**: "
      f"the top 10 of 196 modules hold {c1['top10']:.1%} (seed 1) and {c2['top10']:.1%} "
      f"(seed 2); the top 10% hold {c1['top10pct']:.1%} / {c2['top10pct']:.1%}; "
      f"Gini {c1['gini']:.3f} / {c2['gini']:.3f}.")
    A(f"- Under dimension normalization the picture is "
      f"{'similar' if abs(c1['norm_top10pct'] - c1['top10pct']) < 0.15 else 'different'}: "
      f"normalized top-10% share {c1['norm_top10pct']:.1%} / {c2['norm_top10pct']:.1%} "
      f"(Gini {c1['norm_gini']:.3f} / {c2['norm_gini']:.3f}).")
    A(f"- Median per-module entropy effective rank is **{med_eff:.2f} of the rank-16 cap**, "
      f"so the update is not collapsing to a single direction inside each module.")
    A(f"- Matched-role cross-seed agreement at selection: energy-weighted mean composite "
      f"cosine **{sel_agg['mean_cosine_energy_weighted']:+.4f}** "
      f"(unweighted {sel_agg['mean_cosine_unweighted']:+.4f}).")
    A("")
    A("**Verdict on the original hypothesis.** The hypothesis was that the rapid SFT "
      "behavioral change is concentrated in a small number of layers, modules or "
      "low-rank directions. In weight space, the screening finds **no such "
      "concentration**: energy is spread across depth and across the seven module types "
      "roughly in proportion to matrix size, and the within-module effective rank sits "
      "near the middle of the rank-16 cap rather than near 1. That is a genuine negative "
      "for the weight-space form of the hypothesis.")
    A("")
    A("**What this does and does not license.** These are magnitudes and directions in "
      "weight space. They say where the optimizer moved, not which movement matters for "
      "the measured behavioral change. Fast behavioral movement early in training does "
      "not imply that few parameters carry it, and a large `||dW||` does not imply causal "
      "importance — an update in a direction the task inputs never excite contributes "
      "nothing. The converse also holds: a diffuse weight update is **not** evidence "
      "that the behavioral change is diffuse, because a small number of directions could "
      "still carry all of the behaviorally relevant effect. Only Phase 2 can separate "
      "these. The screening is informative enough to *define* a Phase-2 ablation design "
      "with a small number of pre-registered groups; it cannot substitute for one.")
    A("")

    # --- shortlist ---------------------------------------------------------- #
    A("## Proposed Phase-2 shortlist (NOT executed)")
    A("")
    A("Hierarchical, cross-seed-stable groups for a later necessity (zero-out) / "
      "sufficiency (keep-only) ablation. Nothing below has been run.")
    A("")
    A("**Tier 1 — module-family partitions** (coarse, unambiguous, no selection on "
      "per-module magnitude):")
    A("")
    A("1. `attention-only` — keep LoRA on q/k/v/o, zero gate/up/down.")
    A("2. `mlp-only` — keep LoRA on gate/up/down, zero q/k/v/o.")
    A("3. `qkv-only` vs `o_proj-only` — splits attention into what the block reads "
      "versus how it writes back.")
    A("")
    A("**Tier 2 — depth blocks** (early 0-9 / middle 10-18 / late 19-27), each kept-only "
      "and each zeroed, so necessity and sufficiency are separated.")
    A("")
    A("**Tier 3 — cross-seed-stable module group.** Modules ranked by the *worse* of the "
      "two seeds' standing under BOTH raw energy share and dimension-normalized "
      "`delta_rms`, so nothing enters because it is large in one seed only. Note the "
      "final column: these locations are stable in magnitude but NOT in direction, so "
      "this group must be ablated **per seed**, never pooled, and it is the weakest tier "
      "here precisely because the concentration it presumes is not present in the data:")
    A("")
    A("| # | layer | module | family | worse energy rank | worse normalized rank | "
      "min energy share | min `delta_rms` | cross-seed cosine |")
    A("|---|---|---|---|---|---|---|---|---|")
    for i, d in enumerate(cross_seed_stable_modules(snap, 15), start=1):
        A(f"| {i} | {d['layer']} | `{d['module']}` | {d['family']} | "
          f"{d['energy_rank_worse']} | {d['norm_rank_worse']} | "
          f"{d['energy_share_min']:.4f} | {d['delta_rms_min']:.3e} | "
          f"{d['cross_seed_cosine']:+.3f} |")
    A("")
    A("**Tier 4 — controls that must accompany every group above:**")
    A("")
    A("- A **random module group matched on trainable-parameter count** (and separately "
      "on dense-element count), resampled over several draws — without it, any effect of "
      "a chosen group is confounded with simply removing that many parameters.")
    A("- **Rank truncation** of every retained module to rank 1 / 2 / 4 / 8 (project `dW` "
      "onto its top-k singular directions), which tests the low-rank claim directly "
      "rather than by proxy. Given the depth-dependent effective rank above, run this "
      "both uniformly and split by depth band — the pre-registered prediction is that "
      "a rank-2 truncation costs little in layers 24-27 and much more in layers 0-17.")
    A("- A **full-adapter** arm and a **zero-adapter** (base) arm to bracket the range.")
    A("")
    A("Each arm must be run for **both seeds**, evaluated on the same behavioral "
      "instrument, and read against the base and full-adapter brackets. Group selection "
      "must be frozen before any behavioral number is looked at.")
    A("")

    # --- limitations --------------------------------------------------------- #
    A("## Limitations")
    A("")
    for lim in ["weight-space screening only — no causal necessity or sufficiency result",
                "no activation weighting; unexcited directions are counted the same as "
                "excited ones",
                "no behavioral evaluation in this phase; no suite was opened",
                "no sub-2k localization — the full runs saved no adapter before step 2000",
                "no method comparison (SFT vs GRPO vs sign-only) is made or implied",
                "rank 16 caps every effective rank by construction",
                "base `||W||_2` is a deterministic iterative estimate; `dW` singular "
                "values are exact",
                "adapter-to-historical-prediction binding remains non-cryptographic"]:
        A(f"- {lim}")
    A("")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
FOOTER = ("weight-space screening  ·  no causal importance implied  ·  "
          "full-run SFT adapters only (no pilot)  ·  dW = (alpha/r) B A, alpha/r = 2")


def _grid(snap, seed, step, field):
    rows = _rows(snap, seed, step)
    g = np.full((N_LAYERS, len(MODULES)), np.nan)
    for r in rows:
        g[r["layer"], MODULES.index(r["module"])] = r[field]
    return g


def _heat(ax, g, title, cmap="viridis"):
    im = ax.imshow(g.T, aspect="auto", origin="lower", cmap=cmap, interpolation="nearest")
    ax.set_yticks(range(len(MODULES)))
    ax.set_yticklabels([SHORT[m] for m in MODULES], fontsize=7)
    ax.set_xticks(range(0, N_LAYERS, 3))
    ax.set_xticklabels(range(0, N_LAYERS, 3), fontsize=7)
    ax.set_xlabel("layer", fontsize=8)
    ax.set_title(title, fontsize=9)
    return im


def render_atlas(snap, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(16.5, 15.5))
    gs = fig.add_gridspec(4, 4, hspace=0.55, wspace=0.35,
                          left=0.055, right=0.975, top=0.868, bottom=0.055)

    # rows 1-2: heatmaps at the two selected checkpoints, raw-relative and normalized
    for j, seed in enumerate(SEEDS):
        step = SELECTED[seed]
        ax = fig.add_subplot(gs[0, j * 2:j * 2 + 2])
        im = _heat(ax, _grid(snap, seed, step, "rel_fro") * 1e4,
                   f"A{j + 1}. relative Frobenius update  ||dW||_F/||W||_F  (x1e-4)\n"
                   f"seed {seed} @ {step} (selected)")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.015).ax.tick_params(labelsize=6)
        ax2 = fig.add_subplot(gs[1, j * 2:j * 2 + 2])
        im2 = _heat(ax2, _grid(snap, seed, step, "delta_rms") * 1e4,
                    f"B{j + 1}. dimension-normalized RMS update  ||dW||_F/sqrt(out*in)  (x1e-4)\n"
                    f"seed {seed} @ {step} (selected)", cmap="magma")
        fig.colorbar(im2, ax=ax2, fraction=0.03, pad=0.015).ax.tick_params(labelsize=6)

    colors = {1: "#1f5fd0", 2: "#d02b2b"}
    styles = {"first_saved": ":", "selected": "-", "endpoint": "--"}

    # C. cumulative update-energy curve over rank-ordered modules
    ax = fig.add_subplot(gs[2, 0])
    for a in snap["adapters"]:
        rows = sorted(_rows(snap, a["seed"], a["step"]), key=lambda r: r["energy_rank"])
        y = np.cumsum([r["energy_share"] for r in rows])
        ax.plot(np.arange(1, len(y) + 1), y, styles[a["role"]], color=colors[a["seed"]],
                lw=1.5, label=f"s{a['seed']} @{a['step']} ({a['role']})")
    ax.plot([1, 196], [1 / 196, 1.0], color="#888", lw=0.8, label="uniform")
    ax.set_xscale("log")
    ax.set_xlabel("modules, ranked by energy", fontsize=8)
    ax.set_ylabel("cumulative energy share", fontsize=8)
    ax.set_title("C. cumulative update energy (raw)", fontsize=9)
    ax.grid(alpha=0.25); ax.legend(fontsize=5.6, loc="lower right")
    ax.tick_params(labelsize=7)

    # C2. dimension-normalized cumulative curve
    ax = fig.add_subplot(gs[2, 1])
    for a in snap["adapters"]:
        rows = _rows(snap, a["seed"], a["step"])
        ne = np.sort(np.array([r["delta_rms"] ** 2 for r in rows]))[::-1]
        y = np.cumsum(ne) / ne.sum()
        ax.plot(np.arange(1, len(y) + 1), y, styles[a["role"]], color=colors[a["seed"]], lw=1.5)
    ax.plot([1, 196], [1 / 196, 1.0], color="#888", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("modules, ranked by normalized energy", fontsize=8)
    ax.set_ylabel("cumulative share", fontsize=8)
    ax.set_title("D. cumulative update energy (dimension-normalized)", fontsize=9)
    ax.grid(alpha=0.25); ax.tick_params(labelsize=7)

    # E. per-layer energy profile
    ax = fig.add_subplot(gs[2, 2])
    for a in snap["adapters"]:
        c = _ck(snap, a["seed"], a["step"])
        bl = {int(k): v for k, v in c["by_layer"].items()}
        ax.plot(sorted(bl), [bl[k] for k in sorted(bl)], styles[a["role"]],
                color=colors[a["seed"]], lw=1.4)
    ax.axhline(1 / N_LAYERS, color="#888", lw=0.8, ls="-")
    ax.set_xlabel("layer", fontsize=8); ax.set_ylabel("energy share", fontsize=8)
    ax.set_title("E. energy share by layer (grey = uniform)", fontsize=9)
    ax.grid(alpha=0.25); ax.tick_params(labelsize=7)

    # F. attention vs MLP share
    ax = fig.add_subplot(gs[2, 3])
    labels = [f"s{a['seed']}\n@{a['step']}" for a in snap["adapters"]]
    attn = [_ck(snap, a["seed"], a["step"])["by_family"]["attention"] for a in snap["adapters"]]
    mlp = [_ck(snap, a["seed"], a["step"])["by_family"]["mlp"] for a in snap["adapters"]]
    x = np.arange(len(labels))
    ax.bar(x, attn, 0.62, label="attention (q/k/v/o)", color="#1f5fd0")
    ax.bar(x, mlp, 0.62, bottom=attn, label="MLP (gate/up/down)", color="#e0a13a")
    # parameter-count reference line: attention share of trainable LoRA values
    r0 = _rows(snap, 1, 2000)
    attn_par = sum(r["n_trainable"] for r in r0 if r["family"] == "attention") \
        / sum(r["n_trainable"] for r in r0)
    ax.axhline(attn_par, color="black", ls=":", lw=1.2,
               label=f"attention share of LoRA params ({attn_par:.2f})")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylim(0, 1.42)
    ax.set_ylabel("energy share", fontsize=8)
    ax.set_title("F. attention vs MLP energy share", fontsize=9)
    ax.legend(fontsize=5.8, loc="upper center"); ax.tick_params(labelsize=7)

    # G. per-module-type share across checkpoints
    ax = fig.add_subplot(gs[3, 0:2])
    w = 0.13
    for i, a in enumerate(snap["adapters"]):
        c = _ck(snap, a["seed"], a["step"])
        ax.bar(np.arange(len(MODULES)) + (i - 2.5) * w,
               [c["by_module"][m] for m in MODULES], w,
               color=colors[a["seed"]], alpha=0.45 + 0.25 * ["first_saved", "selected", "endpoint"].index(a["role"]),
               label=f"s{a['seed']} @{a['step']}")
    par = {}
    for m in MODULES:
        par[m] = sum(r["n_trainable"] for r in r0 if r["module"] == m) \
            / sum(r["n_trainable"] for r in r0)
    ax.plot(np.arange(len(MODULES)), [par[m] for m in MODULES], "k:o", ms=3.5, lw=1.1,
            label="share of LoRA params")
    ax.set_xticks(range(len(MODULES)))
    ax.set_xticklabels([SHORT[m] for m in MODULES], fontsize=8)
    ax.set_ylabel("energy share", fontsize=8)
    ax.set_ylim(0, 0.55)
    ax.set_title("G. energy share by module type — all six checkpoints "
                 "(dotted = what an even-per-parameter update would give)", fontsize=9)
    ax.legend(fontsize=6, ncol=4, loc="upper left")
    ax.grid(alpha=0.25, axis="y"); ax.tick_params(labelsize=7)

    # H. 2k / selected / 30k magnitude comparison
    ax = fig.add_subplot(gs[3, 2])
    for seed in SEEDS:
        steps = [2000, SELECTED[seed], 30000]
        vals = [_ck(snap, seed, s)["total_delta_fro"] for s in steps]
        ax.plot(steps, vals, "-o", color=colors[seed], lw=1.6, ms=5, label=f"seed {seed}")
        ax.annotate("selected", (steps[1], vals[1]), textcoords="offset points",
                    xytext=(4, -12), fontsize=6.5, color=colors[seed])
    ax.set_xlabel("step (only 2k / selected / 30k were read)", fontsize=8)
    ax.set_ylabel("total ||dW||_F over 196 modules", fontsize=8)
    ax.set_title("H. total update magnitude: 2k -> selected -> 30k", fontsize=9)
    ax.grid(alpha=0.25); ax.legend(fontsize=7); ax.tick_params(labelsize=7)

    # I. concentration statistics across checkpoints
    ax = fig.add_subplot(gs[3, 3])
    keys = [("top1", "top-1"), ("top10", "top-10"), ("top10pct", "top-10%"),
            ("top25pct", "top-25%"), ("gini", "Gini")]
    for i, a in enumerate(snap["adapters"]):
        c = _ck(snap, a["seed"], a["step"])
        ax.plot(range(len(keys)), [c[k] for k, _ in keys], styles[a["role"]] + "o",
                color=colors[a["seed"]], lw=1.3, ms=3.5, label=f"s{a['seed']} @{a['step']}")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([lab for _, lab in keys], fontsize=7)
    ax.set_ylabel("share of raw energy / Gini", fontsize=8)
    ax.set_title("I. concentration statistics", fontsize=9)
    ax.grid(alpha=0.25); ax.legend(fontsize=5.6); ax.tick_params(labelsize=7)

    fig.suptitle("SFT LoRA parameter atlas — where the full-run SFT update lives in weight space",
                 y=0.982, fontsize=14)
    fig.text(0.5, 0.958, FOOTER, ha="center", va="top", fontsize=8.5, color="#444")
    fig.text(0.5, 0.940,
             "Screening for a later causal ablation. Magnitude in weight space is NOT causal "
             "importance:\nno activations, no ablation, no behavioral evaluation in this phase.",
             ha="center", va="top", fontsize=8, color="#8a1f1f", linespacing=1.5)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def render_spectrum(snap, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.4))
    colors = {1: "#1f5fd0", 2: "#d02b2b"}
    styles = {"first_saved": ":", "selected": "-", "endpoint": "--"}

    rmax = snap["lora"]["rank"]

    def _sv_matrix(rows):
        """[n_modules, rank] spectra, zero-padded if a module's dense shape is
        smaller than the LoRA rank (never the case for Qwen-7B, but keeps the
        renderer total)."""
        m = np.zeros((len(rows), rmax))
        for i, r in enumerate(rows):
            sv = np.asarray(r["singular_values"], dtype=float)
            m[i, :sv.size] = sv[:rmax]
        return m

    # A: mean normalized singular spectrum
    ax = axes[0, 0]
    for a in snap["adapters"]:
        rows = _rows(snap, a["seed"], a["step"])
        m = _sv_matrix(rows)
        sv = m / m[:, :1]
        ax.plot(np.arange(1, rmax + 1), sv.mean(0), styles[a["role"]], color=colors[a["seed"]],
                lw=1.5, label=f"s{a['seed']} @{a['step']}")
    ax.set_xlabel("singular index j"); ax.set_ylabel("mean sigma_j / sigma_1")
    ax.set_title("A. normalized singular spectrum of dW\n(mean over 196 modules)", fontsize=9)
    ax.grid(alpha=0.25); ax.legend(fontsize=6.5)

    # B: cumulative singular energy
    ax = axes[0, 1]
    for a in snap["adapters"]:
        rows = _rows(snap, a["seed"], a["step"])
        e = _sv_matrix(rows) ** 2
        cum = np.cumsum(e, axis=1) / e.sum(axis=1, keepdims=True)
        ax.plot(np.arange(1, rmax + 1), cum.mean(0), styles[a["role"]],
                color=colors[a["seed"]], lw=1.5)
        ax.fill_between(np.arange(1, rmax + 1), np.percentile(cum, 5, axis=0),
                        np.percentile(cum, 95, axis=0), color=colors[a["seed"]], alpha=0.06)
    for lev in (0.90, 0.95, 0.99):
        ax.axhline(lev, color="#888", lw=0.7, ls=":")
    ax.set_xlabel("rank k"); ax.set_ylabel("energy retained by top-k")
    ax.set_title("B. within-module energy retention\n(mean, 5-95% band)", fontsize=9)
    ax.grid(alpha=0.25)

    # C: effective rank distribution
    ax = axes[0, 2]
    data = [[r["eff_rank_entropy"] for r in _rows(snap, a["seed"], a["step"])]
            for a in snap["adapters"]]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                    tick_labels=[f"s{a['seed']}\n@{a['step']}" for a in snap["adapters"]])
    for patch, a in zip(bp["boxes"], snap["adapters"]):
        patch.set_facecolor(colors[a["seed"]]); patch.set_alpha(0.35)
    ax.axhline(rmax, color="#8a1f1f", ls=":", lw=1.1)
    ax.text(0.5, rmax * 1.01, f"rank-{rmax} cap (by construction)", fontsize=6.5, color="#8a1f1f")
    ax.set_ylabel("entropy effective rank")
    ax.set_title("C. effective rank per module", fontsize=9)
    ax.grid(alpha=0.25, axis="y"); ax.tick_params(labelsize=7)

    # D: eff rank by layer, selected checkpoints
    ax = axes[1, 0]
    for seed in SEEDS:
        g = _grid(snap, seed, SELECTED[seed], "eff_rank_entropy")
        ax.plot(range(N_LAYERS), np.nanmean(g, axis=1), "-o", ms=3,
                color=colors[seed], lw=1.4, label=f"seed {seed} @{SELECTED[seed]}")
    ax.set_xlabel("layer"); ax.set_ylabel("mean effective rank")
    ax.set_title("D. effective rank by depth (selected checkpoints)", fontsize=9)
    ax.grid(alpha=0.25); ax.legend(fontsize=7)

    # E: eff rank by module type
    ax = axes[1, 1]
    w = 0.13
    for i, a in enumerate(snap["adapters"]):
        rows = _rows(snap, a["seed"], a["step"])
        vals = [np.mean([r["eff_rank_entropy"] for r in rows if r["module"] == m])
                for m in MODULES]
        ax.bar(np.arange(len(MODULES)) + (i - 2.5) * w, vals, w, color=colors[a["seed"]],
               alpha=0.45 + 0.25 * ["first_saved", "selected", "endpoint"].index(a["role"]),
               label=f"s{a['seed']} @{a['step']}")
    ax.set_xticks(range(len(MODULES)))
    ax.set_xticklabels([SHORT[m] for m in MODULES])
    ax.set_ylabel("mean effective rank")
    ax.set_title("E. effective rank by module type", fontsize=9)
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.25, axis="y")

    # F: rank needed for 90/95/99% energy
    ax = axes[1, 2]
    x = np.arange(len(snap["adapters"]))
    for lev, mark, col in ((("rank90"), "o", "#1f8a4c"), ("rank95", "s", "#e0a13a"),
                           ("rank99", "^", "#8a1f1f")):
        med = [np.median([r[lev] for r in _rows(snap, a["seed"], a["step"])])
               for a in snap["adapters"]]
        p90 = [np.percentile([r[lev] for r in _rows(snap, a["seed"], a["step"])], 90)
               for a in snap["adapters"]]
        ax.plot(x, med, "-" + mark, color=col, lw=1.4, ms=5, label=f"{lev} median")
        ax.plot(x, p90, ":" + mark, color=col, lw=1.0, ms=3.5, alpha=0.7, label=f"{lev} p90")
    ax.set_xticks(x)
    ax.set_xticklabels([f"s{a['seed']}\n@{a['step']}" for a in snap["adapters"]], fontsize=7)
    ax.set_ylabel("rank needed"); ax.set_ylim(0, rmax + 1)
    ax.set_title("F. rank retaining 90/95/99% of module energy", fontsize=9)
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.25, axis="y")

    fig.suptitle("SFT LoRA singular-energy structure — effective rank of the update",
                 y=0.985, fontsize=13)
    fig.text(0.5, 0.945, FOOTER, ha="center", va="top", fontsize=8, color="#444")
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def render_cross_seed(snap, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(17.0, 9.4))

    def cos_vals(prefix):
        g = np.full((N_LAYERS, len(MODULES)), np.nan)
        vals = []
        for c in snap["comparisons"]:
            if c["label"].startswith(prefix):
                g[c["layer"], MODULES.index(c["module"])] = c["cosine"]
                vals.append(c["cosine"])
        return g, np.array(vals)

    # A/B: cross-seed cosine heatmaps. The colour scale is EXPANDED to the data
    # range so structure is visible; the title states how far below 1 that is.
    for j, (prefix, lab) in enumerate([
            ("cross-seed selected", f"A. SELECTED checkpoints "
                                    f"(seed 1 @{SELECTED[1]} vs seed 2 @{SELECTED[2]})"),
            ("cross-seed endpoint", "B. ENDPOINTS (seed 1 @30000 vs seed 2 @30000)")]):
        ax = axes[0, j]
        g, v = cos_vals(prefix)
        m = float(np.nanmax(np.abs(g)))
        im = _heat(ax, g, f"{lab}\ncross-seed cosine — scale EXPANDED to ±{m:.3f}, "
                          f"not ±1", cmap="coolwarm")
        im.set_clim(-m, m)
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.015).ax.tick_params(labelsize=6)

    # C: the honest full-range view — where cross-seed sits against within-seed
    ax = axes[0, 2]
    bins = np.linspace(-1, 1, 81)
    for prefix, col, lab in [
            ("cross-seed selected", "#8a1f1f", "cross-seed (selected)"),
            ("cross-seed endpoint", "#d0722b", "cross-seed (30k)"),
            ("seed1 2k->selected", "#1f5fd0", "within-seed 1 (2k->selected)"),
            ("seed2 2k->selected", "#1f8a4c", "within-seed 2 (2k->selected)")]:
        _, v = cos_vals(prefix)
        ax.hist(v, bins=bins, histtype="step", lw=1.6, color=col, label=lab)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("composite-update cosine (full range)", fontsize=8)
    ax.set_ylabel("modules", fontsize=8)
    ax.set_title("C. the same cosines on the FULL [-1, 1] axis\n"
                 "cross-seed updates are near-orthogonal; within-seed are not", fontsize=9)
    ax.legend(fontsize=6.2, loc="upper left"); ax.grid(alpha=0.25)

    # D: aggregate cosines, unweighted vs energy-weighted, on the full scale
    ax = axes[1, 0]
    aggs = snap["comparison_aggregates"]
    x = np.arange(len(aggs))
    ax.bar(x - 0.2, [a["mean_cosine_unweighted"] for a in aggs], 0.38,
           label="unweighted mean", color="#9aa7bd")
    ax.bar(x + 0.2, [a["mean_cosine_energy_weighted"] for a in aggs], 0.38,
           label="energy-weighted mean", color="#1f5fd0")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylim(-0.05, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([a["label"] for a in aggs], rotation=32, ha="right", fontsize=6.2)
    ax.set_ylabel("mean composite cosine", fontsize=8)
    ax.set_title("D. mean module cosine — unweighted vs energy-weighted\n"
                 "(w = ||dW_a|| ||dW_b||; a high cosine on a tiny update cannot dominate)",
                 fontsize=9)
    ax.legend(fontsize=6.5); ax.grid(alpha=0.25, axis="y")

    # E: per-module energy share, seed 1 vs seed 2 — magnitude agreement
    ax = axes[1, 1]
    keys, e1, e2 = _selected_pair(snap, "energy_share")
    fam = np.array([FAMILY[k[1]] for k in keys])
    for f, col in (("attention", "#1f5fd0"), ("mlp", "#e0a13a")):
        s = fam == f
        ax.scatter(e1[s], e2[s], s=16, alpha=0.75, color=col, label=f)
    lim = max(e1.max(), e2.max()) * 1.08
    ax.plot([0, lim], [0, lim], "k:", lw=0.9)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel(f"seed 1 @{SELECTED[1]} energy share", fontsize=8)
    ax.set_ylabel(f"seed 2 @{SELECTED[2]} energy share", fontsize=8)
    ax.set_title("E. per-module energy share, seed 1 vs seed 2 (selected)\n"
                 f"Spearman rho = {spearman(e1, e2):.3f} — the seeds agree on WHERE, "
                 f"panels A-D show they disagree on DIRECTION", fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.25)

    # F: within-seed drift heatmap for contrast
    ax = axes[1, 2]
    g, _ = cos_vals(f"seed1 selected({SELECTED[1]})->30k")
    im = _heat(ax, g, "F. WITHIN-seed 1 cosine, selected -> 30k (contrast)\n"
                      "same colour convention, full ±1 scale", cmap="coolwarm")
    im.set_clim(-1, 1)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.015).ax.tick_params(labelsize=6)

    fig.suptitle("SFT LoRA cross-seed similarity — do the same modules move in both seeds?",
                 y=0.985, fontsize=13)
    fig.text(0.5, 0.947, FOOTER, ha="center", va="top", fontsize=8, color="#444")
    fig.tight_layout(rect=(0, 0, 1, 0.928))
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def load_snapshot() -> dict:
    if not SNAPSHOT.exists():
        raise SystemExit(f"Snapshot missing: {SNAPSHOT} — run --refresh where the "
                         f"checkpoints and base model live.")
    return json.loads(SNAPSHOT.read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true",
                    help="Read the six untracked adapters + base weights and rewrite the snapshot.")
    ap.add_argument("--check", action="store_true",
                    help="Verify deterministic rendering from the tracked snapshot.")
    args = ap.parse_args()

    if args.refresh:
        snap, manifest = refresh()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(json.dumps(snap, indent=1, sort_keys=False) + "\n")
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Wrote {SNAPSHOT.relative_to(ROOT)} + {MANIFEST.relative_to(ROOT)}")

    snap = load_snapshot()
    stats_s = render_stats_csv(snap)
    comp_s = render_comp_csv(snap)
    md_s = render_summary_md(snap)

    if args.check:
        bad = []
        for p, s in ((STATS_CSV, stats_s), (COMP_CSV, comp_s), (SUMMARY_MD, md_s)):
            if not p.exists():
                bad.append(f"{p.name} missing")
            elif p.read_text() != s:
                bad.append(f"{p.name} differs from snapshot rendering")
        for p, fn in ((ATLAS_PNG, render_atlas), (SPECTRUM_PNG, render_spectrum),
                      (CROSS_PNG, render_cross_seed)):
            fn(snap, p)
            if not p.exists() or p.stat().st_size == 0:
                bad.append(f"{p.name} did not render")
        if bad:
            raise SystemExit("CHECK FAILED: " + "; ".join(bad))
        print("CHECK PASSED: snapshot present; CSV/Markdown byte-identical; all figures render.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATS_CSV.write_text(stats_s)
    COMP_CSV.write_text(comp_s)
    SUMMARY_MD.write_text(md_s)
    render_atlas(snap, ATLAS_PNG)
    render_spectrum(snap, SPECTRUM_PNG)
    render_cross_seed(snap, CROSS_PNG)
    print(f"Wrote {STATS_CSV.name}, {COMP_CSV.name}, {SUMMARY_MD.name}, "
          f"{ATLAS_PNG.name}, {SPECTRUM_PNG.name}, {CROSS_PNG.name} in "
          f"{OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
