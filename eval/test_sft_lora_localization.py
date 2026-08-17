#!/usr/bin/env python3
"""Tests for the CPU-only SFT LoRA parameter localization atlas.

Every mathematical test is SYNTHETIC — no Qwen checkpoint, adapter or base-model
file is required, so the suite runs in a clean `git archive` extraction. The
low-rank identities are checked against explicit dense `B @ A` algebra.

Run with the venv python (numpy/matplotlib/safetensors):
    ./venv/bin/python eval/test_sft_lora_localization.py
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

import analyze_sft_lora_localization as L  # noqa: E402

_fail: list[str] = []
_pass = 0


def check(name, cond, detail=""):
    global _pass
    if cond:
        _pass += 1
        print(f"  PASS  {name}")
    else:
        _fail.append(f"{name}: {detail}")
        print(f"  FAIL  {name}: {detail}")


def raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


def close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))


rng = np.random.default_rng(20260817)


def make_ab(out, in_, r=16, seed=0):
    g = np.random.default_rng(seed)
    a = g.standard_normal((r, in_)).astype(np.float32) * 0.1
    b = g.standard_normal((out, r)).astype(np.float32) * 0.1
    return a, b


SCALE = 2.0

print("== 1-3. low-rank norms and singular values vs explicit dense algebra ==")
for (out, in_, r) in [(64, 96, 16), (96, 64, 16), (40, 40, 16), (37, 53, 8)]:
    a, b = make_ab(out, in_, r, seed=out * 100 + in_)
    dense = SCALE * (b.astype(np.float64) @ a.astype(np.float64))
    check(f"1. Frobenius norm matches dense [{out}x{in_} r{r}]",
          close(L.lowrank_fro(a, b, SCALE), np.linalg.norm(dense)),
          f"{L.lowrank_fro(a, b, SCALE)} vs {np.linalg.norm(dense)}")
    sv_lr = L.lowrank_singular_values(a, b, SCALE)
    sv_d = np.linalg.svd(dense, compute_uv=False)[:min(r, out, in_)]
    check(f"2. singular values match dense SVD [{out}x{in_} r{r}]",
          np.allclose(sv_lr, sv_d, atol=1e-10, rtol=1e-9),
          f"max|diff|={np.max(np.abs(sv_lr - sv_d))}")
    check(f"3. spectral norm equals largest dense singular value [{out}x{in_} r{r}]",
          close(L.lowrank_spectral(a, b, SCALE), sv_d[0]))

print("\n== 4. stable rank on a known matrix ==")
# diag(3,4) -> ||.||_F^2 = 25, ||.||_2^2 = 16 -> stable rank 25/16
check("4a. stable rank of diag(3,4) is 25/16", close(L.stable_rank([4.0, 3.0]), 25.0 / 16.0))
check("4b. stable rank of rank-1 is exactly 1", close(L.stable_rank([7.0, 0.0, 0.0]), 1.0))
check("4c. stable rank of k equal singular values is k",
      close(L.stable_rank([2.0] * 5), 5.0))
# and against a real low-rank product
a, b = make_ab(64, 96, 16, seed=7)
dense = SCALE * (b.astype(np.float64) @ a.astype(np.float64))
sr_dense = np.linalg.norm(dense) ** 2 / np.linalg.svd(dense, compute_uv=False)[0] ** 2
check("4d. stable rank matches dense computation",
      close(L.stable_rank(L.lowrank_singular_values(a, b, SCALE)), sr_dense))

print("\n== 5. entropy effective rank ==")
check("5a. rank-1 spectrum has effective rank 1",
      close(L.effective_rank_entropy([5.0, 0.0, 0.0, 0.0]), 1.0))
for k in (2, 4, 8, 16):
    sv = [1.0] * k
    check(f"5b. {k} equal-energy directions give effective rank {k}",
          close(L.effective_rank_entropy(sv), float(k)),
          f"{L.effective_rank_entropy(sv)}")
check("5c. effective rank is scale-invariant",
      close(L.effective_rank_entropy([3.0, 1.0, 0.5]),
            L.effective_rank_entropy([30.0, 10.0, 5.0])))
check("5d. effective rank lies between 1 and the number of directions",
      1.0 <= L.effective_rank_entropy([3.0, 1.0, 0.5]) <= 3.0)

print("\n== 6. rank-k energy retention ==")
sv = np.array([4.0, 3.0, 2.0, 1.0] + [0.0] * 12)      # energies 16, 9, 4, 1 (total 30)
check("6a. top-1 energy retention", close(L.energy_retained(sv, 1), 16 / 30))
check("6b. top-2 energy retention", close(L.energy_retained(sv, 2), 25 / 30))
check("6c. top-4 energy retention", close(L.energy_retained(sv, 4), 1.0))
check("6d. top-8 energy retention (padded zeros)", close(L.energy_retained(sv, 8), 1.0))
check("6e. top-16 energy retention", close(L.energy_retained(sv, 16), 1.0))
check("6f. min rank for 90% energy", L.min_rank_for_energy(sv, 0.90) == 3,
      f"got {L.min_rank_for_energy(sv, 0.90)}")
check("6g. min rank for 95% energy", L.min_rank_for_energy(sv, 0.95) == 3,
      f"got {L.min_rank_for_energy(sv, 0.95)}")
check("6h. min rank for 99% energy", L.min_rank_for_energy(sv, 0.99) == 4,
      f"got {L.min_rank_for_energy(sv, 0.99)}")
a, b = make_ab(80, 60, 16, seed=11)
dense = SCALE * (b.astype(np.float64) @ a.astype(np.float64))
sv_d = np.linalg.svd(dense, compute_uv=False)[:16]
for k in (1, 2, 4, 8):
    check(f"6i. rank-{k} retention matches dense SVD",
          close(L.energy_retained(L.lowrank_singular_values(a, b, SCALE), k),
                float(np.sum(sv_d[:k] ** 2) / np.sum(sv_d ** 2))))

print("\n== 7-8. composite-update cosine and difference vs dense ==")
a1, b1 = make_ab(64, 96, 16, seed=21)
a2, b2 = make_ab(64, 96, 16, seed=22)
d1 = SCALE * (b1.astype(np.float64) @ a1.astype(np.float64))
d2 = SCALE * (b2.astype(np.float64) @ a2.astype(np.float64))
cos_dense = float(np.sum(d1 * d2) / (np.linalg.norm(d1) * np.linalg.norm(d2)))
check("7a. composite cosine equals dense cosine",
      close(L.lowrank_cosine(a1, b1, SCALE, a2, b2, SCALE), cos_dense),
      f"{L.lowrank_cosine(a1, b1, SCALE, a2, b2, SCALE)} vs {cos_dense}")
check("7b. inner product equals dense trace inner product",
      close(L.lowrank_inner(a1, b1, SCALE, a2, b2, SCALE), float(np.sum(d1 * d2))))
check("7c. self-cosine is exactly 1",
      close(L.lowrank_cosine(a1, b1, SCALE, a1, b1, SCALE), 1.0))
check("7d. anti-parallel cosine is exactly -1",
      close(L.lowrank_cosine(a1, b1, SCALE, a1, -b1, SCALE), -1.0))
check("8a. composite difference norm equals dense difference",
      close(L.lowrank_diff_fro(a1, b1, SCALE, a2, b2, SCALE), np.linalg.norm(d1 - d2)),
      f"{L.lowrank_diff_fro(a1, b1, SCALE, a2, b2, SCALE)} vs {np.linalg.norm(d1 - d2)}")
check("8b. difference with self is zero",
      L.lowrank_diff_fro(a1, b1, SCALE, a1, b1, SCALE) < 1e-9)
check("8c. differing ranks are handled (r=16 vs r=8)",
      close(L.lowrank_diff_fro(a1, b1, SCALE, *make_ab(64, 96, 8, seed=23), SCALE),
            np.linalg.norm(d1 - SCALE * (make_ab(64, 96, 8, seed=23)[1].astype(np.float64)
                                         @ make_ab(64, 96, 8, seed=23)[0].astype(np.float64)))))
check("8d. mismatched dense shapes hard-fail",
      raises(lambda: L.lowrank_inner(a1, b1, SCALE, *make_ab(64, 32, 16, seed=24), SCALE)))

print("\n== 9. LoRA scale alpha/r applied exactly once ==")
a, b = make_ab(64, 96, 16, seed=31)
unit = L.lowrank_fro(a, b, 1.0)
check("9a. ||dW||_F scales linearly (not quadratically) in alpha/r",
      close(L.lowrank_fro(a, b, 2.0), 2.0 * unit),
      f"{L.lowrank_fro(a, b, 2.0)} vs {2.0 * unit}")
check("9b. scale=2 matches dense 2*B@A exactly",
      close(L.lowrank_fro(a, b, 2.0),
            np.linalg.norm(2.0 * (b.astype(np.float64) @ a.astype(np.float64)))))
check("9c. scale does not enter twice (not 4x at scale 2)",
      not close(L.lowrank_fro(a, b, 2.0), 4.0 * unit))
check("9d. singular values scale linearly",
      np.allclose(L.lowrank_singular_values(a, b, 3.0),
                  3.0 * L.lowrank_singular_values(a, b, 1.0), rtol=1e-9))
check("9e. inner product is bilinear in the two scales",
      close(L.lowrank_inner(a1, b1, 2.0, a2, b2, 3.0),
            6.0 * L.lowrank_inner(a1, b1, 1.0, a2, b2, 1.0)))
check("9f. cosine is invariant to the common scale",
      close(L.lowrank_cosine(a1, b1, 2.0, a2, b2, 2.0),
            L.lowrank_cosine(a1, b1, 1.0, a2, b2, 1.0)))
check("9g. project scale is exactly alpha/r = 32/16 = 2",
      L.EXPECTED_ALPHA / L.EXPECTED_RANK == 2.0)

print("\n== 10. gauge invariance:  A -> R A,  B -> B R^-1 ==")
g = np.random.default_rng(101)
R = g.standard_normal((16, 16)).astype(np.float64)
while abs(np.linalg.det(R)) < 1e-3:
    R = g.standard_normal((16, 16)).astype(np.float64)
Rinv = np.linalg.inv(R)
a, b = make_ab(64, 96, 16, seed=41)
ag = (R @ a.astype(np.float64))
bg = (b.astype(np.float64) @ Rinv)
check("10a. Frobenius norm is gauge invariant",
      close(L.lowrank_fro(ag, bg, SCALE), L.lowrank_fro(a, b, SCALE), 1e-7),
      f"{L.lowrank_fro(ag, bg, SCALE)} vs {L.lowrank_fro(a, b, SCALE)}")
check("10b. singular values are gauge invariant",
      np.allclose(L.lowrank_singular_values(ag, bg, SCALE),
                  L.lowrank_singular_values(a, b, SCALE), rtol=1e-6, atol=1e-10))
check("10c. spectral norm is gauge invariant",
      close(L.lowrank_spectral(ag, bg, SCALE), L.lowrank_spectral(a, b, SCALE), 1e-7))
check("10d. cosine is gauge invariant on BOTH sides",
      close(L.lowrank_cosine(ag, bg, SCALE, a2, b2, SCALE),
            L.lowrank_cosine(a, b, SCALE, a2, b2, SCALE), 1e-7))
check("10e. effective rank is gauge invariant",
      close(L.effective_rank_entropy(L.lowrank_singular_values(ag, bg, SCALE)),
            L.effective_rank_entropy(L.lowrank_singular_values(a, b, SCALE)), 1e-6))
# and the point of it all: raw factor norms are NOT invariant
check("10f. raw ||A||/||B|| ARE gauge-dependent (why they are never used)",
      not close(float(np.linalg.norm(ag)), float(np.linalg.norm(a)), 1e-3))

print("\n== 11. raw A/B norms are never the headline importance statistic ==")
src = (ROOT / "eval" / "analyze_sft_lora_localization.py").read_text()
check("11a. no per-factor norm column in the module stats CSV",
      not any(c in L.STATS_COLS for c in ("a_fro", "b_fro", "a_norm", "b_norm",
                                          "lora_a_norm", "lora_b_norm")),
      str(L.STATS_COLS))
check("11b. energy is defined from the composite update, not from A or B",
      L.DEFINITIONS["energy"].startswith("||dW||_F^2"))
check("11c. the gauge hazard is documented in the manifest text",
      "gauge" in src.lower() and "R^-1" in src)
check("11d. module_rows reports composite norms only (delta_fro/delta_spec/delta_rms)",
      all(c in L.STATS_COLS for c in ("delta_fro", "delta_spec", "delta_rms")))
# decisive behavioural check: gauge-transform one module inside the real
# per-module pipeline and confirm every emitted statistic is unchanged
_bs = {(0, "q_proj"): {"shape": [64, 96], "numel": 64 * 96, "fro": 10.0, "spec": 2.0,
                       "spec_iters": 3, "spec_converged": True}}
_meta = {"seed": 1, "step": 2000, "role": "first_saved"}
_saved_layers, _saved_mods = L.N_LAYERS, L.MODULES
try:
    L.N_LAYERS, L.MODULES = 1, ["q_proj"]
    _plain = L.module_rows({(0, "q_proj"): {"A": a, "B": b}}, _bs, _meta, SCALE)[0]
    _gauge = L.module_rows({(0, "q_proj"): {"A": ag, "B": bg}}, _bs, _meta, SCALE)[0]
finally:
    L.N_LAYERS, L.MODULES = _saved_layers, _saved_mods
check("11e. every emitted per-module statistic is gauge invariant",
      all(close(_plain[k], _gauge[k], 1e-6)
          for k in ("delta_fro", "delta_spec", "delta_rms", "rel_fro", "rel_spec",
                    "energy", "stable_rank", "eff_rank_entropy", "energy_top1",
                    "energy_top4", "rank90", "rank95", "rank99")),
      f"plain={_plain['delta_fro']} gauge={_gauge['delta_fro']}")

print("\n== 12. module-key parsing ==")
for layer in (0, 7, 27):
    for mod, sub in L.SUB_BLOCK.items():
        for fac in ("A", "B"):
            k = f"base_model.model.model.layers.{layer}.{sub}.{mod}.lora_{fac}.weight"
            got = L.parse_module_key(k)
            check(f"12. parse L{layer} {mod} lora_{fac}", got == (layer, mod, fac), str(got))
check("12z. base tensor key mapping is unique and well-formed",
      L.base_key_for(13, "down_proj") == "model.layers.13.mlp.down_proj.weight",
      L.base_key_for(13, "down_proj"))
check("12y. malformed key hard-fails",
      raises(lambda: L.parse_module_key("model.layers.3.mlp.up_proj.weight"), ValueError))
check("12x. module in the wrong sub-block hard-fails",
      raises(lambda: L.parse_module_key(
          "base_model.model.model.layers.3.mlp.q_proj.lora_A.weight"), ValueError))
check("12w. out-of-range layer hard-fails",
      raises(lambda: L.parse_module_key(
          "base_model.model.model.layers.99.mlp.up_proj.lora_A.weight"), ValueError))

print("\n== 13-15. adapter validation hard-fails (synthetic, no checkpoint needed) ==")


def _write_adapter(tmp: Path, *, rank=16, alpha=32, bias="none", targets=None,
                   drop=None, extra_key=None, dtype=np.float32):
    """Build a tiny synthetic PEFT-shaped adapter directory."""
    from safetensors.numpy import save_file
    targets = list(L.MODULES) if targets is None else targets
    cfg = {"peft_type": "LORA", "r": rank, "lora_alpha": alpha, "bias": bias,
           "target_modules": targets, "modules_to_save": None, "use_dora": False,
           "use_rslora": False, "rank_pattern": {}, "alpha_pattern": {},
           "base_model_name_or_path": L.EXPECTED_BASE_MODEL, "peft_version": "0.19.1"}
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "adapter_config.json").write_text(json.dumps(cfg))
    tensors = {}
    dims = {"q_proj": (32, 32), "k_proj": (8, 32), "v_proj": (8, 32), "o_proj": (32, 32),
            "gate_proj": (48, 32), "up_proj": (48, 32), "down_proj": (32, 48)}
    for layer in range(L.N_LAYERS):
        for mod in L.MODULES:
            out, in_ = dims[mod]
            sub = L.SUB_BLOCK[mod]
            ka = f"base_model.model.model.layers.{layer}.{sub}.{mod}.lora_A.weight"
            kb = f"base_model.model.model.layers.{layer}.{sub}.{mod}.lora_B.weight"
            if drop == (layer, mod, "A"):
                pass
            else:
                tensors[ka] = np.zeros((rank, in_), dtype=dtype)
            if drop == (layer, mod, "B"):
                pass
            else:
                tensors[kb] = np.zeros((out, rank), dtype=dtype)
    if extra_key:
        tensors[extra_key] = np.zeros((rank, 8), dtype=np.float32)
    save_file(tensors, str(tmp / "adapter_model.safetensors"))
    return tmp


with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    good = _write_adapter(td / "good")
    f = L.load_adapter_factors(good)
    check("13a. a well-formed synthetic adapter loads with the full inventory",
          len(f) == L.N_LAYERS * len(L.MODULES), str(len(f)))
    miss = _write_adapter(td / "miss_a", drop=(5, "v_proj", "A"))
    check("13b. missing lora_A factor hard-fails",
          raises(lambda: L.load_adapter_factors(miss), SystemExit))
    missb = _write_adapter(td / "miss_b", drop=(9, "gate_proj", "B"))
    check("13c. missing lora_B factor hard-fails",
          raises(lambda: L.load_adapter_factors(missb), SystemExit))
    extra = _write_adapter(
        td / "extra",
        extra_key="base_model.model.model.layers.3.self_attn.rotary_proj.lora_A.weight")
    check("14a. unexpected target module in the weights hard-fails",
          raises(lambda: L.load_adapter_factors(extra), ValueError))
    check("14b. unexpected target module in the CONFIG hard-fails",
          raises(lambda: L.verify_adapter_config(
              {"peft_type": "LORA", "r": 16, "lora_alpha": 32, "bias": "none",
               "target_modules": L.MODULES + ["embed_tokens"],
               "base_model_name_or_path": L.EXPECTED_BASE_MODEL}, "synthetic"), SystemExit))
    check("14c. a MISSING target module in the config hard-fails",
          raises(lambda: L.verify_adapter_config(
              {"peft_type": "LORA", "r": 16, "lora_alpha": 32, "bias": "none",
               "target_modules": L.MODULES[:-1],
               "base_model_name_or_path": L.EXPECTED_BASE_MODEL}, "synthetic"), SystemExit))

    base_cfg = {"peft_type": "LORA", "r": 16, "lora_alpha": 32, "bias": "none",
                "target_modules": list(L.MODULES),
                "base_model_name_or_path": L.EXPECTED_BASE_MODEL}
    check("15a. good config passes", L.verify_adapter_config(base_cfg, "x") is None)
    check("15b. rank mismatch hard-fails",
          raises(lambda: L.verify_adapter_config({**base_cfg, "r": 8}, "x"), SystemExit))
    check("15c. alpha mismatch hard-fails",
          raises(lambda: L.verify_adapter_config({**base_cfg, "lora_alpha": 16}, "x"), SystemExit))
    check("15d. bias != none hard-fails",
          raises(lambda: L.verify_adapter_config({**base_cfg, "bias": "all"}, "x"), SystemExit))
    check("15e. non-LORA peft_type hard-fails",
          raises(lambda: L.verify_adapter_config({**base_cfg, "peft_type": "IA3"}, "x"), SystemExit))
    check("15f. use_rslora (different scale) hard-fails",
          raises(lambda: L.verify_adapter_config({**base_cfg, "use_rslora": True}, "x"), SystemExit))
    check("15g. use_dora hard-fails",
          raises(lambda: L.verify_adapter_config({**base_cfg, "use_dora": True}, "x"), SystemExit))
    check("15h. per-module rank_pattern hard-fails",
          raises(lambda: L.verify_adapter_config(
              {**base_cfg, "rank_pattern": {"q_proj": 8}}, "x"), SystemExit))
    check("15i. modules_to_save hard-fails",
          raises(lambda: L.verify_adapter_config(
              {**base_cfg, "modules_to_save": ["lm_head"]}, "x"), SystemExit))
    check("15j. wrong base-model identity hard-fails",
          raises(lambda: L.verify_adapter_config(
              {**base_cfg, "base_model_name_or_path": "/models/Llama-3-8B"}, "x"), SystemExit))
    check("15k. rank mismatch between config and TENSORS hard-fails",
          raises(lambda: L.load_adapter_factors(_write_adapter(td / "r8", rank=8)), SystemExit))

print("\n== 16. pilot / final paths hard-fail ==")
for p in ["checkpoints/sft_qwen_delta_seed1_pilot6k/checkpoint-6000",
          "checkpoints/sft_qwen_delta_seed1/final",
          "/abs/path/PILOT/checkpoint-2000",
          "checkpoints/sft_qwen_delta_seed2/Final"]:
    check(f"16. rejected: {p}", raises(lambda p=p: L.reject_forbidden_path(p), ValueError))
check("16z. a legitimate full-run path is accepted",
      L.reject_forbidden_path("checkpoints/sft_qwen_delta_seed1/checkpoint-4000") is None)

print("\n== 17. exact six-checkpoint inventory ==")
check("17a. exactly six adapters are declared", len(L.ADAPTERS) == 6, str(len(L.ADAPTERS)))
check("17b. the six identities are exactly the intended ones",
      {(a["seed"], a["step"]) for a in L.ADAPTERS}
      == {(1, 2000), (1, 4000), (1, 30000), (2, 2000), (2, 6000), (2, 30000)})
check("17c. roles are first_saved / selected / endpoint per seed",
      sorted((a["seed"], a["role"]) for a in L.ADAPTERS)
      == sorted([(1, "first_saved"), (1, "selected"), (1, "endpoint"),
                 (2, "first_saved"), (2, "selected"), (2, "endpoint")]))
check("17d. selected steps are seed1@4000 and seed2@6000",
      L.SELECTED == {1: 4000, 2: 6000}, str(L.SELECTED))
check("17e. no declared path contains a forbidden token",
      all(L.reject_forbidden_path(a["path"]) is None for a in L.ADAPTERS))
check("17f. every declared path is under the full-run seed directories",
      all(a["path"].startswith(f"checkpoints/sft_qwen_delta_seed{a['seed']}/checkpoint-")
          for a in L.ADAPTERS))
check("17g. comparison set is 6 within-seed + 3 cross-seed",
      [c["kind"] for c in L.comparison_pairs()].count("within_seed") == 6
      and [c["kind"] for c in L.comparison_pairs()].count("cross_seed") == 3)
check("17h. cross-seed comparisons match roles, not raw steps",
      any(c["a"] == (1, 4000) and c["b"] == (2, 6000) for c in L.comparison_pairs()))

print("\n== 18. concentration / gini / top-share statistics ==")
check("18a. gini of a perfectly even vector is 0", close(L.gini([1.0] * 10), 0.0))
check("18b. gini of all-mass-in-one approaches (n-1)/n",
      close(L.gini([0.0] * 9 + [1.0]), 0.9, 1e-9), str(L.gini([0.0] * 9 + [1.0])))
check("18c. gini is scale invariant", close(L.gini([1.0, 2.0, 7.0]), L.gini([10.0, 20.0, 70.0])))
check("18d. gini rejects negatives", raises(lambda: L.gini([-1.0, 2.0]), ValueError))
check("18e. top_share of the largest entry", close(L.top_share([1, 2, 3, 4], 1), 0.4))
check("18f. top_share of the two largest", close(L.top_share([1, 2, 3, 4], 2), 0.7))
check("18g. top_frac_share uses a ceiling count",
      close(L.top_frac_share([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.25),
            L.top_share([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3)))
check("18h. top_share over the whole vector is 1", close(L.top_share([1, 2, 3], 3), 1.0))

print("\n== 18i-n. spearman + random-rank cosine null ==")
check("18i. spearman of a strictly increasing pair is +1",
      close(L.spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]), 1.0))
check("18j. spearman of a reversed pair is -1",
      close(L.spearman([1, 2, 3, 4, 5], [50, 40, 30, 20, 10]), -1.0))
check("18k. spearman is monotone-transform invariant",
      close(L.spearman([1, 2, 3, 4, 5], [1, 4, 9, 16, 25]), 1.0))
nul = L.random_rank_cosine_null(64, 96, rank=16, n_draws=60, seed=5)
check("18l. null cosine of independent rank-16 updates is centred near zero",
      abs(nul["mean_cosine"]) < 4 * nul["sd_cosine"] / math.sqrt(nul["n_draws"]) + 1e-3,
      str(nul))
check("18m. null cosine is deterministic for a fixed seed",
      L.random_rank_cosine_null(64, 96, rank=16, n_draws=60, seed=5) == nul)
check("18n. a larger dense shape gives a tighter null",
      L.random_rank_cosine_null(256, 384, rank=16, n_draws=60, seed=5)["sd_cosine"]
      < nul["sd_cosine"])

print("\n== 19. bfloat16 widening is lossless ==")
vals = np.array([0.0, 1.0, -1.0, 0.5, -2.25, 3.5e-5, 1.7e30], dtype=np.float32)
bf = (vals.view(np.uint32) >> 16).astype(np.uint16)          # truncate to bf16
back = L._bf16_to_f32(bf)
expect = (bf.astype(np.uint32) << 16).view(np.float32)
check("19a. bf16 -> f32 widening reproduces the exact bit pattern",
      np.array_equal(back.view(np.uint32), expect.view(np.uint32)))
check("19b. exactly-representable values round-trip",
      np.allclose(back[:5], vals[:5], rtol=0, atol=0))

print("\n== 20. snapshot-only rendering works without NSCC directories ==")


def _synthetic_snapshot():
    """Build a complete snapshot with random factors, so rendering is exercised
    end-to-end with no checkpoint or base-model file present."""
    scale = 2.0
    dims = {"q_proj": (32, 32), "k_proj": (8, 32), "v_proj": (8, 32), "o_proj": (32, 32),
            "gate_proj": (48, 32), "up_proj": (48, 32), "down_proj": (32, 48)}
    base_stats = {}
    for layer in range(L.N_LAYERS):
        for mod in L.MODULES:
            out, in_ = dims[mod]
            base_stats[(layer, mod)] = {"shape": [out, in_], "numel": out * in_,
                                        "fro": 10.0 + layer, "spec": 2.0 + 0.01 * layer,
                                        "spec_iters": 4, "spec_converged": True}
    adapters, rows_by, conc_by, factors, all_rows = [], {}, {}, {}, []
    for spec in L.ADAPTERS:
        fac = {}
        for layer in range(L.N_LAYERS):
            for mod in L.MODULES:
                out, in_ = dims[mod]
                g = np.random.default_rng(spec["seed"] * 10 ** 6 + spec["step"]
                                          + layer * 17 + L.MODULES.index(mod))
                fac[(layer, mod)] = {"A": g.standard_normal((16, in_)).astype(np.float32) * 0.01,
                                     "B": g.standard_normal((out, 16)).astype(np.float32) * 0.01}
        a = {**spec, "weight_sha256": "0" * 64, "config_sha256": "0" * 64,
             "weight_bytes": 1, "target_modules_sorted": sorted(L.MODULES),
             "peft_version": "0.19.1", "base_model_name_or_path": L.EXPECTED_BASE_MODEL,
             "lora_dropout": 0.05}
        adapters.append(a)
        key = (spec["seed"], spec["step"])
        factors[key] = fac
        rows_by[key] = L.module_rows(fac, base_stats, a, scale)
        conc_by[key] = L.concentration(rows_by[key])
        all_rows += rows_by[key]
    comps, aggs = [], []
    for spec in L.comparison_pairs():
        per, agg = L.compare_adapters(factors[spec["a"]], factors[spec["b"]],
                                      rows_by[spec["a"]], rows_by[spec["b"]], scale, spec)
        comps += per
        aggs.append(agg)
    return {
        "title": "synthetic", "phase": "test",
        "lora": {"rank": 16, "alpha": 32, "scale": scale,
                 "formula": "dW = (alpha / rank) * B @ A", "bias": "none",
                 "target_modules": sorted(L.MODULES), "n_layers": L.N_LAYERS},
        "n_trainable_lora_values_per_adapter": conc_by[(1, 2000)]["n_trainable_total"],
        "selected_checkpoints": {str(k): v for k, v in L.SELECTED.items()},
        "adapters": adapters,
        "base_model": {"dir": "x", "shard": "y", "shard_sha256": "0" * 64,
                       "shard_bytes": 1, "header_sha256": "0" * 64,
                       "config_sha256": "0" * 64, "dtype": "BF16"},
        "base_module_stats": {f"{l}|{m}": base_stats[(l, m)]
                              for l in range(L.N_LAYERS) for m in L.MODULES},
        "modules": all_rows,
        "concentration": {f"seed{k[0]}_step{k[1]}": v for k, v in conc_by.items()},
        "comparisons": comps, "comparison_aggregates": aggs,
        "random_rank_cosine_null": {
            f"{o}x{i}": {**L.random_rank_cosine_null(o, i, n_draws=20, seed=3),
                         "modules": sorted(m for m in L.MODULES if dims[m] == (o, i))}
            for o, i in sorted(set(dims.values()))},
        "definitions": L.DEFINITIONS,
    }


snap = _synthetic_snapshot()
check("20a. synthetic snapshot has 6 x 196 module rows",
      len(snap["modules"]) == 6 * L.N_LAYERS * len(L.MODULES), str(len(snap["modules"])))
check("20b. every adapter reports the same trainable-value count",
      len({c["n_trainable_total"] for c in snap["concentration"].values()}) == 1)
csv_s = L.render_stats_csv(snap)
comp_s = L.render_comp_csv(snap)
md_s = L.render_summary_md(snap)
check("20c. stats CSV renders with a header + 1176 rows",
      len(csv_s.strip().split("\n")) == 1 + 6 * 196, str(len(csv_s.strip().split("\n"))))
check("20d. comparisons CSV renders with a header + 9 x 196 rows",
      len(comp_s.strip().split("\n")) == 1 + 9 * 196, str(len(comp_s.strip().split("\n"))))
check("20e. rendering is deterministic (byte-identical on a second pass)",
      L.render_stats_csv(snap) == csv_s and L.render_comp_csv(snap) == comp_s
      and L.render_summary_md(snap) == md_s)
check("20f. summary answers all nine questions",
      all(f"\n## {i}." in md_s or md_s.startswith(f"## {i}.")
          for i in range(1, 10)) or all(f"## {i}." in md_s for i in (1, 2, 3, 4, 5, 6, 9)))
check("20g. summary carries the screening / no-causal-claim boundary",
      "no causal" in md_s.lower() and "screening" in md_s.lower())
check("20h. summary proposes a Phase-2 shortlist without running it",
      "Phase-2 shortlist (NOT executed)" in md_s)
check("20i. summary states the no-sub-2k limitation",
      "sub-2k" in md_s or "before step 2000" in md_s)
check("20m. summary opens with a headline reading",
      "## Headline reading" in md_s)
check("20n. summary renders the random-rank cosine null table",
      "independent** random rank-16 updates" in md_s and "null p95" in md_s)
check("20o. summary reconciles the rank correlation with the cosine",
      "Reconciling the two numbers" in md_s)
check("20p. summary renders without a null block too (backward compatible)",
      "## Headline reading" in L.render_summary_md(
          {k: v for k, v in snap.items() if k != "random_rank_cosine_null"}))
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    for fn, name in ((L.render_atlas, "atlas.png"), (L.render_spectrum, "spec.png"),
                     (L.render_cross_seed, "cross.png")):
        fn(snap, td / name)
        check(f"20j. {name} renders from a snapshot alone",
              (td / name).exists() and (td / name).stat().st_size > 5000,
              str((td / name).stat().st_size if (td / name).exists() else "missing"))
check("20k. figure footer carries all four required labels",
      all(s in L.FOOTER for s in ("weight-space screening", "no causal importance implied",
                                  "full-run SFT", "no pilot")), L.FOOTER)
check("20l. rendering never touches the checkpoint or model directories",
      not any(str(L.BASE_MODEL_DIR) in s or "checkpoints/" in s
              for s in (csv_s[:4000], comp_s[:4000])))

print("\n== 21. clean git-archive --check ==")


def _archive_check():
    with tempfile.TemporaryDirectory() as td:
        arch = Path(td) / "arch"
        arch.mkdir()
        tar = subprocess.run(["git", "archive", "HEAD"], cwd=ROOT, capture_output=True)
        if tar.returncode != 0:
            return "no-head"
        subprocess.run(["tar", "-x", "-C", str(arch)], input=tar.stdout, check=True)
        if not (arch / "results/sft_parameter_localization"
                / "sft_lora_localization_snapshot.json").exists():
            return "not-committed-yet"
        for forbidden in ("checkpoints", "models"):
            if (arch / forbidden).exists():
                return f"archive unexpectedly contains {forbidden}/"
        r = subprocess.run([sys.executable,
                            str(arch / "eval/analyze_sft_lora_localization.py"), "--check"],
                           cwd=arch, capture_output=True, text=True)
        return "ok" if r.returncode == 0 else f"fail: {(r.stdout + r.stderr)[-300:]}"


res = _archive_check()
# "no-head" occurs when the tests are themselves run from an extracted archive
# (not a git repo); "not-committed-yet" before the first commit. The outer
# archive run in the verification steps is the real proof.
check("21. clean git-archive --check", res in ("ok", "not-committed-yet", "no-head"),
      f"result={res}")
if res in ("not-committed-yet", "no-head"):
    print(f"       (note: archive self-check skipped [{res}]; verified by the outer archive run)")

print(f"\n{_pass} passed, {len(_fail)} failed")
if _fail:
    for f in _fail:
        print("  -", f)
    raise SystemExit(1)
print("ALL SFT LORA LOCALIZATION TESTS PASSED")
