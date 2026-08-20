#!/usr/bin/env python3
"""Training driver for the SFT batch / learning-rate sensitivity experiment.

Governed by SFT_BATCH_LR_SENSITIVITY_PROTOCOL.md. EXPLORATORY / POST-HOC: it
neither revises nor reselects the frozen matched-SFT baseline.

The original train/sft_train.py is imported, never modified, so the frozen
baseline stays exactly reproducible. What this driver adds:

  * one deterministic permutation per seed; each cell consumes a PREFIX of it,
    materialised as an explicitly ordered dataset read by a SEQUENTIAL sampler,
    so ordering is a verified property rather than a hoped-for side effect of
    HF's shuffling;
  * an order hash checked against the frozen values before a GPU is touched;
  * exposure-aligned checkpoints, identical across effective batches;
  * a manifest describing EFFECTIVE batch and PROMPT EXPOSURE (the old
    "1 step = 1 unique prompt at batch 1 / accum 1" wording does not apply here
    and is not reused);
  * full provenance, including library versions and the PBS job id.

    python train/sft_sensitivity_train.py --cell-list
    python train/sft_sensitivity_train.py --cell <name> --validate   # CPU only
    python train/sft_sensitivity_train.py --cell <name>              # GPU
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "train"))

import sft_sensitivity_plan as P                                    # noqa: E402
from sft_train import (CompletionOnlyCollator, assert_clean_training_data,  # noqa: E402
                       build_sft_example, build_sft_records, dataset_stats,
                       git_commit, resolve, sha256_of)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("sft_sens")

DATA_FILE = "data/remaining_goods.json"
FROZEN_ORDER_HASHES = P.RESULT_ROOT / "order_hashes.json"


def all_planned_cells() -> dict[str, P.Cell]:
    """Every cell the protocol can produce, for name lookup and collision checks.

    Phase B and the full runs are enumerated over ALL large batches / LRs because
    the selection has not been made yet; the selected subset is a filter applied
    later, not a different naming scheme.
    """
    cells = list(P.phase_a_cells())
    for b in P.LARGE_BATCHES:
        cells += P.phase_b_cells(b, include_probe=True)
        for lr in P.PHASE_B_LRS:
            cells += P.full_cells(b, lr)
    # Extended-horizon exploratory run (Amendment 4). Its horizon tag h32000
    # keeps it in its own directory, so it can never be confused with the
    # h6016 probe that shares its batch, LR and seed.
    cells += [P.Cell("full", 32, 1.0e-4, 1, P.HORIZON_32K)]
    uniq: dict[str, P.Cell] = {}
    for c in cells:
        uniq.setdefault(c.name, c)
    P.assert_unique_output_dirs(list(uniq.values()))
    return uniq


def library_versions() -> dict:
    out = {"python": platform.python_version()}
    for mod in ("torch", "transformers", "peft", "accelerate", "numpy", "datasets"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception as exc:
            out[mod] = f"unavailable: {exc}"
    try:
        import torch
        out["cuda"] = torch.version.cuda
        out["gpu"] = (torch.cuda.get_device_name(0) if torch.cuda.is_available()
                      else "no CUDA device visible")
    except Exception as exc:
        out["cuda"] = out["gpu"] = f"unavailable: {exc}"
    return out


def verify_config_matches_plan(config_path: str) -> None:
    """The YAML is a human-readable record; the plan module is the source of
    truth. Any divergence between them is a hard failure rather than a silent
    change of what actually ran."""
    import yaml
    cfg = yaml.safe_load(Path(config_path).read_text())
    for key in ("per_device_train_batch_size", "max_grad_norm", "warmup_ratio",
                "lr_scheduler_type", "logging_steps", "lora_r", "lora_alpha",
                "lora_dropout"):
        if key in cfg and key in P.FIXED and cfg[key] != P.FIXED[key]:
            raise SystemExit(
                f"REFUSED: config {config_path} sets {key}={cfg[key]!r} but the "
                f"frozen plan requires {P.FIXED[key]!r}. Reconcile before running.")


def guard_output_dir(cell: P.Cell, allow_existing: bool) -> None:
    """Never write into a frozen run, and never silently overwrite our own."""
    d = cell.output_dir
    s = str(d.resolve())
    for frozen in ("sft_qwen_delta_seed1", "sft_qwen_delta_seed2",
                   "sft_qwen_delta_seed1_pilot6k", "grpo_qwen_delta",
                   "grpo_qwen_delta_sign"):
        if Path(s).name == frozen or f"/{frozen}/" in s + "/":
            raise SystemExit(f"REFUSED: output dir {d} collides with frozen run {frozen}")
    if "sft_sensitivity" not in s:
        raise SystemExit(f"REFUSED: output dir {d} is outside checkpoints/sft_sensitivity")
    # A directory holding only manifests/logs (e.g. from a --validate dry run)
    # is not a run and may be written into. Actual trained artefacts may not.
    artefacts = [p for p in (d.glob("checkpoint-*") if d.exists() else [])] + \
                [p for p in (d.glob("exposure-*") if d.exists() else [])]
    if artefacts and not allow_existing:
        raise SystemExit(
            f"REFUSED: {d} already holds trained artefacts "
            f"({', '.join(p.name for p in artefacts[:4])}). Existing runs are "
            "never overwritten; pass --allow-existing only to resume a run you "
            "intend to continue unchanged.")


def build_ordered_records(cell: P.Cell) -> tuple[list[dict], dict]:
    """The exposure prefix of this seed's single deterministic ordering."""
    data_file = resolve(DATA_FILE)
    assert_clean_training_data(data_file)
    goods = resolve("everyday_goods_full.json")
    delta = resolve("data/deltas/delta_qwen_base.json")
    records = build_sft_records(data_file, goods, delta, resolve("data"))

    order = P.deterministic_order(records, cell.seed)
    P.assert_no_duplicate_exposure(records, order, cell.exposure)
    ordered = [records[i] for i in order[:cell.exposure]]
    if len(ordered) != cell.exposure:
        raise SystemExit(f"REFUSED: got {len(ordered):,} examples, need {cell.exposure:,}")

    oh = P.order_hash(records, order, cell.exposure)
    prov = {
        "algorithm": P.ORDER_ALGORITHM,
        "example_id_format": "{case_id}:{perspective}",
        "pool_size": len(records),
        "prompt_exposure": cell.exposure,
        "order_hash": oh,
        "order_hash_full_pool": P.order_hash(records, order),
        "first_3_example_ids": [P.example_id(r) for r in ordered[:3]],
        "duplicate_prompt_exposure": 0,
    }

    # Fail before a GPU is touched if the ordering is not the frozen one.
    if FROZEN_ORDER_HASHES.is_file():
        frozen = json.loads(FROZEN_ORDER_HASHES.read_text()).get(str(cell.seed), {})
        # An unmapped exposure would leave `expected` None and skip verification
        # silently, which is exactly the provenance hole this check exists to
        # close. Every horizon the driver can run must appear here.
        key = {P.PILOT_EXPOSURE: "order_hash_pilot_6016",
               P.FULL_EXPOSURE: "order_hash_full_30016",
               P.HORIZON_32K: "order_hash_h32000"}.get(cell.exposure)
        if key is None:
            raise SystemExit(
                f"REFUSED: no frozen order hash is registered for exposure "
                f"{cell.exposure:,}. Freeze it in {FROZEN_ORDER_HASHES.name} "
                "before running, so the ordering is pinned rather than merely "
                "recomputed.")
        expected = frozen.get(key)
        prov["frozen_hash_expected"] = expected
        if expected and expected != oh:
            raise SystemExit(
                f"REFUSED: order hash mismatch for seed {cell.seed} at exposure "
                f"{cell.exposure:,}\n  expected {expected}\n  computed {oh}\n"
                "The data ordering is not the one frozen in the protocol.")
        prov["frozen_hash_verified"] = bool(expected)
    else:
        prov["frozen_hash_verified"] = False
    return ordered, prov


def write_manifest(cell: P.Cell, args, stats: dict, order_prov: dict,
                   started: str, finished: str | None, runtime_s: float | None) -> dict:
    d = cell.describe()
    steps = cell.optimizer_steps
    man = {
        "experiment": "SFT batch/learning-rate sensitivity",
        "protocol": "SFT_BATCH_LR_SENSITIVITY_PROTOCOL.md",
        "scientific_status": ("EXPLORATORY / POST-HOC optimization ablation. Does "
                              "not modify or reselect the frozen matched-SFT "
                              "baseline; licenses no SFT-vs-GRPO claim."),
        # git is unavailable on the compute nodes, so the submitting side
        # captures the hash on the login node and passes it through the
        # environment. Falling back to a bare "unknown" here is what produced
        # SFTPROV-001 in the earlier SFT manifests; this run records the real
        # commit and says which side established it.
        "git_commit": os.environ.get("LZ_GIT_COMMIT") or git_commit(),
        "git_commit_source": ("LZ_GIT_COMMIT (captured on the login node at "
                              "submission, where git is available)"
                              if os.environ.get("LZ_GIT_COMMIT")
                              else "git rev-parse on this host"),
        "command": " ".join(sys.argv),
        "cell": d,
        "is_off_grid_probe": P.is_probe(cell),
        "probe_note": ("Off-grid bracketing probe under Amendment 3. May not "
                       "promote a setting or enter the frozen selection rule."
                       if P.is_probe(cell) else None),
        "exposure_accounting": {
            "prompt_exposure": cell.exposure,
            "optimizer_updates": steps,
            "effective_batch": cell.effective_batch,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": cell.effective_batch,
            "identity": ("prompt_exposure = optimizer_updates x "
                         "gradient_accumulation_steps x per_device_train_batch_size"),
            "verified": cell.exposure == steps * cell.effective_batch,
            "note": ("Effective batch is realised purely by accumulation, so the "
                     "per-prompt forward/backward cost is identical across cells "
                     "and only the optimizer-update boundary moves."),
        },
        "schedule": {
            "lr_scheduler_type": "cosine",
            "warmup_ratio": P.FIXED["warmup_ratio"],
            "horizon_optimizer_steps": steps,
            "horizon_prompt_exposure": cell.exposure,
            "learning_rate": cell.learning_rate,
            "peak_learning_rate": cell.learning_rate,
            "warmup_optimizer_steps": int(round(P.FIXED["warmup_ratio"] * steps)),
            "warmup_prompt_exposure": int(round(P.FIXED["warmup_ratio"] * steps)
                                          * cell.effective_batch),
            "max_grad_norm": P.FIXED["max_grad_norm"],
            "trained_from": "base weights (not resumed from any checkpoint)",
            "not_interchangeable": ("Weights at equal prompt exposure but "
                                    "different cosine horizons (h6016 / h30016 / "
                                    "h32000) are NOT interchangeable: each run "
                                    "decays over its own horizon."),
        },
        "checkpoints": (
            {"schedule": "Amendment 4: dense early window then coarse intervals",
             "early_exposure_to_step": P.h32k_early_and_coarse(cell.effective_batch)[0],
             "coarse_exposure_to_step": P.h32k_early_and_coarse(cell.effective_batch)[1],
             "n_save_points": len(P.h32k_checkpoint_steps(cell.effective_batch)),
             "intermediate_kind": "adapter-only (resume_supported: false)",
             "endpoint_kind": ("full resumable checkpoint at max_steps, plus an "
                               "`exposure-32000` adapter for evaluation"),
             "endpoint_always_saved": True}
            if cell.exposure == P.HORIZON_32K else
            {"exposure_to_optimizer_step":
             P.checkpoint_steps(cell.exposure, cell.effective_batch),
             "save_steps": P.save_steps_for(cell.exposure, cell.effective_batch),
             "endpoint_always_saved": True}),
        "data_ordering": order_prov,
        "dataset": stats,
        "sources": {p: {"path": p, "sha256": sha256_of(resolve(p))} for p in
                    (DATA_FILE, "data/deltas/delta_qwen_base.json")},
        "config_sha256": sha256_of(Path(args.config)) if args.config else None,
        "plan_module_sha256": sha256_of(PROJECT_ROOT / "train" / "sft_sensitivity_plan.py"),
        "driver_sha256": sha256_of(Path(__file__).resolve()),
        "environment": library_versions(),
        "pbs_job_id": os.environ.get("PBS_JOBID", "not-under-pbs"),
        "started_utc": started, "finished_utc": finished, "runtime_seconds": runtime_s,
        "suites_not_read": ["data/method_comparison/", "data/frozen_unused_test_goods.json",
                            "data/ood_new_goods_50*", "semantic", "framing", "GSM8K"],
    }
    cell.output_dir.mkdir(parents=True, exist_ok=True)
    out = cell.output_dir / "sft_sensitivity_manifest.json"
    out.write_text(json.dumps(man, indent=2, default=str) + "\n")
    logger.info(f"Wrote manifest: {out.relative_to(PROJECT_ROOT)}")
    return man


def append_execution_manifest(cell: P.Cell, status: str, extra: str = "") -> None:
    """Append-only execution record."""
    f = P.RESULT_ROOT / "EXECUTION_MANIFEST.md"
    f.parent.mkdir(parents=True, exist_ok=True)
    if not f.exists():
        f.write_text("# SFT batch/LR sensitivity — execution manifest (append-only)\n\n"
                     "| utc | cell | pbs job | status | note |\n|---|---|---|---|---|\n")
    with open(f, "a") as fh:
        fh.write(f"| {datetime.now(timezone.utc).isoformat(timespec='seconds')} | "
                 f"{cell.name} | {os.environ.get('PBS_JOBID', '-')} | {status} | {extra} |\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", help="cell name, e.g. sft_sens_phA_eb32_lr1e-6_seed1_h6016")
    ap.add_argument("--cell-list", action="store_true", help="print every planned cell and exit")
    ap.add_argument("--validate", action="store_true",
                    help="CPU only: build ordering + manifest, touch no GPU")
    ap.add_argument("--config", default="train/configs/sft_sensitivity_base.yaml")
    ap.add_argument("--allow-existing", action="store_true")
    args = ap.parse_args()

    planned = all_planned_cells()
    if args.cell_list:
        for n, c in sorted(planned.items()):
            print(f"{n:<48} exposure={c.exposure:>6,}  steps={c.optimizer_steps:>6,}  "
                  f"eb={c.effective_batch:<3} lr={c.learning_rate:g} seed={c.seed}")
        print(f"\n{len(planned)} distinct cells planned "
              f"(Phase A + Phase B over all large batches + full runs).")
        return 0
    if not args.cell:
        ap.error("--cell is required (or use --cell-list)")
    if args.cell not in planned:
        raise SystemExit(f"Unknown cell {args.cell!r}. Use --cell-list.")
    cell = planned[args.cell]

    verify_config_matches_plan(args.config)
    guard_output_dir(cell, args.allow_existing)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if P.is_probe(cell):
        logger.info("PROBE CELL (Amendment 3): off-grid bracketing LR. May "
                    "bracket the optimum; may NOT promote a setting or change "
                    "the frozen selection.")
    logger.info(f"Cell {cell.name}: exposure={cell.exposure:,} prompts, "
                f"steps={cell.optimizer_steps:,}, effective batch={cell.effective_batch}, "
                f"lr={cell.learning_rate:g}, seed={cell.seed}")

    ordered, order_prov = build_ordered_records(cell)
    logger.info(f"Order hash {order_prov['order_hash'][:16]}… "
                f"(frozen-verified: {order_prov.get('frozen_hash_verified')})")
    stats = dataset_stats(ordered)

    if args.validate:
        write_manifest(cell, args, stats, order_prov, started, None, None)
        append_execution_manifest(cell, "VALIDATED (CPU, no training)")
        print(json.dumps({"validate_ok": True, "cell": cell.name,
                          "prompt_exposure": cell.exposure,
                          "optimizer_steps": cell.optimizer_steps,
                          "order_hash": order_prov["order_hash"],
                          "frozen_hash_verified": order_prov.get("frozen_hash_verified"),
                          **stats}, indent=2))
        return 0

    # ── GPU path ────────────────────────────────────────────────────────────
    import time
    import torch
    import datasets
    from torch.utils.data import SequentialSampler
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model

    model_path = resolve("models/Qwen2.5-7B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tok_ds = datasets.Dataset.from_list(ordered).map(
        lambda r: build_sft_example(tokenizer, r["prompt"], r["target"]),
        remove_columns=["prompt", "perspective", "delta", "case_id", "target"],
        desc="Tokenizing (completion-only)")

    model = AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype=torch.bfloat16)
    model = get_peft_model(model, LoraConfig(
        r=P.FIXED["lora_r"], lora_alpha=P.FIXED["lora_alpha"],
        lora_dropout=P.FIXED["lora_dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=str(cell.output_dir), seed=cell.seed, data_seed=cell.seed,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=cell.effective_batch,
        learning_rate=cell.learning_rate, lr_scheduler_type="cosine",
        warmup_ratio=P.FIXED["warmup_ratio"], max_grad_norm=P.FIXED["max_grad_norm"],
        max_steps=cell.optimizer_steps, bf16=True,
        logging_steps=P.FIXED["logging_steps"],
        # At the 32,000 horizon the adapter-only callback owns the intermediate
        # schedule, so HF is told to save exactly ONCE, at max_steps. That keeps
        # the endpoint a full resumable checkpoint (optimizer + scheduler + RNG,
        # and the trainer_state.json the diagnostics read) while preventing HF's
        # 2,048-stride saves from interleaving with — and being mistaken for —
        # the dense-then-coarse grid Amendment 4 defines.
        save_steps=(cell.optimizer_steps if cell.exposure == P.HORIZON_32K
                    else P.save_steps_for(cell.exposure, cell.effective_batch)),
        report_to="none", dataloader_num_workers=0, remove_unused_columns=False)

    class OrderedTrainer(Trainer):
        """Consume the frozen ordering exactly: no reshuffling, ever."""
        def _get_train_sampler(self, *a, **k):
            return SequentialSampler(self.train_dataset)

    trainer = OrderedTrainer(model=model, args=targs, train_dataset=tok_ds,
                             data_collator=CompletionOnlyCollator(tokenizer.pad_token_id))

    # ── Phase-B instrumentation (Amendment 2) ───────────────────────────────
    from sft_sensitivity_callbacks import (EarlyAdapterSnapshotCallback,
                                           UpdateNormCallback)
    update_cb = UpdateNormCallback(model)
    trainer.add_callback(update_cb)
    snap_cb = None
    if cell.phase == "B" or cell.exposure == P.HORIZON_32K:
        # Adapter-only snapshots. Preserved, not evaluated. Adapter-only because
        # these exist to be evaluated later, not resumed from; the resumable
        # state is the endpoint checkpoint.
        #
        # Phase B uses the 128..2,048 early window. The 32,000 horizon uses the
        # Amendment-4 schedule instead: a dense early window (128..1,280) then
        # coarse 4,096 intervals. The 32,000 endpoint is EXCLUDED here — it is
        # written twice already, as HF's full resumable checkpoint-1000 and as
        # the `exposure-32000` adapter saved after training — so including it
        # would produce a third, redundant copy.
        if cell.exposure == P.HORIZON_32K:
            sched = {e: s for e, s in
                     P.h32k_checkpoint_steps(cell.effective_batch).items()
                     if e != P.HORIZON_32K}
        else:
            sched = P.early_snapshot_steps(cell.effective_batch)
        step_to_exposure = {v: k for k, v in sched.items()}
        snap_cb = EarlyAdapterSnapshotCallback(
            model, cell.output_dir, step_to_exposure,
            {"cell": cell.name, "effective_batch": cell.effective_batch,
             "seed": cell.seed, "learning_rate": cell.learning_rate,
             "base_model": "models/Qwen2.5-7B-Instruct",
             "git_commit": os.environ.get("LZ_GIT_COMMIT") or git_commit()})
        trainer.add_callback(snap_cb)
        logger.info(f"Early adapter-only snapshots at prompt exposures "
                    f"{sorted(step_to_exposure.values())}")
    append_execution_manifest(cell, "STARTED",
                              f"exposure {cell.exposure:,}, {cell.optimizer_steps:,} updates")
    t0 = time.time()
    trainer.train()
    trainer.save_model(str(cell.output_dir / f"exposure-{cell.exposure}"))
    runtime = time.time() - t0
    finished = datetime.now(timezone.utc).isoformat(timespec="seconds")
    man = write_manifest(cell, args, stats, order_prov, started, finished, runtime)
    # Diagnostics recorded but explicitly barred from the frozen CE selection.
    extra = {
        "parameter_update_norm": update_cb.summary(),
        "parameter_update_norm_per_step": update_cb.records,
        "early_adapter_snapshots": (snap_cb.saved if snap_cb else []),
        "selection_note": ("Diagnostics only. The frozen selection rule uses "
                           "endpoint validation cross-entropy and nothing here "
                           "may change it."),
    }
    man.update(extra)
    (cell.output_dir / "sft_sensitivity_manifest.json").write_text(
        json.dumps(man, indent=2, default=str) + "\n")
    append_execution_manifest(cell, "COMPLETED", f"{runtime/3600:.2f} GPU-h")
    logger.info(f"Done: {cell.name} in {runtime/3600:.2f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
