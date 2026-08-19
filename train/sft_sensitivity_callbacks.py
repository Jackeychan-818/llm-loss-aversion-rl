#!/usr/bin/env python3
"""Phase-B instrumentation: parameter-update norms and adapter-only snapshots.

Both are Phase-B additions (Amendment 2). They are imported by
train/sft_sensitivity_train.py and are deliberately kept in their own module so
they can be unit-tested on CPU with a toy model, with no GPU and no 7B weights.

Why the update norm matters
---------------------------
Under universal clipping (100% of updates at every large batch) the PRE-clipping
gradient norm no longer tells you how far the parameters actually moved. And the
optimizer is `adamw_torch_fused`, whose update is `lr * m_hat/(sqrt(v_hat)+eps)`
— already scale-free, so a constant rescaling of the gradient largely cancels in
that ratio. Only the realised parameter movement settles what clipping is doing,
so it is measured directly rather than inferred.
"""
from __future__ import annotations

import json
from pathlib import Path

from transformers import TrainerCallback


def trainable_params(model):
    """Only trainable LoRA parameters — the frozen base must never be counted."""
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


class UpdateNormCallback(TrainerCallback):
    """Record ||Delta theta||_2 and ||Delta theta|| / ||theta|| per OPTIMIZER step.

        ||Delta theta_t||_2 = sqrt( sum_j || theta_{j,t+1} - theta_{j,t} ||_2^2 )

    over all trainable LoRA parameters j.

    Measured between `on_pre_optimizer_step` and `on_optimizer_step`, so
    gradient-accumulation microsteps (`on_substep_end`) can never be counted as
    updates: at effective batch 64 there are 64 microsteps per optimizer step,
    and counting them would inflate the record 64-fold and report near-zero
    movement for most entries.
    """

    def __init__(self, model):
        self.model = model
        self._before: dict[str, "object"] = {}
        self.records: list[dict] = []
        self.n_substeps_seen = 0

    def on_substep_end(self, args, state, control, **kw):
        # Counted for the test that proves microsteps are NOT recorded as updates.
        self.n_substeps_seen += 1

    def on_pre_optimizer_step(self, args, state, control, **kw):
        self._before = {n: p.detach().clone()
                        for n, p in trainable_params(self.model)}

    def on_optimizer_step(self, args, state, control, **kw):
        if not self._before:
            return
        import torch
        sq_delta = 0.0
        sq_theta = 0.0
        for n, p in trainable_params(self.model):
            if n not in self._before:
                continue
            d = (p.detach() - self._before[n]).float()
            sq_delta += float(torch.sum(d * d))
            sq_theta += float(torch.sum(p.detach().float() ** 2))
        norm = sq_delta ** 0.5
        theta = sq_theta ** 0.5
        self.records.append({
            "step": int(state.global_step),
            "update_norm_l2": norm,
            "param_norm_l2": theta,
            "relative_update_norm": (norm / theta) if theta > 0 else None,
        })
        self._before = {}

    def summary(self) -> dict:
        vals = [r["update_norm_l2"] for r in self.records]
        rel = [r["relative_update_norm"] for r in self.records
               if r["relative_update_norm"] is not None]
        if not vals:
            return {"available": False, "reason": "no optimizer steps recorded"}
        srt = sorted(vals)
        q = lambda f: srt[min(len(srt) - 1, max(0, int(round(f * (len(srt) - 1)))))]
        return {
            "available": True, "n_optimizer_updates_recorded": len(vals),
            "n_accumulation_substeps_seen": self.n_substeps_seen,
            "update_norm_l2": {"median": q(0.5), "p90": q(0.9), "p99": q(0.99),
                               "max": max(vals), "min": min(vals)},
            "relative_update_norm": ({"median": sorted(rel)[len(rel) // 2],
                                      "max": max(rel)} if rel else None),
            "n_zero_updates": sum(1 for v in vals if v == 0.0),
            "definition": ("L2 over all trainable LoRA parameter changes between "
                           "consecutive optimizer steps; accumulation microsteps "
                           "are not counted"),
        }


class EarlyAdapterSnapshotCallback(TrainerCallback):
    """Save ADAPTER-ONLY snapshots at the frozen early-exposure grid.

    Adapter-only, not a full checkpoint: these exist to be *evaluated* later, not
    resumed from, and a full checkpoint is ~463 MB against ~150 MB here, mostly
    optimizer state. Each snapshot records `resume_supported: false` so nothing
    downstream mistakes it for a resumable state.
    """

    def __init__(self, model, output_dir: Path, step_to_exposure: dict[int, int],
                 provenance: dict):
        self.model = model
        self.output_dir = Path(output_dir)
        self.step_to_exposure = dict(step_to_exposure)
        self.provenance = provenance
        self.saved: list[dict] = []

    def on_step_end(self, args, state, control, **kw):
        step = int(state.global_step)
        exposure = self.step_to_exposure.get(step)
        if exposure is None:
            return
        d = self.output_dir / f"early-exposure-{exposure}"
        d.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(d))          # adapter_config + adapter_model
        meta = {
            "kind": "adapter-only snapshot",
            "resume_supported": False,
            "reason": ("no optimizer/scheduler/RNG state is saved; this snapshot "
                       "is for evaluation only and cannot resume training"),
            "prompt_exposure": exposure,
            "optimizer_step": step,
            **self.provenance,
        }
        (d / "snapshot_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        self.saved.append({"exposure": exposure, "step": step, "path": str(d)})
