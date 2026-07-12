#!/usr/bin/env python3
"""
Paired X/Y GRPO plumbing for Reward Design v2 (core: R_pair + R_neutral).

Two facts about TRL's GRPOTrainer (verified against the installed source) make
this simple and robust:

  1. Each reward function is called ONCE per generation batch with the full
     list of completions plus the dataset columns (perspective, case_id, ...).
     So R_pair can pair the X- and Y-endowed completions of a case *inside* the
     reward function, keyed by case_id — no surgery on generation internals, and
     robust to batch ordering/shuffling.

  2. TRL supports multiple reward functions, logs each as rewards/<name>/mean,
     and passes a `log_metric` callback. So R_pair and R_neutral are reported
     separately for free, and we log the extra v2 diagnostics through log_metric.

To guarantee both perspectives of a case land in the same reward batch (so
R_pair actually fires), PairedRepeatSampler keeps each (X,Y) pair together and
never splits a pair across a batch or a process.

This module is import-light on purpose (no torch/trl at top level) so the pure
logic and the sampler are unit-testable on a login node. The GRPOTrainer
subclass lives behind build_paired_grpo_trainer_cls() with a lazy trl import.
"""

from __future__ import annotations

import random
from typing import Callable, Optional

from reward_functions import parse_response
from reward_functions_v2 import (
    r_pair, r_neutral, r_gated, W_PAIR_DEFAULT, W_NEUTRAL_DEFAULT,
)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset → pair index list
# ─────────────────────────────────────────────────────────────────────────────
def build_pairs_from_columns(case_ids, perspectives) -> list[tuple[int, int]]:
    """
    Map the flat dataset (rows of alternating X/Y perspectives sharing a case_id)
    to a list of (x_row_index, y_row_index). Cases missing a perspective are
    dropped (should not happen with prompt_builder output).
    """
    xmap: dict[int, int] = {}
    ymap: dict[int, int] = {}
    for idx, (cid, persp) in enumerate(zip(case_ids, perspectives)):
        (xmap if persp == "X" else ymap)[int(cid)] = idx
    return [(xmap[c], ymap[c]) for c in xmap if c in ymap]


# ─────────────────────────────────────────────────────────────────────────────
# Reward functions (TRL-compatible; grouped/paired by case_id)
# ─────────────────────────────────────────────────────────────────────────────
def make_r_pair_reward() -> Callable:
    """
    Per-completion R_pair. Groups the batch by case_id, splits each case into its
    X and Y completions (in batch order), pairs the g-th X with the g-th Y, and
    assigns the shared R_pair to both. Unpaired leftovers (a case with only one
    perspective present in this batch) get 0.0 — no R_pair signal, not a penalty.
    """
    def _reward(completions, perspective, case_id, log_metric=None, **kwargs):
        parsed = [parse_response(c) for c in completions]
        groups: dict[int, dict[str, list[int]]] = {}
        for idx, (cid, persp) in enumerate(zip(case_id, perspective)):
            if persp in ("X", "Y"):
                groups.setdefault(int(cid), {"X": [], "Y": []})[persp].append(idx)

        rewards = [0.0] * len(completions)
        consistent = keep_both = trade_both = paired = unpaired = 0
        for g in groups.values():
            xs, ys = g["X"], g["Y"]
            n = min(len(xs), len(ys))
            unpaired += (len(xs) - n) + (len(ys) - n)
            for k in range(n):
                rx, ry = parsed[xs[k]], parsed[ys[k]]
                rp = r_pair(rx, ry)            # pure logic from reward_functions_v2
                rewards[xs[k]] = rewards[ys[k]] = rp
                paired += 1
                if rp > 0:
                    consistent += 1
                elif rx == "No" and ry == "No":
                    keep_both += 1
                elif rx == "Yes" and ry == "Yes":
                    trade_both += 1

        if log_metric is not None and parsed:
            ny = sum(p == "Yes" for p in parsed)
            nn = sum(p == "No" for p in parsed)
            log_metric("v2/yes_rate", ny / len(parsed))
            log_metric("v2/no_rate", nn / len(parsed))
            log_metric("v2/unparseable_rate", (len(parsed) - ny - nn) / len(parsed))
            if paired:
                log_metric("v2/pair_consistent_rate", consistent / paired)
                log_metric("v2/pair_keep_both_rate", keep_both / paired)
                log_metric("v2/pair_trade_both_rate", trade_both / paired)
            log_metric("v2/pairs_scored", float(paired))
            log_metric("v2/unpaired_completions", float(unpaired))
        return rewards

    _reward.__name__ = "r_pair"
    return _reward


def make_r_gated_reward(anchor: dict, mode: str = "shaped") -> Callable:
    """
    Production v2-core reward: ONE gated paired reward (replaces the additive
    r_pair + r_neutral). Groups the batch by case_id, pairs the g-th X-endowed
    completion with the g-th Y-endowed completion, computes r_gated(rx, ry, pref,
    mode) (see reward_functions_v2.r_gated), and assigns that single scalar to
    BOTH members of the pair.

    UNPAIRED COMPLETIONS ARE FATAL. The PairedRepeatSampler guarantees every
    generation batch holds only whole X/Y pairs, so a completion without its
    partner means the sampler/geometry is broken and every downstream reward is
    silently wrong — we raise instead of scoring it 0.0.

    `anchor` is the elicit_neutral_anchor.py output: {case_id: {"pref": "X"/"Y"/None}}
    or a flat {case_id: "X"/"Y"/None}. `mode` is "shaped" or "hard" (see r_gated).
    """
    amap: dict[int, Optional[str]] = {}
    for k, v in anchor.items():
        amap[int(k)] = v["pref"] if isinstance(v, dict) else v

    def _reward(completions, perspective, case_id, log_metric=None, **kwargs):
        parsed = [parse_response(c) for c in completions]
        groups: dict[int, dict[str, list[int]]] = {}
        for idx, (cid, persp) in enumerate(zip(case_id, perspective)):
            if persp in ("X", "Y"):
                groups.setdefault(int(cid), {"X": [], "Y": []})[persp].append(idx)

        # FATAL: any completion whose case/perspective has no aligned partner in
        # this batch means the paired sampler failed — abort loudly.
        bad = {cid: (len(g["X"]), len(g["Y"])) for cid, g in groups.items()
               if len(g["X"]) != len(g["Y"])}
        n_bad_persp = sum(1 for p in perspective if p not in ("X", "Y"))
        if bad or n_bad_persp:
            raise RuntimeError(
                "r_gated: UNPAIRED completions in a generation batch — the paired "
                f"sampler is broken. Cases with |X|!=|Y|: {dict(list(bad.items())[:8])}"
                f"{' ...' if len(bad) > 8 else ''}; completions with bad perspective: "
                f"{n_bad_persp}. Refusing to train on misaligned rewards."
            )

        rewards = [0.0] * len(completions)
        paired = unpaired = 0
        consistent = keep_both = trade_both = 0
        correct = wrong_pref = 0
        neg1 = zero = pos1 = 0
        anchored_pairs = unanchored_pairs = 0
        for cid, g in groups.items():
            pref = amap.get(int(cid))
            xs, ys = g["X"], g["Y"]
            n = min(len(xs), len(ys))          # == len(xs) == len(ys) after the guard
            for k in range(n):
                rx, ry = parsed[xs[k]], parsed[ys[k]]
                rg = r_gated(rx, ry, pref, mode=mode)
                rewards[xs[k]] = rewards[ys[k]] = rg
                paired += 1
                anchored_pairs += pref is not None
                unanchored_pairs += pref is None
                # diagnostics
                if rx in ("Yes", "No") and ry in ("Yes", "No"):
                    if rx != ry:
                        consistent += 1
                        if pref is not None:
                            if ("X" if rx == "No" else "Y") == pref:
                                correct += 1
                            else:
                                wrong_pref += 1
                    elif rx == "No":
                        keep_both += 1
                    else:
                        trade_both += 1
                if rg < 0:
                    neg1 += 1
                elif rg == 0.0:
                    zero += 1
                else:
                    pos1 += 1

        if log_metric is not None and parsed:
            ny = sum(p == "Yes" for p in parsed)
            nn = sum(p == "No" for p in parsed)
            log_metric("v2/yes_rate", ny / len(parsed))
            log_metric("v2/no_rate", nn / len(parsed))
            log_metric("v2/unparseable_rate", (len(parsed) - ny - nn) / len(parsed))
            log_metric("v2/pairs_scored", float(paired))
            log_metric("v2/unpaired_completions", float(unpaired))
            log_metric("v2/anchor_coverage",
                       anchored_pairs / paired if paired else 0.0)
            log_metric("v2/unanchored_pairs", float(unanchored_pairs))
            if paired:
                log_metric("v2/pair_consistent_rate", consistent / paired)
                log_metric("v2/pair_keep_both_rate", keep_both / paired)
                log_metric("v2/pair_trade_both_rate", trade_both / paired)
                # gated-table distribution — should match the reward ladder
                log_metric("v2/gated_neg1_rate", neg1 / paired)
                log_metric("v2/gated_zero_rate", zero / paired)
                log_metric("v2/gated_pos1_rate", pos1 / paired)
            if consistent:
                log_metric("v2/consistent_correct_rate", correct / consistent)
        return rewards

    _reward.__name__ = "r_gated"
    return _reward


def make_r_neutral_reward(anchor: dict) -> Callable:
    """
    Per-completion R_neutral from the frozen ownership-free anchor.
    `anchor` is the elicit_neutral_anchor.py output: {case_id: {"pref": "X"/"Y"/None}}
    or a flat {case_id: "X"/"Y"/None}. Ambiguous (no anchor) → 0.0 (no signal).
    """
    amap: dict[int, Optional[str]] = {}
    for k, v in anchor.items():
        amap[int(k)] = v["pref"] if isinstance(v, dict) else v

    def _reward(completions, perspective, case_id, log_metric=None, **kwargs):
        out = []
        have = correct = 0
        for c, persp, cid in zip(completions, perspective, case_id):
            v = r_neutral(parse_response(c), persp, amap.get(int(cid)))  # pure logic
            if v is None:
                out.append(0.0)
            else:
                out.append(v)
                have += 1
                correct += v > 0
        if log_metric is not None and out:
            log_metric("v2/anchor_coverage", have / len(out))
            if have:
                log_metric("v2/neutral_correct_rate", correct / have)
        return out

    _reward.__name__ = "r_neutral"
    return _reward


# ─────────────────────────────────────────────────────────────────────────────
# Pair-preserving sampler (pure Python; mirrors trl.RepeatSampler at pair grain)
# ─────────────────────────────────────────────────────────────────────────────
class PairedRepeatSampler:
    """
    Yields dataset row indices so that, within every generation batch, each
    case's X and Y prompts appear together, each repeated `mini_repeat_count`
    (= num_generations) times. Mirrors trl.RepeatSampler but the atomic unit is
    an (X, Y) pair rather than a single prompt.

    - shuffle: shuffles at the PAIR level (X and Y stay adjacent).
    - pairs_per_batch: pairs per generation batch, per process
      (= (generation_batch_size // num_generations) // 2).
    - num_processes / process_index: shard whole pairs across processes so a
      pair is never split across GPUs (R_pair must be computable locally).
    - repeat_count: reuse each generation batch this many times (num_iterations
      * steps_per_generation), matching RepeatSampler.
    - drop_last: drop a trailing partial global batch so batches stay aligned.
    """

    def __init__(self, pairs, mini_repeat_count, pairs_per_batch, repeat_count=1,
                 shuffle=True, seed=0, num_processes=1, process_index=0, drop_last=True):
        if pairs_per_batch < 1:
            raise ValueError("pairs_per_batch must be >= 1")
        self.pairs = list(pairs)
        self.G = mini_repeat_count
        self.ppb = pairs_per_batch
        self.repeat_count = repeat_count
        self.shuffle = shuffle
        self.seed = seed
        self.num_processes = num_processes
        self.process_index = process_index
        self.drop_last = drop_last
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def _global_batch_pairs(self) -> int:
        return self.ppb * self.num_processes

    def __iter__(self):
        order = list(range(len(self.pairs)))
        if self.shuffle:
            # same seed across processes → identical global order → consistent shards
            random.Random(self.seed + self._epoch).shuffle(order)
        gbp = self._global_batch_pairs()
        for start in range(0, len(order), gbp):
            chunk = order[start:start + gbp]
            if len(chunk) < gbp:
                if self.drop_last:
                    break
                # pad-free tail: only this process's available slice
            mine = chunk[self.process_index * self.ppb:(self.process_index + 1) * self.ppb]
            for _ in range(self.repeat_count):
                for pi in mine:
                    x_idx, y_idx = self.pairs[pi]
                    for _ in range(self.G):
                        yield x_idx
                    for _ in range(self.G):
                        yield y_idx

    def __len__(self) -> int:
        gbp = self._global_batch_pairs()
        n_batches = len(self.pairs) // gbp if self.drop_last else -(-len(self.pairs) // gbp)
        return n_batches * self.ppb * 2 * self.G * self.repeat_count


# ─────────────────────────────────────────────────────────────────────────────
# GRPOTrainer subclass (lazy trl import)
# ─────────────────────────────────────────────────────────────────────────────
def build_paired_grpo_trainer_cls():
    """Return a GRPOTrainer subclass that uses PairedRepeatSampler. Lazy import
    keeps this module usable (and testable) without loading trl."""
    from trl import GRPOTrainer

    class PairedGRPOTrainer(GRPOTrainer):
        def __init__(self, *args, pairs=None, **kwargs):
            if pairs is None:
                raise ValueError("PairedGRPOTrainer requires pairs=[(x_idx, y_idx), ...]")
            self._pairs = list(pairs)
            super().__init__(*args, **kwargs)

        def _get_train_sampler(self, dataset=None):
            prompts_per_batch = self.args.generation_batch_size // self.num_generations
            if prompts_per_batch % 2 != 0:
                raise ValueError(
                    "generation_batch_size // num_generations must be EVEN so each "
                    f"generation batch holds whole X/Y pairs (got {prompts_per_batch})."
                )
            return PairedRepeatSampler(
                pairs=self._pairs,
                mini_repeat_count=self.num_generations,
                pairs_per_batch=prompts_per_batch // 2,
                repeat_count=self.num_iterations * self.args.steps_per_generation,
                shuffle=self.shuffle_dataset,
                seed=self.args.seed,
                num_processes=self.accelerator.num_processes,
                process_index=self.accelerator.process_index,
            )

    return PairedGRPOTrainer
