"""Matched ESL prototype-count (K) sensitivity for the twelve-game 3×3 protocol.

K=3 is the identity path through ``run_categorical_cell`` so flagship
(instance_id, seed) cells can be reproduced. K≠3 patches only ESL prototype
and belief initialization, plus LTE matching when K is rectangular.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np

from esl.metrics import pairwise_assignment_cost
from esl.res.benchmark import load_suite
from esl.res.runner import game_spec_to_task

K_TRUE = 3
K_GRID = (2, 3, 5)
SEEDS = tuple(range(10))
HORIZON = 1000
H_WINDOW = 100
NUM_ACTIONS = 3
NUM_OPPONENTS = 3


def _patch_k_model(k_model: int):
    import esl.res.categorical_cell as cc

    k_model = int(k_model)
    orig_ms = cc._method_state
    orig_lte = cc._lte

    def wrapped_ms(method, seed, **kwargs):
        state = orig_ms(method, seed, **kwargs)
        if method == "ESL" and k_model != K_TRUE:
            a = int(state.num_actions)
            state.logits = cc._esl_initial_logits(seed, k_model=k_model, num_actions=a)
            state.beliefs = {
                opp: np.full(k_model, 1.0 / float(k_model), dtype=np.float64)
                for opp in state.beliefs
            }
        return state

    def wrapped_lte(belief, *, true_type, learned_logits, true_probs):
        logits = np.asarray(learned_logits, dtype=np.float64)
        km = int(logits.shape[0])
        b = np.clip(np.asarray(belief, dtype=np.float64).ravel(), cc.EPS, 1.0)
        b = b / b.sum()
        if km == K_TRUE and int(b.size) == K_TRUE:
            return orig_lte(
                belief, true_type=true_type, learned_logits=learned_logits, true_probs=true_probs
            )
        cost = pairwise_assignment_cost(np.asarray(true_probs, dtype=np.float64), logits)
        matched_idx = int(np.argmin(cost[int(true_type)]))
        return float(-math.log(float(b[matched_idx])))

    cc._method_state = wrapped_ms  # type: ignore[assignment]
    cc._lte = wrapped_lte  # type: ignore[assignment]
    return orig_ms, orig_lte


def _unpatch(orig_ms, orig_lte) -> None:
    import esl.res.categorical_cell as cc

    cc._method_state = orig_ms
    cc._lte = orig_lte


@contextmanager
def esl_k_model(k_model: int) -> Iterator[None]:
    orig = _patch_k_model(int(k_model))
    try:
        yield
    finally:
        _unpatch(*orig)


def inspect_esl_init(*, seed: int, k_model: int, num_actions: int = NUM_ACTIONS) -> dict[str, Any]:
    import esl.res.categorical_cell as cc

    with esl_k_model(k_model):
        state = cc._method_state(
            "ESL",
            seed + 7919,
            num_actions=num_actions,
            num_opponents=NUM_OPPONENTS,
            p_stay=0.95,
        )
        logits = np.asarray(state.logits)
        beliefs = {int(k): np.asarray(v) for k, v in state.beliefs.items()}
        return {
            "k_model": int(k_model),
            "logits_shape": list(logits.shape),
            "logits_finite": bool(np.isfinite(logits).all()),
            "n_beliefs": len(beliefs),
            "belief_lens": {str(k): int(v.size) for k, v in beliefs.items()},
            "belief_sums": {str(k): float(v.sum()) for k, v in beliefs.items()},
            "uniform_beliefs": all(np.allclose(v, 1.0 / k_model) for v in beliefs.values()),
        }


def prefix_init_rows(seed: int, k_small: int, k_large: int, num_actions: int = NUM_ACTIONS) -> bool:
    import esl.res.categorical_cell as cc

    init_seed = seed + 7919
    a_small = cc._esl_initial_logits(init_seed, k_model=k_small, num_actions=num_actions)
    a_large = cc._esl_initial_logits(init_seed, k_model=k_large, num_actions=num_actions)
    return bool(np.allclose(a_small, a_large[:k_small]))


def run_esl_k_cell(
    *,
    spec,
    seed: int,
    k_model: int,
    horizon: int = HORIZON,
    h_window: int = H_WINDOW,
) -> dict[str, Any]:
    from esl.res.categorical_cell import run_categorical_cell

    task = game_spec_to_task(spec)
    with esl_k_model(int(k_model)):
        result = run_categorical_cell(
            task=task,
            method="ESL",
            seed=int(seed),
            horizon=int(horizon),
            num_actions=int(spec.num_actions),
            num_opponents=NUM_OPPONENTS,
            h_window=int(h_window),
            p_obs=1.0,
            obs_noise=0.0,
        )
    summary = dict(result["summary"])
    summary["family"] = spec.family
    summary["instance_id"] = spec.instance_id
    summary["k_model"] = int(k_model)
    summary["k_true"] = K_TRUE
    summary["focal_role"] = "row"
    return {"trajectory": result["trajectory"], "summary": summary}


def load_suite_3x3(benchmark_root=None):
    specs = load_suite(benchmark_root, num_actions=3) if benchmark_root is not None else load_suite(num_actions=3)
    specs = sorted(specs, key=lambda s: (s.family, s.instance_id))
    if len(specs) != 12:
        raise RuntimeError(f"expected 12 committed 3x3 games, got {len(specs)}")
    return specs
