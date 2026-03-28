"""Recovery and dynamics metrics; permutation-invariant prototype matching."""

from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import linear_sum_assignment

from esl.prototypes import stable_softmax


def cross_entropy(p_true: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """CE(p || q) = -sum_a p(a) log q(a)."""
    q = np.clip(q, eps, 1.0)
    return float(-np.sum(p_true * np.log(q)))


def pairwise_assignment_cost(true_probs: np.ndarray, learned_logits: np.ndarray) -> np.ndarray:
    """
    Cost matrix C[t, k] = CE(true_t || softmax(theta_k)).
    true_probs: (K, A), learned_logits: (K, A)
    """
    learned_p = stable_softmax(learned_logits)
    k = true_probs.shape[0]
    c = np.zeros((k, k), dtype=np.float64)
    for t in range(k):
        for k_idx in range(k):
            c[t, k_idx] = cross_entropy(true_probs[t], learned_p[k_idx])
    return c


def hungarian_min_cost_permutation(cost: np.ndarray) -> tuple[np.ndarray, float]:
    """Returns (perm, total_cost) where perm[t] = assigned column index (learned prototype)."""
    row_ind, col_ind = linear_sum_assignment(cost)
    total = float(cost[row_ind, col_ind].sum())
    k = cost.shape[0]
    perm = np.empty(k, dtype=int)
    perm[row_ind] = col_ind
    return perm, total


def brute_force_min_permutation(cost: np.ndarray) -> tuple[np.ndarray, float]:
    k = cost.shape[0]
    best_p = None
    best_c = np.inf
    for cols in itertools.permutations(range(k)):
        cols_a = np.array(cols, dtype=int)
        c = float(np.sum(cost[np.arange(k), cols_a]))
        if c < best_c:
            best_c = c
            best_p = cols_a
    assert best_p is not None
    return best_p, best_c


def match_prototypes_to_types(
    true_probs: np.ndarray,
    learned_logits: np.ndarray,
    *,
    method: str = "auto",
) -> tuple[np.ndarray, float]:
    """
    Permutation-invariant matching: returns (perm, total_ce) where perm[t] is learned index for true t.
    method: 'hungarian', 'brute', or 'auto' (brute if K<=4 else hungarian).
    """
    cost = pairwise_assignment_cost(true_probs, learned_logits)
    k = cost.shape[0]
    if method == "hungarian":
        return hungarian_min_cost_permutation(cost)
    if method == "brute":
        return brute_force_min_permutation(cost)
    if k <= 4:
        return brute_force_min_permutation(cost)
    return hungarian_min_cost_permutation(cost)


def belief_entropy(beliefs: np.ndarray, num_agents: int, k_proto: int) -> float:
    """Mean entropy of off-diagonal beliefs."""
    ent = 0.0
    count = 0
    for i in range(num_agents):
        for j in range(num_agents):
            if i == j:
                continue
            p = beliefs[i, j]
            ent += float(-np.sum(p * np.log(np.clip(p, 1e-12, 1.0))))
            count += 1
    return ent / max(count, 1)


def belief_argmax_accuracy(
    beliefs: np.ndarray,
    true_types: np.ndarray,
    perm: np.ndarray,
    num_agents: int,
) -> float:
    """
    Fraction of ordered pairs (i,j) where argmax_k b[i,j,k] == perm^{-1}[true_type[j]].
    Map: true type t should place mass on prototype index k with perm[t]=k... 
    Actually perm[t] = learned prototype matched to true t.
    Agent j has true type T_j. Correct prototype index for j is k* such that perm[T_j] == k*.
    """
    correct = 0
    total = 0
    for i in range(num_agents):
        for j in range(num_agents):
            if i == j:
                continue
            t_j = int(true_types[j])
            k_star = int(perm[t_j])
            pred = int(np.argmax(beliefs[i, j]))
            correct += int(pred == k_star)
            total += 1
    return correct / max(total, 1)


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q))))
