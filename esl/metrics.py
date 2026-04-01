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

    The minimized objective is **K × MCE** in the paper sense:
    MCE = (1/K) * total_ce with CE(p* || softmax(θ)) per matched pair.
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


def mce_value(
    true_probs: np.ndarray,
    learned_logits: np.ndarray,
    *,
    method: str = "auto",
) -> float:
    """
    **Matched cross-entropy (MCE)** in the paper normalization:
    min_{σ ∈ S_K} (1/K) Σ_k CE(p*_k || softmax(θ_{σ(k)})).
    """
    _, total = match_prototypes_to_types(true_probs, learned_logits, method=method)
    k = true_probs.shape[0]
    return float(total / max(k, 1))


def belief_cross_entropy_vs_type(
    b_ij: np.ndarray,
    true_type_j: int,
    k_proto: int,
    eps: float = 1e-12,
) -> float:
    """
    CE(e_{z_j} || b_{i→j}) = −log b_{i→j}[z_j] (natural units; clamp for stability).

    **Evaluation only:** uses discrete true type index z_j, not learned prototypes.
    """
    b = np.asarray(b_ij, dtype=np.float64).ravel()
    if b.shape[0] != int(k_proto):
        raise ValueError("b_ij length must equal k_proto")
    z = int(true_type_j)
    if not (0 <= z < k_proto):
        raise ValueError("true_type_j out of range")
    b = np.clip(b, eps, 1.0)
    b = b / b.sum()
    return float(-np.log(b[z]))


def belief_kl_true_vs_belief(
    b_ij: np.ndarray,
    true_type_j: int,
    k_proto: int,
    eps: float = 1e-12,
) -> float:
    """
    KL(e_{z_j} || b_{i→j}). **Evaluation only.**
    """
    p = np.zeros(int(k_proto), dtype=np.float64)
    p[int(true_type_j)] = 1.0
    q = np.asarray(b_ij, dtype=np.float64).ravel()
    if q.shape[0] != int(k_proto):
        raise ValueError("b_ij length must equal k_proto")
    q = np.clip(q, eps, 1.0)
    q = q / q.sum()
    return kl_divergence(p, q, eps=eps)


def mean_belief_ce_vs_types(
    beliefs: np.ndarray,
    true_types: np.ndarray,
    num_agents: int,
    k_proto: int,
) -> float:
    """Mean CE(e_{z_j} || b_{i→j}) over ordered pairs i≠j."""
    s = 0.0
    c = 0
    for i in range(num_agents):
        for j in range(num_agents):
            if i == j:
                continue
            s += belief_cross_entropy_vs_type(beliefs[i, j], int(true_types[j]), k_proto)
            c += 1
    return float(s / max(c, 1))


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
