"""Agent-level Bernoulli mixture EM (marginal P(cooperate|z=k), ignores a_i). Ablation baseline."""

from __future__ import annotations

import time

import numpy as np

from esl.baselines.dataset import RunDataset, observed_rows


def _agent_marginal_stats(ds: RunDataset) -> tuple[np.ndarray, np.ndarray]:
    n_agents = ds.num_agents
    n = np.zeros(n_agents, dtype=np.float64)
    s = np.zeros(n_agents, dtype=np.float64)
    for r in observed_rows(ds):
        j = r.j
        y = 1.0 if r.a_j == 0 else 0.0
        n[j] += 1.0
        s[j] += y
    return n, s


def _log_bernoulli(n: float, s: float, pi: float, eps: float = 1e-12) -> float:
    if n <= 0.0:
        return 0.0
    pi = np.clip(pi, eps, 1.0 - eps)
    return float(s * np.log(pi) + (n - s) * np.log(1.0 - pi))


def em_marginal_bernoulli_mixture(
    ds: RunDataset,
    *,
    max_iter: int = 200,
    tol: float = 1e-6,
    n_restarts: int = 8,
    base_seed: int = 0,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    """Returns (rho (K,), pi (K,), gamma (N,K), best_iter, best_ll)."""
    n_agents = ds.num_agents
    k = ds.num_prototypes
    n_tot, s_tot = _agent_marginal_stats(ds)

    best_ll = -np.inf
    best_rho = np.ones(k) / k
    best_pi = np.full(k, 0.5)
    best_gamma = np.ones((n_agents, k)) / k
    best_iter = 0

    for restart in range(n_restarts):
        rng = np.random.default_rng(base_seed + restart * 10007 + 1)
        rho = np.ones(k) / k
        pi = rng.uniform(0.15, 0.85, size=k)
        ll_prev = -np.inf
        it_used = 0
        gamma = np.ones((n_agents, k)) / k
        ll = -np.inf

        for it in range(max_iter):
            log_lik = np.zeros((n_agents, k), dtype=np.float64)
            for kk in range(k):
                log_lik[:, kk] = np.array(
                    [_log_bernoulli(n_tot[j], s_tot[j], pi[kk], eps) for j in range(n_agents)],
                    dtype=np.float64,
                )
            log_rho = np.log(np.clip(rho, eps, 1.0))
            log_joint = log_lik + log_rho.reshape(1, -1)
            m = np.max(log_joint, axis=1, keepdims=True)
            exp_j = np.exp(log_joint - m)
            gamma = exp_j / np.maximum(exp_j.sum(axis=1, keepdims=True), eps)
            ll = float(np.sum(m.ravel() + np.log(exp_j.sum(axis=1) + eps)))

            rho_new = gamma.mean(axis=0)
            rho_new = np.clip(rho_new, eps, 1.0)
            rho_new /= rho_new.sum()

            pi_new = np.zeros(k, dtype=np.float64)
            for kk in range(k):
                w = np.dot(gamma[:, kk], n_tot)
                num = np.dot(gamma[:, kk], s_tot)
                pi_new[kk] = (num + 1.0) / (w + 2.0) if w > 0 else 0.5

            rho, pi = rho_new, pi_new
            it_used = it + 1
            if abs(ll - ll_prev) < tol:
                break
            ll_prev = ll

        if ll > best_ll:
            best_ll = ll
            best_rho = rho.copy()
            best_pi = pi.copy()
            best_gamma = gamma.copy()
            best_iter = it_used

    return best_rho, best_pi, best_gamma, best_iter, float(best_ll)


def logits_from_marginal_pi(pi: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    k = len(pi)
    logits = np.zeros((k, 2), dtype=np.float64)
    for i in range(k):
        p = float(np.clip(pi[i], eps, 1.0 - eps))
        logits[i, 0] = np.log(p)
        logits[i, 1] = np.log(1.0 - p)
    return logits


def em_marginal_with_timing(
    ds: RunDataset, **kwargs: object
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float, float]:
    t0 = time.perf_counter()
    rho, pi, gamma, iters, ll = em_marginal_bernoulli_mixture(ds, **kwargs)  # type: ignore[arg-type]
    elapsed = time.perf_counter() - t0
    return rho, pi, gamma, iters, ll, elapsed
