"""Agent-level mixture EM: P(cooperate|z=k, a_i=C) and P(cooperate|z=k, a_i=D)."""

from __future__ import annotations

import time

import numpy as np

from esl.baselines.dataset import RunDataset, observed_rows


def _agent_obs_stats(ds: RunDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per agent j: n0, s0 (coop count when ai=0), n1, s1 (when ai=1)."""
    n_agents = ds.num_agents
    n0 = np.zeros(n_agents, dtype=np.float64)
    s0 = np.zeros(n_agents, dtype=np.float64)
    n1 = np.zeros(n_agents, dtype=np.float64)
    s1 = np.zeros(n_agents, dtype=np.float64)
    for r in observed_rows(ds):
        j = r.j
        y = 1.0 if r.a_j == 0 else 0.0
        if r.a_i == 0:
            n0[j] += 1.0
            s0[j] += y
        else:
            n1[j] += 1.0
            s1[j] += y
    return n0, s0, n1, s1


def _log_bernoulli(n: float, s: float, pi: float, eps: float = 1e-12) -> float:
    if n <= 0.0:
        return 0.0
    pi = np.clip(pi, eps, 1.0 - eps)
    return float(s * np.log(pi) + (n - s) * np.log(1.0 - pi))


def em_conditional_bernoulli_mixture(
    ds: RunDataset,
    *,
    max_iter: int = 200,
    tol: float = 1e-6,
    n_restarts: int = 8,
    base_seed: int = 0,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float]:
    """
    Returns (rho (K,), pi_c (K,), pi_d (K,), gamma (N,K), best_iter, best_ll).
    """
    n_agents = ds.num_agents
    k = ds.num_prototypes
    n0, s0, n1, s1 = _agent_obs_stats(ds)

    best_ll = -np.inf
    best_rho = np.ones(k) / k
    best_pi_c = np.full(k, 0.5)
    best_pi_d = np.full(k, 0.5)
    best_gamma = np.ones((n_agents, k)) / k
    best_iter = 0

    for restart in range(n_restarts):
        rng = np.random.default_rng(base_seed + restart * 10007)
        rho = np.ones(k) / k
        pi_c = rng.uniform(0.15, 0.85, size=k)
        pi_d = rng.uniform(0.15, 0.85, size=k)

        ll_prev = -np.inf
        it_used = 0
        gamma = np.ones((n_agents, k)) / k
        ll = -np.inf

        for it in range(max_iter):
            log_lik = np.zeros((n_agents, k), dtype=np.float64)
            for kk in range(k):
                log_lik[:, kk] = np.array(
                    [
                        _log_bernoulli(n0[j], s0[j], pi_c[kk], eps)
                        + _log_bernoulli(n1[j], s1[j], pi_d[kk], eps)
                        for j in range(n_agents)
                    ],
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

            pi_c_new = np.zeros(k, dtype=np.float64)
            pi_d_new = np.zeros(k, dtype=np.float64)
            for kk in range(k):
                w0 = np.dot(gamma[:, kk], n0)
                num0 = np.dot(gamma[:, kk], s0)
                pi_c_new[kk] = (num0 + 1.0) / (w0 + 2.0) if w0 > 0 else 0.5
                w1 = np.dot(gamma[:, kk], n1)
                num1 = np.dot(gamma[:, kk], s1)
                pi_d_new[kk] = (num1 + 1.0) / (w1 + 2.0) if w1 > 0 else 0.5

            rho, pi_c, pi_d = rho_new, pi_c_new, pi_d_new
            it_used = it + 1
            if abs(ll - ll_prev) < tol:
                break
            ll_prev = ll

        if ll > best_ll:
            best_ll = ll
            best_rho = rho.copy()
            best_pi_c = pi_c.copy()
            best_pi_d = pi_d.copy()
            best_gamma = gamma.copy()
            best_iter = it_used

    return best_rho, best_pi_c, best_pi_d, best_gamma, best_iter, float(best_ll)


def marginalize_logits_from_conditional(
    pi_c: np.ndarray,
    pi_d: np.ndarray,
    ds: RunDataset,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Build (K, 2) logits so softmax matches marginal P(C), P(D) under empirical P(a_i) from w>0 rows.
    """
    obs = observed_rows(ds)
    if not obs:
        q = 0.5
    else:
        n_c = sum(1 for r in obs if r.a_i == 0)
        q = n_c / len(obs)
    logits = np.zeros((len(pi_c), 2), dtype=np.float64)
    for k in range(len(pi_c)):
        p_coop = q * pi_c[k] + (1.0 - q) * pi_d[k]
        p_coop = float(np.clip(p_coop, eps, 1.0 - eps))
        p_def = 1.0 - p_coop
        logits[k, 0] = np.log(p_coop)
        logits[k, 1] = np.log(p_def)
    return logits


def em_conditional_with_timing(
    ds: RunDataset, **kwargs: object
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float, float]:
    t0 = time.perf_counter()
    rho, pi_c, pi_d, gamma, iters, ll = em_conditional_bernoulli_mixture(ds, **kwargs)  # type: ignore[arg-type]
    elapsed = time.perf_counter() - t0
    return rho, pi_c, pi_d, gamma, iters, ll, elapsed
