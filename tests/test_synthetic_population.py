"""Synthetic population sampling (evaluation / sim only)."""

import numpy as np
import pytest

from esl.synthetic_population import (
    SoftmaxLogitsPolicy,
    agent_logits_from_star,
    build_softmax_policies,
    sample_gaussian_parameter_noise,
    sample_latent_types,
)


def test_sample_latent_types_matches_rho_empirically():
    rng = np.random.default_rng(0)
    rho = np.array([0.7, 0.2, 0.1])
    z = sample_latent_types(rng, rho, n_agents=30_000)
    counts = np.bincount(z, minlength=3) / len(z)
    assert np.allclose(counts, rho, atol=0.02)


def test_agent_logits_cluster_around_star_rows():
    rng = np.random.default_rng(1)
    theta_star = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    rho = np.array([0.5, 0.5])
    z = sample_latent_types(rng, rho, n_agents=4000)
    phi = sample_gaussian_parameter_noise(rng, len(z), 2, sigma=0.05)
    tilde = agent_logits_from_star(theta_star, z, phi)
    m0 = tilde[z == 0].mean(axis=0)
    m1 = tilde[z == 1].mean(axis=0)
    assert np.linalg.norm(m0 - theta_star[0]) < 0.02
    assert np.linalg.norm(m1 - theta_star[1]) < 0.02


def test_softmax_policy_act_samples():
    rng = np.random.default_rng(2)
    pol = SoftmaxLogitsPolicy(np.array([10.0, -10.0]))
    acts = [pol.act(rng, last_opponent_action=None) for _ in range(50)]
    assert all(a == 0 for a in acts)


def test_build_softmax_policies_len():
    al = np.zeros((3, 2), dtype=np.float64)
    policies = build_softmax_policies(al)
    assert len(policies) == 3
    assert all(isinstance(p, SoftmaxLogitsPolicy) for p in policies)


def test_agent_logits_from_star_shape_errors():
    with pytest.raises(ValueError):
        agent_logits_from_star(np.zeros((2, 2)), np.array([0, 9]), np.zeros((2, 2)))
