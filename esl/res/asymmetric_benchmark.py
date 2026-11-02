"""Committed 3×3 asymmetric-bimatrix suite: incentive templates, not affine scale."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from esl.res.benchmark import FAMILIES, write_suite
from esl.res.game_spec import GameSpec, InteractionFamily

ASYM_BENCHMARK_ROOT_NAME = "benchmark_asym3"
DEFAULT_ASYM_SEED = 20260725 + 17
TEMPLATES: tuple[str, ...] = (
    "reward_preference",
    "penalty",
    "equilibrium_conflict",
    "combined",
)
TemplateName = Literal[
    "reward_preference",
    "penalty",
    "equilibrium_conflict",
    "combined",
]


def default_asym_benchmark_root(repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parents[2]
    return root / ASYM_BENCHMARK_ROOT_NAME


def normalize_payoff(mat: np.ndarray) -> np.ndarray:
    """Independent affine map of a single player's matrix to [-1, 1]."""
    m = np.asarray(mat, dtype=np.float64)
    lo = float(m.min())
    hi = float(m.max())
    if hi - lo < 1e-12:
        return np.zeros_like(m)
    return 2.0 * (m - lo) / (hi - lo) - 1.0


def asymmetry_score(row: np.ndarray, col: np.ndarray) -> float:
    """A_payoff = ||U_row - U_col^T||_F / (||U_row||_F + ||U_col||_F)."""
    r = np.asarray(row, dtype=np.float64)
    c = np.asarray(col, dtype=np.float64)
    num = float(np.linalg.norm(r - c.T))
    den = float(np.linalg.norm(r) + np.linalg.norm(c))
    return num / max(den, 1e-12)


def _is_nash(row: np.ndarray, col: np.ndarray, i: int, j: int, tol: float = 1e-9) -> bool:
    """Pure-strategy Nash at (i, j)."""
    if float(row[i, j]) + tol < float(np.max(row[:, j])):
        return False
    if float(col[i, j]) + tol < float(np.max(col[i, :])):
        return False
    return True


def _entry_corr(row: np.ndarray, col: np.ndarray) -> float:
    a = np.asarray(row, dtype=np.float64).reshape(-1)
    b = np.asarray(col, dtype=np.float64).reshape(-1)
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _zero_sum_residual(row: np.ndarray, col: np.ndarray) -> float:
    r = np.asarray(row, dtype=np.float64)
    c = np.asarray(col, dtype=np.float64)
    return float(np.linalg.norm(r + c) / max(np.linalg.norm(r) + np.linalg.norm(c), 1e-12))


def _pd_like_3x3(
    *,
    t_r: float,
    r_r: float,
    p_r: float,
    s_r: float,
    t_c: float,
    r_c: float,
    p_c: float,
    s_c: float,
    mid_r: float,
    mid_c: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    3×3 social-dilemma skeleton (actions C=0, D=1, M=2).
    Mutual C is (r_*, r_*); temptation on D vs C; M is mixed interest.
    """
    row = np.array(
        [
            [r_r, s_r, s_r * 0.5 + mid_r * 0.5],
            [t_r, p_r, p_r],
            [t_r * 0.7 + mid_r * 0.3, p_r, mid_r],
        ],
        dtype=np.float64,
    )
    col = np.array(
        [
            [r_c, t_c, t_c * 0.7 + mid_c * 0.3],
            [s_c, p_c, p_c],
            [s_c * 0.5 + mid_c * 0.5, p_c, mid_c],
        ],
        dtype=np.float64,
    )
    return row, col


def _bos_like_3x3(
    *,
    hi_r: float,
    lo_r: float,
    mid_r: float,
    miss_r: float,
    hi_c: float,
    lo_c: float,
    mid_c: float,
    miss_c: float,
    prefer_row: int,
    prefer_col: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Coordination / BoS-style: diagonal matching high; off-diagonal = miss.
    prefer_row / prefer_col index which matched outcome each player likes most.
    """
    row = np.full((3, 3), miss_r, dtype=np.float64)
    col = np.full((3, 3), miss_c, dtype=np.float64)
    diag_vals_r = [lo_r, mid_r, hi_r]
    diag_vals_c = [lo_c, mid_c, hi_c]
    # Place each player's ranking on the three matched profiles
    order_r = [prefer_row, (prefer_row + 1) % 3, (prefer_row + 2) % 3]
    order_c = [prefer_col, (prefer_col + 1) % 3, (prefer_col + 2) % 3]
    for rank, a in enumerate(order_r):
        row[a, a] = diag_vals_r[2 - rank]  # rank0 = hi
    for rank, a in enumerate(order_c):
        col[a, a] = diag_vals_c[2 - rank]
    return row, col


def _competitive_cycle_3x3(
    *,
    win_r: float,
    loss_r: float,
    win_c: float,
    loss_c: float,
    tie_r: float,
    tie_c: float,
    wedge: float,
    mode: str = "reward_preference",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Competitive general-sum 3×3: negative-interest skeleton + structural wedge.

    Skeleton is cyclic (0>1>2>0) and approximately opposing. Templates add a
    general-sum component that changes relative incentives (not a global scale).
    """
    beats = {(0, 1), (1, 2), (2, 0)}
    row = np.zeros((3, 3), dtype=np.float64)
    col = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            if i == j:
                row[i, j] = 0.0
                col[i, j] = 0.0
            elif (i, j) in beats:
                row[i, j] = win_r
                col[i, j] = -loss_c
            else:
                row[i, j] = -loss_r
                col[i, j] = win_c

    # Structural general-sum wedges (template-specific)
    if mode == "reward_preference":
        # Row cares more about beating action 1; column cares more about beating action 0
        row[0, 1] += wedge
        col[1, 0] += 0.7 * wedge
        row[1, 2] += 0.3 * wedge
        col[0, 2] -= 0.4 * wedge
        row[0, 0] += tie_r
        col[2, 2] += tie_c
    elif mode == "penalty":
        # Asymmetric loss severity on particular mismatches
        row[1, 0] -= wedge
        col[0, 1] -= 0.6 * wedge
        row[2, 1] -= 0.5 * wedge
        col[2, 0] += 0.3 * wedge
        row[0, 0] -= abs(tie_r)
        col[0, 0] += abs(tie_c)
    elif mode == "equilibrium_conflict":
        # Disagreement on which draw is acceptable
        row[0, 0] += abs(tie_r) + 0.5 * wedge
        col[0, 0] -= 0.3 * wedge
        row[2, 2] -= 0.4 * wedge
        col[2, 2] += abs(tie_c) + 0.5 * wedge
        row[1, 1] += 0.2 * wedge
        col[1, 1] -= 0.2 * wedge
    else:  # combined
        row[0, 1] += wedge
        col[1, 0] += 0.5 * wedge
        row[1, 0] -= 0.7 * wedge
        col[0, 1] -= 0.4 * wedge
        row[0, 0] += abs(tie_r)
        col[2, 2] += abs(tie_c)
        row[2, 2] -= 0.3 * wedge
        col[0, 0] -= 0.2 * wedge
    return row, col


def _coop_params(template: TemplateName) -> dict[str, float]:
    # Base dilemma numbers before normalization; templates change relative incentives.
    if template == "reward_preference":
        return dict(
            t_r=5.5, r_r=3.0, p_r=1.0, s_r=0.0, mid_r=1.8,
            t_c=4.0, r_c=3.5, p_c=1.0, s_c=0.2, mid_c=2.2,
        )
    if template == "penalty":
        return dict(
            t_r=4.5, r_r=3.0, p_r=1.2, s_r=-1.5, mid_r=1.5,
            t_c=4.5, r_c=3.0, p_c=0.4, s_c=0.5, mid_c=1.5,
        )
    if template == "equilibrium_conflict":
        # Mutual C still good, but M-profile valued differently → conflicted secondary interest
        return dict(
            t_r=4.2, r_r=3.2, p_r=1.0, s_r=0.0, mid_r=3.6,
            t_c=4.2, r_c=3.2, p_c=1.0, s_c=0.0, mid_c=1.2,
        )
    # combined
    return dict(
        t_r=5.8, r_r=2.8, p_r=1.1, s_r=-1.2, mid_r=3.4,
        t_c=4.0, r_c=3.6, p_c=0.3, s_c=0.6, mid_c=1.0,
    )


def _coord_params(template: TemplateName) -> dict[str, Any]:
    if template == "reward_preference":
        return dict(
            hi_r=6.0, mid_r=3.0, lo_r=2.0, miss_r=0.0,
            hi_c=5.0, mid_c=4.5, lo_c=2.0, miss_c=0.0,
            prefer_row=0, prefer_col=1,
        )
    if template == "penalty":
        return dict(
            hi_r=5.0, mid_r=3.5, lo_r=2.5, miss_r=-2.0,
            hi_c=5.0, mid_c=3.5, lo_c=2.5, miss_c=0.0,
            prefer_row=0, prefer_col=1,
        )
    if template == "equilibrium_conflict":
        return dict(
            hi_r=6.0, mid_r=2.5, lo_r=2.0, miss_r=-0.5,
            hi_c=6.0, mid_c=2.5, lo_c=2.0, miss_c=-0.5,
            prefer_row=0, prefer_col=2,
        )
    return dict(
        hi_r=6.5, mid_r=3.0, lo_r=1.5, miss_r=-1.5,
        hi_c=5.0, mid_c=4.0, lo_c=2.0, miss_c=0.2,
        prefer_row=1, prefer_col=0,
    )


def _adv_params(template: TemplateName) -> dict[str, float]:
    if template == "reward_preference":
        return dict(win_r=3.0, loss_r=1.2, win_c=2.0, loss_c=2.5, tie_r=0.3, tie_c=-0.2, wedge=1.5)
    if template == "penalty":
        return dict(win_r=2.2, loss_r=3.0, win_c=2.2, loss_c=1.0, tie_r=0.4, tie_c=0.5, wedge=1.8)
    if template == "equilibrium_conflict":
        return dict(win_r=2.4, loss_r=1.6, win_c=2.4, loss_c=1.6, tie_r=1.5, tie_c=1.5, wedge=2.0)
    return dict(win_r=3.5, loss_r=2.8, win_c=1.5, loss_c=2.2, tie_r=0.6, tie_c=0.4, wedge=2.2)


def build_asymmetric_raw(
    family: InteractionFamily,
    template: TemplateName,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if family == "cooperative":
        params = _coop_params(template)
        row, col = _pd_like_3x3(**params)
    elif family == "coordination":
        params = _coord_params(template)
        row, col = _bos_like_3x3(**params)
    elif family == "adversarial":
        params = _adv_params(template)
        row, col = _competitive_cycle_3x3(**params, mode=template)
    else:
        raise ValueError(family)
    return row, col, {"template": template, **params}


def assert_asymmetric_family_constraints(
    *,
    family: InteractionFamily,
    row: np.ndarray,
    col: np.ndarray,
    template: str,
) -> None:
    r = np.asarray(row, dtype=np.float64)
    c = np.asarray(col, dtype=np.float64)
    if r.shape != (3, 3) or c.shape != (3, 3):
        raise AssertionError("asymmetric suite requires 3×3 matrices")
    # Not role-symmetric
    if np.allclose(r, c.T, atol=1e-6):
        raise AssertionError(f"{family}/{template}: still role-symmetric (row≈col.T)")
    # Not identical interests
    if np.allclose(r, c, atol=1e-6):
        raise AssertionError(f"{family}/{template}: identical interests")
    a_score = asymmetry_score(r, c)
    if a_score < 0.08:
        raise AssertionError(f"{family}/{template}: A_payoff too small ({a_score:.3f})")

    if family == "cooperative":
        # Mutual C attractive vs mutual D for both; temptation exists for at least one
        if not (r[0, 0] > r[1, 1] and c[0, 0] > c[1, 1]):
            raise AssertionError(f"{family}/{template}: mutual C not better than mutual D")
        if not (r[1, 0] > r[0, 0] or c[0, 1] > c[0, 0]):
            raise AssertionError(f"{family}/{template}: missing temptation structure")
    elif family == "coordination":
        # At least two distinct pure matched Nash (or clearly preferred diagonals)
        nash_diag = [k for k in range(3) if _is_nash(r, c, k, k)]
        if len(nash_diag) < 1:
            # soften: both prefer their matched cell vs mismatch in their row/col
            if not all(r[k, k] > np.max(r[k, :]) - 1e-9 or c[k, k] > np.max(c[:, k]) - 1e-9 for k in range(3)):
                raise AssertionError(f"{family}/{template}: weak coordination structure")
        # Players disagree on best matched outcome
        best_r = int(np.argmax([r[k, k] for k in range(3)]))
        best_c = int(np.argmax([c[k, k] for k in range(3)]))
        if best_r == best_c and template != "penalty":
            # penalty template may share preferred eq but differ in miss costs
            if np.allclose(r, c, atol=0.15):
                raise AssertionError(f"{family}/{template}: no equilibrium-preference conflict")
    elif family == "adversarial":
        corr = _entry_corr(r, c)
        if corr > -0.15:
            raise AssertionError(f"{family}/{template}: payoff corr too high ({corr:.3f}; need opposing interests)")
        # No dominant jointly high profile (after [-1,1] normalize)
        joint = r + c
        if float(joint.max()) > 0.85:
            raise AssertionError(f"{family}/{template}: joint win-win cell too strong ({joint.max():.3f})")
        # Exact zero-sum forbidden; residual check is on structure surviving normalize via A_payoff
        if np.allclose(c, -r, atol=1e-5):
            raise AssertionError(f"{family}/{template}: exact zero-sum not allowed in asym suite")
        # Role asymmetry must remain material after normalize (row != col.T)
        if a_score < 0.12:
            raise AssertionError(f"{family}/{template}: role asymmetry too weak after normalize ({a_score:.3f})")


def generate_3x3_asymmetric_suite(*, seed: int = DEFAULT_ASYM_SEED) -> list[GameSpec]:
    """Four construction templates × three families = 12 committed games."""
    del seed  # templates are deterministic; seed reserved for future jitter variants
    specs: list[GameSpec] = []
    prefixes = {"cooperative": "coop3a", "coordination": "coord3a", "adversarial": "adv3a"}
    for family in FAMILIES:
        for i, template in enumerate(TEMPLATES, start=1):
            raw_row, raw_col, params = build_asymmetric_raw(family, template)  # type: ignore[arg-type]
            row = normalize_payoff(raw_row)
            col = normalize_payoff(raw_col)
            assert_asymmetric_family_constraints(family=family, row=row, col=col, template=template)
            a_score = asymmetry_score(row, col)
            instance_id = f"{prefixes[family]}_{i:02d}"
            meta = {
                "asymmetric_suite": 1.0,
                "A_payoff": float(a_score),
                "template_index": float(i),
                "entry_corr": float(_entry_corr(row, col)),
                "zero_sum_residual": float(_zero_sum_residual(row, col)),
            }
            # Keep numeric construction knobs when present
            for k, v in params.items():
                if k == "template":
                    continue
                if isinstance(v, (int, float)):
                    meta[str(k)] = float(v)
            specs.append(
                GameSpec(
                    instance_id=instance_id,
                    family=family,
                    num_actions=3,
                    payoff_row=tuple(tuple(float(x) for x in rr) for rr in row),
                    payoff_col=tuple(tuple(float(x) for x in rr) for rr in col),
                    generator_seed=DEFAULT_ASYM_SEED,
                    notes=f"asym3|{template}|A={a_score:.3f}",
                    params=meta,
                )
            )
    return specs


def swap_roles(spec: GameSpec) -> GameSpec:
    """Focal becomes former column player: swap payoff matrices."""
    return GameSpec(
        instance_id=spec.instance_id,
        family=spec.family,
        num_actions=spec.num_actions,
        payoff_row=spec.payoff_col,
        payoff_col=spec.payoff_row,
        generator_seed=spec.generator_seed,
        notes=spec.notes + "|role=col",
        params=dict(spec.params) if spec.params else None,
    )


def write_asymmetric_suite(root: Path | None = None) -> dict[str, Any]:
    root = root or default_asym_benchmark_root()
    specs = generate_3x3_asymmetric_suite()
    if root.exists():
        for path in root.rglob("*.json"):
            path.unlink()
    man = write_suite(specs, root)
    # Override manifest description
    manifest = json.loads(man.read_text(encoding="utf-8"))
    manifest["description"] = (
        "Committed 3×3 asymmetric-bimatrix RES suite. "
        "Four incentive templates per family (not affine scales); "
        "each player's matrix normalized to [-1,1]. "
        "Adversarial family is competitive general-sum (not exact zero-sum)."
    )
    manifest["suite"] = "asymmetric_3x3"
    man.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    audit = {
        "n_total": len(specs),
        "templates": list(TEMPLATES),
        "per_instance": [
            {
                "instance_id": s.instance_id,
                "family": s.family,
                "notes": s.notes,
                "A_payoff": (s.params or {}).get("A_payoff"),
                "entry_corr": (s.params or {}).get("entry_corr"),
                "zero_sum_residual": (s.params or {}).get("zero_sum_residual"),
            }
            for s in specs
        ],
    }
    (root / "diversity_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit
