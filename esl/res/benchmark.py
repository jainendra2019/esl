"""Fixed RES benchmark suite: generate-once, commit, load-only at experiment time."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from esl.res.game_spec import GameSpec, InteractionFamily

BENCHMARK_ROOT_NAME = "benchmark"
FAMILIES: tuple[InteractionFamily, ...] = ("cooperative", "coordination", "adversarial")
DEFAULT_GENERATOR_SEED = 20260725
MIN_PAIRWISE_DISTANCE = 1.25  # reject near-duplicates in feature space


def default_benchmark_root(repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parents[2]
    return root / BENCHMARK_ROOT_NAME


def _pd_matrices(t: float, r: float, p: float, s: float) -> tuple[np.ndarray, np.ndarray]:
    row = np.array([[r, s], [t, p]], dtype=np.float64)
    col = np.array([[r, t], [s, p]], dtype=np.float64)
    return row, col


def _feature_pd(t: float, r: float, p: float, s: float) -> np.ndarray:
    gaps = np.array([t - r, r - p, p - s, t - s], dtype=np.float64)
    return np.concatenate([np.array([t, r, p, s], dtype=np.float64), gaps])


def _feature_zs(row: np.ndarray) -> np.ndarray:
    return np.asarray(row, dtype=np.float64).reshape(-1)


def _greedy_diverse(
    candidates: list[tuple[np.ndarray, dict]],
    *,
    k: int,
    min_dist: float,
) -> list[tuple[np.ndarray, dict]]:
    """Greedy farthest-point selection with near-duplicate rejection."""
    if not candidates:
        return []
    selected: list[tuple[np.ndarray, dict]] = [candidates[0]]
    for feat, meta in candidates[1:]:
        if len(selected) >= k:
            break
        dists = [float(np.linalg.norm(feat - sfeat)) for sfeat, _ in selected]
        if min(dists) < min_dist:
            continue
        selected.append((feat, meta))
    # If too few, fill by max min-distance but prefer remaining distance ≥ 0.5×min_dist
    if len(selected) < k:
        remaining = [(f, m) for f, m in candidates[1:] if all(m is not sm for _, sm in selected)]
        # Re-include paper/first only once; rebuild remaining from all unused
        used_ids = {id(m) for _, m in selected}
        remaining = [(f, m) for f, m in candidates if id(m) not in used_ids]
        while len(selected) < k and remaining:
            best_i = -1
            best_d = -1.0
            for i, (feat, _) in enumerate(remaining):
                d = min(float(np.linalg.norm(feat - sfeat)) for sfeat, _ in selected)
                if d > best_d:
                    best_d = d
                    best_i = i
            if best_d < 0.5 * min_dist and len(selected) >= max(2, k - 1):
                break
            selected.append(remaining.pop(best_i))
    return selected[:k]


def _sample_cooperative(rng: np.random.Generator, n_candidates: int = 200) -> list[tuple[np.ndarray, dict]]:
    out: list[tuple[np.ndarray, dict]] = []
    # Always include paper IPD first
    paper = (5.0, 3.0, 1.0, 0.0)
    out.append((_feature_pd(*paper), {"T": paper[0], "R": paper[1], "P": paper[2], "S": paper[3], "tag": "paper_ipd"}))
    for _ in range(n_candidates):
        s = float(rng.uniform(-1.0, 0.5))
        p = float(rng.uniform(s + 0.5, s + 3.0))
        r = float(rng.uniform(p + 0.5, p + 4.0))
        t = float(rng.uniform(r + 0.5, r + 5.0))
        if not (t > r > p > s):
            continue
        out.append((_feature_pd(t, r, p, s), {"T": t, "R": r, "P": p, "S": s, "tag": "sampled"}))
    return out


def _sample_coordination(rng: np.random.Generator, n_candidates: int = 200) -> list[tuple[np.ndarray, dict]]:
    out: list[tuple[np.ndarray, dict]] = []
    paper = (3.0, 4.0, 2.0, 0.0)  # T,R,P,S with R>T>=P>S for paper stag hunt
    out.append((_feature_pd(*paper), {"T": paper[0], "R": paper[1], "P": paper[2], "S": paper[3], "tag": "paper_stag_hunt"}))
    for _ in range(n_candidates):
        s = float(rng.uniform(-1.0, 0.5))
        p = float(rng.uniform(s + 0.25, s + 2.5))
        t = float(rng.uniform(p, p + 3.0))
        r = float(rng.uniform(t + 0.5, t + 4.0))
        if not (r > t >= p > s):
            continue
        out.append((_feature_pd(t, r, p, s), {"T": t, "R": r, "P": p, "S": s, "tag": "sampled"}))
    return out


def _sample_adversarial(rng: np.random.Generator, n_candidates: int = 200) -> list[tuple[np.ndarray, dict]]:
    out: list[tuple[np.ndarray, dict]] = []
    paper = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float64)
    out.append((_feature_zs(paper), {"row": paper.tolist(), "tag": "paper_matching_pennies"}))
    for _ in range(n_candidates):
        # Biased / scaled zero-sum variants of matching-pennies structure
        a = float(rng.uniform(0.5, 4.0))
        b = float(rng.uniform(0.5, 4.0))
        c = float(rng.uniform(0.5, 4.0))
        d = float(rng.uniform(0.5, 4.0))
        row = np.array([[a, -b], [-c, d]], dtype=np.float64)
        out.append((_feature_zs(row), {"row": row.tolist(), "tag": "sampled_zs"}))
    return out


def generate_2x2_suite(
    *,
    seed: int = DEFAULT_GENERATOR_SEED,
    per_family: int = 4,
    min_dist: float = MIN_PAIRWISE_DISTANCE,
) -> list[GameSpec]:
    """Constraint-satisfying candidates → diversity filter → fixed GameSpec list."""
    rng = np.random.default_rng(seed)
    specs: list[GameSpec] = []

    coop_sel = _greedy_diverse(_sample_cooperative(rng), k=per_family, min_dist=min_dist)
    for i, (_, meta) in enumerate(coop_sel, start=1):
        t, r, p, s = float(meta["T"]), float(meta["R"]), float(meta["P"]), float(meta["S"])
        row, col = _pd_matrices(t, r, p, s)
        specs.append(
            GameSpec(
                instance_id=f"coop_{i:02d}",
                family="cooperative",
                num_actions=2,
                payoff_row=tuple(tuple(float(x) for x in rr) for rr in row),
                payoff_col=tuple(tuple(float(x) for x in rr) for rr in col),
                generator_seed=seed,
                notes=str(meta.get("tag", "")),
                params={"T": t, "R": r, "P": p, "S": s},
            )
        )

    coord_sel = _greedy_diverse(_sample_coordination(rng), k=per_family, min_dist=min_dist)
    for i, (_, meta) in enumerate(coord_sel, start=1):
        t, r, p, s = float(meta["T"]), float(meta["R"]), float(meta["P"]), float(meta["S"])
        row, col = _pd_matrices(t, r, p, s)
        specs.append(
            GameSpec(
                instance_id=f"coord_{i:02d}",
                family="coordination",
                num_actions=2,
                payoff_row=tuple(tuple(float(x) for x in rr) for rr in row),
                payoff_col=tuple(tuple(float(x) for x in rr) for rr in col),
                generator_seed=seed,
                notes=str(meta.get("tag", "")),
                params={"T": t, "R": r, "P": p, "S": s},
            )
        )

    adv_sel = _greedy_diverse(_sample_adversarial(rng), k=per_family, min_dist=min_dist)
    for i, (_, meta) in enumerate(adv_sel, start=1):
        row = np.asarray(meta["row"], dtype=np.float64)
        col = -row
        specs.append(
            GameSpec(
                instance_id=f"adv_{i:02d}",
                family="adversarial",
                num_actions=2,
                payoff_row=tuple(tuple(float(x) for x in rr) for rr in row),
                payoff_col=tuple(tuple(float(x) for x in rr) for rr in col),
                generator_seed=seed,
                notes=str(meta.get("tag", "")),
                params=None,
            )
        )

    return specs


def _sample_3x3_cooperative(rng: np.random.Generator, n: int = 60) -> list[tuple[np.ndarray, dict]]:
    """3×3 mixed-interest: elevate C/C reward, temptation on unilateral D-like action."""
    out: list[tuple[np.ndarray, dict]] = []
    for _ in range(n):
        base = rng.uniform(0.0, 1.0, size=(3, 3))
        # Encourage cooperate-like action 0 mutual reward
        r = float(rng.uniform(2.0, 5.0))
        t = float(rng.uniform(r + 0.5, r + 3.0))
        p = float(rng.uniform(0.0, 1.5))
        s = float(rng.uniform(-1.0, 0.5))
        row = np.array(
            [
                [r, s, s],
                [t, p, p],
                [t * 0.8, p, float(rng.uniform(p, p + 1.0))],
            ],
            dtype=np.float64,
        )
        # Symmetric social dilemma style
        col = row.T.copy()
        feat = np.concatenate([row.reshape(-1), np.array([t - r, r - p, p - s])])
        out.append((feat, {"row": row.tolist(), "col": col.tolist()}))
    return out


def _sample_3x3_coordination(rng: np.random.Generator, n: int = 60) -> list[tuple[np.ndarray, dict]]:
    out: list[tuple[np.ndarray, dict]] = []
    for _ in range(n):
        r = float(rng.uniform(3.0, 6.0))
        t = float(rng.uniform(1.0, r - 0.5))
        p = float(rng.uniform(0.5, t))
        s = float(rng.uniform(-1.0, 0.5))
        mid = float(rng.uniform(p, r))
        row = np.array(
            [
                [r, s, s],
                [s, mid, s],
                [s, s, p],
            ],
            dtype=np.float64,
        )
        col = row.copy()
        feat = np.concatenate([row.reshape(-1), np.array([r - t, t - p, mid - p])])
        out.append((feat, {"row": row.tolist(), "col": col.tolist()}))
    return out


def _sample_3x3_adversarial(rng: np.random.Generator, n: int = 60) -> list[tuple[np.ndarray, dict]]:
    out: list[tuple[np.ndarray, dict]] = []
    for _ in range(n):
        row = rng.normal(0.0, 1.0, size=(3, 3))
        # Zero-sumify
        row = row - row.mean()
        scale = float(rng.uniform(0.5, 3.0))
        row = scale * row
        col = -row
        out.append((_feature_zs(row), {"row": row.tolist(), "col": col.tolist()}))
    return out


def generate_3x3_suite(
    *,
    seed: int = DEFAULT_GENERATOR_SEED + 3,
    per_family: int = 4,
    min_dist: float = MIN_PAIRWISE_DISTANCE,
) -> list[GameSpec]:
    rng = np.random.default_rng(seed)
    specs: list[GameSpec] = []
    samplers = {
        "cooperative": _sample_3x3_cooperative,
        "coordination": _sample_3x3_coordination,
        "adversarial": _sample_3x3_adversarial,
    }
    prefixes = {"cooperative": "coop3", "coordination": "coord3", "adversarial": "adv3"}
    for family in FAMILIES:
        selected = _greedy_diverse(samplers[family](rng), k=per_family, min_dist=min_dist)
        for i, (_, meta) in enumerate(selected, start=1):
            row = np.asarray(meta["row"], dtype=np.float64)
            col = np.asarray(meta["col"], dtype=np.float64)
            specs.append(
                GameSpec(
                    instance_id=f"{prefixes[family]}_{i:02d}",
                    family=family,
                    num_actions=3,
                    payoff_row=tuple(tuple(float(x) for x in rr) for rr in row),
                    payoff_col=tuple(tuple(float(x) for x in rr) for rr in col),
                    generator_seed=seed,
                    notes="3x3_sampled",
                )
            )
    return specs


def write_suite(specs: Iterable[GameSpec], root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    written: list[dict] = []
    for spec in specs:
        fam_dir = root / spec.family
        fam_dir.mkdir(parents=True, exist_ok=True)
        path = fam_dir / f"{spec.instance_id}.json"
        spec.save_json(path)
        written.append(
            {
                "instance_id": spec.instance_id,
                "family": spec.family,
                "num_actions": spec.num_actions,
                "path": str(path.relative_to(root)),
            }
        )
    manifest = {
        "version": 1,
        "description": (
            "Fixed RES benchmark suite. Interaction regime ≠ payoff realization. "
            "Instances were generated once under family constraints with a diversity filter; "
            "experiments must load these files rather than resampling."
        ),
        "generator_module": "esl.res.benchmark",
        "n_instances": len(written),
        "instances": written,
    }
    man_path = root / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return man_path


def load_suite(
    root: Path | None = None,
    *,
    num_actions: int | None = None,
    family: InteractionFamily | None = None,
) -> list[GameSpec]:
    root = root or default_benchmark_root()
    specs: list[GameSpec] = []
    for fam in FAMILIES:
        if family is not None and fam != family:
            continue
        fam_dir = root / fam
        if not fam_dir.is_dir():
            continue
        for path in sorted(fam_dir.glob("*.json")):
            spec = GameSpec.load_json(path)
            if num_actions is not None and spec.num_actions != num_actions:
                continue
            specs.append(spec)
    return specs


def min_pairwise_feature_distance(specs: list[GameSpec]) -> dict[str, float]:
    """Per-(family, A) minimum pairwise L2 distance in feature space (diversity audit)."""
    out: dict[str, float] = {}
    keys = sorted({(s.family, int(s.num_actions)) for s in specs})
    for fam, a in keys:
        group = [s for s in specs if s.family == fam and int(s.num_actions) == a]
        label = f"{fam}_A{a}"
        if len(group) < 2:
            out[label] = float("inf")
            continue
        feats = [s.feature_vector() for s in group]
        dists = [
            float(np.linalg.norm(feats[i] - feats[j]))
            for i in range(len(feats))
            for j in range(i + 1, len(feats))
        ]
        out[label] = float(min(dists))
    return out


def assert_family_constraints(spec: GameSpec) -> None:
    """Validate defining inequalities / zero-sum structure for committed instances."""
    row = np.asarray(spec.payoff_row, dtype=np.float64)
    col = np.asarray(spec.payoff_col, dtype=np.float64)
    if spec.num_actions == 2 and spec.family in ("cooperative", "coordination") and spec.params:
        t, r, p, s = spec.params["T"], spec.params["R"], spec.params["P"], spec.params["S"]
        if spec.family == "cooperative" and not (t > r > p > s):
            raise AssertionError(f"{spec.instance_id}: cooperative requires T>R>P>S")
        if spec.family == "coordination" and not (r > t >= p > s):
            raise AssertionError(f"{spec.instance_id}: coordination requires R>T>=P>S")
    if spec.family == "adversarial":
        # Asymmetric suite uses competitive general-sum (validated separately).
        asym = bool((spec.params or {}).get("asymmetric_suite")) or (
            "asym3|" in str(spec.notes)
        )
        if not asym and not np.allclose(col, -row, atol=1e-8):
            raise AssertionError(f"{spec.instance_id}: adversarial instances must be zero-sum (col=-row)")
