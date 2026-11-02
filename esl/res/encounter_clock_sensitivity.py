"""Encounter-clock opponent-count sensitivity (primary apples-to-apples design).

Switches and post-switch evaluation are defined in *per-opponent encounter time*,
not global rounds. Global-clock switch-matched results remain diagnostic only.

Locked protocol (all J in {3,10,100}):
  pre-switch encounters / opponent / cycle : 60
  post-switch evaluation H_e               : 30
  evaluated switches / opponent            : 5
  total encounters / opponent              : 450
  opponent sampling                        : uniform
"""

from __future__ import annotations

import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from esl.experiments.figure1_dynamic_adaptation import FIGURE1_MAIN_SEEDS
from esl.res.benchmark import load_suite
from esl.res.categorical_cell import (
    ENC_CYCLE,
    ENC_H_E,
    ENC_N_SWITCHES,
    ENC_PER_OPPONENT,
    ENC_PRE_SWITCH,
    ENC_SWITCH_POINTS,
    encounter_clock_horizon,
    opponent_ids,
    run_categorical_cell,
)
from esl.res.game_spec import GameSpec
from esl.res.opponent_count_sensitivity import (
    compute_matched_game_seed_contrast,
    plot_opponent_count_ladder,
    summarize_diagnostics,
)
from esl.res.runner import game_spec_to_task

NUM_ACTIONS = 3
SEEDS: tuple[int, ...] = FIGURE1_MAIN_SEEDS
J_LADDER: tuple[int, ...] = (3, 10, 100)


def compute_encounter_psr_diagnostics(
    rows: list[dict[str, Any]],
    *,
    num_opponents: int,
    h_e: int = ENC_H_E,
) -> dict[str, float]:
    """PSR / DPR in encounter time; discard incomplete post-switch windows."""
    # Group by opponent in encounter order (global round order for that opp).
    by_opp: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_opp[int(r["opponent_id"])].append(r)
    for opp in by_opp:
        by_opp[opp].sort(key=lambda x: int(x["round"]))

    switch_psr: list[float] = []
    switch_dpr: list[float] = []
    recovery_encounters: list[float] = []
    n_incomplete = 0
    n_switches_complete = 0
    switches_per_opp: list[float] = []

    for opp, opp_rows in by_opp.items():
        # Find switch starts: first row with a given switch_id >= 0 after previous.
        by_sid: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for r in opp_rows:
            sid = int(r["switch_id"])
            if sid >= 0:
                by_sid[sid].append(r)
        complete = 0
        for sid in sorted(by_sid):
            window = by_sid[sid]
            # Encounter-time tau is stored in "tau"; take first H_e post-switch encounters.
            post = [r for r in window if 0 <= int(r["tau"]) < int(h_e)]
            if len(post) < int(h_e):
                n_incomplete += 1
                continue
            post = post[: int(h_e)]
            psr_s = float(sum(float(r["psr_gap"]) for r in post))
            dpr_s = float(sum(float(r["dpr_gap"]) for r in post if r.get("dpr_gap") != ""))
            switch_psr.append(psr_s)
            switch_dpr.append(dpr_s)
            n_switches_complete += 1
            complete += 1
            # Recovery: first encounter where belief mode matches true type (if available).
            rec = float(h_e)
            for i, r in enumerate(post):
                belief_raw = r.get("belief", "")
                if not belief_raw:
                    break
                try:
                    belief = np.asarray(json.loads(belief_raw), dtype=np.float64)
                    if int(np.argmax(belief)) == int(r["true_type"]):
                        rec = float(i)
                        break
                except (json.JSONDecodeError, TypeError, ValueError):
                    break
            recovery_encounters.append(rec)
        switches_per_opp.append(float(complete))

    enc = Counter(int(r["opponent_id"]) for r in rows)
    j = int(num_opponents)
    enc_counts = [enc.get(opp, 0) for opp in opponent_ids(j)]

    n_sw = max(n_switches_complete, 1)
    return {
        "psr_enc_per_switch": float(np.mean(switch_psr)) if switch_psr else float("nan"),
        "psr_enc_per_encounter": (
            float(np.mean([s / float(h_e) for s in switch_psr])) if switch_psr else float("nan")
        ),
        "dpr_enc_per_switch": float(np.mean(switch_dpr)) if switch_dpr else float("nan"),
        "recovery_encounters_mean": float(np.mean(recovery_encounters)) if recovery_encounters else float("nan"),
        "n_evaluated_switches": float(n_switches_complete),
        "n_incomplete_switches": float(n_incomplete),
        "switches_per_opponent_mean": float(np.mean(switches_per_opp)) if switches_per_opp else float("nan"),
        "mean_encounters_per_opponent": float(np.mean(enc_counts)),
        "enc_p05": float(np.percentile(enc_counts, 5)) if enc_counts else float("nan"),
        "enc_p50": float(np.percentile(enc_counts, 50)) if enc_counts else float("nan"),
        "enc_p95": float(np.percentile(enc_counts, 95)) if enc_counts else float("nan"),
        "frac_opponents_never_observed": float(np.mean([c == 0 for c in enc_counts])),
        # Alias for ladder summarizer
        "psr_per_switch": float(np.mean(switch_psr)) if switch_psr else float("nan"),
        "psr": float(np.sum(switch_psr)) if switch_psr else float("nan"),
        "n_psr_window_rows": float(n_switches_complete * h_e),
        "psr_per_window_row": (
            float(np.mean([s / float(h_e) for s in switch_psr])) if switch_psr else float("nan")
        ),
        "mean_encounters_between_opponent_switches": float(ENC_CYCLE),
    }


def run_encounter_clock_cell(
    *,
    spec: GameSpec,
    seed: int,
    num_opponents: int,
) -> dict[str, Any]:
    task = game_spec_to_task(spec)
    j = int(num_opponents)
    horizon = encounter_clock_horizon(j)
    t0 = time.perf_counter()
    result = run_categorical_cell(
        task=task,
        method="ESL",
        seed=int(seed),
        horizon=horizon,
        num_actions=NUM_ACTIONS,
        num_opponents=j,
        switch_clock="encounter",
        p_obs=1.0,
        obs_noise=0.0,
    )
    wall = time.perf_counter() - t0
    diag = compute_encounter_psr_diagnostics(result["trajectory"], num_opponents=j)
    return {
        **result["summary"],
        **diag,
        "match_mode": "encounter_clock",
        "switch_clock": "encounter",
        "horizon": horizon,
        "enc_pre_switch": ENC_PRE_SWITCH,
        "enc_h_e": ENC_H_E,
        "enc_n_switches": ENC_N_SWITCHES,
        "enc_per_opponent_target": ENC_PER_OPPONENT,
        "enc_switch_points": json.dumps(list(ENC_SWITCH_POINTS)),
        "family": spec.family,
        "instance_id": spec.instance_id,
        "wall_time_sec": round(wall, 3),
        # placeholders for summarize_diagnostics protocol fields
        "switch_interval": ENC_CYCLE,
        "switch_hazard": 0.0,
        "mean_encounters_between_opponent_switches": float(ENC_CYCLE),
    }


def run_encounter_clock_grid(
    *,
    num_opponents: int,
    out_dir: Path,
    seeds: tuple[int, ...] = SEEDS,
    resume: bool = True,
) -> Path:
    j = int(num_opponents)
    specs = load_suite(num_actions=NUM_ACTIONS)
    if not specs:
        raise RuntimeError("No 3×3 benchmark games found")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"J{j}_encounter_clock_diagnostics.csv"

    existing: dict[tuple[str, int], dict[str, Any]] = {}
    if resume and csv_path.is_file():
        with csv_path.open(newline="") as f:
            for r in csv.DictReader(f):
                existing[(str(r["instance_id"]), int(r["seed"]))] = r

    rows_out: list[dict[str, Any]] = []
    total = len(specs) * len(seeds)
    done0 = len(existing)
    print(f"[encounter J={j}] resume={resume}: {done0}/{total} done", flush=True)

    for spec in specs:
        for seed in seeds:
            key = (spec.instance_id, int(seed))
            if key in existing:
                rows_out.append(existing[key])
                continue
            row = run_encounter_clock_cell(spec=spec, seed=int(seed), num_opponents=j)
            rows_out.append(row)
            ordered = sorted(
                rows_out,
                key=lambda r: (str(r["family"]), str(r["instance_id"]), int(r["seed"])),
            )
            fields = list(ordered[0].keys())
            tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
            with tmp.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(ordered)
            tmp.replace(csv_path)
            print(
                f"[encounter J={j}] {len(rows_out)}/{total} "
                f"{spec.instance_id} seed={seed} "
                f"psr_enc/switch={row['psr_enc_per_switch']:.3f} "
                f"wall={row['wall_time_sec']}s",
                flush=True,
            )

    # Final rewrite sorted
    rows_out = sorted(rows_out, key=lambda r: (str(r["family"]), str(r["instance_id"]), int(r["seed"])))
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    return csv_path


def summarize_encounter_ladder(*, out_dir: Path) -> dict[str, Any]:
    metric = "psr_enc_per_switch"
    paths = {j: out_dir / f"J{j}_encounter_clock_diagnostics.csv" for j in J_LADDER}
    for p in paths.values():
        if not p.is_file():
            raise FileNotFoundError(p)

    summaries = []
    for j in J_LADDER:
        # summarize_diagnostics expects psr_per_switch; we aliased it
        s = summarize_diagnostics(paths[j], metric="psr_per_switch")
        s["primary_metric_name"] = metric
        # overwrite cell mean from psr_enc if present
        import csv as csv_mod

        with open(paths[j]) as f:
            rows = list(csv_mod.DictReader(f))
        vals = np.array([float(r["psr_enc_per_switch"]) for r in rows], dtype=np.float64)
        s["cell_mean"] = float(np.mean(vals))
        s["hier_boot"] = _hier_from_rows(rows, metric="psr_enc_per_switch")
        summaries.append(s)

    contrasts = {
        "J10_minus_J3": compute_matched_game_seed_contrast(
            paths[3], paths[10], contrast="J10_enc minus J3_enc", metric="psr_enc_per_switch"
        ),
        "J100_minus_J3": compute_matched_game_seed_contrast(
            paths[3], paths[100], contrast="J100_enc minus J3_enc", metric="psr_enc_per_switch"
        ),
        "J100_minus_J10": compute_matched_game_seed_contrast(
            paths[10], paths[100], contrast="J100_enc minus J10_enc", metric="psr_enc_per_switch"
        ),
    }

    for out_path in (
        out_dir / "encounter_clock_psr_per_switch.pdf",
        Path("manuscript_bundle/main/figures/encounter_clock_psr_per_switch.pdf"),
        Path("outputs/res/eval/expanded_adaptation/figures/encounter_clock_psr_per_switch.pdf"),
    ):
        plot_opponent_count_ladder(
            summaries,
            out_path=out_path,
            ylabel="Mean PSR per switch (encounter-time H_e) ↓",
        )

    report = {
        "protocol": "encounter_clock",
        "locked": {
            "pre_switch": ENC_PRE_SWITCH,
            "h_e": ENC_H_E,
            "n_switches": ENC_N_SWITCHES,
            "encounters_per_opponent": ENC_PER_OPPONENT,
            "switch_points": list(ENC_SWITCH_POINTS),
        },
        "claim": (
            "Holding per-opponent evidence, switch frequency, and post-switch evaluation "
            "exposure fixed, we evaluate the effect of increasing only the number of "
            "simultaneously tracked opponents."
        ),
        "summaries": {f"J{j}": s for j, s in zip(J_LADDER, summaries)},
        "matched_contrasts": contrasts,
        "paths": {f"J{j}": str(paths[j]) for j in J_LADDER},
    }
    (out_dir / "encounter_clock_analysis.json").write_text(json.dumps(report, indent=2) + "\n")

    tex = [
        "\\begin{table}[t]\\centering\\small",
        "\\caption{Matched game--seed contrasts in encounter-time PSR per switch "
        "(H$_e{=}30$; $n{=}120$ cells). Instance-clustered 95\\% CIs.}",
        "\\begin{tabular}{lcc}\\toprule",
        "Contrast & Mean difference & 95\\% CI \\\\",
        "\\midrule",
    ]
    for key in ("J10_minus_J3", "J100_minus_J3", "J100_minus_J10"):
        c = contrasts[key]
        ci = c["hier_boot_ci95_instance"]
        tex.append(f"{c['contrast'].replace('_', ' ')} & {c['mean_delta']:+.3f} & [{ci[0]:+.3f}, {ci[1]:+.3f}] \\\\")
    tex.append("\\bottomrule\\end{tabular}\\end{table}")
    (out_dir / "encounter_clock_contrasts.tex").write_text("\n".join(tex) + "\n")
    return report


def _hier_from_rows(rows: list[dict[str, Any]], *, metric: str, B: int = 4000, seed: int = 0) -> dict[str, float]:
    by_inst: dict[str, np.ndarray] = {}
    tmp: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        tmp[str(r["instance_id"])].append(float(r[metric]))
    for k, v in tmp.items():
        by_inst[k] = np.asarray(v, dtype=np.float64)
    instances = sorted(by_inst)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(B):
        drawn = rng.choice(instances, size=len(instances), replace=True)
        parts = []
        for i in drawn:
            vals = by_inst[i]
            parts.extend(vals[rng.integers(0, len(vals), size=len(vals))].tolist())
        boots.append(float(np.mean(parts)))
    boots.sort()
    mean = float(np.mean([float(r[metric]) for r in rows]))
    return {"mean": mean, "ci_lo": boots[int(0.025 * B)], "ci_hi": boots[int(0.975 * B)]}


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Encounter-clock opponent-count sensitivity")
    sub = p.add_subparsers(dest="cmd")
    p_run = sub.add_parser("run", help="Run encounter-clock grid for one J")
    p_run.add_argument("--J", type=int, required=True)
    p_run.add_argument("--out-dir", type=Path, default=Path("outputs/opponent_count"))
    p_run.add_argument("--no-resume", action="store_true")
    sub.add_parser("run-all", help="Run J=3,10,100 then summarize")
    sub.add_parser("summarize", help="Summarize existing encounter-clock CSVs")
    args = p.parse_args()
    out = Path("outputs/opponent_count")
    if args.cmd == "run":
        path = run_encounter_clock_grid(
            num_opponents=int(args.J), out_dir=args.out_dir, resume=not args.no_resume
        )
        print(f"Grid → {path}")
    elif args.cmd == "run-all":
        for j in J_LADDER:
            run_encounter_clock_grid(num_opponents=j, out_dir=out, resume=True)
        report = summarize_encounter_ladder(out_dir=out)
        print(json.dumps({k: report["summaries"][k]["hier_boot"] for k in report["summaries"]}, indent=2))
        print(json.dumps(report["matched_contrasts"], indent=2))
    elif args.cmd == "summarize":
        report = summarize_encounter_ladder(out_dir=out)
        print(json.dumps({k: report["summaries"][k]["hier_boot"] for k in report["summaries"]}, indent=2))
        print(json.dumps(report["matched_contrasts"], indent=2))
    else:
        p.print_help()


if __name__ == "__main__":
    main()
