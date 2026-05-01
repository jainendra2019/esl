"""Paired significance tests for manuscript experiment summaries."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

OUTPUT_COLUMNS: tuple[str, ...] = (
    "comparison",
    "metric",
    "n_pairs",
    "mean_esl",
    "mean_baseline",
    "mean_diff",
    "ci95_low",
    "ci95_high",
    "t_stat",
    "p_value",
    "cohen_dz",
)

ESL_CANONICAL = "ESL"
FIXEDK_CANONICAL = "Fixed-K"
METHOD_ALIASES = {
    "esl": ESL_CANONICAL,
    "fixed-k": FIXEDK_CANONICAL,
    "fixed-k bayesian": FIXEDK_CANONICAL,
    "fixedk": FIXEDK_CANONICAL,
    "fixed_k": FIXEDK_CANONICAL,
    "fixed k": FIXEDK_CANONICAL,
}


def canonical_method(method: str) -> str:
    key = " ".join(str(method).strip().replace("_", " ").split()).lower()
    key = key.replace("fixed k", "fixed-k")
    return METHOD_ALIASES.get(key, str(method).strip())


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def _as_int(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def _as_float(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not np.isfinite(value):
        raise ValueError(f"non-finite {key} in row: {row}")
    return value


def load_figure1_seed_pairs(
    summary_csv: Path,
    *,
    metric: str = "psr",
    baseline_method: str = FIXEDK_CANONICAL,
) -> list[dict[str, float]]:
    """Load matched seed-level ESL/baseline pairs after averaging over tasks."""
    rows = _read_csv(Path(summary_csv))
    baseline = canonical_method(baseline_method)
    wanted = {ESL_CANONICAL, baseline}
    filtered: list[dict[str, Any]] = []
    for row in rows:
        method = canonical_method(row.get("method", ""))
        if method not in wanted:
            continue
        if metric not in row:
            raise ValueError(f"missing metric column `{metric}`")
        parsed = dict(row)
        parsed["method"] = method
        parsed["seed"] = _as_int(row, "seed")
        parsed["task"] = str(row.get("task", "")).strip()
        parsed["horizon"] = _as_int(row, "horizon")
        parsed["h_window"] = _as_int(row, "h_window")
        parsed["focal_agent_id"] = _as_int(row, "focal_agent_id")
        parsed[metric] = _as_float(row, metric)
        filtered.append(parsed)

    filtered.sort(key=lambda r: (int(r["seed"]), str(r["task"]), str(r["method"])))
    if not filtered:
        raise ValueError("no ESL/Fixed-K rows found")

    by_key: dict[tuple[int, str, str], dict[str, Any]] = {}
    for row in filtered:
        key = (int(row["seed"]), str(row["task"]), str(row["method"]))
        if key in by_key:
            raise ValueError(f"duplicate row for seed/task/method: {key}")
        by_key[key] = row

    seed_task_keys = sorted({(int(r["seed"]), str(r["task"])) for r in filtered})
    by_seed: dict[int, dict[str, list[float]]] = {}
    for seed, task in seed_task_keys:
        esl = by_key.get((seed, task, ESL_CANONICAL))
        base = by_key.get((seed, task, baseline))
        if esl is None or base is None:
            raise ValueError(f"missing matched method row for seed={seed}, task={task}")
        for field in ("horizon", "h_window", "focal_agent_id"):
            if int(esl[field]) != int(base[field]):
                raise ValueError(f"mismatched {field} for seed={seed}, task={task}")
        bucket = by_seed.setdefault(seed, {"esl": [], "baseline": []})
        bucket["esl"].append(float(esl[metric]))
        bucket["baseline"].append(float(base[metric]))

    pairs: list[dict[str, float]] = []
    expected_tasks: set[str] | None = None
    for seed in sorted(by_seed):
        tasks = {task for s, task in seed_task_keys if s == seed}
        if expected_tasks is None:
            expected_tasks = tasks
        elif tasks != expected_tasks:
            raise ValueError(f"mismatched task set for seed={seed}")
        esl_vals = by_seed[seed]["esl"]
        base_vals = by_seed[seed]["baseline"]
        if len(esl_vals) != len(base_vals):
            raise ValueError(f"unbalanced task count for seed={seed}")
        pairs.append(
            {
                "seed": float(seed),
                "psr_esl": float(np.mean(esl_vals)),
                "psr_baseline": float(np.mean(base_vals)),
                "diff": float(np.mean(base_vals) - np.mean(esl_vals)),
            }
        )

    if len(pairs) < 3:
        raise ValueError("paired test requires at least 3 matched seeds")
    return pairs


def paired_ttest_summary(pairs: list[dict[str, float]], *, comparison: str = "ESL vs Fixed-K", metric: str = "PSR") -> dict[str, float | str]:
    diffs = np.asarray([float(row["diff"]) for row in pairs], dtype=np.float64)
    esl = np.asarray([float(row["psr_esl"]) for row in pairs], dtype=np.float64)
    baseline = np.asarray([float(row["psr_baseline"]) for row in pairs], dtype=np.float64)
    if diffs.size < 3:
        raise ValueError("paired test requires at least 3 matched seeds")
    if not (np.isfinite(diffs).all() and np.isfinite(esl).all() and np.isfinite(baseline).all()):
        raise ValueError("paired values must be finite")
    n = int(diffs.size)
    mean_diff = float(diffs.mean())
    if mean_diff <= 0.0:
        raise ValueError("mean paired difference must be positive for this configured manuscript comparison")
    mean_esl = float(esl.mean())
    mean_baseline = float(baseline.mean())
    std_diff = float(diffs.std(ddof=1))
    if math.isclose(std_diff, 0.0, abs_tol=1e-12):
        if math.isclose(mean_diff, 0.0, abs_tol=1e-12):
            t_stat = 0.0
            p_value = 1.0
            cohen_dz = 0.0
        else:
            t_stat = math.copysign(float("inf"), mean_diff)
            p_value = 0.0
            cohen_dz = math.copysign(float("inf"), mean_diff)
        ci_low = mean_diff
        ci_high = mean_diff
    else:
        sem = std_diff / math.sqrt(n)
        half = float(stats.t.ppf(0.975, n - 1) * sem)
        ci_low = mean_diff - half
        ci_high = mean_diff + half
        t_res = stats.ttest_1samp(diffs, 0.0)
        t_stat = float(t_res.statistic)
        p_value = float(t_res.pvalue)
        cohen_dz = mean_diff / std_diff
    return {
        "comparison": comparison,
        "metric": metric,
        "n_pairs": float(n),
        "mean_esl": mean_esl,
        "mean_baseline": mean_baseline,
        "mean_diff": mean_diff,
        "paired_std_diff": std_diff,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "t_stat": t_stat,
        "p_value": p_value,
        "cohen_dz": cohen_dz,
    }


def _format_p(p_value: float) -> str:
    if p_value < 0.001:
        return "p < 0.001"
    return f"p = {p_value:.3f}"


def _effect_size_label(dz: float) -> str:
    adz = abs(float(dz))
    if adz >= 0.8:
        return "large"
    if adz >= 0.5:
        return "medium"
    if adz >= 0.2:
        return "small"
    return "negligible"


def write_figure1_significance(
    *,
    summary_csv: Path,
    manuscript_bundle: Path,
    table_path: Path | None = None,
    report_path: Path | None = None,
) -> dict[str, Path]:
    pairs = load_figure1_seed_pairs(summary_csv)
    result = paired_ttest_summary(pairs)
    mb = Path(manuscript_bundle)
    table = table_path or mb / "main" / "tables" / "figure1_significance.csv"
    report = report_path or mb / "reports" / "figure1_significance_report.md"
    _write_csv(table, [result], OUTPUT_COLUMNS)
    report.parent.mkdir(parents=True, exist_ok=True)
    p_value = float(result["p_value"])
    if p_value < 0.05:
        manuscript_sentence = (
            "Across matched seeds, ESL significantly reduces PSR relative to the strongest baseline, Fixed-K "
            f"(paired t-test, mean paired reduction = {float(result['mean_diff']):.2f}, "
            f"95% CI [{float(result['ci95_low']):.2f}, {float(result['ci95_high']):.2f}], "
            f"{_format_p(p_value)})."
        )
    else:
        manuscript_sentence = (
            "Across matched seeds, ESL reduces mean PSR relative to Fixed-K by "
            f"{float(result['mean_diff']):.2f}, with a paired 95% CI of "
            f"[{float(result['ci95_low']):.2f}, {float(result['ci95_high']):.2f}]."
        )
    lines = [
        "# Figure 1 Paired Significance",
        "",
        "## Protocol",
        f"- Source: `{summary_csv}`.",
        "- Comparison: ESL vs Fixed-K.",
        "- Metric: PSR.",
        "- Unit of replication: matched seed after averaging PSR across games; timesteps and post-switch rows are not treated as independent samples.",
        "- Test: paired two-sided t-test implemented as a one-sample t-test on paired differences `d_s = PSR_Fixed-K(s) - PSR_ESL(s)`.",
        "",
        "## Results",
        f"- n pairs: `{int(result['n_pairs'])}`.",
        f"- mean ESL PSR: `{float(result['mean_esl']):.4f}`.",
        f"- mean Fixed-K PSR: `{float(result['mean_baseline']):.4f}`.",
        f"- mean paired reduction: `{float(result['mean_diff']):.4f}`.",
        f"- 95% CI: [`{float(result['ci95_low']):.4f}`, `{float(result['ci95_high']):.4f}`].",
        f"- t-statistic: `{float(result['t_stat']):.4f}`.",
        f"- p-value: `{p_value:.6g}`.",
        f"- Cohen's dz: `{float(result['cohen_dz']):.4f}` ({_effect_size_label(float(result['cohen_dz']))}; thresholds: small ~0.2, medium ~0.5, large ~0.8).",
        "",
        "## Manuscript Sentence",
        manuscript_sentence,
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"table": table, "report": report}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Paired significance tests for existing Figure 1 summaries")
    parser.add_argument("--figure1-summary", type=Path, default=Path("runs/figure1_dynamic_adaptation/figure1_dynamic_run_summaries.csv"))
    parser.add_argument("--manuscript-bundle", type=Path, default=Path("manuscript_bundle"))
    args = parser.parse_args(argv)
    outputs = write_figure1_significance(summary_csv=args.figure1_summary, manuscript_bundle=args.manuscript_bundle)
    for key, path in outputs.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
