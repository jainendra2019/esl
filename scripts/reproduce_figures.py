#!/usr/bin/env python3
"""Write supporting evidence-derived plots from the frozen cell tables.

These are bar/line charts computed from ``evidence/``. They are not the exact
manuscript figure PDFs.

Does not rerun simulations.

Inputs:  evidence/
Outputs: outputs/figures/*.pdf and *.png
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
OUT = ROOT / "outputs" / "figures"

METHOD_ORDER = ("ESL", "SA-FP", "Fixed-K Bayesian", "Online EM", "PPO", "SOM", "FP")
COLORS = {
    "ESL": "#1f77b4",
    "SA-FP": "#17becf",
    "Fixed-K Bayesian": "#2ca02c",
    "Online EM": "#7f7f7f",
    "PPO": "#ff7f0e",
    "SOM": "#9467bd",
    "FP": "#8c564b",
    "No-Sharing": "#bcbd22",
    "Fixed-K": "#2ca02c",
    "No-Belief": "#d62728",
}


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=160)
    plt.close(fig)


def _se(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.std(ddof=1) / np.sqrt(arr.size))


def flagship_psr_bars() -> None:
    rows = load_csv(EVIDENCE / "flagship" / "cells.csv")
    means, ses, labels = [], [], []
    for m in METHOD_ORDER:
        vals = [float(r["psr"]) for r in rows if r["method"] == m]
        if not vals:
            continue
        labels.append(m)
        means.append(float(np.mean(vals)))
        ses.append(_se(vals))
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=ses, color=[COLORS.get(m, "#333") for m in labels], capsize=3)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("PSR (lower better)")
    ax.set_title("Twelve-game flagship PSR (n=120 cells/method)")
    _save(fig, "flagship_psr")


def k_sensitivity() -> None:
    rows = load_csv(EVIDENCE / "k_sensitivity" / "summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, metric, title in ((axes[0], "psr", "PSR"), (axes[1], "dpr", "DPR")):
        sub = [r for r in rows if r["metric"] == metric]
        sub.sort(key=lambda r: int(r["k_model"]))
        xs = [int(r["k_model"]) for r in sub]
        ys = [float(r["mean"]) for r in sub]
        lo = [float(r["ci95_low"]) for r in sub]
        hi = [float(r["ci95_high"]) for r in sub]
        yerr = np.vstack([np.asarray(ys) - np.asarray(lo), np.asarray(hi) - np.asarray(ys)])
        ax.errorbar(xs, ys, yerr=yerr, marker="o", color="#1f77b4")
        ax.set_xticks(xs)
        ax.set_xlabel("K")
        ax.set_ylabel(f"{title} (lower better)")
        ax.set_title(title)
    fig.suptitle("ESL prototype-count sensitivity")
    _save(fig, "k_sensitivity")


def omd_bars() -> None:
    rows = load_csv(EVIDENCE / "omd" / "omd_psr_summary.csv")
    primary = [r for r in rows if r["layer"] == "primary_held_out"]
    keep = ["ESL-decision-tuned", "Per-opponent OMD/MWU"]
    labels, means, ses = [], [], []
    for m in keep:
        r = next(x for x in primary if x["method"] == m)
        labels.append("ESL (λ=10)" if "ESL" in m else "OMD/MWU (η=1)")
        means.append(float(r["psr_mean"]))
        ses.append(float(r["psr_se"]))
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=ses, color=["#1f77b4", "#ff7f0e"], capsize=3)
    ax.set_xticks(x, labels)
    ax.set_ylabel("PSR (lower better)")
    ax.set_title("Held-out OMD comparison (n=84)")
    _save(fig, "omd_held_out")


def line_from_summary(path: Path, x_key: str, y_key: str, name: str, title: str, xlabel: str) -> None:
    if not path.is_file():
        return
    rows = load_csv(path)
    if not rows or x_key not in rows[0] or y_key not in rows[0]:
        # try common alternatives
        keys = list(rows[0].keys()) if rows else []
        if "mean_psr" in keys:
            y_key = "mean_psr"
        elif "PSR_mean" in keys:
            y_key = "PSR_mean"
        if x_key not in (rows[0] if rows else {}):
            for cand in ("lam_focal", "alpha", "batch_size", "M", "gamma"):
                if cand in keys:
                    x_key = cand
                    break
    xs, ys = [], []
    for r in rows:
        try:
            xs.append(float(r[x_key]))
            ys.append(float(r[y_key]))
        except (KeyError, ValueError):
            continue
    if not xs:
        return
    order = np.argsort(xs)
    xs = np.asarray(xs)[order]
    ys = np.asarray(ys)[order]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.plot(xs, ys, marker="o", color="#1f77b4")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("PSR (lower better)")
    ax.set_title(title)
    _save(fig, name)


def ablation_bars() -> None:
    path = EVIDENCE / "ablation" / "ablation_summary.csv"
    if not path.is_file():
        return
    rows = load_csv(path)
    labels = [r["Method"] for r in rows]
    means = [float(r["PSR_mean"]) for r in rows]
    ses = [float(r["PSR_se"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=ses, color=[COLORS.get(m, "#4c72b0") for m in labels], capsize=3)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("PSR (lower better)")
    ax.set_title("Matched-component ablations")
    _save(fig, "ablation_psr")


def rm_timeseries() -> None:
    path = EVIDENCE / "rm_diagnostic" / "timeseries.csv"
    if not path.is_file():
        return
    rows = load_csv(path)
    by_step: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        try:
            by_step[int(r["slow_step"])].append(float(r["prototype_step_size"]))
        except (KeyError, ValueError):
            continue
    if not by_step:
        return
    xs = sorted(by_step)
    ys = [float(np.mean(by_step[t])) for t in xs]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(xs, ys, color="#1f77b4")
    ax.set_xlabel("slow step")
    ax.set_ylabel("prototype step size")
    ax.set_title("Diminishing-step Robbins–Monro diagnostic")
    ax.set_yscale("log")
    _save(fig, "rm_diminishing_step")


def main() -> None:
    flagship_psr_bars()
    k_sensitivity()
    omd_bars()
    ablation_bars()
    line_from_summary(
        EVIDENCE / "lambda" / "summary.csv",
        "lam_focal",
        "mean_psr",
        "lambda_sensitivity",
        "Inverse-temperature sensitivity",
        r"$\lambda$",
    )
    line_from_summary(
        EVIDENCE / "alpha" / "alpha_summary.csv",
        "alpha",
        "mean_psr",
        "alpha_sensitivity",
        "Prototype-update coefficient sensitivity",
        r"$\alpha$",
    )
    line_from_summary(
        EVIDENCE / "batch" / "batch_summary.csv",
        "batch_size",
        "mean_psr",
        "batch_sensitivity",
        "Batch-size sensitivity",
        "M",
    )
    rm_timeseries()
    print(f"Wrote supporting evidence-derived plots under {OUT}")
    print("These are not the exact manuscript figure PDFs.")
    for p in sorted(OUT.glob("*")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
