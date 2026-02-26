#!/usr/bin/env python3
"""
44c_plot_h3_residualized_null.py

Plots the residualized permutation null summary for:
- tau(age_resid, phi_resid)
- tau(age_resid, ripeness_resid)

Reads:
  /mnt/g/STAR_HPV/results/ripeness/h3_gyro_age_permutation_null_residualized.yaml

Writes:
  /mnt/g/STAR_HPV/results/figures/h3_null_plots/residualized_null_summary_*.png
  /mnt/g/STAR_HPV/results/figures/h3_null_plots/figures_h3_residualized_null.tex
"""

from __future__ import annotations

import os
import argparse
from typing import Dict, Any

import yaml
import matplotlib.pyplot as plt


DEFAULT_IN = "/mnt/g/STAR_HPV/results/ripeness/h3_gyro_age_permutation_null_residualized.yaml"
DEFAULT_OUTDIR = "/mnt/g/STAR_HPV/results/figures/h3_null_plots"


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def plot_tau_summary(null_stats: Dict[str, Any], observed: float, title: str, out_png: str) -> None:
    vmin = float(null_stats["min"])
    p50 = float(null_stats["p50"])
    p95 = float(null_stats["p95"])
    vmax = float(null_stats["max"])

    plt.figure(figsize=(7, 2.2))
    plt.hlines(0.5, vmin, vmax, linewidth=6, alpha=0.25)
    for x, lab in [(vmin, "min"), (p50, "p50"), (p95, "p95"), (vmax, "max")]:
        plt.plot([x], [0.5], marker="o")
        plt.text(x, 0.62, lab, ha="center", va="bottom", fontsize=9)

    plt.axvline(observed, linewidth=2)
    plt.text(observed, 0.18, "observed", rotation=90, va="bottom", ha="right", fontsize=9)

    plt.yticks([])
    plt.xlabel("Kendall τ")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_IN)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    args = ap.parse_args()

    ensure_dir(args.outdir)

    y = load_yaml(args.input)
    obs = y["observed"]
    nul = y["null"]

    figs = {}
    figs["tau_age_phi_resid"] = os.path.join(args.outdir, "residualized_null_tau_age_phi.png")
    plot_tau_summary(
        nul["tau_age_phi_resid"],
        float(obs["kendall_tau_age_phi_resid"]),
        title="Residualized null: τ(age_resid, φ_resid)",
        out_png=figs["tau_age_phi_resid"],
    )

    figs["tau_age_ripeness_resid"] = os.path.join(args.outdir, "residualized_null_tau_age_ripeness.png")
    plot_tau_summary(
        nul["tau_age_ripeness_resid"],
        float(obs["kendall_tau_age_ripeness_resid"]),
        title="Residualized null: τ(age_resid, ripeness_resid)",
        out_png=figs["tau_age_ripeness_resid"],
    )

    tex = os.path.join(args.outdir, "figures_h3_residualized_null.tex")
    with open(tex, "w") as f:
        f.write("\n".join([
            r"\begin{figure}[t]",
            r"\centering",
            rf"\includegraphics[width=0.80\textwidth]{{{figs['tau_age_phi_resid']}}}",
            r"\caption{Residualized permutation null summary for $\tau(t',\Phi')$, where $t'$ and $\Phi'$ are bin-median residuals within $T_{\rm eff}$ bins.}",
            r"\end{figure}",
            "",
            r"\begin{figure}[t]",
            r"\centering",
            rf"\includegraphics[width=0.80\textwidth]{{{figs['tau_age_ripeness_resid']}}}",
            r"\caption{Residualized permutation null summary for $\tau(t',R'^{\rm time}_{\rm ripe})$ with $R'^{\rm time}_{\rm ripe}=t'\Phi'$.}",
            r"\end{figure}",
            "",
        ]))

    print("saved:", tex)
    for k, v in figs.items():
        print(" ", k, "->", v)


if __name__ == "__main__":
    main()
