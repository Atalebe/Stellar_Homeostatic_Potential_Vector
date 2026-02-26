#!/usr/bin/env python3
"""
44b_plot_h3_window_age_null.py

Plots the permutation null distributions for the H3 window-age test:
- pooled null distribution of delta_median
- per-bin null distributions of delta_median

Reads:
  /mnt/g/STAR_HPV/results/ripeness/h3_gyro_window_age_permutation_null.yaml

Writes:
  /mnt/g/STAR_HPV/results/figures/h3_null_plots/window_age_null_*.png
  /mnt/g/STAR_HPV/results/figures/h3_null_plots/figures_h3_null_plots.tex
  /mnt/g/STAR_HPV/results/figures/h3_null_plots/window_age_null_plots_summary.yaml
"""

from __future__ import annotations

import os
import argparse
from typing import Dict, Any

import numpy as np
import yaml
import matplotlib.pyplot as plt


DEFAULT_IN = "/mnt/g/STAR_HPV/results/ripeness/h3_gyro_window_age_permutation_null.yaml"
DEFAULT_OUTDIR = "/mnt/g/STAR_HPV/results/figures/h3_null_plots"


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def plot_null_hist(null_stats: Dict[str, Any], observed: float, title: str, xlabel: str, out_png: str) -> None:
    # null_stats: {n, min, p50, p95, max} from yaml summary, but we don't have samples.
    # So: we cannot reconstruct histogram. We need samples to plot.
    # HOWEVER: your script stored only summary stats, not the full null vector.
    # Workaround: plot a "summary bar" style: min/p50/p95/max + observed.
    n = null_stats.get("n", None)
    vmin = null_stats.get("min", None)
    p50 = null_stats.get("p50", None)
    p95 = null_stats.get("p95", None)
    vmax = null_stats.get("max", None)

    plt.figure(figsize=(7, 2.2))
    xs = []
    labels = []

    for val, lab in [(vmin, "min"), (p50, "p50"), (p95, "p95"), (vmax, "max")]:
        if val is None:
            continue
        xs.append(float(val))
        labels.append(lab)

    # draw a horizontal range line
    if vmin is not None and vmax is not None:
        plt.hlines(0.5, float(vmin), float(vmax), linewidth=6, alpha=0.25)

    # plot summary markers
    for x, lab in zip(xs, labels):
        plt.plot([x], [0.5], marker="o")
        plt.text(x, 0.62, lab, ha="center", va="bottom", fontsize=9)

    # observed
    plt.axvline(observed, linewidth=2)
    plt.text(observed, 0.18, "observed", rotation=90, va="bottom", ha="right", fontsize=9)

    plt.yticks([])
    plt.xlabel(xlabel)
    plt.title(title + (f" (n={n})" if n is not None else ""))
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

    # pooled
    pooled_obs = float(obs["delta_median_pooled"])
    pooled_null_stats = nul["delta_median_pooled"]
    figs["pooled"] = os.path.join(args.outdir, "window_age_null_pooled_summary.png")
    plot_null_hist(
        pooled_null_stats,
        pooled_obs,
        title="Permutation null: Δmedian(age|window - nonwindow), pooled",
        xlabel="Δmedian (Gyr)",
        out_png=figs["pooled"],
    )

    # per-bin: we only have p-values and observed deltas, but not null distributions.
    # We'll still plot the pooled null summary, and a per-bin "dot chart" of observed deltas with empirical p.
    bins = list(obs["delta_median_by_bin"].keys())
    deltas = [float(obs["delta_median_by_bin"][b]) for b in bins]
    pvals = [float(nul["p_empirical_one_sided_ge"]["by_bin"][b]) for b in bins]

    figs["perbin_dots"] = os.path.join(args.outdir, "window_age_delta_by_bin.png")
    plt.figure(figsize=(7, 4.2))
    yidx = np.arange(len(bins))[::-1]
    plt.scatter(deltas, yidx, s=60)
    for d, yi, b, p in zip(deltas, yidx, bins, pvals):
        plt.text(d, yi + 0.1, f"{b}  p={p:.4g}", fontsize=9)
    plt.axvline(0.0, linewidth=2, linestyle="--")
    plt.yticks(yidx, bins)
    plt.xlabel("Δmedian gyro age (Gyr): window - nonwindow")
    plt.title("Observed window-age shift by Teff bin (permutation p shown)")
    plt.tight_layout()
    plt.savefig(figs["perbin_dots"], dpi=160)
    plt.close()

    # LaTeX
    tex_path = os.path.join(args.outdir, "figures_h3_null_plots.tex")
    with open(tex_path, "w") as f:
        f.write(
            "\n".join([
                r"\begin{figure}[t]",
                r"\centering",
                rf"\includegraphics[width=0.70\textwidth]{{{figs['pooled']}}}",
                r"\caption{H3 permutation null (pooled): summary range of the null distribution for $\Delta\tilde{t}$ compared to the observed value.}",
                r"\end{figure}",
                "",
                r"\begin{figure}[t]",
                r"\centering",
                rf"\includegraphics[width=0.70\textwidth]{{{figs['perbin_dots']}}}",
                r"\caption{Observed $\Delta\tilde{t}$ by $T_{\rm eff}$ bin with empirical permutation p-values.}",
                r"\end{figure}",
                "",
            ])
        )

    # YAML summary
    summary = {
        "input": args.input,
        "figures": figs,
        "latex": tex_path,
        "note": "This plotter uses stored null summary statistics (min/p50/p95/max), not the full null vector.",
    }
    with open(os.path.join(args.outdir, "window_age_null_plots_summary.yaml"), "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    print("saved:", tex_path)
    print("figs:", figs)


if __name__ == "__main__":
    main()
