#!/usr/bin/env python3
"""
H3 figure pack: gyro ages + age vs ripeness_time.

Outputs:
  /mnt/g/STAR_HPV/results/figures/h3_gyro/
    gyro_age_hist_overall.png
    gyro_age_hist_<bin>.png
    age_vs_ripeness_pooled.png
    age_vs_ripeness_<bin>.png
    figures_h3_gyro.tex
    figures_h3_gyro.yaml
"""

from __future__ import annotations

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INP = "/mnt/g/STAR_HPV/processed/gyro/kepler_gmmgate_with_gyro_ages_mamajek.parquet"
OUT_DIR = "/mnt/g/STAR_HPV/results/figures/h3_gyro"
OUT_TEX = os.path.join(OUT_DIR, "figures_h3_gyro.tex")
OUT_YAML = os.path.join(OUT_DIR, "figures_h3_gyro.yaml")

TEFF_BINS = ["3200-4000", "4000-5200", "5200-6000", "6000-7500"]


def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def clean(x):
    return pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def save_hist(x, path, title, xlabel):
    x = clean(x)
    plt.figure()
    plt.hist(x.values, bins=40)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_scatter(x, y, path, title, xlabel, ylabel):
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    m = x.notna() & y.notna()
    plt.figure()
    plt.scatter(x[m].values, y[m].values, s=8, alpha=0.6)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_parquet(INP)

    age_col = pick_col(df, ["age_gyr_gyro", "age_gyr"])
    phi_col = pick_col(df, ["Phi_gmm", "Phi"])
    bin_col = pick_col(df, ["teff_bin_s", "teff_bin"])

    if not all([age_col, phi_col, bin_col]):
        raise RuntimeError(f"Missing columns: age={age_col} phi={phi_col} bin={bin_col}")

    df["ripeness_time"] = pd.to_numeric(df[phi_col], errors="coerce") * pd.to_numeric(df[age_col], errors="coerce")

    figs = {"overall": {}, "per_bin": {}}

    # Age hist overall
    p = os.path.join(OUT_DIR, "gyro_age_hist_overall.png")
    save_hist(df[age_col], p, "Gyro ages (Mamajek) for GMM-gated sample", "age_gyr_gyro")
    figs["overall"]["age_hist"] = p

    # Age vs ripeness pooled
    p = os.path.join(OUT_DIR, "age_vs_ripeness_pooled.png")
    save_scatter(df[age_col], df["ripeness_time"], p, "Gyro age vs time-weighted ripeness (pooled)", "age_gyr_gyro", "ripeness_time = Phi_gmm * age_gyr_gyro")
    figs["overall"]["age_vs_ripeness"] = p

    # Per-bin
    for b in TEFF_BINS:
        sub = df[df[bin_col].astype("string") == b]
        if len(sub) == 0:
            continue

        p1 = os.path.join(OUT_DIR, f"gyro_age_hist_{b.replace('-','_')}.png")
        save_hist(sub[age_col], p1, f"Gyro ages (Mamajek), Teff bin {b}", "age_gyr_gyro")

        p2 = os.path.join(OUT_DIR, f"age_vs_ripeness_{b.replace('-','_')}.png")
        save_scatter(sub[age_col], sub["ripeness_time"], p2, f"Gyro age vs ripeness_time, Teff bin {b}", "age_gyr_gyro", "ripeness_time")

        figs["per_bin"][b] = {"age_hist": p1, "age_vs_ripeness": p2}

    # LaTeX snippet
    tex_lines = []
    tex_lines.append(r"\begin{figure*}[t]")
    tex_lines.append(r"\centering")
    tex_lines.append(r"\includegraphics[width=0.72\textwidth]{results/figures/h3_gyro/gyro_age_hist_overall.png}")
    tex_lines.append(r"\caption{Gyrochronology ages (Mamajek \& Hillenbrand style) for the Kepler--Gaia GMM-gated dwarf sample.}")
    tex_lines.append(r"\label{fig:gyro_age_hist_overall}")
    tex_lines.append(r"\end{figure*}")
    tex_lines.append("")
    tex_lines.append(r"\begin{figure*}[t]")
    tex_lines.append(r"\centering")
    tex_lines.append(r"\includegraphics[width=0.72\textwidth]{results/figures/h3_gyro/age_vs_ripeness_pooled.png}")
    tex_lines.append(r"\caption{Gyro age versus time-weighted ripeness $R^{\rm ripe}_{\rm time}=\Phi_{\rm gmm}t_{\rm gyro}$ (pooled).}")
    tex_lines.append(r"\label{fig:age_vs_ripeness_pooled}")
    tex_lines.append(r"\end{figure*}")
    tex_lines.append("")
    tex_lines.append(r"% Per-bin panels")
    for b in TEFF_BINS:
        k = b.replace("-", "_")
        tex_lines.append(r"\begin{figure*}[t]")
        tex_lines.append(r"\centering")
        tex_lines.append(rf"\includegraphics[width=0.49\textwidth]{{results/figures/h3_gyro/gyro_age_hist_{k}.png}}"
                         rf"\includegraphics[width=0.49\textwidth]{{results/figures/h3_gyro/age_vs_ripeness_{k}.png}}")
        tex_lines.append(rf"\caption{{Gyro-age distribution (left) and gyro-age versus $R^{{\rm ripe}}_{{\rm time}}$ (right) for $T_{{\rm eff}}$ bin {b}.}}")
        tex_lines.append(rf"\label{{fig:h3_gyro_{k}}}")
        tex_lines.append(r"\end{figure*}")
        tex_lines.append("")

    with open(OUT_TEX, "w") as f:
        f.write("\n".join(tex_lines))

    with open(OUT_YAML, "w") as f:
        yaml.safe_dump(
            {"input": INP, "age_col": age_col, "phi_col": phi_col, "bin_col": bin_col, "figures": figs, "latex": OUT_TEX},
            f,
            sort_keys=False,
        )

    print("saved figures to:", OUT_DIR)
    print("saved LaTeX snippet:", OUT_TEX)
    print("saved YAML:", OUT_YAML)


if __name__ == "__main__":
    main()
