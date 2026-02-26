#!/usr/bin/env python3
"""
Plot APOKASC age distributions for any Kepler/Gaia products with non-zero overlap.

Outputs:
- /mnt/g/STAR_HPV/results/figures/apokasc_age_overlap/*.png
- figures_apokasc_age_overlap.tex
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INFILES = {
    "kepler_gmmgate": "/mnt/g/STAR_HPV/processed/ages_merged/kepler_gmmgate_with_apokasc3_ages.parquet",
    "anchor_window": "/mnt/g/STAR_HPV/processed/ages_merged/anchor_window_with_apokasc3_ages.parquet",
    "tail_global_postgate_q90": "/mnt/g/STAR_HPV/processed/ages_merged/tail_global_postgate_q90_with_apokasc3_ages.parquet",
    "tail_global_postgate_q95": "/mnt/g/STAR_HPV/processed/ages_merged/tail_global_postgate_q95_with_apokasc3_ages.parquet",
    "tail_window_perbin_q90": "/mnt/g/STAR_HPV/processed/ages_merged/tail_window_perbin_q90_with_apokasc3_ages.parquet",
    "tail_window_perbin_q95": "/mnt/g/STAR_HPV/processed/ages_merged/tail_window_perbin_q95_with_apokasc3_ages.parquet",
}

OUTDIR = "/mnt/g/STAR_HPV/results/figures/apokasc_age_overlap"
TEXOUT = os.path.join(OUTDIR, "figures_apokasc_age_overlap.tex")

def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    tex_lines = []
    made = 0

    for name, path in INFILES.items():
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path)
        if "age_gyr" not in df.columns:
            continue

        a = pd.to_numeric(df["age_gyr"], errors="coerce").astype(float)
        a = a[np.isfinite(a)]
        if len(a) < 10:
            continue

        figpath = os.path.join(OUTDIR, f"age_hist_{name}.png")
        plt.figure()
        plt.hist(a.values, bins=25)
        plt.xlabel("Age (Gyr)")
        plt.ylabel("Count")
        plt.title(f"APOKASC age overlap: {name} (N={len(a)})")
        plt.tight_layout()
        plt.savefig(figpath, dpi=200)
        plt.close()
        made += 1

        tex_lines.append(r"\begin{figure}[t]")
        tex_lines.append(r"\centering")
        tex_lines.append(rf"\includegraphics[width=0.78\linewidth]{{{figpath}}}")
        tex_lines.append(rf"\caption{{APOKASC-3 age distribution for overlapping subset: \texttt{{{name}}}.}}")
        tex_lines.append(rf"\label{{fig:apokasc_age_{name}}}")
        tex_lines.append(r"\end{figure}")
        tex_lines.append("")

    with open(TEXOUT, "w") as f:
        f.write("\n".join(tex_lines))

    print(f"saved figures to: {OUTDIR}")
    print(f"saved LaTeX snippet: {TEXOUT}")
    print("figures made:", made)

if __name__ == "__main__":
    main()
