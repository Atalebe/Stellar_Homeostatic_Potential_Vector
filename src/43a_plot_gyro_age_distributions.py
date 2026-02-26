#!/usr/bin/env python3
import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INP = "/mnt/g/STAR_HPV/processed/gyro/kepler_gmmgate_with_gyro_ages_mamajek.parquet"
OUTDIR = "/mnt/g/STAR_HPV/results/figures/gyro_ages"
TEX = os.path.join(OUTDIR, "figures_gyro_ages.tex")

BIN_ORDER = ["3200-4000", "4000-5200", "5200-6000", "6000-7500"]

def _qstats(x: pd.Series):
    x = pd.to_numeric(x, errors="coerce")
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0}
    return {
        "n": int(len(x)),
        "min": float(np.min(x)),
        "p10": float(np.quantile(x, 0.10)),
        "p50": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
        "max": float(np.max(x)),
    }

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    df = pd.read_parquet(INP)
    # tolerate either column name
    age_col = None
    for c in ["age_gyro_gyr", "age_gyr_gyro", "age_gyr", "gyro_age_gyr"]:
        if c in df.columns:
            age_col = c
            break
    if age_col is None:
        raise RuntimeError(f"No gyro age column found in {INP}. Columns: {df.columns.tolist()}")

    bin_col = "teff_bin_s" if "teff_bin_s" in df.columns else ("teff_bin" if "teff_bin" in df.columns else None)
    if bin_col is None:
        raise RuntimeError("No teff bin column found (expected teff_bin_s or teff_bin).")

    # Overall plot
    ages = pd.to_numeric(df[age_col], errors="coerce")
    ages = ages[np.isfinite(ages)]

    plt.figure()
    plt.hist(ages, bins=40)
    plt.xlabel("Gyro age (Gyr)")
    plt.ylabel("Count")
    plt.title("Gyro-age distribution (Kepler GMM-gated sample)")
    out_png = os.path.join(OUTDIR, "gyro_age_hist_overall.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    # Per-bin plots
    perbin_pngs = []
    for b in BIN_ORDER:
        m = df[bin_col].astype(str) == b
        x = pd.to_numeric(df.loc[m, age_col], errors="coerce")
        x = x[np.isfinite(x)]
        if len(x) < 5:
            continue
        plt.figure()
        plt.hist(x, bins=30)
        plt.xlabel("Gyro age (Gyr)")
        plt.ylabel("Count")
        plt.title(f"Gyro-age distribution ({b} K)")
        p = os.path.join(OUTDIR, f"gyro_age_hist_{b.replace('-','_')}.png")
        plt.tight_layout()
        plt.savefig(p, dpi=200)
        plt.close()
        perbin_pngs.append((b, p))

    # Summary yaml
    summary = {
        "input": INP,
        "age_col": age_col,
        "bin_col": bin_col,
        "overall": _qstats(pd.Series(ages)),
        "by_bin": {b: _qstats(df.loc[df[bin_col].astype(str) == b, age_col]) for b in BIN_ORDER},
        "figures": {
            "overall": out_png,
            "per_bin": {b: p for b, p in perbin_pngs},
        },
    }
    with open(os.path.join(OUTDIR, "gyro_age_plots_summary.yaml"), "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    # LaTeX snippet
    lines = []
    lines.append(r"\begin{figure*}")
    lines.append(r"\centering")
    lines.append(rf"\includegraphics[width=0.55\textwidth]{{{out_png}}}")
    lines.append(r"\caption{Gyrochronology age distribution for the Kepler--Gaia GMM-gated dwarf sample.}")
    lines.append(r"\label{fig:gyro_age_overall}")
    lines.append(r"\end{figure*}")
    lines.append("")

    if perbin_pngs:
        lines.append(r"\begin{figure*}")
        lines.append(r"\centering")
        # up to 4 panels
        for (b, p) in perbin_pngs[:4]:
            lines.append(rf"\includegraphics[width=0.48\textwidth]{{{p}}}")
        lines.append(r"\caption{Gyrochronology age distributions by effective-temperature bin for the GMM-gated sample.}")
        lines.append(r"\label{fig:gyro_age_by_bin}")
        lines.append(r"\end{figure*}")
        lines.append("")

    with open(TEX, "w") as f:
        f.write("\n".join(lines))

    print(f"saved figures to: {OUTDIR}")
    print(f"saved LaTeX snippet: {TEX}")

if __name__ == "__main__":
    main()
