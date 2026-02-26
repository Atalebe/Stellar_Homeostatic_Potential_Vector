#!/usr/bin/env python3
import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INP = "/mnt/g/STAR_HPV/processed/gyro/kepler_gmmgate_with_gyro_ages_mamajek.parquet"
OUTDIR = "/mnt/g/STAR_HPV/results/figures/gyro_age_ripeness"
TEX = os.path.join(OUTDIR, "figures_gyro_age_ripeness.tex")

BIN_ORDER = ["3200-4000", "4000-5200", "5200-6000", "6000-7500"]

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    df = pd.read_parquet(INP)

    # identify columns
    age_col = None
    for c in ["age_gyro_gyr", "age_gyr_gyro", "age_gyr", "gyro_age_gyr"]:
        if c in df.columns:
            age_col = c
            break
    if age_col is None:
        raise RuntimeError("No gyro age column found.")

    phi_col = "Phi_gmm" if "Phi_gmm" in df.columns else ("Phi" if "Phi" in df.columns else None)
    if phi_col is None:
        raise RuntimeError("No Phi column found (expected Phi_gmm or Phi).")

    rip_col = None
    for c in ["ripeness_time", "ripeness_gyro", "ripeness_gmm_time", "ripeness_gmm"]:
        if c in df.columns:
            rip_col = c
            break
    # If not precomputed, compute time-weighted ripeness now
    if rip_col is None:
        df["ripeness_time"] = pd.to_numeric(df[phi_col], errors="coerce") * pd.to_numeric(df[age_col], errors="coerce")
        rip_col = "ripeness_time"

    bin_col = "teff_bin_s" if "teff_bin_s" in df.columns else ("teff_bin" if "teff_bin" in df.columns else None)
    if bin_col is None:
        raise RuntimeError("No teff bin column found.")

    # pooled scatter (log x helps)
    x = pd.to_numeric(df[age_col], errors="coerce")
    y = pd.to_numeric(df[rip_col], errors="coerce")
    m = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x = x[m]; y = y[m]

    plt.figure()
    plt.scatter(x, y, s=6)
    plt.xscale("log")
    plt.xlabel("Gyro age (Gyr)")
    plt.ylabel(r"Time-weighted ripeness ($\Phi_{\rm gmm}\,t_{\rm gyro}$)")
    plt.title("Age vs time-weighted ripeness (pooled)")
    out_png = os.path.join(OUTDIR, "age_vs_ripeness_pooled.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    # per-bin binned medians
    perbin_pngs = []
    for b in BIN_ORDER:
        m = df[bin_col].astype(str) == b
        x = pd.to_numeric(df.loc[m, age_col], errors="coerce")
        y = pd.to_numeric(df.loc[m, rip_col], errors="coerce")
        ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
        x = x[ok]; y = y[ok]
        if len(x) < 30:
            continue

        # bin in log-age
        lx = np.log10(x)
        bins = np.linspace(lx.min(), lx.max(), 9)
        centers = 10 ** ((bins[:-1] + bins[1:]) / 2)
        med = []
        for i in range(len(bins)-1):
            mm = (lx >= bins[i]) & (lx < bins[i+1])
            if mm.sum() < 10:
                med.append(np.nan)
            else:
                med.append(np.nanmedian(y[mm]))
        med = np.array(med)

        plt.figure()
        plt.scatter(x, y, s=6)
        plt.xscale("log")
        plt.plot(centers, med, linewidth=2)
        plt.xlabel("Gyro age (Gyr)")
        plt.ylabel(r"Time-weighted ripeness ($\Phi_{\rm gmm}\,t_{\rm gyro}$)")
        plt.title(f"{b}: age vs ripeness")
        p = os.path.join(OUTDIR, f"age_vs_ripeness_{b.replace('-','_')}.png")
        plt.tight_layout()
        plt.savefig(p, dpi=200)
        plt.close()
        perbin_pngs.append((b, p))

    summary = {
        "input": INP,
        "age_col": age_col,
        "phi_col": phi_col,
        "ripeness_col": rip_col,
        "bin_col": bin_col,
        "figures": {"pooled": out_png, "per_bin": {b: p for b, p in perbin_pngs}},
    }
    with open(os.path.join(OUTDIR, "age_vs_ripeness_summary.yaml"), "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    # LaTeX snippet
    lines = []
    lines.append(r"\begin{figure*}")
    lines.append(r"\centering")
    lines.append(rf"\includegraphics[width=0.60\textwidth]{{{out_png}}}")
    lines.append(r"\caption{Gyrochronology age vs time-weighted ripeness for the pooled GMM-gated sample.}")
    lines.append(r"\label{fig:age_vs_ripeness_pooled}")
    lines.append(r"\end{figure*}")
    lines.append("")
    if perbin_pngs:
        lines.append(r"\begin{figure*}")
        lines.append(r"\centering")
        for (b, p) in perbin_pngs[:4]:
            lines.append(rf"\includegraphics[width=0.48\textwidth]{{{p}}}")
        lines.append(r"\caption{Age vs time-weighted ripeness by temperature bin. Points show stars with finite gyro ages; the curve shows binned medians in log-age.}")
        lines.append(r"\label{fig:age_vs_ripeness_by_bin}")
        lines.append(r"\end{figure*}")
        lines.append("")
    with open(TEX, "w") as f:
        f.write("\n".join(lines))

    print(f"saved figures to: {OUTDIR}")
    print(f"saved LaTeX snippet: {TEX}")

if __name__ == "__main__":
    main()
