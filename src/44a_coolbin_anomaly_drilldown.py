#!/usr/bin/env python3
"""
44a_coolbin_anomaly_drilldown.py

Drill down into the 3200-4000K bin "window younger than non-window" behavior.

Outputs:
- YAML summary: /mnt/g/STAR_HPV/results/ripeness/coolbin_anomaly_drilldown.yaml
- CSV summary:  /mnt/g/STAR_HPV/results/ripeness/coolbin_anomaly_drilldown_table.csv
- Figures:      /mnt/g/STAR_HPV/results/figures/coolbin_anomaly/*.png
- LaTeX snippet:/mnt/g/STAR_HPV/results/figures/coolbin_anomaly/figures_coolbin_anomaly.tex

Assumes input has columns:
  teff_bin_s, in_window_gmm, age_gyr_gyro, prot_days, Phi_gmm, bp_rp, bv_est
"""

from __future__ import annotations

import os
import argparse
from typing import Dict, Any

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp, mannwhitneyu, kendalltau

DEFAULT_INP = "/mnt/g/STAR_HPV/processed/gyro/kepler_gmmgate_with_gyro_ages_mamajek.parquet"
DEFAULT_OUTDIR = "/mnt/g/STAR_HPV/results/figures/coolbin_anomaly"
DEFAULT_YAML = "/mnt/g/STAR_HPV/results/ripeness/coolbin_anomaly_drilldown.yaml"
DEFAULT_CSV = "/mnt/g/STAR_HPV/results/ripeness/coolbin_anomaly_drilldown_table.csv"
DEFAULT_TEX = "/mnt/g/STAR_HPV/results/figures/coolbin_anomaly/figures_coolbin_anomaly.tex"

BIN = "3200-4000"

WIN_COL = "in_window_gmm"
AGE_COL = "age_gyr_gyro"
PHI_COL = "Phi_gmm"
PROT_COL = "prot_days"
BPRP_COL = "bp_rp"
BV_COL = "bv_est"

# Mamajek-style diagnostic constants
MAMAJEK_C = 0.495
EPS = 1e-12


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def qstats(x: pd.Series) -> Dict[str, float]:
    x = pd.to_numeric(x, errors="coerce").astype(float)
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


def two_sample_tests(a: pd.Series, b: pd.Series) -> Dict[str, Any]:
    a = pd.to_numeric(a, errors="coerce").astype(float)
    b = pd.to_numeric(b, errors="coerce").astype(float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    out: Dict[str, Any] = {"n_a": int(len(a)), "n_b": int(len(b))}
    if len(a) < 10 or len(b) < 10:
        out["note"] = "insufficient sample (need >=10 per group)"
        return out

    out["median_a"] = float(np.median(a))
    out["median_b"] = float(np.median(b))
    out["delta_median"] = float(out["median_a"] - out["median_b"])

    ks = ks_2samp(a, b)
    out["ks_stat"] = float(ks.statistic)
    out["ks_p"] = float(ks.pvalue)

    mw = mannwhitneyu(a, b, alternative="greater")
    out["mw_u"] = float(mw.statistic)
    out["mw_p_greater"] = float(mw.pvalue)
    return out


def save_hist(a: pd.Series, b: pd.Series, xlabel: str, title: str, out_png: str) -> None:
    a = pd.to_numeric(a, errors="coerce").astype(float)
    b = pd.to_numeric(b, errors="coerce").astype(float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    plt.figure(figsize=(7, 5))
    plt.hist(b, bins=40, alpha=0.6, label="non-window")
    plt.hist(a, bins=40, alpha=0.6, label="window")
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def save_scatter(x: pd.Series, y: pd.Series, c: pd.Series, xlabel: str, ylabel: str, title: str, out_png: str) -> None:
    x = pd.to_numeric(x, errors="coerce").astype(float)
    y = pd.to_numeric(y, errors="coerce").astype(float)
    c = c.astype(bool)

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    c = c[m]

    plt.figure(figsize=(7, 5))
    plt.scatter(x[~c], y[~c], s=10, alpha=0.6, label="non-window")
    plt.scatter(x[c], y[c], s=10, alpha=0.7, label="window")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INP)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--out-yaml", default=DEFAULT_YAML)
    ap.add_argument("--out-csv", default=DEFAULT_CSV)
    ap.add_argument("--out-tex", default=DEFAULT_TEX)
    args = ap.parse_args()

    ensure_dir(args.outdir)
    ensure_dir(os.path.dirname(args.out_yaml))
    ensure_dir(os.path.dirname(args.out_csv))
    ensure_dir(os.path.dirname(args.out_tex))

    df = pd.read_parquet(args.input)

    required = ["teff_bin_s", WIN_COL, AGE_COL, PHI_COL, PROT_COL, BPRP_COL, BV_COL]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise RuntimeError(f"Missing required columns: {miss}. Available sample: {list(df.columns)[:50]}")

    cool = df[df["teff_bin_s"] == BIN].copy()
    cool[WIN_COL] = cool[WIN_COL].astype(bool)

    # --- create derived columns on `cool` first (IMPORTANT) ---
    bv = pd.to_numeric(cool[BV_COL], errors="coerce").astype(float)
    cool["bv_minus_C"] = bv - MAMAJEK_C
    cool["near_floor"] = (cool["bv_minus_C"] > 0) & (cool["bv_minus_C"] < 0.05)

    # Now define subsets so they inherit those columns
    w = cool[cool[WIN_COL] == True].copy()
    n = cool[cool[WIN_COL] == False].copy()

    # Kendall correlations (within cool bin)
    tau_age_phi, p_age_phi = kendalltau(
        pd.to_numeric(cool[AGE_COL], errors="coerce"),
        pd.to_numeric(cool[PHI_COL], errors="coerce"),
        nan_policy="omit",
    )
    tau_age_prot, p_age_prot = kendalltau(
        pd.to_numeric(cool[AGE_COL], errors="coerce"),
        pd.to_numeric(cool[PROT_COL], errors="coerce"),
        nan_policy="omit",
    )
    tau_phi_prot, p_phi_prot = kendalltau(
        pd.to_numeric(cool[PHI_COL], errors="coerce"),
        pd.to_numeric(cool[PROT_COL], errors="coerce"),
        nan_policy="omit",
    )

    summary: Dict[str, Any] = {
        "input": args.input,
        "bin": BIN,
        "rows": int(len(cool)),
        "rows_window": int(len(w)),
        "rows_nonwindow": int(len(n)),
        "stats": {
            "age_window": qstats(w[AGE_COL]),
            "age_nonwindow": qstats(n[AGE_COL]),
            "prot_window": qstats(w[PROT_COL]),
            "prot_nonwindow": qstats(n[PROT_COL]),
            "phi_window": qstats(w[PHI_COL]),
            "phi_nonwindow": qstats(n[PHI_COL]),
            "bp_rp_window": qstats(w[BPRP_COL]),
            "bp_rp_nonwindow": qstats(n[BPRP_COL]),
            "bv_est_window": qstats(w[BV_COL]),
            "bv_est_nonwindow": qstats(n[BV_COL]),
            "bv_minus_C_all": qstats(cool["bv_minus_C"]),
            "near_floor_frac_all": float(np.mean(cool["near_floor"].fillna(False))),
            "near_floor_frac_window": float(np.mean(w["near_floor"].fillna(False))) if len(w) else float("nan"),
            "near_floor_frac_nonwindow": float(np.mean(n["near_floor"].fillna(False))) if len(n) else float("nan"),
        },
        "tests": {
            "window_vs_nonwindow_age": two_sample_tests(w[AGE_COL], n[AGE_COL]),
            "window_vs_nonwindow_prot": two_sample_tests(w[PROT_COL], n[PROT_COL]),
            "window_vs_nonwindow_phi": two_sample_tests(w[PHI_COL], n[PHI_COL]),
        },
        "kendall": {
            "tau_age_phi": float(tau_age_phi) if np.isfinite(tau_age_phi) else None,
            "p_age_phi": float(p_age_phi) if np.isfinite(p_age_phi) else None,
            "tau_age_prot": float(tau_age_prot) if np.isfinite(tau_age_prot) else None,
            "p_age_prot": float(p_age_prot) if np.isfinite(p_age_prot) else None,
            "tau_phi_prot": float(tau_phi_prot) if np.isfinite(tau_phi_prot) else None,
            "p_phi_prot": float(p_phi_prot) if np.isfinite(p_phi_prot) else None,
        },
        "notes": [
            "If window appears younger in 3200-4000K, check whether window skews toward faster Prot or lower Phi within the gated dwarf regime.",
            "Check proximity to Mamajek (B-V - C) floor; near-floor values can destabilize inferred ages.",
        ],
    }

    # CSV table
    rows = []
    for lab, sub in [("window", w), ("nonwindow", n)]:
        rows.append({
            "group": lab,
            "n": int(len(sub)),
            "age_med": float(np.nanmedian(pd.to_numeric(sub[AGE_COL], errors="coerce"))),
            "prot_med": float(np.nanmedian(pd.to_numeric(sub[PROT_COL], errors="coerce"))),
            "phi_med": float(np.nanmedian(pd.to_numeric(sub[PHI_COL], errors="coerce"))),
            "bp_rp_med": float(np.nanmedian(pd.to_numeric(sub[BPRP_COL], errors="coerce"))),
            "bv_est_med": float(np.nanmedian(pd.to_numeric(sub[BV_COL], errors="coerce"))),
            "near_floor_frac": float(np.mean(sub["near_floor"].fillna(False))) if len(sub) else float("nan"),
        })
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)

    with open(args.out_yaml, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    # Figures
    figs: Dict[str, str] = {}
    figs["age_hist"] = os.path.join(args.outdir, "coolbin_age_hist_window_vs_nonwindow.png")
    save_hist(w[AGE_COL], n[AGE_COL], "gyro age (Gyr)", f"{BIN}: gyro age (window vs non-window)", figs["age_hist"])

    figs["prot_hist"] = os.path.join(args.outdir, "coolbin_prot_hist_window_vs_nonwindow.png")
    save_hist(w[PROT_COL], n[PROT_COL], "P_rot (days)", f"{BIN}: P_rot (window vs non-window)", figs["prot_hist"])

    figs["phi_hist"] = os.path.join(args.outdir, "coolbin_phi_hist_window_vs_nonwindow.png")
    save_hist(w[PHI_COL], n[PHI_COL], r"$\Phi_{\rm gmm}$", f"{BIN}: Phi_gmm (window vs non-window)", figs["phi_hist"])

    figs["age_vs_prot"] = os.path.join(args.outdir, "coolbin_age_vs_prot_scatter.png")
    save_scatter(cool[PROT_COL], cool[AGE_COL], cool[WIN_COL], "P_rot (days)", "gyro age (Gyr)", f"{BIN}: age vs P_rot", figs["age_vs_prot"])

    figs["age_vs_bprp"] = os.path.join(args.outdir, "coolbin_age_vs_bp_rp_scatter.png")
    save_scatter(cool[BPRP_COL], cool[AGE_COL], cool[WIN_COL], "BP-RP", "gyro age (Gyr)", f"{BIN}: age vs BP-RP", figs["age_vs_bprp"])

    figs["bv_floor_hist"] = os.path.join(args.outdir, "coolbin_bv_minus_C_hist.png")
    plt.figure(figsize=(7, 5))
    xw = pd.to_numeric(w["bv_minus_C"], errors="coerce").astype(float)
    xn = pd.to_numeric(n["bv_minus_C"], errors="coerce").astype(float)
    xw = xw[np.isfinite(xw)]
    xn = xn[np.isfinite(xn)]
    plt.hist(xn, bins=40, alpha=0.6, label="non-window")
    plt.hist(xw, bins=40, alpha=0.6, label="window")
    plt.axvline(0.0, linewidth=2, linestyle="--")
    plt.axvline(0.05, linewidth=2, linestyle="--")
    plt.xlabel(r"$B\!-\!V - C$ (C=0.495)")
    plt.ylabel("count")
    plt.title(f"{BIN}: proximity to (B-V - C) floor")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figs["bv_floor_hist"], dpi=160)
    plt.close()

    # LaTeX snippet
    tex = []
    tex.append(r"\FindingBlock{Cool-bin anomaly drilldown (3200--4000\,K)}{")
    tex.append(r"The 3200--4000\,K bin shows a distinct behavior in the H3 window-age comparison.")
    tex.append(r"Panels below compare window vs non-window distributions and provide gyro-age diagnostics near the Mamajek $(B\!-\!V-C)$ floor.")
    tex.append(r"}")
    tex.append("")
    tex.append(r"\begin{figure}[t]")
    tex.append(r"\centering")
    tex.append(rf"\includegraphics[width=0.32\textwidth]{{{figs['age_hist']}}}")
    tex.append(rf"\includegraphics[width=0.32\textwidth]{{{figs['prot_hist']}}}")
    tex.append(rf"\includegraphics[width=0.32\textwidth]{{{figs['phi_hist']}}}")
    tex.append(r"\caption{3200--4000\,K: gyro age, rotation period, and $\Phi_{\rm gmm}$ distributions (window vs non-window).}")
    tex.append(r"\end{figure}")
    tex.append("")
    tex.append(r"\begin{figure}[t]")
    tex.append(r"\centering")
    tex.append(rf"\includegraphics[width=0.49\textwidth]{{{figs['age_vs_prot']}}}")
    tex.append(rf"\includegraphics[width=0.49\textwidth]{{{figs['age_vs_bprp']}}}")
    tex.append(r"\caption{3200--4000\,K: gyro age vs rotation period and gyro age vs BP-RP, with window membership highlighted.}")
    tex.append(r"\end{figure}")
    tex.append("")
    tex.append(r"\begin{figure}[t]")
    tex.append(r"\centering")
    tex.append(rf"\includegraphics[width=0.65\textwidth]{{{figs['bv_floor_hist']}}}")
    tex.append(r"\caption{3200--4000\,K: diagnostic for Mamajek gyro ages showing $(B\!-\!V-C)$; the near-floor region (0 to 0.05) can destabilize inferred ages.}")
    tex.append(r"\end{figure}")
    tex.append("")

    with open(args.out_tex, "w") as f:
        f.write("\n".join(tex))

    print(f"saved YAML: {args.out_yaml}")
    print(f"saved CSV:  {args.out_csv}")
    print(f"saved TeX:  {args.out_tex}")
    print("figures:")
    for k, v in figs.items():
        print(" ", k, "->", v)


if __name__ == "__main__":
    main()
