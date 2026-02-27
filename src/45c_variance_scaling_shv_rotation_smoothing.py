#!/usr/bin/env python3
"""
src/45c_variance_scaling_shv_rotation_smoothing.py

Rotation-scale coarse-graining (stellar SHV):
- Within each teff_bin_s, treat log10(P_rot) as the "scale axis".
- For each smoothing bandwidth h in logP, compute a smoothed mean field Phi_hat(logP)
  via rolling median in sorted logP.
- Define residual r_h = Phi_gmm - Phi_hat_h(logP).
- The "variance at scale h" is Var(r_h) within each teff_bin_s (and pooled).

Null:
- Permute Phi_gmm within teff_bin_s, recompute Phi_hat_h and Var(r_h)_null.

Outputs:
  /mnt/g/STAR_HPV/results/variance_scaling/variance_scaling_rotation_smoothing.csv
  /mnt/g/STAR_HPV/results/variance_scaling/variance_scaling_rotation_smoothing.yaml
  /mnt/g/STAR_HPV/results/figures/variance_scaling_rotation_smoothing/*.png
  /mnt/g/STAR_HPV/results/figures/variance_scaling_rotation_smoothing/figures_variance_scaling_rotation_smoothing.tex

Notes:
- This is the closest stellar analogue of "variance vs smoothing scale".
- Uses robust rolling median; window size is derived from bandwidth h and
  the empirical spacing in logP for each teff bin.
"""

from __future__ import annotations

import os
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml


# --------------------------
# Utilities
# --------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def safe_var(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    return float(np.var(x, ddof=1))

def mad(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.median(x)
    return float(np.median(np.abs(x - med)))

def robust_var_mad2(x: np.ndarray) -> float:
    m = mad(x)
    return float(m * m) if np.isfinite(m) else np.nan

def rolling_median_on_sorted(x: np.ndarray, y: np.ndarray, w: int) -> np.ndarray:
    """
    Rolling median smoother on y after sorting by x.
    Returns y_smooth aligned to original order.
    """
    n = len(x)
    if n == 0:
        return np.array([])

    if w < 3:
        return y.copy()
    if w % 2 == 0:
        w += 1
    w = min(w, n if n % 2 == 1 else n - 1)
    if w < 3:
        return y.copy()

    order = np.argsort(x)
    y_s = y[order]

    half = w // 2
    y_sm = np.empty_like(y_s, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        window = y_s[lo:hi]
        window = window[np.isfinite(window)]
        y_sm[i] = np.median(window) if window.size else np.nan

    out = np.empty_like(y, dtype=float)
    out[order] = y_sm
    return out

def estimate_window_size_from_bandwidth(logp: np.ndarray, h: float, w_min: int = 21) -> int:
    """
    Convert a bandwidth in logP units into a rolling window size using empirical spacing.

    Heuristic:
      typical spacing ~ median(diff(sorted logp))
      w ~ (2*h)/spacing
    """
    lp = logp[np.isfinite(logp)]
    if lp.size < 50:
        return w_min
    s = np.sort(lp)
    d = np.diff(s)
    d = d[np.isfinite(d) & (d > 0)]
    if d.size == 0:
        return w_min
    spacing = float(np.median(d))
    if spacing <= 0:
        return w_min
    w = int(max(w_min, round((2.0 * h) / spacing)))
    if w % 2 == 0:
        w += 1
    return w


@dataclass
class RotScaleResult:
    teff_bin_s: str
    h_logp: float
    window_size: int
    n: int
    var_resid: float
    mad2_resid: float


def compute_rotation_smoothing(
    df: pd.DataFrame,
    teff_col: str,
    logp_col: str,
    phi_col: str,
    h_list: List[float],
    min_n: int,
) -> List[RotScaleResult]:
    out: List[RotScaleResult] = []

    for teff_bin, g in df.groupby(teff_col):
        g = g[[logp_col, phi_col]].copy()
        g = g.dropna(subset=[logp_col, phi_col])

        if g.shape[0] < min_n:
            continue

        logp = g[logp_col].to_numpy(dtype=float)
        phi = g[phi_col].to_numpy(dtype=float)

        for h in h_list:
            w = estimate_window_size_from_bandwidth(logp, h=h, w_min=21)
            phi_hat = rolling_median_on_sorted(logp, phi, w=w)
            resid = phi - phi_hat

            out.append(
                RotScaleResult(
                    teff_bin_s=str(teff_bin),
                    h_logp=float(h),
                    window_size=int(w),
                    n=int(np.sum(np.isfinite(resid))),
                    var_resid=safe_var(resid),
                    mad2_resid=robust_var_mad2(resid),
                )
            )

    return out


def pooled_curve(df_res: pd.DataFrame, ycol: str) -> pd.DataFrame:
    """
    Simple pooled mean across teff bins at each h.
    """
    return (
        df_res.groupby("h_logp")
        .agg(y=(ycol, "mean"), w=("window_size", "mean"))
        .reset_index()
        .sort_values("h_logp")
    )


def write_tex_snippet(tex_path: str, fig_paths: Dict[str, str]) -> None:
    lines = []
    lines.append(r"% Auto-generated by src/45c_variance_scaling_shv_rotation_smoothing.py")
    for key, path in fig_paths.items():
        label = f"fig:variance_scaling_rotation:{key}"
        caption = key.replace("_", " ")
        lines.append(r"\begin{figure}[t]")
        lines.append(r"  \centering")
        lines.append(rf"  \includegraphics[width=0.92\linewidth]{{{path}}}")
        lines.append(rf"  \caption{{{caption}.}}")
        lines.append(rf"  \label{{{label}}}")
        lines.append(r"\end{figure}")
        lines.append("")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")


# --------------------------
# Main
# --------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/mnt/g/STAR_HPV/processed/gyro/kepler_gmmgate_with_gyro_ages_mamajek.parquet")
    ap.add_argument("--outdir", default="/mnt/g/STAR_HPV/results/variance_scaling")
    ap.add_argument("--figdir", default="/mnt/g/STAR_HPV/results/figures/variance_scaling_rotation_smoothing")
    ap.add_argument("--teff-col", default="teff_bin_s")
    ap.add_argument("--prot-col", default="prot_days",
                    help="Rotation period column (days). If missing, try gate_days_gmm.")
    ap.add_argument("--phi-col", default="Phi_gmm")
    ap.add_argument("--h-list", default="0.05,0.10,0.20,0.30,0.50")
    ap.add_argument("--min-n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    ensure_dir(args.outdir)
    ensure_dir(args.figdir)

    h_list = [float(x.strip()) for x in args.h_list.split(",") if x.strip()]
    rng = np.random.default_rng(args.seed)

    df = pd.read_parquet(args.input).copy()

    # Resolve rotation period column
    prot_col = args.prot_col
    if prot_col not in df.columns:
        if "gate_days_gmm" in df.columns:
            prot_col = "gate_days_gmm"
        else:
            raise SystemExit(f"prot column '{args.prot_col}' not found and gate_days_gmm not found.")

    needed = [args.teff_col, prot_col, args.phi_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in input parquet: {missing}")

    df = df.dropna(subset=[args.teff_col, prot_col, args.phi_col]).copy()
    df["logP"] = np.log10(df[prot_col].to_numpy(dtype=float))

    # Observed
    obs = compute_rotation_smoothing(
        df=df,
        teff_col=args.teff_col,
        logp_col="logP",
        phi_col=args.phi_col,
        h_list=h_list,
        min_n=args.min_n,
    )
    obs_df = pd.DataFrame([r.__dict__ for r in obs])
    obs_df["variant"] = "observed"

    # Null: permute Phi within teff bins
    df_null = df[[args.teff_col, "logP", args.phi_col]].copy()

    def permute_within_group(s: pd.Series) -> pd.Series:
        arr = s.to_numpy()
        perm = rng.permutation(arr)
        return pd.Series(perm, index=s.index)

    df_null["Phi_perm"] = df_null.groupby(args.teff_col)[args.phi_col].transform(permute_within_group)

    null = compute_rotation_smoothing(
        df=df_null.rename(columns={"Phi_perm": args.phi_col}),
        teff_col=args.teff_col,
        logp_col="logP",
        phi_col=args.phi_col,
        h_list=h_list,
        min_n=args.min_n,
    )
    null_df = pd.DataFrame([r.__dict__ for r in null])
    null_df["variant"] = "permute_phi_within_teff"

    out = pd.concat([obs_df, null_df], ignore_index=True)

    out_csv = os.path.join(args.outdir, "variance_scaling_rotation_smoothing.csv")
    out_yaml = os.path.join(args.outdir, "variance_scaling_rotation_smoothing.yaml")
    out.to_csv(out_csv, index=False)

    meta = {
        "input": args.input,
        "prot_col_used": prot_col,
        "h_list": h_list,
        "min_n": args.min_n,
        "seed": args.seed,
        "n_rows_loaded": int(len(df)),
        "outputs": {"csv": out_csv},
        "figdir": args.figdir,
    }
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False)

    # Figures
    fig_paths: Dict[str, str] = {}

    for ycol in ["var_resid", "mad2_resid"]:
        plt.figure()
        p_obs = pooled_curve(out[out["variant"] == "observed"], ycol=ycol)
        p_null = pooled_curve(out[out["variant"] == "permute_phi_within_teff"], ycol=ycol)

        if not p_obs.empty:
            plt.plot(p_obs["h_logp"], p_obs["y"], marker="o", linestyle="-", label="observed (pooled)")
        if not p_null.empty:
            plt.plot(p_null["h_logp"], p_null["y"], marker="s", linestyle="--", label="null (pooled)")

        plt.xlabel("smoothing bandwidth h in log10(P_rot)")
        plt.ylabel(ycol)
        plt.title(f"Rotation-scale coarse graining: residual dispersion vs smoothing scale ({ycol})")
        plt.legend(fontsize=9)
        plt.tight_layout()

        out_png = os.path.join(args.figdir, f"pooled_rotation_smoothing_{ycol}.png")
        plt.savefig(out_png, dpi=200)
        plt.close()
        fig_paths[f"pooled_rotation_smoothing_{ycol}"] = out_png

    # By-bin plot for var_resid (observed only)
    sub = out[out["variant"] == "observed"].copy()
    if not sub.empty:
        plt.figure()
        for teff_bin, g in sub.groupby("teff_bin_s"):
            g = g.sort_values("h_logp")
            plt.plot(g["h_logp"], g["var_resid"], marker="o", linestyle="-", label=str(teff_bin))
        plt.xlabel("smoothing bandwidth h in log10(P_rot)")
        plt.ylabel("var_resid")
        plt.title("Rotation-scale coarse graining by Teff bin: Var(Phi - Phi_hat_h)")
        plt.legend(fontsize=8)
        plt.tight_layout()
        out_png = os.path.join(args.figdir, "byteff_rotation_smoothing_var_resid.png")
        plt.savefig(out_png, dpi=200)
        plt.close()
        fig_paths["byteff_rotation_smoothing_var_resid"] = out_png

    tex_path = os.path.join(args.figdir, "figures_variance_scaling_rotation_smoothing.tex")
    write_tex_snippet(tex_path, fig_paths)

    print(f"[OK] wrote {out_csv}")
    print(f"[OK] wrote {out_yaml}")
    print(f"[OK] wrote figures to {args.figdir}")
    print(f"[OK] wrote TeX snippet {tex_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
