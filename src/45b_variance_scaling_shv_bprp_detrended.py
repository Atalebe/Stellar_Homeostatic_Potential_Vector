#!/usr/bin/env python3
"""
src/45b_variance_scaling_shv_bprp_detrended.py

Variance scaling (stellar SHV) with a *meaningful* residual:
- Within each teff_bin_s, coarse-grain by bp_rp quantile bins (K).
- Fit a smooth trend Phi(bp_rp) within each teff_bin_s (rolling-median smoother).
- Define phi_detrended = Phi_gmm - Phi_smooth(bp_rp).
- Measure within-subbin variance of Phi_gmm and phi_detrended as a function of scale
  (mean Δ(bp_rp) per sub-bin), and compare to a permutation null:
    permute Phi_gmm within teff_bin_s, then recompute smooth trend + detrended residual,
    then rerun the scaling.

Outputs:
  /mnt/g/STAR_HPV/results/variance_scaling/variance_scaling_bprp_detrended.csv
  /mnt/g/STAR_HPV/results/variance_scaling/variance_scaling_bprp_detrended.yaml
  /mnt/g/STAR_HPV/results/figures/variance_scaling_bprp_detrended/*.png
  /mnt/g/STAR_HPV/results/figures/variance_scaling_bprp_detrended/figures_variance_scaling_bprp_detrended.tex

Notes:
- This script avoids the "subtracting a constant doesn't change variance" trap.
- The smoother is nonparametric and robust (rolling median on sorted bp_rp).
"""

from __future__ import annotations

import os
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml


# --------------------------
# Utilities
# --------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def mad(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.median(x)
    return float(np.median(np.abs(x - med)))

def robust_var_mad2(x: np.ndarray) -> float:
    m = mad(x)
    return float(m * m) if np.isfinite(m) else np.nan

def safe_var(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    return float(np.var(x, ddof=1))

def qcut_with_fallback(s: pd.Series, q: int) -> Optional[pd.Series]:
    """
    Quantile binning that gracefully handles duplicated edges.
    Returns None if it cannot create at least 2 bins.
    """
    s2 = s.dropna()
    if s2.size < max(10, q * 5):
        return None
    try:
        bins = pd.qcut(s2, q=q, duplicates="drop")
        if bins.cat.categories.size < 2:
            return None
        return pd.Series(index=s2.index, data=bins)
    except Exception:
        return None

def rolling_median_smoother(x: np.ndarray, y: np.ndarray, frac: float) -> np.ndarray:
    """
    Robust smoother: rolling median on y after sorting by x.
    frac controls the window size as a fraction of N (clipped).
    Returns y_smooth aligned to the original ordering.
    """
    n = len(x)
    if n == 0:
        return np.array([])
    order = np.argsort(x)
    x_s = x[order]
    y_s = y[order]

    # Choose window size
    w = int(max(11, round(frac * n)))
    if w % 2 == 0:
        w += 1
    w = min(w, n if n % 2 == 1 else n - 1)
    if w < 3:
        return y.copy()

    half = w // 2
    y_smooth_sorted = np.empty_like(y_s, dtype=float)

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        window = y_s[lo:hi]
        window = window[np.isfinite(window)]
        y_smooth_sorted[i] = np.median(window) if window.size else np.nan

    # Map back to original order
    y_smooth = np.empty_like(y, dtype=float)
    y_smooth[order] = y_smooth_sorted
    return y_smooth


@dataclass
class ScaleResult:
    teff_bin_s: str
    K: int
    metric: str
    smoother_frac: float
    n_groups: int
    n_total: int
    mean_bin_width: float
    median_bin_width: float
    mean_var: float
    median_var: float
    mean_mad2: float
    median_mad2: float


def compute_scaling_for_metric(
    df: pd.DataFrame,
    metric: str,
    teff_col: str,
    bprp_col: str,
    K: int,
    min_group_n: int,
) -> List[ScaleResult]:
    """
    For each teff bin: subdivide by bp_rp quantile bins (K),
    compute within-subbin variance of `metric`, summarize across subbins.
    """
    results: List[ScaleResult] = []

    for teff_bin, g in df.groupby(teff_col):
        g = g[[metric, bprp_col]].copy()
        g = g.dropna(subset=[metric, bprp_col])

        if g.shape[0] < max(min_group_n, K * 5):
            continue

        bins = qcut_with_fallback(g[bprp_col], q=K)
        if bins is None:
            continue

        g = g.loc[bins.index].copy()
        g["_bprp_qbin"] = bins

        per = []
        for _, gg in g.groupby("_bprp_qbin"):
            x = gg[metric].to_numpy(dtype=float)
            n = int(np.sum(np.isfinite(x)))
            if n < min_group_n:
                continue

            bprp = gg[bprp_col].to_numpy(dtype=float)
            width = float(np.nanmax(bprp) - np.nanmin(bprp))

            per.append({
                "n": n,
                "width": width,
                "var": safe_var(x),
                "mad2": robust_var_mad2(x),
            })

        if len(per) < 2:
            continue

        per_df = pd.DataFrame(per)

        results.append(
            ScaleResult(
                teff_bin_s=str(teff_bin),
                K=int(K),
                metric=str(metric),
                smoother_frac=float(df.attrs.get("smoother_frac", np.nan)),
                n_groups=int(len(per_df)),
                n_total=int(per_df["n"].sum()),
                mean_bin_width=float(per_df["width"].mean()),
                median_bin_width=float(per_df["width"].median()),
                mean_var=float(per_df["var"].mean(skipna=True)),
                median_var=float(per_df["var"].median(skipna=True)),
                mean_mad2=float(per_df["mad2"].mean(skipna=True)),
                median_mad2=float(per_df["mad2"].median(skipna=True)),
            )
        )

    return results


def plot_pooled_observed_vs_null(
    out_png: str,
    df_all: pd.DataFrame,
    metric: str,
    y_col: str,
    title: str,
) -> None:
    sub = df_all[df_all["metric"] == metric].copy()
    if sub.empty:
        return

    def pooled(variant: str) -> pd.DataFrame:
        s = sub[sub["variant"] == variant].copy()
        if s.empty:
            return s
        return (
            s.groupby("K")
            .agg(mean_bin_width=("mean_bin_width", "mean"),
                 y=(y_col, "mean"))
            .reset_index()
            .sort_values("mean_bin_width")
        )

    p_obs = pooled("observed")
    p_null = pooled("permute_phi_within_teff")

    if p_obs.empty:
        return

    plt.figure()
    plt.plot(p_obs["mean_bin_width"], p_obs["y"], marker="o", linestyle="-", label="observed (pooled)")
    if not p_null.empty:
        plt.plot(p_null["mean_bin_width"], p_null["y"], marker="s", linestyle="--", label="null (pooled)")
    plt.xlabel("mean Δ(bp_rp) within sub-bins")
    plt.ylabel(y_col)
    plt.title(title)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def write_tex_snippet(tex_path: str, fig_paths: Dict[str, str]) -> None:
    lines = []
    lines.append(r"% Auto-generated by src/45b_variance_scaling_shv_bprp_detrended.py")
    for key, path in fig_paths.items():
        label = f"fig:variance_scaling_bprp_detrended:{key}"
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
    ap.add_argument("--figdir", default="/mnt/g/STAR_HPV/results/figures/variance_scaling_bprp_detrended")
    ap.add_argument("--teff-col", default="teff_bin_s")
    ap.add_argument("--bprp-col", default="bp_rp")
    ap.add_argument("--phi-col", default="Phi_gmm")
    ap.add_argument("--age-col", default="age_gyr_gyro")
    ap.add_argument("--ripeness-col", default="ripeness_time_gyr")
    ap.add_argument("--K-list", default="5,10,20,40")
    ap.add_argument("--min-group-n", type=int, default=30)
    ap.add_argument("--smoother-frac", type=float, default=0.15,
                    help="Rolling-median smoother window as a fraction of N within each teff bin.")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    ensure_dir(args.outdir)
    ensure_dir(args.figdir)

    K_list = [int(x.strip()) for x in args.K_list.split(",") if x.strip()]
    rng = np.random.default_rng(args.seed)

    df = pd.read_parquet(args.input).copy()

    needed = [args.teff_col, args.bprp_col, args.phi_col, args.age_col, args.ripeness_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in input parquet: {missing}")

    df = df.dropna(subset=[args.teff_col, args.bprp_col, args.phi_col]).copy()

    # Smooth Phi(bp_rp) within each teff bin and detrend
    phi_smooth = np.full(len(df), np.nan, dtype=float)
    for teff_bin, idx in df.groupby(args.teff_col).groups.items():
        ii = np.array(list(idx), dtype=int)
        x = df.loc[ii, args.bprp_col].to_numpy(dtype=float)
        y = df.loc[ii, args.phi_col].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 50:
            continue
        phi_smooth[ii[mask]] = rolling_median_smoother(x[mask], y[mask], frac=args.smoother_frac)

    df["phi_smooth_bprp"] = phi_smooth
    df["phi_detrended"] = df[args.phi_col] - df["phi_smooth_bprp"]

    # Store smoother_frac in attrs for downstream bookkeeping in results rows
    df.attrs["smoother_frac"] = args.smoother_frac

    metrics = [
        args.phi_col,
        "phi_detrended",
        args.age_col,
        args.ripeness_col,
    ]

    # Observed
    all_obs = []
    for K in K_list:
        for metric in metrics:
            all_obs.extend(compute_scaling_for_metric(df, metric, args.teff_col, args.bprp_col, K, args.min_group_n))

    obs_df = pd.DataFrame([r.__dict__ for r in all_obs])
    obs_df["variant"] = "observed"

    # Null: permute Phi within teff, recompute smoother + detrended
    df_null = df[[args.teff_col, args.bprp_col, args.phi_col, args.age_col, args.ripeness_col]].copy()

    def permute_within_group(s: pd.Series) -> pd.Series:
        arr = s.to_numpy()
        perm = rng.permutation(arr)
        return pd.Series(perm, index=s.index)

    df_null["Phi_perm"] = df_null.groupby(args.teff_col)[args.phi_col].transform(permute_within_group)

    # Recompute smoother on permuted Phi
    phi_smooth_null = np.full(len(df_null), np.nan, dtype=float)
    for teff_bin, idx in df_null.groupby(args.teff_col).groups.items():
        ii = np.array(list(idx), dtype=int)
        x = df_null.loc[ii, args.bprp_col].to_numpy(dtype=float)
        y = df_null.loc[ii, "Phi_perm"].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 50:
            continue
        phi_smooth_null[ii[mask]] = rolling_median_smoother(x[mask], y[mask], frac=args.smoother_frac)

    df_null["phi_smooth_bprp"] = phi_smooth_null
    df_null["phi_detrended"] = df_null["Phi_perm"] - df_null["phi_smooth_bprp"]
    df_null.attrs["smoother_frac"] = args.smoother_frac

    # Compute null scaling; map Phi_perm back to the name Phi_gmm in output metric field
    all_null = []
    for K in K_list:
        # Phi-based
        tmp_phi = compute_scaling_for_metric(
            df=df_null.rename(columns={"Phi_perm": "__metric__"}),
            metric="__metric__",
            teff_col=args.teff_col,
            bprp_col=args.bprp_col,
            K=K,
            min_group_n=args.min_group_n,
        )
        for r in tmp_phi:
            r.metric = args.phi_col
        all_null.extend(tmp_phi)

        # Detrended Phi-based
        tmp_det = compute_scaling_for_metric(
            df=df_null,
            metric="phi_detrended",
            teff_col=args.teff_col,
            bprp_col=args.bprp_col,
            K=K,
            min_group_n=args.min_group_n,
        )
        all_null.extend(tmp_det)

        # Non-Phi metrics (these should match observed, but keep for completeness)
        for metric in [args.age_col, args.ripeness_col]:
            all_null.extend(compute_scaling_for_metric(df_null, metric, args.teff_col, args.bprp_col, K, args.min_group_n))

    null_df = pd.DataFrame([r.__dict__ for r in all_null])
    null_df["variant"] = "permute_phi_within_teff"

    out = pd.concat([obs_df, null_df], ignore_index=True)

    out_csv = os.path.join(args.outdir, "variance_scaling_bprp_detrended.csv")
    out_yaml = os.path.join(args.outdir, "variance_scaling_bprp_detrended.yaml")
    out.to_csv(out_csv, index=False)

    meta = {
        "input": args.input,
        "K_list": K_list,
        "min_group_n": args.min_group_n,
        "seed": args.seed,
        "smoother_frac": args.smoother_frac,
        "n_rows_loaded": int(len(df)),
        "metrics": metrics,
        "outputs": {"csv": out_csv},
        "figdir": args.figdir,
    }
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False)

    # Figures (pooled observed vs null)
    fig_paths: Dict[str, str] = {}
    for metric in [args.phi_col, "phi_detrended"]:
        for y_col in ["mean_var", "mean_mad2"]:
            out_png = os.path.join(args.figdir, f"pooled_{metric}_{y_col}.png")
            plot_pooled_observed_vs_null(
                out_png=out_png,
                df_all=out,
                metric=metric,
                y_col=y_col,
                title=f"Variance scaling vs bp_rp (pooled): {metric} | {y_col} | smoother_frac={args.smoother_frac}",
            )
            fig_paths[f"pooled_{metric}_{y_col}"] = out_png

    tex_path = os.path.join(args.figdir, "figures_variance_scaling_bprp_detrended.tex")
    write_tex_snippet(tex_path, fig_paths)

    print(f"[OK] wrote {out_csv}")
    print(f"[OK] wrote {out_yaml}")
    print(f"[OK] wrote figures to {args.figdir}")
    print(f"[OK] wrote TeX snippet {tex_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
