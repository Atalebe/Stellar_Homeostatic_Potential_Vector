#!/usr/bin/env python3
"""
Variance scaling for Stellar Homeostatic Potential Vector Space (SHV).

Method (1): Coarse-grain within each teff_bin_s by bp_rp quantile bins (K),
compute within-subbin variance of selected metrics, and summarize how the
mean within-subbin variance changes with coarse-graining scale (mean Δbp_rp).

Includes a permutation null: permute Phi_gmm within teff_bin_s, recompute residuals,
and rerun the same scaling to produce σ²_null(scale).

Outputs:
  /mnt/g/STAR_HPV/results/variance_scaling/variance_scaling_bprp_quantiles.csv
  /mnt/g/STAR_HPV/results/variance_scaling/variance_scaling_bprp_quantiles.yaml
  /mnt/g/STAR_HPV/results/figures/variance_scaling/*.png
  /mnt/g/STAR_HPV/results/figures/variance_scaling/figures_variance_scaling.tex
"""

from __future__ import annotations

import os
import sys
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml


# --------------------------
# Helpers
# --------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def mad(x: np.ndarray) -> float:
    """Median absolute deviation (unscaled)."""
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.median(x)
    return np.median(np.abs(x - med))

def robust_var_mad2(x: np.ndarray) -> float:
    """Robust variance proxy ~ MAD^2."""
    m = mad(x)
    return m * m if np.isfinite(m) else np.nan

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
        out = pd.Series(index=s2.index, data=bins)
        return out
    except Exception:
        return None

def residualize_within_group(df: pd.DataFrame, group_col: str, x_col: str, out_col: str) -> pd.DataFrame:
    """
    Robust residualization: x - median(x) within each group.
    """
    med = df.groupby(group_col)[x_col].transform("median")
    df[out_col] = df[x_col] - med
    return df

@dataclass
class ScaleResult:
    teff_bin_s: str
    K: int
    metric: str
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
    For each teff bin: subdivide by bp_rp quantile bins (K), compute within-subbin
    variance of `metric`, then summarize (mean/median) across subbins.
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

        # Compute per-subbin stats
        per = []
        for cat, gg in g.groupby("_bprp_qbin"):
            x = gg[metric].to_numpy(dtype=float)
            if np.sum(np.isfinite(x)) < min_group_n:
                continue
            bprp_min = float(np.nanmin(gg[bprp_col].to_numpy(dtype=float)))
            bprp_max = float(np.nanmax(gg[bprp_col].to_numpy(dtype=float)))
            width = bprp_max - bprp_min
            per.append({
                "n": int(np.sum(np.isfinite(x))),
                "width": float(width),
                "var": safe_var(x),
                "mad2": robust_var_mad2(x),
            })

        if len(per) < 2:
            continue

        per_df = pd.DataFrame(per)
        # Summarize across subbins (unweighted means; also medians)
        res = ScaleResult(
            teff_bin_s=str(teff_bin),
            K=int(K),
            metric=str(metric),
            n_groups=int(len(per_df)),
            n_total=int(per_df["n"].sum()),
            mean_bin_width=float(per_df["width"].mean()),
            median_bin_width=float(per_df["width"].median()),
            mean_var=float(per_df["var"].mean(skipna=True)),
            median_var=float(per_df["var"].median(skipna=True)),
            mean_mad2=float(per_df["mad2"].mean(skipna=True)),
            median_mad2=float(per_df["mad2"].median(skipna=True)),
        )
        results.append(res)

    return results


def plot_scaling(
    out_png: str,
    df_sum: pd.DataFrame,
    metric: str,
    y_col: str,
    title: str,
) -> None:
    """
    Plot mean within-subbin variance proxy vs mean bin width, by teff bin, plus pooled.
    """
    plt.figure()
    sub = df_sum[df_sum["metric"] == metric].copy()
    if sub.empty:
        plt.close()
        return

    # By teff bins
    for teff_bin, g in sub.groupby("teff_bin_s"):
        g = g.sort_values("mean_bin_width")
        plt.plot(g["mean_bin_width"], g[y_col], marker="o", linestyle="-", label=str(teff_bin))

    # Pooled: average across teff bins at each K (simple mean)
    pooled = (
        sub.groupby("K")
        .agg(mean_bin_width=("mean_bin_width", "mean"),
             y=(y_col, "mean"))
        .reset_index()
        .sort_values("mean_bin_width")
    )
    plt.plot(pooled["mean_bin_width"], pooled["y"], marker="s", linestyle="--", label="pooled")

    plt.xlabel("mean Δ(bp_rp) within sub-bins")
    plt.ylabel(y_col)
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def write_tex_snippet(tex_path: str, fig_paths: Dict[str, str]) -> None:
    """
    Create a simple LaTeX snippet for including the figures.
    """
    lines = []
    lines.append(r"% Auto-generated by src/45a_variance_scaling_shv.py")
    for key, path in fig_paths.items():
        rel = path  # keep absolute; you can edit later if you prefer relative
        caption = key.replace("_", " ")
        label = f"fig:variance_scaling:{key}"
        block = rf"""
\begin{{figure}}[t]
  \centering
  \includegraphics[width=0.92\linewidth]{{{rel}}}
  \caption{{{caption}.}}
  \label{{{label}}}
\end{{figure}}
""".strip()
        lines.append(block)
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
    ap.add_argument("--figdir", default="/mnt/g/STAR_HPV/results/figures/variance_scaling")
    ap.add_argument("--teff-col", default="teff_bin_s")
    ap.add_argument("--bprp-col", default="bp_rp")
    ap.add_argument("--phi-col", default="Phi_gmm")
    ap.add_argument("--age-col", default="age_gyr_gyro")
    ap.add_argument("--ripeness-col", default="ripeness_time_gyr")
    ap.add_argument("--K-list", default="5,10,20,40")
    ap.add_argument("--min-group-n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    ensure_dir(args.outdir)
    ensure_dir(args.figdir)

    K_list = [int(x.strip()) for x in args.K_list.split(",") if x.strip()]
    rng = np.random.default_rng(args.seed)

    # Load
    df = pd.read_parquet(args.input)

    # Minimal required columns
    needed = [args.teff_col, args.bprp_col, args.phi_col, args.age_col, args.ripeness_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in input parquet: {missing}")

    # Base metrics
    df = df.copy()
    df = df.dropna(subset=[args.teff_col, args.bprp_col, args.phi_col])

    # Residualize Phi within teff bins (robust median)
    df = residualize_within_group(df, args.teff_col, args.phi_col, "phi_resid")

    metrics = [
        args.phi_col,
        "phi_resid",
        args.age_col,
        args.ripeness_col,
    ]

    # Compute observed scaling
    all_obs: List[ScaleResult] = []
    for K in K_list:
        for metric in metrics:
            all_obs.extend(
                compute_scaling_for_metric(
                    df=df,
                    metric=metric,
                    teff_col=args.teff_col,
                    bprp_col=args.bprp_col,
                    K=K,
                    min_group_n=args.min_group_n,
                )
            )

    obs_df = pd.DataFrame([r.__dict__ for r in all_obs])
    obs_df["variant"] = "observed"

    # Permutation null: permute Phi within teff bins, recompute residuals, rerun scaling
    df_null = df[[args.teff_col, args.bprp_col, args.phi_col, args.age_col, args.ripeness_col]].copy()

    def permute_within_group(x: pd.Series) -> pd.Series:
        arr = x.to_numpy()
        perm = rng.permutation(arr)
        return pd.Series(perm, index=x.index)

    df_null["Phi_perm"] = df_null.groupby(args.teff_col)[args.phi_col].transform(permute_within_group)
    df_null["phi_resid"] = df_null["Phi_perm"] - df_null.groupby(args.teff_col)["Phi_perm"].transform("median")

    all_null: List[ScaleResult] = []
    for K in K_list:
        # For the null, only Phi-related metrics matter most, but keep all for completeness
        # Use Phi_perm as the "Phi" metric in null variant
        for metric in metrics:
            if metric == args.phi_col:
                metric_use = "Phi_perm"
                metric_name = args.phi_col
            else:
                metric_use = metric
                metric_name = metric

            tmp = compute_scaling_for_metric(
                df=df_null.rename(columns={metric_use: "__metric__"}),
                metric="__metric__",
                teff_col=args.teff_col,
                bprp_col=args.bprp_col,
                K=K,
                min_group_n=args.min_group_n,
            )
            # Fix metric name back
            for r in tmp:
                r.metric = metric_name
            all_null.extend(tmp)

    null_df = pd.DataFrame([r.__dict__ for r in all_null])
    null_df["variant"] = "permute_phi_within_teff"

    # Merge and save
    out_csv = os.path.join(args.outdir, "variance_scaling_bprp_quantiles.csv")
    out_yaml = os.path.join(args.outdir, "variance_scaling_bprp_quantiles.yaml")

    out = pd.concat([obs_df, null_df], ignore_index=True)
    out.to_csv(out_csv, index=False)

    payload = {
        "input": args.input,
        "K_list": K_list,
        "min_group_n": args.min_group_n,
        "seed": args.seed,
        "n_rows_loaded": int(len(df)),
        "metrics": metrics,
        "outputs": {
            "csv": out_csv,
        },
    }
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)

    # Figures: observed vs null pooled curves (one plot per metric and per variance proxy)
    fig_paths: Dict[str, str] = {}

    for metric in metrics:
        for y_col in ["mean_var", "mean_mad2"]:
            # Build pooled table for observed + null
            def pooled_variant(v: str) -> pd.DataFrame:
                s = out[(out["variant"] == v) & (out["metric"] == metric)].copy()
                if s.empty:
                    return s
                return (
                    s.groupby("K")
                    .agg(mean_bin_width=("mean_bin_width", "mean"),
                         y=(y_col, "mean"))
                    .reset_index()
                    .sort_values("mean_bin_width")
                )

            p_obs = pooled_variant("observed")
            p_null = pooled_variant("permute_phi_within_teff")

            if p_obs.empty:
                continue

            plt.figure()
            plt.plot(p_obs["mean_bin_width"], p_obs["y"], marker="o", linestyle="-", label="observed (pooled)")
            if not p_null.empty:
                plt.plot(p_null["mean_bin_width"], p_null["y"], marker="s", linestyle="--", label="null (pooled)")

            plt.xlabel("mean Δ(bp_rp) within sub-bins")
            plt.ylabel(y_col)
            plt.title(f"Variance scaling (pooled): {metric} | {y_col}")
            plt.legend(fontsize=9)
            plt.tight_layout()

            out_png = os.path.join(args.figdir, f"variance_scaling_pooled_{metric}_{y_col}.png")
            plt.savefig(out_png, dpi=200)
            plt.close()

            fig_key = f"pooled_{metric}_{y_col}"
            fig_paths[fig_key] = out_png

        # By-teff curves (observed only) for mean_var
        sub_obs = out[(out["variant"] == "observed") & (out["metric"] == metric)].copy()
        if not sub_obs.empty:
            out_png2 = os.path.join(args.figdir, f"variance_scaling_byteff_{metric}_mean_var.png")
            plot_scaling(
                out_png=out_png2,
                df_sum=sub_obs,
                metric=metric,
                y_col="mean_var",
                title=f"Variance scaling by Teff bin: {metric} (mean within-subbin var)",
            )
            fig_paths[f"byteff_{metric}_mean_var"] = out_png2

    # TeX snippet
    tex_path = os.path.join(args.figdir, "figures_variance_scaling.tex")
    write_tex_snippet(tex_path, fig_paths)

    print(f"[OK] wrote {out_csv}")
    print(f"[OK] wrote {out_yaml}")
    print(f"[OK] wrote figures to {args.figdir}")
    print(f"[OK] wrote TeX snippet {tex_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
