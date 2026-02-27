#!/usr/bin/env python3
"""
src/45d_make_variance_scaling_tables_tex.py

Auto-generate LaTeX tables for:
- bp_rp detrended variance scaling (45b)
- rotation smoothing variance scaling (45c / 45c2)

Reads:
  /mnt/g/STAR_HPV/results/variance_scaling/variance_scaling_bprp_detrended.csv
  /mnt/g/STAR_HPV/results/variance_scaling/variance_scaling_rotation_smoothing.csv

Writes:
  /mnt/g/STAR_HPV/results/variance_scaling/tables_variance_scaling.tex
  /mnt/g/STAR_HPV/results/variance_scaling/tables_variance_scaling.yaml

Default table content matches what you pasted into the draft:
- bp_rp: only teff bins 4000-5200 and 5200-6000, Ks {5,10,20,40}, variants {observed,null}
- rotation: all teff bins, hs {0.05,0.1} for both variants (keeps table compact)
  (you can expand with --rot-h-list)
"""

from __future__ import annotations

import os
import argparse
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import yaml


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and (not np.isfinite(x))):
        return r"\cdots"
    return f"{x:.{nd}f}"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_bprp_table(
    df: pd.DataFrame,
    teff_keep: List[str],
    k_list: List[int],
    out_label: str,
    out_caption: str,
) -> str:
    # Normalize variants
    df = df.copy()
    df["variant2"] = df["variant"].map({
        "observed": "observed",
        "permute_phi_within_teff": "null",
    }).fillna(df["variant"])

    # Pivot-friendly filter: we need Phi_gmm and phi_detrended
    sub = df[df["metric"].isin(["Phi_gmm", "phi_detrended"])].copy()
    sub = sub[sub["teff_bin_s"].isin(teff_keep)].copy()
    sub = sub[sub["K"].isin(k_list)].copy()
    sub = sub[sub["variant2"].isin(["observed", "null"])].copy()

    # Create a wide frame: one row per (teff, variant, K)
    # Columns: mean_bin_width, mean_var(Phi_gmm), mean_var(phi_detrended)
    rows = []
    for (teff, variant, K), g in sub.groupby(["teff_bin_s", "variant2", "K"]):
        # Extract widths from any metric row (they match by construction)
        width = float(g["mean_bin_width"].iloc[0]) if len(g) else np.nan
        phi_var = g.loc[g["metric"] == "Phi_gmm", "mean_var"]
        det_var = g.loc[g["metric"] == "phi_detrended", "mean_var"]
        rows.append({
            "teff": teff,
            "variant": variant,
            "K": int(K),
            "width": float(phi_var.index.size and width or width),
            "phi_var": float(phi_var.iloc[0]) if len(phi_var) else np.nan,
            "det_var": float(det_var.iloc[0]) if len(det_var) else np.nan,
        })
    tab = pd.DataFrame(rows)
    tab = tab.sort_values(["teff", "variant", "K"])

    # Emit LaTeX
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{out_caption}}}")
    lines.append(rf"\label{{{out_label}}}")
    lines.append(r"\begin{tabular}{llrrrr}")
    lines.append(r"\toprule")
    lines.append(r"\texttt{teff\_bin\_s} & Variant & $K$ & $\overline{\Delta \mathrm{bp\_rp}}$ & $\overline{\sigma^2}(\Phi_{\mathrm{gmm}})$ & $\overline{\sigma^2}(\phi_{\mathrm{det}})$ \\")
    lines.append(r"\midrule")

    for teff in teff_keep:
        block = tab[tab["teff"] == teff]
        if block.empty:
            continue
        for _, r in block.iterrows():
            lines.append(
                f"{teff.replace('-', '--')} & {r['variant']} & {int(r['K'])} & "
                f"{fmt(r['width'], 5)} & {fmt(r['phi_var'], 4)} & {fmt(r['det_var'], 4)} \\\\"
            )
        lines.append(r"\midrule")

    # Replace last midrule with bottomrule
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    else:
        lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


def build_rotation_table(
    df: pd.DataFrame,
    h_list: List[float],
    out_label: str,
    out_caption: str,
    max_rows_per_teff: int = 999999,
) -> str:
    df = df.copy()
    df["variant2"] = df["variant"].map({
        "observed": "observed",
        "permute_phi_within_teff": "null",
    }).fillna(df["variant"])

    sub = df[df["h_logp"].isin(h_list)].copy()
    sub = sub[sub["variant2"].isin(["observed", "null"])].copy()
    sub = sub.sort_values(["teff_bin_s", "variant2", "h_logp"])

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{out_caption}}}")
    lines.append(rf"\label{{{out_label}}}")
    lines.append(r"\begin{tabular}{llrrrr}")
    lines.append(r"\toprule")
    lines.append(r"\texttt{teff\_bin\_s} & Variant & $h$ & window & $N$ & $\mathrm{Var}(r_h)$ \\")
    lines.append(r"\midrule")

    for teff, g in sub.groupby("teff_bin_s"):
        g = g.copy()
        # optional safety cap for huge table explosions
        if len(g) > max_rows_per_teff:
            g = g.iloc[:max_rows_per_teff]
        for _, r in g.iterrows():
            lines.append(
                f"{str(teff).replace('-', '--')} & {r['variant2']} & "
                f"{r['h_logp']:.2f} & {int(r['window_size'])} & {int(r['n'])} & {fmt(float(r['var_resid']), 4)} \\\\"
            )
        lines.append(r"\midrule")

    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    else:
        lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bprp-csv", default="/mnt/g/STAR_HPV/results/variance_scaling/variance_scaling_bprp_detrended.csv")
    ap.add_argument("--rot-csv", default="/mnt/g/STAR_HPV/results/variance_scaling/variance_scaling_rotation_smoothing.csv")
    ap.add_argument("--outdir", default="/mnt/g/STAR_HPV/results/variance_scaling")

    ap.add_argument("--bprp-teff-keep", default="4000-5200,5200-6000")
    ap.add_argument("--bprp-k-list", default="5,10,20,40")

    ap.add_argument("--rot-h-list", default="0.05,0.10")  # compact by default
    args = ap.parse_args()

    ensure_dir(args.outdir)

    teff_keep = [x.strip() for x in args.bprp_teff_keep.split(",") if x.strip()]
    k_list = [int(x.strip()) for x in args.bprp_k_list.split(",") if x.strip()]
    h_list = [float(x.strip()) for x in args.rot_h_list.split(",") if x.strip()]

    df_b = pd.read_csv(args.bprp_csv)
    df_r = pd.read_csv(args.rot_csv)

    tex_parts = []
    tex_parts.append(r"% Auto-generated by src/45d_make_variance_scaling_tables_tex.py")
    tex_parts.append("")

    tex_parts.append(
        build_bprp_table(
            df=df_b,
            teff_keep=teff_keep,
            k_list=k_list,
            out_label="tab:variance_scaling_bprp_summary",
            out_caption=(
                r"Colour coarse-graining variance scaling summary for the dominant temperature bins. "
                r"Values are mean within-sub-bin variances across $K$ $\mathrm{bp\_rp}$ quantile sub-bins. "
                r"The detrended coordinate $\phi_{\mathrm{det}}$ removes a rolling-median colour trend within each "
                r"\texttt{teff\_bin\_s}. The null permutes $\Phi_{\mathrm{gmm}}$ within \texttt{teff\_bin\_s}. "
                r"Full outputs are in \texttt{variance\_scaling\_bprp\_detrended.csv}."
            ),
        )
    )

    tex_parts.append(
        build_rotation_table(
            df=df_r,
            h_list=h_list,
            out_label="tab:variance_scaling_rotation_summary",
            out_caption=(
                r"Rotation-scale coarse-graining summary. The table reports $\mathrm{Var}(r_h)$ where "
                r"$r_h=\Phi_{\mathrm{gmm}}-\widehat{\Phi}_h(\log_{10}P_{\mathrm{rot}})$ and $\widehat{\Phi}_h$ "
                r"is a rolling-median smoother. The null permutes $\Phi_{\mathrm{gmm}}$ within \texttt{teff\_bin\_s} "
                r"prior to smoothing. Full outputs are in \texttt{variance\_scaling\_rotation\_smoothing.csv}."
            ),
        )
    )

    out_tex = os.path.join(args.outdir, "tables_variance_scaling.tex")
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(tex_parts).strip() + "\n")

    meta: Dict[str, Any] = {
        "bprp_csv": args.bprp_csv,
        "rot_csv": args.rot_csv,
        "bprp_teff_keep": teff_keep,
        "bprp_k_list": k_list,
        "rot_h_list": h_list,
        "outputs": {"tex": out_tex},
    }
    out_yaml = os.path.join(args.outdir, "tables_variance_scaling.yaml")
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False)

    print(f"[OK] wrote {out_tex}")
    print(f"[OK] wrote {out_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
