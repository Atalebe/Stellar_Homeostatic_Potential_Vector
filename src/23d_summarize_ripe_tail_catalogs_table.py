#!/usr/bin/env python3
"""
Summarize ripe-tail catalogs into one CSV + LaTeX snippet.

Fixes:
- avoids KeyError('prot_days') by always sourcing prot_days from the base Kepler GMM-gated table.
- robust to different bin-column names across exports.
"""

from __future__ import annotations

import os
import yaml
import numpy as np
import pandas as pd


BASE = "/mnt/g/STAR_HPV"
KEPLER_GMMGATE = f"{BASE}/processed/state_vectors/kepler_led_state_vector_gmmgate.parquet"

CAT_DIR = f"{BASE}/results/anchor_catalogs"
OUT_CSV = f"{BASE}/results/ripeness/ripe_tail_summary_table.csv"
OUT_TEX = f"{BASE}/results/ripeness/ripe_tail_summary_table.tex"
OUT_YAML = f"{BASE}/results/ripeness/ripe_tail_summary_table.yaml"


CATALOGS = [
    # global-postgate (old definition; tail can be outside window)
    ("tail_global_postgate_q90", f"{CAT_DIR}/kepler_ripe_tail_global_postgate_q90.parquet"),
    ("tail_global_postgate_q95", f"{CAT_DIR}/kepler_ripe_tail_global_postgate_q95.parquet"),

    # within-window (global threshold within window)
    ("tail_window_q90", f"{CAT_DIR}/kepler_ripe_tail_window_q90.parquet"),
    ("tail_window_q95", f"{CAT_DIR}/kepler_ripe_tail_window_q95.parquet"),

    # within-window (thresholds per bin)
    ("tail_window_perbin_q90", f"{CAT_DIR}/kepler_ripe_tail_window_perbin_q90.parquet"),
    ("tail_window_perbin_q95", f"{CAT_DIR}/kepler_ripe_tail_window_perbin_q95.parquet"),
]


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def safe_median(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) == 0:
        return float("nan")
    return float(np.median(x.values))


def main() -> None:
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    # Load canonical Kepler post-gate (slow-side) table: has prot_days + Phi_gmm + in_window_gmm
    base = pd.read_parquet(KEPLER_GMMGATE)
    need = ["source_id", "prot_days"]
    for c in need:
        if c not in base.columns:
            raise RuntimeError(f"Missing {c} in {KEPLER_GMMGATE}")

    # These are nice-to-have
    col_phi = pick_col(base, ["Phi_gmm", "Phi"])
    col_win = pick_col(base, ["in_window_gmm", "in_window"])
    col_bin = pick_col(base, ["teff_bin_s", "teff_bin"])

    # Build quick lookup tables
    base_ids = base["source_id"].astype("int64")
    prot_map = pd.Series(base["prot_days"].values, index=base_ids).to_dict()

    if col_phi:
        phi_map = pd.Series(pd.to_numeric(base[col_phi], errors="coerce").values, index=base_ids).to_dict()
    else:
        phi_map = {}

    if col_win:
        win_map = pd.Series(base[col_win].astype(bool).values, index=base_ids).to_dict()
    else:
        win_map = {}

    if col_bin:
        bin_map = pd.Series(base[col_bin].astype(str).values, index=base_ids).to_dict()
    else:
        bin_map = {}

    rows = []
    meta = {"kepler_gmmgate": KEPLER_GMMGATE, "catalogs": []}

    for name, path in CATALOGS:
        if not os.path.exists(path):
            continue

        df = pd.read_parquet(path)
        sid_col = pick_col(df, ["source_id"])
        if sid_col is None:
            raise RuntimeError(f"{path}: missing source_id")

        ids = df[sid_col].astype("int64")
        n = int(len(ids))

        prot_vals = pd.Series([prot_map.get(int(s), np.nan) for s in ids])
        phi_vals  = pd.Series([phi_map.get(int(s), np.nan) for s in ids]) if phi_map else pd.Series([np.nan]*n)
        win_vals  = pd.Series([win_map.get(int(s), np.nan) for s in ids]) if win_map else pd.Series([np.nan]*n)
        bin_vals  = pd.Series([bin_map.get(int(s), "NA") for s in ids]) if bin_map else pd.Series(["NA"]*n)

        rows.append({
            "catalog": name,
            "path": path,
            "rows": n,
            "prot_days_median": safe_median(prot_vals),
            "phi_median": safe_median(phi_vals),
            "in_window_frac": float(np.nanmean(win_vals.astype("float"))) if win_map else float("nan"),
            "bins_present": ",".join(sorted(set(bin_vals.unique().tolist()))),
        })

        meta["catalogs"].append({"name": name, "path": path, "rows": n})

    out = pd.DataFrame(rows).sort_values(["catalog"]).reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)

    # Tiny LaTeX snippet (no deluxetable drama)
    lines = []
    lines.append(r"\FindingBlock{Ripe-tail catalog summary (Kepler, GMM-gated)}{")
    lines.append(r"Summary statistics for ripeness-tail exports (global-postgate vs within-window, and within-window per-bin thresholds).")
    lines.append(r"\begin{itemize}")
    for _, r in out.iterrows():
        lines.append(
            rf"\item \texttt{{{r['catalog']}}}: $N={int(r['rows'])}$, "
            rf"$\tilde P_{{\rm rot}}={r['prot_days_median']:.3f}\,\rm d$, "
            rf"$\tilde\Phi={r['phi_median']:.3f}$, "
            rf"$f_{{\rm window}}={r['in_window_frac']:.3f}$."
        )
    lines.append(r"\end{itemize}")
    lines.append(r"}")
    with open(OUT_TEX, "w") as f:
        f.write("\n".join(lines) + "\n")

    with open(OUT_YAML, "w") as f:
        yaml.safe_dump({"output_csv": OUT_CSV, "output_tex": OUT_TEX, "rows": len(out), "meta": meta}, f, sort_keys=False)

    print("saved:", OUT_CSV)
    print("saved:", OUT_TEX)
    print("saved:", OUT_YAML)


if __name__ == "__main__":
    main()
