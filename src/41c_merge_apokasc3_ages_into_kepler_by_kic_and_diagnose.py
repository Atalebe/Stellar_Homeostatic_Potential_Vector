#!/usr/bin/env python3
"""
Merge APOKASC-3 ages into Kepler rotator-led products by KIC (not Gaia source_id),
then write coverage diagnostics.

Inputs expected to exist (based on your pipeline):
- /mnt/g/STAR_HPV/raw/ages/apokasc3_kic_ages_true.parquet      (kic, age_gyr, age_gyr_err)
- /mnt/g/STAR_HPV/processed/state_vectors/kepler_led_state_vector_gmmgate.parquet
- tail catalogs (global postgate + window + perbin window)

Outputs:
- /mnt/g/STAR_HPV/processed/ages_merged/*_with_apokasc3_ages_by_kic.parquet
- /mnt/g/STAR_HPV/results/ripeness/apokasc3_overlap_by_kic.yaml
- /mnt/g/STAR_HPV/results/ripeness/apokasc3_overlap_by_kic_matches.csv (if any)
"""

from __future__ import annotations
import os
from pathlib import Path
import yaml
import numpy as np
import pandas as pd

AGE_KIC = "/mnt/g/STAR_HPV/raw/ages/apokasc3_kic_ages_true.parquet"

TARGETS = {
    "kepler_gmmgate": "/mnt/g/STAR_HPV/processed/state_vectors/kepler_led_state_vector_gmmgate.parquet",
    "anchor_window": "/mnt/g/STAR_HPV/results/anchor_catalogs/kepler_anchor_catalog.parquet",

    # global post-gate tails (can be outside window, by design)
    "tail_global_postgate_q90": "/mnt/g/STAR_HPV/results/anchor_catalogs/kepler_ripe_tail_global_postgate_q90.parquet",
    "tail_global_postgate_q95": "/mnt/g/STAR_HPV/results/anchor_catalogs/kepler_ripe_tail_global_postgate_q95.parquet",

    # window-only tails
    "tail_window_q90": "/mnt/g/STAR_HPV/results/anchor_catalogs/kepler_ripe_tail_window_q90.parquet",
    "tail_window_q95": "/mnt/g/STAR_HPV/results/anchor_catalogs/kepler_ripe_tail_window_q95.parquet",

    # window-only per-bin tails
    "tail_window_perbin_q90": "/mnt/g/STAR_HPV/results/anchor_catalogs/kepler_ripe_tail_window_perbin_q90.parquet",
    "tail_window_perbin_q95": "/mnt/g/STAR_HPV/results/anchor_catalogs/kepler_ripe_tail_window_perbin_q95.parquet",
}

OUT_DIR = Path("/mnt/g/STAR_HPV/processed/ages_merged")
OUT_YAML = Path("/mnt/g/STAR_HPV/results/ripeness/apokasc3_overlap_by_kic.yaml")
OUT_MATCHES = Path("/mnt/g/STAR_HPV/results/ripeness/apokasc3_overlap_by_kic_matches.csv")


def _to_int_kic(s: pd.Series) -> pd.Series:
    # KIC should be integer-like; keep as Int64 to allow NA.
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _age_stats(x: pd.Series) -> dict:
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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_YAML.parent.mkdir(parents=True, exist_ok=True)

    ages = pd.read_parquet(AGE_KIC)
    if "kic" not in ages.columns:
        raise RuntimeError(f"Missing 'kic' in {AGE_KIC}. Found: {list(ages.columns)[:20]}")

    ages = ages.copy()
    ages["kic"] = _to_int_kic(ages["kic"])
    ages = ages.dropna(subset=["kic"]).drop_duplicates(subset=["kic"])
    ages_kic_set = set(ages["kic"].astype(int).tolist())

    summary = {
        "age_file_kic": AGE_KIC,
        "ages_rows": int(len(ages)),
        "ages_unique_kic": int(ages["kic"].nunique()),
        "ages_age_stats": _age_stats(ages["age_gyr"]),
        "targets": {},
        "notes": [
            "Merged by KIC, not Gaia source_id.",
            "If overlap remains tiny, it's likely a real sample-selection mismatch (rotator dwarfs vs APOKASC evolved stars).",
        ],
    }

    matches_rows = []

    for name, path in TARGETS.items():
        df = pd.read_parquet(path)

        # find KIC column
        kic_col = None
        for cand in ["kic", "KIC", "kic_id", "kic_kepler"]:
            if cand in df.columns:
                kic_col = cand
                break
        if kic_col is None:
            # some of your catalogs might only have source_id; skip gracefully
            summary["targets"][name] = {
                "input_path": path,
                "input_rows": int(len(df)),
                "kic_col": None,
                "rows_with_age": 0,
                "coverage": 0.0,
                "note": "No KIC column present; cannot merge by KIC.",
            }
            continue

        df = df.copy()
        df["kic_tmp"] = _to_int_kic(df[kic_col])
        before = len(df)

        merged = df.merge(ages, left_on="kic_tmp", right_on="kic", how="left", suffixes=("", "_age"))
        rows_with_age = int(merged["age_gyr"].notna().sum())
        cov = float(rows_with_age / max(1, before))

        out_path = OUT_DIR / f"{name}_with_apokasc3_ages_by_kic.parquet"
        merged.to_parquet(out_path, index=False)

        # collect actual matches for a quick look
        mm = merged.loc[merged["age_gyr"].notna(), ["source_id"] + ([kic_col] if kic_col != "source_id" else [])].copy()
        if kic_col != "kic_tmp":
            mm["kic"] = merged.loc[merged["age_gyr"].notna(), "kic_tmp"].astype("Int64").values
        else:
            mm["kic"] = merged.loc[merged["age_gyr"].notna(), "kic_tmp"].astype("Int64").values
        mm["catalog"] = name
        mm["age_gyr"] = merged.loc[merged["age_gyr"].notna(), "age_gyr"].values
        matches_rows.append(mm)

        # extra: how many KICs even intersect (set-based)
        df_kic_set = set(merged["kic_tmp"].dropna().astype(int).unique().tolist())
        set_overlap = len(df_kic_set & ages_kic_set)

        summary["targets"][name] = {
            "input_path": path,
            "input_rows": int(before),
            "kic_col": kic_col,
            "kic_set_overlap": int(set_overlap),
            "rows_with_age": rows_with_age,
            "coverage": cov,
            "age_stats_in_target": _age_stats(merged.loc[merged["age_gyr"].notna(), "age_gyr"]),
            "out_parquet": str(out_path),
        }

    if matches_rows:
        matches = pd.concat(matches_rows, ignore_index=True)
        matches.to_csv(OUT_MATCHES, index=False)
        summary["matches_csv"] = str(OUT_MATCHES)
        summary["matches_rows_total"] = int(len(matches))
    else:
        summary["matches_csv"] = None
        summary["matches_rows_total"] = 0

    with open(OUT_YAML, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    print(f"saved: {OUT_YAML}")
    if summary["matches_csv"]:
        print(f"saved: {OUT_MATCHES}")
    print("done.")


if __name__ == "__main__":
    main()
