#!/usr/bin/env python3
"""
Build KIC->age table from APOKASC3 using a REAL age column (not an E_* error column).

Rules:
- Prefer columns that do NOT start with 'E_'.
- Prefer columns whose median is in a sensible Gyr range (e.g. 0.2..20).
- Use matching E_* column as age_gyr_err when available.
- Drop sentinel values like 99.
"""

from __future__ import annotations
import os
import yaml
import numpy as np
import pandas as pd

INP  = "/mnt/g/STAR_HPV/raw/ages/apokasc3_table5_clean.parquet"
DIAG = "/mnt/g/STAR_HPV/raw/ages/apokasc3_age_column_diagnostics.yaml"

OUT_CSV  = "/mnt/g/STAR_HPV/raw/ages/apokasc3_kic_ages_true.parquet.csv"  # not used
OUT_PARQ = "/mnt/g/STAR_HPV/raw/ages/apokasc3_kic_ages_true.parquet"      # overwritten on purpose
OUT_CSV2 = "/mnt/g/STAR_HPV/raw/ages/apokasc3_kic_ages_true.csv"
SUM      = "/mnt/g/STAR_HPV/raw/ages/apokasc3_kic_ages_true_summary.yaml"

def choose_age_column(diag: dict) -> str:
    top = diag.get("top_plausible_age_columns", [])
    if not top:
        raise RuntimeError("No plausible age columns found in diagnostics.")
    # Prefer non-E_ columns
    non_e = [r for r in top if not str(r["col"]).startswith("E_")]
    if non_e:
        return non_e[0]["col"]
    # fallback
    return top[0]["col"]

def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)

def main() -> None:
    os.makedirs(os.path.dirname(OUT_PARQ), exist_ok=True)

    with open(DIAG, "r") as f:
        diag = yaml.safe_load(f)

    age_col = choose_age_column(diag)
    err_col = f"E_{age_col}" if f"E_{age_col}" in diag.get("candidates", {}) else None

    df = pd.read_parquet(INP)
    df["KIC"] = pd.to_numeric(df["KIC"], errors="coerce").astype("Int64")

    age = to_num(df[age_col])
    err = to_num(df[err_col]) if (err_col and err_col in df.columns) else pd.Series(np.nan, index=df.index)

    # Drop obvious sentinels
    age = age.mask(age >= 90, np.nan)   # 99 shows up as sentinel in your stats
    err = err.mask(err >= 90, np.nan)

    # Basic sanity
    age = age.mask(age <= 0, np.nan)
    age = age.mask(age > 20, np.nan)    # keep within cosmically sane bounds

    x = df.loc[df["KIC"].notna()].copy()
    x["age_gyr"] = age
    x["age_gyr_err"] = err

    x = x.loc[np.isfinite(x["age_gyr"])].copy()
    x = x.drop_duplicates("KIC", keep="first")

    out = pd.DataFrame({
        "kic": x["KIC"].astype("Int64"),
        "age_gyr": x["age_gyr"].astype(float),
        "age_gyr_err": x["age_gyr_err"].astype(float),
        "age_source": f"APOKASC-3 (VizieR table5), age_gyr column={age_col}",
        "age_quality_flag": 1,
    })

    out.to_parquet(OUT_PARQ, index=False)
    out.to_csv(OUT_CSV2, index=False)

    summary = {
        "age_column_used": age_col,
        "age_err_column_used": err_col,
        "rows_out": int(len(out)),
        "unique_kic": int(out["kic"].nunique()),
        "age_gyr_range": [float(out["age_gyr"].min()), float(out["age_gyr"].max())] if len(out) else None,
        "out_parquet": OUT_PARQ,
        "out_csv": OUT_CSV2,
        "note": "Non-E_* column used as age; E_* column used as uncertainty when present. Sentinel 99 removed.",
    }
    with open(SUM, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    print(f"saved: {OUT_PARQ} rows={len(out)}")
    print("age col:", age_col, "err col:", err_col)
    print("age range:", summary["age_gyr_range"])
    print("summary:", SUM)

if __name__ == "__main__":
    main()
