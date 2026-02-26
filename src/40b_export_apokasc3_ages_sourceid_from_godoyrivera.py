#!/usr/bin/env python3
"""
Export APOKASC ages to Gaia source_id using Godoy-Rivera+2025 Kepler-Gaia DR3 crossmatch.

Inputs:
- /mnt/g/STAR_HPV/raw/ages/apokasc3_kic_ages_true.parquet
- /mnt/g/STAR_HPV/raw/kepler/godoyrivera25_tableA1.csv  (cols: KIC, GaiaDR3)

Outputs:
- /mnt/g/STAR_HPV/raw/ages/apokasc3_ages_sourceid_agegyr.parquet
- /mnt/g/STAR_HPV/raw/ages/apokasc3_ages_sourceid_agegyr.csv
- /mnt/g/STAR_HPV/raw/ages/apokasc3_ages_sourceid_summary.yaml
"""

from __future__ import annotations
import os
import yaml
import pandas as pd

AGES_IN = "/mnt/g/STAR_HPV/raw/ages/apokasc3_kic_ages_true.parquet"
XM_IN   = "/mnt/g/STAR_HPV/raw/kepler/godoyrivera25_tableA1.csv"

OUT_PARQ = "/mnt/g/STAR_HPV/raw/ages/apokasc3_ages_sourceid_agegyr.parquet"
OUT_CSV  = "/mnt/g/STAR_HPV/raw/ages/apokasc3_ages_sourceid_agegyr.csv"
SUM      = "/mnt/g/STAR_HPV/raw/ages/apokasc3_ages_sourceid_summary.yaml"

def main() -> None:
    os.makedirs(os.path.dirname(OUT_PARQ), exist_ok=True)

    ages = pd.read_parquet(AGES_IN)
    ages["kic"] = pd.to_numeric(ages["kic"], errors="coerce").astype("Int64")

    x = pd.read_csv(XM_IN, usecols=["KIC","GaiaDR3"], low_memory=False, dtype={"GaiaDR3":"string"})
    x["KIC"] = pd.to_numeric(x["KIC"], errors="coerce").astype("Int64")
    x["source_id"] = x["GaiaDR3"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)

    j = x.merge(ages, left_on="KIC", right_on="kic", how="inner")
    j = j.dropna(subset=["source_id"])
    j = j.drop_duplicates("source_id", keep="first")

    out = j[["source_id","age_gyr","age_gyr_err","age_source","age_quality_flag","KIC"]].rename(columns={"KIC":"kic"}).copy()
    out.to_parquet(OUT_PARQ, index=False)
    out.to_csv(OUT_CSV, index=False)

    summary = {
        "ages_kic_rows": int(len(ages)),
        "xmatch_rows": int(len(x)),
        "matched_rows": int(len(out)),
        "unique_source_id_with_age": int(out["source_id"].nunique()),
        "out_parquet": OUT_PARQ,
        "out_csv": OUT_CSV,
        "note": "This uses Godoy-Rivera+2025 Kepler-Gaia DR3 crossmatch; coverage should be far larger than Prot-only maps.",
    }
    with open(SUM, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    print(f"saved: {OUT_PARQ} rows={len(out)} unique_source_id={out['source_id'].nunique()}")
    print(f"summary: {SUM}")

if __name__ == "__main__":
    main()
