#!/usr/bin/env python3
"""
Convert APOKASC KIC ages into Gaia source_id ages using kepler_gaia_ids.csv.

Inputs:
- /mnt/g/STAR_HPV/raw/ages/apokasc3_kic_ages.parquet
- /mnt/g/STAR_HPV/interim/kepler/kepler_gaia_ids.csv

Outputs:
- /mnt/g/STAR_HPV/raw/ages/apokasc3_ages_sourceid_agegyr.csv
- /mnt/g/STAR_HPV/raw/ages/apokasc3_ages_sourceid_agegyr.parquet
- /mnt/g/STAR_HPV/raw/ages/apokasc3_ages_sourceid_summary.yaml
"""

from __future__ import annotations
import os
import yaml
import pandas as pd

AGES_IN = "/mnt/g/STAR_HPV/raw/ages/apokasc3_kic_ages.parquet"
IDMAP = "/mnt/g/STAR_HPV/interim/kepler/kepler_gaia_ids.csv"

OUT_CSV = "/mnt/g/STAR_HPV/raw/ages/apokasc3_ages_sourceid_agegyr.csv"
OUT_PARQ = "/mnt/g/STAR_HPV/raw/ages/apokasc3_ages_sourceid_agegyr.parquet"
SUM = "/mnt/g/STAR_HPV/raw/ages/apokasc3_ages_sourceid_summary.yaml"

def main() -> None:
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    ages = pd.read_parquet(AGES_IN)
    ages["kic"] = pd.to_numeric(ages["kic"], errors="coerce").astype("Int64")

    m = pd.read_csv(IDMAP, dtype={"source_id":"string"}, low_memory=False)
    # normalize columns
    if "kic" not in m.columns:
        raise RuntimeError(f"Missing kic in {IDMAP}. cols={list(m.columns)}")
    m["kic"] = pd.to_numeric(m["kic"], errors="coerce").astype("Int64")
    m["source_id"] = m["source_id"].astype("string").str.strip()

    j = m.merge(ages, on="kic", how="inner")
    j = j.loc[j["source_id"].notna()].copy()

    out = j[["source_id","age_gyr","age_gyr_err","age_source","age_quality_flag","kic"]].copy()
    out = out.drop_duplicates("source_id", keep="first")

    out.to_csv(OUT_CSV, index=False)
    out.to_parquet(OUT_PARQ, index=False)

    summary = {
        "apokasc_kic_ages_rows": int(len(ages)),
        "idmap_rows": int(len(m)),
        "matched_rows": int(len(j)),
        "unique_source_id_with_age": int(out["source_id"].nunique()),
        "out_csv": OUT_CSV,
        "out_parquet": OUT_PARQ,
    }
    with open(SUM, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    print(f"saved: {OUT_PARQ} rows={len(out)} unique_source_id={out['source_id'].nunique()}")
    print(f"summary: {SUM}")

if __name__ == "__main__":
    main()
