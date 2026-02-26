#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

LEGACY_AGE_SID = "/mnt/g/STAR_HPV/raw/ages/kepler_ages_sourceid_agegyr.csv"  # from r5 map export
LEGACY_R5MAP = "/mnt/g/STAR_HPV/raw/ages/legacy_2017_kic_to_gaia_sourceid_string_r5arcsec.csv"
MCQ = "/mnt/g/STAR_HPV/raw/kepler/mcquillan2014_rotators.parquet"

OUT_DIR = Path("/mnt/g/STAR_HPV/processed/gyro")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PARQ = OUT_DIR / "legacy_mcquillan_gyro_calibration.parquet"
OUT_YAML = OUT_DIR / "legacy_mcquillan_gyro_calibration_summary.yaml"

def main():
    ages = pd.read_csv(LEGACY_AGE_SID, dtype={"source_id": "string"})
    r5 = pd.read_csv(LEGACY_R5MAP, dtype={"source_id": "string"})
    mcq = pd.read_parquet(MCQ)

    # Normalize types
    r5["kic"] = pd.to_numeric(r5["kic"], errors="coerce").astype("Int64")
    mcq["kic"] = pd.to_numeric(mcq["kic"], errors="coerce").astype("Int64")

    # Join: legacy ages (by source_id) -> r5 (source_id,kic) -> mcquillan (kic,prot,Teff)
    tmp = r5.merge(ages, on="source_id", how="inner")
    cal = tmp.merge(mcq[["kic", "prot_days", "Teff"]], on="kic", how="inner")

    # Clean
    cal["prot_days"] = pd.to_numeric(cal["prot_days"], errors="coerce").astype(float)
    cal["Teff"] = pd.to_numeric(cal["Teff"], errors="coerce").astype(float)
    cal["age_gyr"] = pd.to_numeric(cal["age_gyr"], errors="coerce").astype(float)

    cal = cal.replace([np.inf, -np.inf], np.nan).dropna(subset=["prot_days", "Teff", "age_gyr"])
    cal = cal[(cal["prot_days"] > 0) & (cal["age_gyr"] > 0) & (cal["Teff"] > 0)]

    cal.to_parquet(OUT_PARQ, index=False)

    summ = {
        "rows_legacy_ages": int(len(ages)),
        "rows_r5map": int(len(r5)),
        "rows_mcquillan": int(len(mcq)),
        "rows_calibration": int(len(cal)),
        "prot_days_stats": {
            "min": float(cal["prot_days"].min()) if len(cal) else None,
            "p50": float(cal["prot_days"].median()) if len(cal) else None,
            "max": float(cal["prot_days"].max()) if len(cal) else None,
        },
        "age_gyr_stats": {
            "min": float(cal["age_gyr"].min()) if len(cal) else None,
            "p50": float(cal["age_gyr"].median()) if len(cal) else None,
            "max": float(cal["age_gyr"].max()) if len(cal) else None,
        },
        "out": str(OUT_PARQ),
        "note": "This is LEGACY asteroseismic ages intersected with McQuillan rotation periods (not gmmgate). Used to calibrate gyro ages for dwarfs."
    }
    with open(OUT_YAML, "w") as f:
        yaml.safe_dump(summ, f, sort_keys=False)

    print(f"saved: {OUT_PARQ}")
    print(f"summary: {OUT_YAML}")
    print(summ)

if __name__ == "__main__":
    main()
