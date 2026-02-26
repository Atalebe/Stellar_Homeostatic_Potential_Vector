#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

INP = "/mnt/g/STAR_HPV/processed/state_vectors/kepler_led_state_vector_gmmgate.parquet"
MODEL = "/mnt/g/STAR_HPV/results/gyro/gyro_model_teff_fit.yaml"

OUT_DIR = Path("/mnt/g/STAR_HPV/processed/gyro")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PARQ = OUT_DIR / "kepler_gmmgate_with_gyro_ages.parquet"
OUT_YAML = OUT_DIR / "kepler_gmmgate_with_gyro_ages_summary.yaml"

def main():
    df = pd.read_parquet(INP)

    with open(MODEL, "r") as f:
        m = yaml.safe_load(f)

    a = float(m["coeffs"]["a"])
    b = float(m["coeffs"]["b"])
    c = float(m["coeffs"]["c"])

    # Required inputs
    if "prot_days" not in df.columns or "Teff" not in df.columns:
        raise RuntimeError(f"Missing required columns in {INP}. Need prot_days and Teff. Have: {list(df.columns)[:40]}")

    P = pd.to_numeric(df["prot_days"], errors="coerce").astype(float)
    Teff = pd.to_numeric(df["Teff"], errors="coerce").astype(float)

    # Solve for age:
    # log10(P) = a + b*log10(t) + c*log10(Teff/5777)
    # => log10(t) = (log10(P) - a - c*log10(Teff/5777))/b
    logP = np.log10(P)
    logTe = np.log10(Teff / 5777.0)
    logt = (logP - a - c * logTe) / b
    age_gyr = 10 ** logt

    # Mask nonsense
    age_gyr[(~np.isfinite(age_gyr)) | (age_gyr <= 0) | (age_gyr > 25)] = np.nan

    df["age_gyr_gyro"] = age_gyr

    # Updated ripeness definition (time-weighted)
    if "Phi_gmm" in df.columns:
        df["ripeness_time_gyr"] = df["Phi_gmm"].astype(float) * df["age_gyr_gyro"].astype(float)
    else:
        df["ripeness_time_gyr"] = np.nan

    df.to_parquet(OUT_PARQ, index=False)

    cov = float(df["age_gyr_gyro"].notna().mean())

    summ = {
        "input": INP,
        "rows": int(len(df)),
        "age_coverage": cov,
        "age_stats_gyr": {
            "n": int(df["age_gyr_gyro"].notna().sum()),
            "min": float(np.nanmin(df["age_gyr_gyro"])) if df["age_gyr_gyro"].notna().any() else None,
            "p50": float(np.nanmedian(df["age_gyr_gyro"])) if df["age_gyr_gyro"].notna().any() else None,
            "p90": float(np.nanpercentile(df["age_gyr_gyro"], 90)) if df["age_gyr_gyro"].notna().any() else None,
            "max": float(np.nanmax(df["age_gyr_gyro"])) if df["age_gyr_gyro"].notna().any() else None,
        },
        "model_used": m["model"],
        "coeffs": m["coeffs"],
        "out": str(OUT_PARQ),
        "note": "Gyro ages are model-calibrated from LEGACY∩McQuillan and should be interpreted as a relative age scale for your dwarf regime."
    }

    with open(OUT_YAML, "w") as f:
        yaml.safe_dump(summ, f, sort_keys=False)

    print(f"saved: {OUT_PARQ}")
    print(f"summary: {OUT_YAML}")
    print(summ)

if __name__ == "__main__":
    main()
