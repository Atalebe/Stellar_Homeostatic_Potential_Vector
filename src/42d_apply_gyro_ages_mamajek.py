#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

INP = "/mnt/g/STAR_HPV/processed/state_vectors/kepler_led_state_vector_gmmgate.parquet"
OUT_DIR = Path("/mnt/g/STAR_HPV/processed/gyro")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PARQ = OUT_DIR / "kepler_gmmgate_with_gyro_ages_mamajek.parquet"
OUT_YAML = OUT_DIR / "kepler_gmmgate_with_gyro_ages_mamajek_summary.yaml"

# Mamajek & Hillenbrand (2008) coefficients (commonly used)
A = 0.407
B = 0.325
C = 0.495
N = 0.566
AGE_MIN_GYR = 0.1
AGE_MAX_GYR = 13.8
def bp_rp_to_bv(bp_rp: np.ndarray) -> np.ndarray:
    """
    Rough conversion for dwarfs. This is an approximation; good enough for relative gyro ages.
    """
    # simple linear-ish mapping in the dwarf regime
    # clamp to avoid nonsense
    x = np.clip(bp_rp.astype(float), 0.4, 2.0)
    # heuristic: BV ~ 0.5 + 0.6*(BP-RP-0.8)
    bv = 0.5 + 0.6 * (x - 0.8)
    return np.clip(bv, 0.4, 1.6)

def gyro_age_gyr_from_period(P_days: np.ndarray, bv: np.ndarray) -> np.ndarray:
    """
    Returns age in Gyr using the Mamajek-style formula.
    Their 't' is in Myr in the original form; we convert to Gyr.
    """
    P = P_days.astype(float)
    bv = bv.astype(float)

    denom = A * np.power(np.clip(bv - C, 1e-3, None), B)
    t_myr = np.power(P / denom, 1.0 / N)
    t_gyr = t_myr / 1000.0
    t_gyr[(~np.isfinite(t_gyr)) | (t_gyr < AGE_MIN_GYR) | (t_gyr > AGE_MAX_GYR)] = np.nan
    return t_gyr

def main():
    df = pd.read_parquet(INP)

    # Need prot + color
    if "prot_days" not in df.columns:
        raise RuntimeError("prot_days missing in input")
    if "bp_rp" not in df.columns:
        raise RuntimeError("bp_rp missing in input; merge Gaia bp_rp into this parquet first.")

    P = pd.to_numeric(df["prot_days"], errors="coerce").astype(float).to_numpy()
    bprp = pd.to_numeric(df["bp_rp"], errors="coerce").astype(float).to_numpy()

    bv = bp_rp_to_bv(bprp)
    age_gyr = gyro_age_gyr_from_period(P, bv)

    df["bv_est"] = bv
    df["age_gyr_gyro"] = age_gyr

    # time-weighted ripeness (now truly time-based)
    if "Phi_gmm" in df.columns:
        df["ripeness_time_gyr"] = df["Phi_gmm"].astype(float) * df["age_gyr_gyro"].astype(float)
    else:
        df["ripeness_time_gyr"] = np.nan

    df.to_parquet(OUT_PARQ, index=False)

    cov = float(np.nanmean(~np.isnan(age_gyr)))
    stats = df["age_gyr_gyro"].dropna()
    summ = {
        "input": INP,
        "rows": int(len(df)),
        "age_coverage": cov,
        "age_stats_gyr": {
            "n": int(stats.shape[0]),
            "min": float(stats.min()) if len(stats) else None,
            "p50": float(stats.median()) if len(stats) else None,
            "p90": float(stats.quantile(0.9)) if len(stats) else None,
            "max": float(stats.max()) if len(stats) else None,
        },
        "gyro_model": {
            "form": "P = A*(B-V - C)^B * t^N ; t in Myr -> converted to Gyr",
            "A": A, "B": B, "C": C, "N": N,
            "bv_source": "bv_est from Gaia BP-RP (rough conversion, dwarf regime)",
        },
        "out": str(OUT_PARQ),
    }
    with open(OUT_YAML, "w") as f:
        yaml.safe_dump(summ, f, sort_keys=False)

    print(f"saved: {OUT_PARQ}")
    print(f"summary: {OUT_YAML}")
    print(summ)

if __name__ == "__main__":
    main()
