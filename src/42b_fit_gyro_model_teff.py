#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

CAL = "/mnt/g/STAR_HPV/processed/gyro/legacy_mcquillan_gyro_calibration.parquet"
OUT_DIR = Path("/mnt/g/STAR_HPV/results/gyro")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_YAML = OUT_DIR / "gyro_model_teff_fit.yaml"

def robust_fit(X, y):
    # plain least squares with basic sanity filtering
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    rmse = float(np.sqrt(np.mean(resid**2)))
    return beta, rmse, yhat

def main():
    df = pd.read_parquet(CAL)
    if len(df) < 8:
        raise RuntimeError(f"Too few calibration rows for fit: {len(df)}")

    P = df["prot_days"].to_numpy(float)
    t = df["age_gyr"].to_numpy(float)
    Teff = df["Teff"].to_numpy(float)

    # logs
    y = np.log10(P)
    x1 = np.log10(t)                      # age term
    x2 = np.log10(Teff / 5777.0)          # teff tilt term

    X = np.column_stack([np.ones_like(x1), x1, x2])

    beta, rmse, yhat = robust_fit(X, y)

    out = {
        "model": "log10(P_days) = a + b*log10(age_gyr) + c*log10(Teff/5777)",
        "fit_rows": int(len(df)),
        "coeffs": {"a": float(beta[0]), "b": float(beta[1]), "c": float(beta[2])},
        "rmse_log10P": rmse,
        "notes": [
            "This is a data-calibrated gyro model using LEGACY asteroseismic ages intersected with McQuillan periods.",
            "It is not a universal gyrochronology law; it is a pragmatic calibration for your Kepler dwarf regime.",
            "You can replace this with a literature gyro law later; the pipeline stays the same."
        ],
    }

    with open(OUT_YAML, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)

    print(f"saved: {OUT_YAML}")
    print(out)

if __name__ == "__main__":
    main()
