import os, yaml
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

INP = "/mnt/g/STAR_HPV/processed/ages_merged/kepler_gmmgate_with_ages.parquet"
OUT = "/mnt/g/STAR_HPV/results/ripeness/kepler_age_recalibration.yaml"

BINS = ["3200-4000","4000-5200","5200-6000","6000-7500"]

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df = pd.read_parquet(INP)

    # Ensure required columns exist
    for c in ["teff_bin_s","age_gyr","prot_days","Phi_gmm"]:
        if c not in df.columns:
            raise RuntimeError(f"Missing {c} in {INP}")

    # If ripeness_rot missing, compute it
    if "ripeness_rot" not in df.columns:
        df["prot_days"] = pd.to_numeric(df["prot_days"], errors="coerce")
        df["Phi_gmm"] = pd.to_numeric(df["Phi_gmm"], errors="coerce")
        df["ripeness_rot"] = df["prot_days"] * df["Phi_gmm"]

    out = {"bins": {}}

    for tb in BINS:
        sub = df[df["teff_bin_s"].astype(str) == tb].copy()
        sub["age_gyr"] = pd.to_numeric(sub["age_gyr"], errors="coerce")
        sub = sub.dropna(subset=["age_gyr","ripeness_rot","Phi_gmm","prot_days"]).copy()
        if len(sub) < 100:
            continue

        tau_r, p_r = kendalltau(sub["age_gyr"], sub["ripeness_rot"])
        tau_phi, p_phi = kendalltau(sub["age_gyr"], sub["Phi_gmm"])
        tau_p, p_p = kendalltau(sub["age_gyr"], sub["prot_days"])

        med_age = float(np.nanmedian(sub["age_gyr"]))
        old = sub[sub["age_gyr"] > med_age]
        young = sub[sub["age_gyr"] <= med_age]

        out["bins"][tb] = {
            "rows": int(len(sub)),
            "kendall_tau_age_vs_ripeness": float(tau_r),
            "p_value_ripeness": float(p_r),
            "kendall_tau_age_vs_phi": float(tau_phi),
            "p_value_phi": float(p_phi),
            "kendall_tau_age_vs_prot": float(tau_p),
            "p_value_prot": float(p_p),
            "median_age_gyr": med_age,
            "median_ripeness_old": float(np.nanmedian(old["ripeness_rot"])) if len(old) else None,
            "median_ripeness_young": float(np.nanmedian(young["ripeness_rot"])) if len(young) else None,
        }

    with open(OUT, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)

    print("saved:", OUT)
    print(out)

if __name__ == "__main__":
    main()
