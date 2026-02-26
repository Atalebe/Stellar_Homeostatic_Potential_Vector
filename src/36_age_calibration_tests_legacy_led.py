import os, yaml
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

INP = "/mnt/g/STAR_HPV/processed/state_vectors/legacy_led_state_vector.parquet"
OUT = "/mnt/g/STAR_HPV/results/ripeness/legacy_led_age_calibration.yaml"

BINS = ["3200-4000","4000-5200","5200-6000","6000-7500"]
MIN_ROWS_BIN = 6  # LEGACY is tiny; keep threshold low but non-trivial

def kt(x, y):
    tau, p = kendalltau(x, y)
    if tau != tau or p != p:
        return None, None
    return float(tau), float(p)

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df = pd.read_parquet(INP).copy()

    # keep only rows with ages
    df["age_gyr"] = pd.to_numeric(df.get("age_gyr"), errors="coerce")
    df = df.dropna(subset=["age_gyr","prot_days","Phi_gmm","ripeness_rot","teff_bin_s"]).copy()

    out = {"pooled": {}, "bins": {}}

    out["pooled"]["rows"] = int(len(df))
    out["pooled"]["kendall_tau_age_vs_prot"], out["pooled"]["p_age_vs_prot"] = kt(df["age_gyr"], df["prot_days"])
    out["pooled"]["kendall_tau_age_vs_phi"], out["pooled"]["p_age_vs_phi"] = kt(df["age_gyr"], df["Phi_gmm"])
    out["pooled"]["kendall_tau_age_vs_ripeness"], out["pooled"]["p_age_vs_ripeness"] = kt(df["age_gyr"], df["ripeness_rot"])

    # Per-bin if enough points
    for tb in BINS:
        sub = df[df["teff_bin_s"].astype(str) == tb]
        if len(sub) < MIN_ROWS_BIN:
            continue
        out["bins"][tb] = {
            "rows": int(len(sub)),
            "kendall_tau_age_vs_prot": kt(sub["age_gyr"], sub["prot_days"])[0],
            "p_age_vs_prot": kt(sub["age_gyr"], sub["prot_days"])[1],
            "kendall_tau_age_vs_phi": kt(sub["age_gyr"], sub["Phi_gmm"])[0],
            "p_age_vs_phi": kt(sub["age_gyr"], sub["Phi_gmm"])[1],
            "kendall_tau_age_vs_ripeness": kt(sub["age_gyr"], sub["ripeness_rot"])[0],
            "p_age_vs_ripeness": kt(sub["age_gyr"], sub["ripeness_rot"])[1],
        }

    with open(OUT, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)

    print("saved:", OUT)
    print(out)

if __name__ == "__main__":
    main()
