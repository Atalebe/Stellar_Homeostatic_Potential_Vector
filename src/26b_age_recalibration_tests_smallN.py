import os, yaml
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

INP = "/mnt/g/STAR_HPV/processed/ages_merged/kepler_gmmgate_with_ages.parquet"
OUT = "/mnt/g/STAR_HPV/results/ripeness/kepler_age_recalibration_smallN.yaml"

BINS = ["3200-4000","4000-5200","5200-6000","6000-7500"]
MIN_ROWS_BIN = 10

def kt(x, y):
    tau, p = kendalltau(x, y)
    if tau != tau or p != p:
        return None, None
    return float(tau), float(p)

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df = pd.read_parquet(INP).copy()

    df["age_gyr"] = pd.to_numeric(df.get("age_gyr"), errors="coerce")
    df["prot_days"] = pd.to_numeric(df.get("prot_days"), errors="coerce")
    df["Phi_gmm"] = pd.to_numeric(df.get("Phi_gmm"), errors="coerce")
    df["teff_bin_s"] = df.get("teff_bin_s", df.get("teff_bin")).astype(str)

    if "ripeness_rot" not in df.columns:
        df["ripeness_rot"] = df["prot_days"] * df["Phi_gmm"]
    df["ripeness_rot"] = pd.to_numeric(df["ripeness_rot"], errors="coerce")

    d = df.dropna(subset=["age_gyr","prot_days","Phi_gmm","ripeness_rot","teff_bin_s"]).copy()
    d = d[d["prot_days"] > 0].copy()

    out = {"pooled": {}, "bins": {}}

    out["pooled"]["rows"] = int(len(d))
    out["pooled"]["kendall_tau_age_vs_prot"], out["pooled"]["p_age_vs_prot"] = kt(d["age_gyr"], d["prot_days"])
    out["pooled"]["kendall_tau_age_vs_phi"], out["pooled"]["p_age_vs_phi"] = kt(d["age_gyr"], d["Phi_gmm"])
    out["pooled"]["kendall_tau_age_vs_ripeness"], out["pooled"]["p_age_vs_ripeness"] = kt(d["age_gyr"], d["ripeness_rot"])

    if len(d) >= 4:
        med_age = float(np.nanmedian(d["age_gyr"]))
        old = d[d["age_gyr"] > med_age]
        young = d[d["age_gyr"] <= med_age]
        out["pooled"]["median_age_gyr"] = med_age
        out["pooled"]["median_ripeness_old"] = float(np.nanmedian(old["ripeness_rot"])) if len(old) else None
        out["pooled"]["median_ripeness_young"] = float(np.nanmedian(young["ripeness_rot"])) if len(young) else None

    for tb in BINS:
        sub = d[d["teff_bin_s"] == tb]
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
