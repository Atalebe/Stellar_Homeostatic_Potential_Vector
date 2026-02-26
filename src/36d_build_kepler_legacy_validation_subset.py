import os, yaml
import pandas as pd
from scipy.stats import kendalltau

AGES = "/mnt/g/STAR_HPV/raw/ages/kepler_ages_sourceid_agegyr.csv"
KEP = "/mnt/g/STAR_HPV/processed/state_vectors/kepler_led_state_vector_gmmgate.parquet"

OUT_PARQ = "/mnt/g/STAR_HPV/processed/ages_merged/kepler_legacy_validation_subset.parquet"
OUT_YAML = "/mnt/g/STAR_HPV/results/ripeness/kepler_legacy_validation_subset.yaml"

def norm_sid(s):
    return s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)

def main():
    os.makedirs(os.path.dirname(OUT_PARQ), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_YAML), exist_ok=True)

    ages = pd.read_csv(AGES, dtype={"source_id":"string"})
    ages["source_id"] = norm_sid(ages["source_id"])
    ages["age_gyr"] = pd.to_numeric(ages["age_gyr"], errors="coerce")
    ages = ages.dropna(subset=["source_id","age_gyr"]).drop_duplicates(subset=["source_id"]).copy()

    df = pd.read_parquet(KEP)
    df["source_id"] = norm_sid(df["source_id"])

    sub = df.merge(ages, on="source_id", how="inner")
    sub.to_parquet(OUT_PARQ, index=False)

    out = {"rows": int(len(sub))}

    if len(sub) >= 5:
        # pooled correlations
        tau1, p1 = kendalltau(sub["age_gyr"], sub["prot_days"])
        tau2, p2 = kendalltau(sub["age_gyr"], sub["Phi_gmm"])
        tau3, p3 = kendalltau(sub["age_gyr"], sub["ripeness_rot"])
        out["pooled"] = {
            "kendall_tau_age_vs_prot": float(tau1),
            "p": float(p1),
            "kendall_tau_age_vs_phi": float(tau2),
            "p2": float(p2),
            "kendall_tau_age_vs_ripeness": float(tau3),
            "p3": float(p3),
        }

    with open(OUT_YAML, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)

    print("saved:", OUT_PARQ)
    print("saved:", OUT_YAML)
    print(out)

if __name__ == "__main__":
    main()
