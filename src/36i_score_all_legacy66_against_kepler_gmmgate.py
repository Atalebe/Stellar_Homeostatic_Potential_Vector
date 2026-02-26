import os, yaml
import numpy as np
import pandas as pd

# LEGACY ages (source_id, age_gyr)
LEG_AGES = "/mnt/g/STAR_HPV/raw/ages/kepler_ages_sourceid_agegyr.csv"

# r5 map gives KIC and good Gaia source_id
MAP_R5   = "/mnt/g/STAR_HPV/raw/ages/legacy_2017_kic_to_gaia_sourceid_string_r5arcsec.csv"

# Your Kepler rotator-led post-gate sample
KEP_GMM  = "/mnt/g/STAR_HPV/processed/state_vectors/kepler_led_state_vector_gmmgate.parquet"

OUT_DIR  = "/mnt/g/STAR_HPV/results/ripeness/legacy66_scoring"
OUT_CSV  = f"{OUT_DIR}/legacy66_scored_vs_kepler_gmmgate.csv"
OUT_YAML = f"{OUT_DIR}/legacy66_scored_vs_kepler_gmmgate.yaml"

def norm_sid(s):
    return s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    ages = pd.read_csv(LEG_AGES, dtype={"source_id":"string"})
    ages["source_id"] = norm_sid(ages["source_id"])
    ages["age_gyr"] = pd.to_numeric(ages["age_gyr"], errors="coerce")
    ages = ages.dropna(subset=["source_id","age_gyr"]).drop_duplicates("source_id").copy()

    mp = pd.read_csv(MAP_R5, dtype={"source_id":"string"}, low_memory=False)
    mp = mp[mp["status"].astype(str)=="ok"].copy()
    mp["source_id"] = norm_sid(mp["source_id"])
    mp["kic"] = pd.to_numeric(mp["kic"], errors="coerce").astype("Int64")

    legacy = mp.merge(ages, on="source_id", how="inner")  # should be 66
    legacy = legacy[["source_id","kic","age_gyr","age_gyr_err","dist_deg","ruwe","phot_g_mean_mag"]].copy()

    kep = pd.read_parquet(KEP_GMM)
    kep["source_id"] = norm_sid(kep["source_id"])

    # Keep just what we need from Kepler table
    need = ["source_id","teff_bin_s","prot_days","Phi_gmm","in_window_gmm","gate_days_gmm","gate_logP_gmm"]
    keep = [c for c in need if c in kep.columns]
    kep = kep[keep].copy()

    # Merge: legacy66 left join kepler sample
    m = legacy.merge(kep, on="source_id", how="left", indicator=True)
    m["in_kepler_gmmgate"] = (m["_merge"]=="both")
    m.drop(columns=["_merge"], inplace=True)

    # Compute ripeness where possible
    m["ripeness_gmm"] = pd.to_numeric(m.get("Phi_gmm"), errors="coerce") * pd.to_numeric(m.get("prot_days"), errors="coerce")

    # Simple reasons for non-overlap
    m["reason_if_missing"] = np.where(
        m["in_kepler_gmmgate"],
        "",
        "not_in_kepler_rotator_led_sample (likely no McQuillan Prot / filtered out upstream)"
    )

    # Summary
    n_total = len(m)
    n_in = int(m["in_kepler_gmmgate"].sum())
    out = {
        "legacy_total": int(n_total),
        "legacy_found_in_kepler_gmmgate": int(n_in),
        "legacy_missing_from_kepler_gmmgate": int(n_total - n_in),
        "note": "Missing usually means the star is not in the McQuillan rotator-led pipeline (no Prot), not that Gaia ID is invalid."
    }

    with open(OUT_YAML,"w") as f:
        yaml.safe_dump(out, f, sort_keys=False)

    m.to_csv(OUT_CSV, index=False)

    print("saved:", OUT_CSV)
    print("saved:", OUT_YAML)
    print(out)
    print(m.head(10))

if __name__ == "__main__":
    main()
