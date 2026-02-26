import os, yaml
import pandas as pd

LEG_AGES = "/mnt/g/STAR_HPV/raw/ages/kepler_ages_sourceid_agegyr.csv"
MAP_R5  = "/mnt/g/STAR_HPV/raw/ages/legacy_2017_kic_to_gaia_sourceid_string_r5arcsec.csv"
KEP     = "/mnt/g/STAR_HPV/processed/state_vectors/kepler_led_state_vector_gmmgate.parquet"

OUT = "/mnt/g/STAR_HPV/results/ripeness/legacy_vs_kepler_teffbin_comparison.yaml"

def norm_sid(s):
    return s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)

def frac_by_bin(x, col="teff_bin_s"):
    c = x[col].astype(str).value_counts(dropna=False).to_dict()
    n = len(x)
    f = {k: (v / n if n else 0.0) for k, v in c.items()}
    return {"rows": int(n), "counts": {k:int(v) for k,v in c.items()}, "fractions": f}

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    ages = pd.read_csv(LEG_AGES, dtype={"source_id":"string"})
    ages["source_id"] = norm_sid(ages["source_id"])
    ages = ages.dropna(subset=["source_id"]).drop_duplicates("source_id")

    # map gives KIC + source_id; kepler table gives teff_bin_s
    mp = pd.read_csv(MAP_R5, dtype={"source_id":"string"}, low_memory=False)
    mp = mp[mp["status"].astype(str)=="ok"].copy()
    mp["source_id"] = norm_sid(mp["source_id"])

    legacy_ids = set(ages["source_id"].tolist())
    mp = mp[mp["source_id"].isin(legacy_ids)].copy()

    kep = pd.read_parquet(KEP)
    kep["source_id"] = norm_sid(kep["source_id"])

    # build a "legacy-in-kepler" merged view
    legacy_in_kepler = kep.merge(mp[["source_id","kic"]], on="source_id", how="inner")
    legacy_in_kepler = legacy_in_kepler.merge(ages, on="source_id", how="inner")

    out = {
        "legacy_total_ids": int(len(legacy_ids)),
        "legacy_matched_to_kepler_rows": int(len(legacy_in_kepler)),
        "teffbin_distribution_kepler_gmmgate": frac_by_bin(kep, "teff_bin_s"),
        "teffbin_distribution_legacy_in_kepler": frac_by_bin(legacy_in_kepler, "teff_bin_s"),
        "note": "This compares LEGACY asteroseismic-age targets (that overlap the Kepler GMM-gated sample) to the full Kepler GMM-gated sample, to expose selection bias."
    }

    with open(OUT, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)

    print("saved:", OUT)
    print(out)

if __name__ == "__main__":
    main()
