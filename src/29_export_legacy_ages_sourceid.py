import os
import pandas as pd

KIC_AGES = "/mnt/g/STAR_HPV/raw/ages/legacy_2017_kic_ages.csv"
KIC2GAIA = "/mnt/g/STAR_HPV/interim/kepler/kepler_gaia_ids.csv"
OUT = "/mnt/g/STAR_HPV/raw/ages/kepler_ages_sourceid_agegyr.csv"

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    ages = pd.read_csv(KIC_AGES)
    ages["kic"] = pd.to_numeric(ages["kic"], errors="coerce").astype("Int64")
    ages["age_gyr"] = pd.to_numeric(ages["age_gyr"], errors="coerce")
    ages["age_gyr_err"] = pd.to_numeric(ages["age_gyr_err"], errors="coerce")
    ages = ages.dropna(subset=["kic","age_gyr"]).copy()

    m = pd.read_csv(KIC2GAIA, dtype={"source_id":"string"}).copy()
    m["kic"] = pd.to_numeric(m["kic"], errors="coerce").astype("Int64")
    m["source_id"] = m["source_id"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    m["source_id"] = pd.to_numeric(m["source_id"], errors="coerce").astype("Int64")
    m = m.dropna(subset=["kic","source_id"]).copy()

    out = m.merge(ages, on="kic", how="inner")
    out = out[["source_id","age_gyr","age_gyr_err","age_source","age_quality_flag"]].drop_duplicates(subset=["source_id"]).copy()
    out.to_csv(OUT, index=False)

    print(f"saved: {OUT} rows={len(out)}")
    print(out.head(10))

if __name__ == "__main__":
    main()
