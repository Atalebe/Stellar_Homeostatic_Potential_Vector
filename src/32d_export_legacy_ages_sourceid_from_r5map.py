import os
import pandas as pd

AGES_KIC = "/mnt/g/STAR_HPV/raw/ages/legacy_2017_kic_ages.csv"
MAP = "/mnt/g/STAR_HPV/raw/ages/legacy_2017_kic_to_gaia_sourceid_string_r5arcsec.csv"
OUT = "/mnt/g/STAR_HPV/raw/ages/kepler_ages_sourceid_agegyr.csv"

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    ages = pd.read_csv(AGES_KIC)
    ages["kic"] = pd.to_numeric(ages["kic"], errors="coerce").astype("Int64")
    ages["age_gyr"] = pd.to_numeric(ages["age_gyr"], errors="coerce")
    ages["age_gyr_err"] = pd.to_numeric(ages.get("age_gyr_err"), errors="coerce")
    ages = ages.dropna(subset=["kic","age_gyr"]).copy()

    mp = pd.read_csv(MAP, dtype={"source_id":"string"}, low_memory=False)
    mp["kic"] = pd.to_numeric(mp["kic"], errors="coerce").astype("Int64")
    mp["source_id"] = mp["source_id"].astype("string").str.strip()
    mp = mp[(mp["status"]=="ok") & mp["source_id"].notna() & (mp["source_id"]!="")].copy()
    mp = mp.drop_duplicates(subset=["kic"])

    df = ages.merge(mp[["kic","source_id"]], on="kic", how="inner").copy()
    df["age_source"] = "Kepler LEGACY asteroseismology (Silva Aguirre+ 2017)"
    df["age_quality_flag"] = 1

    df = df[["source_id","age_gyr","age_gyr_err","age_source","age_quality_flag"]].copy()
    df.to_csv(OUT, index=False)

    print("saved:", OUT, "rows:", len(df))
    print(df.head(10))

if __name__ == "__main__":
    main()
