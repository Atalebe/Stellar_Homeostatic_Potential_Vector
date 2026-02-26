import os
import pandas as pd

INP = "/mnt/g/STAR_HPV/processed/state_vectors/kepler_led_state_vector_gmmgate.parquet"
OUT = "/mnt/g/STAR_HPV/raw/ages/kepler_age_template_sourceid.csv"

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df = pd.read_parquet(INP, columns=["source_id", "kic", "teff_bin_s", "prot_days"]).copy()

    # Normalize ID type to string for safe exports
    df["source_id"] = df["source_id"].astype("Int64").astype("string")
    df["kic"] = df["kic"].astype("Int64")

    # Unique IDs
    df = df.drop_duplicates(subset=["source_id"]).copy()

    # Empty age columns to be filled later
    df["age_gyr"] = pd.NA
    df["age_gyr_err"] = pd.NA
    df["age_source"] = pd.NA  # e.g. "asteroseismology", "isochrone", etc.
    df["age_quality_flag"] = pd.NA

    df.to_csv(OUT, index=False)
    print("saved:", OUT, "rows:", len(df))
    print(df.head())

if __name__ == "__main__":
    main()
