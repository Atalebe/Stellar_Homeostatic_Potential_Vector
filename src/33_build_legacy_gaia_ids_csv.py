import os
import pandas as pd

INP = "/mnt/g/STAR_HPV/raw/ages/legacy_2017_kic_to_gaia_sourceid.csv"
OUT = "/mnt/g/STAR_HPV/interim/legacy/legacy_gaia_ids.csv"

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df = pd.read_csv(INP)
    df["source_id"] = pd.to_numeric(df["source_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["source_id"]).drop_duplicates(subset=["source_id"]).copy()
    df[["source_id"]].astype("Int64").astype("string").to_csv(OUT, index=False)
    print("saved:", OUT, "rows:", len(df))

if __name__ == "__main__":
    main()
