import os
import pandas as pd

INP = "/mnt/g/STAR_HPV/raw/ages/legacy_2017_kic_to_gaia_sourceid_fixed.csv"
OUT = "/mnt/g/STAR_HPV/interim/legacy/legacy_gaia_ids_fixed.csv"

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df = pd.read_csv(INP, dtype={"source_id":"string"})
    df["source_id"] = df["source_id"].astype("string").str.strip()
    df = df.dropna(subset=["source_id"]).drop_duplicates(subset=["source_id"]).copy()
    df[["source_id"]].to_csv(OUT, index=False)
    print("saved:", OUT, "rows:", len(df))

if __name__ == "__main__":
    main()
