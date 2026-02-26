import os
import pandas as pd

INP = "/mnt/g/STAR_HPV/raw/ages/legacy_2017_kic_to_gaia_sourceid.csv"
OUT = "/mnt/g/STAR_HPV/raw/ages/legacy_2017_kic_to_gaia_sourceid_fixed.csv"

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    df = pd.read_csv(INP)

    # These are currently floats in scientific notation. Convert safely:
    # read as string then to Int64.
    df["source_id"] = df["source_id"].astype("string").str.strip()
    df["source_id"] = df["source_id"].str.replace(r"\.0$", "", regex=True)

    # If it is in sci-notation, pandas numeric->Int64 can recover if the float kept all digits,
    # but safest is: convert numeric then round then Int64.
    sid_num = pd.to_numeric(df["source_id"], errors="coerce")
    sid_int = sid_num.round().astype("Int64")

    # Drop anything that failed conversion
    df["source_id_int"] = sid_int
    df = df.dropna(subset=["source_id_int"]).copy()

    # KIC also integer
    df["kic"] = pd.to_numeric(df["kic"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["kic"]).copy()

    # Keep only needed columns + preserve diagnostics if present
    keep = ["kic", "source_id_int"]
    for c in ["ra", "dec", "phot_g_mean_mag", "ruwe", "dist_deg"]:
        if c in df.columns:
            keep.append(c)
    df = df[keep].copy()

    # Write source_id as exact string, no sci-notation
    df["source_id"] = df["source_id_int"].astype("Int64").astype("string")
    df = df.drop(columns=["source_id_int"])
    df = df.drop_duplicates(subset=["kic"], keep="first")

    df.to_csv(OUT, index=False)
    print("saved:", OUT, "rows:", len(df))
    print(df.head(10))

if __name__ == "__main__":
    main()
