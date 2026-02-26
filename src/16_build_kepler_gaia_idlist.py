import os
import pandas as pd

MCQ = "/mnt/g/STAR_HPV/raw/kepler/mcquillan2014_rotators.parquet"
X   = "/mnt/g/STAR_HPV/raw/kepler/godoyrivera25_tableA1.csv"
OUT = "/mnt/g/STAR_HPV/interim/kepler/kepler_gaia_ids.csv"

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    mcq = pd.read_parquet(MCQ, columns=["kic", "prot_days", "Teff"]).copy()
    mcq["kic"] = pd.to_numeric(mcq["kic"], errors="coerce").astype("Int64")
    mcq = mcq.dropna(subset=["kic"]).copy()

    x = pd.read_csv(
        X,
        usecols=["KIC", "GaiaDR3"],
        dtype={"KIC": "string", "GaiaDR3": "string"},
        low_memory=False,
    ).rename(columns={"KIC": "kic_str", "GaiaDR3": "source_id"})

    # Clean strings
    x["kic_str"] = x["kic_str"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    x["source_id"] = x["source_id"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)

    # Parse numeric KIC exactly
    x["kic"] = pd.to_numeric(x["kic_str"], errors="coerce").astype("Int64")
    x = x.dropna(subset=["kic", "source_id"]).copy()

    # Keep only the required columns (prevents duplicate labels)
    x = x[["kic", "source_id"]].drop_duplicates().copy()

    keep = mcq.merge(x, on="kic", how="inner")
    keep = keep.drop_duplicates(subset=["source_id"]).copy()

    keep.to_csv(OUT, index=False)
    print("saved:", OUT, "rows:", len(keep))
    print(keep.head())

if __name__ == "__main__":
    main()
