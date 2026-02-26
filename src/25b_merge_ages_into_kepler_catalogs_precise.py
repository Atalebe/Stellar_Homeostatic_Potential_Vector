import os
import pandas as pd

AGE_IN = "/mnt/g/STAR_HPV/raw/ages/kepler_ages_sourceid_agegyr.csv"

TARGETS = [
    ("/mnt/g/STAR_HPV/processed/state_vectors/kepler_led_state_vector_gmmgate.parquet",
     "/mnt/g/STAR_HPV/processed/ages_merged/kepler_gmmgate_with_ages.parquet"),
    ("/mnt/g/STAR_HPV/results/anchor_catalogs/kepler_anchor_catalog.parquet",
     "/mnt/g/STAR_HPV/processed/ages_merged/kepler_anchor_with_ages.parquet"),
    ("/mnt/g/STAR_HPV/results/anchor_catalogs/kepler_ripe_tail_q90.parquet",
     "/mnt/g/STAR_HPV/processed/ages_merged/kepler_ripe_tail_q90_with_ages.parquet"),
    ("/mnt/g/STAR_HPV/results/anchor_catalogs/kepler_ripe_tail_q95.parquet",
     "/mnt/g/STAR_HPV/processed/ages_merged/kepler_ripe_tail_q95_with_ages.parquet"),
]

def norm_sid(s):
    s = s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return s

def main():
    os.makedirs("/mnt/g/STAR_HPV/processed/ages_merged", exist_ok=True)

    ages = pd.read_csv(AGE_IN, dtype={"source_id":"string"})
    ages["source_id"] = norm_sid(ages["source_id"])
    ages["age_gyr"] = pd.to_numeric(ages["age_gyr"], errors="coerce")
    ages = ages.dropna(subset=["source_id","age_gyr"]).drop_duplicates(subset=["source_id"]).copy()

    ages_set = set(ages["source_id"].tolist())
    print("AGES rows:", len(ages), "unique source_id:", len(ages_set))

    for inp, outp in TARGETS:
        df = pd.read_parquet(inp)
        if "source_id" not in df.columns:
            raise RuntimeError(f"Missing source_id in {inp}")

        df["source_id"] = norm_sid(df["source_id"])
        df_set = set(df["source_id"].dropna().tolist())

        merged = df.merge(ages, on="source_id", how="left")

        n_total = len(merged)
        n_match = int(merged["age_gyr"].notna().sum())
        cov = n_match / n_total if n_total else 0.0

        # diagnostics
        overlap = len(df_set & ages_set)
        print("\n===", inp, "===")
        print("rows:", n_total)
        print("overlap(source_id sets):", overlap)
        print("rows with age_gyr:", n_match)
        print(f"coverage: {cov:.6f}")

        # show some missing age ids that should have matched if overlap>0
        if overlap > 0 and n_match == 0:
            sample = list(df_set & ages_set)[:10]
            print("WARNING: overlap exists but merge produced 0 matches. sample:", sample)

        merged.to_parquet(outp, index=False)
        print("saved:", outp)

if __name__ == "__main__":
    main()
