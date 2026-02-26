import os
import pandas as pd

AGE_IN = "/mnt/g/STAR_HPV/raw/ages/kepler_ages_sourceid_agegyr.csv"  # you will provide this
GMM_IN = "/mnt/g/STAR_HPV/processed/state_vectors/kepler_led_state_vector_gmmgate.parquet"
ANCH_IN = "/mnt/g/STAR_HPV/results/anchor_catalogs/kepler_anchor_catalog.parquet"
TAIL90_IN = "/mnt/g/STAR_HPV/results/anchor_catalogs/kepler_ripe_tail_q90.parquet"
TAIL95_IN = "/mnt/g/STAR_HPV/results/anchor_catalogs/kepler_ripe_tail_q95.parquet"

OUT_DIR = "/mnt/g/STAR_HPV/processed/ages_merged"

def merge_one(inp, outp, ages):
    df = pd.read_parquet(inp)
    df["source_id"] = pd.to_numeric(df["source_id"], errors="coerce").astype("Int64")
    out = df.merge(ages, on="source_id", how="left", validate="m:1")
    out.to_parquet(outp, index=False)
    return len(out), float(out["age_gyr"].notna().mean())

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    ages = pd.read_csv(AGE_IN, dtype={"source_id":"string"})
    ages["source_id"] = ages["source_id"].astype("string").str.strip().str.replace(r"\.0$","",regex=True)
    ages["source_id"] = pd.to_numeric(ages["source_id"], errors="coerce").astype("Int64")
    ages["age_gyr"] = pd.to_numeric(ages["age_gyr"], errors="coerce")
    ages = ages.dropna(subset=["source_id","age_gyr"]).drop_duplicates(subset=["source_id"]).copy()

    targets = [
        (GMM_IN, os.path.join(OUT_DIR, "kepler_gmmgate_with_ages.parquet")),
        (ANCH_IN, os.path.join(OUT_DIR, "kepler_anchor_with_ages.parquet")),
        (TAIL90_IN, os.path.join(OUT_DIR, "kepler_ripe_tail_q90_with_ages.parquet")),
        (TAIL95_IN, os.path.join(OUT_DIR, "kepler_ripe_tail_q95_with_ages.parquet")),
    ]

    for inp, outp in targets:
        n, frac = merge_one(inp, outp, ages)
        print(f"merged ages: {inp} -> {outp} rows={n} age_coverage={frac:.3f}")

if __name__ == "__main__":
    main()
