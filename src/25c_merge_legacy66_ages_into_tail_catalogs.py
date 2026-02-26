import os
import yaml
import pandas as pd

AGE_IN = "/mnt/g/STAR_HPV/raw/ages/kepler_ages_sourceid_agegyr.csv"
OUT_DIR = "/mnt/g/STAR_HPV/processed/ages_merged"
OUT_YAML = "/mnt/g/STAR_HPV/processed/ages_merged/legacy66_age_merge_summary.yaml"

TARGETS = {
    "kepler_gmmgate": "/mnt/g/STAR_HPV/processed/state_vectors/kepler_led_state_vector_gmmgate.parquet",
    "tail_global_postgate_q90": "/mnt/g/STAR_HPV/results/anchor_catalogs/kepler_ripe_tail_global_postgate_q90.parquet",
    "tail_global_postgate_q95": "/mnt/g/STAR_HPV/results/anchor_catalogs/kepler_ripe_tail_global_postgate_q95.parquet",
    "tail_window_q90": "/mnt/g/STAR_HPV/results/anchor_catalogs/kepler_ripe_tail_window_q90.parquet",
    "tail_window_q95": "/mnt/g/STAR_HPV/results/anchor_catalogs/kepler_ripe_tail_window_q95.parquet",
    "tail_window_perbin_q90": "/mnt/g/STAR_HPV/results/anchor_catalogs/kepler_ripe_tail_window_perbin_q90.parquet",
    "tail_window_perbin_q95": "/mnt/g/STAR_HPV/results/anchor_catalogs/kepler_ripe_tail_window_perbin_q95.parquet",
}

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def norm_sid(s):
    return s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)

def main():
    ensure_dir(OUT_DIR)

    ages = pd.read_csv(AGE_IN, dtype={"source_id": "string"})
    ages["source_id"] = norm_sid(ages["source_id"])
    ages = ages.dropna(subset=["source_id"]).copy()

    summary = {
        "age_file": AGE_IN,
        "ages_rows": int(len(ages)),
        "targets": {},
        "note": "LEGACY66 ages are small-N; coverage is expected to be tiny for rotator-led samples."
    }

    for name, path in TARGETS.items():
        if not os.path.exists(path):
            summary["targets"][name] = {"error": f"missing {path}"}
            continue

        df = pd.read_parquet(path)
        if "source_id" not in df.columns:
            summary["targets"][name] = {"error": "no source_id column"}
            continue

        df["source_id"] = norm_sid(df["source_id"])
        merged = df.merge(ages, on="source_id", how="left", suffixes=("", "_age"))

        outp = f"{OUT_DIR}/{name}_with_legacy66_ages.parquet"
        merged.to_parquet(outp, index=False)

        n = int(len(merged))
        n_age = int(merged["age_gyr"].notna().sum()) if "age_gyr" in merged.columns else 0
        cov = float(n_age / n) if n else 0.0

        bybin = {}
        bincol = "teff_bin_s" if "teff_bin_s" in merged.columns else None
        if bincol and n_age:
            g = merged[merged["age_gyr"].notna()].groupby(bincol)["source_id"].count()
            bybin = {str(k): int(v) for k, v in g.to_dict().items()}

        # also export matched rows as CSV for eyeballing
        matched_csv = f"{OUT_DIR}/{name}_legacy66_matches.csv"
        if n_age:
            merged.loc[merged["age_gyr"].notna()].to_csv(matched_csv, index=False)
        else:
            matched_csv = None

        summary["targets"][name] = {
            "input_rows": n,
            "rows_with_age": n_age,
            "coverage": cov,
            "by_teff_bin_with_age": bybin,
            "out_parquet": outp,
            "out_matched_csv": matched_csv,
        }

    with open(OUT_YAML, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    print("saved:", OUT_YAML)
    for k, v in summary["targets"].items():
        if "error" in v:
            print(k, "ERROR", v["error"])
        else:
            print(k, "coverage", f"{v['coverage']:.6f}", "rows_with_age", v["rows_with_age"])

if __name__ == "__main__":
    main()
