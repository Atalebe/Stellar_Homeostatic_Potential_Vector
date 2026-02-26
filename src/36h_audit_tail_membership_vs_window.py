import pandas as pd
import yaml
import os

KEP = "/mnt/g/STAR_HPV/processed/state_vectors/kepler_led_state_vector_gmmgate.parquet"
Q90 = "/mnt/g/STAR_HPV/results/anchor_catalogs/kepler_ripe_tail_q90.parquet"
Q95 = "/mnt/g/STAR_HPV/results/anchor_catalogs/kepler_ripe_tail_q95.parquet"
OUT = "/mnt/g/STAR_HPV/results/ripeness/tail_window_audit.yaml"

def norm_sid(s):
    return s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    kep = pd.read_parquet(KEP, columns=["source_id","teff_bin_s","in_window_gmm"])
    kep["source_id"] = norm_sid(kep["source_id"])

    q90 = pd.read_parquet(Q90, columns=["source_id"])
    q90["source_id"] = norm_sid(q90["source_id"])

    q95 = pd.read_parquet(Q95, columns=["source_id"])
    q95["source_id"] = norm_sid(q95["source_id"])

    m90 = kep.merge(q90, on="source_id", how="inner")
    m95 = kep.merge(q95, on="source_id", how="inner")

    def pack(df):
        n = len(df)
        n_in = int(df["in_window_gmm"].sum())
        n_out = n - n_in
        by = df.groupby("teff_bin_s")["in_window_gmm"].agg(["count","sum"])
        by = {str(k): {"rows": int(v["count"]), "in_window": int(v["sum"]), "not_in_window": int(v["count"]-v["sum"])}
              for k,v in by.to_dict("index").items()}
        return {"rows": int(n), "in_window": n_in, "not_in_window": n_out, "by_bin": by}

    out = {"q90": pack(m90), "q95": pack(m95)}

    with open(OUT,"w") as f:
        yaml.safe_dump(out, f, sort_keys=False)

    print("saved:", OUT)
    print(out)

if __name__ == "__main__":
    main()
