import os, yaml
import pandas as pd

KEP = "/mnt/g/STAR_HPV/processed/state_vectors/kepler_led_state_vector_gmmgate.parquet"
Q90 = "/mnt/g/STAR_HPV/results/anchor_catalogs/kepler_ripe_tail_q90.parquet"
Q95 = "/mnt/g/STAR_HPV/results/anchor_catalogs/kepler_ripe_tail_q95.parquet"
OUT = "/mnt/g/STAR_HPV/results/ripeness/tail_window_audit_v2.yaml"

def norm_sid(s):
    return s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)

def pick_bin_col(df):
    # prefer explicit suffixes if present
    for c in ["teff_bin_s_kep", "teff_bin_s_tail", "teff_bin_kep", "teff_bin_tail",
              "teff_bin_s", "teff_bin"]:
        if c in df.columns:
            return c
    return None

def pack(df, col, bin_col):
    n = len(df)
    v = df[col].fillna(False).astype(bool)
    n_in = int(v.sum())
    by = df.groupby(bin_col)[col].apply(lambda x: int(x.fillna(False).astype(bool).sum()))
    ct = df.groupby(bin_col)[col].size()
    out_by = {}
    for b in ct.index:
        rows = int(ct.loc[b])
        inn = int(by.loc[b])
        out_by[str(b)] = {"rows": rows, "in_window": inn, "not_in_window": rows - inn}
    return {"rows": int(n), "in_window": n_in, "not_in_window": int(n - n_in), "by_bin": out_by}

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    kep = pd.read_parquet(KEP, columns=["source_id","teff_bin_s","in_window_gmm"])
    kep["source_id"] = norm_sid(kep["source_id"])

    q90 = pd.read_parquet(Q90, columns=["source_id","teff_bin_s","in_window_gmm"])
    q90["source_id"] = norm_sid(q90["source_id"])

    q95 = pd.read_parquet(Q95, columns=["source_id","teff_bin_s","in_window_gmm"])
    q95["source_id"] = norm_sid(q95["source_id"])

    m90 = kep.merge(q90, on="source_id", how="inner", suffixes=("_kep","_tail"))
    m95 = kep.merge(q95, on="source_id", how="inner", suffixes=("_kep","_tail"))

    b90 = pick_bin_col(m90)
    b95 = pick_bin_col(m95)
    if b90 is None or b95 is None:
        raise RuntimeError(f"Could not find bin column after merge. m90 cols: {list(m90.columns)}")

    out = {
        "q90": {
            "bin_col_used": b90,
            "cols": list(m90.columns),
            "window_from_kepler": pack(m90, "in_window_gmm_kep", b90),
            "window_from_tail": pack(m90, "in_window_gmm_tail", b90),
        },
        "q95": {
            "bin_col_used": b95,
            "cols": list(m95.columns),
            "window_from_kepler": pack(m95, "in_window_gmm_kep", b95),
            "window_from_tail": pack(m95, "in_window_gmm_tail", b95),
        },
    }

    with open(OUT, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)

    print("saved:", OUT)
    print("q90 tail window:", out["q90"]["window_from_tail"])
    print("q95 tail window:", out["q95"]["window_from_tail"])

if __name__ == "__main__":
    main()
