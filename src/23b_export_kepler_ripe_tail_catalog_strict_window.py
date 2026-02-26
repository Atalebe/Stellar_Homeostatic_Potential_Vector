import os, yaml
import numpy as np
import pandas as pd

INP = "/mnt/g/STAR_HPV/processed/state_vectors/kepler_led_state_vector_gmmgate.parquet"
OUT_DIR = "/mnt/g/STAR_HPV/results/anchor_catalogs"

Q_LIST = [0.90, 0.95]

def norm_sid(s):
    return s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)

def qstats(x):
    x = pd.to_numeric(x, errors="coerce")
    x = x.dropna()
    if len(x) == 0:
        return {"q10": None, "q50": None, "q90": None}
    return {"q10": float(x.quantile(0.10)), "q50": float(x.quantile(0.50)), "q90": float(x.quantile(0.90))}

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_parquet(INP)
    df["source_id"] = norm_sid(df["source_id"])

    # Require key cols
    need = ["source_id","teff_bin_s","prot_days","Phi_gmm","in_window_gmm"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in {INP}: {missing}")

    # Strict window subset
    w = df[df["in_window_gmm"].astype(bool)].copy()
    if len(w) == 0:
        raise RuntimeError("No rows with in_window_gmm==True. Window logic missing upstream?")

    w["prot_days"] = pd.to_numeric(w["prot_days"], errors="coerce")
    w["Phi_gmm"] = pd.to_numeric(w["Phi_gmm"], errors="coerce")
    w = w.dropna(subset=["prot_days","Phi_gmm"]).copy()

    w["ripeness_gmm"] = w["Phi_gmm"] * w["prot_days"]

    summary = {
        "input_rows_total": int(len(df)),
        "rows_in_window": int(len(w)),
        "window_fraction": float(len(w)/len(df)),
        "tails": {}
    }

    for q in Q_LIST:
        tag = f"q{int(round(q*100))}"
        thr = w["ripeness_gmm"].quantile(q)
        tail = w[w["ripeness_gmm"] >= thr].copy()

        # outputs
        p_csv = f"{OUT_DIR}/kepler_ripe_tail_window_{tag}.csv"
        p_par = f"{OUT_DIR}/kepler_ripe_tail_window_{tag}.parquet"

        tail.to_csv(p_csv, index=False)
        tail.to_parquet(p_par, index=False)

        # summary
        by = tail.groupby("teff_bin_s").size().to_dict()
        summary["tails"][tag] = {
            "q": float(q),
            "ripeness_threshold": float(thr),
            "rows_tail": int(len(tail)),
            "tail_fraction_of_window": float(len(tail)/len(w)),
            "rows_tail_by_bin": {str(k): int(v) for k,v in by.items()},
            "prot_quantiles_tail": qstats(tail["prot_days"]),
            "ripeness_quantiles_tail": qstats(tail["ripeness_gmm"]),
            "outputs": {"csv": p_csv, "parquet": p_par},
        }

    out_yaml = f"{OUT_DIR}/kepler_ripe_tail_window_summary.yaml"
    with open(out_yaml, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    print("saved:", out_yaml)
    print(summary)

if __name__ == "__main__":
    main()
