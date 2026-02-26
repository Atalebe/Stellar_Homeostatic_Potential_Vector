import os
import yaml
import numpy as np
import pandas as pd

INP = "/mnt/g/STAR_HPV/processed/state_vectors/kepler_led_state_vector_gmmgate.parquet"
OUT_DIR = "/mnt/g/STAR_HPV/results/anchor_catalogs"
OUT_YAML = f"{OUT_DIR}/kepler_ripe_tail_window_perbin_summary.yaml"

Q_LIST = [0.90, 0.95]
BINS_ORDER = ["3200-4000", "4000-5200", "5200-6000", "6000-7500"]

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def norm_sid(s):
    return s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)

def qstats(x):
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return {"q10": None, "q50": None, "q90": None}
    return {"q10": float(x.quantile(0.10)),
            "q50": float(x.quantile(0.50)),
            "q90": float(x.quantile(0.90))}

def main():
    ensure_dir(OUT_DIR)

    df = pd.read_parquet(INP)
    df["source_id"] = norm_sid(df["source_id"])

    need = ["source_id", "teff_bin_s", "prot_days", "Phi_gmm", "in_window_gmm"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in {INP}: {missing}")

    # Anchor cohort only
    w = df[df["in_window_gmm"].astype(bool)].copy()
    w["prot_days"] = pd.to_numeric(w["prot_days"], errors="coerce")
    w["Phi_gmm"] = pd.to_numeric(w["Phi_gmm"], errors="coerce")
    w = w.dropna(subset=["prot_days", "Phi_gmm"]).copy()
    w["ripeness_gmm"] = w["prot_days"] * w["Phi_gmm"]

    summary = {
        "input_rows_total": int(len(df)),
        "rows_in_window": int(len(w)),
        "window_fraction": float(len(w) / len(df)) if len(df) else None,
        "definition": {
            "ripeness": "ripeness_gmm = Phi_gmm * prot_days",
            "tail_scope": "within-window, thresholds computed per teff_bin_s",
            "q_list": [float(q) for q in Q_LIST],
        },
        "by_bin": {},
        "outputs": {}
    }

    # Collect per-q tails into two big tables
    tails_by_q = {q: [] for q in Q_LIST}

    for b in BINS_ORDER:
        wb = w[w["teff_bin_s"].astype(str) == b].copy()
        nbin = int(len(wb))
        if nbin == 0:
            summary["by_bin"][b] = {"rows_in_window_bin": 0, "note": "no window rows in bin"}
            continue

        entry = {"rows_in_window_bin": nbin,
                 "prot_quantiles_window": qstats(wb["prot_days"]),
                 "phi_quantiles_window": qstats(wb["Phi_gmm"]),
                 "ripeness_quantiles_window": qstats(wb["ripeness_gmm"]),
                 "tails": {}}

        for q in Q_LIST:
            thr = float(wb["ripeness_gmm"].quantile(q))
            tail = wb[wb["ripeness_gmm"] >= thr].copy()
            entry["tails"][f"q{int(round(q*100))}"] = {
                "q": float(q),
                "ripeness_threshold_bin": thr,
                "rows_tail_bin": int(len(tail)),
                "tail_fraction_of_window_bin": float(len(tail) / nbin) if nbin else None,
                "prot_quantiles_tail": qstats(tail["prot_days"]),
                "phi_quantiles_tail": qstats(tail["Phi_gmm"]),
                "ripeness_quantiles_tail": qstats(tail["ripeness_gmm"]),
            }
            tails_by_q[q].append(tail)

        summary["by_bin"][b] = entry

    # Concatenate and save
    for q in Q_LIST:
        tag = f"q{int(round(q*100))}"
        out_csv = f"{OUT_DIR}/kepler_ripe_tail_window_perbin_{tag}.csv"
        out_par = f"{OUT_DIR}/kepler_ripe_tail_window_perbin_{tag}.parquet"

        if len(tails_by_q[q]) == 0:
            big = pd.DataFrame()
        else:
            big = pd.concat(tails_by_q[q], ignore_index=True)

        # Keep it tidy: retain key columns if present
        keep_cols = [
            "source_id", "kic", "teff_bin_s",
            "prot_days", "Phi_gmm", "ripeness_gmm",
            "in_window_gmm", "gate_days_gmm", "gate_logP_gmm",
        ]
        cols = [c for c in keep_cols if c in big.columns]
        if cols:
            big = big[cols + [c for c in big.columns if c not in cols]]

        big.to_csv(out_csv, index=False)
        big.to_parquet(out_par, index=False)

        summary["outputs"][tag] = {
            "q": float(q),
            "csv": out_csv,
            "parquet": out_par,
            "rows_total_tail": int(len(big)),
        }

    with open(OUT_YAML, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    print("saved:", OUT_YAML)
    print("outputs:", summary["outputs"])

if __name__ == "__main__":
    main()
