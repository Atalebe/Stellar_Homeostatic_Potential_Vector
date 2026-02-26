import os, yaml
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

INP = "/mnt/g/STAR_HPV/processed/state_vectors/kepler_led_state_vector_gmmgate.parquet"
OUT = "/mnt/g/STAR_HPV/results/ripeness/kepler_gmmgate_ripeness_proxy_test.yaml"

BINS = ["3200-4000","4000-5200","5200-6000","6000-7500"]

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df = pd.read_parquet(INP)

    out = {"bins": {}}

    for tb in BINS:
        sub = df[df["teff_bin_s"].astype(str) == tb].copy()
        sub["prot_days"] = pd.to_numeric(sub["prot_days"], errors="coerce")
        sub["Phi_gmm"] = pd.to_numeric(sub["Phi_gmm"], errors="coerce")
        sub = sub.dropna(subset=["prot_days","Phi_gmm"]).copy()
        sub = sub[sub["prot_days"] > 0].copy()
        if len(sub) < 100:
            continue

        tau, p = kendalltau(sub["Phi_gmm"], sub["prot_days"])

        w = sub["in_window_gmm"].astype(bool)
        med_win = float(np.nanmedian(sub.loc[w, "prot_days"])) if w.any() else None
        med_non = float(np.nanmedian(sub.loc[~w, "prot_days"])) if (~w).any() else None

        out["bins"][tb] = {
            "rows": int(len(sub)),
            "kendall_tau_phi_vs_prot": float(tau) if tau == tau else None,
            "p_value": float(p) if p == p else None,
            "median_prot_window_days": med_win,
            "median_prot_nonwindow_days": med_non,
            "delta_median_days": (med_win - med_non) if (med_win is not None and med_non is not None) else None,
        }

    with open(OUT, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)

    print("saved:", OUT)
    print(out)

if __name__ == "__main__":
    main()
