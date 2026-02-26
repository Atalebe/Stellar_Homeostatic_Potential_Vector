import os, yaml
import numpy as np
import pandas as pd

P = "/mnt/g/STAR_HPV/processed/state_vectors/gaia_shv_state_vector_gateaware_v2.parquet"
OUT = "/mnt/g/STAR_HPV/results/gate/postgate_depth_test.yaml"

def main():
    df = pd.read_parquet(P)

    # Use gate-aware fields
    if "R_raw_v2" not in df.columns or "in_window_v2" not in df.columns:
        raise RuntimeError("Expected R_raw_v2 and in_window_v2 in v2 parquet")

    out = {"bins": {}}

    for tb in sorted(df["teff_bin"].astype(str).unique()):
        sub = df[df["teff_bin"].astype(str) == tb].copy()
        if len(sub) < 1000:
            continue

        w = sub["in_window_v2"].astype(bool)
        if w.sum() < 50 or (~w).sum() < 50:
            continue

        a = sub.loc[w, "R_raw_v2"].astype(float)
        b = sub.loc[~w, "R_raw_v2"].astype(float)

        out["bins"][tb] = {
            "rows": int(len(sub)),
            "rows_window": int(w.sum()),
            "Rraw_median_window": float(np.nanmedian(a)),
            "Rraw_median_nonwindow": float(np.nanmedian(b)),
            "Rraw_delta_median": float(np.nanmedian(a) - np.nanmedian(b)),
            "Rraw_q25_window": float(np.nanquantile(a, 0.25)),
            "Rraw_q75_window": float(np.nanquantile(a, 0.75)),
        }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)

    print("saved:", OUT)

if __name__ == "__main__":
    main()
