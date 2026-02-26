import os, yaml
import numpy as np
import pandas as pd

INP = "/mnt/g/STAR_HPV/processed/state_vectors/kepler_led_state_vector_gmmgate.parquet"
OUT_DIR = "/mnt/g/STAR_HPV/results/anchor_catalogs"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_parquet(INP).copy()

    # Ensure numeric
    df["prot_days"] = pd.to_numeric(df["prot_days"], errors="coerce")
    df = df.dropna(subset=["prot_days", "Phi_gmm"]).copy()
    df = df[df["prot_days"] > 0].copy()

    # Rotation-weighted ripeness proxy
    df["ripeness_rot"] = df["Phi_gmm"].astype(float) * df["prot_days"].astype(float)

    # Anchor = window members in post-gate set
    anchor = df[df["in_window_gmm"] == True].copy()

    # Keep compact, high-signal columns
    keep = [
        "source_id", "kic", "Teff", "teff_bin_s",
        "prot_days", "log_prot",
        "gate_days_gmm", "gate_logP_gmm",
        "R_raw_gmm", "R_gmm",
        "M_G", "H_raw", "H_z",
        "Phi_gmm", "in_window_gmm",
        "ripeness_rot",
        "parallax", "parallax_error", "phot_g_mean_mag", "bp_rp", "ruwe",
        "teff_gspphot", "logg_gspphot", "mh_gspphot",
    ]
    keep = [c for c in keep if c in anchor.columns]
    anchor = anchor[keep].copy()

    out_csv = os.path.join(OUT_DIR, "kepler_anchor_catalog.csv")
    out_parq = os.path.join(OUT_DIR, "kepler_anchor_catalog.parquet")

    anchor.to_csv(out_csv, index=False)
    anchor.to_parquet(out_parq, index=False)

    summary = {
        "rows_total_postgate_gmm": int(len(df)),
        "rows_anchor_window": int(len(anchor)),
        "anchor_fraction": float(len(anchor) / max(1, len(df))),
        "by_teff_bin_rows": df.groupby("teff_bin_s").size().to_dict(),
        "by_teff_bin_anchor": anchor.groupby("teff_bin_s").size().to_dict(),
        "ripeness_rot_quantiles": {
            "q10": float(anchor["ripeness_rot"].quantile(0.10)),
            "q50": float(anchor["ripeness_rot"].quantile(0.50)),
            "q90": float(anchor["ripeness_rot"].quantile(0.90)),
        }
    }

    out_yaml = os.path.join(OUT_DIR, "anchor_summary.yaml")
    with open(out_yaml, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    print("saved:", out_csv)
    print("saved:", out_parq)
    print("saved:", out_yaml)
    print(summary)

if __name__ == "__main__":
    main()
