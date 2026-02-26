import os, yaml
import numpy as np
import pandas as pd

INP = "/mnt/g/STAR_HPV/processed/state_vectors/kepler_led_state_vector_gmmgate.parquet"
OUT_DIR = "/mnt/g/STAR_HPV/results/anchor_catalogs"

# Tail thresholds to export
TAIL_QS = [0.90, 0.95]

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_parquet(INP).copy()

    # Basic cleaning
    df["prot_days"] = pd.to_numeric(df["prot_days"], errors="coerce")
    df["Phi_gmm"] = pd.to_numeric(df["Phi_gmm"], errors="coerce")
    df = df.dropna(subset=["prot_days", "Phi_gmm", "teff_bin_s"]).copy()
    df = df[df["prot_days"] > 0].copy()

    # Rotation-weighted ripeness proxy (signed)
    df["ripeness_rot"] = df["Phi_gmm"].astype(float) * df["prot_days"].astype(float)

    summary = {"tails": {}}

    for q in TAIL_QS:
        # Threshold per Teff bin (fair comparison across regimes)
        thr = df.groupby("teff_bin_s")["Phi_gmm"].quantile(q)
        df["phi_tail_thr"] = df["teff_bin_s"].map(thr)

        tail = df[df["Phi_gmm"] >= df["phi_tail_thr"]].copy()

        # Save catalog
        tag = f"q{int(q*100):02d}"
        out_csv = os.path.join(OUT_DIR, f"kepler_ripe_tail_{tag}.csv")
        out_parq = os.path.join(OUT_DIR, f"kepler_ripe_tail_{tag}.parquet")

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
        keep = [c for c in keep if c in tail.columns]
        tail[keep].to_csv(out_csv, index=False)
        tail[keep].to_parquet(out_parq, index=False)

        # Compare medians: tail vs non-tail, per bin
        comp = {}
        for tb, sub in df.groupby("teff_bin_s"):
            t = sub[sub["Phi_gmm"] >= float(thr.loc[tb])]
            n = sub[sub["Phi_gmm"] < float(thr.loc[tb])]
            if len(t) < 20 or len(n) < 20:
                continue
            comp[str(tb)] = {
                "rows_bin": int(len(sub)),
                "rows_tail": int(len(t)),
                "phi_threshold": float(thr.loc[tb]),
                "median_prot_tail": float(np.nanmedian(t["prot_days"])),
                "median_prot_nontail": float(np.nanmedian(n["prot_days"])),
                "delta_median_prot_days": float(np.nanmedian(t["prot_days"]) - np.nanmedian(n["prot_days"])),
                "median_ripeness_tail": float(np.nanmedian(t["ripeness_rot"])),
                "median_ripeness_nontail": float(np.nanmedian(n["ripeness_rot"])),
            }

        summary["tails"][tag] = {
            "q": float(q),
            "rows_total": int(len(df)),
            "rows_tail_total": int(len(tail)),
            "tail_fraction_total": float(len(tail) / max(1, len(df))),
            "rows_tail_by_bin": tail.groupby("teff_bin_s").size().to_dict(),
            "prot_quantiles_tail": {
                "q10": float(tail["prot_days"].quantile(0.10)),
                "q50": float(tail["prot_days"].quantile(0.50)),
                "q90": float(tail["prot_days"].quantile(0.90)),
            },
            "ripeness_quantiles_tail": {
                "q10": float(tail["ripeness_rot"].quantile(0.10)),
                "q50": float(tail["ripeness_rot"].quantile(0.50)),
                "q90": float(tail["ripeness_rot"].quantile(0.90)),
            },
            "comparison_by_bin": comp,
            "outputs": {"csv": out_csv, "parquet": out_parq},
        }

        print(f"saved tail {tag}: rows={len(tail)} -> {out_csv}")

    out_yaml = os.path.join(OUT_DIR, "kepler_ripe_tail_summary.yaml")
    with open(out_yaml, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    print("saved summary:", out_yaml)

if __name__ == "__main__":
    main()
